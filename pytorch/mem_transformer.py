import sys
import math
import functools

import numpy as np
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('utils')
from proj_adaptive_softmax import ProjectedAdaptiveLogSoftmax
from log_uniform_sampler import LogUniformSampler, sample_logits



from utils.fp16_utils import clamp_hidden

LEVEL="O2"

def einsum_custom(expression, mats, opt_level=LEVEL):
    if opt_level == "O1":
        return torch.einsum(expression, tuple(mat.float() for mat in mats))
    
    return torch.einsum(expression, mats)


class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if bsz is not None:
            return pos_emb[:,None,:].expand(-1, bsz, -1)
        else:
            return pos_emb[:,None,:]


class PositionwiseFF(nn.Module):
    def __init__(self, d_model, d_inner, dropout, pre_lnorm=False):
        super(PositionwiseFF, self).__init__()

        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout

        self.CoreNet = nn.Sequential(
            nn.Linear(d_model, d_inner), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_inner, d_model),
            nn.Dropout(dropout),
        )

        self.layer_norm = nn.LayerNorm(d_model)

        self.pre_lnorm = pre_lnorm

    def forward(self, inp):
        if self.pre_lnorm:
            ##### layer normalization + positionwise feed-forward
            core_out = self.CoreNet(self.layer_norm(inp))

            ##### residual connection
            output = core_out + inp
        else:
            ##### positionwise feed-forward
            core_out = self.CoreNet(inp)

            core_out = clamp_hidden(core_out)

            ##### residual connection + layer normalization
            output = self.layer_norm(inp + core_out)

        return output

class MultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0, 
                 pre_lnorm=False):
        super(MultiHeadAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        self.q_net = nn.Linear(d_model, n_head * d_head, bias=False)
        self.kv_net = nn.Linear(d_model, 2 * n_head * d_head, bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm

    def forward(self, h, attn_mask=None, mems=None):
        ##### multihead attention
        # [hlen x bsz x n_head x d_head]

        if mems is not None:
            c = torch.cat([mems, h], 0)
        else:
            c = h

        if self.pre_lnorm:
            ##### layer normalization
            c = self.layer_norm(c)

        head_q = self.q_net(h)
        head_k, head_v = torch.chunk(self.kv_net(c), 2, -1)

        head_q = head_q.view(h.size(0), h.size(1), self.n_head, self.d_head)
        head_k = head_k.view(c.size(0), c.size(1), self.n_head, self.d_head)
        head_v = head_v.view(c.size(0), c.size(1), self.n_head, self.d_head)

        # [qlen x klen x bsz x n_head]
        attn_score = einsum_custom('ibnd,jbnd->ijbn', (head_q, head_k))
        attn_score.mul_(self.scale)
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score.masked_fill_(attn_mask[None,:,:,None].bool(), -float('inf'))
            elif attn_mask.dim() == 3:
                attn_score.masked_fill_(attn_mask[:,:,:,None].bool(), -float('inf'))

        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)

        # [qlen x klen x bsz x n_head] + [klen x bsz x n_head x d_head] -> [qlen x bsz x n_head x d_head]
        attn_vec = einsum_custom('ijbn,jbnd->ibnd', (attn_prob, head_v))
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        ##### linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            ##### residual connection
            output = h + attn_out
        else:
            ##### residual connection + layer normalization
            output = self.layer_norm(h + attn_out)

        return output

class RelMultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0,
                 tgt_len=None, ext_len=None, mem_len=None, pre_lnorm=False, save_analysis=False):
        super(RelMultiHeadAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout
        self.save_analysis = save_analysis

        self.qkv_net = nn.Linear(d_model, 3 * n_head * d_head, bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm

    def _parallelogram_mask(self, h, w, left=False):
        mask = torch.ones((h, w)).byte()
        m = min(h, w)
        mask[:m,:m] = torch.triu(mask[:m,:m])
        mask[-m:,-m:] = torch.tril(mask[-m:,-m:])

        if left:
            return mask
        else:
            return mask.flip(0)

    def _shift(self, x, qlen, klen, mask, left=False):
        if qlen > 1:
            zero_pad = torch.zeros((x.size(0), qlen-1, x.size(2), x.size(3)),
                                    device=x.device, dtype=x.dtype)
        else:
            zero_pad = torch.zeros(0, device=x.device, dtype=x.dtype)

        if left:
            mask = mask.flip(1)
            x_padded = torch.cat([zero_pad, x], dim=1).expand(qlen, -1, -1, -1)
        else:
            x_padded = torch.cat([x, zero_pad], dim=1).expand(qlen, -1, -1, -1)

        x = x_padded.masked_select(mask[:,:,None,None]) \
                    .view(qlen, klen, x.size(2), x.size(3))

        return x

    def _rel_shift(self, x, zero_triu=False): # see Transformer-xl paper appendix B
        zero_pad = torch.zeros((x.size(0), 1, *x.size()[2:]),
                               device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=1)

        x_padded = x_padded.view(x.size(1) + 1, x.size(0), *x.size()[2:])

        x = x_padded[1:].view_as(x)

        if zero_triu:
            ones = torch.ones((x.size(0), x.size(1)))
            x = x * torch.tril(ones, x.size(1) - x.size(0))[:,:,None,None]

        return x

    def forward(self, w, r, attn_mask=None, mems=None):
        raise NotImplementedError

class RelPartialLearnableMultiHeadAttn(RelMultiHeadAttn):
    def __init__(self, *args, **kwargs):
        super(RelPartialLearnableMultiHeadAttn, self).__init__(*args, **kwargs)

        self.r_net = nn.Linear(self.d_model, self.n_head * self.d_head, bias=False)

    def forward(self, w, r, r_w_bias, r_r_bias, attn_mask=None, mems=None):
        qlen, rlen, bsz = w.size(0), r.size(0), w.size(1)

        if mems is not None:
            cat = torch.cat([mems, w], 0)
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(cat))
            else:
                w_heads = self.qkv_net(cat)
            r_head_k = self.r_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
            w_head_q = w_head_q[-qlen:]
        else:
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(w))
            else:
                w_heads = self.qkv_net(w)
            r_head_k = self.r_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        klen = w_head_k.size(0)

        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)           # qlen x bsz x n_head x d_head
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)           # qlen x bsz x n_head x d_head
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)           # qlen x bsz x n_head x d_head

        r_head_k = r_head_k.view(rlen, self.n_head, self.d_head)                # qlen x n_head x d_head

        #### compute attention score
        rw_head_q = w_head_q + r_w_bias                                         # qlen x bsz x n_head x d_head
        AC = einsum_custom('ibnd,jbnd->ijbn', (rw_head_q, w_head_k))             # qlen x klen x bsz x n_head

        rr_head_q = w_head_q + r_r_bias
        BD = einsum_custom('ibnd,jnd->ijbn', (rr_head_q, r_head_k))              # qlen x klen x bsz x n_head
        BD = self._rel_shift(BD)

        # [qlen x klen x bsz x n_head]
        attn_score = AC + BD
        attn_score.mul_(self.scale)

        #### compute attention probability
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score = attn_score.float().masked_fill(
                    attn_mask[None,:,:,None].bool(), -float('inf')).type_as(attn_score)
            elif attn_mask.dim() == 3:
                attn_score = attn_score.float().masked_fill(
                    attn_mask[:,:,:,None].bool(), -float('inf')).type_as(attn_score)

        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)

        #### compute attention vector
        attn_vec = einsum_custom('ijbn,jbnd->ibnd', (attn_prob, w_head_v))

        # [qlen x bsz x n_head x d_head]
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        ##### linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            ##### residual connection
            output = w + attn_out
        else:
            ##### residual connection + layer normalization
            output = self.layer_norm(w + attn_out)

        if self.save_analysis:
            return output, attn_prob

        return output

class LaMemoRelPartialLearnableMultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0,
                 tgt_len=None, ext_len=None, mem_len=None, pre_lnorm=False, 
                 full_mem=False, save_analysis=False):
        super(LaMemoRelPartialLearnableMultiHeadAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.mem_len = mem_len
        self.dropout = dropout
        self.save_analysis = save_analysis
        
        
        self.full_mem = full_mem

        self.q_net = nn.Linear(d_model, n_head * d_head, bias=False)
        
        self.kv_net = nn.Linear(d_model, 2 * n_head * d_head, bias=False)
        
        self.r_net = nn.Linear(self.d_model, self.n_head * self.d_head, bias=False)
        

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm

        self.register_buffer("train_logits_constant", torch.zeros(1, 1, 1, 1, device=self.o_net.weight.data.device))
        self.register_buffer("eval_logits_constant", torch.zeros(1, 1, 1, 1, device=self.o_net.weight.data.device))

    
    def _rel_shift(self, x, zero_triu=False): # see Transformer-xl paper appendix B
        zero_pad = torch.zeros((x.size(0), 1, *x.size()[2:]),
                               device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=1)

        x_padded = x_padded.view(x.size(1) + 1, x.size(0), *x.size()[2:])

        x = x_padded[1:].view_as(x)

        if zero_triu:
            ones = torch.ones((x.size(0), x.size(1)))
            x = x * torch.tril(ones, x.size(1) - x.size(0))[:,:,None,None]

        return x
    
    def _rel_shift_future(self, x, zero_triu=False): # lamemo rel shift
        zero_pad = torch.zeros((x.size(0), 1, *x.size()[2:]),
                                device=x.device, dtype=x.dtype)
        x_padded = torch.cat([x, zero_pad], dim=1)

        x_padded = x_padded.view(x.size(1) + 1, x.size(0), *x.size()[2:])

        x = x_padded[:-1].view_as(x)

        if zero_triu:
            ones = torch.ones((x.size(0), x.size(1)))
            x = x * torch.triu(ones, x.size(1) - x.size(0))[:,:,None,None]

        return x
    
    def _rel_shift_full(self, x): # lamemo rel shift with full history
        # x: m, 2 * m, bsz, n
        zero_pad = torch.zeros((x.size(0), 1, *x.size()[2:]),
                                device=x.device, dtype=x.dtype)
        x_padded = torch.cat([x, zero_pad], dim=1).view(-1, *x.size()[2:]) # m * 2 * m + m, bsz, n

        x_padded = x_padded[:-x.size(0)].view(x.size(0), -1, *x.size()[2:]) # m, 2m, bsz, n

        return x_padded # m, 2 * m, bsz, n

    def forward(self, w, r, r_w_bias, r_r_bias, r_r_bias_future, r_w_bias_future=None, attn_mask=None, mems=None, recur_mems=None, denom=None):
        qlen, rlen, bsz = w.size(0), r.size(0), w.size(1)
        

        if mems is not None:
            cat = torch.cat([mems, w], 0)
            if self.pre_lnorm:
                w_head_q = self.q_net(self.layer_norm(cat))
                w_head_kv = self.kv_net(self.layer_norm(cat))
            else:
                w_head_q = self.q_net(cat)
                w_head_kv = self.kv_net(cat)
            r_head_k = self.r_net(r)

            w_head_k, w_head_v = torch.chunk(w_head_kv, 2, dim=-1)
            
            klen = w_head_k.size(0)
            if w_head_k.size(0) > qlen:
                mem_head_q = w_head_q[:klen - qlen]
                beg = max(- 2 * qlen + 1, - klen + 1)
                if qlen > 1:
                    mem_head_k = w_head_k[beg : (- qlen + 1)]
                    mem_head_v = w_head_v[beg : (- qlen + 1)]
                else:
                    mem_head_k = w_head_k[beg :]
                    mem_head_v = w_head_v[beg :]

            
            w_head_q = w_head_q[-qlen:]

        else:
            if self.pre_lnorm:
                w_head_q = self.q_net(self.layer_norm(w))
                w_head_kv = self.kv_net(self.layer_norm(w))
            else:
                w_head_q = self.q_net(w)
                w_head_kv = self.kv_net(w)
            r_head_k = self.r_net(r)

            w_head_k, w_head_v = torch.chunk(w_head_kv, 2, dim=-1)

            klen = w_head_k.size(0)

        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)           # klen x bsz x n_head x d_head
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)           # qlen x bsz x n_head x d_head
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)           # qlen x bsz x n_head x d_head

        
        r_head_k_causal = r_head_k[:klen]
        

        r_head_k_causal = r_head_k_causal.view(klen, self.n_head, self.d_head)                # klen x n_head x d_head
        
        

        #### compute attention score
        rw_head_q = w_head_q + r_w_bias
        
        AC = einsum_custom('ibnd,jbnd->ijbn', (rw_head_q, w_head_k))             # qlen x klen x bsz x n_head   


        rr_head_q = w_head_q + r_r_bias
        
        
        BD = einsum_custom('ibnd,jnd->ijbn', (rr_head_q, r_head_k_causal))              # qlen x klen x bsz x n_head
        BD = self._rel_shift(BD)

        # [qlen x klen x bsz x n_head]
        attn_score = AC + BD
        attn_score.mul_(self.scale)
        


        #### compute attention probability
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score = attn_score.float().masked_fill(
                    attn_mask[None,:,:,None].bool(), -float('inf')).type_as(attn_score)
            elif attn_mask.dim() == 3:
                attn_score = attn_score.float().masked_fill(
                    attn_mask[:,:,:,None].bool(), -float('inf')).type_as(attn_score)
                
        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)

        #### compute attention vector
        attn_vec = einsum_custom('ijbn,jbnd->ibnd', (attn_prob, w_head_v))

        # [qlen x bsz x n_head x d_head]
        _attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        ##### linear projection
        attn_out = self.o_net(_attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            ##### residual connection
            output = w + attn_out
        else:
            ##### residual connection + layer normalization
            output = self.layer_norm(w + attn_out)


        #### look-ahead memory
        if w_head_k.size(0) > qlen:
            mem_head_q = mem_head_q.view(klen - qlen, bsz, self.n_head, self.d_head)   # klen - qlen, bsz, n_head, d_head
            mem_k_len = min(klen - qlen, qlen)
            mem_head_k = mem_head_k.view(mem_k_len, bsz, self.n_head, self.d_head)          # qlen, bsz, n_head, d_head
            mem_head_v = mem_head_v.view(mem_k_len, bsz, self.n_head, self.d_head)         # qlen, bsz, n_head, d_head
            
            if r_w_bias_future is not None:
                mem_rw_head_q = mem_head_q + r_w_bias_future
            else:
                mem_rw_head_q = mem_head_q + r_w_bias
            

            
            AC_mem = einsum_custom('ibnd,jbnd->ijbn', (mem_rw_head_q, mem_head_k))

            if not self.full_mem:
                r_head_k_ahead = r_head_k[klen:]
            else:
                r_head_k_ahead = r_head_k[- 2 * (klen - qlen):]
            
            r_head_k_ahead = r_head_k_ahead.view(-1, self.n_head, self.d_head) # klen - qlen, bsz, n_head, d_head
            
            

            BD_mem = einsum_custom('ibnd,jnd->ijbn', (mem_head_q + r_r_bias_future, r_head_k_ahead)) # klen - qlen, klen - qlen, bsz, n_head

            
            if self.full_mem:
                BD_mem = self._rel_shift_full(BD_mem)[:, -qlen:].contiguous()
            else:
                BD_mem = self._rel_shift_future(BD_mem)[:, -qlen:].contiguous() # klen - qlen, qlen, bsz, d_head
            
            
            attn_score_mem = AC_mem + BD_mem

            attn_score_mem.mul_(self.scale)


            if not self.full_mem:
                attn_mask_mem = torch.ones(klen - qlen, klen - qlen, dtype=attn_mask.dtype, device=attn_mask.device)
                attn_mask_mem = torch.triu(attn_mask_mem)[:, -qlen:].ne(1).bool()[:, :, None, None]
                
                attn_score_mem = attn_score_mem.float().masked_fill_(attn_mask_mem, -float('inf')).type_as(attn_score_mem)
                
            
            
            attn_prob_mem = F.softmax(attn_score_mem, dim=1)
            attn_prob_mem = self.dropatt(attn_prob_mem)

            attn_vec_mem = einsum_custom('ijbn,jbnd->ibnd', (attn_prob_mem, mem_head_v)) # klen - qlen, bsz, n_head, d_head
            
            
            mem_denom = torch.logsumexp(attn_score_mem.float(), 1)
            
            
            eps = 1e-6

            Constant = torch.stack([mem_denom, denom]).max(0)[0]

            _denom = (denom - Constant).float().exp()
            _mem_denom = (mem_denom - Constant).float().exp()
            
            alpha = _denom / (_denom + _mem_denom + eps)
            alpha = alpha.unsqueeze(-1)            
            
            inter_mems = recur_mems * alpha + attn_vec_mem * (1 - alpha)
            
            inter_mems = inter_mems.type_as(attn_vec_mem)
            inter_mems.nan_to_num_()
            
            
            _inter_mems = inter_mems.contiguous().view(
                attn_vec_mem.size(0), attn_vec_mem.size(1), self.n_head * self.d_head)
            
            
            attn_out_mem = self.o_net(_inter_mems)
            attn_out_mem = self.drop(attn_out_mem)
            
            if self.pre_lnorm:
                ##### residual connection
                output_mem = mems + attn_out_mem
            else:
                ##### residual connection + layer normalization
                output_mem = self.layer_norm(mems + attn_out_mem)
            

        else:
            output_mem = torch.empty(0, dtype=attn_score.dtype, device=attn_score.device)
            mem_denom = torch.empty(0, dtype=attn_score.dtype, device=attn_score.device)
            inter_mems = torch.empty(0, dtype=attn_score.dtype, device=attn_score.device)
            attn_prob, attn_prob_mem, alpha = None, None, None
        cur_denom = torch.logsumexp(attn_score.float(), 1)
        
        prev_denom = torch.logaddexp(denom.float(), mem_denom.float())
        new_denom = torch.cat([prev_denom, cur_denom], dim=0) # mlen + qlen
        
        beg_idx = max(0, klen - self.mem_len)
        
        
        new_denom = new_denom[beg_idx:].detach().type_as(output)
        new_denom.nan_to_num_()
        
        #### update recurrent memory
        new_recur_mems = torch.cat([inter_mems, attn_vec], dim=0)[beg_idx:].detach()
        
            
        # tgt output
        # layer memory output
        # recurrent memory output
        # recurrent denom
        

        if self.save_analysis:
            return output, output_mem, new_recur_mems, new_denom, attn_prob, attn_prob_mem, alpha

        return output, output_mem, new_recur_mems, new_denom



class RelLearnableMultiHeadAttn(RelMultiHeadAttn):
    def __init__(self, *args, **kwargs):
        super(RelLearnableMultiHeadAttn, self).__init__(*args, **kwargs)

    def forward(self, w, r_emb, r_w_bias, r_bias, attn_mask=None, mems=None):
        # r_emb: [klen, n_head, d_head], used for term B
        # r_w_bias: [n_head, d_head], used for term C
        # r_bias: [klen, n_head], used for term D

        qlen, bsz = w.size(0), w.size(1)

        if mems is not None:
            cat = torch.cat([mems, w], 0)
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(cat))
            else:
                w_heads = self.qkv_net(cat)
            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

            w_head_q = w_head_q[-qlen:]
        else:
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(w))
            else:
                w_heads = self.qkv_net(w)
            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        klen = w_head_k.size(0)

        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)

        if klen > r_emb.size(0):
            r_emb_pad = r_emb[0:1].expand(klen-r_emb.size(0), -1, -1)
            r_emb = torch.cat([r_emb_pad, r_emb], 0)
            r_bias_pad = r_bias[0:1].expand(klen-r_bias.size(0), -1)
            r_bias = torch.cat([r_bias_pad, r_bias], 0)
        else:
            r_emb = r_emb[-klen:]
            r_bias = r_bias[-klen:]

        #### compute attention score
        rw_head_q = w_head_q + r_w_bias[None]                                   # qlen x bsz x n_head x d_head

        AC = einsum_custom('ibnd,jbnd->ijbn', (rw_head_q, w_head_k))             # qlen x klen x bsz x n_head
        B_ = einsum_custom('ibnd,jnd->ijbn', (w_head_q, r_emb))                  # qlen x klen x bsz x n_head
        D_ = r_bias[None, :, None]                                              # 1    x klen x 1   x n_head
        BD = self._rel_shift(B_ + D_)

        # [qlen x klen x bsz x n_head]
        attn_score = AC + BD
        attn_score.mul_(self.scale)

        #### compute attention probability
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score.masked_fill_(attn_mask[None,:,:,None].bool(), -float('inf'))
            elif attn_mask.dim() == 3:
                attn_score.masked_fill_(attn_mask[:,:,:,None].bool(), -float('inf'))

        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)

        #### compute attention vector
        attn_vec = einsum_custom('ijbn,jbnd->ibnd', (attn_prob, w_head_v))

        # [qlen x bsz x n_head x d_head]
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        ##### linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            ##### residual connection
            output = w + attn_out
        else:
            ##### residual connection + layer normalization
            output = self.layer_norm(w + attn_out)

        return output

class LaMemoRelLearnableMultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0,
                 tgt_len=None, ext_len=None, mem_len=None, pre_lnorm=False,):
        super(LaMemoRelLearnableMultiHeadAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.mem_len = mem_len
        self.dropout = dropout

        self.q_net = nn.Linear(d_model, n_head * d_head, bias=False)
        self.kv_net = nn.Linear(d_model, 2 * n_head * d_head, bias=False)
        
        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm
    
    def _rel_shift(self, x, zero_triu=False): # see Transformer-xl paper appendix B
        zero_pad = torch.zeros((x.size(0), 1, *x.size()[2:]),
                               device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=1)

        x_padded = x_padded.view(x.size(1) + 1, x.size(0), *x.size()[2:])

        x = x_padded[1:].view_as(x)

        if zero_triu:
            ones = torch.ones((x.size(0), x.size(1)))
            x = x * torch.tril(ones, x.size(1) - x.size(0))[:,:,None,None]

        return x
    
    def _rel_shift_future(self, x, zero_triu=False): # lamemo rel shift
        zero_pad = torch.zeros((x.size(0), 1, *x.size()[2:]),
                                device=x.device, dtype=x.dtype)
        x_padded = torch.cat([x, zero_pad], dim=1)

        x_padded = x_padded.view(x.size(1) + 1, x.size(0), *x.size()[2:])

        x = x_padded[:-1].view_as(x)

        if zero_triu:
            ones = torch.ones((x.size(0), x.size(1)))
            x = x * torch.triu(ones, x.size(1) - x.size(0))[:,:,None,None]

        return x
    
    def _rel_shift_full(self, x): # lamemo rel shift with full history
        # x: m, 2 * m, bsz, n
        zero_pad = torch.zeros((x.size(0), 1, *x.size()[2:]),
                                device=x.device, dtype=x.dtype)
        x_padded = torch.cat([x, zero_pad], dim=1).view(-1, *x.size()[2:]) # m * 2 * m + m, bsz, n

        x_padded = x_padded[:-x.size(0)].view(x.size(0), -1, *x.size()[2:]) # m, 2m, bsz, n

        return x_padded[:, x.size(0) - 1:].contiguous() # m, m + 1, bsz, n

    
    def forward(self, w, r_emb, attn_mask=None, mems=None, recur_mems=None, denom=None):
        qlen, rlen, bsz = w.size(0), r_emb.size(0), w.size(1)

        if mems is not None:
            cat = torch.cat([mems, w], 0)
            if self.pre_lnorm:
                w_head_q = self.q_net(self.layer_norm(cat))
                w_head_kv = self.kv_net(self.layer_norm(cat))
            else:
                w_head_q = self.q_net(cat)
                w_head_kv = self.kv_net(cat)
            
            w_head_k, w_head_v = torch.chunk(w_head_kv, 2, dim=-1)
            
            klen = w_head_k.size(0)

            if w_head_k.size(0) > qlen:
                # mixed with past
                mem_head_q = w_head_q[:klen - qlen]             # mlen, bsz, ...
                mem_head_k = w_head_k[: (- qlen + 1)]           # mlen + 1, bsz, ...
                mem_head_v = w_head_v[: (- qlen + 1)]           # mlen + 1, bsz, ...
            
            w_head_q = w_head_q[-qlen:]
        
        else:
            if self.pre_lnorm:
                w_head_q = self.q_net(self.layer_norm(w))
                w_head_kv = self.kv_net(self.layer_norm(w))
            else:
                w_head_q = self.q_net(w)
                w_head_kv = self.kv_net(w)

            w_head_k, w_head_v = torch.chunk(w_head_kv, 2, dim=-1)

            klen = w_head_k.size(0)

        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)           # klen x bsz x n_head x d_head
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)           # qlen x bsz x n_head x d_head
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)           # qlen x bsz x n_head x d_head

        r_emb_causal = r_emb[:klen]

        AC = einsum_custom('ibnd,jbnd->ijbn', (w_head_q, w_head_k))             # qlen x klen x bsz x n_head
        BD = einsum_custom('ibnd,jnd->ijbn', (w_head_q, r_emb_causal))
        BD = self._rel_shift(BD)

        attn_score = AC + BD
        attn_score.mul_(self.scale)

        #### compute attention probability
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score.masked_fill_(attn_mask[None,:,:,None].bool(), -float('inf'))
            elif attn_mask.dim() == 3:
                attn_score.masked_fill_(attn_mask[:,:,:,None].bool(), -float('inf'))
        
        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)

        #### compute attention vector
        attn_vec = einsum_custom('ijbn,jbnd->ibnd', (attn_prob, w_head_v))

        # [qlen x bsz x n_head x d_head]
        _attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        ##### linear projection
        attn_out = self.o_net(_attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            ##### residual connection
            output = w + attn_out
        else:
            ##### residual connection + layer normalization
            output = self.layer_norm(w + attn_out)
        
        #### look-ahead memory
        if w_head_k.size(0) > qlen:
            mem_head_q = mem_head_q.view(klen - qlen, bsz, self.n_head, self.d_head)   # klen - qlen, bsz, n_head, d_head
            mem_head_k = mem_head_k.view(klen - qlen + 1, bsz, self.n_head, self.d_head)          # klen - qlen + 1, bsz, n_head, d_head
            mem_head_v = mem_head_v.view(klen - qlen + 1, bsz, self.n_head, self.d_head)         # klen - qlen + 1, bsz, n_head, d_head

            r_emb_mem = r_emb[-2 * (klen - qlen) : ] # 2 * (klen - qlen), n_head, d_head

            AC_mem = einsum_custom('ibnd,jbnd->ijbn', (mem_head_q, mem_head_k)) # klen - qlen, klen - qlen + 1, bsz, n_head

            BD_mem = einsum_custom('ibnd,jnd->ijbn', (mem_head_q, r_emb_mem)) # klen - qlen, 2 * (klen - qlen), bsz, n_head

            BD_mem = self._rel_shift_full(BD_mem) # klen - qlen, klen - qlen + 1, bsz, n_head

            attn_score_mem = AC_mem + BD_mem

            attn_score_mem.mul_(self.scale)
            # bi-directional attention

            attn_prob_mem = F.softmax(attn_score_mem, dim=1)
            attn_prob_mem = self.dropatt(attn_prob_mem)

            attn_vec_mem = einsum_custom('ijbn,jbnd->ibnd', (attn_prob_mem, mem_head_v)) # klen - qlen, bsz, n_head, d_head

            ###### alpha
            mem_denom = torch.logsumexp(attn_score_mem.float(), 1)
            
            eps = 1e-6

            Constant = torch.stack([mem_denom, denom]).max(0)[0]

            _denom = (denom - Constant).exp()
            _mem_denom = (mem_denom - Constant).exp()
            
            alpha = _denom / (_denom + _mem_denom + eps)
            
            alpha = (denom - torch.logaddexp(denom, mem_denom)).exp()
            alpha = alpha.unsqueeze(-1)

            ##### interpolation

            inter_mems = recur_mems * alpha + attn_vec_mem * (1 - alpha)

            inter_mems = inter_mems.type_as(attn_vec_mem)

            _inter_mems = inter_mems.contiguous().view(
                attn_vec_mem.size(0), attn_vec_mem.size(1), self.n_head * self.d_head)

            attn_out_mem = self.o_net(_inter_mems)
            
            attn_out_mem = self.drop(attn_out_mem)

            if self.pre_lnorm:
                ##### residual connection
                output_mem = mems + attn_out_mem
            else:
                ##### residual connection + layer normalization
                output_mem = self.layer_norm(mems + attn_out_mem)
            
        else:
            output_mem = torch.empty(0, dtype=attn_score.dtype, device=attn_score.device)
            mem_denom = torch.empty(0, dtype=attn_score.dtype, device=attn_score.device)
            inter_mems = torch.empty(0, dtype=attn_score.dtype, device=attn_score.device)

        cur_denom = torch.logsumexp(attn_score.float(), 1)

        prev_denom = torch.logaddexp(denom, mem_denom)
        new_denom = torch.cat([prev_denom, cur_denom], dim=0) # mlen + qlen
        
        beg_idx = max(0, klen - self.mem_len)
        
        # TODO: Are we going to detach it ?
        # stable log ?
        new_denom = new_denom[beg_idx:].detach()
        
        
        #### update recurrent memory
        new_recur_mems = torch.cat([inter_mems, attn_vec], dim=0)[beg_idx:].detach()

        return output, output_mem, new_recur_mems, new_denom


class LinearAttentionDecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout, **kwargs):
        super(LinearAttentionDecoderLayer, self).__init__()

        self.dec_attn = AutogradLinearMultiHeadAttn(n_head, d_model, d_head, dropout, **kwargs)
        self.pos_ff = PositionwiseFF(d_model, d_inner, dropout, 
                                     pre_lnorm=kwargs.get('pre_lnorm'))
    
    def forward(self, dec_inp, **kwargs):

        output, new_mems = self.dec_attn(dec_inp, **kwargs)

        output = self.pos_ff(output)

        return output, new_mems

class LunaAttentionDecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout, **kwargs):
        super(LunaAttentionDecoderLayer, self).__init__()

        self.dec_attn = LunaAttn(n_head, d_model, d_head, dropout, **kwargs)
        self.pos_ff = PositionwiseFF(d_model, d_inner, dropout, 
                                     pre_lnorm=kwargs.get('pre_lnorm'))
    
    def forward(self, dec_inp, p, **kwargs):

        output = self.dec_attn(dec_inp, p, **kwargs)

        output = self.pos_ff(output)

        return output


class DecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout, **kwargs):
        super(DecoderLayer, self).__init__()

        self.dec_attn = MultiHeadAttn(n_head, d_model, d_head, dropout, **kwargs)
        self.pos_ff = PositionwiseFF(d_model, d_inner, dropout, 
                                     pre_lnorm=kwargs.get('pre_lnorm'))

    def forward(self, dec_inp, dec_attn_mask=None, mems=None):

        output = self.dec_attn(dec_inp, attn_mask=dec_attn_mask,
                               mems=mems)
        output = self.pos_ff(output)

        return output

class RelLearnableDecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout,
                 **kwargs):
        super(RelLearnableDecoderLayer, self).__init__()

        self.dec_attn = RelLearnableMultiHeadAttn(n_head, d_model, d_head, dropout,
                                         **kwargs)
        self.pos_ff = PositionwiseFF(d_model, d_inner, dropout, 
                                     pre_lnorm=kwargs.get('pre_lnorm'))

    def forward(self, dec_inp, r_emb, r_w_bias, r_bias, dec_attn_mask=None, mems=None):

        output = self.dec_attn(dec_inp, r_emb, r_w_bias, r_bias,
                               attn_mask=dec_attn_mask,
                               mems=mems)
        output = self.pos_ff(output)

        return output

class RelPartialLearnableDecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout,
                 **kwargs):
        super(RelPartialLearnableDecoderLayer, self).__init__()
        self.kwargs = kwargs
        self.dec_attn = RelPartialLearnableMultiHeadAttn(n_head, d_model,
                            d_head, dropout, **kwargs)
        self.pos_ff = PositionwiseFF(d_model, d_inner, dropout, 
                                     pre_lnorm=kwargs.get('pre_lnorm'))

    def forward(self, dec_inp, r, r_w_bias, r_r_bias, dec_attn_mask=None, mems=None):

        outputs = self.dec_attn(dec_inp, r, r_w_bias, r_r_bias,
                               attn_mask=dec_attn_mask,
                               mems=mems)
        if self.kwargs["save_analysis"]:
            output, attn_prob = outputs
        else:
            output = outputs
        output = self.pos_ff(output)
        
        if self.kwargs["save_analysis"]:
            return output, attn_prob

        return output

class LaMemoRelPartialLearnableDecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout,
                 **kwargs):
        super(LaMemoRelPartialLearnableDecoderLayer, self).__init__()
        self.kwargs = kwargs
        self.dec_attn = LaMemoRelPartialLearnableMultiHeadAttn(n_head, d_model,
                            d_head, dropout, **kwargs)
        self.pos_ff = PositionwiseFF(d_model, d_inner, dropout, 
                                     pre_lnorm=kwargs.get('pre_lnorm'))
    
    def forward(self, dec_inp, r, r_w_bias, r_r_bias, r_r_bias_future, dec_attn_mask=None, mems=None, recur_mems=None, denom=None, **kwargs):

        
        outputs = self.dec_attn(dec_inp, r, r_w_bias, r_r_bias, r_r_bias_future,
                               attn_mask=dec_attn_mask,
                               mems=mems, recur_mems=recur_mems, denom=denom, **kwargs)
        if self.kwargs["save_analysis"]:
            output, output_mem, new_recur_mems, new_denom, attn_prob, attn_prob_mem, alpha = outputs
        else:
            output, output_mem, new_recur_mems, new_denom = outputs
        shape = [output_mem.size(0), output.size(0)]
        
        output_cat = torch.cat([output_mem, output], 0)

        #print("before ffw max: {}".format(output_cat.max()))
        output_cat = self.pos_ff(output_cat)
        #print("after ffw max: {}".format(output_cat.max()))
        
        output_mem, output = torch.split(output_cat, shape)
        if self.kwargs["save_analysis"]:
            return output, output_mem, new_recur_mems, new_denom, attn_prob, attn_prob_mem, alpha
        return output, output_mem, new_recur_mems, new_denom

class LaMemoRelLearnableDecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout,
                 **kwargs):
        super(LaMemoRelLearnableDecoderLayer, self).__init__()

        self.dec_attn = LaMemoRelLearnableMultiHeadAttn(n_head, d_model,
                            d_head, dropout, **kwargs)
        self.pos_ff = PositionwiseFF(d_model, d_inner, dropout, 
                                     pre_lnorm=kwargs.get('pre_lnorm'))
    
    def forward(self, dec_inp, r_emb, dec_attn_mask, mems=None, recur_mems=None, denom=None, **kwargs):

        output, output_mem, new_recur_mems, new_denom = self.dec_attn(dec_inp, r_emb, attn_mask=dec_attn_mask,
                                                            mems=mems, recur_mems=recur_mems, denom=denom, **kwargs)

        shape = [output_mem.size(0), output.size(0)]
        
        output_cat = torch.cat([output_mem, output], 0)

        #print("before ffw max: {}".format(output_cat.max()))
        output_cat = self.pos_ff(output_cat)
        #print("after ffw max: {}".format(output_cat.max()))
        
        output_mem, output = torch.split(output_cat, shape)

        return output, output_mem, new_recur_mems, new_denom

class AdaptiveEmbedding(nn.Module):
    def __init__(self, n_token, d_embed, d_proj, cutoffs, div_val=1, 
                 sample_softmax=False):
        super(AdaptiveEmbedding, self).__init__()

        self.n_token = n_token
        self.d_embed = d_embed

        self.cutoffs = cutoffs + [n_token]
        self.div_val = div_val
        self.d_proj = d_proj

        self.emb_scale = d_proj ** 0.5

        self.cutoff_ends = [0] + self.cutoffs

        self.emb_layers = nn.ModuleList()
        self.emb_projs = nn.ParameterList()
        if div_val == 1:
            self.emb_layers.append(
                nn.Embedding(n_token, d_embed, sparse=sample_softmax>0)
            )
            if d_proj != d_embed:
                self.emb_projs.append(nn.Parameter(torch.Tensor(d_proj, d_embed)))
        else:
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i+1]
                d_emb_i = d_embed // (div_val ** i)
                self.emb_layers.append(nn.Embedding(r_idx-l_idx, d_emb_i))
                self.emb_projs.append(nn.Parameter(torch.Tensor(d_proj, d_emb_i)))

    def forward(self, inp):
        if self.div_val == 1:
            embed = self.emb_layers[0](inp)
            if self.d_proj != self.d_embed:
                embed  = F.linear(embed, self.emb_projs[0])
        else:
            param = next(self.parameters())
            inp_flat = inp.view(-1)
            emb_flat = torch.zeros([inp_flat.size(0), self.d_proj], 
                dtype=param.dtype, device=param.device)
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]

                mask_i = (inp_flat >= l_idx) & (inp_flat < r_idx)
                indices_i = mask_i.nonzero().squeeze()

                if indices_i.numel() == 0:
                    continue

                inp_i = inp_flat.index_select(0, indices_i) - l_idx
                emb_i = self.emb_layers[i](inp_i)
                emb_i = F.linear(emb_i, self.emb_projs[i])

                emb_flat.index_copy_(0, indices_i, emb_i)

            embed = emb_flat.view(*inp.size(), self.d_proj)

        embed.mul_(self.emb_scale)

        return embed

class MemTransformerLM(nn.Module):
    def __init__(self, n_token, n_layer, n_head, d_model, d_head, d_inner,
                 dropout, dropatt, tie_weight=True, d_embed=None, 
                 div_val=1, tie_projs=[False], pre_lnorm=False,
                 tgt_len=None, ext_len=None, mem_len=None, 
                 cutoffs=[], adapt_inp=False,
                 same_length=False, attn_type=0, clamp_len=-1, 
                 sample_softmax=-1, attn_norm_type=0, half_loss=False, 
                 enhance_recurrence=False, save_analysis=False):
        super(MemTransformerLM, self).__init__()
        self.n_token = n_token

        d_embed = d_model if d_embed is None else d_embed
        self.d_embed = d_embed
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head

        self.save_analysis = save_analysis

        self.word_emb = AdaptiveEmbedding(n_token, d_embed, d_model, cutoffs, 
                                          div_val=div_val)
        self.cutoffs = cutoffs

        self.drop = nn.Dropout(dropout)

        self.n_layer = n_layer

        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.ext_len = ext_len
        self.max_klen = tgt_len + ext_len + mem_len

        self.attn_type = attn_type
        self.attn_norm_type = attn_norm_type
        if self.attn_type in [0,1,2]:
            self.memory_type = "fifo"
        elif self.attn_type in [3,31,32,33]:
            self.memory_type = "lamemo"
        else:
            self.memory_type = "none"

        self.half_loss = half_loss

        self.enhance_recurrence = enhance_recurrence

        self.layers = nn.ModuleList()
        if attn_type == 0: # the default attention
            for i in range(n_layer):
                self.layers.append(
                    RelPartialLearnableDecoderLayer(
                        n_head, d_model, d_head, d_inner, dropout,
                        tgt_len=tgt_len, ext_len=ext_len, mem_len=mem_len,
                        dropatt=dropatt, pre_lnorm=pre_lnorm, save_analysis=self.save_analysis)
                )
        elif attn_type == 1: # learnable embeddings
            for i in range(n_layer):
                self.layers.append(
                    RelLearnableDecoderLayer(
                        n_head, d_model, d_head, d_inner, dropout,
                        tgt_len=tgt_len, ext_len=ext_len, mem_len=mem_len,
                        dropatt=dropatt, pre_lnorm=pre_lnorm)
                )
        elif attn_type == 2: # absolute embeddings
            for i in range(n_layer):
                self.layers.append(
                    DecoderLayer(
                        n_head, d_model, d_head, d_inner, dropout,
                        dropatt=dropatt, pre_lnorm=pre_lnorm)
                )
        
        elif attn_type in [3, 31, 32, 33]: # lamemo 
            for i in range(n_layer):
                self.layers.append(
                    LaMemoRelPartialLearnableDecoderLayer(
                        n_head, d_model, d_head, d_inner, dropout,
                        tgt_len=tgt_len, ext_len=ext_len, mem_len=mem_len,
                        dropatt=dropatt, pre_lnorm=pre_lnorm, 
                        full_mem=attn_type==33, save_analysis=self.save_analysis
                    )
                )
        


        self.sample_softmax = sample_softmax
        # use sampled softmax
        if sample_softmax > 0:
            self.out_layer = nn.Linear(d_model, n_token)
            if tie_weight:
                self.out_layer.weight = self.word_emb.weight
            self.tie_weight = tie_weight
            self.sampler = LogUniformSampler(n_token, sample_softmax)

        # use adaptive softmax (including standard softmax)
        else:
            self.crit = ProjectedAdaptiveLogSoftmax(n_token, d_embed, d_model, 
                                                    cutoffs, div_val=div_val)

            if tie_weight:
                for i in range(len(self.crit.out_layers)):
                    self.crit.out_layers[i].weight = self.word_emb.emb_layers[i].weight

            if tie_projs:
                for i, tie_proj in enumerate(tie_projs):
                    if tie_proj and div_val == 1 and d_model != d_embed:
                        self.crit.out_projs[i] = self.word_emb.emb_projs[0]
                    elif tie_proj and div_val != 1:
                        self.crit.out_projs[i] = self.word_emb.emb_projs[i]

        self.same_length = same_length
        self.clamp_len = clamp_len

        self._create_params()

    def backward_compatible(self):
        self.sample_softmax = -1

    def _create_params(self):
        if self.attn_type in [0,3,31,32,33]: # default attention
            self.pos_emb = PositionalEmbedding(self.d_model)
            self.r_w_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))
            self.r_r_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))
            nn.init.xavier_uniform_(self.r_r_bias.data)
            nn.init.xavier_uniform_(self.r_w_bias.data)
            if self.attn_type in [3, 33]:
                self.r_r_bias_future = nn.Parameter(torch.Tensor(self.n_head, self.d_head))
                nn.init.xavier_uniform_(self.r_r_bias_future.data)
            
            if self.attn_type == 32:
                self.r_r_bias_future = nn.Parameter(torch.Tensor(self.n_head, self.d_head))
                self.r_w_bias_future = nn.Parameter(torch.Tensor(self.n_head, self.d_head))
                nn.init.xavier_uniform_(self.r_r_bias_future.data)
                nn.init.xavier_uniform_(self.r_w_bias_future.data)
        

        elif self.attn_type == 1: # learnable
            self.r_emb = nn.Parameter(torch.Tensor(
                    self.n_layer, self.max_klen, self.n_head, self.d_head))
            self.r_w_bias = nn.Parameter(torch.Tensor(
                    self.n_layer, self.n_head, self.d_head))
            self.r_bias = nn.Parameter(torch.Tensor(
                    self.n_layer, self.max_klen, self.n_head))
        elif self.attn_type == 2: # absolute standard
            self.pos_emb = PositionalEmbedding(self.d_model)
        

    def reset_length(self, tgt_len, ext_len, mem_len):
        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.ext_len = ext_len

    def init_mems(self, memory_type="fifo"):
        param = next(self.parameters())
        if memory_type == "fifo":
            if self.mem_len > 0:
                mems = []
                
                if self.enhance_recurrence:
                    num_mem_layers = self.n_layer
                else:
                    num_mem_layers = self.n_layer+1

                for i in range(num_mem_layers):
                    empty = torch.empty(0, dtype=param.dtype, device=param.device)
                    mems.append(empty)

                return mems
            else:
                return None
        
        
        elif memory_type == "lamemo":
            if self.mem_len > 0:
                
                # the 0-th element of this list is the word embeddings
                # among the 1 to 2*n_layer+1-th elements
                # odd elements are recur_mems
                # even elements are denoms
                mems = []
                n_elements = 2 * self.n_layer + 1

                for i in range(n_elements):
                    empty = torch.empty(0, dtype=param.dtype, device=param.device)
                    mems.append(empty)
                
                # use float32 to store denom
                # prevent explode
                for i in range(2, n_elements, 2):
                    mems[i] = torch.empty(0, dtype=torch.float32, device=param.device)
                
                return mems
            else:
                return None

        elif memory_type == "none":
            return None

    def _update_mems(self, hids, mems, qlen, mlen, memory_type="fifo"):
        # does not deal with None
        if mems is None: return None

        

        # There are `mlen + qlen` steps that can be cached into mems
        # For the next step, the last `ext_len` of the `qlen` tokens
        # will be used as the extended context. Hence, we only cache
        # the tokens from `mlen + qlen - self.ext_len - self.mem_len`
        # to `mlen + qlen - self.ext_len`.
        if memory_type == "fifo":
            # mems is not None
            assert len(hids) == len(mems), 'len(hids) != len(mems)'

            with torch.no_grad():
                new_mems = []
                end_idx = mlen + max(0, qlen - 0 - self.ext_len)
                beg_idx = max(0, end_idx - self.mem_len)
                for i in range(len(hids)):

                    cat = torch.cat([mems[i], hids[i]], dim=0)
                    new_mems.append(cat[beg_idx:end_idx].detach())

        
        elif memory_type == "lamemo":
            with torch.no_grad():
                end_idx = mlen + qlen
                beg_idx = max(0, end_idx - self.mem_len)
                # fifo-style update for the first word embedding layer
                cat = torch.cat([mems[0], hids[0]], dim=0)[beg_idx:end_idx].detach()

                return [cat] + hids[1:]


        elif memory_type == "none":
            return None

        return new_mems

    def _forward(self, dec_inp, mems=None):
        qlen, bsz = dec_inp.size()

        word_emb = self.word_emb(dec_inp)
        ##print(mems)
        
        mlen = mems[0].size(0) if mems is not None else 0
        
        klen = mlen + qlen
        save_results = []

        if self.same_length:
            all_ones = word_emb.new_ones(qlen, klen)
            mask_len = klen - self.mem_len
            if mask_len > 0:
                mask_shift_len = qlen - mask_len
            else:
                mask_shift_len = qlen
            dec_attn_mask = (torch.triu(all_ones, 1+mlen)
                    + torch.tril(all_ones, -mask_shift_len)).byte()[:, :, None] # -1
        else:
            dec_attn_mask = torch.triu(
                word_emb.new_ones(qlen, klen), diagonal=1+mlen).byte()[:,:,None]

        hids = []
        if self.attn_type == 0: # default
            pos_seq = torch.arange(klen-1, -1, -1.0, device=word_emb.device, 
                                   dtype=word_emb.dtype)
            if self.clamp_len > 0:
                pos_seq.clamp_(max=self.clamp_len)
            pos_emb = self.pos_emb(pos_seq)

            core_out = self.drop(word_emb)
            pos_emb = self.drop(pos_emb)

            if not self.enhance_recurrence:
                hids.append(core_out)
            

            for i, layer in enumerate(self.layers):
                mems_i = None if mems is None else mems[i]
                core_outs = layer(core_out, pos_emb, self.r_w_bias,
                        self.r_r_bias, dec_attn_mask=dec_attn_mask, mems=mems_i)
                if self.save_analysis:
                    core_out, attn_prob = core_outs
                    save_results.append(attn_prob)
                else:
                    core_out = core_outs
                hids.append(core_out)
        
        elif self.attn_type == 1: # learnable
            core_out = self.drop(word_emb)
            hids.append(core_out)
            for i, layer in enumerate(self.layers):
                if self.clamp_len > 0:
                    r_emb = self.r_emb[i][-self.clamp_len :]
                    r_bias = self.r_bias[i][-self.clamp_len :]
                else:
                    r_emb, r_bias = self.r_emb[i], self.r_bias[i]

                mems_i = None if mems is None else mems[i]
                core_out = layer(core_out, r_emb, self.r_w_bias[i],
                        r_bias, dec_attn_mask=dec_attn_mask, mems=mems_i)
                hids.append(core_out)

        elif self.attn_type == 2: # absolute
            pos_seq = torch.arange(klen - 1, -1, -1.0, device=word_emb.device,
                                   dtype=word_emb.dtype)
            #pos_seq = torch.arange(klen, device=word_emb.device, dtype=word_emb.dtype)
            if self.clamp_len > 0:
                pos_seq.clamp_(max=self.clamp_len)
            pos_emb = self.pos_emb(pos_seq)
            
            core_out = self.drop(word_emb + pos_emb[-qlen:]) 

            hids.append(core_out)
            for i, layer in enumerate(self.layers):
                mems_i = None if mems is None else mems[i]
                if mems_i is not None and i == 0:
                    if len(mems_i.size()) == len(pos_emb.size()):
                        mems_i += pos_emb[:mlen]

                core_out = layer(core_out, dec_attn_mask=dec_attn_mask,
                                 mems=mems_i)
                hids.append(core_out)

        elif self.attn_type in [3, 31, 32, 33]: # lamemo
            pos_seq = torch.arange(klen-1, -1, -1.0, device=word_emb.device, 
                                   dtype=word_emb.dtype)
            if mlen > 0:
                if self.attn_type in [3, 32, 33]:
                    pos_seq_fut = torch.arange(1, mlen+1, device=word_emb.device, 
                                        dtype=word_emb.dtype)
                elif self.attn_type == 31:
                    pos_seq_fut = torch.arange(-1, -mlen-1, -1.0, device=word_emb.device, 
                                        dtype=word_emb.dtype)
                pos_seq = torch.cat([pos_seq, pos_seq_fut])
            if self.clamp_len > 0:
                pos_seq.clamp_(max=self.clamp_len)
            pos_emb = self.pos_emb(pos_seq)

            core_out = self.drop(word_emb)
            pos_emb = self.drop(pos_emb)

            layer_mems_i = None if mems is None else mems[0]
            recur_mems = None if mems is None else mems[1::2]
            denoms = None if mems is None else mems[2::2]
            # new_mems 0-th element store word embeddings
            new_mems = [core_out]
            
            for i, layer in enumerate(self.layers):
                recur_mems_i = None if recur_mems is None else recur_mems[i]
                denoms_i = None if denoms is None else denoms[i]
                ##print(core_out.size())
                #print("denoms_i dtype: {}".format(denoms_i.dtype))
                r_w_bias_future = None
                if self.attn_type in [3, 33]: # learnable position bias
                    r_r_bias_future = self.r_r_bias_future
                elif self.attn_type == 31: # rpe of xl
                    r_r_bias_future = self.r_r_bias
                elif self.attn_type == 32: # learnable content & position bias
                    r_w_bias_future = self.r_w_bias_future
                    r_r_bias_future = self.r_r_bias_future
                
                core_outs = layer(core_out, pos_emb, self.r_w_bias,
                        self.r_r_bias, r_r_bias_future, 
                        r_w_bias_future=r_w_bias_future, dec_attn_mask=dec_attn_mask, 
                        mems=layer_mems_i, recur_mems=recur_mems_i, denom=denoms_i)
                if self.save_analysis:
                    core_out, layer_mems_i, new_recur_mems_i, new_denoms_i, attn_prob_i, attn_prob_mem_i, alpha_i = core_outs
                    save_results.append([attn_prob_i, attn_prob_mem_i, alpha_i])
                else:
                    core_out, layer_mems_i, new_recur_mems_i, new_denoms_i = core_outs
                new_mems.append(new_recur_mems_i)
                new_mems.append(new_denoms_i)
                
                #print("new_denoms_i dtype: {}".format(new_denoms_i.dtype))

            # remap
            hids = new_mems

            core_out = torch.cat([layer_mems_i, core_out], dim=0)


        
        

        

        core_out = self.drop(core_out)

        new_mems = self._update_mems(hids, mems, qlen, mlen, memory_type=self.memory_type)
        #print("after update denom dtype: {}".format(new_mems[2].dtype))
        return core_out, new_mems, save_results

    def forward(self, data, target, *mems, evaluate_first_tgt=False, self_entropy=False, mem_target=None):
        # nn.DataParallel does not allow size(0) tensors to be broadcasted.
        # So, have to initialize size(0) mems inside the model forward.
        # Moreover, have to return new_mems to allow nn.DataParallel to piece
        # them together.
        

        if not mems: 
            mems = self.init_mems(self.memory_type)
        elif mems[0] == None:
            mems = None

        
        
        hidden, new_mems, save_results = self._forward(data, mems=mems)

        if target == None:
            # generate mode
            pred_hid = hidden[-1:]
            logit = self.crit.compute_logit(pred_hid.view(-1, pred_hid.size(-1)))
            return [logit] + new_mems, save_results

        tgt_len = target.size(0)
        pred_hid = hidden[-tgt_len:]
        mem_hid = hidden[:-tgt_len]
        mem_tgt_len = mem_hid.size(0)
        #print(hidden.size())
        #print(mem_target)
        if evaluate_first_tgt:
            pred_hid = pred_hid[0].unsqueeze(0)
            target = target[0].unsqueeze(0)
            tgt_len = 1
        if self.sample_softmax > 0 and self.training:
            assert self.tie_weight
            logit = sample_logits(self.word_emb,
                self.out_layer.bias, target, pred_hid, self.sampler)
            loss = -F.log_softmax(logit, -1)[:, :, 0]
            if mem_target is not None:
                mem_logit = sample_logits(self.word_emb,
                    self.out_layer.bias, mem_target, mem_hid, self.sampler)
                mem_loss = -F.log_softmax(mem_logit, -1)[:, :, 0]
            else:
                mem_loss = torch.zeros_like(loss, dtype=loss.dtype, device=loss.device)
        else:
            if self_entropy:
                logit = self.crit.compute_logit(pred_hid.view(-1, pred_hid.size(-1)))
                prob = F.softmax(logit, dim=-1)
                logprob = F.log_softmax(logit, dim=-1)
                loss = - (prob * logprob).sum(-1).view(tgt_len, -1)
            else:
                loss = self.crit(pred_hid.view(-1, pred_hid.size(-1)), target.view(-1))
                loss = loss.view(tgt_len, -1)
                if mem_target is not None:
                    mem_logit = self.crit(mem_hid.view(-1, mem_hid.size(-1)), mem_target.view(-1))
                    mem_loss = loss.view(mem_tgt_len, -1)
                else:
                    mem_loss = torch.zeros_like(loss, dtype=loss.dtype, device=loss.device)
        
        if self.save_analysis:
            logit = self.crit.compute_logit(pred_hid.view(-1, pred_hid.size(-1)))
            prob = F.softmax(logit, dim=-1).view(target.size(0), target.size(1), -1)
            save_results.append(target)
            #print(prob.size())
            #print(target.size())
            save_results.append(prob.gather(-1, target.unsqueeze(-1)))


        if self.half_loss and self.training:
            loss_mask = torch.ones_like(loss, dtype=loss.dtype, device=loss.device)
            loss_mask[:tgt_len // 2] = 0.0
            # double for the same scale
            loss = loss * loss_mask * 2.0

        if self.training:
            if new_mems is None:
                return [loss, mem_loss]
            else:
                return [loss, mem_loss] + new_mems
        else:
            if new_mems is None:
                return [loss], save_results
            else:
                return [loss] + new_mems, save_results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='unit test')

    parser.add_argument('--n_layer', type=int, default=4, help='')
    parser.add_argument('--n_rel_layer', type=int, default=4, help='')
    parser.add_argument('--n_head', type=int, default=2, help='')
    parser.add_argument('--d_head', type=int, default=2, help='')
    parser.add_argument('--d_model', type=int, default=200, help='')
    parser.add_argument('--d_embed', type=int, default=200, help='')
    parser.add_argument('--d_inner', type=int, default=200, help='')
    parser.add_argument('--dropout', type=float, default=0.0, help='')
    parser.add_argument('--cuda', action='store_true', help='')
    parser.add_argument('--seed', type=int, default=1111, help='')
    parser.add_argument('--multi_gpu', action='store_true', help='')

    args = parser.parse_args()

    device = torch.device("cuda" if args.cuda else "cpu")

    B = 1
    tgt_len, mem_len, ext_len = 36, 36, 0
    data_len = tgt_len * 20
    args.n_token = 10000

    import data_utils

    data = torch.LongTensor(data_len*B).random_(0, args.n_token).to(device)
    diter = data_utils.LMOrderedIterator(data, B, tgt_len, device=device, ext_len=ext_len)

    cutoffs = [args.n_token // 2]
    tie_projs = [False] + [True] * len(cutoffs)

    for div_val in [1, 2]:
        for d_embed in [200, 100]:
            model = MemTransformerLM(args.n_token, args.n_layer, args.n_head,
                            args.d_model, args.d_head, args.d_inner, args.dropout,
                            dropatt=args.dropout, tie_weight=True, 
                            d_embed=d_embed, div_val=div_val, 
                            tie_projs=tie_projs, pre_lnorm=True,
                            tgt_len=tgt_len, ext_len=ext_len, mem_len=mem_len, 
                            cutoffs=cutoffs, attn_type=0).to(device)

            #print(sum(p.numel() for p in model.parameters()))

            mems = tuple()
            for idx, (inp, tgt, seqlen) in enumerate(diter):
                #print('batch {}'.format(idx))
                out = model(inp, tgt, *mems)
                mems = out[1:]

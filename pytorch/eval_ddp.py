# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import math
import os
import pickle
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import data_utils
import utils
from data_utils import get_lm_corpus, Corpus
#from data_utils import tokenize_raw
from utils.exp_utils_ddp import AverageMeter
from utils.exp_utils_ddp import benchmark
from utils.exp_utils_ddp import create_exp_dir

from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description='PyTorch Transformer Language Model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--work_dir', default='LM-TFM', type=str,
                        help='experiment directory')
    parser.add_argument('--seed', type=int, default=1111,
                          help='Random seed')
    parser.add_argument('--debug', action='store_true',
                        help='run in debug mode (do not create exp dir)')
    parser.add_argument('--data', type=str, default='../data/wikitext-103',
                        help='location of the data corpus')
    parser.add_argument('--manual', type=str, default=None, nargs='+',
                        help='run model on raw input data')
    parser.add_argument('--dataset', type=str, default='wt103',
                        choices=['wt103', 'lm1b', 'enwik8', 'text8'],
                        help='dataset name')
    parser.add_argument('--split', type=str, default='all',
                        choices=['all', 'valid', 'test'],
                        help='which split to evaluate')
    parser.add_argument('--type', type=str, default='pytorch',
                        choices=['pytorch', 'torchscript', 'onnx'],
                        help='type of runtime to use')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='batch size')
    parser.add_argument('--tgt_len', type=int, default=64,
                        help='number of tokens to predict')
    parser.add_argument('--ext_len', type=int, default=0,
                        help='length of the extended context')
    parser.add_argument('--mem_len', type=int, default=640,
                        help='length of the retained previous heads')
    parser.add_argument('--clamp_len', type=int, default=-1,
                        help='max positional embedding index')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--model', type=str, default='',
                        help='path to the checkpoint')
    parser.add_argument('--fp16', action='store_true',
                        help='Run training in fp16/mixed precision')
    parser.add_argument('--log_all_ranks', action='store_true',
                        help='Enable logging for all distributed ranks')
    parser.add_argument('--same_length', action='store_true',
                        help='set same length attention with masking')
    parser.add_argument('--target_perplexity', type=float, default=None,
                        help='target perplexity')
    parser.add_argument('--target_throughput', type=float, default=None,
                        help='target throughput')
    parser.add_argument('--save_data', action='store_true',
                        help='save latency and throughput data to a file')
    parser.add_argument('--repeat', type=int, default=1,
                        help='loop over the dataset REPEAT times')
    parser.add_argument('--max_size', type=int, default=None,
                        help='run inference on up to MAX_SIZE batches')
    parser.add_argument('--percentiles', nargs='+', default=[90, 95, 99],
                        help='percentiles for latency confidence intervals')
    parser.add_argument('--save_torchscript', default=None, type=str,
                        help='save torchscript model to a file')
    parser.add_argument('--load_torchscript', default=None, type=str,
                        help='load torchscript model from a file')
    parser.add_argument('--local_rank', default=0, type=int,
                        help='Used for multi-process training. ' +
                        'Can either be manually set ' +
                        'or automatically set by using \'python -m multiproc\'.')
    parser.add_argument('--save_analysis', action='store_true', help='save attention weights and predict logits')
    parser.add_argument('--generate', action='store_true', help='generate tokens in target-length number')
    parser.add_argument('--top_k', type=int, default=0)
    parser.add_argument('--top_p', type=float, default=0.0)
    parser.add_argument('--temp', type=float, default=1.0)
    parser.add_argument
    args = parser.parse_args()
    assert args.ext_len >= 0, 'extended context length must be non-negative'
    return args


def load_checkpoint(path):
    dst = f'cuda:{torch.cuda.current_device()}'
    logging.info(f'Loading checkpoint from {path}')
    checkpoint = torch.load(path, map_location=dst)
    return checkpoint


def format_log(loss, split, args):
    if args.dataset in ['enwik8', 'text8']:
        log_str = '| {0} loss {1:5.2f} | {0} bpc {2:9.5f} '.format(
            split, loss, loss / math.log(2))
    else:
        log_str = '| {0} loss {1:5.2f} | {0} ppl {2:9.3f} '.format(
            split, loss, math.exp(loss))
    return log_str


def evaluate(eval_iter, model, meters, max_size=None, repeat=1):
    total_len, total_loss = 0, 0.
    saved_results = []
    torch.cuda.synchronize()
    start_time = time.time()
    with torch.no_grad():
        mems = tuple()
        for _ in range(repeat):
            for idx, (data, target, seq_len) in enumerate(eval_iter):
                if max_size and idx >= max_size:
                    break
                torch.cuda.synchronize()
                start_iter = time.time()
                ret, saved_batch = model(data, target, *mems)
                torch.cuda.synchronize()
                elapsed = time.time() - start_iter
                loss, mems = ret[0], ret[1:]
                numel = loss.numel()
                loss = loss.mean()
                total_loss += numel * loss.item()
                total_len += numel
                meters['eval_latency'].update(elapsed)
                target_tokens = target.numel()
                throughput = target_tokens / elapsed
                throughput = utils.distributed.all_reduce_item(throughput, op='sum')
                meters['eval_throughput'].update(throughput)
                saved_results.append(saved_batch)

    utils.distributed.barrier()
    torch.cuda.synchronize()
    total_time = time.time() - start_time
    logging.info('Time : {:.2f}s, {:.2f}ms/segment'.format(
            total_time, 1000 * total_time / (idx+1)))

    avg_loss = total_loss / total_len
    avg_loss = utils.distributed.all_reduce_item(avg_loss, op='mean')
    return avg_loss, saved_results

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    # https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits

def generate(eval_iter, model, max_len, top_k=0, top_p=0.9, temp=1.0):
    torch.cuda.synchronize()
    start_time = time.time()
    with torch.no_grad():
        mems = tuple()
        # feed prompt
        for idx, (data, target, seq_len) in enumerate(eval_iter):
            torch.cuda.synchronize()
            start_iter = time.time()
            ret, _ = model(data, target, *mems)
            torch.cuda.synchronize()
            elapsed = time.time() - start_iter
            loss, mems = ret[0], ret[1:]
        
        # generate
        results = []
        data = torch.tensor(target[-1:], dtype=target.dtype, device=target.device)
        #print(target.size())
        for step in tqdm(range(max_len)):
            ret, _ = model(data, None, *mems)
            logit, mems = ret[0], ret[1:]
            logit = logit.squeeze()
            #print(logit.size())
            filtered_logits = top_k_top_p_filtering(logit / temp, top_k=top_k, top_p=top_p)

            # Sample from the filtered distribution
            probabilities = F.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(probabilities, 1)
            results.append(next_token.item())
            data = torch.tensor(next_token.unsqueeze(1), dtype=target.dtype, device=target.device)
            torch.cuda.synchronize()
            elapsed = time.time() - start_iter
        
        logging.info('Time : {:.2f}s'.format(elapsed))

    return results


def compile_model(model, device, args):
    inp = torch.randint(0, 1000, (args.tgt_len, args.batch_size)).to(device)
    tgt = torch.randint(0, 1000, (args.tgt_len, args.batch_size)).to(device)
    start = time.time()
    with torch.no_grad():
        mems = None
        for _ in range(2):
            ret = model(inp, tgt, mems)
            _, mems = ret[0], ret[1:]
    torch.cuda.synchronize()
    stop = time.time()
    logging.info(f'Building the model took {stop - start:.2f} seconds')


def main():
    args = parse_args()

    if args.type == 'pytorch':
        from mem_transformer import MemTransformerLM
    else:
        from inference.mem_transformer_base_jit import MemTransformerLM

    torch.cuda.set_device(args.local_rank)
    device = torch.device('cuda' if args.cuda else 'cpu')
    utils.distributed.init_distributed(args.cuda)

    with utils.distributed.sync_workers() as rank:
        if rank == 0:
            create_exp_dir(args.work_dir, debug=args.debug)

    # Setup logging
    if args.log_all_ranks:
        log_file = f'log_rank_{utils.distributed.get_rank()}.log'
    else:
        log_file = f'log.log'

    log_file = os.path.join(args.work_dir, log_file)
    if args.debug:
        log_file = os.devnull

    utils.exp_utils_ddp.setup_logging(log_all_ranks=args.log_all_ranks,
                                  filename=log_file,
                                  filemode='a',
                                  )
    logging.info(args)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.model:
        model_path = args.model
    elif args.work_dir:
        model_path = os.path.join(args.work_dir, 'checkpoint_best.pt')
    else:
        raise RuntimeError('Specify path to checkpoint using --model or --work_dir')

    checkpoint = load_checkpoint(model_path)

    if args.manual:
        args.batch_size = 1
        vocab = checkpoint['vocab']

        if hasattr(vocab, 'sym2idx') and not hasattr(vocab, 'unk_idx'):
            vocab.unk_idx = vocab.sym2idx['<unk>']

        print(args.manual)
        if os.path.isfile(args.manual[0]):
            text = open(args.manual[0], "r").read()
        else:
            text = " ".join(args.manual)
        tokenized = text#.split()#tokenize_raw(text)
        symbols = vocab.tokenize(tokenized, add_eos=not args.generate)
        print(symbols)
        print("# tokens of input prompt: {}".format(len(symbols)))
        tensor = vocab.convert_to_tensor(symbols)

        iter = data_utils.LMOrderedIterator(tensor, bsz=args.batch_size,
                                            bptt=args.tgt_len, device=device,
                                            ext_len=args.ext_len)
    else:
        # Load dataset
        corpus = get_lm_corpus(args.data, args.dataset)#, checkpoint['args'].vocab)

        if args.split == 'valid':
            iter = corpus.get_iterator('valid', args.batch_size, args.tgt_len,
                                       device=device, ext_len=args.ext_len)
        elif args.split == 'test':
            iter = corpus.get_iterator('test', args.batch_size, args.tgt_len,
                                       device=device, ext_len=args.ext_len)
        else:
            raise RuntimeError('Unknown split')

    if args.fp16:
        dtype = torch.float16
        math_str = 'fp16'
    else:
        dtype = torch.float32
        math_str = 'fp32'

    if args.load_torchscript:
        model = torch.jit.load(args.load_torchscript)

    else:
        checkpoint['model_config']['tgt_len'] = args.tgt_len
        checkpoint['model_config']['ext_len'] = args.ext_len
        checkpoint['model_config']['mem_len'] = args.mem_len
        checkpoint['model_config']['clamp_len'] = args.clamp_len
        checkpoint['model_config']['same_length'] = args.same_length
        checkpoint['model_config']['save_analysis'] = args.save_analysis
        

        model = MemTransformerLM(**checkpoint['model_config'])
        model.load_state_dict(checkpoint['model_state'])

    model = model.eval()
    model = model.to(device)

    model = model.float()
    if args.fp16:
        model = model.half()

    if args.type != 'pytorch':
        compile_model(model, device, args)

    if args.type == 'torchscript' and args.save_torchscript:
        torch.jit.save(model, args.save_torchscript)

    logging.info(f'Evaluating with: math {math_str} type {args.type} '
                 f'bsz {args.batch_size} tgt_len {args.tgt_len} '
                 f'ext_len {args.ext_len} mem_len {args.mem_len} '
                 f'clamp_len {args.clamp_len}')

    meters = {}
    warmup = args.mem_len // args.tgt_len + 1
    meters['eval_throughput'] = AverageMeter(warmup=warmup, keep=args.save_data)
    meters['eval_latency'] = AverageMeter(warmup=warmup, keep=args.save_data)


    if args.generate:
        gen_results = generate(iter, model, args.tgt_len, top_k=args.top_k, top_p=args.top_p, temp=args.temp)
        detok_results = " ".join([vocab.idx2sym[x] for x in gen_results])
        input_name = args.manual[0].split('/')[-1]
        save_dir = os.path.join(args.work_dir, "result-topk{}-topp{}-temp{}-{}".format(args.top_k, args.top_p, args.temp, input_name))
        with open(save_dir, 'w') as f:
            f.write(detok_results)
        print("Write generation results to {}".format(save_dir))

        return 

    loss, saved_results = evaluate(iter, model, meters, args.max_size, args.repeat)
    perplexity = math.exp(loss)
    log_str = format_log(loss, args.split, args)

    logging.info('=' * 100)
    logging.info(log_str)
    logging.info('=' * 100)

    if args.save_analysis:
        save_path = os.path.join(args.work_dir, 'attn_res-{}.bin'.format(args.max_size))
        torch.save(saved_results, save_path)
        logging.info("Save attentions in {}".format(save_path))


    if args.save_data:
        latency_data = np.array(meters['eval_latency'].vals)
        throughput_data = np.array(meters['eval_throughput'].vals)
        precision = 'fp16' if args.fp16 else 'fp32'
        data_fname = f'eval_data_{args.batch_size}_{precision}_{args.type}'
        data_path = os.path.join(args.work_dir, data_fname)
        data = {
            'args': args,
            'throughput': throughput_data,
            'latency': latency_data,
            }
        with open(data_path, 'wb') as f:
            pickle.dump(data, f)
        logging.info(f'Throughput Avg: {throughput_data.mean():.2f} tok/s')
        logging.info(f'Latency Avg: {1000.0 * latency_data.mean():.2f} ms')
        for p in args.percentiles:
            logging.info(f'Latency {p}%: {1000.0 * np.percentile(latency_data, p):.2f} ms')

        logging.info('=' * 100)

    passed = benchmark(target_perplexity=args.target_perplexity,
                       test_perplexity=perplexity,
                       target_throughput=args.target_throughput,
                       test_throughput=meters['eval_throughput'].avg,
                       )
    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()

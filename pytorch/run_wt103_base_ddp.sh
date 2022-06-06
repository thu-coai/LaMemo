if [[ $1 == 'train' ]]; then
    echo 'Run training...'
    CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=$2 --master_port 29501 train_ddp.py \
        --cuda \
        --data ../data/wikitext-103/ \
        --dataset wt103 \
        --adaptive \
        --n_layer 16 \
        --d_model 410 \
        --n_head 10 \
        --d_head 41 \
        --d_inner 2100 \
        --dropout 0.1 \
        --dropatt 0.0 \
        --optim adam \
        --lr 0.00025 \
        --eta_min 0 \
        --roll \
        --warmup_step 0 \
        --max_step 250000 \
        --tgt_len 150 \
        --mem_len 150 \
        --eval_tgt_len 150 \
        --batch_size 32 \
        --eval_batch_size 16 \
        --batch_chunk 1 \
        --multi_gpu ddp \
        --log_interval 10 \
        --eval_interval 5000 \
        --fp16 "O2" \
        --attn_type 3 \
	    --work_dir ../models/lamemo-base-wt103-fp16-O2 \
        ${@:3}
elif [[ $1 == 'eval' ]]; then
    echo 'Run evaluation...'
    CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=$2 --master_port 29550 eval_ddp.py \
        --cuda \
        --data ../data/wikitext-103/ \
        --dataset wt103 \
        --tgt_len 150 \
        --mem_len 150 \
        --clamp_len -1 \
	    --batch_size 16 \
        --split test \
        ${@:3}
elif [[ $1 == 'gen' ]]; then
    echo 'Run generation...'
    CUDA_VISIBLE_DEVICES=7 python -m torch.distributed.launch --nproc_per_node=$2 --master_port 29554 eval_ddp.py \
        --cuda \
        --data ../data/wikitext-103/ \
        --dataset wt103 \
        --tgt_len 256 \
        --mem_len 512 \
        --clamp_len 400 \
        --seed 42 \
	    --manual ../data/wt103-prompt2.txt \
        --generate \
        --top_k 0 \
        --top_p 0.95 \
        --temp 1.0 \
        ${@:3}
else
    echo 'unknown argment 1'
fi

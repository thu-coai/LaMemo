if [[ $1 == 'train' ]]; then
    echo 'Run training...'
    CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=$2 --master_port 29509 train_ddp.py \
        --cuda \
        --data ../data/enwik8/ \
        --dataset enwik8 \
        --n_layer 12 \
        --d_model 512 \
        --n_head 8 \
        --d_head 64 \
        --d_inner 2048 \
        --dropout 0.1 \
        --dropatt 0.0 \
        --optim adam \
        --lr 0.00025 \
        --eta_min 0 \
        --roll \
        --warmup_step 0 \
        --max_step 50000 \
        --tgt_len 512 \
        --mem_len 512 \
        --eval_tgt_len 512 \
        --batch_size 20 \
        --eval_batch_size 16 \
        --batch_chunk 1 \
        --multi_gpu ddp \
        --log_interval 10 \
        --eval_interval 5000 \
        --fp16 "O2" \
        --attn_type 3 \
	    --work_dir ../models/lamemo-base-enwik8-fp16-O2 \
        ${@:3}
elif [[ $1 == 'eval' ]]; then
    echo 'Run evaluation...'
    CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch --nproc_per_node=$2 --master_port 29550 eval_ddp.py \
        --cuda \
        --data ../data/enwik8/ \
        --dataset enwik8 \
        --tgt_len 512 \
        --mem_len 512 \
        --clamp_len -1 \
	    --batch_size 16 \
        --split test \
        ${@:3}
else
    echo 'unknown argment 1'
fi
#! /bin/bash

# Change for multinode config
MP_SIZE=1

NUM_WORKERS=4
NUM_GPUS_PER_WORKER=8

DATA_PATH=/cfs/aidenliang/corpus/wiki_data/db/wiki_and_book_text_sentence
CHECKPOINT_PATH=/cfs/aidenliang/checkpoints/ds_megatron_grad_acc_test

script_path=$(realpath $0)
script_dir=$(dirname $script_path)
JOB_NAME=DS_ACC

config_json="$script_dir/ds_zero2_grad_acc_64k_config.json"
hostfile="$script_dir/myhostfile"
bert_options=" \
       --model-parallel-size ${MP_SIZE} \
       --num-layers 24 \
       --hidden-size 1024 \
       --num-attention-heads 16 \
       --batch-size 16 \
       --seq-length 512 \
       --max-position-embeddings 512 \
       --train-iters 7500 \
       --save ${CHECKPOINT_PATH} \
       --load ${CHECKPOINT_PATH} \
       --data-path ${DATA_PATH} \
       --vocab-file bert-vocab.txt \
       --data-impl mmap \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 2e-3 \
       --lr-decay-style linear \
       --min-lr 1.0e-4 \
       --lr-decay-iters 7500 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --warmup .01 \
       --log-interval 10 \
       --save-interval 50 \
       --eval-interval 50 \
       --eval-iters 10 \
       --fp16 \
"
bert_options="${bert_options}
               --deepspeed \
               --deepspeed_config ${config_json} \

"


run_cmd="deepspeed --num_nodes ${NUM_WORKERS} --num_gpus ${NUM_GPUS_PER_WORKER} --hostfile=${hostfile} pretrain_bert.py $@ ${bert_options} &> ${JOB_NAME}.log"
echo ${run_cmd}
eval ${run_cmd}

set +x

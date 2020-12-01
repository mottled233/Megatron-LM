#! /bin/bash

# Change for multinode config
MP_SIZE=1

NUM_WORKERS=1
NUM_GPUS_PER_WORKER=8

DATA_PATH=/cfs/aidenliang/corpus/squad
CHECKPOINT_PATH=/cfs/aidenliang/checkpoints/ds_mega_squad
OUTPUT_PATH=/cfs/aidenliang/output/squad
PRETRAINED_PATH=/cfs/aidenliang/checkpoints/ds_megatron_save2

script_path=$(realpath $0)
script_dir=$(dirname $script_path)
JOB_NAME=DS_SQUAD

config_json="$script_dir/ds_squad.json"
hostfile="$script_dir/myhostfile"
bert_options=" \
       --model-parallel-size ${MP_SIZE} \
       --num-layers 24 \
       --hidden-size 1024 \
       --num-attention-heads 16 \
       --batch-size 6 \
       --seq-length 512 \
       --max-position-embeddings 512 \
       --save ${CHECKPOINT_PATH} \
       --load ${CHECKPOINT_PATH} \
       --data-dir ${DATA_PATH} \
       --vocab-file bert-vocab.txt \
       --tokenizer-type BertWordPieceLowerCase\
       --distributed-backend nccl \
       --lr 5e-6 \
       --lr-decay-style linear \
       --min-lr 1.0e-6 \
       --weight-decay 0 \
       --clip-grad 1.0 \
       --warmup .0 \
       --log-interval 100 \
       --save-interval 1000 \
       --eval-interval 500 \
       --fp16 \
       --finetune \
       --attention-dropout 0.1 \
       --hidden-dropout 0.1 \
       --doc-stride 128 \
       --test-batch-size 6 \
       --epochs 3 \
       --pretrained-checkpoint  $PRETRAINED_PATH\
       --train-data train-v1.1.json \
       --valid-data dev-v1.1.json \
       --output-dir $OUTPUT_PATH \
"
bert_options="${bert_options}
               --deepspeed \
               --deepspeed_config ${config_json} \

"


run_cmd="deepspeed --num_nodes ${NUM_WORKERS} --num_gpus ${NUM_GPUS_PER_WORKER} --hostfile=${hostfile} tasks/squad_main.py $@ ${bert_options} &> ${JOB_NAME}.log"
echo ${run_cmd}
eval ${run_cmd}

set +x

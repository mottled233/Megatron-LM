#!/bin/bash

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DATA_PATH=/cfs/aidenliang//corpus/squad
CHECKPOINT_PATH=/cfs/aidenliang/checkpoints/ds_megatron_save
OUTPUT_PATH=/cfs/aidenliang/output/squad

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       tasks/squad_main.py \
       --model-parallel-size 1 \
       --num-layers 24 \
       --hidden-size 1024 \
       --num-attention-heads 16 \
       --batch-size 4 \
       --seq-length 512 \
       --max-position-embeddings 512 \
       --vocab-file bert-vocab.txt \
       --distributed-backend nccl \
       --lr 0.0001 \
       --lr-decay-style linear \
       --min-lr 1.0e-5 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --warmup .1 \
       --log-interval 100 \
       --save-interval 1000 \
       --eval-interval 1000 \
       --fp16 \
       --data-dir $DATA_PATH \
       --epochs 20 \
       --pretrained-checkpoint  $CHECKPOINT_PATH\
       --train-data train-v1.1.json \
       --valid-data dev-v1.1.json \
       --output-dir $OUTPUT_PATH \




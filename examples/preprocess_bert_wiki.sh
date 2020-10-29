#!/bin/bash

VOCAB_FILE=bert-vocab.txt
INPUT_DIR=
OUTPUT_PREFIX=
NUM_OF_WORKER=16

python tools\preprocess_data.py \
       --input $INPUT_DIR \
       --json-keys text \
       --split-sentences \
       --tokenizer-type BertWordPieceCase \
       --vocab-file $VOCAB_FILE \
       --output-prefix $OUTPUT_PREFIX \
       --workers $NUM_OF_WORKER

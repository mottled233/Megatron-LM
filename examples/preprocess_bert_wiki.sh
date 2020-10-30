#!/bin/bash

VOCAB_FILE=bert-vocab.txt
INPUT_DIR=/cfs/aidenliang/corpus/wiki_data/text/AA
OUTPUT_PREFIX=/cfs/aidenliang/corpus/wiki_data/db/wiki
NUM_OF_WORKER=16

python tools/preprocess_data.py \
       --input $INPUT_DIR \
       --json-keys text \
       --split-sentences \
       --tokenizer-type BertWordPieceLowerCase \
       --vocab-file $VOCAB_FILE \
       --output-prefix $OUTPUT_PREFIX \
       --workers $NUM_OF_WORKER

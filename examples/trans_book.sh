#! /bin/bash

NUM_WORKERS=4

DATA_PATH=/cfs/aidenliang/corpus/books1/epubtxt
OUT_PATH=/cfs/aidenliang/corpus/books1/testout

options=" \
       --input ${DATA_PATH} \
       --output-dir ${OUT_PATH} \
       --split-by sentence \
       --remove-empty-lines \
       --num_of_workers ${NUM_WORKERS} \
"

run_cmd="python trans_book_corpus.py $@ ${options}"
echo ${run_cmd}
eval ${run_cmd}

set +x

#! /bin/bash

NUM_WORKERS=16

DATA_PATH=/cfs/aidenliang/corpus/books3/the-eye.eu/public/Books/Bibliotik
OUT_PATH=/cfs/aidenliang/corpus/books3/fine_splitted

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

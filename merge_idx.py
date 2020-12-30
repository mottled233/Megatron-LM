import argparse
import json
import multiprocessing
import os
import shutil
import sys
from functools import partial
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
import threading
import time

import torch
try:
    import nltk
    nltk_available = True
except ImportError:
    nltk_available = False

from megatron.tokenizer import build_tokenizer
from megatron.data import indexed_dataset
from tqdm import tqdm
import tools.preprocess_data as preprocess


def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='input data')
    group.add_argument('--input', type=str, required=True,)

    group.add_argument('--output', type=str, required=True,
                       help='Prefix to binary output file')

    group.add_argument('--dataset-impl', type=str, default="mmap",
                       help='Prefix to binary output file')

    args = parser.parse_args()
    args.rank = 0
    args.vocab_file = "bert-vocab.txt"
    args.tokenizer_type = "BertWordPieceLowerCase"
    args.make_vocab_size_divisible_by = 128
    args.model_parallel_size = 1
    return args


def main():
    args = get_args()
    args.tokenizer_type = "BertWordPieceLowerCase"

    tokenizer = build_tokenizer(args)

    prefixes = args.input.split("@")
    assert len(prefixes) >= 2, "Need at least two index file to merge."

    output_idx_file = f"{args.output}.idx"
    output_bin_file = f"{args.output}.bin"

    builder = indexed_dataset.make_builder(output_bin_file, impl=args.dataset_impl, vocab_size=tokenizer.vocab_size)

    for data_path in tqdm(prefixes):
        builder.merge_file_(another_file=data_path)

    print("Finished merge, start to finalize.")
    builder.finalize(output_idx_file)
    print("Finished.")


if __name__ == "__main__":
    main()
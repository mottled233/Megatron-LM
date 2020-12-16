# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Processing data for pretraining."""

import argparse
import json
import multiprocessing
import os
import sys
from functools import partial
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
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


# https://stackoverflow.com/questions/33139531/preserve-empty-lines-with-nltks-punkt-tokenizer
class CustomLanguageVars(nltk.tokenize.punkt.PunktLanguageVars):

    _period_context_fmt = r"""
        \S*                          # some word material
        %(SentEndChars)s             # a potential sentence ending
        \s*                       #  <-- THIS is what I changed
        (?=(?P<after_tok>
            %(NonWord)s              # either other punctuation
            |
            (?P<next_tok>\S+)     #  <-- Normally you would have \s+ here
        ))"""

class IdentitySplitter(object):
    def tokenize(self, *text):
        return text

class Encoder(object):
    def __init__(self, args):
        self.args = args

    def initializer(self):
        # Use Encoder class as a container for global data
        Encoder.tokenizer = build_tokenizer(self.args)
        if self.args.split_sentences:
            if not nltk_available:
                print("NLTK is not available to split sentences.")
                exit()
            splitter = nltk.load("tokenizers/punkt/english.pickle")
            if self.args.keep_newlines:
                # this prevents punkt from eating newlines after sentences
                Encoder.splitter = nltk.tokenize.punkt.PunktSentenceTokenizer(
                    train_text = splitter._params,
                    lang_vars = CustomLanguageVars())
            else:
                Encoder.splitter = splitter

        else:
            Encoder.splitter = IdentitySplitter()

    def encode(self, json_line):
        data = json.loads(json_line)
        ids = {}
        for key in self.args.json_keys:
            text = data[key]
            doc_ids = []
            for sentence in Encoder.splitter.tokenize(text):
                sentence_ids = Encoder.tokenizer.tokenize(sentence)
                if len(sentence_ids) > 0:
                    doc_ids.append(sentence_ids)
            if self.args.append_eod:
                doc_ids[-1].append(Encoder.tokenizer.eod)
            ids[key] = doc_ids
        return ids, len(json_line)

def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='input data')
    group.add_argument('--input', type=str, required=True,
                       help='Path to input JSON')
    group.add_argument('--json-keys', nargs='+', default=['text'],
                       help='space separate listed of keys to extract from json')
    group.add_argument('--split-sentences', action='store_true',
                       help='Split documents into sentences.')
    group.add_argument('--skip-encode', action='store_true',
                       help='Skip the encode stage and use cache file instead.')
    group.add_argument('--keep-newlines', action='store_true',
                       help='Keep newlines between sentences when splitting.')

    group = parser.add_argument_group(title='tokenizer')
    group.add_argument('--tokenizer-type', type=str, required=True,
                       choices=['BertWordPieceLowerCase','BertWordPieceCase',
                                'GPT2BPETokenizer'],
                       help='What type of tokenizer to use.')
    group.add_argument('--vocab-file', type=str, default=None,
                       help='Path to the vocab file')
    group.add_argument('--merge-file', type=str, default=None,
                       help='Path to the BPE merge file (if necessary).')
    group.add_argument('--append-eod', action='store_true',
                       help='Append an <eod> token to the end of a document.')
    group.add_argument('--cache-dir', type=str, default=None,
                       help='Cache the tokenized doc in case the corpus too big to load into memory.')
    group.add_argument('--cache-size', type=int, default=1000000,
                       help='Number of document per cache file store')

    group = parser.add_argument_group(title='output data')
    group.add_argument('--output-dir', type=str, required=True,
                       help='Path to binary output file')
    group.add_argument('--output-name-prefix', type=str, required=True,
                       help='Filename of binary output file without suffix and path')
    group.add_argument('--dataset-impl', type=str, default='mmap',
                       choices=['lazy', 'cached', 'mmap'])

    group = parser.add_argument_group(title='runtime')
    group.add_argument('--workers', type=int, default=1,
                       help='Number of worker processes to launch')
    group.add_argument('--doc-of-workers', type=int, default=25,
                       help='Number of document per worker to processes')
    group.add_argument('--log-interval', type=int, default=10000,
                       help='Interval between progress updates')
    args = parser.parse_args()
    args.keep_empty = False

    if args.tokenizer_type.lower().startswith('bert'):
        if not args.split_sentences:
            print("Bert tokenizer detected, are you sure you don't want to split sentences?")

    # some default/dummy values for the tokenizer
    args.rank = 0
    args.make_vocab_size_divisible_by = 128
    args.model_parallel_size = 1

    return args


def get_dir_cnt(dir_path):
    global lock
    if lock is not None:
        lock.acquire()

    if not os.path.exists(os.path.join(dir_path, "count")):
        postfix = 0
    else:
        with open(os.path.join(dir_path, "count"), "r") as cnt:
            postfix = int(cnt.read())

    if lock is not None:
        lock.release()

    return postfix


def add_dir_cnt(dir_path, cnt=1):
    global lock
    if lock is not None:
        lock.acquire()
    postfix = get_dir_cnt(dir_path)
    with open(os.path.join(dir_path, "count"), "w") as cnt_file:
        cnt_file.write(f"{postfix + cnt}")
    if lock is not None:
        lock.release()


def cache_docs(docs, cache_dir):
    postfix = get_dir_cnt(cache_dir)

    with open(os.path.join(cache_dir, f"doc_{postfix}"), 'w') as f:
        json.dump({"docs": docs}, f)

    add_dir_cnt(cache_dir)


def encode_doc_generator(encoded_docs, cache_dir=None, filename=None):
    if cache_dir and not filename:
        postfix = get_dir_cnt(cache_dir)
        for i in range(postfix):
            with open(os.path.join(cache_dir, f"doc_{i}")) as f:
                docs = json.load(f)["docs"]
            for doc in docs:
                yield doc[0], int(doc[1])
    elif cache_dir and filename:
        with open(os.path.join(cache_dir, filename)) as f:
            docs = json.load(f)["docs"]
        for doc in docs:
            yield doc[0], int(doc[1])
    else:
        for doc in encoded_docs:
            yield doc[0], doc[1]


def doc_encode(args, tokenizer):
    encoder = Encoder(args)
    startup_start = time.time()
    pool = multiprocessing.Pool(args.workers, initializer=encoder.initializer)
    encoded_docs = []
    inputs = args.input.split("@")

    buff_size = args.doc_of_workers * args.workers
    total_doc_num = 0
    buff_docs = []
    buff_file_num = 0
    proc_start = time.time()
    for input_dir in inputs:
        for parent, dirnames, filenames in os.walk(input_dir):
            for filename in filenames:
                current = os.path.join(parent, filename)
                print("Opening", current)
                fin = open(current, 'r', encoding='utf-8')
                buff_docs.extend(fin.readlines())
                fin.close()
                buff_file_num += 1

                if len(buff_docs) >= buff_size:
                    encoded_docs.extend(pool.imap(encoder.encode, buff_docs, args.doc_of_workers))
                    time_per_file = (time.time()-proc_start)/buff_file_num
                    total_doc_num += len(buff_docs)
                    print(f"Finished {buff_file_num} files, total {total_doc_num} docs, use time per file:{time_per_file}")

                    if args.cache_dir and len(encoded_docs) >= args.cache_size:
                        cache_docs(encoded_docs, args.cache_dir)
                        print(f"cached dir....")
                        del encoded_docs
                        encoded_docs = []
                    proc_start = time.time()
                    buff_file_num = 0
                    del buff_docs
                    buff_docs = []

    if buff_docs:
        encoded_docs.extend(pool.imap(encoder.encode, buff_docs, args.doc_of_workers))
        time_per_file = (time.time() - proc_start) / buff_file_num
        print(f"Finished {buff_file_num} files , use time per file:{time_per_file}")
        print(encoded_docs[0])
    if args.cache_dir and len(encoded_docs) >= 0:
        cache_docs(encoded_docs, args.cache_dir)
        print(f"cached dir....")
        encoded_docs = []

    pool.close()
    return encoded_docs


def database_init(args, tokenizer, local_lock):
    global lock, builders, output_idx_files, output_bin_files, file_id
    lock = local_lock
    builders = {}
    output_bin_files = {}
    output_idx_files = {}
    file_id = get_dir_cnt(args.output_dir)
    for key in args.json_keys:
        output_bin_files[f"{key}_{file_id}"] = os.path.join(args.output_dir, "{}_{}_{}.bin".format(args.output_name_prefix,
                                                                                                   key, file_id))
        output_idx_files[f"{key}_{file_id}"] = os.path.join(args.output_dir, "{}_{}_{}.idx".format(args.output_name_prefix,
                                                                                                   key, file_id))
        builders[f"{key}_{file_id}"] = indexed_dataset.make_builder(output_bin_files[f"{key}_{file_id}"],
                                                                    impl=args.dataset_impl,
                                                                    vocab_size=tokenizer.vocab_size)


def parallel_dataset_builder(cache_file, cache_dir, json_keys=("text", )):
    global builders, output_idx_files, file_id
    for doc, bytes_processed in encode_doc_generator([], cache_dir, cache_file):
        for key, sentences in doc.items():
            for sentence in sentences:
                builders[f"{key}_{file_id}"].add_item(torch.IntTensor(sentence))
            builders[f"{key}_{file_id}"].end_document()
    for key in json_keys:
        builders[f"{key}_{file_id}"].finalize(output_idx_files[f"{key}_{file_id}"])
    return 1


def main():
    args = get_args()
    #
    # # print("Opening", args.input)
    # # fin = open(args.input, 'r', encoding='utf-8')
    # print("setup...")
    # # if nltk_available and args.split_sentences:
    # #     nltk.download("punkt", download_dir="./", quiet=True)
    #

    tokenizer = build_tokenizer(args)
    print("initializing process pool...")

    if not args.skip_encode:
        encoded_docs = doc_encode(args, tokenizer)

    # level = "document"
    # if args.split_sentences:
    #     level = "sentence"

    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Output prefix: {os.path.join(args.output_dir, args.output_name_prefix)}")

    # Cache file level parallel
    cache_cnt = get_dir_cnt(args.cache_dir)
    cache_files = [f"doc_{cache_id}" for cache_id in range(1, cache_cnt + 1)]

    dataset_builder = partial(parallel_dataset_builder, cache_dir=args.cache_dir, json_keys=args.json_keys)

    if args.workers > 1:
        global_lock = multiprocessing.Lock()
        pool = multiprocessing.Pool(processes=args.workers, initializer=database_init, initargs=(args,
                                                                                                      tokenizer,
                                                                                                      global_lock))

        for _ in tqdm(pool.imap(dataset_builder, cache_files)):
            pass

    else:
        database_init(args, tokenizer, None)
        for file in tqdm(cache_files):
            dataset_builder(file)

    print("Finished data preprocess.")


    # output_bin_files = {}
    # output_idx_files = {}
    # builders = {}
    # for key in args.json_keys:
    #     output_bin_files[key] = "{}_{}_{}.bin".format(args.output_name_prefix,
    #                                                   key, level)
    #     output_idx_files[key] = "{}_{}_{}.idx".format(args.output_prefix,
    #                                                   key, level)
    #     builders[key] = indexed_dataset.make_builder(output_bin_files[key],
    #                                            impl=args.dataset_impl,
    #                                            vocab_size=tokenizer.vocab_size)
    #
    # proc_start = time.time()
    # total_bytes_processed = 0
    # log_cnt = 0
    # for i, (doc, bytes_processed) in enumerate(encode_doc_generator(encoded_docs, args.cache_dir), start=1):
    #     total_bytes_processed += bytes_processed
    #     for key, sentences in doc.items():
    #         for sentence in sentences:
    #             builders[key].add_item(torch.IntTensor(sentence))
    #         builders[key].end_document()
    #     log_cnt += 1
    #     if i % args.log_interval == 0:
    #         current = time.time()
    #         elapsed = current - proc_start
    #         mbs = total_bytes_processed/elapsed/1024/1024
    #         print(f"Processed {i} documents",
    #               f"({log_cnt/elapsed} docs/s, {mbs} MB/s).",
    #               file=sys.stderr)
    #         log_cnt = 0
    #         proc_start = time.time()
    #         total_bytes_processed = 0
    #
    # for key in args.json_keys:
    #     builders[key].finalize(output_idx_files[key])


if __name__ == '__main__':
    lock = None
    builders = None
    file_id = None
    output_bin_files = None
    output_idx_files = None
    main()

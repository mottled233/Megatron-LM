import json
import os
import time
import argparse
from tqdm import tqdm
import nltk
import re
import multiprocessing
from functools import partial
from tools.preprocess_data import CustomLanguageVars
from trans_book_corpus import whitespace_tokenize, para_splitter


def parse_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='input data')
    group.add_argument('--input', type=str, required=True,
                       help='Path to input wiki corpus')
    group.add_argument('--output', type=str, required=True,
                       help='Path to output wiki corpus')
    group.add_argument('--json-key', type=str, default="text")
    group.add_argument('--split-by', type=str, default="sentence",
                       choices=['sentence', 'paragraph'])
    group.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    return args


def parse_file(file_name, parent, args, splitter):
    docs = []
    with open(os.path.join(parent, file_name), 'r', encoding='utf-8') as f1:
        for json_line in f1:
            text = json.loads(json_line)[args.json_key]
            if args.split_by == "paragraph":
                text = re.sub("\n+", "\n", text)
            else:
                text = text.replace("\n", "")
                text = "\n".join(splitter(text))
            docs.append(text)
    return docs


def main():
    args = parse_args()

    if args.workers > 1:
        pool = multiprocessing.Pool(args.workers)

    docs = []

    if args.split_by == "paragraph":
        splitter = para_splitter
    elif args.split_by == "sentence":
        splitter = nltk.load("tokenizers/punkt/english.pickle")
        # this prevents punkt from eating newlines after sentences
        splitter = nltk.tokenize.punkt.PunktSentenceTokenizer(
                    train_text=splitter._params,
                    lang_vars=CustomLanguageVars()).tokenize

    for parent, dirnames, filenames in tqdm(os.walk(args.input)):
        print("enter " + parent)
        if args.workers == 1:
            for filename in tqdm(filenames):
                docs.extend(parse_file(filename, parent, args, splitter))
        else:
            parse_file_func = partial(parse_file, parent=parent, args=args, splitter=splitter)
            docs.extend(pool.imap(parse_file_func, filenames))

    print("Starting write")
    buff = []
    with open(os.path.join(args.output), 'w', encoding='utf-8') as f:
        for i, doc in tqdm(enumerate(docs)):
            buff.append(doc + "\n\n")
            i += 1
            if i % 10000 == 0 or i >= len(docs):
                f.write("".join(buff))
                buff = []


if __name__ == "__main__":
    main()

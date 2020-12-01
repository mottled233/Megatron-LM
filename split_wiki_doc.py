import json
import os
import time
import argparse
from tqdm import tqdm
import nltk
import multiprocessing
from functools import partial
from tools.preprocess_data import CustomLanguageVars
from trans_book_corpus import whitespace_tokenize, para_splitter


def parse_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='input data')
    group.add_argument('--input', type=str, required=True,
                       help='Path to input wiki corpus')
    group.add_argument('--output-dir', type=str, required=True,
                       help='Path to output wiki corpus')
    group.add_argument('--json-key', type=str, default="text")
    group.add_argument('--max-seq-len', type=int, default=500)
    group.add_argument('--num-of-workers', type=int, default=1)

    group.add_argument('--split-by', type=str, default="sentence",
                       choices=['sentence', 'paragraph'])
    group.add_argument('--keep-last', action="store_true",
                       help="Keep the last sub-doc of document. It will be short than max-seq-len.")

    args = parser.parse_args()
    return args


def process_wiki(filename, parent_dir, args, splitter, dir_par):
    current = os.path.join(parent_dir, filename)
    buff = []
    with open(current, 'r', encoding='utf-8') as f1:
        lines = f1.readlines()

    for json_line in lines:
        doc = json.loads(json_line)
        doc_text = doc[args.json_key]
        doc_lines = splitter(doc_text)
        if args.split_by == "paragraph":
            doc_lines = [doc_line + "\n" for doc_line in doc_lines]

        sub_file = ""
        wd_count = 0

        for line in doc_lines:
            sub_file += line
            wd_count += len(whitespace_tokenize(line))
            if wd_count >= args.max_seq_len:
                json_data = {args.json_key: sub_file}
                buff.append(json.dumps(json_data, ensure_ascii=False))
                sub_file = ""
                wd_count = 0
        if sub_file != "" and args.keep_last:
            json_data = {args.json_key: sub_file}
            buff.append(json.dumps(json_data, ensure_ascii=False))
    with open(os.path.join(args.output_dir, dir_par, f"{filename}.json"), 'w', encoding='utf-8') as out_f:
        out_f.write("\n".join(buff))

    return 1


def main():
    args = parse_args()

    if args.split_by == "paragraph":
        splitter = para_splitter
    elif args.split_by == "sentence":
        splitter = nltk.load("tokenizers/punkt/english.pickle")
        # this prevents punkt from eating newlines after sentences
        splitter = nltk.tokenize.punkt.PunktSentenceTokenizer(
                    train_text=splitter._params,
                    lang_vars=CustomLanguageVars()).tokenize
    else:
        splitter = None

    if args.num_of_workers > 1:
        pool = multiprocessing.Pool(args.num_of_workers)

    for parent, dirnames, filenames in tqdm(os.walk(args.input)):
        dir_par = parent.split("/")[-1]
        if dir_par == "":
            dir_par = parent.split("/")[-2]
        if not os.path.isdir(os.path.join(args.output_dir, dir_par)):
            os.mkdir(os.path.join(args.output_dir, dir_par))

        parser = partial(process_wiki, parent_dir=parent, args=args, splitter=splitter, dir_par=dir_par)
        if args.num_of_workers > 1:
            for _ in tqdm(pool.imap(parser, filenames)):
                pass
        else:
            for file in tqdm(filenames):
                parser(file)


if __name__ == "__main__":
    main()

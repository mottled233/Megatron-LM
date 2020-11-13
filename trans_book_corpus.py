import json
import os
import argparse
from tqdm import tqdm
import nltk
import multiprocessing
from functools import partial
from tools.preprocess_data import CustomLanguageVars


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


def parse_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='input data')
    group.add_argument('--input', type=str, required=True,
                       help='Path to input book corpus')
    group.add_argument('--output-dir', type=str, required=True,
                       help='Path to output book corpus')
    group.add_argument('--remove-lines', type=int, default=60,
                       help='Skip the first n lines')
    group.add_argument('--split-by', type=str, default="sentence",
                       choices=['sentence', 'paragraph'])
    group.add_argument('--remove-empty-lines', action='store_true')
    group.add_argument('--json-key', type=str, default="text")
    group.add_argument('--max-seq-len', type=int, default=500)
    group.add_argument('--num_of_workers', type=int, default=1)
    args = parser.parse_args()
    return args


def para_splitter(text):
    def no_blank(string):
        string = string.replace("\n", "")
        return len(string) > 0
    return filter(no_blank, text.split("\n"))


def process_book(filename, parent_dir, args, splitter):
    current = os.path.join(parent_dir, filename)
    buff = []
    with open(current, 'r', encoding='utf-8') as f1:
        lines = f1.readlines()
    file = ""
    sub_file = ""

    for idx, line in enumerate(lines):
        if idx < args.remove_lines:
            continue
        if args.remove_empty_lines and line == "\n":
            continue
        file += line
    file = splitter(file)
    if args.split_by == "paragraph":
        file = [line + "\n" for line in file]

    wd_count = 0
    for line in file:
        sub_file += line
        wd_count += len(whitespace_tokenize(line))
        if wd_count >= args.max_seq_len:
            json_data = {args.json_key: sub_file}
            buff.append(json.dumps(json_data))
            sub_file = ""
            wd_count = 0
    with open(f"{args.output_dir}/books_{filename}.json", 'w', encoding='utf-8') as out_f:
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
        parser = partial(process_book, parent_dir=parent, args=args, splitter=splitter)
        if args.num_of_workers > 1:
            for _ in tqdm(pool.imap(parser, filenames, 16)):
                pass
        else:
            for file in tqdm(filenames):
                parser(file)


if __name__ == "__main__":
    main()

import json
import os
import re
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
    group.add_argument('--num_of_workers', type=int, default=1)


    args = parser.parse_args()
    return args


code_pattern = [" = ", ".(", "();", " + ", " - ", "=\"", "http", "://", ".html", ".com"]
image_header = ["[graph", "(Image", "(Photo"]


def code_line_filter(line):
    match_cnt = 0
    for pattern in code_pattern:
        match_cnt += 0 if line.find(pattern) == -1 else 1
    if match_cnt >= 2:
        return False
    return True


def filter_lines(line):
    if re.match(r"^\s*$", line):
        return False
    line_len = len(whitespace_tokenize(line))
    if line_len <= 5 and line.startswith("Advertise"):
        return False

    for header in image_header:
        if line.startswith(header):
            return False

    if not code_line_filter(line):
        return False


def read_open_web_doc(file_lines):
    docs = []
    lines = []
    for line in file_lines:
        # Head of a document.
        header_match = re.search(r"\d{5,}-[\d\w]{10,}\.txt\s+(\d+ )+\s+(ustar)*.+?(\d+ )+", line)
        if header_match:
            st, ed = header_match.span()

            if st != 0:
                lines.append(line[:st].strip())

            docs.append(lines)
            lines = []

            if ed != len(line):
                lines.append(line[ed:].strip())

        if not re.match("^\s*$", line):
            lines.append(line)
    return docs


def list_in_doc(doc_lines):
    short_cnt = 0
    for line in doc_lines:
        if len(whitespace_tokenize(line)) <= 10:
            short_cnt += 1
            if short_cnt > 10:
                return True
        else:
            short_cnt = 0
    return True


def process_doc(filename, parent_dir, args, splitter):
    current = os.path.join(parent_dir, filename)
    buff = []
    with open(current, 'r', encoding='utf-8') as f1:
        lines = f1.readlines()

    docs = read_open_web_doc(lines)

    for doc in docs:
        filter_doc(doc)
        doc_text = "\n".join(doc)
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
        if sub_file != "":
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





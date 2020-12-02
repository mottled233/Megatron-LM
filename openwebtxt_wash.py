import json
import os
import re
import argparse
from tqdm import tqdm
import multiprocessing
from functools import partial
# from trans_book_corpus import whitespace_tokenize
#

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
                       help='Path to input wiki corpus')
    group.add_argument('--output-dir', type=str, required=True,
                       help='Path to output wiki corpus')
    group.add_argument('--json-key', type=str, default="text")
    group.add_argument('--num_of_workers', type=int, default=1)

    args = parser.parse_args()
    return args


code_pattern = [" = ", ".(", "();", " + ", " - ", "=\"", "http", "://", ".html", ".com"]
image_header = ["[graph", "(Image", "(Photo", "Top\n", "| ", "["]


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
    return True


def add_line_to_doc(doc: list, line: str):
    if not re.match(r"^\s*$", line) and filter_lines(line):
        doc.append(line)


def read_open_web_doc(file_lines):
    docs = []
    lines = []
    for line in file_lines:
        line = re.sub("\x00+", " ", line)
        # Head of a document.
        header_match = re.search(r"\d{5,}-[\d\w]{10,}\.txt[\d+ ]+(ustar)*.+?[\d ]+", line)
        if header_match:
            st, ed = header_match.span()

            if st != 0:
                add_line_to_doc(lines, line[:st].strip())
            if lines:
                docs.append(lines)
                lines = []

            if ed != len(line):
                add_line_to_doc(lines, line[ed:].strip())

        else:
            add_line_to_doc(lines, line)
    return docs


def list_in_doc(doc_lines):
    short_cnt = 0
    have_list = False
    long_cnt = 0
    for line in doc_lines:
        line_len = len(whitespace_tokenize(line))
        if line_len <= 10:
            short_cnt += 1
            if short_cnt > 10:
                have_list = True
        else:
            short_cnt = 0
            if line_len >= 25:
                long_cnt += 1
    if have_list and long_cnt <= 10:
        return True
    return False


def process_doc(filename, parent_dir, args):
    current = os.path.join(parent_dir, filename)
    buff = []
    with open(current, 'r', encoding='utf-8') as f1:
        lines = f1.readlines()

    docs = read_open_web_doc(lines)

    for doc in docs:
        if list_in_doc(doc):
            continue

        doc_text = "\n".join(doc)

        json_data = {args.json_key: doc_text}
        buff.append(json.dumps(json_data, ensure_ascii=False))

    with open(os.path.join(args.output_dir, f"{filename}.json"), 'w', encoding='utf-8') as out_f:
        out_f.write("\n".join(buff))

    return 1


def main():
    args = parse_args()

    if args.num_of_workers > 1:
        pool = multiprocessing.Pool(args.num_of_workers)

    for parent, dirnames, filenames in tqdm(os.walk(args.input)):
        parser = partial(process_doc, parent_dir=parent, args=args)
        if args.num_of_workers > 1:
            for _ in tqdm(pool.imap(parser, filenames)):
                pass
        else:
            for file in tqdm(filenames):
                parser(file)


if __name__ == "__main__":
    main()





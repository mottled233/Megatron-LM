import argparse
import json
from tqdm import tqdm
import os
import re

from concurrent.futures import ThreadPoolExecutor


def save(filename, docs):
    with open(filename, "w", encoding="utf-8") as w:
        for doc in docs:
            w.write(doc + "\n")


def read(filename, args):
    pool = ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="test_")
    with open(filename, "r", encoding="utf-8") as f:
        now_file_index = 0
        doc_list = []

        for index, line in tqdm(enumerate(f, start=1)):
            if index % args.per_doc_num == 0:
                writer = args.output_dir + "realnews-" + str(now_file_index)
                now_file_index += 1
                pool.submit(save, writer, doc_list)
                doc_list = []

            data = json.loads(line)
            text = data["text"]
            one_column = dict()
            # text = text.replace('\\"', '\"').replace('\\\'', '\'')
            length = len(text.split())
            if length <= args.min_seq_length:
                continue
            one_column["text"] = text
            json1 = json.dumps(one_column, ensure_ascii=False)
            json1 = re.sub(r'(\\n)+', r'\\n', json1)
            doc_list.append(json1)

        if doc_list:
            writer = args.output_dir + "realnews-" + str(now_file_index)
            pool.submit(save, writer, doc_list)

    pool.shutdown(wait=True)
    return index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filename",
        default=None,
        type=str,
        required=True,
        help="File to be processed"
    )
    parser.add_argument(
        "--min_seq_length",
        default=128,
        type=int,
        required=False,
        help="the shortest sequence length"
    )
    parser.add_argument(
        "--per_doc_num",
        default=100,
        type=int,
        required=False,
        help="doc nums in a file"
    )
    parser.add_argument(
        "--output_dir",
        default="realnews_result/",
        type=str,
        required=False,
        help="output dir"
    )
    parser.add_argument(
        "--workers",
        default=32,
        type=int
    )

    args = parser.parse_args()
    print(args)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    doc_num = read(args.filename, args)
    print("doc num", doc_num)


if __name__ == "__main__":
    main()

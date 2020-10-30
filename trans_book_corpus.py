import json
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='input data')
    group.add_argument('--input', type=str, required=True,
                       help='Path to input book corpus')
    group.add_argument('--output-dir', type=str, required=True,
                       help='Path to output book corpus')
    group.add_argument('--remove-lines', type=int, default=60,
                       help='Skip the first n lines')
    group.add_argument('--remove-empty-lines', type=bool, default=True)
    group.add_argument('--json-key', type=str, default="text")
    group.add_argument('--log-step', type=int, default=100)
    group.add_argument('--buff_file', type=int, default=1000)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    file_count = 0
    count = 0
    buff = []
    for parent, dirnames, filenames in os.walk(args.input):
        for filename in filenames:
            current = os.path.join(parent, filename)
            with open(current, 'r', encoding='utf-8') as f1:
                lines = f1.readlines()
            file = ""
            for idx, line in enumerate(lines):
                if idx < args.remove_lines:
                    continue
                if args.remove_empty_lines and line == "\n":
                    continue
                file += line

            json_data = {args.json_key: file}
            buff.append(json.dumps(json_data))
            count += 1
            if count != 0 and count % args.log_step:
                print(f"{count} books handled...")

            if len(buff) % args.buff_file == 0:
                with open(f"{args.output_dir}/books_{file_count}.json", 'w', encoding='utf-8') as out_f:
                    out_f.write("\n".join(buff))
                file_count += 1
                buff = []


if __name__ == "__main__":
    main()

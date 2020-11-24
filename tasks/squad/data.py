import torch
import os
from megatron import print_rank_0

import transformers
from transformers import squad_convert_examples_to_features
from transformers.data.processors.squad import SquadV1Processor, SquadV2Processor


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def load_and_cache_examples(args, tokenizer, stage="train", output_examples=False):
    if args.rank not in [-1, 0]:
        # Make sure only the first process in distributed training process the dataset,
        # and the others will use the cache
        torch.distributed.barrier()

    # Load data features from cache or dataset file
    input_dir = args.data_dir if args.data_dir else "."
    cached_features_file = os.path.join(
        input_dir,
        "cached_{}_{}".format(
           stage, str(args.seq_length),
        ),
    )

    # Init features and dataset from cache if it exists
    if os.path.exists(cached_features_file):
        print_rank_0(f"Loading features from cached file {cached_features_file}")
        features_and_dataset = torch.load(cached_features_file)
        features, dataset, examples = (
            features_and_dataset["features"],
            features_and_dataset["dataset"],
            features_and_dataset["examples"],
        )
    else:
        print_rank_0(f"Creating features from dataset file at {input_dir}")

        processor = SquadV1Processor()
        if stage == "train":
            examples = processor.get_train_examples(args.data_dir, filename=args.train_data)
        else:
            examples = processor.get_dev_examples(args.data_dir, filename=args.valid_data)
        print_rank_0(examples[0].context_text)
        print_rank_0(examples[0].doc_tokens)
        features, dataset = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=args.seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=(stage != "test"),
            return_dataset="pt",
            threads=args.num_workers,
        )

        if args.rank in [-1, 0]:
            print_rank_0(f"Saving features into cached file {cached_features_file}")
            torch.save({"features": features, "dataset": dataset, "examples": examples}, cached_features_file)

    if args.rank == 0:
        # Make sure only the first process in distributed training process the dataset,
        # and the others will use the cache
        torch.distributed.barrier()

    if output_examples:
        return dataset, examples, features
    return dataset

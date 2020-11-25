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

"""SQUAD."""
import torch
from torch.utils.data import DataLoader, SequentialSampler
from megatron import get_args, get_timers
from megatron import print_rank_0
from megatron import get_tokenizer
from megatron import mpu
from megatron.utils import make_data_loader
from megatron.model.reading import SQUAD
from megatron.utils import reduce_losses
from tasks.eval_utils import accuracy_func_provider
from tasks.finetune_utils import finetune, build_data_loader
from tasks.squad.data import load_and_cache_examples, to_list

from transformers.data.processors.squad import SquadResult
from transformers.data.metrics.squad_metrics import (
    compute_predictions_logits,
    squad_evaluate,
)
from transformers import BertTokenizer

import os


def parse_args(parser):
    group = parser.add_argument_group(title='squad')

    group.add_argument('--test-batch-size', type=int, default=4,
                       help='Batch size on single gpu in test stage.')

    group.add_argument('--data-dir', type=str, required=True,
                       help='Directory of prediction output.')
    group.add_argument('--output-dir', type=str, default="./result/",
                       help='Directory of prediction output.')
    group.add_argument('--n-best-size', type=int, default=20,
                       help='Output the best n results.')
    group.add_argument('--max-answer-length', type=int, default=30)
    group.add_argument('--max-query-length', type=int, default=64)
    group.add_argument('--verbose-logging', action='store_true',
                       help='If set, use verbose log in output prediction.')
    group.add_argument(
        "--null_score_diff_threshold",
        type=float,
        default=0.0,
        help="If null_score - best_non_null is greater than the threshold predict null.",
    )
    return parser


def train_valid_datasets_provider():
    """Provide train and validation datasets."""
    args = get_args()
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # batch: all_input_ids, all_attention_masks, all_token_type_ids, all_start_positions,
    #                 all_end_positions, all_cls_index, all_p_mask, all_is_impossible,
    train_dataset = load_and_cache_examples(args, tokenizer, stage="train", output_examples=False)
    valid_dataset = load_and_cache_examples(args, tokenizer, stage="dev", output_examples=False)

    return train_dataset, valid_dataset


def model_provider():
    """Build the model."""

    print_rank_0('building QA model for squad...')

    return SQUAD(num_tokentypes=2)


def compute_loss(start_logits, end_logits, start_position, end_position):
    criterion = torch.nn.CrossEntropyLoss(reduction='mean', ignore_index=-1)
    loss1 = criterion(start_logits, start_position) + criterion(end_logits, end_position)
    return loss1


def forward_step(batch, model, return_yp=False):
    timers = get_timers()

    # Get the batch.
    timers('batch generator').start()
    try:
        batch_ = next(batch)
    except BaseException:
        batch_ = batch
    batch_ = tuple(t.cuda().contiguous() for t in batch_)
    input_ids = batch_[0]
    attention_mask = batch_[1]
    token_type_ids = batch_[2]
    start_positions = batch_[3]
    end_positions = batch_[4]

    timers('batch generator').stop()

    st_logits, ed_logits = model(input_ids, attention_mask, token_type_ids)

    loss = compute_loss(st_logits, ed_logits, start_positions, end_positions)
    reduced_loss = reduce_losses([loss])

    if not return_yp:
        return loss, {'lm loss': reduced_loss[0]}
    else:
        return loss, {'lm loss': reduced_loss[0]}, st_logits, ed_logits


def test_step(batch, model):
    timers = get_timers()

    # Get the batch.
    timers('batch generator').start()
    try:
        batch_ = next(batch)
    except BaseException:
        batch_ = batch
    batch_ = tuple(t.cuda().contiguous() for t in batch_)
    input_ids = batch_[0]
    attention_mask = batch_[1]
    token_type_ids = batch_[2]
    feature_indices = batch_[3]

    timers('batch generator').stop()

    outputs = model(input_ids, attention_mask, token_type_ids)

    return feature_indices, outputs


def metrics_func_provider():
    """Privde metrics callback function."""
    args = get_args()
    tokenizer = get_tokenizer()
    # batch: all_input_ids, all_attention_masks, all_token_type_ids, all_feature_index, all_cls_index, all_p_mask
    test_dataset, test_examples, test_features\
        = load_and_cache_examples(args, BertTokenizer.from_pretrained("bert-base-uncased"), stage="test", output_examples=True)

    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    # Note that DistributedSampler samples randomly, but does it matter?
    # args.test_batch_size_ = max(1, mpu.get_data_parallel_world_size()) * args.test_batch_size
    # test_sampler = SequentialSampler(test_dataset)
    # test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.test_batch_size_)

    args.test_batch_size_ = args.test_batch_size
    test_dataloader = make_data_loader(test_dataset, batch_size=args.test_batch_size_)

    def test_model_func(model, epoch=-1, output_predictions=True):
        model.eval()
        all_results = []
        # For all the batches in the dataset.
        timers = get_timers()

        for iteration_, batch in enumerate(test_dataloader):
            timers("test iter").start()
            # Predict with bert
            feature_indices, outputs = test_step(batch, model)

            # Processing the output
            for i, feature_index in enumerate(feature_indices):
                eval_feature = test_features[feature_index.item()]
                unique_id = int(eval_feature.unique_id)

                output = [to_list(output[i]) for output in outputs]

                start_logits, end_logits = output
                result = SquadResult(unique_id, start_logits, end_logits)

                all_results.append(result)
            timers("test iter").stop()

            if iteration_+1 % args.log_interval == 0:
                time_per_iter = timers('test iter').elapsed() / args.test_batch_size_
                print_rank_0(f"Test iter {iteration_} finished, times per iter: {time_per_iter}")

        # Compute predictions
        output_prediction_file = os.path.join(args.output_dir, "predictions_epoch{}.json".format(epoch))
        output_nbest_file = os.path.join(args.output_dir, "nbest_predictions_epoch{}.json".format(epoch))
        output_null_log_odds_file = None
        predictions = compute_predictions_logits(
            test_examples,
            test_features,
            all_results,
            args.n_best_size,
            args.max_answer_length,
            True,  # args.do_lower_case,
            output_prediction_file,
            output_nbest_file,
            output_null_log_odds_file,
            args.verbose_logging,
            False,  # args.version_2_with_negative,
            args.null_score_diff_threshold,
            tokenizer,
        )

        # Compute the F1 and exact scores.
        results = squad_evaluate(test_examples, predictions)
        print_rank_0(f"Epoch {epoch} test results: {results}")

    return test_model_func


def main():
    finetune(train_valid_datasets_provider, model_provider,
             end_of_epoch_callback_provider=metrics_func_provider)

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

"""Classification model."""

import torch
from torch.autograd import Variable
import numpy as np

from megatron import get_args, print_rank_0
from megatron.model.bert_model import bert_attention_mask_func, bert_extended_attention_mask, bert_position_ids
from megatron.model.language_model import get_language_model
from megatron.model.utils import get_linear_layer
from megatron.model.utils import init_method_normal
from megatron.model.utils import scaled_init_method_normal
from megatron.module import MegatronModule


class SimplePredictionLayer(MegatronModule):
    def __init__(self):
        super(SimplePredictionLayer, self).__init__()

        args = get_args()
        init_method = init_method_normal(args.init_method_std)

        self.input_dim = args.hidden_size

        self.start_dropout = torch.nn.Dropout(args.hidden_dropout)
        self.start_head = get_linear_layer(args.hidden_size, 1, init_method)
        self._start_head_key = 'start_pos_head'

        self.end_dropout = torch.nn.Dropout(args.hidden_dropout)
        self.end_head = get_linear_layer(args.hidden_size, 1, init_method)
        self._end_head_key = 'end_pos_head'

        # self.type_linear = nn.Linear(self.input_dim, config.label_type_num)   # yes/no/ans

        self.cache_S = 0
        self.cache_mask = None

    def get_output_mask(self, outer):
        # (batch, 512, 512)
        S = outer.size(1)
        if S <= self.cache_S:
            return Variable(self.cache_mask[:S, :S], requires_grad=False)
        self.cache_S = S
        # triu 生成上三角矩阵，tril生成下三角矩阵，这个相当于生成了(512, 512)的矩阵表示开始-结束的位置，答案长度最长为15
        np_mask = np.tril(np.triu(np.ones((S, S)), 0), 15)
        self.cache_mask = outer.data.new(S, S).copy_(torch.from_numpy(np_mask))
        return Variable(self.cache_mask, requires_grad=False)

    def forward(self, input_state, return_yp=False):

        start_logits = self.start_head(self.start_dropout(input_state)).squeeze(2)
        end_logits = self.end_head(self.end_dropout(input_state)).squeeze(2)

        if return_yp:
            # 找结束位置用的开始和结束位置概率之和
            # (batch, 512, 1) + (batch, 1, 512) -> (512, 512)
            outer = start_logits[:, :, None] + end_logits[:, None]
            outer_mask = self.get_output_mask(outer)
            outer = outer - 1e30 * (1 - outer_mask[None].expand_as(outer))

            # 这两句相当于找到了outer中最大值的i和j坐标
            start_position = outer.max(dim=2)[0].max(dim=1)[1]
            end_position = outer.max(dim=1)[0].max(dim=1)[1]

            return start_logits, end_logits, start_position, end_position

        else:
            return start_logits, end_logits

    def state_dict_for_save_checkpoint(self, destination=None, prefix='',
                                       keep_vars=False):
        """For easy load when model is combined with other heads,
        add an extra key."""
        state_dict_ = {}
        state_dict_[self._start_head_key] \
            = self.start_head.state_dict(
                destination, prefix, keep_vars)
        state_dict_[self._end_head_key] \
            = self.end_head.state_dict(
                destination, prefix, keep_vars)
        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""

        self.start_head.load_state_dict(
            state_dict[self._start_head_key], strict=strict)
        self.prediction.load_state_dict(
            state_dict[self._end_head_key], strict=strict)


class SQUAD(MegatronModule):
    def __init__(self, num_tokentypes=2):
        super(SQUAD, self).__init__()
        args = get_args()

        init_method = init_method_normal(args.init_method_std)

        self.language_model, self._language_model_key = get_language_model(
            attention_mask_func=bert_attention_mask_func,
            num_tokentypes=num_tokentypes,
            add_pooler=True,
            init_method=init_method,
            scaled_init_method=scaled_init_method_normal(args.init_method_std,
                                                         args.num_layers))

        # Reading head.
        self.prediction = SimplePredictionLayer()
        self._perdiction_layer_key = 'perdiction_layer'

    def forward(self, input_ids, attention_mask, tokentype_ids, return_yp=False):

        extended_attention_mask = bert_extended_attention_mask(attention_mask)
        position_ids = bert_position_ids(input_ids)

        input_state, pooled_output = self.language_model(input_ids,
                                                         position_ids,
                                                         extended_attention_mask,
                                                         tokentype_ids=tokentype_ids)

        # Output.
        return self.prediction(input_state, return_yp)

    def state_dict_for_save_checkpoint(self, destination=None, prefix='',
                                       keep_vars=False):
        """For easy load when model is combined with other heads,
        add an extra key."""
        state_dict_ = {}
        state_dict_[self._language_model_key] \
            = self.language_model.state_dict_for_save_checkpoint(
                destination, prefix, keep_vars)
        state_dict_[self._perdiction_layer_key] \
            = self.prediction.state_dict(
                destination, prefix, keep_vars)
        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""

        self.language_model.load_state_dict(
            state_dict[self._language_model_key], strict=strict)
        if self._perdiction_layer_key in state_dict:
            self.prediction.load_state_dict(
                state_dict[self._perdiction_layer_key], strict=strict)
        else:
            print_rank_0('***WARNING*** could not find {} in the checkpoint, '
                         'initializing to random'.format(
                             self._perdiction_layer_key))

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

"""Utilities for models."""

import math
from functools import wraps

import torch
from mup.init import normal_

from megatron import get_args

def init_method_normal(sigma):
    """Init method based on N(0, sigma)."""
    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=sigma)

    return init_


def init_method_mup_normal(sigma):
    """Normal initialization with mup rescaling"""
    def init_(tensor):
        return normal_(tensor, mean=0, std=sigma)

    return init_


# hack to detect the last layer norm weight (not given a name by the pipeline)
def is_last_layernorm_weight(name):
    split = name.split(".")
    return len(split) == 2 and split[0].isdigit() and split[1] == "weight"


def mup_init_with_param_name(sigma):
    def init_with_name_(name, param):
        if name.split(".")[-1] == "weight":
            if "layernorm" in name or is_last_layernorm_weight:
                torch.nn.init.constant_(param, 1)
            else:
                normal_(param, mean=0, std=sigma)
        else:
            torch.nn.init.constant_(param, 0)

    return init_with_name_


def scaled_init_method_normal(sigma, num_layers):
    """Init method based on N(0, sigma/sqrt(2*num_layers)."""
    std = sigma / math.sqrt(2.0 * num_layers)

    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=std)

    return init_


def attention_mask_func(attention_scores, attention_mask):
    args = get_args()
    if args.curriculum_learning:
        attention_mask_ = attention_mask
        actual_seqlen = attention_scores.size()[2]
        if actual_seqlen != attention_mask_.size()[2]:
            # attention_mask has size [1, 1, seqlen, seqlen]
            attention_mask_ = attention_mask_[:, :, :actual_seqlen, :actual_seqlen].contiguous()
        attention_scores.masked_fill_(attention_mask_, torch.finfo(attention_scores.dtype).min)
    else:
        attention_scores.masked_fill_(attention_mask, torch.finfo(attention_scores.dtype).min)
    return attention_scores


def get_linear_layer(rows, columns, init_method):
    """Simple linear layer with weight initialization."""
    layer = torch.nn.Linear(rows, columns)
    init_method(layer.weight)
    with torch.no_grad():
        layer.bias.zero_()
    return layer

@torch.jit.script
def gelu_impl(x):
    """OpenAI's gelu implementation."""
    return 0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * x *
                                       (1.0 + 0.044715 * x * x)))
def openai_gelu(x):
    return gelu_impl(x)

#This is actually Python equivalent of torch.nn.functional.gelu(), also with type hints for ONNX exporter
@torch.jit.script
def erf_gelu(x):
    return x * 0.5 * (torch.erf(x / 1.41421).to(dtype=x.dtype)+torch.ones_like(x).to(dtype=x.dtype))

def log_debug_usage(logger, msg: str):
    def log_debug_usage_(func):
        """Helper function in order to log a message when using a function for the first time"""
        func.__logged_message__ = False

        @wraps(func)
        def wrapped(*args, **kwargs):
            if func.__logged_message__ is False:
                logger.debug(msg)
                func.__logged_message__ = True
            return func(*args, **kwargs)

        return wrapped
    return log_debug_usage_

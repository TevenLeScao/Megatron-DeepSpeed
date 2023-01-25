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

"""Learning rate decay functions."""

import math
from typing import List

from megatron import print_rank_0, get_args

class AnnealingLR(object):
    """Anneals the learning rate."""

    def __init__(self, optimizer, max_lr, min_lr,
                 warmup_steps, decay_steps, decay_style,
                 use_checkpoint_lr_scheduler=True,
                 override_lr_scheduler=False):
        args = get_args()
        # Class values.
        self.optimizer = optimizer

        self.n_groups = len(optimizer.param_groups)

        if isinstance(max_lr, list):
            assert len(max_lr) == self.n_groups
            self.max_lr = max_lr
        else:
            self.max_lr = [float(max_lr) for _ in range(self.n_groups)]
        if isinstance(min_lr, list):
            assert len(min_lr) == self.n_groups
            self.min_lr = min_lr
        else:
            self.min_lr = [float(min_lr) for _ in range(self.n_groups)]
        assert [lr >= 0.0 for lr in self.min_lr]
        assert [lr2 >= lr1 for lr2, lr1 in zip(self.max_lr, self.min_lr)]

        self.warmup_steps = warmup_steps
        self.num_steps = 0
        self.decay_steps = decay_steps
        assert self.decay_steps > 0
        assert self.warmup_steps < self.decay_steps

        self.decay_tokens = args.lr_decay_tokens
        self.num_tokens = 0
        self.warmup_tokens = 0

        self.decay_style = decay_style

        self.override_lr_scheduler = override_lr_scheduler
        self.use_checkpoint_lr_scheduler = use_checkpoint_lr_scheduler
        if self.override_lr_scheduler:
            assert not self.use_checkpoint_lr_scheduler, 'both override and '\
                'use-checkpoint are set.'

        # Set the learning rate
        self.step(0)

        print_rank_0('> learning rate decay style: {}'.format(self.decay_style))


    def get_lrs(self) -> List:
        """Learning rate decay functions from:
              https://openreview.net/pdf?id=BJYwwY9ll pg. 4"""

        # Use linear warmup for the initial part.
        if self.warmup_steps > 0 and self.num_steps <= self.warmup_steps:
            if self.num_steps == self.warmup_steps and \
                self.decay_tokens is not None:
                self.warmup_tokens = self.num_tokens
            return [lr * float(self.num_steps) / float(self.warmup_steps) for lr in self.max_lr]

        # If the learning rate is constant, just return the initial value.
        if self.decay_style == 'constant':
            return self.max_lr

        if self.decay_tokens is None:
            # step-based decay
            
            # For any steps larger than `self.decay_steps`, use `self.min_lr`.
            if self.num_steps > self.decay_steps:
                return self.min_lr
            
            # If we are done with the warmup period, use the decay style.
            num_steps_ = self.num_steps - self.warmup_steps
            decay_steps_ = self.decay_steps - self.warmup_steps
            decay_ratio = float(num_steps_) / float(decay_steps_)
        else:
            # token-based decay

            if self.num_tokens > self.decay_tokens:
                return self.min_lr
            num_tokens_ = self.num_tokens - self.warmup_tokens
            decay_tokens_ = self.decay_tokens - self.warmup_tokens
            decay_ratio = float(num_tokens_) / float(decay_tokens_)
        assert decay_ratio >= 0.0
        assert decay_ratio <= 1.0
        delta_lrs = [lr2 - lr1 for lr2, lr1 in zip(self.max_lr, self.min_lr)]

        if self.decay_style == 'linear':
            coeff = (1.0 - decay_ratio)
        elif self.decay_style == 'cosine':
            coeff = 0.5 * (math.cos(math.pi * decay_ratio) + 1.0)
        else:
            raise Exception('{} decay style is not supported.'.format(
                self.decay_style))
       
        return [lr + coeff * delta for lr, delta in zip(self.min_lr, delta_lrs)]


    def step(self, increment, token_num=None):
        """Set lr for all parameters groups."""
        if token_num is None:
            args = get_args()
            token_num = args.consumed_train_tokens
        self.num_tokens = token_num
        self.num_steps += increment
        new_lrs = self.get_lrs()
        for group, new_lr in zip(self.optimizer.param_groups, new_lrs):
            group['lr'] = new_lr


    def state_dict(self):
        state_dict = {
            'max_lr': self.max_lr,
            'warmup_steps': self.warmup_steps,
            'num_steps': self.num_steps,
            'warmup_tokens': self.warmup_tokens,
            'num_tokens': self.num_tokens,
            'decay_style': self.decay_style,
            'decay_steps': self.decay_steps,
            'min_lr': self.min_lr
        }
        return state_dict


    def _check_and_set(self, cls_value, sd_value, name):
        """Auxiliary function for checking the values in the checkpoint and
        setting them."""
        if self.override_lr_scheduler:
            print_rank_0(' > overriding {} value to {}'.format(name, cls_value))
            return cls_value

        if not self.use_checkpoint_lr_scheduler:
            assert cls_value == sd_value, \
                f'AnnealingLR: class input value {cls_value} and checkpoint' \
                f'value {sd_value} for {name} do not match'
        print_rank_0(' > using checkpoint value {} for {}'.format(sd_value,
                                                                  name))
        return sd_value


    def load_state_dict(self, sd):

        if 'start_lr' in sd:
            max_lr_ = sd['start_lr']
        else:
            max_lr_ = sd['max_lr']
        self.max_lr = self._check_and_set(self.max_lr, max_lr_,
                                          'learning rate')
        self.min_lr = self._check_and_set(self.min_lr, sd['min_lr'],
                                          'minimum learning rate')
        # Backwards compatibility
        if not isinstance(self.max_lr, list):
            self.max_lr = [self.max_lr for _ in range(self.n_groups)]
        if not isinstance(self.min_lr, list):
            self.min_lr = [self.min_lr for _ in range(self.n_groups)]

        if 'warmup_iter' in sd:
            warmup_steps_ = sd['warmup_iter']
        else:
            warmup_steps_ = sd['warmup_steps']
        self.warmup_steps = self._check_and_set(self.warmup_steps,
                                                warmup_steps_,
                                                'warmup iterations')

        if 'end_iter' in sd:
            decay_steps_ = sd['end_iter']
        else:
            decay_steps_ = sd['decay_steps']
        self.decay_steps = self._check_and_set(self.decay_steps, decay_steps_,
                                               'total number of iterations')
        self.decay_style = self._check_and_set(self.decay_style,
                                               sd['decay_style'],
                                               'decay style')

        if 'num_iters' in sd:
            num_steps = sd['num_iters']
        else:
            num_steps = sd['num_steps']
        if 'warmup_tokens' in sd:
            self.warmup_tokens = sd['warmup_tokens']
        if 'num_tokens' in sd:
            self.num_tokens = sd['num_tokens']
        self.step(num_steps, self.num_tokens)

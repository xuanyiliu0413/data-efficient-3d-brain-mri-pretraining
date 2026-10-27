# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from logging import getLogger

import torch

logger = getLogger()


def _no_grad_trunc_normal_(tensor, mean, std, a, b):


    def norm_cdf(x):

        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    with torch.no_grad():



        lower = norm_cdf((a - mean) / std)
        upper = norm_cdf((b - mean) / std)



        tensor.uniform_(2 * lower - 1, 2 * upper - 1)



        tensor.erfinv_()


        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)


        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):

    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def repeat_interleave_batch(x, B, repeat):
    N = len(x) // B
    x = torch.cat([torch.cat([x[i * B : (i + 1) * B] for _ in range(repeat)], dim=0) for i in range(N)], dim=0)
    return x

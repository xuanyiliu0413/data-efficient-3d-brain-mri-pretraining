# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch


def apply_masks(x, masks, concat=True):




    all_x = []
    for m in masks:




        batch_idx = torch.arange(x.size(0), device=x.device).unsqueeze(1)
        all_x.append(x[batch_idx, m])
    if not concat:
        return all_x

    return torch.cat(all_x, dim=0)

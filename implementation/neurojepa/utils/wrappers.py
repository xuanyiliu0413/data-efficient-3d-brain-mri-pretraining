# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn


class MultiSeqWrapper(nn.Module):

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x, masks=None):




        if masks is None:
            outs = []
            for xi in x:
                z_i, s_i = self.backbone(xi)
                outs.append(z_i)
            return outs

        outs = [[] for _ in x]
        outs_scores = [[] for _ in x]
        for i, (xi, mi) in enumerate(zip(x, masks)):

            if type(xi) is not torch.Tensor:
                xi = xi.as_subclass(torch.Tensor)
            for mij in mi:
                zij, sij = self.backbone(xi, masks=mij)
                outs[i] += [zij]
                outs_scores[i] += [sij]
        return outs, outs_scores


class PredictorMultiSeqWrapper(nn.Module):

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x, masks_x, masks_y, has_cls=False):
        n = 0
        outs = [[] for _ in x]
        for i, (xi, mxi, myi) in enumerate(zip(x, masks_x, masks_y)):
            for xij, mxij, myij in zip(xi, mxi, myi):
                outs[i] += [self.backbone(xij, mxij, myij, mask_index=i, has_cls=has_cls)]
                n += 1
        return outs


class MultiModalSeqWrapper(nn.Module):










    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x, masks=None, validity_mask=None):





        if masks is None:
            outs = []
            for xi in x:
                z_i, s_i = self.backbone(xi, validity_mask=validity_mask)
                outs.append(z_i)
            return outs

        outs = [[] for _ in x]
        outs_scores = [[] for _ in x]
        for i, (xi, mi) in enumerate(zip(x, masks)):
            for mij in mi:
                zij, sij = self.backbone(xi, masks=mij, validity_mask=validity_mask)
                outs[i] += [zij]
                outs_scores[i] += [sij]
        return outs, outs_scores

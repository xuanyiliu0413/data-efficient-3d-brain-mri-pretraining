import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple, Optional, Literal

import torch.distributed as dist

global world_size, rank
world_size = dist.get_world_size() if dist.is_initialized() else 1
rank = dist.get_rank() if dist.is_initialized() else 0


def linear(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:





















    return F.linear(x, weight, bias)


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Gate(nn.Module):











    def __init__(self,
                 dim: int,
                 n_routed_experts: int,
                 n_activated_experts: int,
                 score_func = "sigmoid",
                 route_scale = 1.0):






        super().__init__()
        self.dim = dim
        self.topk = n_activated_experts
        self.score_func = score_func
        self.gate = nn.Linear(dim, n_routed_experts, bias=False)


        self.register_buffer("bias", torch.zeros(n_routed_experts))

        self.route_scale = route_scale

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:










        logits = self.gate(x)
        if self.score_func == "softmax":
            scores = logits.softmax(dim=-1, dtype=torch.float32)
        else:
            scores = logits.sigmoid()


        original_scores = scores

        scores = scores + self.bias

        indices = torch.topk(scores, self.topk, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        if self.score_func == "sigmoid":

            weights /= weights.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        weights *= self.route_scale


        self.last_indices = indices.detach()
        return weights.type_as(x), indices, original_scores


class MoE(nn.Module):












    def __init__(self,
                 dim: int,
                 n_shared_experts: int,
                 n_routed_experts: int,
                 n_activated_experts: int,
                 moe_inter_dim: int,
                 score_func = "sigmoid",
                 route_scale = 1.0,
                 act_layer = nn.GELU,
                 drop = 0.0,
                 ):






        super().__init__()
        self.dim = dim
        self.n_routed_experts = n_routed_experts
        self.n_activated_experts = n_activated_experts
        self.experts_start_idx = 0
        self.experts_end_idx = n_routed_experts
        self.gate = Gate(
            dim=dim,
            n_routed_experts=n_routed_experts,
            n_activated_experts=n_activated_experts,
            score_func=score_func,
            route_scale=route_scale
        )
        self.experts = nn.ModuleList([MLP(in_features=dim, hidden_features=moe_inter_dim, act_layer=act_layer, drop=drop)
                                      if self.experts_start_idx <= i < self.experts_end_idx else None
                                      for i in range(self.n_routed_experts)])
        self.shared_experts = MLP(in_features=dim, hidden_features=n_shared_experts * moe_inter_dim, act_layer=act_layer, drop=drop)


        self.register_buffer('counts', torch.zeros(self.n_routed_experts, dtype=torch.long))

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:











        shape = x.size()
        x = x.view(-1, self.dim)
        weights, indices, original_scores = self.gate(x)
        y = torch.zeros_like(x)


        flat_indices = indices.flatten()
        if mask is not None:


            mask_flat = mask.view(-1)


            mask_expanded = mask_flat.unsqueeze(-1).expand_as(indices).flatten()
            valid_indices = flat_indices[mask_expanded]
        else:
            valid_indices = flat_indices

        batch_counts = torch.bincount(valid_indices, minlength=self.n_routed_experts)
        self.counts += batch_counts.to(self.counts.device)


        self.last_indices = indices.detach()


        if mask is not None:
            valid_mask = mask.view(-1)
        else:
            valid_mask = None

        for i in range(self.experts_start_idx, self.experts_end_idx):
            expert = self.experts[i]
            if batch_counts[i] == 0:


                y = y + 0.0 * expert(x[:1]).sum()
                continue
            idx, top = torch.where(indices == i)
            if valid_mask is not None:

                valid_sel = valid_mask[idx]
                valid_idx = idx[valid_sel]
                valid_top = top[valid_sel]
                if valid_idx.numel() > 0:
                    y[valid_idx] += expert(x[valid_idx]) * weights[valid_idx, valid_top, None]
            else:
                y[idx] += expert(x[idx]) * weights[idx, top, None]

        if valid_mask is not None:

            valid_token_indices = valid_mask.nonzero(as_tuple=True)[0]
            if valid_token_indices.numel() > 0:
                shared_out = self.shared_experts(x[valid_token_indices])
                z = torch.zeros_like(x, dtype=shared_out.dtype)
                z[valid_token_indices] = shared_out
            else:
                z = torch.zeros_like(x)

            original_scores = original_scores * valid_mask.unsqueeze(-1).float()
        else:
            z = self.shared_experts(x)

        return (y + z).view(shape), original_scores

    def reset_counts(self):

        self.counts.zero_()


def calculate_vio(expert_counts):

    if expert_counts.numel() == 0:
        return [1.0, 1.0]
    total = expert_counts.sum()
    if total == 0:
        return [1.0, 1.0]

    avg_count = expert_counts.float().mean()
    if avg_count == 0:
        return [1.0, 1.0]

    min_violation = torch.min(expert_counts.float()) / avg_count
    max_violation = torch.max(expert_counts.float()) / avg_count

    return [min_violation.item(), max_violation.item()]


def calculate_vio_model(model, used_ddp=False, vit=False):









    if used_ddp:
        model_noddp = model.module
    else:
        model_noddp = model

    backbone = model_noddp if vit else getattr(model_noddp, 'backbone', model_noddp)


    moe_layers = [b.mlp for b in backbone.blocks if hasattr(b, 'mlp') and isinstance(b.mlp, MoE)]

    if not moe_layers:
        return 1.0, 1.0


    all_counts = torch.stack([m.counts for m in moe_layers])

    if used_ddp and dist.is_initialized():
        dist.all_reduce(all_counts, op=dist.ReduceOp.SUM)

    total_minvio = 0.0
    total_maxvio = 0.0

    for i, moe_layer in enumerate(moe_layers):
        min_vio, max_vio = calculate_vio(all_counts[i])
        total_minvio += min_vio
        total_maxvio += max_vio
        moe_layer.reset_counts()

    n = len(moe_layers)
    return total_minvio / n, total_maxvio / n


def unpack_moe_params(moe_params):
    dim = moe_params.dim
    n_shared_experts = moe_params.n_shared_experts
    n_routed_experts = moe_params.n_routed_experts
    n_activated_experts = moe_params.n_activated_experts
    moe_inter_dim = moe_params.moe_inter_dim
    score_func = moe_params.score_func
    route_scale = moe_params.route_scale

    return dim, n_shared_experts, n_routed_experts, n_activated_experts, moe_inter_dim, score_func, route_scale


def moe_bias_update(model, update_rate, used_ddp=False, vit=False, bias_clip=0.3, use_proportional=False):













    if used_ddp:
        model_noddp = model.module
    else:
        model_noddp = model


    backbone = model_noddp if vit else getattr(model_noddp, 'backbone', model_noddp)


    moe_layers = [b.mlp for b in backbone.blocks if hasattr(b, 'mlp') and isinstance(b.mlp, MoE)]

    if not moe_layers:
        return 0.0, 0.0


    all_counts = torch.stack([m.counts for m in moe_layers]).long()

    if used_ddp and dist.is_initialized():
        dist.all_reduce(all_counts, op=dist.ReduceOp.SUM)


    all_counts = all_counts.float()

    total_minvio = 0.0
    total_maxvio = 0.0


    with torch.no_grad():
        for i, moe_layer in enumerate(moe_layers):
            layer_counts = all_counts[i]
            avg_count = layer_counts.mean()


            if avg_count > 0:
                rel_counts = layer_counts / avg_count
                total_minvio += rel_counts.min().item()
                total_maxvio += rel_counts.max().item()


                error = avg_count - layer_counts

                if use_proportional:

                    update = error / (avg_count + 1e-6)
                else:

                    update = torch.sign(error)


                update = update - update.mean()


                moe_layer.gate.bias.data += update_rate * update


                if bias_clip > 0:
                    moe_layer.gate.bias.data.clamp_(-bias_clip, bias_clip)



                moe_layer.gate.bias.data -= moe_layer.gate.bias.data.mean()
            else:

                total_minvio += 1.0
                total_maxvio += 1.0


            moe_layer.reset_counts()

    n = len(moe_layers)
    return total_minvio / n, total_maxvio / n

import torch
import sys
import logging
import os
import pickle
from typing import Any, Dict, Optional

import torch.distributed as dist
import torch.nn.functional as F

from neurojepa.models.utils.moe import MoE
from neurojepa.utils.misc import MetricLogger

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def get_foreground_mask(volumes, patch_size=(16, 16, 4), min_foreground_fraction=0.1):







    B, C, D, H, W = volumes.shape
    pD, pH, pW = patch_size



    vol_max = volumes.abs().amax(dim=1, keepdim=True)
    vol_flat = vol_max.reshape(B, -1)

    p2 = torch.quantile(vol_flat.float(), 0.02, dim=1)
    p98 = torch.quantile(vol_flat.float(), 0.98, dim=1)
    dyn_thresh = 0.1 * (p98 - p2) + p2


    threshold = 0.0
    dyn_thresh = torch.where(p98 > p2, dyn_thresh, torch.full_like(dyn_thresh, threshold))

    dyn_thresh = dyn_thresh.view(B, 1, 1, 1, 1)


    vol = vol_max






    pad_D = (pD - D % pD) % pD
    pad_H = (pH - H % pH) % pH
    pad_W = (pW - W % pW) % pW
    if pad_D > 0 or pad_H > 0 or pad_W > 0:
        vol = F.pad(vol, (0, pad_W, 0, pad_H, 0, pad_D))


    u_vol = vol.unfold(2, pD, pD).unfold(3, pH, pH).unfold(4, pW, pW)



    dt = dyn_thresh.view(B, 1, 1, 1, 1, 1, 1, 1)
    fg_frac = (u_vol > dt).float().mean(dim=(-3, -2, -1))



    return (fg_frac >= min_foreground_fraction).flatten(1)


def infer_one_epoch(
    cfg: Any,
    model: Dict[str, torch.nn.Module],
    loader: torch.utils.data.DataLoader,
    logger: Optional[logging.Logger] = None,
    device: Optional[torch.device] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
    wandb_run: Optional[Any] = None,
) -> Dict[str, float]:

    metric_logger = MetricLogger(delimiter="  ", logger=logger)
    backbone = model['backbone']
    backbone.eval()

    use_amp = cfg.optimization.mixed_precision

    backbone_noddp = backbone.module
    num_blocks = len(backbone_noddp.blocks)
    num_experts = cfg.model.moe_params.n_routed_experts
    epoch_expert_counts_sum = {i: torch.zeros(num_experts, device=device) for i in range(num_blocks)}
    epoch_expert_counts_fg = {i: torch.zeros(num_experts, device=device) for i in range(num_blocks)}
    epoch_expert_counts_bg = {i: torch.zeros(num_experts, device=device) for i in range(num_blocks)}
    total_tokens_epoch = 0


    moe_block_indices = [i for i, b in enumerate(backbone_noddp.blocks) if isinstance(b.mlp, MoE)]
    num_moe_layers = len(moe_block_indices)
    moe_block_to_layer = {bi: li for li, bi in enumerate(moe_block_indices)}
    token_routing_counts = None

    for idx, batch_data in enumerate(loader):
        print(f"Inference step {idx+1}/{len(loader)}", end='\r')
        image, target = batch_data
        image = image.to(device, non_blocking=True)


        patch_size = backbone_noddp.patch_size if hasattr(backbone_noddp, 'patch_size') else (16, 16, 4)
        fg_mask = get_foreground_mask(image, patch_size).to(device)
        fg_mask_flat = fg_mask.view(-1)

        with torch.amp.autocast('cuda', enabled=use_amp, dtype=torch.float16):
            feats = backbone(image)

        torch.cuda.synchronize()


        for block_idx, block in enumerate(backbone_noddp.blocks):

            if not isinstance(block.mlp, MoE):
                continue
            moe_layer = block.mlp
            expert_counts = moe_layer.counts.clone()


            if hasattr(moe_layer, 'last_indices') and moe_layer.last_indices is not None:
                indices = moe_layer.last_indices


                B_cur = image.shape[0]
                N_tokens = indices.shape[0] // B_cur
                k = indices.shape[1]
                if token_routing_counts is None:
                    token_routing_counts = torch.zeros(num_moe_layers, N_tokens, num_experts, device=device)
                moe_idx = moe_block_to_layer[block_idx]
                indices_3d = indices.view(B_cur, N_tokens, k)
                one_hot = F.one_hot(indices_3d.long(), num_classes=num_experts)
                token_routing_counts[moe_idx] += one_hot.sum(dim=(0, 2)).float()



                if indices.shape[0] == fg_mask_flat.shape[0]:
                    fg_indices = indices[fg_mask_flat].flatten()
                    bg_indices = indices[~fg_mask_flat].flatten()

                    fg_counts = torch.bincount(fg_indices, minlength=num_experts)
                    bg_counts = torch.bincount(bg_indices, minlength=num_experts)

                    if dist.is_initialized():
                        dist.all_reduce(fg_counts, op=dist.ReduceOp.SUM)
                        dist.all_reduce(bg_counts, op=dist.ReduceOp.SUM)

                    epoch_expert_counts_fg[block_idx] += fg_counts
                    epoch_expert_counts_bg[block_idx] += bg_counts

            if dist.is_initialized():

                dist.all_reduce(expert_counts, op=dist.ReduceOp.SUM)


            epoch_expert_counts_sum[block_idx] += expert_counts

            if block_idx == list(epoch_expert_counts_sum.keys())[-1]:
                total_tokens_epoch += expert_counts.sum().item()


            if hasattr(moe_layer, 'last_indices'):
                moe_layer.last_indices = None


            moe_layer.reset_counts()



    final_expert_percentages_fg = {}
    final_expert_percentages_bg = {}

    if total_tokens_epoch > 0:
        for block_idx in epoch_expert_counts_fg.keys():
            total_fg = epoch_expert_counts_fg[block_idx].sum().item()
            total_bg = epoch_expert_counts_bg[block_idx].sum().item()

            if total_fg > 0:
                final_expert_percentages_fg[block_idx] = (epoch_expert_counts_fg[block_idx].float() / total_fg).cpu()
            else:
                final_expert_percentages_fg[block_idx] = torch.zeros(num_experts, dtype=torch.float)

            if total_bg > 0:
                final_expert_percentages_bg[block_idx] = (epoch_expert_counts_bg[block_idx].float() / total_bg).cpu()
            else:
                final_expert_percentages_bg[block_idx] = torch.zeros(num_experts, dtype=torch.float)
    else:
        for block_idx in range(num_blocks):
            final_expert_percentages_fg[block_idx] = torch.zeros(num_experts, dtype=torch.float)
            final_expert_percentages_bg[block_idx] = torch.zeros(num_experts, dtype=torch.float)


    if token_routing_counts is not None:
        if dist.is_initialized():
            dist.all_reduce(token_routing_counts, op=dist.ReduceOp.SUM)

        token_routing_probs = token_routing_counts / token_routing_counts.sum(dim=-1, keepdim=True).clamp(min=1e-9)


    if not dist.is_initialized() or dist.get_rank() == 0:
        _prefix = cfg.inference.output_prefix
        output_filename_fg = f'{_prefix}_foreground.pkl'
        output_filename_bg = f'{_prefix}_background.pkl'

        os.makedirs(os.path.dirname(os.path.abspath(output_filename_fg)), exist_ok=True)

        logger.info(f"Saving expert counts for the epoch to {output_filename_fg} and {output_filename_bg}...")
        with open(output_filename_fg, 'wb') as f:
            pickle.dump(final_expert_percentages_fg, f)
        with open(output_filename_bg, 'wb') as f:
            pickle.dump(final_expert_percentages_bg, f)
        logger.info("Done saving expert counts.")


        if token_routing_counts is not None:
            output_filename_routing = f'{_prefix}_token_routing.pkl'
            logger.info(f"Saving per-token-position routing to {output_filename_routing}...")
            with open(output_filename_routing, 'wb') as f:
                pickle.dump({
                    'counts': token_routing_counts.cpu(),
                    'probs': token_routing_probs.cpu(),
                    'moe_block_indices': moe_block_indices,
                }, f)
            logger.info(f"Done saving token routing (shape: {token_routing_counts.shape}).")

    return


def inferencer(
    cfg: Any,
    model: Dict[str, torch.nn.Module],
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    logger: Optional[logging.Logger] = None,
    device: Optional[torch.device] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
    wandb_run: Optional[Any] = None,
) -> float:


    infer_one_epoch(
        cfg=cfg,
        model=model,
        loader=val_loader,
        logger=logger,
        device=device,
        scaler=scaler,
        wandb_run=wandb_run,
    )

    return

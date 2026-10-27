import torch
import math
import time
import sys
import logging
import os
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Callable

import torch.distributed as dist

from torchmetrics import MetricCollection

from neurojepa.utils.init_utils import build_monitor_metrics
from neurojepa.utils.misc import all_reduce_mean, MetricLogger

import copy
import pickle

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def unpack_metrics(results: Dict[str, torch.Tensor]) -> Any:

    perclass_auc  = results["auroc_perclass"].detach().cpu().tolist()
    perclass_ap   = results["auprc_perclass"].detach().cpu().tolist()
    perclass_acc  = results["acc_perclass"].detach().cpu().tolist()
    perclass_f1  = results["f1_perclass"].detach().cpu().tolist()
    macro_auc     = float(results["auroc_macro"].detach().cpu())
    macro_ap      = float(results["auprc_macro"].detach().cpu())
    macro_acc     = float(results["acc_macro"].detach().cpu())
    macro_f1      = float(results["f1_macro"].detach().cpu())
    return perclass_auc, perclass_ap, perclass_acc, perclass_f1, macro_auc, macro_ap, macro_acc, macro_f1

def log_metrics(loss, epoch, results, class_names, wandb_run, mode='train'):

    perclass_auc, perclass_ap, perclass_acc, perclass_f1,\
        macro_auc, macro_ap, macro_acc, macro_f1 = unpack_metrics(results)

    logger.info(f"{mode} epoch {epoch+1} metrics:")
    logger.info(
        "Per-class: " +
        ", ".join([f"{name}: AUC={perclass_auc[i]:.3f}, AP={perclass_ap[i]:.3f}, Acc={perclass_acc[i]:.3f}, F1={perclass_f1[i]:.3f}"
                for i, name in enumerate(class_names)]) +
        f" | MACRO -> AUC={macro_auc:.3f}, AP={macro_ap:.3f}, Acc={macro_acc:.3f}, F1={macro_f1:.3f}"
    )


    if dist.get_rank() == 0 and wandb_run is not None:
        wandb_run.log({
            "epoch": epoch + 1,
            f"{mode}/auc_macro": macro_auc,
            f"{mode}/ap_macro": macro_ap,
            f"{mode}/acc_macro": macro_acc,
            f"{mode}/f1_macro": macro_f1,
            **{f"{mode}/auc_{class_names[i]}": perclass_auc[i] for i in range(len(class_names))},
            **{f"{mode}/ap_{class_names[i]}": perclass_ap[i] for i in range(len(class_names))},
            **{f"{mode}/acc_{class_names[i]}": perclass_acc[i] for i in range(len(class_names))},
            **{f"{mode}/f1_{class_names[i]}": perclass_f1[i] for i in range(len(class_names))},
        })
        if loss != None:
            wandb_run.log({
                f"{mode}/loss": loss,
            })

    return


@torch.no_grad()
def evaluate_epoch(
    cfg: Any,
    model: Dict[str, torch.nn.Module],
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    metrics: MetricCollection,
    logger: Optional[logging.Logger] = None,
    use_amp: bool = False,
) -> Dict[str, float]:

    backbone, classifier = model['backbone'], model['classifier']
    backbone.eval()
    classifier.eval()

    all_probs = []
    all_targets = []
    for idx, batch in enumerate(loader):
        image, target = batch
        image  = image.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.amp.autocast('cuda', enabled=use_amp, dtype=torch.bfloat16):
            feats = backbone(image)
            out_list = classifier(feats)
            logits = torch.stack([out.squeeze(-1) for out in out_list], dim=1)
            probs = torch.sigmoid(logits)
        metrics.update(probs, target.int())
        all_probs.append(probs.detach().cpu())
        all_targets.append(target.detach().cpu())

        logger.info(f"Validation/Test: [{idx+1}/{len(loader)}]")

    results = metrics.compute()
    metrics.reset()


    all_probs_local = torch.cat(all_probs, dim=0)
    all_tgts_local  = torch.cat(all_targets, dim=0)

    world_size = dist.get_world_size()
    if world_size > 1:
        probs_gpu = all_probs_local.to(device)
        tgts_gpu  = all_tgts_local.to(device)
        gathered_probs   = [torch.zeros_like(probs_gpu) for _ in range(world_size)]
        gathered_targets = [torch.zeros_like(tgts_gpu)  for _ in range(world_size)]
        dist.all_gather(gathered_probs,   probs_gpu)
        dist.all_gather(gathered_targets, tgts_gpu)
        dataset_size = len(loader.dataset)
        C = probs_gpu.shape[-1]
        full_probs   = torch.stack(gathered_probs,   dim=0).permute(1, 0, 2).reshape(-1, C)[:dataset_size].cpu()
        full_targets = torch.stack(gathered_targets, dim=0).permute(1, 0, 2).reshape(-1, C)[:dataset_size].cpu()
    else:
        full_probs   = all_probs_local
        full_targets = all_tgts_local

    return {
        "results": results,
        "loss": None,
        "probs": full_probs,
        "targets": full_targets,
    }


def train_one_epoch(
    cfg: Any,
    model: Dict[str, torch.nn.Module],
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    criterion: Callable,
    wd_scheduler: Any,
    epoch: int,
    max_epoch: int,
    logger: Optional[logging.Logger] = None,
    device: Optional[torch.device] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
    wandb_run: Optional[Any] = None,
    metrics: MetricCollection = None,
) -> Dict[str, float]:

















    metric_logger = MetricLogger(delimiter="  ", logger=logger)
    backbone, classifiers = model['backbone'], model['classifier']
    classifiers.train()

    use_amp = cfg.optimization.mixed_precision

    if metrics is None:
        raise ValueError("Pass a MetricCollection via `metrics` (use `build_monitor_metrics(num_labels, device)`).")

    for idx, batch_data in enumerate(loader):

        global_step = epoch * len(loader) + idx


        image, target = batch_data
        image = image.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)


        _new_lr = scheduler.step()
        _new_wd = wd_scheduler.step()


        losses = []
        logits = []
        with torch.amp.autocast('cuda', enabled=use_amp, dtype=torch.bfloat16):

            with torch.no_grad():
                feats = backbone(image)

            out_list = classifiers(feats)
            for c, logit_c in enumerate(out_list):
                logit_c = logit_c.squeeze(-1)
                logits.append(logit_c)
                target_c = target[:, c].float()
                losses.append(criterion(logit_c, target_c))
            loss = torch.stack(losses).mean()


        optimizer.zero_grad()

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()


        torch.cuda.synchronize()


        loss_value = all_reduce_mean(loss).item()


        if not math.isfinite(loss_value):
            logger.info("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        metric_logger.update(
            loss=loss_value,
            lr=_new_lr,
            wd=_new_wd,
            global_step=global_step,
        )

        with torch.no_grad():
            logits = torch.stack(logits, dim=1)
            probs = torch.sigmoid(logits)
            metrics.update(probs, target.int())


        if dist.get_rank() == 0 and wandb_run is not None:
            wandb_run.log({
                "train/loss": loss_value,
                "train/lr":   _new_lr,
                "train/wd":   _new_wd,
                "train/epoch": epoch + 1,
            })

        logger.info(f"Epoch {epoch+1}/{max_epoch} [{idx+1}/{len(loader)}]  Loss: {loss_value:.4f}")


    metric_logger.synchronize_between_processes()
    logger.info(f"Averaged stats: {metric_logger}")
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def compute_subgroup_metrics(
    probs: np.ndarray,
    targets: np.ndarray,
    demo_df: pd.DataFrame,
    demo_cols: List[str],
    class_names: List[str],
    wandb_run: Optional[Any],
    epoch: int,
    min_samples: int = 10,
) -> None:














    from sklearn.metrics import roc_auc_score, average_precision_score

    num_classes = len(class_names)

    for col in demo_cols:
        if col not in demo_df.columns:
            logger.warning(f"Demographic column '{col}' not found in test dataframe; skipping.")
            continue

        logger.info(f"\n=== Subgroup Analysis: {col} ===")
        subgroups = sorted(demo_df[col].dropna().unique().tolist(), key=str)

        for sg in subgroups:
            mask = (demo_df[col] == sg).values
            n = int(mask.sum())
            if n < min_samples:
                logger.info(f"  [{col}={sg}] N={n} — fewer than {min_samples} samples, skipping.")
                continue

            sg_probs   = probs[mask]
            sg_targets = targets[mask]

            auc_list, ap_list = [], []
            for c in range(num_classes):
                try:
                    auc = roc_auc_score(sg_targets[:, c], sg_probs[:, c])
                except ValueError:
                    auc = float("nan")
                try:
                    ap = average_precision_score(sg_targets[:, c], sg_probs[:, c])
                except ValueError:
                    ap = float("nan")
                auc_list.append(auc)
                ap_list.append(ap)

            valid_auc = [v for v in auc_list if not math.isnan(v)]
            valid_ap  = [v for v in ap_list  if not math.isnan(v)]
            macro_auc = float(np.mean(valid_auc)) if valid_auc else float("nan")
            macro_ap  = float(np.mean(valid_ap))  if valid_ap  else float("nan")

            macro_auc_str = f"{macro_auc:.3f}" if not math.isnan(macro_auc) else "N/A"
            macro_ap_str  = f"{macro_ap:.3f}"  if not math.isnan(macro_ap)  else "N/A"

            per_class_parts = []
            for i, name in enumerate(class_names):
                a_s = f"{auc_list[i]:.3f}" if not math.isnan(auc_list[i]) else "N/A"
                p_s = f"{ap_list[i]:.3f}"  if not math.isnan(ap_list[i])  else "N/A"
                per_class_parts.append(f"{name}: AUC={a_s}, AP={p_s}")

            logger.info(
                f"  [{col}={sg}] N={n} | "
                f"MACRO AUC={macro_auc_str}, AP={macro_ap_str} | " +
                ", ".join(per_class_parts)
            )

            if wandb_run is not None:
                sg_key = (
                    str(sg)
                    .replace(" ", "_")
                    .replace("(", "").replace(")", "")
                    .replace("–", "-").replace("/", "_")
                )
                log_dict: Dict[str, Any] = {"epoch": epoch + 1}
                if not math.isnan(macro_auc):
                    log_dict[f"test_subgroup/{col}/{sg_key}/auc_macro"] = macro_auc
                if not math.isnan(macro_ap):
                    log_dict[f"test_subgroup/{col}/{sg_key}/ap_macro"] = macro_ap
                for i, name in enumerate(class_names):
                    if not math.isnan(auc_list[i]):
                        log_dict[f"test_subgroup/{col}/{sg_key}/auc_{name}"] = auc_list[i]
                    if not math.isnan(ap_list[i]):
                        log_dict[f"test_subgroup/{col}/{sg_key}/ap_{name}"] = ap_list[i]
                wandb_run.log(log_dict)


def trainer(
    cfg: Any,
    model: Dict[str, torch.nn.Module],
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    criterion: Callable,
    wd_scheduler: Any,
    start_epoch: int = 0,
    max_epoch: int = 100,
    logger: Optional[logging.Logger] = None,
    device: Optional[torch.device] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
    wandb_run: Optional[Any] = None,
    test_df: Optional[pd.DataFrame] = None,
    demo_cols: Optional[List[str]] = None,
) -> float:



























    num_labels = cfg.data.num_classes
    train_metrics = build_monitor_metrics(num_labels=num_labels, device=device)
    val_metrics = build_monitor_metrics(num_labels=num_labels, device=device)
    test_metrics = build_monitor_metrics(num_labels=num_labels, device=device)

    class_names = cfg.data.label_cols

    best_val_auc_per_class = [-float("inf")] * num_labels
    best_head_state_per_class = [None] * num_labels

    for epoch in range(start_epoch, max_epoch):
        logger.info(f"Epoch: {epoch+1}")
        epoch_time = time.time()


        train_stats = train_one_epoch(
            cfg=cfg,
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            wd_scheduler=wd_scheduler,
            epoch=epoch,
            max_epoch=max_epoch,
            logger=logger,
            device=device,
            scaler=scaler,
            wandb_run=wandb_run,
            metrics=train_metrics,
        )

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}
        logger.info(f"log_stats: {log_stats}")

        if dist.get_rank() == 0:
            logger.info(
                f"Final training stats for epoch {epoch+1}/{max_epoch}: {log_stats}, "
                f"time: {time.time() - epoch_time:.2f}s"
            )


        train_results = train_metrics.compute()
        train_metrics.reset()
        logger.info(f"Training epoch {epoch+1} metrics:")
        log_metrics(train_stats["loss"], epoch, train_results, class_names, wandb_run, mode='train')


        val_stats = evaluate_epoch(
            cfg=cfg,
            model=model,
            loader=val_loader,
            device=device,
            metrics=val_metrics,
            logger=logger,
            use_amp=cfg.optimization.mixed_precision,
        )
        val_auc_pc = val_stats["results"]["auprc_perclass"].detach().cpu().tolist()

        for i in range(num_labels):
            if val_auc_pc[i] > best_val_auc_per_class[i]:
                best_val_auc_per_class[i] = val_auc_pc[i]
                best_head_state_per_class[i] = copy.deepcopy(model['classifier'].classifiers[i].state_dict())


        logger.info(f"Validation epoch {epoch+1} metrics:")
        log_metrics(val_stats['loss'], epoch, val_stats['results'], class_names, wandb_run, mode='val')



    classifiers_best = copy.deepcopy(model['classifier'])


    logger.info("Loading best heads per class for testing...")
    for i in range(num_labels):
        if best_head_state_per_class[i] is not None:
            classifiers_best.classifiers[i].load_state_dict(best_head_state_per_class[i])
        else:
            logger.info(f"Warning: No best head found for class {class_names[i]} (using last epoch head).")


    test_model = {"backbone": model["backbone"], "classifier": classifiers_best}
    test_stats = evaluate_epoch(
        cfg=cfg,
        model=test_model,
        loader=test_loader,
        device=device,
        metrics=test_metrics,
        logger=logger,
        use_amp=cfg.optimization.mixed_precision,
    )

    logger.info(f"Test epoch {epoch+1} metrics:")
    log_metrics(test_stats['loss'], epoch, test_stats['results'], class_names, wandb_run, mode='test')


    if dist.get_rank() == 0:
        pred_results = {}
        test_probs = test_stats['probs'].float()
        test_targets = test_stats['targets'].float()
        for i, name in enumerate(class_names):
            pred_results[name] = {
                'probs': test_probs[:, i].numpy(),
                'targets': test_targets[:, i].numpy(),
            }
        pred_results['probs_all'] = test_probs.numpy()
        pred_results['targets_all'] = test_targets.numpy()
        pred_results['class_names'] = class_names

        pkl_path = os.path.join(cfg.log.output_pkl_dir, f"{cfg.log.filename}.pkl")
        with open(pkl_path, 'wb') as f:
            pickle.dump(pred_results, f)
        logger.info(f"Test predictions saved to {pkl_path}")


        if demo_cols and test_df is not None:
            compute_subgroup_metrics(
                probs=test_probs.numpy(),
                targets=test_targets.numpy(),
                demo_df=test_df.reset_index(drop=True),
                demo_cols=demo_cols,
                class_names=class_names,
                wandb_run=wandb_run,
                epoch=epoch,
            )

    logger.info(f"Training Finished after {max_epoch} epochs.")
    return train_stats['loss']

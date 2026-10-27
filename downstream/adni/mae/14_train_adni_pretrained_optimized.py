"""Fine-tune pretrained MAE models on ADNI classification."""

import os
from pathlib import Path
import argparse
import importlib.util
import json
import math
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    ScaleIntensityRangePercentilesd,
    Resized,
    RandFlipd,
    EnsureTyped,
)
from monai.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, balanced_accuracy_score, confusion_matrix


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_train_module(script_path):
    spec = importlib.util.spec_from_file_location("adni_base_train_module", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def find_column(df, candidates, semantic):
    lower_map = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in lower_map:
            return lower_map[name.lower()]

    if semantic == "path":
        for col in df.columns:
            sample_values = df[col].dropna().astype(str).head(20).tolist()
            if any(value.endswith(".nii") or value.endswith(".nii.gz") for value in sample_values):
                return col

    raise RuntimeError(f"Could not find {semantic} column. Available columns: {list(df.columns)}")


def encode_label(value):
    if isinstance(value, str):
        text = value.strip().upper()
        if text in {"AD", "ALZHEIMER", "ALZHEIMER'S DISEASE", "1"}:
            return 1
        if text in {"CN", "CONTROL", "NORMAL", "0"}:
            return 0
    return int(value)


def read_split(csv_path, adni_root):
    df = pd.read_csv(csv_path)

    path_col = find_column(
        df,
        ["image_path", "nifti_path", "nii_path", "path", "file_path", "filename", "image", "scan_path"],
        "path",
    )
    label_col = find_column(
        df,
        ["label", "target", "class", "y", "diagnosis", "researchGroup", "research_group", "dx"],
        "label",
    )

    items = []
    for _, row in df.iterrows():
        image_path = Path(str(row[path_col]))
        if not image_path.is_absolute():
            image_path = Path(adni_root) / image_path

        label = encode_label(row[label_col])

        items.append({
            "image": str(image_path),
            "image_path": str(image_path),
            "label": label,
        })

    return items


def get_transforms(train, image_size, flip_prob):
    transforms = [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        ScaleIntensityRangePercentilesd(
            keys=["image"],
            lower=1,
            upper=99,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        Resized(
            keys=["image"],
            spatial_size=(image_size, image_size, image_size),
            mode="trilinear",
        ),
    ]

    if train and flip_prob > 0:
        transforms.append(RandFlipd(keys=["image"], prob=flip_prob, spatial_axis=0))

    transforms.append(EnsureTyped(keys=["image"]))
    return Compose(transforms)


def make_loaders(args):
    adni_root = Path(args.adni_root)
    train_csv = adni_root / "splits" / f"adni_ad_cn_fold{args.fold}_train.csv"
    val_csv = adni_root / "splits" / f"adni_ad_cn_fold{args.fold}_val.csv"

    train_items = read_split(train_csv, adni_root)
    val_items = read_split(val_csv, adni_root)

    train_labels = [item["label"] for item in train_items]
    class_counts = np.bincount(train_labels, minlength=2)
    class_weights = class_counts.sum() / (2.0 * np.maximum(class_counts, 1))
    class_weights = torch.tensor(class_weights, dtype=torch.float32)

    train_ds = Dataset(
        data=train_items,
        transform=get_transforms(train=True, image_size=args.image_size, flip_prob=args.flip_prob),
    )
    val_ds = Dataset(
        data=val_items,
        transform=get_transforms(train=False, image_size=args.image_size, flip_prob=0.0),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, class_weights, class_counts.tolist(), len(train_items), len(val_items)


def set_encoder_trainable(model, trainable):
    for name, parameter in model.named_parameters():
        if not name.startswith("head."):
            parameter.requires_grad = trainable


def build_optimizer(model, args):
    encoder_params = []
    head_params = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("head."):
            head_params.append(parameter)
        else:
            encoder_params.append(parameter)

    groups = []
    if encoder_params:
        groups.append({
            "params": encoder_params,
            "lr": args.encoder_lr,
            "weight_decay": args.weight_decay,
        })
    if head_params:
        groups.append({
            "params": head_params,
            "lr": args.head_lr,
            "weight_decay": args.head_weight_decay,
        })

    if not groups:
        raise RuntimeError("No trainable parameters found.")

    optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.999))
    return optimizer


def build_scheduler(optimizer, args, remaining_epochs):
    if args.scheduler == "none":
        return None

    if args.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, remaining_epochs),
            eta_min=args.min_lr,
        )

    raise RuntimeError(f"Unsupported scheduler: {args.scheduler}")


def compute_metrics(labels, probs, threshold=0.5):
    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs).astype(float)
    preds = (probs >= threshold).astype(int)

    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = float("nan")

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, zero_division=0)
    bal_acc = balanced_accuracy_score(labels, preds)

    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "auc": auc,
        "accuracy": acc,
        "f1": f1,
        "balanced_accuracy": bal_acc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp):
    model.eval()

    all_labels = []
    all_probs = []
    total_loss = 0.0
    total_items = 0

    for batch in loader:
        images = batch["image"].to(device)
        labels = torch.as_tensor(batch["label"], dtype=torch.long, device=device)

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        probs = torch.softmax(logits, dim=1)[:, 1]

        batch_size = labels.shape[0]
        total_loss += loss.item() * batch_size
        total_items += batch_size

        all_labels.extend(labels.detach().cpu().numpy().tolist())
        all_probs.extend(probs.detach().cpu().numpy().tolist())

    metrics = compute_metrics(all_labels, all_probs, threshold=0.5)
    metrics["loss"] = total_loss / max(total_items, 1)
    return metrics, all_labels, all_probs


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp):
    model.train()

    total_loss = 0.0
    total_items = 0
    all_labels = []
    all_probs = []

    for batch in loader:
        images = batch["image"].to(device)
        labels = torch.as_tensor(batch["label"], dtype=torch.long, device=device)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()

        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        scaler.step(optimizer)
        scaler.update()

        probs = torch.softmax(logits.detach(), dim=1)[:, 1]

        batch_size = labels.shape[0]
        total_loss += loss.item() * batch_size
        total_items += batch_size

        all_labels.extend(labels.detach().cpu().numpy().tolist())
        all_probs.extend(probs.detach().cpu().numpy().tolist())

    metrics = compute_metrics(all_labels, all_probs, threshold=0.5)
    metrics["loss"] = total_loss / max(total_items, 1)
    return metrics


@torch.no_grad()
def save_best_predictions(model, loader, criterion, device, use_amp, output_dir):
    model.eval()

    rows = []
    all_labels = []
    all_probs = []

    for batch in loader:
        images = batch["image"].to(device)
        labels = torch.as_tensor(batch["label"], dtype=torch.long, device=device)

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(images)
            _ = criterion(logits, labels)

        probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()

        paths = batch.get("image_path", [""] * len(labels_np))
        if isinstance(paths, str):
            paths = [paths]

        for path, label, prob in zip(paths, labels_np, probs):
            rows.append({
                "image_path": str(path),
                "label": int(label),
                "prob_ad": float(prob),
                "pred_0p5": int(prob >= 0.5),
            })
            all_labels.append(int(label))
            all_probs.append(float(prob))

    pred_df = pd.DataFrame(rows)
    pred_path = output_dir / "val_predictions_best.csv"
    pred_df.to_csv(pred_path, index=False)

    threshold_rows = []
    for threshold in np.arange(0.05, 0.951, 0.005):
        metrics = compute_metrics(all_labels, all_probs, threshold=float(threshold))
        threshold_rows.append({
            "threshold": float(threshold),
            **metrics,
            "selection_score": metrics["balanced_accuracy"],
        })

    threshold_df = pd.DataFrame(threshold_rows)
    threshold_path = output_dir / "threshold_metrics_best.csv"
    threshold_df.to_csv(threshold_path, index=False)

    best_row = threshold_df.sort_values(["balanced_accuracy", "f1"], ascending=False).iloc[0]

    print("Prediction file:", pred_path)
    print("Threshold metrics file:", threshold_path)
    print("Best threshold by balanced accuracy:")
    print(best_row.to_string())


def main(args):
    set_seed(args.seed)

    adni_root = Path(args.adni_root)
    output_dir = adni_root / "classification_outputs" / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "config.json").open("w") as f:
        json.dump(vars(args), f, indent=2)

    train_module = load_train_module(adni_root / "scripts" / "11_train_adni_classifier.py")

    model = train_module.MAEViTClassifier3D(
        num_classes=2,
        img_size=args.image_size,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
    )

    load_report = train_module.load_pretrained_encoder(model, args.pretrained_ckpt)
    print("Loaded pretrained tensors:", load_report.get("loaded_tensors"))
    print("Skipped checkpoint tensors:", load_report.get("skipped_tensors"))
    print("Missing model tensors:", load_report.get("missing_tensors"))
    print("Unexpected tensors:", load_report.get("unexpected_tensors"))

    train_loader, val_loader, class_weights, class_counts, n_train, n_val = make_loaders(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = args.amp and device.type == "cuda"

    print("Device:", device)
    print("Train items:", n_train)
    print("Val items:", n_val)
    print("Train class counts:", class_counts)
    print("Class weights:", class_weights.tolist())

    model = model.to(device)
    class_weights = class_weights.to(device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=args.label_smoothing,
    )

    if args.freeze_encoder_epochs > 0:
        set_encoder_trainable(model, False)
    else:
        set_encoder_trainable(model, True)

    optimizer = build_optimizer(model, args)
    scheduler = build_scheduler(optimizer, args, args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_auc = -math.inf
    best_epoch = -1
    bad_epochs = 0
    metrics_rows = []

    for epoch in range(args.epochs):
        if epoch == args.freeze_encoder_epochs and args.freeze_encoder_epochs > 0:
            print("Unfreezing encoder")
            set_encoder_trainable(model, True)
            optimizer = build_optimizer(model, args)
            scheduler = build_scheduler(optimizer, args, args.epochs - epoch)
            scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        start_time = time.time()

        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, use_amp)
        val_metrics, _, _ = evaluate(model, val_loader, criterion, device, use_amp)

        if scheduler is not None:
            scheduler.step()

        elapsed = time.time() - start_time
        current_lrs = [group["lr"] for group in optimizer.param_groups]

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_auc": train_metrics["auc"],
            "train_accuracy": train_metrics["accuracy"],
            "train_f1": train_metrics["f1"],
            "train_balanced_accuracy": train_metrics["balanced_accuracy"],
            "val_loss": val_metrics["loss"],
            "val_auc": val_metrics["auc"],
            "val_accuracy": val_metrics["accuracy"],
            "val_f1": val_metrics["f1"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "val_sensitivity": val_metrics["sensitivity"],
            "val_specificity": val_metrics["specificity"],
            "seconds": elapsed,
            "lrs": "|".join(f"{lr:.8g}" for lr in current_lrs),
        }
        metrics_rows.append(row)
        pd.DataFrame(metrics_rows).to_csv(output_dir / "metrics.csv", index=False)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_auc={val_metrics['auc']:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"val_f1={val_metrics['f1']:.4f} | "
            f"val_bal_acc={val_metrics['balanced_accuracy']:.4f} | "
            f"time={elapsed:.1f}s | "
            f"lrs={row['lrs']}"
        )

        improved = val_metrics["auc"] > best_auc
        if improved:
            best_auc = val_metrics["auc"]
            best_epoch = epoch
            bad_epochs = 0

            torch.save(
                {
                    "epoch": epoch,
                    "best_val_auc": best_auc,
                    "state_dict": model.state_dict(),
                    "model_state_dict": model.state_dict(),
                    "config": vars(args),
                    "load_report": load_report,
                },
                output_dir / "best_model.pth.tar",
            )
        else:
            bad_epochs += 1

        if args.patience > 0 and bad_epochs >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break

    best_path = output_dir / "best_model.pth.tar"
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    print("Best val AUC:", best_auc)
    print("Best epoch:", best_epoch)
    print("Output directory:", output_dir)

    save_best_predictions(model, val_loader, criterion, device, use_amp, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--adni_root", type=str, default=os.environ.get("BRAINMRI_ADNI_ROOT", str(Path(__file__).resolve().parents[3] / "data/adni")))
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--pretrained_ckpt", type=str, required=True)

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--image_size", type=int, default=96)

    parser.add_argument("--encoder_lr", type=float, default=1e-5)
    parser.add_argument("--head_lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=1e-7)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--head_weight_decay", type=float, default=0.0)
    parser.add_argument("--label_smoothing", type=float, default=0.05)

    parser.add_argument("--freeze_encoder_epochs", type=int, default=3)
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "none"])
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--flip_prob", type=float, default=0.5)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")

    args = parser.parse_args()
    main(args)

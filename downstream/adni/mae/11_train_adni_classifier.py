"""Train and evaluate ADNI classification models."""

import os
from pathlib import Path
import argparse
import json
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
)

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    ScaleIntensityRangePercentilesd,
    Resized,
    RandFlipd,
    ToTensord,
)
from monai.data import Dataset, DataLoader


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim=768, num_heads=12, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        batch_size, num_tokens, dim = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, num_tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ v
        x = x.transpose(1, 2).reshape(batch_size, num_tokens, dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim=768, num_heads=12, mlp_ratio=4.0, drop=0.0, attn_drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim=dim, num_heads=num_heads, qkv_bias=True, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), out_features=dim, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed3D(nn.Module):
    def __init__(self, img_size=96, patch_size=16, in_chans=1, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 3
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class ViTEncoder3D(nn.Module):
    def __init__(self, img_size=96, patch_size=16, in_chans=1, embed_dim=768, depth=12, num_heads=12):
        super().__init__()
        self.patch_embed = PatchEmbed3D(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.0)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x, encoder_pos_embed):
        x = self.patch_embed(x)
        x = x + encoder_pos_embed

        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return x[:, 0]


class MAEViTClassifier3D(nn.Module):
    def __init__(self, num_classes=2, img_size=96, patch_size=16, embed_dim=768, depth=12, num_heads=12):
        super().__init__()
        num_patches = (img_size // patch_size) ** 3
        self.encoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.encoder = ViTEncoder3D(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=1,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
        )
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.encoder_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        features = self.encoder(x, self.encoder_pos_embed)
        logits = self.head(features)
        return logits


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_items(csv_path):
    df = pd.read_csv(csv_path)
    items = []
    for _, row in df.iterrows():
        items.append({
            "image": row["nifti_path"],
            "label": int(row["label"]),
            "subject_id": row["subject_id"],
            "label_name": row["label_name"],
        })
    return items


def make_transforms(train, image_size):
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
        Resized(keys=["image"], spatial_size=(image_size, image_size, image_size), mode="trilinear"),
    ]

    if train:
        transforms.extend([
            RandFlipd(keys=["image"], spatial_axis=0, prob=0.5),
        ])

    transforms.append(ToTensord(keys=["image", "label"]))
    return Compose(transforms)


def load_pretrained_encoder(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    model_state = model.state_dict()

    filtered = {}
    skipped = []

    for key, value in state.items():
        if key.startswith("module."):
            key = key[len("module."):]

        if key.startswith("decoder") or key.startswith("mask_token"):
            skipped.append(key)
            continue

        if key in model_state and hasattr(value, "shape") and model_state[key].shape == value.shape:
            filtered[key] = value
        else:
            skipped.append(key)

    missing, unexpected = model.load_state_dict(filtered, strict=False)

    print(f"Loaded pretrained tensors: {len(filtered)}")
    print(f"Skipped checkpoint tensors: {len(skipped)}")
    print(f"Missing model tensors: {len(missing)}")
    print(f"Unexpected tensors: {len(unexpected)}")

    print("First missing tensors:")
    for key in list(missing)[:30]:
        print(key)

    return {
        "loaded_tensors": len(filtered),
        "skipped_tensors": len(skipped),
        "missing_tensors": len(missing),
        "unexpected_tensors": len(unexpected),
    }


def compute_metrics(y_true, y_prob):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {}

    try:
        metrics["auc"] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        metrics["auc"] = float("nan")

    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    metrics["sensitivity"] = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
    metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")
    metrics["tp"] = int(tp)
    metrics["tn"] = int(tn)
    metrics["fp"] = int(fp)
    metrics["fn"] = int(fn)

    return metrics


def run_epoch(model, loader, criterion, optimizer, device, scaler, train):
    model.train(train)

    losses = []
    y_true = []
    y_prob = []

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].long().to(device)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            with torch.amp.autocast(device_type="cuda", enabled=(scaler is not None)):
                logits = model(images)
                loss = criterion(logits, labels)

            if train:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        probs = torch.softmax(logits.detach(), dim=1)[:, 1]
        losses.append(float(loss.detach().cpu()))
        y_true.extend(labels.detach().cpu().numpy().tolist())
        y_prob.extend(probs.cpu().numpy().tolist())

    metrics = compute_metrics(y_true, y_prob)
    metrics["loss"] = float(np.mean(losses)) if losses else float("nan")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adni_root", type=str, default=os.environ.get("BRAINMRI_ADNI_ROOT", str(Path(__file__).resolve().parents[3] / "data/adni")))
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--mode", type=str, choices=["scratch", "pretrained"], required=True)
    parser.add_argument("--pretrained_ckpt", type=str, default="")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--image_size", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--encoder_lr", type=float, default=0.0)
    parser.add_argument("--head_lr", type=float, default=0.0)
    args = parser.parse_args()

    set_seed(args.seed)

    adni_root = Path(args.adni_root)
    train_csv = adni_root / "splits" / f"adni_ad_cn_fold{args.fold}_train.csv"
    val_csv = adni_root / "splits" / f"adni_ad_cn_fold{args.fold}_val.csv"

    run_name = args.run_name if args.run_name else f"{args.mode}_fold{args.fold}_epochs{args.epochs}"
    output_dir = adni_root / "classification_outputs" / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    train_items = read_items(train_csv)
    val_items = read_items(val_csv)

    train_subjects = {item["subject_id"] for item in train_items}
    val_subjects = {item["subject_id"] for item in val_items}
    overlap = train_subjects.intersection(val_subjects)
    if overlap:
        raise RuntimeError(f"Subject overlap detected: {sorted(list(overlap))[:20]}")

    train_labels = [item["label"] for item in train_items]
    class_counts = np.bincount(train_labels, minlength=2)
    class_weights = class_counts.sum() / (2.0 * np.maximum(class_counts, 1))
    class_weights = torch.tensor(class_weights, dtype=torch.float32)

    print(f"Train items: {len(train_items)}")
    print(f"Val items: {len(val_items)}")
    print(f"Train class counts: {class_counts.tolist()}")
    print(f"Class weights: {class_weights.tolist()}")

    train_dataset = Dataset(data=train_items, transform=make_transforms(train=True, image_size=args.image_size))
    val_dataset = Dataset(data=val_items, transform=make_transforms(train=False, image_size=args.image_size))

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = MAEViTClassifier3D(
        num_classes=2,
        img_size=args.image_size,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
    )

    load_report = None
    if args.mode == "pretrained":
        if not args.pretrained_ckpt:
            raise ValueError("--pretrained_ckpt is required for pretrained mode")
        load_report = load_pretrained_encoder(model, args.pretrained_ckpt)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    if args.encoder_lr > 0.0 or args.head_lr > 0.0:
        encoder_lr = args.encoder_lr if args.encoder_lr > 0.0 else args.lr
        head_lr = args.head_lr if args.head_lr > 0.0 else args.lr
        optimizer = torch.optim.AdamW(
            [
                {"params": list(model.encoder.parameters()) + [model.encoder_pos_embed], "lr": encoder_lr},
                {"params": model.head.parameters(), "lr": head_lr},
            ],
            weight_decay=args.weight_decay,
        )
        print(f"Encoder LR: {encoder_lr}")
        print(f"Head LR: {head_lr}")
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda") if args.amp and device.type == "cuda" else None

    best_auc = -1.0
    history = []

    config = vars(args).copy()
    config["train_csv"] = str(train_csv)
    config["val_csv"] = str(val_csv)
    config["output_dir"] = str(output_dir)
    config["load_report"] = load_report

    with open(output_dir / "config.json", "w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)

    for epoch in range(args.epochs):
        start_time = time.time()

        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            train=True,
        )

        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=None,
            train=False,
        )

        row = {"epoch": epoch}
        for key, value in train_metrics.items():
            row[f"train_{key}"] = value
        for key, value in val_metrics.items():
            row[f"val_{key}"] = value
        row["seconds"] = time.time() - start_time
        history.append(row)

        pd.DataFrame(history).to_csv(output_dir / "metrics.csv", index=False)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_auc={val_metrics['auc']:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"val_f1={val_metrics['f1']:.4f} | "
            f"time={row['seconds']:.1f}s"
        )

        val_auc = val_metrics["auc"]
        if not np.isnan(val_auc) and val_auc > best_auc:
            best_auc = val_auc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_auc": best_auc,
                    "config": config,
                    "val_metrics": val_metrics,
                },
                output_dir / "best_model.pth.tar",
            )

    torch.save(
        {
            "epoch": args.epochs - 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_auc": best_auc,
            "config": config,
        },
        output_dir / "last_model.pth.tar",
    )

    print(f"Best val AUC: {best_auc:.4f}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()

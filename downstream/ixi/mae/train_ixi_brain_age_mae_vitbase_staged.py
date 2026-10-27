"""Train the MAE-based model for IXI brain-age regression."""

from pathlib import Path
import argparse
import json
import math
import random
import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class IXIBrainAgeDataset(Dataset):
    def __init__(self, csv_path, age_mean=None, age_std=None, training=False):
        self.csv_path = str(csv_path)
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.training = training

        if "image_npy" not in self.df.columns:
            raise RuntimeError(f"Missing image_npy column in {csv_path}")
        if "age" not in self.df.columns:
            raise RuntimeError(f"Missing age column in {csv_path}")

        self.ages = self.df["age"].astype(float).values

        if age_mean is None:
            age_mean = float(np.mean(self.ages))
        if age_std is None:
            age_std = float(np.std(self.ages))
        if age_std < 1e-8:
            age_std = 1.0

        self.age_mean = float(age_mean)
        self.age_std = float(age_std)

    def __len__(self):
        return len(self.df)

    def _augment(self, x):
        if random.random() < 0.5:
            x = np.flip(x, axis=1).copy()
        if random.random() < 0.5:
            x = np.flip(x, axis=2).copy()
        if random.random() < 0.5:
            scale = random.uniform(0.9, 1.1)
            shift = random.uniform(-0.1, 0.1)
            x = x * scale + shift
        if random.random() < 0.25:
            noise = np.random.normal(0.0, 0.02, size=x.shape).astype(np.float32)
            x = x + noise
        return x.astype(np.float32)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = np.load(row["image_npy"]).astype(np.float32)

        if x.shape != (1, 96, 96, 96):
            raise RuntimeError(f"Unexpected npy shape {x.shape} for {row['image_npy']}")

        if self.training:
            x = self._augment(x)

        age = float(row["age"])
        y = (age - self.age_mean) / self.age_std

        return {
            "image": torch.from_numpy(x),
            "target": torch.tensor(y, dtype=torch.float32),
            "age": torch.tensor(age, dtype=torch.float32),
            "subject_id": str(row.get("subject_id", idx)),
        }


class PatchEmbed3D(nn.Module):
    def __init__(self, img_size=96, patch_size=16, in_chans=1, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 3
        self.proj = nn.Conv3d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
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
    def __init__(self, dim, num_heads=12, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ v
        x = x.transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True, drop=0.0, attn_drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            out_features=dim,
            drop=drop,
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Encoder(nn.Module):
    def __init__(self, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, drop=0.0, attn_drop=0.0):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    drop=drop,
                    attn_drop=attn_drop,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x


class MAEViTBase3DRegressor(nn.Module):
    def __init__(
        self,
        img_size=96,
        patch_size=16,
        in_chans=1,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        dropout=0.1,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed3D(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        self.encoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches, embed_dim)
        )
        self.encoder = Encoder(
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop=dropout,
            attn_drop=0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1),
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.encoder_pos_embed, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.encoder_pos_embed
        x = self.encoder(x)
        x = x.mean(dim=1)
        x = self.head(x).squeeze(1)
        return x


def load_torch_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def choose_state_dict(raw):
    if isinstance(raw, dict):
        for key in ["state_dict", "model", "model_state_dict", "net", "encoder"]:
            if key in raw and isinstance(raw[key], dict):
                return raw[key]
    if isinstance(raw, dict):
        tensor_items = {k: v for k, v in raw.items() if torch.is_tensor(v)}
        if tensor_items:
            return raw
    raise RuntimeError("Could not find a valid state_dict in checkpoint.")


def clean_key(key):
    k = str(key)
    changed = True
    while changed:
        changed = False
        for prefix in ["module.", "model.", "net.", "backbone."]:
            if k.startswith(prefix):
                k = k[len(prefix):]
                changed = True

    if k.startswith("encoder.patch_embed."):
        k = k.replace("encoder.patch_embed.", "patch_embed.", 1)
    if k.startswith("encoder.encoder_pos_embed"):
        k = k.replace("encoder.encoder_pos_embed", "encoder_pos_embed", 1)
    if k.startswith("blocks."):
        k = "encoder." + k
    if k.startswith("norm."):
        k = "encoder." + k
    if k == "pos_embed":
        k = "encoder_pos_embed"

    return k


def load_pretrained_encoder(model, checkpoint_path, min_matched=50):
    raw = load_torch_checkpoint(checkpoint_path)
    state = choose_state_dict(raw)
    model_state = model.state_dict()

    load_state = {}
    skipped_shape = []
    skipped_name = []

    for key, value in state.items():
        if not torch.is_tensor(value):
            continue

        cleaned = clean_key(key)

        if cleaned.startswith("decoder") or "decoder" in cleaned:
            skipped_name.append((key, cleaned))
            continue
        if cleaned.startswith("mask_token") or cleaned.startswith("head"):
            skipped_name.append((key, cleaned))
            continue

        if cleaned in model_state and tuple(model_state[cleaned].shape) == tuple(value.shape):
            load_state[cleaned] = value
        elif cleaned in model_state:
            skipped_shape.append((key, cleaned, tuple(value.shape), tuple(model_state[cleaned].shape)))
        else:
            skipped_name.append((key, cleaned))

    missing, unexpected = model.load_state_dict(load_state, strict=False)

    print("pretrained_checkpoint:", checkpoint_path)
    print("checkpoint_total_tensor_keys:", sum(1 for v in state.values() if torch.is_tensor(v)))
    print("matched_tensor_keys:", len(load_state))
    print("missing_after_partial_load:", len(missing))
    print("unexpected_after_partial_load:", len(unexpected))
    print("skipped_shape_count:", len(skipped_shape))
    print("skipped_name_count:", len(skipped_name))

    print("first_matched_keys:")
    for k in list(load_state.keys())[:30]:
        print("  " + k)

    if len(load_state) < min_matched:
        raise RuntimeError(f"Too few pretrained keys matched: {len(load_state)}")

    return {
        "checkpoint_path": str(checkpoint_path),
        "matched_tensor_keys": len(load_state),
        "checkpoint_total_tensor_keys": sum(1 for v in state.values() if torch.is_tensor(v)),
        "skipped_shape_count": len(skipped_shape),
        "skipped_name_count": len(skipped_name),
    }


def freeze_encoder_train_head(model):
    for p in model.parameters():
        p.requires_grad = False
    for p in model.head.parameters():
        p.requires_grad = True


def unfreeze_all(model):
    for p in model.parameters():
        p.requires_grad = True


def count_trainable_parameters(model):
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def make_head_optimizer(model, lr, weight_decay):
    return torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
    )


def make_finetune_optimizer(model, encoder_lr, head_lr, weight_decay):
    head_param_ids = set(id(p) for p in model.head.parameters())

    encoder_params = []
    head_params = []

    for p in model.parameters():
        if not p.requires_grad:
            continue
        if id(p) in head_param_ids:
            head_params.append(p)
        else:
            encoder_params.append(p)

    return torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": encoder_lr},
            {"params": head_params, "lr": head_lr},
        ],
        weight_decay=weight_decay,
    )


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    if len(y_true) > 2 and np.std(y_pred) > 1e-8:
        r = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        r = float("nan")

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "pearson_r": r,
    }


@torch.no_grad()
def evaluate(model, loader, device, age_mean, age_std):
    model.eval()
    preds = []
    targets = []

    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        age = batch["age"].cpu().numpy()

        pred_z = model(x).detach().cpu().numpy()
        pred_age = pred_z * age_std + age_mean

        preds.extend(pred_age.tolist())
        targets.extend(age.tolist())

    metrics = compute_metrics(targets, preds)
    return metrics, np.asarray(targets), np.asarray(preds)


def save_predictions(path, dataset, targets, preds):
    df = dataset.df.copy()
    df["age_true"] = targets
    df["age_pred"] = preds
    df["age_abs_error"] = np.abs(df["age_true"] - df["age_pred"])
    df.to_csv(path, index=False)


def train_one_epoch(model, loader, optimizer, criterion, scaler, device, grad_accum_steps, use_amp, stage_name, epoch, total_epochs):
    model.train()
    losses = []
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(loader, desc=f"{stage_name} {epoch}/{total_epochs}", leave=False)

    for step, batch in enumerate(pbar, start=1):
        x = batch["image"].to(device, non_blocking=True)
        y = batch["target"].to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            pred = model(x)
            loss = criterion(pred, y)
            loss = loss / grad_accum_steps

        scaler.scale(loss).backward()

        if step % grad_accum_steps == 0 or step == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=1.0,
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        losses.append(float(loss.item() * grad_accum_steps))
        pbar.set_postfix(loss=np.mean(losses))

    return float(np.mean(losses))


def maybe_save_best(model, optimizer, out_dir, epoch_global, stage, val_metrics, val_targets, val_preds, val_ds, best_state, age_mean, age_std, config):
    if val_metrics["mae"] < best_state["best_val_mae"]:
        best_state["best_val_mae"] = float(val_metrics["mae"])
        best_state["best_epoch"] = int(epoch_global)
        best_state["best_stage"] = stage
        best_state["bad_epochs"] = 0

        torch.save(
            {
                "epoch": int(epoch_global),
                "stage": stage,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_val_mae": float(val_metrics["mae"]),
                "age_mean": age_mean,
                "age_std": age_std,
                "config": config,
            },
            out_dir / "best_model.pt",
        )
        save_predictions(out_dir / "val_predictions_best.csv", val_ds, val_targets, val_preds)
        return True

    best_state["bad_epochs"] += 1
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--val_csv", required=True)
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--pretrained_checkpoint", required=True)
    parser.add_argument("--warmup_epochs", type=int, default=15)
    parser.add_argument("--stage2_epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--head_warmup_lr", type=float, default=1e-3)
    parser.add_argument("--encoder_lr", type=float, default=5e-6)
    parser.add_argument("--head_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=35)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dropout", type=float, default=0.1)
    args = parser.parse_args()

    set_seed(args.seed + args.fold)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    if torch.cuda.is_available():
        print("cuda_device_name:", torch.cuda.get_device_name(0))

    train_ds = IXIBrainAgeDataset(args.train_csv, training=True)
    age_mean = train_ds.age_mean
    age_std = train_ds.age_std

    val_ds = IXIBrainAgeDataset(args.val_csv, age_mean=age_mean, age_std=age_std, training=False)
    test_ds = IXIBrainAgeDataset(args.test_csv, age_mean=age_mean, age_std=age_std, training=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model = MAEViTBase3DRegressor(
        img_size=96,
        patch_size=16,
        in_chans=1,
        embed_dim=768,
        depth=12,
        num_heads=12,
        dropout=args.dropout,
    )

    pretrain_info = load_pretrained_encoder(model, args.pretrained_checkpoint)
    model = model.to(device)

    criterion = nn.SmoothL1Loss(beta=1.0)
    use_amp = torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    config = vars(args).copy()
    config["age_mean"] = age_mean
    config["age_std"] = age_std
    config["device"] = str(device)
    config["train_n"] = len(train_ds)
    config["val_n"] = len(val_ds)
    config["test_n"] = len(test_ds)
    config["architecture"] = "mae_vit_base_3d_patch16_96"
    config["pretrain_info"] = pretrain_info
    config["training_strategy"] = "head_warmup_then_encoder_low_lr"

    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    history = []
    best_state = {
        "best_val_mae": float("inf"),
        "best_epoch": -1,
        "best_stage": "",
        "bad_epochs": 0,
    }

    epoch_global = 0

    freeze_encoder_train_head(model)
    print("stage:", "head_warmup")
    print("trainable_parameters:", count_trainable_parameters(model))

    optimizer = make_head_optimizer(model, args.head_warmup_lr, args.weight_decay)

    for epoch in range(1, args.warmup_epochs + 1):
        epoch_global += 1
        start = time.time()

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
            device=device,
            grad_accum_steps=args.grad_accum_steps,
            use_amp=use_amp,
            stage_name="warmup",
            epoch=epoch,
            total_epochs=args.warmup_epochs,
        )

        val_metrics, val_targets, val_preds = evaluate(model, val_loader, device, age_mean, age_std)

        row = {
            "global_epoch": epoch_global,
            "stage": "head_warmup",
            "stage_epoch": epoch,
            "train_loss": train_loss,
            "val_mae": val_metrics["mae"],
            "val_rmse": val_metrics["rmse"],
            "val_r2": val_metrics["r2"],
            "val_pearson_r": val_metrics["pearson_r"],
            "seconds": float(time.time() - start),
            "trainable_parameters": count_trainable_parameters(model),
        }
        history.append(row)
        pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)

        maybe_save_best(
            model=model,
            optimizer=optimizer,
            out_dir=out_dir,
            epoch_global=epoch_global,
            stage="head_warmup",
            val_metrics=val_metrics,
            val_targets=val_targets,
            val_preds=val_preds,
            val_ds=val_ds,
            best_state=best_state,
            age_mean=age_mean,
            age_std=age_std,
            config=config,
        )

        print(
            f"global_epoch={epoch_global} "
            f"stage=head_warmup "
            f"stage_epoch={epoch} "
            f"train_loss={row['train_loss']:.5f} "
            f"val_mae={row['val_mae']:.4f} "
            f"val_rmse={row['val_rmse']:.4f} "
            f"val_r2={row['val_r2']:.4f} "
            f"val_r={row['val_pearson_r']:.4f}"
        )

    unfreeze_all(model)
    print("stage:", "encoder_low_lr_finetune")
    print("trainable_parameters:", count_trainable_parameters(model))

    optimizer = make_finetune_optimizer(
        model=model,
        encoder_lr=args.encoder_lr,
        head_lr=args.head_lr,
        weight_decay=args.weight_decay,
    )

    best_state["bad_epochs"] = 0

    for epoch in range(1, args.stage2_epochs + 1):
        epoch_global += 1
        start = time.time()

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
            device=device,
            grad_accum_steps=args.grad_accum_steps,
            use_amp=use_amp,
            stage_name="finetune",
            epoch=epoch,
            total_epochs=args.stage2_epochs,
        )

        val_metrics, val_targets, val_preds = evaluate(model, val_loader, device, age_mean, age_std)

        row = {
            "global_epoch": epoch_global,
            "stage": "encoder_low_lr_finetune",
            "stage_epoch": epoch,
            "train_loss": train_loss,
            "val_mae": val_metrics["mae"],
            "val_rmse": val_metrics["rmse"],
            "val_r2": val_metrics["r2"],
            "val_pearson_r": val_metrics["pearson_r"],
            "seconds": float(time.time() - start),
            "trainable_parameters": count_trainable_parameters(model),
        }
        history.append(row)
        pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)

        improved = maybe_save_best(
            model=model,
            optimizer=optimizer,
            out_dir=out_dir,
            epoch_global=epoch_global,
            stage="encoder_low_lr_finetune",
            val_metrics=val_metrics,
            val_targets=val_targets,
            val_preds=val_preds,
            val_ds=val_ds,
            best_state=best_state,
            age_mean=age_mean,
            age_std=age_std,
            config=config,
        )

        print(
            f"global_epoch={epoch_global} "
            f"stage=encoder_low_lr_finetune "
            f"stage_epoch={epoch} "
            f"train_loss={row['train_loss']:.5f} "
            f"val_mae={row['val_mae']:.4f} "
            f"val_rmse={row['val_rmse']:.4f} "
            f"val_r2={row['val_r2']:.4f} "
            f"val_r={row['val_pearson_r']:.4f}"
        )

        if best_state["bad_epochs"] >= args.patience:
            print("early_stopping_triggered:", epoch_global)
            break

    ckpt = torch.load(out_dir / "best_model.pt", map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)

    val_metrics, val_targets, val_preds = evaluate(model, val_loader, device, age_mean, age_std)
    test_metrics, test_targets, test_preds = evaluate(model, test_loader, device, age_mean, age_std)

    save_predictions(out_dir / "val_predictions_final_best.csv", val_ds, val_targets, val_preds)
    save_predictions(out_dir / "test_predictions_final_best.csv", test_ds, test_targets, test_preds)

    summary = {
        "run_name": args.run_name,
        "fold": args.fold,
        "architecture": "mae_vit_base_3d_patch16_96",
        "training_strategy": "head_warmup_then_encoder_low_lr",
        "pretrained_checkpoint": args.pretrained_checkpoint,
        "best_epoch": int(best_state["best_epoch"]),
        "best_stage": best_state["best_stage"],
        "best_val_mae": float(best_state["best_val_mae"]),
        "val_mae": val_metrics["mae"],
        "val_rmse": val_metrics["rmse"],
        "val_r2": val_metrics["r2"],
        "val_pearson_r": val_metrics["pearson_r"],
        "test_mae": test_metrics["mae"],
        "test_rmse": test_metrics["rmse"],
        "test_r2": test_metrics["r2"],
        "test_pearson_r": test_metrics["pearson_r"],
        "train_n": len(train_ds),
        "val_n": len(val_ds),
        "test_n": len(test_ds),
        "warmup_epochs": args.warmup_epochs,
        "stage2_epochs": args.stage2_epochs,
        "head_warmup_lr": args.head_warmup_lr,
        "encoder_lr": args.encoder_lr,
        "head_lr": args.head_lr,
    }

    pd.DataFrame([summary]).to_csv(out_dir / "summary.csv", index=False)

    print()
    print("Final summary:")
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()

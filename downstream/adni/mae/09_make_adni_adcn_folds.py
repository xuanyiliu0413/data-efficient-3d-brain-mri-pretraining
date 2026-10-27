"""Prepare subject-level ADNI classification splits."""

import os
from pathlib import Path
import pandas as pd

adni_root = Path(os.environ.get("BRAINMRI_ADNI_ROOT", str(Path(__file__).resolve().parents[3] / "data/adni")))
split_dir = adni_root / "splits"
input_path = split_dir / "adni_ad_cn_subject_one_scan.csv"

df = pd.read_csv(input_path)
df = df.drop_duplicates("subject_id").copy()
df = df[df["label_name"].isin(["AD", "CN"])].copy()

n_splits = 5
seed = 42

try:
    from sklearn.model_selection import StratifiedKFold
except ImportError as error:
    raise ImportError("scikit-learn is required. Install it with: pip install scikit-learn") from error

skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

summary_rows = []

for fold, (train_idx, val_idx) in enumerate(skf.split(df["subject_id"], df["label"])):
    train_df = df.iloc[train_idx].copy()
    val_df = df.iloc[val_idx].copy()

    train_path = split_dir / f"adni_ad_cn_fold{fold}_train.csv"
    val_path = split_dir / f"adni_ad_cn_fold{fold}_val.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    train_counts = train_df["label_name"].value_counts().to_dict()
    val_counts = val_df["label_name"].value_counts().to_dict()

    summary_rows.append({
        "fold": fold,
        "train_n": len(train_df),
        "val_n": len(val_df),
        "train_CN": train_counts.get("CN", 0),
        "train_AD": train_counts.get("AD", 0),
        "val_CN": val_counts.get("CN", 0),
        "val_AD": val_counts.get("AD", 0),
        "train_path": str(train_path),
        "val_path": str(val_path),
    })

summary = pd.DataFrame(summary_rows)
summary_path = split_dir / "adni_ad_cn_5fold_summary.csv"
summary.to_csv(summary_path, index=False)

print(summary.to_string(index=False))
print(f"Saved summary to: {summary_path}")

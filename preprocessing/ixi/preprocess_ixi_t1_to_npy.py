"""Prepare IXI T1 MRI inputs for brain-age experiments."""

from pathlib import Path
from multiprocessing import Pool, cpu_count
import os
import traceback

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import zoom
from tqdm import tqdm

PROJECT_ROOT = Path(os.environ.get("BRAINMRI_PROJECT_ROOT", str(Path(__file__).resolve().parents[2])))
IXI_ROOT = PROJECT_ROOT / "data/downstream/IXI"
MANIFEST_DIR = IXI_ROOT / "processed/t1_manifest_clean"
OUT_ROOT = IXI_ROOT / "processed/t1_npy_96"
NPY_DIR = OUT_ROOT / "volumes"
SPLIT_OUT_DIR = OUT_ROOT / "splits"

TARGET_SHAPE = (96, 96, 96)
LOWER_PERCENTILE = 0.5
UPPER_PERCENTILE = 99.5
EPS = 1e-8

NPY_DIR.mkdir(parents=True, exist_ok=True)
SPLIT_OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_volume(path):
    img = nib.load(str(path))
    img = nib.as_closest_canonical(img)
    arr = img.get_fdata(dtype=np.float32)
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def robust_clip(arr):
    mask = np.isfinite(arr) & (np.abs(arr) > EPS)
    if mask.sum() < 100:
        mask = np.isfinite(arr)

    values = arr[mask]
    lo = np.percentile(values, LOWER_PERCENTILE)
    hi = np.percentile(values, UPPER_PERCENTILE)

    if hi <= lo:
        return arr.astype(np.float32)

    arr = np.clip(arr, lo, hi)
    return arr.astype(np.float32)


def resize_volume(arr, target_shape):
    factors = [target_shape[i] / arr.shape[i] for i in range(3)]
    arr = zoom(arr, zoom=factors, order=1)
    arr = arr.astype(np.float32)

    if arr.shape != target_shape:
        fixed = np.zeros(target_shape, dtype=np.float32)
        common = tuple(min(arr.shape[i], target_shape[i]) for i in range(3))
        fixed[:common[0], :common[1], :common[2]] = arr[:common[0], :common[1], :common[2]]
        arr = fixed

    return arr


def normalize_volume(arr):
    mask = np.isfinite(arr) & (np.abs(arr) > EPS)
    if mask.sum() < 100:
        mask = np.isfinite(arr)

    mean = float(arr[mask].mean())
    std = float(arr[mask].std())

    if std < EPS:
        arr = arr - mean
    else:
        arr = (arr - mean) / std

    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = arr.astype(np.float32)
    return arr


def output_path_for_row(row):
    subject_id = str(row["subject_id"])
    return NPY_DIR / f"{subject_id}_T1_96.npy"


def process_one(record):
    try:
        image_path = Path(record["image"])
        out_path = output_path_for_row(record)

        if out_path.exists():
            arr = np.load(out_path, mmap_mode="r")
            return {
                "subject_id": record["subject_id"],
                "image": str(image_path),
                "image_npy": str(out_path),
                "status": "exists",
                "shape": "x".join(map(str, arr.shape)),
                "error": "",
            }

        arr = load_volume(image_path)
        original_shape = arr.shape

        arr = robust_clip(arr)
        arr = resize_volume(arr, TARGET_SHAPE)
        arr = normalize_volume(arr)

        arr = arr[None, ...]
        np.save(out_path, arr)

        return {
            "subject_id": record["subject_id"],
            "image": str(image_path),
            "image_npy": str(out_path),
            "status": "ok",
            "shape": "x".join(map(str, arr.shape)),
            "original_shape": "x".join(map(str, original_shape)),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "error": "",
        }

    except Exception:
        return {
            "subject_id": record.get("subject_id", "UNKNOWN"),
            "image": record.get("image", "UNKNOWN"),
            "image_npy": "",
            "status": "error",
            "shape": "",
            "error": traceback.format_exc(),
        }


def add_npy_column(input_csv, output_csv, mapping):
    df = pd.read_csv(input_csv)
    df["image_npy"] = df["image"].map(mapping)

    missing = df["image_npy"].isna().sum()
    if missing != 0:
        raise RuntimeError(f"Missing npy mapping for {missing} rows in {input_csv}")

    missing_files = [p for p in df["image_npy"].tolist() if not Path(p).exists()]
    if missing_files:
        raise RuntimeError(f"Missing npy files for {len(missing_files)} rows in {input_csv}")

    df.to_csv(output_csv, index=False)
    return len(df)


manifest_path = MANIFEST_DIR / "ixi_t1_manifest_clean.csv"
trainval_path = MANIFEST_DIR / "ixi_t1_trainval_clean.csv"
test_path = MANIFEST_DIR / "ixi_t1_test_clean.csv"
fold_dir = MANIFEST_DIR / "folds"

manifest = pd.read_csv(manifest_path)
print("Loaded manifest:", manifest_path)
print("Rows:", len(manifest))
print("Unique subjects:", manifest["subject_id"].nunique())
print("Unique images:", manifest["image"].nunique())

if len(manifest) != 563:
    print("Warning: manifest row count is not 563. Continuing with current manifest.")

records = manifest.to_dict(orient="records")
workers = int(os.environ.get("IXI_PREPROCESS_WORKERS", "0"))
if workers <= 0:
    workers = min(8, max(1, cpu_count() // 2))

print("Workers:", workers)
print("Target shape:", TARGET_SHAPE)
print("Output volume directory:", NPY_DIR)

with Pool(processes=workers) as pool:
    results = list(tqdm(pool.imap_unordered(process_one, records), total=len(records)))

results_df = pd.DataFrame(results)
report_path = OUT_ROOT / "preprocess_report.csv"
results_df.to_csv(report_path, index=False)

print()
print("Preprocessing status counts:")
print(results_df["status"].value_counts(dropna=False).to_string())

errors = results_df[results_df["status"] == "error"]
if len(errors) > 0:
    error_path = OUT_ROOT / "preprocess_errors.csv"
    errors.to_csv(error_path, index=False)
    raise RuntimeError(f"Preprocessing failed for {len(errors)} images. See {error_path}")

bad_shape = results_df[results_df["shape"] != "1x96x96x96"]
if len(bad_shape) > 0:
    bad_shape_path = OUT_ROOT / "bad_shape_report.csv"
    bad_shape.to_csv(bad_shape_path, index=False)
    raise RuntimeError(f"Unexpected output shape for {len(bad_shape)} images. See {bad_shape_path}")

mapping = dict(zip(results_df["image"], results_df["image_npy"]))

manifest_out = SPLIT_OUT_DIR / "ixi_t1_manifest_clean_npy.csv"
trainval_out = SPLIT_OUT_DIR / "ixi_t1_trainval_clean_npy.csv"
test_out = SPLIT_OUT_DIR / "ixi_t1_test_clean_npy.csv"

n_manifest = add_npy_column(manifest_path, manifest_out, mapping)
n_trainval = add_npy_column(trainval_path, trainval_out, mapping)
n_test = add_npy_column(test_path, test_out, mapping)

out_fold_dir = SPLIT_OUT_DIR / "folds"
out_fold_dir.mkdir(parents=True, exist_ok=True)

fold_counts = []
for fold in range(5):
    train_csv = fold_dir / f"fold{fold}_train.csv"
    val_csv = fold_dir / f"fold{fold}_val.csv"
    train_out = out_fold_dir / f"fold{fold}_train_npy.csv"
    val_out = out_fold_dir / f"fold{fold}_val_npy.csv"

    n_train = add_npy_column(train_csv, train_out, mapping)
    n_val = add_npy_column(val_csv, val_out, mapping)

    fold_counts.append(
        {
            "fold": fold,
            "train_n": n_train,
            "val_n": n_val,
            "train_csv": str(train_out),
            "val_csv": str(val_out),
        }
    )

fold_counts_df = pd.DataFrame(fold_counts)
fold_counts_path = SPLIT_OUT_DIR / "fold_counts_npy.csv"
fold_counts_df.to_csv(fold_counts_path, index=False)

print()
print("Saved preprocessed split CSV files:")
print("manifest_npy_csv:", manifest_out)
print("trainval_npy_csv:", trainval_out)
print("test_npy_csv:", test_out)
print("fold_npy_dir:", out_fold_dir)
print("fold_counts_csv:", fold_counts_path)
print("preprocess_report_csv:", report_path)

print()
print("Final counts:")
print("manifest_n:", n_manifest)
print("trainval_n:", n_trainval)
print("test_n:", n_test)
print("npy_files:", len(list(NPY_DIR.glob('*.npy'))))
print("expected_shape:", "1x96x96x96")

print()
print("Fold counts:")
print(fold_counts_df.to_string(index=False))

print()
print("IXI T1 preprocessing completed successfully.")

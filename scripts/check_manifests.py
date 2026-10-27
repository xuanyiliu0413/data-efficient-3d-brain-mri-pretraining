import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/fomo50k/manifests"

def read_ids(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "img_path" not in reader.fieldnames:
            raise ValueError(f"Missing img_path: {path}")
        values = [Path(row["img_path"]).name for row in reader]

    if not values or len(values) != len(set(values)):
        raise ValueError(f"Empty or duplicate identifiers: {path}")
    return set(values)

def validate(directory=DATA, counts=None):
    if counts is None:
        counts = {
            "12p5": 6084,
            "25": 12168,
            "50": 24336,
            "100": 48672,
        }

    sets = {}
    for scale, expected in counts.items():
        path = directory / f"jepa_{scale}_train.csv"
        if not path.is_file():
            raise FileNotFoundError(path)

        ids = read_ids(path)
        if len(ids) != expected:
            raise ValueError(
                f"{scale}: expected {expected}, found {len(ids)}"
            )
        sets[scale] = ids

    order = list(counts)

    for small, large in zip(order, order[1:]):
        if not sets[small].issubset(sets[large]):
            raise ValueError(f"{small} is not contained in {large}")

    return {scale: len(sets[scale]) for scale in order}

if __name__ == "__main__":
    summary = validate()
    print(json.dumps(summary, indent=2))

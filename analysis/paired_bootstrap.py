import argparse
import csv
import json
import random
import statistics

def percentile(values, q):
    s = sorted(values)
    pos = (len(s) - 1) * q
    i = int(pos)
    j = min(i + 1, len(s) - 1)
    return s[i] + (s[j] - s[i]) * (pos - i)

def paired_bootstrap(mae, jepa, iterations=20000, seed=42):
    if len(mae) != len(jepa) or not mae:
        raise ValueError("Equal, non-empty paired samples required")

    differences = [b - a for a, b in zip(mae, jepa)]
    n = len(differences)
    rng = random.Random(seed)

    estimates = [
        statistics.fmean(
            differences[rng.randrange(n)] for _ in range(n)
        )
        for _ in range(iterations)
    ]

    return {
        "n_cases": n,
        "estimate": statistics.fmean(differences),
        "ci_lower": percentile(estimates, 0.025),
        "ci_upper": percentile(estimates, 0.975),
        "iterations": iterations,
        "seed": seed,
    }

def load_pairs(path):
    case_ids = set()
    mae, jepa = [], []

    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            case_id = row["case_id"]
            if case_id in case_ids:
                raise ValueError("Duplicate case_id")
            case_ids.add(case_id)
            mae.append(float(row["mae"]))
            jepa.append(float(row["jepa"]))

    return mae, jepa

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--iterations", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    a, b = load_pairs(args.input)
    result = paired_bootstrap(
        a, b, iterations=args.iterations, seed=args.seed
    )
    print(json.dumps(result, indent=2))

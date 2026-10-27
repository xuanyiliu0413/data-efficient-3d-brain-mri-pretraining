"""Check the BraTS JEPA training and validation pipeline."""

import os
import sys
import time
import math
from pathlib import Path
from argparse import Namespace

import torch
import yaml
import wandb
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[3]
AUTO = Path(__file__).resolve().parent
BASE = REPO_ROOT / "implementation/selfmedmae"

for p in (str(AUTO), str(BASE), str(BASE / "lib"),
          str(REPO_ROOT / "implementation")):
    if p not in sys.path:
        sys.path.insert(0, p)

from run_brats_jepa import JepaUNETR3D
from lib.trainers.seg_trainer import SegTrainer

torch.manual_seed(123)
torch.set_num_threads(4)
wandb.init(mode="disabled")

cfg_path = REPO_ROOT / "configs/downstream/brats/jepa_50.yaml"
cfg = yaml.safe_load(cfg_path.read_text())

out = Path(os.environ.get("BRAINMRI_BRATS_PREFLIGHT_OUT", str(REPO_ROOT / "outputs/brats_preflight")))
(out / "ckpts").mkdir(parents=True, exist_ok=True)

cfg.update({
    "run_name": "brats_jepa_preflight",
    "output_dir": str(out),
    "ckpt_dir": str(out / "ckpts"),
    "epochs": 1,
    "eval_freq": 1,
    "save_freq": 999,
    "vis_freq": 999,
    "print_freq": 1,
    "skip_initial_eval": True,
    "gpu": 0,
    "rank": 0,
    "world_size": 1,
    "ngpus_per_node": 1,
    "distributed": False,
    "workers": 0,
    "disable_wandb": True
})

args = Namespace(**cfg)

print("Building BraTS trainer", flush=True)

trainer = SegTrainer(args)
trainer.build_model()
trainer.build_optimizer()
trainer.build_dataloader()

def limited_loader(loader, count):
    """Create a small dataloader for pipeline testing."""
    n = min(count, len(loader.dataset))
    return DataLoader(
        Subset(loader.dataset, list(range(n))),
        batch_size=loader.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=loader.collate_fn
    )

trainer.dataloader = limited_loader(trainer.dataloader, 2)
trainer.val_dataloader = limited_loader(trainer.val_dataloader, 1)
trainer.vis_dataloader = limited_loader(trainer.vis_dataloader, 1)
trainer.iters_per_epoch = len(trainer.dataloader)

print("Training batches:", trainer.iters_per_epoch, flush=True)
print("Validation cases:", len(trainer.val_dataloader), flush=True)

original_evaluate = trainer.evaluate
validation_scores = []

def checked_evaluate(*a, **kw):
    """Run validation and check that Dice is finite."""
    result = original_evaluate(*a, **kw)
    score = float(result[0])
    if not math.isfinite(score):
        raise RuntimeError("Validation Dice is not finite")
    validation_scores.append(score)
    print("Validation Dice:", score, flush=True)
    return result

trainer.evaluate = checked_evaluate

print("Running BraTS pipeline test", flush=True)

start = time.perf_counter()
trainer.run()

if len(validation_scores) != 1:
    raise RuntimeError("Expected exactly one validation")

best = out / "ckpts/best_model.pth.tar"
if not best.is_file() or best.stat().st_size == 0:
    raise RuntimeError("Best model checkpoint not saved")

print("BraTS pipeline test results")
print("Validation Dice:", validation_scores[0])
print("Best checkpoint:", best)
print("Elapsed seconds:", round(time.perf_counter() - start, 1))



print("====================================")
print("BraTS pipeline test completed")
print("====================================")

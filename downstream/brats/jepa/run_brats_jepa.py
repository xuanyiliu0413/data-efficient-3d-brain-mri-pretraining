"""Run BraTS segmentation with the JEPA encoder."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
BASE = REPO_ROOT / "implementation/selfmedmae"
AUTO = Path(__file__).resolve().parent

for p in (str(BASE / "lib"), str(BASE), str(AUTO),
          str(REPO_ROOT / "implementation")):
    if p not in sys.path:
        sys.path.insert(0, p)

import models
from jepa_unetr import JepaUnetr


class JepaUNETR3D(JepaUnetr):
    def __init__(self, encoder=None, decoder=None, args=None):
        if args is None:
            raise ValueError("BraTS args are required")
        fraction = str(args.jepa_fraction)
        super().__init__(fraction=fraction)

    def get_num_layers(self):
        return self.encoder.get_num_layers()

    def no_weight_decay(self):
        if hasattr(self.encoder, "no_weight_decay"):
            return {
                "encoder." + name
                for name in self.encoder.no_weight_decay()
            }
        return set()


models.JepaUNETR3D = JepaUNETR3D


if __name__ == "__main__":
    from main import main
    main()

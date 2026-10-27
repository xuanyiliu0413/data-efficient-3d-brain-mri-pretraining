"""Define the JEPA encoder and UNETR segmentation network."""

import os
import sys
from pathlib import Path

import torch
from torch import nn
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "implementation"))
sys.path.insert(0, str(REPO_ROOT / "implementation/selfmedmae"))
sys.path.insert(0, str(REPO_ROOT / "implementation/selfmedmae/lib"))

from neurojepa.utils.init_utils import init_backbone
from lib.networks.unetr_decoder import UNETR_decoder


CONFIG_DIR = REPO_ROOT / "configs/pretrain/jepa_archived"
ENC = Path(os.environ.get("BRAINMRI_JEPA_ENCODER_DIR", str(REPO_ROOT / "checkpoints/jepa")))


class JepaUnetr(nn.Module):

    def __init__(self, fraction="50"):
        super().__init__()

        if fraction not in ("12p5", "25", "50", "100"):
            raise ValueError("Unknown fraction: " + fraction)

        cfg_path = CONFIG_DIR / f"jepa_{fraction}_paper.yaml"
        if not cfg_path.is_file():
            raise FileNotFoundError(cfg_path)

        cfg = OmegaConf.load(str(cfg_path))
        m = cfg.model
        meta = cfg.meta

        self.encoder = init_backbone(
            device=torch.device("cpu"),
            model_name=m.backbone_name,
            img_size=list(m.img_size),
            patch_size=list(m.patch_size),
            in_chans=int(m.in_chans),
            uniform_power=bool(
                m.get("uniform_power", False)
            ),
            use_sdpa=bool(
                meta.get("use_sdpa", False)
            ),
            use_rope=bool(
                m.get("use_rope", False)
            ),
            use_silu=bool(
                m.get("use_silu", False)
            ),
            wide_silu=bool(
                m.get("wide_silu", False)
            ),
            use_activation_checkpointing=False,
        )

        checkpoint = ENC / (
            "neurojepa_fomo"
            + fraction
            + "_encoder_e100.pt"
        )

        ck = torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=True
        )

        if ck.get("epoch") != 99:
            raise RuntimeError("Unexpected pretraining epoch")

        self.encoder.load_state_dict(
            ck["model"],
            strict=True
        )

        print(
            "Loaded JEPA encoder:",
            fraction,
            flush=True
        )




        old = self.encoder.patch_embed.proj

        if not isinstance(old, nn.Conv3d):
            raise RuntimeError(
                "Expected Conv3d patch embedding"
            )

        if old.in_channels != 1 or old.groups != 1:
            raise RuntimeError(
                "Expected single-channel patch embedding"
            )

        new = nn.Conv3d(
            in_channels=4,
            out_channels=old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            dilation=old.dilation,
            groups=1,
            bias=old.bias is not None
        )

        with torch.no_grad():
            new.weight.copy_(
                old.weight.repeat(1, 4, 1, 1, 1) / 4.0
            )
            if old.bias is not None:
                new.bias.copy_(old.bias)

        self.encoder.patch_embed.proj = new

        self.decoder = UNETR_decoder(
            in_channels=4,
            out_channels=3,
            img_size=(96, 96, 96),
            patch_size=16,
            feature_size=16,
            hidden_size=768,
            spatial_dims=3
        )

        self.hidden = {}




        self.handles = []

        for index in (3, 6, 9):

            def capture(module, inputs, output, i=index):
                if isinstance(output, (tuple, list)):
                    output = output[0]
                self.hidden[i] = output

            handle = self.encoder.blocks[
                index
            ].register_forward_hook(capture)

            self.handles.append(handle)

    def forward(self, x):

        self.hidden = {}

        result = self.encoder(x)

        tokens = (
            result[0]
            if isinstance(result, (tuple, list))
            else result
        )

        if tokens.shape[1:] != (216, 768):
            raise RuntimeError(
                "Invalid JEPA token shape: "
                + str(tuple(tokens.shape))
            )

        states = [None] * 12

        for i in (3, 6, 9):
            if i not in self.hidden:
                raise RuntimeError(
                    "Missing intermediate block " + str(i)
                )
            states[i] = self.hidden[i]

        return self.decoder(x, tokens, states)

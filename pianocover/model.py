"""Style-conditioned Transkun.

The Transkun V2 transformer (semi-CRF interval scoring) is kept as-is and initialised from
its pretrained piano-transcription checkpoint. We add FiLM conditioning: after every
encoder block, the hidden state h (shape [N, T, F, D]) becomes h * (1 + gamma) + beta,
with gamma/beta predicted from the style vector. FiLM layers are zero-initialised, so at
step 0 the model is exactly pretrained Transkun (the AMT-APC "fine-tune a transcriber" idea).

The FiLM is applied through forward hooks, so Transkun's own code is not modified.

Gradient checkpointing: in training mode Transkun wraps every encoder layer in (reentrant)
torch.utils.checkpoint, which re-runs the layer -- and therefore our hook -- during
backward(). The hook reads the current style, so the style set by log_prob() must still be
in place when backward() runs. log_prob() therefore leaves it set (it is replaced by the
next log_prob() call); always call backward() on one log_prob() result before the next
log_prob() with a different style. generate() restores the previous style when it is done.
scripts/check_env.py verifies checkpointed and non-checkpointed gradients agree.
"""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path

import torch
import torch.nn as nn

from . import style as style_mod


def pretrained_paths() -> tuple[Path, Path]:
    """Paths of the checkpoint and conf shipped inside the `transkun` pip package."""
    root = resources.files("transkun") / "pretrained"
    return Path(str(root / "2.0.pt")), Path(str(root / "2.0.conf"))


def load_transkun_conf(conf_path) -> dict:
    with open(conf_path) as f:
        data = json.load(f)
    model = data["Model"]
    if model.get("module") != "transkun.ModelTransformer":
        raise ValueError(f"only transkun.ModelTransformer confs are supported, got {model.get('module')}")
    return model["config"]


def build_transkun(conf_dict: dict):
    from transkun.ModelTransformer import Config, TransKun

    conf = Config()
    conf.__dict__.update(conf_dict)
    return TransKun(conf=conf)


def load_pretrained_transkun(weight_path=None, conf_path=None, device="cpu"):
    """Plain pretrained Transkun (used to transcribe the human covers)."""
    w, c = pretrained_paths()
    weight_path, conf_path = weight_path or w, conf_path or c
    model = build_transkun(load_transkun_conf(conf_path))
    ckpt = torch.load(weight_path, map_location="cpu", weights_only=False)
    state = ckpt.get("best_state_dict") or ckpt["state_dict"]
    model.load_state_dict(state, strict=False)
    return model.to(device).eval()


class PianoCoverModel(nn.Module):
    def __init__(self, conf_dict: dict, style_dim: int = style_mod.DIM, use_amp: bool = True):
        super().__init__()
        self.conf_dict = dict(conf_dict)
        self.style_dim = style_dim
        self.transkun = build_transkun(conf_dict)
        width = self.transkun.conf.baseSize * 4  # Backbone encoder width
        layers = self.transkun.backbone.encoderLayers
        self.film = nn.ModuleList(nn.Linear(style_dim, 2 * width) for _ in layers)
        for lin in self.film:
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)
        self._style: torch.Tensor | None = None
        for i, layer in enumerate(layers):
            layer.register_forward_hook(self._film_hook(i))
        self.use_amp = use_amp
        self._wrap_backbone_amp()

    # ----------------------------------------------------------------- conditioning
    def _film_hook(self, idx: int):
        def hook(_module, _inputs, out):
            if self._style is None:
                return out
            gb = self.film[idx](self._style.to(out.dtype))  # [B, 2D]
            if gb.shape[0] != out.shape[0]:
                if gb.shape[0] == 1:
                    gb = gb.expand(out.shape[0], -1)
                else:
                    gb = gb.repeat_interleave(out.shape[0] // gb.shape[0], dim=0)
            gamma, beta = gb.chunk(2, dim=-1)
            return out * (1 + gamma[:, None, None, :]) + beta[:, None, None, :]
        return hook

    def _wrap_backbone_amp(self):
        """Run only the transformer backbone in bf16; scores and the semi-CRF stay fp32."""
        backbone = self.transkun.backbone
        original = backbone.forward

        def forward(*args, **kwargs):
            enabled = self.use_amp and torch.cuda.is_available() and next(backbone.parameters()).is_cuda
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=enabled):
                out = original(*args, **kwargs)
            return out.float()

        backbone.forward = forward

    def set_style(self, style: torch.Tensor | None):
        if style is not None and style.dim() == 1:
            style = style[None]
        self._style = style

    # ----------------------------------------------------------------- API
    def log_prob(self, audio: torch.Tensor, notes_batch, style: torch.Tensor | None):
        """audio: [B, nSample, nChannel] at 44.1 kHz; notes_batch: list of lists of transkun Notes."""
        # No reset afterwards: checkpoint recomputation during backward() needs this style.
        self.set_style(style)
        return self.transkun.log_prob(audio, notes_batch)

    @torch.no_grad()
    def generate(self, audio: torch.Tensor, style: torch.Tensor | None, step_sec=None, segment_sec=None):
        """audio: [nSample, nChannel] tensor at 44.1 kHz -> list of transkun Notes (absolute times)."""
        previous = self._style
        self.set_style(style)
        try:
            return self.transkun.transcribe(audio, stepInSecond=step_sec, segmentSizeInSecond=segment_sec,
                                            discardSecondHalf=False)
        finally:
            self._style = previous

    def parameter_groups(self, lr_backbone: float, lr_heads: float, lr_film: float, weight_decay: float):
        """Lower LR for the pretrained backbone, higher for heads, highest for the new FiLM layers."""
        film = list(self.film.parameters())
        backbone = list(self.transkun.backbone.parameters()) + list(self.transkun.framewiseFeatureExtractor.parameters())
        ids = {id(p) for p in film + backbone}
        heads = [p for p in self.parameters() if id(p) not in ids]

        def split(params):
            decay = [p for p in params if p.ndim >= 2]
            no_decay = [p for p in params if p.ndim < 2]
            return decay, no_decay

        groups = []
        for params, lr in ((backbone, lr_backbone), (heads, lr_heads), (film, lr_film)):
            decay, no_decay = split([p for p in params if p.requires_grad])
            if decay:
                groups.append({"params": decay, "lr": lr, "weight_decay": weight_decay})
            if no_decay:
                groups.append({"params": no_decay, "lr": lr, "weight_decay": 0.0})
        return groups

    # ----------------------------------------------------------------- persistence
    @classmethod
    def from_pretrained_transkun(cls, weight_path=None, conf_path=None, **kw) -> "PianoCoverModel":
        w, c = pretrained_paths()
        weight_path, conf_path = weight_path or w, conf_path or c
        model = cls(load_transkun_conf(conf_path), **kw)
        ckpt = torch.load(weight_path, map_location="cpu", weights_only=False)
        state = ckpt.get("best_state_dict") or ckpt["state_dict"]
        missing, unexpected = model.transkun.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"[pianocover] pretrained load: {len(missing)} missing, {len(unexpected)} unexpected keys")
        return model

    def checkpoint_payload(self) -> dict:
        return {"conf": self.conf_dict, "style_dim": self.style_dim, "state_dict": self.state_dict()}

    @classmethod
    def from_checkpoint(cls, path, map_location="cpu", use_amp=True):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(ckpt["conf"], style_dim=ckpt["style_dim"], use_amp=use_amp)
        state = ckpt.get("best_state_dict") or ckpt["state_dict"]
        model.load_state_dict(state)
        return model.to(map_location), ckpt

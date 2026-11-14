"""AMT-APC's training loss (train/_loss.py), computed without host synchronisation.

The original selects cells with boolean indexing (dynamic shapes: a GPU->CPU sync per call,
and a torch.compile graph break) and computes F1 with sklearn on the CPU every step. Here the
same cells are selected with a weight mask, so loss values are identical up to float rounding,
and F1 counts stay on the GPU until they are logged.

Selection (as in the original): a cell is used if its label is positive, if the same frame's
neighbouring pitch is positive (torch.roll along the note axis, as AMT-APC does), or at random
with probability theta (onset 0.07, frame 0.2, velocity 0.01).
Loss = beta * (onset + frame + velocity)/3 of the frequency stage
     + (1 - beta) * (same) of the time stage, beta = 0.75. Offsets are not trained, as in AMT-APC.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

BETA = 0.75
THETA_ONSET, THETA_FRAME, THETA_VELOCITY = 0.07, 0.2, 0.01


def _mask(label: torch.Tensor, prob: float) -> torch.Tensor:
    pos = label > 0
    m = pos | torch.roll(pos, 1, dims=-1) | torch.roll(pos, -1, dims=-1)
    return (m | (torch.rand(pos.shape, device=pos.device) < prob)).float()


def _wmean(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return (x * w).sum() / w.sum().clamp_min(1.0)


def loss_fn(out, onset, frame, velocity):
    """out: logits from Spec2MIDI; labels [B, 512, 88] (velocity int). Returns (loss, stats)."""
    on_f, _, fr_f, vel_f, on_t, _, fr_t, vel_t = (o.float() for o in out)
    frame = frame.float()
    w_on, w_fr, w_vel = _mask(onset, THETA_ONSET), _mask(frame, THETA_FRAME), _mask(velocity, THETA_VELOCITY)
    vel = velocity.long()

    def stage(on, fr, ve):
        l_on = _wmean(F.binary_cross_entropy_with_logits(on, onset, reduction="none"), w_on)
        l_fr = _wmean(F.binary_cross_entropy_with_logits(fr, frame, reduction="none"), w_fr)
        l_ve = _wmean(F.cross_entropy(ve.permute(0, 3, 1, 2), vel, reduction="none"), w_vel)
        return (l_on + l_fr + l_ve) / 3

    loss = BETA * stage(on_f, fr_f, vel_f) + (1 - BETA) * stage(on_t, fr_t, vel_t)
    with torch.no_grad():  # F1 counts of the time stage at threshold 0.5, as AMT-APC logs them
        def counts(pred, true):
            return torch.stack([(pred & true).sum(), (pred & ~true).sum(), (~pred & true).sum()]).float()

        stats = torch.stack([
            counts(on_t > 0, onset > 0.5),
            counts(fr_t > 0, frame > 0.5),
            counts(vel_t.argmax(-1) > 0, vel > 0),
        ])
    return loss, stats


def f1_from_counts(c: torch.Tensor) -> list[float]:
    """c [3 kinds, (tp, fp, fn)] -> F1 for onset, frame, velocity."""
    tp, fp, fn = c[:, 0], c[:, 1], c[:, 2]
    return (2 * tp / (2 * tp + fp + fn).clamp_min(1)).tolist()

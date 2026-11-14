"""Note events -> AMT-APC frame labels, and the 24-dim style vector.

Labels follow hFT-Transformer's `conv_note2label` (onset / offset as triangles of +-50 ms
around the event, frame = onset..offset inclusive, velocity where onset >= 0.5), but are
computed for one window at a time, vectorised, so the dataset stores notes, not label maps.

Notes are first lengthened through sustain-pedal intervals: hFT's MIDI reader extends notes
while the pedal is down, and AMT-APC's released model was trained on such labels, while our
cover transcriptions (Transkun, no pedal extension) keep pedal as separate events.
"""

from __future__ import annotations

import numpy as np

from .. import events
from . import FPS, N_NOTE, N_VELOCITY, NOTE_MIN

HOP_MS = 1000.0 / FPS
TOL = int(50.0 / HOP_MS + 0.5)  # 3 frames
_OFFS = np.arange(-TOL, TOL + 1)

# AMT-APC's style-vector normalisation (data/sv/style_vectors.json "params" of the released
# model), kept so that our style vectors mean the same as those apc.pth was trained with.
SV_PARAMS = {"mean_vel": 80.48306532691362, "mean_pitch": 61.196047258765944,
             "mean_onset_rate": 0.1216200295550453, "std_vel": 15.223412313236983,
             "std_pitch": 14.966039734307582, "std_onset_rate": 0.050815288415511056}
_BIN_DIST = np.array([-2, -4 / 3, -2 / 3, 0, 2 / 3, 4 / 3, 2])


def pedal_extend(ev: np.ndarray) -> np.ndarray:
    """Piano notes as [start, end, pitch, velocity], each end moved to the release of a sustain
    pedal held at that moment (but not past the next onset of the same pitch)."""
    notes = ev[events.is_note(ev)].astype(np.float64)
    ped = ev[ev[:, events.PITCH] == -64]
    if len(notes) == 0 or len(ped) == 0:
        return notes
    ped = ped[np.argsort(ped[:, events.START])]
    ends = notes[:, events.END].copy()
    i = np.searchsorted(ped[:, events.START], ends, side="right") - 1
    held = (i >= 0) & (ped[np.maximum(i, 0), events.END] > ends)
    ends[held] = ped[i[held], events.END]
    order = np.lexsort((notes[:, events.START], notes[:, events.PITCH]))
    n = notes[order]
    e = ends[order]
    same = n[1:, events.PITCH] == n[:-1, events.PITCH]
    nxt = np.where(same, n[1:, events.START], np.inf)
    e[:-1] = np.minimum(e[:-1], np.maximum(nxt, n[:-1, events.END]))
    n[:, events.END] = e
    return n[np.argsort(n[:, events.START], kind="stable")]


def window_labels(notes: np.ndarray, first_frame: int, n_frames: int):
    """Labels for frames [first_frame, first_frame + n_frames) from pedal-extended notes.

    Returns onset, offset (float32 in [0, 1]), frame (bool), velocity (int64 0..127),
    each [n_frames, 88].
    """
    onset = np.zeros((n_frames, N_NOTE), np.float32)
    offset = np.zeros((n_frames, N_NOTE), np.float32)
    frame = np.zeros((n_frames, N_NOTE), bool)
    velocity = np.zeros((n_frames, N_NOTE), np.int64)
    if len(notes) == 0:
        return onset, offset, frame, velocity
    t0 = first_frame / FPS
    t1 = (first_frame + n_frames) / FPS
    margin = (TOL + 1) / FPS
    m = (notes[:, events.END] >= t0 - margin) & (notes[:, events.START] < t1 + margin)
    n = notes[m]
    if len(n) == 0:
        return onset, offset, frame, velocity
    pitch = (n[:, events.PITCH] - NOTE_MIN).astype(np.int64)
    on_ms, off_ms = n[:, events.START] * 1000.0, n[:, events.END] * 1000.0
    on_f = (n[:, events.START] * FPS + 0.5).astype(np.int64)
    off_f = (n[:, events.END] * FPS + 0.5).astype(np.int64)
    vel = np.clip(n[:, events.VEL], 1, N_VELOCITY - 1).astype(np.int64)

    def tri(center_f, center_ms):
        f = center_f[:, None] + _OFFS[None, :]
        val = np.maximum(0.0, 1.0 - np.abs(f * HOP_MS - center_ms[:, None]) / (TOL * HOP_MS))
        return f - first_frame, val

    # onset + velocity (velocity of the strongest onset in each cell)
    f, val = tri(on_f, on_ms)
    p = np.broadcast_to(pitch[:, None], f.shape)
    ok = (f >= 0) & (f < n_frames)
    fi, pi, vi = f[ok], p[ok], val[ok].astype(np.float32)
    np.maximum.at(onset, (fi, pi), vi)
    strong = vi >= 0.5
    order = np.argsort(vi[strong], kind="stable")  # later writes (stronger) win
    velocity[fi[strong][order], pi[strong][order]] = np.broadcast_to(vel[:, None], f.shape)[ok][strong][order]

    # offset, unless the same pitch starts again exactly there
    starts = {(int(pp), round(float(s), 6)) for pp, s in zip(pitch, n[:, events.START])}
    keep = np.array([(int(pp), round(float(e), 6)) not in starts for pp, e in zip(pitch, n[:, events.END])])
    if keep.any():
        f, val = tri(off_f[keep], off_ms[keep])
        p = np.broadcast_to(pitch[keep][:, None], f.shape)
        ok = (f >= 0) & (f < n_frames)
        np.maximum.at(offset, (f[ok], p[ok]), val[ok].astype(np.float32))

    # frame: onset frame .. offset frame inclusive
    a = np.clip(on_f - first_frame, 0, n_frames)
    b = np.clip(off_f - first_frame + 1, 0, n_frames)
    for lo, hi, pp in zip(a, b, pitch):
        if hi > lo:
            frame[lo:hi, pp] = True
    return onset, offset, frame, velocity


def _distribution(x: np.ndarray) -> np.ndarray:
    digit = np.digitize(x, _BIN_DIST)
    dist = np.bincount(digit, minlength=len(_BIN_DIST) + 1).astype(np.float64)
    return dist / max(dist.sum(), 1.0)


def style_vector(notes: np.ndarray, duration: float, p=SV_PARAMS) -> np.ndarray:
    """AMT-APC's 24-dim style vector (velocity, pitch and onset-rate distributions, 8 bins each)."""
    if len(notes) == 0:
        return np.zeros(24, np.float32)
    vel = np.clip(notes[:, events.VEL], 1, N_VELOCITY - 1)
    pitch = notes[:, events.PITCH]
    on_f = (notes[:, events.START] * FPS).astype(np.int64)
    n_seg = max(1, int(np.ceil(duration * FPS / 512)))
    rates = np.bincount(np.minimum(on_f // 512, n_seg - 1), minlength=n_seg) / 512.0
    sv = np.concatenate([
        _distribution((vel - p["mean_vel"]) / p["std_vel"]),
        _distribution((pitch - p["mean_pitch"]) / p["std_pitch"]),
        _distribution((rates - p["mean_onset_rate"]) / p["std_onset_rate"]),
    ])
    return sv.astype(np.float32)

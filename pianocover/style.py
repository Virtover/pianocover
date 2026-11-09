"""Style vectors: a small, interpretable description of how a cover is arranged.

During training the model sees the style of the human cover it must reproduce, so at
inference you can steer the arrangement (sparser / denser, louder, lower register ...).
This plays the role of AMT-APC's style vector, but with named, human-readable features.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from .events import END, PITCH, START, VEL

FEATURES = [
    "log_onset_rate",   # log(notes per active second): overall busyness
    "polyphony",        # mean number of sounding keys while anything sounds
    "velocity_mean",    # loudness (0..1)
    "velocity_std",     # dynamic range (0..1)
    "pitch_mean",       # register centre (MIDI / 127)
    "pitch_std",        # register spread (semitones / 12)
    "left_hand_ratio",  # fraction of notes below middle C
    "pedal_ratio",      # fraction of active time with sustain pedal down
]
DIM = len(FEATURES)

# Presets are expressed in standard deviations from the training-set mean.
PRESETS = {
    "normal": {},
    "easy": {"log_onset_rate": -1.5, "polyphony": -1.5, "velocity_std": -0.5},
    "simple": {"log_onset_rate": -0.8, "polyphony": -0.8},
    "full": {"log_onset_rate": 0.8, "polyphony": 0.8, "pitch_std": 0.5},
    "virtuoso": {"log_onset_rate": 1.6, "polyphony": 1.2, "pitch_std": 1.0},
}


def raw_style(ev: np.ndarray, frame: float = 0.05) -> np.ndarray:
    """Compute the raw (unnormalised) style features for one whole cover."""
    notes = ev[ev[:, PITCH] > 0]
    if len(notes) == 0:
        return np.zeros(DIM, dtype=np.float32)
    t0, t1 = notes[:, START].min(), notes[:, END].max()
    n_frames = max(int(np.ceil((t1 - t0) / frame)), 1)
    counts = np.zeros(n_frames + 1, dtype=np.int32)
    a = np.clip(((notes[:, START] - t0) / frame).astype(int), 0, n_frames)
    b = np.clip(((notes[:, END] - t0) / frame).astype(int), 0, n_frames)
    np.add.at(counts, a, 1)
    np.add.at(counts, b, -1)
    active = np.cumsum(counts)[:n_frames]
    active_frames = active > 0
    active_sec = max(active_frames.sum() * frame, 1e-3)

    pedal = ev[ev[:, PITCH] == -64]
    pedal_sec = float(np.clip(pedal[:, END] - pedal[:, START], 0, None).sum()) if len(pedal) else 0.0
    total_sec = max(t1 - t0, 1e-3)

    vel = notes[:, VEL] / 127.0
    pitch = notes[:, PITCH]
    return np.array([
        np.log(len(notes) / active_sec + 1e-3),
        float(active[active_frames].mean()) if active_frames.any() else 0.0,
        float(vel.mean()),
        float(vel.std()),
        float(pitch.mean() / 127.0),
        float(pitch.std() / 12.0),
        float((pitch < 60).mean()),
        min(pedal_sec / total_sec, 1.0),
    ], dtype=np.float32)


@dataclass
class StyleNormalizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, raw: np.ndarray) -> "StyleNormalizer":
        raw = np.asarray(raw, dtype=np.float64)
        return cls(raw.mean(0).astype(np.float32), (raw.std(0) + 1e-6).astype(np.float32))

    def normalize(self, raw: np.ndarray) -> np.ndarray:
        return ((np.asarray(raw) - self.mean) / self.std).astype(np.float32)

    def to_dict(self) -> dict:
        return {"features": FEATURES, "mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "StyleNormalizer":
        if d.get("features", FEATURES) != FEATURES:
            raise ValueError("style feature list mismatch; rebuild the dataset stats")
        return cls(np.array(d["mean"], np.float32), np.array(d["std"], np.float32))

    def save(self, path) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path) -> "StyleNormalizer":
        with open(path) as f:
            return cls.from_dict(json.load(f))


def preset_vector(name: str = "normal", **overrides: float) -> np.ndarray:
    """Normalised style vector for a preset, optionally overriding single features (in std units)."""
    if name not in PRESETS:
        raise KeyError(f"unknown preset {name!r}; choose from {sorted(PRESETS)}")
    values = dict(PRESETS[name])
    values.update(overrides)
    vec = np.zeros(DIM, dtype=np.float32)
    for k, v in values.items():
        if k not in FEATURES:
            raise KeyError(f"unknown style feature {k!r}; choose from {FEATURES}")
        vec[FEATURES.index(k)] = v
    return vec

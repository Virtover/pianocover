"""Training windows for AMT-APC: memory-mapped spectrogram slices + labels built on the fly."""

from __future__ import annotations

import pickle
import random
from pathlib import Path

import numpy as np
import torch

from . import FPS, MARGIN, NUM_FRAME
from .labels import window_labels


def load_rows(path):
    with open(path, "rb") as f:
        return pickle.load(f)


class Windows(torch.utils.data.Dataset):
    """512-frame windows (8.2 s) of every pair.

    Training: window starts get a random offset each epoch (call `reshuffle`), so the model sees
    different cuts of the same song. Windows overlapping the song's unaligned regions by more
    than `max_bad` are skipped, so the model is not taught silence where the cover diverged.
    """

    def __init__(self, rows, spec_dir, jitter=True, max_bad=0.2, seed=0):
        self.rows, self.spec_dir, self.jitter, self.max_bad = rows, Path(spec_dir), jitter, max_bad
        self._mm = {}
        self.reshuffle(seed)

    def reshuffle(self, seed):
        rng = random.Random(seed)
        win = []
        for idx, r in enumerate(self.rows):
            shift = rng.randrange(NUM_FRAME) if self.jitter else 0
            for start in range(-shift if shift else 0, r["n_frames"], NUM_FRAME):
                s = max(start, 0)
                e = min(start + NUM_FRAME, r["n_frames"])
                if e - s < NUM_FRAME // 4:
                    continue
                t0, t1 = s / FPS, e / FPS
                bad = sum(max(0.0, min(t1, b1) - max(t0, b0)) for b0, b1 in r["bad_regions"])
                if bad > self.max_bad * (t1 - t0):
                    continue
                win.append((idx, start))
        self.windows = win

    def __len__(self):
        return len(self.windows)

    def _spec(self, name):
        mm = self._mm.get(name)
        if mm is None:  # opened lazily in each worker process
            mm = self._mm[name] = np.load(self.spec_dir / name, mmap_mode="r")
        return mm

    def __getitem__(self, i):
        idx, start = self.windows[i]
        r = self.rows[idx]
        spec = self._spec(r["spec"])
        # padded index = frame + MARGIN; the window needs MARGIN frames of context each side
        lo, hi = start, start + NUM_FRAME + 2 * MARGIN
        x = np.full((NUM_FRAME + 2 * MARGIN, spec.shape[1]), -18.0, np.float16)
        a, b = max(lo, 0), min(hi, spec.shape[0])
        x[a - lo:b - lo] = spec[a:b]
        onset, offset, frame, velocity = window_labels(r["notes"], start, NUM_FRAME)
        return (torch.from_numpy(x.T.copy()), torch.from_numpy(r["sv"]),
                torch.from_numpy(onset), torch.from_numpy(offset),
                torch.from_numpy(frame), torch.from_numpy(velocity.astype(np.uint8)))

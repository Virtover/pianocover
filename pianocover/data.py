"""Training dataset: chunks of original-song audio + aligned cover notes + style vector."""

from __future__ import annotations

import math
import pickle
import random
from pathlib import Path

import numpy as np
import torch

from . import SAMPLE_RATE, events
from .events import shift_events  # noqa: F401  (re-exported for train.py)
from .audio import read_wav_slice


def load_split(path):
    with open(path, "rb") as f:
        blob = pickle.load(f)
    items = blob["items"]
    for it in items:
        # merged datasets keep a root per item; single datasets share the file's root
        root = Path(it.get("data_root") or blob["data_root"])
        it["audio_path"] = str(root / it["audio"])
        it.setdefault("source", "web")
    return items


class CoverChunks(torch.utils.data.Dataset):
    """Fixed-length chunks (like Transkun's DatasetMaestroIterator) with random dithering.

    Chunks that overlap unaligned regions of the original by more than `max_bad_overlap`
    are skipped, so the model is never taught "play nothing" where the cover simply
    diverged from the song.
    """

    def __init__(self, items, chunk_sec=16.0, hop_sec=8.0, seed=0, dither=True,
                 max_bad_overlap=0.2, style_dropout=0.0, augmentator=None):
        self.items = items
        self.chunk_sec = chunk_sec
        self.style_dropout = style_dropout
        self.augmentator = augmentator
        rng = random.Random(seed)
        chunks = []
        for idx, it in enumerate(items):
            dur = it["duration"]
            n = math.ceil(dur / hop_sec) + 1
            for j in range(n):
                shift = (rng.random() - 0.5) * hop_sec if dither else 0.0
                begin = j * hop_sec + shift - chunk_sec / 4
                end = begin + chunk_sec
                if end <= 0 or begin >= dur:
                    continue
                bad = sum(max(0.0, min(end, e) - max(begin, s)) for s, e in it["bad_regions"])
                if bad > max_bad_overlap * chunk_sec:
                    continue
                chunks.append((idx, begin, end))
        rng.shuffle(chunks)
        self.chunks = chunks

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, i):
        idx, begin, end = self.chunks[i]
        it = self.items[idx]
        audio = read_wav_slice(it["audio_path"], begin, end)
        if self.augmentator is not None:
            audio = self.augmentator(audio)
        ev, has_on, has_off = events.slice_events(it["events"].astype(np.float64), begin, end)
        style = np.asarray(it["style"], dtype=np.float32)
        if self.style_dropout and random.random() < self.style_dropout:
            style = np.zeros_like(style)  # zero = training-set average style
        return {"audio": audio, "events": ev, "has_onset": has_on, "has_offset": has_off,
                "style": style, "key": (it["pop_id"], it["piano_id"], begin)}


def collate(batch):
    n = min(b["audio"].shape[0] for b in batch)
    audio = torch.from_numpy(np.stack([b["audio"][:n, :1] for b in batch]))
    style = torch.from_numpy(np.stack([b["style"] for b in batch]))
    return {"audio": audio, "events": [b["events"] for b in batch],
            "has_onset": [b["has_onset"] for b in batch], "has_offset": [b["has_offset"] for b in batch],
            "style": style, "keys": [b["key"] for b in batch]}


def make_augmentator(noise_folder=None, ir_folder=None):
    """Transkun's own audio augmentation (EQ, small detune, noise, reverb); needs audiomentations."""
    from transkun.Data import AugmentatorAudiomentations

    return AugmentatorAudiomentations(sampleRate=SAMPLE_RATE, noiseFolder=noise_folder, convIRFolder=ir_folder)


def random_key_shift(audio: torch.Tensor, max_semitones: int, prob: float):
    """Shift each item by a random whole number of semitones by *resampling* (varispeed).

    Playing audio r = 2**(k/12) times faster raises it k semitones and shortens it by 1/r;
    `shift_events` applies the identical transform to the targets. Linear interpolation is
    enough: aliasing lands near Nyquist, far above Transkun's 8 kHz mel ceiling. This is
    one GPU op, unlike phase-vocoder pitch shifting with awkward resampling ratios.
    audio: [B, nSample, 1]. Returns (audio, shifts).
    """
    B, n = audio.shape[0], audio.shape[1]
    shifts = [0] * B
    if max_semitones <= 0 or prob <= 0:
        return audio, shifts
    out = audio.clone()
    t = torch.arange(n, device=audio.device, dtype=torch.float64)
    for b in range(B):
        k = random.randint(-max_semitones, max_semitones) if random.random() < prob else 0
        if k == 0:
            continue
        pos = t * (2.0 ** (k / 12.0))  # input sample index for each output sample
        valid = pos < (n - 1)
        i0 = pos.clamp(max=n - 2).floor().long()
        frac = (pos - i0).to(audio.dtype)[:, None]
        x = audio[b]
        y = x[i0] * (1 - frac) + x[i0 + 1] * frac
        out[b] = y * valid[:, None].to(audio.dtype)
        shifts[b] = k
    return out, shifts

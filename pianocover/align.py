"""Align a human piano cover to the original song and map its notes onto the song's timeline.

Steps (per pair):
  1. Detect the cover's key offset against the original (12 transpositions, coarse DTW).
  2. High-resolution sync with synctoolbox's MrMsDTW on chroma + onset (DLNCO) features at
     50 Hz, the method AMT-APC uses (they time-stretch the audio; we warp the MIDI instead,
     which is cheaper and adds no audio artefacts).
  3. Score the alignment locally (chroma similarity along the path, path slope) and mark
     regions where the cover does not follow the original (extra intros, solos, medleys).
"""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass, field

import numpy as np

SYNC_SR = 22050
FEATURE_RATE = 50
STEP_WEIGHTS = np.array([1.5, 1.5, 2.0])
THRESHOLD_REC = 10 ** 6


@dataclass
class Alignment:
    pop_times: np.ndarray        # anchor times in the original (seconds), strictly increasing
    cover_times: np.ndarray      # matching anchor times in the cover (seconds)
    sim: np.ndarray              # smoothed local similarity at each anchor (0..1)
    slope_ok: np.ndarray         # bool: local tempo ratio within limits at each anchor
    transpose: int               # semitones to ADD to cover notes to match the original key
    key_cost_margin: float       # how clearly the chosen transposition won (higher = surer)
    quality: float = 0.0         # mean similarity over the path
    bad_regions: list = field(default_factory=list)  # [(start, end)] in original-song seconds

    def cover_sim_at(self, t: np.ndarray) -> np.ndarray:
        good = (self.sim * self.slope_ok)
        return np.interp(t, self.cover_times, good)


def _features(y: np.ndarray, sr: int):
    from synctoolbox.feature.chroma import pitch_to_chroma, quantize_chroma
    from synctoolbox.feature.dlnco import pitch_onset_features_to_DLNCO
    from synctoolbox.feature.pitch import audio_to_pitch_features
    from synctoolbox.feature.pitch_onset import audio_to_pitch_onset_features
    from synctoolbox.feature.utils import estimate_tuning

    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        tuning = estimate_tuning(y, sr)
        f_pitch = audio_to_pitch_features(f_audio=y, Fs=sr, tuning_offset=tuning,
                                          feature_rate=FEATURE_RATE, verbose=False)
        chroma = quantize_chroma(f_chroma=pitch_to_chroma(f_pitch=f_pitch))
        f_onset = audio_to_pitch_onset_features(f_audio=y, Fs=sr, tuning_offset=tuning, verbose=False)
        dlnco = pitch_onset_features_to_DLNCO(f_peaks=f_onset, feature_rate=FEATURE_RATE,
                                              feature_sequence_length=chroma.shape[1], visualize=False)
    return chroma, dlnco


def _unit(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=0, keepdims=True) + 1e-8)


def _pool(x: np.ndarray, factor: int) -> np.ndarray:
    n = x.shape[1] // factor
    if n == 0:
        return x
    return x[:, : n * factor].reshape(x.shape[0], n, factor).mean(-1)


def detect_transpose(pop_chroma: np.ndarray, cover_chroma: np.ndarray, factor: int = 10):
    """Return (shift, margin): roll cover chroma by -shift to match the original.

    Uses a coarse (5 Hz) DTW per candidate transposition; the margin is the relative cost
    gap between the best and the second-best candidate.
    """
    import librosa

    p = _unit(_pool(pop_chroma, factor))
    costs = []
    for k in range(12):
        c = _unit(_pool(np.roll(cover_chroma, -k, axis=0), factor))
        C = 1.0 - p.T @ c
        D, wp = librosa.sequence.dtw(C=C, subseq=False, backtrack=True)
        costs.append(D[-1, -1] / len(wp))
    costs = np.array(costs)
    order = np.argsort(costs)
    best, second = costs[order[0]], costs[order[1]]
    k = int(order[0])
    return k, float((second - best) / (second + 1e-8))


def _moving_average(x: np.ndarray, n: int) -> np.ndarray:
    if n <= 1 or len(x) < n:
        return x
    kernel = np.ones(n) / n
    return np.convolve(np.pad(x, (n // 2, n - 1 - n // 2), mode="edge"), kernel, mode="valid")


def align(pop_y: np.ndarray, cover_y: np.ndarray, sr: int = SYNC_SR,
          smooth_sec: float = 2.0, sim_threshold: float = 0.35,
          slope_limits=(0.5, 2.0), min_bad_sec: float = 1.0) -> Alignment:
    """pop_y / cover_y: mono float audio at `sr`."""
    from synctoolbox.dtw.mrmsdtw import sync_via_mrmsdtw
    from synctoolbox.dtw.utils import make_path_strictly_monotonic

    pop_chroma, pop_onset = _features(pop_y, sr)
    cov_chroma, cov_onset = _features(cover_y, sr)

    k, margin = detect_transpose(pop_chroma, cov_chroma)
    cov_chroma = np.roll(cov_chroma, -k, axis=0)
    cov_onset = np.roll(cov_onset, -k, axis=0)

    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        wp = sync_via_mrmsdtw(f_chroma1=pop_chroma, f_onset1=pop_onset,
                              f_chroma2=cov_chroma, f_onset2=cov_onset,
                              input_feature_rate=FEATURE_RATE, step_weights=STEP_WEIGHTS,
                              threshold_rec=THRESHOLD_REC, verbose=False)
    wp = make_path_strictly_monotonic(wp).astype(np.int64)
    wp[0] = np.clip(wp[0], 0, pop_chroma.shape[1] - 1)
    wp[1] = np.clip(wp[1], 0, cov_chroma.shape[1] - 1)

    pu, cu = _unit(pop_chroma), _unit(cov_chroma)
    raw_sim = np.einsum("dt,dt->t", pu[:, wp[0]], cu[:, wp[1]])
    # silent frames (all-zero chroma) carry no evidence either way: treat as neutral-good
    silent = (np.abs(pop_chroma[:, wp[0]]).sum(0) < 1e-6) | (np.abs(cov_chroma[:, wp[1]]).sum(0) < 1e-6)
    raw_sim = np.where(silent, sim_threshold + 0.1, raw_sim)
    n_smooth = max(int(smooth_sec * FEATURE_RATE), 1)
    sim = _moving_average(raw_sim, n_smooth)

    pop_t = wp[0] / FEATURE_RATE
    cov_t = wp[1] / FEATURE_RATE
    # local tempo ratio (cover seconds per original second) over the smoothing window
    half = n_smooth // 2
    idx = np.arange(len(pop_t))
    lo, hi = np.clip(idx - half, 0, len(idx) - 1), np.clip(idx + half, 0, len(idx) - 1)
    slope = (cov_t[hi] - cov_t[lo]) / np.maximum(pop_t[hi] - pop_t[lo], 1e-3)
    slope_ok = (slope >= slope_limits[0]) & (slope <= slope_limits[1])

    good = (sim >= sim_threshold) & slope_ok
    bad_regions = []
    start = None
    for i, g in enumerate(good):
        if not g and start is None:
            start = pop_t[i]
        elif g and start is not None:
            if pop_t[i] - start >= min_bad_sec:
                bad_regions.append((float(start), float(pop_t[i])))
            start = None
    if start is not None and pop_t[-1] - start >= min_bad_sec:
        bad_regions.append((float(start), float(pop_t[-1])))

    shift = -k if k <= 6 else 12 - k  # semitones to add to cover notes (smallest magnitude)
    return Alignment(pop_times=pop_t, cover_times=cov_t, sim=sim, slope_ok=slope_ok,
                     transpose=int(shift), key_cost_margin=margin,
                     quality=float(raw_sim.mean()), bad_regions=bad_regions)

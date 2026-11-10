"""Monophonic f0 track (CREPE) -> sung notes. Numpy only.

Why: on the prototype songs basic-pitch's vocal notes agreed with a monophonic pitch
tracker (CREPE, and independently librosa pYIN) on only 41-63% of sung frames, with
octave errors, wrong pitches and missing notes; the melody is what a cover is built on.
Bass did not need this (65-87% agreement, the rest mostly octave errors that the arranger
folds anyway).

Segmentation: voiced frames are grouped into notes; a new note starts when the pitch moves
more than `split_semitones` away from the current note for at least `min_change_sec`
(vibrato and scoops do not split a note), or at a re-articulation onset (same pitch sung
again). Note pitch = median of the note's central frames.
"""

from __future__ import annotations

import numpy as np


def _median_filter(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1 or len(x) < k:
        return x.copy()
    pad = k // 2
    xp = np.pad(x, pad, mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(xp, k)
    return np.nanmedian(win, axis=1)


def f0_to_notes(midi: np.ndarray, voiced: np.ndarray, hop: float, level_db: np.ndarray | None = None,
                onsets: np.ndarray | None = None, split_semitones: float = 0.7, min_change_sec: float = 0.06,
                min_note_sec: float = 0.08, onset_guard_sec: float = 0.1) -> np.ndarray:
    """midi: f0 per frame in MIDI units (nan where unknown); voiced: bool per frame;
    level_db: loudness per frame (for velocity); onsets: re-articulation times (s).
    Returns (N, 4) [start, end, pitch, velocity]."""
    midi = np.asarray(midi, np.float64)
    voiced = np.asarray(voiced, bool) & np.isfinite(midi)
    n = len(midi)
    if n == 0 or not voiced.any():
        return np.zeros((0, 4))
    sm = np.where(voiced, midi, np.nan)
    sm = np.where(voiced, _median_filter(sm, 3), np.nan)
    min_change = max(int(round(min_change_sec / hop)), 1)
    onset_frames = set()
    if onsets is not None and len(onsets):
        onset_frames = set(np.round(np.asarray(onsets) / hop).astype(int).tolist())

    segments = []  # (start_frame, end_frame_exclusive)
    i = 0
    while i < n:
        if not voiced[i]:
            i += 1
            continue
        start = i
        anchor = np.nanmedian(sm[i:min(i + 3, n)])
        j = i + 1
        dev_run = 0
        while j < n and voiced[j]:
            if j - start >= int(onset_guard_sec / hop) and j in onset_frames \
                    and abs(sm[j] - anchor) <= split_semitones:
                break  # same pitch sung again
            if abs(sm[j] - anchor) > split_semitones:
                dev_run += 1
                if dev_run >= min_change:
                    j = j - dev_run + 1  # the new note starts where the deviation began
                    break
            else:
                dev_run = 0
                # slowly follow the note (drift), not the vibrato
                anchor = 0.9 * anchor + 0.1 * sm[j]
            j += 1
        end = j
        segments.append((start, end))
        i = end if end > start else start + 1

    out = []
    for s, e in segments:
        if (e - s) * hop < min_note_sec:
            continue
        core = sm[s:e]
        if e - s >= 6:  # ignore the scoop at the start and the fall at the end
            core = core[1:-1]
        pitch = int(np.round(np.nanmedian(core)))
        vel = 80.0
        if level_db is not None:
            vel = float(np.clip(60 + 1.2 * (np.nanmax(level_db[s:e]) + 30), 30, 120))
        out.append([s * hop, e * hop, pitch, vel])
    out = np.array(out, np.float64).reshape(-1, 4)
    # merge back-to-back fragments of the same pitch that were split by a short unvoiced dip
    merged = []
    for row in out:
        if merged and row[2] == merged[-1][2] and row[0] - merged[-1][1] < 0.05 \
                and not (onset_frames and int(round(row[0] / hop)) in onset_frames):
            merged[-1][1] = row[1]
        else:
            merged.append(row.copy())
    return np.array(merged, np.float64).reshape(-1, 4)

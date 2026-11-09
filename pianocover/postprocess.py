"""Post-processing for generated covers: cleanup, playability limits, beat-grid quantisation.

Everything here is deterministic and cheap. Defaults are conservative; tune per product.
"""

from __future__ import annotations

import numpy as np

from .events import END, PITCH, START, VEL, sanitize


def drop_ghost_notes(ev: np.ndarray, min_dur: float = 0.06, min_vel: int = 30) -> np.ndarray:
    """Remove very short AND quiet notes (typical generation noise). Pedals untouched."""
    notes = ev[:, PITCH] > 0
    ghost = notes & ((ev[:, END] - ev[:, START]) < min_dur) & (ev[:, VEL] < min_vel)
    return ev[~ghost]


def limit_hands(ev: np.ndarray, max_per_hand: int = 4, span: int = 12, split: int = 60,
                cluster_tol: float = 0.035) -> np.ndarray:
    """Make simultaneous onsets playable by two hands.

    Onsets within `cluster_tol` seconds form a chord. Notes >= `split` go to the right hand,
    the rest to the left (if a chord is entirely on one side the split moves to its widest
    gap). Right hand keeps the top note (melody) and the next notes below it within `span`
    semitones, at most `max_per_hand`; left hand keeps the bass note and notes above it.
    """
    notes_mask = ev[:, PITCH] > 0
    notes, other = ev[notes_mask], ev[~notes_mask]
    if len(notes) == 0:
        return ev
    order = np.argsort(notes[:, START], kind="stable")
    notes = notes[order]
    keep = np.zeros(len(notes), bool)
    i = 0
    while i < len(notes):
        j = i + 1
        while j < len(notes) and notes[j, START] - notes[i, START] <= cluster_tol:
            j += 1
        idx = np.arange(i, j)
        pitches = notes[idx, PITCH]
        s = split
        if (pitches >= s).all() or (pitches < s).all():
            if len(idx) > max_per_hand:
                srt = np.sort(pitches)
                gap = int(np.argmax(np.diff(srt)))
                s = srt[gap + 1]
            else:
                s = -1 if (pitches >= split).all() else 1000
        rh = idx[pitches >= s]
        lh = idx[pitches < s]
        for hand, top_down in ((rh, True), (lh, False)):
            if len(hand) == 0:
                continue
            ordered = hand[np.argsort(-notes[hand, PITCH] if top_down else notes[hand, PITCH], kind="stable")]
            anchor = notes[ordered[0], PITCH]
            chosen = [ordered[0]]
            for k in ordered[1:]:
                if len(chosen) >= max_per_hand:
                    break
                if abs(notes[k, PITCH] - anchor) <= span:
                    chosen.append(k)
            keep[chosen] = True
        i = j
    out = np.concatenate([notes[keep], other]) if len(other) else notes[keep]
    return out[np.lexsort((out[:, PITCH], out[:, START]))]


def beat_grid(beats: np.ndarray, subdivisions: int = 4) -> np.ndarray:
    beats = np.asarray(sorted(beats), dtype=np.float64)
    if len(beats) < 2:
        return beats
    grid = [np.linspace(a, b, subdivisions, endpoint=False) for a, b in zip(beats[:-1], beats[1:])]
    grid.append([beats[-1]])
    return np.concatenate(grid)


def quantize(ev: np.ndarray, beats: np.ndarray, subdivisions: int = 4, max_shift: float = 0.07) -> np.ndarray:
    """Snap onsets/offsets to a beat subdivision grid when they are within `max_shift` seconds.

    Keeps notes at least one grid step long. Useful before engraving sheet music.
    """
    grid = beat_grid(beats, subdivisions)
    if len(grid) < 2 or len(ev) == 0:
        return ev
    out = ev.copy()

    def snap(t):
        k = np.clip(np.searchsorted(grid, t), 1, len(grid) - 1)
        left, right = grid[k - 1], grid[k]
        nearest = np.where(t - left <= right - t, left, right)
        return np.where(np.abs(nearest - t) <= max_shift, nearest, t), np.where(t - left <= right - t, k - 1, k)

    out[:, START], si = snap(ev[:, START])
    out[:, END], ei = snap(ev[:, END])
    step = np.diff(grid).mean()
    too_short = out[:, END] - out[:, START] < 0.5 * step
    nxt = np.clip(si + 1, 0, len(grid) - 1)
    out[too_short, END] = np.maximum(grid[nxt[too_short]], out[too_short, START] + 0.5 * step)
    return sanitize(out)


def estimate_beats(audio_path) -> np.ndarray:
    """Beat times via librosa (good enough for a grid; plug in beat_this for better results)."""
    import librosa

    y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
    _, beats = librosa.beat.beat_track(y=y, sr=sr, units="time")
    return np.asarray(beats, dtype=np.float64)


def postprocess(ev: np.ndarray, beats: np.ndarray | None = None, playable: bool = True,
                max_per_hand: int = 4, span: int = 12) -> np.ndarray:
    ev = drop_ghost_notes(ev)
    if playable:
        ev = limit_hands(ev, max_per_hand=max_per_hand, span=span)
    if beats is not None and len(beats) > 1:
        ev = quantize(ev, beats)
    return sanitize(ev)

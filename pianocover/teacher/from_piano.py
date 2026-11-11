"""Good piano arrangement -> song-like parts (the easy direction).

Rules cannot write a good piano arrangement from song parts, but splitting a human-written
piano piece into "what a band would play" needs no taste: it only throws information away.
This builds, from a piano MIDI, the same Analysis the audio pipeline produces for a song
(melody, bass, other, chords, beats, drum-like onsets, loudness curves), so that
(derived parts -> original piano) pairs can train an arranger whose targets are human-made.

Roles (heuristic, per onset cluster):
  melody = the top line: highest note of a cluster, if it stays near the running melody
           register (a much lower "top note" is accompaniment showing through a rest)
  bass   = the lowest note of a cluster in the low register, near the running bass register
  other  = everything else (inner voices, chords, figuration) -> the "guitar / keys" part
Chords are template-matched per two beats from all notes; "drums" are a rhythm skeleton
(kick on bass onsets, snare on the off-beats that are played, hi-hat on every onset).
"""

from __future__ import annotations

import numpy as np

from .analysis import FRAME_HOP, Analysis
from .chords import QUALITIES

NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
CHORD_TYPES = ["maj", "min", "7", "maj7", "min7", "dim", "sus4", "hdim7"]


def read_piano(path) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(str(path))
    rows = [[n.start, n.end, n.pitch, n.velocity] for i in pm.instruments if not i.is_drum for n in i.notes]
    ev = np.array(sorted(rows), np.float64).reshape(-1, 4)
    return ev, np.asarray(pm.get_beats()), np.asarray(pm.get_downbeats()), float(pm.get_end_time())


def split_roles(ev: np.ndarray, tol: float = 0.03) -> np.ndarray:
    """Role per note: 0 melody, 1 bass, 2 other."""
    role = np.full(len(ev), 2)
    order = np.argsort(ev[:, 0], kind="stable")
    clusters, cur = [], [order[0]]
    for i in order[1:]:
        if ev[i, 0] - ev[cur[0], 0] <= tol:
            cur.append(i)
        else:
            clusters.append(cur)
            cur = [i]
    clusters.append(cur)
    tops = np.array([ev[c, 2].max() for c in clusters])
    lows = np.array([ev[c, 2].min() for c in clusters])
    mel_ref = [float(np.median(tops[tops >= 60]))] if (tops >= 60).any() else [72.0]
    bass_ref = [float(np.median(lows[lows <= 55]))] if (lows <= 55).any() else [43.0]
    for c in clusters:
        c = np.array(c)
        top, low = c[np.argmax(ev[c, 2])], c[np.argmin(ev[c, 2])]
        ref = float(np.median(mel_ref[-8:]))
        if ev[top, 2] >= max(ref - 9, 55):
            role[top] = 0
            mel_ref.append(ev[top, 2])
        bref = float(np.median(bass_ref[-8:]))
        if low != top and ev[low, 2] <= min(bref + 9, 57):
            role[low] = 1
            bass_ref.append(ev[low, 2])
    return role


def detect_chords(ev: np.ndarray, role: np.ndarray, beats: np.ndarray, beats_per_chord: int = 2):
    spans, labels = [], []
    for s, e in zip(beats[:-beats_per_chord:beats_per_chord], beats[beats_per_chord::beats_per_chord]):
        ov = np.clip(np.minimum(ev[:, 1], e) - np.maximum(ev[:, 0], s), 0, None)
        if ov.sum() <= 0:
            lab = "N"
        else:
            w = np.zeros(12)
            np.add.at(w, ev[:, 2].astype(int) % 12, ov * np.where(role == 1, 2.0, 1.0))
            w /= w.sum()
            bw = np.zeros(12)
            np.add.at(bw, ev[role == 1, 2].astype(int) % 12, ov[role == 1])
            best, lab = -1e9, "N"
            for root in range(12):
                for q in CHORD_TYPES:
                    pcs = [(root + i) % 12 for i in QUALITIES[q]]
                    score = w[pcs].sum() - 0.6 * (1 - w[pcs].sum()) - 0.04 * len(pcs) \
                        - 0.15 * sum(w[p] < 0.03 for p in pcs) + (0.1 if bw.sum() and np.argmax(bw) == root else 0)
                    if score > best:
                        best, lab = score, NAMES[root] if q == "maj" else f"{NAMES[root]}:{q}"
        if labels and labels[-1] == lab:
            spans[-1][1] = e
        else:
            spans.append([s, e])
            labels.append(lab)
    return np.array(spans, np.float64).reshape(-1, 2), labels


def level_curve(ev: np.ndarray, duration: float, decay: float = 1.5) -> np.ndarray:
    """Loudness curve of piano notes: velocity energy with the piano's decay."""
    n = int(duration / FRAME_HOP) + 1
    t = np.arange(n) * FRAME_HOP
    e = np.full(n, 1e-8)
    for s, en, p, v in ev:
        a, b = int(s / FRAME_HOP), min(int(en / FRAME_HOP) + 1, n)
        if b > a:
            e[a:b] += (v / 127.0) ** 2 * np.exp(-(t[a:b] - s) / decay)
    return np.sqrt(e) * 0.2


def analysis_from_piano(path) -> tuple[Analysis, np.ndarray]:
    ev, beats, downbeats, duration = read_piano(path)
    role = split_roles(ev)
    mel = ev[role == 0].copy()
    mel = mel[np.argsort(mel[:, 0], kind="stable")]
    if len(mel) > 1:  # a melody is one note at a time
        mel[:-1, 1] = np.minimum(mel[:-1, 1], mel[1:, 0])
        mel = mel[mel[:, 1] - mel[:, 0] > 0.02]
    bass, other = ev[role == 1], ev[role == 2]
    chords, labels = detect_chords(ev, role, beats)

    # rhythm skeleton as kick / snare / hi-hat
    drums = [[s, 0, v / 127.0] for s, e, p, v in bass]
    onsets = np.unique(np.round(ev[:, 0], 2))
    drums += [[t, 2, 0.5] for t in onsets]
    for i, b in enumerate(beats):
        k = int(np.searchsorted(downbeats, b + 1e-3)) - 1
        pos = i - int(np.searchsorted(beats, downbeats[max(k, 0)] - 1e-3)) if len(downbeats) else i
        if pos % 2 == 1 and np.any(np.abs(onsets - b) < 0.04):
            near = ev[np.abs(ev[:, 0] - b) < 0.04]
            drums.append([b, 1, float(near[:, 3].max()) / 127.0])
    drums = np.array(sorted(drums), np.float64).reshape(-1, 3)

    a = Analysis(
        duration=duration, beats=beats, downbeats=downbeats, chords=chords, chord_labels=labels,
        vocal=mel, bass=bass, other=other, other_piano=np.zeros((0, 4)), drum_onsets=drums,
        mix_rms=level_curve(ev, duration), vocal_rms=level_curve(mel, duration),
        other_rms=level_curve(other, duration), vocal_f0=mel,
        meta={"source_midi": str(path), "derived_from": "piano"},
    )
    return a, ev

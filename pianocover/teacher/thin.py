"""'Transcribe-and-thin' arranger: stay close to the song.

Listening feedback on the reduction and piano-texture arrangers: "far from the song". This
arranger starts from what a piano transcriber hears in the whole song without drums
(Transkun on vocals+bass+other), which already sounds like the song, then:
  1. puts the sung melody (CREPE notes, shaped) clearly on top: transcribed notes at or just
     below the melody, or a semitone from it, are removed while it sounds;
  2. removes transcription errors (out-of-key AND out-of-chord short notes, harmonics);
  3. thins to the level's density, keeping the bass (lowest note per beat) and the most
     salient notes (loud, long, chord tones);
  4. playable hands, clash-free pedal, melody louder than the accompaniment.
Every note is either the transcribed song or the sung melody.
"""

from __future__ import annotations

import numpy as np

from ..events import END, PITCH, START, VEL, sanitize
from ..postprocess import drop_ghost_notes, limit_hands
from .arrange import LEVELS, Song, _notes, melody_part, pedal_part

# non-melody notes per second kept at each level
DENSITY = {"easy": 1.5, "simple": 3.0, "normal": 5.5, "full": 8.5, "virtuoso": 13.0}


def arrange_thin(a, level: str = "normal", song: Song | None = None, mix_notes: np.ndarray | None = None):
    lv = LEVELS[level]
    song = song or Song(a)
    stats = {"level": level, "texture": "thin"}
    base = _notes(mix_notes if mix_notes is not None else getattr(a, "mix_piano", np.zeros((0, 4))))
    if len(base) == 0:
        raise ValueError("no Transkun transcription of the drumless mix (analysis.mix_piano)")
    melody = melody_part(song, lv, stats)

    # 1. melody on top
    keep = np.ones(len(base), bool)
    for s, e, p, v in melody:
        over = (base[:, START] < e) & (base[:, END] > s)
        too_high = base[:, PITCH] >= p - 2
        rub = np.isin((base[:, PITCH] - p) % 12, (1, 11))
        same_pc_near = (base[:, PITCH] % 12 == p % 12) & (np.abs(base[:, PITCH] - p) < 12)
        keep &= ~(over & (too_high | rub | same_pc_near))
    acc = base[keep]
    stats["removed_for_melody"] = int((~keep).sum())

    # 2. transcription errors: harmonics / bleed that are neither in key nor in the chord
    n0 = len(acc)
    acc, _ = song.clean_pitches(acc, short=0.3, snap_min=10.0)  # drop only, never move
    stats["removed_errors"] = n0 - len(acc)

    # 3. thin to the level's density, beat by beat
    target_per_beat = DENSITY[level] * song.period
    out, carry = [], 0.0
    for b0, b1 in zip(song.beats[:-1], song.beats[1:]):
        m = (acc[:, START] >= b0) & (acc[:, START] < b1)
        if not m.any():
            continue
        cand = acc[m]
        c = song.chord_at(b0 + 0.02)
        dur = np.minimum(cand[:, END] - cand[:, START], 1.5)
        sal = cand[:, VEL] / 127 * (0.3 + dur)
        if c is not None:
            sal *= np.where(np.isin(cand[:, PITCH].astype(int) % 12, list(c[3])), 1.5, 0.7)
        lowest = int(np.argmin(cand[:, PITCH]))
        sal[lowest] += 10.0  # the bass note of the beat always survives
        carry += target_per_beat
        k = int(carry)
        carry -= k
        k = max(k, 1)
        order = np.argsort(-sal)[:k]
        out.append(cand[order])
    acc = np.concatenate(out) if out else np.zeros((0, 4))

    # 4. dynamics, hands, pedal
    melody = melody.copy()
    if len(melody):
        melody[:, VEL] = [70 + 25 * song.energy_at(t) for t in melody[:, START]]
    if len(acc):
        acc = acc.copy()
        acc[:, VEL] = np.clip(acc[:, VEL] * 0.75, 30, 85)
    parts = [melody, acc]
    if lv.pedal:
        parts.append(pedal_part(song, np.concatenate([p.reshape(-1, 4) for p in parts])))
    ev = sanitize(np.concatenate([p.reshape(-1, 4) for p in parts]))
    ev = limit_hands(ev, max_per_hand=lv.max_per_hand, span=lv.span)
    ev = sanitize(drop_ghost_notes(ev))
    ev = ev[ev[:, START] < song.duration]
    ev[:, END] = np.minimum(ev[:, END], song.duration)
    ev = sanitize(ev)
    stats.update(notes=int((ev[:, PITCH] > 0).sum()), melody_notes=len(melody), accomp_notes=len(acc))
    return ev, stats

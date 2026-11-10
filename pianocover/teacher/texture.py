"""Piano-texture arranger: the song's own notes laid out as idiomatic piano figures.

The reduction arranger (arrange.py) copies the band's rhythms (guitar strums, drum hits)
onto the piano; measured flow problems were fixed, but it still does not sound like a
pianist. Pianists flow because of *figures*: left-hand broken chords / arpeggios, bass on 1
with chords on 2 and 4, octaves in loud sections, chord tones under the melody.

What still comes from the song (reduction):
  * the melody (shaped vocal line / lead riff, see arrange.melody_part)
  * the bass note of every beat (the real bass line, so walking bass and inversions stay)
  * the chord's pitch classes, as actually played in that span (verified chords)
  * energy per section, beat grid and downbeats
What is generated: the *layout* (which figure, register, rhythm), chosen per 4-bar section
from the section's energy and the difficulty level.
"""

from __future__ import annotations

import numpy as np

from ..events import END, PITCH, START, VEL, sanitize
from ..postprocess import drop_ghost_notes, limit_hands
from .arrange import LEVELS, Song, _notes, bass_part, melody_part, pedal_part


def _span_pcs(song: Song, s: float, e: float, chord) -> list:
    """Chord pitch classes heard in [s, e), most-played first (verified chord if any)."""
    a = song.a
    pool = np.concatenate([_notes(a.other), _notes(a.other_piano), _notes(a.vocal)])
    ov = np.clip(np.minimum(pool[:, END], e) - np.maximum(pool[:, START], s), 0, None) if len(pool) else []
    w = np.zeros(12)
    if len(pool):
        np.add.at(w, pool[:, PITCH].astype(int) % 12, ov)
    if chord is not None:
        pcs = sorted(chord[3], key=lambda pc: -w[pc])
        return pcs
    order = [int(pc) for pc in np.argsort(-w) if w[pc] > 0]
    chosen = []
    for pc in order:  # no chord label: most-played, no semitone pairs, at most 3
        if len(chosen) == 3:
            break
        if all((pc - q) % 12 not in (1, 11) for q in chosen):
            chosen.append(pc)
    return chosen


def _place(pc: int, lo: int, hi: int):
    opts = [p for p in range(lo, hi + 1) if p % 12 == pc]
    return opts


def _bass_at(bass: np.ndarray, t: float, chord, fallback_pcs):
    """Real bass note sounding at (or starting just after) t, else the chord's bass."""
    if len(bass):
        m = (bass[:, START] <= t + 0.06) & (bass[:, END] > t)
        if m.any():
            return int(bass[m][np.argmax(bass[m][:, START]), PITCH])
    pc = chord[4] if chord is not None else (fallback_pcs[0] if fallback_pcs else None)
    if pc is None:
        return None
    return 36 + (pc - 36) % 12


def _lh_figure(level: str, energy: float):
    """Return (name, subdivision per beat). Pattern choice per level and section energy."""
    loud = energy >= 0.6
    if level == "easy":
        return "hold", 1
    if level == "simple":
        return ("bass_chord", 1) if loud else ("hold_half", 1)
    if level == "normal":
        return ("pop", 2) if loud else ("arp8", 2)
    if level == "full":
        return ("pop_oct", 2) if loud else ("arp8_wide", 2)
    return ("pop_oct", 2) if loud else ("arp16", 4)  # virtuoso


def _figure_notes(fig, sub, beats_in_unit, t0, period, bass, tones):
    """LH notes [start, end, pitch] for one chord unit. tones: chord tones above the bass."""
    step = period / sub
    n = int(round(beats_in_unit * sub))
    out = []
    if not tones:
        tones = [bass + 7]
    if fig == "hold":
        out.append([t0, t0 + beats_in_unit * period, bass])
    elif fig == "hold_half":
        for k in range(0, n, 2):
            out.append([t0 + k * step, t0 + min(k + 2, n) * step, bass])
            if len(tones) and k % 4 == 2:
                out.append([t0 + k * step, t0 + min(k + 2, n) * step, tones[0]])
    elif fig == "bass_chord":  # bass on 1 / 3, chord on 2 / 4
        for k in range(n):
            if k % 2 == 0:
                out.append([t0 + k * step, t0 + (k + 1) * step, bass])
            else:
                out += [[t0 + k * step, t0 + (k + 1) * step, p] for p in tones[:2]]
    elif fig in ("arp8", "arp8_wide", "arp16"):
        seq = [bass] + tones[:3]
        if fig != "arp8":
            seq = [bass] + tones[:4]
        seq = seq + seq[-2:0:-1]  # up and back down: B 5 8 10 8 5 ...
        for k in range(n):
            out.append([t0 + k * step, t0 + (k + 1) * step, seq[k % len(seq)]])
    elif fig in ("pop", "pop_oct"):  # bass (+octave) on beats 1 and 3, chord on the off-beats
        for k in range(n):
            if k % 4 == 0:
                out.append([t0 + k * step, t0 + (k + 2) * step, bass])
                if fig == "pop_oct" and bass - 12 >= 28:
                    out.append([t0 + k * step, t0 + (k + 2) * step, bass - 12])
            elif k % 2 == 1 or k % 4 == 2:
                out += [[t0 + k * step, t0 + (k + 1) * step, p] for p in tones[:3]]
    return [r for r in out if r[1] > r[0]]


def arrange_piano(a, level: str = "normal", song: Song | None = None):
    lv = LEVELS[level]
    song = song or Song(a)
    stats = {"level": level, "texture": "piano"}
    melody = melody_part(song, lv, stats)
    bass_line = bass_part(song, LEVELS["normal"], {})  # the real bass line, legato, on the grid

    # chord units: chord spans split at bar lines; spans without a chord -> bar units
    units = []
    chords = [c for c in song.chords if c[2] is not None]
    cut_times = sorted({0.0, song.duration} | {c[0] for c in chords} | {c[1] for c in chords}
                       | {float(b) for b in song.bars if 0 <= b < song.duration})
    for s, e in zip(cut_times[:-1], cut_times[1:]):
        if e - s >= 0.5 * song.period:
            units.append((s, e))

    lh, rh_harm = [], []
    for s, e in units:
        c = song.chord_at(s + 0.02)
        pcs = _span_pcs(song, s, e, c)
        if not pcs:
            continue
        energy = song.energy_at(s)
        fig, sub = _lh_figure(level, energy)
        beats = (e - s) / song.period
        # walk the unit beat by beat so a moving bass line is followed
        n_beats = max(int(round(beats)), 1)
        beat_len = (e - s) / n_beats
        group = 2 if fig not in ("hold",) else n_beats
        b = 0
        while b < n_beats:
            g = min(group, n_beats - b)
            t0 = s + b * beat_len
            bass = _bass_at(bass_line, t0, c, pcs)
            if bass is None:
                b += g
                continue
            bass = int(np.clip(bass, 28, 50))
            tones = []
            for pc in pcs:
                if pc == bass % 12 and len(pcs) > 2:
                    continue
                opts = _place(pc, bass + 3, bass + 16)
                if opts:
                    tones.append(opts[0])
            tones = sorted(set(tones))
            if fig in ("arp8_wide", "arp16") and tones:
                tones = tones + [tones[0] + 12]
            # keep the LH below the melody
            mel_lo = melody[(melody[:, START] < t0 + g * beat_len) & (melody[:, END] > t0), PITCH] \
                if len(melody) else np.zeros(0)
            ceiling = int(mel_lo.min()) - 2 if len(mel_lo) else 67
            tones = [p for p in tones if p <= ceiling] or tones[:1]
            for st, en, p in _figure_notes(fig, sub, g, t0, beat_len, bass, tones):
                accent = 6 if abs(st - song.bars[song.bar_index(st)]) < 0.03 else 0
                lh.append([st, en, p, 48 + 22 * energy + accent])
            b += g

        # RH harmony under melody notes that start on a beat (normal and harder)
        n_under = {"easy": 0, "simple": 0, "normal": 1, "full": 2, "virtuoso": 2}[level]
        if n_under and len(melody):
            m = melody[(melody[:, START] >= s) & (melody[:, START] < e)]
            for st, en, p, v in m:
                on_beat = np.min(np.abs(song.beats - st)) < 0.03
                if not on_beat:
                    continue
                under = []
                for pc in pcs:
                    opts = [q for q in _place(pc, int(p) - 9, int(p) - 3)]
                    if opts and all((q - p) % 12 not in (1, 11) for q in opts):
                        under.append(opts[-1])
                for q in sorted(set(under), reverse=True)[:n_under]:
                    rh_harm.append([st, min(en, st + 2 * song.period), q, 52 + 20 * energy])

    melody = melody.copy()
    if len(melody):
        melody[:, VEL] = [66 + 24 * song.energy_at(t) for t in melody[:, START]]
    parts = [melody, np.array(lh, np.float64).reshape(-1, 4), np.array(rh_harm, np.float64).reshape(-1, 4)]
    if lv.pedal or level == "simple":
        parts.append(pedal_part(song, np.concatenate([p.reshape(-1, 4) for p in parts])))
    ev = sanitize(np.concatenate([p.reshape(-1, 4) for p in parts]))
    ev = limit_hands(ev, max_per_hand=lv.max_per_hand, span=lv.span)
    ev = sanitize(drop_ghost_notes(ev))
    ev = ev[ev[:, START] < song.duration]
    ev[:, END] = np.minimum(ev[:, END], song.duration)
    ev = sanitize(ev)
    stats.update(notes=int((ev[:, PITCH] > 0).sum()), melody_notes=len(melody), lh_notes=len(lh),
                 rh_harmony_notes=len(rh_harm))
    return ev, stats

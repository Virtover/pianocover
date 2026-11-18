"""Hybrid cover: the teacher's melody and beat grid + AMT-APC's piano accompaniment.

AMT-APC sounds pianistic but has no beat grid and buries the tune; the teacher has a clear
vocal melody and the song's beats but rule-made accompaniment. The hybrid keeps:
  * RH melody  = teacher melody_part (the sung line), on top and a little louder;
  * the rest   = AMT-APC notes, onsets snapped to the 16th grid when within `snap_tol`,
                 minus notes that would sit at or above the melody while it sounds
                 (so the tune stays the top voice); outside sung bars AMT-APC plays as is;
  * pedal      = recomputed with the teacher's clash-free pedal.
"""

from __future__ import annotations

import numpy as np

from ..events import END, PITCH, START, VEL, sanitize
from ..postprocess import limit_hands
from .arrange import LEVELS, Song, bass_part, melody_part, pedal_part


def midi_notes(path) -> np.ndarray:
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(str(path))
    rows = [[n.start, n.end, n.pitch, n.velocity] for i in pm.instruments if not i.is_drum for n in i.notes]
    return np.array(rows, np.float64).reshape(-1, 4)


def snap(ev: np.ndarray, grid: np.ndarray, tol: float) -> np.ndarray:
    ev = ev.copy()
    if not len(ev) or not len(grid):
        return ev
    i = np.clip(np.searchsorted(grid, ev[:, START]), 1, len(grid) - 1)
    near = np.where(np.abs(grid[i - 1] - ev[:, START]) <= np.abs(grid[i] - ev[:, START]), grid[i - 1], grid[i])
    d = near - ev[:, START]
    m = np.abs(d) <= tol
    ev[m, START] += d[m]
    ev[m, END] = np.maximum(ev[m, END] + d[m], ev[m, START] + 0.05)
    return ev


def arrange_hybrid(song: Song, apc: np.ndarray, level: str = "normal", snap_tol: float = 0.06,
                   below: int = 2, apc_gain: float = 0.8):
    stats = {"level": f"hybrid/{level}"}
    melody = melody_part(song, LEVELS[level], stats)
    acc = snap(apc[apc[:, PITCH] > 0], song.grid(4), snap_tol)
    # AMT-APC's right hand usually sits higher than the sung register (measured 7-11
    # semitones on 2 of 3 songs): lift the melody by octaves to its top voice, else the
    # "stay below the melody" rule strips most of AMT-APC's right hand
    if len(melody) and len(acc):
        cl = np.round(acc[:, START] / 0.04)
        top = np.array([acc[cl == k, PITCH].max() for k in np.unique(cl)])
        shift = 12 * int(np.clip(np.round((np.median(top) - np.median(melody[:, PITCH])) / 12), 0, 1))
        if np.percentile(melody[:, PITCH], 95) + shift > 98:
            shift = 0
        melody = melody.copy()
        melody[:, PITCH] += shift
        stats["melody_shift"] = shift
    keep = np.ones(len(acc), bool)
    for s, e, p, _v in melody:  # notes starting while a melody note sounds must stay below it
        m = (acc[:, START] >= s - 0.03) & (acc[:, START] < e) & (acc[:, PITCH] > p - below)
        keep &= ~m
    stats["apc_notes"], stats["apc_removed_above_melody"] = len(acc), int((~keep).sum())
    acc = acc[keep]
    acc[:, VEL] = np.clip(acc[:, VEL] * apc_gain, 1, 127)
    melody = melody.copy()
    if len(melody):
        melody[:, VEL] = [np.clip(50 + 70 * song.dyn_at(t) ** 1.2 + 8 * song.vocal_push_at(t), 1, 127)
                          for t in melody[:, START]]
    notes = sanitize(np.concatenate([melody, acc]))
    notes = limit_hands(notes, max_per_hand=5, span=12)
    ev = sanitize(np.concatenate([notes, pedal_part(song, notes)]))
    ev = ev[ev[:, START] < song.duration]
    stats.update(notes=int((ev[:, PITCH] > 0).sum()), melody_notes=len(melody))
    return ev, stats


# --------------------------------------------------------------------------- role-split variants
# Summing two full arrangements (above) gave clashes (two sources hearing different chords /
# playing the tune in different rhythms) and "fist" chords (up to 10 notes, dense low
# clusters, loud). The variants below take one ROLE from each source instead:
#   melody + bass + harmony = teacher;  middle-voice notes (C) or only their rhythm (D) = AMT-APC.

def lifted_melody(song: Song, apc: np.ndarray, level: str = "normal", stats: dict | None = None):
    """Teacher melody, lifted an octave when AMT-APC's top voice sits that much higher."""
    stats = {} if stats is None else stats
    melody = melody_part(song, LEVELS[level], stats).copy()
    if len(melody) and len(apc):
        cl = np.round(apc[:, START] / 0.04)
        top = np.array([apc[cl == k, PITCH].max() for k in np.unique(cl)])
        shift = 12 * int(np.clip(np.round((np.median(top) - np.median(melody[:, PITCH])) / 12), 0, 1))
        if np.percentile(melody[:, PITCH], 95) + shift > 98:
            shift = 0
        melody[:, PITCH] += shift
        stats["melody_shift"] = shift
    return melody


def _sounding(ev: np.ndarray, t: float, lead: float = 0.06):
    """Pitch of the note of `ev` sounding at t (latest start), or None."""
    if not len(ev):
        return None
    m = (ev[:, START] <= t + lead) & (ev[:, END] > t)
    return float(ev[m][np.argmax(ev[m][:, START]), PITCH]) if m.any() else None


def _pcs_at(song: Song, t: float):
    from .texture import _span_pcs

    c = song.chord_at(t + 0.02)
    return set(c[3]) if c is not None else set(_span_pcs(song, t - 0.3, t + 0.3, None))


def _band(song, melody, bass, t):
    """Middle register at t: above the bass, below the melody."""
    b = _sounding(bass, t)
    m = _sounding(melody, t)
    lo = max(48, (b if b is not None else 40) + 3)
    hi = (m - 3) if m is not None else 76
    return int(lo), int(hi)


def tame(song: Song, melody: np.ndarray, bass: np.ndarray, middle: np.ndarray, stats: dict,
         max_mid: int = 3, dense_pedal: float = 7.0):
    """Shared anti-"fist" rules: compressed dynamics, no close low chords, at most 4 notes
    per hand, no pedal over dense passages (> `dense_pedal` onsets per second)."""
    def dyn(ev, base, span):
        ev = ev.copy()
        if len(ev):
            ev[:, VEL] = [np.clip(base + span * song.dyn_at(t), 1, 127) for t in ev[:, START]]
        return ev

    melody, bass, middle = dyn(melody, 58, 34), dyn(bass, 44, 30), dyn(middle, 34, 26)
    # middle: per onset cluster keep the `max_mid` highest notes (nearest the melody)
    if len(middle):
        cl = np.round(middle[:, START] / 0.04)
        keep = np.zeros(len(middle), bool)
        for k in np.unique(cl):
            idx = np.flatnonzero(cl == k)
            keep[idx[np.argsort(-middle[idx, PITCH])[:max_mid]]] = True
        middle = middle[keep]
    notes = sanitize(np.concatenate([melody, bass, middle]))
    # below C3 only octaves / fifths over the lowest note of a cluster
    cl = np.round(notes[:, START] / 0.04)
    keep = np.ones(len(notes), bool)
    for k in np.unique(cl):
        idx = np.flatnonzero((cl == k) & (notes[:, PITCH] < 48))
        if len(idx) > 1:
            low = notes[idx, PITCH].min()
            ok = np.isin((notes[idx, PITCH] - low) % 12, (0, 7))
            keep[idx[~ok]] = False
    stats["low_cluster_notes_removed"] = int((~keep).sum())
    notes = limit_hands(notes[keep], max_per_hand=4, span=12)
    pedal = pedal_part(song, notes)
    if len(pedal):
        on = notes[:, START]
        rate = np.array([((on >= s) & (on < e)).sum() / max(e - s, 1e-3) for s, e, _p, _v in pedal])
        pedal = pedal[rate <= dense_pedal]
    ev = sanitize(np.concatenate([notes, pedal]))
    ev = ev[ev[:, START] < song.duration]
    ev[:, END] = np.minimum(ev[:, END], song.duration)
    stats.update(notes=int((ev[:, PITCH] > 0).sum()), melody_notes=len(melody), bass_notes=len(bass))
    return sanitize(ev), stats


def arrange_hybrid_filtered(song: Song, apc: np.ndarray, snap_tol: float = 0.06):
    """C: teacher melody + teacher bass; AMT-APC notes only in the middle band and only if
    they are chord tones of the teacher's (verified) chord at that moment."""
    stats = {"level": "hybrid_filtered"}
    apc = apc[apc[:, PITCH] > 0]
    melody = lifted_melody(song, apc, stats=stats)
    bass = bass_part(song, LEVELS["normal"], {})
    acc = snap(apc, song.grid(4), snap_tol)
    keep = np.zeros(len(acc), bool)
    for i, (s, _e, p, _v) in enumerate(acc):
        lo, hi = _band(song, melody, bass, s)
        keep[i] = lo <= p <= hi and int(p) % 12 in _pcs_at(song, s)
    stats["apc_notes"], stats["apc_kept"] = len(acc), int(keep.sum())
    mid = acc[keep]
    mid[:, END] = np.minimum(mid[:, END], mid[:, START] + 2 * song.period)
    return tame(song, melody, bass, mid, stats)


def polish_apc(song: Song, apc: np.ndarray, snap_tol: float = 0.04):
    """E: AMT-APC alone with only the anti-"fist" rules of `tame` (no teacher notes): the
    top voice per cluster is treated as melody for dynamics, notes < 48 as bass."""
    stats = {"level": "apc_polished"}
    acc = snap(apc[apc[:, PITCH] > 0], song.grid(4), snap_tol)
    cl = np.round(acc[:, START] / 0.04)
    top = np.zeros(len(acc), bool)
    for k in np.unique(cl):
        idx = np.flatnonzero(cl == k)
        if acc[idx, PITCH].max() >= 60:
            top[idx[np.argmax(acc[idx, PITCH])]] = True
    low = ~top & (acc[:, PITCH] < 48)
    return tame(song, acc[top], acc[low], acc[~top & ~low], stats, max_mid=4)


def fill_gaps(song: Song, apc: np.ndarray, teacher: np.ndarray, beats_per_unit: int = 2,
              add_melody: bool = False, melody: np.ndarray | None = None, min_gap_units: int = 1,
              correct: bool = False):
    """F / G: AMT-APC is kept untouched; the teacher only fills where AMT-APC is missing.

    A unit (`beats_per_unit` beats) is a GAP when AMT-APC starts no note in it and less
    than half of it is covered by sounding AMT-APC notes. Teacher notes starting in gap
    units are added (cut at the gap's end), at AMT-APC's own loudness.
    add_melody (G): also add the teacher melody where AMT-APC plays but not the tune, i.e.
    no AMT-APC note within 2 semitones (or an octave of that) of the melody note's pitch
    starts during it; those melody notes get AMT-APC's top-voice loudness."""
    stats = {"level": "gap_fill" + ("+melody" if add_melody else "") + ("+correct" if correct else "")}
    apc = apc[apc[:, PITCH] > 0]
    if correct:  # H: rare wrong notes = outside BOTH key and chord; short ones dropped,
        n0 = len(apc)  # long ones moved a semitone to a chord tone (Song.clean_pitches)
        fixed, suspect = song.clean_pitches(apc)
        stats["apc_suspect"], stats["apc_dropped"] = int(suspect.sum()), n0 - len(fixed)
        apc = fixed
    b = song.beats
    units = np.r_[b[::beats_per_unit], song.duration]
    units = units[(units >= 0) & (units <= song.duration)]
    gap = np.zeros(len(units) - 1, bool)
    for i, (s, e) in enumerate(zip(units[:-1], units[1:])):
        starts = ((apc[:, START] >= s) & (apc[:, START] < e)).any()
        cov = np.clip(np.minimum(apc[:, END], e) - np.maximum(apc[:, START], s), 0, None)
        covered = min(cov.max() if len(cov) else 0, e - s) / max(e - s, 1e-3)
        gap[i] = not starts and covered < 0.5
    stats["gap_units"], stats["units"] = int(gap.sum()), len(gap)
    vel = float(np.median(apc[:, VEL])) if len(apc) else 64.0
    t_notes = teacher[teacher[:, PITCH] > 0]
    scale = vel / max(float(np.median(t_notes[:, VEL])), 1.0) if len(t_notes) else 1.0
    add = []
    if len(t_notes):
        u = np.clip(np.searchsorted(units, t_notes[:, START], side="right") - 1, 0, len(gap) - 1)
        m = gap[u]
        f = t_notes[m].copy()
        f[:, END] = np.minimum(f[:, END], units[u[m] + 1] + 0.05)
        f[:, VEL] = np.clip(f[:, VEL] * scale, 1, 127)
        add.append(f)
        stats["filled_notes"] = len(f)
    if add_melody and melody is not None and len(melody):
        top_vel = float(np.percentile(apc[:, VEL], 75)) if len(apc) else 72.0
        miss = []
        for s, e, p, _v in melody:
            u = int(np.clip(np.searchsorted(units, s, side="right") - 1, 0, len(gap) - 1))
            if gap[u]:
                continue  # the gap fill already carries the teacher melody there
            on = apc[(apc[:, START] >= s - 0.08) & (apc[:, START] < max(e, s + 0.15))]
            d = np.abs(on[:, PITCH] - p) if len(on) else np.zeros(0)
            if not len(on) or not np.any((d <= 2) | (np.abs(d - 12) <= 1)):
                miss.append([s, e, p, top_vel])
        miss = np.array(miss, np.float64).reshape(-1, 4)
        add.append(miss)
        stats["melody_added"] = len(miss)
    notes = sanitize(np.concatenate([apc] + add))
    notes = limit_hands(notes, max_per_hand=5, span=12)
    ev = sanitize(np.concatenate([notes, pedal_part(song, notes)]))
    ev = ev[ev[:, START] < song.duration]
    stats["notes"] = int((ev[:, PITCH] > 0).sum())
    return ev, stats


def _voice(pcs, n, lo, hi, prev):
    """n chord tones in [lo, hi], closest to the previous voicing, no semitone neighbours."""
    from itertools import combinations, product

    opts = {pc: [p for p in range(lo, hi + 1) if p % 12 == pc] for pc in pcs}
    opts = {pc: o for pc, o in opts.items() if o}
    best, best_cost = [], np.inf
    for chosen in combinations(sorted(opts), min(n, len(opts))):
        for ps in product(*(opts[pc] for pc in chosen)):
            ps = sorted(ps)
            if ps[-1] - ps[0] > 12 or any(b - a < 2 for a, b in zip(ps, ps[1:])):
                continue
            if prev:
                cost = sum(min(abs(p - q) for q in prev) for p in ps)
            else:
                cost = sum(abs(p - (lo + hi) / 2) for p in ps)
            if cost < best_cost:
                best, best_cost = ps, cost
    return best


def arrange_hybrid_rhythm(song: Song, apc: np.ndarray, max_mid: int = 3):
    """D: AMT-APC supplies only WHEN the middle voices strike and how many notes; the pitches
    are the teacher's chord tones, voiced with smooth voice leading. Melody + bass = teacher."""
    stats = {"level": "hybrid_rhythm"}
    apc = apc[apc[:, PITCH] > 0]
    melody = lifted_melody(song, apc, stats=stats)
    bass = bass_part(song, LEVELS["normal"], {})
    acc = snap(apc, song.grid(4), 0.08)
    cl = np.round(acc[:, START] / 0.04)
    hits = []
    for k in np.unique(cl):
        c = acc[cl == k]
        t = float(c[:, START].min())
        lo, hi = _band(song, melody, bass, t)
        n = int(((c[:, PITCH] >= 48) & (c[:, PITCH] <= hi)).sum())  # AMT-APC's own middle notes
        if n:
            hits.append((t, min(n, max_mid), lo, hi))
    out, prev = [], []
    for j, (t, n, lo, hi) in enumerate(hits):
        nxt = hits[j + 1][0] if j + 1 < len(hits) else t + song.period
        end = min(nxt, t + 2 * song.period)
        ps = _voice(_pcs_at(song, t), n, lo, hi, prev)
        out += [[t, end, p, 0] for p in ps]
        prev = ps or prev
    stats["hits"] = len(hits)
    mid = np.array(out, np.float64).reshape(-1, 4)
    return tame(song, melody, bass, mid, stats)

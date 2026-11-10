"""Measure 'flow' problems of teacher arrangements (normal level).

  python scripts/diag_flow.py [/data/teacher_proto/analysis]

groove     : mean Jaccard similarity of the accompaniment's on-grid onset pattern between
             consecutive bars (1 = same rhythm every bar, like a pianist's pattern)
revoice%   : accompaniment hits whose pitch set differs from the previous hit although the
             chord did not change (restless voicing)
mel_short% : melody notes shorter than an 8th note
mel_gap    : median gap (s) between consecutive melody notes inside phrases
mel_jump%  : melody steps larger than a 6th (>9 semitones)
mel_offgrid: median distance (ms) of melody onsets from the 16th grid
bass_stac% : bass notes shorter than half of the time to the next bass note (choppy)
holes%     : time with no key sounding at all (pedal included), inside the song's active span
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pianocover.teacher.analysis import Analysis  # noqa: E402
from pianocover.teacher.arrange import (LEVELS, Song, accompaniment_part, bass_part, melody_part)  # noqa: E402


def groove(acc, song, subdiv=4):
    if len(acc) == 0:
        return np.nan
    grid_step = song.period / subdiv
    pats = []
    for s in song.bars:
        on = np.unique(acc[(acc[:, 0] >= s) & (acc[:, 0] < s + song.bar_len), 0])
        pats.append(set(np.round((on - s) / grid_step).astype(int)))
    sims = [len(a & b) / len(a | b) for a, b in zip(pats[:-1], pats[1:]) if a or b]
    return float(np.mean(sims)) if sims else np.nan


def revoice(acc, song):
    if len(acc) == 0:
        return np.nan
    hits = {}
    for s, e, p, v in acc:
        hits.setdefault(round(s, 3), set()).add(int(p))
    times = sorted(hits)
    n = ch = 0
    for t0, t1 in zip(times[:-1], times[1:]):
        c0, c1 = song.chord_at(t0 + 0.02), song.chord_at(t1 + 0.02)
        if c0 is not None and c1 is not None and c0[0] == c1[0] and c0[3] == c1[3]:
            n += 1
            ch += hits[t0] != hits[t1]
    return 100 * ch / max(n, 1)


def melody_stats(mel, song):
    if len(mel) < 2:
        return (np.nan,) * 4
    mel = mel[np.argsort(mel[:, 0])]
    dur = mel[:, 1] - mel[:, 0]
    short = 100 * np.mean(dur < song.period / 2)
    gaps = mel[1:, 0] - mel[:-1, 1]
    gap = float(np.median(gaps[gaps < 1.0])) if (gaps < 1.0).any() else np.nan
    jump = 100 * np.mean(np.abs(np.diff(mel[:, 2])) > 9)
    g = song.grid(4)
    k = np.clip(np.searchsorted(g, mel[:, 0]), 1, len(g) - 1)
    off = np.minimum(np.abs(mel[:, 0] - g[k - 1]), np.abs(g[k] - mel[:, 0]))
    return short, gap, jump, 1000 * float(np.median(off))


def bass_staccato(bass):
    if len(bass) < 2:
        return np.nan
    b = bass[np.argsort(bass[:, 0])]
    ioi = b[1:, 0] - b[:-1, 0]
    dur = b[:-1, 1] - b[:-1, 0]
    m = ioi < 2.0
    return 100 * float(np.mean(dur[m] < 0.5 * ioi[m])) if m.any() else np.nan


def holes(ev, a, hop=0.05):
    notes = ev[ev[:, 2] > 0]
    if len(notes) == 0:
        return 100.0
    t0, t1 = notes[:, 0].min(), notes[:, 1].max()
    t = np.arange(t0, t1, hop)
    on = np.zeros(len(t), bool)
    for s, e, p, v in ev:
        on |= (t >= s) & (t < e)
    return 100 * float(np.mean(~on))


d = sys.argv[1] if len(sys.argv) > 1 else "/data/teacher_proto/analysis"
print(f"{'song':<11} {'groove':>6} {'revoice%':>8} {'mel_short%':>10} {'mel_gap':>7} {'mel_jump%':>9} "
      f"{'offgrid_ms':>10} {'bass_stac%':>10} {'holes%':>6}")
for path in sorted(glob.glob(os.path.join(d, "*.npz"))):
    a = Analysis.load(path)
    song = Song(a)
    lv = LEVELS["normal"]
    st = {}
    mel = melody_part(song, lv, st)
    bass = bass_part(song, lv, st)
    acc = accompaniment_part(song, lv, mel, bass, np.zeros(len(song.bars), bool), st)
    from pianocover.teacher.arrange import arrange
    ev, _ = arrange(a, "normal", song)
    ms = melody_stats(mel, song)
    print(f"{os.path.basename(path)[:-4]:<11} {groove(acc, song):6.2f} {revoice(acc, song):8.1f} {ms[0]:10.1f} "
          f"{ms[1]:7.2f} {ms[2]:9.1f} {ms[3]:10.0f} {bass_staccato(bass):10.1f} {holes(ev, a):6.1f}")

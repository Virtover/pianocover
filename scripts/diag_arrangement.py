"""Diagnose audible wrong notes in teacher arrangements (regression check for arrange.py).

  python scripts/diag_arrangement.py /data/teacher/analysis

clash/s   : semitone (or major-7th) rubs between simultaneously SOUNDING notes, per second,
            with sustain pedal extending note ends (what you actually hear)
foreign % : output notes whose pitch class is not sounding anywhere in the song's transcribed
            parts (vocal, bass, other, piano) within +-0.12 s of the onset
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pianocover.teacher.analysis import Analysis  # noqa: E402
from pianocover.teacher.arrange import (LEVELS, Song, accompaniment_part, arrange, bass_part,  # noqa: E402
                                        melody_part, _notes)


def sounding_ends(ev):
    """Note ends extended by the sustain pedal (pitch -64 rows)."""
    notes = ev[ev[:, 2] > 0].copy()
    ped = ev[ev[:, 2] == -64]
    for i, (s, e, p, v) in enumerate(notes):
        for ps, pe in ped[:, :2]:
            if ps <= e < pe:
                notes[i, 1] = max(notes[i, 1], pe)
    return notes


def clashes_per_sec(notes, dur, hop=0.05):
    t = np.arange(0, dur, hop)
    n = 0
    for x in t:
        on = notes[(notes[:, 0] <= x) & (notes[:, 1] > x), 2].astype(int)
        if len(on) < 2:
            continue
        pcs = sorted(set(on % 12))
        n += any(((b - a) % 12) in (1, 11) for i, a in enumerate(pcs) for b in pcs[i + 1:])
    return n * hop / dur  # fraction of time with a semitone rub


def foreign(ev, a, tol=0.12):
    src = np.concatenate([_notes(a.vocal), _notes(a.bass), _notes(a.other), _notes(a.other_piano)])
    if len(ev) == 0:
        return np.nan
    bad = 0
    for s, e, p, v in ev:
        m = (src[:, 0] <= s + tol) & (src[:, 1] >= s - tol)
        bad += int(p) % 12 not in set(src[m, 2].astype(int) % 12)
    return 100 * bad / len(ev)


print(f"{'song':<11} {'level':<8} {'rub% ped':>8} {'rub% noped':>10} | foreign%: {'melody':>6} {'bass':>6} {'accomp':>6} | other-in-chord%")
for path in sorted(glob.glob(os.path.join(sys.argv[1] if len(sys.argv) > 1 else "/data/teacher_proto/analysis", "*.npz"))):
    a = Analysis.load(path)
    song = Song(a)
    name = os.path.basename(path)[:-4]
    st = {}
    lv = LEVELS["normal"]
    mel = melody_part(song, lv, st)
    bass = bass_part(song, lv, st)
    acc = accompaniment_part(song, lv, mel, bass, np.zeros(len(song.bars), bool), st)
    oth = _notes(a.other)
    inch = np.mean([(int(p) % 12) in c[3] for s, e, p, v in oth if (c := song.chord_at(s + 0.02)) is not None])
    for level in ("easy", "normal", "full"):
        ev, _ = arrange(a, level, song)
        with_ped = clashes_per_sec(sounding_ends(ev), a.duration)
        no_ped = clashes_per_sec(ev[ev[:, 2] > 0], a.duration)
        extra = (f"{foreign(mel, a):6.1f} {foreign(bass, a):6.1f} {foreign(acc, a):6.1f} | {100 * inch:5.1f}"
                 if level == "normal" else "")
        print(f"{name:<11} {level:<8} {100 * with_ped:8.1f} {100 * no_ped:10.1f} | {'':9} {extra}")

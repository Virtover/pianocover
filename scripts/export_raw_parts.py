"""Export what the teacher pipeline HEARD, as multi-track MIDI, to judge the inputs by ear.

  python scripts/export_raw_parts.py /data/teacher_proto /out/dir

One <id>_raw_parts.mid per song, on the song's timeline, with separate tracks you can mute:
  1 Melody   CREPE notes from the vocal stem (what the arranger uses as the tune)
  2 Bass     basic-pitch on the bass stem
  3 Other    basic-pitch on the 'other' stem (guitars, keys, synths)
  4 Other(TK) Transkun on the 'other' stem
  5 Chords   detected chords (after the chord check) as held block chords around C4
  6 Drums    detected kick (36) / snare (38) / hi-hat (42) onsets
If these already sound far from the song, the problem is upstream (separation /
transcription), not the arranger.
"""
import glob
import os
import sys

import numpy as np
import pretty_midi

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pianocover.teacher.analysis import Analysis  # noqa: E402
from pianocover.teacher.arrange import Song, _notes  # noqa: E402

root = sys.argv[1] if len(sys.argv) > 1 else "/data/teacher_proto"
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(root, "raw_parts")
os.makedirs(out, exist_ok=True)


def track(name, ev, program=0):
    inst = pretty_midi.Instrument(program=program, name=name)
    for s, e, p, v in _notes(ev):
        if e > s:
            inst.notes.append(pretty_midi.Note(int(np.clip(v, 1, 127)), int(p), float(s), float(e)))
    return inst


for path in sorted(glob.glob(os.path.join(root, "analysis", "fma_*.npz"))):
    if path.endswith("_mixpiano.npy"):
        continue
    sid = os.path.basename(path)[:-4]
    a = Analysis.load(path)
    song = Song(a)
    chords = []
    for s, e, root_pc, pcs, bass_pc in song.chords:
        if root_pc is None:
            continue
        for pc in pcs:
            chords.append([s, e, 60 + (pc - 60) % 12, 50])
    pm = pretty_midi.PrettyMIDI()
    pm.instruments += [
        track("1 Melody (CREPE)", a.vocal_f0 if len(a.vocal_f0) else a.vocal, 0),
        track("2 Bass (basic-pitch)", a.bass, 32),
        track("3 Other (basic-pitch)", a.other, 25),
        track("4 Other (Transkun)", a.other_piano, 0),
        track("5 Chords (detected)", np.array(chords).reshape(-1, 4), 48),
    ]
    drums = pretty_midi.Instrument(program=0, is_drum=True, name="6 Drums (kick/snare/hat)")
    for t, kind, strength in np.asarray(a.drum_onsets).reshape(-1, 3):
        note = {0: 36, 1: 38, 2: 42}.get(int(round(kind)), 38)
        drums.notes.append(pretty_midi.Note(int(np.clip(40 + 80 * strength, 1, 127)), note, float(t), float(t) + 0.1))
    pm.instruments.append(drums)
    pm.write(os.path.join(out, f"{sid}_raw_parts.mid"))
    print("wrote", sid)

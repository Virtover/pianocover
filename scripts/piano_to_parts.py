"""Piano pieces -> song-like parts, for review and (later) arranger training pairs.

  python scripts/piano_to_parts.py <dir with piano .mid> <out dir>

Writes <out>/analysis/<id>.npz (same format as the audio pipeline's analysis) and prints,
per piece, how the notes were split and how well the CURRENT rule arranger rebuilds the
original piano from those parts (note onset F1, 50 ms): the first objective score of the
arranger against human-written piano. Then run scripts/export_raw_parts.py on <out> to get
the multi-track parts MIDI for listening.
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import mir_eval  # noqa: E402

from pianocover.teacher.arrange import Song, arrange  # noqa: E402
from pianocover.teacher.from_piano import analysis_from_piano  # noqa: E402

src, out = sys.argv[1], sys.argv[2]
os.makedirs(os.path.join(out, "analysis"), exist_ok=True)


def f1(ref, est):
    ref, est = ref[ref[:, 2] > 0], est[est[:, 2] > 0]
    hz = mir_eval.util.midi_to_hz
    p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref[:, :2], hz(ref[:, 2]), est[:, :2], hz(est[:, 2]), onset_tolerance=0.05, offset_ratio=None)
    return p, r, f


print(f"{'piece':<22} {'notes':>5} {'mel%':>5} {'bass%':>5} {'other%':>6} {'chords':>6} | arranger vs original: "
      f"{'P':>5} {'R':>5} {'F1':>5} (normal)  {'F1':>5} (full)")
for path in sorted(glob.glob(os.path.join(src, "*.mid"))):
    sid = os.path.basename(path)[:-4]
    a, ev = analysis_from_piano(path)
    a.save(os.path.join(out, "analysis", f"{sid}.npz"))
    n = len(ev)
    song = Song(a)
    p, r, f = f1(ev, arrange(a, "normal", song)[0])
    f_full = f1(ev, arrange(a, "full", song)[0])[2]
    print(f"{sid:<22} {n:5d} {100 * len(a.vocal) / n:5.0f} {100 * len(a.bass) / n:5.0f} {100 * len(a.other) / n:6.0f} "
          f"{len(a.chord_labels):6d} | {'':22}{p:5.2f} {r:5.2f} {f:5.2f}           {f_full:5.2f}")

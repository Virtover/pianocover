"""How much of the sung melody a Transkun model's full output already contains, per test song.

Reference: the CREPE notes on the Demucs vocal stem saved by scripts/vocal_test.py
(<vocal dir>/NN_vocal_notes.npy). Same rule as there: a sung note is covered when the MIDI has a
note of the same pitch class overlapping >= 50 % of it ("same octave" = same key). Songs without
singing (the Toccata) are skipped. Compares any number of MIDI folders written by tk_songs.py.

  PYTHONPATH=/repo python scripts/vocal_coverage.py \
      --model ft=/data/tk_songs/20261125/midi:transkun-ft --model vocal1=/data/tk_vocal1_songs/midi:transkun-ft
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from scripts.vocal_test import coverage, read_events  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
NO_SINGING = {"6"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocal", default="/data/vocal_test/20261125")
    ap.add_argument("--model", action="append", required=True, help="name=midi_dir:label (label as in the file names)")
    ap.add_argument("--out", default=None, help="write the table as JSON here")
    a = ap.parse_args()
    songs = [s for s in csv.DictReader(open(REPO / "data" / "test_songs" / "songs.csv", encoding="utf-8"))
             if s["n"] not in NO_SINGING]
    res = {}
    for spec in a.model:
        name, rest = spec.split("=", 1)
        d, label = rest.rsplit(":", 1)
        res[name] = {}
        for s in songs:
            voc = np.load(Path(a.vocal) / f"{int(s['n']):02d}_vocal_notes.npy")
            f = Path(d) / f"{int(s['n']):02d} - {s['title']} - {label} full.mid".replace("/", "-")
            ev = read_events(f)
            cov, exact = coverage(voc, ev)
            res[name][s["title"]] = {"covered": round(float(cov.mean()), 3), "same_octave": round(float(exact.mean()), 3),
                                     "notes": int((ev[:, 2] > 0).sum())}
    names = list(res)
    print("song".ljust(40) + "".join(f"{n:>22}" for n in names))
    for s in songs:
        t = s["title"]
        print(t[:38].ljust(40) + "".join(f"{res[n][t]['covered']:>10.3f} ({res[n][t]['same_octave']:.3f})" for n in names))
    print("mean".ljust(40) + "".join(f"{np.mean([v['covered'] for v in res[n].values()]):>10.3f} "
                                     f"({np.mean([v['same_octave'] for v in res[n].values()]):.3f})" for n in names))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()

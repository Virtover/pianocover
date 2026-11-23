"""Piano replay set for Transkun (tier 2, "piano against forgetting"): real piano audio.

Real piano covers already in /data/ds_all (web covers, content- and rights-filtered there)
with their Transkun 2.0 transcriptions (cover_events/<piano_id>.npy, notes + pedals) as the
target. Training on them keeps 2.0's piano behaviour (self-distillation) while the model learns
other instruments. Renders are excluded (pdmx_organ "covers" are YDP piano renders of scores),
so only real recordings remain.

Output: <out>/piano/<piano_id>.wav (16-bit mono 44.1 kHz), <out>/piano/{train,val}.pickle in
transkun's DatasetMaestro format (audio_filename relative to <out>), <out>/piano/meta.csv.
The split follows ds_all (its val pieces stay val). Resumable.

  python scripts/piano_replay.py --ds /data/ds_all --out /data/piano_replay [--workers 6]
"""
import argparse
import csv
import os
import pickle
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
FS = 44100
REAL_ROOTS = {"/data/raw", "/data/organ_pilot", "/data/organ_partial"}  # pdmx_organ: renders


def find_audio(root, pid):
    for ext in ("webm", "m4a", "mp3", "wav", "opus", "flac"):
        p = Path(root) / "cover" / f"{pid}.{ext}"
        if p.exists():
            return p
    return None


def work(job):
    root, pid, out = job
    from scipy.io import wavfile

    from pianocover import events

    src = find_audio(root, pid)
    evp = Path(root) / "cover_events" / f"{pid}.npy"
    if src is None or not evp.exists():
        return None
    wav = Path(out) / "piano" / f"{pid}.wav"
    if not wav.exists():
        tmp = wav.with_suffix(".tmp.wav")
        subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(src), "-ac", "1", "-ar", str(FS),
                        "-c:a", "pcm_s16le", str(tmp)], check=True)
        os.replace(tmp, wav)
    sr, x = wavfile.read(wav, mmap=True)
    assert sr == FS and x.dtype == np.int16 and x.ndim == 1
    ev = events.load_npy(evp)
    dur = len(x) / FS
    ev = ev[(ev[:, 0] < dur) & (ev[:, 1] > ev[:, 0])]
    ev[:, 1] = np.minimum(ev[:, 1], dur)
    return {"id": pid, "root": root, "src": src.name, "duration": dur, "nSamples": len(x), "ev": ev}


def to_items(results, split):
    from transkun.Data import resolveOverlapping

    from pianocover import events

    items = []
    for r in results:
        ev = r["ev"]
        notes = events.to_transkun_notes(ev[ev[:, 2] > 0])
        pedals = events.to_transkun_notes(ev[ev[:, 2] < 0])
        allev = resolveOverlapping(notes) + pedals
        allev.sort(key=lambda n: (n.start, n.end, n.pitch))
        items.append({"audio_filename": f"piano/{r['id']}.wav", "duration": r["duration"], "fs": FS,
                      "nSamples": r["nSamples"], "nChannel": 1, "notes": allev, "source": "piano_replay",
                      "split": split})
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="/data/ds_all")
    ap.add_argument("--out", default="/data/piano_replay")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="first N items per split (for a test run)")
    ap.add_argument("--train-hours", type=float, default=0.0,
                    help="random subset of train covers of about this many hours (0 = all); val stays complete. "
                         "Durations come from the existing meta.csv, so the choice needs no decoding")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    known = {}
    meta_old = Path(args.out) / "piano" / "meta.csv"
    if meta_old.exists():
        with open(meta_old, encoding="utf8") as f:
            known = {r["id"]: float(r["duration"]) for r in csv.DictReader(f)}
    out = Path(args.out)
    (out / "piano").mkdir(parents=True, exist_ok=True)
    rows = []
    for split in ("train", "val"):
        blob = pickle.load(open(Path(args.ds) / f"{split}.pkl", "rb"))
        seen = set()
        its = []
        for it in blob["items"]:
            root = it.get("data_root", blob["data_root"])
            if root not in REAL_ROOTS or it["piano_id"] in seen:
                continue
            seen.add(it["piano_id"])
            its.append((root, it["piano_id"], str(out)))
        if args.limit:
            its = its[: args.limit]
        if split == "train" and args.train_hours > 0:
            import random

            random.Random(args.seed).shuffle(its)
            keep, h = [], 0.0
            for j in its:
                if h >= args.train_hours:
                    break
                keep.append(j)
                h += known.get(j[1], 240.0) / 3600  # unknown length: about a typical cover
            its = keep
        with Pool(args.workers) as pool:
            res = [r for r in pool.imap_unordered(work, its, chunksize=4) if r]
        res.sort(key=lambda r: r["id"])
        items = to_items(res, split)
        pickle.dump(items, open(out / "piano" / f"{split}.pickle", "wb"))
        rows += [[r["id"], split, r["root"], r["src"], round(r["duration"], 2), r["nSamples"],
                  int((r["ev"][:, 2] > 0).sum()), int((r["ev"][:, 2] < 0).sum())] for r in res]
        print(split, len(its), "listed,", len(res), "converted,", f"{sum(r['duration'] for r in res) / 3600:.1f} h",
              flush=True)
    with open(out / "piano" / "meta.csv", "w", newline="", encoding="utf8") as f:
        w = csv.writer(f)
        w.writerow(["id", "split", "data_root", "source_file", "duration", "nSamples", "n_notes", "n_pedals"])
        w.writerows(rows)


if __name__ == "__main__":
    main()

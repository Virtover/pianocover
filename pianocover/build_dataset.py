"""Build training / validation sets from downloaded pairs + transcribed covers.

For each (pop, cover) pair: align, transpose the cover to the original's key, warp the
cover notes onto the original's timeline, drop notes from passages that do not follow the
original, compute the style vector, and record unusable regions of the original.

Output (in --out):
  train.pkl, val.pkl     list of dicts, see `process_pair`
  style_stats.json       style normaliser fitted on the training split
  report.csv             per-pair alignment quality (use it to tune the thresholds)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import pickle
import traceback
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from . import events
from . import style as style_mod
from .align import SYNC_SR, align
from .audio import decode, find_audio

ARGS = None


def _load_sync_audio(path):
    return decode(path, SYNC_SR)


def process_pair(item):
    pop_id, piano_id = item
    a = ARGS
    data = Path(a.data)
    pop_wav = find_audio(data / "pop", pop_id)
    cover_wav = find_audio(data / "cover", piano_id)
    ev_path = data / "cover_events" / f"{piano_id}.npy"
    base = {"pop_id": pop_id, "piano_id": piano_id}
    try:
        if not (pop_wav and cover_wav and ev_path.exists()):
            return {**base, "status": "missing_files"}
        ev = events.load_npy(ev_path)
        n_notes = int((ev[:, events.PITCH] > 0).sum())
        if n_notes < a.min_notes:
            return {**base, "status": "too_few_notes", "n_notes": n_notes}

        pop_y = _load_sync_audio(pop_wav)
        cov_y = _load_sync_audio(cover_wav)
        al = align(pop_y, cov_y, sim_threshold=a.local_threshold)
        stats = {"quality": al.quality, "transpose": al.transpose, "key_margin": al.key_cost_margin}
        if al.quality < a.min_quality:
            return {**base, "status": "low_quality", **stats}
        if al.key_cost_margin < a.min_key_margin:
            return {**base, "status": "ambiguous_key", **stats}

        # drop notes played in passages that do not follow the original
        mid = 0.5 * (ev[:, events.START] + ev[:, events.END])
        ev = ev[al.cover_sim_at(mid) >= a.local_threshold]
        ev = events.transpose(ev, al.transpose)
        ev = events.warp(ev, al.cover_times, al.pop_times)
        ev = events.sanitize(ev)

        duration = len(pop_y) / SYNC_SR
        bad = al.bad_regions
        bad_sec = sum(e - s for s, e in bad)
        if bad_sec > a.max_bad_fraction * duration:
            return {**base, "status": "mostly_unaligned", "bad_fraction": bad_sec / duration, **stats}

        kept = int((ev[:, events.PITCH] > 0).sum())
        return {
            **base,
            "status": "ok",
            # pairs downloaded from the web carry no usable license; merge --licensed-only drops them
            "source": "web",
            "license": "unknown",
            "audio": str(Path("pop") / pop_wav.name),
            "duration": float(duration),
            "events": ev.astype(np.float32),
            "bad_regions": bad,
            "raw_style": style_mod.raw_style(ev),
            "n_notes": kept,
            "bad_fraction": bad_sec / duration,
            **stats,
        }
    except Exception as exc:  # keep going; report the failure
        return {**base, "status": "error", "error": f"{type(exc).__name__}: {exc}",
                "trace": traceback.format_exc(limit=3)}


def _is_val(pop_id: str, fraction: float) -> bool:
    h = int(hashlib.md5(pop_id.encode()).hexdigest()[:8], 16)
    return (h % 10000) < fraction * 10000


def _init(args):
    global ARGS
    ARGS = args


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="directory with pop/, cover/, cover_events/, pairs.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--val-fraction", type=float, default=0.03, help="split by original song")
    ap.add_argument("--min-quality", type=float, default=0.5, help="mean chroma similarity along the path")
    ap.add_argument("--local-threshold", type=float, default=0.35, help="local similarity to keep notes")
    ap.add_argument("--min-key-margin", type=float, default=0.02, help="reject unclear key detection")
    ap.add_argument("--max-bad-fraction", type=float, default=0.4)
    ap.add_argument("--min-notes", type=int, default=200)
    args = ap.parse_args(argv)

    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (data / "pairs.csv").open() as f:
        pairs = [(r["pop_id"], r["piano_id"]) for r in csv.DictReader(f)]
    print(f"processing {len(pairs)} pairs with {args.workers} workers", flush=True)

    results = []
    with Pool(args.workers, initializer=_init, initargs=(args,)) as pool:
        for i, r in enumerate(pool.imap_unordered(process_pair, pairs, chunksize=2), 1):
            results.append(r)
            if i % 50 == 0 or i == len(pairs):
                n_ok = sum(x["status"] == "ok" for x in results)
                print(f"[{i}/{len(pairs)}] ok={n_ok}", flush=True)

    with (out / "report.csv").open("w", newline="") as f:
        cols = ["pop_id", "piano_id", "status", "quality", "transpose", "key_margin",
                "bad_fraction", "n_notes", "error"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in sorted(results, key=lambda r: (r["pop_id"], r["piano_id"])):
            w.writerow(r)

    ok = [r for r in results if r["status"] == "ok"]
    train = [r for r in ok if not _is_val(r["pop_id"], args.val_fraction)]
    val = [r for r in ok if _is_val(r["pop_id"], args.val_fraction)]
    if not train:
        raise SystemExit("no usable training pairs; check report.csv")
    norm = style_mod.StyleNormalizer.fit(np.stack([r["raw_style"] for r in train]))
    norm.save(out / "style_stats.json")
    for r in ok:
        r["style"] = norm.normalize(r["raw_style"])
        r.pop("trace", None)
    for name, split in (("train", train), ("val", val)):
        with (out / f"{name}.pkl").open("wb") as f:
            pickle.dump({"data_root": str(data.resolve()), "items": split}, f)

    counts = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    hours = sum(r["duration"] for r in ok) / 3600
    print(f"status counts: {counts}")
    print(f"train={len(train)} val={len(val)} pairs, {hours:.1f} h of original audio")


if __name__ == "__main__":
    main()

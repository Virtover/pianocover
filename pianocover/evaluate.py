"""Evaluate a checkpoint on whole songs from the validation split.

Metrics per song (against the aligned human cover):
  onset_f1        pitch+onset match within 50 ms. Low by nature (many valid covers exist),
                  but useful for comparing checkpoints; AMT-APC reported F1 ~0.27.
  chroma_corr     correlation of 0.5 s pitch-class profiles with the human cover
                  (does it follow the harmony/melody, regardless of exact voicing?)
  density_ratio   generated notes / human cover notes (style control check)

  python -m pianocover.evaluate --val ds/val.pkl --ckpt runs/cover1/best.pt --out eval.csv
  python -m pianocover.evaluate --val ds/val.pkl --baseline --out eval_transkun.csv
"""

from __future__ import annotations

import argparse
import csv

import numpy as np
import torch

from . import events
from .audio import read_wav_slice
from .data import load_split
from .model import PianoCoverModel, load_pretrained_transkun
from .train import onset_f1


def chroma_profile(ev: np.ndarray, duration: float, win: float = 0.5) -> np.ndarray:
    n = int(np.ceil(duration / win)) + 1
    prof = np.zeros((n, 12))
    for s, e, p, v in ev[ev[:, 2] > 0]:
        a, b = int(s / win), int(np.ceil(e / win))
        prof[a:max(b, a + 1), int(p) % 12] += v / 127.0
    return prof


def chroma_corr(est, ref, duration):
    a, b = chroma_profile(est, duration), chroma_profile(ref, duration)
    active = (a.sum(1) > 0) & (b.sum(1) > 0)
    if active.sum() < 4:
        return float("nan")
    a, b = a[active], b[active]
    a = (a - a.mean(1, keepdims=True)) / (a.std(1, keepdims=True) + 1e-8)
    b = (b - b.mean(1, keepdims=True)) / (b.std(1, keepdims=True) + 1e-8)
    return float((a * b).mean())


def mask_bad(ev, bad):
    if not bad or len(ev) == 0:
        return ev
    mid = 0.5 * (ev[:, 0] + ev[:, 1])
    ok = np.ones(len(ev), bool)
    for s, e in bad:
        ok &= ~((mid >= s) & (mid < e))
    return ev[ok]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--val", required=True)
    ap.add_argument("--ckpt", help="pianocover checkpoint")
    ap.add_argument("--baseline", action="store_true", help="plain pretrained Transkun (no fine-tuning)")
    ap.add_argument("--style", choices=["own", "normal"], default="own",
                    help="own = each song's human-cover style; normal = average style")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if not args.baseline and not args.ckpt:
        ap.error("--ckpt or --baseline is required")

    items = load_split(args.val)
    seen, songs = set(), []
    for it in items:  # one cover per original song keeps the evaluation balanced
        if it["pop_id"] not in seen:
            seen.add(it["pop_id"])
            songs.append(it)
    if args.limit:
        songs = songs[: args.limit]

    torch.set_grad_enabled(False)
    if args.baseline:
        base = load_pretrained_transkun(device=args.device)
        run = lambda x, st: base.transcribe(x, discardSecondHalf=False)  # noqa: E731
    else:
        model, _ = PianoCoverModel.from_checkpoint(args.ckpt, map_location=args.device)
        model.eval()
        run = lambda x, st: model.generate(x, st)  # noqa: E731

    rows = []
    for i, it in enumerate(songs, 1):
        x = torch.from_numpy(read_wav_slice(it["audio_path"], 0.0, it["duration"])).to(args.device)
        st = torch.from_numpy(np.asarray(it["style"] if args.style == "own" else np.zeros_like(it["style"]),
                                         dtype=np.float32)).to(args.device)
        est = events.sanitize(events.from_transkun_notes(run(x, st)))
        ref = it["events"].astype(np.float64)
        est, ref = mask_bad(est, it["bad_regions"]), mask_bad(ref, it["bad_regions"])
        p, r, f = onset_f1(est, ref)
        n_ref = max(int((ref[:, 2] > 0).sum()), 1)
        row = {"pop_id": it["pop_id"], "piano_id": it["piano_id"], "onset_p": p, "onset_r": r, "onset_f1": f,
               "chroma_corr": chroma_corr(est, ref, it["duration"]),
               "density_ratio": int((est[:, 2] > 0).sum()) / n_ref}
        rows.append(row)
        print(f"[{i}/{len(songs)}] " + " ".join(f"{k}={v:.3f}" for k, v in row.items() if isinstance(v, float)),
              flush=True)

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for k in ("onset_f1", "chroma_corr", "density_ratio"):
        vals = np.array([r[k] for r in rows], dtype=float)
        print(f"{k}: mean {np.nanmean(vals):.3f}  median {np.nanmedian(vals):.3f}")


if __name__ == "__main__":
    main()

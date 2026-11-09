"""Fine-tune Transkun V2 into a style-conditioned piano cover model.

Single-GPU training. The defaults (batch 2 x accumulation 2) fit a 12 GB card such as the
RTX 5070; on 24 GB use --batch-size 4 --accum 1. Resumes automatically from <out>/last.pt.

Two-stage recipe for the license-clear plan, both stages on ONE merged dataset (so the
style statistics are shared):
  1. --source-weights teacher=1,silver=1,gold=1          (starts from pretrained Transkun)
  2. --init <stage1>/best.pt --source-weights gold=1,teacher=0.1,silver=0 --val-sources gold
     with lower learning rates; a little teacher data limits forgetting

Example:
  python -m pianocover.train --train ds/train.pkl --val ds/val.pkl \
      --style-stats ds/style_stats.json --out runs/cover1
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from . import FRAME_SEC, events
from . import style as style_mod
from .data import CoverChunks, collate, load_split, make_augmentator, random_key_shift, shift_events
from .sampling import chunk_weights, parse_source_weights, source_summary
from .model import PianoCoverModel


class AdaptiveClip:
    """Clip gradients at a running quantile of recent norms (as Transkun's trainer does)."""

    def __init__(self, quantile=0.8, init=40.0, maxlen=2000):
        self.q = quantile
        self.hist = collections.deque([init], maxlen=maxlen)

    def __call__(self, params):
        limit = float(np.quantile(np.array(self.hist), self.q))
        norm = torch.nn.utils.clip_grad_norm_(params, limit)
        if torch.isfinite(norm):
            self.hist.append(float(norm))
        return float(norm), limit


def lr_lambda(step, warmup, total, floor=0.1):
    if step < warmup:
        return (step + 1) / warmup
    p = min((step - warmup) / max(total - warmup, 1), 1.0)
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * p))


def batch_to_notes(batch, shifts, chunk_sec):
    notes_batch = []
    for ev, on, off, k in zip(batch["events"], batch["has_onset"], batch["has_offset"], shifts):
        ev, on, off = shift_events(ev, on, off, k, chunk_sec)
        notes_batch.append(events.to_transkun_notes(ev, on, off))
    return notes_batch


def onset_f1(est: np.ndarray, ref: np.ndarray, tol=0.05):
    """Note-onset F1 (pitch + onset within tol), mir_eval convention, offsets ignored."""
    import mir_eval

    est, ref = est[est[:, 2] > 0], ref[ref[:, 2] > 0]
    if len(ref) == 0 and len(est) == 0:
        return 1.0, 1.0, 1.0
    if len(ref) == 0 or len(est) == 0:
        return 0.0, 0.0, 0.0
    to_hz = mir_eval.util.midi_to_hz
    p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref[:, :2], to_hz(ref[:, 2]), est[:, :2], to_hz(est[:, 2]), onset_tolerance=tol, offset_ratio=None)
    return p, r, f


@torch.no_grad()
def validate(model, loader, device, n_decode, chunk_sec):
    model.eval()
    nll_sum, sec_sum = 0.0, 0.0
    f1s, dens = [], []
    decoded = 0
    for batch in loader:
        audio = batch["audio"].to(device)
        style = batch["style"].to(device)
        notes = batch_to_notes(batch, [0] * audio.shape[0], chunk_sec)
        logp = model.log_prob(audio, notes, style)
        nll_sum += float(-logp.sum())
        sec_sum += audio.shape[0] * audio.shape[1] / model.transkun.fs
        for b in range(audio.shape[0]):
            if decoded >= n_decode:
                break
            est = events.from_transkun_notes(
                model.generate(audio[b], style[b], step_sec=chunk_sec / 2, segment_sec=chunk_sec))
            ref = batch["events"][b]
            f1s.append(onset_f1(est, ref)[2])
            n_ref = max(int((ref[:, 2] > 0).sum()), 1)
            dens.append(int((est[:, 2] > 0).sum()) / n_ref)
            decoded += 1
    model.train()
    return {"val_nll_per_sec": nll_sum / max(sec_sum, 1e-6),
            "val_onset_f1": float(np.mean(f1s)) if f1s else float("nan"),
            "val_density_ratio": float(np.median(dens)) if dens else float("nan")}


def save(path, model, extra):
    payload = model.checkpoint_payload()
    payload.update(extra)
    tmp = Path(str(path) + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--style-stats", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--init", help="init weights: Transkun .pt (default: the pip package's V2) "
                                   "or a pianocover checkpoint to continue from")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--warmup", type=int, default=1000)
    # defaults fit a 12 GB card (RTX 5070); effective batch = batch-size x accum = 4
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--source-weights", help="sampling weights per data source, e.g. gold=4,teacher=1,silver=1 "
                                             "(0 excludes a source)")
    ap.add_argument("--val-sources", help="validate (and pick best.pt) only on these sources, e.g. gold")
    ap.add_argument("--chunk-sec", type=float, default=16.0)
    ap.add_argument("--hop-sec", type=float, default=8.0)
    ap.add_argument("--lr-backbone", type=float, default=5e-5)
    ap.add_argument("--lr-heads", type=float, default=2e-4)
    ap.add_argument("--lr-film", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--clip-quantile", type=float, default=0.8)
    ap.add_argument("--style-dropout", type=float, default=0.1)
    ap.add_argument("--key-shift", type=int, default=2, help="max random transposition (semitones)")
    ap.add_argument("--key-shift-prob", type=float, default=0.5)
    ap.add_argument("--augment", action="store_true", help="Transkun audio augmentation (audiomentations)")
    ap.add_argument("--noise-folder")
    ap.add_argument("--ir-folder")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--val-every", type=int, default=2000)
    ap.add_argument("--val-chunks", type=int, default=400)
    ap.add_argument("--val-decode", type=int, default=48, help="chunks decoded for onset F1")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device(args.device)

    style_stats = style_mod.StyleNormalizer.load(args.style_stats).to_dict()
    source_weights = parse_source_weights(args.source_weights)
    train_items, val_items = load_split(args.train), load_split(args.val)
    if args.val_sources:
        keep = {s.strip() for s in args.val_sources.split(",")}
        val_items = [it for it in val_items if it.get("source", "web") in keep]
        if not val_items:
            raise SystemExit(f"no validation items from sources {sorted(keep)}")
    if source_weights and not any(source_weights.get(it.get("source", "web"), 1.0) > 0 for it in train_items):
        raise SystemExit("--source-weights gives every training item weight 0")
    print(f"train pairs={len(train_items)} val pairs={len(val_items)}")

    last = out / "last.pt"
    if last.exists():
        model, ckpt = PianoCoverModel.from_checkpoint(last, use_amp=not args.no_amp)
        print(f"resuming from {last} at step {ckpt['step']}")
    elif args.init and "style_dim" in torch.load(args.init, map_location="cpu", weights_only=False):
        model, ckpt = PianoCoverModel.from_checkpoint(args.init, use_amp=not args.no_amp)
        ckpt = None
    else:
        model, ckpt = PianoCoverModel.from_pretrained_transkun(args.init, use_amp=not args.no_amp), None
    model.to(device).train()

    opt = torch.optim.AdamW(model.parameter_groups(args.lr_backbone, args.lr_heads, args.lr_film,
                                                   args.weight_decay), betas=(0.9, 0.98))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_lambda(s, args.warmup, args.steps))
    step, best = 0, float("inf")
    clipper = AdaptiveClip(args.clip_quantile)
    if ckpt is not None:
        opt.load_state_dict(ckpt["optimizer"])
        sched.load_state_dict(ckpt["scheduler"])
        step, best = ckpt["step"], ckpt.get("best_val", float("inf"))
        clipper.hist.extend(ckpt.get("clip_hist", []))

    augmentator = make_augmentator(args.noise_folder, args.ir_folder) if args.augment else None
    val_set = CoverChunks(val_items, args.chunk_sec, args.chunk_sec, seed=1234, dither=False)
    val_set.chunks = val_set.chunks[: args.val_chunks]
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                                             collate_fn=collate, num_workers=min(args.workers, 2))

    log_path = out / "log.csv"
    new_log = not log_path.exists()
    log_f = log_path.open("a", newline="")
    log = csv.writer(log_f)
    if new_log:
        log.writerow(["step", "loss_per_sec", "grad_norm", "clip", "lr_backbone", "sec_per_step",
                      "val_nll_per_sec", "val_onset_f1", "val_density_ratio"])

    epoch = step  # only used to vary the chunk dithering seed; resuming gives a fresh order
    t_last = time.time()
    running = collections.deque(maxlen=50)
    while step < args.steps:
        train_set = CoverChunks(train_items, args.chunk_sec, args.hop_sec, seed=args.seed * 1000 + epoch,
                                style_dropout=args.style_dropout, augmentator=augmentator)
        weights = chunk_weights(train_set, source_weights)
        sampler = (torch.utils.data.WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
                   if weights is not None else None)
        loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=sampler is None,
                                             sampler=sampler, collate_fn=collate, num_workers=args.workers,
                                             drop_last=True, persistent_workers=False,
                                             prefetch_factor=4 if args.workers else None)
        print(f"epoch {epoch}: {len(train_set)} chunks, sources {source_summary(train_set, source_weights)}",
              flush=True)
        it = iter(loader)
        while step < args.steps:
            opt.zero_grad(set_to_none=True)
            loss_acc, done = 0.0, True
            for _ in range(args.accum):
                try:
                    batch = next(it)
                except StopIteration:
                    done = False
                    break
                audio = batch["audio"].to(device, non_blocking=True)
                audio, shifts = random_key_shift(audio, args.key_shift, args.key_shift_prob)
                notes = batch_to_notes(batch, shifts, args.chunk_sec)
                style = batch["style"].to(device)
                logp = model.log_prob(audio, notes, style)
                secs = audio.shape[0] * audio.shape[1] / model.transkun.fs
                loss = -logp.sum() / secs
                (loss / args.accum).backward()
                loss_acc += float(loss) / args.accum
            if not done:
                break  # epoch finished; partial accumulation discarded
            if not math.isfinite(loss_acc):
                print(f"step {step}: non-finite loss, skipping batch", flush=True)
                continue
            gnorm, limit = clipper(model.parameters())
            opt.step()
            sched.step()
            step += 1
            running.append(loss_acc)

            if step % 50 == 0:
                dt = (time.time() - t_last) / 50
                t_last = time.time()
                lr0 = opt.param_groups[0]["lr"]
                print(f"step {step} loss/s {np.mean(running):.3f} gnorm {gnorm:.1f} clip {limit:.1f} "
                      f"lr {lr0:.2e} {dt:.2f}s/step", flush=True)
                log.writerow([step, f"{np.mean(running):.4f}", f"{gnorm:.2f}", f"{limit:.2f}", f"{lr0:.3e}",
                              f"{dt:.3f}", "", "", ""])
                log_f.flush()

            if step % args.val_every == 0 or step == args.steps:
                v = validate(model, val_loader, device, args.val_decode, args.chunk_sec)
                print(f"[val] step {step} {v}", flush=True)
                log.writerow([step, "", "", "", "", "", f"{v['val_nll_per_sec']:.4f}",
                              f"{v['val_onset_f1']:.4f}", f"{v['val_density_ratio']:.3f}"])
                log_f.flush()
                extra = {"style_stats": style_stats, "step": step, "frame_sec": FRAME_SEC, "val": v}
                if v["val_nll_per_sec"] < best:
                    best = v["val_nll_per_sec"]
                    save(out / "best.pt", model, extra)
                    print(f"[val] new best {best:.4f} -> {out / 'best.pt'}", flush=True)
                save(last, model, {**extra, "best_val": best, "optimizer": opt.state_dict(),
                                   "scheduler": sched.state_dict(), "clip_hist": list(clipper.hist)})
        epoch += 1
    log_f.close()
    print("training finished")


if __name__ == "__main__":
    main()

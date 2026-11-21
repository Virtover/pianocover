"""Phase 2 capacity test (docs/TRANSKUN_PLAN.md): can Transkun V2 fit any-instrument data?

Fine-tunes the plain pretrained Transkun 2.0.pt (no FiLM style input) on a small subset
(default 2.5 h) of the synthetic renders in /data/synth_tk2, and measures note-onset F1 on
fixed chunks of that same training subset. Pass: training-set onset F1 >= 0.95
(the model has enough capacity). Fail -> larger config or another architecture, before
collecting more data. Onset F1 on a few val chunks is printed too, but only as a hint.

The subset is balanced over the families (piano, organ, choir, ...): from each family a
random segment of --seg-sec seconds is taken from as many pieces as fit its share, so the
test sees many pieces and instruments rather than a few long ones.

Resumable (hibernation, crash, `docker restart`): last.pt is written atomically every
--save-every-min minutes and at every evaluation; running the same command again continues
from it (step, optimizer, schedule, used training minutes). The run ends when the train F1
reaches --target-f1 at an evaluation, or when --max-minutes of training time are used, or at
--steps. Results: <out>/log.csv, <out>/result.json (pass/fail, per-family F1, peak VRAM).

VRAM: --max-vram-gb caps this process (torch memory fraction); by default it is whatever
is free at start minus a 1 GB reserve for the desktop, so the computer does not stutter. Transkun checkpoints its
encoder layers in training mode and the backbone runs in bf16 (PianoCoverModel wrapper).

Example (in pianocover:dev, render paused first):
  python scripts/tk_capacity.py --root /data/synth_tk2 --out /data/runs/tk_capacity
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path
import pickle

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pianocover import events  # noqa: E402
from pianocover.audio import read_wav_slice  # noqa: E402
from pianocover.model import PianoCoverModel  # noqa: E402
from pianocover.train import AdaptiveClip, lr_lambda, onset_f1, save  # noqa: E402

FAMILIES = ["piano", "organ", "choir", "guitar", "band", "chamber", "orchestra"]


def note_array(notes) -> np.ndarray:
    """transkun Notes -> (N, 4) float array start, end, pitch, velocity (piano range only)."""
    a = np.array([(n.start, n.end, n.pitch, n.velocity) for n in notes], dtype=np.float64).reshape(-1, 4)
    return a[(a[:, 2] >= 21) & (a[:, 2] <= 108) & (a[:, 1] > a[:, 0])]


def load_family(root: Path, fam: str, split: str):
    path = root / fam / f"{split}.pickle"
    if not path.exists():
        return []
    with open(path, "rb") as f:
        rows = pickle.load(f)
    return [{"fam": fam, "audio": str(root / r["audio_filename"]), "duration": float(r["duration"]),
             "notes": note_array(r["notes"])} for r in rows]


def pick_segments(pieces, seconds, seg_sec, rng):
    """Random seg_sec windows from distinct pieces until `seconds` are covered."""
    pieces = list(pieces)
    rng.shuffle(pieces)
    out, total = [], 0.0
    for p in pieces:
        if total >= seconds:
            break
        notes = p["notes"]
        if len(notes) < 8:
            continue
        length = min(seg_sec, p["duration"])
        begin = rng.uniform(0.0, max(p["duration"] - length, 0.0))
        inside = notes[(notes[:, 0] < begin + length) & (notes[:, 1] > begin)]
        if len(inside) < 8:
            continue
        out.append({**p, "begin": begin, "end": begin + length})
        total += length
    return out, total


def slice_notes(notes: np.ndarray, begin: float, end: float):
    """Notes overlapping [begin, end) relative to begin, with Transkun's onset/offset flags."""
    from transkun.Data import Note

    sel = notes[(notes[:, 0] < end) & (notes[:, 1] > begin)]
    return [Note(max(s, begin) - begin, min(e, end) - begin, int(p), int(v), bool(s >= begin), bool(e < end))
            for s, e, p, v in sel]


class Chunks(torch.utils.data.Dataset):
    """Random chunk_sec windows inside the chosen segments (new random position per call)."""

    def __init__(self, segs, chunk_sec, n, seed):
        self.segs, self.chunk_sec, self.n, self.seed = segs, chunk_sec, n, seed
        self.w = np.array([s["end"] - s["begin"] for s in segs])
        self.w = self.w / self.w.sum()

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = np.random.default_rng((self.seed, i))
        s = self.segs[rng.choice(len(self.segs), p=self.w)]
        span = s["end"] - s["begin"]
        b = s["begin"] + rng.uniform(0.0, max(span - self.chunk_sec, 0.0))
        e = b + self.chunk_sec
        return read_wav_slice(s["audio"], b, e), slice_notes(s["notes"], b, e)


def collate(batch):
    n = min(a.shape[0] for a, _ in batch)
    return torch.from_numpy(np.stack([a[:n, :1] for a, _ in batch])), [nt for _, nt in batch]


def fixed_chunks(segs, chunk_sec, n, seed):
    """Deterministic evaluation chunks, spread evenly over the families."""
    rng = random.Random(seed)
    by_fam = {}
    for s in segs:
        by_fam.setdefault(s["fam"], []).append(s)
    out, fams = [], sorted(by_fam)
    k = 0
    while len(out) < n and fams:
        fam = fams[k % len(fams)]
        s = rng.choice(by_fam[fam])
        b = s["begin"] + rng.uniform(0.0, max(s["end"] - s["begin"] - chunk_sec, 0.0))
        out.append({"fam": fam, "audio": s["audio"], "begin": b, "end": b + chunk_sec, "notes": s["notes"]})
        k += 1
    return out


@torch.no_grad()
def evaluate(model, chunks, device, chunk_sec):
    model.eval()
    per_fam, dens = {}, []
    for c in chunks:
        audio = torch.from_numpy(read_wav_slice(c["audio"], c["begin"], c["end"])[:, :1]).to(device)
        est = events.from_transkun_notes(model.generate(audio, None, step_sec=chunk_sec / 2, segment_sec=chunk_sec))
        sel = c["notes"]
        sel = sel[(sel[:, 0] >= c["begin"]) & (sel[:, 0] < c["end"])]
        ref = np.column_stack([sel[:, 0] - c["begin"], sel[:, 1] - c["begin"], sel[:, 2], sel[:, 3]])
        ref[:, 1] = np.minimum(ref[:, 1], chunk_sec)
        # notes sounding from before the chunk start have no onset to find: ignore estimates there
        est = est[est[:, 0] >= 0.05] if len(est) else est
        per_fam.setdefault(c["fam"], []).append(onset_f1(est, ref)[2])
        dens.append(int((est[:, 2] > 0).sum()) / max(len(ref), 1))
    model.train()
    fam_f1 = {f: float(np.mean(v)) for f, v in sorted(per_fam.items())}
    allv = [x for v in per_fam.values() for x in v]
    return float(np.mean(allv)), fam_f1, float(np.median(dens))


def probe_batch(model, segs, chunk_sec, device, cap, sizes=(8, 6, 4, 3, 2, 1), margin=0.9):
    """Largest batch whose forward+backward peak stays under margin x the VRAM cap.
    The densest segments are used, since the semi-CRF cost grows with the number of notes."""
    def density(s):
        n = s["notes"]
        return int(((n[:, 0] < s["begin"] + chunk_sec) & (n[:, 1] > s["begin"])).sum())

    dense = sorted(segs, key=density, reverse=True)[:8]
    for bs in sizes:
        batch = [(read_wav_slice(s["audio"], s["begin"], s["begin"] + chunk_sec),
                  slice_notes(s["notes"], s["begin"], s["begin"] + chunk_sec)) for s in (dense * 8)[:bs]]
        audio, notes = collate(batch)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            model.zero_grad(set_to_none=True)
            loss = -model.log_prob(audio.to(device), notes, None).sum()
            loss.backward()
            peak = torch.cuda.max_memory_allocated()
            ok = peak <= margin * cap
            print(f"[probe] batch {bs}: peak {peak / 2**30:.2f} GB{'' if ok else ' (over margin)'}", flush=True)
        except torch.cuda.OutOfMemoryError:
            print(f"[probe] batch {bs}: out of memory", flush=True)
            ok = False
        finally:
            model.zero_grad(set_to_none=True)
            loss = None
            torch.cuda.empty_cache()
        if ok:
            torch.cuda.reset_peak_memory_stats()
            return bs
    raise SystemExit("even batch 1 does not fit the VRAM cap")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="/data/synth_tk2")
    ap.add_argument("--out", default="/data/runs/tk_capacity")
    ap.add_argument("--families", default=",".join(FAMILIES))
    ap.add_argument("--hours", type=float, default=2.5, help="size of the training subset")
    ap.add_argument("--seg-sec", type=float, default=60.0, help="one random segment this long per piece")
    ap.add_argument("--chunk-sec", type=float, default=16.0)
    ap.add_argument("--batch-size", default="auto",
                    help="chunks per forward pass, or 'auto': the largest of 8,6,4,3,2,1 that fits --max-vram-gb")
    ap.add_argument("--eff-batch", type=int, default=8, help="chunks per optimizer step (accumulation fills up)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--steps", type=int, default=20000, help="upper bound; time budget usually ends first")
    ap.add_argument("--max-minutes", type=float, default=120.0, help="training time budget (resumes add up)")
    ap.add_argument("--target-f1", type=float, default=0.95)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-chunks", type=int, default=42, help="fixed training chunks decoded for onset F1")
    ap.add_argument("--val-chunks", type=int, default=14)
    ap.add_argument("--max-vram-gb", default="auto",
                    help="cap for this process; 'auto' = VRAM free at start minus --vram-reserve-gb")
    ap.add_argument("--vram-reserve-gb", type=float, default=1.0,
                    help="left free for the desktop and other programs (auto cap only)")
    ap.add_argument("--min-vram-gb", type=float, default=5.0,
                    help="the auto cap is never below this (owner's choice); fewer GB free -> refuse to start")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--save-every-min", type=float, default=10.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info()  # after CUDA init: context already counted as used
        if args.max_vram_gb == "auto":
            args.max_vram_gb = max(args.min_vram_gb, round((free / 2**30) - args.vram_reserve_gb, 2))
        args.max_vram_gb = float(args.max_vram_gb)
        if free < args.min_vram_gb * 2**30:
            raise SystemExit(f"only {free / 2**30:.1f} GB VRAM free, need {args.min_vram_gb} GB: "
                             "is the vocal render paused?")
        torch.cuda.set_per_process_memory_fraction(min(args.max_vram_gb * 2**30 / total, 1.0))
        print(f"VRAM: {free / 2**30:.2f} of {total / 2**30:.2f} GB free at start, cap {args.max_vram_gb:.2f} GB "
              f"(reserve {args.vram_reserve_gb} GB)", flush=True)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # ---- subset (saved, so a resumed run trains on exactly the same data)
    subset_path = out / "subset.pkl"
    if subset_path.exists():
        with open(subset_path, "rb") as f:
            segs, val_segs = pickle.load(f)
    else:
        rng = random.Random(args.seed)
        fams = [f.strip() for f in args.families.split(",") if f.strip()]
        share = args.hours * 3600 / len(fams)
        segs, val_segs = [], []
        for fam in fams:
            s, got = pick_segments(load_family(Path(args.root), fam, "train"), share, args.seg_sec, rng)
            v, _ = pick_segments(load_family(Path(args.root), fam, "val"), 4 * args.chunk_sec, args.chunk_sec, rng)
            print(f"{fam}: {len(s)} segments, {got / 60:.1f} min; val {len(v)}", flush=True)
            segs += s
            val_segs += v
        with open(subset_path, "wb") as f:
            pickle.dump((segs, val_segs), f)
    hours = sum(s["end"] - s["begin"] for s in segs) / 3600
    print(f"subset: {len(segs)} segments, {hours:.2f} h", flush=True)
    eval_set = fixed_chunks(segs, args.chunk_sec, args.eval_chunks, args.seed + 1)
    val_set = fixed_chunks(val_segs, args.chunk_sec, args.val_chunks, args.seed + 2)

    # ---- model / optimizer, resume
    model = PianoCoverModel.from_pretrained_transkun().to(device).train()
    params = [p for n, p in model.named_parameters() if not n.startswith("film.")]
    for p in model.film.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.98), weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_lambda(s, args.warmup, args.steps))
    clipper = AdaptiveClip(0.8)
    step, used_min, history = 0, 0.0, []
    last = out / "last.pt"
    for path in (last, last.with_suffix(".prev.pt")):
        if not path.exists():
            continue
        try:
            ck = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:
            print(f"[resume] cannot read {path}: {exc}", flush=True)
            continue
        model.load_state_dict(ck["state_dict"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        step, used_min, history = ck["step"], ck["used_min"], ck["history"]
        clipper.hist.clear()
        clipper.hist.extend(ck["clip_hist"] or [40.0])
        print(f"[resume] {path} at step {step}, {used_min:.1f} training min used", flush=True)
        break

    def checkpoint(reason):
        save(last, model, {"step": step, "used_min": used_min, "history": history,
                           "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                           "clip_hist": list(clipper.hist)}, keep_previous=True)
        print(f"[ckpt] step {step} ({reason})", flush=True)

    log_path = out / "log.csv"
    new_log = not log_path.exists()
    log_f = log_path.open("a", newline="")
    log = csv.writer(log_f)
    if new_log:
        log.writerow(["step", "train_min", "loss_per_sec", "train_f1", "val_f1", "density", "peak_vram_gb"])

    def run_eval():
        t = time.time()
        f1, fam_f1, dens = evaluate(model, eval_set, device, args.chunk_sec)
        vf1, vfam, _ = evaluate(model, val_set, device, args.chunk_sec)
        peak = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0
        history.append({"step": step, "train_min": round(used_min, 1), "train_f1": f1, "per_family": fam_f1,
                        "val_f1": vf1, "val_per_family": vfam, "density": dens, "peak_vram_gb": round(peak, 2)})
        log.writerow([step, f"{used_min:.1f}", "", f"{f1:.4f}", f"{vf1:.4f}", f"{dens:.3f}", f"{peak:.2f}"])
        log_f.flush()
        print(f"[eval] step {step} ({used_min:.0f} min): train onset F1 {f1:.3f} {fam_f1} | val {vf1:.3f} "
              f"| density {dens:.2f} | peak VRAM {peak:.2f} GB | {time.time() - t:.0f}s", flush=True)
        return f1

    if not history:
        run_eval()  # step-0 baseline = pretrained piano model on these instruments

    if args.batch_size == "auto":
        args.batch_size = probe_batch(model, segs, args.chunk_sec, device, args.max_vram_gb * 2**30) if device.type == "cuda" else 1
    args.batch_size = int(args.batch_size)
    args.accum = max(1, math.ceil(args.eff_batch / args.batch_size))
    print(f"batch {args.batch_size} x accum {args.accum} (effective {args.batch_size * args.accum})", flush=True)

    loader = torch.utils.data.DataLoader(
        Chunks(segs, args.chunk_sec, 10**9, args.seed * 7919 + step), batch_size=args.batch_size,
        collate_fn=collate, num_workers=args.workers, prefetch_factor=4 if args.workers else None)
    it = iter(loader)
    running, t_saved, reason = [], time.time(), "steps"
    while step < args.steps:
        if used_min >= args.max_minutes:
            reason = "time budget"
            break
        t0 = time.time()
        opt.zero_grad(set_to_none=True)
        loss_acc = 0.0
        for _ in range(args.accum):
            audio, notes = next(it)
            audio = audio.to(device, non_blocking=True)
            logp = model.log_prob(audio, notes, None)
            loss = -logp.sum() / (audio.shape[0] * audio.shape[1] / model.transkun.fs)
            (loss / args.accum).backward()
            loss_acc += float(loss.detach()) / args.accum
        if not math.isfinite(loss_acc):
            print(f"step {step}: non-finite loss, skipped", flush=True)
            continue
        clipper(params)
        opt.step()
        sched.step()
        step += 1
        used_min += (time.time() - t0) / 60
        running = (running + [loss_acc])[-50:]
        if step % 50 == 0:
            peak = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0
            print(f"step {step} loss/s {np.mean(running):.3f} {used_min:.1f} min "
                  f"{(time.time() - t0):.2f}s/step peak VRAM {peak:.2f} GB", flush=True)
            log.writerow([step, f"{used_min:.1f}", f"{np.mean(running):.4f}", "", "", "", f"{peak:.2f}"])
            log_f.flush()
        if step % args.eval_every == 0:
            f1 = run_eval()
            checkpoint("eval")
            t_saved = time.time()
            if f1 >= args.target_f1:
                reason = "target reached"
                break
        elif time.time() - t_saved > args.save_every_min * 60:
            checkpoint("periodic")
            t_saved = time.time()

    if not history or history[-1]["step"] != step:
        run_eval()
    checkpoint("end")
    best = max(history, key=lambda h: h["train_f1"])
    result = {"pass": best["train_f1"] >= args.target_f1, "target_f1": args.target_f1, "stopped_by": reason,
              "subset_hours": round(hours, 2), "steps": step, "train_min": round(used_min, 1),
              "baseline": history[0], "best": best, "final": history[-1], "args": vars(args)}
    tmp = out / "result.json.tmp"
    tmp.write_text(json.dumps(result, indent=2))
    os.replace(tmp, out / "result.json")
    log_f.close()
    print(f"RESULT {'PASS' if result['pass'] else 'FAIL'}: best train onset F1 {best['train_f1']:.3f} "
          f"at step {best['step']} (baseline {history[0]['train_f1']:.3f}), stopped by {reason}", flush=True)


if __name__ == "__main__":
    main()

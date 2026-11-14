"""Fine-tune AMT-APC on prepared cover pairs (see pianocover.apc.prepare).

  python -m pianocover.apc.train --ds /data/ds --init /data/ext/amt-apc/models/params/apc.pth \
      --out /data/runs/apc1 --steps 40000

Speed: bf16 autocast, SDPA attention (model.py), torch.compile, fused AdamW, TF32, a
sync-free loss, mmap'd float16 spectrograms, labels built in DataLoader workers, pinned
memory with non-blocking copies. `--bench 30` times a few steps and prints throughput and
peak memory, to pick --batch / --checkpointing for a GPU.

Resumable: rerun the same command; it continues from <out>/last.pt. Ctrl+C saves first.
Checkpoints <out>/best.pth and last.pth are plain state dicts in apc.pth layout.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import signal
import time
from pathlib import Path

import numpy as np
import torch

from . import model as apc_model
from .dataset import Windows, load_rows
from .loss import f1_from_counts, loss_fn


def lr_at(step, warmup, total, floor=0.1):
    if step < warmup:
        return (step + 1) / warmup
    p = min(1.0, (step - warmup) / max(1, total - warmup))
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * p))


def to_dev(batch, device):
    spec, sv, onset, offset, frame, vel = (t.to(device, non_blocking=True) for t in batch)
    return spec.float(), sv.float(), onset, frame, vel


@torch.no_grad()
def validate(net, loader, device, amp_dtype, max_batches):
    net.eval()
    tot, n, counts = 0.0, 0, torch.zeros(3, 3, device=device)
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        spec, sv, onset, frame, vel = to_dev(batch, device)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            out = net(spec, sv)
        loss, c = loss_fn(out, onset, frame, vel)
        tot += loss.item()
        n += 1
        counts += c
    net.train()
    return tot / max(n, 1), f1_from_counts(counts)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ds", required=True, help="dataset dir containing apc/ (from pianocover.apc.prepare)")
    ap.add_argument("--init", required=True, help="apc.pth (warm start) or amt.pth (style gate starts fresh)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=1, help="one window already fills the GPU (512 frames x 256 bins)")
    ap.add_argument("--accum", type=int, default=4, help="gradient accumulation; batch 1 x 4 = AMT-APC's batch 4")
    ap.add_argument("--steps", type=int, default=40000, help="optimizer steps")
    ap.add_argument("--lr", type=float, default=5e-5, help="AMT-APC used 1e-4 from amt.pth; lower for apc.pth")
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--checkpointing", action="store_true", help="activation checkpointing: less memory, ~30%% slower")
    ap.add_argument("--freeze-encoder", action="store_true", help="train decoder + style gate only (faster)")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 2))
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--val-batches", type=int, default=100)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bench", type=int, default=0, help="time N steps (after warm-up) and exit")
    ap.add_argument("--vram-fraction", type=float, default=0.92,
                    help="cap this process's GPU memory (an OOM error instead of spilling into system RAM, "
                         "which on Windows is ~50x slower and can crash WSL); lower it to keep headroom")
    args = ap.parse_args(argv)

    device = torch.device("cuda")
    torch.cuda.set_per_process_memory_fraction(args.vram_fraction)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[args.precision]
    random.seed(args.seed), np.random.seed(args.seed), torch.manual_seed(args.seed)

    apc_dir = Path(args.ds) / "apc"
    train_ds = Windows(load_rows(apc_dir / "train.pkl"), apc_dir / "spec", jitter=True, seed=args.seed)
    val_rows = load_rows(apc_dir / "val.pkl") if (apc_dir / "val.pkl").exists() else []
    val_ds = Windows(val_rows, apc_dir / "spec", jitter=False) if val_rows else None
    print(f"train: {len(train_ds.rows)} pairs, {len(train_ds)} windows; val: {len(val_rows)} pairs", flush=True)

    loader_kw = dict(batch_size=args.batch, num_workers=args.workers, pin_memory=True,
                     persistent_workers=args.workers > 0, prefetch_factor=4 if args.workers else None)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    net = apc_model.load(args.init, device, dropout=args.dropout)
    net.set_checkpointing(args.checkpointing)
    if args.freeze_encoder:
        for p in net.encoder.parameters():
            p.requires_grad_(False)
    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0, fused=True)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_at(s, args.warmup, args.steps))
    scaler = torch.amp.GradScaler(enabled=args.precision == "fp16")

    step, epoch, best = 0, 0, float("inf")
    if (out / "last.pt").exists() and not args.bench:
        ck = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
        net.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        step, epoch, best = ck["step"], ck["epoch"], ck["best"]
        print(f"resumed at step {step}", flush=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=1))
    fwd = net if args.no_compile else torch.compile(net)

    def save_state(path):
        tmp = path.with_suffix(".tmp")
        torch.save({"model": net.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "step": step, "epoch": epoch, "best": best}, tmp)
        tmp.replace(path)

    stop = {"flag": False}
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.update(flag=True))

    log = (out / "log.csv").open("a")
    counts = torch.zeros(3, 3, device=device)
    run_loss = torch.zeros((), device=device)
    t_last, n_since = time.time(), 0
    bench_t0 = None
    net.train()
    while step < args.steps and not stop["flag"]:
        train_ds.reshuffle(args.seed + epoch)
        loader = torch.utils.data.DataLoader(train_ds, shuffle=True, drop_last=True, **loader_kw)
        for i, batch in enumerate(loader):
            spec, sv, onset, frame, vel = to_dev(batch, device)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                out_ = fwd(spec, sv)
            loss, c = loss_fn(out_, onset, frame, vel)
            scaler.scale(loss / args.accum).backward()
            run_loss += loss.detach()
            counts += c
            n_since += 1
            if (i + 1) % args.accum:
                continue
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, args.clip)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            step += 1

            if args.bench:
                if step == 5:
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    bench_t0 = time.time()
                elif step == 5 + args.bench:
                    torch.cuda.synchronize()
                    dt = (time.time() - bench_t0) / args.bench
                    win_s = args.batch * args.accum / dt
                    print(f"BENCH batch={args.batch} accum={args.accum} precision={args.precision} "
                          f"compile={not args.no_compile} ckpt={args.checkpointing} freeze={args.freeze_encoder}: "
                          f"{dt * 1000:.0f} ms/step, {win_s:.2f} windows/s "
                          f"({win_s * 8.192:.0f} s of audio per s), "
                          f"peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)
                    return
                continue

            if step % args.log_every == 0:
                f1 = f1_from_counts(counts)
                dt = time.time() - t_last
                msg = (f"step {step} epoch {epoch} loss {run_loss.item() / n_since:.4f} "
                       f"F1 on/fr/vel {f1[0]:.3f}/{f1[1]:.3f}/{f1[2]:.3f} lr {sched.get_last_lr()[0]:.2e} "
                       f"{n_since * args.batch / dt:.1f} win/s")
                print(msg, flush=True)
                log.write(f"{step},train,{run_loss.item() / n_since:.5f},{f1[0]:.4f},{f1[1]:.4f},{f1[2]:.4f}\n")
                log.flush()
                counts.zero_(), run_loss.zero_()
                t_last, n_since = time.time(), 0
            if val_ds is not None and step % args.val_every == 0:
                vl, vf1 = validate(fwd, torch.utils.data.DataLoader(val_ds, shuffle=False, **loader_kw),
                                   device, amp_dtype, args.val_batches)
                print(f"VAL step {step} loss {vl:.4f} F1 on/fr/vel {vf1[0]:.3f}/{vf1[1]:.3f}/{vf1[2]:.3f}", flush=True)
                log.write(f"{step},val,{vl:.5f},{vf1[0]:.4f},{vf1[1]:.4f},{vf1[2]:.4f}\n")
                log.flush()
                if vl < best:
                    best = vl
                    apc_model.save(net, out / "best.pth")
                save_state(out / "last.pt")
                apc_model.save(net, out / "last.pth")
            if step >= args.steps or stop["flag"]:
                break
        epoch += 1
    if not args.bench:
        save_state(out / "last.pt")
        apc_model.save(net, out / "last.pth")
        print(f"stopped at step {step}", flush=True)


if __name__ == "__main__":
    main()

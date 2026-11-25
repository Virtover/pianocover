"""Phase 5 (docs/TRANSKUN_PLAN.md): fine-tune Transkun V2 on the any-instrument mix.

Starts from the plain pretrained Transkun 2.0.pt (no FiLM style input; same model path as
scripts/tk_capacity.py: bf16 backbone, Transkun's own encoder checkpointing). Every training
chunk is drawn fresh: a group is picked by its weight (--group), then an item inside the group
with probability proportional to its duration, then a random --chunk-sec window in it. So a
group's share of training is set by its weight, and inside a group every hour counts the same.

Default mix (owner 2026-11-24: tier 1 not strongly oversampled; its natural share is ~18%):
  tier1 0.20  real recordings (musicnet, guitarset, dagstuhl, esmuc, choral, vocadito;
              cantoria left out, its USB copy has the old broken labels)
  piano 0.25  piano replay (real piano covers, Transkun 2.0 labels incl. pedals)
  synth 0.55  synth_tk2 + synth_vocal + voice_data

Pedals: only sources whose labels contain pedal intervals (pitch -64 / -67) train the two
pedal outputs; for all others the pedal terms of the loss are masked, so the model is not
taught "never any pedal" on recordings whose pedal is simply unlabelled.

Validation every --val-every steps on fixed chunks of every source's val split: onset F1 per
source (and per group); best.pt = highest mean of the group F1s. Step 0 is evaluated first
(baseline = plain 2.0.pt). best_transkun.pt + transkun.conf load like the stock checkpoint
(`transkun --weight best_transkun.pt --conf transkun.conf`).

Stop / resume: Ctrl+C or SIGTERM (`docker stop -t 60`) finishes the step, saves last.pt and
exits; running the same command again continues at the exact step and data position.
last.pt is also written atomically every --save-every-min minutes.

VRAM: capped at (free VRAM at start - --vram-reserve-gb), so the desktop keeps some memory.

Example (pianocover image, repo at /w, data volume at /data):
  python scripts/tk_train.py --out /data/runs/tk_train
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import pickle
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pianocover import events  # noqa: E402
from pianocover.audio import read_wav_slice  # noqa: E402
from pianocover.model import PianoCoverModel, pretrained_paths  # noqa: E402
from pianocover.train import AdaptiveClip, StopRequest, ignore_stop_signals, lr_lambda, onset_f1, save  # noqa: E402

DEFAULT_GROUPS = [
    "tier1=0.20:tier1_data/musicnet,tier1_data/guitarset,tier1_data/dagstuhl,tier1_data/esmuc,"
    "tier1_data/choral,tier1_data/vocadito",
    "piano=0.25:piano_replay/piano",
    "synth=0.55:synth_tk2/*,synth_vocal/*,voice_data/*",
]
PEDALS = (-64, -67)  # Transkun's targetMIDIPitch starts with these two, so loss columns 0 and 1


# ------------------------------------------------------------------ data
def note_array(notes) -> np.ndarray:
    """transkun Notes -> (N, 4) start, end, pitch, velocity: piano keys 21-108 and the two pedals."""
    a = np.array([(n.start, n.end, n.pitch, n.velocity) for n in notes], dtype=np.float64).reshape(-1, 4)
    keep = (((a[:, 2] >= 21) & (a[:, 2] <= 108)) | np.isin(a[:, 2], PEDALS)) & (a[:, 1] > a[:, 0])
    return a[keep]


def resolve(pkl: Path, rel: str) -> str | None:
    for base in (pkl.parent, pkl.parent.parent, pkl.parent.parent.parent):
        if (base / rel).exists():
            return str(base / rel)
    return None


def load_source(data: Path, src: str, split: str):
    pkl = data / src / f"{split}.pickle"
    if not pkl.exists():
        return []
    out = []
    for r in pickle.load(open(pkl, "rb")):
        path = resolve(pkl, r["audio_filename"])
        if path is None:
            raise SystemExit(f"{pkl}: audio not found: {r['audio_filename']}")
        out.append({"src": src, "audio": path, "duration": float(r["duration"]), "notes": note_array(r["notes"])})
    return out


def parse_groups(specs, data: Path):
    """'name=weight:src,src_glob,...' -> [(name, weight, [sources])]."""
    groups = []
    for spec in specs:
        head, srcs = spec.split(":", 1)
        name, w = head.split("=")
        found = []
        for pat in srcs.split(","):
            hits = sorted(str(Path(p).relative_to(data)) for p in glob.glob(str(data / pat.strip()))
                          if (Path(p) / "train.pickle").exists())
            if not hits:
                raise SystemExit(f"group {name}: nothing matches {pat}")
            found += [h for h in hits if h not in found]
        groups.append((name.strip(), float(w), found))
    return groups


class MixChunks(torch.utils.data.Dataset):
    """Index i -> one random chunk, a pure function of (seed, i): resuming at index k replays
    exactly the chunks a non-stop run would have drawn from k on."""

    def __init__(self, groups, chunk_sec, seed, augmentator=None):
        self.chunk_sec, self.seed, self.aug = chunk_sec, seed, augmentator
        self.groups = []
        w = np.array([g["weight"] for g in groups], dtype=np.float64)
        self.gw = w / w.sum()
        for g in groups:
            d = np.array([it["duration"] for it in g["items"]])
            self.groups.append((g["items"], d / d.sum()))

    def __len__(self):
        return 10**12

    def __getitem__(self, i):
        rng = np.random.default_rng((self.seed, i))
        items, p = self.groups[rng.choice(len(self.groups), p=self.gw)]
        it = items[rng.choice(len(items), p=p)]
        lo = -self.chunk_sec / 4  # like Transkun: chunks may start a little before the file
        b = rng.uniform(lo, max(it["duration"] - 3 * self.chunk_sec / 4, lo))
        e = b + self.chunk_sec
        audio = read_wav_slice(it["audio"], b, e)[:, :1]
        if self.aug is not None:
            audio = self.aug(audio)
            audio = (audio if audio.ndim == 2 else audio[:, None]).astype(np.float32)
        return audio, slice_notes(it["notes"], b, e), it["has_pedal"]


def slice_notes(notes: np.ndarray, begin: float, end: float):
    from transkun.Data import Note

    sel = notes[(notes[:, 0] < end) & (notes[:, 1] > begin)]
    return [Note(max(s, begin) - begin, min(e, end) - begin, int(p), int(v), bool(s >= begin), bool(e < end))
            for s, e, p, v in sel]


def collate(batch):
    n = min(a.shape[0] for a, _, _ in batch)
    audio = torch.from_numpy(np.stack([a[:n] for a, _, _ in batch]))
    return audio, [nt for _, nt, _ in batch], torch.tensor([hp for _, _, hp in batch])


class Seq(torch.utils.data.Sampler):
    def __init__(self, start):
        self.start = start

    def __iter__(self):
        return iter(range(self.start, 10**12))


# ------------------------------------------------------------------ eval
def val_chunks(items, chunk_sec, n, seed):
    """n fixed chunks spread over the source's val items (longest first, round robin)."""
    rng = np.random.default_rng(seed)
    items = sorted(items, key=lambda it: -it["duration"])
    out = []
    for k in range(n):
        it = items[k % len(items)]
        b = rng.uniform(0.0, max(it["duration"] - chunk_sec, 0.0))
        out.append((it, b))
    return out


@torch.no_grad()
def evaluate(model, val_sets, device, chunk_sec, stop=None):
    model.eval()
    res = {}
    for src, chunks in val_sets.items():
        f1s = []
        for it, b in chunks:
            if stop is not None and stop.requested:
                model.train()
                return None
            audio = torch.from_numpy(read_wav_slice(it["audio"], b, b + chunk_sec)[:, :1]).to(device)
            est = events.from_transkun_notes(model.generate(audio, None, step_sec=chunk_sec / 2, segment_sec=chunk_sec))
            n = it["notes"]
            sel = n[(n[:, 0] >= b) & (n[:, 0] < b + chunk_sec) & (n[:, 2] > 0)]
            ref = np.column_stack([sel[:, 0] - b, np.minimum(sel[:, 1] - b, chunk_sec), sel[:, 2], sel[:, 3]])
            # notes held from before the chunk have no onset to find: ignore estimates right at the start
            est = est[(est[:, 0] >= 0.05) & (est[:, 2] > 0)] if len(est) else est
            f1s.append(onset_f1(est, ref)[2])
        res[src] = float(np.mean(f1s))
    model.train()
    return res


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="/data")
    ap.add_argument("--out", default="/data/runs/tk_train")
    ap.add_argument("--group", action="append",
                    help="name=weight:src,src_glob,... relative to --data (repeatable; default: the mix above)")
    ap.add_argument("--init", help="initial Transkun weights (default: plain 2.0.pt), e.g. a best_transkun.pt")
    ap.add_argument("--init-conf", help="its transkun.conf (default: the stock 2.0.conf)")
    ap.add_argument("--steps", type=int, default=22500)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--chunk-sec", type=float, default=16.0)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--clip-quantile", type=float, default=0.8)
    ap.add_argument("--augment", action="store_true",
                    help="Transkun's audiomentations augmentation (pitch shift +-0.2 st, EQ, Gaussian noise)")
    ap.add_argument("--noise-folder")
    ap.add_argument("--ir-folder")
    ap.add_argument("--val-every", type=int, default=1500)
    ap.add_argument("--val-chunks", type=int, default=8, help="fixed val chunks decoded per source")
    ap.add_argument("--vram-reserve-gb", type=float, default=1.0, help="left free for the desktop")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--save-every-min", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    out, data = Path(args.out), Path(args.data)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info()
        cap = free / 2**30 - args.vram_reserve_gb
        torch.cuda.set_per_process_memory_fraction(min(cap * 2**30 / total, 1.0))
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"VRAM: {free / 2**30:.2f} of {total / 2**30:.2f} GB free, cap {cap:.2f} GB", flush=True)

    # ---- data
    groups, val_sets = [], {}
    for name, w, srcs in parse_groups(args.group or DEFAULT_GROUPS, data):
        items = []
        for src in srcs:
            tr = load_source(data, src, "train")
            hp = any(np.isin(it["notes"][:, 2], PEDALS).any() for it in tr)
            for it in tr:
                it["has_pedal"] = hp
            items += tr
            va = load_source(data, src, "val")
            if va:
                val_sets[f"{name}/{src}"] = val_chunks(va, args.chunk_sec, args.val_chunks, args.seed + len(val_sets))
            print(f"  {name:6s} {src:26s} {sum(i['duration'] for i in tr) / 3600:6.2f} h train, "
                  f"{len(va)} val items{', pedal labels' if hp else ''}", flush=True)
        groups.append({"name": name, "weight": w, "items": items})
        print(f"group {name}: weight {w}, {sum(i['duration'] for i in items) / 3600:.1f} h", flush=True)
    wsum = sum(g["weight"] for g in groups)
    per_step = args.batch_size * args.accum * args.chunk_sec
    for g in groups:
        h = sum(i["duration"] for i in g["items"]) / 3600
        seen = args.steps * per_step * g["weight"] / wsum / 3600
        print(f"  {g['name']}: {seen:.0f} h drawn over the run = {seen / h:.1f}x its data", flush=True)

    # ---- model, optimizer, resume
    # --init: start from other Transkun weights (e.g. an earlier run's best_transkun.pt) instead of
    # plain 2.0.pt; a resume from last.pt below still takes precedence
    model = PianoCoverModel.from_pretrained_transkun(args.init, args.init_conf).to(device).train()
    params = [p for n, p in model.named_parameters() if not n.startswith("film.")]
    for p in model.film.parameters():
        p.requires_grad_(False)
    decay = [p for p in params if p.ndim >= 2]
    no_decay = [p for p in params if p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}], lr=args.lr, betas=(0.9, 0.98))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_lambda(s, args.warmup, args.steps))
    clipper = AdaptiveClip(args.clip_quantile)
    step, best, history = 0, -1.0, []
    for path in (out / "last.pt", out / "last.prev.pt"):
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
        step, best, history = ck["step"], ck["best"], ck["history"]
        clipper.hist.clear()
        clipper.hist.extend(ck["clip_hist"] or [40.0])
        print(f"[resume] {path} at step {step}", flush=True)
        break
    (out / "args.json").write_text(json.dumps(vars(args), indent=2))

    def checkpoint(reason):
        save(out / "last.pt", model, {"step": step, "best": best, "history": history, "optimizer": opt.state_dict(),
                                      "scheduler": sched.state_dict(), "clip_hist": list(clipper.hist)},
             keep_previous=True)
        print(f"[ckpt] step {step} ({reason})", flush=True)

    log_path = out / "log.csv"
    if log_path.exists():  # drop rows past the checkpoint (they will be redone)
        rows = list(csv.reader(log_path.open(newline="")))
        rows = rows[:1] + [r for r in rows[1:] if r and r[0].isdigit() and int(r[0]) <= step]
        csv.writer(log_path.open("w", newline="")).writerows(rows)
    new_log = not log_path.exists()
    log_f = log_path.open("a", newline="")
    log = csv.writer(log_f)
    if new_log:
        log.writerow(["step", "loss_per_sec", "grad_norm", "clip", "lr", "sec_per_step", "peak_vram_gb"])
    stop = StopRequest()

    def run_val():
        nonlocal best
        t = time.time()
        res = evaluate(model, val_sets, device, args.chunk_sec, stop)
        if res is None:
            return
        by_group = {}
        for k, v in res.items():
            by_group.setdefault(k.split("/", 1)[0], []).append(v)
        gf1 = {g: float(np.mean(v)) for g, v in by_group.items()}
        score = float(np.mean(list(gf1.values())))
        history.append({"step": step, "score": score, "groups": gf1, "sources": res})
        with (out / "val.jsonl").open("a") as f:
            f.write(json.dumps(history[-1]) + "\n")
        print(f"[val] step {step}: score {score:.3f} {({g: round(v, 3) for g, v in gf1.items()})} "
              f"({time.time() - t:.0f}s)", flush=True)
        print("      " + " ".join(f"{k.split('/', 1)[1]}={v:.3f}" for k, v in res.items()), flush=True)
        if score > best and step > 0:
            best = score
            save(out / "best.pt", model, {"step": step, "val": history[-1]})
            torch.save({"best_state_dict": model.transkun.state_dict(), "step": step, "val": history[-1]},
                       out / "best_transkun.pt")
            shutil.copy(pretrained_paths()[1], out / "transkun.conf")
            print(f"[val] new best {score:.3f} -> best.pt, best_transkun.pt", flush=True)

    if not history:
        run_val()  # step-0 baseline: plain 2.0.pt

    aug = None
    if args.augment:
        from pianocover.data import make_augmentator
        aug = make_augmentator(args.noise_folder, args.ir_folder)
    per_batch = args.batch_size
    loader = torch.utils.data.DataLoader(
        MixChunks(groups, args.chunk_sec, args.seed, aug), batch_size=per_batch,
        sampler=Seq(step * args.accum * per_batch), collate_fn=collate, num_workers=args.workers,
        worker_init_fn=ignore_stop_signals, prefetch_factor=4 if args.workers else None,
        persistent_workers=bool(args.workers))
    it = iter(loader)
    running, t_last, t_saved = [], time.time(), time.time()
    torch.cuda.reset_peak_memory_stats() if device.type == "cuda" else None
    while step < args.steps and not stop.requested:
        opt.zero_grad(set_to_none=True)
        loss_acc = 0.0
        for _ in range(args.accum):
            audio, notes, has_pedal = next(it)
            audio = audio.to(device, non_blocking=True)
            logp = model.log_prob(audio, notes, None)  # [B, 2 pedals + 88 keys]
            mask = torch.ones_like(logp)
            mask[~has_pedal.to(device), :len(PEDALS)] = 0.0
            loss = -(logp * mask).sum() / (audio.shape[0] * audio.shape[1] / model.transkun.fs)
            (loss / args.accum).backward()
            loss_acc += float(loss.detach()) / args.accum
        if not math.isfinite(loss_acc):
            print(f"step {step}: non-finite loss, skipped", flush=True)
            continue
        gnorm, limit = clipper(params)
        opt.step()
        sched.step()
        step += 1
        running = (running + [loss_acc])[-50:]
        if step % 50 == 0:
            dt = (time.time() - t_last) / 50
            t_last = time.time()
            peak = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0
            eta = (args.steps - step) * dt / 3600
            print(f"step {step} loss/s {np.mean(running):.3f} gnorm {gnorm:.1f} lr {opt.param_groups[0]['lr']:.2e} "
                  f"{dt:.2f}s/step peak {peak:.2f} GB, ETA {eta:.1f} h (without val)", flush=True)
            log.writerow([step, f"{np.mean(running):.4f}", f"{gnorm:.2f}", f"{limit:.2f}",
                          f"{opt.param_groups[0]['lr']:.3e}", f"{dt:.3f}", f"{peak:.2f}"])
            log_f.flush()
        if step % args.val_every == 0 or step == args.steps:
            run_val()
            checkpoint("val")
            t_saved = t_last = time.time()
        elif time.time() - t_saved > args.save_every_min * 60:
            checkpoint("periodic")
            t_saved = time.time()

    checkpoint("stop requested" if stop.requested else "end")
    log_f.close()
    if stop.requested:
        print(f"stopped at step {step}; run the same command again to continue", flush=True)
        raise SystemExit(130)
    print(f"training finished, best score {best:.3f}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (SystemExit, KeyboardInterrupt):
        raise
    except BaseException:
        # e.g. a CUDA error after a GPU driver hiccup: the DataLoader workers can keep a crashed
        # process alive forever, so exit hard; the container's restart policy resumes from last.pt
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)

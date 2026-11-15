"""Transcribe the downloaded human piano covers with pretrained Transkun V2.

These transcriptions become the training targets. Run on a GPU; split across GPUs with
--shard i --num-shards n (one process per GPU).

Output: <data>/cover_events/<piano_id>.npy  (+ .mid for listening checks with --midi)
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch

from . import SAMPLE_RATE, events
from .audio import decode, list_audio
from .model import load_pretrained_transkun


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, help="directory produced by pianocover.download")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--vram-fraction", type=float, default=0.5,
                    help="cap GPU memory so the machine stays usable while the dataset is built")
    ap.add_argument("--midi", action="store_true", help="also write .mid files")
    args = ap.parse_args(argv)
    if str(args.device).startswith("cuda"):
        import torch as _torch

        _torch.cuda.set_per_process_memory_fraction(args.vram_fraction)

    data = Path(args.data)
    out_dir = data / "cover_events"
    out_dir.mkdir(parents=True, exist_ok=True)
    wavs = list_audio(data / "cover")
    wavs = [w for i, w in enumerate(wavs) if i % args.num_shards == args.shard]
    todo = [w for w in wavs if not (out_dir / f"{w.stem}.npy").exists()]
    print(f"{len(todo)} of {len(wavs)} covers to transcribe on {args.device}", flush=True)

    model = load_pretrained_transkun(device=args.device)
    torch.set_grad_enabled(False)
    t_audio = t_wall = 0.0
    for i, wav in enumerate(todo, 1):
        lock = claim(out_dir, wav.stem)
        if lock is None:
            continue
        t0 = time.time()
        try:
            dur = _transcribe_one(model, wav, out_dir, args)
        finally:
            lock.unlink(missing_ok=True)
        if dur:
            t_audio += dur
            t_wall += time.time() - t0
        if i % 20 == 0 or i == len(todo):
            print(f"[{i}/{len(todo)}] {t_audio / max(t_wall, 1e-6):.1f}x realtime", flush=True)


STALE_LOCK_SEC = 1800


def claim(out_dir: Path, stem: str) -> Path | None:
    """Claim one cover for this process; None if it is done or another worker holds it.

    Several workers may share one folder. A claim is an exclusively created `<stem>.lock`;
    one older than STALE_LOCK_SEC belongs to a crashed worker and is taken over by renaming
    it away, which only one process can win.
    """
    done = out_dir / f"{stem}.npy"
    if done.exists():
        return None
    lock = out_dir / f"{stem}.lock"
    try:
        if time.time() - lock.stat().st_mtime > STALE_LOCK_SEC:
            os.replace(lock, out_dir / f"{stem}.lock.stale.{os.getpid()}")
            (out_dir / f"{stem}.lock.stale.{os.getpid()}").unlink(missing_ok=True)
    except FileNotFoundError:
        pass  # no lock, or another worker took the stale one over first
    try:
        os.close(os.open(lock, os.O_CREAT | os.O_EXCL))
    except FileExistsError:
        return None
    if done.exists():  # finished by another worker between the check and the claim
        lock.unlink(missing_ok=True)
        return None
    return lock


def _transcribe_one(model, wav, out_dir, args) -> float:
    """Transcribe one cover into <out_dir>/<id>.npy. Returns its duration (0 if skipped)."""
    try:
        y = decode(wav, SAMPLE_RATE)
    except RuntimeError as e:
        print(f"skip {wav.name}: {e}", flush=True)
        return 0.0
    x = torch.from_numpy(y[:, None]).to(args.device)
    try:
        notes = model.transcribe(x, discardSecondHalf=False)
    except torch.cuda.OutOfMemoryError:
        del x
        torch.cuda.empty_cache()
        print(f"skip {wav.name}: out of GPU memory under the --vram-fraction cap", flush=True)
        return 0.0
    ev = events.sanitize(events.from_transkun_notes(notes))
    tmp = out_dir / f"{wav.stem}.{os.getpid()}.tmp.npy"  # per process: workers never share it
    events.save_npy(tmp, ev)
    tmp.replace(out_dir / f"{wav.stem}.npy")
    if args.midi:
        events.to_pretty_midi(ev).write(str(out_dir / f"{wav.stem}.mid"))
    return len(y) / SAMPLE_RATE


if __name__ == "__main__":
    main()

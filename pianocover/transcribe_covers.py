"""Transcribe the downloaded human piano covers with pretrained Transkun V2.

These transcriptions become the training targets. Run on a GPU; split across GPUs with
--shard i --num-shards n (one process per GPU).

Output: <data>/cover_events/<piano_id>.npy  (+ .mid for listening checks with --midi)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from . import SAMPLE_RATE, events
from .audio import read_wav_slice, wav_info
from .model import load_pretrained_transkun


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, help="directory produced by pianocover.download")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--midi", action="store_true", help="also write .mid files")
    args = ap.parse_args(argv)

    data = Path(args.data)
    out_dir = data / "cover_events"
    out_dir.mkdir(parents=True, exist_ok=True)
    wavs = sorted((data / "cover").glob("*.wav"))
    wavs = [w for i, w in enumerate(wavs) if i % args.num_shards == args.shard]
    todo = [w for w in wavs if not (out_dir / f"{w.stem}.npy").exists()]
    print(f"{len(todo)} of {len(wavs)} covers to transcribe on {args.device}", flush=True)

    model = load_pretrained_transkun(device=args.device)
    torch.set_grad_enabled(False)
    t_audio = t_wall = 0.0
    for i, wav in enumerate(todo, 1):
        fs, dur = wav_info(wav)
        if fs != SAMPLE_RATE:
            print(f"skip {wav.name}: sample rate {fs} != {SAMPLE_RATE}")
            continue
        t0 = time.time()
        x = torch.from_numpy(read_wav_slice(wav, 0.0, dur)).to(args.device)
        notes = model.transcribe(x, discardSecondHalf=False)
        ev = events.sanitize(events.from_transkun_notes(notes))
        tmp = out_dir / f"{wav.stem}.tmp.npy"
        events.save_npy(tmp, ev)
        tmp.replace(out_dir / f"{wav.stem}.npy")
        if args.midi:
            events.to_pretty_midi(ev).write(str(out_dir / f"{wav.stem}.mid"))
        t_audio += dur
        t_wall += time.time() - t0
        if i % 20 == 0 or i == len(todo):
            print(f"[{i}/{len(todo)}] {t_audio / max(t_wall, 1e-6):.1f}x realtime", flush=True)


if __name__ == "__main__":
    main()

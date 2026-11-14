"""Precompute AMT-APC inputs for a dataset built by pianocover.build_dataset.

For every original song: decode once, compute the log-mel spectrogram on the GPU, pad it like
AMT-APC (32 frames of -18 before, up to a multiple of 512 plus 32 after) and store it as a
float16 .npy that training reads with mmap. For every pair: pedal-extended cover notes, the
style vector, and the song's unaligned regions.

  python -m pianocover.apc.prepare --ds /data/ds [--device cuda]

Output in <ds>/apc/: spec/<pop_id>.npy, train.pkl, val.pkl, style_vectors.json.
Spectrograms take about 32 KB per second of audio (200 MB per hour).
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from ..audio import decode
from ..data import load_split
from . import MARGIN, MIN_VALUE, NUM_FRAME, SR
from .features import logmel
from .labels import pedal_extend, style_vector


def padded_spec(feat: np.ndarray) -> np.ndarray:
    """AMT-APC's preprocess_feature: [margin | feat | pad to k*512 | margin], float16."""
    n = feat.shape[0]
    tail = int(np.ceil(n / NUM_FRAME) * NUM_FRAME) - n + MARGIN
    out = np.full((MARGIN + n + tail, feat.shape[1]), MIN_VALUE, np.float16)
    out[MARGIN:MARGIN + n] = feat
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ds", required=True, help="directory with train.pkl / val.pkl from build_dataset")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--vram-fraction", type=float, default=0.5,
                    help="cap GPU memory so the machine stays usable while the dataset is built")
    ap.add_argument("--min-notes", type=int, default=200)
    args = ap.parse_args(argv)
    if str(args.device).startswith("cuda"):
        import torch as _torch

        _torch.cuda.set_per_process_memory_fraction(args.vram_fraction)

    ds = Path(args.ds)
    out = ds / "apc"
    (out / "spec").mkdir(parents=True, exist_ok=True)
    svs = {}
    for split in ("train", "val"):
        items = load_split(ds / f"{split}.pkl") if (ds / f"{split}.pkl").exists() else []
        rows = []
        for i, it in enumerate(items, 1):
            spec_path = out / "spec" / f"{it['pop_id']}.npy"
            if not spec_path.exists():
                feat = logmel(decode(it["audio_path"], SR), args.device)
                tmp = spec_path.with_suffix(".tmp.npy")
                np.save(tmp, padded_spec(feat))
                tmp.replace(spec_path)
            n_frames = np.load(spec_path, mmap_mode="r").shape[0] - 2 * MARGIN
            notes = pedal_extend(np.asarray(it["events"], np.float64))
            if len(notes) < args.min_notes:
                continue
            sv = style_vector(notes, it["duration"])
            svs[it["piano_id"]] = sv.tolist()
            rows.append({"pop_id": it["pop_id"], "piano_id": it["piano_id"], "spec": spec_path.name,
                         "n_frames": int(n_frames), "notes": notes.astype(np.float32), "sv": sv,
                         "bad_regions": it["bad_regions"], "source": it.get("source", "web")})
            if i % 100 == 0:
                print(f"{split}: {i}/{len(items)}", flush=True)
        with (out / f"{split}.pkl").open("wb") as f:
            pickle.dump(rows, f)
        print(f"{split}: {len(rows)} pairs", flush=True)
    (out / "style_vectors.json").write_text(json.dumps(svs))


if __name__ == "__main__":
    main()

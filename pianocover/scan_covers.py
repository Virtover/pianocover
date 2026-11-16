"""Measure, for every downloaded pair, whether the "cover" is really a solo-piano cover.

Two measurements, written to <data>/cover_scan.csv and used by build_dataset to drop pairs:
  cover_vocal   share of the cover's energy that Demucs (htdemucs) puts in the vocal stem,
                median over three 20 s excerpts. Solo piano covers measure ~0; covers with
                singing, even quiet, measure 0.06 or more.
  timbre        cosine similarity of the original's and the cover's mean MFCCs (c1..c19,
                after removing the average over all files). The original re-uploaded as a
                "cover", or a band arrangement with the piano up front, measures 0.95 or more;
                real piano covers measured at most 0.76.
Thresholds were set by listening tests (2026-11-16); see build_dataset --max-cover-vocal / --max-timbre.

Resumable: per-file results are cached in <data>/scan/ (mfcc_stats.npz, vocal.csv).

  python -m pianocover.scan_covers --data /data/raw [--vram-fraction 0.4]
"""

from __future__ import annotations

import argparse
import csv
import queue
import threading
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .audio import decode, find_audio

MFCC_SR = 22050
SEG_SEC = 20
SEG_AT = (0.25, 0.5, 0.75)


def mfcc_stats(path) -> np.ndarray:
    """[38] = mean and std of MFCC c1..c19 over the whole file."""
    import librosa

    y = decode(path, MFCC_SR)
    m = librosa.feature.mfcc(y=y, sr=MFCC_SR, n_mfcc=20, hop_length=512)[1:]
    return np.concatenate([m.mean(1), m.std(1)]).astype(np.float32)


def _mfcc_job(item):
    key, path = item
    try:
        return key, mfcc_stats(path)
    except Exception as e:  # noqa: BLE001  (undecodable file: no measurement)
        print(f"skip {path}: {e}", flush=True)
        return key, None


def timbre_similarity(stats: dict, pop_key: str, cover_key: str, mu: np.ndarray) -> float:
    def unit(k):
        v = stats[k][:19] - mu
        return v / (np.linalg.norm(v) + 1e-8)

    return float(unit(pop_key) @ unit(cover_key))


def vocal_excerpts(path) -> np.ndarray:
    """[3, 2, SEG] stereo 44.1 kHz excerpts around 25/50/75% of the file."""
    from .teacher.models import decode_audio

    x = decode_audio(path, 44100, 2)
    n, seg = x.shape[1], SEG_SEC * 44100
    segs = []
    for f in SEG_AT:
        s = max(0, min(int(f * n) - seg // 2, n - seg))
        e = x[:, s:s + seg]
        segs.append(np.pad(e, ((0, 0), (0, seg - e.shape[1]))))
    return np.ascontiguousarray(np.stack(segs))


def _load_cache(scan: Path):
    stats, vocal = {}, {}
    if (scan / "mfcc_stats.npz").exists():
        z = np.load(scan / "mfcc_stats.npz")
        stats = dict(zip(z["keys"].tolist(), z["vals"]))
    if (scan / "vocal.csv").exists():
        vocal = {r["piano_id"]: float(r["cover_vocal"]) for r in csv.DictReader(open(scan / "vocal.csv"))}
    return stats, vocal


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="directory with pop/, cover/, pairs.csv")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--vram-fraction", type=float, default=0.4)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args(argv)

    data = Path(args.data)
    scan = data / "scan"
    scan.mkdir(exist_ok=True)
    with (data / "pairs.csv").open() as f:
        pairs = [(r["pop_id"], r["piano_id"]) for r in csv.DictReader(f)]
    files = {}
    for p, c in pairs:
        for d, i in (("pop", p), ("cover", c)):
            path = find_audio(data / d, i)
            if path:
                files[f"{d}/{i}"] = path
    stats, vocal = _load_cache(scan)

    todo = [(k, p) for k, p in files.items() if k not in stats]
    print(f"{len(pairs)} pairs; MFCC: {len(todo)} of {len(files)} files to do", flush=True)
    if todo:
        with ProcessPoolExecutor(args.workers) as ex:
            for n, (k, v) in enumerate(ex.map(_mfcc_job, todo, chunksize=8), 1):
                if v is not None:
                    stats[k] = v
                if n % 500 == 0 or n == len(todo):
                    keys = sorted(stats)
                    np.savez(scan / "mfcc_stats.npz", keys=np.array(keys), vals=np.stack([stats[k] for k in keys]))

    covers = sorted({c for _, c in pairs if f"cover/{c}" in files and c not in vocal})
    print(f"vocal: {len(covers)} covers to do", flush=True)
    if covers:
        import torch
        from demucs.apply import apply_model
        from demucs.pretrained import get_model

        if str(args.device).startswith("cuda"):
            torch.cuda.set_per_process_memory_fraction(args.vram_fraction)
        model = get_model("htdemucs").to(args.device).eval()
        vi = model.sources.index("vocals")
        q: queue.Queue = queue.Queue(maxsize=8)

        def loader():
            for c in covers:
                try:
                    q.put((c, vocal_excerpts(files[f"cover/{c}"])))
                except Exception as e:  # noqa: BLE001
                    print(f"skip cover {c}: {e}", flush=True)
            q.put(None)

        threading.Thread(target=loader, daemon=True).start()
        new_file = not (scan / "vocal.csv").exists()
        with (scan / "vocal.csv").open("a", newline="") as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(["piano_id", "cover_vocal"])
            n = 0
            while (job := q.get()) is not None:
                c, x = job
                with torch.inference_mode():
                    wav = torch.from_numpy(x).to(args.device)
                    ref = wav.mean(1, keepdim=True)
                    mean, std = ref.mean(-1, keepdim=True), ref.std(-1, keepdim=True) + 1e-8
                    out = apply_model(model, (wav - mean) / std, shifts=0, split=True, overlap=0.25,
                                      progress=False)
                    e = (out ** 2).mean(dim=(2, 3))  # [excerpts, sources]
                    vocal[c] = float((e[:, vi] / e.sum(1).clamp_min(1e-12)).median())
                w.writerow([c, f"{vocal[c]:.4f}"])
                f.flush()
                n += 1
                if n % 200 == 0:
                    print(f"vocal {n}/{len(covers)}", flush=True)

    mu = np.stack([v[:19] for v in stats.values()]).mean(0)
    n_rows = 0
    with (data / "cover_scan.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pop_id", "piano_id", "cover_vocal", "timbre"])
        for p, c in pairs:
            if f"pop/{p}" in stats and f"cover/{c}" in stats and c in vocal:
                w.writerow([p, c, f"{vocal[c]:.4f}", f"{timbre_similarity(stats, f'pop/{p}', f'cover/{c}', mu):.4f}"])
                n_rows += 1
    print(f"wrote {data / 'cover_scan.csv'}: {n_rows} of {len(pairs)} pairs", flush=True)


if __name__ == "__main__":
    main()

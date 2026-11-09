"""Download pop / piano-cover pairs from YouTube and convert them to 44.1 kHz mono WAV.

Accepted pair lists:
  * CSV with columns `piano_ids,pop_ids` (Pop2Piano's train_dataset.csv, Etude's assets/dataset.csv)
  * AMT-APC's dataset/src.json: {"title": {"original": id, "pianos": [id, ...]}}

Output layout:
  <out>/pop/<pop_id>.wav
  <out>/cover/<piano_id>.wav
  <out>/pairs.csv            (pop_id,piano_id of pairs whose audio both exist)
  <out>/download_failures.txt

Check YouTube's terms and the copyright rules that apply to you before downloading or
training on this material, especially for a commercial product.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import SAMPLE_RATE


def read_pairs(path: str) -> list[tuple[str, str]]:
    """Return a de-duplicated list of (pop_id, piano_id)."""
    p = Path(path)
    pairs: list[tuple[str, str]] = []
    if p.suffix.lower() == ".json":
        data = json.loads(p.read_text())
        for entry in data.values():
            for piano in entry.get("pianos", []):
                pairs.append((entry["original"], piano))
    else:
        with p.open(newline="") as f:
            reader = csv.DictReader(f)
            cols = {c.strip().lower(): c for c in reader.fieldnames or []}
            if "pop_ids" not in cols or "piano_ids" not in cols:
                raise ValueError(f"{path}: expected columns pop_ids,piano_ids; got {reader.fieldnames}")
            for row in reader:
                pop, piano = row[cols["pop_ids"]].strip(), row[cols["piano_ids"]].strip()
                if pop and piano:
                    pairs.append((pop, piano))
    seen, out = set(), []
    for pair in pairs:
        if pair not in seen:
            seen.add(pair)
            out.append(pair)
    return out


def fetch_wav(video_id: str, dst: Path, max_minutes: float, cookies: str | None) -> str | None:
    """Download one video's audio to `dst` (mono, 44.1 kHz, 16-bit). Returns an error string or None."""
    if dst.exists() and dst.stat().st_size > 1000:
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        cmd = [
            sys.executable, "-m", "yt_dlp", "-q", "--no-warnings", "--no-playlist",
            "-f", "bestaudio/best",
            "--match-filter", f"duration < {int(max_minutes * 60)}",
            "-o", os.path.join(tmp, "audio.%(ext)s"),
            f"https://www.youtube.com/watch?v={video_id}",
        ]
        if cookies:
            cmd[3:3] = ["--cookies", cookies]
        r = subprocess.run(cmd, capture_output=True, text=True)
        files = list(Path(tmp).glob("audio.*"))
        if r.returncode != 0 or not files:
            return (r.stderr or "no file (filtered by duration?)").strip().splitlines()[-1][:300]
        part = dst.with_suffix(".part.wav")
        r = subprocess.run(
            ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(files[0]),
             "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", str(part)],
            capture_output=True, text=True)
        if r.returncode != 0:
            return ("ffmpeg: " + r.stderr.strip())[:300]
        part.replace(dst)
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pairs", nargs="+", help="one or more CSV (piano_ids,pop_ids) / AMT-APC src.json files")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="only the first N pairs (0 = all)")
    ap.add_argument("--max-minutes", type=float, default=10.0, help="skip videos longer than this")
    ap.add_argument("--cookies", help="optional cookies.txt for yt-dlp")
    args = ap.parse_args(argv)

    out = Path(args.out)
    pairs, seen = [], set()
    for path in args.pairs:
        for pair in read_pairs(path):
            if pair not in seen:
                seen.add(pair)
                pairs.append(pair)
    if args.limit:
        pairs = pairs[: args.limit]
    jobs = {}
    for pop, piano in pairs:
        jobs.setdefault(("pop", pop), out / "pop" / f"{pop}.wav")
        jobs.setdefault(("cover", piano), out / "cover" / f"{piano}.wav")
    print(f"{len(pairs)} pairs -> {len(jobs)} unique videos", flush=True)

    failures = {}
    with ThreadPoolExecutor(args.workers) as pool:
        futs = {pool.submit(fetch_wav, vid, dst, args.max_minutes, args.cookies): (kind, vid)
                for (kind, vid), dst in jobs.items()}
        for i, fut in enumerate(as_completed(futs), 1):
            key = futs[fut]
            err = fut.result()
            if err:
                failures[key] = err
            if i % 50 == 0 or i == len(futs):
                print(f"[{i}/{len(futs)}] failures so far: {len(failures)}", flush=True)

    ok = [(pop, piano) for pop, piano in pairs
          if ("pop", pop) not in failures and ("cover", piano) not in failures]
    with (out / "pairs.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pop_id", "piano_id"])
        w.writerows(ok)
    with (out / "download_failures.txt").open("w") as f:
        for (kind, vid), err in sorted(failures.items()):
            f.write(f"{kind}\t{vid}\t{err}\n")
    print(f"done: {len(ok)}/{len(pairs)} pairs usable, see {out / 'pairs.csv'}")


if __name__ == "__main__":
    main()

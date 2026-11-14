"""Download pop / piano-cover pairs from YouTube.

By default the audio is kept as YouTube serves it (opus/webm or m4a, about 4 MB per song);
later steps decode it with ffmpeg. `--format wav` converts to 44.1 kHz mono WAV instead
(about 20 MB per song), which Transkun training reads faster.

Accepted pair lists:
  * CSV with columns `piano_ids,pop_ids` (Pop2Piano's train_dataset.csv, Etude's assets/dataset.csv)
  * AMT-APC's dataset/src.json: {"title": {"original": id, "pianos": [id, ...]}}

Output layout:
  <out>/pop/<pop_id>.<ext>
  <out>/cover/<piano_id>.<ext>
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
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import SAMPLE_RATE
from .audio import AUDIO_EXTS, find_audio


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


def fetch_audio(video_id: str, dst_dir: Path, fmt: str, max_minutes: float, cookies: str | None) -> str | None:
    """Download one video's audio into `dst_dir/<video_id>.<ext>`. Returns an error string or None.

    fmt "native" keeps the downloaded stream; "wav" converts to mono 44.1 kHz 16-bit WAV.
    """
    if find_audio(dst_dir, video_id) is not None:
        return None
    dst_dir.mkdir(parents=True, exist_ok=True)
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
        files = [f for f in Path(tmp).glob("audio.*") if ".part" not in f.name]
        if r.returncode != 0 or not files:
            return (r.stderr or "no file (filtered by duration?)").strip().splitlines()[-1][:300]
        src = files[0]
        if fmt == "native":
            ext = src.suffix.lower() if src.suffix.lower() in AUDIO_EXTS else ".mka"
            part = dst_dir / f"{video_id}.part{ext}"
            shutil.move(str(src), part)
            part.replace(dst_dir / f"{video_id}{ext}")
            return None
        part = dst_dir / f"{video_id}.part.wav"
        r = subprocess.run(
            ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(src),
             "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", str(part)],
            capture_output=True, text=True)
        if r.returncode != 0:
            return ("ffmpeg: " + r.stderr.strip())[:300]
        part.replace(dst_dir / f"{video_id}.wav")
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pairs", nargs="+", help="one or more CSV (piano_ids,pop_ids) / AMT-APC src.json files")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--workers", type=int, default=2, help="parallel downloads (more gets rate limited)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N pairs (0 = all)")
    ap.add_argument("--max-minutes", type=float, default=10.0, help="skip videos longer than this")
    ap.add_argument("--cookies", help="optional cookies.txt for yt-dlp")
    ap.add_argument("--format", choices=("native", "wav"), default="native",
                    help="keep the downloaded audio stream (default) or convert to WAV")
    ap.add_argument("--sleep", type=float, default=1.0,
                    help="pause per download and worker, to stay under YouTube's rate limits")
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
        jobs.setdefault(("pop", pop), out / "pop")
        jobs.setdefault(("cover", piano), out / "cover")
    print(f"{len(pairs)} pairs -> {len(jobs)} unique videos", flush=True)

    failures = {}
    with ThreadPoolExecutor(args.workers) as pool:
        def job(vid, dst):
            err = fetch_audio(vid, dst, args.format, args.max_minutes, args.cookies)
            time.sleep(args.sleep)
            return err

        futs = {pool.submit(job, vid, dst): (kind, vid) for (kind, vid), dst in jobs.items()}
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

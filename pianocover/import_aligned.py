"""Import already-aligned (song audio, piano cover MIDI) pairs, with a license gate.

For data whose MIDI is already on the song's timeline, so no DTW alignment is needed:
  * teacher  - covers produced offline by the reduction pipeline from CC BY / CC0 songs
  * silver   - ACE-Step band versions generated from piano arrangements you own
  * gold     - commissioned human covers recorded as MIDI while following the song

Manifest CSV (paths relative to the manifest file):
  id,audio,midi,source,license,song_id,offset
  fma_000123,audio/000123.mp3,midi/000123.mid,teacher,CC-BY-4.0,fma_000123,0
  - song_id  groups versions of one song so they share a train/val split (default: id)
  - offset   seconds added to every MIDI time (default 0)

Licenses accepted by default: CC0, CC-BY (any version), public-domain, owned,
work-for-hire, ace-step. Anything with NC or ND is always refused; ShareAlike (SA) is
refused unless --allow-sharealike. Unrecognised strings are refused, so every training
pair has a recorded, commercially usable license.

Output (in --out): train.pkl, val.pkl, style_stats.json, import_report.csv,
audio/<sha1 of source path>.wav (one per distinct source audio file)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import pickle
import re
import subprocess
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from . import SAMPLE_RATE, events
from . import style as style_mod

ALLOWED_EXACT = {"cc0", "public-domain", "pd", "owned", "work-for-hire", "commissioned", "ace-step"}


def license_ok(text: str, allow_sharealike: bool = False) -> tuple[bool, str]:
    """Return (accepted, reason) for a license string such as 'CC-BY-4.0' or 'CC BY-NC 3.0'."""
    t = (text or "").strip().lower().replace("_", "-").replace(" ", "-")
    if not t or t in {"unknown", "none", "n/a"}:
        return False, "no license recorded"
    parts = set(re.split(r"[-/]+", t))
    if "nc" in parts or "noncommercial" in parts or "non-commercial" in t:
        return False, "non-commercial"
    if "nd" in parts or "noderivatives" in parts or "noderivs" in parts:
        return False, "no-derivatives (a piano cover is an adaptation)"
    if t in ALLOWED_EXACT:
        return True, "ok"
    if t.startswith("cc0"):
        return True, "ok"
    if t.startswith("cc-by") or t.startswith("ccby"):
        if "sa" in parts and not allow_sharealike:
            return False, "share-alike (pass --allow-sharealike to accept)"
        return True, "ok"
    return False, f"unrecognised license {text!r}"


def pedal_intervals(ccs, number: int, end_time: float, threshold: int = 64):
    """Turn (time, value) control changes of one CC number into [start, end) down intervals."""
    out, start = [], None
    for t, v in sorted(ccs):
        down = v >= threshold
        if down and start is None:
            start = t
        elif not down and start is not None:
            if t > start:
                out.append((start, t))
            start = None
    if start is not None and end_time > start:
        out.append((start, end_time))
    return out


def midi_to_events(path, offset: float = 0.0) -> np.ndarray:
    """All non-drum notes plus sustain (CC64) / soft (CC67) pedal intervals as an event array."""
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(str(path))
    end = pm.get_end_time()
    rows = []
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        rows += [[n.start, n.end, n.pitch, n.velocity] for n in inst.notes]
        for number, pitch in ((64, -64), (67, -67)):
            ccs = [(c.time, c.value) for c in inst.control_changes if c.number == number]
            rows += [[s, e, pitch, 127] for s, e in pedal_intervals(ccs, number, end)]
    if not rows:
        return events.empty()
    ev = np.array(rows, dtype=np.float64)
    ev[:, :2] += offset
    return ev


def to_wav(src: Path, dst: Path) -> None:
    if dst.exists() and dst.stat().st_size > 1000:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_suffix(f".part{os.getpid()}.wav")  # several rows may share one source file
    r = subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(src), "-ac", "1",
                        "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", str(part)], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("ffmpeg: " + r.stderr.strip()[:300])
    part.replace(dst)


def process_row(job):
    row, base, out, min_notes = job
    item_id = row["id"]
    info = {"id": item_id, "source": row["source"], "license": row["license"]}
    try:
        from .audio import wav_info

        # one WAV per *source file*: rows sharing audio (e.g. several difficulty levels of one
        # teacher song) must not store it several times
        src = (base / row["audio"]).resolve()
        wav_name = f"{hashlib.sha1(str(src).encode()).hexdigest()[:16]}.wav"
        wav = out / "audio" / wav_name
        to_wav(src, wav)
        _, duration = wav_info(wav)
        ev = midi_to_events(base / row["midi"], float(row.get("offset") or 0.0))
        ev = ev[(ev[:, events.END] > 0) & (ev[:, events.START] < duration)]
        ev[:, events.END] = np.minimum(ev[:, events.END], duration)
        ev = events.sanitize(ev)
        n_notes = int((ev[:, events.PITCH] > 0).sum())
        if n_notes < min_notes:
            return {**info, "status": "too_few_notes", "n_notes": n_notes}
        return {
            **info, "status": "ok",
            "pop_id": row.get("song_id") or item_id, "piano_id": item_id,
            "audio": str(Path("audio") / wav_name), "duration": float(duration),
            "events": ev.astype(np.float32), "bad_regions": [], "raw_style": style_mod.raw_style(ev),
            "n_notes": n_notes, "quality": 1.0,
        }
    except Exception as exc:
        return {**info, "status": "error", "error": f"{type(exc).__name__}: {exc}",
                "trace": traceback.format_exc(limit=2)}


def is_val(group_id: str, fraction: float) -> bool:
    h = int(hashlib.md5(group_id.encode()).hexdigest()[:8], 16)
    return (h % 10000) < fraction * 10000


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifest", help="CSV: id,audio,midi,source,license[,song_id,offset]")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--val-fraction", type=float, default=0.03)
    ap.add_argument("--min-notes", type=int, default=50)
    ap.add_argument("--allow-sharealike", action="store_true")
    args = ap.parse_args(argv)

    manifest = Path(args.manifest)
    base, out = manifest.parent, Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with manifest.open(newline="") as f:
        rows = list(csv.DictReader(f))
    required = {"id", "audio", "midi", "source", "license"}
    if rows and not required <= set(rows[0]):
        raise SystemExit(f"manifest needs columns {sorted(required)}, got {list(rows[0])}")
    if len({r["id"] for r in rows}) != len(rows):
        raise SystemExit("manifest ids must be unique")

    refused, jobs = [], []
    for r in rows:
        ok, why = license_ok(r["license"], args.allow_sharealike)
        if ok:
            jobs.append((r, base, out, args.min_notes))
        else:
            refused.append({"id": r["id"], "source": r["source"], "license": r["license"],
                            "status": "license_refused", "error": why})
    print(f"{len(rows)} rows: {len(jobs)} pass the license gate, {len(refused)} refused", flush=True)

    results = list(refused)
    with ProcessPoolExecutor(args.workers) as pool:
        for i, r in enumerate(pool.map(process_row, jobs, chunksize=4), 1):
            results.append(r)
            if i % 100 == 0 or i == len(jobs):
                print(f"[{i}/{len(jobs)}] ok={sum(x['status'] == 'ok' for x in results)}", flush=True)

    with (out / "import_report.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "source", "license", "status", "n_notes", "error"],
                           extrasaction="ignore")
        w.writeheader()
        w.writerows(results)

    ok = [r for r in results if r["status"] == "ok"]
    if not ok:
        raise SystemExit("nothing imported; see import_report.csv")
    train = [r for r in ok if not is_val(r["pop_id"], args.val_fraction)]
    val = [r for r in ok if is_val(r["pop_id"], args.val_fraction)]
    norm = style_mod.StyleNormalizer.fit(np.stack([r["raw_style"] for r in (train or ok)]))
    norm.save(out / "style_stats.json")
    for r in ok:
        r["style"] = norm.normalize(r["raw_style"])
        r.pop("trace", None)
    for name, split in (("train", train), ("val", val)):
        with (out / f"{name}.pkl").open("wb") as f:
            pickle.dump({"data_root": str(out.resolve()), "items": split}, f)
    by_source = {}
    for r in ok:
        by_source[r["source"]] = by_source.get(r["source"], 0) + 1
    print(f"imported {len(ok)} pairs {by_source}: train={len(train)} val={len(val)}, "
          f"{sum(r['duration'] for r in ok) / 3600:.1f} h")


if __name__ == "__main__":
    main()

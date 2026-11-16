"""Organ pairs from public-domain scores: a real organ recording (input) + the score's notes (target).

A piano version of an organ piece is very often the same notes, so instead of a piano cover the
target is the PDMX score itself. The score is rendered with a piano soundfont and stored as the
"cover" audio, its notes as the "cover transcription"; scan_covers and build_dataset then work
unchanged and align the score to the recording.

  python scripts/pdmx_organ.py search  data/meta/pdmx_organ_candidates.csv --out data/meta/pdmx_organ_pairs.jsonl
  python scripts/pdmx_organ.py prepare data/meta/pdmx_organ_pairs.jsonl --pdmx /data/pdmx \
      --sf2 /sf2/YDP-GrandPiano-20160804.sf2 --data /data/pdmx_organ

Candidates come from PDMX (CC BY 4.0) organ-only scores (GM programs 16-20) in the
no_license_conflict + deduplicated subsets whose composer died before 1955.
"""
import argparse
import csv
import json
import re
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import numpy as np

EXCLUDE_TITLE = re.compile(r"virgil fox|improvvisazione|\(wip\)|exercitium|bagpipe", re.I)
ORGAN = re.compile(r"organ|orgel|orgue|organo|오르간|hauptwerk", re.I)
NOT_REC = re.compile(r"piano|synthesia|midi|tutorial|lesson|score|sheet|noten|partitura|easy|"
                     r"harmonium|accordion|choir|chor\b|sing|vocal|lyrics|guitar|reaction|hour|compilation", re.I)
STOP = {"the", "and", "der", "die", "das", "dich", "ich", "ein", "for", "von", "mit", "organ", "orgel",
        "bach", "major", "minor", "dur", "moll", "professional", "production", "score", "no", "op"}


def words(title):
    t = re.sub(r"\{.*?\}|\(.*?\)|bwv\s*\d+\w?|[^\w\s']", " ", title.lower())
    return [w for w in t.split() if len(w) > 3 and w not in STOP]


def search(args):
    from yt_dlp import YoutubeDL

    ydl = YoutubeDL({"quiet": True, "no_warnings": True, "extract_flat": True, "skip_download": True})
    rows = list(csv.DictReader(open(args.candidates, encoding="utf8")))
    seen_piece, seen_vid, n = set(), set(), 0
    with open(args.out, "w", encoding="utf8") as f:
        for r in rows:
            title, comp = r["title"], r["composer_name"]
            if EXCLUDE_TITLE.search(title):
                continue
            key = f"bwv{r['bwv']}" if r["bwv"] else " ".join(words(title)[:4])
            if not key or key in seen_piece:
                continue
            seen_piece.add(key)
            last = (re.findall(r"[A-Za-zÀ-ÿ'-]{4,}", comp) or [""])[-1]
            q = f"{re.sub(r'{.*?}', '', title)} {last if last.lower() not in title.lower() else ''} organ"
            try:
                res = (ydl.extract_info(f"ytsearch12:{q}", download=False) or {}).get("entries") or []
            except Exception as e:  # noqa: BLE001
                print("search failed:", title, str(e)[:100], flush=True)
                time.sleep(args.sleep * 10)
                continue
            dur_score = float(r["song_length.seconds"])
            need = words(title)
            pick = None
            for e in res:
                vid, t, d = e.get("id"), e.get("title") or "", e.get("duration") or 0
                if not vid or vid in seen_vid or not ORGAN.search(t) or NOT_REC.search(t):
                    continue
                if not (0.6 * dur_score <= d <= 1.6 * dur_score + 30) or d > 600:
                    continue
                if r["bwv"]:
                    if not re.search(rf"bwv\.?\s*0*{re.escape(r['bwv'])}\b", t, re.I):
                        continue
                elif need and sum(w in t.lower() for w in need) < max(1, (len(need) + 1) // 2):
                    continue
                pick = {"id": vid, "title": t, "channel": e.get("channel") or e.get("uploader"), "duration": d}
                break
            time.sleep(args.sleep)
            if pick:
                seen_vid.add(pick["id"])
                f.write(json.dumps({"score": r, "recording": pick}, ensure_ascii=False) + "\n")
                f.flush()
                n += 1
                print(f"+ {title[:50]:50s} -> {pick['title'][:60]}", flush=True)
            if n >= args.max_pairs:
                break
    print(f"{n} pairs", flush=True)


def score_events(midi_bytes) -> np.ndarray:
    """All notes of the score as [start, end, pitch, velocity], folded into the piano's 88 keys."""
    import io

    import pretty_midi

    pm = pretty_midi.PrettyMIDI(io.BytesIO(midi_bytes))
    ev = []
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            p = n.pitch
            while p < 21:
                p += 12
            while p > 108:
                p -= 12
            ev.append([n.start, n.end, p, max(1, min(127, n.velocity))])
    ev = np.array(sorted(ev), dtype=np.float64).reshape(-1, 4)
    return ev


def prepare(args):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from pianocover import events
    from pianocover.download import fetch_audio

    data = Path(args.data)
    (data / "cover").mkdir(parents=True, exist_ok=True)
    (data / "cover_events").mkdir(exist_ok=True)
    (data / "pop").mkdir(exist_ok=True)
    pairs = [json.loads(line) for line in open(args.pairs, encoding="utf8")]
    want = {p["score"]["mid"].lstrip("./") for p in pairs}
    mids = {}
    with tarfile.open(Path(args.pdmx) / "mid.tar.gz") as tar:
        for m in tar:
            name = m.name.lstrip("./")
            if m.isfile() and any(name.endswith(w.split("/", 1)[-1]) for w in want):
                mids[name] = tar.extractfile(m).read()
    rows = []
    for p in pairs:
        s, rec = p["score"], p["recording"]
        sid = "pdmx_" + Path(s["mid"]).stem
        midi = next((v for k, v in mids.items() if k.endswith(s["mid"].lstrip("./").split("/", 1)[-1])), None)
        if midi is None:
            print("no midi:", s["mid"], flush=True)
            continue
        ev = events.sanitize(score_events(midi))
        events.save_npy(data / "cover_events" / f"{sid}.npy", ev)
        mid_path = data / "cover" / f"{sid}.mid"
        events.to_pretty_midi(ev).write(str(mid_path))
        wav = data / "cover" / f"{sid}.wav"
        if not wav.exists():
            subprocess.run(["fluidsynth", "-ni", "-q", "-g", "0.8", "-r", "44100", "-F", str(wav), args.sf2,
                            str(mid_path)], check=True, capture_output=True)
        mid_path.unlink()
        err = fetch_audio(rec["id"], data / "pop", "native", 11.0, None)
        if err:
            print("download failed:", rec["id"], err, flush=True)
            continue
        rows.append((rec["id"], sid))
        print(f"ok {sid} <- {rec['id']}", flush=True)
    with (data / "pairs.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pop_id", "piano_id"])
        w.writerows(rows)
    print(f"{len(rows)} pairs -> {data / 'pairs.csv'}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("search")
    a.add_argument("candidates")
    a.add_argument("--out", required=True)
    a.add_argument("--max-pairs", type=int, default=40)
    a.add_argument("--sleep", type=float, default=3.0)
    b = sub.add_parser("prepare")
    b.add_argument("pairs")
    b.add_argument("--pdmx", required=True)
    b.add_argument("--sf2", required=True)
    b.add_argument("--data", required=True)
    args = ap.parse_args()
    search(args) if args.cmd == "search" else prepare(args)


if __name__ == "__main__":
    main()

"""Search YouTube for more piano covers of songs that already passed the rights filter.

  python scripts/find_covers.py data/meta/train_keep.jsonl --out data/meta/extra_covers.jsonl \
      --exclude pairs/pop2piano_train_dataset.csv

Input: JSON lines with id (original-song video), genie_artist/artist, genie_title/track.
Output: one JSON line per song: {"id", "query", "covers": [{"id", "title", "channel", "duration"}]}.
Use scripts/covers_to_pairs.py to turn it into a piano_ids,pop_ids CSV for pianocover.download.

Only flat search results are fetched (no per-video pages), one search every few seconds.
Title filters drop tutorials, medleys, singing and other-instrument videos; whether a video
really covers the song is checked later by the alignment step in build_dataset.
Resumable: songs already in --out are skipped.
"""
import argparse
import csv
import json
import re
import time
from pathlib import Path

from yt_dlp import YoutubeDL

PIANO = re.compile(r"piano|피아노|ピアノ|钢琴|鋼琴", re.I)
BAD = re.compile(
    r"tutorial|lesson|how to play|easy piano|sheet music only|악보 ?만|강좌|레슨|배우기|"
    r"medley|메들리|playlist|플레이리스트|모음|1 ?hour|hours|시간 연속|compilation|"
    r"reaction|vocal|노래|불러|sing|sung|cover by .*\(vocal\)|guitar|기타|violin|바이올린|cello|첼로|"
    r"orchestra|오케스트라|drum|드럼|band cover|karaoke|노래방|mr|inst\.?\b|instrumental|ai cover|"
    r"with ar|반주|accompaniment|backing|"
    r"duet|4 hands|four hands|연탄|organ|lofi|lo-fi|music box|오르골|8d|slowed|reverb",
    re.I,
)


def norm(s):
    return re.sub(r"[^0-9a-z가-힣぀-ヿ一-鿿]", "", s.lower())


def bare(s):
    return re.sub(r"\([^)]*\)|\[[^\]]*\]", "", s or "").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("songs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--exclude", nargs="*", default=[], help="pair CSVs whose piano_ids are already known")
    ap.add_argument("--per-song", type=int, default=4, help="max new covers kept per song")
    ap.add_argument("--results", type=int, default=15, help="search results inspected per song")
    ap.add_argument("--sleep", type=float, default=3.0)
    a = ap.parse_args()

    known = set()
    for p in a.exclude:
        known |= {r["piano_ids"] for r in csv.DictReader(open(p, encoding="utf8"))}
    out = Path(a.out)
    done = {json.loads(line)["id"] for line in out.open(encoding="utf8")} if out.exists() else set()
    songs = [json.loads(line) for line in open(a.songs, encoding="utf8")]
    songs = [s for s in {s["id"]: s for s in songs}.values() if s["id"] not in done]
    print(f"{len(done)} done, {len(songs)} to search", flush=True)

    ydl = YoutubeDL({"quiet": True, "no_warnings": True, "extract_flat": True, "skip_download": True})
    fails = 0
    with out.open("a", encoding="utf8") as f:
        for n, s in enumerate(songs, 1):
            artist = bare(s.get("genie_artist") or s["artist"])
            title = bare(s.get("genie_title") or s["track"])
            query = f"{artist} {title} piano cover"
            try:
                res = ydl.extract_info(f"ytsearch{a.results}:{query}", download=False)
                fails = 0
            except Exception as e:
                fails += 1
                print("search failed:", s["id"], str(e)[:120], flush=True)
                if fails >= 5:
                    raise SystemExit("too many failures in a row (rate limited?); rerun later to resume")
                time.sleep(a.sleep * 10)
                continue
            t = norm(title)
            covers = []
            for e in res.get("entries") or []:
                vid, vt = e.get("id"), e.get("title") or ""
                dur = e.get("duration") or 0
                if not vid or vid in known or vid == s["id"]:
                    continue
                if not PIANO.search(vt) or BAD.search(vt) or not (90 <= dur <= 480):
                    continue
                if t and t not in norm(vt):  # the song title must appear in the video title
                    continue
                covers.append({"id": vid, "title": vt, "channel": e.get("channel") or e.get("uploader"), "duration": dur})
                known.add(vid)
                if len(covers) >= a.per_song:
                    break
            f.write(json.dumps({"id": s["id"], "query": query, "covers": covers}, ensure_ascii=False) + "\n")
            f.flush()
            if n % 50 == 0:
                print(n, flush=True)
            time.sleep(a.sleep)


if __name__ == "__main__":
    main()

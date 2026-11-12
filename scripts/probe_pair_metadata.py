"""Fetch title + channel (YouTube oEmbed, no media) for the original-song IDs of a pair list.

  python scripts/probe_pair_metadata.py pairs/pop2piano_train_dataset.csv --out data/meta/p2p_pop.jsonl

Resumable: IDs already present in --out are skipped. One JSON object per line; dead or private
videos get an "error" field (HTTP status). oEmbed is used because yt-dlp metadata calls get
bot-blocked after a few hundred requests.
"""
import argparse
import csv
import json
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

URL = "https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={}&format=json"


def pop_ids(path):
    if path.endswith(".json"):
        return [v["original"] for v in json.load(open(path, encoding="utf8")).values()]
    return [r["pop_ids"] for r in csv.DictReader(open(path, encoding="utf8"))]


def probe(vid):
    try:
        with urllib.request.urlopen(URL.format(vid), timeout=20) as r:
            d = json.load(r)
        return {"id": vid, "title": d.get("title"), "channel": d.get("author_name")}
    except urllib.error.HTTPError as e:  # 401/403/404: private, embedding disabled, or deleted
        return {"id": vid, "error": e.code}
    except Exception as e:  # network trouble: leave the ID out so a rerun retries it
        print("retry later:", vid, e, flush=True)
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(line)["id"] for line in out.open(encoding="utf8")} if out.exists() else set()
    todo = [v for v in dict.fromkeys(pop_ids(a.pairs)) if v not in done]
    print(f"{len(done)} done, {len(todo)} to probe", flush=True)
    with out.open("a", encoding="utf8") as f, ThreadPoolExecutor(a.workers) as ex:
        for row in ex.map(probe, todo):
            if row:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()


if __name__ == "__main__":
    main()

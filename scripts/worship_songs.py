"""Build a song list of Korean worship teams from genie.co.kr, then match each song to a YouTube video.

  python scripts/worship_songs.py list  --out data/meta/worship_songs.tsv
  python scripts/genie_rights_lookup.py data/meta/worship_songs.tsv --out data/meta/worship_rights.jsonl \
      --asian-writers data/meta/asian_writers.txt
  python scripts/worship_songs.py match --rights data/meta/worship_rights.jsonl --out data/meta/worship_keep.jsonl

`list` writes the TSV format genie_rights_lookup.py reads (key, artist, title, country, kind);
keys are "g<genie song id>" until a YouTube video is matched. `match` takes songs with verdict
"keep", finds the original recording on YouTube (flat search, the title must contain the song
title and must not be a cover, live or piano video) and writes JSON lines with "id" = that
video, ready for find_covers.py. Many Korean worship songs are translations of English songs;
the rights lookup excludes them through their foreign writer credits.
"""
import argparse
import html
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

TEAMS = {  # Genie search term -> substring the Genie artist name must contain
    "마커스워십": "마커스", "어노인팅": "어노인팅", "제이어스": "제이어스", "아이자야씩스티원": "아이자야",
    "위러브": "위러브", "예수전도단": "예수전도단", "디사이플스": "디사이플스", "시와그림": "시와그림",
    "피아워십": "피아", "레위지파": "레위지파", "캠퍼스워십": "캠퍼스", "옹기장이": "옹기장이",
    "다윗의장막": "다윗의장막", "소진영": "소진영", "심종호": "심종호", "팀룩워십": "팀룩",
}
NOT_ORIGINAL = re.compile(r"piano|피아노|cover|커버|live|라이브|mr\b|inst|반주|악보|가사만|1시간|연속|모음|playlist", re.I)


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf8", "ignore")


def clean(s):
    s = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", s or ""))).strip()
    return re.sub(r"^(TITLE\s*)?(HOT)?\s*-->\s*", "", s)


def norm(s):
    return re.sub(r"[^0-9a-z가-힣]", "", s.lower())


def cmd_list(a):
    seen, rows = set(), []
    for term, key in TEAMS.items():
        n = 0
        for page in range(1, a.pages + 1):
            h = fetch("https://www.genie.co.kr/search/searchSong?" + urllib.parse.urlencode({"query": term, "page": page}))
            time.sleep(1)
            for sid, row in re.findall(r'songid="(\d+)"(.*?)</tr>', h, re.S):
                t = re.search(r'class="title ellipsis"[^>]*>(.*?)</a>', row, re.S)
                ar = re.search(r'class="artist ellipsis"[^>]*>(.*?)</a>', row, re.S)
                title, artist = clean(t.group(1) if t else ""), clean(ar.group(1) if ar else "")
                title = re.sub(r"\s*\((인도|Inst|MR)[^)]*\)", "", title).strip()
                if key not in artist or not title or norm(title) in seen or re.search(r"\b(inst|mr)\b", title, re.I):
                    continue
                seen.add(norm(title))
                rows.append(f"g{sid}\t{artist}\t{title}\tKR\tworship")
                n += 1
        print(term, n, flush=True)
    Path(a.out).write_text("\n".join(rows) + "\n", encoding="utf8")
    print(len(rows), "songs", flush=True)


def cmd_match(a):
    from yt_dlp import YoutubeDL
    ydl = YoutubeDL({"quiet": True, "no_warnings": True, "extract_flat": True, "skip_download": True})
    keep = [json.loads(line) for line in open(a.rights, encoding="utf8")]
    keep = [x for x in keep if x["verdict"] == "keep"]
    out = Path(a.out)
    done = {json.loads(line)["genie_key"] for line in out.open(encoding="utf8")} if out.exists() else set()
    with out.open("a", encoding="utf8") as f:
        for x in keep:
            if x["id"] in done:
                continue
            artist = re.sub(r"\([^)]*\)", "", x.get("genie_artist") or x["artist"]).strip()
            title = re.sub(r"\([^)]*\)", "", x.get("genie_title") or x["track"]).strip()
            try:
                res = ydl.extract_info(f"ytsearch8:{artist} {title}", download=False)
            except Exception as e:
                print("search failed:", x["id"], str(e)[:100], flush=True)
                time.sleep(30)
                continue
            vid = next((e["id"] for e in res.get("entries") or []
                        if norm(title) in norm(e.get("title") or "") and not NOT_ORIGINAL.search(e.get("title") or "")
                        and 90 <= (e.get("duration") or 0) <= 600), None)
            if vid:
                f.write(json.dumps({**x, "genie_key": x["id"], "id": vid}, ensure_ascii=False) + "\n")
                f.flush()
            else:
                print("no original found:", artist, "-", title, flush=True)
            time.sleep(a.sleep)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("list")
    p.add_argument("--out", required=True)
    p.add_argument("--pages", type=int, default=3)
    p = sub.add_parser("match")
    p.add_argument("--rights", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--sleep", type=float, default=3.0)
    a = ap.parse_args()
    {"list": cmd_list, "match": cmd_match}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())

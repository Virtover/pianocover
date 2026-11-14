"""Fetch song lyrics from melon.com and flag demonic / occult / dark-supernatural content.

  python scripts/lyrics_scan.py data/meta/train_keep.jsonl --out data/meta/lyrics.jsonl

Input JSON lines need id, artist/genie_artist, track/genie_title. Output per song: melon id,
lyrics, and keyword hits. Keyword hits are only candidates: review them (in context) before
dropping songs. Resumable.
"""
import argparse
import html
import json
import re
import sys
import urllib.parse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from melon_credits import fetch, norm  # noqa: E402

TERMS = re.compile(
    r"악마|마귀|사탄|악령|귀신|지옥|저주|마녀|주술|무당|부적|저승|영혼을 ?팔|루시퍼|데빌|666|"
    r"구미호|도깨비|유령|좀비|뱀파이어|흡혈|강령|빙의|접신|사령|요괴|"
    r"\bdevils?\b|\bdemons?\b|demonic|satan|lucifer|\bhell\b|witch|voodoo|occult|curse|possess|"
    r"\bghosts?\b|zombie|vampire|haunt|spell|séance|summon|underworld|hades|beelzebub|exorcis",
    re.I,
)


def lyrics(artist, track):
    q = f"{re.sub(r'[(].*', '', artist).strip()} {re.sub(r'[(].*', '', track).strip()}"
    page = fetch("https://www.melon.com/search/song/index.htm?" + urllib.parse.urlencode({"q": q}))
    for sid in list(dict.fromkeys(re.findall(r"goSongDetail\('(\d+)'\)", page)))[:2]:
        d = fetch("https://www.melon.com/song/detail.htm?songId=" + sid)
        title = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", (re.search(r'<div class="song_name">(.*?)</div>', d, re.S) or [0, ""])[1])))
        if norm(re.sub(r"[(].*", "", track)) not in norm(title):
            continue
        m = re.search(r'<div class="lyric"[^>]*id="d_video_summary"[^>]*>(.*?)</div>', d, re.S)
        if m:
            text = html.unescape(re.sub(r"<br\s*/?>", "\n", m.group(1)))
            return sid, re.sub(r"<[^>]+>|<!--.*?-->", "", text, flags=re.S).strip()
        return sid, ""
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("songs", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    done = {json.loads(line)["id"] for line in out.open(encoding="utf8")} if out.exists() else set()
    rows = [json.loads(line) for p in a.songs for line in open(p, encoding="utf8")]
    rows = [x for x in {x["id"]: x for x in rows}.values() if x["id"] not in done]
    print(f"{len(done)} done, {len(rows)} to fetch", flush=True)
    with out.open("a", encoding="utf8") as f:
        for n, x in enumerate(rows, 1):
            artist, track = x.get("genie_artist") or x["artist"], x.get("genie_title") or x["track"]
            try:
                sid, lyr = lyrics(artist, track)
            except RuntimeError as e:
                print("skipped:", x["id"], e, flush=True)
                continue
            hits = sorted({m.group(0).lower() for m in TERMS.finditer((lyr or "") + " " + track)})
            f.write(json.dumps({"id": x["id"], "artist": artist, "track": track, "melon_id": sid,
                                "lyrics": lyr, "hits": hits}, ensure_ascii=False) + "\n")
            f.flush()
            if n % 100 == 0:
                print(n, flush=True)


if __name__ == "__main__":
    main()

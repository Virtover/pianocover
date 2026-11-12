"""Look up songwriters and labels of Korean songs on genie.co.kr and apply the rights filter.

  python scripts/genie_rights_lookup.py data/meta/songs.tsv --out data/meta/rights_kr.jsonl --country KR

MusicBrainz has few writer credits for Korean songs; Genie lists lyricist (작사가), composer
(작곡가), distributor (발매사) and agency (기획사). Input TSV: youtube_id, artist, track, country,
kind. Verdicts (policy 2026-11-12):
  keep        every writer name is in Hangul (or listed in --asian-writers) and no label is a major
  exclude     released only by a major, or a writer with a Latin name not known to be Asian
              (European/US writers: their societies may have opted out)
  major_dist  writers pass, but a major is the distributor or agency alongside others
  unknown     song not found, neither artist nor title matches, or no writer credits
Latin writer names are collected so they can be classified once and passed back via
--asian-writers (text file, one name per line). Resumable; about 3 s per song.
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

sys.path.insert(0, str(Path(__file__).parent))
from mb_rights_lookup import MAJOR  # noqa: E402

BASE = "https://www.genie.co.kr/"
MAJOR_KO = re.compile(r"소니|유니버설|워너")
HANGUL = re.compile(r"[가-힣]")
_last = [0.0]


def fetch(path, **params):
    url = BASE + path + "?" + urllib.parse.urlencode(params)
    for attempt in range(4):
        time.sleep(max(0.0, 1.0 - (time.time() - _last[0])))
        _last[0] = time.time()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read().decode("utf8", "ignore")
        except Exception:
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("genie unreachable: " + url)


def text(s):
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", s))).strip()


def field(page, label):
    """Names in the <li> whose attribute image has alt=label."""
    m = re.search(r'alt="%s".*?<span class="value">(.*?)</li>' % label, page, re.S)
    if not m:
        return []
    names = [text(x) for x in re.findall(r"<a [^>]*>(.*?)</a>", m.group(1), re.S)] or [text(m.group(1))]
    return [n for n in dict.fromkeys(names) if n]


def norm(s):
    return re.sub(r"[^0-9a-z가-힣぀-ヿ一-鿿]", "", s.lower())


NOISE = re.compile(
    r"\[[^\]]*\]|color coded|lyrics?|han/rom/eng|eng/rom/han|official|audio|m/v|mv|eng sub|kor sub|"
    r"가사|해석|번역|자막|한글|한국어|발음|뮤비|듣기",
    re.I,
)


def clean_title(title):
    """YouTube fan-upload title -> search query (keeps the original-language song title)."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", NOISE.sub(" ", title))).strip()


def lookup(artist, track, asian, yt_title=""):
    sids = []
    cleaned = clean_title(yt_title)
    hangul = " ".join(w for w in cleaned.split() if HANGUL.search(w))  # extracted titles are often translations
    for q in (f"{artist} {track}", hangul, cleaned):
        if q:
            sids = list(dict.fromkeys(re.findall(r'songid="(\d+)"', fetch("search/searchSong", query=q))))[:3]
        if sids:
            break
    if not sids:
        return {"verdict": "unknown", "why": "not found"}
    a, t, yt, best = norm(artist), norm(track), norm(yt_title), None
    for sid in sids:  # first hit by the right artist that has writer credits (skips remixes, other versions)
        song = fetch("detail/songInfo", xgnm=sid)
        g_artist = text((re.search(r"fnGoMore\('artistInfo'[^>]*>(.*?)</a>", song, re.S) or [0, ""])[1])
        writers = list(dict.fromkeys(field(song, "작사가") + field(song, "작곡가")))
        ga = norm(g_artist)
        gt = norm(text((re.search(r'<h2 class="name">(.*?)</h2>', song, re.S) or [0, ""])[1]))
        # Genie often shows only the Hangul artist name, so a title match also counts
        match = bool(a and ga and (a in ga or ga in a)) or bool(t and gt and (t in gt or gt in t))
        # ...and so does finding Genie's artist or title (without bracketed parts) in the YouTube title
        bare = [norm(re.sub(r"\([^)]*\)", "", x)) for x in (g_artist, text((re.search(r'<h2 class="name">(.*?)</h2>', song, re.S) or [0, ""])[1]))]
        match = match or any(len(b) >= 2 and b in yt for b in bare)
        if best is None or (match and writers):
            best = (sid, song, g_artist, writers, match)
        if match and writers:
            break
    sid, song, g_artist, writers, match = best
    g_title = text((re.search(r'<h2 class="name">(.*?)</h2>', song, re.S) or [0, ""])[1])
    alb = re.search(r"fnGoMore\('albumInfo','(\d+)", song)
    labels = []
    if alb:
        apage = fetch("detail/albumInfo", axnm=alb.group(1))
        labels = list(dict.fromkeys(field(apage, "발매사") + field(apage, "기획사")))
    out = {"genie_id": sid, "genie_title": g_title, "genie_artist": g_artist, "writers": writers, "labels": labels}
    majors = [x for x in labels if MAJOR.search(x) or MAJOR_KO.search(x)]
    latin = [w for w in writers if not HANGUL.search(w) and w.lower() not in asian]
    if not match:
        out.update(verdict="unknown", why="no artist or title match")
    elif majors and len(majors) == len(labels):
        out.update(verdict="exclude", why="major label: " + ", ".join(majors[:3]))
    elif not writers:
        out.update(verdict="unknown", why="no writer credits")
    elif latin:
        out.update(verdict="exclude", why="non-Korean writer name: " + ", ".join(latin[:3]), latin_writers=latin)
    elif majors:
        out.update(verdict="major_dist", why="major as distributor/agency: " + ", ".join(majors[:3]))
    else:
        out.update(verdict="keep", why="")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("songs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--country", default="KR")
    ap.add_argument("--titles", help="jsonl from probe_pair_metadata.py; raw titles are a fallback search query")
    ap.add_argument("--asian-writers", help="text file of Latin-script writer names known to be KR/JP/CN/TW/HK")
    a = ap.parse_args()
    asian = set()
    if a.asian_writers:
        asian = {x.strip().lower() for x in open(a.asian_writers, encoding="utf8") if x.strip()}
    titles = {}
    if a.titles:
        titles = {x["id"]: x.get("title") or "" for x in map(json.loads, open(a.titles, encoding="utf8"))}
    out = Path(a.out)
    done = {json.loads(line)["id"] for line in out.open(encoding="utf8")} if out.exists() else set()
    rows = [line.rstrip("\n").split("\t") for line in open(a.songs, encoding="utf8")]
    rows = [r for r in rows if len(r) >= 5 and r[3] == a.country and r[0] not in done]
    print(f"{len(done)} done, {len(rows)} to look up", flush=True)
    with out.open("a", encoding="utf8") as f:
        for n, (vid, artist, track, country, kind, *_) in enumerate(rows, 1):
            res = lookup(artist, track, asian, titles.get(vid, ""))
            f.write(json.dumps({"id": vid, "artist": artist, "track": track, **res}, ensure_ascii=False) + "\n")
            f.flush()
            if n % 100 == 0:
                print(n, flush=True)


if __name__ == "__main__":
    main()

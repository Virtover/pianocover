"""Fill missing writer credits from melon.com and re-apply the rights verdict.

  python scripts/melon_credits.py data/meta/worship_rights.jsonl --out data/meta/worship_rights2.jsonl \
      --asian-writers data/meta/asian_writers.txt data/meta/pd_writers.txt

Rows whose Genie verdict was "unknown" (no writer credits / not found) are looked up on Melon
(search, then the song page's 작사/작곡 credits); every other row is copied unchanged.
--asian-writers files list Latin-script names that pass: writers known to be Korean, and
writers who died before 1956 (public-domain hymns). Same verdict rules as genie_rights_lookup.py.
"""
import argparse
import html
import json
import re
import time
import urllib.parse
import urllib.request

HANGUL = re.compile(r"[가-힣]")
_last = [0.0]


def fetch(url):
    for attempt in range(4):
        time.sleep(max(0.0, 1.0 - (time.time() - _last[0])))
        _last[0] = time.time()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read().decode("utf8", "ignore")
        except Exception:
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("melon unreachable: " + url)


def text(s):
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", s))).strip()


def norm(s):
    return re.sub(r"[^0-9a-z가-힣]", "", s.lower())


def credits(artist, track):
    q = f"{re.sub(r'[(].*', '', artist).strip()} {re.sub(r'[(].*', '', track).strip()}"
    page = fetch("https://www.melon.com/search/song/index.htm?" + urllib.parse.urlencode({"q": q}))
    for sid in list(dict.fromkeys(re.findall(r"goSongDetail\('(\d+)'\)", page)))[:2]:
        d = fetch("https://www.melon.com/song/detail.htm?songId=" + sid)
        title = text((re.search(r'<div class="song_name">(.*?)</div>', d, re.S) or [0, ""])[1]).replace("곡명", "").strip()
        if norm(re.sub(r"[(].*", "", track)) not in norm(title):
            continue
        writers = []
        for e in re.findall(r'<div class="entry">(.*?)</div>\s*</div>', d, re.S):
            t = text(e)
            m = re.match(r"(.+?)\s+(작사|작곡)$", t)
            if m:
                writers.append(m.group(1))
        return sid, title, list(dict.fromkeys(writers))
    return None, None, []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rights")
    ap.add_argument("--out", required=True)
    ap.add_argument("--asian-writers", nargs="*", default=[])
    a = ap.parse_args()
    ok = {x.strip().lower() for p in a.asian_writers for x in open(p, encoding="utf8") if x.strip()}
    rows = [json.loads(line) for line in open(a.rights, encoding="utf8")]
    with open(a.out, "w", encoding="utf8", newline="\n") as f:
        for n, x in enumerate(rows, 1):
            if x["verdict"] == "exclude" and x.get("latin_writers"):  # names may now be on an allow list
                if all(w.lower() in ok for w in x["latin_writers"]):
                    x.update(verdict="keep", why="writer names on allow list")
            elif x["verdict"] == "unknown":
                try:
                    sid, title, writers = credits(x.get("genie_artist") or x["artist"], x.get("genie_title") or x["track"])
                except RuntimeError as e:
                    print("skipped:", x["id"], e, flush=True)
                    writers, sid, title = [], None, None
                if writers:
                    x.update(melon_id=sid, melon_title=title, writers=writers)
                    latin = [w for w in writers if not HANGUL.search(w) and w.lower() not in ok]
                    majors = [l for l in x.get("labels", []) if re.search(r"소니|유니버설|워너|sony|universal|warner", l, re.I)]
                    if majors and len(majors) == len(x.get("labels", [])):
                        x.update(verdict="exclude", why="major label: " + ", ".join(majors))
                    elif latin:
                        x.update(verdict="exclude", why="non-Korean writer name: " + ", ".join(latin[:3]), latin_writers=latin)
                    else:
                        x.update(verdict="keep", why="credits from melon")
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
            if n % 100 == 0:
                print(n, flush=True)


if __name__ == "__main__":
    main()

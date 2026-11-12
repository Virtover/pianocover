"""Look up songwriters and labels on MusicBrainz and apply the rights filter (policy 2026-11-12).

  python scripts/mb_rights_lookup.py data/meta/songs.tsv --out data/meta/rights.jsonl

Input TSV: youtube_id, artist, track, country, kind. Output: one JSON per line with
writers (name, country), labels and a verdict:
  keep     every writer is from KR/JP/CN/TW/HK and no label belongs to Sony/Universal/Warner
  exclude  released only on major labels, or a writer from elsewhere (their societies may have opted out)
  major_dist  writers pass, but some releases (usually foreign editions) are on a major label: owner decides
  unknown  no recording, no writer credits, or a writer whose country is not recorded
Resumable. MusicBrainz allows 1 request per second, so expect about 4 s per song.
"""
import argparse
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://musicbrainz.org/ws/2/"
UA = {"User-Agent": "pair-rights-lookup/0.1 (research script)"}
OK_COUNTRIES = {"KR", "JP", "CN", "TW", "HK"}
WRITER_RELS = {"composer", "lyricist", "writer"}
MAJOR = re.compile(
    r"sony|columbia|\brca\b|\bepic\b|arista|ariola|legacy|sme records|ki/oon|sacra|aniplex|"
    r"universal|\bumg\b|republic|interscope|def jam|capitol|\bemi\b|polydor|island|mercury|virgin|"
    r"geffen|decca|verve|motown|"
    r"warner|atlantic|elektra|parlophone|rhino|\bwea\b|asylum|reprise|\bsire\b|unborde",
    re.I,
)
_last = [0.0]
_artists = {}


def get(path, **params):
    url = API + path + "?" + urllib.parse.urlencode({**params, "fmt": "json"})
    for attempt in range(5):
        time.sleep(max(0.0, 1.05 - (time.time() - _last[0])))
        _last[0] = time.time()
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return {}
            time.sleep(5 * (attempt + 1))  # 503 = rate limited
        except Exception:
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("MusicBrainz unreachable: " + url)


def script_country(name):
    """Guess a country from the script a name is written in (None for Latin)."""
    if re.search(r"[가-힣]", name):
        return "KR"
    if re.search(r"[぀-ヿ]", name):
        return "JP"
    return None


def artist_country(aid, name):
    if aid not in _artists:
        a = get("artist/" + aid)
        c = a.get("country") or next(
            (x for x in ((a.get(k) or {}).get("iso-3166-1-codes") or [None] for k in ("area", "begin-area")) if x[0]),
            [None],
        )[0]
        _artists[aid] = c or script_country(name + " " + (a.get("sort-name") or ""))
    return _artists[aid]


def esc(s):
    return re.sub(r'([+\-&|!(){}\[\]^"~*?:\\/])', r"\\\1", s)


_arids = {}


def resolve_artist(artist):
    """MusicBrainz artist ID for a name (artist search also matches aliases such as Hangul names)."""
    if artist not in _arids:
        hits = get("artist", query=f'artist:"{esc(artist)}"', limit=1).get("artists", [])
        _arids[artist] = hits[0]["id"] if hits and hits[0].get("score", 0) >= 90 else None
    return _arids[artist]


def find_recordings(artist, track):
    arid = resolve_artist(artist)
    who = f"arid:{arid}" if arid else f'artist:"{esc(artist)}"'
    for title in (f'"{esc(track)}"', f"({esc(track)})"):
        hits = get("recording", query=f"recording:{title} AND {who}", limit=10).get("recordings", [])
        hits = [h for h in hits if h.get("score", 0) >= 80]
        if hits:
            return hits
    return []


def lookup(artist, track):
    hits = find_recordings(artist, track)
    if not hits:
        return {"verdict": "unknown", "why": "no recording"}
    writers, rec = {}, hits[0]
    for h in hits[:5]:
        full = get("recording/" + h["id"], inc="work-rels+work-level-rels+artist-rels")
        for rel in full.get("relations", []):
            for wr in (rel.get("work") or {}).get("relations", []):
                if wr.get("type") in WRITER_RELS and "artist" in wr:
                    writers[wr["artist"]["id"]] = wr["artist"]["name"]
        if writers:
            rec = h
            break
    rels = get("release", recording=rec["id"], inc="labels", limit=25).get("releases", [])
    labels = sorted({li["label"]["name"] for r in rels for li in r.get("label-info", []) if li.get("label")})
    labels = [x for x in labels if x != "[no label]"]
    wl = [{"name": n, "country": artist_country(i, n)} for i, n in writers.items()]
    out = {"mb_recording": rec["id"], "mb_title": rec.get("title"), "writers": wl, "labels": labels}
    majors = [x for x in labels if MAJOR.search(x)]
    foreign = [w["name"] for w in wl if w["country"] and w["country"] not in OK_COUNTRIES]
    if majors and len(majors) == len(labels):
        out.update(verdict="exclude", why="major label: " + ", ".join(majors[:3]))
    elif foreign:
        out.update(verdict="exclude", why="writer outside KR/JP/CN/TW/HK: " + ", ".join(foreign[:3]))
    elif not wl:
        out.update(verdict="unknown", why="no writer credits")
    elif any(w["country"] is None for w in wl):
        out.update(verdict="unknown", why="writer country not recorded")
    elif majors:
        out.update(verdict="major_dist", why="own label plus major-label releases: " + ", ".join(majors[:3]))
    else:
        out.update(verdict="keep", why="")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("songs")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    done = {json.loads(line)["id"] for line in out.open(encoding="utf8")} if out.exists() else set()
    rows = [line.rstrip("\n").split("\t") for line in open(a.songs, encoding="utf8")]
    rows = [r for r in rows if len(r) >= 5 and r[0] not in done]
    print(f"{len(done)} done, {len(rows)} to look up", flush=True)
    with out.open("a", encoding="utf8") as f:
        for n, (vid, artist, track, country, kind, *_) in enumerate(rows, 1):
            res = lookup(artist, track)
            f.write(json.dumps({"id": vid, "artist": artist, "track": track, "country": country, **res}, ensure_ascii=False) + "\n")
            f.flush()
            if n % 100 == 0:
                print(n, flush=True)


if __name__ == "__main__":
    main()

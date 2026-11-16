"""Pilot: find organ originals (public-domain organ works, hymns) and their piano versions on YouTube.

  python scripts/find_organ_pairs.py --out data/meta/organ_candidates.jsonl

For each piece: up to 2 organ recordings (title must name the piece and say organ/Orgel/오르간)
and up to 4 piano versions (title names the piece and says piano, and no organ, harpsichord,
orchestra, lesson...). Only flat search results are fetched, one search every few seconds.
Output: one JSON line per piece {"piece", "originals": [...], "covers": [...]}; review it, then
turn it into a piano_ids,pop_ids CSV for pianocover.download.
"""
import argparse
import json
import re
import time

from yt_dlp import YoutubeDL

# (name, search terms, regex that a title must match to be this piece)
PIECES = [
    ("Bach - Toccata and Fugue in D minor BWV 565", "Bach Toccata and Fugue in D minor BWV 565", r"565"),
    ("Bach - Ich ruf zu dir BWV 639", "Bach Ich ruf zu dir Herr Jesu Christ BWV 639", r"639"),
    ("Bach - Nun komm der Heiden Heiland BWV 659", "Bach Nun komm der Heiden Heiland BWV 659", r"659"),
    ("Bach - Wachet auf BWV 645", "Bach Wachet auf ruft uns die Stimme BWV 645", r"645"),
    ("Bach - Jesu bleibet meine Freude BWV 147", "Bach Jesu Joy of Man's Desiring BWV 147", r"147|jesu,? joy|jesu bleibet"),
    ("Bach - Little Fugue in G minor BWV 578", "Bach Little Fugue G minor BWV 578", r"578"),
    ("Bach - Prelude and Fugue in D major BWV 532", "Bach Prelude and Fugue D major BWV 532", r"532"),
    ("Bach - Prelude and Fugue in A minor BWV 543", "Bach Prelude and Fugue A minor BWV 543", r"543"),
    ("Bach - Fantasia and Fugue in G minor BWV 542", "Bach Fantasia and Fugue G minor BWV 542", r"542"),
    ("Widor - Toccata (Symphony No. 5)", "Widor Toccata Symphony 5", r"widor"),
    ("Boellmann - Toccata (Suite gothique)", "Boellmann Toccata Suite gothique", r"bo[eë]?llmann"),
    ("Vierne - Carillon de Westminster", "Vierne Carillon de Westminster", r"westminster"),
    ("Hymn - Holy, Holy, Holy (Nicaea)", "Holy Holy Holy hymn", r"holy,? holy,? holy"),
    ("Hymn - A Mighty Fortress / Ein feste Burg", "A Mighty Fortress Is Our God hymn", r"mighty fortress|feste burg"),
    ("Hymn - Now Thank We All Our God / Nun danket", "Now Thank We All Our God hymn", r"now thank we|nun danket"),
    ("Hymn - Amazing Grace", "Amazing Grace hymn", r"amazing grace"),
]
# --partial: public-domain hymns sung with organ (choir / congregation + organ) as originals
PARTIAL = [
    ("Hymn - Holy, Holy, Holy (Nicaea)", "Holy Holy Holy Lord God Almighty", r"holy,? holy,? holy"),
    ("Hymn - A Mighty Fortress / Ein feste Burg", "A Mighty Fortress Is Our God", r"mighty fortress|feste burg"),
    ("Hymn - Now Thank We All Our God", "Now Thank We All Our God", r"now thank we"),
    ("Hymn - Praise to the Lord, the Almighty", "Praise to the Lord the Almighty", r"praise to the lord,? the almighty"),
    ("Hymn - Joyful, Joyful, We Adore Thee", "Joyful Joyful We Adore Thee", r"joyful,? joyful"),
    ("Hymn - Crown Him with Many Crowns", "Crown Him with Many Crowns", r"crown him with many crowns"),
    ("Hymn - It Is Well with My Soul", "It Is Well with My Soul", r"it is well"),
    ("Hymn - O God, Our Help in Ages Past", "O God Our Help in Ages Past", r"o god,? our help"),
    ("Hymn - All Creatures of Our God and King", "All Creatures of Our God and King", r"all creatures of our god"),
    ("Hymn - Jesus Christ Is Risen Today", "Jesus Christ Is Risen Today", r"jesus christ is risen today"),
    ("Hymn - O Come, All Ye Faithful", "O Come All Ye Faithful", r"o come,? all ye faithful"),
    ("Hymn - Hark! The Herald Angels Sing", "Hark the Herald Angels Sing", r"hark,? the herald"),
]
WITH_VOICES = re.compile(r"choir|chorus|congregation|congregational|sing|sung|hymn sing|cathedral|chapel|abbey|lyrics", re.I)
ORGAN = re.compile(r"organ|orgel|orgue|organo|오르간|オルガン", re.I)
PIANO = re.compile(r"piano|klavier|피아노|ピアノ|钢琴|鋼琴", re.I)
NOT_COVER = re.compile(
    r"organ|orgel|orgue|organo|오르간|harpsichord|cembalo|orchestra|orchester|choir|chor\b|vocal|sing|"
    r"guitar|violin|cello|trumpet|brass|tutorial|lesson|how to play|easy|synthesia|midi|medley|"
    r"compilation|playlist|hour|karaoke|lofi|lo-fi|music box|duet|4 hands|four hands|reaction", re.I)
NOT_ORIGINAL = re.compile(r"piano|klavier|tutorial|lesson|synthesia|midi|medley|compilation|playlist|"
                          r"hour|reaction|guitar|metal|remix|8d|slowed", re.I)


def search(ydl, query, n):
    return (ydl.extract_info(f"ytsearch{n}:{query}", download=False) or {}).get("entries") or []


def pick(entries, piece_re, want, reject, need, seen, k):
    out = []
    for e in entries:
        vid, t, dur = e.get("id"), e.get("title") or "", e.get("duration") or 0
        if not vid or vid in seen or not (60 <= dur <= 600):
            continue
        if not re.search(piece_re, t, re.I) or not want.search(t) or reject.search(t):
            continue
        out.append({"id": vid, "title": t, "channel": e.get("channel") or e.get("uploader"), "duration": dur})
        seen.add(vid)
        if len(out) >= k:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--sleep", type=float, default=3.0)
    ap.add_argument("--partial", action="store_true",
                    help="hymns sung with organ (choir/congregation + organ) instead of organ solo")
    a = ap.parse_args()
    ydl = YoutubeDL({"quiet": True, "no_warnings": True, "extract_flat": True, "skip_download": True})
    seen = set()
    with open(a.out, "w", encoding="utf8") as f:
        for name, q, piece_re in (PARTIAL if a.partial else PIECES):
            if a.partial:
                entries = search(ydl, f"{q} choir organ", 25)
                entries = [e for e in entries if WITH_VOICES.search(e.get("title") or "")]
                orig = pick(entries, piece_re, ORGAN, NOT_ORIGINAL, None, seen, 2)
            else:
                orig = pick(search(ydl, f"{q} organ", 20), piece_re, ORGAN, NOT_ORIGINAL, None, seen, 2)
            time.sleep(a.sleep)
            cov = pick(search(ydl, f"{q} piano", 25), piece_re, PIANO, NOT_COVER, None, seen, 4)
            time.sleep(a.sleep)
            f.write(json.dumps({"piece": name, "originals": orig, "covers": cov}, ensure_ascii=False) + "\n")
            f.flush()
            print(f"{name}: {len(orig)} organ, {len(cov)} piano", flush=True)


if __name__ == "__main__":
    main()

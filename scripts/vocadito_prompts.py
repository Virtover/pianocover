"""SoulX-Singer voice prompts from vocadito (CC BY 4.0, Zenodo 5578807): one per English singer.

A prompt is a short sung phrase (audio + note/phoneme metadata). vocadito has note annotations
(annotator 1) and lyrics one line per phrase, but no word timing, so words are laid onto the
notes of their phrase: phrases are note runs split at gaps, line k goes to phrase k, and the
syllables are spread evenly over the phrase's notes (extra notes become slurs).

Output: <out>/<name>.wav (24 kHz mono), <out>/<name>.json (SoulX metadata list),
<out>/prompts.json {name: median MIDI pitch}.

  python scripts/vocadito_prompts.py --src /data/tier1/vocadito/x --out /data/ext/vocadito_prompts
"""
import argparse
import csv
import json
import re
import subprocess
from pathlib import Path

import numpy as np

SR = 24000


def split_lines(notes, syls):
    """Split the note list into len(syls) consecutive phrases (DP): note count close to the line's
    syllable count, boundaries at long gaps. Returns [(phrase notes, cost)] or None."""
    n, L = len(notes), len(syls)
    gap = [9.0] + [notes[i][0] - notes[i - 1][1] for i in range(1, n)]

    def cost(i, j, k):  # notes[i:j] sing line k
        c = j - i - syls[k]
        return (abs(c) if c >= 0 else 4 * -c) - 3 * min(gap[i], 1.0)

    INF = float("inf")
    best = [[INF] * (n + 1) for _ in range(L + 1)]
    back = [[0] * (n + 1) for _ in range(L + 1)]
    best[0][0] = 0.0
    for k in range(L):
        for j in range(1, n + 1):
            for i in range(max(0, j - 3 * syls[k] - 8), j):
                if best[k][i] < INF:
                    v = best[k][i] + cost(i, j, k)
                    if v < best[k + 1][j]:
                        best[k + 1][j], back[k + 1][j] = v, i
    if best[L][n] == INF:
        return None
    out, j = [], n
    for k in range(L, 0, -1):
        i = back[k][j]
        out.append((notes[i:j], cost(i, j, k - 1) + 3 * min(gap[i], 1.0), gap[i], gap[j] if j < n else 9.0))
        j = i
    return out[::-1]


def words_of(line, g2p):
    ws = []
    for w in re.sub(r"[’‘`]", "'", line).split():
        w = re.sub(r"[^A-Za-z']", "", w).strip("'")
        if not w:
            continue
        ph = [p for p in g2p(w.lower()) if re.fullmatch(r"[A-Z]+[0-2]?", p)]
        syl = sum(p[-1].isdigit() for p in ph)
        if ph and syl:
            ws.append((w, ph, syl))
    return ws


def meta_for(ph, ws, t0, t1):
    """SoulX metadata for one phrase; times relative to t0 (the cut start)."""
    syls = [(w, k) for w, _, s in ws for k in range(s)]
    n, s = len(ph), len(syls)
    owner = [int(j * s / n) for j in range(n)]  # note j sings syllable owner[j]
    dur, text, phon, pitch, ntype = [], [], [], [], []

    def rest(d):
        if d > 0.005:
            dur.append(d); text.append("<SP>"); phon.append("<SP>"); pitch.append(0); ntype.append(1)

    widx = {}
    i = 0
    for wi, (w, p, sy) in enumerate(ws):
        for k in range(sy):
            widx[i] = wi
            i += 1
    rest(ph[0][0] - t0)
    for j, (a, b, m) in enumerate(ph):
        if j:
            rest(a - ph[j - 1][1])
        wi = widx[owner[j]]
        w, p, _ = ws[wi]
        first = j == 0 or widx[owner[j - 1]] != wi
        dur.append(b - a); text.append(w); phon.append("en_" + "-".join(p)); pitch.append(int(m))
        ntype.append(2 if first else 3)
    rest(t1 - ph[-1][1])
    return {"index": "prompt", "language": "English", "time": [0, int(sum(dur) * 1000)],
            "duration": " ".join(f"{d:.3f}" for d in dur), "text": " ".join(text), "phoneme": " ".join(phon),
            "note_pitch": " ".join(map(str, pitch)), "note_type": " ".join(map(str, ntype))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/data/tier1/vocadito/x")
    ap.add_argument("--out", default="/data/ext/vocadito_prompts")
    ap.add_argument("--min-s", type=float, default=3.0)
    ap.add_argument("--max-s", type=float, default=9.0)
    args = ap.parse_args()
    from g2p_en import G2p

    g2p = G2p()
    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(src / "vocadito_metadata.csv", encoding="utf8")))
    index, done = {}, set()
    for r in rows:
        if r["language"] != "English" or r["singer_id"] in done:
            continue
        tid = r["track_id"]
        notes = []
        for a, hz, d in csv.reader(open(src / f"Annotations/Notes/vocadito_{tid}_notesA1.csv")):
            a, hz, d = float(a), float(hz), float(d)
            if hz > 0 and d >= 0.05:
                notes.append([a, a + d, round(69 + 12 * np.log2(hz / 440))])
        lines = [x.strip() for x in open(src / f"Annotations/Lyrics/vocadito_{tid}_lyrics.txt", encoding="utf8")
                 if x.strip()]
        wss = [words_of(x, g2p) for x in lines]
        wss = [w for w in wss if w]
        split = split_lines(notes, [sum(w[2] for w in ws) for ws in wss])
        if split is None:
            print(tid, r["singer_id"], "skip: no split")
            continue
        best = None
        for (ph, mis, g0, g1), ws in zip(split, wss):
            span = ph[-1][1] - ph[0][0]
            nsyl = sum(w[2] for w in ws)
            # a clean prompt: notes >= syllables, at most 4 slurred extras, silence on both sides
            if args.min_s <= span <= args.max_s and 0 <= len(ph) - nsyl <= 4 and min(g0, g1) >= 0.15:
                score = (len(ph) - nsyl) + abs(span - 6) * 0.3
                if best is None or score < best[0]:
                    best = (score, ph, ws)
        if best is None:
            print(tid, r["singer_id"], "skip: no usable phrase")
            continue
        _, ph, ws = best
        t0, t1 = max(0.0, ph[0][0] - 0.2), ph[-1][1] + 0.2
        name = f"vocadito_{tid}_{r['singer_id']}"
        subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-ss", f"{t0:.3f}", "-t", f"{t1 - t0:.3f}",
                        "-i", str(src / f"Audio/vocadito_{tid}.wav"), "-ac", "1", "-ar", str(SR),
                        str(out / f"{name}.wav")], check=True)
        m = meta_for(ph, ws, t0, t1)
        json.dump([m], open(out / f"{name}.json", "w", encoding="utf8"), ensure_ascii=False)
        index[name] = int(np.median([n[2] for n in ph]))
        done.add(r["singer_id"])
        print(name, f"{t1 - t0:.1f}s", len(ph), "notes", sum(w[2] for w in ws), "syl", "median", index[name],
              "|", " ".join(w[0] for w in ws))
    json.dump(index, open(out / "prompts.json", "w"), indent=1)
    print(len(index), "prompts")


if __name__ == "__main__":
    main()

"""Turn find_organ_pairs.py candidates into a piano_ids,pop_ids CSV for pianocover.download.

  python scripts/organ_to_pairs.py data/meta/organ_candidates.jsonl --out pairs/organ.csv

Each piano version is paired with one organ recording of the same piece, alternating between
the recordings. Recordings released by a major label are excluded (same rule as the song list,
docs/MODEL_LICENSES.md): EXCLUDE lists them by title fragment.
"""
import argparse
import csv
import json

EXCLUDE = {
    "Trifonov": "Deutsche Grammophon release (Universal Music)",
    "Karl Richter": "Archiv Produktion release (Universal Music)",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("candidates")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows, dropped = [], []
    for line in open(a.candidates, encoding="utf8"):
        j = json.loads(line)
        keep = lambda v: not any(k.lower() in v["title"].lower() for k in EXCLUDE)  # noqa: E731
        orig = [v for v in j["originals"] if keep(v)]
        cov = [v for v in j["covers"] if keep(v)]
        dropped += [v["title"] for v in j["originals"] + j["covers"] if not keep(v)]
        if not orig:
            continue
        rows += [(c["id"], orig[i % len(orig)]["id"]) for i, c in enumerate(cov)]
    with open(a.out, "w", newline="", encoding="utf8") as f:
        w = csv.writer(f)
        w.writerow(["piano_ids", "pop_ids"])
        w.writerows(rows)
    print(f"{len(rows)} pairs, {len({p for _, p in rows})} organ recordings -> {a.out}")
    for t in dropped:
        print("excluded:", t)


if __name__ == "__main__":
    main()

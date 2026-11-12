"""Merge several datasets (from build_dataset / import_aligned) into one training set.

  python -m pianocover.merge_datasets /data/ds_teacher /data/ds_silver /data/ds_gold \
      --out /data/ds_all --licensed-only

Each input directory must contain train.pkl and val.pkl. The style normaliser is refit on
the merged training split, so style vectors stay comparable across sources. With
--licensed-only, every item must carry a license that passes import_aligned's gate
(items from build_dataset, tagged license=unknown, are dropped).
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

from . import style as style_mod
from .import_aligned import license_ok


def load(path: Path):
    with path.open("rb") as f:
        blob = pickle.load(f)
    for it in blob["items"]:
        it.setdefault("data_root", blob["data_root"])
        it.setdefault("source", "web")
        it.setdefault("license", "unknown")
    return blob["items"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--licensed-only", action="store_true")
    ap.add_argument("--allow-sharealike", action="store_true")
    ap.add_argument("--allow-noncommercial", action="store_true")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    splits = {"train": [], "val": []}
    for d in map(Path, args.inputs):
        for name in splits:
            splits[name] += load(d / f"{name}.pkl")

    if args.licensed_only:
        for name in splits:
            before = len(splits[name])
            splits[name] = [it for it in splits[name] if license_ok(it["license"], args.allow_sharealike, args.allow_noncommercial)[0]]
            print(f"{name}: dropped {before - len(splits[name])} items without a usable license")

    # a song must not appear in both splits, even when it came from different sources
    val_songs = {it["pop_id"] for it in splits["val"]}
    leaked = [it for it in splits["train"] if it["pop_id"] in val_songs]
    if leaked:
        splits["train"] = [it for it in splits["train"] if it["pop_id"] not in val_songs]
        splits["val"] += leaked
        print(f"moved {len(leaked)} train items whose song is in val")
    if not splits["train"]:
        raise SystemExit("empty training split")

    norm = style_mod.StyleNormalizer.fit(np.stack([it["raw_style"] for it in splits["train"]]))
    norm.save(out / "style_stats.json")
    for name, items in splits.items():
        for it in items:
            it["style"] = norm.normalize(it["raw_style"])
        with (out / f"{name}.pkl").open("wb") as f:
            pickle.dump({"data_root": "", "items": items}, f)
        counts = {}
        for it in items:
            counts[it["source"]] = counts.get(it["source"], 0) + 1
        hours = sum(it["duration"] for it in items) / 3600
        print(f"{name}: {len(items)} items, {hours:.1f} h, by source {counts}")


if __name__ == "__main__":
    main()

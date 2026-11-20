"""Download tier-1 sources (real recordings with note labels) into <out>/<source>/raw.

Zenodo records are listed through the Zenodo API; files are fetched resumably (a .part file
is renamed when complete) and their md5 is checked against the record. Licences: see
docs/MODEL_LICENSES.md ("Tier 1 sources"); NC sources follow the NC policy there.

  python scripts/tier1_download.py --out /data/tier1 [--only vocadito,guitarset]
"""
import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

# source: (zenodo record id or None, file-name filter: keep a file if any substring matches; None = all)
ZENODO = {
    "vocadito": ("5578807", None),                                   # CC BY 4.0, solo singing
    "guitarset": ("3371780", ["annotation", "audio_mono-mic"]),      # CC BY 4.0, hexaphonic labels
    "dagstuhl": ("4618287", None),                                   # CC BY-NC-SA 4.0, choir multitrack
    "csd": ("2649950", None),                                        # CC BY-NC 4.0, Choral Singing Dataset
    "esmuc": ("5848990", None),                                      # CC BY-NC 4.0, ESMUC choir
    "cantoria": ("5878677", None),                                   # CC BY-NC 4.0, SATB + organ
    "musicnet": ("5120004", ["musicnet.tar.gz", "midis", "metadata"]),  # CC BY 4.0, DTW labels
}
DIRECT = {
    # URMP, CC0 (Dryad). Dryad needs its download API; filled in after checking the record.
}


def zenodo_files(rec):
    with urllib.request.urlopen(f"https://zenodo.org/api/records/{rec}", timeout=60) as r:
        d = json.load(r)
    return [(f["key"], f["links"]["self"], f["size"], f.get("checksum", "")) for f in d["files"]], d["metadata"].get("license")


def fetch(url, dst, size, checksum):
    if dst.exists() and dst.stat().st_size == size:
        return "exists"
    part = dst.with_suffix(dst.suffix + ".part")
    for _ in range(50):  # a dropped connection ends the read early: resume until the size matches
        have = part.stat().st_size if part.exists() else 0
        if have >= size:
            break
        req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
        try:
            with urllib.request.urlopen(req, timeout=120) as r, open(part, "ab" if have else "wb") as f:
                if have and r.status != 206:
                    f.seek(0)
                    f.truncate(0)
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    f.write(b)
        except OSError as e:
            print("retry", dst.name, str(e)[:100], flush=True)
    if part.stat().st_size != size:
        raise RuntimeError(f"size mismatch: {dst.name}")
    if checksum.startswith("md5:"):
        h = hashlib.md5()
        with open(part, "rb") as f:
            for b in iter(lambda: f.read(1 << 22), b""):
                h.update(b)
        if h.hexdigest() != checksum[4:]:
            part.unlink()
            raise RuntimeError(f"md5 mismatch: {dst.name}")
    part.rename(dst)
    return "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/data/tier1")
    ap.add_argument("--only")
    args = ap.parse_args()
    only = set(args.only.split(",")) if args.only else None
    for src, (rec, keep) in ZENODO.items():
        if only and src not in only:
            continue
        files, lic = zenodo_files(rec)
        raw = Path(args.out) / src / "raw"
        raw.mkdir(parents=True, exist_ok=True)
        (raw.parent / "zenodo_record.json").write_text(json.dumps({"record": rec, "license": lic,
                                                                     "files": files}, indent=1))
        for key, url, size, ck in files:
            if keep and not any(k in key for k in keep):
                continue
            try:
                print(src, key, f"{size / 1e6:.0f} MB", fetch(url, raw / key, size, ck), flush=True)
            except Exception as e:  # noqa: BLE001
                print("FAILED", src, key, str(e)[:200], flush=True, file=sys.stderr)


if __name__ == "__main__":
    main()

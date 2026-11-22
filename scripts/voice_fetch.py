"""Download the parts of the ready-made voice datasets that voice_data.py uses (owner 2026-11-21).

Only what is needed, from the official sources (licences: docs/MODEL_LICENSES.md):
  vocalset   VocalSet.zip + Annotated VocalSet.zip (Zenodo 1193957 / 7061507, CC BY 4.0), whole, md5 checked
  csd        CSD.zip v1.1 (Zenodo 4916302, CC BY-NC-SA 4.0), whole, md5 checked
  humtrans   all_midi.zip whole + a diverse ~3.2 h subset of all_wav.zip by HTTP range reads
             (huggingface.co/datasets/dadinghh2/HumTrans, CC BY-NC 4.0): one take per
             person x melody, round-robin over melodies
  ace        2 of the 19 train parquet shards of espnet/ace-kising-segments (CC BY-NC 4.0)
  (ChoralSynth: zenodo 10161065, already in /data/ext/voice_src_samples/choralsynth)
Resumable: finished files are skipped, partial downloads continue with Range requests.

  python scripts/voice_fetch.py --out /data/ext/voice_raw vocalset csd humtrans ace
"""
import argparse
import hashlib
import io
import os
import random
import re
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import requests

S = requests.Session()
S.headers["User-Agent"] = "dataset-fetch/0.1"
HT = "https://huggingface.co/datasets/dadinghh2/HumTrans/resolve/main/"


class HttpFile(io.RawIOBase):
    """Seekable read-only view of a remote file through Range requests."""

    def __init__(self, url):
        r = S.head(url, allow_redirects=True, timeout=60)
        self.url, self.size, self.pos = r.url, int(r.headers["Content-Length"]), 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def readinto(self, b):
        if self.pos >= self.size:
            return 0
        end = min(self.pos + len(b), self.size) - 1
        for k in range(6):
            try:
                r = S.get(self.url, headers={"Range": f"bytes={self.pos}-{end}"}, timeout=120)
                r.raise_for_status()
                break
            except requests.RequestException:
                if k == 5:
                    raise
        d = r.content
        b[: len(d)] = d
        self.pos += len(d)
        return len(d)


def remote_zip(url):
    return zipfile.ZipFile(io.BufferedReader(HttpFile(url), buffer_size=1 << 20))


def download(url, path, md5=None):
    path = Path(path)
    if path.exists() and (md5 is None or path.with_suffix(path.suffix + ".ok").exists()):
        return
    part = path.with_suffix(path.suffix + ".part")
    have = part.stat().st_size if part.exists() else 0
    with S.get(url, headers={"Range": f"bytes={have}-"} if have else {}, stream=True, timeout=120) as r:
        r.raise_for_status()
        mode = "ab" if have and r.status_code == 206 else "wb"
        with open(part, mode) as f:
            for chunk in r.iter_content(1 << 22):
                f.write(chunk)
    if md5:
        h = hashlib.md5()
        with open(part, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                h.update(chunk)
        if h.hexdigest() != md5:
            part.unlink()
            raise SystemExit(f"md5 mismatch for {path.name}")
        path.with_suffix(path.suffix + ".ok").write_text(md5)
    os.replace(part, path)
    print("downloaded", path, path.stat().st_size, flush=True)


def vocalset(out):
    d = out / "vocalset"
    d.mkdir(parents=True, exist_ok=True)
    download("https://zenodo.org/records/1193957/files/VocalSet.zip", d / "VocalSet.zip",
             "c44f60d34b8724b9a6f6d15e0a3158a9")
    download("https://zenodo.org/records/7061507/files/Annotated%20VocalSet.zip", d / "AnnotatedVocalSet.zip",
             "f4324592f5edebcb9e572c01531ac63b")


def csd(out):
    d = out / "csd"
    d.mkdir(parents=True, exist_ok=True)
    download("https://zenodo.org/records/4916302/files/CSD.zip", d / "CSD.zip", "74d121dd8706fded26a15526a379f7a2")


def humtrans(out, hours=3.2, seed=0):
    d = out / "humtrans"
    (d / "wav").mkdir(parents=True, exist_ok=True)
    download(HT + "all_midi.zip", d / "all_midi.zip")
    download(HT + "train_valid_test_keys.json", d / "train_valid_test_keys.json")
    z = remote_zip(HT + "all_wav.zip")
    groups = defaultdict(list)  # (person, melody) -> takes
    for i in z.infolist():
        m = re.match(r".*/((\w\d+)_(\d+)_\d+_\d+(?:_\w+)?)\.wav$", i.filename)
        if m:
            groups[(m.group(2), m.group(3))].append(i)
    rng = random.Random(seed)
    by_mel = defaultdict(list)
    for (person, mel), takes in sorted(groups.items()):
        by_mel[mel].append(rng.choice(takes))
    mels = sorted(by_mel)
    rng.shuffle(mels)
    for v in by_mel.values():
        rng.shuffle(v)
    chosen, secs, r = [], 0.0, 0
    while secs < hours * 3600 and any(by_mel.values()):
        for mel in mels:
            if by_mel[mel] and secs < hours * 3600:
                i = by_mel[mel].pop()
                chosen.append(i)
                secs += (i.file_size - 44) / 88200
        r += 1
    print(f"humtrans: {len(chosen)} takes, {len(set(c.filename.split('_')[1] for c in chosen))} melodies, "
          f"{secs / 3600:.2f} h", flush=True)
    for k, i in enumerate(chosen):
        p = d / "wav" / Path(i.filename).name
        if not p.exists():
            tmp = p.with_suffix(".part")
            tmp.write_bytes(z.read(i))
            os.replace(tmp, p)
        if k % 100 == 0:
            print("humtrans", k, "/", len(chosen), flush=True)


def ace(out, shards=(3, 12)):
    d = out / "ace_kising"
    d.mkdir(parents=True, exist_ok=True)
    for s in shards:
        name = f"train-{s:05d}-of-00019.parquet"
        download(f"https://huggingface.co/datasets/espnet/ace-kising-segments/resolve/main/data/{name}", d / name)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="/data/ext/voice_raw")
    ap.add_argument("sources", nargs="+", choices=["vocalset", "csd", "humtrans", "ace"])
    args = ap.parse_args()
    out = Path(args.out)
    for s in args.sources:
        globals()[s](out)
    print("fetch done", flush=True)


if __name__ == "__main__":
    sys.exit(main())

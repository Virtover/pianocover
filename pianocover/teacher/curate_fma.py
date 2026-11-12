"""Curate a commercially usable (CC0 / CC BY / public domain) subset of FMA full-length tracks.

  python -m pianocover.teacher.curate_fma --meta /data/fma/fma_metadata --out /data/fma_cc \
      --limit 10 --genres Rock,Pop,Folk,Electronic --download

Source: the FMA dataset dump (Defferrard et al., 2017; metadata CC BY 4.0). Every track has
its own license, recorded by the uploader at upload time; we use the license URL from
raw_tracks.csv (exact version and jurisdiction), not the free-text title.

License policy: the same gate as pianocover.import_aligned.license_ok (single source of
truth): CC0, CC BY any version, Public Domain Mark / CC public-domain dedication accepted;
NC and ND always refused; SA refused unless --allow-sharealike. Snapshot statistics of the
dump (2026-11-09): ~5,050 CC BY 4.0, ~1,840 CC BY 3.0 variants, 1,018 CC0, ~380 public
domain, of 106,574 tracks; most of FMA is NC and/or ND.

Audio comes from fma_full.zip (879 GiB) WITHOUT downloading it: the server supports HTTP
range requests, so zipfile reads the central directory and only the selected members.

Output: <out>/songs.csv  id,audio,license,license_url,attribution,title,artist,album,
        track_url,genre,duration,fma_track_id  (input for pianocover.teacher.pipeline)
        <out>/audio/fma_<id>.mp3
        <out>/ATTRIBUTION.md  human-readable credits for every selected track
"""

from __future__ import annotations

import argparse
import csv
import io
import re
import urllib.request
import zipfile
from pathlib import Path

FMA_FULL_URL = "https://os.unil.cloud.switch.ch/fma/fma_full.zip"
DEFAULT_GENRES = "Rock,Pop,Folk,Electronic,Hip-Hop,Soul-RnB,Country,Blues,Jazz,International,Instrumental"


def canonical_license(url: str) -> str:
    """Creative Commons URL -> canonical id understood by import_aligned.license_ok."""
    u = (url or "").strip().lower()
    if "publicdomain/zero" in u:
        return "CC0-1.0"
    if "publicdomain/mark" in u or u.rstrip("/").endswith("licenses/publicdomain"):
        return "public-domain"
    m = re.search(r"creativecommons\.org/licenses/([a-z\-]+)/([\d.]+)(?:/([a-z]{2}))?", u)
    if not m:
        return "unknown"
    kind, ver, juris = m.groups()
    out = f"CC-{kind.upper()}-{ver}"
    return f"{out}-{juris.upper()}" if juris else out


class HTTPRangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP range requests, with a small block cache."""

    def __init__(self, url: str, block: int = 1 << 20):
        self.url, self.block, self.pos = url, block, 0
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as r:
            self.size = int(r.headers["Content-Length"])
            if r.headers.get("Accept-Ranges") != "bytes":
                raise RuntimeError("server does not support range requests")
        self._cache: dict[int, bytes] = {}

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=io.SEEK_SET):
        self.pos = off if whence == io.SEEK_SET else self.pos + off if whence == io.SEEK_CUR else self.size + off
        return self.pos

    def _fetch(self, start: int, end: int) -> bytes:
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end - 1}"})
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    return r.read()
            except Exception:
                if attempt == 3:
                    raise
        return b""

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        if n > 4 * self.block:  # large reads (audio members) go straight through
            data = self._fetch(self.pos, self.pos + n)
        else:
            parts, p, end = [], self.pos, self.pos + n
            while p < end:
                b = p // self.block
                if b not in self._cache:
                    if len(self._cache) > 64:
                        self._cache.clear()
                    self._cache[b] = self._fetch(b * self.block, min((b + 1) * self.block, self.size))
                chunk = self._cache[b]
                off = p - b * self.block
                take = min(end - p, len(chunk) - off)
                parts.append(chunk[off:off + take])
                p += take
            data = b"".join(parts)
        self.pos += len(data)
        return data

    def readinto(self, b):
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)


def select_tracks(meta_dir: Path, genres, min_dur, max_dur, allow_sa, limit, order, seed, allow_nc=False):
    import pandas as pd

    from ..import_aligned import license_ok

    tracks = pd.read_csv(meta_dir / "tracks.csv", index_col=0, header=[0, 1])
    raw = pd.read_csv(meta_dir / "raw_tracks.csv", index_col=0)
    df = pd.DataFrame({
        "genre": tracks[("track", "genre_top")],
        "duration": tracks[("track", "duration")],
        "listens": tracks[("track", "listens")],
        "title": tracks[("track", "title")],
        "artist": tracks[("artist", "name")],
        "album": tracks[("album", "title")],
        "license_title": tracks[("track", "license")],
    }).join(raw[["license_url", "track_url", "artist_url"]], how="left")
    df["license"] = df["license_url"].fillna("").map(canonical_license)
    verdict = df["license"].map(lambda s: license_ok(s, allow_sa, allow_nc))
    df["license_ok"] = verdict.map(lambda v: v[0])
    stats = {"tracks": len(df), "license_ok": int(df["license_ok"].sum())}
    df = df[df["license_ok"]]
    if genres:
        df = df[df["genre"].isin(genres)]
    df = df[(df["duration"] >= min_dur) & (df["duration"] <= max_dur)]
    stats["after_filters"] = len(df)
    if order == "listens":
        df = df.sort_values("listens", ascending=False)
    else:
        df = df.sample(frac=1.0, random_state=seed)
    if limit:
        # round-robin over genres so a small prototype set is not all Rock
        groups = [g for _, g in df.groupby("genre", sort=False)]
        picked, i = [], 0
        while len(picked) < limit and any(i < len(g) for g in groups):
            for g in groups:
                if i < len(g) and len(picked) < limit:
                    picked.append(g.index[i])
            i += 1
        df = df.loc[picked]
    return df, stats


def attribution(row) -> str:
    return (f"\"{row.title}\" by {row.artist} ({row.track_url}), licensed under {row.license_title} "
            f"({row.license_url}); via the FMA dataset (Defferrard et al., 2017)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--meta", required=True, help="extracted fma_metadata directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--genres", default=DEFAULT_GENRES, help="comma list of FMA top genres ('' = all)")
    ap.add_argument("--min-dur", type=float, default=90)
    ap.add_argument("--max-dur", type=float, default=420)
    ap.add_argument("--allow-sharealike", action="store_true")
    ap.add_argument("--allow-noncommercial", action="store_true",
                    help="also keep NC tracks (risk decision; see docs/MODEL_LICENSES.md); ND stays excluded")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--order", choices=["listens", "random"], default="listens")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--download", action="store_true", help="fetch the selected mp3s from fma_full.zip")
    ap.add_argument("--zip-url", default=FMA_FULL_URL)
    args = ap.parse_args(argv)

    out = Path(args.out)
    (out / "audio").mkdir(parents=True, exist_ok=True)
    genres = [g.strip() for g in args.genres.split(",") if g.strip()]
    df, stats = select_tracks(Path(args.meta), genres, args.min_dur, args.max_dur, args.allow_sharealike,
                              args.limit, args.order, args.seed, allow_nc=args.allow_noncommercial)
    print(f"{stats['tracks']} tracks, {stats['license_ok']} pass the license gate, "
          f"{stats['after_filters']} after genre/duration filters; selected {len(df)}", flush=True)

    zf = None
    if args.download:
        zf = zipfile.ZipFile(HTTPRangeFile(args.zip_url))
    rows = []
    for n, (tid, row) in enumerate(df.iterrows(), 1):
        sid = f"fma_{tid:06d}"
        dst = out / "audio" / f"{sid}.mp3"
        if zf is not None and not dst.exists():
            member = f"fma_full/{tid:06d}"[:12] + f"/{tid:06d}.mp3"
            try:
                data = zf.read(member)
            except KeyError:
                print(f"  {sid}: not in fma_full.zip, skipped", flush=True)
                continue
            dst.write_bytes(data)
            print(f"  [{n}/{len(df)}] {sid} {row.genre}: {row.artist} - {row.title} ({len(data) / 2**20:.1f} MiB)",
                  flush=True)
        if not dst.exists():
            continue
        rows.append({"id": sid, "audio": f"audio/{sid}.mp3", "license": row.license,
                     "license_url": row.license_url, "attribution": attribution(row), "title": row.title,
                     "artist": row.artist, "album": row.album, "track_url": row.track_url, "genre": row.genre,
                     "duration": row.duration, "fma_track_id": tid})
    with (out / "songs.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["id"])
        w.writeheader()
        w.writerows(rows)
    with (out / "ATTRIBUTION.md").open("w", encoding="utf-8") as f:
        f.write("# Audio credits\n\nTracks from the Free Music Archive (FMA dataset dump, 2017).\n\n")
        for r in rows:
            f.write(f"- {r['attribution']}\n")
    print(f"wrote {len(rows)} rows -> {out / 'songs.csv'}")


if __name__ == "__main__":
    main()

"""Training items from the ready-made voice datasets (owner 2026-11-21; licences in MODEL_LICENSES.md).

Sources (raw files from voice_fetch.py), each written like the other families: 16-bit mono
44.1 kHz WAV in <out>/<src>/, per-item records in <out>/<src>/items, meta.csv and transkun
train/val pickles:
  vocalset   all non-spoken files that pass the label check: the annotated MIDI of each sung
             note vs the MIDI estimated from its f0; a file is dropped if >10% of its notes
             differ by more than one semitone (the labels have known errors)
  csd        Children's Song Dataset, one of the two keys per song
  humtrans   the ~3.2 h subset fetched (one take per person x melody)
  ace        ACE-KiSing, ~2 h from 2 parquet shards, consecutive segments joined to <= 40 s,
             spread over songs; song lyrics printed for the content check
  choralsynth all 20 pieces as the full mix, plus a second render with a random subset of the
             voices (and the accompaniment when there is one) and other levels / room / EQ

Solo voices (all but choralsynth) get an accompaniment with probability --acc-p: an excerpt of
a synth_tk2 render (same split) whose key matches the voice's key (Krumhansl profiles, relative
major/minor counts as a match), shifted by at most +-2 semitones by resampling. The target is
then the voice notes plus the excerpt's notes (shifted the same way), so the mix is a complete
example. The excerpt starts at a note onset where few notes are held over; held-over notes are
dropped from the target and the excerpt fades in over 30 ms. Voice 0..+8 dB over the
accompaniment, optional synthetic room on the voice, then the same MP3 / EQ / level chain as
the renders (vocal_synth.finish_audio). Resumable per item.

  python scripts/voice_data.py --raw /data/ext/voice_raw --out /data/voice_data vocalset csd humtrans ace choralsynth
"""
import argparse
import csv
import glob
import io
import json
import os
import pickle
import random
import re
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_synth import FS, to_transkun  # noqa: E402
from vocal_synth import finish_audio, resample, rms, synth_ir  # noqa: E402

MAJ = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MIN = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def key_of(notes):
    """Major-equivalent tonic (0-11) of [s, e, p, ...] notes, or None (relative minor -> its major)."""
    if len(notes) == 0:
        return None
    h = np.zeros(12)
    for s, e, p in np.asarray(notes)[:, :3]:
        h[int(p) % 12] += max(e - s, 0.05)
    best, arg = -2.0, None
    for t in range(12):
        for prof, minor in ((MAJ, False), (MIN, True)):
            c = np.corrcoef(h, np.roll(prof, t))[0, 1]
            if c > best:
                best, arg = c, ((t + 3) % 12 if minor else t)
    return arg


def wrap(k):
    return (k + 6) % 12 - 6


class AccPool:
    """synth_tk2 renders, per split, for key-matched accompaniment excerpts."""

    def __init__(self, root):
        self.root = Path(root)
        self.items = {"train": [], "val": []}
        for f in sorted(glob.glob(str(self.root / "*" / "*.pickle"))):
            split = Path(f).stem
            for r in pickle.load(open(f, "rb")):
                ev = np.array([(n.start, n.end, n.pitch, n.velocity) for n in r["notes"]], dtype=np.float64)
                if len(ev) > 20:
                    self.items[split].append((str(self.root / r["audio_filename"]), float(r["duration"]), ev))
        print("acc pool:", {k: len(v) for k, v in self.items.items()}, flush=True)

    def excerpt(self, split, length, vkey, rng, tries=60):
        pool = self.items[split] or self.items["train"]
        best = None
        for _ in range(tries):
            path, dur, ev = rng.choice(pool)
            k0 = rng.choice([-2, -1, 0, 0, 1, 2])
            r = 2 ** (k0 / 12)
            need = length * r
            if dur < need + 1:
                continue
            ons = ev[(ev[:, 0] < dur - need) & (ev[:, 0] > 0.5), 0]
            if len(ons) == 0:
                continue
            b = float(rng.choice(ons))
            win = ev[(ev[:, 0] >= b - 1e-3) & (ev[:, 0] < b + need)]
            akey = key_of(win)
            if akey is None:
                continue
            k = wrap(vkey - akey) if vkey is not None else k0
            if abs(k) > 2:
                continue
            held = int(((ev[:, 0] < b - 0.01) & (ev[:, 1] > b + 0.1)).sum())
            score = held + 0.5 * abs(k)
            if best is None or score < best[0]:
                best = (score, path, b, k, ev)
            if held == 0:
                break
        if best is None:
            return None
        _, path, b, k, ev = best
        r = 2 ** (k / 12)
        n_in = int(length * r * FS) + 2
        x, _ = sf.read(path, start=int(b * FS), frames=n_in, dtype="float32")
        if x.ndim > 1:
            x = x.mean(1)
        n = int(length * FS)
        pos = np.arange(n) * r
        y = np.interp(pos, np.arange(len(x)), x, right=0.0).astype(np.float32)
        fade = min(len(y), int(0.03 * FS))
        y[:fade] *= np.linspace(0, 1, fade)
        sel = ev[(ev[:, 0] >= b - 1e-3) & (ev[:, 0] < b + length * r)]
        notes = np.column_stack([(sel[:, 0] - b) / r, np.minimum((sel[:, 1] - b) / r, length), sel[:, 2] + k,
                                 sel[:, 3]])
        return y, notes, {"acc": Path(path).stem, "acc_at": round(b, 2), "acc_shift": k}


def build_item(out, src, iid, voice, vnotes, split, rng, pool, acc_p, meta):
    """voice: float mono 44.1 kHz; vnotes: [s, e, p] (+v). Writes WAV + record; returns hours."""
    items = out / src / "items"
    items.mkdir(parents=True, exist_ok=True)
    if (items / f"{iid}.pkl").exists():
        return 0.0
    vnotes = np.asarray(vnotes, dtype=np.float64).reshape(-1, len(vnotes[0]) if len(vnotes) else 3)
    if vnotes.shape[1] == 3:
        vnotes = np.column_stack([vnotes, np.full(len(vnotes), 80.0)])
    length = len(voice) / FS
    if rng.random() < 0.5:  # synthetic room on the voice
        ir, rt = synth_ir(rng)
        from scipy.signal import fftconvolve

        wet = rng.uniform(0.1, 0.4)
        voice = (1 - wet) * voice + wet * fftconvolve(voice, ir)[: len(voice)].astype(np.float32)
        meta = {**meta, "vocal_ir_rt60": round(rt, 2), "vocal_wet": round(wet, 2)}
    notes = vnotes
    mix = voice / max(rms(voice), 1e-6) * 0.1
    if pool is not None and rng.random() < acc_p:
        exc = pool.excerpt(split, length, key_of(vnotes), rng)
        if exc is not None:
            acc, anotes, ameta = exc
            acc = np.pad(acc, (0, max(0, len(voice) - len(acc))))[: len(voice)]
            lead = rng.uniform(0, 8)
            mix = acc / max(rms(acc), 1e-6) * 0.1 + voice / max(rms(voice), 1e-6) * 0.1 * 10 ** (lead / 20)
            notes = np.vstack([vnotes, anotes]) if len(anotes) else vnotes
            meta = {**meta, **ameta, "lead_db": round(lead, 1)}
    from pianocover import events

    ev = np.array(sorted(notes.tolist()), dtype=np.float64).reshape(-1, 4)
    while (ev[:, 2] < 21).any() or (ev[:, 2] > 108).any():
        ev[ev[:, 2] < 21, 2] += 12
        ev[ev[:, 2] > 108, 2] -= 12
    ev = events.sanitize(ev)
    x, wav, mp3 = finish_audio(mix.astype(np.float32), rng, out, src, iid)
    dur = len(x) / FS
    ev = ev[ev[:, 0] < dur]
    rec = {"fam": src, "id": iid, "notes": ev, "duration": dur, "nSamples": len(x), "split": split,
           "meta": {**meta, "mp3": int(mp3)}}
    pickle.dump(rec, open(items / f"{iid}.pkl.tmp", "wb"))
    os.replace(items / f"{iid}.pkl.tmp", items / f"{iid}.pkl")
    return dur / 3600


def to_fs(x, sr):
    if x.ndim > 1:
        x = x.mean(1)
    x = x.astype(np.float32)
    return resample(x, sr, FS) if sr != FS else x


# ------------------------------------------------------------------ sources
def vocalset(raw, out, rng, pool, acc_p):
    za = zipfile.ZipFile(raw / "vocalset" / "AnnotatedVocalSet.zip")
    zv = zipfile.ZipFile(raw / "vocalset" / "VocalSet.zip")
    wavs = {Path(n).stem.replace(" ", ""): n for n in zv.namelist() if n.endswith(".wav") and "__MACOSX" not in n}
    anns = [n for n in za.namelist() if "/with file header/" in n and n.endswith(".csv") and "spoken" not in n]
    kept = dropped = 0
    hours = 0.0
    for a in sorted(anns):
        stem = Path(a).stem.replace(" ", "")
        if stem not in wavs:
            continue
        lines = za.read(a).decode("utf-8", "replace").splitlines()
        try:
            h = next(i for i, l in enumerate(lines) if l.startswith("Sequence"))
        except StopIteration:
            continue
        notes, bad = [], 0
        for r in csv.reader(lines[h + 1:]):
            if len(r) > 14 and r[4].strip() == "Sound" and r[14].strip():
                try:
                    gt, est = int(float(r[14])), float(r[11])
                    s, e = float(r[1]), float(r[2])
                except ValueError:
                    continue
                bad += abs(gt - est) > 1.0
                notes.append([s, e, gt])
        if not notes or bad > 0.1 * len(notes):
            dropped += 1
            continue
        kept += 1
        x, sr = sf.read(io.BytesIO(zv.read(wavs[stem])), dtype="float32")
        singer = stem.split("_")[0]
        split = "val" if singer == "m11" else "train"
        hours += build_item(out, "vocalset", stem, to_fs(x, sr), notes, split, rng, pool, acc_p,
                            {"title": stem, "singer": singer})
    print(f"vocalset: kept {kept}, dropped {dropped} (label check); new {hours:.2f} h", flush=True)


def csd(raw, out, rng, pool, acc_p):
    z = zipfile.ZipFile(raw / "csd" / "CSD.zip")
    names = z.namelist()
    meta = {}
    for n in names:
        if n.endswith("metadata.json"):
            meta.update(json.loads(z.read(n)))
    songs = sorted({re.sub(r"[ab]$", "", Path(n).stem) for n in names if n.endswith(".wav")})
    val = set(rng.sample(songs, 5))
    hours = 0.0
    for s in songs:
        k = s + rng.choice("ab")
        wav = next(n for n in names if n.endswith(f"/{k}.wav"))
        rows = list(csv.DictReader(io.StringIO(z.read(next(n for n in names if n.endswith(f"/{k}.csv"))).decode())))
        notes = [[float(r["start"]), float(r["end"]), int(r["pitch"])] for r in rows]
        x, sr = sf.read(io.BytesIO(z.read(wav)), dtype="float32")
        title = meta.get(k, {}).get("songname", k)
        hours += build_item(out, "csd", k, to_fs(x, sr), notes, "val" if s in val else "train", rng, pool, acc_p,
                            {"title": title})
    print(f"csd: {len(songs)} songs; new {hours:.2f} h", flush=True)


def humtrans(raw, out, rng, pool, acc_p):
    import pretty_midi

    d = raw / "humtrans"
    zm = zipfile.ZipFile(d / "all_midi.zip")
    mids = {Path(n).stem: n for n in zm.namelist() if n.endswith(".mid")}
    keys = json.load(open(d / "train_valid_test_keys.json"))
    valset = {k for sp, v in keys.items() if sp.lower() != "train" for k in v}
    hours = 0.0
    for w in sorted((d / "wav").glob("*.wav")):
        if w.stem not in mids:
            continue
        pm = pretty_midi.PrettyMIDI(io.BytesIO(zm.read(mids[w.stem])))
        notes = [[n.start, n.end, n.pitch] for i in pm.instruments for n in i.notes]
        if not notes:
            continue
        x, sr = sf.read(w, dtype="float32")
        split = "val" if w.stem in valset else "train"
        hours += build_item(out, "humtrans", w.stem, to_fs(x, sr), notes, split, rng, pool, acc_p,
                            {"title": w.stem.split("_")[1]})
    print(f"humtrans: new {hours:.2f} h", flush=True)


def ace(raw, out, rng, pool, acc_p, hours_max=2.0, max_len=40.0):
    import pyarrow.parquet as pq

    songs = defaultdict(list)
    for f in sorted((raw / "ace_kising").glob("*.parquet")):
        t = pq.read_table(f, columns=["segment_id", "singer", "note_midi", "note_lyrics", "note_start_times",
                                      "note_end_times", "audio"])
        for row in t.to_pylist():
            m = re.match(r"kising_(.+)_(\d+)_(\d+)$", row["segment_id"])
            if m:
                songs[(m.group(1), m.group(2))].append((int(m.group(3)), row))
    # spread over songs: at most ~6 min per (singer, song), songs in random order
    order = sorted(songs)
    rng.shuffle(order)
    val = set(order[:2])
    done_h, n_items = 0.0, 0
    for key in order:
        if done_h >= hours_max:
            break
        segs = [r for _, r in sorted(songs[key], key=lambda x: x[0])]
        lyr = " ".join(l for r in segs[:12] for l in r["note_lyrics"] if l not in ("<AP>", "<SP>", "-"))
        print("ace song", key, "|", lyr[:160], flush=True)
        if key[1] in BAD_ACE_SONGS:
            continue
        chunk, notes, t, used = [], [], 0.0, 0.0
        for r in segs:
            x, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
            x = to_fs(x, sr)
            seg_len = len(x) / FS
            for p, a, b in zip(r["note_midi"], r["note_start_times"], r["note_end_times"]):
                if p > 0:
                    notes.append([t + a, t + b, p])
            chunk.append(x)
            t += seg_len
            if t >= max_len or r is segs[-1]:
                if notes:
                    iid = f"{key[0]}_{key[1]}_{n_items:04d}"
                    h = build_item(out, "ace_kising", iid, np.concatenate(chunk), notes,
                                   "val" if key in val else "train", rng, pool, acc_p,
                                   {"title": f"kising song {key[1]}", "singer": key[0]})
                    done_h += h if h else t / 3600
                    used += t
                    n_items += 1
                chunk, notes, t = [], [], 0.0
                if used > 360 or done_h >= hours_max:
                    break
    print(f"ace: {n_items} items, {done_h:.2f} h", flush=True)


BAD_ACE_SONGS = set()  # song numbers dropped after reading their lyrics (content filter)


def choralsynth(raw, out, rng, pool, acc_p):
    import pretty_midi
    import subprocess
    import tempfile

    z = zipfile.ZipFile(raw / "choralsynth" / "ChoralSynth.zip")
    pieces = sorted({n.split("/")[1] for n in z.namelist() if n.count("/") >= 3})
    hours = 0.0
    for k, p in enumerate(pieces):
        tracks = [n for n in z.namelist() if n.startswith(f"ChoralSynth/{p}/voices/") and n.endswith(".mp3")]
        pm = pretty_midi.PrettyMIDI(io.BytesIO(z.read(f"ChoralSynth/{p}/score.midi")))
        norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())  # noqa: E731
        inst_by = {norm(i.name): i for i in pm.instruments if not i.is_drum}
        audio = {}
        with tempfile.TemporaryDirectory() as td:
            for n in tracks:
                src = Path(td) / "t.mp3"
                src.write_bytes(z.read(n))
                r = subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(src), "-ac", "1", "-ar",
                                    str(FS), "-f", "f32le", "-"], check=True, capture_output=True)
                audio[n] = np.frombuffer(r.stdout, dtype=np.float32).copy()
        n_out = max(len(a) for a in audio.values())
        matched = all(norm(Path(n).stem) in inst_by for n in tracks) and len(tracks) == len(inst_by)
        split = "val" if k % 10 == 9 else "train"
        for variant in (0, 1):
            if variant == 1 and not matched:
                continue
            use = list(tracks)
            if variant == 1:
                voices = [n for n in tracks if not re.search(r"(?i)organ|piano|accomp|practice", n)]
                accs = [n for n in tracks if n not in voices]
                use = rng.sample(voices, rng.randint(max(1, len(voices) // 2), max(1, len(voices) - 1))) + accs
            mix = np.zeros(n_out, dtype=np.float32)
            for n in use:
                a = audio[n]
                mix[: len(a)] += a * 10 ** (rng.uniform(-4, 2) / 20)
            if variant == 0:
                notes = [[n.start, n.end, n.pitch, n.velocity] for i in pm.instruments if not i.is_drum for n in i.notes]
            else:
                keep = {norm(Path(n).stem) for n in use}
                notes = [[n.start, n.end, n.pitch, n.velocity] for nm, i in inst_by.items() if nm in keep for n in i.notes]
            hours += build_item(out, "choralsynth", f"{k:02d}_v{variant}", mix, notes, split, rng, None, 0.0,
                                {"title": p, "tracks": len(use)})
    print(f"choralsynth: {len(pieces)} pieces; new {hours:.2f} h", flush=True)


def finalize(out, src):
    recs = [pickle.load(open(f, "rb")) for f in sorted((out / src / "items").glob("*.pkl"))]
    if not recs:
        return
    keys = list(dict.fromkeys(k for r in recs for k in r["meta"]))
    with open(out / src / "meta.csv.tmp", "w", newline="", encoding="utf8") as f:
        w = csv.writer(f)
        w.writerow(["id", "split", "duration", "nSamples", *keys])
        for r in recs:
            w.writerow([r["id"], r["split"], round(r["duration"], 2), r["nSamples"], *(r["meta"].get(k, "") for k in keys)])
    os.replace(out / src / "meta.csv.tmp", out / src / "meta.csv")
    for split in ("train", "val"):
        items = [{"audio_filename": f"{src}/{r['id']}.wav", "duration": r["duration"], "fs": FS,
                  "nSamples": r["nSamples"], "nChannel": 1, "notes": to_transkun(r["notes"]), "source": src,
                  "split": split} for r in recs if r["split"] == split]
        pickle.dump(items, open(out / src / f"{split}.pickle.tmp", "wb"))
        os.replace(out / src / f"{split}.pickle.tmp", out / src / f"{split}.pickle")
    h = {sp: sum(r["duration"] for r in recs if r["split"] == sp) / 3600 for sp in ("train", "val")}
    print(f"{src}: {len(recs)} items, train {h['train']:.2f} h, val {h['val']:.2f} h", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default="/data/ext/voice_raw")
    ap.add_argument("--out", default="/data/voice_data")
    ap.add_argument("--acc-root", default="/data/synth_tk2")
    ap.add_argument("--acc-p", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bad-ace", default="", help="comma-separated ACE song numbers to drop (content filter)")
    ap.add_argument("sources", nargs="+", choices=["vocalset", "csd", "humtrans", "ace", "choralsynth"])
    args = ap.parse_args()
    BAD_ACE_SONGS.update(s for s in args.bad_ace.split(",") if s)
    raw, out = Path(args.raw), Path(args.out)
    pool = AccPool(args.acc_root)
    for k, s in enumerate(args.sources):
        rng = random.Random(f"{args.seed}-{s}")
        globals()[s](raw, out, rng, pool, args.acc_p)
        finalize(out, "ace_kising" if s == "ace" else s)


if __name__ == "__main__":
    main()

"""Synthetic 1:1 transcription data for Transkun: public-domain scores rendered as audio.

Input = a PDMX score rendered by fluidsynth (random soundfont, instruments within each track's
family, room, mix, EQ, compression, MP3 round-trip); target = every pitched note of the same
score (drums never), folded into 21-108, unisons merged, score durations kept, no pedal.
Audio and notes match by construction. See docs/TRANSKUN_PLAN.md (tier 3).

  python scripts/score_synth.py --pdmx /data/pdmx --out /data/synth_tk \
      --sf2 /sf2/GeneralUser-GS.sf2 /sf2/TimGM6mb.sf2 /usr/share/sounds/sf2/FluidR3_GM.sf2 \
      [--families organ,chamber,...] [--per-family organ=89,piano=150] [--renders 2] [--workers 6]

Families (from the score's MIDI programs): organ, piano, guitar, choir, chamber, orchestra,
band (has a drum track). Scores: no_license_conflict + deduplicated, publicdomain / cc-zero,
30 s - 10 min, composer died before 1955 (organ_synth.OLD), obvious modern / pop excluded.

Output: <out>/<family>/<id>.wav (16-bit mono 44.1 kHz, what transkun.Data reads),
<out>/<family>/meta.csv, and <out>/<family>/{train,val}.pickle in transkun's DatasetMaestro
format (audio_filename relative to <out>). Resumable: finished renders are skipped.
"""
import argparse
import csv
import io
import os
import pickle
import random
import re
import subprocess
import sys
import tarfile
import tempfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from organ_synth import MODERN, OLD, progs  # noqa: E402
from organ_synth import select as select_organ  # noqa: E402
from pdmx_organ import score_events  # noqa: E402

FS = 44100
DEFAULT_N = {"organ": 89, "piano": 150, "guitar": 40, "choir": 80, "chamber": 120, "orchestra": 150, "band": 80}
# instrument families used to randomise programs: a track keeps its family, gets a random member
SWAP = [list(range(0, 2)), [4, 5], list(range(16, 21)), [24, 25], [26, 27], list(range(32, 36)),
        [40, 41, 42, 43, 44, 45, 48, 49], [52, 53], list(range(56, 62)), list(range(64, 72)),
        list(range(72, 76))]
SWAP_OF = {p: g for g in SWAP for p in g}


def family(prog_set, has_drum):
    p = set(prog_set)
    if not p:
        return None
    if has_drum:
        return "band"
    if p <= set(range(0, 8)):
        return "piano"
    if p <= set(range(16, 21)):
        return "organ"
    if p <= set(range(24, 32)) | set(range(32, 40)) and p & set(range(24, 32)):
        return "guitar"
    if p & {52, 53, 54}:
        return "choir"
    fams = sum(bool(p & set(r)) for r in (range(40, 52), range(56, 64), range(64, 80)))
    if fams >= 2 and len(p) >= 5:
        return "orchestra"
    if p <= set(range(0, 8)) | set(range(40, 48)) | set(range(64, 80)) | {24, 46} and len(p) >= 2:
        return "chamber"
    return None


def candidates(pdmx: Path, families, per_family, seed):
    """{family: [rows]}; the organ family reuses organ_synth.select (Bach organ works too)."""
    import pandas as pd
    import pretty_midi

    df = pd.read_csv(pdmx / "PDMX.csv", low_memory=False)
    df = df[df["subset:no_license_conflict"] & df["subset:deduplicated"] & df["mid"].notna()
            & df["license"].isin(["publicdomain", "cc-zero"])
            & df["song_length.seconds"].between(30, 600) & (df["n_notes"] >= 150)]
    txt = df.title.fillna("") + " | " + df.composer_name.fillna("") + " | " + df.artist_name.fillna("")
    df = df[txt.str.contains(OLD) & ~txt.str.contains(MODERN)]
    # piano-only scores are most of PDMX: sample them before parsing any MIDI
    tr = df["tracks"].map(lambda t: set(progs(t)))
    is_piano = tr.map(lambda p: bool(p) and p <= set(range(8)))
    rng = random.Random(seed)
    piano_idx = list(df.index[is_piano])
    rng.shuffle(piano_idx)
    keep = set(df.index[~is_piano]) | set(piano_idx[: 4 * per_family.get("piano", 150)])
    df = df.loc[sorted(keep)]
    want = {m.lstrip("./"): i for i, m in zip(df.index, df["mid"])}
    out = {f: [] for f in families}
    mids = {}
    with tarfile.open(pdmx / "mid.tar.gz") as tar:
        for m in tar:
            name = m.name.lstrip("./")
            if not (m.isfile() and name in want):
                continue
            data = tar.extractfile(m).read()
            try:
                pm = pretty_midi.PrettyMIDI(io.BytesIO(data))
            except Exception:  # noqa: BLE001
                continue
            pitched = [i for i in pm.instruments if not i.is_drum and i.notes]
            drums = sum(len(i.notes) for i in pm.instruments if i.is_drum)
            f = family([i.program for i in pitched], drums > 50)
            if f in out and f != "organ":
                mids[want[name]] = data
                out[f].append(want[name])
    rows = {}
    for f in families:
        if f == "organ":
            sel = select_organ(pdmx, per_family.get("organ", 89), seed)
            rows[f] = [dict(r, _idx=i) for i, r in sel.iterrows()]
            continue
        idx = out[f]
        sub = df.loc[idx].sort_values("n_views", ascending=False).head(per_family.get(f, 100))
        rows[f] = [dict(r, _idx=i) for i, r in sub.iterrows()]
    need = {r["mid"].lstrip("./") for f in rows for r in rows[f]} - {df.loc[i, "mid"].lstrip("./") for i in mids}
    if need:
        with tarfile.open(pdmx / "mid.tar.gz") as tar:
            for m in tar:
                name = m.name.lstrip("./")
                if m.isfile() and name in need:
                    mids[name] = tar.extractfile(m).read()
    for f in rows:
        for r in rows[f]:
            r["_midi"] = mids.get(r["_idx"]) or mids.get(r["mid"].lstrip("./"))
        rows[f] = [r for r in rows[f] if r["_midi"]]
    return rows


def audio_midi(midi_bytes, fam, rng):
    """The MIDI that is rendered (target notes come from the original score, not this)."""
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(io.BytesIO(midi_bytes))
    cfg = {"programs": [], "ranks": ""}
    for inst in pm.instruments:
        gain = rng.uniform(0.5, 1.0) if not inst.is_drum else rng.uniform(0.25, 1.0)
        inst.control_changes = [c for c in inst.control_changes if c.number not in (0, 32)]
        has7 = False
        for c in inst.control_changes:
            if c.number == 7:
                c.value = int(np.clip(c.value * gain, 1, 127))
                has7 = True
        if not has7:
            inst.control_changes.insert(0, pretty_midi.ControlChange(7, int(100 * gain), 0.0))
        if not inst.is_drum and inst.program in SWAP_OF and rng.random() < 0.6:
            inst.program = rng.choice(SWAP_OF[inst.program])
        cfg["programs"].append("drums" if inst.is_drum else inst.program)
    if fam == "organ":  # extra ranks in the audio only: harmonics the model must not report
        r = rng.random()
        add = [12] if r < 0.3 else ([12, 19] if r < 0.5 else [])
        cfg["ranks"] = "+".join(map(str, add))
        for inst in [i for i in pm.instruments if not i.is_drum]:
            extra = [pretty_midi.Note(int(n.velocity * 0.6), n.pitch + k, n.start, n.end)
                     for n in inst.notes for k in add if n.pitch + k <= 108]
            inst.notes.extend(extra)
    return pm, cfg


def post_chain(rng):
    """ffmpeg filters: EQ tilt, optional compression; the MP3 round-trip is done separately."""
    f = [f"lowshelf=g={rng.uniform(-6, 6):.1f}:f=200", f"highshelf=g={rng.uniform(-8, 4):.1f}:f=4000"]
    if rng.random() < 0.5:
        f.append(f"acompressor=threshold={rng.uniform(0.05, 0.3):.2f}:ratio={rng.uniform(2, 6):.1f}"
                 ":attack=10:release=200")
    return ",".join(f)


def work(job):
    fam, rid, midi_bytes, sf2, seed, out = job
    wav = Path(out) / fam / f"{rid}.wav"
    if wav.exists():
        return None
    rng = random.Random(seed)
    try:
        ev = score_events(midi_bytes)
        from pianocover import events

        ev = events.sanitize(ev)
        if (ev[:, 2] > 0).sum() < 100:
            return None
        pm, cfg = audio_midi(midi_bytes, fam, rng)
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            pm.write(str(td / "a.mid"))
            room = rng.choice([("dry", 0.1, 0.6, 0.1), ("room", 0.4, 0.4, 0.35), ("hall", 0.7, 0.3, 0.55),
                               ("church", 0.95, 0.15, 0.8)])
            subprocess.run(["fluidsynth", "-ni", "-q", "-g", "0.3", "-r", str(FS),
                            "-o", f"synth.reverb.room-size={room[1]}", "-o", f"synth.reverb.damp={room[2]}",
                            "-o", f"synth.reverb.level={room[3]}", "-o", "synth.chorus.active=1",
                            "-F", str(td / "r.wav"), sf2, str(td / "a.mid")], check=True, capture_output=True)
            src = td / "r.wav"
            mp3 = rng.random() < 0.3
            if mp3:
                kbps = rng.choice([64, 96, 128, 192])
                subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(src), "-b:a",
                                f"{kbps}k", str(td / "r.mp3")], check=True)
                src = td / "r.mp3"
            peak_db = rng.uniform(-12, -1)
            tmp = wav.with_suffix(".tmp.wav")
            subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(src), "-af",
                            post_chain(rng) + ",dynaudnorm=f=500:g=31:p=0.95" if rng.random() < 0.2 else post_chain(rng),
                            "-ac", "1", "-ar", str(FS), "-c:a", "pcm_f32le", str(td / "m.wav")], check=True)
            from scipy.io import wavfile

            _, x = wavfile.read(td / "m.wav")
            x = x.astype(np.float32)
            x = x / max(float(np.abs(x).max()), 1e-6) * 10 ** (peak_db / 20)
            wavfile.write(tmp, FS, (x * 32767).astype(np.int16))
            os.replace(tmp, wav)
        dur = len(x) / FS
        ev = ev[ev[:, 0] < dur]
        return {"fam": fam, "id": rid, "notes": ev, "duration": dur, "nSamples": len(x),
                "meta": {"soundfont": Path(sf2).name, "room": room[0], "mp3": int(mp3),
                         "programs": " ".join(map(str, cfg["programs"])), "ranks": cfg["ranks"]}}
    except Exception as e:  # noqa: BLE001
        print("render failed:", fam, rid, str(e)[:200], flush=True)
        return None


def to_transkun(ev):
    from transkun.Data import Note, resolveOverlapping

    notes = [Note(float(s), float(e), int(p), int(np.clip(v, 1, 127))) for s, e, p, v in ev if p > 0 and e > s]
    return resolveOverlapping(notes)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdmx", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sf2", nargs="+", required=True)
    ap.add_argument("--families", default=",".join(DEFAULT_N))
    ap.add_argument("--per-family", default="", help="e.g. organ=89,piano=150 (defaults: %s)" % DEFAULT_N)
    ap.add_argument("--renders", type=int, default=2)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="only print the selection per family")
    args = ap.parse_args()
    fams = args.families.split(",")
    per = dict(DEFAULT_N)
    per.update({k: int(v) for k, v in (x.split("=") for x in args.per_family.split(",") if x)})
    out = Path(args.out)
    rows = candidates(Path(args.pdmx), fams, per, args.seed)
    for f in fams:
        print(f"{f}: {len(rows[f])} scores, {sum(float(r['song_length.seconds']) for r in rows[f]) / 3600:.1f} h",
              flush=True)
    if args.dry_run:
        for f in fams:
            for r in rows[f][:8]:
                print(f"  {f}: {r['title'][:70]} | {r['composer_name']}")
        return
    rng = random.Random(args.seed)
    jobs, info = [], {}
    for f in fams:
        (out / f).mkdir(parents=True, exist_ok=True)
        ids = list(range(len(rows[f])))
        rng.shuffle(ids)
        n_val = max(1, round(args.val_frac * len(ids)))
        for k, j in enumerate(ids):
            r = rows[f][j]
            split = "val" if k < n_val else "train"
            for v in range(args.renders):
                rid = f"{Path(r['mid']).stem[:24]}_{v}"
                info[(f, rid)] = (r, split)
                jobs.append((f, rid, r["_midi"], rng.choice(args.sf2), rng.randrange(2**31), str(out)))
    print(f"{len(jobs)} renders", flush=True)
    done = []
    with Pool(args.workers) as pool:
        for k, res in enumerate(pool.imap_unordered(work, jobs), 1):
            if res:
                done.append(res)
            if k % 50 == 0:
                print(f"{k}/{len(jobs)}", flush=True)
    # (re)build pickles and meta from everything on disk; renders skipped as already done
    # are re-read from their meta rows of a previous run
    for f in fams:
        old = {}
        mpath = out / f / "meta.csv"
        if mpath.exists():
            old = {r["id"]: r for r in csv.DictReader(open(mpath, encoding="utf-8"))}
        new = {d["id"]: d for d in done if d["fam"] == f}
        meta, split_rows = [], {"train": [], "val": []}
        for (ff, rid), (r, split) in info.items():
            if ff != f or not (out / f / f"{rid}.wav").exists():
                continue
            if rid in new:
                d = new[rid]
                notes, dur, ns, m = d["notes"], d["duration"], d["nSamples"], d["meta"]
            elif rid in old:
                from pianocover import events

                notes = events.sanitize(score_events(r["_midi"]))
                dur, ns = float(old[rid]["duration"]), int(old[rid]["nSamples"])
                notes = notes[notes[:, 0] < dur]
                m = {k: old[rid][k] for k in ("soundfont", "room", "mp3", "programs", "ranks")}
            else:
                continue
            split_rows[split].append({"audio_filename": f"{f}/{rid}.wav", "duration": dur, "fs": FS,
                                      "nSamples": ns, "nChannel": 1, "notes": to_transkun(notes),
                                      "source": f"synth_{f}", "split": split})
            meta.append({"id": rid, "split": split, "title": r["title"], "composer": r["composer_name"],
                         "license": r["license"], "pdmx_mid": r["mid"], "duration": round(dur, 2),
                         "nSamples": ns, **m})
        for sp, lst in split_rows.items():
            with open(out / f / f"{sp}.pickle", "wb") as fh:
                pickle.dump(lst, fh, pickle.HIGHEST_PROTOCOL)
        if meta:
            with open(mpath, "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=list(meta[0]))
                w.writeheader()
                w.writerows(meta)
        h = {sp: sum(x["duration"] for x in lst) / 3600 for sp, lst in split_rows.items()}
        print(f"{f}: train {len(split_rows['train'])} ({h['train']:.1f} h), val {len(split_rows['val'])} "
              f"({h['val']:.1f} h)", flush=True)


if __name__ == "__main__":
    main()

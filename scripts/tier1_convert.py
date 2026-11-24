"""Tier 1 (real recordings with note labels, docs/TRANSKUN_PLAN.md) -> transkun training sets.

Raw files from tier1_download.py in /data/tier1/<source>/raw. Each source is written like the
other families: 16-bit mono 44.1 kHz WAV in <out>/<src>/, per-item records in <out>/<src>/items,
meta.csv and transkun DatasetMaestro pickles train/val/test (split by piece, never by excerpt;
test = held-out pieces for phase 6). The audio is left as recorded (no augmentation), only its
peak is set to -1 dBFS. Velocity is unknown in all of these sets and set to 80. Resumable per item.

Sources:
  vocadito   solo voice, 40 excerpts, notes of annotator 1
  guitarset  solo guitar (mono mic), notes from the hexaphonic pickup; split by progression
  musicnet   classical recordings, score-aligned labels (DTW, the original MusicNet labels)
  dagstuhl   choir, stereo room mic (STM, 22.05 kHz in the set), time-aligned score per voice;
             Locus Iste for train/val, Tebe Poem held out as test
  esmuc      choir, room mic (ORTF) where recorded, else the sum of the close mics; manually
             corrected notes of every singer, merged per section; warm-ups (no notes) skipped
  choral     Choral Singing Dataset (Cuesta), 16 singers recorded per section: the full choir
             (all close mics summed) and one quartet variant (one random singer per section),
             notes per section
  cantoria   SATB quartet mix without the organ (the organ has no notes); notes estimated from
             each singer's pYIN F0 (isolated tracks), so weaker labels -> the gate decides

Unison singers of one section give near-identical notes; notes of the same pitch whose onsets are
within 120 ms are merged (median onset and offset).

  python scripts/tier1_convert.py --raw /data/tier1 --out /data/tier1_data vocadito guitarset ...
  python scripts/tier1_convert.py --out /data/tier1_data --gate /data/tier1_gate --sf2 SF2 [sources]
"""
import argparse
import csv
import io
import json
import os
import pickle
import random
import re
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_synth import FS, to_transkun  # noqa: E402
from vocal_synth import resample  # noqa: E402

SOURCES = ["vocadito", "guitarset", "musicnet", "dagstuhl", "esmuc", "choral", "cantoria"]


# ------------------------------------------------------------------ helpers
def hz2midi(hz):
    return 69 + 12 * np.log2(np.maximum(np.asarray(hz, dtype=np.float64), 1e-6) / 440.0)


def read_audio(data):
    """bytes or path -> float32 mono at FS."""
    x, sr = sf.read(io.BytesIO(data) if isinstance(data, bytes) else data, dtype="float32", always_2d=True)
    x = x.mean(1)
    return resample(x, sr, FS) if sr != FS else x


def lab_notes(text, sep=None):
    """onset, Hz, duration lines -> [s, e, midi]."""
    out = []
    for line in text.strip().splitlines():
        try:
            p = [float(v) for v in re.split(r"[,\t ]+", line.strip()) if v]
        except ValueError:  # e.g. ESMUC DH1_SE_short2_S3.lab holds an f0 track ("[3]" columns): no notes
            return []
        if len(p) >= 3 and p[1] > 0 and p[2] > 0:
            out.append([p[0], p[0] + p[2], round(float(hz2midi(p[1])))])
    return out


def merge_unison(notes, tol=0.12):
    """Notes of the same pitch whose onsets are within tol s -> one note (median times)."""
    by = defaultdict(list)
    for s, e, p in notes:
        by[int(p)].append((s, e))
    out = []
    for p, ns in by.items():
        ns.sort()
        group = [ns[0]]
        for n in ns[1:]:
            if n[0] - group[0][0] <= tol:
                group.append(n)
            else:
                out.append([float(np.median([g[0] for g in group])), float(np.median([g[1] for g in group])), p])
                group = [n]
        out.append([float(np.median([g[0] for g in group])), float(np.median([g[1] for g in group])), p])
    out.sort()
    return out


def f0_notes(t, f0, conf=None, min_dur=0.08, jump=0.7, steady=0.05):
    """Monophonic f0 -> notes: voiced runs split where the pitch moves by > jump semitones and stays
    there for >= steady s; pitch = median of the note, rounded."""
    t = np.asarray(t, float)
    f0 = np.asarray(f0, float)
    voiced = f0 > 0
    if conf is not None:
        voiced &= np.asarray(conf, float) > 0.5
    m = np.where(voiced, hz2midi(np.where(voiced, f0, 1)), np.nan)
    hop = float(np.median(np.diff(t))) if len(t) > 1 else 0.01
    k_steady = max(1, int(round(steady / hop)))
    notes, i, n = [], 0, len(t)
    while i < n:
        if np.isnan(m[i]):
            i += 1
            continue
        j = i
        while j < n and not np.isnan(m[j]):
            j += 1
        seg = m[i:j]
        start = 0
        ref = np.median(seg[: k_steady])
        q = 1
        while q < len(seg):
            w = seg[q: q + k_steady]
            if abs(np.median(w) - ref) > jump and len(w) == k_steady and np.all(abs(w - np.median(w)) < jump / 2):
                notes.append((i + start, i + q, seg[start:q]))
                start, ref = q, np.median(w)
                q += k_steady
            else:
                q += 1
        notes.append((i + start, j, seg[start:]))
        i = j
    out = []
    for a, b, seg in notes:
        s, e = t[a], t[b - 1] + hop
        if e - s >= min_dur:
            out.append([float(s), float(e), int(round(np.median(seg)))])
    return out


def split_groups(groups, n_test, n_val, seed, fixed_test=()):
    """groups: list of piece keys -> {key: split}."""
    g = sorted(set(groups))
    rng = random.Random(seed)
    rng.shuffle(g)
    test = set(fixed_test) or set(g[:n_test])
    rest = [k for k in g if k not in test]
    val = set(rest[:n_val])
    return {k: "test" if k in test else "val" if k in val else "train" for k in g}


def clean_sung(notes, gap=0.15, short=0.15):
    """f0 notes of one voice: join same-pitch notes across gaps < `gap` s (consonants, breaths),
    drop fragments shorter than `short` s (slides between notes), then join again."""
    def join(ns, g):
        out = []
        for n in sorted(ns):
            if out and n[2] == out[-1][2] and n[0] - out[-1][1] < g:
                out[-1][1] = max(out[-1][1], n[1])
            else:
                out.append(list(n))
        return out
    return join([n for n in join(notes, gap) if n[1] - n[0] >= short], 2 * gap)


def write_item(out, src, iid, x, notes, split, meta):
    """x: float mono FS; notes [s, e, p] or [s, e, p, v]. Returns hours (0 if it existed)."""
    from pianocover import events

    items = out / src / "items"
    items.mkdir(parents=True, exist_ok=True)
    if (items / f"{iid}.pkl").exists():
        return 0.0
    ev = np.asarray(notes, dtype=np.float64).reshape(-1, len(notes[0]) if len(notes) else 3)
    if ev.shape[1] == 3:
        ev = np.column_stack([ev, np.full(len(ev), 80.0)])
    dur = len(x) / FS
    ev = ev[(ev[:, 0] < dur) & (ev[:, 1] > ev[:, 0])]
    ev[:, 1] = np.minimum(ev[:, 1], dur)
    ev = events.sanitize(ev[np.argsort(ev[:, 0], kind="stable")])
    peak = float(np.abs(x).max()) if len(x) else 0.0
    if peak > 0:
        x = x * (10 ** (-1 / 20) / peak)
    wav = out / src / f"{iid}.wav"
    sf.write(str(wav) + ".tmp.wav", x, FS, subtype="PCM_16")
    os.replace(str(wav) + ".tmp.wav", wav)
    rec = {"fam": src, "id": iid, "notes": ev, "duration": dur, "nSamples": len(x), "split": split,
           "meta": {**meta, "n_notes": int(len(ev))}}
    pickle.dump(rec, open(items / f"{iid}.pkl.tmp", "wb"))
    os.replace(items / f"{iid}.pkl.tmp", items / f"{iid}.pkl")
    return dur / 3600


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
    for split in ("train", "val", "test"):
        items = [{"audio_filename": f"{src}/{r['id']}.wav", "duration": r["duration"], "fs": FS,
                  "nSamples": r["nSamples"], "nChannel": 1, "notes": to_transkun(r["notes"]), "source": src,
                  "split": split} for r in recs if r["split"] == split]
        pickle.dump(items, open(out / src / f"{split}.pickle.tmp", "wb"))
        os.replace(out / src / f"{split}.pickle.tmp", out / src / f"{split}.pickle")
    h = {sp: sum(r["duration"] for r in recs if r["split"] == sp) / 3600 for sp in ("train", "val", "test")}
    n = {sp: sum(r["split"] == sp for r in recs) for sp in ("train", "val", "test")}
    print(f"{src}: " + ", ".join(f"{sp} {n[sp]} items {h[sp]:.2f} h" for sp in h), flush=True)


# ------------------------------------------------------------------ sources
def vocadito(raw, out, seed):
    z = zipfile.ZipFile(raw / "vocadito" / "raw" / "vocadito.zip")
    names = [n for n in z.namelist() if not n.startswith("__MACOSX")]
    ids = sorted({int(m.group(1)) for n in names if (m := re.search(r"Audio/vocadito_(\d+)\.wav$", n))})
    sp = split_groups(ids, 6, 3, seed)
    h = 0.0
    for i in ids:
        notes = lab_notes(z.read(f"Annotations/Notes/vocadito_{i}_notesA1.csv").decode())
        h += write_item(out, "vocadito", f"vocadito_{i:02d}", read_audio(z.read(f"Audio/vocadito_{i}.wav")), notes,
                        sp[i], {"annotator": "A1"})
    print(f"vocadito: new {h:.2f} h", flush=True)


def guitarset(raw, out, seed):
    za = zipfile.ZipFile(raw / "guitarset" / "raw" / "annotation.zip")
    zw = zipfile.ZipFile(raw / "guitarset" / "raw" / "audio_mono-mic.zip")
    stems = sorted(Path(n).stem for n in za.namelist() if n.endswith(".jams"))
    prog = {s: s.split("_")[1] for s in stems}  # e.g. SS3-98-C: one progression, played by all 6 players
    # one held-out progression per style for test, one more for val (style = letters before the digit)
    by_style = defaultdict(list)
    for p in sorted(set(prog.values())):
        by_style[re.match(r"[A-Za-z]+", p).group(0)].append(p)
    rng = random.Random(seed)
    test, val = set(), set()
    for st, ps in sorted(by_style.items()):
        rng.shuffle(ps)
        test.add(ps[0])
        if len(ps) > 2:
            val.add(ps[1])
    h = 0.0
    for s in stems:
        j = json.loads(za.read(f"{s}.jams"))
        notes = [[d["time"], d["time"] + d["duration"], round(d["value"])]
                 for a in j["annotations"] if a["namespace"] == "note_midi" for d in a["data"]]
        split = "test" if prog[s] in test else "val" if prog[s] in val else "train"
        h += write_item(out, "guitarset", s, read_audio(zw.read(f"{s}_mic.wav")), notes, split,
                        {"player": s[:2], "progression": prog[s]})
    print(f"guitarset: new {h:.2f} h; test {sorted(test)}, val {sorted(val)}", flush=True)


def musicnet(raw, out, seed):
    tgz = raw / "musicnet" / "raw" / "musicnet.tar.gz"
    meta = {}
    with open(raw / "musicnet" / "raw" / "musicnet_metadata.csv", encoding="utf8") as f:
        for r in csv.DictReader(f):
            meta[r["id"]] = r
    labels, official_test = {}, set()
    cache = out / "musicnet" / "labels.pkl"
    if cache.exists():
        labels, official_test = pickle.load(open(cache, "rb"))
    else:  # pass 1: the label CSVs (small); the archive is a stream, so read it twice
        with tarfile.open(tgz, "r|gz") as t:
            for m in t:
                if m.isfile() and m.name.endswith(".csv") and "_labels/" in m.name:
                    rid = Path(m.name).stem
                    rows = list(csv.DictReader(io.StringIO(t.extractfile(m).read().decode("utf8"))))
                    labels[rid] = [[int(r["start_time"]) / 44100, int(r["end_time"]) / 44100, int(r["note"]),
                                    int(r["instrument"])] for r in rows]
                    if "test_labels" in m.name:
                        official_test.add(rid)
        (out / "musicnet").mkdir(parents=True, exist_ok=True)
        pickle.dump((labels, official_test), open(cache, "wb"))
    # val: whole compositions (all recordings of a piece go to the same split), test = official test
    comp = {rid: (meta.get(rid, {}).get("composer", ""), meta.get(rid, {}).get("composition", rid)) for rid in labels}
    test_comps = {comp[r] for r in official_test}
    rest = sorted({comp[r] for r in labels if comp[r] not in test_comps})
    random.Random(seed).shuffle(rest)
    val_comps = set(rest[:8])
    h = 0.0
    with tarfile.open(tgz, "r|gz") as t:
        for m in t:
            if not (m.isfile() and m.name.endswith(".wav")):
                continue
            rid = Path(m.name).stem
            if rid not in labels or (out / "musicnet" / "items" / f"mn_{rid}.pkl").exists():
                continue
            split = "test" if comp[rid] in test_comps else "val" if comp[rid] in val_comps else "train"
            notes = [[s, e, p] for s, e, p, _ in labels[rid]]
            md = meta.get(rid, {})
            h += write_item(out, "musicnet", f"mn_{rid}", read_audio(t.extractfile(m).read()), notes, split,
                            {"composer": md.get("composer", ""), "composition": md.get("composition", ""),
                             "ensemble": md.get("ensemble", ""),
                             "instruments": " ".join(str(i) for i in sorted({n[3] for n in labels[rid]}))})
    print(f"musicnet: new {h:.2f} h", flush=True)


def dagstuhl(raw, out, seed):
    z = zipfile.ZipFile(raw / "dagstuhl" / "raw" / "DagstuhlChoirSet_V1.2.3.zip")
    root = "DagstuhlChoirSet_V1.2.3"
    reps = defaultdict(list)
    for n in z.namelist():
        m = re.search(r"annotations_csv_scorerepresentation/(DCS_\w+?_Take\d+)_Stereo_STM_([SATB])\.csv$", n)
        if m:
            reps[m.group(1)].append(n)
    h = 0.0
    for take, files in sorted(reps.items()):
        notes = []
        for fn in files:
            for line in z.read(fn).decode().strip().splitlines():
                p = line.split(",")
                notes.append([float(p[0]), float(p[1]), int(round(float(p[2])))])
        piece = take.split("_")[1]  # LI = Locus Iste, TP = Tebe Poem
        split = "test" if piece == "TP" else "val" if take == "DCS_LI_QuartetB_Take05" else "train"
        x = read_audio(z.read(f"{root}/audio_wav_22050_mono/{take}_Stereo_STM.wav"))
        h += write_item(out, "dagstuhl", take, x, notes, split, {"piece": piece, "mic": "STM", "voices": len(files)})
    print(f"dagstuhl: new {h:.2f} h", flush=True)


def esmuc(raw, out, seed):
    z = zipfile.ZipFile(raw / "esmuc" / "raw" / "EsmucChoirDataset_v1.0.0.zip")
    names = set(z.namelist())
    takes = sorted({n.rsplit("_", 1)[0] for n in names if n.endswith(".lab")})
    h = 0.0
    for take in takes:
        piece = take.split("_")[0]
        labs = sorted(n for n in names if n.startswith(take + "_") and n.endswith(".lab")
                      and re.fullmatch(r"[SATB]\d", n[len(take) + 1:-4]))
        notes = []
        for fn in labs:
            notes += lab_notes(z.read(fn).decode())
        notes = merge_unison(notes)
        room = [f"{take}_{m}.wav" for m in ("ORTF", "AB") if f"{take}_{m}.wav" in names]
        if room:
            x = read_audio(z.read(room[0]))
            mic = room[0].rsplit("_", 1)[1][:-4]
        else:
            tracks = [read_audio(z.read(fn[:-4] + ".wav")) for fn in labs if fn[:-4] + ".wav" in names]
            L = max(len(t) for t in tracks)
            x = sum(np.pad(t / max(np.sqrt(np.mean(t ** 2)), 1e-6), (0, L - len(t))) for t in tracks).astype(np.float32)
            mic = "close-sum"
        split = "test" if piece.startswith("DH") else "val" if take == "SC3_FT_take2" else "train"
        h += write_item(out, "esmuc", take, x, notes, split, {"piece": piece, "mic": mic, "singers": len(labs)})
    print(f"esmuc: new {h:.2f} h", flush=True)


def choral(raw, out, seed):
    z = zipfile.ZipFile(raw / "csd" / "raw" / "ChoralSingingDataset.zip")
    root = "ChoralSingingDataset"
    rng = random.Random(f"{seed}-choral")
    h = 0.0
    for piece in ("ER", "LI", "ND"):  # El Rossinyol, Locus Iste, Nino Dios (test)
        notes = []
        for sec in ("soprano", "alto", "tenor", "bass"):
            notes += lab_notes(z.read(f"{root}/CSD_{piece}_{sec}_notes.lab").decode())
        tracks = {(sec, k): read_audio(z.read(f"{root}/CSD_{piece}_{sec}_{k}.wav"))
                  for sec in ("soprano", "alto", "tenor", "bass") for k in (1, 2, 3, 4)}
        L = max(len(t) for t in tracks.values())

        def mix(keys):
            return sum(np.pad(tracks[k] / max(np.sqrt(np.mean(tracks[k] ** 2)), 1e-6), (0, L - len(tracks[k])))
                       for k in keys).astype(np.float32)

        split = "test" if piece == "ND" else "train"
        h += write_item(out, "choral", f"CSD_{piece}_full", mix(tracks), notes, split, {"piece": piece, "singers": 16})
        quartet = [(sec, rng.randint(1, 4)) for sec in ("soprano", "alto", "tenor", "bass")]
        h += write_item(out, "choral", f"CSD_{piece}_quartet", mix(quartet), notes, split,
                        {"piece": piece, "singers": 4, "quartet": " ".join(f"{s}{k}" for s, k in quartet)})
    print(f"choral: new {h:.2f} h", flush=True)


def cantoria(raw, out, seed):
    z = zipfile.ZipFile(raw / "cantoria" / "raw" / "CantoriaDataset_v1.0.0.zip")
    songs = sorted({m.group(1) for n in z.namelist() if (m := re.match(r"Audio/Cantoria_(\w+?)_Mix\.wav$", n))})
    h = 0.0
    for song in songs:
        notes = []
        for v in "SATB":
            rows = [r.split(",") for r in z.read(f"F0_pyin/Cantoria_{song}_{v}.csv").decode().strip().splitlines()]
            rows = [r for r in rows if r and re.match(r"-?\d", r[0])]
            arr = np.array([[float(c) for c in r[:3]] for r in rows])
            # voiced = pYIN's own decision (f0 > 0, unvoiced frames are negative). The third column is
            # the voicing probability, about 0.25 in sung notes: thresholding it at 0.5 (first build,
            # 2026-11-23) kept 7% of the frames and gave short random fragments.
            notes += clean_sung(f0_notes(arr[:, 0], arr[:, 1], None))
        split = "test" if song in ("VBP", "YSM") else "val" if song == "THM" else "train"
        h += write_item(out, "cantoria", f"Cantoria_{song}", read_audio(z.read(f"Audio/Cantoria_{song}_Mix.wav")),
                        notes, split, {"song": song, "labels": "pYIN f0 -> notes (estimated)"})
    print(f"cantoria: new {h:.2f} h", flush=True)


# ------------------------------------------------------------------ listening gate
def ff(*a):
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", *a], check=True)


def gate(out, gdir, sf2, src, n, seed, seg=45.0):
    """n control files per source: <src>_<id>.mp3 (the recording excerpt), _control.mp3 (left: recording,
    right: the labels played on a piano), _target.mid."""
    import pretty_midi

    recs = [pickle.load(open(f, "rb")) for f in sorted((out / src / "items").glob("*.pkl"))]
    rng = random.Random(f"{seed}-gate-{src}")
    rng.shuffle(recs)
    # prefer different pieces and at least one test item
    pick, seen = [], set()
    for r in sorted(recs, key=lambda r: r["split"] != "test"):
        key = r["meta"].get("piece") or r["meta"].get("composition") or r["meta"].get("progression") or r["id"]
        if key in seen:
            continue
        seen.add(key)
        pick.append(r)
        if len(pick) >= n:
            break
    gdir.mkdir(parents=True, exist_ok=True)
    for r in pick:
        x, _ = sf.read(out / src / f"{r['id']}.wav", dtype="float32")
        dur = len(x) / FS
        b = rng.uniform(0, max(0.0, dur - seg))
        ev = r["notes"]
        sel = ev[(ev[:, 1] > b) & (ev[:, 0] < b + seg)]
        name = f"{src}_{r['id']}"
        with tempfile.TemporaryDirectory() as td:
            sf.write(f"{td}/a.wav", x[int(b * FS): int((b + seg) * FS)], FS, subtype="PCM_16")
            pm = pretty_midi.PrettyMIDI()
            inst = pretty_midi.Instrument(0, name="labels")
            inst.notes = [pretty_midi.Note(90, int(p), float(max(s - b, 0)), float(min(e - b, seg)))
                          for s, e, p, _ in sel if min(e - b, seg) > max(s - b, 0)]
            pm.instruments.append(inst)
            pm.write(str(gdir / f"{name}_target.mid"))
            subprocess.run(["fluidsynth", "-ni", "-q", "-g", "0.5", "-r", str(FS), "-F", f"{td}/t.wav", sf2,
                            str(gdir / f"{name}_target.mid")], check=True, capture_output=True)
            ff("-i", f"{td}/a.wav", "-b:a", "192k", str(gdir / f"{name}.mp3"))
            ff("-i", f"{td}/a.wav", "-i", f"{td}/t.wav", "-filter_complex",
               "[0:a]aformat=channel_layouts=mono,loudnorm[a];[1:a]aformat=channel_layouts=mono,loudnorm[b];"
               "[a][b]join=inputs=2:channel_layout=stereo", "-b:a", "192k", str(gdir / f"{name}_control.mp3"))
        print(f"gate {name} [{r['split']}] at {b:.0f}s, {len(inst.notes)} notes", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default="/data/tier1")
    ap.add_argument("--out", default="/data/tier1_data")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gate", help="write listening-gate control files to this folder instead of converting")
    ap.add_argument("--gate-n", type=int, default=4)
    ap.add_argument("--sf2", default="/data/ext/sf2/GeneralUser-GS.sf2")
    ap.add_argument("sources", nargs="+", choices=SOURCES)
    args = ap.parse_args()
    raw, out = Path(args.raw), Path(args.out)
    for s in args.sources:
        if args.gate:
            gate(out, Path(args.gate), args.sf2, s, args.gate_n, args.seed)
        else:
            globals()[s](raw, out, args.seed)
            finalize(out, s)


if __name__ == "__main__":
    main()

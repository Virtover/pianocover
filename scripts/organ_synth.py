"""Synthetic 1:1 organ pairs: public-domain organ scores rendered with several organ sounds.

Input audio = the score rendered by fluidsynth with an organ preset; target = the very same
notes (folded into the piano's 88 keys). Audio and MIDI match exactly by construction, so no
alignment or filtering is needed (unlike web covers or real recordings + scores).

Variety per render (seeded, recorded in meta.csv):
  * soundfont: GeneralUser GS, TimGM6mb, FluidR3_GM
  * organ preset per track: Church Organ (19) most often, also Reed (20), Drawbar (16),
    Percussive (17); the lowest track (pedal) may get a different preset than the manuals
  * room: fluidsynth reverb size / damping / level, from dry to a large church
  * level: rendered quietly, then peak-normalised to a random level

Scores: PDMX (CC BY 4.0 dataset; the scores' own licenses are publicdomain / cc-zero),
no_license_conflict + deduplicated, 45 s - 10 min, organ-only scores (GM 16-20) plus Bach's
organ works (BWV 525-771) scored for keyboard; obvious modern / pop arrangements excluded.

  python scripts/organ_synth.py --pdmx /data/pdmx --out /data/synth/organ --pieces 250 --renders 2 \
      --sf2 /sf2/GeneralUser-GS.sf2 /sf2/TimGM6mb.sf2 /usr/share/sounds/sf2/FluidR3_GM.sf2

Output: <out>/apc/{spec/, train.pkl, val.pkl, style_vectors.json} in pianocover.apc's format
(source "organ_synth"), <out>/audio/<id>.flac (16 kHz, for listening), <out>/meta.csv.
"""
import argparse
import csv
import io
import json
import pickle
import random
import re
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pdmx_organ import score_events  # noqa: E402

from pianocover import events  # noqa: E402
from pianocover.apc import MARGIN, SR  # noqa: E402
from pianocover.apc.features import logmel  # noqa: E402
from pianocover.apc.labels import pedal_extend, style_vector  # noqa: E402
from pianocover.apc.prepare import padded_spec  # noqa: E402

MODERN = re.compile(r"langlais|howells|messiaen|duruflé|durufle|\balain\b|dupré|dupre|fonsi|zimmer|"
                    r"john williams|myself|^composer$|\(wip\)|\bwip\b|virgil fox|interstellar|"
                    r"hillsong|tomlin|redman|maher|getty|townend|pokemon|pokémon|zelda|mario|minecraft|"
                    r"undertale|anime|disney|star wars|harry potter|game of thrones|tetris|zorba|theodorakis|"
                    r"let it go|piano guys|frozen|titanic|pirates of the caribbean|tribute to the", re.I)
# composers who died before 1955 (or a pre-1900 year in the composer field): the PDMX license
# covers the engraving; this keeps the composition public domain too (the first smoke test
# without it picked Naruto, Green Onions and Profondo Rosso by view count)
OLD = re.compile(r"\bbach\b|buxtehude|pachelbel|widor|vierne|guilmant|boellmann|boëllmann|rheinberger|karg-elert|"
                 r"frescobaldi|sweelinck|scheidt|walther|böhm|bruhns|lübeck|stanley|franck|reger|lefébure|"
                 r"lefebure|dubois|gigout|boely|boëly|tournemire|mendelssohn|brahms|liszt|krebs|zipoli|cabez|"
                 r"titelouze|couperin|grigny|clérambault|clerambault|marchand|dandrieu|daquin|stainer|stanford|"
                 r"parry|wesley|boyce|handel|händel|haendel|telemann|purcell|tallis|byrd|palestrina|schütz|"
                 r"schutz|goudimel|praetorius|hassler|crüger|cruger|neander|luther|gruber|mozart|haydn|"
                 r"beethoven|schubert|bortniansky|elgar|gounod|fauré|faure|saint-saëns|saint-saens|dykes|"
                 r"sullivan|barnby|lowell mason|bradbury|doane|kirkpatrick|sankey|lowry|zundel|scheidemann|"
                 r"froberger|kerll|muffat|pasquini|merulo|gabrieli|cabanilles|correa|arauxo|vivaldi|corelli|"
                 r"albinoni|marcello|scarlatti|walond|croft|attwood|"
                 r"lemmens|hesse|merkel|töpfer|ritter|bruckner|dvořák|dvorak|grieg|tchaikovsky|"
                 r"rachmaninoff|chopin|schumann|vaughan williams|holst|bizet|verdi|wagner|puccini|"
                 r"\b1[0-8]\d\d\b", re.I)
PRESETS = [(19, 0.55), (20, 0.15), (16, 0.15), (17, 0.15)]


def progs(t):
    return [int(x) for x in re.findall(r"-?\d+", str(t))]


def select(pdmx: Path, n: int, seed: int):
    import pandas as pd

    df = pd.read_csv(pdmx / "PDMX.csv", low_memory=False)
    df = df[df["subset:no_license_conflict"] & df["subset:deduplicated"]
            & df["song_length.seconds"].between(45, 600) & (df["n_notes"] >= 200)]
    txt = (df.title.fillna("") + " | " + df.composer_name.fillna("") + " | " + df.artist_name.fillna(""))
    b = txt.str.lower().str.extract(r"bwv\.?\s*0*(\d{3,4})")[0].astype(float)
    organ_only = df["tracks"].map(lambda t: (p := progs(t)) != [] and all(16 <= x <= 20 for x in p))
    keyboard = df["tracks"].map(lambda t: (p := progs(t)) != [] and all(x <= 7 or 16 <= x <= 20 for x in p))
    pick = ((organ_only & txt.str.contains(OLD)) | (keyboard & b.between(525, 771))) & ~txt.str.contains(MODERN)
    df = df[pick].sort_values("n_views", ascending=False)
    return df.head(n)


def pick_preset(rng):
    r, acc = rng.random(), 0.0
    for p, w in PRESETS:
        acc += w
        if r < acc:
            return p
    return 19


def render(midi_bytes, sf2, rng, tmp: Path):
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(io.BytesIO(midi_bytes))
    manual = pick_preset(rng)
    pedal = manual if rng.random() < 0.6 else pick_preset(rng)
    insts = [i for i in pm.instruments if not i.is_drum and i.notes]
    pm.instruments = insts
    low = min(insts, key=lambda i: np.mean([n.pitch for n in i.notes])) if len(insts) > 1 else None
    for i in insts:
        i.program = pedal if i is low else manual
        i.control_changes = [c for c in i.control_changes if c.number not in (0, 32)]  # no bank switches
    mid = tmp / "in.mid"
    pm.write(str(mid))
    room = rng.choice([("dry", 0.1, 0.6, 0.15), ("room", 0.4, 0.4, 0.4), ("hall", 0.7, 0.3, 0.6),
                       ("church", 0.95, 0.15, 0.8)])
    wav = tmp / "out.wav"
    subprocess.run(["fluidsynth", "-ni", "-q", "-g", "0.3", "-r", "44100", "-R", "1", "-C", "0",
                    "-o", f"synth.reverb.room-size={room[1]}", "-o", f"synth.reverb.damp={room[2]}",
                    "-o", f"synth.reverb.level={room[3]}", "-o", "synth.reverb.width=1.0",
                    "-F", str(wav), str(sf2), str(mid)], check=True, capture_output=True)
    return wav, {"manual": manual, "pedal": pedal, "room": room[0]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdmx", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sf2", nargs="+", required=True)
    ap.add_argument("--pieces", type=int, default=250)
    ap.add_argument("--renders", type=int, default=2)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    from pianocover.audio import decode

    out = Path(args.out)
    for d in ("apc/spec", "audio", "tmp"):
        (out / d).mkdir(parents=True, exist_ok=True)
    sel = select(Path(args.pdmx), args.pieces, args.seed)
    want = {m.lstrip("./"): i for i, m in zip(sel.index, sel["mid"])}
    mids = {}
    with tarfile.open(Path(args.pdmx) / "mid.tar.gz") as tar:
        for m in tar:
            name = m.name.lstrip("./")
            if m.isfile() and name in want:
                mids[want[name]] = tar.extractfile(m).read()
    print(f"{len(sel)} scores selected, {len(mids)} MIDI files found", flush=True)
    rng = random.Random(args.seed)
    ids = [i for i in sel.index if i in mids]
    rng.shuffle(ids)
    n_val = max(1, int(round(args.val_frac * len(ids))))
    split_of = {i: ("val" if k < n_val else "train") for k, i in enumerate(ids)}
    rows = {"train": [], "val": []}
    svs, meta = {}, []
    for k, i in enumerate(ids, 1):
        s = sel.loc[i]
        sid = "orgsyn_" + Path(s["mid"]).stem[:20]
        try:
            ev = events.sanitize(score_events(mids[i]))
        except Exception as e:  # noqa: BLE001
            print("bad midi:", s["title"], e, flush=True)
            continue
        notes = pedal_extend(ev)
        if len(notes) < 200:
            continue
        for r in range(args.renders):
            rid = f"{sid}_{r}"
            sf2 = Path(rng.choice(args.sf2))
            try:
                wav, cfg = render(mids[i], sf2, rng, out / "tmp")
                x = decode(wav, SR)
            except Exception as e:  # noqa: BLE001
                print("render failed:", s["title"], e, flush=True)
                continue
            peak = float(np.abs(x).max()) or 1.0
            x = x / peak * rng.uniform(0.3, 0.95)
            spec = padded_spec(logmel(x, args.device))
            np.save(out / "apc" / "spec" / f"{rid}.npy", spec)
            subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-f", "f32le", "-ar", str(SR),
                            "-ac", "1", "-i", "-", str(out / "audio" / f"{rid}.flac")],
                           input=x.astype(np.float32).tobytes(), check=True)
            n_frames = spec.shape[0] - 2 * MARGIN  # as in pianocover.apc.prepare
            dur = len(x) / SR
            sv = style_vector(notes, dur)
            svs[rid] = sv.tolist()
            rows[split_of[i]].append({"pop_id": rid, "piano_id": rid, "spec": f"{rid}.npy",
                                      "n_frames": n_frames, "notes": notes.astype(np.float32), "sv": sv,
                                      "bad_regions": [], "source": "organ_synth"})
            meta.append({"id": rid, "split": split_of[i], "title": s["title"], "composer": s["composer_name"],
                         "license": s["license"], "pdmx_mid": s["mid"], "soundfont": sf2.name,
                         "duration": round(dur, 1), **cfg})
        if k % 20 == 0:
            print(f"{k}/{len(ids)} scores, {len(meta)} renders", flush=True)
    for sp, r in rows.items():
        with open(out / "apc" / f"{sp}.pkl", "wb") as f:
            pickle.dump(r, f)
        print(f"{sp}: {len(r)} renders, {sum(x['n_frames'] for x in r) / 62.5 / 3600:.1f} h", flush=True)
    (out / "apc" / "style_vectors.json").write_text(json.dumps(svs))
    with open(out / "meta.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(meta[0]))
        w.writeheader()
        w.writerows(meta)


if __name__ == "__main__":
    main()

"""Synthetic 1:1 transcription data with a real-sounding singing voice (tier 3, "vocal" family).

A public-domain PDMX song / choral score is split into vocal lines (tracks named Voice, Soprano,
Alto, Tenor, Bass, ...) and accompaniment. The vocal lines are sung by SoulX-Singer (neural
singing voice synthesis, Apache-2.0, score control: notes only, the model makes its own
vibrato and pitch transitions); the accompaniment is rendered by fluidsynth as in
score_synth.py. Lyrics: PDMX MIDI files carry none, so words of public-domain English texts
(hymns, KJV psalms) are laid onto the notes, one syllable per note; the words need not match
the original text, only sound like singing.

Target: every pitched note of the accompaniment (score) plus the sung notes (each vocal line
made monophonic, highest note wins), the whole piece transposed by the same random shift as
the audio. Drums never. Same output format as score_synth.py (16-bit mono 44.1 kHz WAV,
meta.csv, transkun pickles), plus, with --listen, an MP3 of the mix, the target MIDI and a
control MP3 (mix left, target rendered with a piano right).

  python scripts/vocal_synth.py --pdmx /data/pdmx --out /data/synth_vocal --ids 40679,190160 \
      --sf2 /data/ext/sf2/*.sf2 --soulx /data/ext/soulx [--listen /data/synth_vocal_listen]
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
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_synth import FS, audio_midi, post_chain, to_transkun  # noqa: E402

VOX = re.compile(r"(?i)\b(voice|vocal|voix|stimme|singstimme|soprano|sopran|alto|alt|contralto|tenor|tenore|"
                 r"bass|basso|basse|baritone|bariton|baryton|mezzo|mezzosoprano|canto|chor|choir|chorus|"
                 r"[satb]\.)\b")
# example prompts shipped in the SoulX-Singer repo (audio + metadata); median MIDI pitch
PROMPTS = {"en_prompt": 65, "zh_prompt": 67, "en_target": 65, "zh_target": 53, "yue_target": 57}
SVS_SR = 24000
# public-domain English texts (authors died before 1900; KJV 1611)
TEXTS = """
Amazing grace how sweet the sound that saved a wretch like me I once was lost but now am found
was blind but now I see Twas grace that taught my heart to fear and grace my fears relieved
How precious did that grace appear the hour I first believed
O God our help in ages past our hope for years to come our shelter from the stormy blast
and our eternal home Under the shadow of thy throne thy saints have dwelt secure
sufficient is thine arm alone and our defence is sure
The Lord is my shepherd I shall not want He maketh me to lie down in green pastures he leadeth me
beside the still waters He restoreth my soul he leadeth me in the paths of righteousness
Holy holy holy Lord God Almighty early in the morning our song shall rise to thee
Holy holy holy merciful and mighty God in three persons blessed Trinity
Abide with me fast falls the eventide the darkness deepens Lord with me abide
When other helpers fail and comforts flee help of the helpless O abide with me
Joy to the world the Lord is come let earth receive her King let every heart prepare him room
and heaven and nature sing O come all ye faithful joyful and triumphant
Silent night holy night all is calm all is bright round yon virgin mother and child
Make a joyful noise unto the Lord all ye lands Serve the Lord with gladness come before his presence
with singing Know ye that the Lord he is God it is he that hath made us and not we ourselves
""".split()


def is_vocal(inst):
    name = (inst.name or "").replace("\x00", "")
    if inst.is_drum or inst.program in range(0, 8) and not VOX.search(name):
        return False
    return bool(VOX.search(name)) or inst.program in (52, 53, 54, 85)


def mono_line(notes, min_dur=0.08):
    """Highest note wins at overlaps; ends are clipped at the next onset. [start, end, pitch]."""
    notes = sorted(notes, key=lambda n: (n.start, -n.pitch))
    out = []
    for n in notes:
        if out and n.start < out[-1][0] + 1e-3:  # chord in one voice: keep the top note
            continue
        if out and n.start < out[-1][1]:
            out[-1][1] = n.start
        out.append([n.start, n.end, n.pitch])
    return [x for x in out if x[1] - x[0] >= min_dur]


class Lyrics:
    def __init__(self, rng):
        from g2p_en import G2p

        self.g2p = G2p()
        self.rng = rng
        self.pos = rng.randrange(len(TEXTS))
        self.cache = {}

    def phones(self, word):
        if word not in self.cache:
            ph = [p for p in self.g2p(word) if re.fullmatch(r"[A-Z]+[0-2]?", p)]
            self.cache[word] = ph
        return self.cache[word]

    def next_word(self, max_syl):
        for _ in range(50):
            w = TEXTS[self.pos % len(TEXTS)]
            self.pos += 1
            ph = self.phones(w.lower())
            syl = sum(p[-1].isdigit() for p in ph)
            if ph and 1 <= syl <= max_syl:
                return w, ph, syl
        return "la", ["L", "AA1"], 1


def line_meta(line, lyrics, language="English", max_seg=12.0, gap=0.3):
    """SoulX-Singer target metadata: segments split at rests, each a list of note tokens."""
    segs, cur = [], []
    for n in line:
        if cur and (n[0] - cur[-1][1] >= gap or n[1] - cur[0][0] > max_seg):
            segs.append(cur)
            cur = []
        cur.append(n)
    if cur:
        segs.append(cur)
    out = []
    for k, seg in enumerate(segs):
        t0, t1 = max(0.0, seg[0][0] - 0.2), seg[-1][1] + 0.2
        dur, text, phon, pitch, ntype = [], [], [], [], []

        def rest(d):
            if d > 0.005:
                dur.append(d); text.append("<SP>"); phon.append("<SP>"); pitch.append(0); ntype.append(1)

        rest(seg[0][0] - t0)
        i = 0
        while i < len(seg):
            # a word may span several notes, but not a repeated pitch (that onset would vanish)
            room = 1
            while i + room < len(seg) and seg[i + room][2] != seg[i + room - 1][2] and room < 3 \
                    and seg[i + room][0] - seg[i + room - 1][1] < 0.02:
                room += 1
            w, ph, syl = lyrics.next_word(room)
            p = "en_" + "-".join(ph)
            for j in range(syl):
                s, e, m = seg[i + j]
                if j > 0:
                    rest(s - seg[i + j - 1][1])
                dur.append(e - s); text.append(w); phon.append(p); pitch.append(int(m)); ntype.append(2 if j == 0 else 3)
            if i + syl < len(seg):
                rest(seg[i + syl][0] - seg[i + syl - 1][1])
            i += syl
        rest(t1 - seg[-1][1])
        out.append({"index": f"seg_{k}", "language": language, "time": [int(t0 * 1000), int(t0 * 1000 + sum(dur) * 1000)],
                    "duration": " ".join(f"{d:.3f}" for d in dur), "text": " ".join(text),
                    "phoneme": " ".join(phon), "note_pitch": " ".join(map(str, pitch)),
                    "note_type": " ".join(map(str, ntype))})
    return out


class Singer:
    def __init__(self, soulx, device="cuda"):
        import torch

        sys.path.insert(0, soulx)
        from soulxsinger.models.soulxsinger import SoulXSinger
        from soulxsinger.utils.data_processor import DataProcessor
        from soulxsinger.utils.file_utils import load_config

        self.root = Path(soulx)
        self.cfg = load_config(str(self.root / "soulxsinger/config/soulxsinger.yaml"))
        # weights_only=True: the checkpoint is downloaded, so no arbitrary pickle code is run
        ck = torch.load(self.root / "pretrained_models/SoulX-Singer/model.pt", map_location="cpu", weights_only=True)
        m = SoulXSinger(self.cfg)
        m.load_state_dict(ck["state_dict"], strict=True)
        m = m.half().to(device).eval()
        m.mel.float()
        self.model = m
        self.dp = DataProcessor(self.cfg.audio.hop_size, self.cfg.audio.sample_rate,
                                str(self.root / "soulxsinger/utils/phoneme/phone_set.json"), device)
        self.prompts = {}

    def prompt(self, name):
        if name not in self.prompts:
            d = self.root / "example/audio"
            meta = json.load(open(d / f"{name}.json", encoding="utf8"))[0]
            self.prompts[name] = self.dp.process(meta, str(d / f"{name}.mp3"))
        return self.prompts[name]

    def sing(self, metas, prompt, total_s, seed):
        import torch

        torch.manual_seed(seed)
        y = np.zeros(int(total_s * SVS_SR) + SVS_SR, dtype=np.float32)
        for m in metas:
            tgt = self.dp.process(dict(m), None)
            with torch.no_grad():
                a = self.model.infer({"prompt": self.prompt(prompt), "target": tgt}, auto_shift=False, pitch_shift=0,
                                     n_steps=self.cfg.infer.n_steps, cfg=self.cfg.infer.cfg, control="score",
                                     use_fp16=True)
            a = a.squeeze().float().cpu().numpy()
            s = int(m["time"][0] / 1000 * SVS_SR)
            n = min(len(a), len(y) - s)
            # short fades so segment edges do not click
            f = min(240, n // 4)
            a = a[:n].copy()
            a[:f] *= np.linspace(0, 1, f)
            a[n - f:n] *= np.linspace(1, 0, f)
            y[s:s + n] += a
        return y


def synth_ir(rng, fs=FS):
    """Room IR as decaying stereo-free noise (no licence issue): (rt60 s, wet level)."""
    rt = rng.uniform(0.3, 2.2)
    n = int(rt * fs)
    t = np.arange(n) / fs
    ir = rng_np(rng).standard_normal(n) * np.exp(-6.9 * t / rt)
    ir[: int(0.01 * fs)] *= np.linspace(0, 1, int(0.01 * fs))
    ir /= np.sqrt((ir ** 2).sum())
    return ir, rt


def rng_np(rng):
    return np.random.default_rng(rng.randrange(2**31))


def resample(x, sr_in, sr_out):
    from scipy.signal import resample_poly

    g = np.gcd(sr_in, sr_out)
    return resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)


def render_acc(pm, sf2, rng, td):
    room = rng.choice([("dry", 0.1, 0.6, 0.1), ("room", 0.4, 0.4, 0.35), ("hall", 0.7, 0.3, 0.55),
                       ("church", 0.95, 0.15, 0.8)])
    if not any(i.notes for i in pm.instruments):
        return None, room[0]
    pm.write(str(td / "acc.mid"))
    subprocess.run(["fluidsynth", "-ni", "-q", "-g", "0.3", "-r", str(FS),
                    "-o", f"synth.reverb.room-size={room[1]}", "-o", f"synth.reverb.damp={room[2]}",
                    "-o", f"synth.reverb.level={room[3]}", "-o", "synth.chorus.active=1",
                    "-F", str(td / "acc.wav"), sf2, str(td / "acc.mid")], check=True, capture_output=True)
    from scipy.io import wavfile

    _, x = wavfile.read(td / "acc.wav")
    scale = 32768.0 if x.dtype == np.int16 else 1.0
    x = x.astype(np.float32) / scale
    if x.ndim == 2:
        x = x.mean(1)
    return x, room[0]


def rms(x):
    v = x[np.abs(x) > 1e-4]
    return float(np.sqrt((v ** 2).mean())) if len(v) else 1e-6


def work(row, midi_bytes, singer, sf2, seed, out, listen):
    import pretty_midi
    from scipy.io import wavfile
    from scipy.signal import fftconvolve

    from pianocover import events

    rng = random.Random(seed)
    rid = f"{row['_id']}_{seed % 1000}"
    pm = pretty_midi.PrettyMIDI(io.BytesIO(midi_bytes))
    vox = [i for i in pm.instruments if is_vocal(i) and i.notes]
    if not vox:
        return None
    lines = [mono_line(i.notes) for i in vox]
    # one transposition for the whole piece, so the top line sits near a female prompt's range
    allp = [n[2] for ln in lines for n in ln]
    shift = int(np.clip(round(65 - np.median([n[2] for n in lines[0]])), -5, 5)) + rng.randint(-2, 2)
    if len(lines) > 1:  # choir: keep the original key area
        shift = rng.randint(-2, 2)
    shift = int(np.clip(shift, 21 - min(allp), 108 - max(allp)))
    lyr = Lyrics(rng)
    # accompaniment: everything that is not a vocal track, rendered like score_synth
    for i in vox:
        pm.instruments.remove(i)
    for inst in pm.instruments:
        if not inst.is_drum:
            for n in inst.notes:
                n.pitch = int(np.clip(n.pitch + shift, 0, 127))
    buf = io.BytesIO()
    pm.write(buf)
    apm, cfg = audio_midi(buf.getvalue(), "chamber", rng)
    total = max([n[1] for ln in lines for n in ln] + [apm.get_end_time()]) + 1.0
    used = []
    voices = []
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        acc, room = render_acc(apm, sf2, rng, td)
        n_out = int(total * FS)
        mix = np.zeros(n_out, dtype=np.float32)
        if acc is not None:
            mix[: min(n_out, len(acc))] += acc[:n_out]
        acc_rms = rms(mix) if acc is not None else 0.05
        ir, rt = synth_ir(rng)
        wet = rng.uniform(0.1, 0.45)
        vox_sum = np.zeros(n_out, dtype=np.float32)
        for k, ln in enumerate(lines):
            ln = [[s, e, p + shift] for s, e, p in ln]
            med = np.median([n[2] for n in ln])
            pool = [p for p, m in PROMPTS.items() if abs(m - med) <= 7 and p not in used] or \
                   sorted(PROMPTS, key=lambda p: abs(PROMPTS[p] - med))[:2]
            prompt = rng.choice(pool)
            used.append(prompt)
            metas = line_meta(ln, lyr)
            y = resample(singer.sing(metas, prompt, total, rng.randrange(2**31)), SVS_SR, FS)[:n_out]
            y = np.pad(y, (0, n_out - len(y)))
            y = (1 - wet) * y + wet * fftconvolve(y, ir)[:n_out].astype(np.float32)
            gain_db = rng.uniform(-6, 0) if len(lines) > 1 else 0.0
            vox_sum += y / rms(y) * 10 ** (gain_db / 20)
            voices.append({"prompt": prompt, "n": len(ln), "notes": ln})
        # solo voice sits 0..+8 dB over the accompaniment; a choir 0..+4 dB in total
        lead = rng.uniform(0, 8) if len(lines) == 1 else rng.uniform(-2, 4)
        mix += vox_sum / rms(vox_sum) * acc_rms * 10 ** (lead / 20)
        wavfile.write(td / "mix.wav", FS, (mix / max(np.abs(mix).max(), 1e-6) * 0.9).astype(np.float32))
        src = td / "mix.wav"
        mp3 = rng.random() < 0.3
        if mp3:
            kbps = rng.choice([64, 96, 128, 192])
            subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(src), "-b:a", f"{kbps}k",
                            str(td / "mix.mp3")], check=True)
            src = td / "mix.mp3"
        subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(src), "-af", post_chain(rng),
                        "-ac", "1", "-ar", str(FS), "-c:a", "pcm_f32le", str(td / "m.wav")], check=True)
        _, x = wavfile.read(td / "m.wav")
        x = x.astype(np.float32)
        x = x / max(float(np.abs(x).max()), 1e-6) * 10 ** (rng.uniform(-12, -1) / 20)
        wav = Path(out) / "vocal" / f"{rid}.wav"
        wav.parent.mkdir(parents=True, exist_ok=True)
        wavfile.write(wav, FS, (x * 32767).astype(np.int16))
    # target: accompaniment score notes + sung notes, folded into 21-108
    ev = [[n.start, n.end, n.pitch, max(1, n.velocity)] for i in pm.instruments if not i.is_drum for n in i.notes]
    ev += [[s, e, p, 80] for v in voices for s, e, p in v["notes"]]
    ev = np.array(sorted(ev), dtype=np.float64).reshape(-1, 4)
    while (ev[:, 2] < 21).any() or (ev[:, 2] > 108).any():
        ev[ev[:, 2] < 21, 2] += 12
        ev[ev[:, 2] > 108, 2] -= 12
    ev = events.sanitize(ev)
    dur = len(x) / FS
    ev = ev[ev[:, 0] < dur]
    meta = {"soundfont": Path(sf2).name, "room": room, "mp3": int(mp3), "programs": " ".join(map(str, cfg["programs"])),
            "shift": shift, "voices": " ".join(f"{v['prompt']}:{v['n']}" for v in voices), "vocal_ir_rt60": round(rt, 2),
            "vocal_wet": round(wet, 2), "lead_db": round(lead, 1)}
    if listen:
        write_listen(Path(listen), rid, row, wav, ev, sf2)
    return {"fam": "vocal", "id": rid, "notes": ev, "duration": dur, "nSamples": len(x), "meta": meta}


def write_listen(d, rid, row, wav, ev, sf2):
    import pretty_midi

    d.mkdir(parents=True, exist_ok=True)
    name = re.sub(r"[^\w\-]+", "_", str(row["title"]))[:50]
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(0, name="target")
    inst.notes = [pretty_midi.Note(int(v), int(p), float(s), float(e)) for s, e, p, v in ev]
    pm.instruments.append(inst)
    mid = d / f"{name}_{rid}_target.mid"
    pm.write(str(mid))
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(wav), "-b:a", "192k",
                    str(d / f"{name}_{rid}.mp3")], check=True)
    with tempfile.TemporaryDirectory() as td:
        subprocess.run(["fluidsynth", "-ni", "-q", "-g", "0.5", "-r", str(FS), "-F", f"{td}/t.wav", sf2, str(mid)],
                       check=True, capture_output=True)
        subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(wav), "-i", f"{td}/t.wav",
                        "-filter_complex", "[0:a]aformat=channel_layouts=mono,loudnorm[a];"
                        "[1:a]aformat=channel_layouts=mono,loudnorm[b];[a][b]join=inputs=2:channel_layout=stereo",
                        "-b:a", "192k", str(d / f"{name}_{rid}_control.mp3")], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdmx", default="/data/pdmx")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ids", required=True, help="comma-separated PDMX.csv row indices")
    ap.add_argument("--sf2", nargs="+", required=True)
    ap.add_argument("--soulx", default="/data/ext/soulx")
    ap.add_argument("--listen")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-every", type=int, default=20)
    args = ap.parse_args()

    import pandas as pd
    import torch

    torch.cuda.set_per_process_memory_fraction(0.6)
    df = pd.read_csv(Path(args.pdmx) / "PDMX.csv", low_memory=False)
    ids = [int(x) for x in args.ids.split(",")]
    rows = {i: dict(df.loc[i], _id=Path(df.loc[i, "mid"]).stem[:24]) for i in ids}
    want = {rows[i]["mid"].lstrip("./"): i for i in ids}
    mids = {}
    with tarfile.open(Path(args.pdmx) / "mid.tar.gz") as tar:
        for m in tar:
            if m.isfile() and m.name.lstrip("./") in want:
                mids[want[m.name.lstrip("./")]] = tar.extractfile(m).read()
    singer = Singer(args.soulx)
    out = Path(args.out)
    rng = random.Random(args.seed)
    results = []
    for k, i in enumerate(ids):
        seed = rng.randrange(2**31)
        r = work(rows[i], mids[i], singer, rng.choice(args.sf2), seed, out, args.listen)
        if r:
            r["split"] = "val" if k % args.val_every == args.val_every - 1 else "train"
            r["row"] = rows[i]
            results.append(r)
            print("done", i, rows[i]["title"], r["meta"], flush=True)
    with open(out / "vocal" / "meta.csv", "a", newline="", encoding="utf8") as f:
        w = csv.writer(f)
        if f.tell() == 0:
            w.writerow(["id", "split", "title", "composer", "license", "pdmx_mid", "duration", "nSamples",
                        *results[0]["meta"].keys()])
        for r in results:
            w.writerow([r["id"], r["split"], r["row"]["title"], r["row"]["composer_name"], r["row"]["license"],
                        r["row"]["mid"], round(r["duration"], 2), r["nSamples"], *r["meta"].values()])
    for split in ("train", "val"):
        items = [{"audio_filename": f"vocal/{r['id']}.wav", "duration": r["duration"], "fs": FS,
                  "nSamples": r["nSamples"], "nChannel": 1, "notes": to_transkun(r["notes"]), "source": "vocal",
                  "split": split} for r in results if r["split"] == split]
        p = out / "vocal" / f"{split}.pickle"
        old = pickle.load(open(p, "rb")) if p.exists() else []
        names = {it["audio_filename"] for it in items}
        pickle.dump([o for o in old if o["audio_filename"] not in names] + items, open(p, "wb"))


if __name__ == "__main__":
    main()

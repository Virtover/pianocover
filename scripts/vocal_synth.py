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
UNDOUBLE_P = 0.7  # share of renders whose accompaniment loses the notes doubling the voice
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


def load_wav_ffmpeg(path, sample_rate):
    """Mono float waveform (1, T) at sample_rate, decoded by ffmpeg."""
    import torch

    raw = subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(path), "-ac", "1", "-ar",
                          str(sample_rate), "-f", "f32le", "-"], check=True, capture_output=True).stdout
    return torch.from_numpy(np.frombuffer(raw, dtype=np.float32).copy()).unsqueeze(0)


class Singer:
    def __init__(self, soulx, device="cuda"):
        import torch

        sys.path.insert(0, soulx)
        from soulxsinger.models.soulxsinger import SoulXSinger
        from soulxsinger.utils.data_processor import DataProcessor
        from soulxsinger.utils.file_utils import load_config

        import soulxsinger.utils.data_processor as dpm

        dpm.load_wav = load_wav_ffmpeg  # torchaudio 2.9 needs torchcodec for load(); not installed
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
        self.prompt_dir = None  # None: SoulX repo examples; else a vocadito_prompts.py output dir

    def prompt(self, name):
        if name not in self.prompts:
            if self.prompt_dir:
                d, audio = Path(self.prompt_dir), Path(self.prompt_dir) / f"{name}.wav"
            else:
                d = self.root / "example/audio"
                audio = d / f"{name}.mp3"
            meta = json.load(open(d / f"{name}.json", encoding="utf8"))[0]
            self.prompts[name] = self.dp.process(meta, str(audio))
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
    # one transposition for the whole piece; a solo line moves toward its (randomly chosen) voice
    allp = [n[2] for ln in lines for n in ln]
    solo_prompt = rng.choice(sorted(PROMPTS))
    shift = int(np.clip(round(PROMPTS[solo_prompt] - np.median([n[2] for n in lines[0]])), -7, 7)) \
        + rng.randint(-2, 2)
    if len(lines) > 1:  # choir: keep the original key area
        shift = rng.randint(-2, 2)
    shift = int(np.clip(shift, 21 - min(allp), 108 - max(allp)))
    lyr = Lyrics(rng)
    # accompaniment: everything that is not a vocal track, rendered like score_synth
    for i in vox:
        pm.instruments.remove(i)
    # scores often double the voice in the accompaniment (hymn right hand, choir reduction); then
    # the melody is heard from the piano too and the model need not read the voice. Most renders
    # drop the doubling notes (same onset, same pitch or octave) from audio and target alike.
    undoubled = 0
    if rng.random() < UNDOUBLE_P:
        von = {}
        for ln in lines:
            for s, e, p in ln:
                von.setdefault(p, []).append(s)
        von = {p: np.array(v) for p, v in von.items()}

        def doubles(n):
            for p in (n.pitch, n.pitch - 12, n.pitch + 12):
                if p in von and np.abs(von[p] - n.start).min() < 0.05:
                    return True
            return False

        for inst in pm.instruments:
            if inst.is_drum:
                continue
            keep = [n for n in inst.notes if not doubles(n)]
            undoubled += len(inst.notes) - len(keep)
            inst.notes = keep
    for inst in pm.instruments:
        if not inst.is_drum:
            for n in inst.notes:
                n.pitch = int(np.clip(n.pitch + shift, 0, 127))
    buf = io.BytesIO()
    pm.write(buf)
    apm, cfg = audio_midi(buf.getvalue(), "chamber", rng)
    if undoubled:  # check: the rendered accompaniment no longer doubles the voice
        left = 0
        for i in apm.instruments:
            for n in i.notes:
                if not i.is_drum and doubles(pretty_midi.Note(n.velocity, n.pitch - shift, n.start, n.end)):
                    left += 1
        print("undoubled", undoubled, "doubling notes left in rendered accompaniment:", left, flush=True)
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
            prompt = solo_prompt if len(lines) == 1 else rng.choice(sorted(pool))
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
        vox_mixed = vox_sum / rms(vox_sum) * acc_rms * 10 ** (lead / 20)
        if listen:  # stems for checking by ear: accompaniment alone, voices alone
            for tag, sig in (("acc", mix), ("vox", vox_mixed)):
                wavfile.write(td / f"{tag}.wav", FS, (sig / max(np.abs(sig).max(), 1e-6) * 0.9).astype(np.float32))
                Path(listen).mkdir(parents=True, exist_ok=True)
                name = re.sub(r"[^\w\-]+", "_", str(row["title"]))[:50]
                subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(td / f"{tag}.wav"), "-b:a",
                                "160k", str(Path(listen) / f"{name}_{rid}_stem_{tag}.mp3")], check=True)
        mix += vox_mixed
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
            "vocal_wet": round(wet, 2), "lead_db": round(lead, 1), "undoubled": undoubled}
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


# content filter (owner's rules: no occult / witches / glorified ghosts or death figures);
# sacred texts with "Spirit" or "Tod" stay
BAD_TITLE = re.compile(r"(?i)witch|sorceress|wayward sisters|vaulted cell|our next motion|destruction's our delight|"
                       r"harm's our delight|ere we this perform|spirit aeneas|\bho ho ho\b|hexe|erlk[öo]nig|"
                       r"der tod und? das|devil|satan|teufel|demon|dämon|lucifer|ghost|zombie|vampir|idol")
# PD composers missing from organ_synth.OLD that the vocal candidates need (all died before 1700
# or 1900); DA2017 = Purcell, Dido and Aeneas; a bare BWV number = Bach
OLD_EXTRA = re.compile(r"(?i)dowland|weelkes|\barne\b|w\. ?h\. monk|william henry monk|\bbwv ?\d|^da2017")


def candidates(path):
    """PDMX row indices of the vocal candidates that pass the PD-composer and content filters."""
    from organ_synth import MODERN, OLD

    keep, drop = [], []
    for r in pickle.load(open(path, "rb")):  # (views, pdmx_row, title, composer, kind, vox, acc, secs)
        text = f"{r[3]} {r[2]}"
        pd_ok = (OLD.search(text) or OLD_EXTRA.search(str(r[2]))) and not MODERN.search(text)
        (keep if pd_ok and not BAD_TITLE.search(str(r[2])) else drop).append(r)
    return [r[1] for r in keep], drop


def finalize(out, rows_meta):
    """meta.csv and the transkun pickles from all per-piece records in <out>/vocal/items."""
    recs = []
    for f in sorted((out / "vocal" / "items").glob("*.pkl")):
        recs.append(pickle.load(open(f, "rb")))
    if not recs:
        return 0
    with open(out / "vocal" / "meta.csv.tmp", "w", newline="", encoding="utf8") as f:
        w = csv.writer(f)
        w.writerow(["id", "split", "pdmx_row", "title", "composer", "license", "pdmx_mid", "duration", "nSamples",
                    *recs[0]["meta"].keys()])
        for r in recs:
            w.writerow([r["id"], r["split"], r["pdmx_row"], r["row"]["title"], r["row"]["composer_name"],
                        r["row"]["license"], r["row"]["mid"], round(r["duration"], 2), r["nSamples"],
                        *r["meta"].values()])
    os.replace(out / "vocal" / "meta.csv.tmp", out / "vocal" / "meta.csv")
    for split in ("train", "val"):
        items = [{"audio_filename": f"vocal/{r['id']}.wav", "duration": r["duration"], "fs": FS,
                  "nSamples": r["nSamples"], "nChannel": 1, "notes": to_transkun(r["notes"]), "source": "vocal",
                  "split": split} for r in recs if r["split"] == split]
        pickle.dump(items, open(out / "vocal" / f"{split}.pickle.tmp", "wb"))
        os.replace(out / "vocal" / f"{split}.pickle.tmp", out / "vocal" / f"{split}.pickle")
    return len(recs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdmx", default="/data/pdmx")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ids", help="comma-separated PDMX.csv row indices")
    ap.add_argument("--candidates", help="vocal_candidates.pkl: render all that pass the filters (resumable)")
    ap.add_argument("--sf2", nargs="+", required=True)
    ap.add_argument("--soulx", default="/data/ext/soulx")
    ap.add_argument("--prompts", help="vocadito_prompts.py output dir (default: SoulX repo example prompts)")
    ap.add_argument("--undouble-p", type=float, default=0.7, help="share of renders without voice doubling")
    ap.add_argument("--listen")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-every", type=int, default=20)
    ap.add_argument("--max-vram-gb", type=float, default=6.0)
    ap.add_argument("--max-tries", type=int, default=3, help="skip a piece after this many failed starts")
    args = ap.parse_args()

    import pandas as pd
    import torch

    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, args.max_vram_gb * 2**30 / total))
    out = Path(args.out)
    if args.candidates:
        ids, drop = candidates(args.candidates)
        print(len(ids), "candidates pass the filters,", len(drop), "dropped", flush=True)
    else:
        ids = [int(x) for x in args.ids.split(",")]
    # seeds, sound fonts and splits are fixed per position before anything is skipped, so a
    # resumed run renders exactly what an uninterrupted one would
    rng = random.Random(args.seed)
    plan = [(k, i, rng.randrange(2**31), rng.choice(args.sf2)) for k, i in enumerate(ids)]
    items, tries = out / "vocal" / "items", out / "vocal" / "tries"
    items.mkdir(parents=True, exist_ok=True)
    tries.mkdir(parents=True, exist_ok=True)
    todo = []
    for k, i, seed, sf2 in plan:
        if (items / f"{i}.pkl").exists() or (items / f"{i}.none").exists():
            continue
        t = tries / f"{i}.n"
        n = int(t.read_text()) if t.exists() else 0
        if n >= args.max_tries:
            continue
        todo.append((k, i, seed, sf2))
    print(len(plan) - len(todo), "of", len(plan), "already done or skipped;", len(todo), "to render", flush=True)
    if todo:
        df = pd.read_csv(Path(args.pdmx) / "PDMX.csv", low_memory=False)
        rows = {i: dict(df.loc[i], _id=Path(df.loc[i, "mid"]).stem[:24]) for _, i, _, _ in todo}
        want = {rows[i]["mid"].lstrip("./"): i for i in rows}
        mids = {}
        with tarfile.open(Path(args.pdmx) / "mid.tar.gz") as tar:
            for m in tar:
                if m.isfile() and m.name.lstrip("./") in want:
                    mids[want[m.name.lstrip("./")]] = tar.extractfile(m).read()
        global UNDOUBLE_P
        UNDOUBLE_P = args.undouble_p
        singer = Singer(args.soulx)
        if args.prompts:
            PROMPTS.clear()
            PROMPTS.update(json.load(open(Path(args.prompts) / "prompts.json")))
            singer.prompt_dir = args.prompts
        for n_done, (k, i, seed, sf2) in enumerate(todo):
            # count the attempt before starting: a crash, a CUDA error after hibernation or a
            # kill all leave it counted, so a piece that always fails is skipped eventually
            t = tries / f"{i}.n"
            t.write_text(str((int(t.read_text()) if t.exists() else 0) + 1))
            r = work(rows[i], mids[i], singer, sf2, seed, out, args.listen) if i in mids else None
            if r is None:
                (items / f"{i}.none").write_text("no vocal line / no midi")
                continue
            r["split"] = "val" if k % args.val_every == args.val_every - 1 else "train"
            r["row"] = {c: rows[i][c] for c in ("title", "composer_name", "license", "mid")}
            r["pdmx_row"] = i
            pickle.dump(r, open(items / f"{i}.pkl.tmp", "wb"))
            os.replace(items / f"{i}.pkl.tmp", items / f"{i}.pkl")
            print(f"done {n_done + 1}/{len(todo)}", i, rows[i]["title"], r["meta"], flush=True)
            if (n_done + 1) % 25 == 0:
                finalize(out, rows)
    n = finalize(out, None)
    print("finalized", n, "pieces", flush=True)


if __name__ == "__main__":
    main()

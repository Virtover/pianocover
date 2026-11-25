"""Vocal melody add-on test on the 7 test songs (owner 2026-11-25: singing is caught "so-so").

Fine-tuned Transkun's full output (scripts/tk_songs.py, read back from its MIDI) + the sung
melody from the teacher pipeline: Demucs htdemucs vocal stem -> CREPE full f0 -> notes
(pianocover/teacher/models.py TeacherModels.separate / vocal_melody, f0notes.py). A sung note
counts as covered when Transkun has a note of the same pitch class (same key or an octave off,
so a melody Transkun put in another octave is not doubled) overlapping >= 50 % of it. Only
uncovered sung notes are added, on their own track (channel 1) so a player can mute them;
where one overlaps a Transkun note of the same key it is cut short (or dropped if < 80 ms).
Outputs per song: "transkun-ft+vocal full" (2 tracks) and "transkun-ft+vocal reduced"
(reduce() over the union, 1 track). Timings and peak VRAM per stage in bench.json.

  PYTHONPATH=/repo python scripts/vocal_test.py --out /data/vocal_test/20261125
"""
import argparse
import csv
import json
import resource
import time
from pathlib import Path

import numpy as np
import torch

from pianocover.music2midi_fast import check_midi, clean_notes, write_midi
from pianocover.reduce import reduce
from pianocover.teacher.models import TeacherModels, decode_audio

REPO = Path(__file__).resolve().parents[1]
SONGS = REPO / "data" / "test_songs"


def read_events(path):
    """MIDI written by write_midi -> (N, 4) notes + pedal rows (pitch -64 / -67)."""
    import mido

    mf = mido.MidiFile(str(path))
    t, open_n, open_p, out = 0.0, {}, {}, []
    for msg in mido.merge_tracks(mf.tracks):
        t += msg.time / 960
        if msg.type == "note_on" and msg.velocity > 0:
            open_n[msg.note] = (t, msg.velocity)
        elif msg.type in ("note_off", "note_on") and msg.note in open_n:
            s, v = open_n.pop(msg.note)
            out.append([s, t, msg.note, v])
        elif msg.type == "control_change" and msg.control in (64, 67):
            if msg.value >= 64:
                open_p[msg.control] = t
            elif msg.control in open_p:
                out.append([open_p.pop(msg.control), t, -msg.control, 127])
    a = np.array(out, np.float64).reshape(-1, 4)
    return a[np.argsort(a[:, 0], kind="stable")]


@torch.inference_mode()
def separate(tm, mix):
    """TeacherModels.separate, but the song and the 4 output stems stay in CPU memory; only the
    7.8 s chunks Demucs works on go to the GPU (an 18.5 min song otherwise needs ~1.5 GiB of
    VRAM for the output alone). Same computation as the teacher's."""
    from demucs.apply import apply_model

    wav = torch.from_numpy(mix)
    ref = wav.mean(0)
    mean, std = ref.mean(), ref.std() + 1e-8
    out = apply_model(tm.demucs, ((wav - mean) / std)[None], shifts=tm.demucs_shifts, split=True,
                      overlap=0.25, device=tm.device, progress=False)[0]
    out = out * std + mean
    return {name: out[i].mean(0).float().numpy() for i, name in enumerate(tm.demucs.sources)}


def coverage(voc, tk, min_frac=0.5):
    """Bool per sung note: same pitch class in Transkun overlapping >= min_frac of it."""
    tk = tk[tk[:, 2] > 0]
    cov = np.zeros(len(voc), bool)
    exact = np.zeros(len(voc), bool)
    for i, (s, e, p, _) in enumerate(voc):
        m = (tk[:, 0] < e) & (tk[:, 1] > s) & ((tk[:, 2] - p) % 12 == 0)
        if not m.any():
            continue
        ov = np.minimum(tk[m, 1], e) - np.maximum(tk[m, 0], s)
        frac = ov / max(e - s, 1e-6)
        cov[i] = frac.max() >= min_frac
        exact[i] = (frac[tk[m, 2] == p] >= min_frac).any() if (tk[m, 2] == p).any() else False
    return cov, exact


def trim_against(add, tk, min_dur=0.08):
    """Cut added notes where the same key is already sounding in Transkun's track."""
    tk = tk[tk[:, 2] > 0]
    out = []
    for s, e, p, v in add:
        for ts, te, tp, _ in tk[(tk[:, 2] == p) & (tk[:, 0] < e) & (tk[:, 1] > s)]:
            if ts <= s:
                s = max(s, te)
            else:
                e = min(e, ts)
        if e - s >= min_dur:
            out.append([s, e, p, v])
    return np.array(out, np.float64).reshape(-1, 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tk-midi", default="/data/tk_songs/20261125/midi")
    ap.add_argument("--out", default="/data/vocal_test/20261125")
    ap.add_argument("--songs", default="")
    ap.add_argument("--vram-fraction", type=float, default=0.6)
    a = ap.parse_args()
    out = Path(a.out)
    (out / "midi").mkdir(parents=True, exist_ok=True)
    torch.cuda.set_per_process_memory_fraction(a.vram_fraction)
    gpu = torch.cuda.get_device_properties(0)
    dev = torch.device("cuda")
    t = time.time()
    from demucs.pretrained import get_model

    tm = TeacherModels.__new__(TeacherModels)  # only the separator + CREPE, not the full teacher
    tm.device, tm.demucs_shifts = dev, 0
    tm.demucs = get_model("htdemucs").to(dev).eval()
    torch.cuda.synchronize()
    load_s = time.time() - t
    songs = list(csv.DictReader(open(SONGS / "songs.csv", encoding="utf-8")))
    if a.songs:
        songs = [s for s in songs if s["n"] in a.songs.split(",")]
    rows = []
    for s in songs:
        tk = read_events(Path(a.tk_midi) / f"{int(s['n']):02d} - {s['title']} - transkun-ft full.mid".replace("/", "-"))
        t = time.time()
        mix = decode_audio(SONGS / s["file"])
        audio_s = time.time() - t
        dur = mix.shape[1] / 44100
        torch.cuda.reset_peak_memory_stats()
        t = time.time()
        stems = separate(tm, mix)
        torch.cuda.synchronize()
        sep_s = time.time() - t
        sep_vram = torch.cuda.max_memory_allocated() / 2**30
        torch.cuda.reset_peak_memory_stats()
        t = time.time()
        voc = tm.vocal_melody(stems["vocals"], 44100)
        torch.cuda.synchronize()
        crepe_s = time.time() - t
        crepe_vram = torch.cuda.max_memory_allocated() / 2**30
        voc, _ = clean_notes(voc, keep_unclosed=False)
        cov, exact = coverage(voc, tk)
        add = trim_against(voc[~cov], tk)
        sung_sec = float((voc[:, 1] - voc[:, 0]).sum())
        st = dict(n=s["n"], title=s["title"], audio_sec=round(dur, 1), audio_load_s=round(audio_s, 2),
                  demucs_s=round(sep_s, 2), crepe_s=round(crepe_s, 2),
                  demucs_peak_vram_gib=round(sep_vram, 3), crepe_peak_vram_gib=round(crepe_vram, 3),
                  sung_notes=len(voc), sung_sec=round(sung_sec, 1),
                  covered_by_transkun=round(float(cov.mean()) if len(voc) else 0, 3),
                  covered_same_octave=round(float(exact.mean()) if len(voc) else 0, 3),
                  added_notes=len(add), transkun_notes=int((tk[:, 2] > 0).sum()))
        base = f"{int(s['n']):02d} - {s['title']} - transkun-ft+vocal".replace("/", "-")
        p_full = out / "midi" / f"{base} full.mid"
        write_midi(tk, p_full, name=f"{s['title']} (transkun-ft)", extra_tracks=[("added vocal melody (CREPE)", add)])
        union = np.concatenate([tk, add])
        red, rep = reduce(union[np.argsort(union[:, 0], kind="stable")])
        red_notes, _ = clean_notes(red[red[:, 2] > 0], keep_unclosed=False)
        red = np.concatenate([red_notes, red[red[:, 2] < 0]])
        p_red = out / "midi" / f"{base} reduced.mid"
        write_midi(red[np.argsort(red[:, 0], kind="stable")], p_red, name=f"{s['title']} (transkun-ft+vocal reduced)")
        st.update(check_full=check_midi(p_full), check_reduced=check_midi(p_red),
                  reduced_notes=int((red[:, 2] > 0).sum()),
                  reduce_report=dict(changed_onsets=rep.changed_onsets, onsets=rep.onsets, dropped=rep.dropped))
        np.save(out / f"{int(s['n']):02d}_vocal_notes.npy", voc)
        rows.append(st)
        print(json.dumps(st, ensure_ascii=False), flush=True)
    summary = {"gpu": f"{gpu.name} {gpu.total_memory / 2**30:.1f} GiB sm_{gpu.major}{gpu.minor}",
               "torch": torch.__version__, "demucs": "htdemucs (4 stems, shifts 0, overlap 0.25)",
               "crepe": "full, 20 ms hop, 40 dB gate", "model_load_s": round(load_s, 2),
               "total_demucs_s": round(sum(r["demucs_s"] for r in rows), 1),
               "total_crepe_s": round(sum(r["crepe_s"] for r in rows), 1),
               "total_audio_sec": round(sum(r["audio_sec"] for r in rows), 1),
               "peak_rss_gib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2),
               "date": time.strftime("%Y-%m-%d %H:%M")}
    print(json.dumps(summary, indent=1))
    (out / "bench.json").write_text(json.dumps({"summary": summary, "jobs": rows}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()

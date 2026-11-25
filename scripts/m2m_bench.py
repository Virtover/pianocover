"""Music2MIDI (ytinyui/music2midi, MIT) on the 7 test songs (data/test_songs): covers + benchmark.

  check   one song: authors' 16 kHz input vs the 22.05 kHz their training used, fp32 vs bf16
          (chroma agreement with the song, note F1 between precisions, timings)
  run     every song x difficulty, one generate() job each (timings, peak VRAM), MIDI + stats
  (render the MIDIs to MP3 in pianocover:voice with scripts/m2m_render.sh)

  PYTHONPATH=/repo:/data/ext/pt_lib:/data/ext/m2m_lib python scripts/m2m_bench.py run
"""
import argparse
import csv
import json
import resource
import time
from pathlib import Path

import numpy as np
import torch

from pianocover.music2midi_fast import DIFFICULTIES, GENRES, Music2MIDIFast, check_midi, clean_notes, write_midi

REPO = Path(__file__).resolve().parents[1]
SONGS = REPO / "data" / "test_songs"
GENRE = {"WCxx_OVOVUw": "soundtrack", "y3AiGw8mkq0": "classical"}  # the rest: pop (worship)


def songs():
    with open(SONGS / "songs.csv", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def chroma_score(y, sr, notes):
    """Mean frame-wise cosine between the song's CQT chroma and the MIDI's pitch-class roll
    (100 ms frames, frames where either is silent skipped). Pitch/timing errors lower it."""
    import librosa

    hop = 2048 if sr > 20000 else 1600
    ch = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
    fr = hop / sr
    roll = np.zeros_like(ch)
    for on, off, p, _ in notes:
        a, b = int(on / fr), max(int(on / fr) + 1, int(off / fr))
        roll[int(p) % 12, a:min(b, roll.shape[1])] += 1
    num = (ch * roll).sum(0)
    den = np.linalg.norm(ch, axis=0) * np.linalg.norm(roll, axis=0)
    m = den > 1e-6
    return float((num[m] / den[m]).mean()), float(m.mean())


def note_f1(a, b):
    import mir_eval

    def iv(n):
        return n[:, :2], 440.0 * 2 ** ((n[:, 2] - 69) / 12)

    (ia, pa), (ib, pb) = iv(a), iv(b)
    p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(ia, pa, ib, pb, offset_ratio=None)
    return round(f, 4)


def run_one(m, y, sr, cond, batch_size=128, max_length=512, loop_stop=True):
    wav = m.segments(y, sr)
    c = torch.tensor([cond], dtype=torch.long).expand(len(wav), 2)
    torch.cuda.reset_peak_memory_stats()
    sync()
    t = time.time()
    toks = m.generate_tokens(wav, c, batch_size=batch_size, max_length=max_length, loop_stop=loop_stop)
    sync()
    dt = time.time() - t
    t2 = time.time()
    raw, st = m.decode(toks)
    st["decode_s"] = round(time.time() - t2, 3)
    st["generate_s"] = round(dt, 2)
    st["peak_vram_alloc_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
    st["peak_vram_reserved_gib"] = round(torch.cuda.max_memory_reserved() / 2**30, 3)
    st["max_tokens"] = max(len(x) for x in toks)
    return raw, st


def cmd_check(a):
    s = [r for r in songs() if r["video_id"] == a.video][0]
    path = SONGS / s["file"]
    cond = [GENRES.index(GENRE.get(s["video_id"], "pop")), DIFFICULTIES.index("advanced")]
    torch.cuda.set_per_process_memory_fraction(a.vram_fraction)
    res = {}
    keep = {}
    for dtype in (torch.float32, torch.bfloat16):
        m = Music2MIDIFast(a.ckpt, dtype=dtype)
        for sr in (16000, 22050):
            y = m.load_audio(path, sr)
            raw, st = run_one(m, y, sr, cond, a.batch_size, a.max_length, not a.no_loop_stop)
            notes, cst = clean_notes(raw, m.seg_sec)
            # at 22.05 kHz a "3 s" segment of the model is 66150 samples = 3 s of audio too
            onset_in_seg = np.mod(notes[:, 0], m.seg_sec)
            st.update(cst)
            st["chroma_cos"], st["chroma_frames"] = chroma_score(y, sr, notes)
            st["onsets_last_25pct_of_segment"] = round(float((onset_in_seg > 0.75 * m.seg_sec).mean()), 3)
            key = f"{str(dtype).split('.')[-1]}_{sr}"
            res[key] = st
            keep[key] = notes
            print(key, json.dumps(st), flush=True)
        del m
        torch.cuda.empty_cache()
    for sr in (16000, 22050):
        res[f"f1_fp32_vs_bf16_{sr}"] = note_f1(keep[f"float32_{sr}"], keep[f"bfloat16_{sr}"])
    print(json.dumps(res, indent=1))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"check_{a.video}.json").write_text(json.dumps(res, indent=1))
    for k, n in keep.items():
        write_midi(n, out / f"check_{a.video}_{k}.mid")


def cmd_run(a):
    out = Path(a.out)
    (out / "midi").mkdir(parents=True, exist_ok=True)
    torch.cuda.set_per_process_memory_fraction(a.vram_fraction)
    gpu = torch.cuda.get_device_properties(0)
    sync()
    t = time.time()
    m = Music2MIDIFast(a.ckpt, dtype=torch.bfloat16 if a.dtype == "bf16" else torch.float32)
    sync()
    load_s = time.time() - t
    # warm-up (CUDA context, cuBLAS) on 2 segments, not timed
    m.generate_tokens(torch.zeros(2, int(a.sr * m.seg_sec)), torch.zeros(2, 2, dtype=torch.long))
    rows = []
    for s in songs():
        if a.songs and s["n"] not in a.songs.split(","):
            continue
        t = time.time()
        y = m.load_audio(SONGS / s["file"], a.sr)
        audio_s = time.time() - t
        dur = len(y) / a.sr
        genre = GENRE.get(s["video_id"], "pop")
        for diff in a.difficulties.split(","):
            raw, st = run_one(m, y, a.sr, [GENRES.index(genre), DIFFICULTIES.index(diff)], a.batch_size, a.max_length, not a.no_loop_stop)
            notes, cst = clean_notes(raw, m.seg_sec)
            st.update(cst)
            name = f"{int(s['n']):02d} - {s['title']} - music2midi {diff}".replace("/", "-")
            t = time.time()
            write_midi(notes, out / "midi" / f"{name}.mid", name=f"{s['title']} (music2midi {diff})")
            st["write_s"] = round(time.time() - t, 3)
            st["check"] = check_midi(out / "midi" / f"{name}.mid")
            st.update(n=s["n"], video_id=s["video_id"], title=s["title"], genre=genre, difficulty=diff,
                      audio_sec=round(dur, 1), audio_load_s=round(audio_s, 2),
                      x_realtime=round(dur / st["generate_s"], 1), file=f"{name}.mid")
            rows.append(st)
            print(json.dumps(st, ensure_ascii=False), flush=True)
    summary = {
        "gpu": f"{gpu.name} {gpu.total_memory / 2**30:.1f} GiB sm_{gpu.major}{gpu.minor}",
        "torch": torch.__version__, "dtype": a.dtype, "sr": a.sr, "batch_size": a.batch_size,
        "max_length": a.max_length, "loop_stop": not a.no_loop_stop, "model_load_s": round(load_s, 2), "params": sum(p.numel() for p in m.net.parameters()),
        "peak_rss_gib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2),
        "total_generate_s": round(sum(r["generate_s"] for r in rows), 1),
        "total_audio_sec": round(sum(r["audio_sec"] for r in rows), 1),
        "date": time.strftime("%Y-%m-%d %H:%M"),
    }
    print(json.dumps(summary, indent=1))
    (out / "bench.json").write_text(json.dumps({"summary": summary, "jobs": rows}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "run"])
    ap.add_argument("--ckpt", default="/data/ext/music2midi/epoch799.ckpt")
    ap.add_argument("--out", default="/data/m2m/20261124")
    ap.add_argument("--video", default="edZVnKxKEUU")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--no-loop-stop", action="store_true")
    ap.add_argument("--songs", default="", help="comma-separated n from songs.csv (default: all)")
    ap.add_argument("--difficulties", default="beginner,intermediate,advanced")
    ap.add_argument("--vram-fraction", type=float, default=0.6)
    a = ap.parse_args()
    {"check": cmd_check, "run": cmd_run}[a.cmd](a)

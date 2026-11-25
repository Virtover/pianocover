"""Fine-tuned Transkun (phase 5, scripts/tk_train.py) vs plain Transkun 2.0 on the 7 test songs.

For each model and song: transcribe (full mode), then pianocover.reduce.reduce (reduced mode:
changes only what is not playable). Every MIDI goes through the same safety pass as the
Music2MIDI covers (pianocover/music2midi_fast.clean_notes + write_midi + check_midi): no
same-key overlaps, notes in A0..C8, pedals as CC 64 / 67, re-read and checked after writing.
Timings and peak VRAM per job in bench.json. Render to MP3 with scripts/m2m_render.sh.

  PYTHONPATH=/repo:/data/ext/pt_lib:/data/ext/m2m_lib python scripts/tk_songs.py \
      --ft /data/ext/tk_ft_20261125/best_transkun.pt --out /data/tk_songs/20261125
"""
import argparse
import csv
import json
import resource
import time
from pathlib import Path

import numpy as np
import torch

from pianocover import SAMPLE_RATE, events
from pianocover.audio import decode
from pianocover.model import load_pretrained_transkun
from pianocover.music2midi_fast import check_midi, clean_notes, write_midi
from pianocover.reduce import reduce

REPO = Path(__file__).resolve().parents[1]
SONGS = REPO / "data" / "test_songs"


def safe(ev):
    """clean_notes on the notes only; pedal rows pass through (already sanitized)."""
    notes, pedals = ev[ev[:, 2] > 0], ev[ev[:, 2] < 0]
    n, st = clean_notes(notes, keep_unclosed=False)
    out = np.concatenate([n, pedals])
    return out[np.argsort(out[:, 0], kind="stable")], st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ft", required=True, help="fine-tuned best_transkun.pt")
    ap.add_argument("--ft-conf", default=None, help="its transkun.conf (default: next to --ft)")
    ap.add_argument("--out", default="/data/tk_songs/20261125")
    ap.add_argument("--models", default="ft,base")
    ap.add_argument("--songs", default="")
    ap.add_argument("--vram-fraction", type=float, default=0.6)
    a = ap.parse_args()
    out = Path(a.out)
    (out / "midi").mkdir(parents=True, exist_ok=True)
    torch.cuda.set_per_process_memory_fraction(a.vram_fraction)
    torch.set_grad_enabled(False)
    gpu = torch.cuda.get_device_properties(0)
    songs = list(csv.DictReader(open(SONGS / "songs.csv", encoding="utf-8")))
    if a.songs:
        songs = [s for s in songs if s["n"] in a.songs.split(",")]
    ft_conf = a.ft_conf or str(Path(a.ft).with_name("transkun.conf"))
    rows, loads = [], {}
    for mname in a.models.split(","):
        torch.cuda.synchronize()
        t = time.time()
        model = load_pretrained_transkun(a.ft, ft_conf, "cuda") if mname == "ft" else load_pretrained_transkun(device="cuda")
        torch.cuda.synchronize()
        loads[mname] = round(time.time() - t, 2)
        label = {"ft": "transkun-ft", "base": "transkun-2.0"}[mname]
        for s in songs:
            t = time.time()
            y = decode(SONGS / s["file"], SAMPLE_RATE)
            audio_s = time.time() - t
            dur = len(y) / SAMPLE_RATE
            x = torch.from_numpy(y[:, None]).cuda()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            t = time.time()
            notes = model.transcribe(x, discardSecondHalf=False)
            torch.cuda.synchronize()
            gen = time.time() - t
            del x
            ev = events.sanitize(events.from_transkun_notes(notes))
            t = time.time()
            red, rep = reduce(ev)
            red_s = time.time() - t
            st = dict(n=s["n"], title=s["title"], model=label, audio_sec=round(dur, 1),
                      audio_load_s=round(audio_s, 2), transcribe_s=round(gen, 2),
                      x_realtime=round(dur / gen, 1), reduce_s=round(red_s, 2),
                      peak_vram_alloc_gib=round(torch.cuda.max_memory_allocated() / 2**30, 3),
                      peak_vram_reserved_gib=round(torch.cuda.max_memory_reserved() / 2**30, 3),
                      reduce_report=dict(onsets=rep.onsets, changed_onsets=rep.changed_onsets,
                                         dropped=rep.dropped, shortened=rep.shortened))
            for mode, e in (("full", ev), ("reduced", red)):
                e2, cst = safe(e)
                name = f"{int(s['n']):02d} - {s['title']} - {label} {mode}".replace("/", "-")
                path = out / "midi" / f"{name}.mid"
                write_midi(e2, path, name=f"{s['title']} ({label} {mode})")
                st[mode] = dict(notes=int((e2[:, 2] > 0).sum()), pedal_intervals=int((e2[:, 2] < 0).sum()),
                                notes_per_s=round(float((e2[:, 2] > 0).sum()) / dur, 2),
                                max_polyphony=max_poly(e2), safety=cst, check=check_midi(path), file=path.name)
            rows.append(st)
            print(json.dumps(st, ensure_ascii=False), flush=True)
        del model
        torch.cuda.empty_cache()
    summary = {"gpu": f"{gpu.name} {gpu.total_memory / 2**30:.1f} GiB sm_{gpu.major}{gpu.minor}",
               "torch": torch.__version__, "precision": "fp32 (Transkun default)",
               "model_load_s": loads, "ft_weights": a.ft,
               "peak_rss_gib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2),
               "date": time.strftime("%Y-%m-%d %H:%M")}
    print(json.dumps(summary, indent=1))
    (out / "bench.json").write_text(json.dumps({"summary": summary, "jobs": rows}, indent=1, ensure_ascii=False))


def max_poly(ev):
    n = ev[ev[:, 2] > 0]
    if not len(n):
        return 0
    t = np.concatenate([np.stack([n[:, 0], np.ones(len(n))], 1), np.stack([n[:, 1], -np.ones(len(n))], 1)])
    t = t[np.lexsort((t[:, 1], t[:, 0]))]
    return int(np.cumsum(t[:, 1]).max())


if __name__ == "__main__":
    main()

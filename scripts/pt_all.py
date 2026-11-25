"""Every cover of the 7 test songs (Music2MIDI, Transkun 2.0 / fine-tuned full + reduced, +vocal)
through Pianist Transformer (score -> expressive performance; yhj137/pianist-transformer-rendering,
Apache-2.0), all files in ONE render_ids call: the windows of every file are batched together
(pianocover/pianist.py, windows="two_pass": even windows of all files in parallel, then the odd ones
continuing them), chunks sorted by window length. Output MIDIs go through the same safety pass as the
inputs (clean_notes + write_midi + check_midi); the model's sustain pedal (CC 64 values 0-127 per
quarter of each inter-onset interval) becomes on/off intervals at value >= 64.

Note: the model expects a score (quantised, flat dynamics). Music2MIDI output is close to that
(50 ms grid, velocity 80); Transkun output is a transcription that already has performance timing.

  PYTHONPATH=/repo/third_party/PianistTransformer:/data/ext/pt_lib:/repo python scripts/pt_all.py
"""
import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np
import torch

from pianocover.music2midi_fast import check_midi, clean_notes, write_midi

INPUTS = ["/data/m2m/20261124/midi", "/data/tk_songs/20261125/midi", "/data/vocal_test/20261125/midi"]


def to_events(mf):
    """miditoolkit performance (500 ticks/beat, 120 BPM -> 1 tick = 1 ms) -> (N, 4) notes + pedal rows."""
    sec = 1 / 1000
    notes = [[n.start * sec, n.end * sec, n.pitch, n.velocity] for i in mf.instruments for n in i.notes]
    ped, down = [], None
    ccs = sorted((c for i in mf.instruments for c in i.control_changes if c.number == 64), key=lambda c: c.time)
    for c in ccs:
        if c.value >= 64 and down is None:
            down = c.time * sec
        elif c.value < 64 and down is not None:
            if c.time * sec > down:
                ped.append([down, c.time * sec, -64, 127])
            down = None
    if down is not None and notes:
        ped.append([down, max(n[1] for n in notes), -64, 127])
    return np.array(notes, np.float64).reshape(-1, 4), np.array(ped, np.float64).reshape(-1, 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="/data/ext/pianist_transformer/models/sft")
    ap.add_argument("--out", default="/data/pt_all/20261125")
    ap.add_argument("--windows", default="two_pass", choices=["sequential", "two_pass", "parallel"])
    ap.add_argument("--max-batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--single", default="/data/tk_songs/20261125/midi/03 - SO BE IT - transkun-ft full.mid",
                    help="also time this one file alone (cost of one job)")
    ap.add_argument("--vram-fraction", type=float, default=0.6)
    a = ap.parse_args()
    from miditoolkit import MidiFile
    from src.model.pianoformer import PianoT5Gemma
    from src.utils.midi import ids_to_midi, midi_to_ids

    from pianocover.pianist import FastRenderer

    out = Path(a.out)
    (out / "midi").mkdir(parents=True, exist_ok=True)
    torch.cuda.set_per_process_memory_fraction(a.vram_fraction)
    gpu = torch.cuda.get_device_properties(0)
    t = time.time()
    m = PianoT5Gemma.from_pretrained(a.weights, torch_dtype=torch.bfloat16).cuda().eval()
    fr = FastRenderer(m, "cuda")
    load_s = time.time() - t
    files = sorted(p for d in INPUTS for p in Path(d).glob("*.mid"))
    t = time.time()
    ids = [midi_to_ids(fr.cfg, MidiFile(str(p))) for p in files]
    prep_s = time.time() - t
    n_tok = sum(len(x) for x in ids)
    print(f"{len(files)} files, {n_tok // 8} notes, {n_tok} tokens", flush=True)

    # warm-up + one-job timing: a single file alone
    one = [midi_to_ids(fr.cfg, MidiFile(a.single))]
    fr.render_ids(one[:1], seed=0, windows=a.windows, max_batch=a.max_batch)  # CUDA graphs captured
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t = time.time()
    fr.render_ids(one, seed=a.seed, windows=a.windows, max_batch=a.max_batch)
    torch.cuda.synchronize()
    single = {"file": Path(a.single).name, "notes": len(one[0]) // 8, "seconds": round(time.time() - t, 2),
              "peak_vram_alloc_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3)}
    print("single", json.dumps(single), flush=True)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t = time.time()
    toks = fr.render_ids(ids, seed=a.seed, windows=a.windows, max_batch=a.max_batch)
    torch.cuda.synchronize()
    gen_s = time.time() - t
    peak = torch.cuda.max_memory_allocated() / 2**30
    rows = []
    t = time.time()
    for p, tk, i in zip(files, toks, ids):
        perf = ids_to_midi(fr.cfg, tk, ref=i)
        notes, ped = to_events(perf)
        notes, st = clean_notes(notes, keep_unclosed=False)
        ev = np.concatenate([notes, ped])
        ev = ev[np.argsort(ev[:, 0], kind="stable")]
        dst = out / "midi" / f"{p.stem} - pianist.mid"
        write_midi(ev, dst, name=f"{p.stem} (Pianist Transformer)")
        chk = check_midi(dst)
        rows.append({"file": dst.name, "source": p.parent.parent.name, "notes_in": len(i) // 8,
                     "notes_out": int(len(notes)), "pedal_intervals": int(len(ped)),
                     "velocity_mean": round(float(notes[:, 3].mean()), 1) if len(notes) else 0,
                     "velocity_std": round(float(notes[:, 3].std()), 1) if len(notes) else 0,
                     "seconds": chk["seconds"], "errors": chk["errors"], "safety": st})
    post_s = time.time() - t
    summary = {"gpu": f"{gpu.name} {gpu.total_memory / 2**30:.1f} GiB sm_{gpu.major}{gpu.minor}",
               "torch": torch.__version__, "windows": a.windows, "max_batch": a.max_batch, "seed": a.seed,
               "files": len(files), "notes": n_tok // 8, "model_load_s": round(load_s, 2),
               "prep_s": round(prep_s, 2), "render_s": round(gen_s, 1), "post_s": round(post_s, 2),
               "notes_per_s": round(n_tok / 8 / gen_s, 1),
               "music_hours": round(sum(r["seconds"] for r in rows) / 3600, 2),
               "peak_vram_alloc_gib": round(peak, 3),
               "peak_rss_gib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2),
               "single_job": single, "errors_total": sum(r["errors"] for r in rows),
               "date": time.strftime("%Y-%m-%d %H:%M")}
    print(json.dumps(summary, indent=1), flush=True)
    (out / "bench.json").write_text(json.dumps({"summary": summary, "files": rows}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()

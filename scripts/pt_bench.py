"""Pianist Transformer (phase 8, docs/TRANSKUN_PLAN.md): listening samples + benchmark.

Score MIDI -> expressive performance MIDI with yhj137/pianist-transformer-rendering (Apache-2.0).
Code: third_party/PianistTransformer (git clone of github.com/yhj137/PianistTransformer, not in
this repo); weights in /data/ext/pianist_transformer/models/sft; fast batched decoding in
pianocover/pianist.py.

  generate  famous public-domain pieces from PDMX (2 renders each, seeds 1 and 2) and the authors'
            23 test-set scores (1 render), batched on the GPU; plus single-piece timings
  speed     CPU timing (eager decoder, --cpu-threads) on one short test-set score
  metrics   the authors' objective metrics (src/evaluate/evaluate.py: Jensen-Shannon divergence
            and histogram intersection of velocity, duration, IOI and pedal patterns vs. the human
            performances) for: the plain score, the renders shipped in their repo (their original
            sampler), and ours (pianocover/pianist.py)
  audio     (pianocover:voice, needs fluidsynth) every MIDI -> MP3 with the YDP grand piano

  PYTHONPATH=/repo/third_party/PianistTransformer:/data/ext/pt_lib:/repo python scripts/pt_bench.py generate
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PT_CODE = REPO / "third_party" / "PianistTransformer"
TESTSET = PT_CODE / "data" / "midis" / "testset"
SAMPLES = {  # PDMX (CC0 / public-domain scores of public-domain works), mid path inside /data/pdmx/mid.tar.gz
    "chopin_nocturne_op55_2": "mid/15/11/QmXbFWcmEEAe6iErN3yVdL21iXXPk6yktQihaoYcjeMW8s.mid",
    "debussy_clair_de_lune": "mid/10/27/QmSJeDjtEuw3PjP17fheDNXGojcQ76HmntyE2T84D5zJYY.mid",
    "debussy_arabesque_1": "mid/10/25/QmSiufNxXsLotuQDJMEjTj3AbLr5H4K3mfB4LbjeB6UwTm.mid",
    "beethoven_moonlight_1": "mid/4/46/QmeuK3Unr7ybzDCvVVbU4Eu878nUjDhRJpZMyzzMY5EQyr.mid",
    "bach_prelude_c_bwv846": "mid/14/50/QmWwDho7Hbg4WTZLCJRZx3SKuKqbpWSFznDP9na7G2yACn.mid",
    "liszt_liebestraum_3": "mid/15/49/QmXVDzZEQFK4udkxBwDnRhVyFodUByGiwQN9AicG3YAobk.mid",
}
LISTEN_TEST = [0, 3, 11]  # test-set pieces also exported with a human performance


def load(weights, device, fix_prefix=False):
    import torch
    from src.model.pianoformer import PianoT5Gemma

    from pianocover.pianist import FastRenderer

    m = PianoT5Gemma.from_pretrained(weights, torch_dtype=torch.bfloat16).to(device).eval()
    return FastRenderer(m, device, fix_prefix=fix_prefix)


def timed(fr, midis, seed, **kw):
    import torch

    from pianocover.pianist import render_midis

    if fr.dev.type == "cuda":
        torch.cuda.synchronize()
    t = time.time()
    res = render_midis(fr, midis, seed=seed, **kw)
    if fr.dev.type == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t
    secs = 0.0
    for p in res:
        mp = p.get_tick_to_time_mapping()
        secs += float(mp[min(p.max_tick, len(mp) - 1)])
    n = sum(len(i.notes) for s in midis for i in s.instruments)
    return res, {"seconds": round(dt, 2), "pieces": len(midis), "notes": n, "music_sec": round(secs, 1),
                 "notes_per_s": round(n / dt, 1), "x_realtime": round(secs / dt, 1)}


def cmd_generate(a):
    import torch
    from miditoolkit import MidiFile

    out = Path(a.out)
    (out / "midi").mkdir(parents=True, exist_ok=True)
    fr = load(a.weights, "cuda", a.fix_prefix)
    famous = [(k, Path(a.pdmx) / v) for k, v in SAMPLES.items()]
    test = [(f"testset_{i:02d}", TESTSET / "score" / f"{i}.mid") for i in range(23)]
    for name, path in famous + test:
        shutil.copy(path, out / "midi" / f"{name}_score.mid")
    for i in LISTEN_TEST:
        shutil.copy(TESTSET / "human" / f"{i}-0.mid", out / "midi" / f"testset_{i:02d}_human.mid")
    runs = []
    timed(fr, [MidiFile(str(test[16][1]))], 0)  # warm-up (CUDA init, first graph captures)
    for kind, jobs, seed in (("single", [test[3]], 1), ("single", [test[6]], 1),
                             ("batch_famous", famous, 1), ("batch_famous", famous, 2), ("batch_testset", test, 1)):
        res, info = timed(fr, [MidiFile(str(p)) for _, p in jobs], seed)
        suffix = f"_pt{seed}" if kind != "single" else "_single"
        for (n, _), perf in zip(jobs, res):
            perf.dump(str(out / "midi" / f"{n}{suffix}.mid"))
        runs.append({"run": kind, "seed": seed, "names": [n for n, _ in jobs], **info})
        print(json.dumps(runs[-1]), flush=True)
    json.dump({"gpu": torch.cuda.get_device_name(), "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
               "fix_prefix": a.fix_prefix, "runs": runs}, open(out / "timing_gpu.json", "w"), indent=1)


def cmd_windows(a):
    """sequential / two_pass / parallel window scheduling: single-piece time and VRAM, test-set renders."""
    import torch
    from miditoolkit import MidiFile

    out = Path(a.out)
    (out / "midi_windows").mkdir(parents=True, exist_ok=True)
    fr = load(a.weights, "cuda", a.fix_prefix)
    test = [TESTSET / "score" / f"{i}.mid" for i in range(23)]
    runs = []
    timed(fr, [MidiFile(str(test[16]))], 0)  # warm-up
    for mode in ("sequential", "two_pass", "parallel"):
        for i in (3, 6):
            torch.cuda.reset_peak_memory_stats()
            res, info = timed(fr, [MidiFile(str(test[i]))], 1, windows=mode)
            runs.append({"mode": mode, "piece": f"testset_{i:02d}", **info,
                         "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)})
            print(json.dumps(runs[-1]), flush=True)
            res[0].dump(str(out / "midi_windows" / f"testset_{i:02d}_{mode}.mid"))
        (out / "renders" / mode).mkdir(parents=True, exist_ok=True)
        res, _ = timed(fr, [MidiFile(str(p)) for p in test], 1, windows=mode)
        for k, perf in enumerate(res):
            perf.dump(str(out / "renders" / mode / f"{k}.mid"))
    json.dump({"gpu": torch.cuda.get_device_name(), "runs": runs}, open(out / "timing_windows.json", "w"), indent=1)


def cmd_speed(a):
    import torch
    from miditoolkit import MidiFile

    torch.set_num_threads(a.cpu_threads)
    fr = load(a.weights, "cpu")
    _, info = timed(fr, [MidiFile(str(TESTSET / "score" / "16.mid"))], 1)
    info = {"run": "single testset_16", "device": f"cpu x{a.cpu_threads}", **info}
    print(json.dumps(info), flush=True)
    json.dump(info, open(Path(a.out) / "timing_cpu.json", "w"), indent=1)


def cmd_metrics(a):
    import contextlib
    import io

    from miditoolkit import MidiFile
    from src.evaluate import evaluate as ev
    from src.utils.midi import normalize_midi

    out = Path(a.out)
    norm = out / "norm"
    sets = {"human": TESTSET / "human", "score": TESTSET / "score", "authors_render": TESTSET / "performance",
            "our_render": None}
    for mode in ("two_pass", "parallel"):  # from the `windows` command, if run
        if (out / "renders" / mode).is_dir():
            sets[f"our_{mode}"] = out / "renders" / mode
    for name, src in sets.items():
        (norm / name).mkdir(parents=True, exist_ok=True)
        files = ([(out / "midi" / f"testset_{i:02d}_pt1.mid", f"{i}.mid") for i in range(23)] if src is None
                 else [(f, f.name) for f in sorted(src.glob("*.mid"))])
        for f, dst in files:
            normalize_midi(MidiFile(str(f))).dump(str(norm / name / dst))
    os.chdir(out)  # the evaluate functions write their plots to ./results/imgs
    res, lst = {}, []
    for name in [n for n in sets if n != "human"]:
        lst = ev.get_evaluate_list(str(norm / "human"), str(norm / name))
        with contextlib.redirect_stdout(io.StringIO()):
            vals = {"velocity": ev.plot_velocity_distribution_and_calculate_metrics_pt(lst, name),
                    "duration": ev.plot_duration_distribution_and_metrics_pt(lst, name),
                    "ioi": ev.plot_ioi_distribution_and_metrics_pt(lst, name),
                    "pedal": ev.plot_pedal_pattern_distribution(lst, name)}
        res[name] = {k: ({"js": round(float(v[0]), 4), "intersection": round(float(v[1]), 4)}
                         if v and v[0] is not None else None) for k, v in vals.items()}
        print(name, json.dumps(res[name]), flush=True)
    json.dump({"pairs": len(lst), "results": res}, open(out / "metrics.json", "w"), indent=1)


def cmd_audio(a):
    out = Path(a.out)
    (out / "mp3").mkdir(exist_ok=True)
    for mid in sorted((out / "midi").glob("*.mid")):
        mp3 = out / "mp3" / f"{mid.stem}.mp3"
        if mp3.exists() or (mid.stem.startswith("testset_") and int(mid.stem[8:10]) not in LISTEN_TEST):
            continue
        with tempfile.TemporaryDirectory() as td:
            subprocess.run(["fluidsynth", "-ni", "-q", "-g", "0.6", "-r", "44100", "-F", f"{td}/a.wav", a.sf2, str(mid)],
                           check=True, capture_output=True)
            subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", f"{td}/a.wav", "-af",
                            "silenceremove=stop_periods=-1:stop_duration=2:stop_threshold=-70dB,loudnorm=I=-18",
                            "-b:a", "192k", str(mp3)], check=True)
        print("mp3", mp3.name, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["generate", "windows", "speed", "metrics", "audio"])
    ap.add_argument("--weights", default="/data/ext/pianist_transformer/models/sft")
    ap.add_argument("--pdmx", default="/data/pt_bench/pdmx_src", help="the PDMX mids, extracted from /data/pdmx/mid.tar.gz")
    ap.add_argument("--out", default="/data/pt_bench")
    ap.add_argument("--cpu-threads", type=int, default=2)
    ap.add_argument("--fix-prefix", action="store_true", help="feed the window overlap aligned (see pianocover/pianist.py)")
    ap.add_argument("--sf2", default="/data/ext/sf2/YDP-GrandPiano-20160804.sf2")
    a = ap.parse_args()
    sys.path[:0] = [str(PT_CODE), str(REPO)]
    globals()[f"cmd_{a.cmd}"](a)


if __name__ == "__main__":
    main()

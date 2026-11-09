"""Benchmark the teacher pipeline as a resident server would run it.

  python -m pianocover.teacher.bench song1.mp3 song2.mp3 ... [--json bench.json]

1. Loads every model once and reports load time and RESIDENT VRAM, measured two ways:
   torch allocator (models' tensors) and device-level used memory (adds the CUDA context,
   cuDNN/cuBLAS workspaces and onnxruntime's own arena, which torch does not see).
2. For each song runs every step and reports latency and PEAK VRAM per step (torch
   allocator peak; onnxruntime allocations are reported via the device-level delta).
   The first song includes warm-up (cuDNN autotune, ORT kernel selection), so it is
   reported separately from the steady-state mean over the remaining songs.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch


def device_used_mib() -> float:
    free, total = torch.cuda.mem_get_info()
    return (total - free) / 2**20


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio", nargs="+")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--levels", default="easy,simple,normal,full,virtuoso")
    ap.add_argument("--demucs-shifts", type=int, default=0)
    ap.add_argument("--transkun-bf16", action="store_true")
    ap.add_argument("--cpu-basic-pitch", action="store_true")
    ap.add_argument("--json")
    args = ap.parse_args(argv)

    from .models import TeacherModels
    from .pipeline import TeacherPipeline, arrange_levels

    torch.cuda.init()
    base_dev = device_used_mib()
    t = time.perf_counter()
    models = TeacherModels(device=args.device, demucs_shifts=args.demucs_shifts,
                           basic_pitch_gpu=not args.cpu_basic_pitch, transkun_bf16=args.transkun_bf16)
    torch.cuda.synchronize()
    load_sec = time.perf_counter() - t
    resident_torch = torch.cuda.memory_allocated() / 2**20
    resident_dev = device_used_mib() - base_dev
    params = {
        "demucs": sum(p.numel() for p in models.demucs.parameters()) / 1e6,
        "transkun": sum(p.numel() for p in models.transkun.parameters()) / 1e6,
        "beat_this": sum(p.numel() for p in models.beat.model.parameters()) / 1e6,
        "btc": sum(p.numel() for p in models.btc.parameters()) / 1e6,
    }
    print(f"GPU: {torch.cuda.get_device_name(0)}  basic-pitch provider: {models.bp_provider}")
    print(f"load: {load_sec:.1f}s  " + ", ".join(f"{k} {v:.1f}s" for k, v in models.load_times.items()))
    print("params (M): " + ", ".join(f"{k} {v:.1f}" for k, v in params.items()))
    print(f"RESIDENT VRAM: torch tensors {resident_torch:.0f} MiB, device-level {resident_dev:.0f} MiB "
          f"(incl. CUDA context + ORT)")

    pipe = TeacherPipeline(models)
    levels = [s for s in args.levels.split(",") if s]
    runs = []
    for path in args.audio:
        dev_before = device_used_mib()
        t = time.perf_counter()
        a = pipe.analyze(path)
        t_an = time.perf_counter() - t
        t = time.perf_counter()
        res = arrange_levels(a, levels)
        t_arr = time.perf_counter() - t
        steps = {k: dict(v) for k, v in pipe.timings.items()}
        steps["arrange_all_levels"] = {"sec": t_arr, "peak_mib": 0.0}
        run = {"file": path, "duration_sec": a.duration, "analyze_sec": t_an, "arrange_sec": t_arr,
               "rtf": (t_an + t_arr) / a.duration, "steps": steps,
               "device_used_delta_mib": device_used_mib() - dev_before,
               "notes": {lv: s["notes"] for lv, (_, s) in res.items()}}
        runs.append(run)
        print(f"\n{path}: {a.duration:.0f}s audio -> analyze {t_an:.1f}s + arrange {t_arr:.2f}s "
              f"(RTF {run['rtf']:.3f}); device-level growth {run['device_used_delta_mib']:.0f} MiB")
        for k, v in steps.items():
            print(f"  {k:<20} {v['sec']:7.2f}s   peak {v['peak_mib']:7.0f} MiB")

    def summarize(rs):
        keys = list(rs[0]["steps"])
        return {k: {"sec_mean": float(np.mean([r["steps"][k]["sec"] for r in rs])),
                    "peak_mib_max": float(np.max([r["steps"][k]["peak_mib"] for r in rs]))} for k in keys}

    summary = {"gpu": torch.cuda.get_device_name(0), "basic_pitch_provider": models.bp_provider,
               "load_sec": load_sec, "load_times": models.load_times, "params_m": params,
               "resident_torch_mib": resident_torch, "resident_device_mib": resident_dev,
               "peak_device_used_mib": device_used_mib() - base_dev,
               "max_torch_peak_mib": max(v["peak_mib"] for r in runs for v in r["steps"].values()),
               "first_song": summarize(runs[:1]),
               "steady_state": summarize(runs[1:]) if len(runs) > 1 else None,
               "mean_rtf_steady": float(np.mean([r["rtf"] for r in runs[1:]])) if len(runs) > 1 else None,
               "runs": runs}
    if summary["steady_state"]:
        print("\nsteady state (excluding first song):")
        for k, v in summary["steady_state"].items():
            print(f"  {k:<20} {v['sec_mean']:7.2f}s   peak {v['peak_mib_max']:7.0f} MiB")
        print(f"  mean real-time factor {summary['mean_rtf_steady']:.3f}")
    print(f"\nmax torch peak during any step: {summary['max_torch_peak_mib']:.0f} MiB; "
          f"resident {resident_torch:.0f} MiB (torch) / {resident_dev:.0f} MiB (device)")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()

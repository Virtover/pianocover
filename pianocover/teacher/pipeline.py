"""Teacher pipeline CLI: songs -> aligned piano arrangements + an import_aligned manifest.

  # curated manifest from curate_fma (id,audio,license,attribution,...), paths relative to it
  python -m pianocover.teacher.pipeline /data/fma_cc/songs.csv --out /data/teacher \
      --levels simple,normal,full

  # single file (license must be given explicitly)
  python -m pianocover.teacher.pipeline song.mp3 --out /tmp/t --license CC-BY-4.0

  # re-run only the arranger from cached analyses (no GPU, fast) after tuning arrange.py
  python -m pianocover.teacher.pipeline /data/fma_cc/songs.csv --out /data/teacher --rearrange

Output in --out:
  analysis/<id>.npz        cached model outputs (see analysis.Analysis)
  midi/<id>_<level>.mid    arrangement on the song's timeline (offset 0)
  stems/<id>_<stem>.flac   optional (--save-stems), for listening checks
  manifest.csv             rows for `python -m pianocover.import_aligned` (one per level,
                           song_id groups the levels; attribution columns are kept)
  report.csv               per-song timings and arranger stats
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import traceback
from pathlib import Path

import numpy as np

from .. import events as events_mod
from .analysis import Analysis
from .arrange import LEVELS, Song, arrange


class TeacherPipeline:
    """Holds the resident models; `analyze()` is the GPU part, `arrange()` is numpy only."""

    def __init__(self, models=None, device: str = "cuda", **model_kw):
        if models is None:
            from .models import TeacherModels

            models = TeacherModels(device=device, **model_kw)
        self.m = models
        self.timings: dict = {}

    def _step(self, name, fn, *args, **kw):
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t = time.perf_counter()
        out = fn(*args, **kw)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() / 2**20
        else:
            peak = 0.0
        self.timings[name] = {"sec": time.perf_counter() - t, "peak_mib": peak}
        return out

    def analyze(self, audio_path, save_stems_to: Path | None = None) -> Analysis:
        from .models import SR, decode_audio, drum_onsets, rms_curve

        self.timings = {}
        m = self.m
        mix = self._step("decode", decode_audio, audio_path, SR, 2)
        mono = mix.mean(0)
        stems = self._step("demucs", m.separate, mix)
        beats, downbeats = self._step("beat_this", m.beats, mono, SR)
        chord_spans, chord_labels = self._step("btc_chords", m.chords, mono, SR)
        vocal = self._step("bp_vocals", m.basic_pitch, stems["vocals"], SR, min_freq=70, max_freq=1400,
                           min_note_ms=70)
        bass = self._step("bp_bass", m.basic_pitch, stems["bass"], SR, min_freq=28, max_freq=420,
                          min_note_ms=70)
        other = self._step("bp_other", m.basic_pitch, stems["other"], SR, min_note_ms=60, melodia_trick=False)
        other_piano = self._step("transkun_other", m.transcribe_piano, stems["other"])
        vocal_f0 = self._step("crepe_vocals", m.vocal_melody, stems["vocals"], SR,
                              onsets=vocal[:, 0] if len(vocal) else None)
        drums = self._step("drum_onsets", drum_onsets, stems["drums"], SR)
        rms = self._step("rms", lambda: (rms_curve(mono), rms_curve(stems["vocals"]), rms_curve(stems["other"])))
        if save_stems_to is not None:
            import soundfile as sf

            save_stems_to.parent.mkdir(parents=True, exist_ok=True)
            for k, v in stems.items():
                sf.write(f"{save_stems_to}_{k}.flac", v, SR)
        return Analysis(
            duration=mix.shape[1] / SR, beats=beats, downbeats=downbeats, chords=chord_spans,
            chord_labels=chord_labels, vocal=vocal, bass=bass, other=other, other_piano=other_piano,
            drum_onsets=drums, mix_rms=rms[0], vocal_rms=rms[1], other_rms=rms[2], vocal_f0=vocal_f0,
            meta={"source_audio": str(audio_path), "demucs": "htdemucs", "basic_pitch": m.bp_provider},
        )


def arrange_levels(a: Analysis, levels, keep_piano: bool | None = None, texture: str = "reduction"):
    song = Song(a)
    if texture == "piano":
        from .texture import arrange_piano

        return {lv: arrange_piano(a, lv, song) for lv in levels}
    return {lv: arrange(a, lv, song, keep_piano=keep_piano) for lv in levels}


def read_input(path: Path, license_: str | None):
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            r["audio_abs"] = str((path.parent / r["audio"]).resolve())
        return rows
    if not license_:
        raise SystemExit("--license is required for a single audio file (e.g. CC-BY-4.0, owned)")
    return [{"id": path.stem, "audio": str(path), "audio_abs": str(path.resolve()), "license": license_}]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="curated CSV manifest (id,audio,license,...) or one audio file")
    ap.add_argument("--out", required=True)
    ap.add_argument("--levels", default="simple,normal,full", help=f"comma list of {list(LEVELS)}")
    ap.add_argument("--license", help="license for a single audio file")
    ap.add_argument("--rearrange", action="store_true", help="only re-run the arranger from cached analyses")
    ap.add_argument("--reanalyze", action="store_true", help="ignore cached analyses and run the models again")
    ap.add_argument("--refresh-drums", action="store_true",
                    help="recompute only the kick/snare/hi-hat onsets of cached analyses from saved stems")
    ap.add_argument("--refresh-melody", action="store_true",
                    help="recompute only the CREPE vocal melody of cached analyses from saved stems")
    ap.add_argument("--save-stems", action="store_true")
    ap.add_argument("--texture", choices=["reduction", "piano"], default="reduction",
                    help="reduction: band rhythms copied; piano: song's notes laid out as piano figures "
                         "(files get a _piano suffix)")
    ap.add_argument("--keep-piano", action="store_true",
                    help="EXPERIMENTAL: keep piano-like bars of 'other' verbatim (misfires on guitar)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--demucs-shifts", type=int, default=0, help="more = slightly better stems, slower")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args(argv)

    levels = [s.strip() for s in args.levels.split(",") if s.strip()]
    for lv in levels:
        if lv not in LEVELS:
            raise SystemExit(f"unknown level {lv!r}; choose from {list(LEVELS)}")
    from ..import_aligned import license_ok

    out = Path(args.out)
    for sub in ("analysis", "midi"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    rows = read_input(Path(args.input), args.license)[: args.limit]
    pipe = None if (args.rearrange and not args.refresh_melody) else TeacherPipeline(device=args.device, demucs_shifts=args.demucs_shifts)

    manifest, report = [], []
    for i, r in enumerate(rows, 1):
        sid = r["id"]
        ok, why = license_ok(r.get("license", ""))
        if not ok:
            print(f"[{i}/{len(rows)}] {sid}: skipped, license refused ({why})", flush=True)
            continue
        npz = out / "analysis" / f"{sid}.npz"
        info = {"id": sid}
        try:
            t0 = time.perf_counter()
            if npz.exists() and not args.reanalyze:
                a = Analysis.load(npz)
                if args.refresh_drums:
                    stem = out / "stems" / f"{sid}_drums.flac"
                    if stem.exists():
                        import soundfile as sf

                        from .models import drum_onsets

                        y, sr = sf.read(str(stem), dtype="float32")
                        a.drum_onsets = drum_onsets(y.mean(1) if y.ndim > 1 else y, sr)
                        a.save(npz)
                    else:
                        print(f"[{i}/{len(rows)}] {sid}: no saved drum stem for --refresh-drums", flush=True)
                if args.refresh_melody:
                    stem = out / "stems" / f"{sid}_vocals.flac"
                    if not stem.exists():
                        print(f"[{i}/{len(rows)}] {sid}: no saved vocal stem for --refresh-melody "
                              "(use --reanalyze)", flush=True)
                    else:
                        import soundfile as sf

                        y, sr = sf.read(str(stem), dtype="float32")
                        y = y.mean(1) if y.ndim > 1 else y
                        a.vocal_f0 = pipe.m.vocal_melody(y, sr, onsets=a.vocal[:, 0] if len(a.vocal) else None)
                        a.save(npz)
            elif args.rearrange:
                print(f"[{i}/{len(rows)}] {sid}: no cached analysis, skipped", flush=True)
                continue
            else:
                a = pipe.analyze(r["audio_abs"], out / "stems" / sid if args.save_stems else None)
                a.save(npz)
                info.update({f"t_{k}": round(v["sec"], 3) for k, v in pipe.timings.items()})
                info.update({f"vram_{k}": round(v["peak_mib"]) for k, v in pipe.timings.items()})
            t1 = time.perf_counter()
            results = arrange_levels(a, levels, keep_piano=True if args.keep_piano else None,
                                     texture=args.texture)
            info["t_arrange_all"] = round(time.perf_counter() - t1, 3)
            info["t_total"] = round(time.perf_counter() - t0, 3)
            info["duration"] = round(a.duration, 1)
            for lv, (ev, stats) in results.items():
                tag = f"{lv}_piano" if args.texture == "piano" else lv
                mid = out / "midi" / f"{sid}_{tag}.mid"
                events_mod.to_pretty_midi(ev).write(str(mid))
                info.update({f"{lv}_{k}": v for k, v in stats.items() if k != "level"})
                manifest.append({
                    "id": f"{sid}_{tag}", "audio": r["audio_abs"], "midi": str(mid.resolve()),
                    "source": "teacher", "license": r["license"], "song_id": sid, "offset": 0,
                    "level": lv, "attribution": r.get("attribution", ""), "license_url": r.get("license_url", ""),
                })
            info["status"] = "ok"
            print(f"[{i}/{len(rows)}] {sid}: {a.duration:.0f}s audio, {info['t_total']:.1f}s, "
                  + ", ".join(f"{lv}={s['notes']} notes" for lv, (_, s) in results.items()), flush=True)
        except Exception as exc:
            info.update(status="error", error=f"{type(exc).__name__}: {exc}")
            print(f"[{i}/{len(rows)}] {sid}: ERROR {exc}\n{traceback.format_exc(limit=3)}", flush=True)
        report.append(info)

    if manifest:
        with (out / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(manifest[0]))
            w.writeheader()
            w.writerows(manifest)
    if report:
        keys = sorted({k for r in report for k in r}, key=lambda k: (k != "id", k))
        with (out / "report.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(report)
    print(f"done: {sum(r.get('status') == 'ok' for r in report)}/{len(report)} songs -> {out / 'manifest.csv'}")


if __name__ == "__main__":
    main()

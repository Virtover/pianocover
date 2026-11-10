"""Sanity-check the teacher pipeline's INPUTS (shared by every arranger mode).

  python scripts/diag_inputs.py /data/teacher_proto   (needs --save-stems output)

1. basic-pitch wrapper (models.TeacherModels.basic_pitch, batched, in-memory) vs the official
   basic_pitch.inference.predict() on the same vocal stem: note F1 and median onset shift.
2. beat grid vs drum onsets: median |offset| of drum hits from the nearest 8th-note grid
   point, against what a random grid would give (~1/4 of the 8th spacing).
3. transcribed vocal melody vs librosa.pyin f0 of the vocal stem: share of voiced frames
   where the melody has the same pitch (+-50 cents), same pitch class (octave error), or
   is silent; plus the best time lag between them.
"""
import glob
import os
import sys
import types

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pianocover.teacher.analysis import Analysis  # noqa: E402
from pianocover.teacher.arrange import Song, _notes  # noqa: E402

root = sys.argv[1] if len(sys.argv) > 1 else "/data/teacher_proto"
n_songs = int(sys.argv[2]) if len(sys.argv) > 2 else 4

import librosa  # noqa: E402
import mir_eval  # noqa: E402
import onnxruntime as ort  # noqa: E402
from basic_pitch import FilenameSuffix, build_icassp_2022_model_path  # noqa: E402
from basic_pitch.inference import predict  # noqa: E402

from pianocover.teacher.models import TeacherModels  # noqa: E402

model_path = str(build_icassp_2022_model_path(FilenameSuffix.onnx))
fake = types.SimpleNamespace(bp=ort.InferenceSession(model_path, providers=["CPUExecutionProvider"]))


def f1(ref, est):
    if len(ref) == 0 or len(est) == 0:
        return float("nan"), float("nan")
    hz = mir_eval.util.midi_to_hz
    p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref[:, :2], hz(ref[:, 2]), est[:, :2], hz(est[:, 2]), onset_tolerance=0.05, offset_ratio=None)
    m = mir_eval.transcription.match_notes(ref[:, :2], hz(ref[:, 2]), est[:, :2], hz(est[:, 2]),
                                           onset_tolerance=0.2, offset_ratio=None)
    shift = np.median([est[j, 0] - ref[i, 0] for i, j in m]) if m else float("nan")
    return f, shift


for path in sorted(glob.glob(os.path.join(root, "analysis", "*.npz")))[:n_songs]:
    sid = os.path.basename(path)[:-4]
    a = Analysis.load(path)
    song = Song(a)
    vstem = os.path.join(root, "stems", f"{sid}_vocals.flac")
    print(f"\n== {sid}")

    # 1. wrapper vs official basic-pitch
    if os.path.exists(vstem):
        y, sr = librosa.load(vstem, sr=44100, mono=True)
        mine = TeacherModels.basic_pitch(fake, y, sr, min_freq=70, max_freq=1400, min_note_ms=70)
        _, _, ev = predict(vstem, model_path, minimum_frequency=70, maximum_frequency=1400,
                           minimum_note_length=70, multiple_pitch_bends=False)
        official = np.array([[s, e, p, a_ * 127] for s, e, p, a_, _ in ev]).reshape(-1, 4)
        f, shift = f1(official, mine)
        print(f"basic-pitch wrapper vs official: {len(mine)} vs {len(official)} notes, "
              f"F1 {f:.3f}, median onset shift {1000 * shift:+.0f} ms")

    # 2. beat grid vs drums
    g = song.grid(2)
    d = np.asarray(a.drum_onsets).reshape(-1, 3)
    d = d[d[:, 2] > 0.3, 0]
    if len(d) and len(g) > 1:
        k = np.clip(np.searchsorted(g, d), 1, len(g) - 1)
        signed = np.where(d - g[k - 1] <= g[k] - d, d - g[k - 1], d - g[k])
        step = np.median(np.diff(g))
        print(f"beats: {len(a.beats)} beats, tempo {60 / song.period:.1f} bpm; drum hits vs 8th grid: "
              f"median |offset| {1000 * np.median(np.abs(signed)):.0f} ms (random ~{1000 * step / 4:.0f} ms), "
              f"median signed {1000 * np.median(signed):+.0f} ms")

    # 3. melody (basic-pitch notes and CREPE notes) vs pyin on the vocal stem
    if os.path.exists(vstem):
        y22, _ = librosa.load(vstem, sr=22050, mono=True)
        f0, voiced, _ = librosa.pyin(y22, fmin=70, fmax=1100, sr=22050, hop_length=512)
        t = librosa.times_like(f0, sr=22050, hop_length=512)
        midi_f0 = librosa.hz_to_midi(np.where(voiced, f0, np.nan))
        v = voiced & ~np.isnan(midi_f0)
        for src_name, voc in (("basic-pitch", _notes(a.vocal)), ("crepe-notes", _notes(a.vocal_f0))):
            if len(voc) == 0 or not v.any():
                continue
            m = np.full(len(t), np.nan)
            for s, e, p, _v in voc:
                m[(t >= s) & (t < e)] = p
            same = np.abs(m - midi_f0) < 0.5
            octave = (~same) & (np.abs(((m - midi_f0 + 6) % 12) - 6) < 0.5)
            silent = np.isnan(m)
            print(f"{src_name:<11} vs pyin (voiced frames): same pitch {100 * np.mean(same[v]):.0f}%, "
                  f"octave error {100 * np.mean(octave[v]):.0f}%, "
                  f"other pitch {100 * np.mean((~same & ~octave & ~silent)[v]):.0f}%, "
                  f"no note {100 * np.mean(silent[v]):.0f}%")

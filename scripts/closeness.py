"""How close is an arrangement to the SONG? Compares arranger modes objectively.

  python scripts/closeness.py /data/teacher_proto [level]

Needs the saved stems (pipeline --save-stems). Caches Transkun on the drumless mix
(vocals+bass+other) as <out>/analysis/<id>_mixpiano.npy.

harmony : mean cosine similarity between the song's chroma (drumless mix, CQT) and the
          arrangement's pitch-class roll (pedal included), over frames where both sound
melody  : of frames where the sung melody (CREPE notes) sounds, % where the arrangement's
          TOP note has the same pitch class (is the tune audible on top?)
rhythm  : onset F1 (70 ms) between the arrangement's onsets and the FULL song's onsets (drums included)
n/s     : notes per second
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import librosa  # noqa: E402
import mir_eval  # noqa: E402
import soundfile as sf  # noqa: E402

from pianocover.teacher.analysis import Analysis  # noqa: E402
from pianocover.teacher.arrange import Song, arrange, _notes  # noqa: E402
from pianocover.teacher.texture import arrange_piano  # noqa: E402
from pianocover.teacher.thin import arrange_thin  # noqa: E402

root = sys.argv[1] if len(sys.argv) > 1 else "/data/teacher_proto"
level = sys.argv[2] if len(sys.argv) > 2 else "normal"
HOP = 0.1
_tk = None


def drumless(sid):
    ys = []
    for st in ("vocals", "bass", "other"):
        y, sr = sf.read(os.path.join(root, "stems", f"{sid}_{st}.flac"), dtype="float32")
        ys.append(y if y.ndim == 1 else y.mean(1))
    n = min(len(y) for y in ys)
    return sum(y[:n] for y in ys), sr


def mix_piano(sid, y, sr):
    path = os.path.join(root, "analysis", f"{sid}_mixpiano.npy")
    if os.path.exists(path):
        return np.load(path)
    global _tk
    import torch

    from pianocover.events import from_transkun_notes
    from pianocover.model import load_pretrained_transkun

    if _tk is None:
        _tk = load_pretrained_transkun(device="cuda")
    x = torch.from_numpy(librosa.resample(y, orig_sr=sr, target_sr=44100)[:, None]).cuda()
    with torch.inference_mode():
        ev = from_transkun_notes(_tk.transcribe(x, discardSecondHalf=False))
    np.save(path, ev)
    return ev


def roll_chroma(ev, n):
    notes = ev[ev[:, 2] > 0].copy()
    ped = ev[ev[:, 2] == -64]
    for i, (s, e, p, v) in enumerate(notes):  # pedal sustains
        for ps, pe in ped[:, :2]:
            if ps <= e < pe:
                notes[i, 1] = max(notes[i, 1], pe)
    C = np.zeros((12, n))
    for s, e, p, v in notes:
        a, b = int(s / HOP), max(int(e / HOP), int(s / HOP) + 1)
        C[int(p) % 12, a:min(b, n)] += v / 127
    return C


def top_pc(ev, n):
    notes = ev[ev[:, 2] > 0]
    top = np.full(n, -1)
    topp = np.full(n, -1.0)
    for s, e, p, v in notes:
        a, b = int(s / HOP), min(max(int(e / HOP), int(s / HOP) + 1), n)
        upd = p > topp[a:b]
        topp[a:b] = np.where(upd, p, topp[a:b])
    top[topp >= 0] = topp[topp >= 0].astype(int) % 12
    return top


def scores(ev, song_chroma, mel_pc, song_on, dur):
    n = song_chroma.shape[1]
    C = roll_chroma(ev, n)
    both = (C.sum(0) > 0) & (song_chroma.sum(0) > 1e-3)
    cs = (C[:, both] * song_chroma[:, both]).sum(0) / (
        np.linalg.norm(C[:, both], axis=0) * np.linalg.norm(song_chroma[:, both], axis=0) + 1e-9)
    tp = top_pc(ev, n)
    mv = mel_pc >= 0
    mel = 100 * np.mean(tp[mv] == mel_pc[mv]) if mv.any() else np.nan
    on = np.unique(np.round(ev[ev[:, 2] > 0, 0], 2))
    f = mir_eval.onset.f_measure(song_on, on, window=0.07)[0] if len(on) and len(song_on) else np.nan
    return float(np.mean(cs)) if both.any() else np.nan, mel, f, (ev[:, 2] > 0).sum() / dur


rows = []
print(f"level {level}\n{'song':<11} {'mode':<10} {'harmony':>7} {'melody%':>7} {'rhythm':>6} {'n/s':>5}")
for path in sorted(glob.glob(os.path.join(root, "analysis", "fma_*.npz"))):
    sid = os.path.basename(path)[:-4]
    a = Analysis.load(path)
    song = Song(a)
    y, sr = drumless(sid)
    y22 = librosa.resample(y, orig_sr=sr, target_sr=22050)
    chroma = librosa.feature.chroma_cqt(y=y22, sr=22050, hop_length=int(22050 * HOP))
    yd, srd = sf.read(os.path.join(root, "stems", f"{sid}_drums.flac"), dtype="float32")
    yd = yd if yd.ndim == 1 else yd.mean(1)
    full = librosa.resample(y + yd[:len(y)] if len(yd) >= len(y) else y, orig_sr=sr, target_sr=22050)
    song_on = librosa.onset.onset_detect(y=full, sr=22050, units="time")
    n = chroma.shape[1]
    mel_pc = np.full(n, -1)
    for s, e, p, v in _notes(a.vocal_f0):
        mel_pc[int(s / HOP):min(int(e / HOP) + 1, n)] = int(p) % 12
    mp = mix_piano(sid, y, sr)
    cands = {
        "reduction": arrange(a, level, song)[0],
        "piano": arrange_piano(a, level, song)[0],
        "tk_raw": mp,
        "thin": arrange_thin(a, level, song, mix_notes=mp)[0],
    }
    for mode, ev in cands.items():
        h, m, r, d = scores(ev, chroma, mel_pc, song_on, a.duration)
        rows.append((mode, h, m, r, d))
        print(f"{sid:<11} {mode:<10} {h:7.3f} {m:7.1f} {r:6.3f} {d:5.1f}", flush=True)

print("\nmean over songs:")
for mode in ("reduction", "piano", "tk_raw", "thin"):
    r = np.array([x[1:] for x in rows if x[0] == mode], float)
    print(f"  {mode:<10} harmony {np.nanmean(r[:, 0]):.3f}  melody {np.nanmean(r[:, 1]):5.1f}%  "
          f"rhythm {np.nanmean(r[:, 2]):.3f}  n/s {np.nanmean(r[:, 3]):.1f}")

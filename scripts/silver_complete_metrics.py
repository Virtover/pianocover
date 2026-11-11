"""Metrics for ACE-Step complete/lego outputs in data/silver_test/complete/. Run in pianocover:dev."""
import glob, os, sys, subprocess
import numpy as np, librosa
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import silver_feasibility as sf
R = sf.ROOT; SR = sf.SR; HOP = sf.HOP
load = lambda p: librosa.load(p, sr=SR, mono=True)[0]
def logspec(y):
    return np.log1p(np.abs(librosa.stft(y, n_fft=2048, hop_length=HOP)) * 10)
def corr(a, b):
    n = min(a.shape[1], b.shape[1]); a, b = a[:, :n].ravel(), b[:, :n].ravel()
    return float(np.corrcoef(a, b)[0, 1])
def band_onset(y, lo, hi):
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=HOP / 2 if False else HOP)) ** 2
    f = librosa.fft_frequencies(sr=SR, n_fft=2048); m = (f >= lo) & (f < hi)
    e = S[m].sum(0); d = np.maximum(np.diff(np.log1p(e * 1e3)), 0)
    return float(d.mean())
def added(yi, yo):
    n = min(len(yi), len(yo)); yi, yo = yi[:n], yo[:n]
    best = 1e9
    # best scaled input, search small lag too
    for lag in range(-400, 401, 40):
        a = yi[max(0, lag):n + min(0, lag)]; b = yo[max(0, -lag):n - max(0, lag)]
        g = (a @ b) / (a @ a + 1e-9); r = ((b - g * a) ** 2).sum() / ((b ** 2).sum() + 1e-9)
        best = min(best, r)
    return float(best)
inputs = {(i, p): load(f"{R}/{i}/{p}.wav") for i in ("piano", "guitar") for p in ("entertainer", "bethena")}
outs = {os.path.basename(f)[:-5]: load(f) for f in sorted(glob.glob(f"{R}/complete/*.flac"))}
def bounds(name):
    task, piece, inp, tag = name.split("__"); return task, piece, inp, tag
rows = []
for name, y in outs.items():
    task, piece, inp, tag = bounds(name); yi = inputs[(inp, piece)]
    mid = f"{R}/src/{piece}.mid"; other = "bethena" if piece == "entertainer" else "entertainer"
    h = sf.harmony(mid, y); o = sf.onset_f(mid, y)
    lb = sf.harmony(f"{R}/src/{other}.mid", y)[0]  # this output vs OTHER piece's MIDI
    ub = sf.harmony(mid, yi)[0]  # input itself vs MIDI
    ub_o = sf.onset_f(mid, yi)[0]
    # rms ratio, spectrogram corr
    n = min(len(y), len(yi)); rr = float(np.sqrt((y[:n] ** 2).mean() / (yi[:n] ** 2).mean()))
    sc = corr(logspec(yi), logspec(y))
    ad = added(yi, y)
    lo_r = band_onset(y, 20, 150) / (band_onset(yi, 20, 150) + 1e-9)
    hi_r = band_onset(y, 5000, 12000) / (band_onset(yi, 5000, 12000) + 1e-9)
    rows.append((name, h[0], h[2], h[1], lb, ub, o[0], ub_o, sc, rr, ad, lo_r, hi_r))
L = ["| output | harm0 | harm@lag | lag s | harm vs OTHER midi (lb) | harm of input (ub) | onsetF | onsetF input | spec corr | out/in RMS | added frac of energy | <150Hz onset x | >5k onset x |", "|" + "---|" * 13]
for r in rows: L.append(f"| {r[0]} | " + " | ".join(f"{v:.3f}" for v in r[1:]) + " |")
open(f"{R}/listen_complete/metrics.md", "w").write("\n".join(L) + "\n"); print("\n".join(L))

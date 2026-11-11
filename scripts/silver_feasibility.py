"""Feasibility check: does ACE-Step cover mode keep the piano's notes/timing?

Layout under data/silver_test/:
  midi_full/<piece>.mid   downloaded Mutopia MIDIs
  src/<piece>.mid         trimmed (first 120 s), all program 0 (piano)
  piano/<piece>.wav       fluidsynth FluidR3_GM render, 44.1 kHz
  covers/<piece>__s<strength>.flac   ACE-Step output

  python scripts/silver_feasibility.py prep      # trim + render (needs fluidsynth)
  python scripts/silver_feasibility.py analyze   # metrics -> metrics.json / metrics.md
"""
import glob
import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "silver_test")
SF2 = "/usr/share/sounds/sf2/FluidR3_GM.sf2"
MAXLEN = 120.0
SR = 25600
HOP = 640  # 0.025 s
FR = HOP / SR


def prep():
    import pretty_midi
    os.makedirs(f"{ROOT}/src", exist_ok=True)
    os.makedirs(f"{ROOT}/piano", exist_ok=True)
    for f in sorted(glob.glob(f"{ROOT}/midi_full/*.mid")):
        name = os.path.basename(f)[:-4]
        pm = pretty_midi.PrettyMIDI(f)
        out = pretty_midi.PrettyMIDI()
        inst = pretty_midi.Instrument(program=0)
        for i in pm.instruments:
            if i.is_drum:
                continue
            for n in i.notes:
                if n.start < MAXLEN:
                    inst.notes.append(pretty_midi.Note(n.velocity, n.pitch, n.start, min(n.end, MAXLEN)))
        out.instruments.append(inst)
        out.write(f"{ROOT}/src/{name}.mid")
        subprocess.run(["fluidsynth", "-ni", "-q", "-r", "44100", "-F", f"{ROOT}/piano/{name}.wav",
                        SF2, f"{ROOT}/src/{name}.mid"], check=True)
        print(name, len(inst.notes), "notes")


def pc_roll(mid, n):
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(mid)
    r = np.zeros((12, n))
    for nt in pm.instruments[0].notes:
        a, b = int(nt.start / FR), max(int(nt.start / FR) + 1, int(nt.end / FR))
        r[nt.pitch % 12, a:min(b, n)] += 1
    return r


def chroma(y):
    import librosa
    return librosa.feature.chroma_cqt(y=y, sr=SR, hop_length=HOP, n_octaves=6)


def pool4(x):
    n = x.shape[1] // 4 * 4
    return x[:, :n].reshape(12, -1, 4).mean(2)


def cos_mean(a, b):
    n = min(a.shape[1], b.shape[1])
    a, b = a[:, :n], b[:, :n]
    na, nb = np.linalg.norm(a, axis=0), np.linalg.norm(b, axis=0)
    ok = (na > 1e-6) & (nb > 1e-6)
    return float(((a * b).sum(0)[ok] / (na[ok] * nb[ok])).mean())


def harmony(mid, y):
    c = chroma(y)
    r = pc_roll(mid, c.shape[1])
    n = min(c.shape[1], r.shape[1])
    c, r = c[:, :n], r[:, :n]
    zero = cos_mean(pool4(r), pool4(c))
    best = (-1, 0.0)
    for k in range(-20, 21):  # +-0.5 s; k>0: cover lags the MIDI
        if k >= 0:
            rr, cc = r[:, :n - k], c[:, k:]
        else:
            rr, cc = r[:, -k:], c[:, :n + k]
        s = cos_mean(pool4(rr), pool4(cc))
        if s > best[0]:
            best = (s, k * FR)
    return zero, best[1], best[0]


def onset_f(mid, y):
    import librosa
    import mir_eval
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(mid)
    ref = np.unique(np.round([n.start for n in pm.instruments[0].notes], 3))
    est = librosa.onset.onset_detect(y=y, sr=SR, hop_length=HOP // 2, units="time", backtrack=False)
    f, p, r = mir_eval.onset.f_measure(ref, est, window=0.07)
    return float(f), float(p), float(r)


def drift(ya, yb):
    import librosa
    ea = librosa.onset.onset_strength(y=ya, sr=SR, hop_length=HOP // 2)
    eb = librosa.onset.onset_strength(y=yb, sr=SR, hop_length=HOP // 2)
    fr = HOP / 2 / SR
    w, mx = int(10 / fr), int(0.5 / fr)
    lags = []
    for s in range(0, min(len(ea), len(eb)) - w + 1, w):
        a = ea[s:s + w] - ea[s:s + w].mean()
        b = eb[max(0, s - mx):s + w + mx]
        b = b - b.mean()
        if a.std() < 1e-6 or len(b) < w:
            continue
        cc = np.correlate(b, a, mode="valid")  # index i: a aligned to b[i:]
        i = int(np.argmax(cc))
        lags.append((i - (s - max(0, s - mx))) * fr * 1000)
    return (float(np.min(lags)), float(np.median(lags)), float(np.max(lags))) if lags else (np.nan,) * 3


def analyze():
    import librosa
    load = lambda p: librosa.load(p, sr=SR, mono=True)[0]
    pieces = sorted(os.path.basename(f)[:-4] for f in glob.glob(f"{ROOT}/src/*.mid"))
    piano = {p: load(f"{ROOT}/piano/{p}.wav") for p in pieces}
    rows = []
    # upper bound: piano render itself
    for p in pieces:
        h = harmony(f"{ROOT}/src/{p}.mid", piano[p])
        o = onset_f(f"{ROOT}/src/{p}.mid", piano[p])
        rows.append(dict(piece=p, kind="piano_render", strength="-", dur_src=len(piano[p]) / SR,
                         dur_out=len(piano[p]) / SR, harm0=h[0], lag_s=h[1], harm_lag=h[2],
                         onset_f=o[0], onset_p=o[1], onset_r=o[2], drift_min=0, drift_med=0, drift_max=0))
    covers = {}
    for f in sorted(glob.glob(f"{ROOT}/covers/*.flac")):
        p, s = os.path.basename(f)[:-5].split("__s")
        covers[(p, s)] = load(f)
    for (p, s), y in covers.items():
        h = harmony(f"{ROOT}/src/{p}.mid", y)
        o = onset_f(f"{ROOT}/src/{p}.mid", y)
        d = drift(piano[p], y)
        # lower bound: same cover strength of a different piece vs this piece's MIDI
        lb_h, lb_o = [], []
        for (q, s2), y2 in covers.items():
            if s2 == s and q != p:
                lb_h.append(harmony(f"{ROOT}/src/{p}.mid", y2)[0])
                lb_o.append(onset_f(f"{ROOT}/src/{p}.mid", y2)[0])
        rows.append(dict(piece=p, kind="cover", strength=s, dur_src=len(piano[p]) / SR, dur_out=len(y) / SR,
                         harm0=h[0], lag_s=h[1], harm_lag=h[2], onset_f=o[0], onset_p=o[1], onset_r=o[2],
                         drift_min=d[0], drift_med=d[1], drift_max=d[2],
                         lb_harm=float(np.mean(lb_h)) if lb_h else None,
                         lb_onset_f=float(np.mean(lb_o)) if lb_o else None))
    json.dump(rows, open(f"{ROOT}/metrics.json", "w"), indent=1)
    hdr = "piece | kind | str | dur src/out | harm@0 | best lag s | harm@lag | harm LB | onsetF | onsetP | onsetR | onsetF LB | drift ms min/med/max"
    lines = [hdr]
    fmt = lambda v: "-" if v is None else f"{v:.3f}"
    for r in rows:
        lines.append(f"{r['piece']} | {r['kind']} | {r['strength']} | {r['dur_src']:.1f}/{r['dur_out']:.1f} | "
                     f"{r['harm0']:.3f} | {r['lag_s']:+.3f} | {r['harm_lag']:.3f} | {fmt(r.get('lb_harm'))} | "
                     f"{r['onset_f']:.3f} | {r['onset_p']:.3f} | {r['onset_r']:.3f} | {fmt(r.get('lb_onset_f'))} | "
                     f"{r['drift_min']:.0f}/{r['drift_med']:.0f}/{r['drift_max']:.0f}")
    open(f"{ROOT}/metrics.md", "w").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    {"prep": prep, "analyze": analyze}[sys.argv[1]]()

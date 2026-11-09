"""Tests for the torch-free parts. Run: python -m pytest tests/ (or python tests/test_core.py)."""

import os
import sys
import tempfile
import types

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pianocover import FRAME_SEC, events, postprocess, style  # noqa: E402
from pianocover.audio import read_wav_slice  # noqa: E402


def transkun_target_ok(ev, frame=FRAME_SEC):
    """Re-implements transkun.Data.validateNotes + prepareIntervals' merge condition."""
    by_pitch = {}
    for s, e, p, v in ev:
        assert s < e, (s, e)
        assert s >= 0
        by_pitch.setdefault(p, []).append((s, e))
    for p, notes in by_pitch.items():
        notes.sort()
        prev = None
        for s, e in notes:
            sq, eq = round(s / frame), round(e / frame)
            if prev is not None:
                assert s >= prev[1] - 1e-12, f"overlap pitch {p}"
                psq, peq = prev[2], prev[3]
                assert not (sq < peq or (eq == peq and psq == sq)), f"would be merged pitch {p}"
            prev = (s, e, sq, eq)


def random_events(rng, n=3000, dur=200.0):
    s = rng.uniform(0, dur, n)
    d = rng.exponential(0.3, n)
    p = rng.integers(15, 115, n).astype(float)
    p[rng.random(n) < 0.03] = -64
    v = rng.integers(1, 128, n).astype(float)
    return np.stack([s, s + d, p, v], 1)


def test_sanitize_produces_valid_transkun_targets():
    rng = np.random.default_rng(0)
    for _ in range(5):
        ev = events.sanitize(random_events(rng))
        transkun_target_ok(ev)
        assert ((ev[:, 2] == -64) | ((ev[:, 2] >= 21) & (ev[:, 2] <= 108))).all()


def test_warp_then_sanitize_valid_and_monotonic():
    rng = np.random.default_rng(1)
    ev = events.sanitize(random_events(rng))
    src = np.linspace(0, 200, 400)
    dst = np.cumsum(rng.uniform(0.2, 1.0, 400))  # irregular, strictly increasing tempo map
    w = events.sanitize(events.warp(ev, src, dst))
    transkun_target_ok(w)
    # onset order is preserved by a monotonic map
    i = np.argsort(ev[:, 0])[:50]
    assert np.all(np.diff(np.interp(ev[i, 0], src, dst)) >= 0)


def test_slice_flags_and_bounds():
    ev = np.array([[1.0, 3.0, 60, 80], [4.0, 5.0, 62, 70], [9.0, 12.0, 64, 60], [12.5, 13, 65, 50]])
    sub, on, off = events.slice_events(ev, 2.0, 10.0)
    assert len(sub) == 3
    assert on.tolist() == [False, True, True] and off.tolist() == [True, True, False]
    assert sub[:, 0].min() >= 0 and sub[:, 1].max() <= 8.0


def test_transpose_drops_out_of_range():
    ev = np.array([[0, 1, 107, 80], [0, 1, 60, 80], [0, 1, -64, 127]], float)
    out = events.transpose(ev, 3)
    assert sorted(out[:, 2].tolist()) == [-64, 63]


def test_style_and_presets():
    rng = np.random.default_rng(2)
    raws = np.stack([style.raw_style(events.sanitize(random_events(rng, n=500 + 300 * k))) for k in range(6)])
    assert raws.shape == (6, style.DIM) and np.isfinite(raws).all()
    assert raws[-1, 0] > raws[0, 0]  # more notes -> higher onset rate
    norm = style.StyleNormalizer.fit(raws)
    with tempfile.TemporaryDirectory() as d:
        norm.save(os.path.join(d, "s.json"))
        again = style.StyleNormalizer.load(os.path.join(d, "s.json"))
    assert np.allclose(again.normalize(raws), norm.normalize(raws))
    v = style.preset_vector("easy", pitch_mean=0.5)
    assert v[style.FEATURES.index("polyphony")] == -1.5 and v[style.FEATURES.index("pitch_mean")] == 0.5


def test_limit_hands():
    chord = [[0, 1, p, 80] for p in (36, 40, 43, 48, 52, 60, 64, 67, 72, 76, 79, 84)]
    out = postprocess.limit_hands(np.array(chord, float), max_per_hand=4, span=12)
    rh = out[out[:, 2] >= 60][:, 2]
    lh = out[out[:, 2] < 60][:, 2]
    assert len(rh) <= 4 and len(lh) <= 4
    assert rh.max() == 84 and lh.min() == 36            # melody and bass kept
    assert rh.max() - rh.min() <= 12 and lh.max() - lh.min() <= 12


def test_quantize_snaps_and_stays_valid():
    beats = np.arange(0, 20, 0.5)
    rng = np.random.default_rng(3)
    on = np.round(rng.uniform(0, 19, 200) / 0.125) * 0.125 + rng.normal(0, 0.02, 200)
    ev = np.stack([np.abs(on), np.abs(on) + 0.3, rng.integers(40, 90, 200), np.full(200, 70)], 1)
    q = postprocess.quantize(events.sanitize(ev), beats)
    transkun_target_ok(q)
    grid = postprocess.beat_grid(beats)
    dist = np.min(np.abs(q[:, 0][:, None] - grid[None]), 1)
    assert (dist < 1e-9).mean() > 0.9


def test_wav_slice_padding():
    from scipy.io import wavfile

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "a.wav")
        x = (np.sin(np.arange(44100) / 10) * 20000).astype(np.int16)
        wavfile.write(path, 44100, x)
        y = read_wav_slice(path, -0.5, 0.5)
        assert y.shape == (44100, 1)
        assert np.all(y[:22050] == 0) and np.abs(y[22050:]).max() > 0.5
        z = read_wav_slice(path, 0.9, 1.4)
        assert abs(z.shape[0] - 22050) <= 1 and np.all(z[4411:] == 0)  # float rounding: +-1 sample


def _numpy_dtw(C):
    n, m = C.shape
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = C[i - 1, j - 1] + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    i, j, path = n, m, []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        k = np.argmin([D[i - 1, j - 1], D[i - 1, j], D[i, j - 1]])
        i, j = (i - 1, j - 1) if k == 0 else (i - 1, j) if k == 1 else (i, j - 1)
    return D[1:, 1:], np.array(path)


def test_detect_transpose_sign():
    fake = types.ModuleType("librosa")
    fake.sequence = types.SimpleNamespace(dtw=lambda C, subseq=False, backtrack=True: _numpy_dtw(C))
    sys.modules["librosa"] = fake
    from pianocover.align import detect_transpose

    rng = np.random.default_rng(4)
    pop = rng.random((12, 400)) ** 4  # 40 coarse frames after pooling
    for shift_up in (0, 3, 7, 11):
        cover = np.roll(pop, shift_up, axis=0)  # cover played `shift_up` semitones higher
        k, margin = detect_transpose(pop, cover)
        assert k == shift_up and margin > 0
        add = -k if k <= 6 else 12 - k  # same rule as align.align()
        assert (shift_up + add) % 12 == 0 and abs(add) <= 6
    del sys.modules["librosa"]




def test_key_shift_audio_and_targets_agree():
    """Mirror of data.random_key_shift's resampling in numpy + events.shift_events."""
    fs, chunk = 44100, 4.0
    n = int(fs * chunk)
    t = np.arange(n) / fs
    x = np.sin(2 * np.pi * 440 * t) * (t < 2.0)  # A4 sounding 0..2 s
    ev = np.array([[0.0, 2.0, 69, 80], [3.9, 4.0, 108, 60], [1.0, 3.9, 60, 70]])
    on, off = np.ones(3, bool), np.ones(3, bool)
    for k in (-2, 2):
        r = 2 ** (k / 12)
        pos = np.arange(n) * r
        valid = pos < n - 1
        i0 = np.clip(np.floor(pos), 0, n - 2).astype(int)
        fr = pos - i0
        y = (x[i0] * (1 - fr) + x[i0 + 1] * fr) * valid
        f_peak = np.argmax(np.abs(np.fft.rfft(y))) * fs / n
        sounding = np.where(np.abs(y) > 0.1)[0]
        audio_end = sounding[-1] / fs
        out, o_on, o_off = events.shift_events(ev, on, off, k, chunk)
        a4 = out[out[:, 2] == 69 + k]
        assert abs(12 * np.log2(f_peak / 440) - k) < 0.1          # audio moved k semitones
        assert len(a4) == 1 and abs(a4[0, 1] - audio_end) < 0.01  # target offset matches audio
        assert out[:, 1].max() <= chunk + 1e-9
        assert (110 not in out[:, 2]) and (106 not in out[:, 2] or k == -2)
        if k == -2:
            assert (~o_off).any()  # notes cut by the shortened input window lose their offset



def test_license_gate():
    from pianocover.import_aligned import license_ok
    accept = ["CC0", "cc0-1.0", "CC-BY-4.0", "CC BY 3.0", "cc_by_2.5", "public-domain", "owned",
              "work-for-hire", "ace-step", "commissioned"]
    refuse = ["CC-BY-NC-4.0", "CC BY-NC-SA 3.0", "CC-BY-ND-4.0", "cc-by-nc-nd", "unknown", "",
              "All rights reserved", "MIT", "non-commercial use only", "CC-BY-SA-4.0"]
    for lic in accept:
        assert license_ok(lic)[0], lic
    for lic in refuse:
        assert not license_ok(lic)[0], lic
    assert license_ok("CC-BY-SA-4.0", allow_sharealike=True)[0]
    assert not license_ok("CC-BY-NC-SA-4.0", allow_sharealike=True)[0]


def test_pedal_intervals():
    from pianocover.import_aligned import pedal_intervals
    ccs = [(1.0, 127), (1.5, 100), (2.0, 0), (3.0, 64), (3.0, 64), (4.0, 10), (5.0, 90)]
    assert pedal_intervals(ccs, 64, end_time=6.0) == [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]
    assert pedal_intervals([], 64, 6.0) == []
    assert pedal_intervals([(2.0, 0)], 64, 6.0) == []


def test_merge_and_source_weights():
    import pickle
    from pianocover import merge_datasets
    from pianocover.sampling import chunk_weights, parse_source_weights, source_summary
    rng = np.random.default_rng(5)

    def item(i, source, lic, song):
        ev = events.sanitize(random_events(rng, n=300, dur=60))
        return {"pop_id": song, "piano_id": f"{source}{i}", "audio": f"a{i}.wav", "duration": 60.0,
                "events": ev.astype(np.float32), "bad_regions": [], "raw_style": style.raw_style(ev),
                "source": source, "license": lic}

    with tempfile.TemporaryDirectory() as d:
        for name, src, lic in (("t", "teacher", "CC-BY-4.0"), ("g", "gold", "work-for-hire"), ("w", "web", "unknown")):
            os.makedirs(os.path.join(d, name))
            train = [item(i, src, lic, f"song{i}") for i in range(4)]
            val = [item(9, src, lic, "song_val")]
            for split, items in (("train", train), ("val", val)):
                with open(os.path.join(d, name, split + ".pkl"), "wb") as f:
                    pickle.dump({"data_root": f"/root_{name}", "items": items}, f)
        out = os.path.join(d, "all")
        merge_datasets.main([os.path.join(d, n) for n in "tgw"] + ["--out", out, "--licensed-only"])
        with open(os.path.join(out, "train.pkl"), "rb") as f:
            merged = pickle.load(f)["items"]
        with open(os.path.join(out, "val.pkl"), "rb") as f:
            merged_val = pickle.load(f)["items"]
        assert {it["source"] for it in merged} == {"teacher", "gold"}        # web dropped
        assert not ({it["pop_id"] for it in merged} & {it["pop_id"] for it in merged_val})  # no song leak
        assert all(it["data_root"] in ("/root_t", "/root_g") for it in merged)
        assert all(np.isfinite(it["style"]).all() for it in merged)

    sw = parse_source_weights("gold=4, teacher=1,silver=0")
    assert sw == {"gold": 4.0, "teacher": 1.0, "silver": 0.0}

    class DS:
        items = [{"source": "gold"}, {"source": "teacher"}, {"source": "silver"}]
        chunks = [(0, 0, 16), (1, 0, 16), (1, 8, 24), (2, 0, 16)]

    assert chunk_weights(DS, sw) == [4.0, 1.0, 1.0, 0.0]
    assert chunk_weights(DS, {}) is None
    summary = source_summary(DS, sw)
    assert abs(summary["gold"] - 4 / 6) < 1e-3 and summary["silver"] == 0.0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)

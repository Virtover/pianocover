"""Torch-free tests for the teacher pipeline (chords, license mapping, arranger, analysis I/O).

  python tests/test_teacher.py
"""

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pianocover.events import END, PITCH, START, sanitize  # noqa: E402
from pianocover.import_aligned import license_ok  # noqa: E402
from pianocover.teacher.analysis import FRAME_HOP, Analysis  # noqa: E402
from pianocover.teacher.arrange import LEVELS, Song, arrange, monophonize  # noqa: E402
from pianocover.teacher.chords import parse_chord  # noqa: E402
from pianocover.teacher.curate_fma import canonical_license  # noqa: E402

PROG = [("C", (0, 4, 7)), ("G", (7, 11, 2)), ("A:min", (9, 0, 4)), ("F", (5, 9, 0))]


def synthetic_song(seed=0, guitar=True, vocals=True):
    """16 bars of 4/4 at 120 bpm, C-G-Am-F, strummed guitar with injected non-chord noise."""
    rng = np.random.default_rng(seed)
    beat = 0.5
    n_bars = 16
    dur = n_bars * 4 * beat
    beats = np.arange(0, dur, beat)
    downbeats = beats[::4]
    chords, labels, bass, other, vocal = [], [], [], [], []
    for b in range(n_bars):
        lab, pcs = PROG[b % 4]
        s = b * 2.0
        chords.append([s, s + 2.0])
        labels.append(lab)
        root = pcs[0]
        bass += [[s + k * 1.0, s + k * 1.0 + 0.9, 36 + root, 90] for k in range(2)]
        if guitar:
            for k in range(8):  # eighth-note strums, 3-4 strings, tiny strum spread
                t = s + k * 0.25
                for j, pc in enumerate(pcs):
                    other.append([t + 0.008 * j, t + 0.24, 52 + (pc - 52) % 12 + 12 * (j == 2), 70])
            other.append([s + 0.5, s + 0.74, 61, 75])  # C#: wrong note inside a strum, every bar
        if vocals and b % 4 != 3:  # last bar of each phrase: no vocal -> instrumental gap
            for k in range(4):
                pc = pcs[k % 3]
                vocal.append([s + k * 0.5, s + k * 0.5 + 0.45, 60 + (pc - 60) % 12 + 12, 80])
    mix_rms = np.full(int(dur / FRAME_HOP) + 1, 0.1)
    mix_rms[len(mix_rms) // 2:] = 0.3  # second half louder
    return Analysis(
        duration=dur, beats=beats, downbeats=downbeats, chords=np.array(chords), chord_labels=labels,
        vocal=np.array(vocal).reshape(-1, 4), bass=np.array(bass).reshape(-1, 4),
        other=np.array(other).reshape(-1, 4), other_piano=np.zeros((0, 4)),
        drum_onsets=np.array([[t, 1.0 if i % 2 else 0.0, 0.8] for i, t in enumerate(beats)]),
        mix_rms=mix_rms, vocal_rms=mix_rms * 0, other_rms=mix_rms * 0)


def chord_pcs_at(a, t):
    for (s, e), lab in zip(a.chords, a.chord_labels):
        if s <= t < e:
            return parse_chord(lab)[1]
    return None


def test_parse_chord():
    assert parse_chord("N") is None and parse_chord("X") is None
    root, pcs, bass = parse_chord("C#:min7")
    assert root == 1 and pcs == frozenset({1, 4, 8, 11}) and bass == 1
    assert parse_chord("G")[1] == frozenset({7, 11, 2})
    assert parse_chord("Bb:maj7/3")[2] == 2  # D in the bass
    assert parse_chord("A:hdim7")[1] == frozenset({9, 0, 3, 7})
    assert parse_chord("D:sus4")[1] == frozenset({2, 7, 9})


def test_canonical_license_and_gate():
    cases = {
        "http://creativecommons.org/licenses/by/4.0/": ("CC-BY-4.0", True),
        "http://creativecommons.org/licenses/by/3.0/us/": ("CC-BY-3.0-US", True),
        "http://creativecommons.org/publicdomain/zero/1.0/": ("CC0-1.0", True),
        "http://creativecommons.org/publicdomain/mark/1.0/": ("public-domain", True),
        "http://creativecommons.org/licenses/publicdomain/": ("public-domain", True),
        "http://creativecommons.org/licenses/by-nc-sa/3.0/": ("CC-BY-NC-SA-3.0", False),
        "http://creativecommons.org/licenses/by-nd/4.0/": ("CC-BY-ND-4.0", False),
        "http://creativecommons.org/licenses/by-sa/3.0/": ("CC-BY-SA-3.0", False),
        "http://freemusicarchive.org/Free_Music_Philosophy": ("unknown", False),
        "": ("unknown", False),
    }
    for url, (canon, ok) in cases.items():
        assert canonical_license(url) == canon, (url, canonical_license(url))
        assert license_ok(canon)[0] == ok, canon
    assert license_ok("CC-BY-SA-3.0", allow_sharealike=True)[0]


def test_analysis_roundtrip():
    a = synthetic_song()
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "x.npz")
        a.save(p)
        b = Analysis.load(p)
    assert b.chord_labels == a.chord_labels and abs(b.duration - a.duration) < 1e-9
    assert np.allclose(b.other, a.other) and b.other_piano.shape == (0, 4)


def test_monophonize():
    ev = np.array([[0.0, 1.0, 60, 80], [0.01, 0.5, 64, 50], [0.5, 1.5, 62, 80]])
    out = monophonize(ev, prefer="high")
    assert len(out) == 2 and out[0, PITCH] == 64 and out[0, END] <= out[1, START] + 1e-9


def test_arrange_levels_are_valid_and_reductive():
    a = synthetic_song()
    song = Song(a)
    counts = {}
    for name, lv in LEVELS.items():
        ev, stats = arrange(a, name, song)
        notes = ev[ev[:, PITCH] > 0]
        assert len(notes) > 0, name
        assert np.allclose(sanitize(ev), ev), f"{name}: output not a valid semi-CRF target"
        assert ev[:, START].min() >= 0 and ev[:, END].max() <= a.duration + 1e-6
        # reduction: no note outside the chord of its moment (C# noise must be filtered)
        for s, e, p, v in notes:
            pcs = chord_pcs_at(a, s + 1e-3)
            assert pcs is None or int(p) % 12 in pcs, (name, s, p)
        # playability: simultaneous onsets fit two hands of max_per_hand
        on = np.round(notes[:, START], 2)
        _, n_at = np.unique(on, return_counts=True)
        assert n_at.max() <= 2 * lv.max_per_hand, (name, n_at.max())
        counts[name] = len(notes)
        assert stats["accomp_chord_fallback"] == 0 or name == "easy", (name, stats)
    assert counts["easy"] < counts["simple"] < counts["normal"] < counts["full"] <= counts["virtuoso"], counts


def test_guitar_riff_becomes_melody_without_vocals():
    a = synthetic_song(vocals=False)
    # replace the strums with a single-line riff (A minor pentatonic over everything)
    riff = []
    for b in range(16):
        for k, pc in enumerate([9, 0, 2, 4, 7, 4, 2, 0]):
            t = b * 2.0 + k * 0.25
            riff.append([t, t + 0.22, 57 + (pc - 57) % 12, 90])
    a.other = np.array(riff)
    a.chord_labels = ["N"] * len(a.chord_labels)
    ev, stats = arrange(a, "normal")
    assert stats["melody_from_other_bars"] >= 12, stats
    assert stats["melody_notes"] > 50, stats


def test_clean_pitches_drops_snaps_keeps():
    song = Song(synthetic_song())  # C major, bar 0 = C chord (0.0-2.0 s)
    ev = np.array([
        [0.10, 0.20, 66, 80],  # F#: off-key, not in C chord, short -> dropped
        [0.30, 0.80, 61, 80],  # C#: off-key, long, C is a semitone below -> snapped to 60
        [0.90, 1.00, 62, 80],  # D: in key (passing tone) -> kept as is
        [1.10, 1.60, 70, 80],  # Bb: off-key, long, no chord tone a semitone away -> kept
    ])
    out, suspect = song.clean_pitches(ev)
    assert suspect.tolist() == [True, True, False, True], suspect
    assert out[:, PITCH].tolist() == [60, 62, 70], out


def test_energy_drives_density():
    a = synthetic_song()
    ev, _ = arrange(a, "normal")
    notes = ev[ev[:, PITCH] > 0]
    half = a.duration / 2
    quiet, loud = (notes[:, START] < half).sum(), (notes[:, START] >= half).sum()
    assert loud > quiet, (quiet, loud)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)

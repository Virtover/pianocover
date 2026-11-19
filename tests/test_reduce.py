import numpy as np

from pianocover.reduce import Limits, best_split, reduce, rhythm, simplify


def chord(t, pitches, dur=0.5, vel=80):
    return [[t, t + dur, p, vel] for p in pitches]


def playable(ev, lim=Limits()):
    """Every onset has a valid two-hand split of the notes sounding just after it."""
    n = ev[ev[:, 2] > 0]
    for t in np.unique(n[:, 0]):
        s = np.sort(n[(n[:, 0] <= t + lim.onset_tol) & (n[:, 1] > t + lim.onset_tol), 2])
        if best_split(s, lim, 60) is None:
            return False
    return True


def test_playable_input_unchanged():
    ev = np.array(chord(0, [36, 43, 48, 64, 67, 72]) + chord(0.5, [41, 48, 65, 69, 72]) + [[1.0, 1.5, -64, 127]],
                  np.float64)
    out, rep = reduce(ev)
    assert rep.changed_onsets == 0
    assert np.array_equal(out, ev)


def test_dense_chord_becomes_playable_and_keeps_outer_notes():
    ev = np.array(chord(0, [36, 40, 43, 48, 52, 55, 60, 64, 67, 72, 76, 79, 84]), np.float64)
    out, rep = reduce(ev)
    assert rep.changed_onsets == 1 and rep.dropped > 0
    assert playable(out)
    assert out[:, 2].min() == 36 and out[:, 2].max() == 84


def test_held_notes_released_before_dropping():
    held = [[0.0, 4.0, p, 70] for p in (50, 53, 57, 62, 65)]          # a sustained string chord
    ev = np.array(held + chord(1.0, [36, 43, 72, 76, 79, 84]), np.float64)
    out, rep = reduce(ev)
    assert playable(out)
    assert rep.shortened > 0
    assert len(out[(out[:, 0] == 1.0)]) == 6       # the new chord is kept whole


def tremolo_run(p=67, n=16, ioi=0.0625):  # 16 strikes, 16 per second
    return [[k * ioi, k * ioi + ioi * 0.9, p, 80] for k in range(n)]


def test_slow_repetitions_untouched():
    ev = np.array([[k * 0.25, k * 0.25 + 0.2, 64, 80] for k in range(8)], np.float64)  # 4 per second
    for mode in ("hold", "tremolo", "half"):
        out, rep = reduce(ev, repeat=mode)
        assert np.array_equal(out, ev), mode


def test_hold_merges_fast_run():
    out, rep = reduce(np.array(tremolo_run(), np.float64), repeat="hold")
    assert len(out) == 2                      # 1 s run, re-struck at most every 0.5 s
    assert out[0, 0] == 0 and abs(out[-1, 1] - (15 * 0.0625 + 0.05625)) < 1e-6


def test_half_slows_down():
    out, _ = reduce(np.array(tremolo_run(), np.float64), repeat="half")
    assert np.min(np.diff(np.sort(out[:, 0]))) >= 0.11


def test_tremolo_alternates_octave():
    out, _ = reduce(np.array(tremolo_run(), np.float64), repeat="tremolo")
    assert sorted(set(out[:, 2])) == [55, 67] and len(out) == 16


def test_near_duplicate_onsets_merged():
    ev = np.array([[0.0, 0.5, 60, 80], [0.03, 0.6, 60, 90]], np.float64)
    out, rep = reduce(ev, repeat="hold")
    assert len(out) == 1 and out[0, 1] == 0.6 and out[0, 3] == 90


def test_ostinato_only_top_keeps_rhythm():
    ev = np.array(tremolo_run(72) + tremolo_run(64) + tremolo_run(60), np.float64)
    out, rep = reduce(ev, repeat="half")
    assert rep.repeat["ostinato_held"] == 2
    assert (out[:, 2] == 72).sum() > (out[:, 2] == 64).sum()


BEATS = np.arange(0, 8.01, 0.5)          # 120 bpm
DOWN = BEATS[::4]


def test_rhythm_leaves_moderate_hands_alone():
    ev = np.array([[t, t + 0.2, 72 + (k % 2) * 4, 80] for k, t in enumerate(np.arange(0, 4, 0.125))]  # 8/s
                  + [[t, t + 0.45, 40, 80] for t in np.arange(0, 4, 0.5)], np.float64)
    out, st = rhythm(ev, BEATS, DOWN)
    assert st["bars_changed"] == 0 and np.array_equal(out, ev)


def test_rhythm_grid_catches_fast_figure_and_is_even():
    fig = [72, 76, 79, 76]  # a 32nd-note figure on 3 keys: 16 strikes per second
    ev = np.array([[t, t + 0.06, fig[k % 4], 80] for k, t in enumerate(np.arange(0, 4, 0.0625))], np.float64)
    out, st = rhythm(ev, BEATS, DOWN, mode="grid")
    d = np.diff(np.sort(out[:, 0]))
    assert st["bars_changed"] == 2 and np.allclose(d, d[0])        # one even step
    assert len(out) / 4 <= 9


def test_rhythm_grid_no_gap_at_bar_ends_with_jittery_beats():
    rng = np.random.default_rng(1)
    beats = np.arange(0, 16.01, 0.5) + rng.uniform(-0.015, 0.015, 33)   # beat tracker jitter
    beats[0] = 0.0
    fig = [72, 76, 79, 76]
    t = np.arange(0, 16, 0.0625) + rng.uniform(-0.01, 0.01, 256)       # 16 strikes/s, played a bit loose
    ev = np.array([[x, x + 0.05, fig[k % 4], 80] for k, x in enumerate(t)], np.float64)
    out, st = rhythm(ev, beats, beats[::4], mode="grid")
    o = out[np.argsort(out[:, 0])]
    d = np.diff(o[:, 0])
    assert d.max() < 0.125 + 0.05 and d.min() > 0.125 - 0.05           # even 8ths, also across bar lines
    gaps = o[1:, 0] - o[:-1, 1]
    assert gaps.max() < 0.02                                            # every strike held to the next


def test_rhythm_jittery_beats_over_32nds_follow_the_chords():
    # the Carmen case: LH chords on a steady 8th pulse (0.217 s), RH 32nds (0.054 s), and a beat
    # tracker jittering +-40 ms around the 8ths; the kept RH strikes must follow the chords' pulse
    pulse = 0.217
    lh = [[k * pulse, (k + 1) * pulse, p, 80] for k in range(40) for p in (42, 45, 54)]
    rh = [[k * pulse / 4, k * pulse / 4 + 0.05, 71 + 2 * (k % 2), 80] for k in range(160)]
    rng = np.random.default_rng(3)
    beats = np.arange(41) * pulse + rng.uniform(-0.04, 0.04, 41) - 0.02
    out, st = rhythm(np.array(lh + rh, np.float64), beats, beats[::4], mode="grid")
    r = np.sort(np.unique(np.round(out[out[:, 2] > 60, 0], 3)))
    d = np.diff(r)
    assert st["bars_changed"] > 0
    assert d.max() - d.min() < 0.03, d                         # even, no 0.16/0.27 pairs
    assert np.min(np.abs(r[:, None] - np.arange(40)[None, :] * pulse), axis=1).max() < 0.02  # on the pulse


B2 = np.arange(0, 12.01, 0.5)    # 120 bpm, 6 s of beats for the simplify tests


def melody_line():
    return [[k * 0.5, k * 0.5 + 0.45, 76 + (k % 3), 90] for k in range(12)]


def test_simplify_keeps_scale_runs():
    run = [[1.0 + k * 0.03125, 1.0 + k * 0.03125 + 0.03, 48 + k, 80] for k in range(32)]  # 32 keys up, 1 s
    ev = np.array(run + [[0, 3, 40, 80]], np.float64)
    out, st = simplify(ev, B2)
    assert st["kept_gestures"] + st["kept_short"] >= 1
    assert sorted(out[:, 2].tolist()) == sorted(ev[:, 2].tolist())


def test_simplify_keeps_short_bursts():
    burst = [[1.0 + k * 0.05, 1.0 + k * 0.05 + 0.04, 52 + (k % 2) * 4, 80] for k in range(6)]  # 0.25 s
    ev = np.array(burst + melody_line(), np.float64)
    out, st = simplify(ev, B2)
    assert len(out) == len(ev)


def test_simplify_alberti_to_eighths_keeps_melody():
    alb = [48, 55, 52, 55]                       # C G E G in 16ths at 120 bpm = 8/s ... make it 32nds
    acc = [[k * 0.0625, k * 0.0625 + 0.06, alb[k % 4], 70] for k in range(96)]   # 16/s for 6 s
    ev = np.array(acc + melody_line(), np.float64)
    out, st = simplify(ev, B2)
    assert st["figure"] >= 1
    lh = out[out[:, 2] < 60]
    assert len(np.unique(np.round(lh[:, 0], 3))) / 6 <= 9.5      # at most ~9 strikes per second
    assert set(lh[:, 2].astype(int)) <= set(alb)
    mel = out[out[:, 2] >= 76]
    assert len(mel) == 12                                           # melody untouched


def test_simplify_trill_becomes_mordent():
    trill = [[2.0 + k * 0.06, 2.0 + k * 0.06 + 0.055, 79 + (k % 2) * 2, 90] for k in range(16)]   # ~17/s
    ev = np.array(trill + [[0, 4, 50, 70]], np.float64)
    out, st = simplify(ev, B2)
    assert st["trill"] == 1
    hi = out[out[:, 2] >= 79]
    assert len(hi) == 3 and hi[:, 1].max() >= 2.0 + 15 * 0.06


def test_simplify_repeated_chord_held_on_beats():
    trem = [[k * 0.0625, k * 0.0625 + 0.06, p, 70] for k in range(64) for p in (55, 59, 62)]  # 4 s chord tremolo
    ev = np.array(trem + melody_line(), np.float64)
    out, st = simplify(ev, B2, tremolo="hold")
    assert st["repeat"] >= 1
    acc = out[out[:, 2] < 70]
    assert len(np.unique(np.round(acc[:, 0], 3))) <= 9               # re-struck on the beats only


def test_rhythm_block_makes_chords():
    fig = [60, 64, 67, 72]
    ev = np.array([[t, t + 0.06, fig[k % 4], 80] for k, t in enumerate(np.arange(0, 4, 0.0625))], np.float64)
    out, st = rhythm(ev, BEATS, DOWN, mode="block")
    assert st["chords"] > 0
    starts = np.unique(out[:, 0])
    assert len(starts) / 4 <= 9 and all((out[:, 0] == s).sum() >= 2 for s in starts)


def test_idempotent():
    rng = np.random.default_rng(0)
    ev = np.array([[t, t + rng.uniform(0.2, 2), p, 80] for t in np.arange(0, 20, 0.25)
                   for p in rng.choice(np.arange(30, 95), 9, replace=False)], np.float64)
    once, _ = reduce(ev)
    twice, rep2 = reduce(once)
    assert playable(once)
    assert rep2.changed_onsets == 0 and np.allclose(once, twice)

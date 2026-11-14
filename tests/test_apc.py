import numpy as np

from pianocover.apc import FPS, NUM_FRAME
from pianocover.apc.labels import pedal_extend, style_vector, window_labels
from pianocover.apc.prepare import padded_spec


def test_pedal_extend_stops_at_next_onset():
    ev = np.array([
        [1.0, 1.5, 60, 80],   # pedal held 1.2..3.0 -> extends to 2.5 (next C4 onset)
        [2.5, 2.8, 60, 70],   # extends to 3.0
        [0.0, 0.5, 64, 60],   # released before the pedal: unchanged
        [1.2, 3.0, -64, 127],  # sustain pedal
    ], dtype=np.float64)
    n = pedal_extend(ev)
    assert len(n) == 3 and (n[:, 2] > 0).all()
    ends = {(round(s, 3), int(p)): round(e, 3) for s, e, p, _ in n}
    assert ends[(1.0, 60)] == 2.5
    assert ends[(2.5, 60)] == 3.0
    assert ends[(0.0, 64)] == 0.5


def test_window_labels_match_hft_shapes():
    notes = np.array([[64 / FPS, 128 / FPS, 60, 90]], dtype=np.float64)  # on the frame grid
    on, off, fr, vel = window_labels(notes, 0, NUM_FRAME)
    p = 60 - 21
    f_on, f_off = 64, 128
    assert on[f_on, p] == on[:, p].max() == 1.0
    assert on[f_on - 3, p] < on[f_on - 1, p] and on[f_on + 4, p] == 0
    assert off[f_off, p] > 0.9
    assert fr[f_on:f_off + 1, p].all() and not fr[f_off + 1, p] and not fr[f_on - 1, p]
    assert vel[f_on, p] == 90 and vel[f_on + 2, p] == 0  # onset >= 0.5 only within 1 frame
    # the same note seen from a window starting later
    on2, _, fr2, _ = window_labels(notes, 40, NUM_FRAME)
    assert np.allclose(on2[f_on - 40], on[f_on]) and fr2[f_on - 40:f_off - 39, p].all()


def test_style_vector_is_three_distributions():
    rng = np.random.default_rng(0)
    s = np.sort(rng.uniform(0, 100, 500))
    notes = np.stack([s, s + 0.3, rng.integers(40, 80, 500), rng.integers(40, 110, 500)], 1)
    sv = style_vector(notes, 100.0)
    assert sv.shape == (24,)
    assert np.allclose([sv[:8].sum(), sv[8:16].sum(), sv[16:].sum()], 1.0)


def test_padded_spec_layout():
    feat = np.zeros((1000, 256), np.float32)
    out = padded_spec(feat)
    assert out.shape[0] == 32 + 1024 + 32 and out.dtype == np.float16
    assert (out[:32] == -18).all() and (out[32:1032] == 0).all() and (out[1032:] == -18).all()

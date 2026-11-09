"""Pure-numpy note-event utilities.

Events are stored as float arrays of shape (N, 4): [start_sec, end_sec, pitch, velocity].
Pitch > 0 is a piano key (MIDI 21..108); pitch -64 / -67 are sustain / soft pedal
intervals, matching Transkun's convention.
"""

from __future__ import annotations

import numpy as np

from . import FRAME_SEC, PIANO_MAX, PIANO_MIN, PEDAL_PITCHES

START, END, PITCH, VEL = 0, 1, 2, 3


def empty() -> np.ndarray:
    return np.zeros((0, 4), dtype=np.float64)


def from_transkun_notes(notes) -> np.ndarray:
    """Convert a list of transkun.Data.Note to an (N, 4) array."""
    if not notes:
        return empty()
    return np.array([[n.start, n.end, n.pitch, n.velocity] for n in notes], dtype=np.float64)


def is_note(ev: np.ndarray) -> np.ndarray:
    return ev[:, PITCH] > 0


def transpose(ev: np.ndarray, semitones: int) -> np.ndarray:
    """Shift piano notes by `semitones`; pedals untouched; out-of-range notes dropped."""
    if semitones == 0 or len(ev) == 0:
        return ev.copy()
    out = ev.copy()
    notes = is_note(out)
    out[notes, PITCH] += semitones
    keep = ~notes | ((out[:, PITCH] >= PIANO_MIN) & (out[:, PITCH] <= PIANO_MAX))
    return out[keep]


def warp(ev: np.ndarray, src_times: np.ndarray, dst_times: np.ndarray) -> np.ndarray:
    """Map event times through a monotonic time map (piecewise linear).

    src_times / dst_times: strictly increasing anchor points (e.g. a DTW path in seconds).
    Events whose warped span falls outside [dst_times[0], dst_times[-1]] are clipped/dropped.
    """
    if len(ev) == 0:
        return ev.copy()
    out = ev.copy()
    out[:, START] = np.interp(ev[:, START], src_times, dst_times)
    out[:, END] = np.interp(ev[:, END], src_times, dst_times)
    # Notes entirely outside the mapped source range collapse to a point -> dropped by sanitize.
    outside = (ev[:, END] <= src_times[0]) | (ev[:, START] >= src_times[-1])
    return out[~outside]


def sanitize(ev: np.ndarray, frame: float = FRAME_SEC, min_frames: float = 1.0,
             supported_pitches=None) -> np.ndarray:
    """Make events valid for Transkun's semi-CRF targets.

    Per pitch: sort by onset, enforce a minimum length of `min_frames` frames, trim a note's
    end so it never overlaps the next onset of the same pitch, and drop notes that would
    quantize into the same frame as the previous one (Transkun would otherwise merge them
    with a warning). Also drops unsupported pitches.
    """
    if len(ev) == 0:
        return empty()
    if supported_pitches is None:
        supported_pitches = set(range(PIANO_MIN, PIANO_MAX + 1)) | set(PEDAL_PITCHES)
    min_len = frame * min_frames
    kept = []
    for p in np.unique(ev[:, PITCH]):
        if int(p) not in supported_pitches:
            continue
        group = ev[ev[:, PITCH] == p]
        group = group[np.argsort(group[:, START], kind="stable")]
        prev = None
        for row in group:
            s, e = max(row[START], 0.0), row[END]
            if e - s < min_len:
                e = s + min_len
            if prev is not None:
                if round(s / frame) <= round(prev[START] / frame):
                    # same onset frame as the previous note of this pitch: keep the louder one
                    if row[VEL] > prev[VEL]:
                        prev[VEL] = row[VEL]
                    prev[END] = max(prev[END], e)
                    continue
                if prev[END] > s:
                    # trim so the previous note ends exactly at this onset (zero-length
                    # quantized intervals are valid for Transkun's semi-CRF)
                    prev[END] = s
                kept.append(prev)
            prev = np.array([s, e, p, row[VEL]], dtype=np.float64)
        if prev is not None:
            kept.append(prev)
    if not kept:
        return empty()
    out = np.stack(kept)
    out = out[out[:, END] > out[:, START]]
    out[:, VEL] = np.clip(np.round(out[:, VEL]), 1, 127)
    return out[np.lexsort((out[:, PITCH], out[:, START]))]


def slice_events(ev: np.ndarray, begin: float, end: float):
    """Events overlapping [begin, end), shifted to start at 0 and clipped to the window.

    Returns (events, has_onset, has_offset) where the flags tell Transkun whether the true
    onset / offset lies inside the window (same semantics as transkun DatasetMaestro with
    notesStrictlyContained=False).
    """
    if len(ev) == 0:
        return empty(), np.zeros(0, bool), np.zeros(0, bool)
    m = (ev[:, END] > begin) & (ev[:, START] < end)
    sub = ev[m].copy()
    has_onset = sub[:, START] >= begin
    has_offset = sub[:, END] < end
    sub[:, START] = np.maximum(sub[:, START], begin) - begin
    sub[:, END] = np.minimum(sub[:, END], end) - begin
    valid = sub[:, END] > sub[:, START]
    return sub[valid], has_onset[valid], has_offset[valid]


def shift_events(ev, has_onset, has_offset, k: int, chunk_sec: float):
    """Apply the varispeed transform of `random_key_shift` to a chunk's targets."""
    if k == 0:
        return ev, has_onset, has_offset
    r = 2.0 ** (k / 12.0)
    limit = min(chunk_sec, chunk_sec * r)  # input-time span that is still inside the output window
    keep = ev[:, START] < limit
    ev, on, off = ev[keep].copy(), has_onset[keep].copy(), has_offset[keep].copy()
    cut = ev[:, END] > limit
    ev[cut, END] = limit
    off[cut] = False
    ev[:, [START, END]] /= r
    notes = ev[:, PITCH] > 0
    ev[notes, PITCH] += k
    keep = ~notes | ((ev[:, PITCH] >= PIANO_MIN) & (ev[:, PITCH] <= PIANO_MAX))
    keep &= ev[:, END] > ev[:, START]
    return ev[keep], on[keep], off[keep]


def to_transkun_notes(ev: np.ndarray, has_onset=None, has_offset=None):
    """Convert arrays back to transkun.Data.Note objects (imports transkun lazily)."""
    from transkun.Data import Note

    if has_onset is None:
        has_onset = np.ones(len(ev), bool)
    if has_offset is None:
        has_offset = np.ones(len(ev), bool)
    return [Note(float(r[START]), float(r[END]), int(r[PITCH]), int(r[VEL]), bool(a), bool(b))
            for r, a, b in zip(ev, has_onset, has_offset)]


def to_pretty_midi(ev: np.ndarray, resolution: int = 960):
    """Write events (notes + pedal CCs) into a single-piano PrettyMIDI object."""
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(resolution=resolution)
    piano = pretty_midi.Instrument(program=0, name="Piano")
    for s, e, p, v in ev:
        if p > 0:
            piano.notes.append(pretty_midi.Note(velocity=int(v), pitch=int(p), start=float(s), end=float(e)))
        else:
            cc = int(-p)
            piano.control_changes.append(pretty_midi.ControlChange(cc, 127, float(s)))
            piano.control_changes.append(pretty_midi.ControlChange(cc, 0, float(e)))
    piano.notes.sort(key=lambda n: (n.start, n.pitch))
    piano.control_changes.sort(key=lambda c: c.time)
    pm.instruments.append(piano)
    return pm


def save_npy(path, ev: np.ndarray) -> None:
    np.save(path, ev.astype(np.float64))


def load_npy(path) -> np.ndarray:
    ev = np.load(path)
    return ev.reshape(-1, 4).astype(np.float64)

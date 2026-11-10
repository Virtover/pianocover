"""Reduction arranger: transcribed song parts -> a playable piano part. Numpy only.

The principle is *reduction*, not regeneration: every note in the output comes from a
note that was heard in the song (vocal, bass, guitar/keys/synth in 'other', or piano),
moved by octaves at most. Chords (BTC) are used only to *correct and filter*: to drop
transcription errors that are not chord tones and to fix semitone slips. They supply notes
only when nothing was transcribed at a chord change (logged in stats as 'chord_fallback').

What goes where:
  * RH melody   = the vocal line; where there is no vocal for a bar, the lead line of
                  'other' (guitar riff / solo / synth hook) when that part is mostly
                  single-line (riffs and solos are melody; strummed chords are not).
  * LH bass     = the real bass line (bass stem), simplified per level.
  * Middle      = accompaniment hits placed on the song's real onsets (guitar strums,
                  keys, snare / kick, bass), voiced from the real transcribed notes of
                  'other' filtered to chord tones and to one hand span.
  * Piano parts = (opt-in, keep_piano=True) bars where 'other' looks like a piano
                  (Transkun and basic-pitch agree on many notes over a wide range) are
                  taken nearly verbatim. OFF by default: on the first 10 prototype songs
                  this heuristic flagged guitar-heavy songs as piano (e.g. 80/87 bars of a
                  country song), replacing the guitar reduction with a raw transcription.
                  Needs a real piano-presence classifier before it is turned on.
  * Density     = follows each bar's energy (louder sections -> more hits).
  * Pedal       = re-pedalled every chord change / 2 beats, skipped where it would make
                  notes rub (semitone clashes).
  * Chord check = a detected chord that does not match what is played in its span is
                  replaced by the most-played pitch classes there (Song._verify_chords).
  * Difficulty  = LEVELS controls how much of all this is kept.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..events import END, PITCH, START, VEL, empty, sanitize
from ..postprocess import drop_ghost_notes, limit_hands
from .analysis import FRAME_HOP, Analysis
from .chords import parse_chord


@dataclass(frozen=True)
class Level:
    name: str
    melody_min_dur: float       # melody notes shorter than this are dropped (after merging)
    melody_octaves: bool        # double the melody an octave below in loud sections
    bass_mode: str              # 'chord_root' | 'on_beats' | 'line' | 'line_octave'
    accomp_subdiv: int          # accompaniment grid: subdivisions per beat
    accomp_notes: int           # max notes per accompaniment hit
    accomp_threshold: float     # onset-strength threshold for a grid slot (scaled by energy)
    keep_piano: bool            # keep piano-like parts of 'other' nearly verbatim
    pedal: bool
    max_per_hand: int
    span: int = 12


LEVELS = {
    "easy":     Level("easy",     0.20, False, "chord_root",  1, 1, 9.0, False, False, 2),
    "simple":   Level("simple",   0.14, False, "on_beats",    1, 2, 0.9, False, False, 3),
    "normal":   Level("normal",   0.10, False, "line",        2, 3, 0.6, False, True, 4),
    "full":     Level("full",     0.07, False, "line_octave", 4, 3, 0.45, False, True, 5),
    "virtuoso": Level("virtuoso", 0.05, True,  "line_octave", 4, 4, 0.35, False, True, 5),
}

# Krumhansl-Kessler key profiles
_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
_MAJOR_SCALE = (0, 2, 4, 5, 7, 9, 11)
_MINOR_SCALE = (0, 2, 3, 5, 7, 8, 10, 11)  # natural + harmonic minor (raised 7th)


# --------------------------------------------------------------------------- note helpers
def _notes(ev: np.ndarray) -> np.ndarray:
    ev = np.asarray(ev, dtype=np.float64).reshape(-1, 4)
    return ev[ev[:, PITCH] > 0]


def merge_same_pitch(ev: np.ndarray, gap: float = 0.05) -> np.ndarray:
    """Join consecutive notes of the same pitch separated by less than `gap` seconds."""
    if len(ev) < 2:
        return ev.copy()
    ev = ev[np.lexsort((ev[:, START], ev[:, PITCH]))]
    out = [ev[0].copy()]
    for row in ev[1:]:
        prev = out[-1]
        if row[PITCH] == prev[PITCH] and row[START] - prev[END] < gap:
            prev[END] = max(prev[END], row[END])
            prev[VEL] = max(prev[VEL], row[VEL])
        else:
            out.append(row.copy())
    out = np.stack(out)
    return out[np.argsort(out[:, START], kind="stable")]


def monophonize(ev: np.ndarray, prefer: str = "salience", tol: float = 0.04) -> np.ndarray:
    """One note at a time. Near-simultaneous onsets keep one note (by salience, or the
    highest / lowest pitch); an overlapping earlier note is cut at the next onset."""
    if len(ev) == 0:
        return ev.copy()
    ev = ev[np.argsort(ev[:, START], kind="stable")]
    out = []
    i = 0
    while i < len(ev):
        j = i + 1
        while j < len(ev) and ev[j, START] - ev[i, START] <= tol:
            j += 1
        group = ev[i:j]
        if prefer == "high":
            k = int(np.argmax(group[:, PITCH]))
        elif prefer == "low":
            k = int(np.argmin(group[:, PITCH]))
        else:
            k = int(np.argmax(group[:, VEL] * np.minimum(group[:, END] - group[:, START], 1.0)))
        row = group[k].copy()
        if out and out[-1][END] > row[START]:
            out[-1][END] = row[START]
        out.append(row)
        i = j
    out = np.stack(out)
    return out[out[:, END] - out[:, START] > 1e-3]


def drop_short(ev: np.ndarray, min_dur: float) -> np.ndarray:
    return ev[(ev[:, END] - ev[:, START]) >= min_dur]


def fold_phrases(ev: np.ndarray, lo: float, hi: float, gap: float = 1.0,
                 clamp=(21, 108)) -> np.ndarray:
    """Shift each phrase (notes separated by < gap s) by whole octaves so its median pitch
    lands in [lo, hi]; keeps the phrase's contour. Single notes outside `clamp` are folded."""
    if len(ev) == 0:
        return ev.copy()
    ev = ev[np.argsort(ev[:, START], kind="stable")].copy()
    breaks = np.where(ev[1:, START] - np.maximum.accumulate(ev[:-1, END]) > gap)[0] + 1
    for idx in np.split(np.arange(len(ev)), breaks):
        med = np.median(ev[idx, PITCH])
        shift = 0
        while med + shift < lo:
            shift += 12
        while med + shift > hi:
            shift -= 12
        ev[idx, PITCH] += shift
    while (ev[:, PITCH] < clamp[0]).any():
        ev[ev[:, PITCH] < clamp[0], PITCH] += 12
    while (ev[:, PITCH] > clamp[1]).any():
        ev[ev[:, PITCH] > clamp[1], PITCH] -= 12
    return ev


def drop_ornaments(ev: np.ndarray, max_dur: float = 0.12, max_step: int = 2) -> np.ndarray:
    """Remove scoops / slides / vibrato fragments of a monophonic line: a very short note
    that leads straight (<60 ms) into a longer note at most `max_step` semitones away."""
    if len(ev) < 2:
        return ev.copy()
    ev = ev[np.argsort(ev[:, START], kind="stable")]
    keep = np.ones(len(ev), bool)
    for i in range(len(ev) - 1):
        d = ev[i, END] - ev[i, START]
        nxt = ev[i + 1]
        if d < max_dur and nxt[START] - ev[i, END] < 0.06 and abs(nxt[PITCH] - ev[i, PITCH]) <= max_step \
                and nxt[END] - nxt[START] > d:
            keep[i] = False
            ev[i + 1, START] = min(ev[i + 1, START], ev[i, START])  # the note starts where the scoop did
    return ev[keep]


def smooth_octaves(ev: np.ndarray, max_leap: int = 9, lo: int = 55, hi: int = 96, gap: float = 1.0) -> np.ndarray:
    """Inside a phrase, a leap larger than `max_leap` is almost always an octave error
    (harmonic picked by the transcriber, or a vocal/riff splice): move the note by octaves
    to the closest position to the previous note."""
    if len(ev) < 2:
        return ev.copy()
    ev = ev[np.argsort(ev[:, START], kind="stable")].copy()
    for i in range(1, len(ev)):
        if ev[i, START] - ev[i - 1, END] > gap:
            continue  # new phrase: keep its own register
        prev, p = ev[i - 1, PITCH], ev[i, PITCH]
        if abs(p - prev) <= max_leap:
            continue
        cands = [p + 12 * k for k in range(-3, 4) if lo <= p + 12 * k <= hi]
        if cands:
            ev[i, PITCH] = min(cands, key=lambda c: abs(c - prev))
    return ev


def snap_to_grid(ev: np.ndarray, grid: np.ndarray, tol: float) -> np.ndarray:
    """Snap onsets to the nearest grid point when within `tol` seconds."""
    if len(ev) == 0 or len(grid) < 2:
        return ev.copy()
    ev = ev.copy()
    k = np.clip(np.searchsorted(grid, ev[:, START]), 1, len(grid) - 1)
    nearest = np.where(ev[:, START] - grid[k - 1] <= grid[k] - ev[:, START], grid[k - 1], grid[k])
    ok = np.abs(nearest - ev[:, START]) <= tol
    ev[ok, START] = nearest[ok]
    return ev


def legato(ev: np.ndarray, max_gap: float, max_len: float, barriers=()) -> np.ndarray:
    """Extend each note of a monophonic line to the next onset when the gap is small, so the
    line sings instead of being chopped. Never extends across a barrier time (e.g. a chord
    change) the note did not already reach."""
    if len(ev) < 2:
        return ev.copy()
    ev = ev[np.argsort(ev[:, START], kind="stable")].copy()
    barriers = np.asarray(sorted(barriers), np.float64)
    for i in range(len(ev) - 1):
        nxt = ev[i + 1, START]
        if 0 <= nxt - ev[i, END] < max_gap:
            new_end = min(nxt, ev[i, START] + max_len)
            if len(barriers):
                j = np.searchsorted(barriers, ev[i, END] + 1e-6)
                if j < len(barriers) and barriers[j] < new_end:
                    new_end = max(ev[i, END], barriers[j])
            ev[i, END] = max(ev[i, END], new_end)
        ev[i, END] = min(ev[i, END], nxt)  # never overlap the next note of the line
    return ev


def place_in_range(pitch: float, lo: float, hi: float) -> float:
    while pitch < lo:
        pitch += 12
    while pitch > hi:
        pitch -= 12
    return pitch


# --------------------------------------------------------------------------- song structure
class Song:
    """Pre-computed structure shared by all levels: bars, grid, chords, key, energy."""

    def __init__(self, a: Analysis):
        self.a = a
        self.duration = a.duration
        beats = np.asarray(a.beats, np.float64)
        if len(beats) < 4:  # no usable beat track: assume 120 bpm
            beats = np.arange(0.0, a.duration + 0.5, 0.5)
        self.period = float(np.median(np.diff(beats)))
        # extend the beat grid to cover the whole song
        pre = np.arange(beats[0] - self.period, -self.period, -self.period)[::-1]
        post = np.arange(beats[-1] + self.period, a.duration + self.period, self.period)
        self.beats = np.concatenate([pre[pre > -self.period], beats, post])
        down = np.asarray(a.downbeats, np.float64)
        if len(down) < 2:
            down = self.beats[::4]
        bar = float(np.median(np.diff(down)))
        pre = np.arange(down[0] - bar, -bar, -bar)[::-1]
        post = np.arange(down[-1] + bar, a.duration + bar, bar)
        self.bars = np.concatenate([pre[pre > -bar], down, post])
        self.bar_len = bar

        # chords: merge consecutive identical labels
        self.chords = []  # (start, end, root, pcs, bass_pc) ; root None for 'N'
        for (s, e), lab in zip(np.asarray(a.chords).reshape(-1, 2), a.chord_labels):
            p = parse_chord(lab)
            key = None if p is None else (p[0], p[1], p[2])
            if self.chords and self.chords[-1][2:] == (key or (None, None, None)) and s - self.chords[-1][1] < 0.2:
                self.chords[-1] = (self.chords[-1][0], e) + self.chords[-1][2:]
            else:
                self.chords.append((s, e) + (key or (None, None, None)))
        self.chords_replaced = self._verify_chords()
        self._chord_starts = np.array([c[0] for c in self.chords]) if self.chords else np.zeros(0)

        # energy per bar, 0..1 within the song
        rms = np.asarray(a.mix_rms, np.float64)
        db = 20 * np.log10(rms + 1e-5)
        lo, hi = np.percentile(db, 10), np.percentile(db, 95)
        e = np.clip((db - lo) / max(hi - lo, 1e-3), 0, 1)
        self.bar_energy = np.array([self._mean_over(e, s, s + self.bar_len) for s in self.bars])
        if len(self.bar_energy) >= 3:  # light smoothing over neighbouring bars
            k = np.array([0.25, 0.5, 0.25])
            padded = np.pad(self.bar_energy, 1, mode="edge")
            self.bar_energy = np.convolve(padded, k, mode="valid")

        self.key_pcs = self._estimate_key()

    def _verify_chords(self, min_agree: float = 0.55, min_weight: float = 0.5) -> int:
        """Check each detected chord against what the song actually plays in that span
        (other + piano + bass notes, weighted by sounding time). If the chord explains less
        than `min_agree` of it, the label is wrong (seen on 2 of 10 prototype songs, where
        filtering to it produced notes the song never plays): replace its pitch classes with
        the most-played ones (at most 4, covering 75% of the time). Returns the count."""
        a = self.a
        src = np.concatenate([_notes(a.other), _notes(a.other_piano), _notes(a.bass)])
        bass_src = _notes(a.bass)
        if len(src) == 0:
            return 0
        replaced = 0
        for i, (s, e, root, pcs, bass_pc) in enumerate(self.chords):
            if root is None:
                continue
            ov = np.clip(np.minimum(src[:, END], e) - np.maximum(src[:, START], s), 0, None)
            if ov.sum() < min_weight:
                continue
            w = np.zeros(12)
            np.add.at(w, src[:, PITCH].astype(int) % 12, ov)
            if w[list(pcs)].sum() / w.sum() >= min_agree:
                continue
            order = np.argsort(-w)
            new, acc = [], 0.0
            for pc in order[:4]:
                if w[pc] <= 0 or acc >= 0.75 * w.sum():
                    break
                new.append(int(pc))
                acc += w[pc]
            bov = np.clip(np.minimum(bass_src[:, END], e) - np.maximum(bass_src[:, START], s), 0, None) \
                if len(bass_src) else np.zeros(0)
            if bov.sum() > 0:
                bw = np.zeros(12)
                np.add.at(bw, bass_src[:, PITCH].astype(int) % 12, bov)
                bass_pc = int(np.argmax(bw))
            else:
                bass_pc = new[0]
            self.chords[i] = (s, e, new[0], frozenset(new), bass_pc)
            replaced += 1
        return replaced

    @staticmethod
    def _mean_over(curve, s, e):
        a, b = int(max(s, 0) / FRAME_HOP), int(max(e, 0) / FRAME_HOP) + 1
        seg = curve[a:b]
        return float(seg.mean()) if len(seg) else 0.0

    def _estimate_key(self):
        hist = np.zeros(12)
        for ev in (self.a.bass, self.a.other, self.a.vocal):
            n = _notes(ev)
            np.add.at(hist, n[:, PITCH].astype(int) % 12, np.minimum(n[:, END] - n[:, START], 2.0))
        if hist.sum() == 0:
            return frozenset(range(12))
        best, best_pcs = -np.inf, None
        for tonic in range(12):
            for prof, scale in ((_MAJOR, _MAJOR_SCALE), (_MINOR, _MINOR_SCALE)):
                r = np.corrcoef(hist, np.roll(prof, tonic))[0, 1]
                if r > best:
                    best, best_pcs = r, frozenset((tonic + d) % 12 for d in scale)
        return best_pcs

    def chord_at(self, t: float):
        if not len(self._chord_starts):
            return None
        i = int(np.searchsorted(self._chord_starts, t, side="right")) - 1
        if i < 0:
            return None
        c = self.chords[i]
        return c if c[0] <= t < c[1] and c[2] is not None else None

    def bar_index(self, t: float) -> int:
        return int(np.clip(np.searchsorted(self.bars, t, side="right") - 1, 0, len(self.bars) - 1))

    def energy_at(self, t: float) -> float:
        return float(self.bar_energy[self.bar_index(t)])

    def grid(self, subdiv: int) -> np.ndarray:
        b = self.beats
        g = [np.linspace(x, y, subdiv, endpoint=False) for x, y in zip(b[:-1], b[1:])]
        g.append(b[-1:])
        g = np.concatenate(g)
        return g[(g >= 0) & (g < self.duration)]

    def clean_pitches(self, ev: np.ndarray, short: float = 0.25, snap_min: float = 0.15):
        """Remove / fix transcription errors without inventing notes.

        Only notes that are BOTH outside the key and outside the current chord are suspects
        (in-key passing tones and chromatic chord tones are kept). A suspect is:
          * moved one semitone to a chord tone if that neighbour exists and it lasts >= snap_min;
          * otherwise dropped if shorter than `short` (scoops, slides, vibrato, bleed);
          * otherwise kept (long deliberate notes, e.g. blue notes).
        Returns (cleaned events, suspect mask over the INPUT rows)."""
        ev = ev.copy()
        keep = np.ones(len(ev), bool)
        suspect = np.zeros(len(ev), bool)
        for i in range(len(ev)):
            pc = int(ev[i, PITCH]) % 12
            if pc in self.key_pcs:
                continue
            c = self.chord_at(0.5 * (ev[i, START] + ev[i, END]))
            if c is not None and pc in c[3]:
                continue
            suspect[i] = True
            dur = ev[i, END] - ev[i, START]
            if c is not None and dur >= snap_min:
                d = next((d for d in (-1, 1) if (pc + d) % 12 in c[3]), 0)
                if d:
                    ev[i, PITCH] += d
                    continue
            if dur < short:
                keep[i] = False
        return ev[keep], suspect


# --------------------------------------------------------------------------- parts
def bar_coverage(ev: np.ndarray, start: float, end: float) -> float:
    if len(ev) == 0:
        return 0.0
    s = np.clip(ev[:, START], start, end)
    e = np.clip(ev[:, END], start, end)
    return float(np.clip(e - s, 0, None).sum() / max(end - start, 1e-6))


def mean_polyphony(ev: np.ndarray, start: float, end: float, hop: float = 0.05) -> float:
    if len(ev) == 0:
        return 0.0
    t = np.arange(start, end, hop)
    counts = ((ev[:, START][None] <= t[:, None]) & (ev[:, END][None] > t[:, None])).sum(1)
    active = counts > 0
    return float(counts[active].mean()) if active.any() else 0.0


def piano_bars(song: Song, min_notes: int = 6, min_agree: float = 0.4, min_range: int = 15) -> np.ndarray:
    """Bars where 'other' looks like a piano part (HEURISTIC; guitar can pass too)."""
    tk, bp = _notes(song.a.other_piano), _notes(song.a.other)
    out = np.zeros(len(song.bars), bool)
    for i, s in enumerate(song.bars):
        e = s + song.bar_len
        n = tk[(tk[:, START] >= s) & (tk[:, START] < e)]
        if len(n) < min_notes or np.ptp(n[:, PITCH]) < min_range:
            continue
        m = bp[(bp[:, START] >= s - 0.06) & (bp[:, START] < e + 0.06)]
        if len(m) == 0:
            continue
        agree = np.mean([np.any((m[:, PITCH] == r[PITCH]) & (np.abs(m[:, START] - r[START]) < 0.06)) for r in n])
        out[i] = agree >= min_agree
    return out


def melody_part(song: Song, lv: Level, stats: dict) -> np.ndarray:
    a = song.a
    # CREPE f0 notes (monophonic tracker) when available: far more faithful than basic-pitch
    # on vocals (basic-pitch agreed with CREPE on only 41-63% of sung frames)
    from_f0 = len(getattr(a, "vocal_f0", ())) > 0
    src = _notes(a.vocal_f0) if from_f0 else _notes(a.vocal)
    vocal = drop_short(src if from_f0 else merge_same_pitch(monophonize(src)), 0.06)
    stats["melody_source"] = "crepe" if from_f0 else "basic-pitch"
    cleaned, suspect = song.clean_pitches(vocal)
    # bars where most of the "melody" fails the check are not a sung melody (rap, speech,
    # bleed into the vocal stem): drop the vocal there, the lead of 'other' may take over
    bad_bars = 0
    for s in song.bars:
        in_bar = (vocal[:, START] >= s) & (vocal[:, START] < s + song.bar_len) if len(vocal) else np.zeros(0, bool)
        if in_bar.sum() >= 4 and suspect[in_bar].mean() > 0.35:
            cleaned = cleaned[(cleaned[:, START] < s) | (cleaned[:, START] >= s + song.bar_len)]
            bad_bars += 1
    stats["vocal_bars_rejected"] = bad_bars
    stats["vocal_notes_removed"] = int(len(vocal) - len(cleaned))
    vocal = cleaned
    parts = [vocal] if len(vocal) else []
    other = _notes(a.other)
    filled = 0
    for s in song.bars:
        e = s + song.bar_len
        if bar_coverage(vocal, s, e) >= 0.12:
            continue
        seg = other[(other[:, START] >= s) & (other[:, START] < e)]
        # a lead line (riff / solo / hook) is mostly single-line; strummed chords are not
        if len(seg) < 3 or mean_polyphony(seg, s, e) > 2.2:
            continue
        lead, _ = song.clean_pitches(monophonize(seg, prefer="high"), short=0.15)
        if len(lead):
            parts.append(lead)
            filled += 1
    stats["melody_from_other_bars"] = filled
    stats["vocal_bars"] = int(sum(bar_coverage(vocal, s, s + song.bar_len) >= 0.12 for s in song.bars))
    if not parts:
        return empty()
    mel = monophonize(np.concatenate(parts))
    # shape the transcribed line into a playable, flowing melody (same notes, cleaned up):
    # basic-pitch splits one sung note into several (vibrato); CREPE notes are already whole
    # and a same-pitch repeat there is a real re-articulation, so only merge tiny gaps
    merge_gap = 0.03 if from_f0 else 0.15
    mel = merge_same_pitch(mel, gap=merge_gap)
    mel = drop_ornaments(mel)                       # scoops / slides into a note
    mel = fold_phrases(mel, 64, 79, clamp=(55, 96))
    mel = smooth_octaves(mel)                       # octave errors -> no wild leaps
    mel = merge_same_pitch(mel, gap=merge_gap)
    mel = snap_to_grid(mel, song.grid(4), tol=0.4 * song.period / 4)
    mel = monophonize(mel)                          # snapping may stack two onsets
    changes = [c[0] for c in song.chords if c[2] is not None]
    mel = legato(mel, max_gap=min(0.35, song.period), max_len=4 * song.period, barriers=changes)
    mel = drop_short(mel, lv.melody_min_dur)
    if lv.melody_octaves:
        loud = np.array([song.energy_at(t) > 0.7 for t in mel[:, START]], bool)
        dbl = mel[loud & (mel[:, PITCH] - 12 >= 55)].copy()
        dbl[:, PITCH] -= 12
        mel = np.concatenate([mel, dbl])
    return mel


def follow_kick(song: Song, bass: np.ndarray, on_beats_only: bool = False, min_strength: float = 0.3,
                stats: dict | None = None) -> np.ndarray:
    """The left hand follows the kick drum, as pop pianists do: where a kick hits while a
    bass note is held (and the bass player does not re-attack), re-strike that bass note.
    A sustained bass-guitar note sounds alive; a held piano note just decays."""
    d = np.asarray(song.a.drum_onsets).reshape(-1, 3)
    kicks = d[(d[:, 1] < 0.5) & (d[:, 2] >= min_strength), 0] if len(d) else np.zeros(0)
    if on_beats_only and len(kicks):
        near = np.min(np.abs(kicks[:, None] - song.beats[None, :]), axis=1) < 0.06
        kicks = kicks[near]
    if len(bass) == 0 or len(kicks) == 0:
        return bass
    grid = song.grid(4)
    k = np.clip(np.searchsorted(grid, kicks), 1, len(grid) - 1)
    kicks = np.where(kicks - grid[k - 1] <= grid[k] - kicks, grid[k - 1], grid[k])  # on the grid
    out = [row.copy() for row in bass[np.argsort(bass[:, START], kind="stable")]]
    added = 0
    for t in np.unique(np.round(kicks, 4)):
        if any(abs(r[START] - t) < 0.06 for r in out):
            continue
        for r in out:
            if r[START] + 0.08 < t < r[END] - 0.08:
                new = r.copy()
                new[START] = t
                r[END] = t
                out.append(new)
                added += 1
                break
    if stats is not None:
        stats["bass_kick_restrikes"] = added
    out = np.array(out)
    return out[np.argsort(out[:, START], kind="stable")]


def bass_part(song: Song, lv: Level, stats: dict) -> np.ndarray:
    a = song.a
    bass = drop_short(merge_same_pitch(monophonize(_notes(a.bass), prefer="low")), 0.08)
    n_raw = len(bass)
    bass, _ = song.clean_pitches(bass, short=0.2, snap_min=0.2)
    stats["bass_notes_removed"] = n_raw - len(bass)
    bass = fold_phrases(bass, 36, 47, clamp=(28, 55))
    fallback = 0

    if lv.bass_mode == "chord_root" or len(bass) == 0:
        out = []
        for s, e, root, pcs, bass_pc in song.chords:
            if root is None:
                continue
            # split long chords at bar lines so the bass is re-struck every bar
            cuts = [s] + [b for b in song.bars if s + 0.1 < b < e - 0.1] + [e]
            for cs, ce in zip(cuts[:-1], cuts[1:]):
                seg = bass[(bass[:, END] > cs) & (bass[:, START] < ce)]
                if len(seg):  # the pitch the bass actually plays most in this span
                    dur = np.minimum(seg[:, END], ce) - np.maximum(seg[:, START], cs)
                    pitch = seg[int(np.argmax(dur)), PITCH]
                else:
                    pitch = place_in_range(36 + (bass_pc - 36) % 12, 36, 47)
                    fallback += 1
                out.append([cs, ce, pitch, 0])
        stats["bass_chord_fallback"] = fallback
        return np.array(out, np.float64).reshape(-1, 4)

    if lv.bass_mode == "on_beats":
        g, tol = song.beats, 0.2 * song.period
    else:
        g, tol = song.grid(2), 0.2 * song.period / 2
    k = np.clip(np.searchsorted(g, bass[:, START]), 1, len(g) - 1)
    nearest = np.where(bass[:, START] - g[k - 1] <= g[k] - bass[:, START], g[k - 1], g[k])
    on_grid = np.abs(nearest - bass[:, START]) <= tol
    if lv.bass_mode == "on_beats":
        bass = bass[on_grid].copy()
        bass[:, START] = nearest[on_grid]
        nxt = np.append(bass[1:, START], song.duration)
        bass[:, END] = np.minimum(nxt, bass[:, START] + 2 * song.period)
    else:
        bass = bass.copy()
        bass[on_grid, START] = nearest[on_grid]
        bass = monophonize(bass, prefer="low")
        changes = [c[0] for c in song.chords if c[2] is not None]
        bass = legato(bass, max_gap=song.period, max_len=2 * song.period, barriers=changes)
    if lv.bass_mode in ("on_beats", "line", "line_octave"):
        bass = follow_kick(song, bass, on_beats_only=lv.bass_mode == "on_beats", stats=stats)
    if lv.bass_mode == "line_octave":
        bar_starts = np.array([abs(song.bars[song.bar_index(t)] - t) < 0.08 for t in bass[:, START]], bool)
        octv = bass[bar_starts].copy()
        octv[:, PITCH] = np.where(octv[:, PITCH] - 12 >= 28, octv[:, PITCH] - 12, octv[:, PITCH] + 12)
        bass = np.concatenate([bass, octv])
    return bass


def onset_strength(song: Song, grid: np.ndarray) -> np.ndarray:
    """How strongly the song itself plays on each grid slot: guitar strums / keys
    (clustered onsets of 'other'), piano, snare, kick and bass onsets."""
    a = song.a
    step = float(np.median(np.diff(grid))) if len(grid) > 1 else song.period
    tol = 0.35 * step
    strength = np.zeros(len(grid))

    def add(times, weights):
        if len(times) == 0 or len(grid) == 0:
            return
        k = np.clip(np.searchsorted(grid, times), 1, len(grid) - 1)
        left = times - grid[k - 1] <= grid[k] - times
        idx = np.where(left, k - 1, k)
        ok = np.abs(grid[idx] - times) <= tol
        np.add.at(strength, idx[ok], np.asarray(weights)[ok])

    for ev, w in ((a.other, 1.0), (a.other_piano, 0.7)):
        n = _notes(ev)
        if len(n) == 0:
            continue
        on = np.sort(n[:, START])
        # a strum is several onsets within ~60 ms -> one hit, stronger with more strings
        clusters = np.split(on, np.where(np.diff(on) > 0.06)[0] + 1)
        add(np.array([c[0] for c in clusters]), np.array([w * min(len(c), 3) / 3 for c in clusters]))
    d = np.asarray(a.drum_onsets).reshape(-1, 3)
    if len(d):
        # snare backbeat drives the chord hits most, kick a little, hi-hats only lightly
        # (kind: 0 kick, 1 snare, 2 hi-hat; old 2-band caches only have 0 / 1)
        w_kind = np.select([d[:, 1] < 0.5, d[:, 1] < 1.5], [0.4, 0.8], 0.15)
        add(d[:, 0], w_kind * np.clip(d[:, 2], 0, 1))
    b = _notes(a.bass)
    if len(b):
        add(b[:, START], np.full(len(b), 0.3))
    nz = strength[strength > 0]
    if len(nz):
        strength /= np.percentile(nz, 90)
    return strength


def accompaniment_part(song: Song, lv: Level, melody: np.ndarray, bass: np.ndarray,
                       skip_bars: np.ndarray, stats: dict) -> np.ndarray:
    if lv.accomp_notes <= 0:
        return empty()
    a = song.a
    grid = song.grid(lv.accomp_subdiv)
    if len(grid) == 0:
        return empty()
    strength = onset_strength(song, grid)
    pool = np.concatenate([_notes(a.other), _notes(a.other_piano)])
    pool_sal = np.concatenate([np.full(len(_notes(a.other)), 1.0), np.full(len(_notes(a.other_piano)), 0.8)])

    chord_starts = sorted(c[0] for c in song.chords if c[2] is not None)
    step = song.period / lv.accomp_subdiv

    # --- groove: ONE rhythm pattern per 4-bar section, from that section's combined onsets,
    # repeated in each of its bars (copying every bar's onsets separately gave a different
    # rhythm every bar: groove similarity 0.23-0.66 on the prototype songs)
    bar_idx = np.array([song.bar_index(t) for t in grid])
    pos = np.round((grid - song.bars[bar_idx]) / step).astype(int)
    group = bar_idx // 4
    pattern = {}
    for g in np.unique(group):
        m = group == g
        e_g = float(np.mean([song.bar_energy[b] for b in np.unique(bar_idx[m])]))
        thr = lv.accomp_threshold * (1.3 - 0.6 * e_g)
        for p in np.unique(pos[m]):
            mp = m & (pos == p)
            pattern[(g, p)] = float(strength[mp].mean()) >= thr
    slots = []
    for i, t in enumerate(grid):
        if skip_bars[bar_idx[i]]:
            continue
        at_change = any(abs(t - cs) < 0.5 * step for cs in chord_starts)
        at_bar = pos[i] == 0 and song.bar_energy[bar_idx[i]] > 0.3
        if pattern.get((group[i], pos[i]), False) or at_change or at_bar:
            slots.append(t)
    slots = np.array(sorted(set(np.round(slots, 4))))

    # --- voicing: chosen ONCE per chord span from the notes the song plays in that span
    # (restless re-voicing on every hit was 13-67% of hits on the prototype songs)
    span_pcs = {}

    def pcs_for(t):
        c = song.chord_at(t + 0.02)
        if c is not None:
            key, s, e, allowed = ("c", round(c[0], 3)), c[0], c[1], c[3]
        else:
            b = song.bar_index(t)
            key, s, e, allowed = ("b", b), song.bars[b], song.bars[b] + song.bar_len, None
        if key not in span_pcs:
            ov = np.clip(np.minimum(pool[:, END], e) - np.maximum(pool[:, START], s), 0, None)
            w = {}
            for row, o, sal in zip(pool, ov, pool_sal):
                pc = int(row[PITCH]) % 12
                if o <= 0 or (allowed is not None and pc not in allowed):
                    continue
                w[pc] = w.get(pc, 0.0) + o * sal * row[VEL]
            chosen = []
            for pc, _ in sorted(w.items(), key=lambda x: -x[1]):
                if len(chosen) >= lv.accomp_notes:
                    break
                if any((pc - q) % 12 in (1, 11) for q in chosen):
                    continue  # never voice two pitch classes a semitone apart
                chosen.append(pc)
            span_pcs[key] = chosen
        pcs = span_pcs[key]
        # of the span's voicing, keep what is actually heard within a beat of this hit (the
        # band may play a riff elsewhere in the span); nothing heard -> whole span voicing
        win = (pool[:, START] < t + song.period) & (pool[:, END] > t - song.period)
        heard = set(pool[win, PITCH].astype(int) % 12)
        return ([pc for pc in pcs if pc in heard] or pcs), c

    out, prev, empty_spans = [], None, 0
    for j, t in enumerate(slots):
        nxt = slots[j + 1] if j + 1 < len(slots) else song.duration
        pcs, c = pcs_for(t)
        if not pcs:
            empty_spans += 1
            continue  # the song plays nothing harmonic here: no invented chord
        e = song.energy_at(t)
        # range: above the bass, below the melody, at most one hand span wide
        mel_now = melody[(melody[:, START] <= t + 0.05) & (melody[:, END] > t)] if len(melody) else melody
        hi = min((mel_now[:, PITCH].min() - 1) if len(mel_now) else 76, 76)
        b_now = bass[(bass[:, START] <= t + 0.05) & (bass[:, END] > t)] if len(bass) else bass
        lo = max(48, (b_now[:, PITCH].max() + 3) if len(b_now) else 48)
        lo = min(lo, hi - 11)
        voicing = []
        for pc in pcs:
            options = [p for p in range(int(lo), int(hi) + 1) if p % 12 == pc]
            if not options:
                continue
            # keep the previous position when possible (stable, smooth voice leading)
            p = min(options, key=lambda q: min(abs(q - r) for r in prev)) if prev else max(options)
            voicing.append(p)
        if not voicing:
            continue
        top = max(voicing)
        voicing = sorted({p for p in voicing if top - p <= lv.span})
        end = min(nxt, t + 2 * song.period) - 0.02
        if c is not None:
            end = min(end, max(c[1], t + 0.1))
        # melody and bass win: drop voicing notes a semitone from anything they sound meanwhile
        others = [x for x in (melody, bass) if len(x)]
        if others:
            o = np.concatenate(others)
            o_pcs = set(o[(o[:, START] < end) & (o[:, END] > t), PITCH].astype(int) % 12)
            voicing = [p for p in voicing if not any((p - q) % 12 in (1, 11) for q in o_pcs)]
        if not voicing:
            continue
        prev = voicing
        vel = 42 + 30 * e
        for p in voicing:
            out.append([t, max(end, t + 0.08), p, vel])
    stats["accomp_hits"] = len(slots)
    stats["accomp_empty_spans"] = empty_spans
    return np.array(out, np.float64).reshape(-1, 4)


def verbatim_piano(song: Song, bars_mask: np.ndarray, melody: np.ndarray) -> np.ndarray:
    tk = np.asarray(song.a.other_piano, np.float64).reshape(-1, 4)
    keep = np.zeros(len(tk), bool)
    for i in np.where(bars_mask)[0]:
        s, e = song.bars[i], song.bars[i] + song.bar_len
        keep |= (tk[:, START] >= s) & (tk[:, START] < e)
    out = tk[keep & (tk[:, PITCH] > 0)]
    if len(melody) and len(out):  # stay under the melody so it remains on top
        ok = []
        for row in out:
            m = melody[(melody[:, START] <= row[START] + 0.03) & (melody[:, END] > row[START])]
            ok.append(not len(m) or row[PITCH] < m[:, PITCH].max())
        out = out[np.array(ok, bool)]
    return out


def pedal_part(song: Song, notes: np.ndarray, beats_per_pedal: int = 2) -> np.ndarray:
    """Sustain pedal that never creates a semitone rub.

    Re-pedal at every chord change and every `beats_per_pedal` beats; a span gets pedal
    only if all notes sounding in it (held by the pedal) are free of semitone / major-7th
    clashes. Holding one pedal over a whole chord made passing tones and the moving bass
    ring together: 19-31% of the time had a semitone rub vs 1-7% without pedal (10 songs)."""
    notes = notes[notes[:, PITCH] > 0]
    changes = sorted({c[0] for c in song.chords if c[2] is not None})
    cuts = np.unique(np.concatenate([song.beats[::beats_per_pedal], changes, [0.0, song.duration]]))
    cuts = cuts[(cuts >= 0) & (cuts <= song.duration)]
    out = []
    for cs, ce in zip(cuts[:-1], cuts[1:]):
        if ce - cs < 0.25:
            continue
        inside = notes[(notes[:, START] < ce) & (notes[:, END] > cs)]
        pcs = sorted(set(inside[:, PITCH].astype(int) % 12))
        if any((b - a) % 12 in (1, 11) for i, a in enumerate(pcs) for b in pcs[i + 1:]):
            continue
        out.append([cs + 0.03, ce - 0.04, -64, 127])
    return np.array(out, np.float64).reshape(-1, 4)


# --------------------------------------------------------------------------- entry point
def arrange(a: Analysis, level: str = "normal", song: Song | None = None, keep_piano: bool | None = None):
    """Return (events, stats) for one difficulty level. Pass `song` to reuse the shared
    structure across levels; keep_piano overrides the level's (experimental) setting."""
    lv = LEVELS[level]
    if keep_piano is not None:
        from dataclasses import replace

        lv = replace(lv, keep_piano=keep_piano)
    song = song or Song(a)
    stats = {"level": level}
    melody = melody_part(song, lv, stats)
    bass = bass_part(song, lv, stats)
    pb = piano_bars(song) if lv.keep_piano else np.zeros(len(song.bars), bool)
    stats["piano_bars"] = int(pb.sum())
    stats["bars"] = len(song.bars)
    accomp = accompaniment_part(song, lv, melody, bass, pb, stats)
    piano = verbatim_piano(song, pb, melody) if pb.any() else empty()

    e_mel = np.array([song.energy_at(t) for t in melody[:, START]]) if len(melody) else np.zeros(0)
    melody = melody.copy()
    melody[:, VEL] = 62 + 30 * e_mel
    bass = bass.copy()
    if len(bass):
        bass[:, VEL] = 50 + 30 * np.array([song.energy_at(t) for t in bass[:, START]])

    parts = [melody, bass, accomp, piano]
    stats["chords_replaced"] = song.chords_replaced
    if lv.pedal:
        parts.append(pedal_part(song, np.concatenate([p.reshape(-1, 4) for p in parts])))
    ev = sanitize(np.concatenate([p.reshape(-1, 4) for p in parts]))
    ev = limit_hands(ev, max_per_hand=lv.max_per_hand, span=lv.span)
    ev = sanitize(drop_ghost_notes(ev))
    ev = ev[(ev[:, START] < song.duration)]
    ev[:, END] = np.minimum(ev[:, END], song.duration)
    ev = sanitize(ev)
    stats.update(notes=int((ev[:, PITCH] > 0).sum()), melody_notes=len(melody), bass_notes=len(bass),
                 accomp_notes=len(accomp), piano_notes=len(piano))
    return ev, stats

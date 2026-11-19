"""Playability-first reduction: change only what two hands cannot play.

A transcription of an ensemble ("full") can have more simultaneous notes, or wider chords, than
two hands can play. `reduce()` walks through the onsets; at each one it looks at every note
sounding at that moment (held notes count: no pedal is assumed) and checks whether some split
into a left and a right hand satisfies the limits (notes per hand, span per hand). If one does,
nothing is changed there. Only where none does, it applies the smallest fix first, re-checking
after each step:

  1. end inner held notes at this onset (a pianist would release them);
  2. drop octave doublings among the new notes (keep the outer one);
  3. drop new inner notes whose pitch class is already sounding;
  4. drop new inner notes, middle of the chord first.

The highest and the lowest sounding notes (melody and bass) are never touched. Input that is
already playable comes back unchanged (same array values). See docs/TRANSKUN_PLAN.md.

Repeated notes (`repeat=`, applied before the playability pass). Merged unisons of an ensemble
and string tremolos turn into a "machine gun" on one piano key. First, two onsets of the same
pitch closer than `dup_tol` are always merged (an artefact of instruments entering a few ms
apart). Then a run of >= `min_run` re-strikes of one key faster than `min_ioi` apart is:
  "hold"     one long note, re-struck at most every `accent_sec` (the default for "reduced");
  "tremolo"  the same rhythm alternating the key with its octave (towards the middle);
  "half"     every other note dropped (repeatedly, until slower than `min_ioi`).
When several keys repeat together (a chordal ostinato), only the highest keeps the chosen
treatment; the others are held. Slower repetitions are left alone.

Not covered yet (planned): fast leaps between onsets.

  python -m pianocover.reduce in.mid -o out.mid [--max-notes 5] [--max-span 16] [--repeat hold]
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import numpy as np

from .events import END, PITCH, START, sanitize


@dataclass
class Limits:
    max_notes: int = 5      # notes per hand
    max_span: int = 16      # semitones per hand (a 10th)
    onset_tol: float = 0.035


@dataclass
class Report:
    onsets: int = 0
    changed_onsets: int = 0
    shortened: int = 0
    dropped: int = 0
    reasons: dict = field(default_factory=dict)
    repeat: dict | None = None   # stats of the repeated-note pass, when it ran

    def add(self, reason, n=1):
        self.reasons[reason] = self.reasons.get(reason, 0) + n


def _valid(p, lim):
    return len(p) == 0 or (len(p) <= lim.max_notes and p[-1] - p[0] <= lim.max_span)


def best_split(pitches, lim, prev):
    """Index k so that pitches[:k] (left) and pitches[k:] (right) are both playable, the split
    point closest to the previous one; None if no split works. pitches must be sorted."""
    best, best_d = None, None
    n = len(pitches)
    for k in range(n + 1):
        if _valid(pitches[:k], lim) and _valid(pitches[k:], lim):
            point = (pitches[k - 1] + pitches[k]) / 2 if 0 < k < n else (pitches[0] - 1 if k == 0 else pitches[-1] + 1)
            d = abs(point - prev)
            if best is None or d < best_d:
                best, best_d = k, d
    return best


def repetitions(ev: np.ndarray, mode: str = "hold", min_ioi: float = 0.11, dup_tol: float = 0.035,
                accent_sec: float = 0.5, min_run: int = 3):
    """Calm fast repeated notes (see the module docstring). Returns (events, stats)."""
    assert mode in ("hold", "tremolo", "half")
    ev = np.asarray(ev, np.float64)
    is_note = ev[:, PITCH] > 0
    notes, other = ev[is_note].copy(), ev[~is_note]
    stats = {"merged_duplicates": 0, "runs": 0, "ostinato_held": 0, "removed": 0, "moved_octave": 0}
    if len(notes) == 0:
        return ev.copy(), stats
    keep = np.ones(len(notes), bool)
    by_pitch = {}
    for i in np.lexsort((notes[:, START], notes[:, PITCH])):
        by_pitch.setdefault(int(notes[i, PITCH]), []).append(i)
    # 1. near-simultaneous onsets of one key -> one strike
    for p, idx in by_pitch.items():
        last = None
        kept = []
        for i in idx:
            if last is not None and notes[i, START] - notes[last, START] < dup_tol:
                notes[last, END] = max(notes[last, END], notes[i, END])
                notes[last, 3] = max(notes[last, 3], notes[i, 3])
                keep[i] = False
                stats["merged_duplicates"] += 1
            else:
                last = i
                kept.append(i)
        by_pitch[p] = kept
    # 2. runs of fast re-strikes
    runs = []
    for p, idx in by_pitch.items():
        cur = idx[:1]
        for i in idx[1:]:
            if notes[i, START] - notes[cur[-1], START] < min_ioi:
                cur.append(i)
            else:
                if len(cur) >= min_run:
                    runs.append((p, cur))
                cur = [i]
        if len(cur) >= min_run:
            runs.append((p, cur))
    stats["runs"] = len(runs)
    span = [(notes[r[0], START], notes[r[-1], END]) for _, r in runs]
    leader = [True] * len(runs)
    for a in range(len(runs)):  # chordal ostinato: only the highest key keeps the rhythm
        for b in range(len(runs)):
            if a == b or runs[b][0] <= runs[a][0]:
                continue
            ov = min(span[a][1], span[b][1]) - max(span[a][0], span[b][0])
            if ov > 0.5 * min(span[a][1] - span[a][0], span[b][1] - span[b][0]):
                leader[a] = False
                break
    for (p, run), lead in zip(runs, leader):
        m = mode if lead else "hold"
        stats["ostinato_held"] += int(not lead)
        end_run = notes[run[-1], END]
        if m == "tremolo":
            step = -12 if p >= 60 else 12
            for k, i in enumerate(run):
                if k % 2 == 1 and 21 <= p + step <= 108:
                    notes[i, PITCH] = p + step
                    stats["moved_octave"] += 1
            continue
        if m == "hold":
            kept = [run[0]]
            for i in run[1:]:
                if notes[i, START] - notes[kept[-1], START] >= accent_sec:
                    kept.append(i)
        else:  # half: drop every other note until slower than min_ioi
            kept = list(run)
            while len(kept) > 1 and min(np.diff(notes[kept, START])) < min_ioi:
                kept = kept[::2]
        for k, i in enumerate(kept):
            notes[i, END] = notes[kept[k + 1], START] if k + 1 < len(kept) else max(end_run, notes[i, END])
            notes[i, 3] = max(notes[j, 3] for j in run if notes[j, START] >= notes[i, START]
                              and (k + 1 == len(kept) or notes[j, START] < notes[kept[k + 1], START]))
        dropped = set(run) - set(kept)
        keep[list(dropped)] = False
        stats["removed"] += len(dropped)
    if keep.all() and stats["moved_octave"] == 0 and stats["merged_duplicates"] == 0:
        return ev.copy(), stats
    return sanitize(np.concatenate([notes[keep], other])), stats


def assign_hands(notes: np.ndarray, lim: Limits) -> np.ndarray:
    """0 = left, 1 = right for every note, from the split chosen at its onset (as in reduce)."""
    order = np.argsort(notes[:, START], kind="stable")
    hand = np.zeros(len(notes), int)
    prev = 60.0
    g0 = 0
    for i in range(1, len(order) + 1):
        if i < len(order) and notes[order[i], START] - notes[order[g0], START] <= lim.onset_tol:
            continue
        grp = order[g0:i]
        t = notes[order[g0], START] + lim.onset_tol
        snd = np.flatnonzero((notes[:, START] <= t) & (notes[:, END] > t))
        p = np.sort(notes[snd, PITCH])
        k = best_split(p, lim, prev)
        if k is not None and 0 < k < len(p):
            point = (p[k - 1] + p[k]) / 2
        elif k == 0:
            point = p[0] - 0.5
        elif k is not None:
            point = p[-1] + 0.5
        else:
            point = prev
        prev = 0.7 * prev + 0.3 * point if k is not None and 0 < k < len(p) else prev
        hand[grp] = (notes[grp, PITCH] > point).astype(int)
        g0 = i
    return hand


def _extend(t, period, lo, hi):
    """Beat times padded with `period` steps so they cover [lo, hi]."""
    t = np.sort(np.asarray(t, np.float64))
    pre = np.arange(t[0] - period, lo - period, -period)[::-1]
    post = np.arange(t[-1] + period, hi + 2 * period, period)
    return np.concatenate([pre, t, post])


def regularize_beats(beats, downbeats, notes, period, window=4):
    """Make a beat tracker's beats usable as a grid for these notes.

    beat_this beats jitter (seen: 0.20/0.26/0.16/0.24 s around a steady 0.217 s pulse) and sit
    11-39 ms before the notes. Snapping each beat to its nearest onset made it worse in dense
    figures (it lands on a random 32nd). Instead:
      1. de-jitter: each beat becomes the local linear fit of its 2*window+1 neighbours;
      2. phase: shift by the local median offset to STRONG onsets only (chords of >= 3 notes or
         strikes with a bass note), within 30% of a beat.
    Downbeats are mapped to the nearest regularised beat."""
    b = np.sort(np.asarray(beats, np.float64))
    n = len(b)
    reg = b.copy()
    if n >= 5:
        for i in range(n):
            lo_, hi_ = max(0, i - window), min(n, i + window + 1)
            k = np.arange(lo_, hi_)
            reg[i] = np.polyval(np.polyfit(k, b[lo_:hi_], 1), i)
    st = np.sort(notes[:, START])
    strong = []
    g0 = 0
    for i in range(1, len(st) + 1):
        if i == len(st) or st[i] - st[g0] > 0.035:
            grp = notes[(notes[:, START] >= st[g0] - 1e-9) & (notes[:, START] <= st[i - 1] + 1e-9)]
            if len(grp) >= 3 or grp[:, PITCH].min() < 55:
                strong.append(st[g0])
            g0 = i
    strong = np.array(strong)
    if len(strong) >= 2 and n:
        j = np.clip(np.searchsorted(strong, reg), 1, len(strong) - 1)
        left, right = strong[j - 1], strong[j]
        cand = np.where(np.abs(left - reg) <= np.abs(right - reg), left, right)
        res = cand - reg
        valid = np.abs(res) <= 0.3 * period
        off = np.zeros(n)
        for i in range(n):
            lo_, hi_ = max(0, i - window), min(n, i + window + 1)
            v = res[lo_:hi_][valid[lo_:hi_]]
            off[i] = float(np.median(v)) if len(v) >= 3 else 0.0
        reg = reg + off
    if downbeats is not None and len(downbeats) >= 2:
        downbeats = np.unique(reg[np.argmin(np.abs(reg[None, :] - np.asarray(downbeats)[:, None]), axis=1)])
    return reg, downbeats


def rhythm(ev: np.ndarray, beats: np.ndarray, downbeats: np.ndarray | None = None, mode: str = "grid",
           max_rate: float = 9.0, limits: Limits | None = None, dup_tol: float = 0.035):
    """Limit how fast each HAND strikes, on the beat grid (fast figures and repeated notes alike).

    Per hand and bar, the strike rate is measured. A bar at most `max_rate` strikes per second is
    untouched. Bars above it (plus single bars between two such bars, so the texture does not
    flip) form runs; a run gets ONE grid, the finest of 16ths / 8ths / quarters slower than
    `max_rate` that fits every bar in the run. Slots are spaced per beat (beat_this beats are not
    perfectly even) and every slot owns the strikes within half a step of it, so a note played
    slightly before a bar line still belongs to that downbeat.
      "grid":  per slot the strike nearest to it is kept and held until the hand's next strike
               (across bar lines too), the hand's other strikes in the run are dropped;
      "block": the notes a hand starts within one slot (one grid level coarser) become one chord
               placed on the slot; the playability pass trims it afterwards.
    Returns (events, stats)."""
    assert mode in ("grid", "block")
    lim = limits or Limits()
    ev, dstats = repetitions(ev, "hold", min_ioi=0.0, dup_tol=dup_tol, min_run=10 ** 9)  # duplicates only
    is_note = ev[:, PITCH] > 0
    notes, other = ev[is_note].copy(), ev[~is_note]
    stats = {"merged_duplicates": dstats["merged_duplicates"], "bars": 0, "bars_changed": 0, "removed": 0,
             "chords": 0}
    if len(notes) == 0 or len(beats) < 2:
        return ev, stats
    period = float(np.median(np.diff(np.sort(beats))))
    lo, hi = notes[:, START].min(), notes[:, START].max()
    beats, downbeats = regularize_beats(beats, downbeats, notes, period)
    stats["beats_regularized"] = True
    beats = _extend(beats, period, lo, hi)
    if downbeats is None or len(downbeats) < 2:
        downbeats = beats[::4]
    bars = _extend(downbeats, float(np.median(np.diff(np.sort(downbeats)))), lo, hi)
    bars = bars[(bars <= hi + 1e-6) | (np.arange(len(bars)) == np.searchsorted(bars, hi, side="right"))]
    stats["bars"] = len(bars) - 1
    hand = assign_hands(notes, lim)
    keep = np.ones(len(notes), bool)
    add = []
    changed_bars = set()
    for h in (0, 1):
        hidx = np.flatnonzero(hand == h)
        if len(hidx) == 0:
            continue
        # strike clusters of this hand
        order = hidx[np.argsort(notes[hidx, START], kind="stable")]
        cl_t, cl_m = [], []
        for i in order:
            if cl_t and notes[i, START] - cl_t[-1] <= lim.onset_tol:
                cl_m[-1].append(i)
            else:
                cl_t.append(notes[i, START])
                cl_m.append([i])
        cl_t = np.array(cl_t)
        nb = len(bars) - 1
        rate = np.array([((cl_t >= bars[k]) & (cl_t < bars[k + 1])).sum() / max(bars[k + 1] - bars[k], 1e-3)
                         for k in range(nb)])
        need = rate > max_rate
        # runs meet unreduced bars at their edges, where a fine and a coarse rhythm collide
        # (measured: most of the remaining long-short pairs): join runs at most 2 bars apart,
        # and grow runs into neighbouring bars that are almost as dense (> 70% of the limit)
        for gap in (1, 2):
            for k in range(1, nb - gap):
                if need[k - 1] and need[k + gap] and not need[k:k + gap].all():
                    need[k:k + gap] = True
        busy = rate > 0.7 * max_rate
        grown = True
        while grown:
            grown = False
            for k in range(nb):
                if not need[k] and busy[k] and ((k > 0 and need[k - 1]) or (k + 1 < nb and need[k + 1])):
                    need[k] = grown = True
        # beat-wise slots for each run of needed bars, one grid per run
        k = 0
        while k < nb:
            if not need[k]:
                k += 1
                continue
            r0 = k
            while k < nb and need[k]:
                k += 1
            r1 = k                                     # bars r0 .. r1-1
            t0, t1 = bars[r0], bars[r1]
            bt = beats[(beats >= t0 - 1e-6) & (beats <= t1 + 1e-6)]
            if len(bt) < 2:
                bt = np.array([t0, t1])
            shortest = float(np.min(np.diff(bt)))
            # binary (16ths / 8ths) or ternary (triplets) subdivision: whichever fine grid the
            # run's strikes sit closer to (residual / step; 0.25 = chance for both)
            run_t = cl_t[(cl_t >= t0 - 0.05) & (cl_t < t1)]

            def fit(n):
                g = np.concatenate([x + np.arange(n) * (y - x) / n for x, y in zip(bt[:-1], bt[1:])])
                if not len(run_t) or not len(g):
                    return 1.0
                j = np.clip(np.searchsorted(g, run_t), 1, len(g) - 1)
                r = np.minimum(np.abs(run_t - g[j - 1]), np.abs(run_t - g[j]))
                return float(np.mean(r) / (np.median(np.diff(bt)) / n))

            family = (6, 3, 1) if fit(6) < fit(4) - 0.05 else (4, 2, 1)
            sub = next((s for s in family if shortest / s >= 1.0 / max_rate), 1)
            if mode == "block":
                sub = max(1, sub // 2)
            slots, steps = [], []
            for x, y in zip(bt[:-1], bt[1:]):
                st = (y - x) / sub
                for j in range(sub):
                    slots.append(x + j * st)
                    steps.append(st)
            slots, steps = np.array(slots), np.array(steps)
            # the run owns its strikes, minus those up to ~55 ms before its end (an early-played
            # next downbeat) plus those up to ~55 ms before its start (its own early downbeat)
            early = lim.onset_tol + 0.02
            own_lo, own_hi = slots[0] - min(steps[0] / 2, early), t1 - min(steps[-1] / 2, early)
            own = np.flatnonzero((cl_t >= own_lo) & (cl_t < own_hi))
            if len(own) == 0:
                continue
            changed_bars.update(range(r0, r1))
            if mode == "grid":
                # keep only strikes ON a slot (an empty slot = the previous note goes on), so the
                # result is even by construction; "nearest strike" made 0.3/0.1 s patterns
                # wherever a slot had a rest or a syncopation (measured: long-short pairs in a
                # hand 0.1-2% before, 7.7-11.6% after). The run's grid phase (on the slots or
                # half a step later) is the one that catches more strikes.
                tol = np.minimum(steps / 4, 0.04)

                def on_slot(phase):
                    hit = {}
                    for s, st, tl in zip(slots + phase * steps, steps, tol):
                        cand = own[np.abs(cl_t[own] - s) <= tl]
                        if len(cand):
                            hit[s] = int(cand[np.argmin(np.abs(cl_t[cand] - s))])
                    return set(hit.values())

                kept = max((on_slot(0.0), on_slot(0.5)), key=len)
                for c in own:
                    if c not in kept:
                        keep[cl_m[c]] = False
                        stats["removed"] += len(cl_m[c])
            else:  # every owned strike goes to its nearest slot of this run
                slot_of = np.argmin(np.abs(cl_t[own][:, None] - slots[None, :]), axis=1)
                for si, (s, st) in enumerate(zip(slots, steps)):
                    win = own[slot_of == si]
                    if len(win) == 0 or (len(win) == 1 and abs(cl_t[win[0]] - s) < lim.onset_tol):
                        continue
                    members = [i for c in win for i in cl_m[c]]
                    for p in np.unique(notes[members, PITCH]):
                        w = [i for i in members if notes[i, PITCH] == p]
                        add.append([s, s + st, p, float(notes[w, 3].max())])
                    keep[members] = False
                    stats["removed"] += len(members)
                    stats["chords"] += 1
        if mode == "grid":  # hold every kept strike of this hand until its next kept strike
            alive_c = [c for c in range(len(cl_t)) if keep[cl_m[c]].any()]
            for a_, b_ in zip(alive_c, alive_c[1:] + [None]):
                nxt = cl_t[b_] if b_ is not None else None
                for i in cl_m[a_]:
                    if keep[i] and nxt is not None:
                        notes[i, END] = max(notes[i, END], min(nxt, notes[i, START] + 2 * period))
    stats["bars_changed"] = len(changed_bars)
    if not changed_bars:
        return ev, stats
    out = np.concatenate([notes[keep], np.array(add, np.float64).reshape(-1, 4), other])
    return sanitize(out), stats


def _clusters(notes, idx, tol):
    """Strike clusters of the given note indices: (times, [member index lists])."""
    order = idx[np.argsort(notes[idx, START], kind="stable")]
    t, m = [], []
    for i in order:
        if t and notes[i, START] - t[-1] <= tol:
            m[-1].append(i)
        else:
            t.append(notes[i, START])
            m.append([i])
    return np.array(t), m


def simplify(ev: np.ndarray, beats: np.ndarray, downbeats: np.ndarray | None = None, tremolo: str = "hold",
             max_rate: float = 9.0, trill_rate: float = 12.0, min_beats: float = 1.0,
             limits: Limits | None = None, dup_tol: float = 0.035):
    """Simplify repetitive fast textures, keep characteristic gestures (docs/TRANSKUN_PLAN.md).

    A *segment* is a chain of strikes of one hand and one stream (melody = the top voice, or
    accompaniment = everything else) closer than 1/max_rate apart. Then:
      * trill (2 keys a step apart, >= trill_rate per second), in any voice: shortened to a
        mordent: the first 3 strikes, then the main note held;
      * melody: otherwise untouched, except one key re-struck for >= min_beats beats (held,
        re-struck on the beats);
      * accompaniment shorter than min_beats beats (run, ornament, pickup): untouched;
      * accompaniment "repeat" (the same note / chord, or 2 notes alternating):
        tremolo="hold": the chord held and re-struck on every beat; tremolo="octave": its
        lowest note alternating with the octave above, in 8ths;
      * accompaniment "figure" (<= 4 keys cycling, each used >= 3 times on average: Alberti,
        broken chords, ostinato): the same figure on a coarser beat grid (16ths -> 8ths),
        strikes on the grid kept and held to the next one;
      * anything else (scales, passages over many different keys): untouched.
    Returns (events, stats); run `reduce()` afterwards for playability."""
    assert tremolo in ("hold", "octave")
    lim = limits or Limits()
    ev, dstats = repetitions(ev, "hold", min_ioi=0.0, dup_tol=dup_tol, min_run=10 ** 9)
    is_note = ev[:, PITCH] > 0
    notes, other = ev[is_note].copy(), ev[~is_note]
    stats = {"merged_duplicates": dstats["merged_duplicates"], "trill": 0, "melody_repeat": 0, "repeat": 0,
             "figure": 0, "kept_gestures": 0, "kept_short": 0, "removed": 0, "added": 0}
    beats = np.asarray(beats, np.float64)
    if len(notes) == 0 or len(beats) < 2:
        return ev, stats
    period = float(np.median(np.diff(np.sort(beats))))
    beats, _ = regularize_beats(beats, None, notes, period)
    beats = _extend(beats, period, notes[:, START].min(), notes[:, END].max())
    hand = assign_hands(notes, lim)
    # melody = the top note of a strike when it is the highest key sounding, in the right hand
    protected = np.zeros(len(notes), bool)
    ct, cm = _clusters(notes, np.arange(len(notes)), lim.onset_tol)
    for t_, mem_ in zip(ct, cm):
        top = max(mem_, key=lambda i: notes[i, PITCH])
        snd = (notes[:, START] <= t_ + lim.onset_tol) & (notes[:, END] > t_ + lim.onset_tol)
        if hand[top] == 1 and notes[top, PITCH] >= notes[snd, PITCH].max():
            protected[top] = True
    keep = np.ones(len(notes), bool)
    add = []
    gap = 1.0 / max_rate

    def beat_slots(t0, t1, sub=1):
        b = beats[(beats >= t0 - lim.onset_tol) & (beats <= t1 + lim.onset_tol)]
        if not len(b) or b[0] - t0 > lim.onset_tol:
            b = np.concatenate([[t0], b])
        out_ = []
        for x, y in zip(b, np.append(b[1:], b[-1] + period)):
            out_ += [x + j * (y - x) / sub for j in range(sub)]
        return np.array([x for x in out_ if x <= t1 + lim.onset_tol])

    for h in (0, 1):
        for melody in (False, True):
            idx = np.flatnonzero((hand == h) & (protected == melody))
            if len(idx) < 4:
                continue
            t, mem = _clusters(notes, idx, lim.onset_tol)
            # a strike taken by the melody (a figure note that was briefly the top voice) must
            # not cut an accompaniment segment: bridge gaps up to 2*gap that hold such a strike
            prot_t = np.sort(notes[(hand == h) & protected, START]) if not melody else np.zeros(0)

            def bridged(a, b):
                if b - a >= 2 * gap or not len(prot_t):
                    return False
                j = np.searchsorted(prot_t, a + lim.onset_tol)
                return j < len(prot_t) and prot_t[j] < b - lim.onset_tol

            segs, cur = [], [0]
            for k in range(1, len(t)):
                if t[k] - t[k - 1] < gap or bridged(t[k - 1], t[k]):
                    cur.append(k)
                else:
                    segs.append(cur)
                    cur = [k]
            segs.append(cur)
            for seg in segs:
                if len(seg) < 4:
                    continue
                t0, t1 = t[seg[0]], t[seg[-1]]
                members = [i for k in seg for i in mem[k]]
                end = float(notes[members, END].max())
                sets = [frozenset(int(notes[i, PITCH]) for i in mem[k]) for k in seg]
                union = sorted(set().union(*sets))
                rate = (len(seg) - 1) / max(t1 - t0, 1e-3)
                single = all(len(x) == 1 for x in sets)
                vel = float(notes[members, 3].max())
                long_enough = t1 - t0 >= min_beats * period
                if single and len(union) == 2 and union[1] - union[0] <= 2 and rate >= trill_rate:
                    counts = {p: sum(p in x for x in sets) for p in union}
                    main = max(union, key=lambda p: counts[p])
                    for k in seg[3:]:
                        keep[mem[k]] = False
                    last = mem[seg[2]][0]
                    notes[last, PITCH] = main
                    notes[last, END] = end
                    stats["trill"] += 1
                    stats["removed"] += len(seg) - 3
                    continue
                if melody:
                    if len(union) == 1 and long_enough:
                        slots = beat_slots(t0, t1)
                        keep[members] = False
                        for a, b in zip(slots, np.append(slots[1:], end)):
                            add.append([a, max(b, a + 0.05), union[0], vel])
                        stats["melody_repeat"] += 1
                        stats["removed"] += len(members)
                        stats["added"] += len(slots)
                    continue
                if not long_enough:
                    stats["kept_short"] += 1
                    continue
                # figure notes that were briefly the top voice belong to the figure, not the melody
                absorbed = np.flatnonzero((hand == h) & protected & keep & (notes[:, START] > t0 - lim.onset_tol)
                                          & (notes[:, START] < t1 + lim.onset_tol)
                                          & np.isin(notes[:, PITCH].astype(int), union))
                if all(x == sets[0] for x in sets) or (single and len(union) <= 2):
                    keep[absorbed] = False
                    keep[members] = False
                    stats["repeat"] += 1
                    stats["removed"] += len(members)
                    if tremolo == "hold":
                        slots = beat_slots(t0, t1)
                        for a, b in zip(slots, np.append(slots[1:], end)):
                            for p in union:
                                add.append([a, max(b, a + 0.05), p, vel])
                        stats["added"] += len(slots) * len(union)
                    else:
                        slots = beat_slots(t0, t1, sub=2)
                        low = union[0]
                        up = low + 12 if low + 12 <= 108 else low - 12
                        for j, (a, b) in enumerate(zip(slots, np.append(slots[1:], end))):
                            add.append([a, max(b, a + 0.05), low if j % 2 == 0 else up, vel])
                        stats["added"] += len(slots)
                    continue
                if len(union) <= 4 and len(seg) / len(union) >= 3:
                    bs = beat_slots(t0, t1)
                    shortest = float(np.min(np.diff(bs))) if len(bs) > 1 else period
                    sub = next((s_ for s_ in (4, 2, 1) if shortest / s_ >= gap), 1)
                    slots = beat_slots(t0, t1, sub)
                    step = shortest / sub
                    tol = min(step / 4, 0.04)

                    def hits(ph, slots=slots, step=step, tol=tol, seg=seg, t=t):
                        got = []
                        for s_ in slots + step * ph:
                            d = np.abs(t[seg] - s_)
                            j = int(np.argmin(d))
                            if d[j] <= tol and (not got or got[-1] != seg[j]):
                                got.append(seg[j])
                        return got

                    kept = max((hits(0.0), hits(0.5)), key=len)
                    if not kept:
                        stats["kept_gestures"] += 1
                        continue
                    on_grid = [s_ + step * (0.5 if kept != hits(0.0) else 0.0) for s_ in slots]
                    for i in absorbed:  # absorbed figure notes: kept only if they sit on the grid
                        if np.min(np.abs(np.array(on_grid) - notes[i, START])) > tol:
                            keep[i] = False
                            stats["removed"] += 1
                    for k in seg:
                        if k not in kept:
                            keep[mem[k]] = False
                            stats["removed"] += len(mem[k])
                    for a, b in zip(kept, kept[1:] + [None]):
                        nxt = t[b] if b is not None else end
                        for i in mem[a]:
                            notes[i, END] = max(notes[i, END], min(nxt, notes[i, START] + 2 * period))
                    stats["figure"] += 1
                    continue
                stats["kept_gestures"] += 1
    out = np.concatenate([notes[keep], np.array(add, np.float64).reshape(-1, 4), other])
    return sanitize(out), stats


def reduce(ev: np.ndarray, limits: Limits | None = None, repeat: str | None = None, **repeat_kw):
    """Return (events, Report). Non-note rows (pedal etc.) are passed through untouched.
    repeat: None (repeated notes untouched), "hold", "tremolo" or "half" (see `repetitions`)."""
    lim = limits or Limits()
    ev = np.asarray(ev, np.float64)
    rep_stats = None
    if repeat:
        ev, rep_stats = repetitions(ev, repeat, **repeat_kw)
    is_note = ev[:, PITCH] > 0
    notes, other = ev[is_note].copy(), ev[~is_note]
    rep = Report(repeat=rep_stats)
    if len(notes) == 0:
        return ev.copy(), rep
    order = np.argsort(notes[:, START], kind="stable")
    notes = notes[order]
    alive = np.ones(len(notes), bool)
    starts = notes[:, START]
    # onset clusters
    groups, g0 = [], 0
    for i in range(1, len(notes) + 1):
        if i == len(notes) or starts[i] - starts[g0] > lim.onset_tol:
            groups.append((g0, i))
            g0 = i
    prev_split = 60.0
    changed_any = False
    for a, b in groups:
        t = starts[a] + lim.onset_tol
        rep.onsets += 1
        new = np.arange(a, b)

        def sounding():
            idx = np.flatnonzero(alive[:b] & (notes[:b, END] > t))
            return idx[np.argsort(notes[idx, PITCH], kind="stable")]

        idx = sounding()
        k = best_split(notes[idx, PITCH], lim, prev_split)
        changed = False
        while k is None:
            p = notes[idx, PITCH]
            outer = {idx[0], idx[-1]}
            inner = [i for i in idx if i not in outer]
            held = [i for i in inner if i < a]
            fresh = [i for i in inner if a <= i < b]
            if held:  # 1. release an inner held note (the one nearest the middle first)
                mid = (p[0] + p[-1]) / 2
                i = min(held, key=lambda j: abs(notes[j, PITCH] - mid))
                notes[i, END] = starts[a]
                rep.shortened += 1
                rep.add("released held note")
            else:
                octave = [i for i in fresh if any(j != i and (notes[j, PITCH] - notes[i, PITCH]) % 12 == 0
                                                  for j in idx)]
                if octave:  # 2. octave doubling
                    i = octave[len(octave) // 2]
                    reason = "octave doubling"
                else:
                    dup = [i for i in fresh if sum(int(notes[j, PITCH]) % 12 == int(notes[i, PITCH]) % 12
                                                   for j in idx) > 1]
                    if dup:  # 3. repeated pitch class
                        i = dup[len(dup) // 2]
                        reason = "repeated pitch class"
                    elif fresh:  # 4. inner voice, middle first
                        mid = (p[0] + p[-1]) / 2
                        i = min(fresh, key=lambda j: abs(notes[j, PITCH] - mid))
                        reason = "inner voice"
                    else:
                        break  # only outer notes left: two notes are always playable
                alive[i] = False
                rep.dropped += 1
                rep.add(reason)
            changed = True
            idx = sounding()
            k = best_split(notes[idx, PITCH], lim, prev_split)
        if k is not None:
            p = notes[idx, PITCH]
            if 0 < k < len(p):
                prev_split = 0.7 * prev_split + 0.3 * (p[k - 1] + p[k]) / 2
        if changed:
            rep.changed_onsets += 1
            changed_any = True
    if not changed_any:
        return ev.copy(), rep
    out = np.concatenate([notes[alive], other])
    return sanitize(out), rep


def main(argv=None):
    import pretty_midi

    from . import events

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("midi")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--max-notes", type=int, default=5)
    ap.add_argument("--max-span", type=int, default=16)
    ap.add_argument("--repeat", choices=("hold", "tremolo", "half", "none"), default="hold")
    a = ap.parse_args(argv)
    pm = pretty_midi.PrettyMIDI(a.midi)
    ev = np.array([[n.start, n.end, n.pitch, n.velocity] for i in pm.instruments if not i.is_drum for n in i.notes],
                  np.float64).reshape(-1, 4)
    out, rep = reduce(ev, Limits(a.max_notes, a.max_span), repeat=None if a.repeat == "none" else a.repeat)
    if rep.repeat:
        print("repeated notes:", rep.repeat)
    events.to_pretty_midi(out).write(a.out)
    print(f"{rep.changed_onsets}/{rep.onsets} onsets changed, {rep.dropped} notes dropped, "
          f"{rep.shortened} shortened: {rep.reasons}")


if __name__ == "__main__":
    main()

"""Adaptive difficulty: choose the level per section from the song's intensity.

Fixed levels use one density everywhere, so 'full' is too busy in quiet verses and
'simple' too thin in choruses. Here every section (a run of bars with the same intensity
tier, see arrange.Song.bar_tier) takes its notes from the level matching that tier:
quiet -> simple, medium -> normal, loud and climax -> full. Runs shorter than
`min_bars` join the previous section, so the texture does not flip every bar.
Pedal is recomputed over the spliced result (fixed levels differ in pedal use).
"""

from __future__ import annotations

import numpy as np

from ..events import PITCH, START, sanitize
from ..postprocess import limit_hands
from .arrange import LEVELS, Song, pedal_part

TIER_LEVEL = ("simple", "normal", "full", "full")


def section_tiers(song: Song, min_bars: int = 4) -> np.ndarray:
    """Per-bar tier with runs shorter than `min_bars` merged into a neighbour."""
    tier = np.asarray(song.bar_tier, int).copy()
    changed = True
    while changed and len(tier):
        changed = False
        starts = np.flatnonzero(np.r_[True, tier[1:] != tier[:-1]])
        ends = np.r_[starts[1:], len(tier)]
        if len(starts) == 1:
            break
        for s, e in zip(starts, ends):
            if e - s < min_bars:
                tier[s:e] = tier[s - 1] if s > 0 else tier[e]
                changed = True
                break
    return tier


def arrange_adaptive(fixed: dict, song: Song, min_bars: int = 4, tier_level=TIER_LEVEL):
    """fixed: {level: (events, stats)} that includes every level in `tier_level`."""
    tier = section_tiers(song, min_bars)
    bar_level = [tier_level[t] for t in tier]
    parts = []
    for name in set(bar_level):
        ev = fixed[name][0]
        ev = ev[ev[:, PITCH] > 0]  # notes only; pedal is redone below
        bars = np.clip(np.searchsorted(song.bars, ev[:, START], side="right") - 1, 0, len(song.bars) - 1)
        parts.append(ev[np.array([bar_level[b] == name for b in bars], bool)])
    notes = sanitize(np.concatenate(parts)) if parts else np.zeros((0, 4))
    notes = limit_hands(notes, max_per_hand=LEVELS["full"].max_per_hand, span=LEVELS["full"].span)
    ev = sanitize(np.concatenate([notes, pedal_part(song, notes)]))
    used = {n: bar_level.count(n) for n in dict.fromkeys(tier_level)}
    return ev, {"level": "adaptive", "notes": int((ev[:, PITCH] > 0).sum()), "bars_per_level": used,
                "sections": int(1 + np.count_nonzero(np.diff(tier)))}

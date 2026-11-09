"""Chord labels (Harte / MIREX syntax as produced by BTC) -> pitch-class sets. Pure Python."""

from __future__ import annotations

ROOTS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

QUALITIES = {
    "maj": (0, 4, 7), "min": (0, 3, 7), "dim": (0, 3, 6), "aug": (0, 4, 8),
    "min6": (0, 3, 7, 9), "maj6": (0, 4, 7, 9), "min7": (0, 3, 7, 10), "minmaj7": (0, 3, 7, 11),
    "maj7": (0, 4, 7, 11), "7": (0, 4, 7, 10), "dim7": (0, 3, 6, 9), "hdim7": (0, 3, 6, 10),
    "sus2": (0, 2, 7), "sus4": (0, 5, 7), "9": (0, 4, 7, 10, 2), "maj9": (0, 4, 7, 11, 2),
    "min9": (0, 3, 7, 10, 2), "5": (0, 7), "1": (0,),
}
DEGREES = {"1": 0, "b2": 1, "2": 2, "b3": 3, "3": 4, "4": 5, "#4": 6, "b5": 6, "5": 7, "#5": 8, "b6": 8,
           "6": 9, "bb7": 9, "b7": 10, "7": 11, "9": 2, "b9": 1, "#9": 3, "11": 5, "#11": 6, "13": 9}


def parse_root(text: str) -> int:
    pc = ROOTS[text[0].upper()]
    for acc in text[1:]:
        pc += 1 if acc == "#" else -1 if acc == "b" else 0
    return pc % 12


def parse_chord(label: str):
    """'C#:min7/b3' -> (root_pc, frozenset of pcs, bass_pc). No chord ('N', 'X', '') -> None."""
    label = (label or "").strip()
    if label in ("", "N", "X"):
        return None
    bass = None
    if "/" in label:
        label, inv = label.split("/", 1)
        bass = DEGREES.get(inv)
    if ":" in label:
        root_s, qual = label.split(":", 1)
    else:
        root_s, qual = label, "maj"
    root = parse_root(root_s)
    extra = ()
    if "(" in qual:
        qual, ext = qual.split("(", 1)
        extra = tuple(DEGREES[d] for d in ext.rstrip(")").split(",") if d in DEGREES)
    intervals = QUALITIES.get(qual or "maj", QUALITIES["maj"]) + extra
    pcs = frozenset((root + i) % 12 for i in intervals)
    bass_pc = (root + bass) % 12 if bass is not None else root
    return root, pcs, bass_pc

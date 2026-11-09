"""Per-source sampling weights (torch-free helpers for train.py)."""

from __future__ import annotations


def parse_source_weights(text: str | None) -> dict:
    """'gold=4,teacher=1,silver=1' -> {'gold': 4.0, ...}; sources not listed get weight 1."""
    out = {}
    for part in (text or "").split(","):
        if part.strip():
            k, v = part.split("=", 1)
            out[k.strip()] = float(v)
            if out[k.strip()] < 0:
                raise ValueError(f"negative weight for {k}")
    return out


def chunk_weights(dataset, source_weights: dict):
    """Per-chunk sampling weights so each source's share of training is controlled by the user."""
    if not source_weights:
        return None
    return [source_weights.get(dataset.items[idx].get("source", "web"), 1.0) for idx, _, _ in dataset.chunks]


def source_summary(dataset, source_weights: dict) -> dict:
    """Fraction of sampled chunks per source after weighting."""
    totals = {}
    for idx, _, _ in dataset.chunks:
        src = dataset.items[idx].get("source", "web")
        totals[src] = totals.get(src, 0.0) + source_weights.get(src, 1.0)
    z = sum(totals.values()) or 1.0
    return {k: round(v / z, 3) for k, v in sorted(totals.items())}

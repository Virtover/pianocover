"""The analysis of one song: everything the arranger needs, cached as .npz.

Caching the analysis means the (cheap, CPU-only) arranger can be re-tuned and re-run on
thousands of songs without touching the GPU models again.

Note arrays are (N, 4): [start_sec, end_sec, pitch, velocity], as in pianocover.events.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields

import numpy as np

FRAME_HOP = 0.05  # seconds per frame of the energy curves


@dataclass
class Analysis:
    duration: float
    beats: np.ndarray                 # (B,) beat times
    downbeats: np.ndarray             # (D,) downbeat times (subset of beats, approximately)
    chords: np.ndarray                # (C, 2) [start, end]
    chord_labels: list                # C labels, e.g. 'A:min7', 'N'
    vocal: np.ndarray                 # basic-pitch on the vocal stem
    bass: np.ndarray                  # basic-pitch on the bass stem
    other: np.ndarray                 # basic-pitch on 'other' (guitars, keys, synths, strings)
    other_piano: np.ndarray           # Transkun on 'other' (incl. pedal rows, pitch -64/-67)
    drum_onsets: np.ndarray           # (K, 3) [time, kind (0 low/kick, 1 high/snare), strength]
    mix_rms: np.ndarray               # (F,) RMS of the mix, FRAME_HOP spacing
    vocal_rms: np.ndarray             # (F,)
    other_rms: np.ndarray             # (F,)
    meta: dict = field(default_factory=dict)

    def save(self, path) -> None:
        arrays = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if f.name == "meta":
                arrays["meta_json"] = np.array(__import__("json").dumps(v))
            elif f.name == "chord_labels":
                arrays[f.name] = np.array(list(v), dtype=str)
            else:
                arrays[f.name] = np.asarray(v)
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path) -> "Analysis":
        import json

        z = np.load(path, allow_pickle=False)
        kw = {}
        for f in fields(cls):
            if f.name == "meta":
                kw["meta"] = json.loads(str(z["meta_json"])) if "meta_json" in z else {}
            elif f.name == "chord_labels":
                kw[f.name] = [str(x) for x in z[f.name]]
            elif f.name == "duration":
                kw[f.name] = float(z[f.name])
            else:
                kw[f.name] = np.asarray(z[f.name], dtype=np.float64)
        for name in ("vocal", "bass", "other", "other_piano"):
            kw[name] = kw[name].reshape(-1, 4)
        kw["chords"] = kw["chords"].reshape(-1, 2)
        kw["drum_onsets"] = kw["drum_onsets"].reshape(-1, 3)
        return cls(**kw)

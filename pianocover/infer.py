"""Generate a piano cover MIDI from any song.

  python -m pianocover.infer song.mp3 cover.mid --ckpt runs/cover1/best.pt --style easy
  python -m pianocover.infer song.mp3 cover.mid --ckpt best.pt --set log_onset_rate=-1 polyphony=-0.5

Style presets: normal, easy, simple, full, virtuoso (see pianocover.style). --set overrides
single features in standard deviations from the training average.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from . import SAMPLE_RATE, events
from . import style as style_mod
from .audio import load_any
from .model import PianoCoverModel
from .postprocess import estimate_beats, postprocess

_CACHE: dict = {}


def load_model(ckpt_path, device="cuda"):
    key = (str(ckpt_path), str(device))
    if key not in _CACHE:
        model, _ = PianoCoverModel.from_checkpoint(ckpt_path, map_location=device)
        _CACHE[key] = model.eval()
    return _CACHE[key]


def generate_cover(audio_path, ckpt_path, device="cuda", style="normal", overrides=None,
                   playable=True, quantize=False, beats=None, max_per_hand=4) -> np.ndarray:
    """Return cover events (N, 4) for one audio file. Reuses the loaded model across calls."""
    model = load_model(ckpt_path, device)
    vec = style_mod.preset_vector(style, **(overrides or {}))
    audio = torch.from_numpy(load_any(audio_path, SAMPLE_RATE, mono=True)).to(device)
    notes = model.generate(audio, torch.from_numpy(vec).to(device))
    ev = events.sanitize(events.from_transkun_notes(notes))
    if quantize and beats is None:
        beats = estimate_beats(audio_path)
    return postprocess(ev, beats=beats if quantize else None, playable=playable, max_per_hand=max_per_hand)


def _parse_overrides(items):
    out = {}
    for item in items or []:
        k, v = item.split("=", 1)
        out[k.strip()] = float(v)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio")
    ap.add_argument("out_mid")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--style", default="normal", choices=sorted(style_mod.PRESETS))
    ap.add_argument("--set", nargs="*", metavar="FEATURE=STD", help=f"features: {', '.join(style_mod.FEATURES)}")
    ap.add_argument("--raw", action="store_true", help="skip playability limits")
    ap.add_argument("--quantize", action="store_true", help="snap to a beat grid (for sheet music)")
    ap.add_argument("--beats", help="text file with one beat time (s) per line, e.g. from beat_this")
    ap.add_argument("--max-per-hand", type=int, default=4)
    args = ap.parse_args(argv)

    beats = np.loadtxt(args.beats, ndmin=1) if args.beats else None
    torch.set_grad_enabled(False)
    ev = generate_cover(args.audio, args.ckpt, args.device, args.style, _parse_overrides(args.set),
                        playable=not args.raw, quantize=args.quantize or beats is not None, beats=beats,
                        max_per_hand=args.max_per_hand)
    events.to_pretty_midi(ev).write(args.out_mid)
    n = int((ev[:, events.PITCH] > 0).sum())
    print(f"wrote {args.out_mid}: {n} notes")


if __name__ == "__main__":
    main()

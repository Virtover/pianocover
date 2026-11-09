"""pianocover: style-conditioned piano cover generation built on Transkun.

Pipeline: pop audio -> fine-tuned Transkun (semi-CRF note intervals) -> piano cover MIDI.
"""

__version__ = "0.1.0"

# Transkun's model works at 44.1 kHz with 1024-sample hops (~23 ms frames).
SAMPLE_RATE = 44100
HOP_SIZE = 1024
FRAME_SEC = HOP_SIZE / SAMPLE_RATE

PIANO_MIN = 21
PIANO_MAX = 108
PEDAL_PITCHES = (-64, -67)  # Transkun encodes sustain / soft pedal as negative "pitches"

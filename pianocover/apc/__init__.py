"""AMT-APC backend: fine-tune AMT-APC (hFT-Transformer + style vector) on aligned cover pairs.

Pipeline (after pianocover.download / transcribe_covers / build_dataset):
  python -m pianocover.apc.prepare --ds /data/ds            # spectrograms + style vectors
  python -m pianocover.apc.train   --ds /data/ds --init apc.pth --out /data/runs/apc1

Constants are AMT-APC's config.json (github.com/310hz/amt-apc).
"""

SR = 16000
HOP = 256
FPS = SR / HOP  # 62.5 frames per second, 16 ms
N_FFT = 2048
N_BINS = 256
LOG_OFFSET = 1e-8
MIN_VALUE = -18.0  # padding value of the log-mel input
MARGIN = 32  # context frames on each side of the window
NUM_FRAME = 512  # output frames per window (8.192 s)
N_NOTE = 88
NOTE_MIN = 21
N_VELOCITY = 128
HID_DIM = 256
PF_DIM = 512
N_HEADS = 4
N_LAYERS = 3
SV_DIM = 24

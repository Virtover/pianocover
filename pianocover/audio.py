"""Audio loading helpers."""

from __future__ import annotations

import numpy as np

from . import SAMPLE_RATE


def read_wav_slice(path, begin: float, end: float) -> np.ndarray:
    """Read [begin, end) seconds of a 16-bit WAV via mmap, zero-padding outside the file.

    Returns float32 array [nSample, nChannel] in [-1, 1].
    """
    from scipy.io import wavfile

    fs, data = wavfile.read(path, mmap=True)
    if data.ndim == 1:
        data = data[:, None]
    b = int(np.floor(begin * fs))
    n = int(np.floor(end * fs)) - b
    lo, hi = max(b, 0), min(b + n, data.shape[0])
    chunk = np.asarray(data[lo:hi], dtype=np.float32)
    if np.issubdtype(data.dtype, np.integer):
        chunk /= np.iinfo(data.dtype).max
    pad_l, pad_r = lo - b, (b + n) - hi
    if pad_l > 0 or pad_r > 0:
        chunk = np.pad(chunk, ((max(pad_l, 0), max(pad_r, 0)), (0, 0)))
    return chunk[:n]


def wav_info(path) -> tuple[int, float]:
    from scipy.io import wavfile

    fs, data = wavfile.read(path, mmap=True)
    return fs, data.shape[0] / fs


def load_any(path, sr: int = SAMPLE_RATE, mono: bool = True) -> np.ndarray:
    """Load any audio file (mp3/m4a/wav/...) as float32 [nSample, nChannel] at `sr`."""
    import librosa

    y, _ = librosa.load(str(path), sr=sr, mono=mono)
    if y.ndim == 1:
        y = y[:, None]
    else:
        y = y.T
    return np.ascontiguousarray(y, dtype=np.float32)

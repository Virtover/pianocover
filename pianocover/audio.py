"""Audio loading helpers.

Downloads are kept in the compressed form YouTube serves (opus/webm or m4a); WAV is still
accepted everywhere. `decode` turns any file into float32 samples with ffmpeg.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from . import SAMPLE_RATE

AUDIO_EXTS = (".wav", ".webm", ".opus", ".m4a", ".mp4", ".mp3", ".ogg", ".flac", ".aac", ".mka")


def find_audio(directory, stem: str) -> Path | None:
    """The audio file `<directory>/<stem>.<ext>` for any known extension, or None."""
    directory = Path(directory)
    for ext in AUDIO_EXTS:
        p = directory / f"{stem}{ext}"
        if p.exists() and p.stat().st_size > 1000:
            return p
    return None


def list_audio(directory) -> list[Path]:
    """All finished audio files in a directory (partial downloads excluded), sorted."""
    return sorted(p for p in Path(directory).glob("*")
                  if p.suffix.lower() in AUDIO_EXTS and ".part" not in p.name and p.stat().st_size > 1000)


def decode(path, sr: int = SAMPLE_RATE, begin: float = 0.0, duration: float | None = None) -> np.ndarray:
    """Decode (a span of) any audio file to mono float32 [nSample] at `sr` with ffmpeg."""
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error"]
    if begin > 0:
        cmd += ["-ss", f"{begin:.6f}"]
    cmd += ["-i", str(path)]
    if duration is not None:
        cmd += ["-t", f"{duration:.6f}"]
    cmd += ["-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on {path}: {r.stderr.decode(errors='ignore').strip()[-300:]}")
    return np.frombuffer(r.stdout, dtype=np.float32).copy()


def read_wav_slice(path, begin: float, end: float) -> np.ndarray:
    """Read [begin, end) seconds, zero-padding outside the file. Float32 [nSample, nChannel].

    WAV files are read via mmap (fast random access). Other formats are decoded with an
    ffmpeg seek, which is much slower: cache them as WAV for chunk-wise training.
    """
    if Path(path).suffix.lower() != ".wav":
        n = int(np.floor(end * SAMPLE_RATE)) - int(np.floor(begin * SAMPLE_RATE))
        y = decode(path, SAMPLE_RATE, max(begin, 0.0), end - max(begin, 0.0))
        pad_l = int(round(max(-begin, 0.0) * SAMPLE_RATE))
        y = np.concatenate([np.zeros(pad_l, np.float32), y])[:n]
        return np.pad(y, (0, n - len(y)))[:, None]

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
    """(sample rate, duration in seconds); any format (ffprobe for non-WAV)."""
    if Path(path).suffix.lower() != ".wav":
        r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=sample_rate",
                            "-of", "default=nw=1", str(path)], capture_output=True, text=True)
        vals = dict(line.split("=", 1) for line in r.stdout.splitlines() if "=" in line)
        return int(vals.get("sample_rate", 0) or 0), float(vals.get("duration", 0) or 0)
    from scipy.io import wavfile

    fs, data = wavfile.read(path, mmap=True)
    return fs, data.shape[0] / fs


def load_any(path, sr: int = SAMPLE_RATE, mono: bool = True) -> np.ndarray:
    """Load any audio file (mp3/m4a/webm/wav/...) as float32 [nSample, 1] at `sr` (mono)."""
    return np.ascontiguousarray(decode(path, sr)[:, None])

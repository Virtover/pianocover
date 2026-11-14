"""Log-mel input features, identical to AMT-APC's `wav2feature` (torchaudio, slaney norm)."""

from __future__ import annotations

import numpy as np
import torch

from . import HOP, LOG_OFFSET, N_BINS, N_FFT, SR

_MEL = {}


def _mel(device):
    key = str(device)
    if key not in _MEL:
        import torchaudio

        _MEL[key] = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR, n_fft=N_FFT, win_length=N_FFT, hop_length=HOP, pad_mode="constant",
            n_mels=N_BINS, norm="slaney").to(device)
    return _MEL[key]


@torch.no_grad()
def logmel(wave_16k: np.ndarray, device="cpu") -> np.ndarray:
    """Mono 16 kHz samples -> log-mel [n_frames, 256] float32 (n_frames = len // 256 + 1)."""
    x = torch.from_numpy(np.ascontiguousarray(wave_16k, dtype=np.float32)).to(device)
    spec = _mel(device)(x)
    return torch.log(spec + LOG_OFFSET).T.float().cpu().numpy()

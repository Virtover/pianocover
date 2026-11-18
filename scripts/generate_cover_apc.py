"""Generate a piano cover (MIDI) from an audio file with AMT-APC, using a chosen model file.

Runs AMT-APC's own `infer` (in /data/ext/amt-apc of the pc-data volume), but loads audio with
soundfile / librosa: torchaudio 2.9 in the pianocover image needs the torchcodec package for
torchaudio.load, which the image doesn't have.

    # inside the pianocover image, from AMT-APC's directory:
    cd /data/ext/amt-apc && python /w/scripts/generate_cover_apc.py song.wav -o cover.mid -s level2 \
        --path_model /data/runs/apc_5070/best.pth

Styles: level1 (simple), level2 (medium, default), level3 (rich). Without --path_model it uses the
original apc.pth. Convert webm/opus downloads to WAV first (ffmpeg); soundfile reads WAV directly.
"""
import sys

import numpy as np
import torch
import torchaudio

AMT_APC = "/data/ext/amt-apc"


def _load(path, *args, **kwargs):
    try:
        import soundfile as sf
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)  # [frames, channels]
        return torch.from_numpy(np.ascontiguousarray(data.T)), sr
    except Exception:
        import librosa
        data, sr = librosa.load(str(path), sr=None, mono=False)  # mp3 etc. through audioread/ffmpeg
        data = np.atleast_2d(data).astype(np.float32)
        return torch.from_numpy(data), sr


torchaudio.load = _load

sys.path.insert(0, AMT_APC)
sys.argv[0] = AMT_APC + "/infer"
import runpy  # noqa: E402

runpy.run_path(AMT_APC + "/infer/__main__.py", run_name="__main__")

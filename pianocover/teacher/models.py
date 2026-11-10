"""All teacher-pipeline models, loaded once and kept resident (server style).

  separator  Demucs htdemucs (4 stems: drums, bass, other, vocals)   MIT code, weights see docs
  notes      basic-pitch ICASSP-2022 ONNX via onnxruntime (CUDA EP)  Apache-2.0
  piano      Transkun V2 (pip package checkpoint)                    MIT
  beats      beat_this final0                                         MIT (code + weights)
  chords     BTC large-vocabulary (vendored repo, commit 2682317)     MIT

License notes and check dates: docs/MODEL_LICENSES.md. htdemucs_6s is deliberately NOT
supported (its weights are likely CC BY-NC per the Demucs maintainer).
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

import numpy as np
import torch

SR = 44100
BP_SR = 22050
BTC_SR = 22050
BTC_DIR = os.environ.get("BTC_DIR", "/opt/btc")


def decode_audio(path, sr: int = SR, channels: int = 2) -> np.ndarray:
    """Any audio file -> float32 [channels, n] at `sr` via ffmpeg."""
    r = subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(path), "-f", "f32le",
                        "-ac", str(channels), "-ar", str(sr), "-"], capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on {path}: {r.stderr.decode()[:300]}")
    return np.frombuffer(r.stdout, np.float32).reshape(-1, channels).T.copy()


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    import soxr

    return x if sr_in == sr_out else soxr.resample(x, sr_in, sr_out).astype(np.float32)


class TeacherModels:
    def __init__(self, device: str = "cuda", demucs_name: str = "htdemucs", demucs_shifts: int = 0,
                 beat_ckpt: str = "final0", basic_pitch_gpu: bool = True, transkun_bf16: bool = False):
        if demucs_name == "htdemucs_6s":
            raise ValueError("htdemucs_6s weights are likely CC BY-NC; see docs/MODEL_LICENSES.md")
        self.device = torch.device(device)
        self.demucs_shifts = demucs_shifts
        self.load_times = {}

        t = time.time()
        from demucs.pretrained import get_model

        self.demucs = get_model(demucs_name).to(self.device).eval()
        self.load_times["demucs"] = time.time() - t

        t = time.time()
        from ..model import load_pretrained_transkun

        self.transkun = load_pretrained_transkun(device=self.device)
        self.transkun_bf16 = transkun_bf16
        self.load_times["transkun"] = time.time() - t

        t = time.time()
        from beat_this.inference import Audio2Beats

        self.beat = Audio2Beats(checkpoint_path=beat_ckpt, device=str(self.device), dbn=False)
        self.load_times["beat_this"] = time.time() - t

        t = time.time()
        self._load_btc()
        self.load_times["btc"] = time.time() - t

        t = time.time()
        self._load_basic_pitch(basic_pitch_gpu and self.device.type == "cuda")
        self.load_times["basic_pitch"] = time.time() - t

    # ------------------------------------------------------------------ loading
    def _load_btc(self):
        if BTC_DIR not in sys.path:
            sys.path.insert(0, BTC_DIR)
        from btc_model import BTC_model  # noqa: E402  (vendored, MIT)
        from utils.hparams import HParams
        from utils.mir_eval_modules import idx2voca_chord

        import yaml

        with open(os.path.join(BTC_DIR, "run_config.yaml")) as f:  # HParams.load uses a removed yaml API
            cfg = HParams(**yaml.safe_load(f))
        cfg.feature["large_voca"] = True
        cfg.model["num_chords"] = 170
        self.btc_cfg = cfg
        self.btc = BTC_model(config=cfg.model).to(self.device).eval()
        ckpt = torch.load(os.path.join(BTC_DIR, "test", "btc_model_large_voca.pt"), map_location="cpu",
                          weights_only=False)
        self.btc.load_state_dict(ckpt["model"])
        self.btc_mean, self.btc_std = float(ckpt["mean"]), float(ckpt["std"])
        self.btc_labels = idx2voca_chord()

    def _load_basic_pitch(self, gpu: bool):
        import onnxruntime as ort
        from basic_pitch import build_icassp_2022_model_path, FilenameSuffix

        path = str(build_icassp_2022_model_path(FilenameSuffix.onnx))
        providers = ["CPUExecutionProvider"]
        if gpu and "CUDAExecutionProvider" in ort.get_available_providers():
            try:
                ort.preload_dlls()  # CUDA / cuDNN from the nvidia-* wheels that torch already uses
            except Exception:
                pass
            providers = [("CUDAExecutionProvider", {"device_id": self.device.index or 0,
                                                    "arena_extend_strategy": "kSameAsRequested"}),
                         "CPUExecutionProvider"]
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.bp = ort.InferenceSession(path, sess_options=so, providers=providers)
        self.bp_provider = self.bp.get_providers()[0]

    # ------------------------------------------------------------------ inference
    @torch.inference_mode()
    def separate(self, mix: np.ndarray) -> dict:
        """mix: [2, n] float32 at 44.1 kHz -> {'drums','bass','other','vocals'}: mono [n]."""
        from demucs.apply import apply_model

        wav = torch.from_numpy(mix).to(self.device)
        ref = wav.mean(0)
        mean, std = ref.mean(), ref.std() + 1e-8
        out = apply_model(self.demucs, ((wav - mean) / std)[None], shifts=self.demucs_shifts, split=True,
                          overlap=0.25, device=self.device, progress=False)[0]
        out = out * std + mean
        return {name: out[i].mean(0).float().cpu().numpy() for i, name in enumerate(self.demucs.sources)}

    def basic_pitch(self, mono: np.ndarray, sr: int, onset_threshold=0.5, frame_threshold=0.3,
                    min_note_ms=80.0, min_freq=None, max_freq=None, melodia_trick=True,
                    batch_size=64) -> np.ndarray:
        """basic-pitch on an in-memory signal (batched windows) -> (N, 4) notes."""
        from basic_pitch.constants import AUDIO_N_SAMPLES, AUDIO_SAMPLE_RATE, FFT_HOP
        from basic_pitch.inference import unwrap_output
        from basic_pitch.note_creation import model_output_to_notes

        x = resample(np.asarray(mono, np.float32), sr, AUDIO_SAMPLE_RATE)
        n_olap = 30
        overlap_len = n_olap * FFT_HOP
        hop = AUDIO_N_SAMPLES - overlap_len
        orig_len = len(x)
        x = np.concatenate([np.zeros(overlap_len // 2, np.float32), x])
        n_win = max(int(np.ceil(len(x) / hop)), 1)
        x = np.pad(x, (0, n_win * hop + AUDIO_N_SAMPLES - len(x)))
        idx = np.arange(n_win)[:, None] * hop + np.arange(AUDIO_N_SAMPLES)[None]
        windows = x[idx][..., None].astype(np.float32)
        outs = {"note": [], "onset": [], "contour": []}
        names = ["StatefulPartitionedCall:1", "StatefulPartitionedCall:2", "StatefulPartitionedCall:0"]
        for b in range(0, n_win, batch_size):
            res = self.bp.run(names, {"serving_default_input_2:0": windows[b:b + batch_size]})
            for k, v in zip(("note", "onset", "contour"), res):
                outs[k].append(v)
        output = {k: unwrap_output(np.concatenate(v), orig_len, n_olap) for k, v in outs.items()}
        min_len = int(np.round(min_note_ms / 1000 * (AUDIO_SAMPLE_RATE / FFT_HOP)))
        _, events = model_output_to_notes(output, onset_thresh=onset_threshold, frame_thresh=frame_threshold,
                                          min_note_len=min_len, min_freq=min_freq, max_freq=max_freq,
                                          include_pitch_bends=False, melodia_trick=melodia_trick)
        if not events:
            return np.zeros((0, 4))
        ev = np.array([[s, e, p, np.clip(a * 127, 1, 127)] for s, e, p, a, _ in events], np.float64)
        return ev[np.argsort(ev[:, 0], kind="stable")]

    @torch.inference_mode()
    def vocal_melody(self, mono: np.ndarray, sr: int, onsets=None, hop_sec: float = 0.02,
                     gate_db: float = 40.0, voiced_threshold: float = 0.3) -> np.ndarray:
        """CREPE 'full' f0 on the vocal stem -> sung notes (see f0notes). Only regions where
        the stem is not silent are analysed (saves 30-50% on typical songs).
        Measured on a 3060 Ti: full model, 20 ms hop agrees with 10 ms on 99.9% of frames
        at half the cost (~6.6 s for a 228 s stem without gating)."""
        import torchcrepe

        from .f0notes import f0_to_notes

        y = resample(np.asarray(mono, np.float32), sr, 16000)
        hop = int(round(hop_sec * 16000))
        n_frames = len(y) // hop + 1
        level = 20 * np.log10(rms_curve(y, 16000, hop_sec) + 1e-7)
        level = np.pad(level, (0, max(n_frames - len(level), 0)), constant_values=-140)[:n_frames]
        active = level > level.max() - gate_db
        midi = np.full(n_frames, np.nan)
        per = np.zeros(n_frames)
        # contiguous active regions, padded and merged across short gaps
        idx = np.where(active)[0]
        if len(idx):
            runs = np.split(idx, np.where(np.diff(idx) > int(0.3 / hop_sec))[0] + 1)
            for r in runs:
                a0, a1 = max(r[0] - 5, 0), min(r[-1] + 6, n_frames)
                seg = torch.from_numpy(y[a0 * hop: a1 * hop])[None].to(self.device)
                if seg.shape[1] < 1024:
                    continue
                f0, p = torchcrepe.predict(seg, 16000, hop, 65.0, 1100.0, "full", batch_size=512,
                                           device=self.device, return_periodicity=True)
                f0, p = f0[0].float().cpu().numpy(), p[0].float().cpu().numpy()
                k = min(len(f0), a1 - a0)
                midi[a0:a0 + k] = 12 * np.log2(np.maximum(f0[:k], 1e-3) / 440.0) + 69
                per[a0:a0 + k] = p[:k]
        voiced = (per >= voiced_threshold) & active
        return f0_to_notes(midi, voiced, hop_sec, level_db=level - level.max(), onsets=onsets)

    @torch.inference_mode()
    def transcribe_piano(self, mono: np.ndarray) -> np.ndarray:
        """Transkun on a 44.1 kHz mono signal -> (N, 4) incl. pedal rows (pitch -64/-67)."""
        from ..events import from_transkun_notes

        x = torch.from_numpy(np.asarray(mono, np.float32)[:, None]).to(self.device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.transkun_bf16 and self.device.type == "cuda"):
            notes = self.transkun.transcribe(x, discardSecondHalf=False)
        return from_transkun_notes(notes)

    @torch.inference_mode()
    def beats(self, mono: np.ndarray, sr: int):
        b, d = self.beat(np.asarray(mono, np.float32), sr)
        return np.asarray(b, np.float64), np.asarray(d, np.float64)

    @torch.inference_mode()
    def chords(self, mono: np.ndarray, sr: int):
        """BTC large-vocabulary chords -> (C, 2) [start, end] array and labels."""
        import librosa

        cfg = self.btc_cfg
        y = resample(np.asarray(mono, np.float32), sr, BTC_SR)
        feat = librosa.cqt(y, sr=BTC_SR, n_bins=cfg.feature["n_bins"], bins_per_octave=cfg.feature["bins_per_octave"],
                           hop_length=cfg.feature["hop_length"])
        feat = (np.log(np.abs(feat) + 1e-6).T - self.btc_mean) / self.btc_std
        T = cfg.model["timestep"]
        n = feat.shape[0]
        feat = np.pad(feat, ((0, (-n) % T), (0, 0)))
        x = torch.from_numpy(feat.astype(np.float32)).to(self.device).view(-1, T, feat.shape[1])
        h, _ = self.btc.self_attn_layers(x)
        pred, _ = self.btc.output_layer(h)
        pred = pred.reshape(-1)[:n].cpu().numpy()
        frame = cfg.feature["hop_length"] / BTC_SR
        bounds = np.concatenate([[0], np.where(np.diff(pred) != 0)[0] + 1, [n]])
        spans = np.array([[bounds[i] * frame, bounds[i + 1] * frame] for i in range(len(bounds) - 1)])
        labels = [self.btc_labels[int(pred[bounds[i]])] for i in range(len(bounds) - 1)]
        return spans, labels


# ---------------------------------------------------------------------- cheap DSP features
KICK, SNARE, HAT = 0, 1, 2


def drum_onsets(drums: np.ndarray, sr: int = SR) -> np.ndarray:
    """Kick / snare / hi-hat onsets from the drum stem -> (K, 3) [time, kind, strength].

    One onset detector per band, each where that drum has energy the others mostly lack:
    kick < 120 Hz, snare body 180-1200 Hz (hi-hats have almost nothing there), hi-hat > 7 kHz.
    Heuristic (no drum-transcription model): toms land in 'snare', crashes in 'hat'."""
    import librosa

    y = resample(drums, sr, 22050)
    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=256))
    freqs = librosa.fft_frequencies(sr=22050, n_fft=1024)
    out = []
    bands = ((KICK, freqs < 120), (SNARE, (freqs >= 180) & (freqs <= 1200)), (HAT, freqs > 7000))
    for kind, band in bands:
        env = librosa.onset.onset_strength(S=librosa.amplitude_to_db(S[band] + 1e-6), sr=22050, hop_length=256)
        if env.max() <= 0:
            continue
        env = env / env.max()
        # delta 0.2 / 46 ms spacing / strength >= 0.2: the looser default (delta 0.1) gave
        # 2-3 kicks per beat on prototype songs (bass bleed, toms); this gives ~0.5-1 per beat
        peaks = librosa.onset.onset_detect(onset_envelope=env, sr=22050, hop_length=256, units="frames",
                                           backtrack=False, delta=0.2, wait=4, pre_max=3, post_max=3)
        peaks = peaks[env[peaks] >= 0.2]
        t = librosa.frames_to_time(peaks, sr=22050, hop_length=256)
        out += [[ti, kind, env[p]] for ti, p in zip(t, peaks)]
    out = np.array(out, np.float64).reshape(-1, 3)
    return out[np.argsort(out[:, 0], kind="stable")]


def rms_curve(x: np.ndarray, sr: int = SR, hop_sec: float = 0.05) -> np.ndarray:
    hop = int(sr * hop_sec)
    n = len(x) // hop
    if n == 0:
        return np.zeros(1)
    return np.sqrt((x[: n * hop].reshape(n, hop) ** 2).mean(1))

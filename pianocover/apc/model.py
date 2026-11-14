"""AMT-APC network (hFT-Transformer + style vector), rewritten for fast training.

Adapted from sony/hFT-Transformer `model/model_spec2midi.py` (MIT, Sony Group Corporation
2023) and misya11p/amt-apc `models/_models.py` (MIT, komiya 2024); see THIRD_PARTY_NOTICES.md.
Parameter names are unchanged, so AMT-APC's released `apc.pth` / `amt.pth` load with
strict=True. Changes, none of which alter the function computed:

* attention uses torch's scaled_dot_product_attention (flash / memory-efficient kernels):
  the [batch*frames, heads, 256, 256] attention maps are never stored, which is most of the
  original's memory, and the attention map the original returned for plotting is dropped;
* position tables are built once and broadcast instead of re-created with repeat();
* optional activation checkpointing per transformer layer (`checkpoint=True`) trades about
  30 % more compute for a much larger batch;
* the heads return logits as well (`logits=True`), so the loss can use the numerically safe
  BCE-with-logits under bf16 autocast. sigmoid(logit) equals the original output.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint as _ckpt

from . import HID_DIM, MARGIN, N_BINS, N_HEADS, N_LAYERS, N_NOTE, N_VELOCITY, NUM_FRAME, PF_DIM, SV_DIM


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout):
        super().__init__()
        self.hid_dim, self.n_heads, self.head_dim = hid_dim, n_heads, hid_dim // n_heads
        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        self.p = dropout

    def forward(self, query, key, value):
        b = query.shape[0]
        q = self.fc_q(query).view(b, -1, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.fc_k(key).view(b, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.fc_v(value).view(b, -1, self.n_heads, self.head_dim).transpose(1, 2)
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.p if self.training else 0.0)
        return self.fc_o(x.transpose(1, 2).reshape(b, -1, self.hid_dim))


class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc_2(self.dropout(torch.relu(self.fc_1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hid_dim)  # shared by both sublayers, as in the original
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        src = self.layer_norm(src + self.dropout(self.self_attention(src, src, src)))
        return self.layer_norm(src + self.dropout(self.positionwise_feedforward(src)))


class DecoderLayer_Zero(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hid_dim)
        self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, enc_src, trg):
        trg = self.layer_norm(trg + self.dropout(self.encoder_attention(trg, enc_src, enc_src)))
        return self.layer_norm(trg + self.dropout(self.positionwise_feedforward(trg)))


class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout)
        self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, enc_src, trg):
        trg = self.layer_norm(trg + self.dropout(self.self_attention(trg, trg, trg)))
        trg = self.layer_norm(trg + self.dropout(self.encoder_attention(trg, enc_src, enc_src)))
        return self.layer_norm(trg + self.dropout(self.positionwise_feedforward(trg)))


def _run(layer, use_ckpt, *args):
    if use_ckpt and torch.is_grad_enabled():
        return _ckpt(layer, *args, use_reentrant=False)
    return layer(*args)


class Encoder(nn.Module):
    def __init__(self, n_margin=MARGIN, n_frame=NUM_FRAME, n_bin=N_BINS, cnn_channel=4, cnn_kernel=5,
                 hid_dim=HID_DIM, n_layers=N_LAYERS, n_heads=N_HEADS, pf_dim=PF_DIM, dropout=0.1):
        super().__init__()
        self.n_frame, self.n_bin, self.hid_dim = n_frame, n_bin, hid_dim
        self.conv = nn.Conv2d(1, cnn_channel, kernel_size=(1, cnn_kernel))
        self.n_proc = n_margin * 2 + 1
        self.cnn_dim = cnn_channel * (self.n_proc - (cnn_kernel - 1))
        self.tok_embedding_freq = nn.Linear(self.cnn_dim, hid_dim)
        self.pos_embedding_freq = nn.Embedding(n_bin, hid_dim)
        self.layers_freq = nn.ModuleList([EncoderLayer(hid_dim, n_heads, pf_dim, dropout) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.scale_freq = math.sqrt(hid_dim)
        self.checkpoint = False

    def forward(self, spec_in):
        # spec_in [B, n_bin, margin + n_frame + margin]
        b = spec_in.shape[0]
        spec = spec_in.unfold(2, self.n_proc, 1).permute(0, 2, 1, 3)  # [B, n_frame, n_bin, n_proc]
        x = self.conv(spec.reshape(b * self.n_frame, 1, self.n_bin, self.n_proc))
        x = x.permute(0, 2, 1, 3).reshape(b * self.n_frame, self.n_bin, self.cnn_dim)
        x = self.dropout(self.tok_embedding_freq(x) * self.scale_freq + self.pos_embedding_freq.weight)
        for layer in self.layers_freq:
            x = _run(layer, self.checkpoint, x)
        return x.reshape(b, self.n_frame, self.n_bin, self.hid_dim)


class Decoder(nn.Module):
    def __init__(self, n_frame=NUM_FRAME, n_bin=N_BINS, n_note=N_NOTE, n_velocity=N_VELOCITY, hid_dim=HID_DIM,
                 n_layers=N_LAYERS, n_heads=N_HEADS, pf_dim=PF_DIM, dropout=0.1):
        super().__init__()
        self.n_note, self.n_frame, self.n_velocity, self.n_bin, self.hid_dim = n_note, n_frame, n_velocity, n_bin, hid_dim
        self.dropout = nn.Dropout(dropout)
        self.pos_embedding_freq = nn.Embedding(n_note, hid_dim)
        self.layer_zero_freq = DecoderLayer_Zero(hid_dim, n_heads, pf_dim, dropout)
        self.layers_freq = nn.ModuleList([DecoderLayer(hid_dim, n_heads, pf_dim, dropout) for _ in range(n_layers - 1)])
        self.fc_onset_freq = nn.Linear(hid_dim, 1)
        self.fc_offset_freq = nn.Linear(hid_dim, 1)
        self.fc_mpe_freq = nn.Linear(hid_dim, 1)
        self.fc_velocity_freq = nn.Linear(hid_dim, n_velocity)
        self.scale_time = math.sqrt(hid_dim)
        self.pos_embedding_time = nn.Embedding(n_frame, hid_dim)
        self.layers_time = nn.ModuleList([EncoderLayer(hid_dim, n_heads, pf_dim, dropout) for _ in range(n_layers)])
        self.fc_onset_time = nn.Linear(hid_dim, 1)
        self.fc_offset_time = nn.Linear(hid_dim, 1)
        self.fc_mpe_time = nn.Linear(hid_dim, 1)
        self.fc_velocity_time = nn.Linear(hid_dim, n_velocity)
        self.checkpoint = False

    def forward(self, enc):
        """enc [B, n_frame, n_bin, hid] -> logits; each [B, n_frame, n_note] (velocity: [..., 128]).

        Returns (onset_f, offset_f, mpe_f, velocity_f, onset_t, offset_t, mpe_t, velocity_t),
        "f" from the frequency (note-query) stage, "t" from the time stage.
        """
        b, F_, N = enc.shape[0], self.n_frame, self.n_note
        enc = enc.reshape(b * F_, self.n_bin, self.hid_dim)
        midi = self.pos_embedding_freq.weight.unsqueeze(0).expand(b * F_, N, self.hid_dim)
        midi = _run(self.layer_zero_freq, self.checkpoint, enc, midi)
        for layer in self.layers_freq:
            midi = _run(layer, self.checkpoint, enc, midi)
        onset_f = self.fc_onset_freq(midi).reshape(b, F_, N)
        offset_f = self.fc_offset_freq(midi).reshape(b, F_, N)
        mpe_f = self.fc_mpe_freq(midi).reshape(b, F_, N)
        velocity_f = self.fc_velocity_freq(midi).reshape(b, F_, N, self.n_velocity)

        t = midi.reshape(b, F_, N, self.hid_dim).permute(0, 2, 1, 3).reshape(b * N, F_, self.hid_dim)
        t = self.dropout(t * self.scale_time + self.pos_embedding_time.weight)
        for layer in self.layers_time:
            t = _run(layer, self.checkpoint, t)
        onset_t = self.fc_onset_time(t).reshape(b, N, F_).transpose(1, 2)
        offset_t = self.fc_offset_time(t).reshape(b, N, F_).transpose(1, 2)
        mpe_t = self.fc_mpe_time(t).reshape(b, N, F_).transpose(1, 2)
        velocity_t = self.fc_velocity_time(t).reshape(b, N, F_, self.n_velocity).transpose(1, 2)
        return onset_f, offset_f, mpe_f, velocity_f, onset_t, offset_t, mpe_t, velocity_t


class Spec2MIDI(nn.Module):
    """spec [B, 256 mel bins, 32 + 512 + 32 frames], style vector [B, 24] -> 8 logit maps."""

    def __init__(self, sv_dim: int = SV_DIM, dropout: float = 0.1):
        super().__init__()
        self.encoder = Encoder(dropout=dropout)
        self.decoder = Decoder(dropout=dropout)
        self.sv_dim = sv_dim
        if sv_dim:
            self.fc_sv = nn.Linear(sv_dim, HID_DIM)
            self.gate_sv = nn.Sequential(nn.Linear(HID_DIM, HID_DIM), nn.ReLU(),
                                         nn.Linear(HID_DIM, HID_DIM), nn.Sigmoid())

    def set_checkpointing(self, on: bool) -> None:
        self.encoder.checkpoint = self.decoder.checkpoint = on

    def forward(self, spec, sv=None):
        h = self.encoder(spec)
        if self.sv_dim and sv is not None:
            s = self.fc_sv(sv)[:, None, None, :]
            z = self.gate_sv(h)
            h = z * h + (1 - z) * s
        return self.decoder(h)


def load(path, device="cpu", dropout: float = 0.1) -> Spec2MIDI:
    """Spec2MIDI with weights from an AMT-APC checkpoint (apc.pth: style gate included;
    amt.pth: no style gate, which is then freshly initialised). Prefixes added by
    torch.compile / DDP are stripped."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    state = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in state.items()}
    model = Spec2MIDI(dropout=dropout)
    missing, unexpected = model.load_state_dict(state, strict=False)
    allowed_missing = {k for k in model.state_dict() if k.startswith(("fc_sv.", "gate_sv."))}
    if unexpected or set(missing) - allowed_missing:
        raise RuntimeError(f"checkpoint mismatch: missing={missing} unexpected={unexpected}")
    return model.to(device)


def save(model: nn.Module, path) -> None:
    """Save weights in apc.pth layout (loadable by AMT-APC's own code as well)."""
    state = {k.replace("_orig_mod.", "").replace("module.", ""): v.detach().cpu()
             for k, v in model.state_dict().items()}
    torch.save(state, path)

"""Music2MIDI inference without PyTorch Lightning, plus a MIDI writer that always plays back.

Model: ytinyui/music2midi (MIT code; checkpoint `epoch.799-step.119200.ckpt` from release 0.1.0,
no separate weights license). Code: third_party/music2midi (git clone, not in this repo); weights
in /data/ext/music2midi/epoch799.ckpt. T5 (6+6 layers, d_model 384, 31 M parameters) over a log-mel
spectrogram of 3 s audio segments, each decoded independently into note tokens (50 ms grid).

Differences to the authors' `Music2MIDI.generate`:
  - no Lightning: the T5Transformer module is built from config.yaml and the checkpoint's
    state_dict is loaded into it directly;
  - optional bf16 weights (`dtype`), the mel spectrogram stays fp32;
  - segments of several songs and conditionings go through `generate` in one batch;
  - `sr`: the authors' generate() loads audio at config.model.sample_rate (16 kHz) while their
    training dataset loads it at config.dataset.sample_rate (22.05 kHz); both are selectable;
  - own token decoder (the authors' uses `np.float_`, removed in NumPy 2) with the same semantics,
    followed by `clean_notes` (see there) instead of pretty_midi's remove_invalid_notes only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

M2M_CODE = Path(__file__).resolve().parents[1] / "third_party" / "music2midi"
PAD, BOS, EOS, ONSET, OFFSET = 0, 1, 2, 3, 4
GENRES = ["electronic", "pop", "rock", "soundtrack", "world_music", "classical"]
DIFFICULTIES = ["beginner", "intermediate", "advanced"]


class LoopStop:
    """Stops a row once its last 2k tokens (k = 1..max_period) are pitch tokens only (no time,
    ONSET/OFFSET token in between) and the last k repeat the k before: the decoder loop seen on
    silent segments (one chord emitted again and again at the same instant). A valid sequence
    never names the same pitch twice in one ONSET/OFFSET group. The row gets no EOS, so decode()
    treats it as degenerate, as it would on reaching max_length."""

    def __init__(self, pitch_off, time_off, max_period=16):
        self.lo, self.hi, self.maxk = pitch_off, time_off, max_period

    def __call__(self, input_ids, scores, **kw):
        L = input_ids.shape[1]
        done = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        for k in range(1, min(self.maxk, L // 2) + 1):
            w = input_ids[:, -2 * k:]
            pitch = ((w >= self.lo) & (w < self.hi)).all(1)
            done |= pitch & (w[:, :k] == w[:, k:]).all(1)
        return done


class Music2MIDIFast:
    def __init__(self, ckpt, device="cuda", dtype=torch.bfloat16, config=None):
        if str(M2M_CODE) not in sys.path:
            sys.path.insert(0, str(M2M_CODE))
        from omegaconf import OmegaConf
        from music2midi.transformer import T5Transformer

        config = config or str(M2M_CODE / "config.yaml")
        self.cfg = OmegaConf.load(config)
        net = T5Transformer(config)
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
        sd = {k.removeprefix("model."): v for k, v in sd.items()}
        missing, unexpected = net.load_state_dict(sd, strict=False)
        # mel filterbank / window buffers are not in the checkpoint; nothing else may be missing
        bad = [k for k in missing if not k.startswith("spectrogram.")] + list(unexpected)
        assert not bad, bad
        self.net = net.to(device).eval()
        self.net.transformer.to(dtype)
        self.net.conditioning.to(dtype)
        self.dtype = dtype
        self.dev = torch.device(device)
        t = self.cfg.tokenizer
        self.step = t.midi_quantize_ms / 1000
        self.pitch_off = t.vocab_size.special
        self.time_off = self.pitch_off + t.vocab_size.pitch
        self.velocity = int(t.default_velocity)
        self.seg_sec = float(self.cfg.dataset.segment_duration)

    def load_audio(self, path, sr):
        import librosa

        y, _ = librosa.load(str(path), sr=sr)
        return y

    def segments(self, y, sr):
        n = int(sr * self.seg_sec)
        y = np.pad(y, (0, -len(y) % n))
        return torch.from_numpy(y).view(-1, n)

    @torch.no_grad()
    def generate_tokens(self, wav, cond, batch_size=128, max_length=512, loop_stop=True):
        """max_length 512 (authors: 1024): normal segments end by ~250 tokens (training labels have
        at most 90 notes per segment); longer ones are decoder loops, and every row of the batch
        waits for the longest one.
        wav (N, samples) float32, cond (N, 2) long -> list of N token arrays."""
        from music2midi.input import ModelInputs

        out = []
        for i in range(0, len(wav), batch_size):
            w = wav[i:i + batch_size].to(self.dev)
            c = cond[i:i + batch_size].to(self.dev)
            mel = self.net.spectrogram(w).to(self.dtype)
            emb = self.net.conditioning(mel, c)
            from transformers import StoppingCriteriaList

            toks = self.net.transformer.generate(inputs_embeds=emb, max_length=max_length,
                                                 do_sample=False, num_beams=1,
                                                 stopping_criteria=StoppingCriteriaList(
                                                     [LoopStop(self.pitch_off, self.time_off)] if loop_stop else []))
            out += [t.cpu().numpy() for t in toks]
        return out

    def decode(self, tokens_list):
        """Tokens of consecutive segments -> (notes (n, 4) [onset, offset, pitch, velocity] in
        seconds, stats). Same rules as MidiTokenizer._decode_tokens: a time token sets the current
        time (segment start + index), ONSET/OFFSET switch the mode, a pitch token opens a note or
        closes every open note of that pitch. Notes left open are returned with offset -1."""
        notes, stats = [], {"segments": len(tokens_list), "hit_max_length": 0, "degenerate_dropped": 0,
                            "max_tokens_eos": 0}
        seg_steps = round(self.seg_sec / self.step)
        for si, toks in enumerate(tokens_list):
            base = si * seg_steps
            n0 = len(notes)
            if EOS not in toks:
                stats["hit_max_length"] += 1
            else:
                stats["max_tokens_eos"] = max(stats["max_tokens_eos"], int(np.argmax(toks == EOS)) + 1)
            t_idx, mode, open_ = -1, -1, {}
            for tok in toks:
                tok = int(tok)
                if tok == EOS:
                    break
                if tok in (BOS, PAD):
                    continue
                if tok == ONSET:
                    mode = 1
                elif tok == OFFSET:
                    mode = 0
                elif tok >= self.time_off:
                    t_idx, mode = base + tok - self.time_off, -1
                elif tok >= self.pitch_off and t_idx >= 0 and mode >= 0:
                    p = tok - self.pitch_off
                    if mode == 1:
                        notes.append([t_idx, -1, p, self.velocity])
                        open_.setdefault(p, []).append(len(notes) - 1)
                    else:
                        for j in open_.pop(p, []):
                            if notes[j][0] < t_idx:
                                notes[j][1] = t_idx
                            else:  # authors: only notes with onset < offset time are closed
                                open_.setdefault(p, []).append(j)
            if EOS not in toks and open_:
                # no EOS = the decoder looped until max_length (seen on silent / padded tail
                # segments: the same chord repeated ~1000 times at one instant). The authors'
                # decoder drops these notes as unclosed; do the same for this segment only.
                dead = {j for js in open_.values() for j in js}
                stats["degenerate_dropped"] += len(dead)
                notes[n0:] = [n for k, n in enumerate(notes[n0:], n0) if k not in dead]
        a = np.array(notes, dtype=np.float64).reshape(-1, 4)
        m = a[:, 1] >= 0
        a[m, :2] *= self.step
        a[~m, 0] *= self.step
        stats["notes_raw"] = len(a)
        stats["notes_unclosed"] = int((~m).sum())
        return a, stats


def clean_notes(notes, seg_sec=3.0, keep_unclosed=True, lo=21, hi=108, min_dur=0.05):
    """Make the note list safe for any MIDI player / synth:
      - unclosed notes (the authors drop them): kept, ending at the end of their 3 s segment
        (or dropped if keep_unclosed=False);
      - pitches outside the 88 keys moved by octaves into A0..C8;
      - same-pitch overlaps (two note-ons of one key before its note-off, which many players cut
        or hang on): the earlier note ends at the later onset; exact duplicates are removed;
      - every note at least `min_dur` long; velocity 1..127.
    Returns (notes sorted by onset, stats)."""
    a = notes.copy()
    st = {}
    open_ = a[:, 1] < 0
    if keep_unclosed:
        a[open_, 1] = (np.floor(a[open_, 0] / seg_sec + 1e-9) + 1) * seg_sec
    else:
        a = a[~open_]
    st["unclosed_" + ("kept" if keep_unclosed else "dropped")] = int(open_.sum())
    p = a[:, 2].copy()
    moved = (p < lo) | (p > hi)
    while (p < lo).any():
        p[p < lo] += 12
    while (p > hi).any():
        p[p > hi] -= 12
    a[:, 2] = p
    st["pitch_moved_into_range"] = int(moved.sum())
    a[:, 1] = np.maximum(a[:, 1], a[:, 0] + min_dur)
    a[:, 3] = np.clip(a[:, 3], 1, 127)
    a = a[np.lexsort((a[:, 0], a[:, 2]))]  # by pitch, then onset
    keep = np.ones(len(a), bool)
    trimmed = dups = 0
    for i in range(1, len(a)):
        j = i - 1
        while j >= 0 and not keep[j]:
            j -= 1
        if j < 0 or a[j, 2] != a[i, 2]:
            continue
        if abs(a[j, 0] - a[i, 0]) < 1e-6:  # same key struck twice at once: keep the longer
            a[j, 1] = max(a[j, 1], a[i, 1])
            keep[i] = False
            dups += 1
        elif a[j, 1] > a[i, 0]:
            a[j, 1] = a[i, 0]
            trimmed += 1
    a = a[keep]
    a = a[a[:, 1] - a[:, 0] >= 0.01]
    st["duplicates_removed"] = dups
    st["overlaps_trimmed"] = trimmed
    st["notes_final"] = len(a)
    return a[np.argsort(a[:, 0], kind="stable")], st


def write_midi(notes, path, name="Piano", extra_tracks=()):
    """Plain type-1 SMF: tempo track + one piano track (channel 0, program 0), 480 PPQ at 120 BPM,
    so 1 tick = 1/960 s. Written with mido so note-offs come before note-ons at equal ticks.
    Rows with pitch -64 / -67 (Transkun convention) are sustain / soft pedal intervals (CC 64 / 67);
    at one tick the order is note-offs, pedal ups, pedal downs, note-ons.
    extra_tracks: [(name, notes), ...] written as further piano tracks on channels 1, 2, ...
    (so a player can mute them separately)."""
    import mido

    mf = mido.MidiFile(type=1, ticks_per_beat=480)
    t0 = mido.MidiTrack()
    t0.append(mido.MetaMessage("set_tempo", tempo=500000, time=0))
    t0.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))
    t0.append(mido.MetaMessage("end_of_track", time=0))
    mf.tracks.append(t0)
    for ch, (tname, tnotes) in enumerate([(name, notes), *extra_tracks]):
        mf.tracks.append(_track(tname, tnotes, ch))
    mf.save(str(path))


def _track(name, notes, ch):
    import mido

    tr = mido.MidiTrack()
    tr.append(mido.MetaMessage("track_name", name=name, time=0))
    tr.append(mido.Message("program_change", program=0, channel=ch, time=0))
    ev = []
    for on, off, p, v in notes:
        a, b = int(round(on * 960)), int(round(off * 960))
        b = max(b, a + 1)
        if p > 0:
            ev.append((a, 3, int(p), int(v)))
            ev.append((b, 0, int(p), 0))
        else:
            ev.append((a, 2, int(-p), 127))
            ev.append((b, 1, int(-p), 0))
    ev.sort(key=lambda e: (e[0], e[1]))  # same tick: note off, pedal up, pedal down, note on
    now = 0
    for tick, kind, p, v in ev:
        if kind in (1, 2):
            msg = mido.Message("control_change", control=p, value=v, channel=ch, time=tick - now)
        else:
            msg = mido.Message("note_on" if kind else "note_off", note=p, velocity=v if kind else 0,
                               channel=ch, time=tick - now)
        tr.append(msg)
        now = tick
    tr.append(mido.MetaMessage("end_of_track", time=960))
    return tr


def check_midi(path):
    """Re-read the file and check every note_on is closed before the same key (per channel) is
    struck again."""
    import mido

    mf = mido.MidiFile(str(path))
    held, n, errs = {}, 0, 0
    for msg in mido.merge_tracks(mf.tracks):
        if msg.type == "note_on" and msg.velocity > 0:
            if held.get((msg.channel, msg.note)):
                errs += 1
            held[(msg.channel, msg.note)] = True
            n += 1
        elif msg.type in ("note_off", "note_on"):
            if not held.get((msg.channel, msg.note)):
                errs += 1
            held[(msg.channel, msg.note)] = False
    errs += sum(held.values())
    return {"notes": n, "errors": errs, "seconds": round(mf.length, 2)}

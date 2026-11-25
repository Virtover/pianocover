"""Fast batched inference for Pianist Transformer (yhj137/pianist-transformer-rendering, Apache-2.0).

The authors' `batch_performance_render` (third_party/PianistTransformer/src/model/generate.py)
uses HF `generate` with a Python logits processor that loops over the batch at every token, so
it is CPU-bound: ~18 ms per token on a 3060 Ti whatever the batch. Here the same sampling runs in
an own decode loop:
  - encoder (HF module) once per window; cross-attention K/V precomputed;
  - decoder step re-implemented from the model's weights (T5Gemma: RMSNorm (1 + w), RoPE, GQA,
    attention softcap 50, final logit softcap 30) on a static KV cache, captured as one CUDA graph;
  - token constraints (every 8th token forced to the score's pitch token, the others limited to
    their field's id range) and temperature / top-p sampling on the GPU, no per-row Python;
  - pieces are batched by window round: window r of every piece depends only on its own window
    r - 1, so all pieces advance together; windows, overlap and stitching as in the original
    (4096 tokens, 50 % overlap, the last 20 % of the previous window regenerated);
  - optionally (windows="parallel" / "two_pass", see render_ids) the windows of one piece run
    in the same batch too: a long piece takes one or two windows of time instead of one window
    per 256 notes, at the price of less context across window boundaries.

Differences to the original, both off by default (`fix_prefix`, the padding mask is always fixed):
  - original feeds the previous window's tokens shifted by one (its `generate` output starts with
    BOS); `fix_prefix=True` feeds them aligned;
  - original batches pass the token-level padding mask to the cross-attention, whose keys are the
    8x compressed encoder states, so short pieces in a batch attend to padding. Here each piece
    gets its own compressed mask and its own window schedule, so a batch gives the same
    distribution as rendering each piece alone.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

W = 4096  # window (tokens), as in the original
OVERLAP = 0.5


def slide_window(total_len, window_len=W, overlap_ratio=OVERLAP):
    if total_len <= window_len:
        return [(0, total_len)]
    window_len = window_len // 8 * 8
    out, start = [], 0
    while start + window_len <= total_len:
        out.append((start, start + window_len))
        start += int(window_len * (1 - overlap_ratio)) // 8 * 8
    if out[-1][1] != total_len:
        out.append((start, total_len))
    return out


def _rms(x, w, eps):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * (1.0 + w.float())).type_as(x)


def _rot(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


class FastRenderer:
    def __init__(self, model, device="cuda", fix_prefix=False):
        self.m = model.eval()
        self.dev = torch.device(device)
        self.cfg = model.config
        dec = model.model.decoder
        self.dec = dec
        dc = dec.config
        self.nh, self.nkv, self.hd = dc.num_attention_heads, dc.num_key_value_heads, dc.head_dim
        self.rep = self.nh // self.nkv
        self.scale = dc.query_pre_attn_scalar ** -0.5
        self.softcap = dc.attn_logit_softcapping if model.config._attn_implementation == "eager" else None
        self.kv_len = None
        self.fcap = dc.final_logit_softcapping
        self.eps = dc.rms_norm_eps
        self.sliding = [dc.sliding_window if t == "sliding_attention" else None for t in dc.layer_types]
        self.dtype = dec.embed_tokens.weight.dtype
        self.norm_c = torch.tensor(dc.hidden_size ** 0.5, dtype=self.dtype, device=self.dev)
        self.fix_prefix = fix_prefix
        # RoPE table (float32 like HF, then cast to the hidden dtype)
        inv = dec.rotary_emb.inv_freq.float().to(self.dev)
        pos = torch.arange(W + 8, device=self.dev).float()
        emb = torch.cat([torch.outer(pos, inv)] * 2, dim=-1)
        s = getattr(dec.rotary_emb, "attention_scaling", 1.0)
        self.cos = (emb.cos() * s).to(self.dtype)
        self.sin = (emb.sin() * s).to(self.dtype)
        # field constraints: additive mask per position mod 8
        V = self.cfg.vocab_size
        mask = torch.full((8, V), float("-inf"), device=self.dev)
        for k, (a, b) in enumerate(self.cfg.valid_id_range):
            mask[k, a:b] = 0.0
        self.field_mask = mask
        self.graphs = {}

    # ------------------------------------------------------------------ decoder math
    def _self_attn(self, layer, li, h, pos0, T, kc, vc, valid_len=None):
        """h: [B, T, D] at positions pos0..pos0+T-1 (pos0 python int or 0-dim tensor for T=1)."""
        B = h.shape[0]
        a = layer.self_attn
        q = a.q_proj(h).view(B, T, self.nh, self.hd).transpose(1, 2)
        k = a.k_proj(h).view(B, T, self.nkv, self.hd).transpose(1, 2)
        v = a.v_proj(h).view(B, T, self.nkv, self.hd).transpose(1, 2)
        step = torch.is_tensor(pos0)  # one token at a tensor position (graph-safe) vs. a block at an int
        if step:
            cos = self.cos.index_select(0, pos0.view(1))[None, None]
            sin = self.sin.index_select(0, pos0.view(1))[None, None]
        else:
            cos = self.cos[pos0: pos0 + T][None, None]
            sin = self.sin[pos0: pos0 + T][None, None]
        q = q * cos + _rot(q) * sin
        k = k * cos + _rot(k) * sin
        if step:
            kc.index_copy_(2, pos0.view(1), k)
            vc.index_copy_(2, pos0.view(1), v)
            n = self.kv_len or kc.shape[2]  # static cache length bucket of this graph
            keys, vals = kc[:, :, :n], vc[:, :, :n]
            kpos = torch.arange(n, device=self.dev)
            allowed = kpos <= pos0
            if self.sliding[li] is not None:
                allowed = allowed & (pos0 - kpos < self.sliding[li])
            amask = allowed[None, None, None, :]
        else:
            kc[:, :, pos0: pos0 + T] = k
            vc[:, :, pos0: pos0 + T] = v
            keys, vals = kc[:, :, : pos0 + T], vc[:, :, : pos0 + T]
            qp = torch.arange(pos0, pos0 + T, device=self.dev)[:, None]
            kp = torch.arange(pos0 + T, device=self.dev)[None, :]
            allowed = kp <= qp
            if self.sliding[li] is not None:
                allowed = allowed & (qp - kp < self.sliding[li])
            amask = allowed[None, None]
        out = self._attend(q, keys, vals, amask)
        return a.o_proj(out.transpose(1, 2).reshape(B, T, -1))

    def _attend(self, q, k, v, amask):
        """amask: bool, True = attend. HF loads this model with SDPA, which ignores the attention
        softcap of the config; the authors' inference therefore runs without it, and so does this."""
        if self.softcap is None and q.shape[2] == 1:
            # one query: grouped heads as a [rep x S] product per KV head, reads the cache once
            B, H, _, D = q.shape
            qg = q.view(B, self.nkv, self.rep, D)
            w = torch.matmul(qg, k.transpose(-1, -2)) * self.scale  # [B, nkv, rep, S]
            w = w.float().masked_fill(~amask.view(amask.shape[0], 1, 1, -1), float("-inf"))
            w = torch.softmax(w, dim=-1).to(q.dtype)
            return torch.matmul(w, v).view(B, H, 1, D)
        if self.softcap is None:
            return F.scaled_dot_product_attention(q, k, v, attn_mask=amask, scale=self.scale, enable_gqa=True)
        k = k.repeat_interleave(self.rep, dim=1)
        v = v.repeat_interleave(self.rep, dim=1)
        w = torch.matmul(q, k.transpose(2, 3)) * self.scale
        w = torch.tanh(w / self.softcap) * self.softcap
        w = F.softmax(w.float().masked_fill(~amask, float("-inf")), dim=-1).to(q.dtype)
        return torch.matmul(w, v)

    def _cross(self, layer, h, ck, cv, cmask):
        B, T, _ = h.shape
        a = layer.cross_attn
        q = a.q_proj(h).view(B, T, self.nh, self.hd).transpose(1, 2)
        out = self._attend(q, ck, cv, cmask)
        return a.o_proj(out.transpose(1, 2).reshape(B, T, -1))

    def _decode(self, tok, pos0, T, st):
        """tok [B, T] -> logits [B, T, V] (float32, softcapped), updates st['kc'] / st['vc']."""
        x = self.dec.embed_tokens(tok) * self.norm_c
        for li, L in enumerate(self.dec.layers):
            r = x
            h = _rms(x, L.pre_self_attn_layernorm.weight, self.eps)
            h = self._self_attn(L, li, h, pos0, T, st["kc"][li], st["vc"][li])
            x = r + _rms(h, L.post_self_attn_layernorm.weight, self.eps)
            r = x
            h = _rms(x, L.pre_cross_attn_layernorm.weight, self.eps)
            h = self._cross(L, h, st["ck"][li], st["cv"][li], st["cmask"])
            x = r + _rms(h, L.post_cross_attn_layernorm.weight, self.eps)
            r = x
            h = _rms(x, L.pre_feedforward_layernorm.weight, self.eps)
            h = L.mlp(h)
            x = r + _rms(h, L.post_feedforward_layernorm.weight, self.eps)
        x = _rms(x, self.dec.norm.weight, self.eps)
        logits = self.m.lm_head(x)
        if self.fcap is not None:  # in the hidden dtype like the model's forward, then float32 like generate
            logits = torch.tanh(logits / self.fcap) * self.fcap
        return logits.float()

    # ------------------------------------------------------------------ one window round
    def _encode(self, ids, lens):
        """ids [B, W] (padded), lens [B] real tokens -> cross K/V per layer + additive cross mask."""
        B = ids.shape[0]
        am = (torch.arange(ids.shape[1], device=self.dev)[None] < lens[:, None]).long()
        enc = self.m.model.encoder(input_ids=ids, attention_mask=am).last_hidden_state
        cm = am.view(B, -1, 8).any(-1)  # compressed, one per encoder state
        cmask = cm[:, None, None, :]
        ck, cv = [], []
        for L in self.dec.layers:
            a = L.cross_attn
            ck.append(a.k_proj(enc).view(B, -1, self.nkv, self.hd).transpose(1, 2).contiguous())
            cv.append(a.v_proj(enc).view(B, -1, self.nkv, self.hd).transpose(1, 2).contiguous())
        return ck, cv, cmask

    def _sample(self, logits, pos, forced, temperature, top_p, gen):
        """logits [B, V] for window position `pos` (tensor); forced [B, W] score tokens."""
        lg = logits + self.field_mask.index_select(0, (pos % 8).view(1))
        lg = lg / temperature
        sl, si = torch.sort(lg, descending=True, dim=-1)
        cp = torch.softmax(sl, dim=-1).cumsum(-1)
        drop = (cp - torch.softmax(sl, dim=-1)) > top_p  # keep the smallest set reaching top_p (>= 1 token)
        sl = sl.masked_fill(drop, float("-inf"))
        probs = torch.softmax(sl, dim=-1)
        pick = torch.multinomial(probs, 1, generator=gen)
        tok = si.gather(-1, pick)
        f = forced.index_select(1, pos.view(1))
        return torch.where((pos % 8) == 0, f, tok)

    @torch.no_grad()
    def render_ids(self, ids_list, temperature=1.0, top_p=0.95, seed=0, use_graph=True, max_batch=32, progress=None,
                   windows="sequential"):
        """ids_list: score token lists (multiples of 8) -> performance token lists, same lengths.

        windows: how the windows of one piece are scheduled
          sequential  as the original: window r starts from the end of window r - 1 (rounds = windows)
          parallel    every window independently in one batch, consecutive windows cut at the most
                      similar note of their 50 % overlap (1 round; no context across windows)
          two_pass    even windows independently, then the odd ones continuing the even window before
                      them (as the original) and cut into the even window after them (2 rounds)
        """
        gen = torch.Generator(device=self.dev)
        gen.manual_seed(seed)

        def run(w, p):
            return self._run_windows(w, p, temperature, top_p, gen, use_graph, max_batch, progress)

        scheds = [slide_window(len(x)) for x in ids_list]

        def win(i, r):
            return ids_list[i][scheds[i][r][0]: scheds[i][r][1]]

        def pre(i, r):
            return self._prefix(outs[i][r - 1], scheds[i][r], scheds[i][r - 1])

        outs = [[None] * len(s) for s in scheds]
        if windows == "sequential":
            for r in range(max(len(s) for s in scheds)):
                act = [i for i, s in enumerate(scheds) if len(s) > r]
                for i, o in zip(act, run([win(i, r) for i in act], [pre(i, r) for i in act] if r else None)):
                    outs[i][r] = o
        elif windows in ("parallel", "two_pass"):
            step = 2 if windows == "two_pass" else 1
            jobs = [(i, r) for i, s in enumerate(scheds) for r in range(0, len(s), step)]
            for (i, r), o in zip(jobs, run([win(i, r) for i, r in jobs], None)):
                outs[i][r] = o
            jobs = [(i, r) for i, s in enumerate(scheds) for r in range(1, len(s), 2)] if step == 2 else []
            if jobs:
                for (i, r), o in zip(jobs, run([win(i, r) for i, r in jobs], [pre(i, r) for i, r in jobs])):
                    outs[i][r] = o
        else:
            raise ValueError(windows)
        res = []
        for i, sc in enumerate(scheds):
            x = outs[i][0]
            for r in range(1, len(sc)):
                (s, e), (ls, le) = sc[r], sc[r - 1]
                if windows == "parallel" or (windows == "two_pass" and r % 2 == 0):
                    c = self._cut(x, outs[i][r], s, le)
                    x = x[:c] + outs[i][r][c - s:]
                else:  # the original stitching: this window continued the previous one
                    length = int(((le - ls) - (s - ls)) * 0.2)
                    x = x[:-length] + outs[i][r][-(e - le + length):]
            res.append(x[: len(ids_list[i])])
        return res

    def _prefix(self, prev, cur, last):
        """decoder context for window `cur` = (s, e) from the output `prev` of window `last`."""
        (s, _), (ls, le) = cur, last
        length = int(((le - ls) - (s - ls)) * 0.2)
        m = (le - ls - length) - (s - ls)
        # the original takes output[:, s-ls : le-ls-length] of a BOS-prefixed output
        return prev[s - ls: s - ls + m] if self.fix_prefix else ([self.cfg.bos_token_id] + prev)[s - ls: s - ls + m]

    def _cut(self, a, b, s, le, smooth=16, lo=0.5, hi=0.9):
        """Cut position (global token index, multiple of 8) for `a` (covers [0, le)) followed by `b`
        (window from s): the note in [lo, hi] of the overlap [s, le) where both renders agree most in
        velocity, inter-onset interval and duration, averaged over `smooth` notes. Late in the
        overlap, because `b` starts without context."""
        n = (le - s) // 8
        A = torch.tensor(a[s: s + 8 * n], dtype=torch.float).view(n, 8)
        B = torch.tensor(b[: 8 * n], dtype=torch.float).view(n, 8)
        ts = self.cfg.valid_id_range[1][0]

        def lt(x):
            return torch.log1p((x - ts).clamp(min=0))

        d = (A[:, 2] - B[:, 2]).abs() / 10 + (lt(A[:, 1]) - lt(B[:, 1])).abs() + 0.5 * (lt(A[:, 3]) - lt(B[:, 3])).abs()
        k = torch.ones(1, 1, smooth) / smooth
        d = F.conv1d(F.pad(d[None, None], (smooth // 2, smooth - 1 - smooth // 2), mode="replicate"), k)[0, 0]
        i0, i1 = int(n * lo), max(int(n * lo) + 1, int(n * hi))
        return s + 8 * (i0 + int(d[i0:i1].argmin()))

    def _run_windows(self, wins, prefixes, temperature, top_p, gen, use_graph, max_batch, progress):
        """wins: score-token lists (len <= W); prefixes: None or equal-length decoder contexts (window
        coordinates, without BOS) -> one sampled token list per window (len of the window)."""
        # Chunks are formed after sorting by (prefix length, window length), so a chunk holds windows
        # of similar length and _run_chunk can stop at its longest window instead of always W.
        # Results come back in the caller's order. (Sampling draws from one generator, so a
        # different chunking gives different samples from the same distribution; with greedy
        # decoding the output is identical.)
        plen = [len(p) for p in prefixes] if prefixes else [0] * len(wins)
        order = sorted(range(len(wins)), key=lambda i: (plen[i], -len(wins[i])))
        out = [None] * len(wins)
        c = 0
        while c < len(order):
            idx = order[c: c + max_batch]
            idx = [i for i in idx if plen[i] == plen[idx[0]]]  # one prefix length per chunk
            res = self._run_chunk([wins[i] for i in idx], prefixes and [prefixes[i] for i in idx],
                                  temperature, top_p, gen, use_graph, progress)
            for i, r in zip(idx, res):
                out[i] = r
            c += len(idx)
        return out

    def _run_chunk(self, wins, prefixes, temperature, top_p, gen, use_graph, progress):
        dev = self.dev
        B = len(wins)
        bos = self.cfg.bos_token_id
        ids = torch.zeros(B, W, dtype=torch.long, device=dev)
        lens = torch.zeros(B, dtype=torch.long, device=dev)
        for b, w in enumerate(wins):
            ids[b, : len(w)] = torch.tensor(w, device=dev)
            lens[b] = len(w)
        ck, cv, cmask = self._encode(ids, lens)
        st = {"ck": ck, "cv": cv, "cmask": cmask,
              "kc": [torch.zeros(B, self.nkv, W, self.hd, dtype=self.dtype, device=dev) for _ in self.dec.layers],
              "vc": [torch.zeros(B, self.nkv, W, self.hd, dtype=self.dtype, device=dev) for _ in self.dec.layers]}
        # decoder prefix: BOS (+ the overlap from the previous window)
        m = len(prefixes[0]) if prefixes else 0
        assert not prefixes or all(len(p) == m for p in prefixes)
        prefix = torch.tensor([[bos] + (prefixes[b] if prefixes else []) for b in range(B)], dtype=torch.long, device=dev)
        # prefill (eager, in row chunks to bound the T x T attention memory)
        logits = torch.empty(B, self.cfg.vocab_size, device=dev)
        T = prefix.shape[1]
        for c in range(0, B, 8):
            sub = {k: (v[c: c + 8] if not isinstance(v, list) else [t[c: c + 8] for t in v]) for k, v in st.items()}
            logits[c: c + 8] = self._decode(prefix[c: c + 8], 0, T, sub)[:, -1]
        gen_tok = torch.empty(B, W, dtype=torch.long, device=dev)
        if m:
            gen_tok[:, :m] = prefix[:, 1:]
        # window token y_q is sampled from the logits of decoder position q (BOS at 0) and then
        # fed at decoder position q + 1
        pos = torch.tensor(m, device=dev)
        feed = torch.tensor(m + 1, device=dev)
        step_fn = self._graph_step(B, st) if (use_graph and dev.type == "cuda") else None
        L = max(len(w) for w in wins)  # tokens past a window's length are discarded anyway
        for q in range(m, L):
            pos.fill_(q)
            tok = self._sample(logits, pos, ids, temperature, top_p, gen)
            gen_tok[:, q] = tok[:, 0]
            if q == L - 1:
                break
            feed.fill_(q + 1)
            if step_fn is not None:
                logits = step_fn(tok, feed, q + 1)
            else:
                logits = self._decode(tok, feed, 1, st)[:, -1]
            if progress and q % 512 == 0:
                progress(q, W)
        g = gen_tok.cpu().tolist()
        return [g[b][: len(w)] for b, w in enumerate(wins)]

    def _graph_step(self, B, st, bucket=512):
        """One decoder step as CUDA graphs on this round's static buffers, one graph per cache-length
        bucket (attention reads only the first n positions). Captured lazily, at the step that first
        needs a bucket: its warm-up computes exactly that step, so the cache stays correct."""
        tok_in = torch.zeros(B, 1, dtype=torch.long, device=self.dev)
        pos_in = torch.zeros((), dtype=torch.long, device=self.dev)
        graphs = {}

        def run():
            return self._decode(tok_in, pos_in, 1, st)[:, -1]

        def step(tok, pos, p):
            """pos: position tensor, p: the same as int (no device sync)"""
            tok_in.copy_(tok)
            pos_in.copy_(pos)
            n = min(W, (p // bucket + 1) * bucket)
            self.kv_len = n
            if n not in graphs:
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    run()
                torch.cuda.current_stream().wait_stream(s)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    out = run()
                graphs[n] = (g, out)
            self.kv_len = None
            g, out = graphs[n]
            g.replay()
            return out

        return step


def render_midis(renderer, midis, temperature=1.0, top_p=0.95, seed=0, **kw):
    """miditoolkit score MidiFiles -> performance MidiFiles (the authors' token conversion)."""
    from src.utils.midi import ids_to_midi, midi_to_ids

    ids = [midi_to_ids(renderer.cfg, m) for m in midis]
    toks = renderer.render_ids(ids, temperature=temperature, top_p=top_p, seed=seed, **kw)
    return [ids_to_midi(renderer.cfg, t, ref=i) for t, i in zip(toks, ids)]

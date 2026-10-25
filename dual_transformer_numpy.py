"""
Dual Transformer (NumPy reference implementation, no torch)
===========================================================
Features:
- Hand-written forward and backward passes (reverse-mode, explicit per-op derivatives)
- Pre-norm blocks, RoPE, causal self-attention, optional RMSNorm (else LayerNorm)
- Merge modes:
    "single"  - one branch only (baseline)
    "average" - two branches, mean of outputs
    "concat"  - two branches, concatenated outputs (larger LM head)
- Consolidated options: grouped-query attention (n_kv_heads < n_heads), RoPE or
  learned position embeddings (pos_mode), GELU or SwiGLU feed-forward (ff_mode),
  GPT-2 style N(0, 0.02) initialization (init_style="gpt2"; "legacy" restores
  the original initialization)
- Byte-level UTF-8 tokenizer (no external vocabulary files)
- AdamW with decoupled weight decay and global-norm gradient clipping
- Next-token training, greedy/temperature/top-k/top-p generation, KV-cached generation
- Optional label smoothing, warmup + cosine LR schedule, gradient accumulation, CSV logs
- Toy tasks (arith / copy) for quick sanity checks
- Borrowed from transformer_core.py: final norm before the output head, GPT-2 style scaled
  residual init, weight decay on matrices only, optional tied embeddings, optional char-level
  tokenizer (better for Chinese text than bytes: about 3x shorter sequences)
- Not included: dropout, post-norm, learned position embeddings (they existed only in the
  older train_dual_transformer.py and are not part of this file)
- Checkpoints: .npz files holding parameters, optimizer state, step, RNG state.
  With a validation split, the best-validation weights go to --checkpoint and the
  latest state (used for resume) goes to <stem>_last<suffix>.

Differences from the torch version
- GELU uses the tanh approximation (torch's default is the exact erf form).
- Initialization follows the same distributions as torch's defaults, but the random
  streams differ, so the two backends do not produce identical weights.
- The model runs in float32 by default; gradient checks run in float64.

Requirements: Python 3.10+, numpy
Run smoke tests: python dual_transformer_numpy.py --test
Train demo: python dual_transformer_numpy.py --train-text corpus.txt --steps 500
Toy task:   python dual_transformer_numpy.py --task arith --seq-len 8 --steps 200 --lr 1e-2
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import math
from pathlib import Path
from typing import Optional

import numpy as np

# Byte-level tokenizer: IDs 0..255 are UTF-8 bytes; 256=BOS, 257=EOS.
BOS_ID = 256
EOS_ID = 257
VOCAB_SIZE = 258
MERGE_METHODS = ("single", "average", "concat")
LN_EPS = 1e-5
ROPE_THETA = 10000.0
_GELU_C = math.sqrt(2.0 / math.pi)


class ByteTokenizer:
    """Lossless UTF-8 byte tokenizer. Simple and robust, but not compression-efficient."""
    vocab_size = VOCAB_SIZE
    bos_id = BOS_ID
    eos_id = EOS_ID

    def to_meta(self):
        return {"type": "byte"}

    def encode(self, text: str, add_bos: bool = False,
               add_eos: bool = False) -> list[int]:
        ids = list(text.encode("utf-8"))
        if add_bos:
            ids.insert(0, BOS_ID)
        if add_eos:
            ids.append(EOS_ID)
        return ids

    def decode(self, ids, skip_special: bool = True) -> str:
        raw = bytearray()
        for item in ids:
            i = int(item)
            if i in (BOS_ID, EOS_ID) and skip_special:
                continue
            if 0 <= i <= 255:
                raw.append(i)
            elif not skip_special:
                raw.extend({BOS_ID: b"<BOS>", EOS_ID: b"<EOS>"}.get(i, f"<tok:{i}>".encode()))
        return raw.decode("utf-8", errors="replace")


class CharTokenizer:
    """Character-level tokenizer built from the training text.

    IDs: 0 = BOS, 1 = EOS, 2 = UNK, 3.. = characters in sorted order.
    The character table is stored in the checkpoint, so generation can be reproduced after resume.
    """
    BOS, EOS, UNK = 0, 1, 2

    def __init__(self, chars):
        self.chars = list(chars)
        self.stoi = {c: i + 3 for i, c in enumerate(self.chars)}
        self.itos = {i + 3: c for i, c in enumerate(self.chars)}
        self.vocab_size = 3 + len(self.chars)
        self.bos_id, self.eos_id = self.BOS, self.EOS

    @classmethod
    def from_text(cls, text: str):
        return cls(sorted(set(text)))

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = [self.stoi.get(c, self.UNK) for c in text]
        if add_bos:
            ids.insert(0, self.BOS)
        if add_eos:
            ids.append(self.EOS)
        return ids

    def decode(self, ids, skip_special: bool = True) -> str:
        names = {self.BOS: "<BOS>", self.EOS: "<EOS>", self.UNK: "<UNK>"}
        out = []
        for item in ids:
            i = int(item)
            if i < 3:
                if not skip_special:
                    out.append(names[i])
                continue
            out.append(self.itos.get(i, f"<tok:{i}>"))
        return "".join(out)

    def to_meta(self):
        return {"type": "char", "chars": self.chars}


def tokenizer_from_meta(meta: dict):
    """Rebuild the tokenizer a checkpoint was trained with (byte tokenizer if none is stored)."""
    tok = (meta.get("extra") or {}).get("tokenizer")
    if tok and tok.get("type") == "char":
        return CharTokenizer(tok["chars"])
    return ByteTokenizer()


# ----------------------------------------------------------------------------
# Primitive ops: each forward returns (output, cache); each backward returns
# the input gradient and accumulates parameter gradients into `grads`.
# ----------------------------------------------------------------------------

def _linear_fwd(x, params, name):
    y = x @ params[name + ".weight"].T
    if name + ".bias" in params:
        y = y + params[name + ".bias"]
    return y


def _linear_bwd(dy, x, params, grads, name):
    W = params[name + ".weight"]
    x2 = x.reshape(-1, x.shape[-1])
    dy2 = dy.reshape(-1, dy.shape[-1])
    grads[name + ".weight"] += dy2.T @ x2
    if name + ".bias" in params:
        grads[name + ".bias"] += dy2.sum(axis=0)
    return dy @ W


def _norm_fwd(x, params, name, use_rms):
    if use_rms:
        r = 1.0 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + LN_EPS)
        xn = x * r
        return xn * params[name + ".weight"], (x, r, xn)
    mu = x.mean(axis=-1, keepdims=True)
    xc = x - mu
    var = np.mean(xc * xc, axis=-1, keepdims=True)
    r = 1.0 / np.sqrt(var + LN_EPS)
    xh = xc * r
    return xh * params[name + ".weight"] + params[name + ".bias"], (r, xh)


def _norm_bwd(dy, cache, params, grads, name, use_rms):
    w = params[name + ".weight"]
    D = dy.shape[-1]
    if use_rms:
        x, r, xn = cache
        grads[name + ".weight"] += (dy * xn).reshape(-1, D).sum(axis=0)
        dn = dy * w
        # d/dx of x * r(x) with r = (mean(x^2)+eps)^(-1/2)
        return r * dn - x * (r ** 3 / D) * np.sum(dn * x, axis=-1, keepdims=True)
    r, xh = cache
    grads[name + ".weight"] += (dy * xh).reshape(-1, D).sum(axis=0)
    grads[name + ".bias"] += dy.reshape(-1, D).sum(axis=0)
    dxh = dy * w
    return r * (dxh - dxh.mean(axis=-1, keepdims=True)
                - xh * np.mean(dxh * xh, axis=-1, keepdims=True))


def _gelu_fwd(x):
    t = np.tanh(_GELU_C * (x + 0.044715 * x * x * x))
    return 0.5 * x * (1.0 + t), t


def _gelu_bwd(dy, x, t):
    dg = 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * _GELU_C * (1.0 + 3.0 * 0.044715 * x * x)
    return dy * dg


def _rope_tables(positions, dh, dtype):
    inv = 1.0 / (ROPE_THETA ** (np.arange(0, dh, 2, dtype=np.float64) / dh))
    ang = positions.astype(np.float64)[:, None] * inv[None, :]
    return np.cos(ang).astype(dtype), np.sin(ang).astype(dtype)


def _rope(x, cos, sin):
    # Interleaved even/odd rotation. x: (B,H,S,Dh); cos/sin: (S,Dh/2).
    out = np.empty_like(x)
    xe, xo = x[..., 0::2], x[..., 1::2]
    out[..., 0::2] = xe * cos - xo * sin
    out[..., 1::2] = xe * sin + xo * cos
    return out


def _rope_bwd(dy, cos, sin):
    # Transpose of a rotation is the inverse rotation.
    out = np.empty_like(dy)
    de, do = dy[..., 0::2], dy[..., 1::2]
    out[..., 0::2] = de * cos + do * sin
    out[..., 1::2] = -de * sin + do * cos
    return out


def _attn_fwd(h, params, pre, n_heads, n_kv_heads, use_rope, offset, past):
    B, S, D = h.shape
    H, Hkv = n_heads, n_kv_heads
    Dh = D // H
    Dkv = Hkv * Dh
    group = H // Hkv
    qkv = h @ params[pre + "attn.qkv.weight"].T
    q = qkv[..., :D].reshape(B, S, H, Dh).transpose(0, 2, 1, 3)
    k = qkv[..., D:D + Dkv].reshape(B, S, Hkv, Dh).transpose(0, 2, 1, 3)
    v = qkv[..., D + Dkv:].reshape(B, S, Hkv, Dh).transpose(0, 2, 1, 3)
    pos = np.arange(offset, offset + S)
    cos = sin = None
    if use_rope:
        cos, sin = _rope_tables(pos, Dh, q.dtype)
        q, k = _rope(q, cos, sin), _rope(k, cos, sin)
    if past is not None:
        k_u = np.concatenate([past[0], k], axis=2)
        v_u = np.concatenate([past[1], v], axis=2)
    else:
        k_u, v_u = k, v
    # GQA: each KV head serves `group` query heads; repeat for the attention math.
    k_all = np.repeat(k_u, group, axis=1) if group > 1 else k_u
    v_all = np.repeat(v_u, group, axis=1) if group > 1 else v_u
    Tk = k_all.shape[2]
    # Offset-aware causal mask: a query at absolute position p sees keys at positions <= p.
    mask = np.arange(Tk)[None, :] <= pos[:, None]
    scale = 1.0 / math.sqrt(Dh)
    scores = (q @ k_all.transpose(0, 1, 3, 2)) * scale
    scores = np.where(mask[None, None], scores, np.array(-1e30, dtype=scores.dtype))
    scores = scores - scores.max(axis=-1, keepdims=True)
    e = np.exp(scores)
    P = e / e.sum(axis=-1, keepdims=True)
    o = P @ v_all
    o2 = o.transpose(0, 2, 1, 3).reshape(B, S, D)
    out = o2 @ params[pre + "attn.proj.weight"].T
    cache = (h, q, k_all, v_all, P, o2, cos, sin, scale, Hkv)
    return out, cache, (k_u, v_u)


def _attn_bwd(dout, cache, params, grads, pre):
    h, q, k, v, P, o2, cos, sin, scale, Hkv = cache
    B, S, D = h.shape
    H, Dh = q.shape[1], q.shape[3]
    Dkv = Hkv * Dh
    group = H // Hkv
    grads[pre + "attn.proj.weight"] += dout.reshape(-1, D).T @ o2.reshape(-1, D)
    do = (dout @ params[pre + "attn.proj.weight"]).reshape(B, S, H, Dh).transpose(0, 2, 1, 3)
    dP = do @ v.transpose(0, 1, 3, 2)
    dv = P.transpose(0, 1, 3, 2) @ do
    dS = P * (dP - np.sum(dP * P, axis=-1, keepdims=True)) * scale
    if cos is not None:
        dq = _rope_bwd(dS @ k, cos, sin)
        dk = _rope_bwd(dS.transpose(0, 1, 3, 2) @ q, cos, sin)
    else:
        dq = dS @ k
        dk = dS.transpose(0, 1, 3, 2) @ q
    if group > 1:
        # Sum the gradients of the repeated KV heads back into their source head.
        dk = dk.reshape(B, Hkv, group, S, Dh).sum(axis=2)
        dv = dv.reshape(B, Hkv, group, S, Dh).sum(axis=2)

    def merge(z, width):
        return z.transpose(0, 2, 1, 3).reshape(B, S, width)

    dqkv = np.concatenate([merge(dq, D), merge(dk, Dkv), merge(dv, Dkv)], axis=-1)
    grads[pre + "attn.qkv.weight"] += dqkv.reshape(-1, D + 2 * Dkv).T @ h.reshape(-1, D)
    return dqkv @ params[pre + "attn.qkv.weight"]


def _sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def _block_fwd(params, pre, x, n_heads, n_kv_heads, use_rms, use_rope, ff_mode, offset, past):
    n1, c_n1 = _norm_fwd(x, params, pre + "norm1", use_rms)
    a, c_a, kv = _attn_fwd(n1, params, pre, n_heads, n_kv_heads, use_rope, offset, past)
    x1 = x + a
    n2, c_n2 = _norm_fwd(x1, params, pre + "norm2", use_rms)
    if ff_mode == "swiglu":
        g1 = _linear_fwd(n2, params, pre + "ff_gate")
        u = _linear_fwd(n2, params, pre + "ff_up")
        act = g1 * _sigmoid(g1) * u
        f2 = _linear_fwd(act, params, pre + "ff2")
        ff_cache = ("swiglu", n2, g1, u, act)
    else:
        f1 = _linear_fwd(n2, params, pre + "ff1")
        g, t = _gelu_fwd(f1)
        f2 = _linear_fwd(g, params, pre + "ff2")
        ff_cache = ("gelu", n2, f1, g, t)
    y = x1 + f2
    cache = (c_n1, c_a, c_n2, ff_cache)
    return y, cache, kv


def _block_bwd(dy, cache, params, grads, pre, use_rms):
    c_n1, c_a, c_n2, ff_cache = cache
    if ff_cache[0] == "swiglu":
        _, n2, g1, u, act = ff_cache
        dact = _linear_bwd(dy, act, params, grads, pre + "ff2")
        s = _sigmoid(g1)
        du = dact * (g1 * s)
        dg1 = (dact * u) * s * (1.0 + g1 * (1.0 - s))
        dn2 = _linear_bwd(dg1, n2, params, grads, pre + "ff_gate") \
            + _linear_bwd(du, n2, params, grads, pre + "ff_up")
    else:
        _, n2, f1, g, t = ff_cache
        dg = _linear_bwd(dy, g, params, grads, pre + "ff2")
        df1 = _gelu_bwd(dg, f1, t)
        dn2 = _linear_bwd(df1, n2, params, grads, pre + "ff1")
    dx1 = dy + _norm_bwd(dn2, c_n2, params, grads, pre + "norm2", use_rms)
    dn1 = _attn_bwd(dx1, c_a, params, grads, pre)
    return dx1 + _norm_bwd(dn1, c_n1, params, grads, pre + "norm1", use_rms)


def _cross_entropy(logits, targets, label_smoothing=0.0):
    """Mean next-token cross-entropy and its gradient w.r.t. logits.

    With label smoothing eps, the target distribution is (1 - eps) on the true token plus
    eps / V spread uniformly over the vocabulary. Loss and gradient follow that definition.
    """
    B, S, V = logits.shape
    flat = logits.reshape(-1, V)
    t = targets.reshape(-1)
    N = flat.shape[0]
    idx = np.arange(N)
    m = flat.max(axis=1, keepdims=True)
    e = np.exp(flat - m)
    s = e.sum(axis=1, keepdims=True)
    logp = flat - m - np.log(s)
    eps = float(label_smoothing)
    nll = -logp[idx, t]
    if eps > 0.0:
        loss = ((1.0 - eps) * nll + eps * (-logp.mean(axis=1))).mean()
    else:
        loss = nll.mean()
    p = e / s
    if eps > 0.0:
        p -= eps / V
    p[idx, t] -= 1.0 - eps
    dlogits = (p / N).reshape(B, S, V).astype(logits.dtype, copy=False)
    return float(loss), dlogits


def _zeros_like_params(params):
    return {k: np.zeros_like(v) for k, v in params.items()}


def clip_grad_norm(grads, max_norm: float) -> float:
    """Scale gradients in place so their global L2 norm is at most max_norm. Returns the pre-clip norm."""
    total = math.sqrt(sum(float(np.sum(g.astype(np.float64) ** 2)) for g in grads.values()))
    if total > max_norm:
        scale = max_norm / (total + 1e-6)
        for g in grads.values():
            g *= scale
    return total


class AdamWNP:
    """AdamW (decoupled weight decay), matching torch.optim.AdamW's update rule."""

    def __init__(self, params, lr=3e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        self.lr, self.betas, self.eps, self.weight_decay = lr, betas, eps, weight_decay
        self.t = 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params, grads):
        self.t += 1
        b1, b2 = self.betas
        bc1 = 1.0 - b1 ** self.t
        bc2 = 1.0 - b2 ** self.t
        for k in params:
            g = grads[k]
            m, v = self.m[k], self.v[k]
            m *= b1
            m += (1.0 - b1) * g
            v *= b2
            v += (1.0 - b2) * g * g
            mhat = m / bc1
            vhat = v / bc2
            wd = self.weight_decay if params[k].ndim >= 2 else 0.0
            params[k] = params[k] * (1.0 - self.lr * wd) \
                - self.lr * mhat / (np.sqrt(vhat) + self.eps)

    def load_state(self, m, v, t):
        self.m = {k: np.array(x) for k, x in m.items()}
        self.v = {k: np.array(x) for k, x in v.items()}
        self.t = int(t)


def lr_at(step: int, base_lr: float, schedule: str = "constant", warmup_steps: int = 0,
          total_steps: int = 1, min_lr_ratio: float = 0.1) -> float:
    """Learning rate for optimizer update `step` (1-indexed).

    Pure function of the step index, so resuming from a checkpoint needs no scheduler state.
    Warmup (linear, from 0 to base_lr) applies to both schedules.
    """
    if warmup_steps > 0 and step <= warmup_steps:
        return base_lr * step / warmup_steps
    if schedule == "constant":
        return base_lr
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    min_lr = base_lr * min_lr_ratio
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def _filter_probs(p, top_k: int, top_p: float):
    """Keep the top-k entries and/or the smallest nucleus with mass >= top_p, then renormalise."""
    if top_k and 0 < top_k < p.size:
        cutoff = np.partition(p, -top_k)[-top_k]
        p = np.where(p >= cutoff, p, 0.0)
    if 0.0 < top_p < 1.0:
        order = np.argsort(-p)
        cum_before = np.cumsum(p[order]) - p[order]
        mask = np.zeros(p.shape, dtype=bool)
        mask[order[cum_before < top_p]] = True
        p = np.where(mask, p, 0.0)
    return p / p.sum()


def _pick_token(logits, temperature: float, rng, top_k: int = 0, top_p: float = 1.0) -> int:
    if temperature == 0:
        return int(np.argmax(logits))
    z = np.asarray(logits, dtype=np.float64) / temperature
    p = np.exp(z - z.max())
    p /= p.sum()
    p = _filter_probs(p, top_k, top_p)
    rng = rng if rng is not None else np.random.default_rng(0)
    return int(rng.choice(len(p), p=p))


# ----------------------------------------------------------------------------
# Runtime decode control: entropy monitoring, loop detection, no-repeat-ngram
# blocking and a temperature controller. These are pure functions over logits
# and token IDs; they never touch model state, so they wrap any sampler.
# ----------------------------------------------------------------------------

def token_entropy(logits, temperature: float = 1.0) -> float:
    """Shannon entropy (nats) of softmax(logits / temperature). temperature<=0 is treated as 1."""
    z = np.asarray(logits, dtype=np.float64) / (temperature if temperature > 0 else 1.0)
    z = z - z.max()
    p = np.exp(z)
    p /= p.sum()
    return float(-(p * np.log(p + 1e-12)).sum())


def suffix_repeats(ids, n: int = 8, window: int = 128) -> bool:
    """True if the last n tokens already occurred earlier inside the recent window (a loop)."""
    if n <= 0 or len(ids) <= n:
        return False
    tail = list(ids[-window:])
    suffix = tuple(tail[-n:])
    return any(tuple(tail[i:i + n]) == suffix for i in range(len(tail) - n))


def banned_next_tokens(ids, n: int = 4, window: int = 64) -> set:
    """Tokens that would complete an n-gram already seen in the recent window.

    Blocking every such token guarantees the output never repeats an n-gram inside the window.
    """
    if n <= 0:
        return set()
    tail = list(ids[-window:])
    if len(tail) < n:
        return set()
    prefix = tuple(tail[len(tail) - (n - 1):]) if n > 1 else ()
    banned = set()
    for i in range(len(tail) - n + 1):
        if tuple(tail[i:i + n - 1]) == prefix:
            banned.add(int(tail[i + n - 1]))
    return banned


class DecodeController:
    """Runtime controller for sampling.

    - Blocks tokens that would recreate an n-gram already in the recent window (block_ngram).
    - Flags looping when the last `loop_len` tokens already occurred earlier in the window.
      While looping, the temperature is pushed up; with base temperature 0 the step samples
      at loop_temperature instead of taking the argmax.
    - Tracks the rolling mean entropy of the distribution actually sampled from and adjusts
      the temperature multiplicatively toward target_entropy: low entropy (collapse) raises
      it, high entropy (drift) lowers it. Disable with adapt=False.
    """

    def __init__(self, base_temperature: float = 0.0, target_entropy: float = 1.0,
                 gain: float = 0.5, window: int = 16, block_ngram: int = 4,
                 loop_len: int = 8, loop_temperature: float = 0.8,
                 t_min: float = 0.2, t_max: float = 1.5, adapt: bool = True):
        if base_temperature < 0 or t_min <= 0 or t_max < t_min:
            raise ValueError("need base_temperature >= 0 and 0 < t_min <= t_max")
        self.base = float(base_temperature)
        self.target = float(target_entropy)
        self.gain = float(gain)
        self.hist = collections.deque(maxlen=window)
        self.block_ngram = block_ngram
        self.loop_len = loop_len
        self.loop_temperature = float(loop_temperature)
        self.t_min, self.t_max = float(t_min), float(t_max)
        self.adapt = adapt
        self.temp = self.base
        self.trace = []  # one (entropy, temperature, looping, n_blocked) tuple per step

    def _next_temperature(self, looping: bool) -> float:
        if not self.adapt:
            return self.base
        if self.base == 0:
            return self.loop_temperature if looping else 0.0
        if looping:
            self.temp = min(self.temp * 1.25, self.t_max)
        elif self.hist:
            mean_h = float(np.mean(self.hist))
            self.temp = float(np.clip(self.temp * np.exp(self.gain * (self.target - mean_h)),
                                      self.t_min, self.t_max))
        return self.temp

    def choose(self, logits, ids, rng=None, top_k: int = 0, top_p: float = 1.0) -> int:
        logits = np.asarray(logits, dtype=np.float64).copy()
        looping = suffix_repeats(ids, self.loop_len) if self.loop_len else False
        banned = banned_next_tokens(ids, self.block_ngram) if self.block_ngram else set()
        if banned:
            masked = logits.copy()
            for t in banned:
                masked[t] = -np.inf
            if np.isfinite(masked).any():
                logits = masked
        temp = self._next_temperature(looping)
        h = token_entropy(logits, temp)
        self.hist.append(h)
        self.trace.append((h, temp, looping, len(banned)))
        return _pick_token(logits, temp, rng, top_k, top_p)

    def summary(self) -> str:
        if not self.trace:
            return "decode: no steps"
        hs = [t[0] for t in self.trace]
        ts = [t[1] for t in self.trace]
        return (f"decode: steps={len(hs)} mean_entropy={np.mean(hs):.3f} "
                f"loop_steps={sum(t[2] for t in self.trace)} "
                f"blocked_candidates={sum(t[3] for t in self.trace)} "
                f"temp_range=[{min(ts):.2f},{max(ts):.2f}]")


class DualTransformerNP:
    def __init__(self, d_model=64, n_blocks=2, n_heads=4,
                 merge_method="average", use_rms=True, max_seq=512,
                 seed=0, dtype=np.float32, vocab_size=VOCAB_SIZE, final_norm=True,
                 tie_embeddings=False, scaled_init=True,
                 n_kv_heads=None, pos_mode="rope", ff_mode="gelu",
                 init_style="gpt2"):
        if merge_method not in MERGE_METHODS:
            raise ValueError(f"merge_method must be one of {MERGE_METHODS}")
        if d_model <= 0 or n_blocks <= 0 or max_seq <= 0:
            raise ValueError("d_model, n_blocks, max_seq must be positive")
        if d_model % n_heads or (d_model // n_heads) % 2:
            raise ValueError("d_model/n_heads must be even")
        self.d_model, self.max_seq = d_model, max_seq
        self.n_blocks, self.n_heads, self.use_rms = n_blocks, n_heads, use_rms
        if tie_embeddings and merge_method == "concat":
            raise ValueError("tie_embeddings needs an output width equal to d_model (not concat)")
        if vocab_size < 2:
            raise ValueError("vocab_size must be >= 2")
        if n_kv_heads is None:
            n_kv_heads = n_heads
        if n_kv_heads <= 0 or n_heads % n_kv_heads:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if pos_mode not in ("rope", "learned"):
            raise ValueError('pos_mode must be "rope" or "learned"')
        if ff_mode not in ("gelu", "swiglu"):
            raise ValueError('ff_mode must be "gelu" or "swiglu"')
        if init_style not in ("gpt2", "legacy"):
            raise ValueError('init_style must be "gpt2" or "legacy"')
        self.merge_method = merge_method
        self.vocab_size = int(vocab_size)
        self.final_norm = bool(final_norm)
        self.tie_embeddings = bool(tie_embeddings)
        self.scaled_init = bool(scaled_init)
        self.n_kv_heads = int(n_kv_heads)
        self.pos_mode = pos_mode
        self.ff_mode = ff_mode
        self.init_style = init_style
        self.dtype = np.dtype(dtype)
        self.params = self._init_params(seed)

    # ----- structure -----

    def config(self) -> dict:
        return {
            "d_model": self.d_model,
            "n_blocks": self.n_blocks,
            "n_heads": self.n_heads,
            "merge_method": self.merge_method,
            "use_rms": self.use_rms,
            "max_seq": self.max_seq,
            "vocab_size": self.vocab_size,
            "final_norm": self.final_norm,
            "tie_embeddings": self.tie_embeddings,
            "scaled_init": self.scaled_init,
            "n_kv_heads": self.n_kv_heads,
            "pos_mode": self.pos_mode,
            "ff_mode": self.ff_mode,
        }

    def _head_weight(self):
        return self.params["emb"] if self.tie_embeddings else self.params["lm_head.weight"]

    def _branches(self):
        return ("left",) if self.merge_method == "single" else ("left", "right")

    def _ff_dim(self):
        if self.ff_mode == "swiglu":
            return max(1, int(round(8 * self.d_model / 3)))
        return 4 * self.d_model

    def _init_params(self, seed):
        rng = np.random.default_rng(seed)
        D, V, dt = self.d_model, self.vocab_size, self.dtype
        gpt2 = self.init_style == "gpt2"
        if gpt2:
            p = {"emb": (rng.standard_normal((V, D)) * 0.02).astype(dt)}
        else:
            p = {"emb": rng.standard_normal((V, D)).astype(dt)}
        if self.pos_mode == "learned":
            if gpt2:
                p["wpe"] = (rng.standard_normal((self.max_seq, D)) * 0.02).astype(dt)
            else:
                p["wpe"] = rng.standard_normal((self.max_seq, D)).astype(dt)
        # GPT-2 style: outputs of residual branches are scaled by 1/sqrt(2 * n_blocks).
        res_scale = 1.0 / math.sqrt(2 * self.n_blocks) if self.scaled_init else 1.0

        def linear(name, out_f, in_f, bias, scale=1.0):
            if gpt2:
                p[name + ".weight"] = (rng.standard_normal((out_f, in_f)) * (0.02 * scale)).astype(dt)
                if bias:
                    p[name + ".bias"] = np.zeros(out_f, dt)
            else:
                bound = scale / math.sqrt(in_f)
                p[name + ".weight"] = rng.uniform(-bound, bound, (out_f, in_f)).astype(dt)
                if bias:
                    p[name + ".bias"] = rng.uniform(-bound, bound, (out_f,)).astype(dt)

        Dh = D // self.n_heads
        qkv_out = D + 2 * self.n_kv_heads * Dh
        H = self._ff_dim()
        for name in self._branches():
            for i in range(self.n_blocks):
                pre = f"{name}.{i}."
                p[pre + "norm1.weight"] = np.ones(D, dt)
                if not self.use_rms:
                    p[pre + "norm1.bias"] = np.zeros(D, dt)
                linear(pre + "attn.qkv", qkv_out, D, bias=False)
                linear(pre + "attn.proj", D, D, bias=False, scale=res_scale)
                p[pre + "norm2.weight"] = np.ones(D, dt)
                if not self.use_rms:
                    p[pre + "norm2.bias"] = np.zeros(D, dt)
                if self.ff_mode == "swiglu":
                    linear(pre + "ff_gate", H, D, bias=True)
                    linear(pre + "ff_up", H, D, bias=True)
                    linear(pre + "ff2", D, H, bias=True, scale=res_scale)
                else:
                    linear(pre + "ff1", 4 * D, D, bias=True)
                    linear(pre + "ff2", D, 4 * D, bias=True, scale=res_scale)
        out_dim = 2 * D if self.merge_method == "concat" else D
        if self.final_norm:
            p["final_norm.weight"] = np.ones(out_dim, dt)
            if not self.use_rms:
                p["final_norm.bias"] = np.zeros(out_dim, dt)
        if not self.tie_embeddings:
            linear("lm_head", V, out_dim, bias=False)
        return p

    def num_params(self) -> int:
        return int(sum(v.size for v in self.params.values()))

    def _merge(self, outs):
        if self.merge_method == "single":
            return outs["left"]
        if self.merge_method == "average":
            return (outs["left"] + outs["right"]) * 0.5
        return np.concatenate([outs["left"], outs["right"]], axis=-1)

    def _check_tokens(self, tokens):
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape (batch, sequence)")
        if tokens.shape[1] == 0:
            raise ValueError("empty sequences are not supported")
        if tokens.shape[1] > self.max_seq:
            raise ValueError(f"sequence length {tokens.shape[1]} exceeds max_seq={self.max_seq}")
        if tokens.min() < 0 or tokens.max() >= self.vocab_size:
            raise ValueError(f"token IDs must be in [0, {self.vocab_size - 1}]")

    def _forward(self, tokens, pasts=None):
        B, S = tokens.shape
        offset = 0
        if pasts is not None and pasts.get("left") and pasts["left"][0] is not None:
            offset = pasts["left"][0][0].shape[2]
        if offset + S > self.max_seq:
            raise ValueError("KV cache plus input chunk exceeds max_seq; reset/re-prefill cache")
        x = self.params["emb"][tokens]
        if self.pos_mode == "learned":
            pos = np.arange(offset, offset + S)
            x = x + self.params["wpe"][pos][None, :, :]
        outs, caches, new_kv = {}, {}, {}
        for name in self._branches():
            h = x
            bc, kvs = [], []
            for i in range(self.n_blocks):
                past = pasts[name][i] if pasts is not None else None
                h, cache, kv = _block_fwd(self.params, f"{name}.{i}.", h,
                                          self.n_heads, self.n_kv_heads, self.use_rms,
                                          self.pos_mode == "rope", self.ff_mode, offset, past)
                bc.append(cache)
                kvs.append(kv)
            outs[name] = h
            caches[name] = bc
            new_kv[name] = kvs
        merged = self._merge(outs)
        h, fcache = merged, None
        if self.final_norm:
            h, fcache = _norm_fwd(merged, self.params, "final_norm", self.use_rms)
        logits = h @ self._head_weight().T
        return logits, h, caches, new_kv, merged, fcache

    # ----- public inference API -----

    def forward(self, tokens) -> np.ndarray:
        tokens = np.asarray(tokens, dtype=np.int64)
        self._check_tokens(tokens)
        return self._forward(tokens)[0]

    def forward_cached(self, tokens, pasts=None):
        """Incremental forward; returns logits and the updated per-branch/per-layer KV cache."""
        tokens = np.asarray(tokens, dtype=np.int64)
        if tokens.ndim != 2 or tokens.shape[1] == 0:
            raise ValueError("tokens must be non-empty shape (batch, sequence)")
        if tokens.min() < 0 or tokens.max() >= self.vocab_size:
            raise ValueError(f"token IDs must be in [0, {self.vocab_size - 1}]")
        logits, _, _, new_kv, _, _ = self._forward(tokens, pasts)
        return logits, new_kv

    def loss(self, tokens, targets, label_smoothing=0.0) -> float:
        logits = self.forward(tokens)
        return _cross_entropy(logits, np.asarray(targets, dtype=np.int64), label_smoothing)[0]

    def loss_and_grads(self, tokens, targets, label_smoothing=0.0):
        tokens = np.asarray(tokens, dtype=np.int64)
        targets = np.asarray(targets, dtype=np.int64)
        self._check_tokens(tokens)
        D, V = self.d_model, self.vocab_size
        logits, h, caches, _, _, fcache = self._forward(tokens)
        loss, dlogits = _cross_entropy(logits, targets, label_smoothing)
        grads = _zeros_like_params(self.params)
        out_dim = h.shape[-1]
        head_key = "emb" if self.tie_embeddings else "lm_head.weight"
        grads[head_key] += dlogits.reshape(-1, V).T @ h.reshape(-1, out_dim)
        dh = dlogits @ self._head_weight()
        if self.final_norm:
            dmerged = _norm_bwd(dh, fcache, self.params, grads, "final_norm", self.use_rms)
        else:
            dmerged = dh

        if self.merge_method == "single":
            d_branch = {"left": dmerged}
        elif self.merge_method == "average":
            d_branch = {"left": dmerged * 0.5, "right": dmerged * 0.5}
        else:
            d_branch = {"left": dmerged[..., :D], "right": dmerged[..., D:]}

        dx_total = np.zeros(tokens.shape + (D,), dtype=self.dtype)
        for name in self._branches():
            dy = d_branch[name]
            for i in reversed(range(self.n_blocks)):
                dy = _block_bwd(dy, caches[name][i], self.params, grads,
                                f"{name}.{i}.", self.use_rms)
            dx_total += dy
        np.add.at(grads["emb"], tokens.reshape(-1), dx_total.reshape(-1, D))
        if self.pos_mode == "learned":
            B, S = tokens.shape
            np.add.at(grads["wpe"], np.tile(np.arange(S), B), dx_total.reshape(-1, D))
        return loss, grads

    def _check_prompt(self, prompt_ids):
        if not prompt_ids:
            raise ValueError("prompt_ids must not be empty")
        if any(int(t) < 0 or int(t) >= self.vocab_size for t in prompt_ids):
            raise ValueError("prompt contains invalid token IDs")

    def generate(self, prompt_ids, max_new_tokens=100, temperature=0.0, rng=None, top_k=0, top_p=1.0):
        """Full-context greedy/temperature generation over a sliding window."""
        self._check_prompt(prompt_ids)
        if max_new_tokens < 0 or temperature < 0:
            raise ValueError("max_new_tokens must be non-negative and temperature >= 0")
        result = [int(t) for t in prompt_ids]
        for _ in range(max_new_tokens):
            ctx = np.array([result[-self.max_seq:]], dtype=np.int64)
            logits = self.forward(ctx)[0, -1]
            result.append(_pick_token(logits, temperature, rng, top_k, top_p))
        return result

    def generate_cached(self, prompt_ids, max_new_tokens=100, temperature=0.0, rng=None, top_k=0, top_p=1.0,
                        controller: Optional[DecodeController] = None,
                        logit_hook=None, stop_fn=None):
        """KV-cached generation; the cache is rebuilt when the sliding window rolls.

        If a DecodeController is given it picks each token (its base temperature replaces
        `temperature`); otherwise the plain sampler is used.

        logit_hook(result, logits) -> logits may mask tokens before sampling (grammar constraints).
        stop_fn(result) -> bool ends generation early after the token just appended.
        """
        self._check_prompt(prompt_ids)
        if max_new_tokens < 0 or temperature < 0:
            raise ValueError("max_new_tokens must be non-negative and temperature >= 0")
        result = [int(t) for t in prompt_ids]
        if max_new_tokens == 0:
            return result
        ctx = result[-self.max_seq:]
        logits, pasts = self.forward_cached(np.array([ctx], dtype=np.int64))
        for generated_i in range(max_new_tokens):
            step_logits = logits[0, -1]
            if logit_hook is not None:
                step_logits = logit_hook(result, step_logits)
            if controller is not None:
                nxt = controller.choose(step_logits, result, rng, top_k, top_p)
            else:
                nxt = _pick_token(step_logits, temperature, rng, top_k, top_p)
            result.append(nxt)
            if stop_fn is not None and stop_fn(result):
                break
            if generated_i + 1 == max_new_tokens:
                break
            if len(ctx) < self.max_seq:
                logits, pasts = self.forward_cached(np.array([[nxt]], dtype=np.int64), pasts)
                ctx.append(nxt)
            else:
                # Window is full: re-prefill from the last max_seq tokens (positions restart at 0,
                # matching generate()).
                ctx = result[-self.max_seq:]
                logits, pasts = self.forward_cached(np.array([ctx], dtype=np.int64))
        return result

    # ----- checkpoints -----

    def save_checkpoint(self, path, optimizer: Optional[AdamWNP] = None, step: int = 0,
                        extra: Optional[dict] = None, rng_state: Optional[dict] = None):
        arrays = {"p_" + k: v for k, v in self.params.items()}
        meta = {
            "format_version": 1,
            "config": self.config(),
            "dtype": self.dtype.name,
            "step": int(step),
            "has_optimizer": optimizer is not None,
            "rng_state": rng_state,
            "extra": extra or {},
        }
        if optimizer is not None:
            meta["optimizer_t"] = optimizer.t
            for k in self.params:
                arrays["m_" + k] = optimizer.m[k]
                arrays["v_" + k] = optimizer.v[k]
        arrays["meta_json"] = np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8)
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        # Passing a file object stops np.savez from appending ".npz" to the name.
        with open(target, "wb") as f:
            np.savez(f, **arrays)

    @classmethod
    def load_checkpoint(cls, path):
        ck = read_checkpoint(path)
        meta = ck["meta"]
        model = cls(**meta["config"], dtype=np.dtype(meta["dtype"]))
        if set(ck["params"]) != set(model.params):
            raise ValueError("checkpoint parameter names do not match the architecture")
        for k in model.params:
            if ck["params"][k].shape != model.params[k].shape:
                raise ValueError(f"shape mismatch for {k}")
            model.params[k] = ck["params"][k].astype(model.dtype)
        return model, ck


def read_checkpoint(path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(bytes(data["meta_json"]).decode("utf-8"))
        if meta.get("format_version") != 1:
            raise ValueError("unsupported checkpoint format")
        params = {k[2:]: np.array(data[k]) for k in data.files if k.startswith("p_")}
        m = {k[2:]: np.array(data[k]) for k in data.files if k.startswith("m_")}
        v = {k[2:]: np.array(data[k]) for k in data.files if k.startswith("v_")}
    return {"meta": meta, "params": params, "m": m, "v": v}


def last_checkpoint_path(path) -> Path:
    """Path of the latest-state checkpoint that belongs to a best-checkpoint path."""
    p = Path(path)
    return p.with_name(f"{p.stem}_last{p.suffix}")


def make_batch(data: np.ndarray, batch_size: int, seq_len: int, rng: np.random.Generator):
    starts = rng.integers(0, len(data) - seq_len, size=batch_size)
    x = np.stack([data[int(i):int(i) + seq_len] for i in starts])
    y = np.stack([data[int(i) + 1:int(i) + seq_len + 1] for i in starts])
    return x, y


def evaluate_loss(model: DualTransformerNP, token_ids, seq_len=64,
                  batch_size=8, max_batches=20) -> float:
    """Deterministic held-out next-token loss over consecutive non-overlapping windows."""
    data = np.asarray(token_ids, dtype=np.int64)
    if len(data) < seq_len + 1:
        raise ValueError(f"validation text needs at least {seq_len + 1} tokens")
    starts = list(range(0, len(data) - seq_len, seq_len))[:max_batches]
    losses = []
    for start in starts:
        x = data[start:start + seq_len][None, :]
        y = data[start + 1:start + seq_len + 1][None, :]
        losses.append(model.loss(x, y))
    if not losses:
        raise ValueError("validation set produced no complete windows")
    return float(sum(losses) / len(losses))


TOY_VOCAB = 16


def make_toy_batch(batch_size, seq_len, rng, task="arith", vocab=TOY_VOCAB):
    """Synthetic next-token tasks over token IDs [0, vocab).

    arith: each sequence counts up modulo vocab; the target is the next value.
    copy:  the second half repeats the first half; the last target wraps to the first token.
    """
    if task == "copy":
        half = seq_len // 2
        first = rng.integers(0, vocab, (batch_size, half))
        tokens = np.concatenate([first, first[:, : seq_len - half]], axis=1)
        targets = np.roll(tokens, -1, axis=1)
        targets[:, -1] = tokens[:, 0]
        return tokens.astype(np.int64), targets.astype(np.int64)
    starts = rng.integers(0, vocab, (batch_size, 1))
    tokens = (starts + np.arange(seq_len)[None, :]) % vocab
    targets = (tokens + 1) % vocab
    return tokens.astype(np.int64), targets.astype(np.int64)


def train_steps(model: DualTransformerNP, token_ids=None, steps=500, batch_size=8, seq_len=64,
                lr=3e-4, log_every=50, val_token_ids=None, checkpoint_path=None,
                eval_every=50, resume_from=None, seed=0, weight_decay=0.01, max_grad_norm=1.0,
                label_smoothing=0.0, grad_accum_steps=1, schedule="constant", warmup_steps=0,
                min_lr_ratio=0.1, log_csv=None, batch_fn=None, extra_meta=None):
    """Train with optional validation, checkpointing, LR schedule, gradient accumulation and CSV logs.

    steps counts optimizer updates. Each update averages gradients over grad_accum_steps
    micro-batches, so it sees batch_size * grad_accum_steps sequences.
    schedule: "constant", or "cosine" (decays to min_lr_ratio * lr). warmup_steps applies to both.
    The learning rate is a pure function of the update index, so resume needs no scheduler state.
    The schedule's total length is stored in the checkpoint, so a resumed run continues on the
    original curve (steps past the scheduled end stay at the minimum learning rate).
    batch_fn(rng) -> (inputs, targets) replaces corpus batching (used by the toy tasks).

    With validation:
      - checkpoint_path receives the best-validation weights (written on improvement);
      - last_checkpoint_path(checkpoint_path) receives the latest state at every
        evaluation and at the end of training. Resume uses this latest state, so no
        steps are repeated or silently discarded. The RNG state is saved too, so the
        batch sequence continues where it left off.
    Without validation:
      - checkpoint_path receives the final state.
    Optimizer hyperparameters come from the arguments; the Adam moments and step count
    come from the checkpoint on resume.
    """
    if grad_accum_steps < 1:
        raise ValueError("grad_accum_steps must be >= 1")
    if schedule not in ("constant", "cosine"):
        raise ValueError("schedule must be 'constant' or 'cosine'")
    if batch_fn is None:
        if token_ids is None:
            raise ValueError("provide token_ids or batch_fn")
        data = np.asarray(token_ids, dtype=np.int64)
        if len(data) < seq_len + 2:
            raise ValueError(f"training text needs at least {seq_len + 2} tokens")

        def batch_fn(r):
            return make_batch(data, batch_size, seq_len, r)

    if steps <= 0 or batch_size <= 0 or seq_len <= 0 or eval_every <= 0 or log_every <= 0:
        raise ValueError("steps, batch_size, seq_len, eval_every and log_every must be positive")
    if seq_len > model.max_seq:
        raise ValueError("seq_len must be <= model.max_seq")
    val = None if val_token_ids is None else np.asarray(val_token_ids, dtype=np.int64)
    if val is not None and len(val) < seq_len + 1:
        raise ValueError(f"validation text needs at least {seq_len + 1} tokens")

    optimizer = AdamWNP(model.params, lr=lr, weight_decay=weight_decay)
    rng = np.random.default_rng(seed)
    start_step, best_val, last_loss = 0, math.inf, math.inf
    schedule_total = None

    if resume_from:
        ck = read_checkpoint(resume_from)
        meta = ck["meta"]
        saved_cfg = dict(meta["config"])
        # Checkpoints written before the GQA/SwiGLU/pos options existed lack the
        # new keys; their values are the legacy defaults filled in here.
        saved_cfg.setdefault("n_kv_heads", saved_cfg.get("n_heads"))
        saved_cfg.setdefault("pos_mode", "rope")
        saved_cfg.setdefault("ff_mode", "gelu")
        if saved_cfg != model.config():
            raise ValueError(f"resume checkpoint config {meta['config']} "
                             f"does not match model config {model.config()}")
        if not meta["has_optimizer"]:
            raise ValueError("resume checkpoint has no optimizer state")
        if set(ck["params"]) != set(model.params):
            raise ValueError("resume checkpoint parameter names do not match the model")
        for k in model.params:
            if ck["params"][k].shape != model.params[k].shape:
                raise ValueError(f"shape mismatch for {k}")
            model.params[k] = ck["params"][k].astype(model.dtype)
        optimizer.load_state(ck["m"], ck["v"], meta["optimizer_t"])
        start_step = int(meta["step"])
        best_val = float(meta["extra"].get("best_val_loss", math.inf))
        last_loss = float(meta["extra"].get("last_train_loss", math.inf))
        schedule_total = int(meta["extra"].get("schedule_total_steps", start_step + steps))
        if meta.get("rng_state") is not None:
            rng.bit_generator.state = meta["rng_state"]
        print(f"resumed from step={start_step}; best_val_loss={best_val:.4f}")

    end_step = start_step + steps
    if schedule_total is None:
        schedule_total = end_step
    latest_path = last_checkpoint_path(checkpoint_path) if (checkpoint_path and val is not None) else None
    completed_step = start_step
    last_saved_step = None
    csv_file = csv_writer = None
    if log_csv:
        Path(log_csv).parent.mkdir(parents=True, exist_ok=True)
        fresh = not (resume_from and Path(log_csv).exists())
        csv_file = open(log_csv, "w" if fresh else "a", newline="")
        csv_writer = csv.writer(csv_file, lineterminator="\n")
        if fresh:
            csv_writer.writerow(["step", "train_loss", "val_loss", "grad_norm", "lr"])
    try:
        for step in range(start_step + 1, end_step + 1):
            optimizer.lr = lr_at(step, lr, schedule, warmup_steps, schedule_total, min_lr_ratio)
            grads_acc, loss_acc = None, 0.0
            for _ in range(grad_accum_steps):
                x, y = batch_fn(rng)
                loss, grads = model.loss_and_grads(x, y, label_smoothing=label_smoothing)
                if not math.isfinite(loss):
                    raise FloatingPointError(f"non-finite training loss at step {step}")
                loss_acc += float(loss) / grad_accum_steps
                if grads_acc is None:
                    grads_acc = {k: g / grad_accum_steps for k, g in grads.items()}
                else:
                    for k, g in grads.items():
                        grads_acc[k] += g / grad_accum_steps
            grad_norm = clip_grad_norm(grads_acc, max_grad_norm)
            if not math.isfinite(grad_norm):
                raise FloatingPointError(f"non-finite gradient norm at step {step}")
            optimizer.step(model.params, grads_acc)
            completed_step, last_loss = step, float(loss_acc)
            should_eval = val is not None and (step % eval_every == 0 or step == end_step)
            if should_eval:
                val_loss = evaluate_loss(model, val, seq_len, batch_size)
                improved = val_loss < best_val
                if improved:
                    best_val = val_loss
                print(f"step={step:5d} train_loss={last_loss:.4f} val_loss={val_loss:.4f} "
                      f"grad_norm={grad_norm:.4f} lr={optimizer.lr:.2e}" + (" [best]" if improved else ""))
                if csv_writer:
                    csv_writer.writerow([step, f"{last_loss:.6f}", f"{val_loss:.6f}",
                                         f"{grad_norm:.6f}", f"{optimizer.lr:.8g}"])
                extra = {"best_val_loss": float(best_val), "last_train_loss": float(last_loss),
                     "schedule_total_steps": schedule_total, **(extra_meta or {})}
                rng_state = rng.bit_generator.state
                if checkpoint_path and improved:
                    model.save_checkpoint(checkpoint_path, optimizer, step, extra, rng_state)
                if latest_path is not None:
                    model.save_checkpoint(latest_path, optimizer, step, extra, rng_state)
                    last_saved_step = step
            elif step == 1 or step % log_every == 0:
                print(f"step={step:5d} train_loss={last_loss:.4f} grad_norm={grad_norm:.4f} "
                      f"lr={optimizer.lr:.2e}")
                if csv_writer:
                    csv_writer.writerow([step, f"{last_loss:.6f}", "", f"{grad_norm:.6f}",
                                         f"{optimizer.lr:.8g}"])
            if csv_file:
                csv_file.flush()
    finally:
        if csv_file:
            csv_file.close()

    if checkpoint_path:
        extra = {"best_val_loss": float(best_val), "last_train_loss": float(last_loss),
                     "schedule_total_steps": schedule_total, **(extra_meta or {})}
        rng_state = rng.bit_generator.state
        if val is None:
            model.save_checkpoint(checkpoint_path, optimizer, completed_step, extra, rng_state)
        elif last_saved_step != completed_step:
            model.save_checkpoint(latest_path, optimizer, completed_step, extra, rng_state)
    return optimizer, last_loss, best_val, completed_step


# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------

def numerical_gradient_test():
    """Compare analytic gradients with central finite differences (float64).

    Samples a few entries per parameter tensor for every merge mode and both norm types.
    This is a sanity check on representative entries, not a proof for every element.
    """
    rng = np.random.default_rng(123)
    x = np.array([[BOS_ID, 97, 98]], dtype=np.int64)
    y = np.array([[97, 98, 99]], dtype=np.int64)
    eps = 1e-6
    checked = 0
    for merge in MERGE_METHODS:
        for use_rms in (True, False):
            model = DualTransformerNP(d_model=8, n_blocks=1, n_heads=2, merge_method=merge,
                                      use_rms=use_rms, max_seq=8, seed=123, dtype=np.float64)
            _, grads = model.loss_and_grads(x, y)
            for name, arr in model.params.items():
                for _ in range(min(arr.size, 4)):
                    idx = np.unravel_index(int(rng.integers(arr.size)), arr.shape)
                    orig = arr[idx]
                    arr[idx] = orig + eps
                    lp = model.loss(x, y)
                    arr[idx] = orig - eps
                    lm = model.loss(x, y)
                    arr[idx] = orig
                    numeric = (lp - lm) / (2 * eps)
                    analytic = float(grads[name][idx])
                    abs_err = abs(analytic - numeric)
                    rel_err = abs_err / max(abs(analytic), abs(numeric), 1e-12)
                    if not (abs_err < 1e-7 or rel_err < 1e-4):
                        raise AssertionError(
                            f"[{merge}, rms={use_rms}] {name}{idx} gradient check failed: "
                            f"analytic={analytic:.12g}, numeric={numeric:.12g}, "
                            f"abs_err={abs_err:.3g}, rel_err={rel_err:.3g}")
                    checked += 1
    print(f"Gradient checks OK: {checked} sampled entries across merge modes and norm types")


def checkpoint_round_trip_test():
    import tempfile
    model = DualTransformerNP(d_model=8, n_blocks=1, n_heads=2, max_seq=8, seed=9)
    x = np.array([[BOS_ID, 97, 98]], dtype=np.int64)
    y = np.array([[97, 98, 99]], dtype=np.int64)
    optimizer = AdamWNP(model.params, lr=1e-3)
    _, grads = model.loss_and_grads(x, y)
    optimizer.step(model.params, grads)
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "checkpoint.npz"
        model.save_checkpoint(path, optimizer=optimizer, step=1)
        loaded, ck = DualTransformerNP.load_checkpoint(path)
        assert ck["meta"]["step"] == 1 and ck["meta"]["has_optimizer"]
        assert np.array_equal(model.forward(x), loaded.forward(x))
    print("Checkpoint round-trip OK (logits identical after load)")


def training_validation_resume_test():
    import tempfile
    tokenizer = ByteTokenizer()
    ids = tokenizer.encode("abcabc你好Omega0123456789" * 12, add_bos=True, add_eos=True)
    split = int(len(ids) * 0.8)
    train_ids, val_ids = ids[:split], ids[split:]
    model = DualTransformerNP(d_model=8, n_blocks=1, n_heads=2, max_seq=8, seed=41)
    with tempfile.TemporaryDirectory() as td:
        best = Path(td) / "nested" / "best.npz"
        latest = last_checkpoint_path(best)
        _, loss, best_val, step = train_steps(
            model, train_ids, steps=2, batch_size=2, seq_len=4, lr=1e-3, log_every=1,
            val_token_ids=val_ids, checkpoint_path=best, eval_every=1)
        assert step == 2 and math.isfinite(loss) and math.isfinite(best_val)
        assert best.exists() and latest.exists()
        assert math.isfinite(float(read_checkpoint(best)["meta"]["extra"]["best_val_loss"]))
        latest_meta = read_checkpoint(latest)["meta"]
        assert latest_meta["step"] == 2 and latest_meta["has_optimizer"]
        restored, _ = DualTransformerNP.load_checkpoint(latest)
        _, loss2, best2, step2 = train_steps(
            restored, train_ids, steps=1, batch_size=2, seq_len=4, lr=1e-3, log_every=1,
            val_token_ids=val_ids, checkpoint_path=best, eval_every=1, resume_from=latest)
        assert step2 == 3, f"expected resume to continue at step 3, got {step2}"
        assert math.isfinite(loss2) and math.isfinite(best2)
    print("Training/validation/best+latest-checkpoint/resume loop OK")


def no_validation_resume_test():
    import tempfile
    ids = ByteTokenizer().encode("abcdefgh" * 20, add_bos=True, add_eos=True)
    model = DualTransformerNP(d_model=8, n_blocks=1, n_heads=2, max_seq=8, seed=43)
    with tempfile.TemporaryDirectory() as td:
        ckpt = Path(td) / "final.npz"
        train_steps(model, ids, steps=2, batch_size=2, seq_len=4, lr=1e-3, log_every=1,
                    checkpoint_path=ckpt)
        assert read_checkpoint(ckpt)["meta"]["step"] == 2
        restored, _ = DualTransformerNP.load_checkpoint(ckpt)
        _, _, _, step2 = train_steps(restored, ids, steps=1, batch_size=2, seq_len=4, lr=1e-3,
                                     log_every=1, checkpoint_path=ckpt, resume_from=ckpt)
        assert step2 == 3
    print("No-validation final checkpoint and resume OK")


def kv_cache_parity_test():
    """Full-context logits must match incremental cache logits at every position."""
    for merge in MERGE_METHODS:
        model = DualTransformerNP(d_model=16, n_blocks=2, n_heads=4, merge_method=merge,
                                  max_seq=6, seed=77, dtype=np.float64)
        tokens = np.array([[BOS_ID, 97, 98, 99, 100, EOS_ID]], dtype=np.int64)
        full = model.forward(tokens)
        pasts, parts = None, []
        for i in range(tokens.shape[1]):
            part, pasts = model.forward_cached(tokens[:, i:i + 1], pasts)
            parts.append(part)
        cached = np.concatenate(parts, axis=1)
        max_err = float(np.abs(full - cached).max())
        if not np.allclose(full, cached, atol=1e-10, rtol=1e-10):
            raise AssertionError(f"KV cache parity failed for {merge}: max_err={max_err}")
        full_gen = model.generate(tokens[0].tolist(), max_new_tokens=4)
        cache_gen = model.generate_cached(tokens[0].tolist(), max_new_tokens=4)
        if full_gen != cache_gen:
            raise AssertionError(f"cached generation mismatch for {merge}")
        print(f"KV cache parity [{merge}] OK: max_logit_error={max_err:.2e}")


def sliding_window_generation_test():
    """Generate past max_seq so the cache window rolls; cached and full-context paths must agree."""
    prompt = [BOS_ID, 97, 98, 99]
    for merge in MERGE_METHODS:
        model = DualTransformerNP(d_model=16, n_blocks=2, n_heads=4, merge_method=merge,
                                  max_seq=8, seed=78, dtype=np.float64)
        full = model.generate(prompt, max_new_tokens=20)
        cached = model.generate_cached(prompt, max_new_tokens=20)
        if full != cached:
            raise AssertionError(f"sliding-window generation mismatch for {merge}")
        assert len(full) == len(prompt) + 20
    print("Sliding-window generation test OK (window rolled past max_seq for all merge modes)")


def learning_sanity_test():
    """A tiny model should fit a repeating pattern: loss must drop clearly."""
    data = ByteTokenizer().encode("abcd" * 60, add_bos=False, add_eos=False)
    model = DualTransformerNP(d_model=16, n_blocks=1, n_heads=2, max_seq=16, seed=5)
    x = np.array([data[:12]], dtype=np.int64)
    y = np.array([data[1:13]], dtype=np.int64)
    first = model.loss(x, y)
    train_steps(model, data, steps=80, batch_size=4, seq_len=12, lr=1e-2, log_every=1000)
    last = model.loss(x, y)
    assert last < 0.5 * first, f"loss did not drop enough: {first:.3f} -> {last:.3f}"
    print(f"Learning sanity OK: loss {first:.3f} -> {last:.3f}")


def label_smoothing_gradient_test():
    """Finite-difference check of the label-smoothing gradient (float64)."""
    x = np.array([[BOS_ID, 97, 98]], dtype=np.int64)
    y = np.array([[97, 98, 99]], dtype=np.int64)
    ls, eps = 0.1, 1e-6
    rng = np.random.default_rng(321)
    model = DualTransformerNP(d_model=8, n_blocks=1, n_heads=2, merge_method="average",
                              use_rms=False, max_seq=8, seed=321, dtype=np.float64)
    _, grads = model.loss_and_grads(x, y, label_smoothing=ls)
    checked = 0
    for name, arr in model.params.items():
        for _ in range(min(arr.size, 3)):
            idx = np.unravel_index(int(rng.integers(arr.size)), arr.shape)
            orig = arr[idx]
            arr[idx] = orig + eps
            lp = model.loss(x, y, ls)
            arr[idx] = orig - eps
            lm = model.loss(x, y, ls)
            arr[idx] = orig
            numeric = (lp - lm) / (2 * eps)
            analytic = float(grads[name][idx])
            abs_err = abs(analytic - numeric)
            rel_err = abs_err / max(abs(analytic), abs(numeric), 1e-12)
            if not (abs_err < 1e-7 or rel_err < 1e-4):
                raise AssertionError(f"label-smoothing gradient failed for {name}{idx}: "
                                     f"analytic={analytic:.12g}, numeric={numeric:.12g}")
            checked += 1
    print(f"Label-smoothing gradient checks OK: {checked} sampled entries")


def sampling_and_schedule_test():
    logits = np.array([0.0, 5.0, 4.9, -3.0, -10.0])
    rng = np.random.default_rng(0)
    assert {_pick_token(logits, 1.0, rng, top_k=2) for _ in range(300)} <= {1, 2}
    assert {_pick_token(logits, 1.0, rng, top_p=0.5) for _ in range(100)} == {1}
    assert _pick_token(logits, 0.0, rng, top_k=2) == 1
    assert abs(lr_at(1, 1.0, "cosine", 2, 10) - 0.5) < 1e-12
    assert abs(lr_at(2, 1.0, "cosine", 2, 10) - 1.0) < 1e-12
    assert abs(lr_at(10, 1.0, "cosine", 2, 10, 0.1) - 0.1) < 1e-9
    assert lr_at(7, 0.3, "constant") == 0.3
    print("Top-k/top-p filters and LR schedule OK")


def toy_training_test():
    import tempfile
    model = DualTransformerNP(d_model=16, n_blocks=1, n_heads=2, max_seq=8, seed=11)
    x, y = make_toy_batch(8, 8, np.random.default_rng(0), "arith")
    first = model.loss(x, y)

    def batch_fn(r):
        return make_toy_batch(8, 8, r, "arith")

    with tempfile.TemporaryDirectory() as td:
        log = Path(td) / "nested" / "log.csv"
        _, loss, _, step = train_steps(model, steps=60, batch_fn=batch_fn, seq_len=8, lr=1e-2,
                                       warmup_steps=5, schedule="cosine", grad_accum_steps=2,
                                       log_every=20, log_csv=str(log))
        rows = log.read_text().strip().splitlines()
        assert step == 60
        assert rows[0] == "step,train_loss,val_loss,grad_norm,lr"
        assert len(rows) == 5, rows
    last = model.loss(x, y)
    assert last < 0.7 * first, f"toy loss did not drop: {first:.3f} -> {last:.3f}"
    print(f"Toy arith training with cosine schedule + grad accumulation OK: loss {first:.3f} -> {last:.3f}")


def tie_and_final_norm_gradient_test():
    """Finite-difference check with tied embeddings and with the final norm, both norm types (float64)."""
    x = np.array([[BOS_ID, 97, 98]], dtype=np.int64)
    y = np.array([[97, 98, 99]], dtype=np.int64)
    eps, rng = 1e-6, np.random.default_rng(555)
    cases = [dict(merge_method="average", use_rms=True, tie_embeddings=True),
             dict(merge_method="average", use_rms=False, tie_embeddings=True),
             dict(merge_method="concat", use_rms=True, tie_embeddings=False),
             dict(merge_method="single", use_rms=False, tie_embeddings=False, scaled_init=False)]
    checked = 0
    for case in cases:
        model = DualTransformerNP(d_model=8, n_blocks=1, n_heads=2, max_seq=8, seed=555,
                                  dtype=np.float64, **case)
        _, grads = model.loss_and_grads(x, y)
        for name, arr in model.params.items():
            for _ in range(min(arr.size, 3)):
                idx = np.unravel_index(int(rng.integers(arr.size)), arr.shape)
                orig = arr[idx]
                arr[idx] = orig + eps
                lp = model.loss(x, y)
                arr[idx] = orig - eps
                lm = model.loss(x, y)
                arr[idx] = orig
                numeric = (lp - lm) / (2 * eps)
                analytic = float(grads[name][idx])
                abs_err = abs(analytic - numeric)
                rel_err = abs_err / max(abs(analytic), abs(numeric), 1e-12)
                if not (abs_err < 1e-7 or rel_err < 1e-4):
                    raise AssertionError(f"{case} {name}{idx}: analytic={analytic:.12g}, numeric={numeric:.12g}")
                checked += 1
    print(f"Tied-embedding / final-norm gradient checks OK: {checked} sampled entries")


def tokenizer_and_checkpoint_meta_test():
    import tempfile
    tok = CharTokenizer.from_text("你好，世界！Hello")
    text = "世界你好"
    assert tok.decode(tok.encode(text, add_bos=True, add_eos=True)) == text
    assert tok.encode("猫")[0] == CharTokenizer.UNK          # unseen character -> UNK
    assert tok.decode(tok.encode("猫"), skip_special=False) == "<UNK>"
    assert ByteTokenizer().to_meta() == {"type": "byte"}
    model = DualTransformerNP(d_model=8, n_blocks=1, n_heads=2, max_seq=8, seed=7,
                              vocab_size=tok.vocab_size, tie_embeddings=True)
    assert model.num_params() < DualTransformerNP(d_model=8, n_blocks=1, n_heads=2, max_seq=8,
                                                  seed=7, vocab_size=tok.vocab_size).num_params()
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "ck.npz"
        ids = tok.encode("你好世界" * 10, add_bos=True, add_eos=True)
        train_steps(model, ids, steps=2, batch_size=2, seq_len=4, lr=1e-3, log_every=1,
                    checkpoint_path=path, extra_meta={"tokenizer": tok.to_meta()})
        meta = read_checkpoint(path)["meta"]
        restored = tokenizer_from_meta(meta)
        assert isinstance(restored, CharTokenizer) and restored.vocab_size == tok.vocab_size
        loaded, _ = DualTransformerNP.load_checkpoint(path)
        assert loaded.vocab_size == tok.vocab_size and loaded.tie_embeddings
        xx = np.array([ids[:4]], dtype=np.int64)
        assert np.array_equal(model.forward(xx), loaded.forward(xx))
    print("Char tokenizer, tokenizer metadata in checkpoint, tied-weight round trip OK")


def extended_modes_test():
    """Finite-difference checks for the consolidated options: GQA, SwiGLU,
    learned positions, and combinations (float64), plus GQA KV-cache parity."""
    rng = np.random.default_rng(999)
    x = np.array([[BOS_ID, 97, 98]], dtype=np.int64)
    y = np.array([[97, 98, 99]], dtype=np.int64)
    eps = 1e-6
    cases = [
        dict(merge_method="single", n_kv_heads=1, ff_mode="swiglu", pos_mode="rope", use_rms=True),
        dict(merge_method="average", n_kv_heads=1, ff_mode="gelu", pos_mode="learned", use_rms=False),
        dict(merge_method="average", n_kv_heads=2, ff_mode="swiglu", pos_mode="learned",
             use_rms=True, tie_embeddings=True),
        dict(merge_method="concat", n_kv_heads=1, ff_mode="swiglu", pos_mode="rope", use_rms=False),
    ]
    checked = 0
    for case in cases:
        model = DualTransformerNP(d_model=8, n_blocks=1, n_heads=2, max_seq=8,
                                  seed=999, dtype=np.float64, **case)
        _, grads = model.loss_and_grads(x, y)
        assert set(grads) == set(model.params)
        for name, arr in model.params.items():
            for _ in range(min(arr.size, 3)):
                idx = np.unravel_index(int(rng.integers(arr.size)), arr.shape)
                orig = arr[idx]
                arr[idx] = orig + eps
                lp = model.loss(x, y)
                arr[idx] = orig - eps
                lm = model.loss(x, y)
                arr[idx] = orig
                numeric = (lp - lm) / (2 * eps)
                analytic = float(grads[name][idx])
                abs_err = abs(analytic - numeric)
                rel_err = abs_err / max(abs(analytic), abs(numeric), 1e-12)
                if not (abs_err < 1e-7 or rel_err < 1e-4):
                    raise AssertionError(f"{case} {name}{idx}: analytic={analytic:.12g}, numeric={numeric:.12g}")
                checked += 1
    print(f"Extended-mode gradient checks OK: {checked} sampled entries (GQA/SwiGLU/learned-pos combos)")

    model = DualTransformerNP(d_model=16, n_blocks=2, n_heads=4, merge_method="average",
                              n_kv_heads=2, ff_mode="swiglu", pos_mode="learned",
                              max_seq=6, seed=78, dtype=np.float64)
    tokens = np.array([[BOS_ID, 97, 98, 99, 100, EOS_ID]], dtype=np.int64)
    full = model.forward(tokens)
    pasts, parts = None, []
    for i in range(tokens.shape[1]):
        part, pasts = model.forward_cached(tokens[:, i:i + 1], pasts)
        parts.append(part)
    cached = np.concatenate(parts, axis=1)
    max_err = float(np.abs(full - cached).max())
    if not np.allclose(full, cached, atol=1e-10, rtol=1e-10):
        raise AssertionError(f"GQA KV cache parity failed: max_err={max_err}")
    print(f"Extended-mode KV parity OK (GQA+learned): max_logit_error={max_err:.2e}")


def decode_control_test():
    """Entropy bounds, loop detection, no-repeat blocking, controller direction and bounds."""
    V = 12
    assert abs(token_entropy(np.zeros(V)) - math.log(V)) < 1e-9
    peaked = np.full(V, -50.0)
    peaked[3] = 0.0
    assert token_entropy(peaked) < 1e-6
    assert suffix_repeats([1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8], 8)
    assert not suffix_repeats(list(range(16)), 8)
    assert banned_next_tokens([1, 2, 3, 4, 1, 2, 3], 4) == {4}
    assert banned_next_tokens([1, 2, 3], 4) == set()

    # The controller must refuse the token that would complete a repeated 4-gram.
    ctrl = DecodeController(base_temperature=0.0, block_ngram=4, loop_len=0)
    logits = np.zeros(V)
    logits[8] = 10.0
    assert ctrl.choose(logits, [5, 6, 7, 8, 5, 6, 7], rng=None) != 8

    # Temperature direction: collapsed (peaked) output pushes T up, drifting (uniform) pushes it down.
    up = DecodeController(base_temperature=0.5, target_entropy=1.0, block_ngram=0, loop_len=0,
                          t_min=0.1, t_max=3.0)
    for _ in range(5):
        up.choose(peaked, [0], rng=np.random.default_rng(0))
    assert up.temp > 0.5, up.temp
    down = DecodeController(base_temperature=0.5, target_entropy=1.0, block_ngram=0, loop_len=0,
                            t_min=0.1, t_max=3.0)
    for _ in range(5):
        down.choose(np.zeros(V), [0], rng=np.random.default_rng(0))
    assert down.temp < 0.5, down.temp

    # Greedy controlled generation: no 4-gram may repeat anywhere in the output (guaranteed by blocking).
    model = DualTransformerNP(d_model=16, n_blocks=2, n_heads=4, merge_method="average",
                              max_seq=8, seed=79, dtype=np.float64)
    prompt = [BOS_ID, 97, 98, 99]
    out = model.generate_cached(prompt, 40, controller=DecodeController(0.0, block_ngram=4))
    assert len(out) == len(prompt) + 40
    grams = [tuple(out[i:i + 4]) for i in range(len(out) - 3)]
    assert len(set(grams)) == len(grams), "repeated 4-gram despite no-repeat blocking"

    # Sampled controlled generation: correct length, temperatures stay within bounds, deterministic.
    c1 = DecodeController(base_temperature=0.9, target_entropy=2.0, block_ngram=4, loop_len=8)
    o1 = model.generate_cached(prompt, 40, temperature=0.9, rng=np.random.default_rng(3), controller=c1)
    c2 = DecodeController(base_temperature=0.9, target_entropy=2.0, block_ngram=4, loop_len=8)
    o2 = model.generate_cached(prompt, 40, temperature=0.9, rng=np.random.default_rng(3), controller=c2)
    assert len(o1) == len(prompt) + 40 and o1 == o2
    assert all(c1.t_min - 1e-9 <= t[1] <= c1.t_max + 1e-9 for t in c1.trace)
    print(f"Decode control test OK: {c1.summary()}")


def smoke_test():
    tok = ByteTokenizer()
    sample = "你好，Omega!\n"
    assert tok.decode(tok.encode(sample)) == sample
    assert tok.decode(tok.encode(sample, add_bos=True, add_eos=True)) == sample
    assert tok.encode("x", add_bos=True, add_eos=True) == [BOS_ID, ord("x"), EOS_ID]
    for merge in MERGE_METHODS:
        model = DualTransformerNP(d_model=32, n_blocks=2, n_heads=4, merge_method=merge,
                                  max_seq=64, seed=0)
        x = np.array([[BOS_ID, *tok.encode("abc")]], dtype=np.int64)
        logits = model.forward(x)
        assert logits.shape == (1, x.shape[1], VOCAB_SIZE)
        loss, grads = model.loss_and_grads(x[:, :-1], x[:, 1:])
        assert math.isfinite(loss)
        assert set(grads) == set(model.params)
        assert all(np.all(np.isfinite(g)) for g in grads.values())
        assert any(np.abs(g).sum() > 0 for g in grads.values())
        generated = model.generate(x[0].tolist(), max_new_tokens=3)
        assert len(generated) == x.shape[1] + 3
        print(f"{merge}: forward/backward/generate OK; params={model.num_params():,}; loss={loss:.4f}")
    print("Tokenizer UTF-8 round-trip OK")
    numerical_gradient_test()
    checkpoint_round_trip_test()
    training_validation_resume_test()
    no_validation_resume_test()
    kv_cache_parity_test()
    sliding_window_generation_test()
    learning_sanity_test()
    label_smoothing_gradient_test()
    sampling_and_schedule_test()
    toy_training_test()
    tie_and_final_norm_gradient_test()
    tokenizer_and_checkpoint_meta_test()
    extended_modes_test()
    decode_control_test()
    print("ALL SMOKE TESTS PASSED")


def generalization_report(model, train_ids, held_ids, seq_len, batch_size, seed, max_batches=64):
    """Compare trained vs untrained loss on training text and on unseen held-out text."""
    untrained = DualTransformerNP(**model.config(), seed=seed)
    kw = dict(seq_len=seq_len, batch_size=batch_size, max_batches=max_batches)
    base_held = evaluate_loss(untrained, held_ids, **kw)
    trained_train = evaluate_loss(model, train_ids, **kw)
    trained_held = evaluate_loss(model, held_ids, **kw)
    print("---- generalization (lower is better) ----")
    print(f"untrained, held-out text : {base_held:.4f}")
    print(f"trained,   training text : {trained_train:.4f}")
    print(f"trained,   held-out text : {trained_held:.4f}")
    print(f"held-out improvement     : {base_held - trained_held:.4f}")
    print(f"held-out minus training   : {trained_held - trained_train:.4f}  (large = memorizing)")
    return {"untrained_held": base_held, "trained_train": trained_train, "trained_held": trained_held}


def make_controller(args):
    """None when no decode control was requested, so default sampling is unchanged."""
    if not (args.entropy_control or args.no_repeat_ngram):
        return None
    return DecodeController(base_temperature=args.temperature, target_entropy=args.target_entropy,
                            block_ngram=args.no_repeat_ngram,
                            loop_len=8 if args.entropy_control else 0,
                            adapt=args.entropy_control)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--train-text", type=str)
    ap.add_argument("--task", choices=("corpus", "arith", "copy"), default="corpus",
                    help="corpus: train on --train-text; arith/copy: synthetic tasks over IDs 0-15")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=1, help="micro-batches per optimizer update")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--schedule", choices=("constant", "cosine"), default="constant")
    ap.add_argument("--warmup-steps", type=int, default=0)
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--log-csv", type=str, default=None)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--merge", choices=MERGE_METHODS, default="average")
    ap.add_argument("--norm", choices=("rms", "layer"), default="rms")
    ap.add_argument("--max-seq", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--checkpoint", type=str, default="dual_transformer.npz",
                    help="best-validation weights; latest state goes to <stem>_last<suffix>")
    ap.add_argument("--resume", action="store_true",
                    help="resume from the latest-state checkpoint (architecture comes from it)")
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--prompt", type=str, default="你好")
    ap.add_argument("--new-tokens", type=int, default=100)
    ap.add_argument("--temperature", type=float, default=0.0, help="0 = greedy")
    ap.add_argument("--top-k", type=int, default=0, help="0 = disabled")
    ap.add_argument("--top-p", type=float, default=1.0, help="1.0 = disabled")
    ap.add_argument("--tokenizer", choices=("byte", "char"), default="byte",
                    help="byte: UTF-8 bytes (vocab 258); char: characters of --train-text")
    ap.add_argument("--tie-embeddings", action="store_true", help="share input embedding and output head (not with concat)")
    ap.add_argument("--no-final-norm", action="store_true", help="disable the norm before the output head")
    ap.add_argument("--no-scaled-init", action="store_true", help="disable 1/sqrt(2*blocks) residual init scaling")
    ap.add_argument("--held-out", type=str, default=None,
                    help="optional text file never used for training; reports generalization vs an untrained model")
    ap.add_argument("--entropy-control", action="store_true",
                    help="adapt temperature toward --target-entropy and detect loops")
    ap.add_argument("--target-entropy", type=float, default=1.0, help="nats; used with --entropy-control")
    ap.add_argument("--no-repeat-ngram", type=int, default=0,
                    help="block any token completing an already-seen n-gram (0 = off)")
    ap.add_argument("--kv-heads", type=int, default=None,
                    help="KV heads for grouped-query attention (default: same as --heads)")
    ap.add_argument("--pos", choices=("rope", "learned"), default="rope",
                    help="position scheme: RoPE or learned position embeddings")
    ap.add_argument("--ff", choices=("gelu", "swiglu"), default="gelu",
                    help="feed-forward: GELU MLP or SwiGLU")
    ap.add_argument("--init", choices=("gpt2", "legacy"), default="gpt2",
                    help="gpt2: N(0,0.02) initialization (tie-safe); legacy: original initialization")
    args = ap.parse_args()
    if args.test:
        smoke_test()
        return
    if args.grad_accum < 1:
        ap.error("--grad-accum must be >= 1")
    if not 0.0 <= args.label_smoothing < 1.0:
        ap.error("--label-smoothing must be in [0, 1)")
    if args.temperature < 0:
        ap.error("--temperature must be >= 0")
    if args.top_k < 0:
        ap.error("--top-k must be >= 0")
    if not 0.0 < args.top_p <= 1.0:
        ap.error("--top-p must be in (0, 1]")
    if args.no_repeat_ngram < 0:
        ap.error("--no-repeat-ngram must be >= 0")
    sample_rng = np.random.default_rng(args.seed)

    if args.task != "corpus":
        if args.resume or args.held_out:
            ap.error("--resume and --held-out apply only to --task corpus")
        model = DualTransformerNP(args.d_model, args.blocks, args.heads, args.merge,
                                  args.norm == "rms", args.max_seq, seed=args.seed,
                                  n_kv_heads=args.kv_heads, pos_mode=args.pos,
                                  ff_mode=args.ff, init_style=args.init)

        def toy_batch(r):
            return make_toy_batch(args.batch_size, args.seq_len, r, args.task)

        _, loss, _, completed_step = train_steps(
            model, steps=args.steps, batch_size=args.batch_size, seq_len=args.seq_len, lr=args.lr,
            checkpoint_path=args.checkpoint, seed=args.seed, weight_decay=args.weight_decay,
            max_grad_norm=args.max_grad_norm, label_smoothing=args.label_smoothing,
            grad_accum_steps=args.grad_accum, schedule=args.schedule,
            warmup_steps=args.warmup_steps, log_csv=args.log_csv, batch_fn=toy_batch)
        print(f"toy task={args.task}: completed_step={completed_step}; final_train_loss={loss:.4f}")
        prompt = [0, 1, 2, 3]
        ctrl = make_controller(args)
        out = model.generate_cached(prompt, 8, temperature=args.temperature, rng=sample_rng,
                                    top_k=args.top_k, top_p=args.top_p, controller=ctrl)
        if ctrl is not None:
            print(ctrl.summary())
        print(f"prompt={prompt} output={out}")
        return

    if not args.train_text:
        ap.error("provide --train-text corpus.txt, --task arith/copy, or --test")
    if not 0.0 < args.val_fraction < 0.5:
        ap.error("--val-fraction must be between 0 and 0.5")
    if args.tie_embeddings and args.merge == "concat":
        ap.error("--tie-embeddings cannot be used with --merge concat")
    text = Path(args.train_text).read_text(encoding="utf-8")
    resume_path = None
    if args.resume:
        resume_path = last_checkpoint_path(args.checkpoint)
        if not resume_path.exists():
            ap.error(f"no latest-state checkpoint at {resume_path}")
        tokenizer = tokenizer_from_meta(read_checkpoint(resume_path)["meta"])
    elif args.tokenizer == "char":
        tokenizer = CharTokenizer.from_text(text)
    else:
        tokenizer = ByteTokenizer()
    ids = tokenizer.encode(text, add_bos=True, add_eos=True)
    split = int(len(ids) * (1.0 - args.val_fraction))
    train_ids, val_ids = ids[:split], ids[split:]
    if len(train_ids) < args.seq_len + 2 or len(val_ids) < args.seq_len + 1:
        ap.error("corpus too short for requested --seq-len and --val-fraction; use more text or a shorter sequence")
    if args.resume:
        model, _ = DualTransformerNP.load_checkpoint(resume_path)
    else:
        model = DualTransformerNP(args.d_model, args.blocks, args.heads, args.merge,
                                  args.norm == "rms", args.max_seq, seed=args.seed,
                                  vocab_size=tokenizer.vocab_size,
                                  final_norm=not args.no_final_norm,
                                  tie_embeddings=args.tie_embeddings,
                                  scaled_init=not args.no_scaled_init,
                                  n_kv_heads=args.kv_heads, pos_mode=args.pos,
                                  ff_mode=args.ff, init_style=args.init)
    if model.vocab_size != tokenizer.vocab_size:
        ap.error("checkpoint vocabulary does not match its stored tokenizer")
    _, loss, best_val, completed_step = train_steps(
        model, train_ids, args.steps, args.batch_size, args.seq_len, lr=args.lr,
        val_token_ids=val_ids, checkpoint_path=args.checkpoint,
        eval_every=args.eval_every, resume_from=resume_path, seed=args.seed,
        weight_decay=args.weight_decay, max_grad_norm=args.max_grad_norm,
        label_smoothing=args.label_smoothing, grad_accum_steps=args.grad_accum,
        schedule=args.schedule, warmup_steps=args.warmup_steps, log_csv=args.log_csv,
        extra_meta={"tokenizer": tokenizer.to_meta()})
    print(f"checkpoint saved: {args.checkpoint} (best), {last_checkpoint_path(args.checkpoint)} (latest); "
          f"completed_step={completed_step}; final_train_loss={loss:.4f}; best_val_loss={best_val:.4f}")
    if args.held_out:
        held_text = Path(args.held_out).read_text(encoding="utf-8")
        held_ids = tokenizer.encode(held_text, add_bos=True, add_eos=True)
        if len(held_ids) < args.seq_len + 1:
            ap.error("held-out text is shorter than --seq-len + 1")
        generalization_report(model, train_ids, held_ids, args.seq_len, args.batch_size, args.seed)
    prompt = tokenizer.encode(args.prompt, add_bos=True)
    ctrl = make_controller(args)
    output = model.generate_cached(prompt, args.new_tokens, temperature=args.temperature,
                                   rng=sample_rng, top_k=args.top_k, top_p=args.top_p,
                                   controller=ctrl)
    if ctrl is not None:
        print(ctrl.summary())
    print(tokenizer.decode(output))


if __name__ == "__main__":
    main()

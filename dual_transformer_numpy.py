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
- Byte-level UTF-8 tokenizer (no external vocabulary files)
- AdamW with decoupled weight decay and global-norm gradient clipping
- Next-token training, greedy/temperature generation, KV-cached generation
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
"""
from __future__ import annotations

import argparse
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


def _attn_fwd(h, params, pre, n_heads, offset, past):
    B, S, D = h.shape
    H, Dh = n_heads, D // n_heads
    qkv = h @ params[pre + "attn.qkv.weight"].T
    q, k, v = np.split(qkv, 3, axis=-1)

    def heads(z):
        return z.reshape(B, S, H, Dh).transpose(0, 2, 1, 3)

    q, k, v = heads(q), heads(k), heads(v)
    pos = np.arange(offset, offset + S)
    cos, sin = _rope_tables(pos, Dh, q.dtype)
    q, k = _rope(q, cos, sin), _rope(k, cos, sin)
    if past is not None:
        k_all = np.concatenate([past[0], k], axis=2)
        v_all = np.concatenate([past[1], v], axis=2)
    else:
        k_all, v_all = k, v
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
    cache = (h, q, k_all, v_all, P, o2, cos, sin, scale)
    return out, cache, (k_all, v_all)


def _attn_bwd(dout, cache, params, grads, pre):
    h, q, k, v, P, o2, cos, sin, scale = cache
    B, S, D = h.shape
    H, Dh = q.shape[1], q.shape[3]
    grads[pre + "attn.proj.weight"] += dout.reshape(-1, D).T @ o2.reshape(-1, D)
    do = (dout @ params[pre + "attn.proj.weight"]).reshape(B, S, H, Dh).transpose(0, 2, 1, 3)
    dP = do @ v.transpose(0, 1, 3, 2)
    dv = P.transpose(0, 1, 3, 2) @ do
    dS = P * (dP - np.sum(dP * P, axis=-1, keepdims=True)) * scale
    dq = _rope_bwd(dS @ k, cos, sin)
    dk = _rope_bwd(dS.transpose(0, 1, 3, 2) @ q, cos, sin)

    def merge(z):
        return z.transpose(0, 2, 1, 3).reshape(B, S, D)

    dqkv = np.concatenate([merge(dq), merge(dk), merge(dv)], axis=-1)
    grads[pre + "attn.qkv.weight"] += dqkv.reshape(-1, 3 * D).T @ h.reshape(-1, D)
    return dqkv @ params[pre + "attn.qkv.weight"]


def _block_fwd(params, pre, x, n_heads, use_rms, offset, past):
    n1, c_n1 = _norm_fwd(x, params, pre + "norm1", use_rms)
    a, c_a, kv = _attn_fwd(n1, params, pre, n_heads, offset, past)
    x1 = x + a
    n2, c_n2 = _norm_fwd(x1, params, pre + "norm2", use_rms)
    f1 = _linear_fwd(n2, params, pre + "ff1")
    g, t = _gelu_fwd(f1)
    f2 = _linear_fwd(g, params, pre + "ff2")
    y = x1 + f2
    cache = (c_n1, c_a, c_n2, n2, f1, g, t)
    return y, cache, kv


def _block_bwd(dy, cache, params, grads, pre, use_rms):
    c_n1, c_a, c_n2, n2, f1, g, t = cache
    dg = _linear_bwd(dy, g, params, grads, pre + "ff2")
    df1 = _gelu_bwd(dg, f1, t)
    dn2 = _linear_bwd(df1, n2, params, grads, pre + "ff1")
    dx1 = dy + _norm_bwd(dn2, c_n2, params, grads, pre + "norm2", use_rms)
    dn1 = _attn_bwd(dx1, c_a, params, grads, pre)
    return dx1 + _norm_bwd(dn1, c_n1, params, grads, pre + "norm1", use_rms)


def _cross_entropy(logits, targets):
    B, S, V = logits.shape
    flat = logits.reshape(-1, V)
    t = targets.reshape(-1)
    N = flat.shape[0]
    idx = np.arange(N)
    m = flat.max(axis=1, keepdims=True)
    e = np.exp(flat - m)
    s = e.sum(axis=1, keepdims=True)
    loss = -(flat - m - np.log(s))[idx, t].mean()
    p = e / s
    p[idx, t] -= 1.0
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
            params[k] = params[k] * (1.0 - self.lr * self.weight_decay) \
                - self.lr * mhat / (np.sqrt(vhat) + self.eps)

    def load_state(self, m, v, t):
        self.m = {k: np.array(x) for k, x in m.items()}
        self.v = {k: np.array(x) for k, x in v.items()}
        self.t = int(t)


def _pick_token(logits, temperature: float, rng) -> int:
    if temperature == 0:
        return int(np.argmax(logits))
    z = logits.astype(np.float64) / temperature
    p = np.exp(z - z.max())
    p /= p.sum()
    rng = rng if rng is not None else np.random.default_rng(0)
    return int(rng.choice(len(p), p=p))


class DualTransformerNP:
    def __init__(self, d_model=64, n_blocks=2, n_heads=4,
                 merge_method="average", use_rms=True, max_seq=512,
                 seed=0, dtype=np.float32):
        if merge_method not in MERGE_METHODS:
            raise ValueError(f"merge_method must be one of {MERGE_METHODS}")
        if d_model <= 0 or n_blocks <= 0 or max_seq <= 0:
            raise ValueError("d_model, n_blocks, max_seq must be positive")
        if d_model % n_heads or (d_model // n_heads) % 2:
            raise ValueError("d_model/n_heads must be even")
        self.d_model, self.max_seq = d_model, max_seq
        self.n_blocks, self.n_heads, self.use_rms = n_blocks, n_heads, use_rms
        self.merge_method = merge_method
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
        }

    def _branches(self):
        return ("left",) if self.merge_method == "single" else ("left", "right")

    def _init_params(self, seed):
        rng = np.random.default_rng(seed)
        D, V, dt = self.d_model, VOCAB_SIZE, self.dtype
        p = {"emb": rng.standard_normal((V, D)).astype(dt)}

        def linear(name, out_f, in_f, bias):
            bound = 1.0 / math.sqrt(in_f)
            p[name + ".weight"] = rng.uniform(-bound, bound, (out_f, in_f)).astype(dt)
            if bias:
                p[name + ".bias"] = rng.uniform(-bound, bound, (out_f,)).astype(dt)

        for name in self._branches():
            for i in range(self.n_blocks):
                pre = f"{name}.{i}."
                p[pre + "norm1.weight"] = np.ones(D, dt)
                if not self.use_rms:
                    p[pre + "norm1.bias"] = np.zeros(D, dt)
                linear(pre + "attn.qkv", 3 * D, D, bias=False)
                linear(pre + "attn.proj", D, D, bias=False)
                p[pre + "norm2.weight"] = np.ones(D, dt)
                if not self.use_rms:
                    p[pre + "norm2.bias"] = np.zeros(D, dt)
                linear(pre + "ff1", 4 * D, D, bias=True)
                linear(pre + "ff2", D, 4 * D, bias=True)
        out_dim = 2 * D if self.merge_method == "concat" else D
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
        if tokens.min() < 0 or tokens.max() >= VOCAB_SIZE:
            raise ValueError(f"token IDs must be in [0, {VOCAB_SIZE - 1}]")

    def _forward(self, tokens, pasts=None):
        B, S = tokens.shape
        offset = 0
        if pasts is not None and pasts.get("left") and pasts["left"][0] is not None:
            offset = pasts["left"][0][0].shape[2]
        if offset + S > self.max_seq:
            raise ValueError("KV cache plus input chunk exceeds max_seq; reset/re-prefill cache")
        x = self.params["emb"][tokens]
        outs, caches, new_kv = {}, {}, {}
        for name in self._branches():
            h = x
            bc, kvs = [], []
            for i in range(self.n_blocks):
                past = pasts[name][i] if pasts is not None else None
                h, cache, kv = _block_fwd(self.params, f"{name}.{i}.", h,
                                          self.n_heads, self.use_rms, offset, past)
                bc.append(cache)
                kvs.append(kv)
            outs[name] = h
            caches[name] = bc
            new_kv[name] = kvs
        merged = self._merge(outs)
        logits = merged @ self.params["lm_head.weight"].T
        return logits, merged, caches, new_kv

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
        if tokens.min() < 0 or tokens.max() >= VOCAB_SIZE:
            raise ValueError(f"token IDs must be in [0, {VOCAB_SIZE - 1}]")
        logits, _, _, new_kv = self._forward(tokens, pasts)
        return logits, new_kv

    def loss(self, tokens, targets) -> float:
        logits = self.forward(tokens)
        return _cross_entropy(logits, np.asarray(targets, dtype=np.int64))[0]

    def loss_and_grads(self, tokens, targets):
        tokens = np.asarray(tokens, dtype=np.int64)
        targets = np.asarray(targets, dtype=np.int64)
        self._check_tokens(tokens)
        D = self.d_model
        logits, merged, caches, _ = self._forward(tokens)
        loss, dlogits = _cross_entropy(logits, targets)
        grads = _zeros_like_params(self.params)
        V = VOCAB_SIZE
        out_dim = merged.shape[-1]
        grads["lm_head.weight"] += dlogits.reshape(-1, V).T @ merged.reshape(-1, out_dim)
        dmerged = dlogits @ self.params["lm_head.weight"]

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
        return loss, grads

    @staticmethod
    def _check_prompt(prompt_ids):
        if not prompt_ids:
            raise ValueError("prompt_ids must not be empty")
        if any(int(t) < 0 or int(t) >= VOCAB_SIZE for t in prompt_ids):
            raise ValueError("prompt contains invalid token IDs")

    def generate(self, prompt_ids, max_new_tokens=100, temperature=0.0, rng=None):
        """Full-context greedy/temperature generation over a sliding window."""
        self._check_prompt(prompt_ids)
        if max_new_tokens < 0 or temperature < 0:
            raise ValueError("max_new_tokens must be non-negative and temperature >= 0")
        result = [int(t) for t in prompt_ids]
        for _ in range(max_new_tokens):
            ctx = np.array([result[-self.max_seq:]], dtype=np.int64)
            logits = self.forward(ctx)[0, -1]
            result.append(_pick_token(logits, temperature, rng))
        return result

    def generate_cached(self, prompt_ids, max_new_tokens=100, temperature=0.0, rng=None):
        """KV-cached generation; the cache is rebuilt when the sliding window rolls."""
        self._check_prompt(prompt_ids)
        if max_new_tokens < 0 or temperature < 0:
            raise ValueError("max_new_tokens must be non-negative and temperature >= 0")
        result = [int(t) for t in prompt_ids]
        if max_new_tokens == 0:
            return result
        ctx = result[-self.max_seq:]
        logits, pasts = self.forward_cached(np.array([ctx], dtype=np.int64))
        for generated_i in range(max_new_tokens):
            nxt = _pick_token(logits[0, -1], temperature, rng)
            result.append(nxt)
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


def train_steps(model: DualTransformerNP, token_ids, steps=500, batch_size=8, seq_len=64,
                lr=3e-4, log_every=50, val_token_ids=None, checkpoint_path=None,
                eval_every=50, resume_from=None, seed=0, weight_decay=0.01, max_grad_norm=1.0):
    """Train with optional validation and checkpointing.

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
    data = np.asarray(token_ids, dtype=np.int64)
    if len(data) < seq_len + 2:
        raise ValueError(f"training text needs at least {seq_len + 2} tokens")
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

    if resume_from:
        ck = read_checkpoint(resume_from)
        meta = ck["meta"]
        if meta["config"] != model.config():
            raise ValueError(f"resume checkpoint config {meta['config']} "
                             f"does not match model config {model.config()}")
        if not meta["has_optimizer"]:
            raise ValueError("resume checkpoint has no optimizer state")
        if set(ck["params"]) != set(model.params):
            raise ValueError("resume checkpoint parameter names do not match the model")
        for k in model.params:
            model.params[k] = ck["params"][k].astype(model.dtype)
        optimizer.load_state(ck["m"], ck["v"], meta["optimizer_t"])
        start_step = int(meta["step"])
        best_val = float(meta["extra"].get("best_val_loss", math.inf))
        last_loss = float(meta["extra"].get("last_train_loss", math.inf))
        if meta.get("rng_state") is not None:
            rng.bit_generator.state = meta["rng_state"]
        print(f"resumed from step={start_step}; best_val_loss={best_val:.4f}")

    latest_path = last_checkpoint_path(checkpoint_path) if (checkpoint_path and val is not None) else None
    completed_step = start_step
    last_saved_step = None
    for step in range(start_step + 1, start_step + steps + 1):
        x, y = make_batch(data, batch_size, seq_len, rng)
        loss, grads = model.loss_and_grads(x, y)
        if not math.isfinite(loss):
            raise FloatingPointError(f"non-finite training loss at step {step}")
        grad_norm = clip_grad_norm(grads, max_grad_norm)
        if not math.isfinite(grad_norm):
            raise FloatingPointError(f"non-finite gradient norm at step {step}")
        optimizer.step(model.params, grads)
        completed_step, last_loss = step, float(loss)
        should_eval = val is not None and (step % eval_every == 0 or step == start_step + steps)
        if should_eval:
            val_loss = evaluate_loss(model, val, seq_len, batch_size)
            improved = val_loss < best_val
            if improved:
                best_val = val_loss
            print(f"step={step:5d} train_loss={last_loss:.4f} val_loss={val_loss:.4f} "
                  f"grad_norm={grad_norm:.4f}" + (" [best]" if improved else ""))
            extra = {"best_val_loss": float(best_val), "last_train_loss": float(last_loss)}
            rng_state = rng.bit_generator.state
            if checkpoint_path and improved:
                model.save_checkpoint(checkpoint_path, optimizer, step, extra, rng_state)
            if latest_path is not None:
                model.save_checkpoint(latest_path, optimizer, step, extra, rng_state)
                last_saved_step = step
        elif step == 1 or step % log_every == 0:
            print(f"step={step:5d} train_loss={last_loss:.4f} grad_norm={grad_norm:.4f}")

    if checkpoint_path:
        extra = {"best_val_loss": float(best_val), "last_train_loss": float(last_loss)}
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--train-text", type=str)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
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
    ap.add_argument("--held-out", type=str, default=None,
                    help="optional text file never used for training; reports generalization vs an untrained model")
    args = ap.parse_args()
    if args.test:
        smoke_test()
        return
    if not args.train_text:
        ap.error("provide --train-text corpus.txt or use --test")
    if not 0.0 < args.val_fraction < 0.5:
        ap.error("--val-fraction must be between 0 and 0.5")
    text = Path(args.train_text).read_text(encoding="utf-8")
    tokenizer = ByteTokenizer()
    ids = tokenizer.encode(text, add_bos=True, add_eos=True)
    split = int(len(ids) * (1.0 - args.val_fraction))
    train_ids, val_ids = ids[:split], ids[split:]
    if len(train_ids) < args.seq_len + 2 or len(val_ids) < args.seq_len + 1:
        ap.error("corpus too short for requested --seq-len and --val-fraction; use more text or a shorter sequence")
    resume_path = None
    if args.resume:
        resume_path = last_checkpoint_path(args.checkpoint)
        if not resume_path.exists():
            ap.error(f"no latest-state checkpoint at {resume_path}")
        model, _ = DualTransformerNP.load_checkpoint(resume_path)
    else:
        model = DualTransformerNP(args.d_model, args.blocks, args.heads, args.merge,
                                  args.norm == "rms", args.max_seq, seed=args.seed)
    _, loss, best_val, completed_step = train_steps(
        model, train_ids, args.steps, args.batch_size, args.seq_len, lr=args.lr,
        val_token_ids=val_ids, checkpoint_path=args.checkpoint,
        eval_every=args.eval_every, resume_from=resume_path, seed=args.seed)
    print(f"checkpoint saved: {args.checkpoint} (best), {last_checkpoint_path(args.checkpoint)} (latest); "
          f"completed_step={completed_step}; final_train_loss={loss:.4f}; best_val_loss={best_val:.4f}")
    if args.held_out:
        held_text = Path(args.held_out).read_text(encoding="utf-8")
        held_ids = tokenizer.encode(held_text, add_bos=True, add_eos=True)
        if len(held_ids) < args.seq_len + 1:
            ap.error("held-out text is shorter than --seq-len + 1")
        generalization_report(model, train_ids, held_ids, args.seq_len, args.batch_size, args.seed)
    prompt = tokenizer.encode(args.prompt, add_bos=True)
    output = model.generate_cached(prompt, args.new_tokens)
    print(tokenizer.decode(output))


if __name__ == "__main__":
    main()

"""
Modern Transformer (GQA + SwiGLU + RMSNorm + RoPE) - PURE NumPy version.
Same parameter layout / architecture as modern_dual_transformer.py, but with
NO numba import, so it runs in Pythonista on iOS (numpy only).
Attention is vectorised NumPy (KV heads repeated to Q heads) instead of the
Numba triple-loop kernel. Training: full hand-written backward (GQA /
RoPE / SwiGLU / RMSNorm / embedding / LM head) via loss_and_grads(), with
the same API shape as DualTransformerNP (loss_and_grads / loss), verified
by float64 numerical gradient checks (see check_modern_grads.py).
"""
import math
import numpy as np

LN_EPS = 1e-5
ROPE_THETA = 10000.0


def rmsnorm_fwd(x, w):
    inv = 1.0 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + LN_EPS)
    xn = x * inv
    return xn * w


def swiglu_fwd(x, w_gate, b_gate, w_up, b_up):
    h_gate = x @ w_gate.T + b_gate
    sig = 1.0 / (1.0 + np.exp(-h_gate))
    return (h_gate * sig) * (x @ w_up.T + b_up)


def rope_apply(x, cos, sin):
    # x: (B,H,S,Dh); cos/sin: (S, Dh/2)
    xe = x[..., 0::2]
    xo = x[..., 1::2]
    c = cos[None, None, :, :]
    s = sin[None, None, :, :]
    out = np.empty_like(x)
    out[..., 0::2] = xe * c - xo * s
    out[..., 1::2] = xe * s + xo * c
    return out


def _rms_fwd_2d(x2, w):
    """RMSNorm on rows of a 2D array. Returns (out, inv_rms (N,1), xn)."""
    inv = 1.0 / np.sqrt(np.mean(x2 * x2, axis=-1, keepdims=True) + LN_EPS)
    xn = x2 * inv
    return xn * w, inv, xn


def _rms_bwd_2d(dy2, x2, w, inv, xn):
    """Returns (dx2, grad_w) for _rms_fwd_2d."""
    n = x2.shape[-1]
    grad_w = np.sum(dy2 * xn, axis=0)
    dn = dy2 * w
    dx = inv * dn - x2 * (inv ** 3 / n) * np.sum(dn * x2, axis=-1, keepdims=True)
    return dx, grad_w


def rope_bwd(dout, cos, sin):
    """Transpose (inverse) of rope_apply: rotation by -angle per pair."""
    de = dout[..., 0::2]
    do = dout[..., 1::2]
    c = cos[None, None, :, :]
    s = sin[None, None, :, :]
    out = np.empty_like(dout)
    out[..., 0::2] = de * c + do * s
    out[..., 1::2] = -de * s + do * c
    return out


def quantize_weight_int8(w):
    maxv = np.max(np.abs(w), axis=1)
    scales = np.where(maxv > 0, maxv / 127.0, 1.0).astype(np.float32)
    q = np.clip(np.round(w / scales[:, None]), -128, 127).astype(np.int8)
    return q, scales


class ModernTransformerNumpy:
    """Parameter names/shapes identical to ModernDualTransformer (numba ver)."""

    def __init__(self, d_model=64, n_blocks=2, n_q_heads=8, n_kv_heads=2,
                 vocab_size=258, max_seq=512, is_quantized=False, seed=42):
        assert d_model % n_q_heads == 0 and n_q_heads % n_kv_heads == 0
        self.d_model, self.n_blocks = d_model, n_blocks
        self.n_q_heads, self.n_kv_heads = n_q_heads, n_kv_heads
        self.vocab_size, self.max_seq = vocab_size, max_seq
        self.is_quantized = is_quantized
        rng = np.random.default_rng(seed)
        D, V = d_model, vocab_size
        Dh = D // n_q_heads
        p = {}
        p["emb"] = (rng.standard_normal((V, D)) * 0.02).astype(np.float32)
        for b in range(n_blocks):
            pre = f"block.{b}."
            p[pre + "norm1.weight"] = np.ones(D, dtype=np.float32)
            p[pre + "norm2.weight"] = np.ones(D, dtype=np.float32)
            p[pre + "attn.q.weight"] = (rng.standard_normal((n_q_heads * Dh, D)) * 0.02).astype(np.float32)
            p[pre + "attn.k.weight"] = (rng.standard_normal((n_kv_heads * Dh, D)) * 0.02).astype(np.float32)
            p[pre + "attn.v.weight"] = (rng.standard_normal((n_kv_heads * Dh, D)) * 0.02).astype(np.float32)
            p[pre + "attn.proj.weight"] = (rng.standard_normal((D, D)) * 0.02).astype(np.float32)
            d_ffn = int(8 / 3 * D)
            p[pre + "ff1_gate.weight"] = (rng.standard_normal((d_ffn, D)) * 0.02).astype(np.float32)
            p[pre + "ff1_gate.bias"] = np.zeros(d_ffn, dtype=np.float32)
            p[pre + "ff1_up.weight"] = (rng.standard_normal((d_ffn, D)) * 0.02).astype(np.float32)
            p[pre + "ff1_up.bias"] = np.zeros(d_ffn, dtype=np.float32)
            p[pre + "ff2.weight"] = (rng.standard_normal((D, d_ffn)) * 0.02).astype(np.float32)
            p[pre + "ff2.bias"] = np.zeros(D, dtype=np.float32)
        p["final_norm.weight"] = np.ones(D, dtype=np.float32)
        p["lm_head.weight"] = (rng.standard_normal((V, D)) * 0.02).astype(np.float32)
        self.params = p
        self.q_params, self.q_scales = {}, {}
        if is_quantized:
            for k, v in p.items():
                if "weight" in k and v.ndim == 2:
                    self.q_params[k], self.q_scales[k] = quantize_weight_int8(v)

    def _linear(self, x, name, bias_name=None):
        if self.is_quantized and name in self.q_params:
            w = self.q_params[name].astype(np.float32) * self.q_scales[name][:, None]
            out = x @ w.T
        else:
            out = x @ self.params[name].T
        if bias_name and bias_name in self.params:
            out = out + self.params[bias_name]
        return out

    def forward(self, tokens, pasts=None):
        tokens = np.asarray(tokens)
        if tokens.min() < 0 or tokens.max() >= self.vocab_size:
            raise ValueError(f"token IDs must be in [0, {self.vocab_size - 1}]")
        B, S = tokens.shape
        D, Dh = self.d_model, self.d_model // self.n_q_heads
        x = self.params["emb"][tokens]
        offset = 0 if (pasts is None or pasts[0] is None) else pasts[0][0].shape[2]
        pos = np.arange(offset, offset + S)
        inv = 1.0 / (ROPE_THETA ** (np.arange(0, Dh, 2, dtype=np.float32) / Dh))
        ang = pos.astype(np.float32)[:, None] * inv[None, :]
        cos, sin = np.cos(ang), np.sin(ang)
        group = self.n_q_heads // self.n_kv_heads
        new_kvs = []
        for b in range(self.n_blocks):
            pre = f"block.{b}."
            h = rmsnorm_fwd(x, self.params[pre + "norm1.weight"])
            q = rope_apply(self._linear(h, pre + "attn.q.weight").reshape(B, S, self.n_q_heads, Dh).transpose(0, 2, 1, 3), cos, sin)
            k = rope_apply(self._linear(h, pre + "attn.k.weight").reshape(B, S, self.n_kv_heads, Dh).transpose(0, 2, 1, 3), cos, sin)
            v = self._linear(h, pre + "attn.v.weight").reshape(B, S, self.n_kv_heads, Dh).transpose(0, 2, 1, 3)
            if pasts is not None and pasts[b] is not None:
                k = np.concatenate([pasts[b][0], k], axis=2)
                v = np.concatenate([pasts[b][1], v], axis=2)
            new_kvs.append((k, v))
            Tk = k.shape[2]
            kk = np.repeat(k, group, axis=1)
            vv = np.repeat(v, group, axis=1)
            scores = (q @ kk.transpose(0, 1, 3, 2)) / math.sqrt(Dh)
            mask = (np.arange(Tk)[None, :] <= pos[:, None])  # (S, Tk)
            scores = np.where(mask[None, None, :, :], scores, -1e30)
            scores = scores - scores.max(axis=-1, keepdims=True)
            e = np.exp(scores) * mask[None, None, :, :]
            attn = (e / e.sum(axis=-1, keepdims=True)) @ vv
            x = x + self._linear(attn.transpose(0, 2, 1, 3).reshape(B, S, D), pre + "attn.proj.weight")
            h2 = rmsnorm_fwd(x, self.params[pre + "norm2.weight"])
            if self.is_quantized:
                # quantised path: FFN gate/up/ff2 also go through _linear here
                # (the numba draft bypassed quantisation for the FFN entirely)
                gate_in = h2.reshape(-1, D)
                # reuse swiglu math on dequantised weights via _linear on 2D
                hg = self._linear(gate_in, pre + "ff1_gate.weight", pre + "ff1_gate.bias")
                hu = self._linear(gate_in, pre + "ff1_up.weight", pre + "ff1_up.bias")
                sig = 1.0 / (1.0 + np.exp(-hg))
                ffn = (hg * sig) * hu
                proj = self._linear(ffn, pre + "ff2.weight", pre + "ff2.bias")
                x = x + proj.reshape(B, S, D)
            else:
                ffn = swiglu_fwd(h2.reshape(-1, D), self.params[pre + "ff1_gate.weight"],
                                 self.params[pre + "ff1_gate.bias"], self.params[pre + "ff1_up.weight"],
                                 self.params[pre + "ff1_up.bias"])
                x = x + (ffn @ self.params[pre + "ff2.weight"].T + self.params[pre + "ff2.bias"]).reshape(B, S, D)
        logits = self._linear(rmsnorm_fwd(x, self.params["final_norm.weight"]), "lm_head.weight")
        return logits, new_kvs

    # ------------------------------------------------------------------
    # Training: forward with activation cache + full analytic backward.
    # Float training path only (is_quantized must be False).
    # ------------------------------------------------------------------
    def num_params(self):
        return int(sum(v.size for v in self.params.values()))

    def _forward_train(self, tokens):
        if self.is_quantized:
            raise ValueError("training requires is_quantized=False")
        tokens = np.asarray(tokens, dtype=np.int64)
        if tokens.min() < 0 or tokens.max() >= self.vocab_size:
            raise ValueError(f"token IDs must be in [0, {self.vocab_size - 1}]")
        P = self.params
        B, S = tokens.shape
        D, Dh = self.d_model, self.d_model // self.n_q_heads
        group = self.n_q_heads // self.n_kv_heads
        x = P["emb"][tokens]
        pos = np.arange(S)
        invf = 1.0 / (ROPE_THETA ** (np.arange(0, Dh, 2, dtype=np.float64) / Dh))
        ang = pos.astype(np.float64)[:, None] * invf[None, :]
        cos = np.cos(ang).astype(x.dtype)
        sin = np.sin(ang).astype(x.dtype)
        mask = np.arange(S)[None, :] <= np.arange(S)[:, None]  # (S,S) causal
        caches = []
        for b in range(self.n_blocks):
            pre = f"block.{b}."
            c = {"x0": x, "cos": cos, "sin": sin}
            h, inv1, xn1 = _rms_fwd_2d(x.reshape(-1, D), P[pre + "norm1.weight"])
            c.update(inv1=inv1, xn1=xn1)
            h3 = h.reshape(B, S, D)
            q = (h3 @ P[pre + "attn.q.weight"].T).reshape(B, S, self.n_q_heads, Dh).transpose(0, 2, 1, 3)
            k = (h3 @ P[pre + "attn.k.weight"].T).reshape(B, S, self.n_kv_heads, Dh).transpose(0, 2, 1, 3)
            v = (h3 @ P[pre + "attn.v.weight"].T).reshape(B, S, self.n_kv_heads, Dh).transpose(0, 2, 1, 3)
            q = rope_apply(q, cos, sin)
            k = rope_apply(k, cos, sin)
            c.update(q=q, k=k, v=v)
            kk = np.repeat(k, group, axis=1)
            vv = np.repeat(v, group, axis=1)
            scores = (q @ kk.transpose(0, 1, 3, 2)) / math.sqrt(Dh)
            scores = np.where(mask[None, None], scores, -1e30)
            scores = scores - np.max(scores, axis=-1, keepdims=True)
            e = np.exp(scores) * mask[None, None]
            probs = e / np.sum(e, axis=-1, keepdims=True)
            c["probs"] = probs
            attn = probs @ vv  # (B,Hq,S,Dh)
            attn_r = attn.transpose(0, 2, 1, 3).reshape(B, S, D)
            c["attn_r"] = attn_r
            x1 = x + attn_r @ P[pre + "attn.proj.weight"].T
            c["x1"] = x1
            h2, inv2, xn2 = _rms_fwd_2d(x1.reshape(-1, D), P[pre + "norm2.weight"])
            c.update(inv2=inv2, xn2=xn2)
            hg = h2 @ P[pre + "ff1_gate.weight"].T + P[pre + "ff1_gate.bias"]
            hu = h2 @ P[pre + "ff1_up.weight"].T + P[pre + "ff1_up.bias"]
            sig = 1.0 / (1.0 + np.exp(-hg))
            swish = hg * sig
            mid = swish * hu
            c.update(h2=h2, hg=hg, hu=hu, sig=sig, swish=swish, mid=mid)
            x = x1 + (mid @ P[pre + "ff2.weight"].T + P[pre + "ff2.bias"]).reshape(B, S, D)
            caches.append(c)
        hf, invf_rms, xnf = _rms_fwd_2d(x.reshape(-1, D), P["final_norm.weight"])
        logits = (hf @ P["lm_head.weight"].T).reshape(B, S, self.vocab_size)
        cache = {"blocks": caches, "x_final": x, "hf": hf, "invf": invf_rms,
                 "xnf": xnf, "tokens": tokens}
        return logits, cache

    def loss_and_grads(self, tokens, targets):
        """Mean next-token cross-entropy and gradients for every parameter."""
        P = self.params
        logits, cache = self._forward_train(tokens)
        targets = np.asarray(targets, dtype=np.int64)
        B, S, V = logits.shape
        D, Dh = self.d_model, self.d_model // self.n_q_heads
        group = self.n_q_heads // self.n_kv_heads
        # --- cross entropy (stable) ---
        lg = logits.reshape(-1, V)
        mx = np.max(lg, axis=-1, keepdims=True)
        ex = np.exp(lg - mx)
        probs = ex / np.sum(ex, axis=-1, keepdims=True)
        n = B * S
        flat_t = targets.reshape(-1)
        loss = -float(np.sum(np.log(probs[np.arange(n), flat_t] + 1e-12)) / n)
        dlogits = probs
        dlogits[np.arange(n), flat_t] -= 1.0
        dlogits = (dlogits / n).reshape(B, S, V)
        grads = {k: np.zeros_like(v) for k, v in P.items()}
        # --- LM head + final RMSNorm ---
        grads["lm_head.weight"] += dlogits.reshape(-1, V).T @ cache["hf"]
        dhf = (dlogits @ P["lm_head.weight"]).reshape(-1, D)
        dx, gw = _rms_bwd_2d(dhf, cache["x_final"].reshape(-1, D),
                             P["final_norm.weight"], cache["invf"], cache["xnf"])
        grads["final_norm.weight"] += gw
        dx = dx.reshape(B, S, D)
        # --- blocks, reversed ---
        for b in reversed(range(self.n_blocks)):
            pre = f"block.{b}."
            c = cache["blocks"][b]
            # FFN branch: x_out = x1 + proj(mid); dx flows to both
            dmid = dx.reshape(-1, D) @ P[pre + "ff2.weight"]
            grads[pre + "ff2.weight"] += dx.reshape(-1, D).T @ c["mid"]
            grads[pre + "ff2.bias"] += np.sum(dx.reshape(-1, D), axis=0)
            dswish = dmid * c["hu"]
            dhu = dmid * c["swish"]
            dhg = dswish * (c["sig"] + c["swish"] * (1.0 - c["sig"]))
            grads[pre + "ff1_gate.weight"] += dhg.T @ c["h2"]
            grads[pre + "ff1_gate.bias"] += np.sum(dhg, axis=0)
            grads[pre + "ff1_up.weight"] += dhu.T @ c["h2"]
            grads[pre + "ff1_up.bias"] += np.sum(dhu, axis=0)
            dh2 = dhg @ P[pre + "ff1_gate.weight"] + dhu @ P[pre + "ff1_up.weight"]
            dx1_ffn, gw2 = _rms_bwd_2d(dh2, c["x1"].reshape(-1, D),
                                       P[pre + "norm2.weight"], c["inv2"], c["xn2"])
            grads[pre + "norm2.weight"] += gw2
            dx1 = dx + dx1_ffn.reshape(B, S, D)  # residual + FFN-norm path
            # Attention branch: x1 = x0 + attn_r @ Wo.T
            dattn_r = dx1
            grads[pre + "attn.proj.weight"] += dattn_r.reshape(-1, D).T @ c["attn_r"].reshape(-1, D)
            dattn = (dattn_r @ P[pre + "attn.proj.weight"]).reshape(B, S, self.n_q_heads, Dh).transpose(0, 2, 1, 3)
            probs = c["probs"]
            kk = np.repeat(c["k"], group, axis=1)
            vv = np.repeat(c["v"], group, axis=1)
            dprobs = dattn @ vv.transpose(0, 1, 3, 2)
            dvv = probs.transpose(0, 1, 3, 2) @ dattn
            dscores = probs * (dprobs - np.sum(dprobs * probs, axis=-1, keepdims=True))
            scale = 1.0 / math.sqrt(Dh)
            dq = (dscores @ kk) * scale
            dkk = (dscores.transpose(0, 1, 3, 2) @ c["q"]) * scale
            dk_rep = dkk.reshape(B, self.n_kv_heads, group, S, Dh).sum(axis=2)
            dv_rep = dvv.reshape(B, self.n_kv_heads, group, S, Dh).sum(axis=2)
            dq = rope_bwd(dq, c["cos"], c["sin"])
            dk = rope_bwd(dk_rep, c["cos"], c["sin"])
            dv = dv_rep
            h3 = c["xn1"] * P[pre + "norm1.weight"]  # normed input, (B*S, D)
            h3 = h3.reshape(B, S, D)
            dq_r = dq.transpose(0, 2, 1, 3).reshape(-1, self.n_q_heads * Dh)
            dk_r = dk.transpose(0, 2, 1, 3).reshape(-1, self.n_kv_heads * Dh)
            dv_r = dv.transpose(0, 2, 1, 3).reshape(-1, self.n_kv_heads * Dh)
            grads[pre + "attn.q.weight"] += dq_r.T @ h3.reshape(-1, D)
            grads[pre + "attn.k.weight"] += dk_r.T @ h3.reshape(-1, D)
            grads[pre + "attn.v.weight"] += dv_r.T @ h3.reshape(-1, D)
            dh = (dq_r @ P[pre + "attn.q.weight"] + dk_r @ P[pre + "attn.k.weight"]
                  + dv_r @ P[pre + "attn.v.weight"]).reshape(-1, D)
            dx0_attn, gw1 = _rms_bwd_2d(dh, c["x0"].reshape(-1, D),
                                        P[pre + "norm1.weight"], c["inv1"], c["xn1"])
            grads[pre + "norm1.weight"] += gw1
            dx = dx1 + dx0_attn.reshape(B, S, D)  # x1's grad to x0: residual path is dx1 itself? see below
            # NOTE: x1 = x0 + attn_out, so dL/dx0 gets dx1 (identity) + norm path.
            # dx above already = dx1 + norm-path; the line below is kept explicit:
            # (dx1 was added once, in `dx = dx1 + ...` -- do not add twice.)
        # --- embedding ---
        np.add.at(grads["emb"], cache["tokens"].reshape(-1),
                  dx.reshape(-1, D))
        return loss, grads

    def loss(self, tokens, targets):
        logits, _ = self._forward_train(tokens)
        targets = np.asarray(targets, dtype=np.int64)
        lg = logits.reshape(-1, self.vocab_size)
        mx = np.max(lg, axis=-1, keepdims=True)
        ex = np.exp(lg - mx)
        probs = ex / np.sum(ex, axis=-1, keepdims=True)
        return -float(np.mean(np.log(probs[np.arange(lg.shape[0]), targets.reshape(-1)] + 1e-12)))

    def generate(self, prompt_tokens, max_new=40, temperature=0.8, top_p=0.95, seed=0, eos=None):
        rng = np.random.default_rng(seed)
        toks = np.asarray(prompt_tokens, dtype=np.int64).reshape(1, -1)
        logits, pasts = self.forward(toks)
        out = list(int(t) for t in prompt_tokens)
        for _ in range(max_new):
            last = logits[0, -1].astype(np.float64)
            if temperature is None or temperature <= 0:
                nxt = int(np.argmax(last))
            else:
                z = last / temperature
                z = z - z.max()
                p = np.exp(z); p /= p.sum()
                order = np.argsort(-p); cp = np.cumsum(p[order])
                keep = cp <= top_p; keep[0] = True
                filt = np.zeros_like(p); filt[order[keep]] = p[order[keep]]; filt /= filt.sum()
                nxt = int(rng.choice(len(filt), p=filt))
            out.append(nxt)
            if eos is not None and nxt == eos:
                break
            logits, pasts = self.forward(np.array([[nxt]], dtype=np.int64), pasts=pasts)
        return out


if __name__ == "__main__":
    import time
    m = ModernTransformerNumpy(d_model=128, n_blocks=4, n_q_heads=8, n_kv_heads=2, vocab_size=258)
    toks = np.random.randint(0, 255, (2, 128))
    t0 = time.time(); logits, _ = m.forward(toks); print(f"numpy f32 (2,128) d128x4: {time.time()-t0:.4f}s shape={logits.shape} finite={np.isfinite(logits).all()}")
    t0 = time.time(); logits, _ = m.forward(toks); print(f"again: {time.time()-t0:.4f}s")
    seq = np.random.randint(0, 258, (1, 9))
    full, _ = m.forward(seq); _, pasts = m.forward(seq[:, :8]); one, _ = m.forward(seq[:, 8:], pasts=pasts)
    print(f"kv-cache max diff: {np.max(np.abs(full[0,-1]-one[0,-1])):.2e}")
    mq = ModernTransformerNumpy(d_model=128, n_blocks=4, n_q_heads=8, n_kv_heads=2, vocab_size=258, is_quantized=True)
    mq.params = m.params  # same weights, quantised copies already differ; rebuild quant from m
    mq.q_params, mq.q_scales = {}, {}
    for k, v in m.params.items():
        if "weight" in k and v.ndim == 2:
            mq.q_params[k], mq.q_scales[k] = quantize_weight_int8(v)
    lq, _ = mq.forward(toks)
    print(f"int8 (full, incl FFN) max|f32-i8|: {np.max(np.abs(logits-lq)):.4f}")
    print("generate:", m.generate([10, 20, 30], max_new=12, temperature=0.8, seed=1))
    # Xiyouji char-vocab shape check
    big = ModernTransformerNumpy(d_model=32, n_blocks=1, n_q_heads=4, n_kv_heads=2, vocab_size=4513)
    lb, _ = big.forward(np.array([[100, 2000, 4512]]))
    print(f"xiyouji vocab 4513 check: logits {lb.shape} finite={np.isfinite(lb).all()}")
    print("OK - pure numpy, no numba")

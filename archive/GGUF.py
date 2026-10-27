#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
现代架构训练 + GGUF 导出(单文件,纯 NumPy)。

架构与 llama.cpp 的 llama 架构逐项对齐:
  RMSNorm(eps=1e-5) / GQA / RoPE(相邻配对, theta=10000) / SwiGLU / 注意力与 FFN 均无偏置 /
  独立的输出头(output.weight)。
参数字典的键直接使用 GGUF 张量名(如 blk.0.attn_q.weight),导出时无需重命名。

和旧版的区别:
- 去掉了 FFN 偏置(llama 架构没有对应张量)。旧检查点无法加载,需要重新训练。
- 分词器是字符级,词表写入 GGUF 的 tokenizer 字段(特殊符号在前三位)。
- 删除了 int8 量化路径(GGUF 自己做量化)和无关的分词器/模式分支。
- 训练循环、AdamW、余弦学习率、last/best 检查点、续跑、早停、Ctrl-C 保存都保留。
- 新增:--selftest(float64 梯度检查 + KV 缓存一致性)、--export-gguf、--verify-gguf。

用法:
    python modern_gguf.py --selftest                    # 先跑自检
    python modern_gguf.py                               # 续跑 CONFIG["steps"] 步
    python modern_gguf.py --fresh --steps 2000          # 从头训练
    python modern_gguf.py --generate-only               # 用最佳检查点生成
    python modern_gguf.py --export-gguf model.gguf      # 从最佳检查点导出 GGUF
    python modern_gguf.py --verify-gguf model.gguf      # 读回校验 GGUF 与检查点一致

依赖: numpy;导出与校验需要 pip install gguf
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
LOG_PATH = BASE / "boot_log.txt"
LAST_CKPT = BASE / "modern_gguf_last.npz"
BEST_CKPT = BASE / "modern_gguf_best.npz"

ROPE_THETA = 10000.0
RMS_EPS = 1e-5
ARCH_NAME = "llama"
SPECIAL_TOKENS = ("<BOS>", "<EOS>", "<UNK>")

CONFIG = {
    "steps": 500,            # 本次增训步数
    "d_model": 128,
    "n_blocks": 4,
    "n_heads": 8,            # Q 头数, d_model 必须能被它整除, 且 d_model/n_heads 为偶数
    "n_kv_heads": 4,         # KV 头数, n_heads 必须能被它整除
    "ffn_dim": 512,            # 0 = 自动取 int(8/3 * d_model)
    "max_seq": 512,          # 写入 GGUF 的 context_length
    "batch_size": 8,
    "seq_len": 256,
    "lr": 1e-3,
    "min_lr_ratio": 0.1,
    "warmup_steps": 50,      # 仅全新训练使用
    "weight_decay": 0.1,
    "max_grad_norm": 0.5,
    "eval_every": 100,
    "log_every": 20,
    "patience": 0,           # 连续多少次评估无改善就早停; 0 = 关闭
    "seed": 42,
    "resume": False,
    "text_file": "Xiyouji.txt",
    "prompt": "菩提祖师",
    "new_tokens": 128,
}


# ============================================================================
# 日志
# ============================================================================
_t0 = time.time()


def log(msg: str) -> None:
    line = f"[modern {time.time() - _t0:9.3f}s] {msg}"
    print(line, flush=True)
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


# ============================================================================
# 分词器
# ============================================================================
class CharTokenizer:
    """字符级分词器。ID: 0=BOS, 1=EOS, 2=UNK, 3.. = 训练文本中的字符(排序后)。"""
    BOS, EOS, UNK = 0, 1, 2

    def __init__(self, chars):
        self.chars = list(chars)
        self.stoi = {c: i + 3 for i, c in enumerate(self.chars)}
        self.itos = {i + 3: c for i, c in enumerate(self.chars)}
        self.vocab_size = 3 + len(self.chars)

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

    def gguf_tokens(self) -> list[str]:
        """按 ID 顺序返回词表,供写入 GGUF。"""
        return list(SPECIAL_TOKENS) + self.chars


# ============================================================================
# 算子: 每个前向返回 (输出, 缓存), 反向返回输入梯度并累加参数梯度
# ============================================================================
def rms_norm_fwd(x2: np.ndarray, w: np.ndarray):
    """x2: (N, D). 返回 (输出, inv_rms (N,1), xn (N,D))。"""
    inv = 1.0 / np.sqrt(np.mean(x2 * x2, axis=-1, keepdims=True) + RMS_EPS)
    xn = x2 * inv
    return xn * w, inv, xn


def rms_norm_bwd(dy2: np.ndarray, x2: np.ndarray, w: np.ndarray, inv: np.ndarray, xn: np.ndarray):
    """返回 (dx2, dw)。"""
    n = x2.shape[-1]
    dw = np.sum(dy2 * xn, axis=0)
    dn = dy2 * w
    dx = inv * dn - x2 * (inv ** 3 / n) * np.sum(dn * x2, axis=-1, keepdims=True)
    return dx, dw


def rope_tables(positions: np.ndarray, d_head: int, dtype=np.float32):
    inv = 1.0 / (ROPE_THETA ** (np.arange(0, d_head, 2, dtype=np.float64) / d_head))
    ang = positions.astype(np.float64)[:, None] * inv[None, :]
    return np.cos(ang).astype(dtype), np.sin(ang).astype(dtype)


def rope_apply(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """x: (B, H, S, Dh), 相邻配对 (0,1), (2,3), ... 旋转。"""
    xe, xo = x[..., 0::2], x[..., 1::2]
    c, s = cos[None, None], sin[None, None]
    out = np.empty_like(x)
    out[..., 0::2] = xe * c - xo * s
    out[..., 1::2] = xe * s + xo * c
    return out


def rope_bwd(dy: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """rope_apply 的转置(旋转角度取反)。"""
    de, do = dy[..., 0::2], dy[..., 1::2]
    c, s = cos[None, None], sin[None, None]
    out = np.empty_like(dy)
    out[..., 0::2] = de * c + do * s
    out[..., 1::2] = -de * s + do * c
    return out


def _softmax_causal(scores, mask):
    scores = np.where(mask, scores, -1e30)
    scores = scores - scores.max(axis=-1, keepdims=True)
    e = np.exp(scores) * mask
    return e / e.sum(axis=-1, keepdims=True)


# ============================================================================
# 模型
# ============================================================================
class LlamaStyleModel:
    """与 llama.cpp 的 llama 架构对齐的参数布局与前向计算。"""

    def __init__(self, vocab_size: int, d_model: int, n_blocks: int, n_heads: int,
                 n_kv_heads: int, ffn_dim: int, max_seq: int = 512, seed: int = 42):
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"
        assert n_heads % n_kv_heads == 0, "n_heads 必须能被 n_kv_heads 整除"
        assert (d_model // n_heads) % 2 == 0, "每个头的维度必须为偶数(RoPE)"
        self.vocab_size, self.d_model, self.n_blocks = vocab_size, d_model, n_blocks
        self.n_heads, self.n_kv_heads, self.ffn_dim, self.max_seq = n_heads, n_kv_heads, ffn_dim, max_seq
        self.d_head = d_model // n_heads

        rng = np.random.default_rng(seed)
        D, F, V = d_model, ffn_dim, vocab_size
        kvd = n_kv_heads * self.d_head

        def w(shape):
            return (rng.standard_normal(shape) * 0.02).astype(np.float32)

        p = {"token_embd.weight": w((V, D))}
        for b in range(n_blocks):
            pre = f"blk.{b}."
            p[pre + "attn_norm.weight"] = np.ones(D, dtype=np.float32)
            p[pre + "attn_q.weight"] = w((D, D))
            p[pre + "attn_k.weight"] = w((kvd, D))
            p[pre + "attn_v.weight"] = w((kvd, D))
            p[pre + "attn_output.weight"] = w((D, D))
            p[pre + "ffn_norm.weight"] = np.ones(D, dtype=np.float32)
            p[pre + "ffn_gate.weight"] = w((F, D))
            p[pre + "ffn_up.weight"] = w((F, D))
            p[pre + "ffn_down.weight"] = w((D, F))
        p["output_norm.weight"] = np.ones(D, dtype=np.float32)
        p["output.weight"] = w((V, D))
        self.params = p

    def arch(self) -> dict:
        return {"vocab_size": self.vocab_size, "d_model": self.d_model, "n_blocks": self.n_blocks,
                "n_heads": self.n_heads, "n_kv_heads": self.n_kv_heads,
                "ffn_dim": self.ffn_dim, "max_seq": self.max_seq}

    def num_params(self) -> int:
        return int(sum(v.size for v in self.params.values()))

    # ---------------- 推理(带 KV 缓存) ----------------
    def forward(self, tokens, pasts=None):
        tokens = np.asarray(tokens, dtype=np.int64)
        if tokens.min() < 0 or tokens.max() >= self.vocab_size:
            raise ValueError(f"token ID 必须在 [0, {self.vocab_size - 1}] 内")
        P = self.params
        B, S = tokens.shape
        D, H, KV, Dh = self.d_model, self.n_heads, self.n_kv_heads, self.d_head
        group = H // KV
        offset = 0 if (pasts is None or pasts[0] is None) else pasts[0][0].shape[2]
        pos = np.arange(offset, offset + S)
        cos, sin = rope_tables(pos, Dh)
        x = P["token_embd.weight"][tokens]
        new_kv = []
        for b in range(self.n_blocks):
            pre = f"blk.{b}."
            h = rms_norm_fwd(x.reshape(-1, D), P[pre + "attn_norm.weight"])[0].reshape(B, S, D)
            q = (h @ P[pre + "attn_q.weight"].T).reshape(B, S, H, Dh).transpose(0, 2, 1, 3)
            k = (h @ P[pre + "attn_k.weight"].T).reshape(B, S, KV, Dh).transpose(0, 2, 1, 3)
            v = (h @ P[pre + "attn_v.weight"].T).reshape(B, S, KV, Dh).transpose(0, 2, 1, 3)
            q = rope_apply(q, cos, sin)
            k = rope_apply(k, cos, sin)
            if pasts is not None and pasts[b] is not None:
                k = np.concatenate([pasts[b][0], k], axis=2)
                v = np.concatenate([pasts[b][1], v], axis=2)
            new_kv.append((k, v))
            Tk = k.shape[2]
            kk = np.repeat(k, group, axis=1)
            vv = np.repeat(v, group, axis=1)
            mask = (np.arange(Tk)[None, :] <= pos[:, None])  # (S, Tk)
            probs = _softmax_causal(q @ kk.transpose(0, 1, 3, 2) / math.sqrt(Dh), mask[None, None])
            attn = (probs @ vv).transpose(0, 2, 1, 3).reshape(B, S, D)
            x = x + attn @ P[pre + "attn_output.weight"].T
            h2 = rms_norm_fwd(x.reshape(-1, D), P[pre + "ffn_norm.weight"])[0]
            g = h2 @ P[pre + "ffn_gate.weight"].T
            u = h2 @ P[pre + "ffn_up.weight"].T
            ffn = (g / (1.0 + np.exp(-g))) * u
            x = x + (ffn @ P[pre + "ffn_down.weight"].T).reshape(B, S, D)
        hf = rms_norm_fwd(x.reshape(-1, D), P["output_norm.weight"])[0]
        logits = (hf @ P["output.weight"].T).reshape(B, S, self.vocab_size)
        return logits, new_kv

    # ---------------- 训练前向(保存缓存) ----------------
    def _forward_train(self, tokens):
        tokens = np.asarray(tokens, dtype=np.int64)
        if tokens.min() < 0 or tokens.max() >= self.vocab_size:
            raise ValueError(f"token ID 必须在 [0, {self.vocab_size - 1}] 内")
        P = self.params
        B, S = tokens.shape
        D, H, KV, Dh = self.d_model, self.n_heads, self.n_kv_heads, self.d_head
        group = H // KV
        x = P["token_embd.weight"][tokens]
        cos, sin = rope_tables(np.arange(S), Dh, dtype=x.dtype)
        mask = np.tril(np.ones((S, S), dtype=bool))[None, None]
        blocks = []
        for b in range(self.n_blocks):
            pre = f"blk.{b}."
            c = {"x0": x}
            h, inv1, xn1 = rms_norm_fwd(x.reshape(-1, D), P[pre + "attn_norm.weight"])
            h3 = h.reshape(B, S, D)
            q = rope_apply((h3 @ P[pre + "attn_q.weight"].T).reshape(B, S, H, Dh).transpose(0, 2, 1, 3), cos, sin)
            k = rope_apply((h3 @ P[pre + "attn_k.weight"].T).reshape(B, S, KV, Dh).transpose(0, 2, 1, 3), cos, sin)
            v = (h3 @ P[pre + "attn_v.weight"].T).reshape(B, S, KV, Dh).transpose(0, 2, 1, 3)
            kk = np.repeat(k, group, axis=1)
            vv = np.repeat(v, group, axis=1)
            probs = _softmax_causal(q @ kk.transpose(0, 1, 3, 2) / math.sqrt(Dh), mask)
            attn_r = (probs @ vv).transpose(0, 2, 1, 3).reshape(B, S, D)
            x1 = x + attn_r @ P[pre + "attn_output.weight"].T
            h2, inv2, xn2 = rms_norm_fwd(x1.reshape(-1, D), P[pre + "ffn_norm.weight"])
            hg = h2 @ P[pre + "ffn_gate.weight"].T
            hu = h2 @ P[pre + "ffn_up.weight"].T
            sig = 1.0 / (1.0 + np.exp(-hg))
            swish = hg * sig
            mid = swish * hu
            x = x1 + (mid @ P[pre + "ffn_down.weight"].T).reshape(B, S, D)
            c.update(inv1=inv1, xn1=xn1, h3=h3, q=q, k=k, v=v, probs=probs, attn_r=attn_r,
                     x1=x1, inv2=inv2, xn2=xn2, h2=h2, hg=hg, hu=hu, sig=sig, swish=swish, mid=mid)
            blocks.append(c)
        hf, invf, xnf = rms_norm_fwd(x.reshape(-1, D), P["output_norm.weight"])
        logits = (hf @ P["output.weight"].T).reshape(B, S, self.vocab_size)
        cache = {"blocks": blocks, "x_final": x, "hf": hf, "invf": invf, "xnf": xnf,
                 "tokens": tokens, "cos": cos, "sin": sin}
        return logits, cache

    def loss(self, tokens, targets) -> float:
        logits, _ = self._forward_train(tokens)
        targets = np.asarray(targets, dtype=np.int64).reshape(-1)
        lg = logits.reshape(-1, self.vocab_size)
        mx = lg.max(axis=-1, keepdims=True)
        ex = np.exp(lg - mx)
        probs = ex / ex.sum(axis=-1, keepdims=True)
        return -float(np.mean(np.log(probs[np.arange(lg.shape[0]), targets] + 1e-12)))

    # ---------------- 反向传播 ----------------
    def loss_and_grads(self, tokens, targets):
        P = self.params
        logits, cache = self._forward_train(tokens)
        targets = np.asarray(targets, dtype=np.int64)
        B, S, V = logits.shape
        D, H, KV, Dh = self.d_model, self.n_heads, self.n_kv_heads, self.d_head
        group = H // KV
        n = B * S

        lg = logits.reshape(-1, V)
        mx = lg.max(axis=-1, keepdims=True)
        ex = np.exp(lg - mx)
        probs = ex / ex.sum(axis=-1, keepdims=True)
        flat_t = targets.reshape(-1)
        loss = -float(np.sum(np.log(probs[np.arange(n), flat_t] + 1e-12))) / n
        dl = probs
        dl[np.arange(n), flat_t] -= 1.0
        dl = dl / n

        grads = {k: np.zeros_like(v) for k, v in P.items()}

        # 输出头 + 最终 RMSNorm
        grads["output.weight"] += dl.T @ cache["hf"]
        dhf = dl @ P["output.weight"]
        dx, dw = rms_norm_bwd(dhf, cache["x_final"].reshape(-1, D), P["output_norm.weight"],
                              cache["invf"], cache["xnf"])
        grads["output_norm.weight"] += dw

        cos, sin = cache["cos"], cache["sin"]
        for b in reversed(range(self.n_blocks)):
            pre = f"blk.{b}."
            c = cache["blocks"][b]
            dxf = dx.reshape(-1, D)

            # FFN 分支: x_out = x1 + down(mid)
            grads[pre + "ffn_down.weight"] += dxf.T @ c["mid"]
            dmid = dxf @ P[pre + "ffn_down.weight"]
            dswish = dmid * c["hu"]
            dhu = dmid * c["swish"]
            dhg = dswish * (c["sig"] + c["swish"] * (1.0 - c["sig"]))
            grads[pre + "ffn_gate.weight"] += dhg.T @ c["h2"]
            grads[pre + "ffn_up.weight"] += dhu.T @ c["h2"]
            dh2 = dhg @ P[pre + "ffn_gate.weight"] + dhu @ P[pre + "ffn_up.weight"]
            dx1_ffn, dw2 = rms_norm_bwd(dh2, c["x1"].reshape(-1, D), P[pre + "ffn_norm.weight"],
                                        c["inv2"], c["xn2"])
            grads[pre + "ffn_norm.weight"] += dw2
            dx1 = dxf + dx1_ffn

            # 注意力分支: x1 = x0 + attn_r @ Wo.T
            grads[pre + "attn_output.weight"] += dx1.T @ c["attn_r"].reshape(-1, D)
            dattn = (dx1 @ P[pre + "attn_output.weight"]).reshape(B, S, H, Dh).transpose(0, 2, 1, 3)
            probs_b = c["probs"]
            kk = np.repeat(c["k"], group, axis=1)
            vv = np.repeat(c["v"], group, axis=1)
            dprobs = dattn @ vv.transpose(0, 1, 3, 2)
            dvv = probs_b.transpose(0, 1, 3, 2) @ dattn
            dscores = probs_b * (dprobs - np.sum(dprobs * probs_b, axis=-1, keepdims=True))
            scale = 1.0 / math.sqrt(Dh)
            dq = (dscores @ kk) * scale
            dkk = (dscores.transpose(0, 1, 3, 2) @ c["q"]) * scale
            dk_rep = dkk.reshape(B, KV, group, S, Dh).sum(axis=2)
            dv_rep = dvv.reshape(B, KV, group, S, Dh).sum(axis=2)
            dq = rope_bwd(dq, cos, sin)
            dk = rope_bwd(dk_rep, cos, sin)

            dq_r = dq.transpose(0, 2, 1, 3).reshape(-1, H * Dh)
            dk_r = dk.transpose(0, 2, 1, 3).reshape(-1, KV * Dh)
            dv_r = dv_rep.transpose(0, 2, 1, 3).reshape(-1, KV * Dh)
            h3 = c["h3"].reshape(-1, D)
            grads[pre + "attn_q.weight"] += dq_r.T @ h3
            grads[pre + "attn_k.weight"] += dk_r.T @ h3
            grads[pre + "attn_v.weight"] += dv_r.T @ h3
            dh = dq_r @ P[pre + "attn_q.weight"] + dk_r @ P[pre + "attn_k.weight"] \
                + dv_r @ P[pre + "attn_v.weight"]
            dx0_attn, dw1 = rms_norm_bwd(dh, c["x0"].reshape(-1, D), P[pre + "attn_norm.weight"],
                                         c["inv1"], c["xn1"])
            grads[pre + "attn_norm.weight"] += dw1

            # 残差: x1 的梯度 = 来自 x2 的残差 dx1 (已在 dx1 中) + 注意力分支回传
            dx = (dx1 + dx0_attn).reshape(B, S, D)

        np.add.at(grads["token_embd.weight"], cache["tokens"].reshape(-1), dx.reshape(-1, D))
        return loss, grads

    # ---------------- 生成 ----------------
    def generate(self, prompt_ids, max_new=40, temperature=0.8, top_p=0.95, seed=0, eos=None):
        rng = np.random.default_rng(seed)
        toks = np.asarray(prompt_ids, dtype=np.int64).reshape(1, -1)
        logits, pasts = self.forward(toks)
        out = [int(t) for t in prompt_ids]
        for _ in range(max_new):
            last = logits[0, -1].astype(np.float64)
            if temperature is None or temperature <= 0:
                nxt = int(np.argmax(last))
            else:
                z = last / temperature
                z = z - z.max()
                p = np.exp(z)
                p /= p.sum()
                order = np.argsort(-p)
                cp = np.cumsum(p[order])
                keep = cp <= top_p
                keep[0] = True
                filt = np.zeros_like(p)
                filt[order[keep]] = p[order[keep]]
                filt /= filt.sum()
                nxt = int(rng.choice(len(filt), p=filt))
            out.append(nxt)
            if eos is not None and nxt == eos:
                break
            logits, pasts = self.forward(np.array([[nxt]], dtype=np.int64), pasts=pasts)
        return out


# ============================================================================
# 优化器与学习率
# ============================================================================
class AdamW:
    """AdamW(解耦权重衰减,只作用于二维及以上的参数),与 torch.optim.AdamW 的更新规则一致。"""

    def __init__(self, params, lr, weight_decay, betas=(0.9, 0.999), eps=1e-8):
        self.lr, self.wd = lr, weight_decay
        self.b1, self.b2 = betas
        self.eps = eps
        self.t = 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params, grads):
        self.t += 1
        bc1 = 1.0 - self.b1 ** self.t
        bc2 = 1.0 - self.b2 ** self.t
        for k in params:
            g = grads[k]
            m, v = self.m[k], self.v[k]
            m *= self.b1
            m += (1.0 - self.b1) * g
            v *= self.b2
            v += (1.0 - self.b2) * g * g
            mhat = m / bc1
            vhat = v / bc2
            wd = self.wd if params[k].ndim >= 2 else 0.0
            params[k] = params[k] * (1.0 - self.lr * wd) - self.lr * mhat / (np.sqrt(vhat) + self.eps)


def clip_grad_norm(grads, max_norm: float) -> float:
    total = math.sqrt(sum(float(np.sum(g.astype(np.float64) ** 2)) for g in grads.values()))
    if total > max_norm:
        scale = max_norm / (total + 1e-6)
        for g in grads.values():
            g *= scale
    return total


def lr_schedule(rel: int, base: float, min_lr: float, warmup: int, total_rel: int) -> float:
    """rel 从 1 开始。预热线性上升,之后余弦衰减到 min_lr。"""
    if warmup and rel <= warmup:
        return base * rel / warmup
    span = max(1, total_rel - warmup)
    p = min(1.0, (rel - warmup) / span)
    return min_lr + 0.5 * (base - min_lr) * (1.0 + math.cos(math.pi * p))


# ============================================================================
# 配置与检查点
# ============================================================================
def resolve_ffn(cfg: dict) -> int:
    return int(cfg["ffn_dim"]) or int(8 * cfg["d_model"] / 3)


def expected_arch(cfg: dict, vocab_size: int) -> dict:
    return {"vocab_size": int(vocab_size), "d_model": int(cfg["d_model"]),
            "n_blocks": int(cfg["n_blocks"]), "n_heads": int(cfg["n_heads"]),
            "n_kv_heads": int(cfg["n_kv_heads"]), "ffn_dim": resolve_ffn(cfg),
            "max_seq": int(cfg["max_seq"])}


def _atomic_savez(path: Path, **arrays) -> None:
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def save_checkpoint(path: Path, model: LlamaStyleModel, tok: CharTokenizer, *, params: dict,
                    step: int, best_val: float, best_step: int, last_lr=None,
                    rng=None, opt=None) -> None:
    arrays = {f"param::{k}": v for k, v in params.items()}
    meta = {
        "format": 3,
        "arch": model.arch(),
        "tokenizer": {"type": "char", "chars": tok.chars},
        "step": int(step),
        "best_val": float(best_val),
        "best_step": int(best_step),
        "last_lr": None if last_lr is None else float(last_lr),
        "rng_state": rng.bit_generator.state if rng is not None else None,
        "opt_t": None,
    }
    if opt is not None:
        for k, v in opt.m.items():
            arrays[f"opt_m::{k}"] = v
        for k, v in opt.v.items():
            arrays[f"opt_v::{k}"] = v
        meta["opt_t"] = int(opt.t)
    arrays["meta"] = np.asarray(json.dumps(meta, ensure_ascii=False))
    _atomic_savez(path, **arrays)


def load_params(z, model: LlamaStyleModel) -> dict:
    out = {}
    for name, cur in model.params.items():
        key = f"param::{name}"
        if key not in z.files:
            raise RuntimeError(f"检查点缺少参数 {name}")
        arr = np.asarray(z[key], dtype=np.float32)
        if arr.shape != cur.shape:
            raise RuntimeError(f"参数 {name} 形状不符: 检查点 {arr.shape}, 当前 {cur.shape}")
        out[name] = arr
    return out


def restore(path: Path, seed: int = 0):
    """从检查点还原模型与分词器。返回 (model, tok, meta, npz 句柄)。"""
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    model = LlamaStyleModel(**meta["arch"], seed=seed)
    model.params = load_params(z, model)
    tok = CharTokenizer(meta["tokenizer"]["chars"])
    return model, tok, meta, z


# ============================================================================
# GGUF 导出与读回校验
# ============================================================================
def export_gguf(model: LlamaStyleModel, tok: CharTokenizer, out_path: Path) -> None:
    try:
        import gguf
    except ImportError:
        raise SystemExit("导出 GGUF 需要先安装: pip install gguf")

    a = model.arch()
    w = gguf.GGUFWriter(str(out_path), ARCH_NAME)
    w.add_name("modern-char-llama")
    w.add_context_length(a["max_seq"])
    w.add_embedding_length(a["d_model"])
    w.add_block_count(a["n_blocks"])
    w.add_feed_forward_length(a["ffn_dim"])
    w.add_head_count(a["n_heads"])
    w.add_head_count_kv(a["n_kv_heads"])
    w.add_rope_dimension_count(a["d_model"] // a["n_heads"])
    w.add_rope_freq_base(ROPE_THETA)
    w.add_layer_norm_rms_eps(RMS_EPS)
    w.add_file_type(gguf.LlamaFileType.ALL_F32)

    # 字符级词表: 特殊符号为 CONTROL, 其余为 NORMAL。调用时需直接传 token ID, 不能传原始文本。
    tokens = tok.gguf_tokens()
    w.add_tokenizer_model("llama")
    w.add_token_list(tokens)
    w.add_token_scores([0.0] * len(tokens))
    types = [int(gguf.TokenType.CONTROL)] * len(SPECIAL_TOKENS) + \
            [int(gguf.TokenType.NORMAL)] * (len(tokens) - len(SPECIAL_TOKENS))
    w.add_token_types(types)
    w.add_bos_token_id(tok.BOS)
    w.add_eos_token_id(tok.EOS)
    w.add_unk_token_id(tok.UNK)

    for name, arr in model.params.items():
        w.add_tensor(name, np.ascontiguousarray(arr, dtype=np.float32))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def verify_gguf(path: Path, model: LlamaStyleModel) -> None:
    try:
        import gguf
    except ImportError:
        raise SystemExit("校验 GGUF 需要先安装: pip install gguf")

    reader = gguf.GGUFReader(str(path))
    fields = set(reader.fields.keys())
    for key in ("llama.block_count", "llama.embedding_length", "llama.attention.head_count",
                "llama.attention.head_count_kv", "llama.rope.dimension_count",
                "tokenizer.ggml.tokens", "tokenizer.ggml.bos_token_id"):
        if key not in fields:
            raise RuntimeError(f"GGUF 缺少元数据字段 {key}")

    tensors = {t.name: t for t in reader.tensors}
    bad = []
    for name, arr in model.params.items():
        if name not in tensors:
            bad.append(f"缺少张量 {name}")
            continue
        got = np.asarray(tensors[name].data, dtype=np.float32).reshape(-1)
        if got.size != arr.size or not np.array_equal(got, arr.reshape(-1)):
            bad.append(f"数值不一致 {name}")
    extra = set(tensors) - set(model.params)
    if extra:
        bad.append(f"多出的张量: {sorted(extra)[:5]}")
    if bad:
        raise RuntimeError("GGUF 校验失败:\n  " + "\n  ".join(bad))
    log(f"GGUF 校验通过: {len(model.params)} 个张量与检查点逐值一致")


# ============================================================================
# 自检
# ============================================================================
def selftest() -> None:
    ok = True
    rng = np.random.default_rng(0)

    # 1) 梯度检查(float64, 中心差分)
    m = LlamaStyleModel(vocab_size=20, d_model=16, n_blocks=2, n_heads=4, n_kv_heads=2,
                        ffn_dim=24, max_seq=64, seed=1)
    m.params = {k: v.astype(np.float64) for k, v in m.params.items()}
    tokens = rng.integers(0, 20, size=(2, 6))
    targets = rng.integers(0, 20, size=(2, 6))
    _, grads = m.loss_and_grads(tokens, targets)
    # eps=1e-5 时中心差分的截断误差与舍入误差都远小于判据; 判据为 allclose 形式
    eps, n_bad, worst = 1e-5, 0, 0.0
    for name, arr in m.params.items():
        flat = arr.reshape(-1)
        gflat = grads[name].reshape(-1)
        for i in rng.choice(flat.size, size=min(6, flat.size), replace=False):
            old = flat[i]
            flat[i] = old + eps
            lp = m.loss(tokens, targets)
            flat[i] = old - eps
            lm = m.loss(tokens, targets)
            flat[i] = old
            num = (lp - lm) / (2 * eps)
            ana = gflat[i]
            absdiff = abs(num - ana)
            worst = max(worst, absdiff)
            if absdiff > 1e-7 + 1e-4 * max(abs(num), abs(ana)):
                n_bad += 1
                print(f"  不一致: {name}[{i}] 数值={num:.3e} 解析={ana:.3e}")
    passed = n_bad == 0
    ok &= passed
    print(f"[梯度检查] 抽样 {len(m.params)} 类参数, 最大绝对误差 = {worst:.2e}, "
          f"不一致 {n_bad} 处  {'通过' if passed else '失败'}")

    # 2) KV 缓存与整段前向一致(float32)
    m32 = LlamaStyleModel(vocab_size=20, d_model=16, n_blocks=2, n_heads=4, n_kv_heads=2,
                          ffn_dim=24, max_seq=64, seed=2)
    seq = rng.integers(0, 20, size=(1, 9))
    full, _ = m32.forward(seq)
    _, pasts = m32.forward(seq[:, :8])
    one, _ = m32.forward(seq[:, 8:], pasts=pasts)
    diff = float(np.max(np.abs(full[0, -1] - one[0, -1])))
    passed = diff < 1e-4
    ok &= passed
    print(f"[KV 缓存] 最大差值 = {diff:.2e}  {'通过' if passed else '失败'}")

    # 3) 训练前向与推理前向一致
    lt, _ = m32._forward_train(seq)
    diff = float(np.max(np.abs(lt - full)))
    passed = diff < 1e-4
    ok &= passed
    print(f"[训练/推理前向] 最大差值 = {diff:.2e}  {'通过' if passed else '失败'}")

    if not ok:
        raise SystemExit(1)
    print("全部自检通过")


# ============================================================================
# 训练
# ============================================================================
def parse_args(argv: list[str]):
    cfg = dict(CONFIG)
    flags = {"generate_only": False, "selftest": False, "export_gguf": None, "verify_gguf": None}
    positional: list[int] = []
    i = 0
    while i < len(argv):
        tok = argv[i]
        i += 1
        if tok == "--resume":
            cfg["resume"] = True
            continue
        if tok in ("--fresh", "--no-resume"):
            cfg["resume"] = False
            continue
        if tok == "--generate-only":
            flags["generate_only"] = True
            continue
        if tok == "--selftest":
            flags["selftest"] = True
            continue
        if tok in ("--export-gguf", "--verify-gguf"):
            if i >= len(argv):
                raise SystemExit(f"{tok} 缺少文件路径")
            flags[tok[2:].replace("-", "_")] = argv[i]
            i += 1
            continue
        if tok.startswith("--"):
            key = tok[2:].replace("-", "_")
            if i >= len(argv):
                raise SystemExit(f"参数 {tok} 缺少取值")
            value = argv[i]
            i += 1
        elif "=" in tok:
            key, value = tok.split("=", 1)
        elif tok.isdigit():
            positional.append(int(tok))
            continue
        else:
            raise SystemExit(f"无法识别的参数: {tok}")
        if key not in CONFIG:
            raise SystemExit(f"未知参数: {key}\n可用参数: {', '.join(CONFIG)}")
        default = CONFIG[key]
        if isinstance(default, bool):
            cfg[key] = value.lower() in ("1", "true", "yes", "on")
        else:
            cfg[key] = type(default)(value)
    if positional:
        cfg["steps"] = positional[0]
    if len(positional) >= 2:
        cfg["d_model"] = positional[1]
    return cfg, flags


def generate_samples(model: LlamaStyleModel, tok: CharTokenizer, cfg: dict, params: dict) -> None:
    model.params = {k: v.copy() for k, v in params.items()}
    prompt_ids = tok.encode(cfg["prompt"], add_bos=True)
    gen = model.generate(prompt_ids, max_new=cfg["new_tokens"], temperature=0.8,
                         top_p=0.95, seed=1, eos=tok.EOS)
    log(f"生成示例(prompt={cfg['prompt']!r},温度0.8 / top-p0.95):")
    print(tok.decode(gen), flush=True)
    gen_greedy = model.generate(prompt_ids, max_new=40, temperature=0)
    log("生成示例(greedy):")
    print(tok.decode(gen_greedy), flush=True)


def main(argv: list[str]) -> None:
    cfg, flags = parse_args(argv)

    if flags["selftest"]:
        selftest()
        return

    if flags["export_gguf"] or flags["verify_gguf"]:
        ckpt = BEST_CKPT if BEST_CKPT.exists() else LAST_CKPT
        if not ckpt.exists():
            raise SystemExit("没有找到检查点,请先训练")
        model, tok, meta, _ = restore(ckpt, seed=cfg["seed"])
        log(f"已读取 {ckpt.name}: 第 {meta['step']} 步, 参数量 {model.num_params():,}")
        if flags["export_gguf"]:
            export_gguf(model, tok, Path(flags["export_gguf"]))
            log(f"已导出 GGUF: {flags['export_gguf']}")
        if flags["verify_gguf"]:
            verify_gguf(Path(flags["verify_gguf"]), model)
        return

    from_disk_text = (BASE / cfg["text_file"]).read_text(encoding="utf-8").replace("\r\n", "\n")

    log("=== 现代架构训练启动(llama 兼容架构 · 纯NumPy) ===")
    start_step, best_val, best_step, last_lr = 0, None, 0, None
    best_params, rng_state, z, resumed = None, None, None, False

    if cfg["resume"] and LAST_CKPT.exists():
        model, tok, meta, z = restore(LAST_CKPT, seed=cfg["seed"])
        if meta["arch"] != expected_arch(cfg, tok.vocab_size):
            raise RuntimeError(f"检查点架构 {meta['arch']} 与当前配置不一致。"
                               f"请恢复原配置,或加 --fresh 从头训练。")
        start_step = meta["step"]
        best_val, best_step = meta["best_val"], meta["best_step"]
        last_lr = meta.get("last_lr")
        rng_state = meta.get("rng_state")
        resumed = True
        log(f"从 {LAST_CKPT.name} 续跑: 第 {start_step} 步, 最佳验证 loss = {best_val:.4f}(第 {best_step} 步)")
        if BEST_CKPT.exists():
            bz = np.load(BEST_CKPT, allow_pickle=False)
            best_params = load_params(bz, model)
    else:
        tok = CharTokenizer.from_text(from_disk_text)
        model = LlamaStyleModel(**expected_arch(cfg, tok.vocab_size), seed=cfg["seed"])

    text_ids = np.asarray(tok.encode(from_disk_text, add_bos=True, add_eos=True), dtype=np.int64)
    split = int(len(text_ids) * 0.95)
    train_ids, val_ids = text_ids[:split], text_ids[split:]
    log(f"词表={tok.vocab_size} 训练token={len(train_ids)} 验证token={len(val_ids)}")

    if flags["generate_only"]:
        generate_samples(model, tok, cfg, best_params if best_params is not None else model.params)
        log("=== 全部完成 ===")
        return

    rng = np.random.default_rng(cfg["seed"] + start_step)
    if rng_state is not None:
        rng.bit_generator.state = rng_state

    opt = AdamW(model.params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    if resumed and z is not None:
        meta_t = json.loads(str(z["meta"])).get("opt_t")
        if meta_t is not None and any(f.startswith("opt_m::") for f in z.files):
            for k in model.params:
                opt.m[k] = np.array(z[f"opt_m::{k}"])
                opt.v[k] = np.array(z[f"opt_v::{k}"])
            opt.t = int(meta_t)
            log(f"已恢复优化器状态(第 {opt.t} 次更新)")
        else:
            log("检查点不含优化器状态,动量从零开始")

    if resumed:
        base_lr = min(cfg["lr"], last_lr) if last_lr else cfg["lr"]
        warmup = 0
    else:
        base_lr = cfg["lr"]
        warmup = cfg["warmup_steps"]
    min_lr = base_lr * cfg["min_lr_ratio"]

    val_now = _eval(model, val_ids, cfg["seq_len"])
    if best_val is None:
        best_val, best_step = float(val_now), start_step
    if best_params is None:
        best_params = {k: v.copy() for k, v in model.params.items()}
    log(f"当前验证 loss = {val_now:.4f}(最佳 {best_val:.4f} @ 第 {best_step} 步)")
    log(f"参数量={model.num_params():,} 配置: d={cfg['d_model']} blocks={cfg['n_blocks']} "
        f"heads={cfg['n_heads']}/{cfg['n_kv_heads']} ffn={resolve_ffn(cfg)} "
        f"batch={cfg['batch_size']} seq={cfg['seq_len']} lr={base_lr:g} warmup={warmup} "
        f"增训={cfg['steps']} 步 模式={'续跑' if resumed else '全新'}")

    total = start_step + cfg["steps"]
    run_start = start_step
    lr_now = base_lr
    cur_step = start_step
    no_improve, nan_count = 0, 0
    t_run = time.time()

    def checkpoint_last():
        save_checkpoint(LAST_CKPT, model, tok, params=model.params, step=cur_step,
                        best_val=best_val, best_step=best_step, last_lr=lr_now, rng=rng, opt=opt)

    def checkpoint_best():
        save_checkpoint(BEST_CKPT, model, tok, params=best_params, step=best_step,
                        best_val=best_val, best_step=best_step, last_lr=None)

    try:
        for step in range(start_step + 1, total + 1):
            rel = step - run_start
            lr_now = lr_schedule(rel, base_lr, min_lr, warmup, cfg["steps"])
            opt.lr = lr_now

            x, y = _make_batch(train_ids, cfg["batch_size"], cfg["seq_len"], rng)
            loss, grads = model.loss_and_grads(x, y)
            if not np.isfinite(loss):
                nan_count += 1
                log(f"step {step} loss 非有限({loss}),跳过该步(累计 {nan_count} 次)")
                if nan_count >= 10:
                    raise RuntimeError("非有限 loss 累计达到 10 次,已停止")
                continue

            gn = clip_grad_norm(grads, cfg["max_grad_norm"])
            opt.step(model.params, grads)
            cur_step = step

            if step % cfg["log_every"] == 0 or step == start_step + 1:
                log(f"step {step}/{total} train loss={loss:.4f} |grad|={gn:.2f} "
                    f"lr={lr_now:.5f} ({time.time() - t_run:.1f}s)")

            if step % cfg["eval_every"] == 0 or step == total:
                val = _eval(model, val_ids, cfg["seq_len"])
                log(f"step {step} 验证 loss = {val:.4f}(最佳 {best_val:.4f} @ 第 {best_step} 步)")
                if val < best_val - 1e-4:
                    best_val, best_step = float(val), step
                    best_params = {k: v.copy() for k, v in model.params.items()}
                    checkpoint_best()
                    no_improve = 0
                    log(f"  ↑ 新最佳,已保存 {BEST_CKPT.name}")
                else:
                    no_improve += 1
                checkpoint_last()
                if cfg["patience"] and no_improve >= cfg["patience"]:
                    log(f"早停: 连续 {no_improve} 次评估无改善")
                    break
    except KeyboardInterrupt:
        log("⏹ 手动停止,正在保存当前进度…")
        checkpoint_last()
        raise

    checkpoint_last()
    log(f"本次训练结束: 训练至第 {cur_step} 步(目标 {total}), 用时 {time.time() - t_run:.1f}s, "
        f"最佳验证 loss = {best_val:.4f}(第 {best_step} 步)")
    generate_samples(model, tok, cfg, best_params)
    log("=== 全部完成 ===")


def _make_batch(data: np.ndarray, batch_size: int, seq_len: int, rng: np.random.Generator):
    starts = rng.integers(0, len(data) - seq_len, size=batch_size)
    x = np.stack([data[int(i):int(i) + seq_len] for i in starts])
    y = np.stack([data[int(i) + 1:int(i) + seq_len + 1] for i in starts])
    return x, y


def _eval(model: LlamaStyleModel, token_ids: np.ndarray, seq_len: int, max_batches: int = 20) -> float:
    """确定性验证: 对不重叠的连续窗口求平均 loss。"""
    data = np.asarray(token_ids, dtype=np.int64)
    if len(data) < seq_len + 1:
        raise ValueError(f"验证集至少需要 {seq_len + 1} 个 token")
    starts = list(range(0, len(data) - seq_len, seq_len))[:max_batches]
    losses = [model.loss(data[s:s + seq_len][None], data[s + 1:s + seq_len + 1][None]) for s in starts]
    return float(sum(losses) / len(losses))


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except KeyboardInterrupt:
        log("⏹ 被手动停止: 进度已保存到 modern_gguf_last.npz, 直接运行即可续跑")
    except SystemExit:
        raise
    except Exception:
        log("❌ 失败:\n" + traceback.format_exc())
        raise

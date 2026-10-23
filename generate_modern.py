#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 modern_ckpt.npz 加载已训练的现代架构模型并生成文本（纯 NumPy，无 UI）。
用法：直接运行 = 用默认 prompt「悟空」生成；或 python generate_modern.py [prompt] [个数]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))


def main() -> None:
    from dual_transformer_numpy import CharTokenizer
    from modern_transformer_numpy import ModernTransformerNumpy

    ckpt = BASE / "modern_ckpt.npz"
    if not ckpt.exists():
        print("找不到 modern_ckpt.npz，请先运行 train_modern.py 训练一次")
        return
    argv = [a for a in sys.argv[1:] if a]
    prompt = argv[0] if len(argv) >= 1 else "悟空"
    n = int(argv[1]) if len(argv) >= 2 and argv[1].isdigit() else 100

    z = np.load(ckpt, allow_pickle=False)
    cfg = z["cfg"]
    d, blocks, qh, kvh, vocab = (int(cfg[0]), int(cfg[1]), int(cfg[2]),
                                 int(cfg[3]), int(cfg[4]))
    tok = CharTokenizer([str(c) for c in z["chars"]])
    assert tok.vocab_size == vocab, "检查点词表与文件不一致"
    model = ModernTransformerNumpy(d_model=d, n_blocks=blocks, n_q_heads=qh,
                                   n_kv_heads=kvh, vocab_size=vocab)
    for k in model.params:
        model.params[k] = z[f"param::{k}"]
    print(f"已加载检查点：d={d} blocks={blocks} q/kv={qh}/{kvh} "
          f"训练步数={int(cfg[5])} 最佳验证loss={float(z['val'][0]):.4f}")
    ids = tok.encode(prompt, add_bos=True)
    out = model.generate(ids, max_new=n, temperature=0.8, top_p=0.95, seed=1)
    print(tok.decode(out))


if __name__ == "__main__":
    main()

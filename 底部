#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
老代码安全全马版：完全对齐原本能跑的加载逻辑，无缝嵌入实时对话框。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))


def main() -> None:
    from dual_transformer_numpy import CharTokenizer, DualTransformerNP

    ckpt = BASE / "modern_ckpt.npz"
    if not ckpt.exists():
        ckpt = BASE / "ckpt_dual" / "modern_ckpt_last.npz"
        
    if not ckpt.exists():
        print("找不到权重文件，请确认配置路径*")
        return

    chars_file = BASE / "chars.txt"
    if chars_file.exists():
        with open(chars_file, "r", encoding="utf-8") as f:
            chars = [c for c in f.read()]
    else:
        z = np.load(ckpt, allow_pickle=True)
        if "chars" in z:
            chars = [str(c) for c in z["chars"]]
        else:
            chars = [chr(i) for i in range(19968, 40870)] 

    tok = CharTokenizer(chars)
    z = np.load(ckpt, allow_pickle=False)

    d = 128
    blocks = 4
    qh = 4
    vocab = tok.vocab_size

    print(f"正在以老代码标准实例化模型: d={d}, blocks={blocks}, heads={qh} ...")
    model = DualTransformerNP(
        d_model=d,
        n_blocks=blocks,
        n_heads=qh,
        pos_mode="rope",
        ff_mode="gelu",
        init_style="gpt2",
        seed=42
    )
    
    for k in model.params:
        if f"param::{k}" in z:
            model.params[k][:] = z[f"param::{k}"]
        elif f"param:{k}" in z:
            model.params[k][:] = z[f"param:{k}"]
        elif k in z:
            model.params[k][:] = z[k]

    print("\n====== 权重加载成功！老笔记本生存模式就绪 ======")
    print("====== 已进入实时对话模式（输入 exit 退出） ======\n")

    while True:
        try:
            user_input = input("用户 (User) >>> ")
            
            if user_input.strip().lower() == "exit":
                print("已退出对话。")
                break
                
            if not user_input.strip():
                continue

            ids = tok.encode(user_input, add_bos=True)
            # 💡 这里已将 seed=None 彻底剔除，只保留你的老代码最稳妥支持的原生参数
            out = model.generate(ids, max_new_tokens=32, temperature=0.8, top_p=0.95)
            response = tok.decode(out)
            
            print(f"模型 (Bot)  >>> {response}\n")
            
        except KeyboardInterrupt:
            print("\n已退出对话。")
            break


if __name__ == "__main__":
    main()

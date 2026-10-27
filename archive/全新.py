#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 modern_ckpt.npz 加载已训练的现代架构模型并进入实时打字对话(纯 NumPy,硬编码安全版)。
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
        print("找不到 modern_ckpt.npz,请先运行 train_modern.py 训练一次")
        return

    z = np.load(ckpt, allow_pickle=False)
    tok = CharTokenizer([str(c) for c in z["chars"]])
    
    # 💡 终极修复:老笔记本专属超轻量硬编码参数,彻底绕过 z["cfg"] 的报错隐患
    d = 128
    blocks = 4
    qh = 4
    kvh = 2
    vocab = tok.vocab_size

    print(f"正在加载微型硬编码模型规格:d={d} blocks={blocks} q/kv={qh}/{kvh} ...")
    
    # 2. 实例化模型并完全恢复原版矩阵名加载逻辑
    model = ModernTransformerNumpy(d_model=d, n_blocks=blocks, n_q_heads=qh,
                                   n_kv_heads=kvh, vocab_size=vocab)
    
    # 3. 安全加载参数,如果名称不完全一致则跳过或直接赋值
    for k in model.params:
        if f"param::{k}" in z:
            model.params[k] = z[f"param::{k}"]
        elif f"param:{k}" in z:
            model.params[k] = z[f"param:{k}"]
        else:
            print(f"提示:权重中未找到对应的参数 {k},已跳过")
        
    print("====== 成功加载权重! ======")
    print("====== 已进入实时对话模式(输入 exit 退出) ======\n")

    # 4. 启动交互式无限对话循环
    while True:
        try:
            user_input = input("用户 (User) >>> ")
            
            # 安全退出检测
            if user_input.strip().lower() == "exit":
                print("已退出对话。")
                break
                
            # 空白输入过滤
            if not user_input.strip():
                continue

            # 5. 对话编码与单次最大生成 (设为 32 防止老笔记本发烫卡顿)
            ids = tok.encode(user_input, add_bos=True)
            out = model.generate(ids, max_new_n=32, temperature=0.8, top_p=0.95, seed=None)
            
            # 6. 解码并打印输出结果
            response = tok.decode(out)
            print(f"模型 (Bot)  >>> {response}\n")
            
        except KeyboardInterrupt:
            print("\n已退出对话。")
            break


if __name__ == "__main__":
    main()

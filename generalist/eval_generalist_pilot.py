#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""全能试点门禁：产品出来后跑，不提前下结论。

做三件事，只呈证据、不自动判 PASS/FAIL（100 步从头训的模型，
分数一定难看，判读交给看样本的人）：
  1. 经 world_loop.load_world_model 加载产品（验证循环路径能读它）；
  2. 每域一个贪心续写样本（经典/代码/对话/推理/世界），看格式对不对、
     回没回在题上、五个域的输出有没有塌成同一个东西；
  3. 每域一句训练文件里没有的新句子，算 loss（指示用，不是严格验证集，
     因为试点没有留验证集）。

用法：python3 eval_generalist_pilot.py [out/generalist_pilot.npz]
  写出 GENERALIST_PILOT_GATE.md，并把样本打印出来。
"""
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser("~/workspace/omega-chat"))
sys.path.insert(0, os.path.expanduser("~/workspace/omega-chat/tools"))
sys.path.insert(0, os.path.expanduser("~/workspace/dual_side/xiyouji_modern_trainer_1"))

import numpy as np

import world_loop as wl

HERE = Path(__file__).resolve().parent

PROMPTS = {
    "经典": "话说",
    "代码": "def add(a, b):",
    "对话": "用户：你好。\n助手：",
    "推理": "问：三加四等于几？答案是",
    "世界": ("我是欧米伽。当前人格悟空。\n世界状态。你在花果山。在场人物孙悟空、"
             "唐僧、猪八戒、沙僧、白龙马。可见物品蟠桃、金箍棒。随身物品无。"
             "时间第一步白天。\n问：我在哪？"),
}

# 训练文件里没有的新句子（指示用 val）。
FRESH = {
    "经典": "山风吹过松林，松涛声一阵一阵。",
    "代码": "def multiply(x, y):\n    return x * y",
    "对话": "用户：今天吃什么。\n助手：吃面条吧。",
    "推理": "问：二十五加三十七等于几？答案是六十二。",
    "世界": ("世界状态。你在天宫。在场人物孙悟空、唐僧、猪八戒、沙僧、白龙马。"
             "可见物品无。随身物品金箍棒。时间第五步夜晚。"),
}


def cjk_ratio(text):
    if not text:
        return 0.0
    return sum(1 for c in text if "\u4e00" <= c <= "\u9fff") / len(text)


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else str(HERE / "out/generalist_pilot.npz")
    model, tok, step, arch = wl.load_world_model(ckpt)
    print(f"loaded: step={step} arch={arch} vocab={tok.vocab_size} "
          f"config={model.config()}")

    lines = [f"# 全能试点门禁（产品 {ckpt}，step {step}）\n"]
    samples = {}
    for name, prompt in PROMPTS.items():
        ids = tok.encode(prompt)
        try:
            out = model.generate_cached(ids, max_new_tokens=32, temperature=0.0)
            path = "generate_cached"
        except Exception as e:
            print(f"generate_cached failed ({e}), falling back to generate")
            out = model.generate(ids, max_new_tokens=32, temperature=0.0)
            path = "generate"
        text = tok.decode(out)
        samples[name] = text
        lines.append(f"## {name}（经 {path}）")
        lines.append(f"提示：{prompt!r}")
        lines.append(f"续写：{text!r}")
        lines.append(f"汉字占比：{cjk_ratio(text):.2f}")
        lines.append("")
        print(f"[{name}] {text!r}")

    lines.append("## 塌缩观察")
    outs = list(samples.values())
    lines.append(f"- 5 个输出互不相同：{len(set(outs)) == len(outs)}"
                 f"（相同对数：{len(outs) - len(set(outs))}）")
    lines.append(f"- 空输出：{[k for k, v in samples.items() if not v.strip()]}")
    code = samples["代码"]
    lines.append("- 代码续写含代码字符 "
                 f"{any(c in code for c in '():=')}")
    lines.append("")
    lines.append("## 新句子 loss（指示用）")
    for name, text in FRESH.items():
        ids = np.asarray(tok.encode(text), dtype=np.int64)
        loss = model.loss(ids[:-1][None, :], ids[1:][None, :])
        lines.append(f"- {name}：{loss:.4f}（{len(ids)} tokens）")
        print(f"[val {name}] {loss:.4f}")

    log = HERE / "GENERALIST_PILOT_RUN.log"
    if log.exists():
        tail = [l for l in log.read_text(encoding="utf-8").splitlines()
                if "train_loss" in l][-5:]
        lines.append("")
        lines.append("## 训练 loss 尾巴")
        lines.extend(f"- {l}" for l in tail)

    out_path = HERE / "GENERALIST_PILOT_GATE.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

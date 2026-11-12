#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""全能线全面评估（只呈证据，不判 PASS/FAIL，不拦训练）。

对指定产品做四件事：
  1. 经 world_loop.load_world_model 加载（验证循环路径能读它），
     核对 meta（step/format/config/params/词表/配比）；
  2. 6 域各一个贪心续写样本，看格式对不对、回没回在题上、
     6 个输出有没有塌成同一个东西；
  3. 每域 5 句训练文件里没有的新句子，算平均 loss（指示用）；
     常识域用留出的新维基导言（knowledge_val_heldout.json），
     其余 5 域用手写新句子；
  4. 读全部 190 块产品的 meta，做 loss 轨迹小结。

用法：python3 eval_generalist_full.py out/generalist_s57300.npz
  写出 GENERALIST_EVAL_S<累计步数>.md。
"""
import glob
import json
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
SPLIT_RE = re.compile(r"(?<=[。！？；\n])")

PROMPTS = {
    "经典": "话说",
    "代码": "def add(a, b):",
    "对话": "用户：你好。\n助手：",
    "推理": "问：三加四等于几？答案是",
    "世界": ("我是欧米伽。当前人格悟空。\n世界状态。你在花果山。在场人物孙悟空、"
             "唐僧、猪八戒、沙僧、白龙马。可见物品蟠桃、金箍棒。随身物品无。"
             "时间第一步白天。\n问：我在哪？"),
    "常识": "地球是",
}

# 手写新句子（评估用，不进训练）。
FRESH = {
    "经典": [
        "山风吹过松林，松涛声一阵一阵。",
        "夜色渐深，山间起了薄雾，远处传来一声虎啸。",
        "老翁拄着拐杖，慢慢走过石桥，望着河水出神。",
        "秋风起时，黄叶满地，道观里静得只听见钟声。",
        "那人推开柴门，见院中一株老梅开得正好。",
    ],
    "代码": [
        "def subtract(a, b):\n    return a - b",
        "def square(n):\n    return n * n",
        "x = [1, 2, 3]\nprint(len(x))",
        "for i in range(5):\n    print(i)",
        "if x > 0:\n    print(x)",
    ],
    "对话": [
        "用户：明天天气怎么样。\n助手：明天晴天，适合出门。",
        "用户：帮我讲个笑话。\n助手：好啊，你听好了。",
        "用户：我饿了。\n助手：去吃碗面吧。",
        "用户：现在几点了。\n助手：快到中午了。",
        "用户：你喜欢什么颜色。\n助手：我喜欢蓝色。",
    ],
    "推理": [
        "问：五加七等于几？答案是十二。",
        "问：九乘三等于几？答案是二十七。",
        "问：四十减十五等于几？答案是二十五。",
        "问：八加九等于几？答案是十七。",
        "问：六加六等于几？答案是十二。",
    ],
    "世界": [
        "世界状态。你在天宫。在场人物孙悟空、唐僧、猪八戒、沙僧、白龙马。可见物品无。随身物品无。时间第八步白天。",
        "世界状态。你在流沙河。在场人物孙悟空、唐僧、猪八戒、沙僧、白龙马。可见物品无。随身物品月牙铲。时间第三步白天。",
        "世界状态。你在高老庄。在场人物孙悟空、唐僧、猪八戒、沙僧、白龙马。可见物品九齿钉耙。随身物品无。时间第六步夜晚。",
        "世界状态。你在南海。在场人物孙悟空、唐僧、猪八戒、沙僧、白龙马、观音。可见物品紫金钵盂。随身物品蟠桃。时间第十步白天。",
        "世界状态。你在水帘洞。在场人物孙悟空、唐僧、猪八戒、沙僧、白龙马。可见物品无。随身物品金箍棒。时间第二步白天。",
    ],
}


def cjk_ratio(text):
    if not text:
        return 0.0
    return sum(1 for c in text if "\u4e00" <= c <= "\u9fff") / len(text)


def heldout_knowledge_sentences(vocab, want=5):
    path = HERE / "generalist_corpus/knowledge_val_heldout.json"
    articles = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for a in articles:
        for chunk in SPLIT_RE.split(a.get("extract", "")):
            s = chunk.strip()
            if len(s) >= 4 and all(c in vocab for c in s):
                out.append(s)
            if len(out) >= want:
                return out
    return out


def read_meta(path):
    with np.load(path, allow_pickle=False) as data:
        return json.loads(bytes(data["meta_json"]).decode("utf-8"))


def trajectory():
    rows = []
    for p in sorted(glob.glob(str(HERE / "out/generalist_s*.npz"))):
        if "_last" in p:
            continue
        try:
            m = read_meta(p)
        except Exception:
            continue
        extra = m.get("extra", {})
        rows.append({
            "file": Path(p).name,
            "step": m.get("step"),
            "last_loss": extra.get("last_train_loss"),
            "ratios": extra.get("ratios"),
        })
    rows.sort(key=lambda r: r["file"])
    return rows


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else str(HERE / "out/generalist_s57000.npz")
    model, tok, step, arch = wl.load_world_model(ckpt)
    meta = read_meta(ckpt)
    extra = meta.get("extra", {})
    cum = int(Path(ckpt).stem.split("_s")[1])
    print(f"loaded: step={step} arch={arch} vocab={tok.vocab_size}")
    print(f"meta: format={meta.get('format_version')} "
          f"params={sum(int(v.size) for v in model.params.values()):,} "
          f"ratios={extra.get('ratios')}")

    lines = [f"# 全能线全面评估（产品 {Path(ckpt).name}，累计 {cum} 步）\n"]
    lines.append("## 产品身份")
    lines.append(f"- step（块内）: {step}；format: {meta.get('format_version')}；"
                 f"arch: {arch}")
    lines.append(f"- config: {model.config()}")
    lines.append(f"- 词表: {tok.vocab_size}；配比: {extra.get('ratios')}")
    lines.append("")

    lines.append("## 贪心续写样本（32 tokens）")
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
        lines.append(f"### {name}（经 {path}）")
        lines.append(f"提示：{prompt!r}")
        lines.append(f"续写：{text!r}")
        lines.append(f"汉字占比：{cjk_ratio(text):.2f}")
        lines.append("")
        print(f"[{name}] {text!r}")

    lines.append("## 塌缩观察")
    outs = list(samples.values())
    lines.append(f"- 6 个输出互不相同：{len(set(outs)) == len(outs)}"
                 f"（相同对数：{len(outs) - len(set(outs))}）")
    lines.append(f"- 空输出：{[k for k, v in samples.items() if not v.strip()]}")
    lines.append(f"- 代码续写含代码字符：{any(c in samples['代码'] for c in '():=')}")
    lines.append("")

    lines.append("## 新句子 loss（每域 5 句，指示用）")
    vocab = set(json.loads((HERE / "generalist_vocab.json").read_text(
        encoding="utf-8"))["chars"])
    fresh = dict(FRESH)
    fresh["常识"] = heldout_knowledge_sentences(vocab)
    print(f"[val 常识] held-out sentences: {len(fresh['常识'])}")
    val_means = {}
    for name, sents in fresh.items():
        losses = []
        for text in sents:
            ids = np.asarray(tok.encode(text), dtype=np.int64)
            if len(ids) < 2:
                continue
            loss = model.loss(ids[:-1][None, :], ids[1:][None, :])
            losses.append(loss)
        mean = float(np.mean(losses)) if losses else float("nan")
        val_means[name] = mean
        lines.append(f"- {name}：平均 {mean:.4f}（{len(losses)} 句，"
                     f"各句：{', '.join(f'{x:.2f}' for x in losses)}）")
        print(f"[val {name}] {mean:.4f} over {len(losses)}")
    lines.append("")

    lines.append("## 190 块 loss 轨迹（各产品 meta 的 last_train_loss）")
    rows = trajectory()
    finals = [r["last_loss"] for r in rows if r["last_loss"] is not None]
    lines.append(f"- 产品数：{len(rows)}；有 last_train_loss：{len(finals)}")
    if finals:
        arr = np.asarray(finals, dtype=np.float64)
        lines.append(f"- 全体：平均 {arr.mean():.3f}，中位 {np.median(arr):.3f}，"
                     f"最高 {arr.max():.3f}，最低 {arr.min():.3f}")
        lines.append(f"- 终值 >4 的块：{int((arr > 4).sum())} 个；"
                     f"终值 <0.5 的块：{int((arr < 0.5).sum())} 个")
        if len(arr) >= 10:
            lines.append(f"- 前 10 块终值平均：{arr[:10].mean():.3f}；"
                         f"后 10 块终值平均：{arr[-10:].mean():.3f}")
        worst = sorted(rows, key=lambda r: (r["last_loss"] is None, r["last_loss"]),
                       reverse=True)[:5]
        lines.append("- 终值最高的 5 块："
                     + "、".join(f"{r['file']}={r['last_loss']:.2f}" for r in worst))
    lines.append("")

    out_path = HERE / f"GENERALIST_EVAL_S{cum}.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

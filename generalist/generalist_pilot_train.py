#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""全能试点训练器：从头训一个 d512 通用小模型（100 步试点）。

复用 dual 模型文件本身的全部机制（DualTransformerNP、AdamWNP、
train_steps、save_checkpoint），本脚本只做三件新事：
  1. 分词器：generalist_vocab.json（通用词表），不用 from_text 现算；
  2. 架构：d512/4 层/8 头/kv 4、merge single、RMS、RoPE、SwiGLU、
     不 tied（与老 d128 同家族，只放大）；
  3. 混合采样：batch_fn 按设计配比抽域（经典 25 / 代码 20 / 对话 20 /
     常识 15 / 推理 10 / 世界 10），再在域内随机取窗口——配比是采样权重，
     不是文件大小比，小文件循环采样。常识域 2026-11-11 落地（A 维基
     百科导言＋B《幼学琼林》，见 GENERALIST_CORPUS.md），配比自第 192 块
     起切回设计值；此前 1–191 块为 5 域试点配比。

域文本来源（GENERALIST_CORPUS.md）：
  经典：西游记全文＋《儒林外史》（Gutenberg #24032 繁体本经 OpenCC t2s
    转简体，均按 4622 词表整句过滤，见 generalist_corpus/）；
  常识：knowledge_wiki.txt（维基百科导言，CC BY-SA 4.0）＋
    knowledge_youxue.txt（《幼学琼林》Gutenberg 公版），两者均已按
    4622 词表整句过滤（见 knowledge_provenance.md）；
  推理：现生成的中文算术/抄写/记忆格式（与协议记录同分布，种子固定）；
  世界：world_tool_records_v8 与 self_report_records_full 的文本字段。
  明确不含：grok 包、用户粘贴的手机对话、旧检查点。

用法：
  python3 generalist_pilot_train.py --steps 100 --out out/generalist_pilot.npz
  python3 generalist_pilot_train.py --steps 3 --out /tmp/bench.npz  # 步速实测
"""
import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.expanduser("~/workspace/dual_side/xiyouji_modern_trainer_1"))
sys.path.insert(0, os.path.expanduser("~/workspace/omega-chat"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from dual_transformer_numpy import CharTokenizer, DualTransformerNP, read_checkpoint, train_steps

HERE = Path(__file__).resolve().parent
MIRROR = Path(os.path.expanduser("~/workspace/omega-chat"))
VOCAB_JSON = HERE / "generalist_vocab.json"

RATIOS = [("经典", 0.25), ("代码", 0.20), ("对话", 0.20),
          ("常识", 0.15), ("推理", 0.10), ("世界", 0.10)]

_ZH = "零一二三四五六七八九十"


def zh(n):
    """0-99 整数 -> 中文读法（与协议记录同款，不用阿拉伯数字）。"""
    if n < 10:
        return _ZH[n]
    tens, ones = divmod(n, 10)
    head = "十" if tens == 1 else _ZH[tens] + "十"
    return head + (_ZH[ones] if ones else "")


def reasoning_text(seed=7):
    """中文算术/抄写/记忆格式文本（现生成，种子固定，与协议同分布）。"""
    rng = np.random.default_rng(seed)
    lines = []
    for _ in range(400):
        a, b = int(rng.integers(1, 20)), int(rng.integers(1, 20))
        lines.append(f"问：{zh(a)}加{zh(b)}等于几？答案是{zh(a + b)}。")
    for _ in range(200):
        a, b = int(rng.integers(2, 10)), int(rng.integers(2, 10))
        lines.append(f"问：{zh(a)}乘{zh(b)}等于几？答案是{zh(a * b)}。")
    for _ in range(300):  # 减法（2026-11-12 补）：被减数 1..19，减数 ≤ 被减数，结果非负
        a = int(rng.integers(1, 20))
        b = int(rng.integers(1, a + 1))
        lines.append(f"问：{zh(a)}减{zh(b)}等于几？答案是{zh(a - b)}。")
    for _ in range(200):  # 除法（2026-11-12 补）：整除，商 1..9
        b = int(rng.integers(2, 10))
        k = int(rng.integers(1, 10))
        lines.append(f"问：{zh(b * k)}除以{zh(b)}等于几？答案是{zh(k)}。")
    for _ in range(200):
        w = rng.choice(["苹果", "花果山", "金箍棒", "蟠桃", "南海", "取经路"])
        lines.append(f"记住这个词：{w}。")
        lines.append(f"刚才让你记住什么。{w}。")
    return "\n".join(lines) + "\n"


def domain_texts(tok):
    texts = {}
    texts["经典"] = ((MIRROR / "Xiyouji.txt").read_text(encoding="utf-8")
                     + (HERE / "generalist_corpus/classics_rulin.txt").read_text(
                         encoding="utf-8"))
    main_py = [p for p in glob.glob(str(MIRROR / "**/*.py"), recursive=True)
               if "__pycache__" not in p
               and "/archive/" not in p and "/legacy/" not in p]
    texts["代码"] = "".join(Path(p).read_text(encoding="utf-8")
                            for p in sorted(main_py))
    texts["对话"] = (HERE / "generalist_corpus/dialogue_pilot.txt").read_text(
        encoding="utf-8")
    texts["推理"] = reasoning_text()
    texts["常识"] = (
        (HERE / "generalist_corpus/knowledge_wiki.txt").read_text(
            encoding="utf-8")
        + (HERE / "generalist_corpus/knowledge_youxue.txt").read_text(
            encoding="utf-8"))
    world = []
    # v9 修法 A（2026-11-12，用户选 C）：摘要地点压后，问答距离 ~53→7 字
    # v10（2026-11-12，用户选 B＋探针格式）：加探针格式记录 600 条
    # （摘要＋探针原问＋标准答案，无锚定行、无“问：”包装，与循环探针同格式）
    for name in ("world_tool_records_v9.jsonl", "self_report_records_v9.jsonl",
                 "probe_format_records.jsonl"):
        for line in open(HERE / name, encoding="utf-8"):
            world.append(json.loads(line)["text"])
    texts["世界"] = "\n".join(world)
    ids = {}
    for name, text in texts.items():
        arr = np.asarray(tok.encode(text), dtype=np.int64)
        assert len(arr) > 200, f"domain {name} too short: {len(arr)}"
        ids[name] = arr
        print(f"domain {name}: {len(text)} chars -> {len(arr)} tokens")
    return ids


def make_batch_fn(ids, seq_len, batch_size=8, seed=7):
    names = [n for n, _ in RATIOS]
    weights = np.asarray([w for _, w in RATIOS], dtype=np.float64)
    weights = weights / weights.sum()
    rng = np.random.default_rng(seed)

    def batch_fn(r):
        # 注意：train_steps 传进来的 r 就是它自己的 rng；这里用独立 rng
        # 抽域，保证配比不受训练 rng 影响。
        doms = rng.choice(len(names), size=batch_size, p=weights)
        inputs, targets = [], []
        for d in doms:
            arr = ids[names[d]]
            s = int(rng.integers(0, len(arr) - seq_len - 1))
            inputs.append(arr[s:s + seq_len])
            targets.append(arr[s + 1:s + seq_len + 1])
        return np.stack(inputs), np.stack(targets)

    return batch_fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=None,
                    help="本块步数；不给则默认 100，断点续跑时默认跑满"
                         "原块剩余步数（schedule_total_steps - 已跑步数）")
    ap.add_argument("--out", default="out/generalist_pilot.npz")
    ap.add_argument("--save-every", type=int, default=50,
                    help="每 N 步把带优化器与随机数状态的最新状态存到"
                         "out 的 _last 文件（如 out/generalist_s600_last.npz），"
                         "被重启杀掉时最多丢 N 步；0 表示不存断点")
    ap.add_argument("--resume", default=None,
                    help="断点续跑：_last 文件路径，接着原块的优化器与"
                         "cosine 曲线继续跑（与 --init 互斥，--init 只取参数、"
                         "优化器全新）。--steps 不给时自动跑满剩余步数")
    ap.add_argument("--d_model", type=int, default=512)
    ap.add_argument("--n_blocks", type=int, default=4)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--n_kv_heads", type=int, default=4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=128)  # B（2026-11-12）：64→128
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--init", default=None,
                    help="上一块产品路径：只取参数热启动，优化器与学习率"
                         "曲线全新（与老线补丁轮同款做法）")
    args = ap.parse_args()
    if args.init and args.resume:
        raise SystemExit("--init 与 --resume 互斥：新块热启动用 --init，"
                         "同块断点续跑用 --resume")

    meta = json.loads(VOCAB_JSON.read_text(encoding="utf-8"))
    assert meta["type"] == "char"
    tok = CharTokenizer(meta["chars"])
    print(f"vocab: {tok.vocab_size} (BOS/EOS/UNK 0/1/2)")
    ids = domain_texts(tok)

    model = DualTransformerNP(
        d_model=args.d_model, n_blocks=args.n_blocks, n_heads=args.n_heads,
        merge_method="single", use_rms=True, max_seq=512, seed=args.seed,
        dtype=np.float32, vocab_size=tok.vocab_size, tie_embeddings=False,
        n_kv_heads=args.n_kv_heads, pos_mode="rope", ff_mode="swiglu")
    n_params = sum(int(v.size) for v in model.params.values())
    print(f"model: d={args.d_model} blocks={args.n_blocks} heads={args.n_heads} "
          f"kv={args.n_kv_heads} params={n_params:,}")
    print(f"train: seq_len={args.seq} batch_size={args.batch}")
    resume_from = None
    steps = args.steps
    if args.init:
        ck = read_checkpoint(args.init)
        assert ck["meta"]["config"] == model.config(), (
            f"init config mismatch: {ck['meta']['config']} vs {model.config()}")
        assert (ck["meta"]["extra"]["tokenizer"]["chars"] == meta["chars"]), (
            "init tokenizer mismatch")
        for k in model.params:
            model.params[k] = ck["params"][k].astype(model.dtype)
        print(f"warm start from {args.init} (step {ck['meta']['step']}), "
              "fresh optimizer + fresh cosine schedule")
    if args.resume:
        ck = read_checkpoint(args.resume)
        assert ck["meta"]["config"] == model.config(), (
            f"resume config mismatch: {ck['meta']['config']} vs {model.config()}")
        assert (ck["meta"]["extra"]["tokenizer"]["chars"] == meta["chars"]), (
            "resume tokenizer mismatch")
        assert ck["meta"]["has_optimizer"], "断点文件没有优化器状态，无法续跑"
        resume_from = args.resume
        start_step = int(ck["meta"]["step"])
        total = int(ck["meta"]["extra"].get("schedule_total_steps", start_step))
        if steps is None:
            steps = total - start_step
            print(f"resume from {args.resume} (step {start_step}/{total}), "
                  f"remaining steps = {steps}")
        print(f"resume keeps optimizer + cosine schedule (total {total})")
    if steps is None:
        steps = 100
    if steps <= 0:
        raise SystemExit(f"步数为 {steps}，断点已跑满，无需续跑")
    batch_fn = make_batch_fn(ids, args.seq, batch_size=args.batch,
                             seed=args.seed)
    t0 = time.time()
    train_steps(
        model, steps=steps, batch_size=args.batch, seq_len=args.seq,
        lr=args.lr, log_every=max(1, steps // 10),
        schedule="cosine", warmup_steps=10, seed=args.seed,
        batch_fn=batch_fn, checkpoint_path=args.out,
        resume_from=resume_from, save_every=(args.save_every or None),
        extra_meta={"tokenizer": tok.to_meta(), "recipe": "generalist-pilot",
                    "ratios": dict(RATIOS), "seq_len": args.seq})
    dt = time.time() - t0
    print(f"done: {steps} steps in {dt:.1f}s "
          f"({dt / max(steps, 1):.2f} s/step) -> {args.out}")


if __name__ == "__main__":
    main()

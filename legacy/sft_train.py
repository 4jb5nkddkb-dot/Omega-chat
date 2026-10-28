#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""sft_train.py — 协议 SFT 训练器（设备版，配方已在本地替身检查点上全量验证）

做什么：从 checkpoints/modern_ckpt.npz 续训，让模型学会工具协议——自发发出
《计算》/《记住》/《回忆》调用、把题面操作数原样抄进调用、读回工具
结果作答。训练完自带验收（三例 + 两探针 + 162 组合抄写探针）。

需要（仓库文件）：legacy/modern_transformer_numpy.py、src/omega_chat/dual_transformer_numpy.py、
src/omega_chat/agent_runtime.py、tools/make_tool_data.py、src/omega_chat/memory_store.py、data/Xiyouji.txt、checkpoints/modern_ckpt.npz

安全：开训前自动把 checkpoints/modern_ckpt.npz 复制为 checkpoints/modern_ckpt_v1_backup.npz 并打印
sha256（已存在则不覆盖）；成果写 checkpoints/modern_ckpt_sft.npz，绝不覆盖原检查点。

配方（v5 定版，v4 + 答案复制值加权）：记录对齐采样（问题与调用同窗）、
加权 masked loss（操作数 ×5 / 调用其余 ×2 / 答案模板 ×2 / 答案中被复制
的值 ×5 / 语料回放 ×1）、协议占比 0.5、lr 5e-4 cosine、2000 步、batch 8。
本地验证（替身 d64 ckpt, val 5.379）：三例全通，抄写探针 162/162，
162 道计算题走 run_agent 全通（v4 是 161/162，挂在九乘九的第二位数字上），
val 5.284 无退化。真检查点（val 4.688）余量更足。

用法：直接运行；或 python sft_train.py steps=200 lr=5e-4 之类临时改配置。
预计耗时：手机上约 10–25 分钟（2000 步）。
"""
from __future__ import annotations

import hashlib
import inspect
import shutil
import sys
import tempfile
import textwrap
import time
from pathlib import Path
_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent if _HERE.name in ("tests", "tools", "legacy") else _HERE
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np

from omega_chat import dual_transformer_numpy as dtn
import modern_transformer_numpy as mtn
from omega_chat.agent_runtime import (Protocol, ToolRouter, calc_zh, contains_verifier,
                           generate_segment, load_modern_model, run_agent)
from make_tool_data import build_samples, check_records
from omega_chat.memory_store import MemoryStore, make_memory_tools

BASE = ROOT
CONFIG = dict(steps=2000, lr=5e-4, batch=8, seq=64, proto_share=0.5,
              w_op=5.0, w_call=2.0, w_ans=2.0, seed=7, probe=True)
P = Protocol()
CN = "一二三四五六七八九"


def build_weighted_loss():
    """模型自带 loss_and_grads 的加权版：只改 loss 头（dlogits 按权重缩放），
    反向传播逐字沿用模型源码——模型怎么改，这里自动跟随。"""
    src = textwrap.dedent(inspect.getsource(mtn.ModernTransformerNumpy.loss_and_grads))
    src = src.replace("def loss_and_grads(self, tokens, targets):",
                      "def weighted_loss_and_grads(self, tokens, targets, weights):")
    old_loss = "loss = -float(np.sum(np.log(probs[np.arange(n), flat_t] + 1e-12)) / n)"
    new_loss = ("Wflat = np.asarray(weights, dtype=np.float64).reshape(-1)\n"
                "    Wsum = float(np.sum(Wflat)) + 1e-12\n"
                "    loss = -float(np.sum(Wflat * np.log(probs[np.arange(n), flat_t] + 1e-12)) / Wsum)")
    assert old_loss in src, "模型 loss 源码已变，需同步更新本文件"
    src = src.replace(old_loss, new_loss)
    old_dl = ("dlogits = probs\n"
              "    dlogits[np.arange(n), flat_t] -= 1.0\n"
              "    dlogits = (dlogits / n).reshape(B, S, V)")
    new_dl = ("dlogits = probs * Wflat[:, None]\n"
              "    dlogits[np.arange(n), flat_t] -= Wflat\n"
              "    dlogits = (dlogits / Wsum).reshape(B, S, V)")
    assert old_dl in src, "模型 loss 源码已变，需同步更新本文件"
    src = src.replace(old_dl, new_dl)
    ns = dict(vars(mtn))
    exec(src, ns)
    return ns["weighted_loss_and_grads"]


def archive_base_ckpt():
    src = BASE / "checkpoints" / "modern_ckpt.npz"
    bak = BASE / "checkpoints" / "modern_ckpt_v1_backup.npz"
    assert src.exists(), "缺少 checkpoints/modern_ckpt.npz"
    h = hashlib.sha256(src.read_bytes()).hexdigest()
    if bak.exists():
        print(f"[存档] 备份已存在，跳过：{bak.name}")
    else:
        shutil.copy2(src, bak)
        print(f"[存档] 已备份原检查点 -> {bak.name}")
    print(f"[存档] checkpoints/modern_ckpt.npz sha256 = {h[:16]}...（完整值见 sft 报告）")
    return h


def val_loss(model, val_ids, cfg, n_batches=16, seed=0):
    rng = np.random.default_rng(seed)
    tot = 0.0
    for _ in range(n_batches):
        xs, ys = [], []
        for _ in range(cfg["batch"]):
            s = int(rng.integers(0, len(val_ids) - cfg["seq"] - 1))
            xs.append(val_ids[s:s + cfg["seq"]])
            ys.append(val_ids[s + 1:s + cfg["seq"] + 1])
        loss, _ = model.loss_and_grads(np.asarray(xs), np.asarray(ys))
        tot += float(loss)
    return tot / n_batches


def record_example(rec, tok, cfg):
    ids = tok.encode(rec["text"], add_bos=True, add_eos=True)
    w = np.zeros(len(ids), dtype=np.float64)
    for s, e in rec["spans"]:
        seg = rec["text"][s:e]
        is_call = seg.startswith(P.call_open)
        for j in range(s, e):
            pos = 1 + j  # +1 BOS；权重落在“预测 ids[pos]”的位置上
            if pos >= len(ids):
                break
            if not is_call:
                # v5：答案里被复制的值（“答案是”之后、句号之前）也是从结果里
                # 抄来的目标，和操作数同级 ×5；其余答案模板 ×2。
                if seg.startswith("答案是") and 3 <= j - s < len(seg) - 1:
                    w[pos] = cfg["w_op"]
                else:
                    w[pos] = cfg["w_ans"]
            else:
                after_name = j > s + len(P.call_open) + 2  # 《 + 两字工具名之后
                w[pos] = cfg["w_op"] if (after_name and rec["text"][j] != P.call_close) \
                    else cfg["w_call"]
    w[-1] = cfg["w_ans"]  # EOS
    return ids[:-1], ids[1:], w[1:]


def pad_batch(items, pad_id):
    L = max(len(x) for x, _, _ in items)
    xs, ys, ws = [], [], []
    for x, y, w in items:
        n = L - len(x)
        xs.append(list(x) + [pad_id] * n)
        ys.append(list(y) + [pad_id] * n)
        ws.append(list(w) + [0.0] * n)
    return np.asarray(xs), np.asarray(ys), np.asarray(ws)


def acceptance(model, tok, label):
    store = MemoryStore(Path(tempfile.mkdtemp()) / "m.jsonl")
    router = ToolRouter({"计算": calc_zh, **make_memory_tools(store)}, tok)
    cases = {}
    r1 = run_agent(model, router, "问：七加八等于几？", contains_verifier("一五"),
                   max_new=64, max_turns=4, max_retries=2, temperature=0.0)
    cases["计算"] = r1.ok
    print(f"[{label}] 计算 七加八: {'✓' if r1.ok else '✗'} 答={r1.text[:20]!r}", flush=True)
    r2 = run_agent(model, router, "请记住：绿茶是不加糖。", contains_verifier("记下了"),
                   max_new=64, max_turns=4, max_retries=2, temperature=0.0)
    r3 = run_agent(model, router, "问：绿茶是什么？", contains_verifier("不加糖"),
                   max_new=64, max_turns=4, max_retries=2, temperature=0.0)
    cases["记忆"] = r2.ok and r3.ok
    print(f"[{label}] 记忆 记住/回忆: {'✓' if cases['记忆'] else '✗'}", flush=True)
    r4 = run_agent(model, router, "问：你好？", contains_verifier("你好"),
                   max_new=32, max_turns=2, max_retries=1, temperature=0.0)
    cases["问候"] = r4.ok
    print(f"[{label}] 问候: {'✓' if r4.ok else '✗'} 答={r4.text[:20]!r}", flush=True)
    for q, want in [("问：五加四等于几？", "九"), ("问：三乘七等于几？", "二一")]:
        rr = run_agent(model, ToolRouter({"计算": calc_zh, **make_memory_tools(
            MemoryStore(Path(tempfile.mkdtemp()) / "m.jsonl"))}, tok), q,
            contains_verifier(want), max_new=64, max_turns=4, max_retries=2,
            temperature=0.0)
        cases[q] = rr.ok
        print(f"[{label}] 探针 {q} {'✓' if rr.ok else '✗'}", flush=True)
    return cases


def operand_probe(model, tok):
    ok = 0
    for a in range(1, 10):
        for b in range(1, 10):
            for op in ("加", "乘"):
                prefix = f"问：{CN[a-1]}{op}{CN[b-1]}等于几？" + \
                    f"{P.call_open}计算{P.name_end}"
                cur = list(tok.encode(prefix, add_bos=True))
                gen = []
                for _ in range(3):
                    logits, _ = model.forward(np.asarray([cur], dtype=np.int64))
                    nxt = int(np.argmax(logits[0, -1]))
                    gen.append(nxt)
                    cur.append(nxt)
                ok += int(gen == tok.encode(f"{CN[a-1]}{op}{CN[b-1]}"))
    return ok


def main():
    cfg = dict(CONFIG)
    for arg in sys.argv[1:]:
        if "=" in arg:
            k, v = arg.split("=", 1)
            cfg[k] = type(CONFIG[k])(v)
    t0 = time.time()
    print("=" * 56)
    print(f"SFT 协议训练（设备版）配置：{cfg}")
    base_hash = archive_base_ckpt()

    model, tok, step0 = load_modern_model(BASE / "checkpoints" / "modern_ckpt.npz", BASE / "data" / "Xiyouji.txt")
    text = (BASE / "data" / "Xiyouji.txt").read_text(encoding="utf-8").replace("\r\n", "\n")
    all_ids = np.asarray(tok.encode(text, add_bos=True, add_eos=True), dtype=np.int64)
    split = int(len(all_ids) * 0.95)
    train_ids, val_ids = all_ids[:split], all_ids[split:]
    v0 = val_loss(model, val_ids, cfg)
    print(f"基座检查点 step={step0} 西游记 val={v0:.4f}", flush=True)
    print("--- 训练前验收（预期全 ✗，模型还没见过协议）---", flush=True)
    acceptance(model, tok, "训前")

    records = build_samples(n_calc=3000, n_mem=1500, seed=cfg["seed"])
    greet = [r for r in records if r["text"].startswith("问：你好？")]
    records = records + greet * 100
    problems = check_records(records, tok)
    assert not problems, problems[:3]
    examples = [record_example(r, tok, cfg) for r in records]
    print(f"协议记录 {len(records)} 条（问候已补 ×100）", flush=True)

    wlag = build_weighted_loss()
    opt = dtn.AdamWNP(model.params, lr=cfg["lr"], weight_decay=0.01)
    rng = np.random.default_rng(0)
    pad_id = int(tok.encode("。")[0])
    SEQ = cfg["seq"]

    def corpus_item():
        s = int(rng.integers(0, len(train_ids) - SEQ - 1))
        return (train_ids[s:s + SEQ], train_ids[s + 1:s + SEQ + 1],
                np.ones(SEQ, dtype=np.float64))

    run_loss, nb = 0.0, 0
    for step in range(1, cfg["steps"] + 1):
        opt.lr = dtn.lr_at(step, cfg["lr"], "cosine", warmup_steps=20,
                           total_steps=cfg["steps"])
        items = [examples[int(rng.integers(0, len(examples)))]
                 if rng.random() < cfg["proto_share"] else corpus_item()
                 for _ in range(cfg["batch"])]
        xs, ys, ws = pad_batch(items, pad_id)
        loss, grads = wlag(model, xs, ys, ws)
        dtn.clip_grad_norm(grads, 1.0)
        opt.step(model.params, grads)
        run_loss += float(loss)
        nb += 1
        if step % 100 == 0:
            el = time.time() - t0
            eta = el / step * (cfg["steps"] - step)
            print(f"  step {step}/{cfg['steps']} loss={run_loss / nb:.4f} "
                  f"已用 {el / 60:.1f} 分 剩约 {eta / 60:.1f} 分", flush=True)
            run_loss, nb = 0.0, 0

    out = BASE / "checkpoints" / "modern_ckpt_sft.npz"
    v1 = val_loss(model, val_ids, cfg)
    payload = {f"param::{k}": v for k, v in model.params.items()}
    payload["cfg"] = np.asarray([64, 2, 4, 2, tok.vocab_size, step0 + cfg["steps"]])
    payload["chars"] = np.asarray(tok.chars)
    payload["val"] = np.asarray([v1])
    np.savez(out, **payload)
    print(f"已保存 {out.name}：西游记 val {v0:.4f} -> {v1:.4f}", flush=True)

    model2, tok2, _ = load_modern_model(out, BASE / "data" / "Xiyouji.txt")
    print("--- 训练后验收 ---", flush=True)
    cases = acceptance(model2, tok2, "训后")
    if cfg["probe"]:
        n_ok = operand_probe(model2, tok2)
        print(f"[训后] 抄写探针：{n_ok}/162 组合全对" if n_ok == 162 else
              f"[训后] 抄写探针：{n_ok}/162", flush=True)
        cases["抄写探针"] = n_ok == 162
    p = tok2.encode("悟空", add_bos=True)
    gen = generate_segment(model2, p, 60, temperature=0.8, rng=np.random.default_rng(1))
    print(f"[生成抽查] {tok2.decode(gen[len(p):])[:50]!r}", flush=True)
    all_ok = all(cases.values())
    print("=" * 56)
    print(f"SFT 验收总结果：{'全部通过 ✓' if all_ok else '未全通 ✗（见上）'}"
          f"  基座 sha256={base_hash}")
    print(f"成果文件：checkpoints/modern_ckpt_sft.npz（原检查点与备份未动）"
          f"  总耗时 {(time.time() - t0) / 60:.1f} 分")
    print("=" * 56)


if __name__ == "__main__":
    main()

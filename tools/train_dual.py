#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
合并版双头的直跑训练脚本 —— 无 UI,Pythonista 里直接运行即可。

依赖：src/omega_chat/dual_transformer_numpy.py（合并版）/ data/Xiyouji.txt。
和 train_modern.py 一样的用法:改下方 CONFIG,或 key=value 覆盖。

默认配方 = “现代味”(实测与现代架构同尺度打平):
  merge=single / GQA kv=2 / SwiGLU / RMSNorm / RoPE / GPT-2 初始化 / 不绑定
想换味只改 CONFIG 三四个键:
  双头味:merge="average"(其余不动,就是双分支版)
  GPT 味:merge="single", kv_heads=4, ff="gelu", pos="learned",
          use_rms=False, tie=True

检查点(ckpt_dual/ 目录):
  model.npz       最佳验证权重
  model_last.npz  最新状态(含优化器状态和 RNG 状态)
resume=True 是真续训:权重、Adam 动量、步数、原来的学习率曲线都接着走,
steps 表示“再加多少步”。resume=False 则从头训,会覆盖同名检查点——
想留旧模型先把 ckpt_dual 改个名。

训完自动生成两段:温度采样一段;greedy + 解码控制一段(治复读)。
"""
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path
_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent if _HERE.name in ("tests", "tools", "legacy") else _HERE

import numpy as np

BASE = ROOT
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE / "src"))

LOG_PATH = BASE / "boot_log.txt"
_t0 = time.time()

CONFIG = {
    "steps": 100,
    "d_model": 128,
    "blocks": 4,
    "heads": 4,
    "kv_heads": 2,
    "merge": "single",        # single / average / concat
    "use_rms": True,          # True=RMSNorm,False=LayerNorm
    "pos": "rope",            # rope / learned
    "ff": "swiglu",           # swiglu / gelu
    "init": "gpt2",           # gpt2 / legacy
    "tie": False,
    "batch_size": 1,
    "seq_len": 64,
    "lr": 2e-3,
    "warmup_steps": 10,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "eval_every": 50,
    "seed": 42,
    "resume": True,
    "decode_control": True,   # 生成时开解码控制(防复读)
    "prompt": "悟空",
    "new_tokens": 32,
}

CKPT_DIR = BASE / "ckpt_dual"
BEST = CKPT_DIR / "modern_ckpt.npz"


def log(msg: str) -> None:
    line = f"[dual {time.time() - _t0:7.3f}s] {msg}"
    print(line, flush=True)
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
    except Exception:
        pass


def main() -> None:
    cfg = dict(CONFIG)
    argv = [a for a in sys.argv[1:] if a]
    for a in list(argv):  # key=value overrides, e.g. steps=3000 merge=average
        if "=" in a:
            k, v = a.split("=", 1)
            if k in cfg:
                if isinstance(cfg[k], bool):
                    cfg[k] = v.lower() in ("true", "1", "yes")
                else:
                    cfg[k] = type(cfg[k])(v)
                argv.remove(a)

    from omega_chat.dual_transformer_numpy import (CharTokenizer, DecodeController,
                                        DualTransformerNP, last_checkpoint_path,
                                        train_steps)

    log("=== 合并版双头直跑训练启动 ===")
    text = (BASE / "data" / "Xiyouji.txt").read_text(encoding="utf-8").replace("\r\n", "\n")
    tok = CharTokenizer.from_text(text)
    ids = np.asarray(tok.encode(text, add_bos=True, add_eos=True), dtype=np.int64)
    split = int(len(ids) * 0.95)
    train_ids, val_ids = ids[:split], ids[split:]
    log(f"词表={tok.vocab_size} 训练token={len(train_ids)} 验证token={len(val_ids)}")

    resume_from = None
    latest = last_checkpoint_path(BEST)
    if cfg["resume"] and latest.exists():
        model, _ = DualTransformerNP.load_checkpoint(latest)
        resume_from = latest
        log(f"续训模式:从 {latest.name} 接着训(配置以检查点为准)")
    else:
        if cfg["resume"]:
            log("没有可续训的最新检查点,从头训练")
        model = DualTransformerNP(
            d_model=cfg["d_model"], n_blocks=cfg["blocks"], n_heads=cfg["heads"],
            merge_method=cfg["merge"], use_rms=cfg["use_rms"],
            vocab_size=tok.vocab_size, tie_embeddings=cfg["tie"],
            n_kv_heads=cfg["kv_heads"], pos_mode=cfg["pos"], ff_mode=cfg["ff"],
            init_style=cfg["init"], seed=cfg["seed"])
    log(f"参数量={model.num_params():,} 配置: d={cfg['d_model']} blocks={cfg['blocks']} "
        f"q/kv头={cfg['heads']}/{cfg['kv_heads']} merge={cfg['merge']} ff={cfg['ff']} "
        f"pos={cfg['pos']} batch={cfg['batch_size']} seq={cfg['seq_len']} "
        f"lr={cfg['lr']} steps={cfg['steps']} resume={cfg['resume']}")

    _, _, best_val, done_step = train_steps(
        model, train_ids, steps=cfg["steps"], batch_size=cfg["batch_size"],
        seq_len=cfg["seq_len"], lr=cfg["lr"], val_token_ids=val_ids,
        checkpoint_path=BEST, eval_every=cfg["eval_every"],
        resume_from=resume_from, seed=cfg["seed"],
        weight_decay=cfg["weight_decay"], max_grad_norm=cfg["max_grad_norm"],
        schedule="cosine", warmup_steps=cfg["warmup_steps"],
        extra_meta={"tokenizer": tok.to_meta()})
    log(f"训练完成:累计步数={done_step},最佳验证 loss = {best_val:.4f}")
    log(f"检查点:{BEST}(最佳) / {latest}(最新,可续训)")

    # 生成一律用最佳检查点
    best_model, _ = DualTransformerNP.load_checkpoint(BEST)
    prompt_ids = tok.encode(cfg["prompt"], add_bos=True)
    gen = best_model.generate_cached(prompt_ids, max_new_tokens=cfg["new_tokens"],
                                     temperature=0.8, top_p=0.95,
                                     rng=np.random.default_rng(1))
    log(f"生成示例(prompt={cfg['prompt']!r},温度0.8/top-p0.95):")
    print(tok.decode(gen), flush=True)
    if cfg["decode_control"]:
        ctrl = DecodeController(base_temperature=0.0, block_ngram=4, loop_len=8)
        gen2 = best_model.generate_cached(prompt_ids, max_new_tokens=60,
                                          temperature=0.0, controller=ctrl)
        log("生成示例(greedy + 解码控制):")
        print(tok.decode(gen2), flush=True)
        log(ctrl.summary())
    else:
        gen2 = best_model.generate_cached(prompt_ids, max_new_tokens=40, temperature=0.0)
        log("生成示例(greedy):")
        print(tok.decode(gen2), flush=True)
    log("=== 全部完成 ===")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("⏹ 被手动停止:最新状态在 ckpt_dual/model_last.npz,下次 resume=True 可续跑")
    except Exception:
        log("❌ 失败:\n" + traceback.format_exc())
        raise

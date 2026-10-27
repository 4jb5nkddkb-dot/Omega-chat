#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
现代架构训练 —— GQA + SwiGLU + RMSNorm + RoPE,纯 NumPy。

依赖(同目录):modern_transformer_numpy.py / dual_transformer_numpy.py / Xiyouji.txt

改进要点:
1. 检查点分两份:
   - modern_ckpt_last.npz:续跑用,每次评估都更新,续跑从这里接着训。
   - modern_ckpt_best.npz:验证 loss 最佳时更新,用于生成文本。
   旧版只存最佳点,续跑会丢掉最佳之后的全部进度;现在不会。
2. 保存优化器状态(动量等)与随机数状态,续跑不再"从零开始建动量"。
3. 续跑不再从 warmup 把学习率重新拉高到 3e-3,而是接着上次的学习率衰减,避免验证 loss 震荡。
4. 检查点元信息用 JSON 存储;架构或参数形状不一致时直接报错并给出提示,不再静默忽略。
5. 原子写入(先写临时文件再替换),写入中途断电或 Ctrl-C 不会损坏检查点。
6. Ctrl-C 时自动保存当前进度后退出。
7. 可选早停(patience);loss 非有限时跳过该步而不是污染参数。
8. 命令行支持 --key value 与 key=value;保留旧版"纯数字 = 步数、第二个纯数字 = d_model"的用法。
   另外支持 --generate-only,只用最佳检查点生成文本,不训练。

用法示例:
    python train_modern.py                       # 按 CONFIG 续跑 500 步
    python train_modern.py --steps 1000          # 续跑 1000 步
    python train_modern.py steps=2000 lr=0.001   # key=value 写法
    python train_modern.py --fresh --d-model 32  # 忽略检查点,从头训练
    python train_modern.py --generate-only       # 只生成文本
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
_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent if _HERE.name in ("tests", "tools", "legacy") else _HERE

BASE = ROOT
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

LOG_PATH = BASE / "boot_log.txt"
LAST_CKPT = BASE / "modern_ckpt_last.npz"
BEST_CKPT = BASE / "modern_ckpt_best.npz"
LEGACY_CKPT = BASE / "modern_ckpt.npz"  # 旧版检查点,仅用于兼容读取

CONFIG = {
    "steps": 50000,            # 本次运行增训的步数
    "d_model": 4096,
    "blocks": 2,
    "q_heads": 32,
    "kv_heads": 32,
    "batch_size": 16,
    "seq_len": 1024,
    "lr": 5e-4,              # 全新训练的峰值学习率;续跑时取 min(lr, 上次学习率)
    "min_lr_ratio": 0.1,     # 余弦衰减的下限 = lr * min_lr_ratio
    "warmup_steps": 100,      # 仅全新训练使用
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "eval_every": 200,
    "log_every": 20,
    "patience": 0,           # 连续多少次评估无改善就早停;0 = 关闭
    "seed": 42,
    "resume": True,          # 默认续跑:有检查点就接着训
    "text_file": "Xiyouji.txt",
    "prompt": "菩提祖师",
    "new_tokens": 128,
}

_t0 = time.time()


# ----------------------------------------------------------------------------
# 日志
# ----------------------------------------------------------------------------
def log(msg: str) -> None:
    line = f"[modern {time.time() - _t0:9.3f}s] {msg}"
    print(line, flush=True)
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
    except Exception:
        pass


# ----------------------------------------------------------------------------
# 命令行解析
# ----------------------------------------------------------------------------
def _convert(key: str, value: str):
    default = CONFIG[key]
    if isinstance(default, bool):
        return value.lower() in ("1", "true", "yes", "on")
    return type(default)(value)


def parse_args(argv: list[str]) -> tuple[dict, dict]:
    cfg = dict(CONFIG)
    flags = {"generate_only": False}
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
        cfg[key] = _convert(key, value)
    if positional:
        cfg["steps"] = positional[0]          # 兼容旧版:第一个纯数字 = 增训步数
    if len(positional) >= 2:
        cfg["d_model"] = positional[1]        # 兼容旧版:第二个纯数字 = d_model
    return cfg, flags


# ----------------------------------------------------------------------------
# 检查点:原子写入、JSON 元信息、架构校验、优化器状态
# ----------------------------------------------------------------------------
def _atomic_savez(path: Path, **arrays) -> None:
    tmp = path.with_name(path.stem + ".tmp.npz")  # np.savez 要求后缀为 .npz
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def arch_of(cfg: dict, vocab_size: int) -> dict:
    return {
        "d_model": int(cfg["d_model"]),
        "blocks": int(cfg["blocks"]),
        "q_heads": int(cfg["q_heads"]),
        "kv_heads": int(cfg["kv_heads"]),
        "vocab_size": int(vocab_size),
    }


def _opt_export(opt) -> tuple[dict, dict]:
    """自动枚举优化器的状态字段(不依赖具体字段名)。

    数组字典 -> 逐项存为 npz;单个数组 -> 直接存;标量 -> 写进 JSON。
    """
    arrays: dict = {}
    scalars: dict = {"_dict_fields": [], "_array_fields": []}
    for name, val in getattr(opt, "__dict__", {}).items():
        if name in ("params", "lr", "weight_decay"):
            continue  # 参数引用与配置项不属于状态
        if isinstance(val, dict) and val and all(isinstance(v, np.ndarray) for v in val.values()):
            for k, v in val.items():
                arrays[f"opt::{name}::{k}"] = v
            scalars["_dict_fields"].append(name)
        elif isinstance(val, np.ndarray):
            arrays[f"opt::{name}"] = val
            scalars["_array_fields"].append(name)
        elif isinstance(val, (int, float, bool, str)):
            scalars[name] = val
    return arrays, scalars


def _opt_import(opt, z, scalars: dict) -> list[str]:
    restored = []
    for name in scalars.get("_dict_fields", []):
        prefix = f"opt::{name}::"
        setattr(opt, name, {k[len(prefix):]: z[k] for k in z.files if k.startswith(prefix)})
        restored.append(name)
    for name in scalars.get("_array_fields", []):
        setattr(opt, name, z[f"opt::{name}"])
        restored.append(name)
    for name, val in scalars.items():
        if not name.startswith("_"):
            setattr(opt, name, val)
            restored.append(name)
    return restored


def save_checkpoint(path: Path, model, tok, cfg: dict, *, params: dict, step: int,
                    best_val: float, best_step: int, last_lr: float | None,
                    rng=None, opt=None) -> None:
    arrays = {f"param::{k}": v for k, v in params.items()}
    arrays["chars"] = np.asarray(tok.chars)
    opt_scalars: dict = {}
    if opt is not None:
        opt_arrays, opt_scalars = _opt_export(opt)
        arrays.update(opt_arrays)
    meta = {
        "format": 2,
        "arch": arch_of(cfg, model.vocab_size),
        "step": int(step),
        "best_val": float(best_val),
        "best_step": int(best_step),
        "last_lr": None if last_lr is None else float(last_lr),
        "rng_state": rng.bit_generator.state if rng is not None else None,
        "opt": opt_scalars,
    }
    arrays["meta"] = np.asarray(json.dumps(meta, ensure_ascii=False))
    _atomic_savez(path, **arrays)


def load_checkpoint(path: Path, model, cfg: dict, vocab_size: int):
    """返回 (npz 句柄, 元信息, 参数字典)。架构或形状不符时抛出 RuntimeError。"""
    z = np.load(path, allow_pickle=False)
    if "meta" in z.files:
        meta = json.loads(str(z["meta"]))
    else:  # 旧版格式:cfg = [d, blocks, q, kv, vocab, 步数],val = [最佳 loss]
        arr = z["cfg"]
        meta = {
            "format": 1,
            "arch": {"d_model": int(arr[0]), "blocks": int(arr[1]), "q_heads": int(arr[2]),
                     "kv_heads": int(arr[3]), "vocab_size": int(arr[4])},
            "step": int(arr[5]),
            "best_val": float(z["val"][0]),
            "best_step": int(arr[5]),
            "last_lr": None,
            "rng_state": None,
            "opt": {},
        }
    expected = arch_of(cfg, vocab_size)
    if meta["arch"] != expected:
        raise RuntimeError(
            f"检查点架构 {meta['arch']} 与当前配置 {expected} 不一致。"
            f"请恢复原配置,或加 --fresh 从头训练。")
    params = {}
    for k, cur in model.params.items():
        key = f"param::{k}"
        if key not in z.files:
            raise RuntimeError(f"检查点缺少参数 {k}")
        if z[key].shape != cur.shape:
            raise RuntimeError(f"参数 {k} 形状不符: {z[key].shape} vs {cur.shape}")
        params[k] = z[key]
    return z, meta, params


# ----------------------------------------------------------------------------
# 学习率:预热 + 余弦衰减
# ----------------------------------------------------------------------------
def lr_schedule(rel: int, base: float, min_lr: float, warmup: int, total_rel: int) -> float:
    if warmup and rel <= warmup:
        return base * rel / warmup
    span = max(1, total_rel - warmup)
    p = min(1.0, (rel - warmup) / span)
    return min_lr + 0.5 * (base - min_lr) * (1.0 + math.cos(math.pi * p))


# ----------------------------------------------------------------------------
# 生成
# ----------------------------------------------------------------------------
def generate_samples(model, tok, cfg: dict, params: dict) -> None:
    model.params = {k: v.copy() for k, v in params.items()}
    prompt_ids = tok.encode(cfg["prompt"], add_bos=True)
    gen = model.generate(prompt_ids, max_new=cfg["new_tokens"],
                         temperature=0.8, top_p=0.95, seed=1)
    log(f"生成示例(prompt={cfg['prompt']!r},温度0.8 / top-p0.95):")
    print(tok.decode(gen), flush=True)
    gen_greedy = model.generate(prompt_ids, max_new=40, temperature=0)
    log("生成示例(greedy):")
    print(tok.decode(gen_greedy), flush=True)


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------
def main(argv: list[str]) -> None:
    cfg, flags = parse_args(argv)

    from dual_transformer_numpy import AdamWNP, CharTokenizer, clip_grad_norm, evaluate_loss, make_batch
    from modern_transformer_numpy import ModernTransformerNumpy

    log("=== 现代架构训练启动(GQA+SwiGLU+RMSNorm+RoPE,纯NumPy) ===")
    text = (BASE / cfg["text_file"]).read_text(encoding="utf-8").replace("\r\n", "\n")
    tok = CharTokenizer.from_text(text)
    ids = np.asarray(tok.encode(text, add_bos=True, add_eos=True), dtype=np.int64)
    split = int(len(ids) * 0.95)
    train_ids, val_ids = ids[:split], ids[split:]
    log(f"词表={tok.vocab_size} 训练token={len(train_ids)} 验证token={len(val_ids)}")

    model = ModernTransformerNumpy(d_model=cfg["d_model"], n_blocks=cfg["blocks"],
                                   n_q_heads=cfg["q_heads"], n_kv_heads=cfg["kv_heads"],
                                   vocab_size=tok.vocab_size, seed=cfg["seed"])

    # ---- 1. 读取检查点 ----
    start_step = 0
    best_val = None
    best_step = 0
    last_lr = None
    resumed = False
    z = None
    opt_meta: dict = {}
    rng_state = None
    best_params = None

    if cfg["resume"]:
        src = LAST_CKPT if LAST_CKPT.exists() else (LEGACY_CKPT if LEGACY_CKPT.exists() else None)
        if src is not None:
            z, meta, params = load_checkpoint(src, model, cfg, tok.vocab_size)
            for k in model.params:
                model.params[k] = params[k]
            start_step = meta["step"]
            best_val = meta["best_val"]
            best_step = meta["best_step"]
            last_lr = meta.get("last_lr")
            opt_meta = meta.get("opt") or {}
            rng_state = meta.get("rng_state")
            resumed = True
            log(f"从 {src.name} 续跑:第 {start_step} 步,最佳验证 loss = {best_val:.4f}(第 {best_step} 步)")
            if src == LEGACY_CKPT:
                log("提示:读取的是旧版检查点,无优化器状态,动量将从零开始;下次保存起将切换为新格式")
            if src == LEGACY_CKPT:
                best_params = {k: v.copy() for k, v in model.params.items()}

        if BEST_CKPT.exists():
            _, bmeta, bparams = load_checkpoint(BEST_CKPT, model, cfg, tok.vocab_size)
            best_params = bparams
            best_val = bmeta["best_val"]
            best_step = bmeta["best_step"]

    # ---- 2. 仅生成 ----
    if flags["generate_only"]:
        if best_params is None:
            if resumed:
                best_params = {k: v.copy() for k, v in model.params.items()}
            else:
                log("没有找到检查点,无法生成")
                return
        generate_samples(model, tok, cfg, best_params)
        log("=== 全部完成 ===")
        return

    # ---- 3. 构建优化器与随机数 ----
    rng = np.random.default_rng(cfg["seed"] + start_step)
    if rng_state is not None:
        rng.bit_generator.state = rng_state

    opt = AdamWNP(model.params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    if z is not None and opt_meta is not None and any(f.startswith("opt::") for f in z.files):
        fields = _opt_import(opt, z, opt_meta)
        log(f"已恢复优化器状态:{', '.join(fields)}")
    elif resumed:
        log("检查点不含优化器状态,动量从零开始")

    # 续跑:接着上次学习率衰减,不做预热;全新训练:完整预热 + 余弦
    if resumed:
        base_lr = min(cfg["lr"], last_lr) if last_lr else cfg["lr"]
        warmup = 0
    else:
        base_lr = cfg["lr"]
        warmup = cfg["warmup_steps"]
    min_lr = base_lr * cfg["min_lr_ratio"]

    # ---- 4. 基准评估 ----
    val_now = evaluate_loss(model, val_ids, cfg["seq_len"], max_batches=20)
    if best_val is None:
        best_val = float(val_now)
        best_step = start_step
        best_params = {k: v.copy() for k, v in model.params.items()}
    if best_params is None:
        best_params = {k: v.copy() for k, v in model.params.items()}
    log(f"当前验证 loss = {val_now:.4f}(检查点记录的最佳 {best_val:.4f} @ 第 {best_step} 步)")

    n_params = model.num_params()
    log(f"参数量={n_params:,} 配置: d={cfg['d_model']} blocks={cfg['blocks']} "
        f"q/kv头={cfg['q_heads']}/{cfg['kv_heads']} batch={cfg['batch_size']} seq={cfg['seq_len']} "
        f"lr={base_lr:g} warmup={warmup} 本次增训={cfg['steps']} 步 "
        f"模式={'续跑' if resumed else '全新'}")

    total = start_step + cfg["steps"]
    run_start = start_step
    lr_now = base_lr
    cur_step = start_step
    no_improve = 0
    nan_count = 0
    t_run = time.time()

    def save_last() -> None:
        save_checkpoint(LAST_CKPT, model, tok, cfg, params=model.params, step=cur_step,
                        best_val=best_val, best_step=best_step, last_lr=lr_now,
                        rng=rng, opt=opt)

    def save_best() -> None:
        save_checkpoint(BEST_CKPT, model, tok, cfg, params=best_params, step=best_step,
                        best_val=best_val, best_step=best_step, last_lr=None)

    # ---- 5. 训练循环 ----
    try:
        for step in range(start_step + 1, total + 1):
            rel = step - run_start
            lr_now = lr_schedule(rel, base_lr, min_lr, warmup, cfg["steps"])
            opt.lr = lr_now

            x, y = make_batch(train_ids, cfg["batch_size"], cfg["seq_len"], rng)
            loss, grads = model.loss_and_grads(x, y)
            if not np.isfinite(loss):
                nan_count += 1
                log(f"step {step} loss 非有限({loss}),跳过该步(累计 {nan_count} 次)")
                if nan_count >= 10:
                    raise RuntimeError("非有限 loss 累计达到 10 次,已停止训练")
                continue

            gn = clip_grad_norm(grads, cfg["max_grad_norm"])
            opt.step(model.params, grads)
            cur_step = step

            if step % cfg["log_every"] == 0 or step == start_step + 1:
                log(f"step {step}/{total} train loss={loss:.4f} |grad|={gn:.2f} "
                    f"lr={lr_now:.5f} ({time.time() - t_run:.1f}s)")

            if step % cfg["eval_every"] == 0 or step == total:
                val = evaluate_loss(model, val_ids, cfg["seq_len"], max_batches=20)
                log(f"step {step} 验证 loss = {val:.4f}(当前最佳 {best_val:.4f} @ 第 {best_step} 步)")
                if val < best_val - 1e-4:
                    best_val = float(val)
                    best_step = step
                    best_params = {k: v.copy() for k, v in model.params.items()}
                    save_best()
                    no_improve = 0
                    log(f"  ↑ 新最佳,已保存 {BEST_CKPT.name}")
                else:
                    no_improve += 1
                save_last()
                if cfg["patience"] and no_improve >= cfg["patience"]:
                    log(f"早停:连续 {no_improve} 次评估无改善")
                    break
    except KeyboardInterrupt:
        log("⏹ 手动停止,正在保存当前进度…")
        save_last()
        raise

    # ---- 6. 收尾 ----
    save_last()
    total_t = time.time() - t_run
    log(f"本次训练结束:训练至第 {cur_step} 步(目标 {total}),用时 {total_t:.1f}s,"
        f"最佳验证 loss = {best_val:.4f}(第 {best_step} 步)")
    generate_samples(model, tok, cfg, best_params)
    log("=== 全部完成 ===")


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except KeyboardInterrupt:
        log("⏹ 被手动停止:进度已保存到 modern_ckpt_last.npz,直接运行即可续跑")
    except SystemExit:
        raise
    except Exception:
        log("❌ 失败:\n" + traceback.format_exc())
        raise

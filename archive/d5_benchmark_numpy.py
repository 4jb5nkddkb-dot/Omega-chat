"""
D5 benchmark (NumPy backend): single-branch vs dual-average vs dual-concat, plus a
parameter-matched single-branch control, on the same corpus and the same training budget.

Fairness rules
- Same corpus bytes (copied to run_dir/corpus.bin and verified by SHA-256),
  same train/validation split, same seeds.
- Same optimizer steps (warm-up steps count toward the budget), batch size,
  sequence length, learning rate, and gradient clipping.
  Equal steps is NOT equal FLOPs: dual models run two branches per layer, so they
  do roughly twice the block compute of the single-branch model.
- Parameter counts are recorded for every configuration. "single_param_matched"
  widens the single-branch model until its parameter count is as close as possible
  to dual_average.

Metrics
- validation loss (nats and bits per byte) on the full held-out split
- training throughput (tokens/s, timed after warm-up, evaluation excluded)
- peak Python+numpy allocation during training (tracemalloc; the numpy backend's
  memory measure, since process RSS is dominated by interpreter overhead at small sizes)
- process peak RSS growth (reference only)
- generation latency (ms/token) for cached decoding, within the window and rolling
- cache correctness: cached vs full-context greedy output over a window-rolling run
- long-context stability: validation loss at several evaluation lengths, as a ratio
  to the loss at the training length

Each configuration runs in its own subprocess so peak-memory numbers are not
contaminated by earlier runs.

Usage
  python d5_benchmark_numpy.py --corpus corpus.txt --steps 1000 --seeds 3 --out-dir d5_numpy_results
  python d5_benchmark_numpy.py --report-only --out-dir d5_numpy_results
Without --corpus, a fixed set of Python standard-library sources is used; pass --corpus
for reproducibility across machines.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np

# Make the sibling module importable no matter where this script is launched from.
_HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
try:
    from dual_transformer_numpy import (
        DualTransformerNP, AdamWNP, clip_grad_norm, evaluate_loss, make_batch, VOCAB_SIZE,
    )
except ModuleNotFoundError as e:
    raise SystemExit(
        f"cannot import dual_transformer_numpy ({e}). Put dual_transformer_numpy.py in the same "
        f"folder as d5_benchmark_numpy.py, and run the benchmark as a file, not via 'python -c' or stdin.")

LN2 = math.log(2.0)
DEFAULT_CONFIGS = ("single", "dual_average", "dual_concat", "single_param_matched")


def peak_rss_mb() -> float:
    # ru_maxrss is KiB on Linux and bytes on macOS.
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1024.0 if sys.platform != "darwin" else r / (1024.0 * 1024.0)


def count_params(model) -> int:
    return model.num_params()


def _stdlib_dir() -> Path | None:
    """Locate the standard-library directory, or None if the runtime does not expose it."""
    module_file = getattr(os, "__file__", None)
    if module_file:
        return Path(module_file).parent
    import sysconfig
    candidate = Path(sysconfig.get_paths()["stdlib"])
    return candidate if candidate.is_dir() else None


def load_corpus(path, max_bytes: int):
    if path:
        data = Path(path).read_bytes()
        source = str(path)
    else:
        std = _stdlib_dir()
        if std is None:
            raise SystemExit("this Python runtime does not expose its standard-library directory; "
                             "pass --corpus path/to/text_file")
        chunks, total = [], 0
        for f in sorted(std.glob("*.py")):
            b = f.read_bytes()
            chunks.append(b)
            total += len(b)
            if total >= max_bytes:
                break
        data = b"".join(chunks)
        source = f"python-stdlib:{std}"
    data = data[:max_bytes]
    return data, {"source": source, "bytes": len(data),
                  "sha256": hashlib.sha256(data).hexdigest()}


def match_single_width(target_params: int, heads: int, d_max: int, blocks: int,
                       use_rms: bool, max_seq: int):
    """Return (d_model, params) for a single-branch model closest to target_params."""
    best = None
    step = 2 * heads  # keeps head_dim even
    for d in range(step, d_max + 1, step):
        m = DualTransformerNP(d_model=d, n_blocks=blocks, n_heads=heads,
                              merge_method="single", use_rms=use_rms, max_seq=max_seq)
        p = count_params(m)
        if best is None or abs(p - target_params) < abs(best[1] - target_params):
            best = (d, p)
    return best


def build_run_config(args, data: bytes, corpus_meta: dict):
    use_rms = args.norm == "rms"
    eval_lengths = sorted(set(int(x) for x in args.eval_lengths.split(",") if x) | {args.seq_len})
    max_seq = max(eval_lengths)
    split = int(len(data) * 0.9)
    val_len = len(data) - split
    if val_len < max_seq + 1:
        raise SystemExit(f"validation split has {val_len} bytes; need >= {max_seq + 1}. Use a larger corpus.")

    dual_ref = DualTransformerNP(d_model=args.d_model, n_blocks=args.blocks, n_heads=args.heads,
                                 merge_method="average", use_rms=use_rms, max_seq=max_seq)
    target = count_params(dual_ref)

    configs = {}
    for name in [n for n in args.configs.split(",") if n]:
        if name == "single":
            configs[name] = {"merge": "single", "d_model": args.d_model}
        elif name == "dual_average":
            configs[name] = {"merge": "average", "d_model": args.d_model}
        elif name == "dual_concat":
            configs[name] = {"merge": "concat", "d_model": args.d_model}
        elif name == "single_param_matched":
            d, _ = match_single_width(target, args.heads, 4 * args.d_model, args.blocks, use_rms, max_seq)
            configs[name] = {"merge": "single", "d_model": d}
        else:
            raise SystemExit(f"unknown config: {name}")

    for arch in configs.values():
        m = DualTransformerNP(d_model=arch["d_model"], n_blocks=args.blocks, n_heads=args.heads,
                              merge_method=arch["merge"], use_rms=use_rms, max_seq=max_seq)
        arch["n_params"] = count_params(m)
        arch["params_vs_dual_average"] = arch["n_params"] / target

    return {
        "backend": "numpy",
        "corpus": corpus_meta,
        "split": split,
        "steps": args.steps,
        "warmup_steps": args.warmup_steps,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "lr": args.lr,
        "blocks": args.blocks,
        "heads": args.heads,
        "use_rms": use_rms,
        "max_seq": max_seq,
        "eval_lengths": eval_lengths,
        "eval_max_batches": args.eval_max_batches,
        "gen_repeats": args.gen_repeats,
        "seeds": args.seeds,
        "configs": configs,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }


def run_worker(run_dir: Path, name: str, seed: int) -> None:
    rss_start = peak_rss_mb()
    cfg = json.loads((run_dir / "run_config.json").read_text())
    data = (run_dir / "corpus.bin").read_bytes()
    if hashlib.sha256(data).hexdigest() != cfg["corpus"]["sha256"]:
        raise SystemExit("corpus.bin does not match run_config.json")
    arch = cfg["configs"][name]
    split = cfg["split"]
    train = np.frombuffer(data[:split], dtype=np.uint8).astype(np.int64)
    val = list(data[split:])
    batch, seq = cfg["batch_size"], cfg["seq_len"]

    model = DualTransformerNP(d_model=arch["d_model"], n_blocks=cfg["blocks"], n_heads=cfg["heads"],
                              merge_method=arch["merge"], use_rms=cfg["use_rms"],
                              max_seq=cfg["max_seq"], seed=seed)
    n_params = count_params(model)
    optimizer = AdamWNP(model.params, lr=cfg["lr"])
    rng = np.random.default_rng(seed)
    tracemalloc.start()

    def one_step():
        x, y = make_batch(train, batch, seq, rng)
        loss, grads = model.loss_and_grads(x, y)
        clip_grad_norm(grads, 1.0)
        optimizer.step(model.params, grads)
        return loss

    traced_base, _ = tracemalloc.get_traced_memory()
    tracemalloc.reset_peak()
    for _ in range(cfg["warmup_steps"]):
        one_step()
    rss_before_train = peak_rss_mb()

    losses, diverged = [], False
    t0 = time.perf_counter()
    for _ in range(cfg["steps"]):
        lv = float(one_step())
        if not math.isfinite(lv):
            diverged = True
            break
        losses.append(lv)
    train_time = time.perf_counter() - t0
    _, traced_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_alloc_mb = (traced_peak - traced_base) / 2**20
    trained_tokens = len(losses) * batch * seq
    tokens_per_sec = trained_tokens / train_time if train_time > 0 else float("nan")
    rss_after_train = peak_rss_mb()

    val_loss = evaluate_loss(model, val, seq_len=seq, batch_size=batch,
                             max_batches=cfg["eval_max_batches"])
    train_eval_loss = spaced_train_loss(model, data[:split], seq, cfg["eval_max_batches"])

    long_ctx = {}
    for L in cfg["eval_lengths"]:
        loss_L = evaluate_loss(model, val, seq_len=L, batch_size=batch,
                               max_batches=cfg["eval_max_batches"])
        long_ctx[str(L)] = {"val_loss": loss_L, "finite": math.isfinite(loss_L)}
    base = long_ctx[str(seq)]["val_loss"]
    for v in long_ctx.values():
        v["ratio_to_train_len"] = v["val_loss"] / base if base else float("nan")

    max_seq = cfg["max_seq"]
    prompt_len = max_seq // 2
    prompt = val[:prompt_len]

    def time_gen(n_tokens: int) -> float:
        model.generate_cached(prompt, 4)  # warm-up
        samples = []
        for _ in range(cfg["gen_repeats"]):
            t = time.perf_counter()
            model.generate_cached(prompt, n_tokens)
            samples.append((time.perf_counter() - t) * 1000.0 / n_tokens)
        return statistics.median(samples)

    gen_within_ms = time_gen(max_seq - prompt_len - 1)  # never rolls the window
    gen_roll_ms = time_gen(max_seq)                     # forces at least one window roll

    full = model.generate(prompt, max_seq)
    cached = model.generate_cached(prompt, max_seq)

    result = {
        "name": name,
        "seed": seed,
        "merge": arch["merge"],
        "d_model": arch["d_model"],
        "n_params": n_params,
        "params_vs_dual_average": arch["params_vs_dual_average"],
        "train_steps": len(losses),
        "warmup_steps": cfg["warmup_steps"],
        "diverged": diverged,
        "final_train_loss": losses[-1] if losses else None,
        "val_loss": val_loss,
        "val_bits_per_byte": val_loss / LN2,
        "train_eval_loss": train_eval_loss,
        "val_minus_train": val_loss - train_eval_loss,
        "tokens_per_sec": tokens_per_sec,
        "train_time_s": train_time,
        "peak_rss_mb": peak_rss_mb(),
        "peak_alloc_train_mb": peak_alloc_mb,
        "rss_growth_during_train_mb": rss_after_train - rss_before_train,
        "rss_growth_total_mb": peak_rss_mb() - rss_start,
        "gen_ms_per_token_within_window": gen_within_ms,
        "gen_ms_per_token_rolling": gen_roll_ms,
        "cache_matches_full_rolling": full == cached,
        "long_context": long_ctx,
    }
    out = run_dir / "results" / f"{name}_seed{seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"[{name} seed={seed}] val_loss={val_loss:.4f} tok/s={tokens_per_sec:.0f} "
          f"params={n_params} cache_ok={result['cache_matches_full_rolling']}")


def spaced_train_loss(model, train_bytes: bytes, seq_len: int, n_windows: int) -> float:
    """Mean loss over n_windows evenly spaced across the whole training split.

    Taking only the first windows would compare the file's opening text with the
    validation text, which can differ in difficulty. Spacing the windows across the
    full split keeps the train/validation gap about memorization, not position.
    """
    data = np.frombuffer(train_bytes, dtype=np.uint8).astype(np.int64)
    starts = np.linspace(0, len(data) - seq_len - 2, n_windows).astype(int)
    losses = [model.loss(data[s:s + seq_len][None, :], data[s + 1:s + seq_len + 1][None, :])
              for s in starts]
    return float(sum(losses) / len(losses))


def agg(values):
    vals = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return None
    return {"mean": statistics.fmean(vals),
            "std": statistics.stdev(vals) if len(vals) > 1 else None,
            "n": len(vals)}


def fmt(a, digits=4, suffix=""):
    if a is None:
        return "—"
    if a["std"] is None:
        return f"{a['mean']:.{digits}f}{suffix}（1 个种子，无标准差）"
    return f"{a['mean']:.{digits}f} ± {a['std']:.{digits}f}{suffix}"


def build_report(run_dir: Path) -> None:
    cfg = json.loads((run_dir / "run_config.json").read_text())
    rows = {}
    for f in sorted((run_dir / "results").glob("*.json")):
        r = json.loads(f.read_text())
        rows.setdefault(r["name"], []).append(r)

    summary = {}
    for name in cfg["configs"]:
        rs = rows.get(name, [])
        if not rs:
            continue
        arch = cfg["configs"][name]
        lc = {}
        for L in cfg["eval_lengths"]:
            k = str(L)
            lc[k] = {
                "val_loss": agg([r["long_context"][k]["val_loss"] for r in rs]),
                "ratio": agg([r["long_context"][k]["ratio_to_train_len"] for r in rs]),
                "all_finite": all(r["long_context"][k]["finite"] for r in rs),
            }
        summary[name] = {
            "merge": arch["merge"],
            "d_model": arch["d_model"],
            "n_params": arch["n_params"],
            "params_vs_dual_average": arch["params_vs_dual_average"],
            "n_seeds": len(rs),
            "diverged_any": any(r["diverged"] for r in rs),
            "val_loss": agg([r["val_loss"] for r in rs]),
            "val_bits_per_byte": agg([r["val_bits_per_byte"] for r in rs]),
            "train_eval_loss": agg([r["train_eval_loss"] for r in rs]),
            "val_minus_train": agg([r["val_minus_train"] for r in rs]),
            "tokens_per_sec": agg([r["tokens_per_sec"] for r in rs]),
            "peak_alloc_train_mb": agg([r["peak_alloc_train_mb"] for r in rs]),
            "rss_growth_total_mb": agg([r["rss_growth_total_mb"] for r in rs]),
            "gen_ms_per_token_within_window": agg([r["gen_ms_per_token_within_window"] for r in rs]),
            "gen_ms_per_token_rolling": agg([r["gen_ms_per_token_rolling"] for r in rs]),
            "cache_matches_full_rolling_all": all(r["cache_matches_full_rolling"] for r in rs),
            "long_context": lc,
        }
    (run_dir / "results.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    c, env = cfg["corpus"], cfg["environment"]
    lines = ["# D5 性能与泛化评估报告（NumPy 后端）\n", "## 1. 实验设置\n",
             f"- 语料：`{c['source']}`，{c['bytes']} 字节，SHA-256 `{c['sha256']}`",
             f"- 划分：前 90% 训练 / 后 10% 验证（验证集 {c['bytes'] - cfg['split']} 字节）",
             f"- 训练预算：每个配置 {cfg['steps']} 步 + {cfg['warmup_steps']} 步热身，"
             f"batch={cfg['batch_size']}，seq_len={cfg['seq_len']}，lr={cfg['lr']}，梯度裁剪 1.0",
             f"- 结构：blocks={cfg['blocks']}，heads={cfg['heads']}，"
             f"norm={'RMSNorm' if cfg['use_rms'] else 'LayerNorm'}，max_seq={cfg['max_seq']}",
             f"- 种子数：{cfg['seeds']}",
             f"- 环境：Python {env['python']}，numpy {env['numpy']}，{env['platform']}\n",
             "## 2. 架构与参数量\n",
             "| 配置 | merge | d_model | 参数量 | 相对 dual_average |",
             "|---|---|---|---|---|"]
    for name, s in summary.items():
        lines.append(f"| {name} | {s['merge']} | {s['d_model']} | {s['n_params']:,} | "
                     f"{s['params_vs_dual_average']:.3f}× |")
    lines += ["",
              "参数量包含嵌入层、所有分支、输出头。dual_concat 的输出头输入维度是 2·d_model，"
              "因此参数量高于 dual_average。single_param_matched 是参数量最接近 dual_average 的单分支对照。\n",
              "## 3. 主要指标\n",
              "| 配置 | 验证 loss (nats) | 训练集 loss (nats) | 验证−训练差距 | 验证 bits/byte | 训练 tok/s | 训练峰值分配 (MB) | 进程 RSS 增长 (MB，参考) | "
              "生成 ms/token（窗口内） | 生成 ms/token（滚动） | 缓存与全上下文一致 |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, s in summary.items():
        lines.append(
            f"| {name} | {fmt(s['val_loss'])} | {fmt(s['train_eval_loss'])} | {fmt(s['val_minus_train'])} | "
            f"{fmt(s['val_bits_per_byte'])} | "
            f"{fmt(s['tokens_per_sec'], 0)} | {fmt(s['peak_alloc_train_mb'], 2)} | {fmt(s['rss_growth_total_mb'], 1)} | "
            f"{fmt(s['gen_ms_per_token_within_window'], 3)} | "
            f"{fmt(s['gen_ms_per_token_rolling'], 3)} | "
            f"{'是' if s['cache_matches_full_rolling_all'] else '否'} |")
    lines.append("")
    if any(s["diverged_any"] for s in summary.values()):
        lines.append("> 注意：至少一个配置在训练中出现非有限 loss，相关数值不可直接比较。\n")

    lines += ["## 4. 长上下文稳定性\n",
              "评估长度超过训练长度时，损失相对训练长度的比值。比值接近 1 且全部有限，说明未明显崩溃。\n"]
    lens = cfg["eval_lengths"]
    lines.append("| 配置 | " + " | ".join(f"L={L} 损失 / 比值" for L in lens) + " |")
    lines.append("|---|" + "---|" * len(lens))
    for name, s in summary.items():
        cells = []
        for L in lens:
            e = s["long_context"][str(L)]
            cells.append(f"{fmt(e['val_loss'])} / {fmt(e['ratio'], 3)}" + ("" if e["all_finite"] else " ⚠非有限"))
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines.append("")

    lines += ["## 5. 解释与局限\n",
              "- **等步数不等计算量。** dual 模型每层运行两个分支，块计算量约为单分支的两倍；"
              "比较时需同时看参数量和计算量。single_param_matched 参数量接近，但宽度和深度不同。",
              "- **种子数量。** 样本少时，标准差只是粗略估计。若两个配置的差距小于标准差，不应下结论。",
              "- **吞吐与内存是本机测量。** NumPy 后端只在 CPU 上运行，计时受后台负载影响。"
              "训练峰值分配由 tracemalloc 统计（包含 numpy 数组），它会略微拖慢计时，但对所有配置一致。"
              "进程 RSS 包含解释器开销，小模型上几乎无法分辨差异，仅作参考。",
              "- **验证−训练差距。** 训练集 loss 在整个训练段上均匀取样的同样长度窗口求平均。差距明显大于 0 且随训练增大，"
              "说明模型在记忆训练数据；差距小则泛化更好。它只是参考信号，不能单独定论。",
              "- **泛化范围有限。** 这里的“泛化”仅指同分布的留出字节和更长上下文，"
              "不代表通用语言能力，也不能外推到 GPT 级模型。",
              "- **长上下文比较的窗口不同。** 不同长度使用不同的验证窗口起点（前 "
              f"{cfg['eval_max_batches']} 个窗口），比值反映的是同一验证段上的相对变化，但并非逐位置对齐。",
              "- **生成延迟包含 Python 循环开销。** 滚动窗口阶段每个 token 都会重新预填充整个窗口，"
              "因此滚动延迟高于窗口内延迟，这是正确性优先实现的预期结果。",
              "- **与 torch 后端不可直接比较数值。** 两个后端的随机初始化流、GELU 近似和 "
              "数值精度不同，只能在各自后端内部比较配置。"]
    (run_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"report written: {run_dir / 'report.md'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=str, default=None)
    ap.add_argument("--max-corpus-bytes", type=int, default=1_000_000)
    ap.add_argument("--out-dir", type=str, default="d5_numpy_results")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--warmup-steps", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--norm", choices=("rms", "layer"), default="rms")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--configs", type=str, default=",".join(DEFAULT_CONFIGS))
    ap.add_argument("--eval-lengths", type=str, default="64,128,256")
    ap.add_argument("--eval-max-batches", type=int, default=64)
    ap.add_argument("--gen-repeats", type=int, default=3)
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--no-subprocess", action="store_true",
                    help="run every configuration in this process (for runtimes that cannot "
                         "start child processes, e.g. some mobile Python apps); per-run memory "
                         "numbers are then less isolated")
    # internal: worker mode
    ap.add_argument("--worker", type=str, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run_dir = Path(args.out_dir)

    if args.worker:
        run_worker(run_dir, args.worker, args.seed)
        return
    if args.report_only:
        build_report(run_dir)
        return

    run_dir.mkdir(parents=True, exist_ok=True)
    data, corpus_meta = load_corpus(args.corpus, args.max_corpus_bytes)
    cfg = build_run_config(args, data, corpus_meta)
    (run_dir / "corpus.bin").write_bytes(data)
    (run_dir / "run_config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
    print(f"corpus: {corpus_meta['bytes']} bytes, sha256={corpus_meta['sha256'][:16]}...")
    for name, arch in cfg["configs"].items():
        print(f"config {name}: merge={arch['merge']} d_model={arch['d_model']} params={arch['n_params']:,}")

    logs = run_dir / "logs"
    logs.mkdir(exist_ok=True)
    for seed in range(args.seeds):
        for name in cfg["configs"]:
            if args.no_subprocess:
                run_worker(run_dir, name, seed)
                continue
            log_path = logs / f"{name}_seed{seed}.log"
            with open(log_path, "w", encoding="utf-8") as log:
                if "__file__" not in globals():
                    raise SystemExit("worker processes need this script as a file; save it as "
                                     "d5_benchmark_numpy.py and run `python d5_benchmark_numpy.py ...`")
                subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                "--out-dir", str(run_dir), "--worker", name, "--seed", str(seed)],
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            print(log_path.read_text(encoding="utf-8").strip().splitlines()[-1])
    build_report(run_dir)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
train.py —— 统一训练入口：DualTransformerNP（模型） + Omega（闭环门控）

流程（每个 chunk）：
    Omega 观察验证 loss → 策略提出 train(k 步) → 预测 → 授权 → 执行训练 →
    归因 → 决策。若 Omega 升级(ESCALATE)或验证 loss 变差，则回滚该 chunk 的参数。

用法：
    python train.py train --text Xiyouji.txt --steps 2000 --ckpt ckpt/model.npz
    python train.py train --no-omega ...      # 不用 Omega 门控，直接调用 train_steps
    python train.py train --resume ...        # 从 *_last.npz 继续
    python train.py generate --ckpt ckpt/model.npz --prompt "悟空"
    python train.py test                      # 端到端冒烟测试
依赖：Python 3.10+, numpy
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

from dual_transformer_numpy import (
    AdamWNP, ByteTokenizer, CharTokenizer, DualTransformerNP, clip_grad_norm,
    evaluate_loss, last_checkpoint_path, lr_at, make_batch, read_checkpoint,
    tokenizer_from_meta, train_steps,
)
from omega import Boundary, Config, Intention, Omega, Prediction

BASE = Path(__file__).resolve().parent
SECRET_PATH = BASE / "omega_secret.txt"


def load_secret() -> str:
    import secrets
    if SECRET_PATH.exists():
        s = SECRET_PATH.read_text().strip()
        if len(s) >= 16:
            return s
    s = secrets.token_hex(32)
    SECRET_PATH.write_text(s)
    return s


# ---------------------------------------------------------------- data
def prepare_data(text_path, tokenizer_kind="char", val_fraction=0.05, resume_meta=None):
    text = Path(text_path).read_text(encoding="utf-8").replace("\r\n", "\n")
    if resume_meta is not None:
        tok = tokenizer_from_meta(resume_meta)
    elif tokenizer_kind == "char":
        tok = CharTokenizer.from_text(text)
    else:
        tok = ByteTokenizer()
    ids = np.asarray(tok.encode(text, add_bos=True, add_eos=True), dtype=np.int64)
    split = int(len(ids) * (1 - val_fraction))
    return tok, ids[:split], ids[split:]


# ---------------------------------------------------- Omega 接入层
class TrainWorld:
    """Omega 的 World：observe() 给出验证 loss；execute('train', k) 真正训练 k 步。"""

    def __init__(self, model, train_ids, val_ids, args):
        self.model, self.train_ids, self.val_ids, self.args = model, train_ids, val_ids, args
        self.rng = np.random.default_rng(args.seed)
        self.opt = AdamWNP(model.params, lr=args.lr, weight_decay=args.weight_decay)
        self.step = 0
        self.total = args.steps
        self.snapshot = None

    def observe(self):
        return {"loss": evaluate_loss(self.model, self.val_ids, self.args.seq_len)}

    def execute(self, action, args):
        if action == "measure":
            return self.observe()
        if action != "train":
            raise ValueError(f"unsupported action: {action}")
        a = self.args
        self.snapshot = ({k: v.copy() for k, v in self.model.params.items()},
                         {k: v.copy() for k, v in self.opt.m.items()},
                         {k: v.copy() for k, v in self.opt.v.items()}, self.opt.t)
        for _ in range(int(args.get("steps", a.chunk))):
            self.step += 1
            self.opt.lr = lr_at(self.step, a.lr, a.schedule, a.warmup_steps, self.total)
            x, y = make_batch(self.train_ids, a.batch_size, a.seq_len, self.rng)
            _, grads = self.model.loss_and_grads(x, y, a.label_smoothing)
            clip_grad_norm(grads, a.max_grad_norm)
            self.opt.step(self.model.params, grads)
        return self.observe()

    def keep(self):
        self.snapshot = None

    def restore(self):
        if self.snapshot is None:
            return
        params, m, v, t = self.snapshot
        for k, val in params.items():
            self.model.params[k] = val
        self.opt.load_state(m, v, t)
        self.snapshot = None


class TrainPolicy:
    """Omega 的 Policy：提出 train/measure，并用最近 loss 变化趋势预测下一步。"""
    policy_id, policy_version = "train-policy", "3.0"
    prefer_policy_intent = True   # 跳过 Omega 内置 planner，使用本策略的 train 动作

    def __init__(self, chunk):
        self.mode, self.k, self.chunk = "train", chunk, chunk
        self.deltas = []

    def record(self, delta):
        self.deltas = (self.deltas + [float(delta)])[-8:]

    def intend(self, state, goal, **_):
        if self.mode == "train":
            return Intention("train", {"steps": self.k}, self.policy_id, self.policy_version,
                             "advance training by one chunk")
        return Intention("measure", {}, self.policy_id, self.policy_version, "observe only")

    def predict(self, state, intention, goal, **_):
        loss = float(state.get("loss", 0.0))
        if intention.action == "train":
            if len(self.deltas) >= 3:
                mean = sum(self.deltas) / len(self.deltas)
                std = math.sqrt(sum((d - mean) ** 2 for d in self.deltas) / len(self.deltas))
                conf = min(0.9, max(0.3, 1.0 - 0.5 * std / (abs(mean) + 1e-3)))
            else:
                mean, conf = -0.01, 0.5
            expected = {"loss": max(0.0, loss + mean)}
        else:
            expected, conf = {"loss": loss}, 0.95
        return Prediction(expected, conf, "trend-predictor", "3.0")


class OmegaTrainer:
    def __init__(self, world, policy, storage_dir, max_error=2.0):
        self.world, self.policy = world, policy
        self.omega = Omega(secret=load_secret(), storage_dir=str(storage_dir),
                           config=Config(max_error_apply=max_error),
                           boundary=Boundary(frozenset({"train", "measure"})),
                           auto_restore=False)
        self.owner = self.omega.owner()
        self.omega.set_goal(self.owner, {"loss": 0.0})
        self.omega.set_limits(self.owner, low={"loss": 0.0}, high={"loss": 50.0})

    def cycle(self, mode="train", k=None):
        """跑一个 Omega 循环，返回 dict(accepted, loss, before, decision, reason)。"""
        self.policy.mode, self.policy.k = mode, k or self.policy.chunk
        before = self.world.observe()["loss"]
        self.world.snapshot = None
        res = self.omega.run_autonomous(
            self.owner, self.world, self.policy, max_steps=1,
            max_error=1e9, max_risk=1.0, auto_actions={"train", "measure"})
        h = res["history"][-1] if res["history"] else {}
        executed = res["executed"] > 0
        escalated = res["stopped_reason"] == "escalate" or self.omega.escalated
        after = self.world.observe()["loss"]
        accepted = executed and not escalated and (mode != "train" or after <= before)
        if mode == "train":
            if executed:
                self.policy.record(after - before)
            self.world.keep() if accepted else self.world.restore()
            if not accepted:
                after = before
        return {"accepted": accepted, "loss": after, "before": before, "escalated": escalated,
                "decision": h.get("decision", "-"), "reason": res["stopped_reason"]}


# ---------------------------------------------------------------- 命令
def cmd_train(a):
    ckpt = Path(a.ckpt)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    resume_path = last_checkpoint_path(ckpt) if a.resume else None
    if resume_path is not None and not resume_path.exists():
        sys.exit(f"没有可恢复的检查点: {resume_path}")
    meta = read_checkpoint(resume_path)["meta"] if resume_path else None
    tok, train_ids, val_ids = prepare_data(a.text, a.tokenizer, a.val_fraction, meta)
    if min(len(train_ids), len(val_ids)) < a.seq_len + 2:
        sys.exit("语料太短，请减小 --seq-len 或 --val-fraction")
    print(f"tokenizer={type(tok).__name__} vocab={tok.vocab_size} train={len(train_ids)} val={len(val_ids)}")

    if a.no_omega:
        model = (DualTransformerNP.load_checkpoint(resume_path)[0] if resume_path else
                 DualTransformerNP(a.d_model, a.blocks, a.heads, a.merge, True, a.max_seq, seed=a.seed,
                                   vocab_size=tok.vocab_size, tie_embeddings=a.tie_embeddings))
        print(f"params={model.num_params():,}")
        train_steps(model, train_ids, a.steps, a.batch_size, a.seq_len, lr=a.lr, val_token_ids=val_ids,
                    checkpoint_path=str(ckpt), eval_every=a.eval_every, resume_from=resume_path,
                    seed=a.seed, weight_decay=a.weight_decay, max_grad_norm=a.max_grad_norm,
                    label_smoothing=a.label_smoothing, schedule=a.schedule,
                    warmup_steps=a.warmup_steps, log_csv=a.log_csv,
                    extra_meta={"tokenizer": tok.to_meta()})
        return model, tok

    if resume_path:
        model, _ = DualTransformerNP.load_checkpoint(resume_path)
    else:
        model = DualTransformerNP(a.d_model, a.blocks, a.heads, a.merge, True, a.max_seq, seed=a.seed,
                                  vocab_size=tok.vocab_size, tie_embeddings=a.tie_embeddings)
    print(f"params={model.num_params():,}")
    world = TrainWorld(model, train_ids, val_ids, a)
    trainer = OmegaTrainer(world, TrainPolicy(a.chunk), ckpt.parent / "omega_state")
    best, kept, streak = math.inf, 0, 0
    rows = []
    cycles = max(1, a.steps // a.chunk)
    for i in range(1, cycles + 1):
        r = trainer.cycle("train")
        kept += a.chunk if r["accepted"] else 0
        streak = 0 if r["accepted"] else streak + 1
        print(f"[{i}/{cycles}] loss={r['loss']:.4f} {'保留' if r['accepted'] else '回滚'} "
              f"decision={r['decision']} reason={r['reason']}")
        rows.append((i, world.step, r["loss"], int(r["accepted"])))
        if r["accepted"] and r["loss"] < best:
            best = r["loss"]
            model.save_checkpoint(ckpt, world.opt, world.step, extra={"tokenizer": tok.to_meta()})
        if r["accepted"] or i % 5 == 0 or i == cycles:
            model.save_checkpoint(last_checkpoint_path(ckpt), world.opt, world.step,
                                  extra={"tokenizer": tok.to_meta()})
        if r["escalated"]:
            print("Omega 已升级(ESCALATE)，训练暂停，需人工检查。")
            break
        if streak >= 5:
            world.opt.lr *= 0.5
            a.lr *= 0.5
            streak = 0
            print(f"连续回滚，学习率降为 {a.lr:.3g}")
    if not ckpt.exists():
        model.save_checkpoint(ckpt, world.opt, world.step, extra={"tokenizer": tok.to_meta()})
    if a.log_csv:
        with open(a.log_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["cycle", "step", "val_loss", "accepted"])
            w.writerows(rows)
    try:
        trainer.omega.save_cas_head()
    except Exception as e:
        print(f"Omega 状态保存失败: {e}")
    print(f"完成：保留步数={kept} best_val={best:.4f} ckpt={ckpt}")
    return model, tok


def cmd_generate(a):
    path = Path(a.ckpt)
    model, _ = DualTransformerNP.load_checkpoint(path)
    tok = tokenizer_from_meta(read_checkpoint(path)["meta"])
    out = model.generate_cached(tok.encode(a.prompt, add_bos=True), a.new_tokens,
                                temperature=a.temperature, rng=np.random.default_rng(a.seed),
                                top_k=a.top_k, top_p=a.top_p)
    print(tok.decode(out))


def cmd_test(_):
    with tempfile.TemporaryDirectory() as d:
        txt = Path(d) / "c.txt"
        txt.write_text(("悟空大闹天宫，八戒背媳妇。" * 30 + "\n") * 20, encoding="utf-8")
        ns = build_parser().parse_args([
            "train", "--text", str(txt), "--steps", "40", "--chunk", "10", "--d-model", "16",
            "--blocks", "1", "--heads", "2", "--seq-len", "16", "--batch-size", "4",
            "--max-seq", "32", "--ckpt", str(Path(d) / "m.npz"), "--lr", "3e-3"])
        cmd_train(ns)
        assert Path(d, "m.npz").exists()
        cmd_generate(build_parser().parse_args(["generate", "--ckpt", str(Path(d) / "m.npz"),
                                                "--prompt", "悟空", "--new-tokens", "10"]))
    print("train.py 冒烟测试通过")


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--text", default=str(BASE / "Xiyouji.txt"))
    t.add_argument("--ckpt", default=str(BASE / "ckpt" / "model.npz"))
    t.add_argument("--tokenizer", choices=["char", "byte"], default="char")
    t.add_argument("--val-fraction", type=float, default=0.05)
    t.add_argument("--d-model", type=int, default=64)
    t.add_argument("--blocks", type=int, default=2)
    t.add_argument("--heads", type=int, default=4)
    t.add_argument("--merge", choices=["single", "average", "concat"], default="average")
    t.add_argument("--max-seq", type=int, default=256)
    t.add_argument("--tie-embeddings", action="store_true")
    t.add_argument("--steps", type=int, default=2000)
    t.add_argument("--chunk", type=int, default=50, help="每个 Omega 循环训练的步数")
    t.add_argument("--batch-size", type=int, default=8)
    t.add_argument("--seq-len", type=int, default=64)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--schedule", choices=["constant", "cosine"], default="cosine")
    t.add_argument("--warmup-steps", type=int, default=50)
    t.add_argument("--weight-decay", type=float, default=0.01)
    t.add_argument("--max-grad-norm", type=float, default=1.0)
    t.add_argument("--label-smoothing", type=float, default=0.0)
    t.add_argument("--eval-every", type=int, default=50)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--resume", action="store_true")
    t.add_argument("--no-omega", action="store_true")
    t.add_argument("--log-csv", default=None)
    t.set_defaults(fn=cmd_train)
    g = sub.add_parser("generate")
    g.add_argument("--ckpt", default=str(BASE / "ckpt" / "model.npz"))
    g.add_argument("--prompt", default="")
    g.add_argument("--new-tokens", type=int, default=100)
    g.add_argument("--temperature", type=float, default=0.8)
    g.add_argument("--top-k", type=int, default=20)
    g.add_argument("--top-p", type=float, default=1.0)
    g.add_argument("--seed", type=int, default=0)
    g.set_defaults(fn=cmd_generate)
    sub.add_parser("test").set_defaults(fn=cmd_test)
    return ap


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.fn(args)

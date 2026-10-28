#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""omega_governance 五项能力的断言测试。

运行：python3 tests/omega_governance_test.py
全部断言通过则打印 ALL GOVERNANCE TESTS PASSED，任一失败即抛 AssertionError。
只用标准库 + NumPy，不依赖仓库其他模块（检查点由测试现造最小 dual 格式文件）。
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path
_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent if _HERE.name in ("tests", "tools", "legacy") else _HERE

import numpy as np

sys.path.insert(0, str(ROOT / "src"))

from omega_chat.omega_governance import (  # noqa: E402
    ALLOW, BROKEN, DENY, HOLD, RELEASE, AuditLog, CheckpointGate, Governance,
    MemoryGate, OwnerAuth, Policy, Rollback, ToolGate,
)

TINY_CHARS = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]


def make_tiny_checkpoint(path: Path, val: float = 5.0, step: int = 5,
                         corrupt: bool = False, vocab_size: int = 10,
                         chars: list = TINY_CHARS) -> None:
    """现造最小 dual meta_json 检查点：d=16, 1 层, 2 头, kv=1, 词表 10。
    vocab_size 与 chars 可分开指定，用于验证特殊位缺口规则。"""
    rng = np.random.default_rng(0)
    d, hidden = 16, round(8 * 16 / 3)
    arrays = {
        "emb": rng.normal(0, 0.1, (vocab_size, d)).astype(np.float32),
        "lm_head.weight": rng.normal(0, 0.1, (vocab_size, d)).astype(np.float32),
        "left.0.attn.qkv.weight": rng.normal(0, 0.1, (32, d)).astype(np.float32),
        "left.0.attn.proj.weight": rng.normal(0, 0.1, (d, d)).astype(np.float32),
        "left.0.ff_gate.weight": rng.normal(0, 0.1, (hidden, d)).astype(np.float32),
        "left.0.ff_up.weight": rng.normal(0, 0.1, (hidden, d)).astype(np.float32),
        "left.0.ff2.weight": rng.normal(0, 0.1, (d, hidden)).astype(np.float32),
    }
    if corrupt:
        arrays["left.0.ff2.weight"][0, 0] = np.nan
    meta = {
        "format_version": 1,
        "config": {"d_model": d, "n_blocks": 1, "n_heads": 2, "n_kv_heads": 1,
                   "vocab_size": vocab_size, "ff_mode": "swiglu"},
        "dtype": "float32",
        "step": step,
        "extra": {"best_val_loss": val, "tokenizer": {"chars": chars}},
    }
    arrays = {("p_" + k if k != "emb" else "p_emb"): v for k, v in arrays.items()}
    arrays["meta_json"] = np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8)
    np.savez(path, **arrays)


def test_checkpoint_gate(tmp: Path) -> None:
    audit = AuditLog(tmp / "audit.jsonl")
    gate = CheckpointGate(audit, tmp / "releases")
    good = tmp / "good.npz"
    make_tiny_checkpoint(good, val=5.0)
    res = gate.evaluate(good, reference_chars=TINY_CHARS,
                        recompute_val=lambda p: 5.02,
                        generate=lambda prompt: "悟空道行者三藏八戒沙僧白马经文取经大唐东土")
    assert res["verdict"] == RELEASE, res["checks"]
    rel = gate.release(good, "tiny", reference_chars=TINY_CHARS,
                       recompute_val=lambda p: 5.02,
                       generate=lambda prompt: "悟空道行者三藏八戒沙僧白马经文取经大唐东土")
    assert rel["released"] is True, rel
    assert (tmp / "releases" / rel["release_name"]).exists()
    # 同名同版本再放行：进步项拦截（同一版本不是进步），这是正确的 HOLD
    rel2 = gate.release(good, "tiny", reference_chars=TINY_CHARS,
                        recompute_val=lambda p: 5.02,
                        generate=lambda prompt: "悟空道行者三藏八戒沙僧白马经文取经大唐东土")
    assert rel2["released"] is False and rel2["verdict"] == HOLD
    # 永不覆盖分支：目标文件名已存在时加后缀（用全新审计走首次放行路径）
    gate2 = CheckpointGate(AuditLog(tmp / "audit2.jsonl"), tmp / "releases")
    rel3 = gate2.release(good, "tiny", reference_chars=TINY_CHARS,
                         recompute_val=lambda p: 5.02,
                         generate=lambda prompt: "悟空道行者三藏八戒沙僧白马经文取经大唐东土")
    assert rel3["released"] is True and rel3["release_name"] != rel["release_name"]
    assert len(list((tmp / "releases").glob("tiny_step5_val5.0000*.npz"))) == 2
    # 更差的 val：进步项拦截
    worse = tmp / "worse.npz"
    make_tiny_checkpoint(worse, val=5.5, step=6)
    res = gate.evaluate(worse, reference_chars=TINY_CHARS,
                        recompute_val=lambda p: 5.5)
    assert res["verdict"] == HOLD, res["checks"]
    assert any(c["name"] == "improvement" and c["status"] == "fail" for c in res["checks"])
    # 损坏：NaN 权重
    bad = tmp / "bad.npz"
    make_tiny_checkpoint(bad, val=4.0, corrupt=True)
    res = gate.evaluate(bad)
    assert res["verdict"] == BROKEN, res["checks"]
    # 记录不诚实：重算差太多
    res = gate.evaluate(good, recompute_val=lambda p: 6.0)
    assert res["verdict"] == HOLD, res["checks"]
    # 生成退化：单字占满
    res = gate.evaluate(good, recompute_val=lambda p: 5.0,
                        generate=lambda prompt: "啊" * 80)
    assert res["verdict"] == HOLD, res["checks"]
    # 特殊位缺口：词表 13、记录字 10（缺口 3，如真实 4510 = 4507 + 3），应放行
    # 用全新审计走首次放行路径，避免被前面已放行的 5.0 挡在进步项
    gate_gap = CheckpointGate(AuditLog(tmp / "audit_gap.jsonl"), tmp / "releases_gap")
    gap_ok = tmp / "gap_ok.npz"
    make_tiny_checkpoint(gap_ok, val=5.0, step=7, vocab_size=13, chars=TINY_CHARS)
    res = gate_gap.evaluate(gap_ok, reference_chars=TINY_CHARS,
                            recompute_val=lambda p: 5.0,
                            generate=lambda prompt: "悟空道行者三藏八戒沙僧白马经文取经大唐东土")
    assert res["verdict"] == RELEASE, res["checks"]
    assert any(c["name"] == "vocab" and c["status"] == "pass"
               and "presumed specials" in c["detail"] for c in res["checks"])
    # 大缺口：词表 30、记录字 10（缺口 20），应拦截
    gap_big = tmp / "gap_big.npz"
    make_tiny_checkpoint(gap_big, val=5.0, step=8, vocab_size=30, chars=TINY_CHARS)
    res = gate_gap.evaluate(gap_big, reference_chars=TINY_CHARS,
                            recompute_val=lambda p: 5.0,
                            generate=lambda prompt: "悟空道行者三藏八戒沙僧白马经文取经大唐东土")
    assert res["verdict"] == HOLD, res["checks"]
    assert any(c["name"] == "vocab" and c["status"] == "fail" for c in res["checks"])
    print("checkpoint gate: PASS")


def test_tool_gate(tmp: Path) -> None:
    policy = Policy()
    auth = OwnerAuth(tmp)
    auth.create_owner("correct-horse")
    audit = AuditLog(tmp / "audit.jsonl")
    gate = ToolGate(policy, auth, audit)
    # 低风险：直接放行并可执行
    r = gate.authorize("计算", {"arg": "二加三"})
    assert r["decision"] == ALLOW and r["token"] is None
    out = gate.execute(None, "计算", {"arg": "二加三"}, lambda a: "五")
    assert out["executed"] is True and out["result"] == "五"
    # 需确认：无密钥 / 错密钥 → HOLD
    assert gate.authorize("记住", {"text": "x"})["decision"] == HOLD
    assert gate.authorize("记住", {"text": "x"}, "wrong-secret")["decision"] == HOLD
    # 正确密钥 → 发令牌；执行一次成功，复用拒绝，参数不符拒绝
    r = gate.authorize("记住", {"text": "hello"}, "correct-horse")
    assert r["decision"] == ALLOW and r["token"]
    out = gate.execute(r["token"], "记住", {"text": "hello"}, lambda a: "stored")
    assert out["executed"] is True
    out = gate.execute(r["token"], "记住", {"text": "hello"}, lambda a: "stored")
    assert out["executed"] is False and "used" in out["error"]
    r2 = gate.authorize("记住", {"text": "hello"}, "correct-horse")
    out = gate.execute(r2["token"], "记住", {"text": "other"}, lambda a: "stored")
    assert out["executed"] is False and "mismatch" in out["error"]
    # 未知动作拒绝；暂停时全部 HOLD
    assert gate.authorize("删库", {})["decision"] == DENY
    policy.pause()
    assert gate.authorize("计算", {})["decision"] == HOLD
    assert gate.execute(None, "计算", {}, lambda a: "x")["executed"] is False
    policy.resume()
    assert gate.authorize("计算", {})["decision"] == ALLOW
    print("tool gate: PASS")


def test_memory_gate(tmp: Path) -> None:
    policy = Policy()
    audit = AuditLog(tmp / "audit.jsonl")
    gate = MemoryGate(policy, audit)
    assert gate.authorize_write("不加糖")["decision"] == ALLOW
    assert gate.authorize_write("x" * 501)["decision"] == DENY
    assert gate.authorize_write("记住《计算》")["decision"] == DENY
    policy.pause()
    assert gate.authorize_write("不加糖")["decision"] == DENY
    print("memory gate: PASS")


def test_audit_log(tmp: Path) -> None:
    audit = AuditLog(tmp / "audit.jsonl")
    audit.append("tool", "owner", "计算", ALLOW, {})
    audit.append("memory", "owner", "记住", ALLOW, {})
    ok, msg = audit.verify_chain()
    assert ok, msg
    # 篡改内存记录 → 验链失败
    audit.records[0]["detail"] = {"tampered": True}
    ok, _ = audit.verify_chain()
    assert not ok
    # HMAC 版本同样可验
    audit2 = AuditLog(tmp / "audit2.jsonl", key=b"test-key")
    audit2.append("tool", "owner", "计算", ALLOW, {})
    ok, msg = audit2.verify_chain()
    assert ok, msg
    # 已放行最佳 val 取最小值
    audit3 = AuditLog(tmp / "audit3.jsonl")
    audit3.append("release", "governance", "a", RELEASE, {"val": 5.0, "release_name": "a"})
    audit3.append("release", "governance", "b", RELEASE, {"val": 4.5, "release_name": "b"})
    best, name = audit3.best_released_val()
    assert best == 4.5 and name == "b"
    print("audit log: PASS")


def test_rollback(tmp: Path) -> None:
    audit = AuditLog(tmp / "audit.jsonl")
    rb = Rollback(audit, tmp / "snapshots")
    work = tmp / "work"
    work.mkdir()
    (work / "a.txt").write_text("v1", encoding="utf-8")
    (work / "b.txt").write_text("v1", encoding="utf-8")
    snap = rb.snapshot([work / "a.txt", work / "b.txt"], label="before change")
    assert snap in rb.list_snapshots()
    (work / "a.txt").write_text("v2", encoding="utf-8")
    (work / "b.txt").write_text("v2", encoding="utf-8")
    out = rb.restore(snap, work)
    assert out["restored"] is True
    assert (work / "a.txt").read_text(encoding="utf-8") == "v1"
    assert (work / "b.txt").read_text(encoding="utf-8") == "v1"
    # 未知快照拒绝
    assert rb.restore("no-such-snap", work)["restored"] is False
    # 快照文件被篡改 → 恢复拒绝
    snap_dir = tmp / "snapshots" / snap
    (snap_dir / "a.txt").write_text("tampered", encoding="utf-8")
    assert rb.restore(snap, work)["restored"] is False
    # 取代标记只写审计，不删文件
    out = rb.supersede("tiny_step5_val5.0000.npz", "superseded by newer release")
    assert out["superseded"] is True
    assert any(r["kind"] == "supersede" for r in audit.records)
    print("rollback: PASS")


def test_facade(tmp: Path) -> None:
    gov = Governance(tmp)
    gov.owner_auth.create_owner("correct-horse")
    st = gov.status()
    assert st["paused"] is False and st["audit_chain_ok"] is True
    assert "计算" in st["actions"] and "记住" in st["actions"]
    gov.pause()
    assert gov.status()["paused"] is True
    gov.resume()
    assert gov.status()["paused"] is False
    print("facade: PASS")


def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        test_checkpoint_gate(tmp / "ckpt")
        test_tool_gate(tmp / "tool")
        test_memory_gate(tmp / "mem")
        test_audit_log(tmp / "audit")
        test_rollback(tmp / "roll")
        test_facade(tmp / "facade")
    print("ALL GOVERNANCE TESTS PASSED")


if __name__ == "__main__":
    main()

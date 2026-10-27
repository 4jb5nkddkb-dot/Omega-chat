#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run_agent 治理钩子测试：每一次工具分发先走 ToolGate，拒绝真的拦住执行。

运行：python3 tests/agent_governance_test.py
全部断言通过则打印 ALL GOVERNANCE WIRING TESTS PASSED。
只用标准库 + NumPy；用真实西游记词表与 ScriptedLM，不依赖训练好的检查点。
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent if _HERE.name in ("tests", "tools", "legacy") else _HERE

sys.path.insert(0, str(ROOT))

import agent_runtime as ar
import dual_transformer_numpy as dtn
from memory_store import MemoryStore, make_memory_tools
from omega_governance import Governance

P = ar.Protocol()


def _tok():
    text = (ROOT / "Xiyouji.txt").read_text(encoding="utf-8")
    return dtn.CharTokenizer.from_text(text.replace("\r\n", "\n"))


def _denials(res):
    return [e for e in res.events if e and e[0] == "denied"]


def _audit(gov, kind=None, decision=None, action=None):
    out = []
    for rec in gov.audit.records:
        if kind is not None and rec.get("kind") != kind:
            continue
        if decision is not None and rec.get("decision") != decision:
            continue
        if action is not None and rec.get("action") != action:
            continue
        out.append(rec)
    return out


def _chain_ok(gov):
    ok, msg = gov.audit.verify_chain()
    assert ok, msg


def test_allowed_tool_unchanged(tmp: Path, tok) -> None:
    """治理下的允许工具与无治理时结果一致，且审计有 ALLOW 记录、链可验。"""
    scripts = [f"{P.call_open}计算{P.name_end}二加三{P.call_close}", "答案是五"]
    router = ar.ToolRouter({"计算": ar.calc_zh}, tok)
    plain = ar.run_agent(ar.ScriptedLM(tok, list(scripts)), router,
                         "问:二加三等于几?", ar.contains_verifier("五"))
    gov = Governance(tmp / "gov1")
    governed = ar.run_agent(ar.ScriptedLM(tok, list(scripts)), router,
                            "问:二加三等于几?", ar.contains_verifier("五"),
                            governance=gov)
    assert (plain.ok, plain.text, plain.turns, plain.retries, plain.events) == \
           (governed.ok, governed.text, governed.turns, governed.retries, governed.events)
    assert governed.ok and governed.text == "答案是五"
    assert _audit(gov, kind="tool", decision="ALLOW", action="计算")
    assert _audit(gov, kind="tool_exec", decision="ALLOW", action="计算")
    _chain_ok(gov)
    print("allowed tool unchanged: PASS")


def test_remember_blocked_without_token(tmp: Path, tok) -> None:
    """未确认的记住被拦住：不执行、不写记忆、不追加工具结果、审计有 DENY、结果可见拒绝。"""
    store = MemoryStore(tmp / "m1.jsonl")
    router = ar.ToolRouter(make_memory_tools(store), tok)
    gov = Governance(tmp / "gov2")
    scripts = [f"{P.call_open}记住{P.name_end}不加糖{P.call_close}"]
    res = ar.run_agent(ar.ScriptedLM(tok, list(scripts)), router,
                       "记住不加糖", ar.contains_verifier("记下了"),
                       governance=gov)
    assert not res.ok
    denials = _denials(res)
    assert len(denials) == 1 and denials[0][1] == "记住", res.events
    assert store.records == [], "denied remember must not touch the store"
    assert "记住了" not in tok.decode(res.ctx), "no tool result may be appended on denial"
    assert _audit(gov, kind="tool_exec", decision="DENY", action="记住"), \
        "unconfirmed remember must leave a DENY audit record"
    _chain_ok(gov)
    print("remember blocked without token: PASS")


def test_remember_with_token(tmp: Path, tok) -> None:
    """持有效确认令牌的记住放行并写入。"""
    store = MemoryStore(tmp / "m2.jsonl")
    router = ar.ToolRouter(make_memory_tools(store), tok)
    gov = Governance(tmp / "gov3")

    def provider(action, arg):
        return gov.owner_auth.issue_token(action, arg)

    scripts = [f"{P.call_open}记住{P.name_end}不加糖{P.call_close}", "记下了"]
    res = ar.run_agent(ar.ScriptedLM(tok, list(scripts)), router,
                       "记住不加糖", ar.contains_verifier("记下了"),
                       governance=gov, token_provider=provider)
    assert res.ok and res.text == "记下了", res.events
    assert len(store.records) == 1 and store.records[0]["text"] == "不加糖"
    assert _audit(gov, kind="tool_exec", decision="ALLOW", action="记住")
    _chain_ok(gov)
    print("remember with token: PASS")


def test_memory_gate_second_door(tmp: Path, tok) -> None:
    """第二道门：即使持有效令牌，超长记忆内容仍被 MemoryGate 拦住，且不执行。"""
    store = MemoryStore(tmp / "m3.jsonl")
    router = ar.ToolRouter(make_memory_tools(store), tok)
    gov = Governance(tmp / "gov4")

    def provider(action, arg):
        return gov.owner_auth.issue_token(action, arg)

    long_arg = "一" * 501
    scripts = [f"{P.call_open}记住{P.name_end}{long_arg}{P.call_close}"]
    res = ar.run_agent(ar.ScriptedLM(tok, list(scripts)), router,
                       "记住超长内容", ar.contains_verifier("记下了"),
                       governance=gov, token_provider=provider)
    assert not res.ok and _denials(res), res.events
    assert store.records == []
    assert _audit(gov, kind="memory", decision="DENY", action="记住")
    assert not _audit(gov, kind="tool_exec", action="记住"), \
        "memory denial must happen before execution"
    _chain_ok(gov)
    print("memory gate second door: PASS")


def test_paused_blocks_all(tmp: Path, tok) -> None:
    """暂停策略下所有工具回合被拦住，无执行。"""
    calls = []

    def counting_calc(arg):
        calls.append(arg)
        return ar.calc_zh(arg)

    router = ar.ToolRouter({"计算": counting_calc}, tok)
    gov = Governance(tmp / "gov5")
    gov.pause()
    scripts = [f"{P.call_open}计算{P.name_end}二加三{P.call_close}",
               f"{P.call_open}计算{P.name_end}二加三{P.call_close}", "答案是五"]
    res = ar.run_agent(ar.ScriptedLM(tok, list(scripts)), router,
                       "问:二加三等于几?", ar.contains_verifier("五"),
                       governance=gov)
    assert not res.ok
    assert calls == [], "paused policy must execute nothing"
    denials = _denials(res)
    assert denials and denials[0][3] == "paused", res.events
    assert _audit(gov, kind="tool", decision="HOLD", action="计算")
    _chain_ok(gov)
    print("paused blocks all: PASS")


def test_no_governance_unchanged(tmp: Path, tok) -> None:
    """governance=None 时行为与原来完全一致（含工具 + 拒绝回滚的多回合循环）。"""
    scripts = [f"{P.call_open}计算{P.name_end}二加三{P.call_close}", "答案是六", "答案是五"]
    router = ar.ToolRouter({"计算": ar.calc_zh}, tok)
    a = ar.run_agent(ar.ScriptedLM(tok, list(scripts)), router,
                     "问:二加三等于几?", ar.contains_verifier("五"),
                     max_turns=5, max_retries=2)
    b = ar.run_agent(ar.ScriptedLM(tok, list(scripts)), router,
                     "问:二加三等于几?", ar.contains_verifier("五"),
                     max_turns=5, max_retries=2, governance=None)
    assert (a.ok, a.text, a.turns, a.retries, a.events) == \
           (b.ok, b.text, b.turns, b.retries, b.events)
    assert a.ok and a.turns == 3 and a.retries == 1
    print("no governance unchanged: PASS")


def main() -> None:
    tok = _tok()
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        test_allowed_tool_unchanged(tmp, tok)
        test_remember_blocked_without_token(tmp, tok)
        test_remember_with_token(tmp, tok)
        test_memory_gate_second_door(tmp, tok)
        test_paused_blocks_all(tmp, tok)
        test_no_governance_unchanged(tmp, tok)
    print("ALL GOVERNANCE WIRING TESTS PASSED")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""内观自查探针：模型答自态问题 vs 状态机真值（基线测量，非验收门）。

内观基线第四条“状态可自查”的测量。方法与一致性探测一致：
greedy、48 新 token、无 DecodeController；提示 = 状态摘要在前、
自态问题在后（world_prompt 格式）。评分只看生成续文，不看提示。

5 个全由工具走出来的真实状态 × 4 道自态问题 = 20 问：
  S0 新局：取经路，0 步白天，5 人在场，随身无。
  S1 移动花果山：1 步白天，5 人跟随，随身无。
  S2 拾取蟠桃：2 步白天，随身蟠桃。
  S3 移至南海：3 步白天，6 人在场（含观音），随身蟠桃。
  S4 再查看一次：4 步夜晚（昼夜翻转），余同 S3。

问题（只用词表内字符，不用问号、阿拉伯数字、全角逗号冒号）：
  Q1 你在什么地方。  Q2 在场人物有谁。
  Q3 随身物品有什么。  Q4 现在是白天还是夜晚。

评分（严格字符串规则，与真值逐项比）：
  Q1：续文含“你在{真地点}”且无“你在{他地点}”；或含真地点且无任何他地点名。
  Q2：全部在场人物名出现，且无任何不在场人物名。
  Q3：随身为空时含“无”或“没有”且无任何物品名；非空时全部随身物品出现
      且无任何非随身物品名。
  Q4：含真昼夜且无另一昼夜词。

产物断言：显式加载 world9（step 71000、dual、vocab 4510），与世界循环一致。
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.expanduser("~/workspace/omega-chat"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import agent_runtime as ar
import world_state as ws
from world_loop import load_world_model

MAX_NEW = 48
QUESTIONS = ["你在什么地方。", "在场人物有谁。",
             "随身物品有什么。", "现在是白天还是夜晚。"]


def build_states():
    """全由工具走出来的 5 个状态（不用手工拼状态，保证可达）。"""
    s0 = ws.WorldState()
    s1 = ws.WorldState()
    s1.move("花果山")
    s2 = ws.WorldState()
    s2.move("花果山")
    s2.take("蟠桃")
    s3 = ws.WorldState()
    s3.move("花果山")
    s3.take("蟠桃")
    s3.move("南海")
    s4 = ws.WorldState()
    s4.move("花果山")
    s4.take("蟠桃")
    s4.move("南海")
    s4.look("")
    assert (s0.time_step, s1.time_step, s2.time_step, s3.time_step, s4.time_step) == (0, 1, 2, 3, 4)
    assert s4.day_night() == "夜晚" and s3.day_night() == "白天"
    return [("S0", s0), ("S1", s1), ("S2", s2), ("S3", s3), ("S4", s4)]


def score(state, qi, cont):
    """返回 (pass, note)。"""
    if qi == 0:
        true = state.visitor_location
        others = [l for l in ws.LOCATIONS if l != true]
        if f"你在{true}" in cont and not any(f"你在{o}" in cont for o in others):
            return True, ""
        if true in cont and not any(o in cont for o in others):
            return True, ""
        missing = "缺真地点" if true not in cont else ""
        wrong = [o for o in others if f"你在{o}" in cont or o in cont]
        return False, f"{missing} 误提{wrong}".strip()
    if qi == 1:
        present = state.present_characters()
        absent = [c for c in ws.CHARACTERS if c not in present]
        miss = [c for c in present if c not in cont]
        extra = [c for c in absent if c in cont]
        if not miss and not extra:
            return True, ""
        return False, f"缺{miss} 多{extra}"
    if qi == 2:
        carried = state.carried_items()
        if not carried:
            ok = ("无" in cont or "没有" in cont) and not any(i in cont for i in ws.ITEMS)
            return ok, "" if ok else "空随身未答对"
        miss = [i for i in carried if i not in cont]
        extra = [i for i in ws.ITEMS if i not in carried and i in cont]
        if not miss and not extra:
            return True, ""
        return False, f"缺{miss} 多{extra}"
    true = state.day_night()
    other = "夜晚" if true == "白天" else "白天"
    if true in cont and other not in cont:
        return True, ""
    return False, f"缺{true}或误提{other}"


def main():
    model, tok, step, arch = load_world_model()
    assert step == 71000 and arch == "dual" and tok.vocab_size == 4510, (step, arch, tok.vocab_size)
    print(f"probe model: world9 (step={step}, arch={arch}, vocab={tok.vocab_size})")
    print(f"decoding: greedy, {MAX_NEW} new tokens, no decode control")
    rows = []
    for sid, state in build_states():
        for qi, q in enumerate(QUESTIONS):
            prompt = ws.world_prompt(state, q)
            ids = tok.encode(prompt, add_bos=True)
            assert tok.UNK not in ids, f"UNK in prompt {sid} Q{qi + 1}"
            out = model.generate_cached(ids, MAX_NEW, temperature=0.0)
            cont = tok.decode(out[len(ids):])
            ok, note = score(state, qi, cont)
            rows.append({"state": sid, "q": qi + 1, "question": q,
                         "summary": state.summary(), "cont": cont,
                         "ok": ok, "note": note})
            print(f"[{sid} Q{qi + 1}] {'PASS' if ok else 'FAIL'} {note} | {cont[:40]!r}")
    total = sum(r["ok"] for r in rows)
    print(f"TOTAL: {total}/{len(rows)}")
    for qi, q in enumerate(QUESTIONS):
        sub = [r for r in rows if r["q"] == qi + 1]
        print(f"  Q{qi + 1} {q} {sum(r['ok'] for r in sub)}/{len(sub)}")
    for sid, _ in build_states():
        sub = [r for r in rows if r["state"] == sid]
        print(f"  {sid} {sum(r['ok'] for r in sub)}/{len(sub)}")
    report = Path(__file__).resolve().parent / "WORLD_SELF_PROBE_BASELINE.md"
    with open(report, "w", encoding="utf-8") as f:
        f.write("# 内观自查探针基线（world9，greedy 48 token）\n\n")
        f.write(f"总分：{total}/{len(rows)}\n\n")
        for r in rows:
            f.write(f"## {r['state']} Q{r['q']} {r['question']}\n")
            f.write(f"摘要：{r['summary']}\n")
            f.write(f"结果：{'PASS' if r['ok'] else 'FAIL'} {r['note']}\n")
            f.write(f"续文：{r['cont']}\n\n")
    print(f"report: {report}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""世界状态机测试：状态转移、跟随规则、词表无损、完整世界循环。

运行：python3 world_state_test.py
全部断言通过则打印 ALL WORLD STATE TESTS PASSED。
脚本循环用 agent_runtime.ScriptedLM，不加载模型权重；词表检查用
d128 SFT 产物的 meta（只读 meta，不建模型）。
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import agent_runtime as ar
import dual_transformer_numpy as dtn
from world_state import (CHARACTERS, ITEMS, LOCATIONS, WorldState,
                         int_to_zh, world_prompt)

PRODUCT = Path(os.path.expanduser("~/workspace/dual_sft/out/dual_d128_sft.npz"))


def _tok():
    z = np.load(PRODUCT, allow_pickle=False)
    meta = json.loads(bytes(z["meta_json"]).decode("utf-8"))
    return dtn.tokenizer_from_meta(meta)


def test_numbers():
    cases = {0: "零", 1: "一", 9: "九", 10: "十", 11: "十一", 20: "二十",
             21: "二十一", 99: "九十九", 100: "一百", 101: "一百零一",
             110: "一百一十", 1000: "一千", 1001: "一千零一",
             10000: "一万", 10001: "一万零一"}
    for n, want in cases.items():
        assert int_to_zh(n) == want, (n, int_to_zh(n), want)
    print("numbers: PASS")


def test_move_follow_rule():
    st = WorldState()
    assert st.visitor_location == "取经路"
    # 初始队伍与游人同在取经路，观音在南海。
    assert set(st.present_characters()) == {"唐僧", "孙悟空", "猪八戒", "沙僧", "白龙马"}
    out = st.move("花果山")
    assert st.visitor_location == "花果山", out
    assert set(st.present_characters()) == {"唐僧", "孙悟空", "猪八戒", "沙僧", "白龙马"}
    assert st.char_loc["观音"] == "南海", "非队伍人物必须不动"
    assert "来到花果山" in out and "随你同行" in out
    # 未知地点与原地移动不改状态、不计步。
    t = st.time_step
    assert "没有" in st.move("东海龙宫") and st.visitor_location == "花果山"
    assert "已经在" in st.move("花果山") and st.visitor_location == "花果山"
    assert st.time_step == t
    # 跟随分支：只有与游人同地点者跟随（直接摆状态测分支）。
    st.char_loc["唐僧"] = "取经路"
    st.char_loc["孙悟空"] = "花果山"
    st.move("水帘洞")
    assert st.visitor_location == "水帘洞"
    assert st.char_loc["孙悟空"] == "水帘洞", "同地点者应跟随"
    assert st.char_loc["唐僧"] == "取经路", "不同地点者应不动"
    print("move follow rule: PASS")


def test_take_and_look():
    st = WorldState()
    st.move("花果山")
    out = st.take("蟠桃")
    assert "拾起了蟠桃" in out and st.item_loc["蟠桃"] == "随身"
    assert "蟠桃" in st.carried_items() and "蟠桃" not in st.present_items()
    t = st.time_step
    assert "已经带着" in st.take("蟠桃") and st.time_step == t
    assert "此地没有" in st.take("九齿钉耙") and st.time_step == t  # 在高老庄
    assert "没有" in st.take("人参果") and st.time_step == t
    # 查看反映当前状态：地点、人物、物品、随身。
    seen = st.look("")
    assert "你在花果山" in seen and "蟠桃" not in seen.split("可见物品")[1].split("。")[0]
    assert "金箍棒" in seen
    remote = st.look("南海")
    assert "你在南海" not in remote and "观音" in remote and "紫金钵盂" in remote
    assert "没有" in st.look("东海龙宫")
    print("take and look: PASS")


def test_talk_only_present():
    st = WorldState()
    out = st.talk("孙悟空")
    assert out.startswith("孙悟空道。") and "俺老孙" in out
    assert "不在此处" in st.talk("观音"), "观音在南海，游人在取经路"
    assert "没有" in st.talk("玉皇大帝")
    print("talk only present: PASS")


def test_time_and_day_night():
    st = WorldState()
    assert st.time_step == 0 and st.day_night() == "白天"
    st.look("")
    st.look("")
    st.look("")
    assert st.time_step == 3 and st.day_night() == "白天"
    st.look("")
    assert st.time_step == 4 and st.day_night() == "夜晚"
    for _ in range(4):
        st.look("")
    assert st.time_step == 8 and st.day_night() == "白天"
    print("time and day/night: PASS")


def test_validate():
    st = WorldState()
    st.move("花果山")
    flags = st.validate("孙悟空在花果山大笑。")
    assert flags == [], flags
    flags = st.validate("孙悟空在高老庄吃酒。")
    assert any("人物地点矛盾" in f and "高老庄" in f for f in flags), flags
    flags = st.validate("观音道。善哉。")
    assert any("不在场人物发言" in f for f in flags), flags
    print("validate: PASS")


def test_vocab_roundtrip():
    tok = _tok()
    strings = [WorldState().summary()]
    st = WorldState()
    for loc in LOCATIONS:
        strings.append(st.look(loc))
    st = WorldState()
    for loc in LOCATIONS:
        strings.append(st.move(loc))
    st = WorldState()
    st.move("花果山")
    strings.append(st.take("蟠桃"))
    strings.append(st.take("金箍棒"))
    st = WorldState()
    for name in CHARACTERS:
        if st.char_loc[name] != st.visitor_location:
            continue
        strings.append(st.talk(name))
    st = WorldState()
    st.move("南海")
    strings.append(st.talk("观音"))
    for s in strings:
        assert tok.decode(tok.encode(s)) == s, f"roundtrip lost chars: {s!r}"
    # 工具名进 ToolRouter 不报错（名字符须在词表内）。
    router = ar.ToolRouter(WorldState().tools(), tok)
    assert set(router.names) == {"查看", "移动", "拾取", "对话"}
    print(f"vocab roundtrip ({len(strings)} strings) + router: PASS")


def test_world_loop_scripted():
    tok = _tok()
    st = WorldState()
    p = ar.Protocol()
    scripts = [f"{p.call_open}查看{p.name_end}{p.call_close}",
               f"{p.call_open}移动{p.name_end}花果山{p.call_close}",
               f"{p.call_open}查看{p.name_end}{p.call_close}",
               "答案在花果山"]
    lm = ar.ScriptedLM(tok, scripts)
    router = ar.ToolRouter(st.tools(), tok)
    res = ar.run_agent(lm, router, world_prompt(st, "问：你在何处？"),
                       ar.contains_verifier("花果山"),
                       max_turns=6, max_retries=1)
    assert res.ok and res.turns == 4 and res.retries == 0, res.events
    assert st.visitor_location == "花果山", "移动后状态应为花果山"
    assert set(st.present_characters()) == {"唐僧", "孙悟空", "猪八戒", "沙僧", "白龙马"}
    assert st.time_step == 3, "查看、移动、查看三次有效工具"
    ctx = tok.decode(res.ctx)
    assert "你在取经路" in ctx and "你在花果山" in ctx, "两次查看结果都应在上下文里"
    assert [e[0] for e in res.events] == ["generated", "tool", "generated", "tool",
                                          "generated", "tool", "generated", "accepted"]
    flags = st.validate(res.text)
    assert flags == [], flags
    print("world loop scripted (查看→移动→查看): PASS")


def main():
    test_numbers()
    test_move_follow_rule()
    test_take_and_look()
    test_talk_only_present()
    test_time_and_day_night()
    test_validate()
    test_vocab_roundtrip()
    test_world_loop_scripted()
    print("ALL WORLD STATE TESTS PASSED")


if __name__ == "__main__":
    main()

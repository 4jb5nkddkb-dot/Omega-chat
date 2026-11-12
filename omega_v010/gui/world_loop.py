#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""小世界循环草稿（未进镜像、未提交）：把已有的三件组装成可玩的一回合。

组装件（都不改）：
  - agent_runtime.load_checkpoint：显式加载世界循环产物
    dual_d128_sft_selfreport_full.npz（2026-10-29 用户拍板切换后，
    全量自报产品接替 world9 成为世界循环产物；仅显式加载，
    默认主力仍是 dual_d128_sft.npz，不改任何默认）
  - agent_runtime.ToolRouter + run_agent：工具分发与闭环
  - world_state.WorldState：地点、人物、物品、时间 + world_prompt 摘要
  - world_anchor.SelfState + anchor_prompt：每回合提示最前面带锚点行
    （2026-10-29 用户拍板接线前，循环只发摘要；接线后锚点行 + 摘要 +
    用户话三段进提示，默认自我为 Omega/omega，调用方可传 SelfState
    换人格、换意图）

关键点（2026-10-25 晚复核时发现）：
  agent_runtime.select_checkpoint 的默认列表里只有 dual_d128_sft.npz，
  没有 dual_d128_sft_world.npz——默认走下去仍会选中旧产物。所以世界循环
  必须显式传新产物文件名，不能依赖默认选择；旧产物继续留给无摘要的
  自由生成。这是按 2026-10-25 晚助手建议的分工（世界循环用新产物、
  自由生成用旧产物）写的第一版，未经用户拍板前不改任何默认。

用法（真实模型）：
  model, tok, step, arch = load_world_model()
  state = WorldState()
  router = make_world_router(tok, state)
  result, flags = world_turn(model, router, state, "去花果山看看。")
  # result.ok / result.text / flags（validate 的矛盾标记，只记录不改写）

治理说明：world_state 的说明已写明——治理开启时，查看、移动、拾取、
对话须在策略里显式列入，否则 ToolGate 按白名单外拒绝；本草稿默认
governance=None，与 world_state 的默认一致，不改 omega_governance 的
默认策略。
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.expanduser("~/workspace/omega-chat"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import agent_runtime as ar
import world_anchor as wa
import world_state as ws

WORLD_CKPT = Path(os.path.expanduser(
    "~/workspace/dual_sft/out/dual_d128_sft_selfreport_move.npz"))
OLD_CKPT = Path(os.path.expanduser(
    "~/workspace/dual_sft/out/dual_d128_sft.npz"))


def load_world_model(ckpt_path=None):
    """显式加载世界产物（不走 select_checkpoint 的默认列表）。

    默认值在调用时解析 WORLD_CKPT，不在定义时绑定——第七轮的教训：
    定义时绑定的默认值改 WORLD_CKPT 后仍指向旧产物，会静默加载错误模型。
    """
    if ckpt_path is None:
        ckpt_path = WORLD_CKPT
    model, tok, step, arch = ar.load_checkpoint(str(ckpt_path))
    assert arch == "dual", f"world product must be dual, got {arch}"
    return model, tok, step, arch


def make_world_router(tok, state):
    return ar.ToolRouter(state.tools(), tok)


def _accept_any(text):
    return True, ""


def world_turn(model, router, state, prompt, *, self_state=None,
               max_new=96, max_turns=6,
               governance=None, token_provider=None):
    """一回合：锚点行 + 摘要在前、用户话在后；跑完返回 (AgentResult, 矛盾标记列表)。

    self_state 为空时用默认自我（Omega/omega、无意图）；调用方可持一个
    SelfState 在回合之间 set_persona/set_intent，锚点行随之换。
    提示结构与锚点探针一致：锚点行 + "\\n" + 摘要 + "\\n" + 用户话。
    """
    if self_state is None:
        self_state = wa.SelfState()
    full_prompt = wa.anchor_prompt(self_state, state) + "\n" + prompt
    result = ar.run_agent(model, router, full_prompt, _accept_any,
                          max_new=max_new, max_turns=max_turns,
                          governance=governance, token_provider=token_provider)
    flags = state.validate(result.text)
    return result, flags


if __name__ == "__main__":
    # 确定性自测：不用真实模型，用 ScriptedLM 证明组装逻辑本身通——
    # 模型先发一个移动工具调用，状态真的移动；再发叙述，循环接受。
    _, tok, step, arch = load_world_model()
    print(f"loaded world product: step={step} arch={arch} vocab={tok.vocab_size}")
    assert step == 74373 and arch == "dual" and tok.vocab_size == 4510
    state = ws.WorldState()
    router = make_world_router(tok, state)
    p = ar.Protocol()
    scripts = [f"{p.call_open}移动{p.name_end}花果山{p.call_close}",
               "你来到花果山。水帘洞天。孙悟空在此。"]
    lm = ar.ScriptedLM(tok, scripts)
    result, flags = world_turn(lm, router, state, "去花果山看看。")
    assert result.ok, result.events
    assert state.visitor_location == "花果山", state.visitor_location
    assert "花果山" in state.summary(), state.summary()
    assert flags == [], flags
    print("WORLD LOOP DRAFT TEST PASSED")
    print("summary:", state.summary())
    print("events:", result.events)

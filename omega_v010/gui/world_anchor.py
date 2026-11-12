#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""内观锚点：主体 + 当前人格 + 当前意图（纯标准库）。

主体（永远不变的“我”）与人格（当前戴的面具）分开：
  主体：Omega —— 机器名 "Omega"，模型内写法 "欧米伽"（词表可编码）。
  人格：omega（Omega 自身）、紫薇、悟空、一一、Dayan（暂定 5 个）。
    机器名进存档（任何写法都行）；模型内写法必须词表可编码：
    omega -> 欧米伽，紫薇 -> 紫薇，悟空 -> 悟空，一一 -> 一一，
    Dayan -> 大眼（全名贾大眼；拉丁写法缺 D、n 两个字母，词表写不出来，
    故模型内一律用中文名）。

SelfState 与 WorldState 是两回事：世界状态是外面的世界，
自我状态是里面的“我”。存档 v2 把两者并排放（见 world_persist）。

锚点行格式（只用词表内字符）：
  我是欧米伽。当前人格悟空。我要去花果山看看。
意图为空时不写意图句。锚点行永远放在状态摘要之前。
"""
import os
import sys

sys.path.insert(0, os.path.expanduser("~/workspace/omega-chat"))

SUBJECT = "Omega"
SUBJECT_MODEL = "欧米伽"
PERSONAS = ("omega", "紫薇", "悟空", "一一", "Dayan")
PERSONA_MODEL = {
    "omega": "欧米伽",
    "紫薇": "紫薇",
    "悟空": "悟空",
    "一一": "一一",
    "Dayan": "大眼",
}
# 全名（只有 Dayan 由用户给了全名；其余人格全名即模型名）。
PERSONA_FULLNAME = {
    "Dayan": "贾大眼",
}


def persona_full_name(persona):
    """人格全名（正式自我介绍用）；未给全名的即模型名。"""
    if persona not in PERSONAS:
        raise ValueError(f"unknown persona {persona!r}")
    return PERSONA_FULLNAME.get(persona, PERSONA_MODEL[persona])
DEFAULT_PERSONA = "omega"


class SelfState:
    """自我状态：主体（恒）+ 当前人格（可换）+ 当前意图（可换）。"""

    def __init__(self, subject=SUBJECT, persona=DEFAULT_PERSONA, intent=""):
        if subject != SUBJECT:
            raise ValueError(f"unknown subject {subject!r}")
        if persona not in PERSONAS:
            raise ValueError(f"unknown persona {persona!r}")
        if not isinstance(intent, str):
            raise ValueError("intent must be a string")
        self.subject = subject
        self.persona = persona
        self.intent = intent

    def set_persona(self, persona):
        if persona not in PERSONAS:
            raise ValueError(f"unknown persona {persona!r}")
        self.persona = persona

    def set_intent(self, intent):
        if not isinstance(intent, str):
            raise ValueError("intent must be a string")
        self.intent = intent

    def persona_model(self):
        """人格的模型内写法（词表可编码的中文名）。"""
        return PERSONA_MODEL[self.persona]

    def anchor_lines(self):
        """锚点行（模型内文本）。"""
        lines = f"我是{SUBJECT_MODEL}。当前人格{self.persona_model()}。"
        if self.intent:
            lines += f"我要{self.intent}。"
        return lines

    def to_dict(self):
        return {"subject": self.subject, "persona": self.persona,
                "intent": self.intent}

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict):
            raise ValueError("self state must be a dict")
        return cls(subject=data.get("subject", SUBJECT),
                   persona=data.get("persona", DEFAULT_PERSONA),
                   intent=data.get("intent", ""))


def anchor_prompt(self_state, state):
    """锚点行在前、状态摘要在后（换行分隔），再由调用方接问题。"""
    import world_state as ws
    return self_state.anchor_lines() + "\n" + state.summary()

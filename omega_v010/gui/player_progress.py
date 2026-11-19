#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""玩家进度（PlayerProgress）：flutter/games basic 模板的进度模式，
抄设计不抄代码。

Flutter 那边只记 highestLevelReached（最高到第几关，只进不退），
持久化抽象分 LocalStorage / Memory 两套实现。Omega 这边 7 个地点
就是 7 关，不用定取经路线顺序——记“到过几处”就行：

  - visited：本次取经到过的地点集合；集满 7 处算一次取经完成，
    pilgrimages +1，visited 重置为当前地点（新一轮取经开始）。
  - collected：累计拾到过的物品名集合（只进不退）。
  - pilgrimages：取经完成次数。

存 player_progress.json（和 dialog_log.jsonl 同目录）。path=None 时
是纯内存实现（测试用，不写盘），对应 Flutter 的 Memory 实现。
文件丢了、坏了都当新进度，不报错、不崩界面。
"""
from __future__ import annotations

import json
from pathlib import Path

TOTAL_LOCATIONS = 7
TOTAL_ITEMS = 6


class PlayerProgress:
    """只进不退的进度 + JSON 持久化。"""

    def __init__(self, path=None, locations=None):
        self.path = Path(path) if path is not None else None
        # locations：本轮要走遍的地点集合；不给则默认 7 处全算。
        self.locations = set(locations) if locations else None
        self.visited = set()
        self.collected = set()
        self.pilgrimages = 0
        self._load()

    # ---- 持久化 ----
    def _load(self):
        if self.path is None:
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return  # 没有文件 / 坏了都当新进度
        if not isinstance(data, dict):
            return
        self.visited = set(data.get("visited", []))
        self.collected = set(data.get("collected", []))
        try:
            self.pilgrimages = int(data.get("pilgrimages", 0))
        except Exception:
            self.pilgrimages = 0

    def _save(self):
        if self.path is None:
            return
        try:
            self.path.write_text(json.dumps({
                "visited": sorted(self.visited),
                "collected": sorted(self.collected),
                "pilgrimages": self.pilgrimages,
            }, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass  # 写盘失败不崩界面（只读目录等情况）

    def visit(self, location):
        """到过一处。返回 True 表示本轮刚集满、取经完成。"""
        self.visited.add(location)
        target = self.locations
        if target is not None:
            done = bool(target) and target <= self.visited
        else:
            # 没给定地点集合时按个数算：世界一共 7 处，到过 7 处就是走遍。
            done = len(self.visited) >= TOTAL_LOCATIONS
        if done:
            self.pilgrimages += 1
            self.visited = {location}
            self._save()
            return True
        self._save()
        return False

    def collect(self, item):
        """拾到一件。返回 True 表示是新物品（之前没拾过）。"""
        if item in self.collected:
            return False
        self.collected.add(item)
        self._save()
        return True

    # ---- 展示 ----
    def progress_text(self, total_locations=TOTAL_LOCATIONS,
                      total_items=TOTAL_ITEMS):
        """顶栏 pill 用：一行进度。"""
        text = (f"足迹 {len(self.visited)}/{total_locations}"
                f" · 拾取 {len(self.collected)}/{total_items}")
        if self.pilgrimages:
            text += f" · 取经 {self.pilgrimages} 次"
        return text

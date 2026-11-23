#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Omega 小世界抽卡：60 张卡池 + 开包经济（GACHA_DESIGN.md §7 最终版）。
纯标准库，不联网、不改训练、不动模型。

卡池 60 张（人物 30 / 法宝 15 / 地点 15），名字、简介取自小世界
已有的 world_state.py；稀有度按名字在 Xiyouji.txt 里的全名出现次数
取 log10 分档：>=2.9 SSR，2.0~2.89 SR，1.0~1.99 R，<1.0 C
（SSR 2 / SR 8 / R 29 / C 21）。攻 = min(1000, 100 + round(900 *
log10(1+count) / 3))；守 = 种类底分（地点 600 / 人物 400 / 法宝 500）
+ 稀有加成（SSR 300 / SR 200 / R 100 / C 0）。
包：1 包 5 张；开包概率 C 55% / R 28% / SR 13% / SSR 4%；
每 10 包保底 1 张 SR 以上（第 10 包第 5 张必 SR+）。
包的来源：每日登录 5 包（同日只领一次）+ 走路每 4 步 1 包（余数
攒着）+ 到新地点 1 包 + 取经完成 3 包，上限攒 10 包；起手 1 包。
不卖包、不接广告（离线买断制，卖包会破承诺）。
重复卡变 10 尘，尘目前没有花处，先只攒着。
持久化复用 player_progress.json 的做法：同目录一个 JSON
（gacha_progress.json），丢了、坏了都当新进度。
"""
from __future__ import annotations

import json
import math
import random
from datetime import date
from pathlib import Path

# 卡池：名字、种类、原文出现次数（2026-11-19 实数，见 GACHA_DESIGN.md）
# 注意：九齿钉耙 count=0 是真发现——原文全名 0 次，原文作“九齿钉钯”。
POOL = [
    ('孙悟空', '人物', 126, '石猴出世，大闹天宫，取经大师兄。'),
    ('唐僧', '人物', 1005, '金蝉子转世，西天取经的师父。'),
    ('猪八戒', '人物', 131, '天蓬下凡，好吃懒做，耙子在手。'),
    ('沙僧', '人物', 816, '卷帘大将，挑担牵马，老实本分。'),
    ('白龙马', '人物', 5, '西海龙王三太子，驮师父西行。'),
    ('观音', '人物', 115, '南海普陀落伽，慈悲救苦。'),
    ('如来', '人物', 213, '西天灵山佛祖，五指山镇住悟空。'),
    ('玉帝', '人物', 189, '天宫之主，统领三界。'),
    ('太上老君', '人物', 22, '兜率宫炼丹，八卦炉炼出火眼金睛。'),
    ('哪吒', '人物', 73, '托塔天王之子，脚踏风火轮。'),
    ('二郎神', '人物', 4, '灌江口显圣真君，三尖两刃刀。'),
    ('牛魔王', '人物', 39, '积雷山大力王，七十二变。'),
    ('铁扇公主', '人物', 6, '罗刹女，芭蕉扇能灭火焰山大火。'),
    ('红孩儿', '人物', 14, '牛魔王之子，号山枯松涧，三昧真火。'),
    ('白骨精', '人物', 2, '白虎岭白骨化形，三戏唐僧。'),
    ('金角大王', '人物', 2, '平顶山莲花洞，紫金红葫芦。'),
    ('银角大王', '人物', 2, '平顶山莲花洞，羊脂玉净瓶。'),
    ('黄风怪', '人物', 7, '黄风岭得道老妖，三昧神风。'),
    ('蜘蛛精', '人物', 7, '盘丝洞七姐妹，盘丝岭结网。'),
    ('镇元子', '人物', 13, '万寿山五庄观，地仙之祖，人参果树。'),
    ('菩提祖师', '人物', 2, '斜月三星洞，悟空的授业恩师。'),
    ('黑熊精', '人物', 2, '黑风山妖，偷了锦襕袈裟。'),
    ('玉兔', '人物', 24, '广寒宫捣药，思凡下界。'),
    ('嫦娥', '人物', 11, '月宫仙子，广寒宫主人。'),
    ('土地', '人物', 180, '一方土地公，报信指路。'),
    ('东海龙王', '人物', 3, '四海龙王之首，金箍棒原主。'),
    ('唐太宗', '人物', 11, '大唐天子，送唐僧西行。'),
    ('文殊', '人物', 14, '文殊菩萨，青狮坐骑。'),
    ('托塔天王', '人物', 13, '李靖，统兵捉拿悟空。'),
    ('弥勒佛', '人物', 16, '未来佛，黄眉老佛的师父。'),
    ('金箍棒', '法宝', 124, '定海神针，如意金箍棒。'),
    ('九齿钉耙', '法宝', 0, '八戒的兵器，太上老君亲造。'),
    ('月牙铲', '法宝', 5, '沙僧的兵器，降妖宝杖。'),
    ('锦襕袈裟', '法宝', 11, '观音所赐，唐僧的袈裟。'),
    ('紫金钵盂', '法宝', 6, '唐僧的化缘钵盂。'),
    ('蟠桃', '法宝', 56, '王母蟠桃园，三千年一熟。'),
    ('芭蕉扇', '法宝', 35, '铁扇公主的宝扇，一扇熄火。'),
    ('金刚琢', '法宝', 3, '太上老君的钢圈，能套万物。'),
    ('幌金绳', '法宝', 11, '金角银角的捆仙绳。'),
    ('照妖镜', '法宝', 7, '照出妖怪原形的宝镜。'),
    ('紫金红葫芦', '法宝', 3, '金角大王的宝贝，装天收人。'),
    ('羊脂玉净瓶', '法宝', 2, '银角大王的宝贝，收人化水。'),
    ('七星剑', '法宝', 11, '宝剑七星，辟邪斩妖。'),
    ('人参果', '法宝', 33, '万寿山五庄观，三千年一开花。'),
    ('紧箍儿', '法宝', 29, '观音所赐，念紧箍咒便收紧。'),
    ('花果山', '地点', 79, '猴王故地，水帘洞天。'),
    ('水帘洞', '地点', 52, '花果山中洞府，瀑布遮门。'),
    ('高老庄', '地点', 12, '庄户人家，八戒旧居。'),
    ('流沙河', '地点', 28, '河水汹涌，沙僧旧地。'),
    ('取经路', '地点', 8, '西行大道，师徒同行。'),
    ('南海', '地点', 63, '观音道场，普陀落伽。'),
    ('天宫', '地点', 116, '玉皇居所，云雾缭绕。'),
    ('五指山', '地点', 11, '如来镇压悟空五百年的山。'),
    ('火焰山', '地点', 27, '烈火熊熊，芭蕉扇方能熄灭。'),
    ('盘丝洞', '地点', 7, '七个蜘蛛精的洞府。'),
    ('狮驼岭', '地点', 3, '狮驼三妖，八百里狮驼城。'),
    ('通天河', '地点', 17, '河面结冰，老鼋驮师徒过河。'),
    ('雷音寺', '地点', 37, '西天灵山，大雷音寺。'),
    ('天竺国', '地点', 34, '西行终点附近的国度。'),
    ('灵山', '地点', 70, '西天佛地，取经终点。'),
]

RARITY_BONUS = {"SSR": 300, "SR": 200, "R": 100, "C": 0}
KIND_DEF = {"地点": 600, "人物": 400, "法宝": 500}
PACK_ODDS = (("C", 0.55), ("R", 0.28), ("SR", 0.13), ("SSR", 0.04))
MAX_PACKS = 10
PITY_EVERY = 10
DAILY_PACKS = 5
PILGRIMAGE_PACKS = 3
STEPS_PER_PACK = 4


def rarity_of(count):
    v = math.log10(1 + count)
    if v >= 2.9:
        return "SSR"
    if v >= 2.0:
        return "SR"
    if v >= 1.0:
        return "R"
    return "C"


def atk_of(count):
    return min(1000, 100 + round(900 * math.log10(1 + count) / 3))


def card(name):
    for n, kind, count, flavor in POOL:
        if n == name:
            r = rarity_of(count)
            return {"name": n, "kind": kind, "rarity": r,
                    "atk": atk_of(count),
                    "def": KIND_DEF[kind] + RARITY_BONUS[r],
                    "flavor": flavor, "mentions": count}
    raise ValueError("unknown card %r" % (name,))


CARDS = [card(n) for n, _, _, _ in POOL]
BY_RARITY = {}
for c in CARDS:
    BY_RARITY.setdefault(c["rarity"], []).append(c)


def open_pack(rng, pity=False):
    """开 1 包 5 张。pity=True 时第 5 张必为 SR 以上。"""
    out = []
    for i in range(5):
        if pity and i == 4:
            pool = BY_RARITY["SR"] + BY_RARITY["SSR"]
            out.append(rng.choice(pool))
            continue
        x = rng.random()
        acc = 0.0
        for r, p in PACK_ODDS:
            acc += p
            if x < acc:
                out.append(rng.choice(BY_RARITY[r]))
                break
    return out


def share_text(c):
    return ("【%s】%s（%s） 攻 %d 守 %d —— %s\nOmega 小世界卡册：60 张已集 %s/60"
            % (c["rarity"], c["name"], c["kind"], c["atk"], c["def"],
               c["flavor"], c.get("_owned", "?")))


class Collection:
    """卡册持久化：cards（卡名->张数）、packs、dust（重复卡化成的尘）、
    packs_opened、last_daily（上次领每日包的日期）。"""

    def __init__(self, path=None):
        self.path = Path(path) if path is not None else None
        self.cards = {}
        self.packs = 1
        self.dust = 0
        self.packs_opened = 0
        self.last_daily = ""
        self.steps_bank = 0
        self._load()

    def _load(self):
        if self.path is None:
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return
        if not isinstance(data, dict):
            return
        self.cards = {k: int(v) for k, v in data.get("cards", {}).items()}
        self.packs = int(data.get("packs", 1))
        self.dust = int(data.get("dust", 0))
        self.packs_opened = int(data.get("packs_opened", 0))
        self.last_daily = str(data.get("last_daily", ""))
        self.steps_bank = int(data.get("steps_bank", 0))

    def _save(self):
        if self.path is None:
            return
        try:
            self.path.write_text(json.dumps({
                "cards": self.cards, "packs": self.packs, "dust": self.dust,
                "packs_opened": self.packs_opened,
                "last_daily": self.last_daily,
                "steps_bank": self.steps_bank,
            }, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    def owned_count(self):
        return len(self.cards)

    def _grant(self, n):
        """发包：不超过上限 10 包，返回实际发出的包数。"""
        room = MAX_PACKS - self.packs
        got = max(0, min(n, room))
        self.packs += got
        return got

    def daily_pack(self, today=None):
        """每日登录：同一天只领一次，给 5 包（上限 10）。返回实际领到的包数。"""
        today = (today or date.today()).isoformat()
        if self.last_daily == today:
            return 0
        self.last_daily = today
        got = self._grant(DAILY_PACKS)
        self._save()
        return got

    def earn_for_steps(self, steps):
        """走路：每 4 步得 1 包，未满 4 步的余数攒着。返回新得的包数。"""
        self.steps_bank += int(steps)
        n, self.steps_bank = divmod(self.steps_bank, STEPS_PER_PACK)
        got = self._grant(n) if n else 0
        self._save()
        return got

    def earn_for_location(self):
        """到新地点：1 包。返回实际得包数。"""
        got = self._grant(1)
        self._save()
        return got

    def earn_for_pilgrimage(self):
        """取经完成（7 地走遍）：3 包。返回实际得包数。"""
        got = self._grant(PILGRIMAGE_PACKS)
        self._save()
        return got


    def open_one(self, rng):
        """开 1 包：无包返回 None；重复卡变 10 尘；每 10 包保底 SR+。"""
        if self.packs <= 0:
            return None
        self.packs -= 1
        self.packs_opened += 1
        pity = (self.packs_opened % PITY_EVERY == 0)
        drawn = open_pack(rng, pity=pity)
        new = []
        for c in drawn:
            if c["name"] in self.cards:
                self.dust += 10
            else:
                new.append(c["name"])
            self.cards[c["name"]] = self.cards.get(c["name"], 0) + 1
        self._save()
        return {"drawn": drawn, "new": new, "pity": pity}


if __name__ == "__main__":
    from collections import Counter
    print("卡池 60 张：", Counter(c["kind"] for c in CARDS),
          "稀有：", Counter(c["rarity"] for c in CARDS))
    rng = random.Random(20261119)
    col = Collection()
    print("起手 1 包 + 每日 5 包 + 走 12 步得 3 包：",
          "每日", col.daily_pack(date(2026, 11, 19)),
          "走路", col.earn_for_steps(12), "余包", col.packs)
    for i in range(9):
        r = col.open_one(rng)
        names = "、".join("%s%s" % (c["rarity"], c["name"]) for c in r["drawn"])
        print("  包%d%s：%s 新卡：%s" % (
            i + 1, "（保底）" if r["pity"] else "", names, r["new"] or "无"))
    print("集齐 %d/60，尘 %d，余包 %d" % (col.owned_count(), col.dust, col.packs))
    print("分享文案示例：", share_text(dict(CARDS[0], _owned=col.owned_count())))
    days = []
    for seed in range(2000):
        r2 = random.Random(seed)
        c2 = Collection()
        d = 0
        while c2.owned_count() < 60:
            d += 1
            c2.packs = min(MAX_PACKS, c2.packs + 8)
            while c2.packs > 0 and c2.owned_count() < 60:
                c2.open_one(r2)
        days.append(d)
    days.sort()
    print("蒙特卡洛 2000 次（每天登录 5 包 + 走 12 步 3 包 = 8 包/天）："
          "集齐中位 %d 天，p10 %d 天，p90 %d 天" % (days[1000], days[200], days[1800]))

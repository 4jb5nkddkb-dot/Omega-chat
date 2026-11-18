#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Omega 小世界 GUI v0.1.2 — Windows 可玩版。

纯标准库 tkinter + NumPy，双击 run_gui.bat 运行；启动时选一个模型
产品 .npz（全能线任意一块都行，Drive 备份里的 s57300 就行）。

能玩什么：
  - 走动：7 个地点按钮，人物会跟你走，昼夜每 4 步一翻。
  - 查看 / 拾取 / 对话：可见物品和在场人物都是按钮，点就行。
  - 问问题：4 道自态问题按钮，或自己打字问。
  - 自动玩：按“自动玩 12 步”，桥接器把每步的接地分交给内核，内核逐
    步选出下一个动作（observe / explore / exploit / stabilize），不
    再固定轮换。需要界面旁有桥接器（omega-world-bridge）；没有找到
    时自动玩会明说，不会悄悄轮换。桥接器的配置（omega.toml）保持
    self_modify 关闭：内核只选动作，不自己改策略。
  - 接地计分：每句模型回话都当场判对错（和世界状态对不对得上），
    答对数实时显示；对话默认存进 dialog_log.jsonl（可关）。

人格：启动时默认 omega；下拉可换紫薇 / 悟空 / 一一 / Dayan，
意图框里写一句想干的事（比如“去花果山找金箍棒”），点应用。
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

import agent_runtime as ar
import kernel_autoplay as ka
import world_anchor as wa
import world_loop as wl
import world_state as ws
import world_self_probe as wsp

PERSONAS = ["omega", "紫薇", "悟空", "一一", "Dayan"]
LOCATIONS = ["花果山", "水帘洞", "高老庄", "流沙河", "取经路", "南海", "天宫"]
AUTO_STEPS = 12

# ---- 外观（对齐设计稿 ~/workspace/user/files/omega_ui.html；只改表现层）----
# token 取自稿子：底色 #03050a / #010206，青 #7ff6ff / #38d6e8，
# 橙 #ffb35c / #ff7a3d，文字 #d9f6ff，等宽字体做标签。tkinter 没有
# 真圆角/发光/渐变：圆角用 Frame 描边 + Canvas 画，渐变用纯色近似，
# 发光用描边色 + 浅色字模拟。纯标准库，不引新依赖。
BG = "#03050a"        # 页面底
BG_PANEL = "#05060e"  # 星图 / 终端底
BG_CARD = "#0a1220"   # 按钮 / 卡片底
BG_INPUT = "#03060c"  # 输入框底
BG_GLOW = "#2a1a06"   # 星图中心暖光
CYAN = "#7ff6ff"      # 主青
CYAN_LINE = "#38d6e8" # 描边青
CYAN_DIM = "#6fa8ba"  # 暗青（次要文字）
AMBER = "#ffb35c"     # 主橙
ORANGE = "#ff7a3d"    # 橙（自动运行按钮）
TEXT = "#d9f6ff"      # 主文字
TEXT_SOFT = "#b5e6f2" # 终端文字
GREEN = "#6dffb0"     # 答对
RED = "#ff6b86"       # 答错
BORDER = "#1f4a55"    # 深青描边
FONT_UI = ("Microsoft YaHei", 10)
FONT_MONO = ("Consolas", 10)
FONT_MONO_BIG = ("Consolas", 12, "bold")
# 星图地点英文副标（稿子里的 Pilgrim Road 一类，只做展示）
LOC_EN = {"花果山": "FLOWER FRUIT MT", "水帘洞": "WATER CURTAIN",
          "高老庄": "GAO VILLAGE", "流沙河": "SAND RIVER",
          "取经路": "PILGRIM ROAD", "南海": "SOUTH SEA", "天宫": "HEAVEN"}


class Game:
    """不依赖界面的游戏逻辑：世界 + 模型 + 接地计分 + 存档。"""

    def __init__(self, ckpt):
        self.ckpt = str(ckpt)
        self.model, self.tok, self.step, self.arch = wl.load_world_model(self.ckpt)
        self.state = ws.WorldState()
        self.me = wa.SelfState()
        self.qi = 0
        self.loc_idx = 4  # 取经路在轮换表里的位置
        self.correct = 0
        self.total = 0
        self.last_reply = ""  # 上一步模型回话（内核自动玩的观察用）
        self.last_err = 0.0  # 上一步接地分（内核自动玩的观察用）
        self.archive_path = HERE / "dialog_log.jsonl"
        self.archive_on = True
        self.archive_warned = False
        self.memory = ar.CognitiveMemoryBuffer(capacity=3)
        self._last_turn = None  # 最近一次模型调用的输入（供 record 写档用）

    def generate(self, prompt, max_new):
        # 在线记忆防复读：把缓存的答对示例拼在提示最前面，再送入模型。
        # 只拼与当前状态相同的示例——状态变了，旧答案就是错答案。
        # 空缓存（或同状态无示例）时 format_examples() 返回空串，提示
        # 与原来一字不差。
        prefix = self.memory.format_examples(state=self.state.summary())
        full = prefix + prompt
        self._last_turn = {"prompt": full, "base_prompt": prompt,
                           "memory_examples": self.memory.examples()}
        ids = self.tok.encode(full, add_bos=True)
        try:
            out = self.model.generate_cached(ids, max_new, temperature=0.0)
        except Exception:
            out = self.model.generate(ids, max_new_tokens=max_new,
                                      temperature=0.0)
        return self.tok.decode(out[len(ids):])

    def _state_snapshot(self):
        """当前真实标准状态（写档用：判卷的 ground truth，不是模型说的）。"""
        st = self.state
        return {"location": st.visitor_location,
                "time_step": st.time_step,
                "day_night": st.day_night(),
                "present_characters": st.present_characters(),
                "present_items": st.present_items(),
                "carried_items": st.carried_items(),
                "summary": st.summary()}

    def _remember(self, prompt_text, reply):
        """答对才记：把（当前真实状态摘要，提问/动作，正确回复）写进记忆缓存。"""
        if reply:
            self.memory.add_success(self.state.summary(), prompt_text, reply)

    def anchor_prompt(self, user_text):
        return wa.anchor_prompt(self.me, self.state) + "\n" + user_text

    def ask(self, qi):
        q = wsp.QUESTIONS[qi]
        cont = self.generate(ws.world_prompt(self.state, q), 32)
        ok, note = wsp.score(self.state, qi, cont)
        if ok:
            self._remember(q, cont)
        return cont, (0.0 if ok else 1.0), note

    def ask_free(self, text):
        cont = self.generate(ws.world_prompt(self.state, text), 32)
        flags = self.state.validate(cont)
        if not flags:
            self._remember(text, cont)
        return cont, (1.0 if flags else 0.0), "; ".join(flags)

    def move(self, dest):
        res = self.state.move(dest)
        cont = self.generate(self.anchor_prompt(f"你来到了{dest}，说说这里。"), 16)
        flags = self.state.validate(cont)
        if not flags:
            self._remember(f"你来到了{dest}，说说这里。", cont)
        return res + cont, (1.0 if flags else 0.0), "; ".join(flags)

    def look(self):
        return self.state.look(""), 0.0, ""

    def take(self, item):
        res = self.state.take(item)
        return res, (0.0 if "拾起了" in res else 1.0), ""

    def talk(self, name):
        cont = self.generate(self.anchor_prompt(f"和{name}说说话。"), 24)
        flags = self.state.validate(cont)
        if not flags:
            self._remember(f"和{name}说说话。", cont)
        return cont, (1.0 if flags else 0.0), "; ".join(flags)

    def observation(self, step):
        """当前世界快照，写给桥接器看：和 v0.1.0 sidecar 的 observation
        同格式，内核据此选下一步动作。"""
        return {"step": step,
                "location": self.state.visitor_location,
                "time_step": self.state.time_step,
                "daynight": self.state.day_night(),
                "present": self.state.present_characters(),
                "visible": self.state.present_items(),
                "carried": self.state.carried_items(),
                "persona": self.me.persona if hasattr(self.me, "persona") else "omega",
                "intent": getattr(self.me, "intent", ""),
                "last_reply": self.last_reply,
                "grounded_error": self.last_err,
                "world_summary": self.state.summary()}

    def run_kernel_action(self, action):
        """执行内核选出的一个动作；返回 (label, reply, err, note)。

        每个动作的做法和旧版固定轮换那一支一字不差，唯一的区别是动作
        来自内核，不再按固定顺序轮换。白名单外的动作一律按 noop 处理：
        什么都不做，接地分记 0。"""
        if action in ("observe", "stabilize"):
            qi = self.qi % len(wsp.QUESTIONS)
            self.qi += 1
            reply, err, note = self.ask(qi)
            return f"[{action}] {wsp.QUESTIONS[qi]}", reply, err, note
        if action == "explore":
            cur = self.state.visitor_location
            for _ in range(len(LOCATIONS)):
                self.loc_idx = (self.loc_idx + 1) % len(LOCATIONS)
                dest = LOCATIONS[self.loc_idx]
                if dest != cur:
                    break
            reply, err, note = self.move(dest)
            return f"[explore] 去{dest}", reply, err, note
        if action == "exploit":
            visible = self.state.present_items()
            if visible:
                reply, err, note = self.take(visible[0])
                return f"[exploit] 拾取{visible[0]}", reply, err, note
            present = self.state.present_characters()
            if present:
                reply, err, note = self.talk(present[0])
                return f"[exploit] 和{present[0]}说话", reply, err, note
            reply, err, note = self.ask(1)
            return "[exploit] 问在场人物", reply, err, note
        return "[noop]", "", 0.0, "noop"

    def record(self, action, reply, err, note=""):
        self.total += 1
        if err == 0.0:
            self.correct += 1
        # 先取走本轮输入再判断是否存档：查看/拾取不调模型，_last_turn 为空，
        # 不会把上一轮的输入错贴到这一条上；存档关掉时也不留陈旧输入。
        last = self._last_turn
        self._last_turn = None
        if not self.archive_on:
            return
        rec = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"),
               "action": action, "reply": reply,
               "grounded_error": err, "note": note,
               "verdict": "correct" if err == 0.0 else "wrong",
               "state": self._state_snapshot()}
        if last is not None:
            # prompt 是实际送入模型的完整输入（含记忆示例）；base_prompt
            # 是去掉示例后的原始提示；memory_examples 是当轮生效的示例。
            # 微调导出时用 base_prompt + reply + state 拼对齐语料。
            rec["prompt"] = last["prompt"]
            rec["base_prompt"] = last["base_prompt"]
            rec["memory_examples"] = last["memory_examples"]
        try:
            with open(self.archive_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception as e:
            if not self.archive_warned:
                self.archive_warned = True
                print(f"存档写失败，已暂停存档：{e}", flush=True)
            self.archive_on = False


class App:
    def __init__(self, root, ckpt=None):
        self.root = root
        self.root.title("Omega 小世界 v0.1.2")
        self.root.geometry("780x880")
        self.root.configure(background=BG)
        try:
            import tkinter.font as tkfont
            tkfont.nametofont("TkDefaultFont").configure(family="Microsoft YaHei", size=10)
            tkfont.nametofont("TkTextFont").configure(family="Microsoft YaHei", size=10)
            tkfont.nametofont("TkFixedFont").configure(family="Consolas", size=10)
        except Exception:
            pass
        self._apply_theme()
        self.game = None
        self.busy = False
        self.auto_running = False
        self.auto_left = 0
        self.driver = None  # 内核自动玩的桥接器驱动（未启动时为空）
        self.move_buttons = {}  # 地点 -> 走动按钮（当前位置高亮用）
        self._build()
        self._set_status("正在准备…")
        self.root.after(200, lambda: self._startup(ckpt))

    def _apply_theme(self):
        """clam 主题 + 稿子配色：深底、青描边、橙点缀。失败时静默回退
        系统默认主题（功能不受影响）。"""
        try:
            style = ttk.Style(self.root)
            style.theme_use("clam")
            style.configure("TFrame", background=BG)
            style.configure("TLabel", background=BG, foreground=TEXT,
                            font=FONT_UI)
            style.configure("TLabelframe", background=BG, foreground=CYAN,
                            font=FONT_MONO, bordercolor=BORDER)
            style.configure("TLabelframe.Label", background=BG,
                            foreground=CYAN, font=FONT_MONO)
            style.configure("TButton", background=BG_CARD, foreground=TEXT,
                            font=FONT_UI, bordercolor=BORDER,
                            focuscolor=CYAN, lightcolor=BG_CARD,
                            darkcolor=BG_CARD)
            style.map("TButton",
                      background=[("active", "#12233a"),
                                  ("disabled", BG_CARD)],
                      foreground=[("disabled", CYAN_DIM)])
            style.configure("Current.TButton", background="#12233a",
                            foreground=CYAN, bordercolor=CYAN)
            style.configure("TEntry", fieldbackground=BG_INPUT,
                            foreground=TEXT, bordercolor=BORDER,
                            insertcolor=CYAN)
            style.configure("TCombobox", fieldbackground=BG_INPUT,
                            foreground=TEXT, bordercolor=BORDER,
                            arrowcolor=CYAN)
            style.map("TCombobox",
                      fieldbackground=[("readonly", BG_INPUT)])
            style.configure("TCheckbutton", background=BG,
                            foreground=TEXT_SOFT)
            style.configure("TNotebook", background=BG,
                            bordercolor=BORDER)
            style.configure("TNotebook.Tab", background=BG_CARD,
                            foreground=CYAN_DIM, font=FONT_UI,
                            padding=(12, 6), bordercolor=BORDER)
            style.map("TNotebook.Tab",
                      background=[("selected", "#12233a")],
                      foreground=[("selected", CYAN)])
        except Exception:
            pass

    def _build(self):
        # ---- 顶栏两行：第一行 logo + 人格 / 意图，第二行状态 pill ----
        top = ttk.Frame(self.root)
        top.pack(fill="x", padx=8, pady=(6, 2))
        logo = tk.Frame(top, bg=BG)
        logo.pack(side="left", padx=(0, 8))
        tk.Label(logo, text="OMEGA // 小世界", font=FONT_MONO_BIG,
                 fg=CYAN, bg=BG).pack(anchor="w")
        tk.Label(logo, text="WORLD OF OMEGA", font=("Consolas", 8),
                 fg=CYAN_DIM, bg=BG).pack(anchor="w")
        ttk.Label(top, text="人格：").pack(side="left")
        self.persona_var = tk.StringVar(value="omega")
        self.persona_cb = ttk.Combobox(top, textvariable=self.persona_var,
                                       values=PERSONAS, width=8, state="readonly")
        self.persona_cb.pack(side="left", padx=4)
        ttk.Label(top, text="我想：").pack(side="left", padx=(8, 0))
        self.intent_var = tk.StringVar()
        ttk.Entry(top, textvariable=self.intent_var, width=20).pack(side="left", padx=4)
        ttk.Button(top, text="应用", command=self._apply_self).pack(side="left", padx=4)
        self.archive_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text="记录对话", variable=self.archive_var,
                        command=self._toggle_archive).pack(side="left", padx=6)
        pills = ttk.Frame(self.root)
        pills.pack(fill="x", padx=8, pady=(0, 4))
        self.step_var = tk.StringVar(value="第 0 步 STEP 0")
        self._pill(pills, self.step_var, CYAN).pack(side="left", padx=(0, 6))
        self.day_var = tk.StringVar(value="白天 DAY")
        self._pill(pills, self.day_var, AMBER).pack(side="left", padx=3)
        self.score_var = tk.StringVar(value="答对 0 / 0")
        self._pill(pills, self.score_var, CYAN).pack(side="left", padx=3)

        wf = ttk.LabelFrame(self.root, text="世界状态")
        wf.pack(fill="x", padx=8, pady=4)
        # ---- 星图：稿子里的轨道 + 中心核 + 角色节点 + 位置标题 + 物品角标 ----
        self.map_canvas = tk.Canvas(wf, height=220, bg=BG_PANEL,
                                    highlightthickness=1,
                                    highlightbackground=BORDER)
        self.map_canvas.pack(fill="x", padx=6, pady=(6, 2))
        self.map_canvas.bind("<Configure>", lambda e: self._draw_map())
        self.world_vars = {}
        for key in ("地点", "时间", "在场人物", "可见物品", "随身物品", "自我"):
            row = ttk.Frame(wf)
            row.pack(fill="x", padx=6, pady=1)
            ttk.Label(row, text=key + "：", width=8).pack(side="left")
            v = tk.StringVar(value="—")
            ttk.Label(row, textvariable=v).pack(side="left")
            self.world_vars[key] = v
        self._draw_map()

        # ---- 终端：稿子里的 // TERMINAL 框，等宽字体，发言按角色着色 ----
        self.log = scrolledtext.ScrolledText(self.root, height=12, state="disabled",
                                             wrap="word", bg=BG_INPUT, fg=TEXT_SOFT,
                                             font=FONT_MONO, insertbackground=CYAN,
                                             selectbackground="#12233a",
                                             relief="flat", borderwidth=1,
                                             highlightbackground=BORDER,
                                             highlightthickness=1)
        self.log.pack(fill="both", expand=True, padx=8, pady=4)
        self.log.tag_configure("you", foreground=CYAN)
        self.log.tag_configure("omega", foreground=AMBER)
        self.log.tag_configure("ok", foreground=GREEN)
        self.log.tag_configure("bad", foreground=RED)
        self.log.tag_configure("note", foreground=CYAN_DIM)

        # ---- 三页签：稿子里的走动 / 动作 / 问问题分段控件 ----
        nb = ttk.Notebook(self.root)
        nb.pack(fill="x", padx=8, pady=4)
        mf = ttk.Frame(nb)
        nb.add(mf, text="走动 MOVE")
        af = ttk.Frame(nb)
        nb.add(af, text="动作 ACTION")
        qf = ttk.Frame(nb)
        nb.add(qf, text="问问题 ASK")

        for loc in LOCATIONS:
            b = ttk.Button(mf, text=loc, width=8,
                           command=lambda d=loc: self._do(f"去{loc}", lambda: self.game.move(d)))
            b.pack(side="left", padx=2, pady=4)
            self.move_buttons[loc] = b

        ttk.Button(af, text="查看", width=8,
                   command=lambda: self._do("查看", self.game.look)).pack(side="left", padx=2, pady=4)
        self.take_frame = ttk.Frame(af)
        self.take_frame.pack(side="left", padx=6, pady=4)
        self.talk_frame = ttk.Frame(af)
        self.talk_frame.pack(side="left", padx=6, pady=4)

        for qi, q in enumerate(wsp.QUESTIONS):
            ttk.Button(qf, text=q, width=16,
                       command=lambda i=qi: self._do(f"问：{wsp.QUESTIONS[i]}",
                                                      lambda: self.game.ask(i))).pack(side="left", padx=2, pady=4)
        self.free_var = tk.StringVar()
        ttk.Entry(qf, textvariable=self.free_var, width=24).pack(side="left", padx=4, pady=4)
        ttk.Button(qf, text="发送", command=self._ask_free).pack(side="left", pady=4)

        bf = ttk.Frame(self.root)
        bf.pack(fill="x", padx=8, pady=6)
        # ---- 自动运行：稿子里的橙色渐变大按钮（tkinter 无渐变，用纯色近似）----
        self.auto_btn = tk.Button(bf, text=">> 内核自动运行 · AUTO-RUN 12 STEPS",
                                  command=self._auto_start,
                                  bg=ORANGE, fg="#1a0d02",
                                  activebackground=AMBER,
                                  activeforeground="#1a0d02",
                                  disabledforeground=CYAN_DIM,
                                  font=("Consolas", 11, "bold"),
                                  relief="flat", padx=16, pady=6)
        self.auto_btn.pack(side="left", padx=2)
        self.stop_btn = tk.Button(bf, text="停", command=self._auto_stop,
                                  state="disabled",
                                  bg=BG_CARD, fg=TEXT,
                                  activebackground="#12233a",
                                  disabledforeground=CYAN_DIM,
                                  font=FONT_UI, relief="flat", padx=12, pady=6)
        self.stop_btn.pack(side="left", padx=2)
        self.auto_var = tk.StringVar(value="")
        tk.Label(bf, textvariable=self.auto_var, font=FONT_MONO,
                 fg=CYAN_DIM, bg=BG).pack(side="left", padx=8)

        self.status_var = tk.StringVar(value="")
        tk.Label(self.root, textvariable=self.status_var, font=FONT_MONO,
                 fg=CYAN_DIM, bg=BG_PANEL, anchor="w").pack(fill="x", padx=8, pady=(0, 6))
        self._set_enabled(False)

    @staticmethod
    def _pill(parent, textvar, color):
        """稿子里的状态 pill：深底 + 彩色细框 + 等宽字。"""
        return tk.Label(parent, textvariable=textvar, font=FONT_MONO,
                        fg=color, bg=BG_CARD, relief="solid",
                        borderwidth=1, padx=8, pady=2)

    def _set_enabled(self, on):
        for w in (self.persona_cb, self.auto_btn):
            try:
                w.configure(state="normal" if on else "disabled")
            except Exception:
                pass

    def _set_status(self, text):
        self.status_var.set(text)
        self.root.update_idletasks()

    def _startup(self, ckpt):
        if ckpt is None:
            ckpt = filedialog.askopenfilename(
                title="选一个模型产品（.npz）",
                initialdir=str(HERE),
                filetypes=[("NumPy 模型", "*.npz"), ("全部文件", "*.*")])
            if not ckpt:
                messagebox.showinfo("Omega 小世界", "没有选模型，退出。")
                self.root.destroy()
                return
        self._set_status(f"正在加载模型 {Path(ckpt).name}…")
        try:
            self.game = Game(ckpt)
        except Exception as e:
            messagebox.showerror("加载失败", f"模型加载失败：\n{e}")
            self.root.destroy()
            return
        self._set_enabled(True)
        self._set_status(f"模型 {Path(ckpt).name}（{self.game.step} 步）已加载，开始玩吧。")
        self._log(f"欢迎来到小世界。你在{self.game.state.visitor_location}，"
                  f"{self.game.state.day_night()}。")
        self._refresh_world()

    def _apply_self(self):
        if not self.game:
            return
        self.game.me.set_persona(self.persona_var.get())
        self.game.me.set_intent(self.intent_var.get().strip())
        self._log(f"你现在是{self.game.me.persona}。想：{self.intent_var.get().strip() or '（无）'}")
        self._refresh_world()

    def _toggle_archive(self):
        if self.game:
            self.game.archive_on = self.archive_var.get()

    def _refresh_world(self):
        if not self.game:
            return
        st = self.game.state
        join = lambda xs: "、".join(xs) if xs else "无"
        self.world_vars["地点"].set(st.visitor_location)
        self.world_vars["时间"].set(f"第 {st.time_step} 步 {st.day_night()}")
        self.world_vars["在场人物"].set(join(st.present_characters()))
        self.world_vars["可见物品"].set(join(st.present_items()))
        self.world_vars["随身物品"].set(join(st.carried_items()))
        self.world_vars["自我"].set(self.game.me.anchor_lines())
        self.score_var.set(f"答对 {self.game.correct} / {self.game.total}")
        self.step_var.set(f"第 {st.time_step} 步 STEP {st.time_step}")
        self.day_var.set(f"{st.day_night()} "
                         f"{'NIGHT' if st.day_night() == '夜晚' else 'DAY'}")
        # 当前位置的走动按钮高亮（稿子里的 YOU ARE HERE）
        for loc, b in self.move_buttons.items():
            try:
                b.configure(style="Current.TButton"
                            if loc == st.visitor_location else "TButton")
            except Exception:
                pass
        self._draw_map()
        for frame, items, verb, fn in (
                (self.take_frame, st.present_items(), "拾取", self.game.take),
                (self.talk_frame, st.present_characters(), "对话", self.game.talk)):
            for w in frame.winfo_children():
                w.destroy()
            if not items:
                ttk.Label(frame, text=f"{verb}：无").pack(side="left")
            for name in items:
                ttk.Button(frame, text=f"{verb}{name}", width=12,
                           command=lambda n=name, f=fn, v=verb: self._do(
                               f"{v}{n}", lambda: f(n))).pack(side="left", padx=2)

    def _log(self, text):
        """终端着色：你（青）、欧米伽（橙）、答对（绿）、答错（红），
        其余默认终端色。签名与旧版一致，仍是一次一行纯文本。"""
        self.log.configure(state="normal")
        for line in str(text).split("\n"):
            if line.startswith("你："):
                self.log.insert("end", "你：", "you")
                self._insert_verdict(line[2:])
            elif line.startswith("欧米伽："):
                self.log.insert("end", "欧米伽：", "omega")
                self._insert_verdict(line[4:])
            elif line:
                self._insert_verdict(line)
            self.log.insert("end", "\n")
        self.log.configure(state="disabled")
        self.log.see("end")

    def _insert_verdict(self, text):
        """把一行里的 [✓ 答对] / [✗ 答错] 标绿 / 标红，括号备注标暗青。"""
        import re
        for part in re.split(r"(\[✓ 答对\]|\[✗ 答错\])", text):
            if part == "[✓ 答对]":
                self.log.insert("end", part, "ok")
            elif part == "[✗ 答错]":
                self.log.insert("end", part, "bad")
            elif part.startswith("（") and part.endswith("）"):
                self.log.insert("end", part, "note")
            elif part:
                self.log.insert("end", part)

    def _draw_map(self):
        """画稿子里的星图：星星 + 暖光 + 双轨道 + 中心核 + 角色节点
        + 左上地点标题 + 右下可拾取物品角标。没加载模型时只画底图
        （轨道、中心核、占位标题），不读游戏状态。"""
        cv = getattr(self, "map_canvas", None)
        if cv is None:
            return
        try:
            cv.delete("all")
            w = cv.winfo_width() or 748
            h = 220
            cx, cy = w // 2, h // 2
            import math
            import random
            # 星星（固定种子，位置不跳）
            rng = random.Random(42)
            for _ in range(14):
                x = rng.randint(20, max(21, w - 20))
                y = rng.randint(12, h - 12)
                r = rng.choice((1, 2))
                cv.create_oval(x - r, y - r, x + r, y + r,
                               fill="#d9f6ff", outline="")
            # 中心暖光：三层同心圆由深到浅模拟径向光晕
            for rx, ry, col in ((150, 88, "#141005"), (115, 68, "#1d1507"),
                                (85, 50, BG_GLOW)):
                cv.create_oval(cx - rx, cy - ry, cx + rx, cy + ry,
                               fill=col, outline="")
            # 双轨道（虚线圆）
            cv.create_oval(cx - 175, cy - 88, cx + 175, cy + 88,
                           outline="#6b4a22", dash=(5, 4), width=1)
            cv.create_oval(cx - 112, cy - 56, cx + 112, cy + 56,
                           outline="#2a5a66", dash=(5, 4), width=1)
            # 四角定位框（稿子里的 corner bracket）
            for x0, y0, dx, dy in ((8, 8, 1, 1), (w - 8, 8, -1, 1),
                                   (8, h - 8, 1, -1), (w - 8, h - 8, -1, -1)):
                cv.create_line(x0, y0 + 14 * dy, x0, y0, x0 + 14 * dx, y0,
                               fill=CYAN, width=2)
            # 中心核：橙色圆 + 亮心 + 人名
            cv.create_oval(cx - 34, cy - 34, cx + 34, cy + 34,
                           fill=AMBER, outline="#fff1d6", width=1)
            cv.create_oval(cx - 20, cy - 20, cx + 20, cy + 20,
                           fill="#fff1d6", outline="")
            cv.create_text(cx, cy - 4, text="欧米伽", font=("Microsoft YaHei", 11, "bold"),
                           fill="#1a0d02")
            cv.create_text(cx, cy + 12, text="OMEGA · YOU",
                           font=("Consolas", 7), fill="#3a1a04")
            # 状态（有模型才读；没模型画占位）
            loc, step, chars, items = "—", 0, [], []
            if getattr(self, "game", None) is not None:
                st = self.game.state
                loc = st.visitor_location
                step = st.time_step
                chars = st.present_characters()
                items = st.present_items()
            # 在场人物 → 内轨道节点（青框圆，字取名字首字）；随步数缓慢转
            n = len(chars)
            for i, name in enumerate(chars):
                ang = math.radians(200 + i * (360.0 / max(n, 1)) + step * 10)
                x = cx + 112 * math.cos(ang)
                y = cy + 56 * math.sin(ang)
                self._draw_node(x, y, name[0], CYAN)
            # 首件可见物品 → 外轨道橙节点（稿子里的龙节点位）+ 右下角标
            if items:
                ang = math.radians(45 + step * 10)
                x = cx + 175 * math.cos(ang)
                y = cy + 88 * math.sin(ang)
                self._draw_node(x, y, items[0][0], AMBER)
                tid = cv.create_text(w - 16, h - 23,
                                     text=f"{items[0]} · 可拾取", anchor="e",
                                     font=FONT_MONO, fill=AMBER)
                x0, y0, x1, y1 = cv.bbox(tid)
                cv.create_rectangle(x0 - 8, y0 - 4, x1 + 8, y1 + 4,
                                    fill="#120a04", outline=AMBER, width=1)
                cv.tag_raise(tid)
            # 左上地点标题：地点名 + 英文副标 + 步数/在场人数
            cv.create_text(16, 16, text=loc, anchor="nw",
                           font=("Consolas", 14, "bold"), fill=CYAN)
            cv.create_text(16, 40, text=LOC_EN.get(loc, ""),
                           anchor="nw", font=("Consolas", 9), fill="#9fd8e6")
            cv.create_text(16, 56,
                           text=f"STEP {step} · 在场 {len(chars)} 人",
                           anchor="nw", font=("Consolas", 9), fill=CYAN_DIM)
        except Exception:
            # 画布画失败不影响玩：星图只是展示，静默跳过
            pass

    def _draw_node(self, x, y, label, color):
        """星图节点：深底圆 + 彩色描边 + 一个字。"""
        cv = self.map_canvas
        cv.create_oval(x - 16, y - 16, x + 16, y + 16,
                       fill="#070d18", outline=color, width=1)
        cv.create_text(x, y, text=label, font=("Microsoft YaHei", 13),
                       fill=color)

    def _do(self, label, fn):
        if not self.game or self.busy:
            return
        self.busy = True
        self._set_status(f"{label}…（模型在想）")
        self.root.update()
        try:
            reply, err, note = fn()
        except Exception as e:
            reply, err, note = f"出错了：{e}", 1.0, ""
        self.game.record(label, reply, err, note)
        mark = "✓ 答对" if err == 0.0 else "✗ 答错"
        self._log(f"你：{label}")
        self._log(f"欧米伽：{reply}  [{mark}]" + (f"（{note}）" if note else ""))
        self._set_status("就绪。")
        self._refresh_world()
        self.busy = False

    def _ask_free(self):
        text = self.free_var.get().strip()
        if not text or not self.game:
            return
        self.free_var.set("")
        self._do(f"问：{text}", lambda: self.game.ask_free(text))

    def _auto_start(self, steps=AUTO_STEPS):
        """内核自动玩：每一步由桥接器里的内核按接地分选动作。

        没有桥接器就明说、不启动——固定轮换已经拿掉，不会悄悄轮换。
        手动查看、移动、拾取和对话不受影响。"""
        if not self.game or self.auto_running:
            return
        bridge = ka.find_bridge()
        if bridge is None:
            messagebox.showinfo(
                "内核自动玩",
                "界面旁没有找到桥接器（omega-world-bridge），这一步没法"
                "由内核选，所以自动玩没有启动。\n\n手动查看、移动、拾取"
                "和对话不受影响。")
            self._set_status("自动玩未启动：没有找到桥接器。")
            return
        self.driver = ka.KernelDriver(HERE / "run_kernel", steps=steps,
                                      bridge=bridge)
        try:
            self.driver.start()
        except Exception as e:
            messagebox.showerror("内核自动玩", f"桥接器启动失败：\n{e}")
            return
        self.auto_running = True
        self.auto_left = steps
        self._k_step = 1
        self._k_deadline = time.time() + ka.ACTION_TIMEOUT_SECS
        self.auto_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.auto_var.set(f"内核自动玩：剩 {self.auto_left} 步")
        self.game.last_reply = ""
        self.game.last_err = 0.0
        self.driver.write_observation(self.game.observation(0))
        self._log("内核自动玩开始：每一步由内核按接地分选。")
        self._kernel_tick()

    def _kernel_tick(self):
        """一轮内核自动玩的心跳（after 驱动，不卡界面）：
        等内核给动作 -> 执行 -> 把新状态和接地分交回桥接器 -> 下一步。"""
        if not self.auto_running:
            return
        if self.auto_left <= 0:
            self._kernel_finish()
            return
        if not self.driver.alive():
            self._log("桥接器已退出，自动玩停止。")
            self._kernel_finish()
            return
        step = self._k_step
        action = self.driver.read_action(step)
        if action is None:
            if time.time() > self._k_deadline:
                self._log(f"第 {step} 步：等不到内核的动作，自动玩停止。")
                self._kernel_finish()
                return
            self.root.after(200, self._kernel_tick)
            return
        if self.busy:
            self.root.after(300, self._kernel_tick)
            return
        raw = self.driver.last_raw_action
        label, reply, err, note = self.game.run_kernel_action(action)
        self.game.last_reply = reply
        self.game.last_err = err
        # 先写新状态的观察，再写本步的结果：桥接器下一步读到的永远是新状态。
        self.driver.write_observation(self.game.observation(step))
        self.driver.write_outcome(step, action, reply, err, note)
        self.driver.record_step(step, action, err, reply)
        self.game.record(f"[内核]{label}", reply, err, note)
        mark = "✓ 答对" if err == 0.0 else "✗ 答错"
        extra = f"（内核给了白名单外的动作 {raw}，按 noop 处理）" if raw != action else ""
        self._log(f"内核选了 {action}：{label}{extra}")
        self._log(f"欧米伽：{reply}  [{mark}]" + (f"（{note}）" if note else ""))
        self.auto_left -= 1
        self._k_step += 1
        self._k_deadline = time.time() + ka.ACTION_TIMEOUT_SECS
        self.auto_var.set(f"内核自动玩：剩 {self.auto_left} 步")
        self._refresh_world()
        self.root.after(1200, self._kernel_tick)

    def _kernel_finish(self):
        self.auto_running = False
        if getattr(self, "driver", None) is not None:
            self.driver.stop()
            self.driver = None
        self.auto_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.auto_var.set("")
        self._set_status("就绪。")
        self._log(f"内核自动玩结束：答对 {self.game.correct} / {self.game.total}。")

    def _auto_stop(self):
        self._kernel_finish()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--auto", type=int, default=0,
                    help="启动后自动玩 N 步（测试用）")
    ap.add_argument("--quit-after-auto", action="store_true")
    args = ap.parse_args()
    root = tk.Tk()
    app = App(root, ckpt=args.ckpt)
    if args.auto:
        def _kick():
            if app.game is not None:
                app._auto_start(steps=args.auto)
            else:
                root.after(500, _kick)
        root.after(500, _kick)
        if args.quit_after_auto:
            def _watch():
                if not app.auto_running and app.game is not None:
                    root.after(500, root.destroy)
                else:
                    root.after(500, _watch)
            root.after(1000, _watch)
    root.mainloop()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Omega 小世界 GUI v0.1.0 — Windows 可玩版。

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
        self.root.title("Omega 小世界 v0.1.0")
        self.root.geometry("780x760")
        try:
            import tkinter.font as tkfont
            tkfont.nametofont("TkDefaultFont").configure(family="Microsoft YaHei", size=10)
            tkfont.nametofont("TkTextFont").configure(family="Microsoft YaHei", size=10)
        except Exception:
            pass
        self.game = None
        self.busy = False
        self.auto_running = False
        self.auto_left = 0
        self.driver = None  # 内核自动玩的桥接器驱动（未启动时为空）
        self._build()
        self._set_status("正在准备…")
        self.root.after(200, lambda: self._startup(ckpt))

    def _build(self):
        top = ttk.Frame(self.root)
        top.pack(fill="x", padx=8, pady=6)
        ttk.Label(top, text="人格：").pack(side="left")
        self.persona_var = tk.StringVar(value="omega")
        self.persona_cb = ttk.Combobox(top, textvariable=self.persona_var,
                                       values=PERSONAS, width=8, state="readonly")
        self.persona_cb.pack(side="left", padx=4)
        ttk.Label(top, text="我想：").pack(side="left", padx=(8, 0))
        self.intent_var = tk.StringVar()
        ttk.Entry(top, textvariable=self.intent_var, width=28).pack(side="left", padx=4)
        ttk.Button(top, text="应用", command=self._apply_self).pack(side="left", padx=4)
        self.score_var = tk.StringVar(value="答对 0 / 0")
        ttk.Label(top, textvariable=self.score_var, foreground="green").pack(side="right")
        self.archive_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text="记录对话", variable=self.archive_var,
                        command=self._toggle_archive).pack(side="right", padx=6)

        wf = ttk.LabelFrame(self.root, text="世界状态")
        wf.pack(fill="x", padx=8, pady=4)
        self.world_vars = {}
        for key in ("地点", "时间", "在场人物", "可见物品", "随身物品", "自我"):
            row = ttk.Frame(wf)
            row.pack(fill="x", padx=6, pady=1)
            ttk.Label(row, text=key + "：", width=8).pack(side="left")
            v = tk.StringVar(value="—")
            ttk.Label(row, textvariable=v).pack(side="left")
            self.world_vars[key] = v

        self.log = scrolledtext.ScrolledText(self.root, height=16, state="disabled",
                                             wrap="word")
        self.log.pack(fill="both", expand=True, padx=8, pady=4)

        mf = ttk.LabelFrame(self.root, text="走动")
        mf.pack(fill="x", padx=8, pady=4)
        for loc in LOCATIONS:
            ttk.Button(mf, text=loc, width=8,
                       command=lambda d=loc: self._do(f"去{loc}", lambda: self.game.move(d))).pack(side="left", padx=2)

        af = ttk.LabelFrame(self.root, text="动作")
        af.pack(fill="x", padx=8, pady=4)
        ttk.Button(af, text="查看", width=8,
                   command=lambda: self._do("查看", self.game.look)).pack(side="left", padx=2)
        self.take_frame = ttk.Frame(af)
        self.take_frame.pack(side="left", padx=6)
        self.talk_frame = ttk.Frame(af)
        self.talk_frame.pack(side="left", padx=6)

        qf = ttk.LabelFrame(self.root, text="问问题")
        qf.pack(fill="x", padx=8, pady=4)
        for qi, q in enumerate(wsp.QUESTIONS):
            ttk.Button(qf, text=q, width=16,
                       command=lambda i=qi: self._do(f"问：{wsp.QUESTIONS[i]}",
                                                      lambda: self.game.ask(i))).pack(side="left", padx=2)
        self.free_var = tk.StringVar()
        ttk.Entry(qf, textvariable=self.free_var, width=24).pack(side="left", padx=4)
        ttk.Button(qf, text="发送", command=self._ask_free).pack(side="left")

        bf = ttk.Frame(self.root)
        bf.pack(fill="x", padx=8, pady=6)
        self.auto_btn = ttk.Button(bf, text="内核自动玩 12 步", command=self._auto_start)
        self.auto_btn.pack(side="left", padx=2)
        self.stop_btn = ttk.Button(bf, text="停", command=self._auto_stop, state="disabled")
        self.stop_btn.pack(side="left", padx=2)
        self.auto_var = tk.StringVar(value="")
        ttk.Label(bf, textvariable=self.auto_var).pack(side="left", padx=8)

        self.status_var = tk.StringVar(value="")
        ttk.Label(self.root, textvariable=self.status_var, relief="sunken",
                  anchor="w").pack(fill="x", padx=8, pady=(0, 6))
        self._set_enabled(False)

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
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n")
        self.log.configure(state="disabled")
        self.log.see("end")

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

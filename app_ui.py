"""
app_ui.py —— MiniTransformer 图形界面（Pythonista）

设计要点：全部在主线程运行，不使用后台线程。
训练、生成、保存、心跳都是“协作式任务”（生成器），由定时器每次执行一小段，
界面始终可响应，同时避免多线程并发访问模型和 UI 导致的崩溃。

功能
  生成   提示词历史、温度 / Top-K 滑块、长度、复制、导出 txt
  训练   实时 loss 曲线、进度条、ETA、可随时停止、自动保存
  日志   捕获全部控制台输出（含心跳），支持复制 / 清空 / 导出
  设置   心跳开关与间隔、加载 / 备份权重、重新初始化、模型信息
  设置持久化到 ui_settings.json

依赖：同目录下的 app.py 与 transformer_core.py
"""

import os
import sys
import json
import time
import shutil
import secrets
import collections
from datetime import datetime

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE)

import ui
import clipboard
import app  # 复用 app.py 中的 omega 桩对象与心跳所需函数
from omega import (
    MiniTransformer, Adam, sample_token, encode, decode,
    save_checkpoint, load_checkpoint,
    VOCAB, D, N_HEADS, SEQ_LEN, STEPS, LR, CHECKPOINT_PATH, PARAM_NAMES,
    data, rng,
)

SETTINGS_PATH = os.path.join(BASE, "ui_settings.json")
EXPORT_DIR = os.path.join(BASE, "exports")
BACKUP_DIR = os.path.join(BASE, "backups")
for _d in (EXPORT_DIR, BACKUP_DIR):
    os.makedirs(_d, exist_ok=True)

# ---------------- 外观 ----------------
ACCENT = "#0a84ff"
GOOD = "#30d158"
BG = "#f2f2f7"
CARD = "#ffffff"
MUTED = "#8e8e93"
FONT = ("<System>", 15)
MONO = ("Menlo", 13)

TICK_BUDGET = 0.04      # 每个定时器周期最多占用主线程的秒数
TICK_BUSY = 0.0         # 有任务时的定时器间隔
TICK_IDLE = 0.2         # 空闲时的定时器间隔

# =========================================================
# 设置（持久化）
# =========================================================

DEFAULTS = {
    "prompt_history": [],
    "temperature": 0.8,
    "top_k": 8,
    "gen_length": 40,
    "train_steps": 100,
    "lr": app.DEFAULT_LR,
    "hb_on": True,
    "hb_interval": 12,
    "autosave_exit": True,
    "total_steps": 0,
    "last_saved": "",
}


class Settings:
    def __init__(self):
        self.d = dict(DEFAULTS)
        try:
            with open(SETTINGS_PATH, encoding="utf-8") as f:
                self.d.update(json.load(f))
        except (OSError, ValueError):
            pass

    def __getitem__(self, k):
        return self.d[k]

    def set(self, k, v):
        self.d[k] = v
        self._save()

    def _save(self):
        try:
            with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
                json.dump(self.d, f, ensure_ascii=False, indent=2)
        except OSError:
            pass

    def push_prompt(self, p):
        rest = [x for x in self.d["prompt_history"] if x != p]
        self.set("prompt_history", ([p] + rest)[:30])


S = Settings()


# =========================================================
# 控制台捕获
# =========================================================

class LogStream:
    def __init__(self, orig):
        self.orig = orig
        self.lines = collections.deque(maxlen=600)
        self.version = 0
        self._buf = ""

    def write(self, s):
        try:
            self.orig.write(s)
        except Exception:
            pass
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line.strip():
                self.lines.append(f"[{datetime.now():%H:%M:%S}] {line}")
                self.version += 1

    def flush(self):
        try:
            self.orig.flush()
        except Exception:
            pass


LOG = LogStream(sys.stdout)
sys.stdout = LOG


# =========================================================
# 模型与任务生成器（全部在主线程执行）
# =========================================================

class Engine:
    def __init__(self):
        self.model = None
        self.cancel = False  # 训练停止标志


ENGINE = Engine()


def set_model(m):
    ENGINE.model = m
    app.set_model(m)


def losses():
    return app.LOSS_HISTORY


def save_model():
    save_checkpoint(ENGINE.model, CHECKPOINT_PATH)
    app.save_state()
    S.set("last_saved", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    return True


def train_and_save(steps, lr):
    """OMEGA 门控训练：每个循环由 app.train_job 执行并判定是否保留。"""
    kept = yield from app.train_job(steps, lr, should_stop=lambda: ENGINE.cancel)
    S.set("total_steps", S["total_steps"] + kept)
    save_model()
    return kept


def generate_gen(model, prompt, length, temperature, top_k):
    """逐字生成：每生成一个字 yield 一次，保证界面不卡。"""
    ids = list(encode(prompt))
    for _ in range(length):
        ctx = np.array(ids[-SEQ_LEN:], dtype=np.int64)
        logits, _ = model.forward(ctx)
        ids.append(sample_token(logits[-1], temperature=temperature, top_k=top_k))
        yield None
    return decode(ids)


def run_sync(gen):
    """同步执行一个生成器（用于心跳这类短任务）"""
    while True:
        try:
            next(gen)
        except StopIteration as e:
            return e.value


def once(fn):
    """把一次性同步函数包装成任务生成器"""
    def g():
        return fn()
        yield  # noqa: unreachable，使其成为生成器
    return g()


def init_gen():
    app.init_omega()
    m = MiniTransformer()
    set_model(m)
    if load_checkpoint(m, CHECKPOINT_PATH):
        print("已加载现有权重")
        return True
    print(f"未找到权重，首次训练 {STEPS} 步")
    yield from train_and_save(STEPS, S["lr"])
    return True


def reset_model():
    set_model(MiniTransformer())
    losses().clear()
    return True


def load_weights():
    if ENGINE.model is None:
        raise RuntimeError("模型未就绪")
    if not load_checkpoint(ENGINE.model, CHECKPOINT_PATH):
        raise RuntimeError("权重不存在或形状不匹配")
    return True


def backup_weights():
    save_checkpoint(ENGINE.model, CHECKPOINT_PATH)
    name = datetime.now().strftime("model_%Y%m%d_%H%M%S.npz")
    shutil.copy2(CHECKPOINT_PATH, os.path.join(BACKUP_DIR, name))
    return name


# =========================================================
# 心跳（主线程定时执行，不占用后台线程）
# =========================================================

HB = {"last": 0.0, "n": 0}


def heartbeat_once():
    if ENGINE.model is None:
        return
    r = app.measure_cycle()
    if r is None:
        return
    HB["n"] += 1
    loss = "-" if r["loss"] is None else f"{r['loss']:.4f}"
    print(f"[心跳#{HB['n']}] {r['decision']} loss={loss} reason={r['reason']}")
    text = run_sync(generate_gen(ENGINE.model, "话说", 20, 0.7, 8))
    print("[生成]", text)


def maybe_heartbeat(now):
    if not S["hb_on"] or ENGINE.model is None:
        return
    if now - HB["last"] < S["hb_interval"]:
        return
    HB["last"] = now
    try:
        heartbeat_once()
    except Exception as e:
        print(f"[心跳] 出错: {type(e).__name__}: {e}")


# =========================================================
# 调度器：主线程上的协作式任务队列
# =========================================================

class Scheduler:
    def __init__(self):
        self.cur = None      # {"op", "gen", "on_done"} 或 None
        self.subs = []       # 事件订阅者 fn(ev)，只在主线程调用
        self.tickers = []    # 每次 tick 调用的回调

    @property
    def op(self):
        return self.cur["op"] if self.cur else None

    def publish(self, ev):
        for cb in list(self.subs):
            try:
                cb(ev)
            except Exception as e:
                print(f"[ui] {type(e).__name__}: {e}")

    def start(self, op, gen, on_done=None):
        if self.cur is not None:
            return False, self.cur["op"]
        ENGINE.cancel = False
        self.cur = {"op": op, "gen": gen, "on_done": on_done}
        self.publish({"kind": "state"})
        return True, None

    def _finish(self, value, err):
        job = self.cur
        self.cur = None
        if err is not None:
            print(f"[{job['op']}] 出错: {type(err).__name__}: {err}")
        self.publish({"kind": "state"})
        if job["on_done"]:
            try:
                job["on_done"](value, err)
            except Exception as e:
                print(f"[回调] {type(e).__name__}: {e}")

    def tick(self):
        t0 = time.time()
        if self.cur is not None:
            while time.time() - t0 < TICK_BUDGET:
                try:
                    ev = next(self.cur["gen"])
                except StopIteration as e:
                    self._finish(e.value, None)
                    break
                except Exception as e:
                    self._finish(None, e)
                    break
                if isinstance(ev, dict):
                    self.publish(ev)
        else:
            maybe_heartbeat(time.time())
        for t in self.tickers:
            try:
                t()
            except Exception as e:
                print(f"[tick] {type(e).__name__}: {e}")


SCHED = Scheduler()


def _tick():
    try:
        SCHED.tick()
    except Exception as e:
        print(f"[tick] {type(e).__name__}: {e}")
    ui.delay(_tick, TICK_BUSY if SCHED.cur is not None else TICK_IDLE)


# =========================================================
# 通用工具
# =========================================================

MAIN = None


def notify(msg):
    if MAIN is not None:
        MAIN.status.text = msg


def _int(text, default):
    try:
        return int(text)
    except (TypeError, ValueError):
        return default


def _float(text, default):
    try:
        return float(text)
    except (TypeError, ValueError):
        return default


def fmt_secs(s):
    m, sec = divmod(int(s), 60)
    return f"{m:02d}:{sec:02d}"


def make_label(text="", size=13, color=MUTED, mono=False):
    lbl = ui.Label(font=MONO if mono else ("<System>", size), text_color=color)
    lbl.text = text
    lbl.number_of_lines = 0
    return lbl


def make_button(title, action, primary=True):
    b = ui.Button(title=title, font=("<System>", 16))
    b.background_color = ACCENT if primary else "#e5e5ea"
    b.tint_color = "white" if primary else ACCENT
    b.corner_radius = 10
    b.action = action
    return b


class SliderRow(ui.View):
    """标题 + 当前值 + 滑块"""

    def __init__(self, title, lo, hi, value, fmt, on_change, integer=False):
        self.background_color = "clear"
        self.lo, self.hi = lo, hi
        self.fmt, self.on_change, self.integer = fmt, on_change, integer
        self.cur = value
        self.title_lbl = make_label(title, 14, "#000")
        self.value_lbl = make_label(fmt(value), 14, ACCENT)
        self.value_lbl.alignment = ui.ALIGN_RIGHT
        self.slider = ui.Slider()
        self.slider.value = self._norm(value)
        self.slider.continuous = True
        self.slider.action = self._changed
        for v in (self.title_lbl, self.value_lbl, self.slider):
            self.add_subview(v)

    def _norm(self, v):
        return (v - self.lo) / (self.hi - self.lo)

    def get(self):
        return self.cur

    def _changed(self, sender):
        v = self.lo + sender.value * (self.hi - self.lo)
        if self.integer:
            v = int(round(v))
        self.cur = v
        self.value_lbl.text = self.fmt(v)
        self.on_change(v)

    def layout(self):
        w = self.width
        self.title_lbl.frame = (0, 0, w * 0.6, 22)
        self.value_lbl.frame = (w * 0.6, 0, w * 0.4, 22)
        self.slider.frame = (0, 26, w, 30)


class LossChart(ui.View):
    def __init__(self):
        self.background_color = CARD
        self.corner_radius = 10
        self.points = []

    def set_points(self, pts):
        self.points = list(pts)
        self.set_needs_display()

    def draw(self):
        w, h, pad = self.width, self.height, 10
        ui.set_color("#e5e5ea")
        for i in range(1, 4):
            y = pad + i * (h - 2 * pad) / 4
            grid = ui.Path()
            grid.move_to(pad, y)
            grid.line_to(w - pad, y)
            grid.line_width = 0.5
            grid.stroke()
        pts = self.points[-400:]
        if len(pts) < 2:
            return
        lo, hi = min(pts), max(pts)
        if hi - lo < 1e-9:
            hi = lo + 1e-3
        path = ui.Path()
        n = len(pts)
        for i, v in enumerate(pts):
            x = pad + i * (w - 2 * pad) / (n - 1)
            y = pad + (hi - v) / (hi - lo) * (h - 2 * pad)
            if i == 0:
                path.move_to(x, y)
            else:
                path.line_to(x, y)
        path.line_width = 2
        ui.set_color(ACCENT)
        path.stroke()


class ProgressBar(ui.View):
    def __init__(self):
        self.background_color = "#e5e5ea"
        self.corner_radius = 4
        self.frac = 0.0

    def set(self, f):
        self.frac = max(0.0, min(1.0, f))
        self.set_needs_display()

    def draw(self):
        ui.set_color(GOOD)
        ui.fill_rect(0, 0, self.width * self.frac, self.height)


# =========================================================
# 页面：生成
# =========================================================

class GeneratePage(ui.View):
    def __init__(self):
        self.background_color = BG
        self.prompt = ui.TextField(placeholder="提示词（留空则为“话说”）", font=FONT)

        self.hist_ds = ui.ListDataSource([])
        self.hist_ds.action = self.on_pick
        self.hist = ui.TableView()
        self.hist.data_source = self.hist_ds
        self.hist.delegate = self.hist_ds
        self.refresh_history()

        self.temp = SliderRow(
            "温度 temperature", 0.1, 1.5, S["temperature"],
            lambda v: f"{v:.2f}", lambda v: S.set("temperature", round(v, 2)))
        self.topk = SliderRow(
            "Top-K", 1, VOCAB, S["top_k"],
            lambda v: f"{int(v)}", lambda v: S.set("top_k", int(v)), integer=True)

        self.length = ui.TextField(
            placeholder="长度", text=str(S["gen_length"]),
            keyboard_type=ui.KEYBOARD_NUMBER_PAD, font=FONT)
        self.btn_gen = make_button("生成", self.on_generate)
        self.btn_copy = make_button("复制", self.on_copy, primary=False)
        self.btn_export = make_button("导出 txt", self.on_export, primary=False)

        self.output = ui.TextView(editable=False, font=FONT, background_color=CARD)

        for v in (self.prompt, self.hist, self.temp, self.topk, self.length,
                  self.btn_gen, self.btn_copy, self.btn_export, self.output):
            self.add_subview(v)

    def refresh_history(self):
        self.hist_ds.items = list(S["prompt_history"])
        self.hist.reload_data()

    def on_pick(self, sender):
        self.prompt.text = sender.items[sender.selected_row]

    def layout(self):
        w, h = self.width, self.height
        p = 16
        iw = w - 2 * p
        half = (iw - 10) / 2
        self.prompt.frame = (p, 10, iw, 34)
        self.hist.frame = (p, 52, iw, 90)
        self.temp.frame = (p, 150, iw, 56)
        self.topk.frame = (p, 212, iw, 56)
        self.length.frame = (p, 278, half, 34)
        self.btn_gen.frame = (p + half + 10, 278, half, 34)
        self.btn_copy.frame = (p, 322, half, 34)
        self.btn_export.frame = (p + half + 10, 322, half, 34)
        top = 368
        self.output.frame = (p, top, iw, max(h - top - p, 80))

    def on_generate(self, sender):
        if ENGINE.model is None:
            return notify("模型尚未就绪")
        prompt = self.prompt.text.strip() or "话说"
        length = max(1, min(300, _int(self.length.text, 40)))
        temp = self.temp.get()
        topk = int(self.topk.get())
        S.set("gen_length", length)
        S.push_prompt(prompt)
        self.refresh_history()

        def done(text, err):
            if err:
                notify(f"生成失败：{err}")
            else:
                self.output.text = text
                notify("生成完成")

        ok, busy = SCHED.start(
            "生成", generate_gen(ENGINE.model, prompt, length, temp, topk), done)
        if not ok:
            notify(f"忙碌中：{busy}")

    def on_copy(self, sender):
        text = self.output.text.strip()
        if not text:
            return notify("没有可复制的内容")
        clipboard.set(text)
        notify("已复制到剪贴板")

    def on_export(self, sender):
        text = self.output.text.strip()
        if not text:
            return notify("没有可导出的内容")
        name = datetime.now().strftime("gen_%Y%m%d_%H%M%S.txt")
        with open(os.path.join(EXPORT_DIR, name), "w", encoding="utf-8") as f:
            f.write(text)
        notify(f"已导出：exports/{name}")


# =========================================================
# 页面：训练
# =========================================================

class TrainPage(ui.View):
    def __init__(self):
        self.background_color = BG
        self.steps = ui.TextField(
            placeholder="训练步数", text=str(S["train_steps"]),
            keyboard_type=ui.KEYBOARD_NUMBER_PAD, font=FONT)
        self.lr = ui.TextField(
            placeholder="学习率", text=str(S["lr"]),
            keyboard_type=ui.KEYBOARD_DECIMAL_PAD, font=FONT)
        self.btn_start = make_button("开始训练", self.on_start)
        self.btn_stop = make_button("停止", self.on_stop, primary=False)
        self.btn_save = make_button("保存", self.on_save, primary=False)
        self.bar = ProgressBar()
        self.stats = make_label("空闲", 13, "#000", mono=True)
        self.chart = LossChart()
        self.chart.set_points(losses())
        for v in (self.steps, self.lr, self.btn_start, self.btn_stop,
                  self.btn_save, self.bar, self.stats, self.chart):
            self.add_subview(v)
        SCHED.subs.append(self.on_event)
        self.refresh()

    def layout(self):
        w, h = self.width, self.height
        p = 16
        iw = w - 2 * p
        half = (iw - 10) / 2
        third = (iw - 20) / 3
        self.steps.frame = (p, 10, half, 34)
        self.lr.frame = (p + half + 10, 10, half, 34)
        self.btn_start.frame = (p, 56, third, 40)
        self.btn_stop.frame = (p + third + 10, 56, third, 40)
        self.btn_save.frame = (p + 2 * (third + 10), 56, third, 40)
        self.bar.frame = (p, 112, iw, 8)
        self.stats.frame = (p, 126, iw, 40)
        chart_top = 172
        self.chart.frame = (p, chart_top, iw, max(h - chart_top - p, 120))

    def on_event(self, ev):
        if ev["kind"] == "step":
            self.show_step(ev)
        elif ev["kind"] == "state":
            self.refresh()

    def show_step(self, ev):
        self.bar.set(ev["step"] / ev["total"])
        recent = losses()[-10:]
        avg = sum(recent) / len(recent) if recent else ev["loss"]
        kept = "保留" if ev.get("accepted", True) else "回滚"
        self.stats.text = (f"cycle {ev['step']}/{ev['total']}  "
                           f"val {ev['loss']:.4f}  avg10 {avg:.4f}\n"
                           f"{ev.get('decision', '')} {kept}  ETA {fmt_secs(ev['eta'])}")
        self.chart.set_points(losses())

    def refresh(self):
        busy = SCHED.op
        escalated = app.OMEGA is not None and app.OMEGA.escalated
        self.btn_start.enabled = ENGINE.model is not None and busy is None and not escalated
        self.btn_stop.enabled = busy == "训练"
        self.btn_save.enabled = busy is None
        if busy is None:
            self.chart.set_points(losses())
            if escalated:
                self.stats.text = "OMEGA 已升级，训练已暂停。请到设置页解除升级。"
            elif losses():
                self.stats.text = (f"已完成  最新 loss {losses()[-1]:.4f}  "
                                   f"累计 {S['total_steps']} 步")
            else:
                self.stats.text = "空闲"

    def on_start(self, sender):
        if ENGINE.model is None:
            return notify("模型尚未就绪")
        steps = max(1, min(20000, _int(self.steps.text, 100)))
        lr = _float(self.lr.text, S["lr"])
        S.set("train_steps", steps)
        S.set("lr", lr)
        self.bar.set(0)

        def done(res, err):
            if err:
                notify(f"训练失败：{err}")
            else:
                notify(f"训练结束：OMEGA 保留 {res} 步，已保存")

        ok, busy = SCHED.start("训练", train_and_save(steps, lr), done)
        if not ok:
            notify(f"忙碌中：{busy}")

    def on_stop(self, sender):
        if SCHED.op == "训练":
            ENGINE.cancel = True
            notify("正在停止，完成当前步后保存...")

    def on_save(self, sender):
        ok, busy = SCHED.start(
            "保存", once(save_model),
            lambda r, e: notify("权重已保存" if not e else f"保存失败：{e}"))
        if not ok:
            notify(f"忙碌中：{busy}")


# =========================================================
# 页面：日志
# =========================================================

class LogPage(ui.View):
    def __init__(self):
        self.background_color = BG
        self.tv = ui.TextView(editable=False, font=MONO, background_color=CARD)
        self.btn_copy = make_button("复制", self.on_copy, primary=False)
        self.btn_clear = make_button("清空", self.on_clear, primary=False)
        self.btn_export = make_button("导出 txt", self.on_export, primary=False)
        for v in (self.tv, self.btn_copy, self.btn_clear, self.btn_export):
            self.add_subview(v)
        self.seen = -1
        SCHED.tickers.append(self.tick)
        self.redraw()

    def tick(self):
        if LOG.version != self.seen:
            self.redraw()

    def redraw(self):
        self.tv.text = "\n".join(LOG.lines)
        self.seen = LOG.version

    def layout(self):
        w, h = self.width, self.height
        p = 16
        iw = w - 2 * p
        third = (iw - 20) / 3
        self.btn_copy.frame = (p, 10, third, 34)
        self.btn_clear.frame = (p + third + 10, 10, third, 34)
        self.btn_export.frame = (p + 2 * (third + 10), 10, third, 34)
        self.tv.frame = (p, 56, iw, max(h - 56 - p, 100))

    def on_copy(self, sender):
        clipboard.set(self.tv.text)
        notify("日志已复制")

    def on_clear(self, sender):
        LOG.lines.clear()
        self.redraw()

    def on_export(self, sender):
        name = datetime.now().strftime("log_%Y%m%d_%H%M%S.txt")
        with open(os.path.join(EXPORT_DIR, name), "w", encoding="utf-8") as f:
            f.write("\n".join(LOG.lines))
        notify(f"日志已导出：exports/{name}")


# =========================================================
# 页面：设置
# =========================================================

class SettingsPage(ui.View):
    def __init__(self):
        self.background_color = BG
        self.reset_armed = False

        self.sec_hb = make_label("OMEGA 心跳", 13, MUTED)
        self.lbl_hb = make_label("后台心跳（定时生成并记录）", 14, "#000")
        self.sw_hb = ui.Switch()
        self.sw_hb.value = S["hb_on"]
        self.sw_hb.action = self.on_hb

        self.interval = SliderRow(
            "心跳间隔", 3, 60, S["hb_interval"],
            lambda v: f"{int(v)} 秒", lambda v: S.set("hb_interval", int(v)),
            integer=True)

        self.lbl_exit = make_label("退出时自动保存权重", 14, "#000")
        self.sw_exit = ui.Switch()
        self.sw_exit.value = S["autosave_exit"]
        self.sw_exit.action = lambda sender: S.set("autosave_exit", bool(sender.value))

        self.sec_model = make_label("模型信息", 13, MUTED)
        self.info = make_label("", 13, "#000", mono=True)

        self.btn_load = make_button("加载权重", self.on_load, primary=False)
        self.btn_backup = make_button("备份权重", self.on_backup, primary=False)
        self.btn_reset = make_button("重新初始化", self.on_reset, primary=False)
        self.btn_clear = make_button("清空提示词历史", self.on_clear_hist, primary=False)
        self.btn_unesc = make_button("解除 OMEGA 升级", self.on_unescalate, primary=False)

        for v in (self.sec_hb, self.lbl_hb, self.sw_hb, self.interval,
                  self.lbl_exit, self.sw_exit, self.sec_model, self.info,
                  self.btn_load, self.btn_backup, self.btn_reset, self.btn_clear,
                  self.btn_unesc):
            self.add_subview(v)
        self.refresh_info()

    def layout(self):
        w = self.width
        p = 16
        iw = w - 2 * p
        half = (iw - 10) / 2
        self.sec_hb.frame = (p, 12, iw, 18)
        self.lbl_hb.frame = (p, 36, iw - 70, 30)
        self.sw_hb.frame = (w - p - 51, 36, 51, 31)
        self.interval.frame = (p, 74, iw, 56)
        self.lbl_exit.frame = (p, 140, iw - 70, 30)
        self.sw_exit.frame = (w - p - 51, 140, 51, 31)
        self.sec_model.frame = (p, 186, iw, 18)
        self.info.frame = (p, 208, iw, 120)
        self.btn_load.frame = (p, 338, half, 36)
        self.btn_backup.frame = (p + half + 10, 338, half, 36)
        self.btn_reset.frame = (p, 384, half, 36)
        self.btn_clear.frame = (p + half + 10, 384, half, 36)
        self.btn_unesc.frame = (p, 430, iw, 36)

    def refresh_info(self):
        m = ENGINE.model
        if m is None:
            self.info.text = "模型尚未就绪"
            return
        n = sum(getattr(m, k).size for k in PARAM_NAMES)
        size = os.path.getsize(CHECKPOINT_PATH) / 1024 if os.path.exists(CHECKPOINT_PATH) else 0
        backups = len([f for f in os.listdir(BACKUP_DIR) if f.endswith(".npz")])
        om = app.omega_summary()
        omega_line = (f"OMEGA epoch {app.OMEGA.epoch}  通过率 {om.get('apply_rate', 0):.2f}  "
                      f"误差 {om.get('mean_error', 0):.4f}  "
                      f"{'健康' if om.get('healthy') else '异常'}"
                      f"{'  已升级' if app.OMEGA.escalated else ''}")
        self.info.text = (
            f"词表 {VOCAB}  D {D}  头 {N_HEADS}  序列 {SEQ_LEN}\n"
            f"参数量 {n:,}   权重 {size:.1f} KB\n"
            f"累计保留训练 {S['total_steps']} 步   备份 {backups} 个\n"
            f"上次保存 {S['last_saved'] or '—'}\n"
            f"{omega_line}")

    def on_hb(self, sender):
        S.set("hb_on", bool(sender.value))
        notify(f"OMEGA 心跳已{'开启' if sender.value else '关闭'}")

    def on_load(self, sender):
        ok, busy = SCHED.start(
            "加载权重", once(load_weights),
            lambda r, e: notify("权重已加载" if not e else f"加载失败：{e}"))
        if not ok:
            notify(f"忙碌中：{busy}")

    def on_backup(self, sender):
        ok, busy = SCHED.start(
            "备份", once(backup_weights),
            lambda r, e: notify(f"已备份：{r}" if not e else f"备份失败：{e}"))
        if not ok:
            notify(f"忙碌中：{busy}")

    def on_reset(self, sender):
        if not self.reset_armed:
            self.reset_armed = True
            self.btn_reset.title = "再点一次确认"
            return notify("再次点击将重新初始化模型（未保存的权重会丢失）")
        self.reset_armed = False
        self.btn_reset.title = "重新初始化"
        ok, busy = SCHED.start(
            "重置", once(reset_model),
            lambda r, e: notify("模型已重新初始化（未训练）" if not e else f"失败：{e}"))
        if not ok:
            notify(f"忙碌中：{busy}")

    def on_unescalate(self, sender):
        app.clear_escalation()
        self.refresh_info()
        notify("已解除 OMEGA 升级")

    def on_clear_hist(self, sender):
        S.set("prompt_history", [])
        if MAIN is not None:
            MAIN.gen.refresh_history()
        notify("提示词历史已清空")


# =========================================================
# 主窗口
# =========================================================

class MainView(ui.View):
    def __init__(self):
        self.name = "MiniTransformer"
        self.background_color = BG
        self.seg = ui.SegmentedControl(segments=["生成", "训练", "日志", "设置"])
        self.seg.selected_index = 0
        self.seg.action = self.on_seg
        self.status = make_label("初始化中...", 13, ACCENT)

        self.gen = GeneratePage()
        self.train = TrainPage()
        self.log = LogPage()
        self.settings = SettingsPage()
        self.pages = [self.gen, self.train, self.log, self.settings]

        for v in self.pages:
            self.add_subview(v)
        self.add_subview(self.seg)
        self.add_subview(self.status)
        self.show(0)

        SCHED.subs.append(self.on_event)

    def on_seg(self, sender):
        self.show(sender.selected_index)

    def show(self, i):
        for idx, page in enumerate(self.pages):
            page.hidden = idx != i
        if i == 3:
            self.settings.refresh_info()

    def layout(self):
        w, h = self.width, self.height
        self.seg.frame = (16, 12, w - 32, 30)
        self.status.frame = (16, 48, w - 32, 18)
        for page in self.pages:
            page.frame = (0, 74, w, h - 74)

    def on_event(self, ev):
        if ev["kind"] == "state":
            self.refresh()

    def refresh(self):
        op = SCHED.op
        if op:
            self.status.text = f"忙碌：{op}"
        elif ENGINE.model is None:
            self.status.text = "初始化中..."
        else:
            self.status.text = "就绪"
        self.train.refresh()
        self.settings.refresh_info()

    def will_close(self):
        if S["autosave_exit"] and ENGINE.model is not None and SCHED.op is None:
            try:
                save_model()
            except Exception as e:
                print(f"退出保存失败: {e}")


# =========================================================
# 入口
# =========================================================

if __name__ == "__main__":
    MAIN = MainView()
    MAIN.present("fullscreen")
    ui.delay(_tick, 0)

    def after_init(res, err):
        if err is None:
            notify("就绪")
        else:
            notify(f"初始化失败：{err}")

    SCHED.start("初始化", init_gen(), after_init)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Kernel-driven auto-play driver for the Omega GUI.

Spawns the Rust bridge (omega-world-bridge) and speaks its
file-exchange protocol, the same one the v0.1.0 sidecar uses:

  GUI writes observation.json  ->  bridge's kernel picks an action
  and writes action.json       ->  GUI executes it in the world,
  writes observation.json (new state) and then outcome.json
  (grounded error -> reward for the kernel).

Only whitelisted actions are accepted; anything else is treated as
noop and noted. The bridge's config (omega.toml, next to this file)
keeps self_modify off: the kernel only selects, it never rewrites
its own strategy.

Experiment A (verified on the workbench 2026-11-14, s2100: step 1
stabilize, steps 2-12 explore 11/11, grounded 12/12) is wired in
below: the driver passes --stab-weight 0.5 --explore-bonus 1.5
--explore-weight 2.5 to the bridge. Without those flags the kernel
only ever picks stabilize (a correct stabilize answer keeps raising
its weight x1.05 per success, while explore is never registered, so
set_strategy_weight("explore", _) silently returns false) and the
world never moves.

The bridge binary ships next to the GUI (Windows: _internal/omega-world-bridge.exe,
verified running on real Windows 2026-11-15). With no bridge next to the GUI,
kernel auto-play reports that instead of silently falling back to a fixed
rotation (which no longer exists). On the workbench, point OMEGA_BRIDGE at the
Linux bridge binary to run it.
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
WHITELIST = ["observe", "explore", "exploit", "stabilize", "noop"]
BRIDGE_ENV = "OMEGA_BRIDGE"
ACTION_TIMEOUT_SECS = 300
# 视频接地账本（grounding_ledger.jsonl，见 omega_v010/GROUNDING_LEDGER.md）。
# 消费端只读它，从不写它；账本本身仍由 verify_to_grounding.py 写。
VIDEO_GROUNDING_LEDGER = HERE.parent / "grounding_ledger.jsonl"


def load_video_grounding(ledger_path=None):
    """读视频接地账本，返回摘要 dict，读不到时返回 None，绝不抛错。

    摘要见 grounding_ledger.summarize：{episodes, n, mean_reward,
    min_reward, worst, latest}。账本不存在、读不出、或全部是坏行时
    返回 None——消费端缺失不是启动失败，驱动照常跑，只是不带视频侧状态。
    """
    try:
        parent = Path(__file__).resolve().parent.parent
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))
        import grounding_ledger as _gl
        path = Path(ledger_path) if ledger_path else VIDEO_GROUNDING_LEDGER
        rows, _skipped, _notes = _gl.load_ledger(path)
        if not rows:
            return None
        return _gl.summarize(rows)
    except Exception:
        return None


def find_bridge():
    """Locate the bridge binary: OMEGA_BRIDGE first, then next to the GUI."""
    p = os.environ.get(BRIDGE_ENV)
    if p and Path(p).is_file():
        return str(p)
    for name in ("omega-world-bridge.exe", "omega-world-bridge"):
        q = HERE / name
        if q.is_file():
            return str(q)
    return None


def write_json(path, obj):
    """Atomic JSON write (tmp + replace), so the bridge never reads a half file."""
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(tmp, path)


class KernelDriver:
    """Owns one bridge process and one run dir for a kernel auto-play run."""

    def __init__(self, run_dir, steps=12, bridge=None, config=None,
                 goal="act coherently in the little world as Omega"):
        self.dir = Path(run_dir)
        self.steps = steps
        self.bridge = bridge or find_bridge()
        self.config = config or str(HERE / "omega.toml")
        self.goal = goal
        self.proc = None
        self._logf = None
        self.steps_log = []
        self.last_raw_action = None
        # 视频接地摘要（load_video_grounding_history 填入）；None 表示
        # 账本不可用。只读，不参与任何 step 的 reward 或决策。
        self.video_grounding = None

    def load_video_grounding_history(self, ledger_path=None):
        """读视频接地账本，写 run 目录的 history 文件并记一行驱动日志。

        可单独调用（不启动桥），start() 也会调它。绝不抛错：
        账本缺失或全坏时返回 None，删掉可能残留的旧 history 文件，
        记一行 `video_grounding: none`，驱动照常启动。
        成功时写 video_grounding_history.json（注意它不在 start() 的
        清理名单里，写入发生在清理之后，所以不会被下一轮清掉），
        并记一行 `video_grounding: episodes=N mean_reward=X.XX worst=EP(R)`。
        """
        summary = load_video_grounding(ledger_path)
        if summary is None:
            self.video_grounding = None
            try:
                (self.dir / "video_grounding_history.json").unlink()
            except (FileNotFoundError, OSError):
                pass
            print("video_grounding: none", flush=True)
            return None
        self.video_grounding = summary
        self.dir.mkdir(parents=True, exist_ok=True)
        write_json(self.dir / "video_grounding_history.json", summary)
        worst = summary["worst"]
        print("video_grounding: episodes=%d mean_reward=%.2f worst=%s(%.2f)"
              % (summary["n"], summary["mean_reward"],
                 worst["episode"], worst["reward"]),
              flush=True)
        return summary

    def start(self):
        if self.bridge is None:
            raise FileNotFoundError("no bridge binary found (see find_bridge)")
        # Fresh run: clear stale exchange files and kernel state so each
        # auto-play run starts from a fresh kernel.
        self.dir.mkdir(parents=True, exist_ok=True)
        for name in ("observation.json", "action.json", "outcome.json",
                     "state.json", "audit.jsonl", "trajectory.jsonl"):
            try:
                (self.dir / name).unlink()
            except FileNotFoundError:
                pass
        shutil.rmtree(self.dir / "checkpoints", ignore_errors=True)
        try:
            (self.dir / "gui_kernel_steps.jsonl").unlink()
        except FileNotFoundError:
            pass
        # 视频接地历史：清理之后再写（video_grounding_history.json 不在
        # 上面的清理名单里，写完也不会被下一轮 start() 清掉）。读不到账本
        # 时只记一行 none，不阻断启动。
        self.load_video_grounding_history()
        self._logf = open(self.dir / "bridge.log", "w", encoding="utf-8")
        cmd = [self.bridge, "--dir", str(self.dir),
               "--steps", str(self.steps), "--goal", self.goal]
        if Path(self.config).is_file():
            cmd += ["--config", self.config]
        # Experiment A (see module docstring): pin stabilize's weight
        # down and explore's bonus up so the kernel actually moves.
        cmd += ["--stab-weight", "0.5",
                "--explore-bonus", "1.5",
                "--explore-weight", "2.5"]
        print(f"kernel driver: {' '.join(cmd)}", flush=True)
        # Windows: the bridge is a console-subsystem exe; without this it
        # pops a black console window over the GUI (seen on real Windows
        # 2026-11-15). CREATE_NO_WINDOW keeps it invisible; its stdout
        # still goes to bridge.log.
        popen_kwargs = {}
        if sys.platform == "win32":
            popen_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        self.proc = subprocess.Popen(cmd, stdout=self._logf,
                                     stderr=subprocess.STDOUT, **popen_kwargs)
        return self.proc

    def write_observation(self, obs):
        # video_grounding：视频侧状态的只读搬运。桥的 FileWorldObserver
        # 用 serde_json::Value 整文件通读，多余字段原样带过、不报错，所以
        # 加这个顶层字段在传输上是安全的。但内核策略代码目前不读任何视频
        # 字段——它只是让每一轮的 observation 都带上视频接地历史，以后有
        # 规则要用时不用再接线。它不改 outcome、step reward 或任何决策。
        if getattr(self, "video_grounding", None) is not None:
            obs = dict(obs)
            obs["video_grounding"] = self.video_grounding
        write_json(self.dir / "observation.json", obs)

    def write_outcome(self, step, action, reply, err, note=""):
        write_json(self.dir / "outcome.json", {
            "step": step, "action": action, "reply": reply,
            "grounded_error": err, "reward": 1.0 - err, "note": note,
        })

    def read_action(self, step):
        """Non-blocking: return the kernel's action for this step, or None.

        Returns the validated action (whitelist only); the raw action is
        kept in self.last_raw_action so an off-whitelist choice can be
        noted instead of silently executed.
        """
        try:
            act = json.loads((self.dir / "action.json").read_text(encoding="utf-8"))
        except Exception:
            return None
        if act.get("step") != step:
            return None
        raw = act.get("action", "noop")
        self.last_raw_action = raw
        return raw if raw in WHITELIST else "noop"

    def wait_action(self, step, timeout=ACTION_TIMEOUT_SECS):
        """Blocking version (headless tests): poll until the action arrives."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            action = self.read_action(step)
            if action is not None:
                return action
            if not self.alive():
                return None
            time.sleep(0.2)
        return None

    def record_step(self, step, action, err, reply=""):
        """GUI-side audit row, appended incrementally so a crash still
        leaves a trail (the bridge keeps its own audit.jsonl too)."""
        row = {"step": step, "action": action, "grounded_error": err,
               "reward": 1.0 - err, "reply": reply[:80]}
        self.steps_log.append(row)
        with open(self.dir / "gui_kernel_steps.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return row

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def stop(self):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except Exception:
                self.proc.kill()
        if self._logf is not None:
            try:
                self._logf.close()
            except Exception:
                pass
            self._logf = None

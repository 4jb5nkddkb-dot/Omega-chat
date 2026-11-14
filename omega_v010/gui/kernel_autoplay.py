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

The bridge binary is NOT bundled with the GUI. On Windows there is
no bridge binary that has ever run on real Windows, so none is
shipped: with no bridge next to the GUI, kernel auto-play reports
that instead of silently falling back to a fixed rotation (which no
longer exists). On the workbench, point OMEGA_BRIDGE at the Linux
bridge binary to run it.
"""
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
WHITELIST = ["observe", "explore", "exploit", "stabilize", "noop"]
BRIDGE_ENV = "OMEGA_BRIDGE"
ACTION_TIMEOUT_SECS = 300


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
        self._logf = open(self.dir / "bridge.log", "w", encoding="utf-8")
        cmd = [self.bridge, "--dir", str(self.dir),
               "--steps", str(self.steps), "--goal", self.goal]
        if Path(self.config).is_file():
            cmd += ["--config", self.config]
        print(f"kernel driver: {' '.join(cmd)}", flush=True)
        self.proc = subprocess.Popen(cmd, stdout=self._logf,
                                     stderr=subprocess.STDOUT)
        return self.proc

    def write_observation(self, obs):
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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Omega v0.1.0 Python sidecar: the world + the brain.

Loads the latest generalist product, holds the little world
(world_state.WorldState) and one SelfState, and executes the kernel's
chosen actions. Each step writes observation.json (world summary +
last model reply + grounded error) BEFORE outcome.json (grounded
error + reward), so the Rust bridge never reads a stale observation.

Action mapping (kernel action -> world/model behaviour):
  observe   -> ask a probe question (rotate Q0..Q3), score vs world
  explore   -> move to the next location, narrate arrival, check
               the narration for wrong-location mentions
  exploit   -> take a visible item if any, else talk to a present
               character; score the reply against the world
  stabilize -> ask the location question again, score vs world
  noop      -> nothing; grounded error 0

Grounded error is 0.0 (reply consistent with the world) or 1.0
(contradicted it), from world_state.validate() and the probe
score() in world_self_probe.py. Reward = 1 - grounded error.
"""
import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKBENCH = HERE.parent
sys.path.insert(0, str(WORKBENCH))
sys.path.insert(0, os.path.expanduser("~/workspace/omega-chat"))
sys.path.insert(0, os.path.expanduser("~/workspace/omega-chat/tools"))
sys.path.insert(0, os.path.expanduser("~/workspace/dual_side/xiyouji_modern_trainer_1"))

import world_anchor as wa
import world_loop as wl
import world_state as ws
import world_self_probe as wsp

LOC_ROTATION = ["花果山", "水帘洞", "高老庄", "流沙河", "取经路", "南海", "天宫"]


def latest_product():
    paths = [p for p in glob.glob(str(WORKBENCH / "out" / "generalist_s*.npz"))
             if "_last" not in p]
    if not paths:
        raise FileNotFoundError("no generalist product in out/")
    return max(paths, key=lambda p: int(p.split("_s")[-1].split(".")[0]))


def write_json(path, obj):
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(tmp, path)


class Sidecar:
    def __init__(self, run_dir, ckpt, persona, intent, max_new):
        self.dir = Path(run_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.max_new = max_new
        ckpt = ckpt or latest_product()
        self.model, self.tok, self.step, self.arch = wl.load_world_model(ckpt)
        print(f"model: {ckpt} step={self.step} arch={self.arch}", flush=True)
        self.state = ws.WorldState()
        self.me = wa.SelfState()
        if persona:
            self.me.set_persona(persona)
        if intent:
            self.me.set_intent(intent)
        self.last_reply = ""
        self.last_err = 0.0
        self.qi = 0
        self.loc_idx = 0

    def prompt(self, user_text):
        return wa.anchor_prompt(self.me, self.state) + "\n" + user_text

    def generate(self, prompt, max_new=None):
        ids = self.tok.encode(prompt, add_bos=True)
        if self.tok.UNK in ids:
            print("WARN: UNK in prompt", flush=True)
        try:
            out = self.model.generate_cached(ids, max_new or self.max_new,
                                             temperature=0.0)
        except Exception as e:
            print(f"generate_cached failed ({e}), falling back to generate",
                  flush=True)
            out = self.model.generate(ids, max_new_tokens=max_new or self.max_new,
                                      temperature=0.0)
        return self.tok.decode(out[len(ids):])

    def ask_question(self, qi):
        # Probe-exact format: summary + question sentence, no anchor
        # lines, no 问： wrapper. This is the format the world domain
        # was trained on (world_self_probe.py).
        q = wsp.QUESTIONS[qi]
        cont = self.generate(ws.world_prompt(self.state, q), 32)
        ok, note = wsp.score(self.state, qi, cont)
        return cont, (0.0 if ok else 1.0), note

    def narrate_arrival(self, dest):
        cont = self.generate(self.prompt(f"你来到了{dest}，说说这里。"), 16)
        flags = self.state.validate(cont)
        return cont, (1.0 if flags else 0.0), "; ".join(flags)

    def talk(self, name):
        cont = self.generate(self.prompt(f"和{name}说说话。"))
        flags = self.state.validate(cont)
        return cont, (1.0 if flags else 0.0), "; ".join(flags)

    def observation(self, step):
        return {
            "step": step,
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
            "world_summary": self.state.summary(),
        }

    def execute(self, action, step):
        """Run one kernel action; return (reply, grounded_error, note)."""
        if action == "observe":
            qi = self.qi % len(wsp.QUESTIONS)
            self.qi += 1
            return self.ask_question(qi)
        if action == "stabilize":
            # Same question pool as observe, rotating: otherwise the
            # dominant action repeats one question forever and the
            # world never gets asked anything else.
            qi = self.qi % len(wsp.QUESTIONS)
            self.qi += 1
            return self.ask_question(qi)
        if action == "explore":
            cur = self.state.visitor_location
            for _ in range(len(LOC_ROTATION)):
                self.loc_idx = (self.loc_idx + 1) % len(LOC_ROTATION)
                dest = LOC_ROTATION[self.loc_idx]
                if dest != cur:
                    break
            res = self.state.move(dest)
            cont, err, note = self.narrate_arrival(dest)
            return f"{res}{cont}", err, note
        if action == "exploit":
            visible = self.state.present_items()
            if visible:
                res = self.state.take(visible[0])
                err = 0.0 if "拾起了" in res else 1.0
                return res, err, ""
            present = self.state.present_characters()
            if present:
                return self.talk(present[0])
            return self.ask_question(1)
        return "", 0.0, "noop"

    def run(self, steps):
        write_json(self.dir / "observation.json", self.observation(0))
        print("sidecar ready: initial observation written", flush=True)
        last_seen = 0
        for step in range(1, steps + 1):
            act_path = self.dir / "action.json"
            action = None
            t0 = time.time()
            while time.time() - t0 < 600:
                try:
                    act = json.loads(act_path.read_text(encoding="utf-8"))
                    if act.get("step") == step:
                        action = act.get("action", "noop")
                        break
                except Exception:
                    pass
                time.sleep(0.2)
            if action is None:
                print(f"step {step}: no action.json in time; stopping", flush=True)
                break
            t1 = time.time()
            reply, err, note = self.execute(action, step)
            self.last_reply = reply
            self.last_err = err
            obs = self.observation(step)
            write_json(self.dir / "observation.json", obs)
            write_json(self.dir / "outcome.json", {
                "step": step,
                "action": action,
                "reply": reply,
                "grounded_error": err,
                "reward": 1.0 - err,
                "note": note,
            })
            print(f"step {step} act={action:<9} err={err:.0f} {note} "
                  f"({time.time() - t1:.1f}s) reply={reply[:40]!r}", flush=True)
        print("sidecar done", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="run")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--persona", default="omega")
    ap.add_argument("--intent", default="")
    ap.add_argument("--max-new", type=int, default=24)
    args = ap.parse_args()
    ckpt = args.ckpt or latest_product()
    sc = Sidecar(args.dir, ckpt, args.persona, args.intent, args.max_new)
    sc.run(args.steps)


if __name__ == "__main__":
    main()

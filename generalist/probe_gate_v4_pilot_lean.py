#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lean gate-split probe for a gated checkpoint (p_* only, no optimizer m/v).

Usage: python3 probe_gate_v4_pilot_lean.py [checkpoint.npz]
Default checkpoint: out/gated_v4_pilot_s3.npz (the v4 3-step pilot product).
Loads ONLY the p_* parameter arrays, builds the model once, and probes
gate_values() in batches of 2. Read-only; never touches any trainer.
Pass criteria (v4): left-domain mean g > 0.6 AND world mean g < 0.4.
"""
import json
import os
import sys

sys.path.insert(0, os.path.expanduser("~/workspace/dual_side/xiyouji_modern_trainer_1"))
sys.path.insert(0, os.path.expanduser("~/workspace/omega-chat"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from dual_transformer_numpy import CharTokenizer, DualTransformerNP
import generalist_pilot_train as gpt

CKPT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "out/gated_v4_pilot_s3.npz")
SEQ = 128
N_WIN = 4
BATCH = 2


def main():
    with np.load(CKPT, allow_pickle=False) as data:
        meta = json.loads(bytes(data["meta_json"]).decode("utf-8"))
        step = int(meta["step"])
        last_loss = meta.get("extra", {}).get("last_train_loss")
        recipe = meta.get("extra", {}).get("recipe")
        cfg = meta["config"]
        print(f"checkpoint step={step} last_train_loss={last_loss} recipe={recipe}")
        print(f"config merge={cfg.get('merge_method')} d={cfg.get('d_model')}")
        model = DualTransformerNP(**cfg, dtype=np.dtype(meta["dtype"]))
        assert model.merge_method == "gated"
        for k in list(data.files):
            if k.startswith("p_"):
                model.params[k[2:]] = np.array(data[k], dtype=model.dtype)
        print("params loaded (p_* only, no optimizer state)")
    del data

    vocab_meta = json.loads(gpt.VOCAB_JSON.read_text(encoding="utf-8"))
    tok = CharTokenizer(vocab_meta["chars"])
    ids = gpt.domain_texts(tok)

    left = ("经典", "代码", "对话", "常识", "推理")
    results = {}
    for name, arr in ids.items():
        starts = np.linspace(0, len(arr) - SEQ - 1, N_WIN).astype(int)
        wins = np.stack([arr[s:s + SEQ] for s in starts])
        gs = []
        for i in range(0, N_WIN, BATCH):
            gs.append(model.gate_values(wins[i:i + BATCH]))
        g = np.concatenate(gs, axis=0)  # (N_WIN, S, 1)
        mean_g = float(g.mean())
        results[name] = mean_g
        side = "LEFT " if name in left else "RIGHT"
        print(f"{side} {name}: mean g={mean_g:.4f} (std {float(g.std()):.4f}) "
              f"n_tokens={g.size}", flush=True)

    left_gs = [results[n] for n in left]
    world_g = results["世界"]
    left_avg = float(np.mean(left_gs))
    print(f"LEFT mean g avg={left_avg:.4f} | WORLD mean g={world_g:.4f} "
          f"| separation={left_avg - world_g:.4f}")
    passed = left_avg > 0.6 and world_g < 0.4
    print(f"PASS CRITERIA (left > 0.6 and world < 0.4): {'PASS' if passed else 'FAIL'}")
    print(f"step={step} last_train_loss={last_loss}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lean gate-split probe for a v5 (hidden-state gate) checkpoint (p_* only, no optimizer m/v).

Usage: python3 probe_gate_v5_pilot_lean.py [checkpoint.npz]
Default checkpoint: out/gated_v5_pilot_s3.npz (the v5 3-step pilot product).
Loads ONLY the p_* parameter arrays, builds the model once, and probes
gate_values() in batches of 2. Read-only; never touches any trainer.
Also measures the gate-input magnitude ||h_pool|| per domain and
||mean_left - mean_world|| of the pooled stem hidden states — the v5 fix
stands or falls on the gate input being O(10) instead of v4's ||x_pool|| ~ 0.16.
Pass criteria (v5 trend): separation > 0.05 (left mean g minus world mean g).
Full pass (as before): left-domain mean g > 0.6 AND world mean g < 0.4.
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
    os.path.dirname(os.path.abspath(__file__)), "out/gated_v5_pilot_s3.npz")
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
        print(f"config merge={cfg.get('merge_method')} d={cfg.get('d_model')} "
              f"n_stem_blocks={cfg.get('n_stem_blocks')}")
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
    hpool_mean_vec = {}
    hpool_norm = {}
    for name, arr in ids.items():
        starts = np.linspace(0, len(arr) - SEQ - 1, N_WIN).astype(int)
        wins = np.stack([arr[s:s + SEQ] for s in starts])
        gs, hps = [], []
        for i in range(0, N_WIN, BATCH):
            _, _, _, _, _, _, gcache = model._forward(wins[i:i + BATCH])
            gs.append(gcache["g"])
            hps.append(gcache["h_pool"])
        g = np.concatenate(gs, axis=0)  # (N_WIN, S, 1)
        hp = np.concatenate(hps, axis=0)  # (N_WIN, 1, D)
        mean_g = float(g.mean())
        results[name] = mean_g
        hpool_mean_vec[name] = hp.mean(axis=0).reshape(-1)  # (D,)
        hpool_norm[name] = float(np.linalg.norm(hp, axis=-1).mean())
        side = "LEFT " if name in left else "RIGHT"
        print(f"{side} {name}: mean g={mean_g:.4f} (std {float(g.std()):.4f}) "
              f"||h_pool||={hpool_norm[name]:.4f} n_tokens={g.size}", flush=True)

    left_gs = [results[n] for n in left]
    world_g = results["世界"]
    left_avg = float(np.mean(left_gs))
    mean_left_vec = np.mean([hpool_mean_vec[n] for n in left], axis=0)
    mean_world_vec = hpool_mean_vec["世界"]
    d_norm = float(np.linalg.norm(mean_left_vec - mean_world_vec))
    hpool_avg = float(np.mean([hpool_norm[n] for n in left + ("世界",)]))
    print(f"LEFT mean g avg={left_avg:.4f} | WORLD mean g={world_g:.4f} "
          f"| separation={left_avg - world_g:.4f}")
    print(f"||h_pool|| avg={hpool_avg:.4f} (v4 ||x_pool|| was ~0.158) "
          f"| ||mean_left - mean_world||={d_norm:.4f} (v4 was 0.1541)")
    print(f"PASS CRITERIA (left > 0.6 and world < 0.4): "
          f"{'PASS' if (left_avg > 0.6 and world_g < 0.4) else 'FAIL'}")
    print(f"TREND BAR (separation > 0.05): "
          f"{'PASS' if (left_avg - world_g) > 0.05 else 'FAIL'}")
    print(f"step={step} last_train_loss={last_loss}")


if __name__ == "__main__":
    main()

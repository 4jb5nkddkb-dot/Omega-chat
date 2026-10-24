"""Protocol SFT data for the tool / memory protocol, built from the real tools.

Every sample is one short dialogue:  question -> tool call -> tool result -> answer.
Tool results are produced by actually running calc_zh and the memory tools, so the
training data and the runtime cannot disagree about formats.

Output: a JSONL file. Each record has
  text   : the full sample (question + call + result + answer)
  spans  : [start, end) character offsets of the assistant-written parts (call and answer)
Characters are one token each with CharTokenizer, so spans map directly to token positions.

Usage:
  python make_tool_data.py                      # writes protocol_sft.jsonl
  python make_tool_data.py --n 400 --seed 1
  python make_tool_data.py --test
"""
from __future__ import annotations

import argparse
import json
import random
import tempfile
from pathlib import Path

import numpy as np

from agent_runtime import Protocol, ToolRouter, calc_zh, load_modern_model
from memory_store import MemoryStore, make_memory_tools

P = Protocol()
BASE = Path(__file__).resolve().parent

DIGITS = "一二三四五六七八九"          # calc only handles single digits
OPS = [("加", "加"), ("乘", "乘")]    # question word, calc word (both non-negative results)
FACTS = [("座右铭", "慢慢来"), ("生日", "三月五日"), ("绿茶", "不加糖")]
GREETINGS = [("问：你好？", "你好。")]  # no-tool sample: answer directly, no call


def _cn(n: int) -> str:
    return "一二三四五六七八九"[n - 1]


def calc_samples(rng: random.Random, n: int):
    templates = ["问：{a}{op}{b}等于几？", "问：{a}{op}{b}是多少？", "问：算一下{a}{op}{b}。"]
    out = []
    for _ in range(n):
        a, b = rng.randint(1, 9), rng.randint(1, 9)
        if rng.random() < 0.5:
            op_word, expr_word = "加", "加"
            value = a + b
        else:
            op_word, expr_word = "乘", "乘"
            value = a * b
        if value > 9 * 9:            # keep results single-character-friendly
            continue
        q = rng.choice(templates).format(a=_cn(a), op=op_word, b=_cn(b))
        call = f"{P.call_open}计算{P.name_end}{_cn(a)}{expr_word}{_cn(b)}{P.call_close}"
        result = calc_zh(f"{_cn(a)}{expr_word}{_cn(b)}")
        answer = f"答案是{result}。"
        out.append((q, call, f"{P.result_open}{result}{P.result_close}", answer))
    return out


def remember_recall_samples(rng: random.Random, n: int):
    out = []
    for _ in range(n):
        key, value = rng.choice(FACTS)
        with tempfile.TemporaryDirectory() as d:
            tools = make_memory_tools(MemoryStore(Path(d, "m.jsonl")))
            result_store = tools["记住"](f"{key}是{value}。")
            result_recall = tools["回忆"](key)
        q_store = f"请记住：{key}是{value}。"
        call_store = f"{P.call_open}记住{P.name_end}{key}是{value}。{P.call_close}"
        out.append((q_store, call_store, f"{P.result_open}{result_store}{P.result_close}", "记下了。"))

        q_recall = f"问：{key}是什么？"
        call_recall = f"{P.call_open}回忆{P.name_end}{key}{P.call_close}"
        out.append((q_recall, call_recall, f"{P.result_open}{result_recall}{P.result_close}",
                    f"答案是{value}。"))
    return out


def build_samples(n_calc=200, n_mem=100, seed=0):
    rng = random.Random(seed)
    items = calc_samples(rng, n_calc) + remember_recall_samples(rng, n_mem)
    rng.shuffle(items)
    records = []
    for q, call, result, answer in items:
        text = q + call + result + answer
        call_start = len(q)
        answer_start = len(q) + len(call) + len(result)
        spans = [[call_start, call_start + len(call)], [answer_start, len(text)]]
        records.append({"text": text, "spans": spans})
    # No-tool samples: the model should answer directly.
    for q, a in GREETINGS:
        records.append({"text": q + a, "spans": [[len(q), len(q) + len(a)]]})
    return records


def load_tokenizer():
    _, tok, _ = load_modern_model(BASE / "modern_ckpt.npz", BASE / "Xiyouji.txt")
    return tok


def check_records(records, tok):
    """Every record must encode without UNK, and every call must dispatch to its result."""
    router = ToolRouter({"计算": calc_zh, "记住": lambda a: "", "回忆": lambda a: ""}, tok)
    problems = []
    for r in records:
        ids = tok.encode(r["text"])
        if any(i == dtn_unk() for i in ids):
            problems.append(("unk", r["text"]))
            continue
        for s, e in r["spans"]:
            seg = r["text"][s:e]
            if seg.startswith(P.call_open):
                if router.dispatch(seg) is None:
                    problems.append(("call", seg))
    return problems


def dtn_unk():
    import dual_transformer_numpy as dtn
    return dtn.CharTokenizer.UNK


def build_stream(records, tok, mix_text_ids=None, seed=0):
    """Token ids and a loss mask (1 = assistant-written token). Records are separated by EOS."""
    rng = random.Random(seed)
    ids, mask = [], []
    order = list(range(len(records)))
    rng.shuffle(order)
    for idx in order:
        r = records[idx]
        t = tok.encode(r["text"], add_bos=True, add_eos=True)
        m = np.zeros(len(t), dtype=np.float32)
        for s, e in r["spans"]:
            m[1 + s:1 + e] = 1.0          # +1 for BOS
        m[-1] = 1.0                       # EOS is assistant-side too
        ids.extend(t)
        mask.extend(m.tolist())
    return np.asarray(ids, dtype=np.int64), np.asarray(mask, dtype=np.float32)


def test():
    tok = load_tokenizer()
    records = build_samples(n_calc=60, n_mem=20, seed=3)
    problems = check_records(records, tok)
    assert not problems, problems[:5]
    # Spans are correct: the call span is exactly a dispatchable call.
    kinds = {"call": 0, "answer": 0}
    for r in records:
        for s, e in r["spans"]:
            seg = r["text"][s:e]
            kinds["call" if seg.startswith(P.call_open) else "answer"] += 1
    assert kinds["call"] > 0 and kinds["answer"] > 0
    ids, mask = build_stream(records, tok, seed=1)
    assert ids.shape == mask.shape and mask.sum() > 0 and mask.sum() < len(mask)
    # The masked span must hold exactly the call text (one token per character).
    r = next(x for x in records if P.call_open in x["text"])
    s, e = r["spans"][0]
    t = tok.encode(r["text"], add_bos=True, add_eos=True)
    assert tok.decode(t[1 + s:1 + e]) == r["text"][s:e], "span does not line up with tokens"
    print(f"Tool data test OK: {len(records)} records, {ids.size} tokens, "
          f"assistant fraction {mask.mean():.2%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300, help="calc samples (memory samples are n/2)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="protocol_sft.jsonl")
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()
    if args.test:
        test()
        return
    tok = load_tokenizer()
    records = build_samples(n_calc=args.n, n_mem=max(1, args.n // 2), seed=args.seed)
    problems = check_records(records, tok)
    if problems:
        raise SystemExit(f"{len(problems)} records failed checks, first: {problems[0]}")
    with open(BASE / args.out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(records)} records to {args.out}")


if __name__ == "__main__":
    main()

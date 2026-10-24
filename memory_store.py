"""Long-term memory for the agent: an append-only JSONL store with TF-IDF recall.

Features are character bigrams, so Chinese and English work without a vocabulary.
The model reaches memory through two tools (`remember`, `recall`) on the normal tool
protocol, so memories are loaded on demand instead of being pasted into every context.

Caveat: recalled text is data returned inside <result> tags. It can still contain
misleading or adversarial instructions, so the model must not treat it as commands.
"""
from __future__ import annotations

import collections
import json
import math
import time
from pathlib import Path

# Characters reserved by the tool protocol (see agent_runtime.Protocol.reserved_chars()).
RESERVED_CHARS = "《》】〔〕（）"


def _grams(text: str) -> list[str]:
    """Character unigrams and bigrams: no vocabulary needed, works for Chinese and English."""
    t = "".join(text.lower().split())
    uni = list(t)
    bi = [t[i:i + 2] for i in range(len(t) - 1)]
    return uni + bi


class MemoryStore:
    def __init__(self, path, max_items: int = 10000, max_chars: int = 500,
                 reserved: str = RESERVED_CHARS):
        self.path = Path(path)
        self.reserved = reserved
        self.max_items, self.max_chars = max_items, max_chars
        self.records: list[dict] = []
        self.skipped = 0          # corrupt lines found on load
        self._next_id = 1
        self._dirty = True
        if self.path.exists():
            with self.path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        self.skipped += 1
                        continue
                    if not isinstance(rec, dict) or not isinstance(rec.get("text"), str) \
                            or not isinstance(rec.get("id"), int):
                        self.skipped += 1
                        continue
                    self.records.append(rec)
                    self._next_id = max(self._next_id, rec["id"] + 1)

    def add(self, text: str) -> int:
        text = text.strip()
        if not text:
            raise ValueError("empty memory")
        if len(text) > self.max_chars:
            raise ValueError(f"memory longer than {self.max_chars} characters")
        if any(c in text for c in self.reserved):
            raise ValueError("memory contains a character reserved by the tool protocol")
        for rec in self.records:
            if rec["text"] == text:
                return rec["id"]              # duplicates are not written twice
        if len(self.records) >= self.max_items:
            raise RuntimeError("memory is full")
        rec = {"id": self._next_id, "ts": round(time.time(), 3), "text": text}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
        self._next_id += 1
        self.records.append(rec)
        self._dirty = True
        return rec["id"]

    def _build(self):
        df = collections.Counter()
        counts = []
        for rec in self.records:
            c = collections.Counter(_grams(rec["text"]))
            counts.append(c)
            df.update(c.keys())
        n = len(self.records)
        self._idf = {g: math.log((1 + n) / (1 + d)) + 1.0 for g, d in df.items()}
        self._doc_vecs = [self._vec(c) for c in counts]
        self._dirty = False

    def _vec(self, counts) -> dict:
        # Bigrams are more discriminative than single characters, so unigrams get half weight.
        w = {g: (1.0 + math.log(tf)) * self._idf[g] * (0.5 if len(g) == 1 else 1.0)
             for g, tf in counts.items() if g in self._idf}
        norm = math.sqrt(sum(v * v for v in w.values())) or 1.0
        return {g: v / norm for g, v in w.items()}

    def search(self, query: str, k: int = 3, min_score: float = 0.18) -> list[dict]:
        if not self.records:
            return []
        if self._dirty:
            self._build()
        qv = self._vec(collections.Counter(_grams(query)))
        scored = []
        for i, dv in enumerate(self._doc_vecs):
            s = sum(v * dv.get(g, 0.0) for g, v in qv.items())
            if s > min_score:
                scored.append((s, i))
        scored.sort(key=lambda t: (-t[0], -t[1]))   # best score first, newer wins ties
        return [dict(self.records[i], score=round(s, 4)) for s, i in scored[:k]]


_CN = "零一二三四五六七八九"


def _cn_int(n: int) -> str:
    """Digit-by-digit Chinese numeral, so tool output stays inside the corpus vocabulary."""
    return "".join(_CN[int(d)] for d in str(n))


def make_memory_tools(store: MemoryStore, k: int = 3, remember_name: str = "记住",
                      recall_name: str = "回忆") -> dict:
    """Tool callables for ToolRouter: name -> f(argument) -> text. Names are corpus characters."""
    def remember(arg: str) -> str:
        return f"记住了第{_cn_int(store.add(arg))}条"

    def recall(arg: str) -> str:
        hits = store.search(arg, k=k)
        if not hits:
            return "没有找到记录"
        return "；".join(f"第{_cn_int(h['id'])}条：{h['text']}" for h in hits)
    return {remember_name: remember, recall_name: recall}


# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------

def memory_test():
    import tempfile
    from agent_runtime import Protocol, ScriptedLM, ToolRouter, contains_verifier, \
        load_modern_model, run_agent, calc_zh
    base = Path(__file__).resolve().parent
    _, tok, _ = load_modern_model(base / "modern_ckpt.npz", base / "Xiyouji.txt")
    p = Protocol()
    assert set(RESERVED_CHARS) == set(p.reserved_chars()), "memory and protocol must reserve the same characters"

    with tempfile.TemporaryDirectory() as d:
        path = Path(d, "memory.jsonl")
        store = MemoryStore(path)
        a = store.add("用户喜欢喝绿茶。")
        c = store.add("用户的生日是三月五日。")
        b = store.add("项目的截止日期是十一月十五日。")
        assert store.add("用户喜欢喝绿茶。") == a                       # duplicate
        assert len(path.read_text(encoding="utf-8").splitlines()) == 3

        hits = store.search("绿茶怎么样", k=2)
        assert hits and hits[0]["id"] == a, hits
        assert store.search("截止日期")[0]["id"] == b
        assert store.search("天气怎么样") == []

        # Persistence across instances; corrupt lines are skipped and counted.
        with path.open("a", encoding="utf-8") as f:
            f.write("not json\n{\"no_text\": 1}\n")
        reloaded = MemoryStore(path)
        assert reloaded.skipped == 2 and len(reloaded.records) == 3
        assert reloaded.search("生日")[0]["id"] == c
        assert reloaded.add("新的记忆。") == 4                            # ids continue

        for bad in ("", "x" * 600, "记住《这个》"):
            try:
                reloaded.add(bad)
                raise AssertionError(f"accepted bad memory {bad[:20]!r}")
            except ValueError:
                pass

        # End to end with the trainer's model and vocabulary: session 1 remembers,
        # session 2 (new store, new router) recalls.
        mem_path = Path(d, "s2.jsonl")
        r1 = ToolRouter({"计算": calc_zh, **make_memory_tools(MemoryStore(mem_path))}, tok)
        call_remember = f"{p.call_open}记住{p.name_end}座右铭是慢慢来。{p.call_close}"
        lm1 = ScriptedLM(tok, [call_remember, "记下了"])
        res1 = run_agent(lm1, r1, "问：", contains_verifier("记下"))
        assert res1.ok, res1.events
        assert any("记住了" in e[2] for e in res1.events if e[0] == "tool"), res1.events
        assert MemoryStore(mem_path).records, "remember tool did not persist"

        store2 = MemoryStore(mem_path)
        r2 = ToolRouter({"计算": calc_zh, **make_memory_tools(store2)}, tok)
        call_recall = f"{p.call_open}回忆{p.name_end}座右铭{p.call_close}"
        lm2 = ScriptedLM(tok, [call_recall, "座右铭是慢慢来"])
        res2 = run_agent(lm2, r2, "问：", contains_verifier("慢慢来"))
        assert res2.ok, res2.events
        assert any("慢慢来" in e[2] for e in res2.events if e[0] == "tool"), res2.events
        assert "没有找到" not in " ".join(e[2] for e in res2.events if e[0] == "tool")
        print(f"Memory test OK: {len(store.records)} records, {reloaded.skipped} corrupt lines skipped, "
              f"cross-session recall works")


if __name__ == "__main__":
    import sys
    if "--test" in sys.argv:
        memory_test()
        print("MEMORY TESTS PASSED")

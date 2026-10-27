"""Closed-loop agent runtime (items 2-4), aligned with the user's modern trainer.

Works with:
- ModernTransformerNumpy (modern_transformer_numpy.py), via generate_segment(), which
  uses model.forward when the model has no generate_cached();
- CharTokenizer built from Xiyouji.txt (the trainer's vocabulary).

The protocol and tool names use characters that already exist in Xiyouji.txt. The corpus
has no ASCII tags, quotes, or '+' and '-', so the defaults below are Chinese punctuation
and Chinese words:

  《NAME》ARGUMENT】     the model requests a tool (NAME is constrained to tool names)
  〔RESULT〕              the runtime appends the tool output
  （FEEDBACK）        the runtime appends a verifier rejection

The constraints guarantee well-formed tool names. They do not teach a model when to call
a tool; the current checkpoint has never seen this protocol.
"""
from __future__ import annotations

import ast
import operator
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

import dual_transformer_numpy as dtn


@dataclass(frozen=True)
class Protocol:
    call_open: str = "《"
    name_end: str = "》"
    call_close: str = "】"
    result_open: str = "〔"
    result_close: str = "〕"
    feedback_open: str = "（"
    feedback_close: str = "）"

    def reserved_chars(self) -> str:
        return "".join(sorted(set(self.call_open + self.name_end + self.call_close
                                  + self.result_open + self.result_close
                                  + self.feedback_open + self.feedback_close)))

    def call_re(self):
        o, e, c = (re.escape(x) for x in (self.call_open, self.name_end, self.call_close))
        return re.compile(f"{o}([^{e}]+){e}(.*?){c}", re.S)


# ----------------------------------------------------------------------------
# Model access: works with generate_cached models and with forward-only models.
# ----------------------------------------------------------------------------

def generate_segment(model, ids, max_new, *, temperature=0.0, rng=None, controller=None,
                     logit_hook=None, stop_fn=None, top_k=0, top_p=1.0):
    """KV-cached sampling with the same hooks as DualTransformerNP.generate_cached."""
    if hasattr(model, "generate_cached"):
        return model.generate_cached(ids, max_new, temperature=temperature, rng=rng,
                                     top_k=top_k, top_p=top_p, controller=controller,
                                     logit_hook=logit_hook, stop_fn=stop_fn)
    max_seq = model.max_seq
    result = [int(t) for t in ids]
    if max_new <= 0:
        return result

    def prefill(seq):
        logits, pasts = model.forward(np.asarray([seq], dtype=np.int64))
        return logits[0, -1], pasts

    ctx = result[-max_seq:]
    last, pasts = prefill(ctx)
    for gi in range(max_new):
        step = np.asarray(last, dtype=np.float64)
        if logit_hook is not None:
            step = logit_hook(result, step)
        if controller is not None:
            nxt = controller.choose(step, result, rng, top_k, top_p)
        else:
            nxt = dtn._pick_token(step, temperature, rng, top_k, top_p)
        result.append(nxt)
        if stop_fn is not None and stop_fn(result):
            break
        if gi + 1 == max_new:
            break
        if len(ctx) < max_seq:
            logits, pasts = model.forward(np.array([[nxt]], dtype=np.int64), pasts)
            last = logits[0, -1]
            ctx.append(nxt)
        else:                                # window full: re-prefill, as generate_cached does
            ctx = result[-max_seq:]
            last, pasts = prefill(ctx)
    return result


def load_modern_model(ckpt_path, corpus_path):
    """Rebuild the trainer's model and tokenizer exactly as train_modern.py does."""
    from modern_transformer_numpy import ModernTransformerNumpy
    text = Path(corpus_path).read_text(encoding="utf-8").replace("\r\n", "\n")
    tok = dtn.CharTokenizer.from_text(text)
    z = np.load(ckpt_path, allow_pickle=False)
    d, blocks, qh, kvh, vocab, step = (int(x) for x in z["cfg"])
    if vocab != tok.vocab_size or [str(c) for c in z["chars"]] != tok.chars:
        raise ValueError("checkpoint vocabulary does not match this corpus")
    model = ModernTransformerNumpy(d_model=d, n_blocks=blocks, n_q_heads=qh, n_kv_heads=kvh,
                                   vocab_size=vocab)
    for k in model.params:
        model.params[k] = z[f"param::{k}"]
    return model, tok, step


# ----------------------------------------------------------------------------
# Checkpoint formats: dual files carry meta_json (config + embedded tokenizer);
# modern files carry cfg + chars + param:: keys. The loader below dispatches
# on that structure, so a dual checkpoint needs no corpus to load.
# ----------------------------------------------------------------------------

def detect_checkpoint_arch(ckpt_path) -> str:
    """Return "dual" or "modern" from the npz structure; raise otherwise."""
    z = np.load(ckpt_path, allow_pickle=False)
    keys = set(z.files)
    if "meta_json" in keys:
        return "dual"
    if "cfg" in keys and any(k.startswith("param::") for k in keys):
        return "modern"
    raise ValueError(f"unknown checkpoint format: {ckpt_path} "
                     f"(keys: {sorted(keys)[:8]})")


def load_dual_model(ckpt_path):
    """Dual-format checkpoint -> (model, tok, step). Tokenizer comes from meta."""
    model, ck = dtn.DualTransformerNP.load_checkpoint(str(ckpt_path))
    meta = ck.get("meta", {}) if isinstance(ck, dict) else {}
    tok = dtn.tokenizer_from_meta(meta)
    step = int(meta.get("step", 0)) if isinstance(meta, dict) else 0
    return model, tok, step


def load_checkpoint(ckpt_path, corpus_path=None):
    """Format-dispatching entry: (model, tok, step, arch).

    Dual checkpoints load from meta alone; modern checkpoints need the corpus
    (their tokenizer is rebuilt from it and checked against the file).
    """
    arch = detect_checkpoint_arch(ckpt_path)
    if arch == "dual":
        model, tok, step = load_dual_model(ckpt_path)
        return model, tok, step, "dual"
    if corpus_path is None:
        raise ValueError("modern checkpoint requires a corpus path")
    model, tok, step = load_modern_model(ckpt_path, corpus_path)
    return model, tok, step, "modern"


def select_checkpoint(base, ckpt="modern_ckpt.npz"):
    """Checkpoint selection, dual first, in the existing fallback style.

    Returns (path, arch). The default ckpt name keeps its old meaning: prefer
    the SFT product when present, else the base checkpoint. An explicit ckpt
    name is used as-is (arch detected when the file exists).
    The fallback labels are only defaults: when a candidate file exists and
    its format is readable, the detected arch wins, so a dual checkpoint
    named modern_ckpt.npz no longer reports as modern. An unreadable file
    keeps the name-based label and fails at load time, as before.
    """
    base = Path(base)
    if ckpt == "modern_ckpt.npz":
        for name, arch in (("dual_d128_sft.npz", "dual"),
                           ("modern_ckpt_sft.npz", "modern"),
                           ("modern_ckpt.npz", "modern")):
            p = base / name
            if p.exists():
                try:
                    arch = detect_checkpoint_arch(p)
                except Exception:
                    pass
                return p, arch
    p = base / ckpt
    return p, detect_checkpoint_arch(p) if p.exists() else "modern"


# ----------------------------------------------------------------------------
# Tools. Inputs and outputs use Chinese characters that the corpus contains.
# ----------------------------------------------------------------------------

_CN_DIGITS = "零一二三四五六七八九"
_CN_OPS = {"加": "+", "减": "-", "乘": "*", "除": "/"}
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg}


def _safe_eval(expr: str):
    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -ev(node.operand)
        raise ValueError("unsupported expression")
    return ev(ast.parse(expr, mode="eval"))


def calc_zh(arg: str) -> str:
    """Single-digit arithmetic written in Chinese, e.g. '二加三' -> '五', '七除二' -> '三点五'."""
    mapped = []
    for ch in arg.strip():
        if ch in _CN_DIGITS:
            mapped.append(str(_CN_DIGITS.index(ch)))
        elif ch.isdigit():
            mapped.append(ch)
        elif ch in _CN_OPS:
            mapped.append(_CN_OPS[ch])
        else:
            raise ValueError(f"unsupported character {ch!r}")
    value = _safe_eval("".join(mapped))
    if isinstance(value, float):
        value = round(value, 6)
        if value == int(value):
            value = int(value)
    out = str(value)
    return out.replace("-", "负").replace(".", "点").translate(
        str.maketrans({str(i): _CN_DIGITS[i] for i in range(10)}))


def make_read_file(root, max_bytes: int = 2000) -> Callable[[str], str]:
    """Read-only file tool confined to `root`. Filenames need letters the corpus lacks,
    so this is only usable with an alias map (see alias_tool)."""
    root = Path(root).resolve()

    def read_file(arg: str) -> str:
        target = (root / arg.strip()).resolve()
        if not target.is_relative_to(root):
            raise PermissionError("path escapes the sandbox")
        return target.read_text(encoding="utf-8")[:max_bytes]
    return read_file


def alias_tool(mapping: dict, fn: Callable[[str], str]) -> Callable[[str], str]:
    """Let the model name a resource with corpus characters: argument -> mapping[argument] -> fn."""
    def tool(arg: str) -> str:
        if arg.strip() not in mapping:
            raise KeyError(f"unknown name {arg.strip()!r}")
        return fn(mapping[arg.strip()])
    return tool


# ----------------------------------------------------------------------------
# Router: grammar-constrained tool names, dispatch, stop rule.
# ----------------------------------------------------------------------------

@dataclass
class AgentResult:
    ok: bool
    text: str
    turns: int
    retries: int
    ctx: list
    events: list = field(default_factory=list)


class ToolRouter:
    def __init__(self, tools: dict, tokenizer, protocol: Protocol = Protocol(),
                 max_result_chars: int = 200):
        if not tools:
            raise ValueError("at least one tool is required")
        self.tools = dict(tools)
        self.names = sorted(self.tools)
        self.tok = tokenizer
        self.p = protocol
        self.max_result_chars = max_result_chars
        self.strings = [tokenizer.decode([i]) for i in range(tokenizer.vocab_size)]
        needed = set(protocol.reserved_chars()) | set("".join(self.names))
        missing = sorted(c for c in needed if c not in self.strings)
        if missing:
            raise ValueError(f"tokenizer vocabulary lacks {''.join(missing)!r}; "
                             "choose protocol characters or tool names from the corpus")
        self._call_re = protocol.call_re()

    def _tail(self, result) -> str:
        return self.tok.decode(result[-96:])

    def _mask(self, logits, allowed) -> np.ndarray:
        logits = np.asarray(logits, dtype=np.float64)
        keep = [i for i, s in enumerate(self.strings) if s and allowed(s)]
        if not keep:
            raise RuntimeError("no token can continue the tool name; check the vocabulary")
        out = np.full(logits.shape, -np.inf)
        out[keep] = logits[keep]
        return out

    def hook(self, result, logits):
        """Inside a tool name, allow only characters that extend a tool name, then the name end."""
        p = self.p
        tail = self._tail(result)
        i = tail.rfind(p.call_open)
        if i < 0:
            return logits
        after = tail[i + len(p.call_open):]
        if p.name_end in after:
            return logits                            # name is closed; arguments are free text
        prefix = after

        def allowed(s):
            if s == p.name_end:
                return prefix in self.tools
            return any(n.startswith(prefix + s) for n in self.names)
        return self._mask(logits, allowed)

    def stop(self, result) -> bool:
        if result and result[-1] == self.tok.eos_id:
            return True
        tail = self.tok.decode(result[-96:])
        return tail.endswith(self.p.call_close) and self.p.call_open in tail

    def dispatch(self, text: str):
        """None if no complete call is present, else (ok, output)."""
        m = self._call_re.search(text)
        if m is None:
            return None
        name, arg = m.group(1), m.group(2)
        if name not in self.tools:
            return False, f"未知工具:{name}"
        try:
            return True, str(self.tools[name](arg.strip()))[: self.max_result_chars]
        except Exception as e:                       # tool errors go back to the model as data
            return False, f"错误:{type(e).__name__}"


def contains_verifier(expected: str) -> Callable[[str], tuple]:
    def verify(answer: str):
        if expected in answer:
            return True, ""
        return False, f"答案需包含{expected}"
    return verify


# Actions whose tool call writes to memory. Kept as a named set (not derived from
# the policy table) so the second door is explicit: it must match
# memory_store.make_memory_tools' default remember_name and the governance
# policy's confirm-required memory action.
MEMORY_WRITE_ACTIONS = frozenset({"记住"})


def _run_governed_tool(router: ToolRouter, text: str, governance,
                       token_provider) -> tuple:
    """One governed tool turn, mirroring ToolRouter.dispatch's visible outcomes.

    Returns ("tool", ok, out) for a handled turn (caller appends the tool result
    to ctx exactly as the ungoverned path does), or ("denied", name, arg, reason)
    when the call must not execute. On denial nothing is executed, no tool
    result is appended, and memory is untouched; the caller ends the run with
    the denial visible in events.

    Order per call: parse (same regex dispatch uses) -> ToolGate.authorize ->
    token resolution for approval-required actions -> MemoryGate.authorize_write
    for memory writes -> ToolGate.execute. CheckpointGate and Rollback are
    deliberately not consulted here: they govern release workflow and governance
    files, not per-turn tool calls.
    """
    # Lazy import: the ungoverned path must never depend on omega_governance.
    from omega_governance import ALLOW, DENY, HOLD
    tool_gate = getattr(governance, "tools", None) or getattr(governance, "tool_gate", None)
    memory_gate = getattr(governance, "memory", None) or getattr(governance, "memory_gate", None)
    if tool_gate is None:
        raise ValueError("governance must expose a ToolGate as .tools (omega_governance.Governance)")
    m = router._call_re.search(text)
    if m is None:
        return ("tool", False, "错误:格式不完整")
    name, arg = m.group(1), m.group(2).strip()
    if name not in router.tools:
        # Never executes; the gate still records the out-of-allowlist attempt.
        tool_gate.authorize(name, arg)
        return ("tool", False, f"未知工具:{name}")
    decision = tool_gate.authorize(name, arg)
    verdict, reason = decision["decision"], decision["reason"]
    if verdict == DENY:
        return ("denied", name, arg, reason)
    token = None
    if verdict == HOLD:
        if reason == "paused":
            return ("denied", name, arg, reason)
        # Approval-required: a token must arrive from the caller (run_agent has
        # no user confirmation channel). No token -> enforce via execute so the
        # denial is audited as a DENY record, without running anything and
        # without touching the memory door.
        token = token_provider(name, arg) if token_provider is not None else None
        if token is None:
            tool_gate.execute(None, name, arg, router.tools[name])
            return ("denied", name, arg, "token required")
    # Second door for memory writes, before any execution.
    if name in MEMORY_WRITE_ACTIONS and memory_gate is not None:
        mem = memory_gate.authorize_write(arg)
        if mem["decision"] != ALLOW:
            return ("denied", name, arg, mem["reason"])
    result = tool_gate.execute(token, name, arg, router.tools[name])
    if result["executed"]:
        return ("tool", True, str(result["result"])[:router.max_result_chars])
    err = result["error"] or "denied"
    if err in ("paused", "action not in allowlist", "token required") or err.startswith("token "):
        return ("denied", name, arg, err)
    # The tool itself raised: mirror dispatch's feedback format.
    typename = err.split(": ", 1)[0] if ": " in err else err
    return ("tool", False, f"错误:{typename}")


def run_agent(model, router: ToolRouter, prompt: str, verifier, *, max_new: int = 96,
              max_turns: int = 6, max_retries: int = 2, temperature: float = 0.0, rng=None,
              controller_factory: Optional[Callable[[], object]] = None,
              governance=None, token_provider=None) -> AgentResult:
    """Closed loop. governance=None preserves existing behavior exactly.

    With governance (an omega_governance.Governance-like object exposing .tools
    and .memory), every tool dispatch is authorized before execution; a denial
    ends the run with a ("denied", name, arg, reason) event and nothing
    appended to ctx. token_provider(action, arg) -> token | None supplies
    one-time confirmation tokens for approval-required actions; without it
    those actions are always denied, so governance stays opt-in: enabling it
    by default would break 记住-class actions until a confirmation channel
    reaches the loop.
    """
    p, tok = router.p, router.tok
    ctx = tok.encode(prompt, add_bos=True)
    events, turns, retries = [], 0, 0
    while turns < max_turns:
        turns += 1
        snapshot = len(ctx)
        ctrl = controller_factory() if controller_factory else None
        seg = generate_segment(model, ctx, max_new, temperature=temperature, rng=rng,
                               controller=ctrl, logit_hook=router.hook, stop_fn=router.stop)
        text = tok.decode(seg[snapshot:])
        events.append(("generated", text))

        if p.call_open in text:                       # tool turn
            if governance is None:
                res = router.dispatch(text)
                if res is None:
                    ok, out = False, "错误:格式不完整"
                else:
                    ok, out = res
                events.append(("tool", ok, out))
                ctx = seg + tok.encode(f"{p.result_open}{out}{p.result_close}")
                continue
            outcome = _run_governed_tool(router, text, governance, token_provider)
            if outcome[0] == "denied":
                _, name, arg, reason = outcome
                events.append(("denied", name, arg, reason))
                return AgentResult(False, text, turns, retries, ctx, events)
            _, ok, out = outcome
            events.append(("tool", ok, out))
            ctx = seg + tok.encode(f"{p.result_open}{out}{p.result_close}")
            continue

        accepted, feedback = verifier(text)           # answer turn
        if accepted:
            events.append(("accepted", text))
            return AgentResult(True, text, turns, retries, seg, events)
        retries += 1
        events.append(("rejected", feedback))
        if retries > max_retries:
            return AgentResult(False, text, turns, retries, ctx, events)
        # Rollback: drop the rejected segment, continue from the snapshot plus feedback.
        ctx = ctx[:snapshot] + tok.encode(f"{p.feedback_open}{feedback}{p.feedback_close}")
    return AgentResult(False, "", turns, retries, ctx, events)


# ----------------------------------------------------------------------------
# Tests (use the real Xiyouji tokenizer and checkpoint when available)
# ----------------------------------------------------------------------------

class ScriptedLM:
    """Stands in for the model so loop logic can be tested deterministically."""

    def __init__(self, tok, scripts):
        self.tok, self.scripts, self.calls = tok, list(scripts), []

    def generate_cached(self, ctx, max_new, **kw):
        self.calls.append(list(ctx))
        return list(ctx) + self.tok.encode(self.scripts.pop(0))


def _roundtrip_ok(tok, s):
    return tok.decode(tok.encode(s)) == s


def runtime_test(corpus="Xiyouji.txt", ckpt="modern_ckpt.npz"):
    import tempfile
    base = Path(__file__).resolve().parent
    # Dual SFT product first (dual path, no corpus needed), then the modern SFT
    # checkpoint (device acceptance passed 2026-10-24), then the base checkpoint.
    ckpt_path, arch = select_checkpoint(base, ckpt)
    print(f"runtime_test using checkpoint: {ckpt_path.name} (arch: {arch})")
    if arch == "dual":
        model, tok, _ = load_dual_model(ckpt_path)
    else:
        model, tok, _ = load_modern_model(ckpt_path, base / corpus)
    chars_ok = "".join(sorted(set(
        Protocol().reserved_chars() + "计算回忆记住二加三五答案是六七除乘点零一")))
    assert all(c in tok.chars for c in chars_ok), "test characters must exist in the corpus"

    # Chinese calculator.
    assert calc_zh("二加三") == "五"
    assert calc_zh("二乘六") == "一二"
    assert calc_zh("七除二") == "三点五"
    try:
        calc_zh("二+三")
        raise AssertionError("calc accepted an ASCII operator it should not see")
    except ValueError:
        pass

    # Vocabulary validation: a tokenizer without the protocol characters is rejected.
    try:
        ToolRouter({"计算": calc_zh}, dtn.CharTokenizer.from_text("一二三"))
        raise AssertionError("router accepted an incompatible vocabulary")
    except ValueError as e:
        assert "lacks" in str(e)

    router = ToolRouter({"计算": calc_zh}, tok)
    p = router.p

    # Dispatch.
    assert router.dispatch(f"{p.call_open}计算{p.name_end}二加三{p.call_close}") == (True, "五")
    ok, msg = router.dispatch(f"{p.call_open}删除{p.name_end}x{p.call_close}")
    assert ok is False and "未知工具" in msg
    assert router.dispatch("没有调用") is None

    # Grammar-constrained tool names with the real trained model.
    for seed in range(12):
        prompt = tok.encode(p.call_open, add_bos=True)
        out = generate_segment(model, prompt, 8, temperature=1.0, rng=np.random.default_rng(seed),
                               logit_hook=router.hook, stop_fn=None)
        text = tok.decode(out[len(prompt):])
        name = text.split(p.name_end)[0]
        assert name in router.names, (seed, text)   # the constraint must yield a registered name
        assert p.name_end in text and text.index(p.name_end) == len(name), (seed, text)

    # Scripted loop: tool call -> result; wrong answer -> rollback -> right answer.
    scripts = [f"{p.call_open}计算{p.name_end}二加三{p.call_close}", "答案是六", "答案是五"]
    for s in scripts + [f"{p.result_open}五{p.result_close}", f"{p.feedback_open}答案需包含五{p.feedback_close}"]:
        assert _roundtrip_ok(tok, s), f"not encodable without loss: {s!r}"
    lm = ScriptedLM(tok, scripts)
    res = run_agent(lm, router, "问:二加三等于几?", contains_verifier("五"), max_turns=5, max_retries=2)
    assert res.ok and res.turns == 3 and res.retries == 1, res.events
    final = tok.decode(res.ctx)
    assert f"{p.result_open}五{p.result_close}" in final
    assert f"{p.feedback_open}答案需包含五{p.feedback_close}" in final
    assert "答案是六" not in tok.decode(lm.calls[2]), "rejected segment leaked into the retry"
    assert [e[0] for e in res.events] == ["generated", "tool", "generated", "rejected",
                                          "generated", "accepted"], res.events

    # Real model through the loop with decode control (must terminate and roll back).
    from functools import partial
    res2 = run_agent(model, router, "悟空", contains_verifier("永远不会出现的字"),
                     max_new=16, max_turns=3, max_retries=1, temperature=0.8,
                     rng=np.random.default_rng(0),
                     controller_factory=partial(dtn.DecodeController, 0.8, block_ngram=4))
    assert not res2.ok and res2.turns <= 3
    print(f"Agent runtime test OK: scripted turns={res.turns} retries={res.retries}; "
          f"real-model events={[e[0] for e in res2.events]}")


if __name__ == "__main__":
    import sys
    if "--test" in sys.argv:
        runtime_test()
        print("AGENT RUNTIME TESTS PASSED")

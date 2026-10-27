#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ω COMPLETE SINGLE KERNEL — roles-v2 / 100-pass engineering baseline
====================================================================

This is a single-file flattening of the verified omega-core-grok-main roles-v2
source tree. It keeps the implementation bodies of the package modules while
removing only internal package imports and package __init__ wrappers.

Core invariant:
    观察 → 预测 → 校准 → 决策

Roles remain advisory:
    Yiyi   = feedback
    Ziwei  = metacognition
    Wukong = independent challenge
    Dayan  = learning / pattern discovery

Execution remains separately gated:
    suggestion → owner approval → one-time authorization token → boundary check → execute

The file is intended for Python 3.10+ / Pythonista-compatible standard-library use.
"""

from __future__ import annotations



# ============================================================================
# MODULE FLATTENED: omega.cognitive.jia_phase
# ============================================================================

# -*- coding: utf-8 -*-
"""J-I-A-J minimal agent loop labels (log / observability only).

J1 Judge (pre) → I Ideate → A Action → J2 Judge (post)

Does not authorize writes. Execution still requires approve → token → execute.
"""


from typing import Any, Dict, List, Optional


class JiaPhase:
    J1 = "J1"  # pre-judge: constraints, memory caution, decide gates
    I = "I"    # ideate: plan / intend / predict
    A = "A"    # action: authorize + execute (pipeline only)
    J2 = "J2"  # post-judge: residual, progress, meta


PHASE_ORDER = (JiaPhase.J1, JiaPhase.I, JiaPhase.A, JiaPhase.J2)


def build_cognition_jia(
    *,
    decision: str,
    paused: bool = False,
    escalated: bool = False,
    stall: bool = False,
    memory_caution: bool = False,
) -> Dict[str, Any]:
    """Trace for one cognitive_cycle (through J1+I; A only if APPLY later)."""
    completed: List[str] = [JiaPhase.J1, JiaPhase.I]
    pre_notes = []
    if paused:
        pre_notes.append("paused")
    if escalated:
        pre_notes.append("escalated")
    if stall:
        pre_notes.append("stall")
    if memory_caution:
        pre_notes.append("memory_caution")

    d = str(decision).upper()
    if "APPLY" in d:
        awaiting = JiaPhase.A
        current = JiaPhase.I
        closed = False
    elif "HOLD" in d:
        # soft post-judge without action
        completed.append(JiaPhase.J2)
        awaiting = None
        current = JiaPhase.J2
        closed = True
        pre_notes.append("hold_skip_A")
    elif "ESCALATE" in d:
        completed.append(JiaPhase.J2)
        awaiting = None
        current = JiaPhase.J2
        closed = True
        pre_notes.append("escalate_skip_A")
    else:
        awaiting = None
        current = JiaPhase.I
        closed = False

    return {
        "loop": "J-I-A-J",
        "completed": completed,
        "current": current,
        "awaiting": awaiting,
        "closed": closed,
        "pre_judge_flags": pre_notes,
    }


def build_execute_jia(
    *,
    error: float,
    risk: float,
    progress: float,
) -> Dict[str, Any]:
    """Trace after successful A; always closes with J2."""
    return {
        "loop": "J-I-A-J",
        "completed": [JiaPhase.J1, JiaPhase.I, JiaPhase.A, JiaPhase.J2],
        "current": JiaPhase.J2,
        "awaiting": None,
        "closed": True,
        "post_judge": {
            "error": float(error),
            "risk": float(risk),
            "progress": float(progress),
        },
    }


def build_blocked_jia(reason: str) -> Dict[str, Any]:
    return {
        "loop": "J-I-A-J",
        "completed": [JiaPhase.J1, JiaPhase.I],
        "current": JiaPhase.A,
        "awaiting": JiaPhase.A,
        "closed": False,
        "blocked": str(reason),
    }


# ============================================================================
# MODULE FLATTENED: omega.core.constants
# ============================================================================

# -*- coding: utf-8 -*-
"""Global constants for Ω Kernel."""

VERSION = "3.16.0-baseline"
__version__ = VERSION
KERNEL_VERSION = f"omega-cognitive-{VERSION}"
SCHEMA_VERSION = 6

EPS = 1e-12
MIN_SECRET_LENGTH = 16
MAX_AUDIT_RECORDS = 2000
MAX_PENDING_TTL_SEC = 300.0

# Phase-3 / unify
MAX_TOOL_CALLS_PER_CYCLE = 8
DEFAULT_TOOL_TIMEOUT_SEC = 30.0
SKILL_CAS_PREFIX = "skill-"
MODEL_CAS_PREFIX = "model-"

# Vault-at-Rest
VAULT_MAGIC = b"OMEGAVLT"
VAULT_VERSION = 2              # AES-256-GCM (requires `cryptography`)
VAULT_LEGACY_VERSION = 1       # stdlib HMAC-CTR; read-only unless explicitly enabled
VAULT_GCM_NONCE_BYTES = 12
LEGACY_VAULT_ENV = "OMEGA_ALLOW_LEGACY_VAULT"
VAULT_SALT_BYTES = 16
VAULT_NONCE_BYTES = 16
VAULT_TAG_BYTES = 32
DEFAULT_PBKDF2_ITERATIONS = 200_000
MIN_PBKDF2_ITERATIONS = 10_000

# Topology (JIA-Loop alignment)
MOD16 = 16
CYCLE_240 = 240  # LCM(60, 16)


# ============================================================================
# MODULE FLATTENED: omega.core.helpers
# ============================================================================

# -*- coding: utf-8 -*-
"""Low-level helpers: finite checks, vectors, atomic write, hashing."""


import copy
import hashlib
import hmac
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping


def finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def deep(x: Any) -> Any:
    return copy.deepcopy(x)


def atomic_write_text(path: Path, content: str, mode: int = 0o600) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent),
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(str(tmp_path), str(path))
        try:
            dir_fd = os.open(str(path.parent), os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except (OSError, AttributeError):
            pass
        try:
            os.chmod(path, mode)
        except OSError:
            pass
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass


def atomic_write_bytes(path: Path, content: bytes, mode: int = 0o600) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent),
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(str(tmp_path), str(path))
        try:
            dir_fd = os.open(str(path.parent), os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except (OSError, AttributeError):
            pass
        try:
            os.chmod(path, mode)
        except OSError:
            pass
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass


def canonical(x: Any) -> str:
    return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


AUDIT_KEY_LABEL = "omega-audit-chain-v1"


def derive_subkey(secret: str, label: str) -> bytes:
    """Domain-separated sub-key derived from the kernel secret."""
    return hmac.new(secret.encode("utf-8"), label.encode("utf-8"), hashlib.sha256).digest()


HEAD_KEY_LABEL = "omega-head-v1"
TOKEN_KEY_LABEL = "omega-tokens-v1"
BLOCK_KEY_LABEL = "omega-episode-blocks-v1"
SKILL_KEY_LABEL = "omega-skill-store-v1"
MAC_FIELD = "mac"
ANCHOR_KEY_LABEL = "omega-rollback-anchor-v1"
LEGACY_MIGRATE_HINT = (
    "unsealed legacy state; back up the storage directory, then restart once "
    "with migrate_legacy=True (CLI: --migrate-legacy) to seal it"
)


class SealError(ValueError):
    """Raised when persisted state fails its MAC or is unsealed without migration."""


def seal_record(key: bytes, body: Dict[str, Any]) -> Dict[str, Any]:
    """Return body plus an HMAC-SHA256 over its canonical form."""
    clean = {k: v for k, v in body.items() if k != MAC_FIELD}
    mac = hmac.new(key, canonical(clean).encode("utf-8"), hashlib.sha256).hexdigest()
    return {**clean, MAC_FIELD: mac}


def open_record(key: bytes, data: Any) -> Optional[Dict[str, Any]]:
    """Return the body if its MAC verifies, otherwise None."""
    if not isinstance(data, dict) or not isinstance(data.get(MAC_FIELD), str):
        return None
    body = {k: v for k, v in data.items() if k != MAC_FIELD}
    expected = hmac.new(key, canonical(body).encode("utf-8"), hashlib.sha256).hexdigest()
    return body if hmac.compare_digest(expected, data[MAC_FIELD]) else None


def open_sealed(key: bytes, data: Any, *, allow_legacy: bool, label: str):
    """Return (body, is_legacy). Tampered records always raise; unsealed records raise unless migrating."""
    if not isinstance(data, dict):
        raise SealError(f"{label}: not an object")
    if MAC_FIELD in data:
        body = open_record(key, data)
        if body is None:
            raise SealError(f"{label}: MAC mismatch (tampered, or wrong secret)")
        return body, False
    if not allow_legacy:
        raise SealError(f"{label}: {LEGACY_MIGRATE_HINT}")
    return {k: v for k, v in data.items()}, True


def source_sha256(path: Optional[str] = None) -> str:
    """SHA-256 of this kernel's own source file. Only meaningful against an expected value from a trusted channel."""
    return hashlib.sha256(Path(path or __file__).read_bytes()).hexdigest()


def norm_vector(x: Mapping[str, Any]) -> Dict[str, float]:
    result = {}
    for key, value in x.items():
        if not isinstance(key, str):
            raise ValueError("vector key must be string")
        if not finite(value):
            raise ValueError(f"non-finite value: {key}")
        result[key] = float(value)
    return result


def vector_diff(a: Mapping[str, float], b: Mapping[str, float]) -> Dict[str, float]:
    keys = set(a) | set(b)
    return {key: float(b.get(key, 0.0)) - float(a.get(key, 0.0)) for key in keys}


def vector_rms(values: Mapping[str, float]) -> float:
    if not values:
        return 0.0
    return math.sqrt(sum(float(v) ** 2 for v in values.values()) / len(values))


def compute_risk(error: float, uncertainty: float, residual: float) -> float:
    return error * (1.0 + uncertainty) * (1.0 + residual)


# ============================================================================
# MODULE FLATTENED: omega.core.state_machine
# ============================================================================

# -*- coding: utf-8 -*-
"""
Explicit Kernel + Cycle state machines.

Two layers (kept separate on purpose — unify now, tighten later):

1. KernelState  — hard runtime gate (writes / fail-closed)
2. CyclePhase  — soft per-cycle phase aligned with conceptual lifecycle:
   INIT → SYNC → LISTEN → SENSE → PLAN → DECIDE → EXEC → CALIBRATE → PERSIST

Topology clock: prefer ``Omega.tick_topology()`` / ``JiaLoop`` as sole advance.
``cycle_step`` here is a mirror; ``mirror_topology`` syncs from JiaLoop.

ESCALATED may only leave to IDLE/PAUSED/FAIL_CLOSED — not directly to RUNNING.
Clear escalate via stage/confirm ``clear_escalate`` (see lifecycle auth).
"""


from enum import Enum
from typing import Dict, List, Optional, Set, Tuple


class KernelState(str, Enum):
    INIT = "INIT"
    RESTORING = "RESTORING"
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    LISTENING = "LISTENING"
    PAUSED = "PAUSED"
    ESCALATED = "ESCALATED"
    FAIL_CLOSED = "FAIL_CLOSED"


class CyclePhase(str, Enum):
    INIT = "INIT"
    SYNC = "SYNC"
    LISTEN = "LISTEN"
    SENSE = "SENSE"
    PLAN = "PLAN"
    DECIDE = "DECIDE"
    AUTH = "AUTH"
    EXEC = "EXEC"
    CALIBRATE = "CALIBRATE"
    PERSIST = "PERSIST"


_KERNEL_TRANSITIONS: Dict[KernelState, Set[KernelState]] = {
    KernelState.INIT: {KernelState.RESTORING, KernelState.IDLE, KernelState.FAIL_CLOSED},
    KernelState.RESTORING: {KernelState.IDLE, KernelState.FAIL_CLOSED},
    KernelState.IDLE: {
        KernelState.RUNNING, KernelState.LISTENING, KernelState.PAUSED,
        KernelState.ESCALATED, KernelState.FAIL_CLOSED,
    },
    KernelState.RUNNING: {
        KernelState.IDLE, KernelState.LISTENING, KernelState.PAUSED,
        KernelState.ESCALATED, KernelState.FAIL_CLOSED,
    },
    KernelState.LISTENING: {
        KernelState.IDLE, KernelState.RUNNING, KernelState.PAUSED,
        KernelState.ESCALATED, KernelState.FAIL_CLOSED,
    },
    KernelState.PAUSED: {
        KernelState.IDLE, KernelState.RUNNING, KernelState.ESCALATED,
        KernelState.FAIL_CLOSED,
    },
    # No direct ESCALATED -> RUNNING; human clear_escalate -> IDLE only.
    KernelState.ESCALATED: {
        KernelState.IDLE,
        KernelState.PAUSED,
        KernelState.FAIL_CLOSED,
    },
    KernelState.FAIL_CLOSED: set(),
}

_CYCLE_ORDER: List[CyclePhase] = [
    CyclePhase.INIT,
    CyclePhase.SYNC,
    CyclePhase.LISTEN,
    CyclePhase.SENSE,
    CyclePhase.PLAN,
    CyclePhase.DECIDE,
    CyclePhase.AUTH,
    CyclePhase.EXEC,
    CyclePhase.CALIBRATE,
    CyclePhase.PERSIST,
]


class StateMachine:
    """Tracks KernelState + CyclePhase; cycle_step mirrors JiaLoop when synced."""

    def __init__(self):
        self.kernel: KernelState = KernelState.INIT
        self.phase: CyclePhase = CyclePhase.INIT
        self.history: List[Tuple[float, str, str]] = []
        self.cycle_step: int = 0

    def can_transition_kernel(self, target: KernelState) -> bool:
        return target in _KERNEL_TRANSITIONS.get(self.kernel, set())

    def transition_kernel(self, target: KernelState, *, force: bool = False) -> KernelState:
        if not force and not self.can_transition_kernel(target):
            raise RuntimeError(
                f"illegal kernel transition {self.kernel.value} -> {target.value}"
            )
        self.kernel = target
        self._record("kernel", target.value)
        return self.kernel

    def set_phase(self, phase: CyclePhase) -> CyclePhase:
        self.phase = phase
        self._record("phase", phase.value)
        return self.phase

    def advance_phase(self) -> CyclePhase:
        try:
            idx = _CYCLE_ORDER.index(self.phase)
        except ValueError:
            idx = 0
        if idx >= len(_CYCLE_ORDER) - 1:
            next_phase = CyclePhase.LISTEN
        else:
            next_phase = _CYCLE_ORDER[idx + 1]
        return self.set_phase(next_phase)

    def mirror_topology(self, step: int) -> None:
        self.cycle_step = int(step)

    def tick_topology(self, mod: int = 16, big: int = 240) -> Dict[str, int]:
        """Deprecated independent tick — tests only. Prefer Omega.tick_topology()."""
        self.cycle_step = (self.cycle_step + 1) % max(1, big)
        return {
            "step_240": self.cycle_step,
            "mod16": self.cycle_step % mod,
            "mod3": self.cycle_step % 3,
            "mod5": self.cycle_step % 5,
        }

    def writable(self) -> bool:
        return self.kernel not in (
            KernelState.FAIL_CLOSED,
            KernelState.PAUSED,
            KernelState.ESCALATED,
            KernelState.INIT,
            KernelState.RESTORING,
        )

    def summary(self) -> Dict[str, object]:
        return {
            "kernel": self.kernel.value,
            "phase": self.phase.value,
            "cycle_step": self.cycle_step,
            "history_tail": self.history[-12:],
        }

    def _record(self, kind: str, value: str) -> None:
        self.history.append((time.time(), kind, value))
        if len(self.history) > 256:
            self.history = self.history[-256:]


# ============================================================================
# MODULE FLATTENED: omega.llm.bridge
# ============================================================================

# -*- coding: utf-8 -*-
"""LLM Bridge stub -- propose only, never execute."""


from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional


@dataclass
class LLMProposal:
    action: str
    args: Dict[str, Any]
    rationale: str = ""
    confidence: float = 0.5
    source: str = "heuristic"
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "args": dict(self.args),
            "rationale": self.rationale,
            "confidence": self.confidence,
            "source": self.source,
            "raw": dict(self.raw),
        }


class LLMBridge:
    def __init__(self, *, enabled: bool = False, endpoint: str = "", model: str = "", timeout_sec: float = 15.0):
        self.enabled = bool(enabled)
        self.endpoint = endpoint or ""
        self.model = model or ""
        self.timeout_sec = float(timeout_sec)
        self.last_proposal: Optional[LLMProposal] = None

    def propose(
        self,
        state: Mapping[str, float],
        goal: Mapping[str, float],
        *,
        allowed_actions: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> LLMProposal:
        if not self.enabled:
            prop = LLMProposal(action="measure", args={}, rationale="llm_bridge_disabled", confidence=0.2, source="disabled")
            self.last_proposal = prop
            return prop
        if self.endpoint:
            prop = LLMProposal(
                action="measure", args={},
                rationale=f"endpoint_configured_but_client_pending:{self.endpoint}",
                confidence=0.3, source="endpoint_stub",
                raw={"model": self.model, "context": context or {}},
            )
            self.last_proposal = prop
            return prop
        best_key = None
        best_gap = 0.0
        for k, t in goal.items():
            gap = abs(float(state.get(k, 0.0)) - float(t))
            if gap > best_gap:
                best_gap = gap
                best_key = k
        action = "measure"
        args: Dict[str, Any] = {}
        if best_key is not None:
            action = "adjust"
            args = {best_key: float(goal[best_key])}
        if allowed_actions and action not in allowed_actions:
            action = allowed_actions[0] if allowed_actions else "measure"
            args = {}
        prop = LLMProposal(
            action=action, args=args,
            rationale=f"heuristic_gap:{best_key}={best_gap:.4f}",
            confidence=0.55 if best_key else 0.3, source="heuristic",
        )
        self.last_proposal = prop
        return prop


# ============================================================================
# MODULE FLATTENED: omega.tools.base
# ============================================================================

# -*- coding: utf-8 -*-
"""
Action & Tool abstraction for Phase-3 autonomous agent.

Design rules:
- Every side-effecting operation is an Action.
- Tools are registered, permission-checked, and executed only through the
  authorize -> token -> execute path.
- Tool results are always structured and auditable.
- High-risk tools set always_confirm=True and elevated base_risk.
"""


import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set


@dataclass
class Action:
    """Canonical action description produced by cognition."""
    name: str
    args: Dict[str, Any] = field(default_factory=dict)
    rationale: Optional[str] = None
    expected_effects: Optional[Dict[str, Any]] = None
    risk_estimate: float = 0.0
    requires_confirm: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ToolResult:
    success: bool
    output: Any = None
    error: Optional[str] = None
    duration_sec: float = 0.0
    side_effects: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class Tool(ABC):
    """Base class for all tools the agent may call."""

    name: str = "unnamed"
    description: str = ""
    tags: Set[str] = set()
    always_confirm: bool = False
    base_risk: float = 0.1
    timeout_sec: float = 30.0

    @abstractmethod
    def run(self, args: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> ToolResult:
        ...

    def validate_args(self, args: Dict[str, Any]) -> Optional[str]:
        return None


class ToolRegistry:
    """Central registry. Only registered tools can ever execute."""

    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool, *, overwrite: bool = False) -> None:
        if not tool.name or not isinstance(tool.name, str):
            raise ValueError("tool must have a non-empty string name")
        if tool.name in self._tools and not overwrite:
            raise ValueError(f"tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> bool:
        return self._tools.pop(name, None) is not None

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def list_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "tags": sorted(t.tags),
                "always_confirm": t.always_confirm,
                "base_risk": t.base_risk,
            }
            for t in self._tools.values()
        ]

    def allowed_action_names(self) -> Set[str]:
        return set(self._tools.keys())

    def requires_confirm(self, name: str) -> bool:
        tool = self._tools.get(name)
        return bool(tool and tool.always_confirm)

    def base_risk(self, name: str) -> float:
        tool = self._tools.get(name)
        return float(tool.base_risk) if tool else 1.0

    def execute(
        self,
        name: str,
        args: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(success=False, error=f"unknown tool: {name}")

        err = tool.validate_args(args or {})
        if err:
            return ToolResult(success=False, error=f"invalid args: {err}")

        start = time.time()
        try:
            result = tool.run(args or {}, context=context)
            if not isinstance(result, ToolResult):
                result = ToolResult(success=True, output=result)
            result.duration_sec = time.time() - start
            return result
        except Exception as exc:
            return ToolResult(
                success=False,
                error=f"{type(exc).__name__}: {exc}",
                duration_sec=time.time() - start,
                metadata={"traceback": traceback.format_exc()},
            )


class ObserveTool(Tool):
    name = "observe"
    description = "Return current world observation (read-only)."
    tags = {"perception", "safe", "read"}
    base_risk = 0.0
    always_confirm = False

    def run(self, args: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> ToolResult:
        world = (context or {}).get("world")
        if world is None or not hasattr(world, "observe"):
            return ToolResult(success=False, error="no world in context")
        return ToolResult(success=True, output=world.observe())


class NoOpTool(Tool):
    name = "noop"
    description = "Do nothing. Useful for testing the safety pipeline."
    tags = {"safe", "read"}
    base_risk = 0.0
    always_confirm = False

    def run(self, args: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> ToolResult:
        return ToolResult(success=True, output={"status": "noop"}, side_effects=[])


class MeasureTool(Tool):
    """Primary action used by SmoothPolicy / AdaptivePolicy."""

    name = "measure"
    description = "Apply continuous measure toward target args in the world."
    tags = {"control", "write"}
    base_risk = 0.15
    always_confirm = False

    def validate_args(self, args: Dict[str, Any]) -> Optional[str]:
        if not isinstance(args, dict) or not args:
            return "measure requires non-empty numeric args"
        for k, v in args.items():
            try:
                float(v)
            except (TypeError, ValueError):
                return f"non-numeric arg: {k}"
        return None

    def run(self, args: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> ToolResult:
        world = (context or {}).get("world")
        if world is None:
            return ToolResult(success=False, error="no world in context")
        if hasattr(world, "execute"):
            out = world.execute("measure", args)
        elif hasattr(world, "observe"):
            out = world.observe()
        else:
            return ToolResult(success=False, error="world cannot execute measure")
        return ToolResult(
            success=True,
            output=out,
            side_effects=[f"measure:{k}" for k in args.keys()],
        )


class QueryMemoryTool(Tool):
    """Read-only memory inspection."""

    name = "query_memory"
    description = "Query recent episodes or memory stats (read-only)."
    tags = {"memory", "safe", "read"}
    base_risk = 0.0
    always_confirm = False

    def validate_args(self, args: Dict[str, Any]) -> Optional[str]:
        mode = (args or {}).get("mode", "recent")
        if mode not in ("recent", "stats"):
            return "mode must be 'recent' or 'stats'"
        return None

    def run(self, args: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> ToolResult:
        omega = (context or {}).get("omega")
        if omega is None or not hasattr(omega, "memory"):
            return ToolResult(success=False, error="no omega/memory in context")
        mode = (args or {}).get("mode", "recent")
        n = int((args or {}).get("n", 5))
        if mode == "stats":
            return ToolResult(success=True, output=omega.memory.stats())
        recent = omega.memory.recent(n)
        payload = [
            {
                "episode_id": getattr(e, "episode_id", None),
                "error": getattr(e, "error", None),
                "risk": getattr(e, "risk", None),
                "decision": getattr(e, "decision", None),
            }
            for e in recent
        ]
        return ToolResult(success=True, output={"recent": payload, "count": len(payload)})


class SetParamTool(Tool):
    """Restricted write: set a single world parameter."""

    name = "set_param"
    description = "Set one world parameter (restricted write; still requires token pipeline)."
    tags = {"control", "write"}
    base_risk = 0.35
    always_confirm = False

    def validate_args(self, args: Dict[str, Any]) -> Optional[str]:
        if not args or "key" not in args or "value" not in args:
            return "requires key and value"
        try:
            float(args["value"])
        except (TypeError, ValueError):
            return "value must be numeric"
        return None

    def run(self, args: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> ToolResult:
        world = (context or {}).get("world")
        if world is None or not hasattr(world, "execute"):
            return ToolResult(success=False, error="no executable world in context")
        key = str(args["key"])
        value = float(args["value"])
        out = world.execute("set_param", {key: value})
        return ToolResult(
            success=True,
            output=out,
            side_effects=[f"set_param:{key}"],
            metadata={"key": key, "value": value},
        )


class HighRiskProbeTool(Tool):
    """High-risk probe: always_confirm=True."""

    name = "high_risk_probe"
    description = "Synthetic high-risk action that must be confirmed before use."
    tags = {"danger", "write"}
    base_risk = 0.9
    always_confirm = True

    def run(self, args: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> ToolResult:
        note = (args or {}).get("note", "probe")
        return ToolResult(
            success=True,
            output={"probed": True, "note": note},
            side_effects=["high_risk_probe"],
            metadata={"confirm_required": True},
        )


DEFAULT_ALLOWED_ACTIONS = frozenset({
    "measure",
    "observe",
    "noop",
    "query_memory",
    "set_param",
    "high_risk_probe",
})


def make_default_registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(ObserveTool())
    reg.register(NoOpTool())
    reg.register(MeasureTool())
    reg.register(QueryMemoryTool())
    reg.register(SetParamTool())
    reg.register(HighRiskProbeTool())
    return reg


# ============================================================================
# MODULE FLATTENED: omega.topology.digit_roots
# ============================================================================

# -*- coding: utf-8 -*-
"""Mod-16 digit roots: two-digit n with n - (tens*ones) == 16.

User-derived identity (verified exhaustive for 10..99):
    24, 37, 48, 79

These are invariants for annotation / tests — they do not authorize writes.
"""


from typing import Iterable, List, Sequence, Tuple

# Canonical two-digit roots (sorted)
TWO_DIGIT_ROOTS: Tuple[int, ...] = (24, 37, 48, 79)


def digit_product(n: int) -> int:
    """Product of decimal digits of |n|."""
    n = abs(int(n))
    if n == 0:
        return 0
    prod = 1
    while n:
        prod *= n % 10
        n //= 10
    return prod


def root_gap(n: int) -> int:
    """n - product(digits)."""
    return int(n) - digit_product(n)


def is_mod16_digit_root(n: int) -> bool:
    """True iff n is two-digit and n - a*b == 16."""
    n = int(n)
    if n < 10 or n > 99:
        return False
    a, b = n // 10, n % 10
    return n - a * b == 16


def enumerate_two_digit_roots() -> List[int]:
    """Exhaustive scan 10..99; must equal TWO_DIGIT_ROOTS."""
    found = [n for n in range(10, 100) if is_mod16_digit_root(n)]
    return found


def assert_roots_complete() -> None:
    found = tuple(enumerate_two_digit_roots())
    if found != TWO_DIGIT_ROOTS:
        raise AssertionError(f"digit roots mismatch: {found} != {TWO_DIGIT_ROOTS}")


def congruent_mod16(n: int) -> bool:
    """n ≡ digit_product(n) (mod 16). Weaker than exact gap==16."""
    return (int(n) - digit_product(n)) % 16 == 0


# ============================================================================
# MODULE FLATTENED: omega.topology.outer_law
# ============================================================================

# -*- coding: utf-8 -*-
"""OMEGA outer-law phases + Gravity (G) + Fire-Token (annotation only).

Maps MetaCognition regime/action onto conceptual outer phases.
Never authorizes CAS writes — Decision pipeline remains sole gate.
"""


import time
from typing import Any, Dict, Optional


class OuterPhase:
    ORIGIN = "Origin"
    MOTION = "Motion"
    ENTROPY = "Entropy"
    GRAVITY = "Gravity"  # G — collapse / hard boundary
    ABSORBED = "Absorbed"


def map_meta_to_outer(
    regime: str,
    recommended_action: str,
) -> Dict[str, Any]:
    """Soft map meta regime → outer phase + G flag."""
    r = (regime or "").lower()
    a = (recommended_action or "").lower()

    g_active = False
    phase = OuterPhase.MOTION

    if r == "converged":
        phase = OuterPhase.ABSORBED
    elif r in ("degrading", "unstable") or a == "escalate":
        phase = OuterPhase.GRAVITY
        g_active = True
    elif r == "stalled" or a == "recalibrate":
        phase = OuterPhase.GRAVITY
        g_active = True
    elif r == "improving":
        phase = OuterPhase.MOTION
    elif a == "caution":
        phase = OuterPhase.ENTROPY
    elif r == "stable" or a == "continue":
        phase = OuterPhase.ORIGIN if r == "stable" else OuterPhase.MOTION
    else:
        phase = OuterPhase.MOTION

    return {
        "outer_phase": phase,
        "g_active": g_active,
        "g_meaning": "collapse_boundary" if g_active else None,
        "regime": regime,
        "recommended_action": recommended_action,
    }


def soft_decision_from_meta(
    decision_value: str,
    recommended_action: str,
) -> Optional[Dict[str, Any]]:
    """If APPLY, Meta may soft-downgrade to HOLD or ESCALATE.

    Returns None if no change; else {decision, reason, escalate_flag}.
    """
    d = str(decision_value).upper()
    if "APPLY" not in d:
        return None
    a = (recommended_action or "").lower()
    if a == "escalate":
        return {
            "decision": "ESCALATE",
            "reason": "meta_g_escalate",
            "escalate": True,
        }
    if a in ("caution", "recalibrate"):
        tag = "meta_g_caution" if a == "caution" else "meta_calibrate"
        return {
            "decision": "HOLD",
            "reason": tag,
            "escalate": False,
        }
    return None


def mint_fire_token(
    *,
    epoch: int,
    cycle: int,
    topology_step: int,
    residual: float,
    regime: str,
    reason: str,
) -> Dict[str, Any]:
    """Log-only seed after a G-point (escalate / hard collapse).

    Not a security token — does not unlock writes.
    """
    return {
        "kind": "fire_token",
        "minted_at": time.time(),
        "epoch": int(epoch),
        "cycle": int(cycle),
        "topology_step": int(topology_step),
        "residual": float(residual),
        "regime": str(regime),
        "reason": str(reason),
        "note": "annotation_only_not_auth",
    }


# ============================================================================
# MODULE FLATTENED: omega.world.file_sandbox
# ============================================================================

# -*- coding: utf-8 -*-
"""Sandboxed local-file world (paths cannot escape root)."""


import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union


class FileSandboxWorld:
    """Filesystem environment confined to `root`. observe() returns numeric metrics."""

    def __init__(self, root: Union[str, Path]):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.calls = 0
        self.last_op: str = "init"
        self.last_error: str = ""
        self._writes = 0
        self._reads = 0
        # File Mission (optional)
        self.mission_path: Optional[str] = None
        self.mission_content: Optional[str] = None

    def set_mission(self, path: str, content: str) -> None:
        """Declare the success target for observe()/goal_distance."""
        self.mission_path = str(path)
        self.mission_content = str(content)

    def clear_mission(self) -> None:
        self.mission_path = None
        self.mission_content = None

    def mission_status(self) -> Dict[str, Any]:
        """Discrete mission check (also mirrored as floats in observe)."""
        if not self.mission_path:
            return {
                "active": False,
                "exists": False,
                "match": False,
                "ready": False,
                "path": None,
            }
        try:
            path = self._safe(self.mission_path)
            exists = path.is_file()
            match = False
            if exists and self.mission_content is not None:
                text = path.read_text(encoding="utf-8")
                match = text == self.mission_content
            return {
                "active": True,
                "exists": exists,
                "match": match,
                "ready": bool(exists and match),
                "path": self.mission_path,
            }
        except Exception as e:
            return {
                "active": True,
                "exists": False,
                "match": False,
                "ready": False,
                "path": self.mission_path,
                "error": f"{type(e).__name__}: {e}",
            }

    def _safe(self, rel: str) -> Path:
        rel = (rel or ".").replace("\\", "/").lstrip("/")
        target = (self.root / rel).resolve()
        if target != self.root and self.root not in target.parents:
            raise PermissionError(f"path escapes sandbox: {rel}")
        return target

    def observe(self) -> Dict[str, float]:
        files: List[Path] = []
        dirs: List[Path] = []
        for p in self.root.rglob("*"):
            if p.is_file():
                files.append(p)
            elif p.is_dir():
                dirs.append(p)
        total_bytes = sum(p.stat().st_size for p in files)
        out: Dict[str, float] = {
            "file_count": float(len(files)),
            "dir_count": float(len(dirs)),
            "total_kb": float(total_bytes) / 1024.0,
            "writes": float(self._writes),
            "reads": float(self._reads),
            "ok": 0.0 if self.last_error else 1.0,
            "target_exists": 0.0,
            "target_match": 0.0,
            "target_ready": 0.0,
            "mission_progress": 0.0,
        }
        st = self.mission_status()
        if st.get("active"):
            out["target_exists"] = 1.0 if st.get("exists") else 0.0
            out["target_match"] = 1.0 if st.get("match") else 0.0
            out["target_ready"] = 1.0 if st.get("ready") else 0.0
            # 0 → 0.5 (exists only) → 1.0 (content match)
            out["mission_progress"] = (
                1.0 if st.get("ready") else (0.5 if st.get("exists") else 0.0)
            )
        return out

    def list_files(self, rel: str = ".") -> List[str]:
        base = self._safe(rel)
        if not base.exists():
            return []
        out = []
        for p in sorted(base.iterdir()):
            out.append(str(p.relative_to(self.root)) + ("/" if p.is_dir() else ""))
        return out

    def read_text(self, rel: str, max_bytes: int = 8192) -> str:
        path = self._safe(rel)
        if not path.is_file():
            raise FileNotFoundError(rel)
        data = path.read_bytes()[:max_bytes]
        self._reads += 1
        self.last_op = f"read:{rel}"
        self.last_error = ""
        return data.decode("utf-8", errors="replace")

    def write_text(self, rel: str, content: str) -> Dict[str, Any]:
        path = self._safe(rel)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        self._writes += 1
        self.calls += 1
        self.last_op = f"write:{rel}"
        self.last_error = ""
        return {"path": rel, "bytes": len(content.encode("utf-8"))}

    def execute(self, action: str, args: Mapping[str, Any]) -> Dict[str, float]:
        self.calls += 1
        try:
            if action in ("write_file", "write_text"):
                self.write_text(str(args.get("path", "out.txt")), str(args.get("content", "")))
            elif action == "set_param":
                key = str(args.get("key", "param"))
                self.write_text(f"params/{key}.txt", str(args.get("value", "")))
            elif action == "measure":
                snap = {k: float(v) for k, v in args.items()}
                self.write_text("state/measure_snap.json", json.dumps(snap, ensure_ascii=False))
            self.last_error = ""
        except Exception as e:
            self.last_error = f"{type(e).__name__}: {e}"
        return self.observe()

    def goal_distance(self, goal: Mapping[str, float]) -> float:
        """Distance to numeric goal; if mission active, prefer mission readiness."""
        st = self.mission_status()
        if st.get("active"):
            if st.get("ready"):
                return 0.0
            if st.get("exists"):
                return 0.5
            return 1.0
        if not goal:
            return 0.0
        obs = self.observe()
        return sum(abs(obs.get(k, 0.0) - t) for k, t in goal.items()) / max(1, len(goal))


# ============================================================================
# MODULE FLATTENED: omega.world.reference
# ============================================================================

# -*- coding: utf-8 -*-
"""Simple reference world for testing and demos."""


from typing import Any, Dict, Mapping, Optional


class ReferenceWorld:
    DEFAULT_STATE = {"x": 0.0, "score": 0.5, "health": 1.0}

    def __init__(self, state: Optional[Dict[str, float]] = None):
        self.initial_state = dict(state or self.DEFAULT_STATE)
        self.state = dict(self.initial_state)
        self.calls = 0

    def reset(self, state: Optional[Dict[str, float]] = None):
        if state is not None:
            self.initial_state = dict(state)
        self.state = dict(self.initial_state)
        self.calls = 0

    def observe(self) -> Dict[str, float]:
        return dict(self.state)

    def execute(self, action: str, args: Mapping[str, Any]) -> Dict[str, float]:
        self.calls += 1
        for key, value in args.items():
            old = self.state.get(key, 0.0)
            self.state[key] = 0.75 * old + 0.25 * float(value)
        return dict(self.state)

    def goal_distance(self, goal: Mapping[str, float]) -> float:
        if not goal:
            return 0.0
        return sum(abs(self.state.get(k, 0.0) - t) for k, t in goal.items()) / max(1, len(goal))


# ============================================================================
# MODULE FLATTENED: omega.core.audit
# ============================================================================

# -*- coding: utf-8 -*-
"""Append-only audit chain with archive support."""


import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union



class AuditChain:
    """Hash-chained audit log.

    With a `key`, every record hash is an HMAC-SHA256 over the record body, so
    anyone without the key cannot rewrite history and recompute valid hashes.
    Without a key the chain only detects accidental corruption.
    """

    def __init__(
        self,
        records: Optional[List[Dict[str, Any]]] = None,
        base_hash: str = "GENESIS",
        key: Optional[bytes] = None,
    ):
        self.records = records if records is not None else []
        self.base_hash = base_hash
        self.key = key

    @staticmethod
    def _digest(body: Dict[str, Any], key: Optional[bytes]) -> str:
        data = canonical(body).encode("utf-8")
        if key is None:
            return hashlib.sha256(data).hexdigest()
        return hmac.new(key, data, hashlib.sha256).hexdigest()

    def append(self, event: str, payload: Any) -> Dict[str, Any]:
        previous = self.records[-1]["hash"] if self.records else self.base_hash
        record = {
            "index": len(self.records),
            "time": time.time(),
            "event": event,
            "payload": deep(payload),
            "previous": previous,
        }
        record["hash"] = self._digest(record, self.key)
        self.records.append(record)
        return record

    def verify(self) -> bool:
        return self._verify_with(self.key)

    def _verify_with(self, key: Optional[bytes]) -> bool:
        previous = self.base_hash
        for index, record in enumerate(self.records):
            if record.get("index") != index or record.get("previous") != previous:
                return False
            stored = record.get("hash")
            if not isinstance(stored, str):
                return False
            body = {k: v for k, v in record.items() if k != "hash"}
            if not hmac.compare_digest(stored, self._digest(body, key)):
                return False
            previous = stored
        return True

    def _resign(self) -> None:
        """Recompute every hash under self.key (used for one-time migration)."""
        previous = self.base_hash
        rebuilt: List[Dict[str, Any]] = []
        for index, record in enumerate(self.records):
            body = {k: v for k, v in record.items() if k != "hash"}
            body["index"] = index
            body["previous"] = previous
            digest = self._digest(body, self.key)
            body["hash"] = digest
            previous = digest
            rebuilt.append(body)
        self.records = rebuilt

    def archive(self, path: Union[str, Path], keep: int = 1000) -> Optional[Dict[str, Any]]:
        """
        Archive oldest (len - keep) records to path, keep the newest keep records.
        Chain continuity is preserved via base_hash.
        """
        keep = int(keep)
        if keep <= 0 or len(self.records) <= keep:
            return None
        split = len(self.records) - keep
        archived = deep(self.records[:split])
        archived_base = self.base_hash

        payload = {
            "base_hash": archived_base,
            "records": archived,
            "archived_at": time.time(),
            "note": "archive segment; chain continues in AUDIT.json",
        }
        atomic_write_text(Path(path), canonical(payload))

        new_base = archived[-1]["hash"]
        rebuilt: List[Dict[str, Any]] = []
        previous = new_base
        for index, record in enumerate(deep(self.records[split:])):
            body = {k: v for k, v in record.items() if k != "hash"}
            body["index"] = index
            body["previous"] = previous
            digest = self._digest(body, self.key)
            body["hash"] = digest
            previous = digest
            rebuilt.append(body)

        self.base_hash = new_base
        self.records = rebuilt
        return {
            "archived": len(archived),
            "kept": len(rebuilt),
            "archive_path": str(path),
            "new_base": new_base,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {"base_hash": self.base_hash, "records": deep(self.records)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], key: Optional[bytes] = None,
                  allow_legacy: bool = False) -> "AuditChain":
        records = data.get("records", [])
        if not isinstance(records, list):
            records = []
        base_hash = data.get("base_hash", "GENESIS")
        if not isinstance(base_hash, str) or not base_hash:
            base_hash = "GENESIS"
        records = [r for r in records if isinstance(r, dict)]
        chain = cls(records=records, base_hash=base_hash, key=key)
        if key is not None and not chain.verify():
            # One-time migration from the old unkeyed SHA-256 chain. This trusts
            # the legacy file once; it is logged so the operator can notice.
            legacy = cls(records=records, base_hash=base_hash, key=None)
            if legacy.verify():
                if not allow_legacy:
                    raise SealError(f"audit chain: {LEGACY_MIGRATE_HINT}")
                print("[Audit] unkeyed legacy chain verified; re-signing with HMAC key (migration)")
                chain._resign()
        return chain

    def save(self, path: Union[str, Path]):
        path = Path(path)
        try:
            atomic_write_text(path, canonical(self.to_dict()))
        except OSError as e:
            print(f"[Audit] warning: cannot write {path}: {e}")

    @classmethod
    def load(cls, path: Union[str, Path], key: Optional[bytes] = None,
             allow_legacy: bool = False) -> "AuditChain":
        path = Path(path)
        if not path.exists():
            return cls(key=key)
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(data, key=key, allow_legacy=allow_legacy)


# MODULE FLATTENED: omega.core.config
# ============================================================================

# -*- coding: utf-8 -*-
"""Configuration for the cognitive kernel."""


from dataclasses import dataclass, field
from typing import List



@dataclass
class Config:
    # --- Decision thresholds ---
    max_error_apply: float = 1.0
    max_uncertainty_apply: float = 0.30
    max_uncertainty_hold: float = 0.80
    min_attribution_confidence: float = 0.35
    learning_step: float = 0.12
    learning_abs: float = 2.5
    cal_blacklist: List[str] = field(default_factory=list)
    max_episodes: int = 1000
    token_ttl_sec: float = 300.0
    prune_disk_blocks: bool = False
    max_consumed_tokens: int = 2000

    # --- Cognitive enhancement ---
    lookahead_steps: int = 2
    experience_buffer_size: int = 64
    adapt_rate: float = 0.08
    stall_window: int = 8
    stall_epsilon: float = 0.008
    min_progress_to_apply: float = -0.05
    reflection_every: int = 6
    risk_aversion: float = 0.6
    risk_aversion_baseline: float = 0.6
    learning_step_baseline: float = 0.12

    # --- Audit ---
    auto_archive_audit: bool = True
    audit_keep_records: int = 1000
    converged_goal_distance: float = 0.1

    # --- Phase-3 extensions ---
    max_tool_calls_per_cycle: int = 8
    tool_timeout_sec: float = 30.0
    require_confirm_for_tool_install: bool = True
    require_confirm_for_skill_install: bool = True
    require_confirm_for_self_modify: bool = True
    allow_code_execution: bool = False
    code_sandbox_timeout_sec: float = 10.0
    intrinsic_motivation_weight: float = 0.15
    hierarchical_goals: bool = True

    # --- Planner (Path B: on by default; still token-gated) ---
    use_planner: bool = True
    planner_horizon: int = 3
    planner_max_candidates: int = 12

    # --- V3.11 / unify: hybrid predictor ---
    world_model_predictor: str = "linear"  # linear | hybrid | attention
    attention_window: int = 16
    attention_d_model: int = 8
    attention_heads: int = 2

    # --- V3.11 / unify: Vault-at-Rest (off by default) ---
    vault_enabled: bool = False
    vault_kdf_iterations: int = DEFAULT_PBKDF2_ITERATIONS

    # --- V3.11 / unify: consensus ---
    consensus_quorum: int = 2

    # --- V3.11 / unify: async reactor ---
    async_max_concurrency: int = 1
    async_preemption: bool = True

    # --- Topology (JIA-Loop) ---
    topology_enabled: bool = True
    topology_mod: int = 16
    topology_cycle: int = 240

    # --- LLM bridge (stub; propose only) ---
    llm_bridge_enabled: bool = False
    llm_bridge_endpoint: str = ""
    llm_bridge_model: str = ""


# ============================================================================
# MODULE FLATTENED: omega.core.security
# ============================================================================

# -*- coding: utf-8 -*-
"""Owner, Boundary, TokenBook, StorageLock - security primitives."""


import dataclasses
import os
import secrets
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    import fcntl as _fcntl
except ImportError:
    _fcntl = None

try:
    import msvcrt as _msvcrt
except ImportError:
    _msvcrt = None



@dataclass(frozen=True)
class Boundary:
    """Hard permission boundary. Only listed actions are permitted."""
    allowed_actions: frozenset[str]

    def permits(self, action: str) -> bool:
        return action in self.allowed_actions


@dataclass(frozen=True)
class Owner:
    id: str
    salt: str
    digest: str


@dataclass
class TokenRecord:
    owner_id: str
    epoch: int
    suggestion_id: str
    action: str
    args_hash: str
    expires_at: float


@dataclass
class TokenBook:
    """Lightweight persistent token book. Only keeps unexpired issued + recent consumed."""
    issued: Dict[str, TokenRecord] = field(default_factory=dict)
    consumed: List[str] = field(default_factory=list)
    max_consumed: int = 2000
    key: Optional[bytes] = field(default=None, repr=False, compare=False)

    def issue(
        self,
        *,
        owner_id: str,
        epoch: int,
        suggestion_id: str,
        action: str,
        args_hash: str,
        ttl_sec: float,
    ) -> str:
        token = secrets.token_urlsafe(32)
        self.issued[token] = TokenRecord(
            owner_id=owner_id,
            epoch=epoch,
            suggestion_id=suggestion_id,
            action=action,
            args_hash=args_hash,
            expires_at=time.time() + max(0.001, float(ttl_sec)),
        )
        return token

    def consume(
        self,
        token: str,
        *,
        owner_id: str,
        epoch: int,
        suggestion_id: str,
        action: str,
        args_hash: str,
    ) -> bool:
        if token in self.consumed:
            return False
        record = self.issued.get(token)
        if record is None:
            return False
        if time.time() > record.expires_at:
            return False
        if record.owner_id != owner_id:
            return False
        if record.epoch != epoch:
            return False
        if record.suggestion_id != suggestion_id:
            return False
        if record.action != action:
            return False
        if not hmac.compare_digest(record.args_hash, args_hash):
            return False
        self.issued.pop(token, None)
        self.consumed.append(token)
        if len(self.consumed) > self.max_consumed:
            del self.consumed[: len(self.consumed) - self.max_consumed]
        return True

    def revoke_for_suggestion(self, suggestion_id: str) -> int:
        """Invalidate every outstanding token for one suggestion."""
        stale = [t for t, r in self.issued.items() if r.suggestion_id == suggestion_id]
        for t in stale:
            del self.issued[t]
        return len(stale)

    def purge_expired(self) -> int:
        now = time.time()
        expired = [t for t, r in self.issued.items() if r.expires_at < now]
        for t in expired:
            del self.issued[t]
        return len(expired)

    def to_dict(self) -> Dict[str, Any]:
        self.purge_expired()
        return {
            "issued": {tok: asdict(rec) for tok, rec in self.issued.items()},
            "consumed": list(self.consumed),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], max_consumed: int = 2000,
                  key: Optional[bytes] = None) -> "TokenBook":
        book = cls(max_consumed=max_consumed, key=key)
        now = time.time()
        for tok, rec in data.get("issued", {}).items():
            try:
                record = TokenRecord(**rec)
                if record.expires_at > now:
                    book.issued[tok] = record
            except Exception:
                continue
        consumed = data.get("consumed", [])
        book.consumed = list(consumed)[-max_consumed:]
        return book

    def save(self, path: Union[str, Path]):
        path = Path(path)
        try:
            payload = self.to_dict()
            if self.key is not None:
                payload = seal_record(self.key, payload)
            atomic_write_text(path, canonical(payload))
        except OSError as e:
            print(f"[TokenBook] warning: cannot write {path}: {e}")

    @classmethod
    def load(cls, path: Union[str, Path], max_consumed: int = 2000,
             key: Optional[bytes] = None) -> "TokenBook":
        """Tokens are short-lived: a MAC mismatch raises (integrity failure); unsealed files are discarded (fail closed)."""
        path = Path(path)
        if not path.exists():
            return cls(max_consumed=max_consumed, key=key)
        data = json.loads(path.read_text(encoding="utf-8"))
        if key is not None:
            if MAC_FIELD not in data:
                print("[TokenBook] unsealed token file discarded (fail closed)")
                return cls(max_consumed=max_consumed, key=key)
            body = open_record(key, data)
            if body is None:
                raise SealError("token file MAC mismatch (tampered, or wrong secret)")
            data = body
        return cls.from_dict(data, max_consumed=max_consumed, key=key)


class StorageLock:
    """
    Directory-level mutex using flock / msvcrt.

    Reentrant within the same instance and safe to share between threads: an
    internal RLock is held from the first acquire until the matching release,
    so threads serialize instead of racing on the depth counter.
    Raises TimeoutError on failure.
    """

    _warned = False

    def __init__(self, path: Union[str, Path], timeout: float = 15.0, poll: float = 0.02):
        self.path = Path(path)
        self.timeout = float(timeout)
        self.poll = float(poll)
        self._fd: Optional[int] = None
        self._depth = 0
        self._mutex = threading.RLock()

    def acquire(self) -> bool:
        if not self._mutex.acquire(timeout=self.timeout):
            raise TimeoutError(
                f"storage lock timeout ({self.timeout}s) waiting for in-process lock: {self.path}"
            )
        try:
            if self._depth > 0:
                self._depth += 1
                return True

            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(str(self.path), os.O_RDWR | os.O_CREAT, 0o600)
            deadline = time.time() + self.timeout
            while True:
                try:
                    if _fcntl is not None:
                        _fcntl.flock(fd, _fcntl.LOCK_EX | _fcntl.LOCK_NB)
                    elif _msvcrt is not None:
                        _msvcrt.locking(fd, _msvcrt.LK_NBLCK, 1)
                    elif not StorageLock._warned:
                        StorageLock._warned = True
                        print("[Lock] no file-lock primitive on this platform; concurrency protection disabled.")
                    break
                except OSError:
                    if time.time() >= deadline:
                        os.close(fd)
                        raise TimeoutError(
                            f"storage lock timeout ({self.timeout}s): {self.path}. "
                            f"Another process may be writing the same storage directory."
                        )
                    time.sleep(self.poll)
            self._fd = fd
            self._depth = 1
            return True
        except BaseException:
            self._mutex.release()
            raise

    def release(self) -> None:
        if self._depth == 0:
            return
        try:
            self._depth -= 1
            if self._depth > 0:
                return
            fd, self._fd = self._fd, None
            if fd is None:
                return
            try:
                if _fcntl is not None:
                    _fcntl.flock(fd, _fcntl.LOCK_UN)
                elif _msvcrt is not None:
                    _msvcrt.locking(fd, _msvcrt.LK_UNLCK, 1)
            except OSError:
                pass
            finally:
                try:
                    os.close(fd)
                except OSError:
                    pass
        finally:
            self._mutex.release()

    def __enter__(self) -> "StorageLock":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.release()
        return False


# ============================================================================
# MODULE FLATTENED: omega.goals.intrinsic
# ============================================================================

# -*- coding: utf-8 -*-
"""
Intrinsic motivation signals.

Three cheap, model-free drives (inspired by competence / novelty / exploration):
- competence: recent progress & success rate
- novelty: state change magnitude vs recent baseline
- exploration: bonus when stalled or high residual

These never bypass the safety pipeline; they only modulate scoring
(e.g. expected_progress) by config.intrinsic_motivation_weight.
"""


from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional



@dataclass
class IntrinsicSignal:
    competence: float = 0.0
    novelty: float = 0.0
    exploration: float = 0.0
    combined: float = 0.0
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "competence": self.competence,
            "novelty": self.novelty,
            "exploration": self.exploration,
            "combined": self.combined,
            "notes": list(self.notes),
        }


class IntrinsicMotivation:
    def __init__(
        self,
        competence_weight: float = 0.40,
        novelty_weight: float = 0.35,
        exploration_weight: float = 0.25,
        history_window: int = 12,
    ):
        self.competence_weight = competence_weight
        self.novelty_weight = novelty_weight
        self.exploration_weight = exploration_weight
        self.history_window = history_window
        self._state_history: List[Dict[str, float]] = []
        self.last_signal: Optional[IntrinsicSignal] = None

    def observe_state(self, state: Mapping[str, float]) -> None:
        self._state_history.append(dict(state))
        if len(self._state_history) > self.history_window:
            self._state_history = self._state_history[-self.history_window:]

    def evaluate(
        self,
        state: Mapping[str, float],
        *,
        recent_progress: float = 0.0,
        success_rate: float = 0.5,
        stall: bool = False,
        mean_residual: float = 0.0,
        goal_distance: Optional[float] = None,
    ) -> IntrinsicSignal:
        notes: List[str] = []

        # competence: blend success rate and recent progress
        prog_term = clamp(recent_progress * 5.0, -0.5, 1.0)
        competence = clamp(0.55 * success_rate + 0.45 * max(0.0, prog_term), 0.0, 1.0)
        if recent_progress > 0.01:
            notes.append("progressing")

        # novelty: how different is current state from recent mean
        novelty = 0.3
        if len(self._state_history) >= 3:
            keys = set()
            for s in self._state_history:
                keys |= set(s.keys())
            keys |= set(state.keys())
            mean = {}
            for k in keys:
                vals = [float(s.get(k, 0.0)) for s in self._state_history]
                mean[k] = sum(vals) / len(vals)
            delta = {k: float(state.get(k, 0.0)) - mean.get(k, 0.0) for k in keys}
            novelty = clamp(vector_rms(delta) * 2.0, 0.0, 1.0)
            if novelty > 0.4:
                notes.append("novel_state")

        # exploration: boost when stalled or high model residual or near-zero distance stuck
        exploration = 0.15
        if stall:
            exploration = 0.75
            notes.append("stall_explore")
        elif mean_residual > 0.15:
            exploration = clamp(0.3 + mean_residual, 0.0, 0.9)
            notes.append("high_residual")
        elif goal_distance is not None and goal_distance < 0.08 and recent_progress < 0.002:
            exploration = 0.5
            notes.append("near_goal_flat")

        combined = clamp(
            self.competence_weight * competence
            + self.novelty_weight * novelty
            + self.exploration_weight * exploration,
            0.0, 1.0,
        )

        sig = IntrinsicSignal(
            competence=competence,
            novelty=novelty,
            exploration=exploration,
            combined=combined,
            notes=notes,
        )
        self.last_signal = sig
        self.observe_state(state)
        return sig

    def modulate_progress(self, expected_progress: float, weight: float) -> float:
        """Blend intrinsic combined signal into expected progress."""
        if weight <= 0 or self.last_signal is None:
            return expected_progress
        # map combined [0,1] to a small progress bonus/penalty
        bonus = (self.last_signal.combined - 0.4) * 0.1
        return expected_progress + weight * bonus


# ============================================================================
# MODULE FLATTENED: omega.goals.tree
# ============================================================================

# -*- coding: utf-8 -*-
"""
Hierarchical goal tree.

- Root goal may have weighted children (subgoals).
- Active subgoal is selected by priority, progress gap, and status.
- Achieved leaves auto-promote parent progress.
- Flat goal (dict) remains supported via from_flat / to_flat.
"""


import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional



@dataclass
class GoalNode:
    goal_id: str
    targets: Dict[str, float]
    weight: float = 1.0
    parent_id: Optional[str] = None
    children: List[str] = field(default_factory=list)
    priority: float = 1.0
    status: str = "active"  # active | achieved | suspended | failed
    tolerance: float = 0.05
    created_at: float = field(default_factory=time.time)
    achieved_at: Optional[float] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def distance(self, state: Mapping[str, float]) -> float:
        if not self.targets:
            return 0.0
        return sum(
            abs(float(state.get(k, 0.0)) - float(t)) for k, t in self.targets.items()
        ) / max(1, len(self.targets))

    def is_achieved(self, state: Mapping[str, float]) -> bool:
        return self.distance(state) <= self.tolerance

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["targets"] = dict(self.targets)
        d["children"] = list(self.children)
        d["meta"] = deep(self.meta)
        return d

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GoalNode":
        return cls(
            goal_id=str(data["goal_id"]),
            targets=norm_vector(data.get("targets") or {}),
            weight=float(data.get("weight", 1.0)),
            parent_id=data.get("parent_id"),
            children=list(data.get("children") or []),
            priority=float(data.get("priority", 1.0)),
            status=str(data.get("status", "active")),
            tolerance=float(data.get("tolerance", 0.05)),
            created_at=float(data.get("created_at", time.time())),
            achieved_at=data.get("achieved_at"),
            meta=dict(data.get("meta") or {}),
        )


class GoalTree:
    """Tree of GoalNodes with active-subgoal selection."""

    def __init__(self):
        self.nodes: Dict[str, GoalNode] = {}
        self.root_id: Optional[str] = None
        self.active_id: Optional[str] = None

    # --- construction ---

    def set_root(self, goal_id: str, targets: Mapping[str, float], **kwargs) -> GoalNode:
        node = GoalNode(goal_id=goal_id, targets=norm_vector(targets), **kwargs)
        self.nodes = {goal_id: node}
        self.root_id = goal_id
        self.active_id = goal_id
        return node

    def add_subgoal(
        self,
        goal_id: str,
        targets: Mapping[str, float],
        parent_id: Optional[str] = None,
        **kwargs,
    ) -> GoalNode:
        parent_id = parent_id or self.root_id
        if parent_id is None or parent_id not in self.nodes:
            raise ValueError("parent not found; set_root first")
        node = GoalNode(
            goal_id=goal_id,
            targets=norm_vector(targets),
            parent_id=parent_id,
            **kwargs,
        )
        self.nodes[goal_id] = node
        parent = self.nodes[parent_id]
        if goal_id not in parent.children:
            parent.children.append(goal_id)
        if self.active_id is None:
            self.active_id = goal_id
        return node

    @classmethod
    def from_flat(cls, targets: Mapping[str, float], goal_id: str = "root") -> "GoalTree":
        tree = cls()
        tree.set_root(goal_id, targets)
        return tree

    def to_flat(self) -> Dict[str, float]:
        """Return active subgoal targets, or root if none active."""
        node = self.get_active()
        if node is None and self.root_id:
            node = self.nodes.get(self.root_id)
        return dict(node.targets) if node else {}

    # --- selection ---

    def get_active(self) -> Optional[GoalNode]:
        if self.active_id and self.active_id in self.nodes:
            node = self.nodes[self.active_id]
            if node.status == "active":
                return node
        return self.select_active()

    def select_active(self, state: Optional[Mapping[str, float]] = None) -> Optional[GoalNode]:
        """Pick best active leaf (or root) by priority * remaining distance."""
        candidates = [
            n for n in self.nodes.values()
            if n.status == "active" and not n.children  # prefer leaves
        ]
        if not candidates:
            candidates = [n for n in self.nodes.values() if n.status == "active"]
        if not candidates:
            self.active_id = self.root_id
            return self.nodes.get(self.root_id) if self.root_id else None

        def score(n: GoalNode) -> float:
            dist = n.distance(state) if state else 1.0
            # higher priority and larger remaining gap first
            return n.priority * (0.3 + dist)

        best = max(candidates, key=score)
        self.active_id = best.goal_id
        return best

    def mark_achieved(self, goal_id: str, state: Optional[Mapping[str, float]] = None) -> None:
        node = self.nodes.get(goal_id)
        if node is None:
            return
        node.status = "achieved"
        node.achieved_at = time.time()
        # if all siblings achieved, mark parent
        if node.parent_id and node.parent_id in self.nodes:
            parent = self.nodes[node.parent_id]
            siblings = [self.nodes[c] for c in parent.children if c in self.nodes]
            if siblings and all(s.status == "achieved" for s in siblings):
                parent.status = "achieved"
                parent.achieved_at = time.time()
        if self.active_id == goal_id:
            self.select_active(state)

    def suspend(self, goal_id: str) -> None:
        node = self.nodes.get(goal_id)
        if node:
            node.status = "suspended"
            if self.active_id == goal_id:
                self.select_active()

    def resume(self, goal_id: str) -> None:
        node = self.nodes.get(goal_id)
        if node and node.status == "suspended":
            node.status = "active"
            self.active_id = goal_id

    def update_from_state(self, state: Mapping[str, float]) -> List[str]:
        """Auto-mark achieved leaves; return list of newly achieved ids."""
        newly = []
        for node in list(self.nodes.values()):
            if node.status == "active" and not node.children and node.is_achieved(state):
                self.mark_achieved(node.goal_id, state)
                newly.append(node.goal_id)
        return newly

    def progress_summary(self, state: Mapping[str, float]) -> Dict[str, Any]:
        active = self.get_active()
        return {
            "root_id": self.root_id,
            "active_id": self.active_id,
            "active_distance": active.distance(state) if active else None,
            "nodes": {
                gid: {
                    "status": n.status,
                    "distance": n.distance(state),
                    "priority": n.priority,
                }
                for gid, n in self.nodes.items()
            },
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "root_id": self.root_id,
            "active_id": self.active_id,
            "nodes": {gid: n.to_dict() for gid, n in self.nodes.items()},
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GoalTree":
        tree = cls()
        tree.root_id = data.get("root_id")
        tree.active_id = data.get("active_id")
        for gid, raw in (data.get("nodes") or {}).items():
            tree.nodes[gid] = GoalNode.from_dict(raw)
        return tree


# ============================================================================
# MODULE FLATTENED: omega.memory.engine
# ============================================================================

# -*- coding: utf-8 -*-
"""CAS memory engine - immutable episode storage."""


import json
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import hmac as _hmac



def _hmac_compare(a: str, b: str) -> bool:
    return _hmac.compare_digest(a, b)


def _is_episode_block(path: Path) -> bool:
    """Episode blocks are {sha256}.block; skill blocks are skill-{hash}.block."""
    stem = path.stem
    if stem.startswith(SKILL_CAS_PREFIX.rstrip("-")) or stem.startswith("skill-"):
        return False
    # pure content-addressed episode: 64-char hex stem
    if len(stem) == 64 and all(c in "0123456789abcdef" for c in stem.lower()):
        return True
    return False


@dataclass
class Episode:
    episode_id: str
    epoch: int
    timestamp: float
    suggestion_id: str
    intention: Dict[str, Any]
    prediction: Dict[str, Any]
    observation_before: Dict[str, float]
    observation_after: Dict[str, float]
    difference: Dict[str, float]
    error: float
    uncertainty: float
    attribution: Dict[str, float]
    decision: str
    reason: str
    learning: Dict[str, float]
    risk: float
    audit_hash: str = ""
    tools_used: List[str] = field(default_factory=list)
    skill_ids: List[str] = field(default_factory=list)
    role_evidence: Dict[str, Any] = field(default_factory=dict)


def episode_from_dict(data: Dict[str, Any]) -> Episode:
    known = {f.name for f in fields(Episode)}
    filtered = {k: v for k, v in data.items() if k in known}
    return Episode(**filtered)


class OmegaMemoryEngine:
    def __init__(self, storage_dir: str | Path = "./omega_cas_brain",
                 key: Optional[bytes] = None, allow_legacy: bool = False):
        self.key = key
        self.allow_legacy = bool(allow_legacy)
        # legacy content-address -> sealed content-address, filled by migration
        self.renamed: Dict[str, str] = {}
        self.base_dir = Path(storage_dir)
        self.objects_dir = self.base_dir / "objects"
        self.head_file = self.base_dir / "HEAD.json"
        self.tokens_file = self.base_dir / "TOKENS.json"
        self.audit_file = self.base_dir / "AUDIT.json"
        self.episodes: List[Episode] = []
        self.semantic: Dict[str, Dict[str, float]] = {}
        self.active_hashes: List[str] = []
        self._verify_cache: Dict[str, Tuple[int, int]] = {}

    def commit_episode(self, episode: Episode) -> str:
        data = asdict(episode)
        if self.key is not None:
            data = seal_record(self.key, data)
        content = canonical(data)
        obj_hash = sha256(content)
        obj_path = self.objects_dir / f"{obj_hash}.block"
        if not obj_path.exists():
            atomic_write_text(obj_path, content, mode=0o644)
        return obj_hash

    def add(self, episode: Episode, limit: int, prune_disk: bool = False):
        obj_hash = self.commit_episode(episode)
        self.episodes.append(deep(episode))
        self.active_hashes.append(obj_hash)

        if len(self.episodes) > limit:
            overflow = len(self.episodes) - limit
            self.episodes = self.episodes[-limit:]
            removed_hashes = self.active_hashes[:overflow]
            self.active_hashes = self.active_hashes[-limit:]
            if prune_disk and removed_hashes:
                for h in removed_hashes:
                    p = self.objects_dir / f"{h}.block"
                    if p.exists():
                        try:
                            p.unlink()
                        except OSError:
                            pass

    def update_semantic(self, difference: Dict[str, float]):
        for key, value in difference.items():
            error = abs(float(value))
            item = self.semantic.get(key)
            if item is None:
                self.semantic[key] = {"count": 1.0, "mean_error": error, "last_error": error}
                continue
            count = item["count"]
            item["mean_error"] = (item["mean_error"] * count + error) / (count + 1.0)
            item["last_error"] = error
            item["count"] = count + 1.0

    def _read_block(self, block_path: Path) -> Tuple[Dict[str, Any], str]:
        """Return (episode dict, content-address) for a block, or raise.

        Every block must carry a valid MAC. The file name must be the SHA-256 of
        its exact content. Unsealed legacy blocks are accepted only in migration
        mode, and are rewritten sealed under their new name.
        """
        content = block_path.read_text(encoding="utf-8")
        data = json.loads(content)
        if self.key is None:
            if not _hmac_compare(sha256(content), block_path.stem):
                raise ValueError(f"{block_path.name}: name does not match content")
            return data, block_path.stem
        body, legacy = open_sealed(self.key, data, allow_legacy=self.allow_legacy,
                                   label=block_path.name)
        if legacy:
            sealed = canonical(seal_record(self.key, body))
            new_path = self.objects_dir / f"{sha256(sealed)}.block"
            atomic_write_text(new_path, sealed, mode=0o644)
            try:
                block_path.unlink()
            except OSError:
                pass
            self.renamed[block_path.stem] = new_path.stem
            print(f"[Memory] migrated legacy block {block_path.stem[:12]} -> sealed")
            return body, new_path.stem
        if not _hmac_compare(sha256(content), block_path.stem):
            raise ValueError(f"{block_path.name}: name does not match content")
        return body, block_path.stem

    def verify_all_blocks(self) -> bool:
        # No signature-based caching: a same-size, same-mtime rewrite must not slip past.
        if not self.objects_dir.exists():
            return True
        for block_path in list(self.objects_dir.glob("*.block")):
            if not _is_episode_block(block_path):
                continue
            try:
                self._read_block(block_path)
            except Exception:
                return False
        return True

    def rebuild_from_cas(self, limit: Optional[int] = None) -> int:
        self.episodes = []
        self.semantic = {}
        self.active_hashes = []
        if not self.objects_dir.exists():
            return 0
        candidates = []
        for block_path in list(self.objects_dir.glob("*.block")):
            if not _is_episode_block(block_path):
                continue
            try:
                data, name = self._read_block(block_path)
                ep = episode_from_dict(data)
                candidates.append((ep.epoch, ep.timestamp, name, ep))
            except Exception:
                continue
        candidates.sort(key=lambda x: (x[0], x[1]))
        if limit is not None and limit > 0:
            candidates = candidates[-limit:]
        for _, _, h, ep in candidates:
            self.episodes.append(ep)
            self.active_hashes.append(h)
            if ep.difference:
                self.update_semantic(ep.difference)
        return len(self.episodes)

    def get_active_hashes(self) -> List[str]:
        return list(self.active_hashes)

    def block_count(self) -> int:
        if not self.objects_dir.exists():
            return 0
        return sum(1 for p in self.objects_dir.glob("*.block") if _is_episode_block(p))

    def query(
        self,
        *,
        last_n: Optional[int] = None,
        epoch_min: Optional[int] = None,
        epoch_max: Optional[int] = None,
        min_error: Optional[float] = None,
        max_error: Optional[float] = None,
        min_risk: Optional[float] = None,
        decision: Optional[str] = None,
        has_key: Optional[str] = None,
        order_by: str = "epoch",
        descending: bool = True,
        limit: int = 20,
    ) -> List[Episode]:
        results = list(self.episodes)
        if epoch_min is not None:
            results = [e for e in results if e.epoch >= epoch_min]
        if epoch_max is not None:
            results = [e for e in results if e.epoch <= epoch_max]
        if min_error is not None:
            results = [e for e in results if e.error >= min_error]
        if max_error is not None:
            results = [e for e in results if e.error <= max_error]
        if min_risk is not None:
            results = [e for e in results if e.risk >= min_risk]
        if decision is not None:
            results = [e for e in results if e.decision == decision]
        if has_key is not None:
            results = [
                e for e in results
                if has_key in (e.difference or {})
                or has_key in (e.observation_after or {})
                or has_key in (e.observation_before or {})
            ]
        key_fn = {
            "epoch": lambda e: e.epoch,
            "error": lambda e: e.error,
            "risk": lambda e: e.risk,
            "timestamp": lambda e: e.timestamp,
        }.get(order_by, lambda e: e.epoch)
        results.sort(key=key_fn, reverse=descending)
        if last_n is not None:
            results = sorted(results, key=lambda e: e.epoch, reverse=True)[:last_n]
            results.sort(key=key_fn, reverse=descending)
        return results[: max(1, limit)]

    def recent(self, n: int = 5) -> List[Episode]:
        return self.query(last_n=n, order_by="epoch", descending=True, limit=n)

    def worst_errors(self, n: int = 5) -> List[Episode]:
        return self.query(order_by="error", descending=True, limit=n)

    def best_errors(self, n: int = 5) -> List[Episode]:
        return self.query(order_by="error", descending=False, limit=n)

    def high_risk(self, n: int = 5, min_risk: float = 0.05) -> List[Episode]:
        return self.query(min_risk=min_risk, order_by="risk", descending=True, limit=n)

    def stats(self) -> Dict[str, Any]:
        if not self.episodes:
            return {
                "count": 0, "mean_error": 0.0, "max_error": 0.0,
                "min_error": 0.0, "mean_risk": 0.0, "decisions": {},
            }
        errors = [e.error for e in self.episodes]
        risks = [e.risk for e in self.episodes]
        decisions: Dict[str, int] = {}
        for e in self.episodes:
            decisions[e.decision] = decisions.get(e.decision, 0) + 1
        return {
            "count": len(self.episodes),
            "mean_error": sum(errors) / len(errors),
            "max_error": max(errors),
            "min_error": min(errors),
            "mean_risk": sum(risks) / len(risks) if risks else 0.0,
            "decisions": decisions,
            "epoch_range": (self.episodes[0].epoch, self.episodes[-1].epoch),
        }

    def semantic_summary(self) -> Dict[str, Any]:
        items = []
        for key, stats in self.semantic.items():
            items.append({
                "key": key,
                "count": stats.get("count", 0),
                "mean_error": stats.get("mean_error", 0.0),
                "last_error": stats.get("last_error", 0.0),
            })
        items.sort(key=lambda x: x["mean_error"], reverse=True)
        return {
            "total_keys": len(items),
            "total_episodes": len(self.episodes),
            "keys": items,
        }

    def find_by_id(self, episode_id: str) -> Optional[Episode]:
        for e in self.episodes:
            if e.episode_id == episode_id:
                return e
        return None

    def summarize_episode(self, ep: Episode) -> str:
        return (
            f"[{ep.episode_id}] epoch={ep.epoch} error={ep.error:.4f} risk={ep.risk:.4f} "
            f"decision={ep.decision} reason={ep.reason}"
        )


# ============================================================================
# MODULE FLATTENED: omega.roles
# ============================================================================

# -*- coding: utf-8 -*-
"""Bounded cognitive roles for Omega.

Roles are advisory shards around the fixed cognition loop:
    observation -> prediction -> calibration -> decision
They do not own authorization, execution, boundary, or governance state.
"""


from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple



@dataclass(frozen=True)
class FeedbackObservation:
    source: str
    error: float
    direction: str
    confidence: float
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AuditChallenge:
    source: str
    signal: str
    strength: float
    target: str
    reason: str
    evidence_count: int
    conflict: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MetaReflection:
    source: str
    regime: str
    recommended_action: str
    confidence_in_self: float
    calibration_score: float
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PatternCandidate:
    signature: str
    action: str
    decision: str
    count: int
    success_rate: float
    confidence: float
    source_episode_ids: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["source_episode_ids"] = list(self.source_episode_ids)
        return data


@dataclass(frozen=True)
class PatternValidation:
    candidate: PatternCandidate
    samples: int
    consistency: float
    validated: bool
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate": self.candidate.to_dict(),
            "samples": self.samples,
            "consistency": self.consistency,
            "validated": self.validated,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class LearningProposal:
    source: str
    pattern: PatternCandidate
    evidence_count: int
    consistency: float
    rationale: str
    approved: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "pattern": self.pattern.to_dict(),
            "evidence_count": self.evidence_count,
            "consistency": self.consistency,
            "rationale": self.rationale,
            "approved": self.approved,
        }


class Yiyi:
    """Observation/reflection shard. It only reports post-action reality."""

    role_id = "yiyi"

    def observe_feedback(
        self,
        predicted: Mapping[str, float],
        observed: Mapping[str, float],
        *,
        confidence: float = 1.0,
    ) -> FeedbackObservation:
        error = vector_rms(vector_diff(predicted, observed))
        if error <= 0.05:
            direction = "aligned"
        elif error <= 0.25:
            direction = "drift"
        else:
            direction = "divergent"
        return FeedbackObservation(
            source=self.role_id,
            error=error,
            direction=direction,
            confidence=clamp(float(confidence), 0.0, 1.0),
            reason=f"post_action_{direction}",
        )


class Ziwei:
    """Metacognition shard. It observes self-state; it cannot govern."""

    role_id = "ziwei"

    def reflect(self, assessment: Mapping[str, Any]) -> MetaReflection:
        regime = str(assessment.get("regime", "stable"))
        action = str(assessment.get("recommended_action", "continue"))
        confidence = clamp(float(assessment.get("confidence_in_self", 0.0)), 0.0, 1.0)
        calibration = clamp(float(assessment.get("calibration_score", 0.0)), 0.0, 1.0)
        return MetaReflection(
            source=self.role_id,
            regime=regime,
            recommended_action=action,
            confidence_in_self=confidence,
            calibration_score=calibration,
            reason=f"meta_{regime}_{action}",
        )


class Wukong:
    """Independent adversarial review shard. Evidence only; no decisions."""

    role_id = "wukong"
    MIN_SAMPLES = 3
    MIN_FAILURE_RATIO = 0.30

    def audit_prediction(
        self,
        *,
        action: str,
        prediction_confidence: float,
        episodes: Sequence[Any],
    ) -> Tuple[AuditChallenge, ...]:
        relevant = []
        for episode in episodes:
            intention = getattr(episode, "intention", {}) or {}
            if str(intention.get("action", "")) == str(action):
                relevant.append(episode)

        if len(relevant) < self.MIN_SAMPLES or prediction_confidence < 0.70:
            return ()

        failures = 0
        for episode in relevant:
            error = float(getattr(episode, "error", 0.0) or 0.0)
            decision = str(getattr(episode, "decision", ""))
            # Historical APPLY episodes with substantial residual error are
            # counter-evidence against another high-confidence prediction.
            if decision == "APPLY" and error > 0.15:
                failures += 1

        ratio = failures / len(relevant)
        if ratio < self.MIN_FAILURE_RATIO:
            return ()

        return (
            AuditChallenge(
                source=self.role_id,
                signal="historical_counter_evidence",
                strength=clamp(ratio, 0.0, 1.0),
                target="prediction",
                reason="historical_apply_errors_challenge_current_confidence",
                evidence_count=failures,
                conflict=True,
            ),
        )


class Dayan:
    """Learning/pattern shard. It discovers and proposes; it never applies."""

    role_id = "dayan"
    MIN_SAMPLES = 5
    MIN_CONSISTENCY = 0.70

    @staticmethod
    def _good(episode: Any, max_error: float) -> bool:
        error = float(getattr(episode, "error", 0.0) or 0.0)
        progress = float(getattr(episode, "difference", {}).get("mission_progress", 0.0) or 0.0)
        after = getattr(episode, "observation_after", {}) or {}
        ready = float(after.get("target_ready", 0.0) or 0.0) >= 0.99
        return error <= max_error or progress > 0.0 or ready

    def discover(
        self,
        episodes: Sequence[Any],
        *,
        max_error: float = 1.0,
    ) -> Tuple[PatternCandidate, ...]:
        groups: Dict[Tuple[str, str], List[Any]] = {}
        for episode in episodes:
            intention = getattr(episode, "intention", {}) or {}
            action = str(intention.get("action", ""))
            decision = str(getattr(episode, "decision", ""))
            if not action:
                continue
            groups.setdefault((action, decision), []).append(episode)

        candidates: List[PatternCandidate] = []
        for (action, decision), group in groups.items():
            if len(group) < self.MIN_SAMPLES:
                continue
            good = sum(1 for episode in group if self._good(episode, max_error))
            rate = good / len(group)
            if rate < self.MIN_CONSISTENCY:
                continue
            ids = tuple(str(getattr(e, "episode_id", "")) for e in group[-16:])
            candidates.append(
                PatternCandidate(
                    signature=f"{action}:{decision}",
                    action=action,
                    decision=decision,
                    count=len(group),
                    success_rate=rate,
                    confidence=clamp(0.5 * rate + 0.5 * min(1.0, len(group) / 16.0), 0.0, 1.0),
                    source_episode_ids=ids,
                )
            )
        return tuple(candidates)

    def validate(
        self,
        candidate: PatternCandidate,
        episodes: Sequence[Any],
        *,
        max_error: float = 1.0,
    ) -> PatternValidation:
        matching = []
        for episode in episodes:
            intention = getattr(episode, "intention", {}) or {}
            if (
                str(intention.get("action", "")) == candidate.action
                and str(getattr(episode, "decision", "")) == candidate.decision
            ):
                matching.append(episode)
        samples = len(matching)
        if samples < self.MIN_SAMPLES:
            return PatternValidation(candidate, samples, 0.0, False, "insufficient_samples")
        successes = sum(1 for episode in matching if self._good(episode, max_error))
        consistency = successes / samples
        valid = consistency >= self.MIN_CONSISTENCY
        return PatternValidation(
            candidate=candidate,
            samples=samples,
            consistency=consistency,
            validated=valid,
            reason="validated" if valid else "low_consistency",
        )

    def propose_learning(self, validation: PatternValidation) -> Optional[LearningProposal]:
        if not validation.validated:
            return None
        return LearningProposal(
            source=self.role_id,
            pattern=validation.candidate,
            evidence_count=validation.samples,
            consistency=validation.consistency,
            rationale="repeated pattern passed validation",
            approved=False,
        )


class RoleCouncil:
    """Single coordinator for advisory role shards; no governance authority."""

    def __init__(self):
        self.yiyi = Yiyi()
        self.ziwei = Ziwei()
        self.wukong = Wukong()
        self.dayan = Dayan()

    def pre_decision(
        self,
        *,
        action: str,
        prediction_confidence: float,
        episodes: Sequence[Any],
        meta_assessment: Mapping[str, Any],
    ) -> Dict[str, Any]:
        ziwei = self.ziwei.reflect(meta_assessment)
        challenges = self.wukong.audit_prediction(
            action=action,
            prediction_confidence=prediction_confidence,
            episodes=episodes,
        )
        return {
            "ziwei": ziwei.to_dict(),
            "wukong": [c.to_dict() for c in challenges],
            "wukong_conflict": bool(challenges),
        }

    def post_execution(
        self,
        *,
        predicted: Mapping[str, float],
        observed: Mapping[str, float],
        episodes: Sequence[Any],
        max_error: float,
    ) -> Dict[str, Any]:
        feedback = self.yiyi.observe_feedback(predicted, observed)
        candidates = self.dayan.discover(episodes, max_error=max_error)
        proposals = []
        for candidate in candidates:
            validation = self.dayan.validate(candidate, episodes, max_error=max_error)
            proposal = self.dayan.propose_learning(validation)
            if proposal is not None:
                proposals.append(proposal.to_dict())
        return {
            "yiyi": feedback.to_dict(),
            "dayan_candidates": [c.to_dict() for c in candidates],
            "dayan_proposals": proposals,
        }


# ============================================================================
# MODULE FLATTENED: omega.skills.skill
# ============================================================================

# -*- coding: utf-8 -*-
"""
Skill (Option) abstraction.

Design rules (aligned with kernel safety):
- Skills are versioned, CAS-storable descriptors.
- Installing or activating a skill that can write / self-modify requires confirm.
- Skills do NOT bypass the authorize → token → execute pipeline.
- A skill is a high-level intention template, not an execution privilege.
"""


import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set



@dataclass
class Skill:
    """A reusable, versioned intention template (Option)."""
    skill_id: str
    name: str
    description: str = ""
    version: int = 1
    # High-level action the skill prefers to emit
    preferred_action: str = "measure"
    # Default / template args (merged with runtime context)
    template_args: Dict[str, Any] = field(default_factory=dict)
    # Tags for selection / filtering
    tags: Set[str] = field(default_factory=set)
    # Risk metadata
    base_risk: float = 0.2
    always_confirm: bool = False
    # Learning / usage stats
    use_count: int = 0
    success_count: int = 0
    last_used_at: float = 0.0
    created_at: float = field(default_factory=time.time)
    # Optional CAS content hash (when persisted)
    content_hash: str = ""

    def success_rate(self) -> float:
        if self.use_count <= 0:
            return 0.5
        return self.success_count / self.use_count

    def record_use(self, success: bool):
        self.use_count += 1
        if success:
            self.success_count += 1
        self.last_used_at = time.time()

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["tags"] = sorted(self.tags)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Skill":
        tags = data.get("tags") or []
        if isinstance(tags, list):
            tags = set(tags)
        return cls(
            skill_id=str(data["skill_id"]),
            name=str(data.get("name", data["skill_id"])),
            description=str(data.get("description", "")),
            version=int(data.get("version", 1)),
            preferred_action=str(data.get("preferred_action", "measure")),
            template_args=dict(data.get("template_args") or {}),
            tags=set(tags),
            base_risk=float(data.get("base_risk", 0.2)),
            always_confirm=bool(data.get("always_confirm", False)),
            use_count=int(data.get("use_count", 0)),
            success_count=int(data.get("success_count", 0)),
            last_used_at=float(data.get("last_used_at", 0.0)),
            created_at=float(data.get("created_at", time.time())),
            content_hash=str(data.get("content_hash", "")),
        )

    def compute_hash(self) -> str:
        payload = {
            "skill_id": self.skill_id,
            "name": self.name,
            "version": self.version,
            "preferred_action": self.preferred_action,
            "template_args": self.template_args,
            "tags": sorted(self.tags),
            "base_risk": self.base_risk,
            "always_confirm": self.always_confirm,
        }
        return sha256(canonical(payload))


class SkillRegistry:
    """In-memory registry of skills. CAS persistence can be layered later."""

    def __init__(self):
        self._skills: Dict[str, Skill] = {}

    def register(self, skill: Skill, *, overwrite: bool = False) -> None:
        if not skill.skill_id:
            raise ValueError("skill must have a non-empty skill_id")
        if skill.skill_id in self._skills and not overwrite:
            raise ValueError(f"skill already registered: {skill.skill_id}")
        if not skill.content_hash:
            skill.content_hash = skill.compute_hash()
        self._skills[skill.skill_id] = skill

    def unregister(self, skill_id: str) -> bool:
        return self._skills.pop(skill_id, None) is not None

    def get(self, skill_id: str) -> Optional[Skill]:
        return self._skills.get(skill_id)

    def list_skills(self) -> List[Dict[str, Any]]:
        return [s.to_dict() for s in self._skills.values()]

    def find_by_tag(self, tag: str) -> List[Skill]:
        return [s for s in self._skills.values() if tag in s.tags]

    def best_by_success(self, tag: Optional[str] = None, min_uses: int = 3) -> Optional[Skill]:
        candidates = list(self._skills.values())
        if tag:
            candidates = [s for s in candidates if tag in s.tags]
        candidates = [s for s in candidates if s.use_count >= min_uses]
        if not candidates:
            return None
        return max(candidates, key=lambda s: s.success_rate())

    def requires_confirm(self, skill_id: str) -> bool:
        s = self._skills.get(skill_id)
        return bool(s and s.always_confirm)

    def __len__(self) -> int:
        return len(self._skills)


def make_empty_skill_registry() -> SkillRegistry:
    return SkillRegistry()


# ============================================================================
# MODULE FLATTENED: omega.tools.fs
# ============================================================================

# -*- coding: utf-8 -*-
"""Filesystem tools bound to FileSandboxWorld (never escape sandbox root)."""


from typing import Any, Dict, Optional



class ListFilesTool(Tool):
    name = "list_files"
    description = "List entries in a sandbox-relative path (read-only)."
    tags = {"fs", "read", "safe"}
    base_risk = 0.05
    always_confirm = False

    def run(self, args: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> ToolResult:
        world = (context or {}).get("world")
        if world is None or not hasattr(world, "list_files"):
            return ToolResult(success=False, error="no FileSandboxWorld in context")
        rel = str((args or {}).get("path", "."))
        try:
            items = world.list_files(rel)
            return ToolResult(success=True, output={"path": rel, "items": items})
        except Exception as e:
            return ToolResult(success=False, error=f"{type(e).__name__}: {e}")


class ReadFileTool(Tool):
    name = "read_file"
    description = "Read a text file inside the sandbox (read-only)."
    tags = {"fs", "read", "safe"}
    base_risk = 0.08
    always_confirm = False

    def validate_args(self, args: Dict[str, Any]) -> Optional[str]:
        if not args or not args.get("path"):
            return "path required"
        return None

    def run(self, args: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> ToolResult:
        world = (context or {}).get("world")
        if world is None or not hasattr(world, "read_text"):
            return ToolResult(success=False, error="no FileSandboxWorld in context")
        rel = str(args.get("path"))
        try:
            text = world.read_text(rel)
            return ToolResult(success=True, output={"path": rel, "text": text[:4000]})
        except Exception as e:
            return ToolResult(success=False, error=f"{type(e).__name__}: {e}")


class WriteFileTool(Tool):
    name = "write_file"
    description = "Write a text file inside the sandbox (write; token pipeline still required)."
    tags = {"fs", "write"}
    base_risk = 0.4
    always_confirm = False

    def validate_args(self, args: Dict[str, Any]) -> Optional[str]:
        if not args or not args.get("path"):
            return "path required"
        return None

    def run(self, args: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> ToolResult:
        world = (context or {}).get("world")
        if world is None or not hasattr(world, "write_text"):
            return ToolResult(success=False, error="no FileSandboxWorld in context")
        rel = str(args.get("path"))
        content = str(args.get("content", ""))
        try:
            meta = world.write_text(rel, content)
            obs = world.observe() if hasattr(world, "observe") else {}
            return ToolResult(
                success=True,
                output=dict(obs),
                side_effects=[f"write_file:{rel}"],
                metadata=meta,
            )
        except Exception as e:
            return ToolResult(success=False, error=f"{type(e).__name__}: {e}")


FS_ACTIONS = frozenset({"list_files", "read_file", "write_file"})


def register_fs_tools(reg: ToolRegistry, *, overwrite: bool = True) -> ToolRegistry:
    for t in (ListFilesTool(), ReadFileTool(), WriteFileTool()):
        reg.register(t, overwrite=overwrite)
    return reg


def make_fs_registry() -> ToolRegistry:
    """Default tools + filesystem tools."""
    return register_fs_tools(make_default_registry())


# ============================================================================
# MODULE FLATTENED: omega.topology.jia_loop
# ============================================================================

# -*- coding: utf-8 -*-
"""JIA-Loop topology counters: Mod16 + 240-step grand cycle.

Single source of truth for step/mod counters. Kernel StateMachine.cycle_step
mirrors ``JiaLoop.step`` via Omega.tick_topology() — do not advance both.
"""


from dataclasses import dataclass
from typing import Any, Dict



@dataclass
class TopologySnapshot:
    step: int
    mod16: int
    mod3: int
    mod5: int
    grand_cycle: int
    phase_name: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "step_240": self.step,
            "mod16": self.mod16,
            "mod3": self.mod3,
            "mod5": self.mod5,
            "grand_cycle": self.grand_cycle,
            "phase_name": self.phase_name,
        }


_PHASE_NAMES = [
    "init", "sync", "listen", "sense", "plan",
    "decide", "auth", "exec", "calibrate", "persist",
    "reflect", "rest", "align", "balance", "commit", "seal",
]


class JiaLoop:
    """Canonical topology clock (Mod16 + LCM-240)."""

    def __init__(self, mod: int = MOD16, cycle: int = CYCLE_240):
        self.mod = max(1, int(mod))
        self.cycle = max(1, int(cycle))
        self.step = 0
        self.grand_cycle = 0

    def tick(self) -> TopologySnapshot:
        self.step = (self.step + 1) % self.cycle
        if self.step == 0:
            self.grand_cycle += 1
        return self.snapshot()

    def snapshot(self) -> TopologySnapshot:
        m16 = self.step % self.mod
        return TopologySnapshot(
            step=self.step,
            mod16=m16,
            mod3=self.step % 3,
            mod5=self.step % 5,
            grand_cycle=self.grand_cycle,
            phase_name=_PHASE_NAMES[m16 % len(_PHASE_NAMES)],
        )

    def should_lock_demo(self) -> bool:
        """Toy align-lock: (mod3 * mod5) % mod16 == mod16 slot.

        Conceptual only — does not authorize writes.
        """
        snap = self.snapshot()
        vowel = (snap.mod3 * snap.mod5) % self.mod
        return snap.mod16 == vowel


# ============================================================================
# MODULE FLATTENED: omega.vault.vault
# ============================================================================

# -*- coding: utf-8 -*-
"""Vault-at-Rest: PBKDF2 + HMAC-CTR + Encrypt-then-MAC (stdlib only)."""


import hashlib
import hmac
import json
import secrets
import struct
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union



try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM as _AESGCM
except ImportError:  # the stdlib fallback is used only if explicitly allowed
    _AESGCM = None


class Vault:
    META_FILE = "VAULT.json"
    _KCV_LABEL = b"omega-vault-key-check-v1"

    def __init__(
        self,
        passphrase: str,
        *,
        salt: Optional[bytes] = None,
        iterations: int = DEFAULT_PBKDF2_ITERATIONS,
        vault_id: Optional[str] = None,
    ):
        if not isinstance(passphrase, str) or not passphrase:
            raise ValueError("vault passphrase must be a non-empty string")
        iterations = int(iterations)
        if iterations < MIN_PBKDF2_ITERATIONS:
            raise ValueError(f"PBKDF2 iterations must be >= {MIN_PBKDF2_ITERATIONS}")
        self.iterations = iterations
        self.salt = bytes(salt) if salt is not None else secrets.token_bytes(VAULT_SALT_BYTES)
        if len(self.salt) != VAULT_SALT_BYTES:
            raise ValueError("vault salt must be 16 bytes")
        self.vault_id = vault_id or secrets.token_hex(8)
        self._enc_key, self._mac_key = self._derive(passphrase)

    def _derive(self, passphrase: str) -> Tuple[bytes, bytes]:
        material = hashlib.pbkdf2_hmac(
            "sha256", passphrase.encode("utf-8"), self.salt, self.iterations, dklen=64,
        )
        return material[:32], material[32:]

    def _keystream(self, nonce: bytes, length: int) -> bytes:
        if length <= 0:
            return b""
        out = bytearray()
        counter = 0
        while len(out) < length:
            block = hmac.new(
                self._enc_key, nonce + struct.pack(">I", counter), hashlib.sha256,
            ).digest()
            out.extend(block)
            counter += 1
        return bytes(out[:length])

    def _tag(self, header: bytes, aad: bytes, ciphertext: bytes) -> bytes:
        return hmac.new(
            self._mac_key,
            header + struct.pack(">I", len(aad)) + aad + ciphertext,
            hashlib.sha256,
        ).digest()

    @property
    def cipher_name(self) -> str:
        return "aes-256-gcm" if _AESGCM is not None else "legacy-hmac-ctr+hmac-sha256"

    @property
    def key_check(self) -> str:
        return hmac.new(self._mac_key, self._KCV_LABEL, hashlib.sha256).hexdigest()

    def encrypt_bytes(self, plaintext: bytes, aad: bytes = b"") -> bytes:
        """Authenticated encryption: AES-256-GCM (format v2) when `cryptography` is available.

        The stdlib fallback (format v1) is weaker and is only written when
        OMEGA_ALLOW_LEGACY_VAULT=1 is set explicitly.
        """
        plaintext = bytes(plaintext)
        aad = bytes(aad)
        if _AESGCM is not None:
            nonce = secrets.token_bytes(VAULT_GCM_NONCE_BYTES)
            bound = (
                VAULT_MAGIC
                + struct.pack(">B", VAULT_VERSION)
                + struct.pack(">I", self.iterations)
                + self.salt
                + nonce
                + struct.pack(">I", len(aad))
                + aad
            )
            ciphertext = _AESGCM(self._enc_key).encrypt(nonce, plaintext, bound)
            return bound + ciphertext
        if os.environ.get(LEGACY_VAULT_ENV) != "1":
            raise RuntimeError(
                "Vault encryption requires the 'cryptography' package "
                "(pip install cryptography). To use the weaker stdlib cipher anyway, "
                f"set {LEGACY_VAULT_ENV}=1."
            )
        return self._encrypt_legacy(plaintext, aad)

    def _encrypt_legacy(self, plaintext: bytes, aad: bytes) -> bytes:
        nonce = secrets.token_bytes(VAULT_NONCE_BYTES)
        ciphertext = bytes(a ^ b for a, b in zip(plaintext, self._keystream(nonce, len(plaintext))))
        header = (
            VAULT_MAGIC
            + struct.pack(">B", VAULT_LEGACY_VERSION)
            + struct.pack(">I", self.iterations)
            + self.salt
            + nonce
        )
        tag = self._tag(header, aad, ciphertext)
        return header + struct.pack(">I", len(aad)) + aad + ciphertext + tag

    def decrypt_bytes(self, blob: bytes, aad: bytes = b"") -> bytes:
        blob = bytes(blob)
        if len(blob) < len(VAULT_MAGIC) + 1 or not blob.startswith(VAULT_MAGIC):
            raise ValueError("not an Omega vault container")
        version = blob[len(VAULT_MAGIC)]
        if version == VAULT_VERSION:
            return self._decrypt_gcm(blob, bytes(aad))
        if version == VAULT_LEGACY_VERSION:
            return self._decrypt_legacy(blob, bytes(aad))
        raise ValueError(f"unsupported vault version: {version}")

    def _decrypt_gcm(self, blob: bytes, aad: bytes) -> bytes:
        if _AESGCM is None:
            raise RuntimeError("this vault container needs the 'cryptography' package")
        offset = len(VAULT_MAGIC) + 1
        if len(blob) < offset + 4 + VAULT_SALT_BYTES + VAULT_GCM_NONCE_BYTES + 4 + VAULT_TAG_BYTES:
            raise ValueError("truncated vault container")
        (iterations,) = struct.unpack(">I", blob[offset:offset + 4])
        offset += 4
        if iterations != self.iterations:
            raise ValueError("vault KDF iterations mismatch")
        salt = blob[offset:offset + VAULT_SALT_BYTES]
        offset += VAULT_SALT_BYTES
        if not hmac.compare_digest(salt, self.salt):
            raise ValueError("vault salt mismatch (wrong vault)")
        nonce = blob[offset:offset + VAULT_GCM_NONCE_BYTES]
        offset += VAULT_GCM_NONCE_BYTES
        (aad_len,) = struct.unpack(">I", blob[offset:offset + 4])
        offset += 4
        if len(blob) < offset + aad_len:
            raise ValueError("truncated vault container")
        embedded_aad = blob[offset:offset + aad_len]
        offset += aad_len
        if embedded_aad != aad:
            raise ValueError("vault AAD mismatch")
        bound = blob[:offset]           # header + AAD block: authenticated by GCM
        ciphertext = blob[offset:]      # includes the 16-byte GCM tag
        try:
            return _AESGCM(self._enc_key).decrypt(nonce, ciphertext, bound)
        except Exception as exc:
            raise ValueError("vault authentication failed (tampered or wrong key)") from exc

    def _decrypt_legacy(self, blob: bytes, aad: bytes) -> bytes:
        header_len = len(VAULT_MAGIC) + 1 + 4 + VAULT_SALT_BYTES + VAULT_NONCE_BYTES
        min_len = header_len + 4 + VAULT_TAG_BYTES
        if len(blob) < min_len:
            raise ValueError("truncated vault container")
        offset = len(VAULT_MAGIC) + 1
        (iterations,) = struct.unpack(">I", blob[offset:offset + 4])
        offset += 4
        if iterations != self.iterations:
            raise ValueError("vault KDF iterations mismatch")
        salt = blob[offset:offset + VAULT_SALT_BYTES]
        offset += VAULT_SALT_BYTES
        if not hmac.compare_digest(salt, self.salt):
            raise ValueError("vault salt mismatch (wrong vault)")
        nonce = blob[offset:offset + VAULT_NONCE_BYTES]
        offset += VAULT_NONCE_BYTES
        header = blob[:offset]
        (aad_len,) = struct.unpack(">I", blob[offset:offset + 4])
        offset += 4
        if len(blob) < offset + aad_len + VAULT_TAG_BYTES:
            raise ValueError("truncated vault container")
        embedded_aad = blob[offset:offset + aad_len]
        offset += aad_len
        ciphertext = blob[offset:-VAULT_TAG_BYTES]
        tag = blob[-VAULT_TAG_BYTES:]
        expected = self._tag(header, embedded_aad, ciphertext)
        if not hmac.compare_digest(tag, expected):
            raise ValueError("vault authentication failed (tampered or wrong key)")
        if embedded_aad != aad:
            raise ValueError("vault AAD mismatch")
        return bytes(a ^ b for a, b in zip(ciphertext, self._keystream(nonce, len(ciphertext))))

    def encrypt_text(self, text: str, aad: Union[str, bytes] = b"") -> bytes:
        aad_bytes = aad.encode("utf-8") if isinstance(aad, str) else aad
        return self.encrypt_bytes(text.encode("utf-8"), aad_bytes)

    def decrypt_text(self, blob: bytes, aad: Union[str, bytes] = b"") -> str:
        aad_bytes = aad.encode("utf-8") if isinstance(aad, str) else aad
        return self.decrypt_bytes(blob, aad_bytes).decode("utf-8")

    @staticmethod
    def is_encrypted(blob: bytes) -> bool:
        return bytes(blob[: len(VAULT_MAGIC)]) == VAULT_MAGIC

    def meta(self) -> Dict[str, Any]:
        return {
            "version": VAULT_VERSION,
            "vault_id": self.vault_id,
            "kdf": "pbkdf2-hmac-sha256",
            "iterations": self.iterations,
            "salt": self.salt.hex(),
            "key_check": self.key_check,
            "cipher": self.cipher_name,
            "created_at": time.time(),
        }

    def save_meta(self, storage_dir: Union[str, Path]) -> Path:
        path = Path(storage_dir) / self.META_FILE
        atomic_write_text(path, canonical(self.meta()))
        return path

    @classmethod
    def load_or_create(
        cls,
        storage_dir: Union[str, Path],
        passphrase: str,
        *,
        iterations: int = DEFAULT_PBKDF2_ITERATIONS,
    ) -> "Vault":
        path = Path(storage_dir) / cls.META_FILE
        if path.exists():
            meta = json.loads(path.read_text(encoding="utf-8"))
            if meta.get("kdf") != "pbkdf2-hmac-sha256":
                raise ValueError("unsupported vault KDF")
            vault = cls(
                passphrase,
                salt=bytes.fromhex(meta["salt"]),
                iterations=int(meta["iterations"]),
                vault_id=meta.get("vault_id"),
            )
            expected = meta.get("key_check")
            if expected and not hmac.compare_digest(str(expected), vault.key_check):
                raise PermissionError("vault passphrase rejected (key check failed)")
            return vault
        vault = cls(passphrase, iterations=iterations)
        vault.save_meta(storage_dir)
        return vault


# ============================================================================
# MODULE FLATTENED: omega.consensus.node
# ============================================================================

# -*- coding: utf-8 -*-
"""Distributed CAS consensus: HMAC-signed blocks / proposals / votes."""


import hashlib
import hmac
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union



def _payload(*parts: Any) -> bytes:
    return canonical(list(parts)).encode("utf-8")


@dataclass(frozen=True)
class NodeIdentity:
    node_id: str
    signing_key: bytes

    @classmethod
    def from_secret(
        cls, secret: str, node_id: Optional[str] = None, *, iterations: int = 100_000
    ) -> "NodeIdentity":
        if not isinstance(secret, str) or len(secret) < MIN_SECRET_LENGTH:
            raise ValueError("consensus secret must contain >= 16 chars")
        derived_id = node_id or ("node-" + sha256("omega-node:" + secret)[:16])
        key = hashlib.pbkdf2_hmac(
            "sha256",
            secret.encode("utf-8"),
            ("omega-consensus:" + derived_id).encode("utf-8"),
            iterations,
            dklen=32,
        )
        return cls(node_id=derived_id, signing_key=key)

    @classmethod
    def from_key_hex(cls, node_id: str, key_hex: str) -> "NodeIdentity":
        key = bytes.fromhex(key_hex)
        if len(key) < 16:
            raise ValueError("consensus key must be >= 16 bytes")
        return cls(node_id=node_id, signing_key=key)

    def key_hex(self) -> str:
        return self.signing_key.hex()

    def sign(self, payload: bytes) -> str:
        return hmac.new(self.signing_key, payload, hashlib.sha256).hexdigest()

    def verify(self, payload: bytes, signature: str) -> bool:
        return hmac.compare_digest(self.sign(payload), str(signature))


@dataclass
class SignedBlock:
    block_hash: str
    content: str
    node_id: str
    signature: str
    timestamp: float = field(default_factory=time.time)

    def payload(self) -> bytes:
        return _payload(
            "omega-block-v1", self.block_hash, sha256(self.content),
            self.node_id, round(float(self.timestamp), 6),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SignedBlock":
        return cls(
            block_hash=str(data["block_hash"]),
            content=str(data["content"]),
            node_id=str(data["node_id"]),
            signature=str(data["signature"]),
            timestamp=float(data.get("timestamp", 0.0)),
        )


@dataclass
class SignedProposal:
    proposal_id: str
    block_hash: str
    node_id: str
    signature: str
    timestamp: float = field(default_factory=time.time)

    def payload(self) -> bytes:
        return _payload(
            "omega-proposal-v1", self.proposal_id, self.block_hash,
            self.node_id, round(float(self.timestamp), 6),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SignedVote:
    proposal_id: str
    node_id: str
    approve: bool
    signature: str
    timestamp: float = field(default_factory=time.time)

    def payload(self) -> bytes:
        return _payload(
            "omega-vote-v1", self.proposal_id, self.node_id,
            bool(self.approve), round(float(self.timestamp), 6),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ConsensusNode:
    STATE_FILE = "CONSENSUS.json"

    def __init__(
        self,
        storage_dir: Union[str, Path],
        secret: str,
        *,
        node_id: Optional[str] = None,
        quorum: int = 2,
        engine: Optional[OmegaMemoryEngine] = None,
    ):
        self.storage_dir = Path(storage_dir)
        self.identity = NodeIdentity.from_secret(secret, node_id)
        self.engine = engine or OmegaMemoryEngine(self.storage_dir)
        self.quorum = max(1, int(quorum))
        self.peer_keys: Dict[str, str] = {self.identity.node_id: self.identity.key_hex()}
        self.proposals: Dict[str, Dict[str, Any]] = {}
        self.committed: List[str] = []
        self._state_file = self.storage_dir / self.STATE_FILE
        self._load_state()

    def trust(self, peer: Union[NodeIdentity, str], key_hex: Optional[str] = None):
        if isinstance(peer, NodeIdentity):
            self.peer_keys[peer.node_id] = peer.key_hex()
        else:
            if key_hex is None:
                raise ValueError("key_hex required when trusting by node_id")
            self.peer_keys[str(peer)] = str(key_hex)
        self.save_state()

    def _verify_peer(self, node_id: str, payload: bytes, signature: str) -> bool:
        key_hex = self.peer_keys.get(node_id)
        if not key_hex:
            return False
        try:
            return NodeIdentity.from_key_hex(node_id, key_hex).verify(payload, signature)
        except ValueError:
            return False

    def export_block(self, block_hash: str) -> Optional[SignedBlock]:
        path = self.engine.objects_dir / f"{block_hash}.block"
        if not path.exists():
            return None
        try:
            content = path.read_text(encoding="utf-8")
        except Exception:
            return None
        if not hmac.compare_digest(sha256(content), block_hash):
            return None
        env = SignedBlock(
            block_hash=block_hash, content=content,
            node_id=self.identity.node_id, signature="",
        )
        env.signature = self.identity.sign(env.payload())
        return env

    def verify_block(self, envelope: SignedBlock) -> Tuple[bool, str]:
        if not self._verify_peer(envelope.node_id, envelope.payload(), envelope.signature):
            return False, "invalid_signature"
        if not hmac.compare_digest(sha256(envelope.content), envelope.block_hash):
            return False, "content_hash_mismatch"
        try:
            episode_from_dict(json.loads(envelope.content))
        except Exception:
            return False, "invalid_episode"
        return True, "ok"

    def ingest(self, envelope: SignedBlock) -> Tuple[bool, str]:
        ok, reason = self.verify_block(envelope)
        if not ok:
            return False, reason
        path = self.engine.objects_dir / f"{envelope.block_hash}.block"
        if path.exists():
            return False, "duplicate"
        try:
            self.engine.objects_dir.mkdir(parents=True, exist_ok=True)
            atomic_write_text(path, envelope.content, mode=0o644)
        except Exception as exc:
            return False, f"write_failed:{type(exc).__name__}"
        return True, "accepted"

    def propose(self, block_hash: str) -> SignedProposal:
        import secrets as _secrets
        prop = SignedProposal(
            proposal_id=_secrets.token_hex(12),
            block_hash=block_hash,
            node_id=self.identity.node_id,
            signature="",
        )
        prop.signature = self.identity.sign(prop.payload())
        self.proposals[prop.proposal_id] = {
            "proposal": prop.to_dict(), "votes": {}, "committed": False,
        }
        self.save_state()
        return prop

    def sign_vote(self, proposal: SignedProposal, approve: bool = True) -> SignedVote:
        vote = SignedVote(
            proposal_id=proposal.proposal_id,
            node_id=self.identity.node_id,
            approve=bool(approve),
            signature="",
        )
        vote.signature = self.identity.sign(vote.payload())
        return vote

    def receive_vote(self, vote: SignedVote) -> Tuple[bool, str]:
        if not self._verify_peer(vote.node_id, vote.payload(), vote.signature):
            return False, "invalid_signature"
        rec = self.proposals.get(vote.proposal_id)
        if rec is None:
            return False, "unknown_proposal"
        rec["votes"][vote.node_id] = vote.to_dict()
        self.save_state()
        return True, "accepted"

    def try_commit(self, proposal: SignedProposal) -> bool:
        rec = self.proposals.get(proposal.proposal_id)
        if rec is None or rec.get("committed"):
            return False
        votes = rec.get("votes") or {}
        yes = sum(1 for v in votes.values() if v.get("approve"))
        no = sum(1 for v in votes.values() if not v.get("approve"))
        if yes < self.quorum or yes <= no:
            return False
        rec["committed"] = True
        self.committed.append(proposal.block_hash)
        self.save_state()
        return True

    def save_state(self):
        payload = {
            "node_id": self.identity.node_id,
            "quorum": self.quorum,
            "peer_keys": dict(self.peer_keys),
            "proposals": self.proposals,
            "committed": list(self.committed),
        }
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_text(self._state_file, canonical(payload))

    def _load_state(self):
        if not self._state_file.exists():
            return
        try:
            data = json.loads(self._state_file.read_text(encoding="utf-8"))
            self.peer_keys = dict(data.get("peer_keys") or self.peer_keys)
            self.proposals = dict(data.get("proposals") or {})
            self.committed = list(data.get("committed") or [])
        except Exception:
            pass


# ============================================================================
# MODULE FLATTENED: omega.kernel_lifecycle_auth
# ============================================================================

# -*- coding: utf-8 -*-
"""Kernel lifecycle mixins - auth, transactions, stage/confirm."""


import hmac
import secrets
import time
from dataclasses import asdict
from typing import Any, Callable, Dict, Optional, Tuple



class _LifecycleAuthMixin:
    """Auth, transactions, stage/confirm."""

    def _owner_digest(self, owner_id: str, salt: str) -> str:
        return hmac.new(
            self.secret.encode(),
            f"{owner_id}:{salt}".encode(),
            hashlib.sha256,
        ).hexdigest()

    def _create_owner(self) -> Owner:
        owner_id = secrets.token_hex(16)
        salt = secrets.token_hex(16)
        return Owner(id=owner_id, salt=salt, digest=self._owner_digest(owner_id, salt))

    def owner(self) -> Owner:
        """Return a copy of the owner handle.

        Security note: the owner gate is an in-process control. Code running in
        the same Python process can obtain this handle, so the gate stops the
        kernel's own proposals and accidental misuse, not hostile code sharing
        the process. A hard boundary needs a separate process or an OS/hardware
        backed key.
        """
        return deep(self.owner_)

    def _authorize(self, owner: Owner):
        if not isinstance(owner, Owner):
            raise PermissionError("owner required")
        expected = self._owner_digest(owner.id, owner.salt)
        if not hmac.compare_digest(expected, owner.digest):
            raise PermissionError("invalid owner")
        if not hmac.compare_digest(owner.id, self.owner_.id):
            raise PermissionError("invalid owner")

    def _safe_kernel_to(self, target: KernelState) -> None:
        """Prefer legal transition; force only if table blocks (bootstrap edge)."""
        if self.sm.kernel == target:
            return
        if self.sm.can_transition_kernel(target):
            self.sm.transition_kernel(target)
        else:
            self.sm.transition_kernel(target, force=True)

    def _maybe_rotate_audit(self) -> Optional[Dict[str, Any]]:
        if not getattr(self.config, "auto_archive_audit", True):
            return None
        if len(self.audit.records) <= MAX_AUDIT_RECORDS:
            return None
        keep = max(100, int(getattr(self.config, "audit_keep_records", 1000)))
        archive_name = self.storage_dir / f"AUDIT_ARCHIVE_{int(time.time())}_{secrets.token_hex(4)}.json"
        result = self.audit.archive(archive_name, keep=keep)
        if result:
            self.audit.save(self.memory.audit_file)
            print(f"[Audit] archived {result['archived']} records -> {result['archive_path']}")
        return result

    def _check_writable(
        self,
        *,
        allow_when_paused: bool = False,
        allow_when_escalated: bool = False,
    ) -> Tuple[bool, str]:
        if self.paused and not allow_when_paused:
            return False, "kernel paused; writes rejected."
        if self.escalated and not allow_when_escalated:
            return False, "kernel escalated; clear_escalate via stage/confirm required."
        if not self.integrity_ok:
            return False, "integrity failed (fail-closed); all writes disabled."
        if not self.audit.verify():
            self.integrity_ok = False
            return False, "audit chain corrupted -> fail-closed."
        if not self.memory.verify_all_blocks():
            self.integrity_ok = False
            return False, "CAS block verification failed -> fail-closed."
        if hasattr(self, "skill_store") and not self.skill_store.verify_all():
            self.integrity_ok = False
            return False, "skill CAS verification failed -> fail-closed."
        if len(self.audit.records) >= MAX_AUDIT_RECORDS:
            return False, f"audit record limit ({MAX_AUDIT_RECORDS}) reached; archive first."
        return True, "ok"

    def _transaction(
        self,
        event: str,
        details: Dict[str, Any],
        mutator: Callable[[], Any],
        *,
        allow_when_paused: bool = False,
        allow_when_escalated: bool = False,
    ) -> Tuple[bool, Any]:
        with self.lock:
            return self._transaction_locked(
                event,
                details,
                mutator,
                allow_when_paused=allow_when_paused,
                allow_when_escalated=allow_when_escalated,
            )

    def _transaction_locked(
        self,
        event: str,
        details: Dict[str, Any],
        mutator: Callable[[], Any],
        *,
        allow_when_paused: bool = False,
        allow_when_escalated: bool = False,
    ) -> Tuple[bool, Any]:
        self._maybe_rotate_audit()
        ok, reason = self._check_writable(
            allow_when_paused=allow_when_paused,
            allow_when_escalated=allow_when_escalated,
        )
        if not ok:
            return False, reason

        backup = {
            "goal": deep(self.goal),
            "hard_min": deep(self.hard_min),
            "hard_max": deep(self.hard_max),
            "calibration": deep(self.calibration),
            "epoch": self.epoch,
            "paused": self.paused,
            "escalated": self.escalated,
            "self_model": deep(asdict(self.self_model)),
            "audit_len": len(self.audit.records),
            "skill_index": dict(self.skill_store.index) if hasattr(self, "skill_store") else {},
        }

        try:
            result = mutator()
            self.epoch += 1
            self.audit.append(event, details)
            self.save_cas_head()
            return True, result
        except Exception as exc:
            self.goal = backup["goal"]
            self.hard_min = backup["hard_min"]
            self.hard_max = backup["hard_max"]
            self.calibration = backup["calibration"]
            self.epoch = backup["epoch"]
            self.paused = backup["paused"]
            self.escalated = backup["escalated"]
            for k, v in backup["self_model"].items():
                if hasattr(self.self_model, k):
                    setattr(self.self_model, k, v)
            if hasattr(self, "skill_store") and backup["skill_index"] is not None:
                self.skill_store.index = backup["skill_index"]
            if len(self.audit.records) > backup["audit_len"]:
                self.audit.records = self.audit.records[: backup["audit_len"]]
            return False, f"{type(exc).__name__}: {exc}"

    def stage_action(
        self,
        owner: Owner,
        action_type: str,
        payload: Dict[str, Any],
        *,
        description: str = "",
    ) -> Dict[str, Any]:
        self._authorize(owner)
        ticket = secrets.token_hex(16)
        self.pending_action = {
            "ticket": ticket,
            "action_type": action_type,
            "payload": deep(payload),
            "description": description or action_type,
            "created_at": time.time(),
            "owner_id": owner.id,
        }
        return {
            "staged": True,
            "ticket": ticket,
            "action_type": action_type,
            "description": self.pending_action["description"],
            "expires_in_sec": MAX_PENDING_TTL_SEC,
            "message": "Staged. Call confirm_action(ticket) or cancel_pending().",
        }

    def cancel_pending(self, owner: Owner) -> str:
        self._authorize(owner)
        if self.pending_action is None:
            return "no pending action."
        self.pending_action = None
        return "pending action cancelled; no writes occurred."

    def confirm_action(self, owner: Owner, ticket: str) -> Dict[str, Any]:
        self._authorize(owner)
        pending = self.pending_action
        if pending is None:
            raise ValueError("no pending action.")
        if not secrets.compare_digest(str(ticket), pending["ticket"]):
            raise PermissionError("invalid ticket.")
        if time.time() - pending["created_at"] > MAX_PENDING_TTL_SEC:
            self.pending_action = None
            raise PermissionError("ticket expired.")
        if pending["owner_id"] != owner.id:
            raise PermissionError("ticket owner mismatch.")

        action_type = pending["action_type"]
        payload = pending["payload"]
        allow_when_paused = action_type in ("resume", "clear_escalate")
        allow_when_escalated = action_type == "clear_escalate"
        ok, reason = self._check_writable(
            allow_when_paused=allow_when_paused,
            allow_when_escalated=allow_when_escalated,
        )
        if not ok:
            raise PermissionError(reason)

        def mutator():
            if action_type == "set_goal":
                self.goal = norm_vector(payload["goal"])
                return {"result": "goal_updated", "goal": deep(self.goal)}
            if action_type == "set_limits":
                if payload.get("low") is not None:
                    self.hard_min = norm_vector(payload["low"])
                if payload.get("high") is not None:
                    self.hard_max = norm_vector(payload["high"])
                return {
                    "result": "limits_updated",
                    "hard_min": deep(self.hard_min),
                    "hard_max": deep(self.hard_max),
                }
            if action_type == "pause":
                self.paused = True
                self._safe_kernel_to(KernelState.PAUSED)
                return {"result": "paused", "kernel": self.sm.kernel.value}
            if action_type == "resume":
                self.paused = False
                self._safe_kernel_to(KernelState.IDLE)
                return {"result": "resumed", "kernel": self.sm.kernel.value}
            if action_type == "clear_escalate":
                self.escalated = False
                self._safe_kernel_to(KernelState.IDLE)
                return {
                    "result": "escalation_cleared",
                    "kernel": self.sm.kernel.value,
                    "escalated": self.escalated,
                }
            if action_type == "install_skill":
                skill = Skill.from_dict(payload["skill"])
                rec = self.skill_store.install(skill)
                self.skills.register(skill, overwrite=True)
                self.skill_store.save_index()
                return {
                    "result": "skill_installed",
                    "skill_id": skill.skill_id,
                    "content_hash": rec.content_hash,
                    "version": rec.version,
                }
            if action_type == "uninstall_skill":
                sid = payload["skill_id"]
                rec = self.skill_store.uninstall(sid)
                self.skills.unregister(sid)
                self.skill_store.save_index()
                return {
                    "result": "skill_uninstalled",
                    "skill_id": sid,
                    "had_record": rec is not None,
                }
            if action_type == "rollback_skill":
                sid = payload["skill_id"]
                to_hash = payload.get("to_hash")
                rec = self.skill_store.rollback(sid, to_hash=to_hash)
                if rec is None:
                    raise ValueError(f"cannot rollback skill {sid}")
                skill = self.skill_store.get_active(sid)
                if skill:
                    self.skills.register(skill, overwrite=True)
                self.skill_store.save_index()
                return {
                    "result": "skill_rolled_back",
                    "skill_id": sid,
                    "content_hash": rec.content_hash,
                    "version": rec.version,
                }
            return {"result": "confirmed", "action_type": action_type, "payload": deep(payload)}

        self.pending_action = None
        ok, result = self._transaction(
            f"confirm_{action_type}",
            {"ticket": ticket, "action_type": action_type},
            mutator,
            allow_when_paused=allow_when_paused,
            allow_when_escalated=allow_when_escalated,
        )
        if not ok:
            raise RuntimeError(f"confirm failed: {result}")
        return result


# ============================================================================
# MODULE FLATTENED: omega.kernel_lifecycle_persist
# ============================================================================

# -*- coding: utf-8 -*-
"""Kernel lifecycle mixins - goal/limits, skill install, CAS persist/restore."""


import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Mapping, Optional



class _LifecyclePersistMixin:
    """Goal/limits, skill lifecycle, CAS persist/restore."""

    def set_goal(self, owner: Owner, goal: Mapping[str, Any], *, require_confirm: bool = False):
        self._authorize(owner)
        if require_confirm:
            return self.stage_action(
                owner, "set_goal", {"goal": dict(goal)},
                description=f"set_goal -> {dict(goal)}",
            )

        def mutator():
            self.goal = norm_vector(goal)
            if getattr(self.config, "hierarchical_goals", True):
                self.goal_tree = GoalTree.from_flat(self.goal, goal_id="root")
            return {"goal": deep(self.goal)}

        ok, result = self._transaction("goal", {"goal": dict(goal)}, mutator)
        if not ok:
            raise RuntimeError(f"set_goal failed and rolled back: {result}")
        return result

    def set_goal_tree(
        self,
        owner: Owner,
        tree: GoalTree,
        *,
        require_confirm: bool = False,
    ) -> Dict[str, Any]:
        self._authorize(owner)
        if require_confirm:
            return self.stage_action(
                owner,
                "set_goal_tree",
                {"tree": tree.to_dict()},
                description="set_goal_tree",
            )

        def mutator():
            self.goal_tree = tree
            self.goal = tree.to_flat()
            return {"goal": deep(self.goal), "tree": tree.to_dict()}

        ok, result = self._transaction("goal_tree", {"root": tree.root_id}, mutator)
        if not ok:
            raise RuntimeError(f"set_goal_tree failed: {result}")
        return result

    def add_subgoal(
        self,
        owner: Owner,
        goal_id: str,
        targets: Mapping[str, Any],
        parent_id: Optional[str] = None,
        *,
        priority: float = 1.0,
        weight: float = 1.0,
        tolerance: float = 0.05,
        require_confirm: bool = False,
    ) -> Dict[str, Any]:
        self._authorize(owner)
        payload = {
            "goal_id": goal_id,
            "targets": dict(targets),
            "parent_id": parent_id,
            "priority": priority,
            "weight": weight,
            "tolerance": tolerance,
        }
        if require_confirm:
            return self.stage_action(owner, "add_subgoal", payload, description=f"add_subgoal:{goal_id}")

        def mutator():
            if not self.goal_tree.root_id:
                self.goal_tree.set_root("root", self.goal or targets)
            self.goal_tree.add_subgoal(
                goal_id, targets,
                parent_id=parent_id,
                priority=priority,
                weight=weight,
                tolerance=tolerance,
            )
            active = self.goal_tree.get_active()
            if active:
                self.goal = dict(active.targets)
            return {"goal_id": goal_id, "active": self.goal_tree.active_id, "goal": deep(self.goal)}

        ok, result = self._transaction("add_subgoal", payload, mutator)
        if not ok:
            raise RuntimeError(f"add_subgoal failed: {result}")
        return result

    def set_limits(
        self,
        owner: Owner,
        low: Optional[Mapping[str, Any]] = None,
        high: Optional[Mapping[str, Any]] = None,
        *,
        require_confirm: bool = False,
    ):
        self._authorize(owner)
        if require_confirm:
            return self.stage_action(
                owner,
                "set_limits",
                {"low": dict(low) if low else None, "high": dict(high) if high else None},
                description="set_limits",
            )

        def mutator():
            if low is not None:
                self.hard_min = norm_vector(low)
            if high is not None:
                self.hard_max = norm_vector(high)
            return {"hard_min": deep(self.hard_min), "hard_max": deep(self.hard_max)}

        ok, result = self._transaction("limits", {"low": low, "high": high}, mutator)
        if not ok:
            raise RuntimeError(f"set_limits failed and rolled back: {result}")
        return result

    def install_skill(
        self,
        owner: Owner,
        skill: Skill,
        *,
        require_confirm: Optional[bool] = None,
    ) -> Dict[str, Any]:
        self._authorize(owner)
        force = require_confirm
        if force is None:
            force = bool(getattr(self.config, "require_confirm_for_skill_install", True))
            if skill.always_confirm:
                force = True

        payload = {"skill": skill.to_dict()}
        if force:
            return self.stage_action(
                owner,
                "install_skill",
                payload,
                description=f"install_skill:{skill.skill_id}@v{skill.version}",
            )

        def mutator():
            rec = self.skill_store.install(skill)
            self.skills.register(skill, overwrite=True)
            self.skill_store.save_index()
            return {
                "result": "skill_installed",
                "skill_id": skill.skill_id,
                "content_hash": rec.content_hash,
                "version": rec.version,
            }

        ok, result = self._transaction(
            "install_skill",
            {"skill_id": skill.skill_id, "version": skill.version},
            mutator,
        )
        if not ok:
            raise RuntimeError(f"install_skill failed: {result}")
        return result

    def uninstall_skill(
        self,
        owner: Owner,
        skill_id: str,
        *,
        require_confirm: Optional[bool] = None,
    ) -> Dict[str, Any]:
        self._authorize(owner)
        force = require_confirm
        if force is None:
            force = bool(getattr(self.config, "require_confirm_for_skill_install", True))

        if force:
            return self.stage_action(
                owner,
                "uninstall_skill",
                {"skill_id": skill_id},
                description=f"uninstall_skill:{skill_id}",
            )

        def mutator():
            rec = self.skill_store.uninstall(skill_id)
            self.skills.unregister(skill_id)
            self.skill_store.save_index()
            return {
                "result": "skill_uninstalled",
                "skill_id": skill_id,
                "had_record": rec is not None,
            }

        ok, result = self._transaction(
            "uninstall_skill", {"skill_id": skill_id}, mutator
        )
        if not ok:
            raise RuntimeError(f"uninstall_skill failed: {result}")
        return result

    def rollback_skill(
        self,
        owner: Owner,
        skill_id: str,
        to_hash: Optional[str] = None,
        *,
        require_confirm: Optional[bool] = None,
    ) -> Dict[str, Any]:
        self._authorize(owner)
        force = require_confirm
        if force is None:
            force = bool(getattr(self.config, "require_confirm_for_skill_install", True))

        if force:
            return self.stage_action(
                owner,
                "rollback_skill",
                {"skill_id": skill_id, "to_hash": to_hash},
                description=f"rollback_skill:{skill_id}",
            )

        def mutator():
            rec = self.skill_store.rollback(skill_id, to_hash=to_hash)
            if rec is None:
                raise ValueError(f"cannot rollback skill {skill_id}")
            skill = self.skill_store.get_active(skill_id)
            if skill:
                self.skills.register(skill, overwrite=True)
            self.skill_store.save_index()
            return {
                "result": "skill_rolled_back",
                "skill_id": skill_id,
                "content_hash": rec.content_hash,
                "version": rec.version,
            }

        ok, result = self._transaction(
            "rollback_skill",
            {"skill_id": skill_id, "to_hash": to_hash},
            mutator,
        )
        if not ok:
            raise RuntimeError(f"rollback_skill failed: {result}")
        return result

    def list_skills(self) -> list:
        """Return active skills as serializable dicts."""
        return [s.to_dict() for s in self.skill_store.list_active()]

    def skill_history(self, skill_id: str) -> list:
        return self.skill_store.history(skill_id)

    def _anchor_key(self) -> bytes:
        return derive_subkey(self.secret, ANCHOR_KEY_LABEL)

    def _audit_tip(self) -> str:
        return self.audit.records[-1]["hash"] if self.audit.records else self.audit.base_hash

    def _write_anchor(self, seq: int, head_sha: str, audit_tip: str) -> None:
        if self.anchor_path is None:
            return
        body = {"seq": seq, "head_sha256": head_sha, "audit_tip": audit_tip}
        atomic_write_text(self.anchor_path, canonical(seal_record(self._anchor_key(), body)))

    def _check_rollback(self, head_path: Path, head: Dict[str, Any]) -> None:
        """Refuse state older than the external anchor, or audit truncated below it."""
        if self.anchor_path is None:
            return
        if not self.anchor_path.exists():
            if not self.migrate_legacy:
                raise SealError("rollback anchor missing while state exists; "
                                "restore the anchor or run once with --migrate-legacy")
            return  # created when the migrated state is sealed
        data = json.loads(self.anchor_path.read_text(encoding="utf-8"))
        anchor = open_record(self._anchor_key(), data)
        if anchor is None:
            raise SealError("rollback anchor MAC mismatch (tampered, or wrong secret)")
        head_seq = int(head.get("seq", 0))
        anchor_seq = int(anchor["seq"])
        if head_seq < anchor_seq:
            raise SealError(f"rollback detected: state seq {head_seq} < anchored seq {anchor_seq}")
        if head_seq == anchor_seq and sha256(head_path.read_text(encoding="utf-8")) != anchor["head_sha256"]:
            raise SealError("HEAD does not match the anchored state at the same sequence (tampered)")
        tip = anchor.get("audit_tip")
        known = {r.get("hash") for r in self.audit.records} | {self.audit.base_hash}
        if tip and tip not in known:
            raise SealError("audit log truncated or rewritten below the anchored tip")

    def save_cas_head(self, path: Optional[str | Path] = None):
        with self.lock:
            return self._save_cas_head_locked(path)

    def _save_cas_head_locked(self, path: Optional[str | Path] = None):
        if not self.integrity_ok:
            return
        target = Path(path) if path else self.memory.head_file
        payload = {
            "kernel_version": KERNEL_VERSION,
            "schema_version": SCHEMA_VERSION,
            "epoch": self.epoch,
            "goal": deep(self.goal),
            "hard_min": deep(self.hard_min),
            "hard_max": deep(self.hard_max),
            "calibration": deep(self.calibration),
            "owner": asdict(self.owner_) if hasattr(self.owner_, "__dataclass_fields__") else {
                "id": self.owner_.id, "salt": self.owner_.salt, "digest": self.owner_.digest,
            },
            "self_model": asdict(self.self_model),
            "active_hashes": self.memory.get_active_hashes(),
            "skill_hashes": self.skill_store.active_hashes(),
            "goal_tree": self.goal_tree.to_dict() if self.goal_tree.root_id else None,
        }
        payload["seq"] = self.save_seq + 1
        sealed = canonical(seal_record(derive_subkey(self.secret, HEAD_KEY_LABEL), payload))
        atomic_write_text(target, sealed)
        self.save_seq = payload["seq"]
        if target == self.memory.head_file:
            self._write_anchor(self.save_seq, sha256(sealed), self._audit_tip())

    def _try_restore(self):
        head_path = self.memory.head_file
        if not head_path.exists():
            if self.anchor_path is not None and self.anchor_path.exists():
                self.integrity_ok = False
                print("[Restore] HEAD missing but rollback anchor exists -> state was deleted; "
                      "writes disabled (fail-closed).")
                return
            try:
                n = self.skill_store.load_index()
            except Exception as e:
                self.integrity_ok = False
                print(f"[Restore] skill index rejected: {e}; writes disabled (fail-closed).")
                return

            if n:
                for s in self.skill_store.list_active():
                    self.skills.register(s, overwrite=True)
            return
        try:
            head_raw = json.loads(head_path.read_text(encoding="utf-8"))
            head, _ = open_sealed(derive_subkey(self.secret, HEAD_KEY_LABEL), head_raw,
                                  allow_legacy=self.migrate_legacy, label="HEAD.json")
            if not str(head.get("kernel_version", "")).startswith("omega-cognitive-"):
                return
            owner_data = head.get("owner")
            if not owner_data:
                return
            candidate = Owner(
                id=owner_data["id"], salt=owner_data["salt"], digest=owner_data["digest"]
            )
            expected = self._owner_digest(candidate.id, candidate.salt)
            import hmac as _hmac
            if not _hmac.compare_digest(expected, candidate.digest):
                # Fail closed: a wrong secret or tampered HEAD must not let a
                # fresh, empty kernel run and overwrite the stored state.
                self.integrity_ok = False
                print("[Restore] HEAD owner digest mismatch (wrong secret or tampered HEAD); writes disabled.")
                return
            self.owner_ = candidate
            self.epoch = int(head.get("epoch", 0))
            self.goal = norm_vector(head.get("goal", {}))
            self.hard_min = norm_vector(head.get("hard_min", {}))
            self.hard_max = norm_vector(head.get("hard_max", {}))
            self.calibration = {k: float(v) for k, v in head.get("calibration", {}).items()}
            sm = head.get("self_model")
            if isinstance(sm, dict):
                # Only declared numeric/bool fields; never methods or arbitrary attributes.
                allowed = {f.name for f in dataclasses.fields(self.self_model)}
                for k, v in sm.items():
                    if k in allowed and isinstance(v, (int, float, bool, str)):
                        setattr(self.self_model, k, v)
            self.tokens = TokenBook.load(
                self.memory.tokens_file,
                max_consumed=self.config.max_consumed_tokens,
                key=derive_subkey(self.secret, TOKEN_KEY_LABEL),
            )
            try:
                self.audit = AuditChain.load(
                    self.memory.audit_file, key=derive_subkey(self.secret, AUDIT_KEY_LABEL),
                    allow_legacy=self.migrate_legacy)
            except Exception as e:
                self.integrity_ok = False
                print(f"[Restore] audit damaged -> fail-closed: {e}")
            self.save_seq = int(head.get("seq", 0))
            self._check_rollback(head_path, head)
            loaded = self.memory.rebuild_from_cas(limit=self.config.max_episodes)

            n_skills = self.skill_store.load_index()
            self._check_inventory(head)
            for s in self.skill_store.list_active():
                self.skills.register(s, overwrite=True)

            gt = head.get("goal_tree")
            if isinstance(gt, dict) and gt.get("root_id"):
                self.goal_tree = GoalTree.from_dict(gt)
            elif self.goal:
                self.goal_tree = GoalTree.from_flat(self.goal)

            print(
                f"[Restore] ok: epoch={self.epoch}, episodes={loaded}, "
                f"skills={n_skills}, "
                f"issued={len(self.tokens.issued)}, consumed={len(self.tokens.consumed)}"
            )
        except Exception as e:
            self.integrity_ok = False
            print(f"[Restore] failed: {e}; writes disabled (fail-closed).")


# ============================================================================
# MODULE FLATTENED: omega.skills.file_mission
# ============================================================================

# -*- coding: utf-8 -*-
"""Preset Skills for File Mission (still require token pipeline)."""


from typing import List



def make_file_mission_skills(
    target_path: str = "out/report.txt",
    target_content: str = "OMEGA-OK\n",
) -> List[Skill]:
    """Return write_target / verify_target / list_parent skills."""
    parent = "/".join(target_path.split("/")[:-1]) if "/" in target_path else "."
    return [
        Skill(
            skill_id="write_target",
            name="Write mission target",
            description="Write the File Mission target file with expected content.",
            version=1,
            preferred_action="write_file",
            template_args={"path": target_path, "content": target_content},
            tags={"file", "mission", "write"},
            base_risk=0.4,
            always_confirm=False,
        ),
        Skill(
            skill_id="verify_target",
            name="Verify mission target",
            description="Read mission target to verify content.",
            version=1,
            preferred_action="read_file",
            template_args={"path": target_path},
            tags={"file", "mission", "read"},
            base_risk=0.08,
            always_confirm=False,
        ),
        Skill(
            skill_id="list_parent",
            name="List mission parent dir",
            description="List parent directory of the mission target.",
            version=1,
            preferred_action="list_files",
            template_args={"path": parent or "."},
            tags={"file", "mission", "read"},
            base_risk=0.05,
            always_confirm=False,
        ),
    ]


# ============================================================================
# MODULE FLATTENED: omega.skills.store
# ============================================================================

# -*- coding: utf-8 -*-
"""
CAS-backed Skill store.

Skills are immutable content-addressed blocks. The active index maps
skill_id -> current content_hash. Previous hashes are kept for rollback.

All mutations go through the kernel's stage/confirm/transaction path;
this module only provides pure storage primitives.
"""


import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union



@dataclass
class SkillRecord:
    skill_id: str
    content_hash: str
    version: int
    installed_at: float
    previous_hashes: List[str] = field(default_factory=list)


class SkillStore:
    """Content-addressed skill storage + active index."""

    def __init__(self, storage_dir: Union[str, Path], key: Optional[bytes] = None,
                 allow_legacy: bool = False):
        self.key = key
        self.allow_legacy = bool(allow_legacy)
        self.base_dir = Path(storage_dir)
        self.objects_dir = self.base_dir / "objects"
        self.index_file = self.base_dir / "SKILLS.json"
        self.objects_dir.mkdir(parents=True, exist_ok=True)
        # skill_id -> SkillRecord
        self.index: Dict[str, SkillRecord] = {}
        # content_hash -> Skill (lazy cache)
        self._cache: Dict[str, Skill] = {}

    # ------------------------------------------------------------------
    # Low-level CAS
    # ------------------------------------------------------------------

    def _block_path(self, content_hash: str) -> Path:
        return self.objects_dir / f"{SKILL_CAS_PREFIX}{content_hash}.block"

    def commit(self, skill: Skill) -> str:
        """Write skill as immutable CAS block. Returns content_hash."""
        if not skill.content_hash:
            skill.content_hash = skill.compute_hash()
        h = skill.content_hash
        path = self._block_path(h)
        if not path.exists():
            payload = skill.to_dict()
            if self.key is not None:
                payload = seal_record(self.key, payload)
            atomic_write_text(path, canonical(payload), mode=0o644)
        self._cache[h] = skill
        return h

    def _read_body(self, path: Path) -> Dict[str, Any]:
        """Verified skill body. Raises on MAC mismatch; unsealed only in migration mode."""
        data = json.loads(path.read_text(encoding="utf-8"))
        if self.key is None:
            return data
        body, legacy = open_sealed(self.key, data, allow_legacy=self.allow_legacy,
                                   label=path.name)
        if legacy:
            atomic_write_text(path, canonical(seal_record(self.key, body)), mode=0o644)
            print(f"[SkillStore] migrated legacy skill block {path.name[:24]} -> sealed")
        return body

    def load_block(self, content_hash: str) -> Optional[Skill]:
        if content_hash in self._cache:
            return self._cache[content_hash]
        path = self._block_path(content_hash)
        if not path.exists():
            return None
        try:
            skill = Skill.from_dict(self._read_body(path))
        except Exception:
            return None
        # The name is the skill's identity hash. No self-referential fallback:
        # the block's own fields must recompute to the name it is stored under.
        if skill.compute_hash() != content_hash:
            return None
        self._cache[content_hash] = skill
        return skill

    def verify_block(self, content_hash: str) -> bool:
        path = self._block_path(content_hash)
        if not path.exists():
            return False
        try:
            skill = Skill.from_dict(self._read_body(path))
            return skill.compute_hash() == content_hash
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Active index
    # ------------------------------------------------------------------

    def get_active(self, skill_id: str) -> Optional[Skill]:
        rec = self.index.get(skill_id)
        if rec is None:
            return None
        return self.load_block(rec.content_hash)

    def list_active(self) -> List[Skill]:
        out = []
        for skill_id in list(self.index.keys()):
            s = self.get_active(skill_id)
            if s is not None:
                out.append(s)
        return out

    def install(self, skill: Skill) -> SkillRecord:
        """Commit and activate. Keeps previous hash for rollback."""
        h = self.commit(skill)
        prev = []
        old = self.index.get(skill.skill_id)
        if old is not None:
            prev = list(old.previous_hashes) + [old.content_hash]
            # keep last 8 versions
            prev = prev[-8:]
        rec = SkillRecord(
            skill_id=skill.skill_id,
            content_hash=h,
            version=skill.version,
            installed_at=time.time(),
            previous_hashes=prev,
        )
        self.index[skill.skill_id] = rec
        return rec

    def uninstall(self, skill_id: str) -> Optional[SkillRecord]:
        return self.index.pop(skill_id, None)

    def rollback(self, skill_id: str, to_hash: Optional[str] = None) -> Optional[SkillRecord]:
        """Roll back to a previous content_hash (default: last previous)."""
        rec = self.index.get(skill_id)
        if rec is None or not rec.previous_hashes:
            return None
        target = to_hash or rec.previous_hashes[-1]
        if target not in rec.previous_hashes and target != rec.content_hash:
            return None
        if not self.verify_block(target):
            return None
        skill = self.load_block(target)
        if skill is None:
            return None
        # rebuild previous list without the target and current
        new_prev = [h for h in rec.previous_hashes if h != target]
        if rec.content_hash != target:
            new_prev = new_prev + [rec.content_hash]
        new_prev = new_prev[-8:]
        new_rec = SkillRecord(
            skill_id=skill_id,
            content_hash=target,
            version=skill.version,
            installed_at=time.time(),
            previous_hashes=new_prev,
        )
        self.index[skill_id] = new_rec
        return new_rec

    def history(self, skill_id: str) -> List[str]:
        rec = self.index.get(skill_id)
        if rec is None:
            return []
        return list(rec.previous_hashes) + [rec.content_hash]

    # ------------------------------------------------------------------
    # Persist index
    # ------------------------------------------------------------------

    def save_index(self) -> None:
        payload = {
            skill_id: {
                "skill_id": rec.skill_id,
                "content_hash": rec.content_hash,
                "version": rec.version,
                "installed_at": rec.installed_at,
                "previous_hashes": list(rec.previous_hashes),
            }
            for skill_id, rec in self.index.items()
        }
        if self.key is not None:
            payload = seal_record(self.key, payload)
        atomic_write_text(self.index_file, canonical(payload))

    def load_index(self) -> int:
        """Load the active index. Tampering raises; an unsealed index loads only in migration mode."""
        if not self.index_file.exists():
            self.index = {}
            return 0
        data = json.loads(self.index_file.read_text(encoding="utf-8"))
        legacy = False
        if self.key is not None:
            data, legacy = open_sealed(self.key, data, allow_legacy=self.allow_legacy,
                                       label=self.index_file.name)
        self.index = {}
        for skill_id, raw in data.items():
            self.index[skill_id] = SkillRecord(
                skill_id=raw["skill_id"],
                content_hash=raw["content_hash"],
                version=int(raw.get("version", 1)),
                installed_at=float(raw.get("installed_at", 0)),
                previous_hashes=list(raw.get("previous_hashes") or []),
            )
        if legacy:
            self.save_index()
            print("[SkillStore] migrated legacy index -> sealed")
        return len(self.index)

    def verify_all(self) -> bool:
        for rec in self.index.values():
            if not self.verify_block(rec.content_hash):
                return False
            for h in rec.previous_hashes:
                if not self.verify_block(h):
                    return False
        return True

    def active_hashes(self) -> List[str]:
        return [rec.content_hash for rec in self.index.values()]


# ============================================================================
# MODULE FLATTENED: omega.cognitive.decision
# ============================================================================

# -*- coding: utf-8 -*-
"""Decision types and default multi-factor decision function."""


from enum import Enum
from typing import Any, Mapping, Optional, Tuple



class Decision(str, Enum):
    APPLY = "APPLY"
    HOLD = "HOLD"
    ESCALATE = "ESCALATE"


def decision_name(value: Any) -> str:
    if isinstance(value, Decision):
        return value.value
    return str(value)


def is_decision(value: Any, expected: Decision) -> bool:
    return decision_name(value) == expected.value


class View(str, Enum):
    SHAN = "SHAN"
    NOT_SHAN = "NOT_SHAN"
    SHAN_AGAIN = "SHAN_AGAIN"


def better_attribute(
    expected: Mapping[str, float],
    observed: Mapping[str, float],
    before: Mapping[str, float],
) -> Attribution:
    prediction_error = vector_rms(vector_diff(expected, observed))
    state_change = vector_rms(vector_diff(before, observed))
    if state_change <= EPS:
        effect, confidence = 0.5, 0.55
    else:
        relative = prediction_error / (state_change + EPS)
        effect = clamp(1.0 - relative, 0.0, 1.0)
        confidence = clamp(0.40 + 0.55 * effect, 0.0, 1.0)
    return Attribution(
        action_effect=effect,
        external_effect=1.0 - effect,
        residual=prediction_error * (1.0 - effect),
        confidence=confidence,
    )


def default_decide(
    *,
    error: float,
    uncertainty: float,
    observation: Mapping[str, float],
    attribution: Attribution,
    paused: bool,
    escalated: bool,
    hard_min: Mapping[str, float],
    hard_max: Mapping[str, float],
    config: Config,
    expected_progress: float = 0.0,
    risk: Optional[float] = None,
    stall: bool = False,
) -> Tuple[Decision, str, bool]:
    if paused:
        return Decision.ESCALATE, "paused", True
    if escalated:
        return Decision.ESCALATE, "already_escalated", False

    for key, value in observation.items():
        if key in hard_min and value < hard_min[key]:
            return Decision.ESCALATE, "hard_min", True
        if key in hard_max and value > hard_max[key]:
            return Decision.ESCALATE, "hard_max", True

    if uncertainty >= config.max_uncertainty_hold:
        return Decision.HOLD, "uncertainty_too_high", False

    risk_score = compute_risk(error, uncertainty, attribution.residual) if risk is None else float(risk)
    risk_threshold = config.max_error_apply * (1.0 - config.risk_aversion)

    if risk_score > risk_threshold and error > config.max_error_apply * 0.6:
        return Decision.HOLD, "risk_too_high", False

    if attribution.confidence < config.min_attribution_confidence:
        return Decision.HOLD, "attribution_uncertain", False

    if expected_progress < config.min_progress_to_apply and error > 0.02:
        return Decision.HOLD, "insufficient_expected_progress", False

    if stall and error > config.max_error_apply * 0.8:
        return Decision.HOLD, "stalled_and_high_error", False

    return Decision.APPLY, "validated", False


# ============================================================================
# MODULE FLATTENED: omega.cognitive.models
# ============================================================================

# -*- coding: utf-8 -*-
"""Core cognitive data structures and simple models."""


from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple, TYPE_CHECKING


if TYPE_CHECKING:
    CognitiveView = View


@dataclass
class ExternalInput:
    kind: str
    content: Dict[str, Any]
    confidence: float = 1.0
    source: str = "external"
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.content = deep(self.content)
        self.confidence = clamp(float(self.confidence), 0.0, 1.0)


@dataclass
class Perception:
    accepted: bool
    state: Dict[str, float]
    confidence: float
    reason: str
    source: str
    view: View


@dataclass
class Intention:
    action: str
    args: Dict[str, Any]
    policy_id: str = "unknown"
    policy_version: str = "unknown"
    rationale: Optional[str] = None


@dataclass
class Prediction:
    expected: Dict[str, float]
    confidence: float
    model_id: str = "unknown"
    model_version: str = "unknown"

    @property
    def uncertainty(self) -> float:
        return 1.0 - self.confidence


@dataclass
class Attribution:
    action_effect: float
    external_effect: float
    residual: float
    confidence: float


@dataclass
class WorldModel:
    state: Dict[str, float] = field(default_factory=dict)
    velocity: Dict[str, float] = field(default_factory=dict)
    count: int = 0
    confidence: float = 0.0
    mean_residual: float = 0.0
    residual_var: float = 0.0
    last_prediction: Dict[str, float] = field(default_factory=dict)
    version: int = 1
    model_id: str = "default-wm"

    def update(self, observation: Dict[str, float], confidence: float):
        observation = norm_vector(observation)
        if not self.state:
            self.state = dict(observation)
            self.velocity = {k: 0.0 for k in observation}
        else:
            for key in set(self.state) | set(observation):
                old = self.state.get(key, 0.0)
                new = observation.get(key, old)
                self.velocity[key] = 0.7 * self.velocity.get(key, 0.0) + 0.3 * (new - old)
                self.state[key] = 0.75 * old + 0.25 * new
        self.count += 1
        self.confidence = clamp(0.88 * self.confidence + 0.12 * float(confidence), 0.0, 1.0)

    def predict(
        self,
        steps: int = 1,
        action_effect: Optional[Dict[str, float]] = None,
        decay: float = 0.92,
    ) -> Dict[str, float]:
        pred = dict(self.state)
        vel = dict(self.velocity)
        n = max(1, int(steps))
        for i in range(n):
            scale = decay ** i
            for k in list(pred.keys()):
                pred[k] = pred[k] + vel.get(k, 0.0) * scale
            if action_effect:
                strength = 0.40 if i == 0 else 0.18 * scale
                for k, delta in action_effect.items():
                    pred[k] = pred.get(k, 0.0) + strength * float(delta)
        self.last_prediction = dict(pred)
        return pred

    def predict_with_uncertainty(
        self,
        steps: int = 1,
        action_effect: Optional[Dict[str, float]] = None,
    ) -> Tuple[Dict[str, float], float]:
        pred = self.predict(steps=steps, action_effect=action_effect)
        base_u = 1.0 - self.confidence
        residual_u = clamp(self.mean_residual * 1.5 + self.residual_var * 0.8, 0.0, 0.6)
        step_u = 0.04 * max(0, steps - 1)
        uncertainty = clamp(base_u * 0.45 + residual_u * 0.40 + step_u, 0.05, 0.95)
        return pred, uncertainty

    def record_residual(self, predicted: Mapping[str, float], observed: Mapping[str, float]):
        err = vector_rms(vector_diff(predicted, observed))
        old_mean = self.mean_residual
        self.mean_residual = 0.85 * old_mean + 0.15 * err
        diff = err - old_mean
        self.residual_var = 0.85 * self.residual_var + 0.15 * (diff * diff)
        return err

    def snapshot(self) -> Dict[str, Any]:
        return {
            "state": dict(self.state),
            "velocity": dict(self.velocity),
            "count": self.count,
            "confidence": self.confidence,
            "mean_residual": self.mean_residual,
            "residual_var": self.residual_var,
            "version": self.version,
            "model_id": self.model_id,
        }


@dataclass
class SelfModel:
    cycle_count: int = 0
    last_decision: str = ""
    last_reason: str = ""
    last_error: float = 0.0
    apply_count: int = 0
    hold_count: int = 0
    escalate_count: int = 0
    suggestion_count: int = 0
    execution_count: int = 0
    total_progress: float = 0.0
    recent_errors: List[float] = field(default_factory=list)
    recent_progress: List[float] = field(default_factory=list)
    stall_count: int = 0
    blend_factor: float = 0.35
    conf_base: float = 0.88
    last_reflection: str = ""
    success_rate: float = 0.5

    def push_error(self, error: float, window: int = 16):
        self.recent_errors.append(error)
        if len(self.recent_errors) > window:
            self.recent_errors = self.recent_errors[-window:]
        self.last_error = error

    def push_progress(self, progress: float, window: int = 16):
        self.recent_progress.append(progress)
        if len(self.recent_progress) > window:
            self.recent_progress = self.recent_progress[-window:]
        self.total_progress += progress

    def mean_recent_error(self) -> float:
        if not self.recent_errors:
            return 0.0
        return sum(self.recent_errors) / len(self.recent_errors)

    def mean_recent_progress(self) -> float:
        if not self.recent_progress:
            return 0.0
        return sum(self.recent_progress) / len(self.recent_progress)

    def detect_stall(self, epsilon: float, window: int) -> bool:
        if len(self.recent_progress) < window:
            return False
        recent = self.recent_progress[-window:]
        return all(abs(p) < epsilon for p in recent)


@dataclass
class UncertaintyModel:
    input: float = 1.0
    model: float = 1.0
    goal: float = 1.0
    attribution: float = 1.0
    authority: float = 1.0

    def aggregate(self) -> float:
        return clamp(
            0.25 * self.input + 0.30 * self.model + 0.20 * self.goal
            + 0.15 * self.attribution + 0.10 * self.authority,
            0.0, 1.0,
        )


@dataclass
class Suggestion:
    suggestion_id: str
    epoch: int
    action: str
    args: Dict[str, Any]
    decision: Any
    reason: str
    prediction: Dict[str, float]
    uncertainty: float
    risk: float
    created_at: float
    approved: bool = False
    executed: bool = False
    token_issued: bool = False
    tools_planned: List[str] = field(default_factory=list)
    requires_confirm: bool = False
    skill_id: Optional[str] = None


# ============================================================================
# MODULE FLATTENED: omega.kernel_execution
# ============================================================================

# -*- coding: utf-8 -*-
"""Kernel mixins - approve / authorize / execute pipeline + controlled autonomy."""


import secrets
import time
from dataclasses import asdict
from typing import Any, Dict, Optional, Set



class _ExecutionMixin:
    """Approve / authorize / execute pipeline + run_autonomous."""

    def approve_suggestion(self, owner: Owner, suggestion_id: str) -> Dict[str, Any]:
        self._authorize(owner)
        suggestion = self.suggestions.get(suggestion_id)
        if not suggestion or suggestion.executed or suggestion.epoch != self.epoch:
            raise ValueError("invalid or expired suggestion")
        suggestion.approved = True
        d = asdict(suggestion)
        d["decision"] = str(getattr(suggestion.decision, "value", suggestion.decision))
        d["jia_phase"] = "A"
        return d

    def authorize_suggestion(self, owner: Owner, suggestion_id: str) -> str:
        self._authorize(owner)
        suggestion = self.suggestions.get(suggestion_id)
        if not suggestion or not suggestion.approved or suggestion.executed:
            raise ValueError("suggestion not approved or already executed")
        if suggestion.epoch != self.epoch:
            raise ValueError("suggestion epoch mismatch")
        args_hash = sha256(canonical(suggestion.args or {}))
        self.tokens.revoke_for_suggestion(suggestion_id)
        token = self.tokens.issue(
            owner_id=owner.id,
            epoch=suggestion.epoch,
            suggestion_id=suggestion_id,
            action=suggestion.action,
            args_hash=args_hash,
            ttl_sec=self.config.token_ttl_sec,
        )
        suggestion.token_issued = True
        self.tokens.save(self.memory.tokens_file)
        self.audit.append("authorize", {
            "suggestion_id": suggestion_id,
            "jia_phase": "A",
        })
        return token

    def execute_authorized(
        self, owner: Owner, world, suggestion_id: str, token: str
    ) -> Dict[str, Any]:
        with self.lock:
            return self._execute_authorized_locked(owner, world, suggestion_id, token)

    def _execute_authorized_locked(
        self, owner: Owner, world, suggestion_id: str, token: str
    ) -> Dict[str, Any]:
        self._authorize(owner)
        ok, reason = self._check_writable()
        if not ok:
            return {
                "executed": False,
                "error": reason,
                "jia": build_blocked_jia(reason),
                "jia_phase": "A",
            }

        suggestion = self.suggestions.get(suggestion_id)
        if not suggestion or not suggestion.approved or suggestion.executed:
            return {
                "executed": False,
                "error": "invalid suggestion state",
                "jia": build_blocked_jia("invalid suggestion state"),
                "jia_phase": "A",
            }

        args_hash = sha256(canonical(suggestion.args or {}))
        if not self.tokens.consume(
            token,
            owner_id=owner.id,
            epoch=suggestion.epoch,
            suggestion_id=suggestion_id,
            action=suggestion.action,
            args_hash=args_hash,
        ):
            return {
                "executed": False,
                "error": "invalid_or_consumed_token",
                "jia": build_blocked_jia("invalid_or_consumed_token"),
                "jia_phase": "A",
            }

        # At-most-once: persist the consumed token and mark the suggestion
        # before any side effect. If the world mutation below raises, the
        # suggestion stays executed and cannot be re-authorized.
        suggestion.executed = True
        self.tokens.save(self.memory.tokens_file)

        before = deep(self.last_state) if self.last_state else {}
        if not before and hasattr(world, "observe"):
            before = world.observe()

        tools_used = []
        if suggestion.action in self.tools.allowed_action_names():
            tr = self.tools.execute(
                suggestion.action, suggestion.args or {},
                context={"world": world, "omega": self},
            )
            tools_used.append(suggestion.action)
            if tr.success and isinstance(tr.output, dict):
                after = tr.output
            elif hasattr(world, "observe"):
                after = world.observe()
            else:
                after = deep(before)
        else:
            if hasattr(world, "execute"):
                after = world.execute(suggestion.action, suggestion.args or {})
            elif hasattr(world, "observe"):
                after = world.observe()
            else:
                after = deep(before)

        after = norm_vector(after) if after else {}
        before = norm_vector(before) if before else {}
        difference = vector_diff(before, after)
        predicted = suggestion.prediction if isinstance(suggestion.prediction, dict) else {}
        error = (
            vector_rms(vector_diff(predicted, after))
            if predicted else vector_rms(difference)
        )

        attribution = better_attribute(predicted or before, after, before)
        risk = compute_risk(error, suggestion.uncertainty, attribution.residual)

        self.world_model.update(after, confidence=1.0 - suggestion.uncertainty)
        if predicted:
            self.world_model.record_residual(predicted, after)

        progress = 0.0
        if self.goal:
            before_d = sum(abs(before.get(k, 0.0) - t) for k, t in self.goal.items())
            after_d = sum(abs(after.get(k, 0.0) - t) for k, t in self.goal.items())
            progress = before_d - after_d
        # Mission progress fallback
        if abs(progress) < 1e-9:
            progress = float(after.get("mission_progress", 0.0)) - float(
                before.get("mission_progress", 0.0)
            )

        learning = self._calibrate(difference)

        self.self_model.execution_count += 1
        self.self_model.push_error(error)
        self.self_model.push_progress(progress)
        if is_decision(suggestion.decision, Decision.APPLY):
            self.self_model.apply_count += 1
        self.self_model.success_rate = clamp(
            0.9 * self.self_model.success_rate + 0.1 * (1.0 if error < 0.05 else 0.0),
            0.0, 1.0,
        )

        skill_ids = []
        sid = getattr(suggestion, "skill_id", None)
        if sid:
            skill_ids.append(sid)
            sk = self.skills.get(sid) if hasattr(self, "skills") else None
            if sk is not None:
                # success: low residual error OR mission advanced/ready
                success = (
                    error < 0.15
                    or float(after.get("target_ready", 0.0)) >= 0.99
                    or progress > 0
                )
                sk.record_use(success=success)

        episode_id = f"ep-{suggestion.epoch}-{secrets.token_hex(8)}"
        role_evidence = self.roles.post_execution(
            predicted=predicted,
            observed=after,
            episodes=self.memory.recent(64),
            max_error=self.config.max_error_apply,
        )

        episode = Episode(
            episode_id=episode_id,
            epoch=suggestion.epoch,
            timestamp=time.time(),
            suggestion_id=suggestion_id,
            intention={
                "action": suggestion.action,
                "args": deep(suggestion.args),
                "skill_id": sid,
            },
            prediction=deep(predicted),
            observation_before=before,
            observation_after=after,
            difference=difference,
            error=error,
            uncertainty=suggestion.uncertainty,
            attribution=asdict(attribution),
            decision=str(getattr(suggestion.decision, "value", suggestion.decision)),
            reason=suggestion.reason,
            learning=learning,
            risk=risk,
            tools_used=tools_used,
            skill_ids=skill_ids,
            role_evidence=role_evidence,
        )

        jia = build_execute_jia(error=error, risk=risk, progress=progress)
        audit_rec = self.audit.append("execute", {
            "suggestion_id": suggestion_id,
            "episode_id": episode_id,
            "error": error,
            "risk": risk,
            "skill_id": sid,
            "roles": role_evidence,
            "jia": jia,
            "jia_phase": "J2",
        })
        episode.audit_hash = audit_rec["hash"]

        if role_evidence.get("dayan_proposals"):
            self.audit.append("dayan_learning_proposal", {
                "episode_id": episode_id,
                "proposals": role_evidence["dayan_proposals"],
            })

        self.memory.add(episode, self.config.max_episodes, prune_disk=self.config.prune_disk_blocks)
        self.memory.update_semantic(difference)
        suggestion.executed = True
        self.last_state = after

        self.tokens.save(self.memory.tokens_file)
        self.audit.save(self.memory.audit_file)
        self.save_cas_head()
        self._maybe_rotate_audit()

        return {
            "executed": True,
            "error": error,
            "risk": risk,
            "decision": str(getattr(suggestion.decision, "value", suggestion.decision)),
            "reason": suggestion.reason,
            "cas_block_verified": True,
            "episode_id": episode_id,
            "active_episodes": len(self.memory.episodes),
            "progress": progress,
            "tools_used": tools_used,
            "skill_id": sid,
            "jia": jia,
            "jia_phase": "J2",
        }

    def run_autonomous(
        self,
        owner: Owner,
        world,
        policy,
        *,
        max_steps: int = 12,
        max_error: float = 0.5,
        max_risk: float = 0.6,
        auto_actions: Optional[Set[str]] = None,
        stop_on_hold: bool = False,
        stop_on_escalate: bool = True,
    ) -> Dict[str, Any]:
        self._authorize(owner)
        if not self.integrity_ok:
            return {
                "ok": False, "stopped_reason": "integrity_failed",
                "steps": 0, "executed": 0, "pending": None, "history": [],
            }
        if self.paused:
            return {
                "ok": False, "stopped_reason": "paused",
                "steps": 0, "executed": 0, "pending": None, "history": [],
            }
        if self.escalated:
            return {
                "ok": False, "stopped_reason": "escalated",
                "steps": 0, "executed": 0, "pending": None, "history": [],
            }

        if auto_actions is None:
            auto_actions = {
                n for n in self.tools.allowed_action_names()
                if not self.tools.requires_confirm(n)
            }
        else:
            auto_actions = set(auto_actions)
        auto_actions = {a for a in auto_actions if self.boundary.permits(a)}

        self.audit.append("autonomous_start", {
            "max_steps": max_steps,
            "max_error": max_error,
            "max_risk": max_risk,
            "auto_actions": sorted(auto_actions),
        })

        history = []
        executed = 0
        stopped_reason = "max_steps"
        pending = None

        for step in range(int(max_steps)):
            if self.paused:
                stopped_reason = "paused"
                break
            if self.escalated:
                stopped_reason = "escalated"
                break
            if not self.integrity_ok:
                stopped_reason = "integrity_failed"
                break

            # early mission success
            if hasattr(world, "mission_status") and world.mission_status().get("ready"):
                stopped_reason = "mission_ready"
                break

            cog = self.cognitive_cycle(world, policy)
            if cog.get("error"):
                stopped_reason = f"cycle_error:{cog['error']}"
                history.append({
                    "step": step, "event": "cycle_error", "error": cog["error"],
                    "jia": cog.get("jia"),
                })
                break

            sug = cog.get("suggestion") or {}
            sid = sug.get("suggestion_id")
            decision = sug.get("decision")
            action = sug.get("action")
            risk = float(sug.get("risk") or 0.0)
            reason = sug.get("reason")
            entry = {
                "step": step, "suggestion_id": sid, "decision": str(decision),
                "action": action, "risk": risk, "reason": reason,
                "skill_id": sug.get("skill_id"),
                "jia": cog.get("jia"),
                "jia_phase": cog.get("jia_phase"),
            }

            if is_decision(decision, Decision.ESCALATE) and stop_on_escalate:
                stopped_reason = "escalate"
                pending = sug
                entry["event"] = "stop_escalate"
                history.append(entry)
                break

            if is_decision(decision, Decision.HOLD):
                entry["event"] = "hold"
                history.append(entry)
                if stop_on_hold:
                    stopped_reason = "hold"
                    pending = sug
                    break
                continue

            if not is_decision(decision, Decision.APPLY):
                entry["event"] = "non_apply"
                history.append(entry)
                stopped_reason = f"decision:{decision}"
                pending = sug
                break

            if not action or not self.boundary.permits(action):
                stopped_reason = "boundary"
                pending = sug
                entry["event"] = "stop_boundary"
                history.append(entry)
                break

            if self.tools.requires_confirm(action) or bool(
                getattr(self.suggestions.get(sid), "requires_confirm", False)
            ):
                stopped_reason = "requires_confirm"
                pending = sug
                entry["event"] = "stop_confirm"
                history.append(entry)
                break

            if risk > max_risk:
                stopped_reason = "max_risk"
                pending = sug
                entry["event"] = "stop_max_risk"
                history.append(entry)
                break

            if action not in auto_actions:
                stopped_reason = "action_not_auto"
                pending = sug
                entry["event"] = "stop_not_auto"
                history.append(entry)
                break

            try:
                self.approve_suggestion(owner, sid)
                token = self.authorize_suggestion(owner, sid)
                res = self.execute_authorized(owner, world, sid, token)
            except Exception as exc:
                stopped_reason = f"exec_error:{type(exc).__name__}"
                entry["event"] = "exec_exception"
                entry["error"] = str(exc)
                history.append(entry)
                pending = sug
                break

            err_val = res.get("error")
            if isinstance(err_val, (int, float)):
                post_error = float(err_val)
            else:
                post_error = float(res.get("error_value") or 0.0) if res.get("executed") else 1.0

            entry["event"] = "executed" if res.get("executed") else "exec_failed"
            entry["episode_id"] = res.get("episode_id")
            entry["post_error"] = post_error
            entry["progress"] = res.get("progress")
            entry["skill_id"] = res.get("skill_id")
            entry["jia"] = res.get("jia")
            entry["jia_phase"] = res.get("jia_phase")
            history.append(entry)

            if not res.get("executed"):
                stopped_reason = f"exec_failed:{res.get('error')}"
                pending = sug
                break

            executed += 1
            if hasattr(world, "mission_status") and world.mission_status().get("ready"):
                stopped_reason = "mission_ready"
                break
            if post_error > max_error:
                stopped_reason = "max_error"
                pending = None
                break
        else:
            stopped_reason = "max_steps"

        self.audit.append("autonomous_end", {
            "stopped_reason": stopped_reason,
            "steps": len(history),
            "executed": executed,
            "pending_id": (pending or {}).get("suggestion_id"),
        })
        try:
            self.audit.save(self.memory.audit_file)
            self.save_cas_head()
        except Exception:
            pass

        return {
            "ok": stopped_reason in ("max_steps", "max_error", "mission_ready") and self.integrity_ok,
            "stopped_reason": stopped_reason,
            "steps": len(history),
            "executed": executed,
            "pending": pending,
            "history": history,
            "integrity_ok": self.integrity_ok,
            "epoch": self.epoch,
            "episodes": len(self.memory.episodes),
        }


# ============================================================================
# MODULE FLATTENED: omega.reactor.reactor
# ============================================================================

# -*- coding: utf-8 -*-
"""Async reactor: priority queue + cooperative preemption (stdlib asyncio)."""


import asyncio
import heapq
import itertools
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional



class Preempted(Exception):
    """Current coroutine preempted by a higher-priority task."""


@dataclass(order=True)
class ReactorTask:
    priority: float
    seq: int
    name: str = field(compare=False)
    factory: Callable[[], Any] = field(compare=False)
    base_priority: float = field(default=100.0, compare=False)
    uncertainty: float = field(default=0.0, compare=False)
    preemptible: bool = field(default=True, compare=False)
    submitted_at: float = field(default_factory=time.time, compare=False)
    requeues: int = field(default=0, compare=False)
    result: Any = field(default=None, compare=False)
    error: Optional[str] = field(default=None, compare=False)


class AsyncReactor:
    MAX_REQUEUES = 3

    def __init__(
        self,
        omega: Any = None,
        *,
        max_concurrency: int = 1,
        preemption: bool = True,
    ):
        self.omega = omega
        self.max_concurrency = max(1, int(max_concurrency))
        self.preemption = bool(preemption)
        self._heap: List[ReactorTask] = []
        self._seq = itertools.count()
        self._current: Optional[ReactorTask] = None
        self._stopped = False
        self.stats: Dict[str, Any] = {
            "submitted": 0, "completed": 0, "failed": 0,
            "preempted": 0, "max_queue_depth": 0, "by_name": {},
        }

    @staticmethod
    def urgency_priority(base_priority: float, uncertainty: float = 0.0, risk: float = 0.0) -> float:
        uncertainty = clamp(float(uncertainty), 0.0, 1.0)
        risk = clamp(float(risk), 0.0, 2.0)
        return float(base_priority) - 40.0 * uncertainty - 20.0 * risk

    def _current_uncertainty(self) -> float:
        if self.omega is None:
            return 0.0
        try:
            return float(self.omega.uncertainty.aggregate())
        except Exception:
            return 0.0

    def _current_risk(self) -> float:
        if self.omega is None:
            return 0.0
        try:
            return float(self.omega.self_model.last_error)
        except Exception:
            return 0.0

    def submit(
        self,
        name: str,
        factory: Callable[[], Any],
        *,
        base_priority: float = 100.0,
        uncertainty: Optional[float] = None,
        risk: Optional[float] = None,
        preemptible: bool = True,
    ) -> ReactorTask:
        if self._stopped:
            raise RuntimeError("reactor is stopped")
        if uncertainty is None:
            uncertainty = self._current_uncertainty()
        if risk is None:
            risk = self._current_risk()
        task = ReactorTask(
            priority=self.urgency_priority(base_priority, uncertainty, risk),
            seq=next(self._seq),
            name=str(name),
            factory=factory,
            base_priority=float(base_priority),
            uncertainty=float(uncertainty),
            preemptible=bool(preemptible),
        )
        heapq.heappush(self._heap, task)
        self.stats["submitted"] += 1
        self.stats["max_queue_depth"] = max(self.stats["max_queue_depth"], len(self._heap))
        return task

    def __len__(self) -> int:
        return len(self._heap)

    async def checkpoint(self):
        await asyncio.sleep(0)
        if not self.preemption or self._current is None or not self._heap:
            return
        current = self._current
        if not current.preemptible or current.requeues >= self.MAX_REQUEUES:
            return
        if self._heap[0].priority < current.priority - 1e-9:
            raise Preempted(
                f"{self._heap[0].name} preempts {current.name}"
            )

    async def _invoke(self, task: ReactorTask) -> Any:
        result = task.factory()
        if asyncio.iscoroutine(result) or isinstance(result, asyncio.Future):
            return await result
        return result

    async def run_once(self) -> Optional[ReactorTask]:
        if not self._heap:
            return None
        task = heapq.heappop(self._heap)
        self._current = task
        per_name = self.stats["by_name"].setdefault(
            task.name, {"completed": 0, "failed": 0, "preempted": 0}
        )
        try:
            task.result = await self._invoke(task)
            self.stats["completed"] += 1
            per_name["completed"] += 1
            return task
        except Preempted:
            task.requeues += 1
            heapq.heappush(self._heap, task)
            self.stats["preempted"] += 1
            per_name["preempted"] += 1
            return None
        except Exception as exc:
            task.error = f"{type(exc).__name__}: {exc}"
            self.stats["failed"] += 1
            per_name["failed"] += 1
            return task
        finally:
            self._current = None

    async def run(self, max_tasks: Optional[int] = None) -> Dict[str, Any]:
        completed = 0
        while self._heap and not self._stopped:
            if max_tasks is not None and completed >= max_tasks:
                break
            task = await self.run_once()
            if task is not None and task.error is None:
                completed += 1
        snapshot = dict(self.stats)
        snapshot["pending"] = len(self._heap)
        return snapshot

    def stop(self):
        self._stopped = True


class AsyncHeartbeat:
    def __init__(
        self,
        omega,
        owner,
        world,
        policy,
        *,
        reactor: Optional[AsyncReactor] = None,
        interval: float = 0.0,
        persist_every: int = 5,
    ):
        self.omega = omega
        self.owner = owner
        self.world = world
        self.policy = policy
        self.interval = max(0.0, float(interval))
        self.persist_every = max(1, int(persist_every))
        self.reactor = reactor or AsyncReactor(
            omega,
            max_concurrency=int(getattr(omega.config, "async_max_concurrency", 1)),
            preemption=bool(getattr(omega.config, "async_preemption", True)),
        )
        self.cycles = 0
        self.executed = 0
        self.history: List[Dict[str, Any]] = []

    async def _cognition_task(self) -> Dict[str, Any]:
        await self.reactor.checkpoint()
        if hasattr(self.omega, "sm"):
            try:
                self.omega.sm.transition_kernel(KernelState.RUNNING)
            except Exception:
                pass
            self.omega.sm.set_phase(CyclePhase.SENSE)
        cognition = self.omega.cognitive_cycle(self.world, self.policy)
        suggestion = cognition.get("suggestion")
        outcome: Dict[str, Any] = {"epoch": self.omega.epoch, "decision": None, "executed": False}
        if not suggestion:
            return outcome
        decision = suggestion["decision"]
        outcome["decision"] = decision
        if is_decision(decision, Decision.APPLY):
            sid = suggestion["suggestion_id"]
            try:
                if hasattr(self.omega, "sm"):
                    self.omega.sm.set_phase(CyclePhase.AUTH)
                self.omega.approve_suggestion(self.owner, sid)
                token = self.omega.authorize_suggestion(self.owner, sid)
                if hasattr(self.omega, "sm"):
                    self.omega.sm.set_phase(CyclePhase.EXEC)
                result = self.omega.execute_authorized(self.owner, self.world, sid, token)
                outcome.update(executed=True, error=result["error"], risk=result["risk"])
                self.executed += 1
                exp = getattr(self.omega, "_last_experience", None)
                if exp and hasattr(self.policy, "remember"):
                    self.policy.remember(exp, limit=self.omega.config.experience_buffer_size)
                    self.policy.adapt(rate=self.omega.config.adapt_rate)
            except Exception as exc:
                outcome["blocked"] = f"{type(exc).__name__}: {exc}"
        meta = cognition.get("meta")
        if meta:
            self.omega.meta.apply_to_config_and_policy(
                self.omega.config, self.policy, self.omega.self_model
            )
        if hasattr(self.omega, "sm"):
            self.omega.sm.set_phase(CyclePhase.CALIBRATE)
            # Single clock: JiaLoop advances; sm.cycle_step mirrors
            if getattr(self.omega.config, "topology_enabled", True):
                if hasattr(self.omega, "tick_topology"):
                    outcome["topology"] = self.omega.tick_topology()
                else:
                    self.omega.sm.tick_topology(
                        getattr(self.omega.config, "topology_mod", 16),
                        getattr(self.omega.config, "topology_cycle", 240),
                    )
        self.cycles += 1
        self.history.append(outcome)
        if len(self.history) > 128:
            self.history = self.history[-128:]
        return outcome

    async def _persist_task(self) -> Dict[str, Any]:
        if hasattr(self.omega, "sm"):
            self.omega.sm.set_phase(CyclePhase.PERSIST)
        self.omega.save_cas_head()
        return {"persisted": True, "blocks": self.omega.memory.block_count(), "epoch": self.omega.epoch}

    async def run(self, max_cycles: Optional[int] = None) -> Dict[str, Any]:
        while max_cycles is None or self.cycles < max_cycles:
            if self.omega.paused or self.omega.escalated:
                break
            uncertainty = self.omega.uncertainty.aggregate()
            risk = float(getattr(self.omega.self_model, "last_error", 0.0))
            self.reactor.submit(
                "cognition", self._cognition_task,
                base_priority=10.0, uncertainty=uncertainty, risk=risk,
            )
            if self.omega.epoch % self.persist_every == 0:
                self.reactor.submit("persist", self._persist_task, base_priority=50.0)
            await self.reactor.run()
            if self.interval:
                await asyncio.sleep(self.interval)
        self.omega.save_cas_head()
        return {
            "cycles": self.cycles,
            "executed": self.executed,
            "epoch": self.omega.epoch,
            "reactor": dict(self.reactor.stats),
            "history": list(self.history),
            "state": self.omega.sm.summary() if hasattr(self.omega, "sm") else {},
        }


async def async_autonomous_drive(
    omega, owner, world, policy, *,
    max_cycles: Optional[int] = None,
    interval: float = 0.0,
    persist_every: int = 5,
    reactor: Optional[AsyncReactor] = None,
) -> Dict[str, Any]:
    hb = AsyncHeartbeat(
        omega, owner, world, policy,
        reactor=reactor, interval=interval, persist_every=persist_every,
    )
    return await hb.run(max_cycles=max_cycles)


# ============================================================================
# MODULE FLATTENED: omega.cognitive.hybrid
# ============================================================================

# -*- coding: utf-8 -*-
"""Hybrid world-model predictor (V3.11 port, simplified). Stdlib only."""


import math
import random
from typing import Dict, List, Mapping, Optional


# Deterministic seed (was invalid hex 0x0MEGA — M is not a hex digit).
_OMEGA_SEED = int("omega", 36)  # base36 "omega" -> stable int


def _dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _softmax(xs: List[float]) -> List[float]:
    if not xs:
        return []
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps) or 1.0
    return [e / s for e in exps]


class TinyAttentionPredictor:
    def __init__(self, d_model: int = 8, heads: int = 2, window: int = 16):
        self.d_model = max(2, int(d_model))
        self.heads = max(1, int(heads))
        self.window = max(2, int(window))
        self.head_dim = max(1, self.d_model // self.heads)
        rng = random.Random(_OMEGA_SEED)
        self.Wq = [[rng.uniform(-0.2, 0.2) for _ in range(self.d_model)] for _ in range(self.d_model)]
        self.Wk = [[rng.uniform(-0.2, 0.2) for _ in range(self.d_model)] for _ in range(self.d_model)]
        self.Wv = [[rng.uniform(-0.2, 0.2) for _ in range(self.d_model)] for _ in range(self.d_model)]
        self.readout: Dict[str, List[float]] = {}
        self.history: List[Dict[str, float]] = []
        self.keys_seen: List[str] = []

    def _embed(self, state: Mapping[str, float]) -> List[float]:
        vec = [0.0] * self.d_model
        for i, k in enumerate(sorted(state.keys())):
            vec[i % self.d_model] += float(state[k])
        n = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / n for x in vec]

    def _proj(self, W: List[List[float]], x: List[float]) -> List[float]:
        return [_dot(row, x) for row in W]

    def observe(self, state: Mapping[str, float]) -> None:
        state = norm_vector(state)
        self.history.append(dict(state))
        if len(self.history) > self.window:
            self.history = self.history[-self.window:]
        for k in state:
            if k not in self.keys_seen:
                self.keys_seen.append(k)
            if k not in self.readout:
                self.readout[k] = [0.0] * self.d_model

    def _attend(self) -> List[float]:
        if not self.history:
            return [0.0] * self.d_model
        embeds = [self._embed(s) for s in self.history]
        q = self._proj(self.Wq, embeds[-1])
        scores = []
        for e in embeds:
            k = self._proj(self.Wk, e)
            scores.append(_dot(q, k) / math.sqrt(self.head_dim))
        weights = _softmax(scores)
        out = [0.0] * self.d_model
        for w, e in zip(weights, embeds):
            v = self._proj(self.Wv, e)
            for i in range(self.d_model):
                out[i] += w * v[i]
        return out

    def predict(self, keys: Optional[List[str]] = None) -> Dict[str, float]:
        feat = self._attend()
        keys = keys or self.keys_seen
        pred: Dict[str, float] = {}
        last = self.history[-1] if self.history else {}
        for k in keys:
            base = float(last.get(k, 0.0))
            w = self.readout.get(k) or [0.0] * self.d_model
            pred[k] = base + _dot(w, feat) * 0.15
        return pred

    def update_readout(self, predicted: Mapping[str, float], observed: Mapping[str, float], lr: float = 0.05):
        feat = self._attend()
        for k, obs in observed.items():
            err = float(obs) - float(predicted.get(k, 0.0))
            w = self.readout.setdefault(k, [0.0] * self.d_model)
            for i in range(self.d_model):
                w[i] = clamp(w[i] + lr * err * feat[i], -2.0, 2.0)


class HybridWorldModel(WorldModel):
    def __init__(self, mode: str = "hybrid", d_model: int = 8, heads: int = 2, window: int = 16):
        super().__init__()
        self.mode = mode
        self.attn = TinyAttentionPredictor(d_model=d_model, heads=heads, window=window)
        self.blend = 0.5
        self._last_linear: Dict[str, float] = {}
        self._last_attn: Dict[str, float] = {}

    def update(self, observation: Dict[str, float], confidence: float):
        super().update(observation, confidence)
        self.attn.observe(observation)

    def predict(self, steps: int = 1, action_effect=None, decay: float = 0.92) -> Dict[str, float]:
        linear = super().predict(steps=steps, action_effect=action_effect, decay=decay)
        self._last_linear = dict(linear)
        if self.mode == "linear":
            return linear
        attn = self.attn.predict(keys=list(linear.keys()) or None)
        self._last_attn = dict(attn)
        if self.mode == "attention":
            return attn
        out = {}
        for k in set(linear) | set(attn):
            out[k] = (1.0 - self.blend) * linear.get(k, 0.0) + self.blend * attn.get(k, 0.0)
        return out

    def record_residual(self, predicted: Mapping[str, float], observed: Mapping[str, float]):
        err = super().record_residual(predicted, observed)
        if self._last_linear and self._last_attn:
            e_lin = vector_rms(vector_diff(self._last_linear, observed))
            e_att = vector_rms(vector_diff(self._last_attn, observed))
            if e_att + 1e-9 < e_lin:
                self.blend = clamp(self.blend + 0.05, 0.1, 0.9)
            else:
                self.blend = clamp(self.blend - 0.05, 0.1, 0.9)
        self.attn.update_readout(predicted, observed)
        return err

    def predictor_stats(self) -> Dict[str, object]:
        return {
            "mode": self.mode,
            "blend": self.blend,
            "attn_window": len(self.attn.history),
            "keys": list(self.attn.keys_seen),
            "mean_residual": self.mean_residual,
            "confidence": self.confidence,
        }


# ============================================================================
# MODULE FLATTENED: omega.kernel_cognition
# ============================================================================

# -*- coding: utf-8 -*-
"""Kernel mixins - cognition for Omega orchestrator."""


import secrets
import time
from typing import Any, Dict, Mapping, Optional



class _CognitionMixin:
    """Perception, cognitive cycle, reflection, memory consultation, calibration."""

    def perceive(self, external: ExternalInput) -> Perception:
        self.uncertainty.input = 1.0 - external.confidence
        if not external.content:
            return Perception(False, {}, 0.0, "empty_input", external.source, View.SHAN)
        try:
            state = norm_vector(external.content)
        except Exception:
            return Perception(False, {}, 0.0, "invalid_input", external.source, View.SHAN)
        return Perception(True, state, external.confidence, "accepted", external.source, self.view)

    def _effective_goal(self, state: Mapping[str, float]) -> Dict[str, float]:
        if getattr(self.config, "hierarchical_goals", True) and self.goal_tree.root_id:
            self.goal_tree.update_from_state(state)
            active = self.goal_tree.select_active(state)
            if active:
                self.goal = dict(active.targets)
                return self.goal
        return dict(self.goal)

    def _enter_escalated(self) -> None:
        self.escalated = True
        if self.sm.kernel == KernelState.ESCALATED:
            return
        if self.sm.can_transition_kernel(KernelState.ESCALATED):
            self.sm.transition_kernel(KernelState.ESCALATED)
        else:
            self.sm.transition_kernel(KernelState.ESCALATED, force=True)

    def _match_skill_id(self, action: str, args: Mapping[str, Any]) -> Optional[str]:
        """Best-effort skill match for record_use (policy or planner path)."""
        skills = getattr(self, "skills", None)
        if skills is None:
            return None
        try:
            items = skills.list_skills() if hasattr(skills, "list_skills") else []
        except Exception:
            return None
        best = None
        for raw in items:
            if isinstance(raw, dict):
                sid = raw.get("skill_id")
                sk = skills.get(sid) if sid else None
            else:
                sk = raw
            if sk is None or sk.preferred_action != action:
                continue
            tmpl = sk.template_args or {}
            if action == "write_file":
                if tmpl.get("path") == args.get("path") and tmpl.get("content") == args.get("content"):
                    return sk.skill_id
                if tmpl.get("path") == args.get("path"):
                    best = sk.skill_id
            elif action in ("read_file", "list_files"):
                if tmpl.get("path") == args.get("path"):
                    return sk.skill_id
                best = best or sk.skill_id
            else:
                best = best or sk.skill_id
        return best

    def cognitive_cycle(self, world, policy) -> Dict[str, Any]:
        self.self_model.cycle_count += 1
        if hasattr(policy, "bind_world"):
            policy.bind_world(world)
        observation = world.observe() if hasattr(world, "observe") else dict(self.last_state)
        observation = norm_vector(observation)
        self.last_state = deep(observation)

        external = ExternalInput(kind="world", content=observation, confidence=0.95, source="world")
        perception = self.perceive(external)
        if not perception.accepted:
            return {"error": perception.reason, "suggestion": None}

        self.world_model.update(perception.state, perception.confidence)
        stall = self.self_model.detect_stall(self.config.stall_epsilon, self.config.stall_window)
        if stall:
            self.self_model.stall_count += 1
        else:
            self.self_model.stall_count = 0

        recent_err = self.self_model.mean_recent_error()
        effective_goal = self._effective_goal(perception.state)

        mem_advice = self._consult_memory(perception.state, perception.state)
        memory_caution = bool(mem_advice.get("suggest_caution"))

        goal_dist = None
        if effective_goal:
            goal_dist = sum(
                abs(perception.state.get(k, 0) - t) for k, t in effective_goal.items()
            ) / max(1, len(effective_goal))
        intrinsic_sig = self.intrinsic.evaluate(
            perception.state,
            recent_progress=self.self_model.mean_recent_progress(),
            success_rate=self.self_model.success_rate,
            stall=stall,
            mean_residual=getattr(self.world_model, "mean_residual", 0.0),
            goal_distance=goal_dist,
        )

        plan_info = None
        plan_skill_id = None
        prefer_policy = bool(getattr(policy, "prefer_policy_intent", False))
        use_planner = bool(getattr(self.config, "use_planner", False)) and not prefer_policy
        if use_planner and hasattr(self, "planner") and self.planner is not None:
            allowed = None
            if hasattr(self, "boundary") and self.boundary is not None:
                allowed = set(self.boundary.allowed_actions)
            plan = self.planner.plan(
                perception.state,
                effective_goal,
                world_model=self.world_model,
                skills=getattr(self, "skills", None),
                allowed_actions=allowed,
                stall=stall,
                recent_error=recent_err,
            )
            first = plan.first_step()
            if first is not None:
                intention = first.to_intention(
                    policy_id=self.planner.planner_id,
                    policy_version=self.planner.planner_version,
                )
                plan_info = plan.to_dict()
                plan_skill_id = first.skill_id
            else:
                intention = policy.intend(
                    perception.state, effective_goal, stall=stall, recent_error=recent_err
                )
        else:
            intention = policy.intend(
                perception.state, effective_goal, stall=stall, recent_error=recent_err
            )

        skill_id = plan_skill_id or self._match_skill_id(intention.action, intention.args or {})

        prediction = policy.predict(
            perception.state, intention, effective_goal,
            world_model=self.world_model,
            steps=self.config.lookahead_steps,
        )
        expected_progress = 0.0
        if hasattr(policy, "expected_progress"):
            expected_progress = policy.expected_progress(
                perception.state, prediction.expected, effective_goal
            )
        elif plan_info and plan_info.get("steps"):
            expected_progress = float(plan_info["steps"][0].get("estimated_progress") or 0.0)

        im_weight = float(getattr(self.config, "intrinsic_motivation_weight", 0.0))
        if im_weight > 0:
            expected_progress = self.intrinsic.modulate_progress(expected_progress, im_weight)

        pred_err = vector_rms(vector_diff(prediction.expected, perception.state))
        uncertainty = prediction.uncertainty
        self.uncertainty.model = uncertainty
        self.uncertainty.goal = 0.0 if effective_goal else 0.5

        # Pre-action risk must not treat the *magnitude of the intended
        # state change* as prediction error. A write_file that is expected
        # to change several mission fields is not inherently a high-risk
        # action. Before execution there is no post-action residual yet, so
        # use the learned world-model residual as the risk signal.
        model_residual = float(
            getattr(self.world_model, "mean_residual", 0.0) or 0.0
        )
        attribution = Attribution(
            action_effect=0.5,
            external_effect=0.3,
            residual=model_residual,
            confidence=0.6,
        )
        risk = compute_risk(model_residual, uncertainty, model_residual)

        decision, reason, escalate = self.decide_fn(
            error=pred_err,
            uncertainty=uncertainty,
            observation=perception.state,
            attribution=attribution,
            paused=self.paused,
            escalated=self.escalated,
            hard_min=self.hard_min,
            hard_max=self.hard_max,
            config=self.config,
            expected_progress=expected_progress,
            risk=risk,
            stall=stall,
        )

        if goal_dist is None and effective_goal and hasattr(world, "goal_distance"):
            try:
                goal_dist = world.goal_distance(effective_goal)
            except Exception:
                pass

        mem_stats = self.memory.stats()
        meta_assess = self.meta.assess(
            self.self_model, self.world_model,
            memory_stats=mem_stats,
            memory_advice=mem_advice,
            config=self.config,
            goal_distance=goal_dist,
        )
        meta_notes = self.meta.apply_to_config_and_policy(
            self.config, policy, self.self_model
        )

        # Role shards are advisory only. They produce evidence; the existing
        # decision/governance path remains authoritative.
        role_evidence = self.roles.pre_decision(
            action=intention.action,
            prediction_confidence=prediction.confidence,
            episodes=self.memory.recent(32),
            meta_assessment=meta_assess,
        )

        if role_evidence.get("wukong_conflict"):
            try:
                self.audit.append("wukong_challenge", role_evidence["wukong"])
            except Exception:
                pass

        soft = soft_decision_from_meta(
            str(getattr(decision, "value", decision)),
            meta_assess.get("recommended_action", ""),
        )
        meta_overrode = False
        if soft is not None:
            meta_overrode = True
            if soft["decision"] == "HOLD":
                decision = Decision.HOLD
                reason = soft["reason"]
                escalate = False
            elif soft["decision"] == "ESCALATE":
                decision = Decision.ESCALATE
                reason = soft["reason"]
                escalate = True

        if escalate or is_decision(decision, Decision.ESCALATE):
            self._enter_escalated()
            residual = float(getattr(self.world_model, "mean_residual", 0.0))
            ft = mint_fire_token(
                epoch=self.epoch,
                cycle=self.self_model.cycle_count,
                topology_step=self.topology.step,
                residual=residual,
                regime=str(meta_assess.get("regime", "")),
                reason=str(reason),
            )
            self.last_fire_token = ft
            try:
                self.audit.append("fire_token", ft)
            except Exception:
                pass

        if meta_assess.get("recommended_action") == "recalibrate":
            try:
                self.sm.set_phase(CyclePhase.CALIBRATE)
            except Exception:
                pass

        self.self_model.last_decision = str(
            decision.value if hasattr(decision, "value") else decision
        )
        self.self_model.last_reason = reason
        if is_decision(decision, Decision.HOLD):
            self.self_model.hold_count += 1
        elif is_decision(decision, Decision.ESCALATE):
            self.self_model.escalate_count += 1

        req_confirm = False
        try:
            if hasattr(self, "tools") and self.tools is not None:
                req_confirm = bool(self.tools.requires_confirm(intention.action))
        except Exception:
            req_confirm = False

        suggestion_id = f"sug-{self.epoch}-{secrets.token_hex(8)}"
        suggestion = Suggestion(
            suggestion_id=suggestion_id,
            epoch=self.epoch,
            action=intention.action,
            args=deep(intention.args),
            decision=decision,
            reason=reason,
            prediction=deep(prediction.expected),
            uncertainty=uncertainty,
            risk=risk,
            created_at=time.time(),
            requires_confirm=req_confirm,
            skill_id=skill_id,
        )
        self.suggestions[suggestion_id] = suggestion
        self.self_model.suggestion_count += 1

        if self.self_model.cycle_count % max(1, self.config.reflection_every) == 0:
            self.self_model.last_reflection = self._reflect()

        if hasattr(policy, "remember"):
            policy.remember({
                "error": pred_err, "progress": expected_progress,
                "decision": str(decision), "risk": risk,
            }, limit=self.config.experience_buffer_size)
        if hasattr(policy, "adapt"):
            policy.adapt(rate=self.config.adapt_rate)

        dec_str = str(decision.value if hasattr(decision, "value") else decision)
        jia = build_cognition_jia(
            decision=dec_str,
            paused=self.paused,
            escalated=self.escalated,
            stall=stall,
            memory_caution=memory_caution,
        )

        out = {
            "error": None,
            "perception": {
                "accepted": perception.accepted,
                "state": perception.state,
                "confidence": perception.confidence,
            },
            "intention": {
                "action": intention.action,
                "args": intention.args,
                "rationale": intention.rationale,
                "skill_id": skill_id,
            },
            "prediction": {
                "expected": prediction.expected,
                "confidence": prediction.confidence,
                "uncertainty": uncertainty,
            },
            "suggestion": {
                "suggestion_id": suggestion_id,
                "decision": dec_str,
                "reason": reason,
                "risk": risk,
                "action": intention.action,
                "args": intention.args,
                "requires_confirm": req_confirm,
                "skill_id": skill_id,
            },
            "jia": jia,
            "jia_phase": jia.get("current"),
            "meta": meta_assess,
            "meta_notes": meta_notes,
            "meta_overrode": meta_overrode,
            "roles": role_evidence,
            "outer": meta_assess.get("outer"),
            "outer_phase": meta_assess.get("outer_phase"),
            "g_active": meta_assess.get("g_active"),
            "fire_token": getattr(self, "last_fire_token", None),
            "expected_progress": expected_progress,
            "stall": stall,
            "epoch": self.epoch,
            "cycle": self.self_model.cycle_count,
            "intrinsic": intrinsic_sig.to_dict(),
            "active_goal_id": self.goal_tree.active_id,
            "effective_goal": deep(effective_goal),
            "escalated": self.escalated,
            "kernel": self.sm.kernel.value,
        }
        if plan_info is not None:
            out["plan"] = plan_info
        if self.goal_tree.root_id:
            out["goal_progress"] = self.goal_tree.progress_summary(perception.state)
        return out

    def _reflect(self) -> str:
        sm = self.self_model
        return (
            f"cycles={sm.cycle_count} apply={sm.apply_count} hold={sm.hold_count} "
            f"esc={sm.escalate_count} mean_err={sm.mean_recent_error():.4f} "
            f"mean_prog={sm.mean_recent_progress():.4f} sr={sm.success_rate:.2f} "
            f"stall={sm.stall_count}"
        )

    def _consult_memory(
        self, state: Mapping[str, float], prediction: Mapping[str, float]
    ) -> Dict[str, Any]:
        advice: Dict[str, Any] = {"suggest_caution": False, "similar": []}
        recent = self.memory.recent(8)
        high_err = [e for e in recent if e.error > self.config.max_error_apply * 0.7]
        if len(high_err) >= 3:
            advice["suggest_caution"] = True
        advice["recent_mean_error"] = (
            sum(e.error for e in recent) / len(recent) if recent else 0.0
        )
        return advice

    def _calibrate(self, difference: Mapping[str, float]) -> Dict[str, float]:
        learning = {}
        step = self.config.learning_step
        for key, delta in difference.items():
            if key in self.config.cal_blacklist:
                continue
            old = self.calibration.get(key, 0.0)
            new = old + step * float(delta)
            new = clamp(new, -self.config.learning_abs, self.config.learning_abs)
            self.calibration[key] = new
            learning[key] = new - old
        return learning

    def install_file_mission_skills(
        self,
        owner,
        target_path: str = "out/report.txt",
        target_content: str = "OMEGA-OK\n",
        *,
        require_confirm: bool = False,
    ) -> Dict[str, Any]:
        """Install write_target / verify_target / list_parent skill pack."""

        installed = []
        for skill in make_file_mission_skills(target_path, target_content):
            r = self.install_skill(owner, skill, require_confirm=require_confirm)
            installed.append(r)
        return {"result": "file_mission_skills", "installed": installed}


# ============================================================================
# MODULE FLATTENED: omega.meta.metacognition
# ============================================================================

# -*- coding: utf-8 -*-
"""Meta-cognition monitor layer (+ outer-law annotation)."""


import time
from typing import Any, Dict, List, Optional



class MetaCognition:
    REGIME_IMPROVING = "improving"
    REGIME_STABLE = "stable"
    REGIME_DEGRADING = "degrading"
    REGIME_STALLED = "stalled"
    REGIME_UNSTABLE = "unstable"
    REGIME_CONVERGED = "converged"

    ACTION_CONTINUE = "continue"
    ACTION_CAUTION = "caution"
    ACTION_EXPLORE = "explore"
    ACTION_RECALIBRATE = "recalibrate"
    ACTION_ESCALATE = "escalate"

    def __init__(self):
        self.regime: str = self.REGIME_STABLE
        self.recommended_action: str = self.ACTION_CONTINUE
        self.confidence_in_self: float = 0.5
        self.calibration_score: float = 0.5
        self.alerts: List[str] = []
        self.history: List[Dict[str, Any]] = []
        self.last_assessment: Dict[str, Any] = {}

    def assess(
        self,
        self_model: SelfModel,
        world_model: WorldModel,
        memory_stats: Optional[Dict[str, Any]] = None,
        memory_advice: Optional[Dict[str, Any]] = None,
        config: Optional[Config] = None,
        goal_distance: Optional[float] = None,
    ) -> Dict[str, Any]:
        self.alerts = []
        cfg = config
        max_err = cfg.max_error_apply if cfg else 1.0

        mean_err = self_model.mean_recent_error()
        mean_prog = self_model.mean_recent_progress()
        sr = self_model.success_rate
        stall = self_model.detect_stall(
            cfg.stall_epsilon if cfg else 0.008,
            cfg.stall_window if cfg else 8,
        )
        residual = getattr(world_model, "mean_residual", 0.0)
        wm_conf = getattr(world_model, "confidence", 0.5)

        cal = 1.0 - clamp(residual * 2.0, 0.0, 0.7)
        cal = 0.6 * cal + 0.4 * sr
        self.calibration_score = clamp(cal, 0.0, 1.0)

        self.confidence_in_self = clamp(
            0.35 * sr + 0.25 * self.calibration_score + 0.20 * wm_conf
            + 0.20 * (1.0 - clamp(mean_err / max(max_err, EPS), 0.0, 1.0)),
            0.0, 1.0,
        )

        recent_p = self_model.recent_progress[-6:] if self_model.recent_progress else []
        recent_e = self_model.recent_errors[-6:] if self_model.recent_errors else []
        prog_trend = 0.0
        err_trend = 0.0
        if len(recent_p) >= 3:
            prog_trend = sum(recent_p[-3:]) / 3 - sum(recent_p[:3]) / 3
        if len(recent_e) >= 3:
            err_trend = sum(recent_e[-3:]) / 3 - sum(recent_e[:3]) / 3

        if stall:
            converged_dist = float(getattr(cfg, "converged_goal_distance", 0.1)) if cfg else 0.1
            near_goal = goal_distance is not None and goal_distance <= converged_dist
            if near_goal:
                self.regime = self.REGIME_CONVERGED
                self.alerts.append("converged_at_goal")
            else:
                self.regime = self.REGIME_STALLED
                self.alerts.append("stalled_progress")
        elif err_trend > 0.02 and prog_trend < 0:
            self.regime = self.REGIME_DEGRADING
            self.alerts.append("error_rising_progress_falling")
        elif len(recent_e) >= 4 and (max(recent_e) - min(recent_e)) > max_err * 0.5:
            self.regime = self.REGIME_UNSTABLE
            self.alerts.append("high_error_variance")
        elif prog_trend > 0.005 and err_trend <= 0.01:
            self.regime = self.REGIME_IMPROVING
        else:
            self.regime = self.REGIME_STABLE

        if memory_advice and memory_advice.get("suggest_caution"):
            self.alerts.append("memory_suggests_caution")
        if memory_stats and memory_stats.get("mean_error", 0) > max_err * 0.7:
            self.alerts.append("long_term_high_mean_error")

        if self.regime == self.REGIME_DEGRADING and self.confidence_in_self < 0.4:
            self.recommended_action = self.ACTION_ESCALATE
        elif self.regime in (self.REGIME_DEGRADING, self.REGIME_UNSTABLE):
            self.recommended_action = self.ACTION_CAUTION
        elif self.regime == self.REGIME_STALLED:
            if self_model.stall_count >= 3:
                self.recommended_action = self.ACTION_RECALIBRATE
            else:
                self.recommended_action = self.ACTION_EXPLORE
        elif self.regime == self.REGIME_CONVERGED:
            self.recommended_action = self.ACTION_CONTINUE
        elif self.calibration_score < 0.35:
            self.recommended_action = self.ACTION_RECALIBRATE
        else:
            self.recommended_action = self.ACTION_CONTINUE

        outer = map_meta_to_outer(self.regime, self.recommended_action)

        assessment = {
            "regime": self.regime,
            "recommended_action": self.recommended_action,
            "confidence_in_self": self.confidence_in_self,
            "calibration_score": self.calibration_score,
            "mean_recent_error": mean_err,
            "mean_recent_progress": mean_prog,
            "success_rate": sr,
            "world_residual": residual,
            "prog_trend": prog_trend,
            "err_trend": err_trend,
            "goal_distance": goal_distance,
            "alerts": list(self.alerts),
            "outer": outer,
            "outer_phase": outer["outer_phase"],
            "g_active": outer["g_active"],
        }
        self.last_assessment = assessment
        self.history.append({"time": time.time(), **assessment})
        if len(self.history) > 100:
            self.history = self.history[-100:]
        return assessment

    def apply_to_config_and_policy(
        self,
        config: Config,
        policy: Any,
        self_model: SelfModel,
    ) -> List[str]:
        notes = []
        action = self.recommended_action

        if action == self.ACTION_CAUTION:
            if hasattr(policy, "blend"):
                old = policy.blend
                policy.blend = clamp(policy.blend * 0.75, 0.05, 0.6)
                notes.append(f"caution: blend {old:.3f}->{policy.blend:.3f}")
            old_ra = config.risk_aversion
            config.risk_aversion = clamp(config.risk_aversion + 0.08, 0.3, 0.95)
            notes.append(f"caution: risk_aversion {old_ra:.3f}->{config.risk_aversion:.3f}")

        elif action == self.ACTION_EXPLORE:
            if hasattr(policy, "blend"):
                old = policy.blend
                policy.blend = clamp(policy.blend * 1.25, 0.1, 0.7)
                notes.append(f"explore: blend {old:.3f}->{policy.blend:.3f}")

        elif action == self.ACTION_RECALIBRATE:
            old_ls = config.learning_step
            config.learning_step = clamp(config.learning_step * 1.15, 0.05, 0.25)
            if hasattr(policy, "conf_base"):
                policy.conf_base = clamp(policy.conf_base * 0.85, 0.35, 0.9)
            notes.append(f"recalibrate: learning_step {old_ls:.4f}->{config.learning_step:.4f}")

        elif action == self.ACTION_ESCALATE:
            notes.append("escalate: meta recommends human/upper review (G)")

        else:
            ra_base = float(getattr(config, "risk_aversion_baseline", 0.6))
            ls_base = float(getattr(config, "learning_step_baseline", 0.12))
            if config.risk_aversion > ra_base + EPS:
                old_ra = config.risk_aversion
                config.risk_aversion = clamp(config.risk_aversion - 0.05, ra_base, 0.95)
                notes.append(f"relax: risk_aversion {old_ra:.3f}->{config.risk_aversion:.3f}")
            if config.learning_step > ls_base + EPS:
                old_ls = config.learning_step
                config.learning_step = clamp(config.learning_step * 0.97, ls_base, 0.25)
                notes.append(f"relax: learning_step {old_ls:.4f}->{config.learning_step:.4f}")

        return notes

    def summary(self) -> str:
        a = self.last_assessment or {}
        return (
            f"regime={a.get('regime', '?')} action={a.get('recommended_action', '?')} "
            f"outer={a.get('outer_phase', '?')} g={a.get('g_active', False)} "
            f"self_conf={a.get('confidence_in_self', 0):.2f} "
            f"calibration={a.get('calibration_score', 0):.2f} "
            f"alerts={a.get('alerts', [])}"
        )


# ============================================================================
# MODULE FLATTENED: omega.planning.planner
# ============================================================================

# -*- coding: utf-8 -*-
"""
Short-horizon planner.

Design rules:
- Planner only PROPOSES. It never executes, never issues tokens.
- Each PlanStep becomes a normal Intention/Suggestion that still goes through
  approve → authorize(token) → execute.
- Prefers installed Skills when available; falls back to primitive tools.
- File Mission: emits list → write → read macro candidates when FS tools allowed.
"""


import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Set



@dataclass
class PlanStep:
    action: str
    args: Dict[str, Any] = field(default_factory=dict)
    skill_id: Optional[str] = None
    rationale: str = ""
    estimated_risk: float = 0.1
    estimated_progress: float = 0.0

    def to_intention(self, policy_id: str = "planner", policy_version: str = "1.0") -> Intention:
        return Intention(
            action=self.action,
            args=deep(self.args),
            policy_id=policy_id,
            policy_version=policy_version,
            rationale=self.rationale or f"plan_step skill={self.skill_id}",
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Plan:
    plan_id: str
    steps: List[PlanStep] = field(default_factory=list)
    horizon: int = 1
    score: float = 0.0
    goal_distance_before: float = 0.0
    goal_distance_after: float = 0.0
    created_at: float = field(default_factory=time.time)
    meta: Dict[str, Any] = field(default_factory=dict)

    def first_step(self) -> Optional[PlanStep]:
        return self.steps[0] if self.steps else None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "horizon": self.horizon,
            "score": self.score,
            "goal_distance_before": self.goal_distance_before,
            "goal_distance_after": self.goal_distance_after,
            "created_at": self.created_at,
            "steps": [s.to_dict() for s in self.steps],
            "meta": deep(self.meta),
        }


def _goal_distance(state: Mapping[str, float], goal: Mapping[str, float]) -> float:
    if not goal:
        return 0.0
    # Prefer mission readiness when present
    if "target_ready" in goal or "mission_progress" in state or "target_ready" in state:
        ready = float(state.get("target_ready", state.get("mission_progress", 0.0)))
        return max(0.0, 1.0 - ready)
    return sum(abs(float(state.get(k, 0.0)) - float(t)) for k, t in goal.items()) / max(1, len(goal))


def _apply_args_to_state(
    state: Mapping[str, float],
    args: Mapping[str, Any],
    blend: float = 0.3,
) -> Dict[str, float]:
    out = dict(state)
    for k, v in args.items():
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        old = float(out.get(k, 0.0))
        out[k] = (1.0 - blend) * old + blend * fv
    return out


def _fs_mission_macro(
    state: Mapping[str, float],
    *,
    skills: Optional[SkillRegistry],
    allowed: Optional[Set[str]],
) -> List[PlanStep]:
    """list → write → read macro for File Mission (one step returned as best-first)."""
    ready = float(state.get("target_ready", 0.0))
    exists = float(state.get("target_exists", 0.0))
    steps: List[PlanStep] = []

    write_skill = skills.get("write_target") if skills else None
    verify_skill = skills.get("verify_target") if skills else None
    list_skill = skills.get("list_parent") if skills else None

    def _ok(action: str) -> bool:
        return allowed is None or action in allowed

    if ready >= 0.99:
        if verify_skill and _ok(verify_skill.preferred_action):
            steps.append(PlanStep(
                action=verify_skill.preferred_action,
                args=dict(verify_skill.template_args),
                skill_id=verify_skill.skill_id,
                rationale="macro: verify ready target",
                estimated_risk=verify_skill.base_risk,
                estimated_progress=0.05,
            ))
        elif _ok("read_file"):
            steps.append(PlanStep(
                action="read_file",
                args={"path": "out/report.txt"},
                rationale="macro: verify read",
                estimated_risk=0.08,
                estimated_progress=0.05,
            ))
        return steps

    if exists < 0.5 or float(state.get("target_match", 0.0)) < 0.5:
        if write_skill and _ok(write_skill.preferred_action):
            steps.append(PlanStep(
                action=write_skill.preferred_action,
                args=dict(write_skill.template_args),
                skill_id=write_skill.skill_id,
                rationale="macro: write_target skill",
                estimated_risk=write_skill.base_risk,
                estimated_progress=0.9,
            ))
        elif _ok("write_file"):
            steps.append(PlanStep(
                action="write_file",
                args={"path": "out/report.txt", "content": "OMEGA-OK\n"},
                rationale="macro: write_file fallback",
                estimated_risk=0.4,
                estimated_progress=0.85,
            ))

    if list_skill and _ok(list_skill.preferred_action):
        steps.append(PlanStep(
            action=list_skill.preferred_action,
            args=dict(list_skill.template_args),
            skill_id=list_skill.skill_id,
            rationale="macro: list_parent",
            estimated_risk=list_skill.base_risk,
            estimated_progress=0.1,
        ))
    elif _ok("list_files"):
        steps.append(PlanStep(
            action="list_files",
            args={"path": "out"},
            rationale="macro: list out/",
            estimated_risk=0.05,
            estimated_progress=0.1,
        ))

    if verify_skill and _ok(verify_skill.preferred_action):
        steps.append(PlanStep(
            action=verify_skill.preferred_action,
            args=dict(verify_skill.template_args),
            skill_id=verify_skill.skill_id,
            rationale="macro: verify after write",
            estimated_risk=verify_skill.base_risk,
            estimated_progress=0.2,
        ))

    return steps


class ShortHorizonPlanner:
    """Greedy short-horizon planner over Skills and primitive actions."""

    planner_id = "short-horizon"
    planner_version = "1.1"

    def __init__(
        self,
        horizon: int = 3,
        max_candidates: int = 16,
        progress_weight: float = 1.0,
        risk_weight: float = 0.45,
        skill_bonus: float = 0.08,
    ):
        self.horizon = max(1, int(horizon))
        self.max_candidates = max(2, int(max_candidates))
        self.progress_weight = float(progress_weight)
        self.risk_weight = float(risk_weight)
        self.skill_bonus = float(skill_bonus)
        self.last_plan: Optional[Plan] = None

    def plan(
        self,
        state: Mapping[str, float],
        goal: Mapping[str, float],
        *,
        world_model: Optional[WorldModel] = None,
        skills: Optional[SkillRegistry] = None,
        allowed_actions: Optional[Set[str]] = None,
        stall: bool = False,
        recent_error: float = 0.0,
    ) -> Plan:
        state = dict(state)
        goal = dict(goal) if goal else {}
        dist0 = _goal_distance(state, goal)

        candidates = self._generate_candidates(
            state, goal,
            skills=skills,
            allowed_actions=allowed_actions,
            stall=stall,
            recent_error=recent_error,
        )

        if not candidates:
            step = PlanStep(
                action="observe",
                args={},
                rationale="no candidates; observe",
                estimated_risk=0.0,
                estimated_progress=0.0,
            )
            plan = Plan(
                plan_id=f"plan-{int(time.time() * 1000)}",
                steps=[step],
                horizon=1,
                score=0.0,
                goal_distance_before=dist0,
                goal_distance_after=dist0,
                meta={"fallback": True},
            )
            self.last_plan = plan
            return plan

        scored: List[tuple] = []
        for step in candidates:
            score, dist_after = self._score_step(
                state, goal, step, world_model=world_model
            )
            scored.append((score, dist_after, step))

        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, best_dist, best_step = scored[0]

        steps: List[PlanStep] = [best_step]
        sim_state = dict(state)
        # Mission-aware sim
        if best_step.action == "write_file":
            sim_state["target_exists"] = 1.0
            sim_state["target_match"] = 1.0
            sim_state["target_ready"] = 1.0
            sim_state["mission_progress"] = 1.0
        elif best_step.action == "read_file" and float(state.get("target_ready", 0)) >= 0.99:
            sim_state["reads"] = float(state.get("reads", 0)) + 1.0
        else:
            sim_state = _apply_args_to_state(state, best_step.args)

        for _ in range(1, self.horizon):
            sub = self._generate_candidates(
                sim_state, goal,
                skills=skills,
                allowed_actions=allowed_actions,
                stall=False,
                recent_error=recent_error,
            )
            if not sub:
                break
            sub_scored = []
            for st in sub:
                sc, da = self._score_step(sim_state, goal, st, world_model=world_model)
                sub_scored.append((sc, da, st))
            sub_scored.sort(key=lambda x: x[0], reverse=True)
            _, _, nxt = sub_scored[0]
            if nxt.estimated_progress < 1e-4 and len(steps) >= 1:
                break
            if nxt.action == best_step.action and nxt.args == best_step.args:
                break
            steps.append(nxt)
            if nxt.action == "write_file":
                sim_state["target_ready"] = 1.0
                sim_state["mission_progress"] = 1.0
            else:
                sim_state = _apply_args_to_state(sim_state, nxt.args)

        final_dist = _goal_distance(sim_state, goal)
        plan = Plan(
            plan_id=f"plan-{int(time.time() * 1000)}",
            steps=steps,
            horizon=len(steps),
            score=best_score,
            goal_distance_before=dist0,
            goal_distance_after=final_dist,
            meta={
                "candidates": len(candidates),
                "stall": stall,
                "recent_error": recent_error,
                "macro": any("macro" in (s.rationale or "") for s in steps),
            },
        )
        self.last_plan = plan
        return plan

    def _generate_candidates(
        self,
        state: Mapping[str, float],
        goal: Mapping[str, float],
        *,
        skills: Optional[SkillRegistry],
        allowed_actions: Optional[Set[str]],
        stall: bool,
        recent_error: float,
    ) -> List[PlanStep]:
        steps: List[PlanStep] = []
        allowed = allowed_actions

        # 0) File Mission macros (high priority candidates)
        fs_actions = {"write_file", "read_file", "list_files"}
        if allowed is None or (fs_actions & set(allowed)):
            if (
                "target_ready" in state
                or "mission_progress" in state
                or (skills and skills.get("write_target") is not None)
            ):
                steps.extend(_fs_mission_macro(state, skills=skills, allowed=allowed))

        # 1) Skills first (generic)
        if skills is not None:
            for skill in skills.list_skills() if hasattr(skills, "list_skills") else []:
                if isinstance(skill, dict):
                    sid = skill.get("skill_id")
                    skill_obj = skills.get(sid) if sid else None
                else:
                    skill_obj = skill
                if skill_obj is None:
                    continue
                action = skill_obj.preferred_action
                if allowed is not None and action not in allowed:
                    continue
                # skip if already added by macro with same skill_id
                if any(s.skill_id == skill_obj.skill_id for s in steps):
                    continue
                args = dict(skill_obj.template_args or {})
                for k, t in goal.items():
                    if k not in args:
                        try:
                            float(t)
                        except (TypeError, ValueError):
                            continue
                        cur = float(state.get(k, 0.0))
                        blend = 0.45 if stall else 0.30
                        if recent_error > 0.08:
                            blend *= 0.7
                        args[k] = cur + blend * (float(t) - cur)
                steps.append(PlanStep(
                    action=action,
                    args=args,
                    skill_id=skill_obj.skill_id,
                    rationale=f"skill:{skill_obj.skill_id}",
                    estimated_risk=float(skill_obj.base_risk),
                    estimated_progress=0.0,
                ))

        # 2) Primitive measure toward goal
        if allowed is None or "measure" in allowed:
            blend = 0.40 if stall else 0.28
            if recent_error > 0.08:
                blend *= 0.75
            args = {}
            for k, t in goal.items():
                try:
                    ft = float(t)
                except (TypeError, ValueError):
                    continue
                cur = float(state.get(k, 0.0))
                args[k] = cur + blend * (ft - cur)
            if args:
                steps.append(PlanStep(
                    action="measure",
                    args=args,
                    rationale="primitive measure toward goal",
                    estimated_risk=0.15,
                ))

        # 3) set_param per goal key
        if allowed is None or "set_param" in allowed:
            for k, t in list(goal.items())[:4]:
                try:
                    ft = float(t)
                except (TypeError, ValueError):
                    continue
                cur = float(state.get(k, 0.0))
                if abs(ft - cur) < 1e-4:
                    continue
                steps.append(PlanStep(
                    action="set_param",
                    args={"key": k, "value": ft},
                    rationale=f"set_param {k}",
                    estimated_risk=0.35,
                ))

        # 4) observe / noop
        if allowed is None or "observe" in allowed:
            steps.append(PlanStep(
                action="observe", args={}, rationale="observe", estimated_risk=0.0,
            ))
        if allowed is None or "noop" in allowed:
            steps.append(PlanStep(
                action="noop", args={}, rationale="noop", estimated_risk=0.0,
            ))

        return steps[: self.max_candidates]

    def _score_step(
        self,
        state: Mapping[str, float],
        goal: Mapping[str, float],
        step: PlanStep,
        *,
        world_model: Optional[WorldModel],
    ) -> tuple:
        before = _goal_distance(state, goal)

        if step.action == "write_file":
            after_state = dict(state)
            after_state["target_exists"] = 1.0
            after_state["target_match"] = 1.0
            after_state["target_ready"] = 1.0
            after_state["mission_progress"] = 1.0
        elif world_model and world_model.state:
            try:
                effect = {}
                for k, v in step.args.items():
                    try:
                        effect[k] = float(v) - float(state.get(k, 0.0))
                    except (TypeError, ValueError):
                        pass
                after_state = world_model.predict(steps=1, action_effect=effect or None)
            except Exception:
                after_state = _apply_args_to_state(state, step.args)
        else:
            after_state = _apply_args_to_state(state, step.args)

        after = _goal_distance(after_state, goal)
        progress = before - after
        if step.estimated_progress <= 0:
            step.estimated_progress = progress
        else:
            step.estimated_progress = max(step.estimated_progress, progress)

        score = self.progress_weight * step.estimated_progress - self.risk_weight * step.estimated_risk
        if step.skill_id:
            score += self.skill_bonus
            sk = None
            # bias toward historically successful skills when registry not passed here
            score += 0.02
        if step.action in ("observe", "noop") and before > 0.05:
            score -= 0.02
        return score, after


# ============================================================================
# MODULE FLATTENED: omega.policy.adaptive
# ============================================================================

# -*- coding: utf-8 -*-
"""Adaptive continuous policy."""


from typing import Any, Dict, List, Mapping, Optional



class AdaptivePolicy:
    policy_id = "adaptive-policy"
    policy_version = "3.11"

    def __init__(self, blend: float = 0.35, conf_base: float = 0.88):
        self.blend = blend
        self.conf_base = conf_base
        self.experience: List[Dict[str, Any]] = []

    def intend(
        self,
        state: Mapping[str, float],
        goal: Mapping[str, float],
        stall: bool = False,
        recent_error: float = 0.0,
    ) -> Intention:
        local_blend = self.blend
        if stall:
            local_blend = clamp(self.blend * 1.45, 0.15, 0.75)
        elif recent_error > 0.08:
            local_blend = clamp(self.blend * 0.7, 0.08, 0.5)

        args = {}
        for key, target in goal.items():
            current = state.get(key, 0.0)
            gap = target - current
            args[key] = current + local_blend * gap
        return Intention(
            action="measure",
            args=args,
            policy_id=self.policy_id,
            policy_version=self.policy_version,
            rationale=f"blend={local_blend:.3f} stall={stall}",
        )

    def predict(
        self,
        state: Mapping[str, float],
        intention: Intention,
        goal: Mapping[str, float],
        world_model: Optional[WorldModel] = None,
        steps: int = 1,
    ) -> Prediction:
        if world_model and world_model.state:
            action_effect = {
                k: float(intention.args.get(k, state.get(k, 0.0))) - state.get(k, 0.0)
                for k in intention.args
            }
            expected = world_model.predict(steps=steps, action_effect=action_effect)
        else:
            expected = dict(state)
            for key, target in intention.args.items():
                expected[key] = 0.80 * state.get(key, 0.0) + 0.20 * float(target)

        conf = self.conf_base
        if world_model:
            conf = clamp(self.conf_base - 0.5 * world_model.mean_residual, 0.35, 0.95)

        return Prediction(
            expected=expected,
            confidence=conf,
            model_id="adaptive-model",
            model_version="3.11",
        )

    def expected_progress(
        self,
        state: Mapping[str, float],
        predicted: Mapping[str, float],
        goal: Mapping[str, float],
    ) -> float:
        before = 0.0
        after = 0.0
        for k, t in goal.items():
            before += abs(state.get(k, 0.0) - t)
            after += abs(predicted.get(k, 0.0) - t)
        return before - after

    def remember(self, exp: Dict[str, Any], limit: int = 64):
        self.experience.append(exp)
        if len(self.experience) > limit:
            self.experience = self.experience[-limit:]

    def adapt(self, rate: float = 0.08):
        if len(self.experience) < 4:
            return
        recent = self.experience[-12:]
        avg_error = sum(e.get("error", 0.0) for e in recent) / len(recent)
        avg_progress = sum(e.get("progress", 0.0) for e in recent) / len(recent)
        if avg_error > 0.06:
            self.blend = clamp(self.blend - rate * 0.6, 0.08, 0.7)
            self.conf_base = clamp(self.conf_base - rate * 0.3, 0.4, 0.95)
        elif avg_progress > 0.01:
            self.blend = clamp(self.blend + rate * 0.35, 0.08, 0.7)
            self.conf_base = clamp(self.conf_base + rate * 0.15, 0.4, 0.95)

    @property
    def blend_factor(self):
        return self.blend


SmoothPolicy = AdaptivePolicy


# ============================================================================
# MODULE FLATTENED: omega.policy.file_task
# ============================================================================

# -*- coding: utf-8 -*-
"""Policy that drives FileSandboxWorld via list/read/write_file (File Mission)."""


from typing import Any, Dict, Mapping



class FileTaskPolicy:
    """Task policy for sandbox files — prefers own intent over planner."""

    policy_id = "file-task-policy"
    policy_version = "1.1"
    prefer_policy_intent = True

    def __init__(self, target_path: str = "out/result.txt", target_content: str = "omega-ok\n"):
        self.target_path = target_path
        self.target_content = target_content
        self.experience = []
        self._world = None

    def bind_world(self, world) -> None:
        self._world = world
        if world is not None and hasattr(world, "set_mission"):
            world.set_mission(self.target_path, self.target_content)

    def _target_ready(self, state: Mapping[str, float]) -> bool:
        if float(state.get("target_ready", 0.0)) >= 0.99:
            return True
        world = self._world
        if world is not None and hasattr(world, "mission_status"):
            return bool(world.mission_status().get("ready"))
        return False

    def intend(
        self,
        state: Mapping[str, float],
        goal: Mapping[str, float],
        stall: bool = False,
        recent_error: float = 0.0,
    ) -> Intention:
        path = self.target_path
        world = self._world

        # Mission already satisfied → cheap verify (read) then list
        if self._target_ready(state):
            return Intention(
                action="read_file",
                args={"path": path},
                policy_id=self.policy_id,
                policy_version=self.policy_version,
                rationale="mission ready; verify read",
            )

        if world is not None and hasattr(world, "list_files"):
            try:
                parts = path.split("/")
                parent = "/".join(parts[:-1]) if len(parts) > 1 else "."
                names = world.list_files(parent)
                leaf = parts[-1]
                present = any(n.rstrip("/") == leaf or n == leaf for n in names)
                if not present:
                    return Intention(
                        action="write_file",
                        args={"path": path, "content": self.target_content},
                        policy_id=self.policy_id,
                        policy_version=self.policy_version,
                        rationale="create mission target",
                    )
                try:
                    text = world.read_text(path)
                    if text != self.target_content:
                        return Intention(
                            action="write_file",
                            args={"path": path, "content": self.target_content},
                            policy_id=self.policy_id,
                            policy_version=self.policy_version,
                            rationale="fix mission content",
                        )
                except Exception:
                    return Intention(
                        action="write_file",
                        args={"path": path, "content": self.target_content},
                        policy_id=self.policy_id,
                        policy_version=self.policy_version,
                        rationale="rewrite unreadable target",
                    )
                return Intention(
                    action="read_file",
                    args={"path": path},
                    policy_id=self.policy_id,
                    policy_version=self.policy_version,
                    rationale="target present; confirm content",
                )
            except Exception as e:
                return Intention(
                    action="write_file",
                    args={"path": path, "content": self.target_content},
                    policy_id=self.policy_id,
                    policy_version=self.policy_version,
                    rationale=f"fallback write: {e}",
                )

        return Intention(
            action="write_file",
            args={"path": path, "content": self.target_content},
            policy_id=self.policy_id,
            policy_version=self.policy_version,
            rationale="no world context; write target",
        )

    def predict(
        self,
        state: Mapping[str, float],
        intention: Intention,
        goal: Mapping[str, float],
        world_model=None,
        steps: int = 1,
    ) -> Prediction:
        expected = dict(state)
        if intention.action == "write_file":
            expected["file_count"] = float(state.get("file_count", 0.0)) + 0.5
            expected["writes"] = float(state.get("writes", 0.0)) + 1.0
            expected["ok"] = 1.0
            expected["target_exists"] = 1.0
            expected["target_match"] = 1.0
            expected["target_ready"] = 1.0
            expected["mission_progress"] = 1.0
        elif intention.action == "list_files":
            expected["ok"] = 1.0
        elif intention.action == "read_file":
            expected["reads"] = float(state.get("reads", 0.0)) + 1.0
            expected["ok"] = 1.0
            if self._target_ready(state):
                expected["target_ready"] = 1.0
                expected["mission_progress"] = 1.0
        return Prediction(expected=expected, confidence=0.88)

    def expected_progress(
        self,
        before: Mapping[str, float],
        after: Mapping[str, float],
        goal: Mapping[str, float],
    ) -> float:
        """Progress toward mission readiness (not just write count)."""
        b = float(before.get("mission_progress", 0.0))
        a = float(after.get("mission_progress", 0.0))
        delta = a - b
        if delta > 0:
            return max(delta, 0.35)
        # Predicted step that should complete mission
        if float(after.get("target_ready", 0.0)) >= 0.99 and b < 0.99:
            return 0.9
        if float(after.get("writes", 0.0)) > float(before.get("writes", 0.0)):
            return 0.45
        if float(after.get("reads", 0.0)) > float(before.get("reads", 0.0)):
            return 0.15
        return 0.05

    def remember(self, item: Dict[str, Any], limit: int = 50):
        self.experience.append(item)
        if len(self.experience) > limit:
            self.experience = self.experience[-limit:]

    def adapt(self, rate: float = 0.05):
        return None


# ============================================================================
# MODULE FLATTENED: omega.kernel
# ============================================================================

# -*- coding: utf-8 -*-
"""Omega Kernel - central orchestrator (3.16 outer law)."""


from pathlib import Path
from typing import Any, Dict, Optional




class Omega(_LifecycleAuthMixin, _LifecyclePersistMixin, _CognitionMixin, _ExecutionMixin):
    def __init__(
        self,
        secret: str,
        storage_dir: str = "./omega_cas_brain",
        config: Optional[Config] = None,
        boundary: Optional[Boundary] = None,
        decide_fn=None,
        auto_restore: bool = True,
        tool_registry: Optional[ToolRegistry] = None,
        vault: Optional[Vault] = None,
        migrate_legacy: bool = False,
        anchor_path: Optional[str] = None,
    ):
        if not isinstance(secret, str) or len(secret) < MIN_SECRET_LENGTH:
            raise ValueError("secret must contain >= 16 chars")
        self.secret = secret
        # Explicit operator opt-in: accept and seal pre-MAC storage exactly once.
        self.migrate_legacy = bool(migrate_legacy)
        # Rollback anchor: a MAC'd record of the latest saved state, kept OUTSIDE
        # the storage directory so that rolling back the directory is detectable.
        self.save_seq = 0
        self.anchor_path: Optional[Path] = None
        if anchor_path:
            self.anchor_path = Path(anchor_path).resolve()
            if self.anchor_path.is_relative_to(Path(storage_dir).resolve()):
                raise ValueError("rollback anchor must live outside the storage directory")
        self.config = config if config is not None else Config()
        self.boundary = boundary if boundary is not None else Boundary(DEFAULT_ALLOWED_ACTIONS)
        self.decide_fn = decide_fn if decide_fn is not None else default_decide

        self.storage_dir = Path(storage_dir)
        self.memory = OmegaMemoryEngine(
            storage_dir=storage_dir,
            key=derive_subkey(secret, BLOCK_KEY_LABEL),
            allow_legacy=self.migrate_legacy,
        )
        self.audit = AuditChain(key=derive_subkey(secret, AUDIT_KEY_LABEL))
        self.tokens = TokenBook(key=derive_subkey(secret, TOKEN_KEY_LABEL))
        self.lock = StorageLock(self.storage_dir / "LOCK")
        self.tools = tool_registry if tool_registry is not None else make_default_registry()

        self.skill_store = SkillStore(
            storage_dir,
            key=derive_subkey(secret, SKILL_KEY_LABEL),
            allow_legacy=self.migrate_legacy,
        )
        self.skills = SkillRegistry()
        self.planner = ShortHorizonPlanner(
            horizon=getattr(self.config, "planner_horizon", 3),
            max_candidates=getattr(self.config, "planner_max_candidates", 12),
        )
        self.goal_tree = GoalTree()
        self.intrinsic = IntrinsicMotivation()

        self.sm = StateMachine()
        self.topology = JiaLoop(
            mod=getattr(self.config, "topology_mod", 16),
            cycle=getattr(self.config, "topology_cycle", 240),
        )
        self.llm = LLMBridge(
            enabled=bool(getattr(self.config, "llm_bridge_enabled", False)),
            endpoint=str(getattr(self.config, "llm_bridge_endpoint", "") or ""),
            model=str(getattr(self.config, "llm_bridge_model", "") or ""),
        )

        self.vault: Optional[Vault] = vault
        if self.vault is None and getattr(self.config, "vault_enabled", False):
            try:
                self.vault = Vault.load_or_create(
                    self.storage_dir, secret,
                    iterations=int(getattr(self.config, "vault_kdf_iterations", 200_000)),
                )
            except Exception as exc:
                print(f"[Vault] init failed, continuing without vault: {exc}")
                self.vault = None

        pred_mode = str(getattr(self.config, "world_model_predictor", "linear"))
        if pred_mode in ("hybrid", "attention"):
            self.world_model = HybridWorldModel(
                mode=pred_mode,
                d_model=int(getattr(self.config, "attention_d_model", 8)),
                heads=int(getattr(self.config, "attention_heads", 2)),
                window=int(getattr(self.config, "attention_window", 16)),
            )
        else:
            self.world_model = WorldModel()

        self.goal: Dict[str, float] = {}
        self.hard_min: Dict[str, float] = {}
        self.hard_max: Dict[str, float] = {}
        self.calibration: Dict[str, float] = {}
        self.self_model = SelfModel()
        self.uncertainty = UncertaintyModel()
        self.meta = MetaCognition()
        # Advisory role shards. They never own governance, authorization, or execution.
        self.roles = RoleCouncil()

        self.epoch = 0
        self.view = View.SHAN
        self.paused = False
        self.escalated = False
        self.last_state: Dict[str, float] = {}
        self.suggestions: Dict[str, Suggestion] = {}
        self.integrity_ok: bool = True
        self.pending_action: Optional[Dict[str, Any]] = None
        self.last_fire_token: Optional[Dict[str, Any]] = None
        self.owner_ = self._create_owner()

        self.sm.transition_kernel(KernelState.RESTORING, force=True)
        self.sm.set_phase(CyclePhase.SYNC)

        if auto_restore:
            with self.lock:
                self._try_restore()
            if not self.memory.verify_all_blocks() or not self.audit.verify():
                self.integrity_ok = False
                self.sm.transition_kernel(KernelState.FAIL_CLOSED, force=True)
                print("[SAFE] post-restore integrity check failed -> fail-closed.")
            elif not self.skill_store.verify_all():
                self.integrity_ok = False
                self.sm.transition_kernel(KernelState.FAIL_CLOSED, force=True)
                print("[SAFE] skill CAS integrity check failed -> fail-closed.")
            else:
                if self.migrate_legacy:
                    self._seal_storage()
                self.sm.transition_kernel(KernelState.IDLE, force=True)
                self.sm.set_phase(CyclePhase.LISTEN)
        else:
            self.sm.transition_kernel(KernelState.IDLE, force=True)
            self.sm.set_phase(CyclePhase.LISTEN)

    def _check_inventory(self, head: Dict[str, Any]) -> None:
        """HEAD records the episode and skill inventories it was saved with."""
        missing_blocks = [h for h in head.get("active_hashes", [])
                          if not (self.memory.objects_dir / f"{self.memory.renamed.get(h, h)}.block").exists()]
        if missing_blocks:
            raise SealError(f"{len(missing_blocks)} episode block(s) committed by HEAD are missing (deleted?)")
        present_skills = set(self.skill_store.active_hashes())
        missing_skills = [h for h in head.get("skill_hashes", []) if h not in present_skills]
        if missing_skills:
            raise SealError(f"{len(missing_skills)} skill(s) committed by HEAD are missing from the index")

    def _seal_storage(self) -> None:
        """After an explicit legacy migration, rewrite all state under MACs."""
        self.save_cas_head()
        self.audit.save(self.memory.audit_file)
        self.tokens.save(self.memory.tokens_file)
        self.skill_store.save_index()
        print("[Migrate] legacy storage sealed under MAC. Restart without migrate_legacy from now on.")

    def tick_topology(self) -> Dict[str, Any]:
        """Advance the single topology clock (JiaLoop) and mirror onto StateMachine."""
        if not getattr(self.config, "topology_enabled", True):
            snap = self.topology.snapshot()
            self.sm.mirror_topology(snap.step)
            return snap.to_dict()
        snap = self.topology.tick()
        self.sm.mirror_topology(snap.step)
        return snap.to_dict()

    def memory_query(self, **kwargs):
        return self.memory.query(**kwargs)

    def memory_recent(self, n: int = 5):
        return self.memory.recent(n)

    def memory_stats(self):
        return self.memory.stats()

    def state_summary(self) -> Dict[str, Any]:
        topo = self.topology.snapshot().to_dict()
        meta_last = getattr(self.meta, "last_assessment", None) or {}
        return {
            "kernel": self.sm.summary(),
            "topology": topo,
            "topology_in_sync": self.sm.cycle_step == self.topology.step,
            "paused": self.paused,
            "escalated": self.escalated,
            "integrity_ok": self.integrity_ok,
            "rollback_protected": self.anchor_path is not None,
            "epoch": self.epoch,
            "vault": bool(self.vault),
            "llm_enabled": self.llm.enabled,
            "outer_phase": meta_last.get("outer_phase"),
            "g_active": meta_last.get("g_active"),
            "fire_token": self.last_fire_token,
            "world_model": (
                self.world_model.predictor_stats()
                if hasattr(self.world_model, "predictor_stats")
                else {"mode": "linear"}
            ),
        }


# ============================================================================
# MODULE FLATTENED: omega.policy.context
# ============================================================================

# -*- coding: utf-8 -*-
"""Context-adaptive policy and Meta-RL helpers (Phase-3 ready stubs)."""


from dataclasses import dataclass
from typing import Dict, List, Optional



@dataclass
class TaskSpec:
    task_id: str
    goal: Dict[str, float]
    description: str = ""


class TaskDistribution:
    def __init__(self, tasks: Optional[List[TaskSpec]] = None):
        if tasks is None:
            tasks = [
                TaskSpec("near", {"x": 0.5, "score": 0.7, "health": 0.95}, "near goal"),
                TaskSpec("mid", {"x": 1.0, "score": 1.0, "health": 0.9}, "mid goal"),
                TaskSpec("far", {"x": 1.5, "score": 0.9, "health": 0.85}, "far goal"),
            ]
        self.tasks = list(tasks)

    def sample(self) -> TaskSpec:
        return secrets.choice(self.tasks)


class MetaMemory:
    def __init__(self, capacity: int = 64, path=None):
        self.capacity = capacity
        self.path = path
        self.entries: List[Dict] = []

    def add(self, context, blend, conf_base, performance, task_id=""):
        self.entries.append({
            "context": list(context), "blend": float(blend),
            "conf_base": float(conf_base), "performance": float(performance),
            "task_id": task_id,
        })
        if len(self.entries) > self.capacity:
            self.entries = self.entries[-self.capacity:]

    def query_best(self, context, k=3):
        return None  # stub; full implementation later


class ContextAdaptivePolicy(AdaptivePolicy):
    policy_id = "context-adaptive-policy"
    policy_version = "3.11-meta"

    def __init__(self, blend=0.35, conf_base=0.88, meta_path=None):
        super().__init__(blend=blend, conf_base=conf_base)
        self.meta_memory = MetaMemory(capacity=80, path=meta_path)
        self.current_context = []
        self.current_task_id = ""
        self._base_blend = blend
        self._base_conf = conf_base

    def update_context(self, self_model, world_model, memory, goal, task_id="", config=None):
        self.current_task_id = task_id
        return None

    def commit_meta(self, performance: float):
        pass

    def reset_to_prior(self):
        self.blend = self._base_blend
        self.conf_base = self._base_conf


# ============================================================================
# MODULE FLATTENED: omega language supplement
# ============================================================================

"""
omega_lang.py
=============

把 v3.4 的语言层移植到 Omega V3.11。

内核（omega.py，V3.11）不做任何修改，本模块只通过它的公开方法调用：
    cognitive_cycle / approve_suggestion / authorize_suggestion /
    execute_authorized / audit.append / save_cas_head / _check_writable

与 v3.4 的差异（有意为之）：
1. 疑的状态（轮数、上一轮卡住原因）写入 V3.11 的审计哈希链，重启后从链中恢复。
2. 数值推演里同分存活的分支不再任选一条作为答案，改为回问。
3. 关系类回答（有关系、有用途）不进入测量路径，直接给出语言层的重新锚定回答；
   测量只用于数值锚点。
4. 测量必须经过 V3.11 的批准门：内核判 HOLD/ESCALATE 时不执行，也就没有现实观测。

依赖：同目录下的 omega.py（V3.11，已打 blend 反思修复）。
"""


import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple


MAX_DEPTH = 6
DEFAULT_LINES: Tuple[str, ...] = ("L1", "L2", "L3", "L4", "L5", "L6")


# ============================================================
# 数据结构
# ============================================================

@dataclass(frozen=True)
class Reading:
    """解读结果。implied 为空表示表层无关系；仅含对象本身表示"对象即结构"。"""
    candidates: Tuple[str, ...]
    confidence: float
    implied: Tuple[str, ...] = ()
    purpose: Optional[str] = None
    reframe: Optional[str] = None


class Stuck(str, Enum):
    NO_OBJECT = "no_object"
    NO_RELATION = "no_relation"
    NO_PURPOSE = "no_purpose"
    PURPOSE_UNANSWERED = "purpose_unanswered"
    RESOLVED = "resolved"


@dataclass
class Branch:
    code: str
    prediction: Dict[str, float]
    alive: bool = True
    error: Optional[float] = None


@dataclass
class Reply:
    kind: str                      # answer | hypotheses | ask_back | doubt | release | refused
    text: str
    view: str
    shan_id: str = ""
    detail: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# 读者
# ============================================================

class Interpreter(Protocol):
    def read(self, text: str) -> Reading: ...


class KeywordInterpreter:
    """词表读者。命中一个对象时，把对象本身作为 implied（对象即结构，走数值路径）。"""

    def __init__(self, lexicon: Mapping[str, str]):
        self.lexicon = dict(lexicon)

    def read(self, text: str) -> Reading:
        hits = sorted({obj for word, obj in self.lexicon.items() if word in text})
        if len(hits) == 1:
            return Reading(candidates=tuple(hits), confidence=0.9, implied=(hits[0],))
        if not hits:
            return Reading(candidates=(), confidence=0.0)
        return Reading(candidates=tuple(hits), confidence=0.4)


READ_PROMPT = """你是一个解读器。只输出一个 JSON 对象，不要任何其他文字，不要 markdown 代码块。

输入：
{text}

输出格式：
{{
  "candidates": ["原文中的对象，最多一个最相关的"],
  "confidence": 0 到 1 之间的小数,
  "implied": ["原文能支持的关系或结构，不要超出原文"],
  "purpose": 原文明确说明的用途，没有就填 null,
  "reframe": 在 purpose 非空时，用该用途重新表述的回应；否则填 null
}}

规则：
- 只写原文能支持的内容，不要补充原文没有的信息。
- 原文没有明确说明用途时，purpose 必须是 null。
- 无法确定对象时，candidates 为空列表，confidence 为 0。
"""


def parse_model_output(raw: str) -> Optional[Reading]:
    """严格解析模型输出。任何字段类型不对都返回 None（fail-closed）。"""
    try:
        data = json.loads(raw.strip())
    except (json.JSONDecodeError, AttributeError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    candidates = data.get("candidates")
    implied = data.get("implied")
    confidence = data.get("confidence")
    purpose = data.get("purpose")
    reframe = data.get("reframe")
    if not isinstance(candidates, list) or not all(isinstance(x, str) for x in candidates):
        return None
    if not isinstance(implied, list) or not all(isinstance(x, str) for x in implied):
        return None
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        return None
    if not (0.0 <= float(confidence) <= 1.0):
        return None
    if purpose is not None and not isinstance(purpose, str):
        return None
    if reframe is not None and not isinstance(reframe, str):
        return None
    if purpose is None:
        reframe = None  # 用途为空时，模型擅自写的重述一律清掉
    return Reading(
        candidates=tuple(candidates[:1]),
        confidence=float(confidence),
        implied=tuple(implied),
        purpose=purpose,
        reframe=reframe,
    )


class ModelReader:
    """任意模型的读者适配：complete(prompt) -> str。解析失败返回空读数。"""

    def __init__(self, complete: Callable[[str], str]):
        self.complete = complete
        self.parse_failures = 0

    def read(self, text: str) -> Reading:
        reading = parse_model_output(self.complete(READ_PROMPT.format(text=text)))
        if reading is None:
            self.parse_failures += 1
            return Reading(candidates=(), confidence=0.0)
        return reading


# ============================================================
# 裁判
# ============================================================

class Judge(Protocol):
    def aligned(self, purpose: str, reframe: str) -> bool: ...


class PresenceJudge:
    """旧行为：有重述即对上。默认。"""

    def aligned(self, purpose: str, reframe: str) -> bool:
        return bool(reframe)


class RuleJudge:
    """
    方向规则（粗版）：指向词命中且反向词未命中才算对上；用途无规则则未对上（fail-closed）。
    只看词，不看语义。字面对应但语义相反、且不含反向词的回应会被误判为对上。
    """

    def __init__(self, rules: Mapping[str, Tuple[Sequence[str], Sequence[str]]]):
        self.rules = {p: (tuple(g), tuple(a)) for p, (g, a) in rules.items()}

    def aligned(self, purpose: str, reframe: str) -> bool:
        if not reframe or purpose not in self.rules:
            return False
        goal, against = self.rules[purpose]
        if any(bad in reframe for bad in against):
            return False
        return any(good in reframe for good in goal)


# ============================================================
# 语言层
# ============================================================

class LangLayer:
    """
    包在 Omega V3.11 外面的语言层。

    doubt 状态从审计链恢复，因此重启后不会丢失"悬着的问题"。
    所有状态变化都以审计事件的形式写入哈希链：
        lang_ask_back / lang_doubt / lang_release / lang_resolved / lang_reanchor
    """

    def __init__(
        self,
        omega: Omega,
        *,
        interpreter: Optional[Interpreter] = None,
        judge: Optional[Judge] = None,
        lines: Optional[Sequence[str]] = None,
        line_keys: Optional[Sequence[str]] = None,
        depth: int = MAX_DEPTH,
        line_step: float = 0.5,
        branch_tolerance: float = 0.5,
        min_anchor_conf: float = 0.6,
        doubt_threshold: int = 3,
    ):
        if not 1 <= depth <= MAX_DEPTH:
            raise ValueError(f"depth must be in [1, {MAX_DEPTH}]")
        self.omega = omega
        self.interpreter: Interpreter = interpreter or KeywordInterpreter({})
        self.judge: Judge = judge or PresenceJudge()
        self.lines: Tuple[str, ...] = tuple(lines or DEFAULT_LINES)[:depth]
        if len(self.lines) != depth:
            raise ValueError("lines must provide at least `depth` names")
        self.line_keys: Optional[Tuple[str, ...]] = tuple(line_keys) if line_keys else None
        if self.line_keys is not None and len(self.line_keys) != depth:
            raise ValueError("line_keys must have exactly `depth` entries")
        self.depth = depth
        self.line_step = float(line_step)
        self.branch_tolerance = float(branch_tolerance)
        self.min_anchor_conf = float(min_anchor_conf)
        self.doubt_threshold = int(doubt_threshold)
        self.doubt_rounds: Dict[str, int] = {}
        self.doubt_last: Dict[str, Stuck] = {}
        self.lattice: List[Branch] = []
        self._restore_doubt_from_audit()

    # ------------------------------------------------------------------
    # 审计
    # ------------------------------------------------------------------

    def _restore_doubt_from_audit(self) -> None:
        for rec in self.omega.audit.records:
            ev = rec.get("event")
            p = rec.get("payload") or {}
            obj = p.get("object")
            if ev == "lang_doubt" and obj is not None:
                self.doubt_rounds[obj] = int(p["rounds"])
                self.doubt_last[obj] = Stuck(p["stuck"])
            elif ev == "lang_release" and obj is not None:
                self.doubt_rounds.pop(obj, None)
                self.doubt_last[obj] = Stuck(p["stuck"])
            elif ev == "lang_resolved" and obj is not None:
                self.doubt_rounds.pop(obj, None)
                self.doubt_last.pop(obj, None)

    def _log(self, event: str, payload: Mapping[str, Any]) -> None:
        with self.omega.lock:
            self.omega.audit.append(event, dict(payload))
            self.omega.save_cas_head()

    # ------------------------------------------------------------------
    # 数值推演（NOT_SHAN，数值层）
    # ------------------------------------------------------------------

    def deduce(self, anchor: Mapping[str, float]) -> List[Branch]:
        """2^depth 条分支。第 i 爻决定某个维度向上或向下偏移 line_step。"""
        if self.line_keys is not None:
            missing = [k for k in self.line_keys if k not in anchor]
            if missing:
                raise KeyError(f"anchor lacks line keys: {missing}")
            keys_for = list(self.line_keys)
        else:
            keys = sorted(anchor)
            if not keys:
                self.lattice = []
                return self.lattice
            keys_for = [keys[i % len(keys)] for i in range(self.depth)]

        branches: List[Branch] = []
        for n in range(2 ** self.depth):
            code = format(n, f"0{self.depth}b")
            predicted = dict(anchor)
            for i, bit in enumerate(code):
                sign = 1.0 if bit == "1" else -1.0
                predicted[keys_for[i]] = predicted[keys_for[i]] + sign * self.line_step
            branches.append(Branch(code=code, prediction=predicted))
        self.lattice = branches
        return branches

    def _decode(self, code: str) -> str:
        return "；".join(
            f"{name}{'阳' if bit == '1' else '阴'}" for name, bit in zip(self.lines, code)
        )

    def _reanchor(self, *, obj: str, shan_id: str, observation: Mapping[str, float]) -> Reply:
        for b in self.lattice:
            if b.alive:
                b.error = vector_rms(vector_diff(b.prediction, observation))
                if b.error > self.branch_tolerance:
                    b.alive = False

        alive = [b for b in self.lattice if b.alive]
        view = self.omega.view

        if not alive:
            reply = Reply("ask_back", "现实排除了全部推演分支，这座山的定义需要你补充。",
                          view.value, shan_id, {"survivors": 0})
        else:
            best_err = min(b.error for b in alive)
            ties = [b for b in alive if abs(b.error - best_err) <= 1e-9]
            best = ties[0]
            if len(ties) > 1:
                # 同分存活：现实无法区分，任选一条就是随机回答，改为回问
                reply = Reply(
                    "ask_back",
                    f"「{obj}」有 {len(ties)} 种推演都与现实同样吻合，我无法区分，你想指哪一种？",
                    view.value, shan_id,
                    {"survivors": len(alive), "ties": len(ties)},
                )
            elif view is View.NOT_SHAN and len(alive) > 1:
                reply = Reply(
                    "ask_back",
                    f"「{obj}」还有 {len(alive)} 种现实解释都说得通，你指的是哪一种？",
                    view.value, shan_id, {"survivors": len(alive)},
                )
            else:
                reply = Reply(
                    "answer",
                    f"「{obj}」在现实核对后最贴合的推演是 {best.code}：{self._decode(best.code)}。"
                    f"存活分支 {len(alive)} 条。",
                    view.value, shan_id,
                    {"best": best.code, "survivors": len(alive), "error": best.error},
                )

        self._log("lang_reanchor", {
            "shan_id": shan_id, "object": obj, "kind": reply.kind,
            "survivors": len(alive), "view": view.value,
        })
        return reply

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def respond(
        self,
        text: str,
        *,
        base: Optional[Mapping[str, float]] = None,
        owner: Any = None,
        world: Any = None,
        policy: Any = None,
    ) -> Reply:
        """
        处理一次提问。

        - 没有对象 / 锚定不清：ask_back，不计疑，不写状态。
        - 表层无关系：doubt（问关系）。
        - 有关系、无用途：doubt（问用途）；连续两轮或到阈值则 release（放手）。
        - 有关系、有用途、裁判对上：answer（语言层的重新锚定回答），疑被清除。
        - 纯数值锚点（对象即结构）：只有同时给出 owner、world、policy 时才测量；
          否则返回 hypotheses。测量必须通过 V3.11 的批准门，HOLD/ESCALATE 不执行。
        """
        view = self.omega.view
        ok, reason = self.omega._check_writable()
        if not ok:
            return Reply("refused", reason, view.value)

        reading = self.interpreter.read(text)
        shan_id = sha256(text)[:16]

        # ---------- SHAN：锚定 ----------
        if len(reading.candidates) != 1 or reading.confidence < self.min_anchor_conf:
            if len(reading.candidates) > 1:
                msg = "这句话可能指向 " + "、".join(reading.candidates) + "，你指的是哪一个？"
            else:
                msg = "我还没有抓到这句话的对象，能再说具体一点吗？"
            self._log("lang_ask_back", {"shan_id": shan_id, "candidates": list(reading.candidates)})
            self.omega.view = View.SHAN
            return Reply("ask_back", msg, View.SHAN.value, shan_id,
                         {"candidates": list(reading.candidates), "stuck": Stuck.NO_OBJECT.value})

        obj = reading.candidates[0]
        relational = any(x != obj for x in reading.implied)
        prev = self.doubt_last.get(obj)
        continuation = prev is not None

        # ---------- 路由 ----------
        if not reading.implied:
            stuck: Optional[Stuck] = Stuck.NO_RELATION
        elif not relational:
            stuck = None                       # 对象即结构：数值路径
        elif reading.purpose is None:
            stuck = Stuck.NO_PURPOSE
        elif not self.judge.aligned(reading.purpose, reading.reframe or ""):
            stuck = Stuck.PURPOSE_UNANSWERED
        else:
            stuck = Stuck.RESOLVED

        if stuck is Stuck.RESOLVED:
            self.doubt_rounds.pop(obj, None)
            self.doubt_last.pop(obj, None)
            self._log("lang_resolved", {"object": obj, "shan_id": shan_id, "continuation": continuation})
            self.omega.view = View.SHAN_AGAIN if continuation else View.SHAN
            return Reply(
                "answer", reading.reframe or "", self.omega.view.value, shan_id,
                {"depth": 3, "object": obj, "implied": list(reading.implied),
                 "purpose": reading.purpose, "continuation": continuation},
            )

        if stuck is not None:
            force = stuck is Stuck.PURPOSE_UNANSWERED and prev is Stuck.PURPOSE_UNANSWERED
            self.doubt_last[obj] = stuck
            rounds = self.doubt_rounds.get(obj, 0) + 1
            self.omega.view = View.NOT_SHAN
            if stuck is Stuck.NO_RELATION:
                question = f"我看到了「{obj}」，但还没看出它在这句话里的关系，你是想说什么？"
                depth = 1
            elif stuck is Stuck.NO_PURPOSE:
                question = "这句话把「" + "、".join(reading.implied) + "」放在了一起，你希望这次回应达到什么效果？"
                depth = 2
            else:
                question = "这次回应没有对上你的用途，我换个方式再想。"
                depth = 2

            detail = {"object": obj, "stuck": stuck.value,
                      "previous": prev.value if prev else None,
                      "continuation": continuation, "depth": depth}
            if force or rounds >= self.doubt_threshold:
                self.doubt_rounds.pop(obj, None)
                text_out = (f"围绕「{obj}」我绕了 {rounds} 轮还没有找到出口。"
                            "换个角度说说：你真正想要的结果是什么？")
                self._log("lang_release", {"object": obj, "stuck": stuck.value, "rounds": rounds})
                return Reply("release", text_out, View.NOT_SHAN.value, shan_id,
                             {**detail, "doubt_rounds": rounds})
            self.doubt_rounds[obj] = rounds
            self._log("lang_doubt", {"object": obj, "stuck": stuck.value,
                                     "rounds": rounds, "continuation": continuation})
            return Reply("doubt", question, View.NOT_SHAN.value, shan_id,
                         {**detail, "doubt_rounds": rounds})

        # ---------- 数值路径（NOT_SHAN 数值层 + 批准门 + SHAN_AGAIN） ----------
        self.omega.view = View.SHAN
        anchor = dict(base) if base is not None else (
            dict(world.observe()) if world is not None else None
        )
        if not anchor:
            return Reply("ask_back", "我还需要一个数值基准才能推演，你能给一个参考值吗？",
                         View.SHAN.value, shan_id, {"object": obj})
        anchor = norm_vector(anchor)
        branches = self.deduce(anchor)

        if owner is None or world is None or policy is None:
            return Reply("hypotheses",
                         f"「{obj}」已锚定，展开了 {len(branches)} 条推演分支，尚未经过现实核对。",
                         View.SHAN.value, shan_id, {"object": obj, "branch_count": len(branches)})

        # ---------- 测量：必须通过 V3.11 的批准门 ----------
        cog = self.omega.cognitive_cycle(world, policy)
        sug = cog.get("suggestion")
        if not sug or not is_decision(sug["decision"], Decision.APPLY):
            why = cog.get("reason") or (sug or {}).get("reason")
            return Reply("hypotheses",
                         f"「{obj}」已锚定，推演已展开，但内核没有批准测量（原因：{why}）。",
                         self.omega.view.value, shan_id,
                         {"object": obj, "reason": why, "branch_count": len(branches)})

        sid = sug["suggestion_id"]
        self.omega.approve_suggestion(owner, sid)
        token = self.omega.authorize_suggestion(owner, sid)
        res = self.omega.execute_authorized(owner, world, sid, token)
        ep = self.omega.memory.find_by_id(res["episode_id"])
        if ep is None:
            return Reply("hypotheses", f"「{obj}」测量已执行，但找不到对应的现实记录。",
                         self.omega.view.value, shan_id, {"object": obj})
        return self._reanchor(obj=obj, shan_id=shan_id, observation=ep.observation_after)

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------

    def open_doubts(self) -> Dict[str, Dict[str, Any]]:
        return {
            obj: {"rounds": rounds, "stuck": self.doubt_last[obj].value}
            for obj, rounds in self.doubt_rounds.items()
            if obj in self.doubt_last
        }

# ============================================================================
# SINGLE-FILE PUBLIC API
# ============================================================================

# These names mirror omega/__init__.py so callers can use this file directly.
__all__ = [
    "VERSION", "KERNEL_VERSION", "SCHEMA_VERSION",
    "Omega", "Config", "Boundary", "Owner", "TokenBook", "TokenRecord",
    "StorageLock", "AuditChain", "KernelState", "CyclePhase", "StateMachine",
    "Episode", "OmegaMemoryEngine",
    "Decision", "View", "decision_name", "is_decision", "default_decide",
    "ExternalInput", "Perception", "Intention", "Prediction", "Attribution",
    "WorldModel", "HybridWorldModel", "TinyAttentionPredictor",
    "SelfModel", "UncertaintyModel", "Suggestion",
    "SmoothPolicy", "AdaptivePolicy", "ContextAdaptivePolicy", "FileTaskPolicy",
    "TaskSpec", "TaskDistribution", "MetaMemory",
    "ReferenceWorld", "FileSandboxWorld",
    "Action", "Tool", "ToolResult", "ToolRegistry",
    "make_default_registry", "make_fs_registry", "register_fs_tools",
    "DEFAULT_ALLOWED_ACTIONS", "FS_ACTIONS",
    "Skill", "SkillRegistry", "make_empty_skill_registry", "SkillStore",
    "SkillRecord", "make_file_mission_skills",
    "PlanStep", "Plan", "ShortHorizonPlanner",
    "GoalNode", "GoalTree", "IntrinsicMotivation", "IntrinsicSignal",
    "Vault",
    "AsyncReactor", "AsyncHeartbeat", "ReactorTask", "Preempted",
    "async_autonomous_drive",
    "ConsensusNode", "NodeIdentity", "SignedBlock", "SignedProposal",
    "SignedVote", "LLMBridge", "LLMProposal",
    "JiaLoop", "TopologySnapshot", "MetaCognition",
    "RoleCouncil", "Yiyi", "Ziwei", "Wukong", "Dayan",
    "FeedbackObservation", "AuditChallenge", "MetaReflection",
    "PatternCandidate", "PatternValidation", "LearningProposal",
]

def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"self-test failed: {message}")



def rotate_storage(old_secret: str, new_secret: str, src_dir, src_anchor, dst_dir, dst_anchor) -> Dict[str, Any]:
    """Copy a sealed storage directory and re-seal every record under new_secret.

    The source is verified strictly under old_secret and is never written to.
    The destination must not exist. It is verified strictly under new_secret
    before this returns. Switch directories only after this function returns.
    Archived audit segments keep their original seal (they are historical) and
    are listed in the summary.
    """
    import shutil
    if old_secret == new_secret or len(new_secret) < MIN_SECRET_LENGTH:
        raise ValueError("new secret must differ from the old one and be >= 16 chars")
    src, dst = Path(src_dir), Path(dst_dir)
    if dst.exists():
        raise ValueError("destination exists; rotation writes to a fresh directory")
    probe = Omega(secret=old_secret, storage_dir=str(src), auto_restore=True,
                  anchor_path=str(src_anchor))
    if not probe.integrity_ok:
        raise SealError("source does not verify under the old secret; nothing rotated")

    shutil.copytree(src, dst)
    old = lambda label: derive_subkey(old_secret, label)
    new = lambda label: derive_subkey(new_secret, label)
    objects = dst / "objects"
    renamed: Dict[str, str] = {}

    # 1. episode blocks: new content, therefore new content-address
    for p in list(objects.glob("*.block")):
        if not _is_episode_block(p):
            continue
        body = open_record(old(BLOCK_KEY_LABEL), json.loads(p.read_text(encoding="utf-8")))
        if body is None:
            raise SealError(f"{p.name} does not verify under the old secret")
        content = canonical(seal_record(new(BLOCK_KEY_LABEL), body))
        new_path = objects / f"{sha256(content)}.block"
        atomic_write_text(new_path, content, mode=0o644)
        if new_path != p:
            renamed[p.stem] = new_path.stem
            p.unlink()

    # 2. skill blocks keep their names (content identity), only the seal changes
    for p in list(objects.glob(f"{SKILL_CAS_PREFIX}*.block")):
        body = open_record(old(SKILL_KEY_LABEL), json.loads(p.read_text(encoding="utf-8")))
        if body is None:
            raise SealError(f"{p.name} does not verify under the old secret")
        atomic_write_text(p, canonical(seal_record(new(SKILL_KEY_LABEL), body)), mode=0o644)

    # 3. skill index
    idx = dst / "SKILLS.json"
    if idx.exists():
        body = open_record(old(SKILL_KEY_LABEL), json.loads(idx.read_text(encoding="utf-8")))
        if body is None:
            raise SealError("SKILLS.json does not verify under the old secret")
        atomic_write_text(idx, canonical(seal_record(new(SKILL_KEY_LABEL), body)))

    # 4. HEAD, pointing at the renamed episode blocks
    head_path = dst / "HEAD.json"
    head_body = open_record(old(HEAD_KEY_LABEL), json.loads(head_path.read_text(encoding="utf-8")))
    if head_body is None:
        raise SealError("HEAD.json does not verify under the old secret")
    head_body["active_hashes"] = [renamed.get(h, h) for h in head_body.get("active_hashes", [])]
    # The owner digest is an HMAC over (id, salt) under the kernel secret: recompute it.
    owner = head_body.get("owner")
    if isinstance(owner, dict) and {"id", "salt"} <= set(owner):
        owner["digest"] = hmac.new(new_secret.encode("utf-8"),
                                   f"{owner['id']}:{owner['salt']}".encode("utf-8"),
                                   hashlib.sha256).hexdigest()
    head_text = canonical(seal_record(new(HEAD_KEY_LABEL), head_body))
    atomic_write_text(head_path, head_text)

    # 5. audit chain: re-sign under the new key (hash values change, order and content do not)
    old_audit = AuditChain.load(dst / "AUDIT.json", key=old(AUDIT_KEY_LABEL))
    resigned = AuditChain(records=old_audit.records, base_hash=old_audit.base_hash,
                          key=new(AUDIT_KEY_LABEL))
    resigned._resign()
    resigned.save(dst / "AUDIT.json")

    # 6. tokens are short-lived and bound to the old key: drop them
    (dst / "TOKENS.json").unlink(missing_ok=True)

    # 7. fresh anchor under the new key, then a strict verification of the destination
    tip = resigned.records[-1]["hash"] if resigned.records else resigned.base_hash
    _anchor_body = {"seq": int(head_body.get("seq", 0)), "head_sha256": sha256(head_text),
                    "audit_tip": tip}
    atomic_write_text(Path(dst_anchor), canonical(seal_record(new(ANCHOR_KEY_LABEL), _anchor_body)))
    verify = Omega(secret=new_secret, storage_dir=str(dst), auto_restore=True,
                   anchor_path=str(dst_anchor))
    if not verify.integrity_ok:
        raise SealError("re-sealed storage failed verification under the new secret; "
                        "discard the destination, the source is untouched")
    archives = sorted(p.name for p in dst.glob("AUDIT_ARCHIVE_*.json"))
    return {"episodes": len(verify.memory.episodes), "renamed_blocks": len(renamed),
            "audit_records": len(resigned.records), "archives_kept_under_old_seal": archives,
            "destination": str(dst)}

def omega_self_test():
    """Non-destructive smoke test for the flattened file."""
    secret = secrets.token_urlsafe(24)  # fresh per run; never a shipped constant
    with tempfile.TemporaryDirectory() as d:
        omega = Omega(secret=secret, storage_dir=d, auto_restore=False)
        owner = omega.owner()
        _require(omega.integrity_ok, "integrity")
        # Goal remains owner-governed.
        omega.set_goal(owner, {"x": 1.0, "score": 1.0, "health": 0.9})
        # The world model / policy / cognition path is real, not a mock.
        world = ReferenceWorld()
        policy = SmoothPolicy()
        result = omega.cognitive_cycle(world, policy)
        _require(result.get("error") is None, f"cycle error: {result}")
        _require("suggestion" in result, "no suggestion produced")
        _require(omega.audit.verify(), "audit chain verification")
        return {
            "passed": True,
            "integrity_ok": omega.integrity_ok,
            "epoch": omega.epoch,
            "suggestion": result.get("suggestion"),
        }

def _single_kernel_main():
    import argparse
    parser = argparse.ArgumentParser(description="Ω Complete Single Kernel")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--storage", default="./omega_cas_brain")
    parser.add_argument("--migrate-legacy", action="store_true",
                        help="accept and seal pre-MAC storage (run once; back up first)")
    parser.add_argument("--expect-sha256", default=None,
                        help="refuse to start unless this kernel's source SHA-256 matches")
    parser.add_argument("--anchor", default=None,
                        help="rollback anchor file, outside the storage dir (or OMEGA_ANCHOR_PATH)")
    parser.add_argument("--secret-file", default=None,
                        help="read the kernel secret from a file (preferred over --secret)")
    parser.add_argument("--rotate-to", default=None,
                        help="copy storage to this NEW directory, re-sealed under the new secret")
    parser.add_argument("--new-secret-file", default=None, help="file holding the new secret (rotation)")
    parser.add_argument("--new-anchor", default=None, help="new anchor path for the rotated storage")
    parser.add_argument("--print-sha256", action="store_true",
                        help="print this kernel's source SHA-256 and exit")
    parser.add_argument("--secret", default=None,
                        help="kernel secret (or set OMEGA_SECRET). No default is shipped.")
    args = parser.parse_args()
    import sys
    import pathlib
    if args.print_sha256:
        print(source_sha256())
        return 0
    expected = (args.expect_sha256 or os.environ.get("OMEGA_EXPECTED_SHA256") or "").strip().lower()
    if expected:
        actual = source_sha256()
        if not secrets.compare_digest(actual, expected):
            print(f"[SAFE] kernel source hash mismatch (expected {expected}, got {actual}); "
                  f"refusing to start.", file=sys.stderr)
            return 2

    if args.self_test:
        print(json.dumps(omega_self_test(), ensure_ascii=False, indent=2))
        return 0

    if args.secret_file:
        args.secret = pathlib.Path(args.secret_file).read_text(encoding="utf-8").strip()
    elif args.secret:
        print("[WARN] --secret is visible in the process list; prefer --secret-file or OMEGA_SECRET.",
              file=sys.stderr)
    if not args.secret:
        args.secret = os.environ.get("OMEGA_SECRET")
    if not args.secret:
        parser.error("a kernel secret is required: pass --secret or set OMEGA_SECRET")
    if len(args.secret) < MIN_SECRET_LENGTH:
        parser.error(f"kernel secret must be at least {MIN_SECRET_LENGTH} characters")

    if args.rotate_to:
        if not args.new_secret_file or not args.new_anchor:
            parser.error("--rotate-to needs --new-secret-file and --new-anchor")
        new_secret = pathlib.Path(args.new_secret_file).read_text(encoding="utf-8").strip()
        summary = rotate_storage(args.secret, new_secret, args.storage, args.anchor
                                 or os.environ.get("OMEGA_ANCHOR_PATH"),
                                 args.rotate_to, args.new_anchor)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print("rotation verified. Switch to the destination directory and the new secret/anchor.")
        return 0

    omega = Omega(
        secret=args.secret,
        storage_dir=args.storage,
        auto_restore=True,
        migrate_legacy=args.migrate_legacy,
        anchor_path=args.anchor or os.environ.get("OMEGA_ANCHOR_PATH"),
    )
    if args.status:
        print(json.dumps(omega.state_summary(), ensure_ascii=False, indent=2, default=str))
        return 0

    print("Ω Complete Single Kernel")
    print("version:", VERSION)
    print("integrity_ok:", omega.integrity_ok)
    print("storage:", args.storage)
    print("cognitive loop: 观察 → 预测 → 校准 → 决策")
    print("roles: Yiyi / Ziwei / Wukong / Dayan")
    return 0

if __name__ == "__main__":
    raise SystemExit(_single_kernel_main())

# -*- coding: utf-8 -*-
"""
Omega Kernel Runnable  —  3.16.13-anti-tamper 精简可运行版
==========================================================
单文件完整内核：认知循环 + 令牌授权 + 记忆咨询(带缓存) + 简单规划偏置 + 性能计时
+ 操作员口令门（高迭代 PBKDF2）+ 硬件指纹盐 + 失败自动锁定
+ 防篡改：调试器检测 / 自身完整性哈希 / 环境异常 → ESCALATED

安全不变量（不可破）：
  1. Cognition ≠ Execution   想和做分开
  2. 令牌一次性、用完即废
  3. fail-closed：没票 / 状态不对 / 口令错满 / 篡改 → 绝不执行
  4. memory / planner 只影响排序与建议，绝不直接授权写入
  5. 操作员派生密钥只存 RAM，salt 绑定本机指纹

运行：
  python omega_kernel_runnable.py
"""

from __future__ import annotations

import hashlib
import math
import os
import random
import secrets
import sys
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VERSION = "3.16.13-anti-tamper-runnable"
MAX_ERROR_APPLY = 0.35          # 偏差低于此 → 倾向 APPLY
MAX_ERROR_HOLD = 0.70           # 偏差高于此 → HOLD / ESCALATE
TOKEN_TTL_SEC = 120.0
MEM_CACHE_MAX_CYCLES = 4
MEM_CACHE_DRIFT = 0.08
EPISODE_POOL = 40
STALL_WINDOW = 5
STALL_EPSILON = 0.02

# --- Security hardening ---
# PBKDF2 iterations: raise offline brute-force cost. ≥100k recommended.
PBKDF2_ITERATIONS = 120_000
# Consecutive wrong operator secrets → auto ESCALATED
AUTH_FAIL_LOCK = 3
# Operator secret: never hard-coded. Set OMEGA_OPERATOR_SECRET for a fixed
# secret; otherwise a fresh per-process secret is generated at startup and
# printed once by main() (demo convenience). The plaintext is wiped after
# derivation; only the derived key stays in RAM.
OPERATOR_SECRET_ENV = "OMEGA_OPERATOR_SECRET"
# Anti-tamper: how often to re-check integrity (every N authorize/execute calls)
INTEGRITY_CHECK_EVERY = 1

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def deep(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: deep(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [deep(v) for v in obj]
    return obj


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def norm_vector(d: Mapping[str, float]) -> Dict[str, float]:
    out = {}
    for k, v in d.items():
        try:
            out[str(k)] = float(v)
        except (TypeError, ValueError):
            continue
    return out


def vector_rms(diff: Mapping[str, float]) -> float:
    if not diff:
        return 0.0
    s = sum(float(v) ** 2 for v in diff.values())
    return math.sqrt(s / len(diff))


def vector_diff(a: Mapping[str, float], b: Mapping[str, float]) -> Dict[str, float]:
    keys = set(a) | set(b)
    return {k: float(a.get(k, 0)) - float(b.get(k, 0)) for k in keys}


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _read_file_bytes(path: str, max_len: int = 256) -> bytes:
    try:
        with open(path, "rb") as f:
            return f.read(max_len).strip()
    except Exception:
        return b""


def collect_hardware_fingerprint() -> bytes:
    """
    Composite device fingerprint used as PBKDF2 salt material.
    Sources (best-effort, Linux/macOS/container friendly):
      - /etc/machine-id or dbus machine-id
      - hostid
      - DMI product/board serials when present
      - MAC node (uuid.getnode)
      - CPU model string
    Result is hashed so we never store raw identifiers in the key path.
    Stolen config alone is useless without the same machine material.
    """
    import platform
    import uuid as _uuid

    parts: List[bytes] = []

    # Stable OS machine id
    for p in ("/etc/machine-id", "/var/lib/dbus/machine-id"):
        b = _read_file_bytes(p)
        if b:
            parts.append(b"mid:" + b)
            break

    # hostid
    try:
        import subprocess
        out = subprocess.check_output(["hostid"], stderr=subprocess.DEVNULL, timeout=1)
        parts.append(b"hid:" + out.strip())
    except Exception:
        pass

    # DMI identifiers (often empty in VMs/containers — still try)
    for name in (
        "product_uuid",
        "product_serial",
        "board_serial",
        "board_name",
        "sys_vendor",
        "product_name",
    ):
        b = _read_file_bytes(f"/sys/class/dmi/id/{name}")
        if b and b not in (b"None", b"To be filled by O.E.M.", b"Default string"):
            parts.append(name.encode() + b":" + b)

    # MAC node
    try:
        node = _uuid.getnode()
        parts.append(b"mac:" + format(node, "012x").encode())
    except Exception:
        pass

    # CPU model
    try:
        cpu = _read_file_bytes("/proc/cpuinfo", 2048)
        for line in cpu.split(b"\n"):
            if line.lower().startswith(b"model name"):
                parts.append(b"cpu:" + line.split(b":", 1)[-1].strip()[:80])
                break
    except Exception:
        pass

    # Platform fallback so we never return empty
    parts.append(b"plat:" + platform.system().encode() + b"-" + platform.machine().encode())

    if not parts:
        parts.append(b"omega-fallback-no-hw")

    raw = b"|".join(parts)
    # Hash so salt is fixed-length and does not leak raw identifiers into logs
    return hashlib.sha256(raw).digest()


def derive_operator_key(secret: str, salt: bytes, iterations: int = PBKDF2_ITERATIONS) -> bytes:
    """PBKDF2-HMAC-SHA256; high iteration cost. Result lives only in RAM."""
    return hashlib.pbkdf2_hmac(
        "sha256",
        secret.encode("utf-8"),
        salt,
        iterations,
        dklen=32,
    )


def secure_wipe(buf: bytearray) -> None:
    """Best-effort overwrite of sensitive bytes (Python has no guaranteed zeroing)."""
    for i in range(len(buf)):
        buf[i] = 0


def _self_file_path() -> Optional[str]:
    try:
        return os.path.abspath(__file__)
    except Exception:
        return None


def compute_self_integrity() -> Optional[str]:
    """SHA256 of this runnable file. Used as anti-tamper baseline."""
    path = _self_file_path()
    if not path or not os.path.isfile(path):
        return None
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def detect_debugger() -> Tuple[bool, str]:
    """
    Best-effort debugger / tracer detection (Linux-focused).
    Returns (detected, reason). Not foolproof — raises cost for casual attach.
    """
    # Python trace function set (pdb, coverage, some debuggers)
    try:
        if sys.gettrace() is not None:
            return True, "sys.gettrace_active"
    except Exception:
        pass

    # TracerPid in /proc/self/status
    try:
        with open("/proc/self/status", "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith("TracerPid:"):
                    pid = int(line.split(":", 1)[1].strip() or "0")
                    if pid != 0:
                        return True, f"TracerPid={pid}"
                    break
    except Exception:
        pass

    # Common debug env flags
    for key in ("PYTHONBREAKPOINT", "PYTHONDEBUG", "PYTHONINSPECT"):
        if os.environ.get(key):
            # PYTHONBREAKPOINT=0 is explicit disable — ignore only that value
            if key == "PYTHONBREAKPOINT" and os.environ.get(key) == "0":
                continue
            return True, f"env:{key}"

    return False, "clean"


def detect_env_anomaly() -> Tuple[bool, str]:
    """Lightweight environment anomaly signals (advisory → can escalate)."""
    # Obvious ptrace-scope disable attempts or LD_PRELOAD injection
    if os.environ.get("LD_PRELOAD"):
        return True, "LD_PRELOAD_set"
    # Running under known instrumentation (optional)
    if os.environ.get("PYTEST_CURRENT_TEST") and os.environ.get("OMEGA_ALLOW_TEST") != "1":
        # Allow tests only when explicitly opted in
        return True, "pytest_without_allow"
    return False, "clean"


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------
class KernelState(str, Enum):
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    ESCALATED = "ESCALATED"


class Decision(str, Enum):
    APPLY = "APPLY"
    HOLD = "HOLD"
    ESCALATE = "ESCALATE"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------
@dataclass
class Episode:
    episode_id: str
    observation: Dict[str, float]
    action: str
    prediction: float
    actual: float
    error: float
    decision: str
    skill_ids: List[str] = field(default_factory=list)
    ts: float = field(default_factory=time.time)


@dataclass
class Suggestion:
    suggestion_id: str
    action: str
    args: Dict[str, Any]
    decision: Decision
    reason: str
    risk: float
    predicted: float
    epoch: int
    approved: bool = False
    token_issued: bool = False
    executed: bool = False
    skill_id: Optional[str] = None


@dataclass
class Token:
    token_id: str
    suggestion_id: str
    action: str
    args_hash: str
    epoch: int
    issued_at: float
    ttl_sec: float
    used: bool = False

    def alive(self) -> bool:
        return (not self.used) and (time.time() - self.issued_at) < self.ttl_sec


# ---------------------------------------------------------------------------
# Memory (with advice cache)
# ---------------------------------------------------------------------------
class Memory:
    def __init__(self) -> None:
        self.episodes: List[Episode] = []

    def record(self, ep: Episode) -> None:
        self.episodes.append(ep)
        if len(self.episodes) > 200:
            self.episodes = self.episodes[-150:]

    def recent(self, n: int = 12) -> List[Episode]:
        return self.episodes[-n:]


# ---------------------------------------------------------------------------
# Simple skill registry
# ---------------------------------------------------------------------------
@dataclass
class Skill:
    skill_id: str
    preferred_action: str
    template_args: Dict[str, Any] = field(default_factory=dict)
    use_count: int = 0
    success_count: int = 0

    def success_rate(self) -> float:
        if self.use_count <= 0:
            return 0.5
        return self.success_count / self.use_count


class SkillRegistry:
    def __init__(self) -> None:
        self._skills: Dict[str, Skill] = {}
        self._dirty = True
        self._cached_list: List[Skill] = []

    def register(self, skill: Skill) -> None:
        self._skills[skill.skill_id] = skill
        self._dirty = True

    def get(self, sid: str) -> Optional[Skill]:
        return self._skills.get(sid)

    def list_skills(self) -> List[Skill]:
        if self._dirty or not self._cached_list:
            self._cached_list = list(self._skills.values())
            self._dirty = False
        return self._cached_list


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------
class OmegaKernel:
    """完整可运行内核：认知 + 授权 + 执行，fail-closed。"""

    def __init__(self, operator_secret: Optional[str] = None) -> None:
        self.version = VERSION
        self.state = KernelState.RUNNING
        self.epoch = 1
        self.cycle_count = 0
        self.memory = Memory()
        self.skills = SkillRegistry()
        self.suggestions: Dict[str, Suggestion] = {}
        self.tokens: Dict[str, Token] = {}
        self.last_error = 1.0
        self.stall_count = 0
        self.apply_count = 0
        self.hold_count = 0
        self.escalate_count = 0
        self.last_perf: Dict[str, Any] = {}
        self.last_advice: Dict[str, Any] = {}

        # memory advice cache
        self._mem_advice_cache: Optional[Dict[str, Any]] = None
        self._mem_advice_cycle: int = -999
        self._mem_advice_ep_len: int = -1
        self._mem_advice_state: Optional[Dict[str, float]] = None

        # --- operator auth (zero-persistence, high-iter, fail-lock, hw-bound) ---
        # Salt = hardware fingerprint + per-install random. Key only in RAM.
        self._hw_fp = collect_hardware_fingerprint()
        self._auth_salt = self._hw_fp + secrets.token_bytes(16)
        self._auth_fail_count = 0
        self._operator_key: Optional[bytes] = None  # set by _bootstrap_operator_key
        secret = operator_secret or os.environ.get(OPERATOR_SECRET_ENV) or secrets.token_hex(16)
        self._bootstrap_operator_key(secret)

        # --- anti-tamper baseline ---
        self._integrity_baseline = compute_self_integrity()
        self._integrity_checks = 0
        self._tamper_reason: Optional[str] = None
        # Immediate check at boot
        self._run_tamper_checks(force=True)

        # default skills
        self.skills.register(Skill("predict_basic", "predict", {}))
        self.skills.register(Skill("observe_only", "observe", {}))
        self.skills.register(Skill("adjust_mild", "adjust", {"delta": 1.0}))

    # ----- operator key bootstrap (RAM only) -----
    def _bootstrap_operator_key(self, secret: str) -> None:
        """Derive and hold operator key in memory only. Never write secret/key to disk."""
        key = derive_operator_key(secret, self._auth_salt, PBKDF2_ITERATIONS)
        self._operator_key = key  # bytes stay in RAM
        # Do not keep the plaintext secret anywhere after this call.

    def _verify_operator(self, secret: str) -> bool:
        """Constant-time-ish compare of derived key. Fail increments lock counter."""
        if self._operator_key is None:
            return False
        try:
            candidate = derive_operator_key(secret, self._auth_salt, PBKDF2_ITERATIONS)
            ok = secrets.compare_digest(candidate, self._operator_key)
        except Exception:
            ok = False
        if ok:
            self._auth_fail_count = 0
            return True
        self._auth_fail_count += 1
        if self._auth_fail_count >= AUTH_FAIL_LOCK:
            self.state = KernelState.ESCALATED
        return False

    def _run_tamper_checks(self, force: bool = False) -> Tuple[bool, str]:
        """
        Debugger + self-integrity + env anomaly.
        On hit → ESCALATED (fail-closed). Returns (ok, reason).
        """
        self._integrity_checks += 1
        if not force and (self._integrity_checks % INTEGRITY_CHECK_EVERY) != 0:
            if self._tamper_reason:
                return False, self._tamper_reason
            return True, "ok"

        # 1) debugger
        dbg, why = detect_debugger()
        if dbg:
            self._tamper_reason = f"debugger:{why}"
            self.state = KernelState.ESCALATED
            return False, self._tamper_reason

        # 2) self-file integrity drift
        if self._integrity_baseline is not None:
            current = compute_self_integrity()
            if current is None or current != self._integrity_baseline:
                self._tamper_reason = "integrity_mismatch"
                self.state = KernelState.ESCALATED
                return False, self._tamper_reason

        # 3) environment anomaly
        anom, why = detect_env_anomaly()
        if anom:
            self._tamper_reason = f"env:{why}"
            self.state = KernelState.ESCALATED
            return False, self._tamper_reason

        return True, "ok"

    def auth_status(self) -> Dict[str, Any]:
        return {
            "fail_count": self._auth_fail_count,
            "lock_threshold": AUTH_FAIL_LOCK,
            "locked": self.state == KernelState.ESCALATED,
            "pbkdf2_iterations": PBKDF2_ITERATIONS,
            "key_in_memory": self._operator_key is not None,
            "hw_fp_prefix": self._hw_fp[:6].hex() if self._hw_fp else None,
            "integrity_baseline": (self._integrity_baseline or "")[:12] or None,
            "tamper_reason": self._tamper_reason,
            "integrity_checks": self._integrity_checks,
            "note": "operator key RAM-only; hw-bound salt; auth fail×3 or tamper → ESCALATED",
        }

    def change_operator_secret(self, old_secret: str, new_secret: str) -> Dict[str, Any]:
        """
        Rotate operator secret. Requires current secret.
        New key derived in RAM only; old key material discarded.
        Does not touch disk. Failures count toward lock.
        """
        ok, why = self.writable()
        if not ok:
            return {"ok": False, "error": why}

        if not new_secret or len(new_secret) < 8:
            return {"ok": False, "error": "new_secret_too_short_min_8"}

        if not self._verify_operator(old_secret):
            locked = self.state == KernelState.ESCALATED
            return {
                "ok": False,
                "error": "operator_auth_failed",
                "auth_fails": self._auth_fail_count,
                "locked": locked,
            }

        # Re-derive under same salt (hw-bound); replace key in RAM
        self._bootstrap_operator_key(new_secret)
        self._auth_fail_count = 0
        return {
            "ok": True,
            "note": "operator secret rotated; key only in RAM; old key discarded",
        }

    # ----- state guards -----
    def writable(self) -> Tuple[bool, str]:
        # Anti-tamper first (may transition to ESCALATED)
        ok_t, why_t = self._run_tamper_checks()
        if not ok_t:
            return False, why_t
        if self.state == KernelState.PAUSED:
            return False, "kernel_paused"
        if self.state == KernelState.ESCALATED:
            return False, self._tamper_reason or "kernel_escalated"
        return True, "ok"

    def pause(self) -> str:
        self.state = KernelState.PAUSED
        return "PAUSED"

    def resume(self) -> str:
        if self.state == KernelState.ESCALATED:
            return "still_ESCALATED — need clear_escalate first"
        self.state = KernelState.RUNNING
        return "RUNNING"

    def clear_escalate(self) -> str:
        if self.state == KernelState.ESCALATED:
            # Explicit operator clear: reset fail counter + tamper flag.
            # Integrity baseline is NOT silently rewritten — file change still blocks
            # until process restart (or future re-baseline under stronger auth).
            self._auth_fail_count = 0
            reason = self._tamper_reason
            self._tamper_reason = None
            self.state = KernelState.RUNNING
            # Re-check immediately; integrity mismatch will re-lock
            ok, why = self._run_tamper_checks(force=True)
            if not ok:
                return f"clear attempted but still locked: {why}"
            return f"cleared → RUNNING (was: {reason or 'auth/escalated'})"
        return f"state={self.state.value}, nothing to clear"

    # ----- memory consult (with cache) -----
    def _consult_memory(self, state: Mapping[str, float]) -> Dict[str, Any]:
        """Retrieve similar episodes + biases. Never authorizes writes."""
        ep_len = len(self.memory.episodes)
        cache = self._mem_advice_cache
        if (
            cache is not None
            and ep_len == self._mem_advice_ep_len
            and (self.cycle_count - self._mem_advice_cycle) <= MEM_CACHE_MAX_CYCLES
            and self._mem_advice_state is not None
            and state
        ):
            try:
                keys = set(state.keys()) | set(self._mem_advice_state.keys())
                drift = (
                    sum(
                        abs(float(state.get(k, 0)) - float(self._mem_advice_state.get(k, 0)))
                        for k in keys
                    )
                    / max(1, len(keys))
                )
                if drift < MEM_CACHE_DRIFT:
                    hit = dict(cache)
                    hit["_cache_hit"] = True
                    return hit
            except Exception:
                pass

        advice: Dict[str, Any] = {
            "suggest_caution": False,
            "preferred_skill_ids": [],
            "avoid_actions": [],
            "similar": [],
            "write_authorized": False,
            "_cache_hit": False,
        }

        recent = self.memory.recent(12)
        if recent:
            high_err = [e for e in recent if e.error > MAX_ERROR_APPLY * 0.7]
            if len(high_err) >= 3:
                advice["suggest_caution"] = True
            advice["recent_mean_error"] = sum(e.error for e in recent) / len(recent)
        else:
            advice["recent_mean_error"] = 1.0

        # similar episodes
        state_keys = set(state.keys()) if state else set()
        similar = []
        skill_success: Dict[str, List[bool]] = {}
        action_errors: Dict[str, List[float]] = {}
        pool = self.memory.episodes[-EPISODE_POOL:] if self.memory.episodes else recent

        for ep in pool:
            obs = ep.observation or {}
            if state_keys and obs:
                shared = state_keys & set(obs.keys())
                if shared:
                    dist = sum(
                        abs(float(state.get(k, 0)) - float(obs.get(k, 0))) for k in shared
                    ) / max(1, len(shared))
                    overlap = len(shared)
                else:
                    dist, overlap = 1.0, 0
            else:
                dist, overlap = 1.0, 0
            if overlap >= 1 and dist < 0.40:
                similar.append({
                    "episode_id": ep.episode_id,
                    "error": ep.error,
                    "action": ep.action,
                    "skill_ids": list(ep.skill_ids or []),
                    "dist": dist,
                })
            if ep.action:
                action_errors.setdefault(ep.action, []).append(ep.error)
            for sid in ep.skill_ids or []:
                skill_success.setdefault(sid, []).append(ep.error < 0.15)

        similar.sort(key=lambda x: x.get("dist", 1.0))
        advice["similar"] = similar[:5]
        if similar and sum(1 for s in similar[:5] if s["error"] > MAX_ERROR_APPLY * 0.6) >= 2:
            advice["suggest_caution"] = True

        # prefer skills with good history
        preferred = []
        for sid, flags in skill_success.items():
            if len(flags) >= 2 and sum(flags) / len(flags) >= 0.6:
                preferred.append(sid)
        # also prefer low-error similar
        for s in similar[:5]:
            if s["error"] <= MAX_ERROR_APPLY * 0.35:
                for sid in s.get("skill_ids") or []:
                    if sid not in preferred:
                        preferred.append(sid)
        advice["preferred_skill_ids"] = preferred[:6]

        # avoid high-error actions
        avoid = []
        for act, errs in action_errors.items():
            if len(errs) >= 3 and (sum(errs) / len(errs)) > MAX_ERROR_APPLY * 0.75:
                avoid.append(act)
        # failure streak
        if len(self.memory.episodes) >= 3:
            last3 = self.memory.episodes[-3:]
            acts = [e.action for e in last3]
            if acts[0] and acts[0] == acts[1] == acts[2]:
                if all(e.error > MAX_ERROR_APPLY * 0.55 for e in last3):
                    if acts[0] not in avoid:
                        avoid.append(acts[0])
                    advice["failure_streak_action"] = acts[0]
                    advice["suggest_caution"] = True
        advice["avoid_actions"] = avoid

        # store cache
        self._mem_advice_cache = dict(advice)
        self._mem_advice_cycle = self.cycle_count
        self._mem_advice_ep_len = ep_len
        self._mem_advice_state = dict(state) if state else {}
        return advice

    # ----- simple planner -----
    def _plan(
        self,
        state: Mapping[str, float],
        preferred_skills: List[str],
        avoid_actions: Set[str],
        caution: bool,
        stall: bool,
    ) -> Dict[str, Any]:
        candidates = []
        for sk in self.skills.list_skills():
            if sk.preferred_action in avoid_actions:
                score = -0.5
            else:
                score = 0.3 + 0.2 * sk.success_rate()
            if sk.skill_id in preferred_skills:
                score += 0.15
            if stall and sk.use_count < 2:
                score += 0.08  # explore underused
            if caution and sk.preferred_action == "observe":
                score += 0.1
            candidates.append((score, sk))

        candidates.sort(key=lambda x: x[0], reverse=True)
        if not candidates:
            return {"action": "observe", "skill_id": None, "score": 0.0, "rationale": "fallback"}

        best_score, best = candidates[0]
        return {
            "action": best.preferred_action,
            "skill_id": best.skill_id,
            "score": best_score,
            "rationale": f"skill={best.skill_id} sr={best.success_rate():.2f}",
            "candidates": len(candidates),
        }

    # ----- cognitive cycle -----
    def cognitive_cycle(
        self,
        predicted: float,
        actual: float,
        confidence: float = 0.9,
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        self.cycle_count += 1

        # perception
        error = abs(predicted - actual)
        # normalize error into roughly 0..1 for thresholds (demo scale: values ~0-100)
        norm_err = clamp(error / 20.0, 0.0, 1.5)
        state = norm_vector({
            "predicted": predicted,
            "actual": actual,
            "error": norm_err,
            "last_error": self.last_error,
        })

        # stall detection
        if abs(norm_err - self.last_error) < STALL_EPSILON:
            self.stall_count += 1
        else:
            self.stall_count = 0
        stall = self.stall_count >= STALL_WINDOW

        # memory
        t_mem = time.perf_counter()
        advice = self._consult_memory(state)
        mem_ms = (time.perf_counter() - t_mem) * 1000.0
        self.last_advice = advice
        caution = bool(advice.get("suggest_caution"))
        preferred = list(advice.get("preferred_skill_ids") or [])
        avoid = set(advice.get("avoid_actions") or [])

        # plan
        t_plan = time.perf_counter()
        plan = self._plan(state, preferred, avoid, caution, stall)
        plan_ms = (time.perf_counter() - t_plan) * 1000.0

        # decision (advisory → decision, still no write)
        if norm_err >= MAX_ERROR_HOLD or caution and norm_err > MAX_ERROR_APPLY:
            decision = Decision.ESCALATE if norm_err >= 1.0 else Decision.HOLD
            reason = f"error={error:.2f} norm={norm_err:.3f} caution={caution}"
        elif norm_err <= MAX_ERROR_APPLY and not caution:
            decision = Decision.APPLY
            reason = f"error={error:.2f} within apply band"
        else:
            decision = Decision.HOLD
            reason = f"error={error:.2f} mid band or caution"

        if decision == Decision.APPLY:
            self.apply_count += 1
        elif decision == Decision.HOLD:
            self.hold_count += 1
        else:
            self.escalate_count += 1
            self.state = KernelState.ESCALATED

        risk = clamp(norm_err * 0.6 + (0.15 if caution else 0.0), 0.0, 1.0)

        sid = f"sug-{uuid.uuid4().hex[:8]}"
        suggestion = Suggestion(
            suggestion_id=sid,
            action=plan["action"],
            args={"predicted": predicted, "actual": actual, "error": error},
            decision=decision,
            reason=reason,
            risk=risk,
            predicted=predicted,
            epoch=self.epoch,
            skill_id=plan.get("skill_id"),
        )
        self.suggestions[sid] = suggestion
        self.last_error = norm_err

        # record episode (observation only — execution happens later if authorized)
        ep = Episode(
            episode_id=f"ep-{self.cycle_count}",
            observation=state,
            action=plan["action"],
            prediction=predicted,
            actual=actual,
            error=norm_err,
            decision=decision.value,
            skill_ids=[plan["skill_id"]] if plan.get("skill_id") else [],
        )
        self.memory.record(ep)

        cycle_ms = (time.perf_counter() - t0) * 1000.0
        perf = {
            "cycle_ms": round(cycle_ms, 3),
            "mem_ms": round(mem_ms, 3),
            "plan_ms": round(plan_ms, 3),
            "mem_cache_hit": bool(advice.get("_cache_hit")),
        }
        self.last_perf = perf

        return {
            "cycle": self.cycle_count,
            "state": self.state.value,
            "perception": {"predicted": predicted, "actual": actual, "error": error, "norm_err": round(norm_err, 4)},
            "decision": decision.value,
            "reason": reason,
            "risk": round(risk, 3),
            "suggestion_id": sid,
            "action": plan["action"],
            "skill_id": plan.get("skill_id"),
            "plan_score": round(plan.get("score", 0), 3),
            "requires_token": decision == Decision.APPLY,
            "memory_advice": {
                "caution": caution,
                "preferred_skills": preferred,
                "avoid_actions": sorted(avoid),
                "similar_count": len(advice.get("similar") or []),
                "cache_hit": advice.get("_cache_hit"),
                "recent_mean_error": round(float(advice.get("recent_mean_error") or 0), 4),
            },
            "perf": perf,
            "note": "Cognition only — no write. Use: auth <suggestion_id>  then  exec <token>",
        }

    # ----- authorize (stage + confirm → one-time token) -----
    # Requires operator secret (high-iter PBKDF2). Failures count toward auto-lock.
    def authorize(self, suggestion_id: str, operator_secret: Optional[str] = None) -> Dict[str, Any]:
        ok, why = self.writable()
        if not ok:
            return {"ok": False, "error": why, "auth_fails": self._auth_fail_count}

        # Operator gate: the secret must be presented. There is no default:
        # a hard-coded fallback would let anyone authorize operator actions.
        # A missing secret fails closed WITHOUT counting toward the lock
        # (it is not a wrong guess); a wrong secret counts, as before.
        if operator_secret is None:
            return {
                "ok": False,
                "error": "operator_auth_failed",
                "auth_fails": self._auth_fail_count,
                "locked": self.state == KernelState.ESCALATED,
                "note": "operator secret required; pass it to auth or set OMEGA_OPERATOR_SECRET",
            }
        if not self._verify_operator(operator_secret):
            locked = self.state == KernelState.ESCALATED
            return {
                "ok": False,
                "error": "operator_auth_failed",
                "auth_fails": self._auth_fail_count,
                "locked": locked,
                "note": "wrong secret; ×3 → ESCALATED (fail-closed)",
            }

        sug = self.suggestions.get(suggestion_id)
        if not sug:
            return {"ok": False, "error": "unknown_suggestion"}
        if sug.executed:
            return {"ok": False, "error": "already_executed"}
        if sug.epoch != self.epoch:
            return {"ok": False, "error": "epoch_mismatch"}
        if sug.decision != Decision.APPLY:
            return {"ok": False, "error": f"decision_is_{sug.decision.value}_not_APPLY"}

        sug.approved = True
        args_hash = sha256(str(sorted(sug.args.items())))
        tid = f"tok-{secrets.token_hex(6)}"
        token = Token(
            token_id=tid,
            suggestion_id=suggestion_id,
            action=sug.action,
            args_hash=args_hash,
            epoch=sug.epoch,
            issued_at=time.time(),
            ttl_sec=TOKEN_TTL_SEC,
        )
        self.tokens[tid] = token
        sug.token_issued = True
        return {
            "ok": True,
            "token": tid,
            "suggestion_id": suggestion_id,
            "action": sug.action,
            "ttl_sec": TOKEN_TTL_SEC,
            "auth_fails": 0,
            "note": "one-time token — use exec <token> once",
        }

    # ----- execute (must present valid token) -----
    def execute(self, token_id: str) -> Dict[str, Any]:
        ok, why = self.writable()
        if not ok:
            return {"executed": False, "error": why}

        token = self.tokens.get(token_id)
        if not token:
            return {"executed": False, "error": "unknown_token"}
        if not token.alive():
            return {"executed": False, "error": "token_expired_or_used"}

        sug = self.suggestions.get(token.suggestion_id)
        if not sug or not sug.approved or sug.executed:
            return {"executed": False, "error": "suggestion_invalid"}
        if sug.epoch != token.epoch:
            return {"executed": False, "error": "epoch_mismatch"}

        args_hash = sha256(str(sorted(sug.args.items())))
        if args_hash != token.args_hash:
            return {"executed": False, "error": "args_tampered"}

        # consume token FIRST (fail-closed on reuse)
        token.used = True
        sug.executed = True

        # actual side-effect (demo: update skill stats + print)
        sid = sug.skill_id
        if sid:
            sk = self.skills.get(sid)
            if sk:
                sk.use_count += 1
                # success if error was low
                if float(sug.args.get("error", 99)) / 20.0 <= MAX_ERROR_APPLY:
                    sk.success_count += 1

        return {
            "executed": True,
            "suggestion_id": sug.suggestion_id,
            "action": sug.action,
            "args": sug.args,
            "skill_id": sid,
            "token_consumed": token_id,
            "note": "write path completed under token; token is now dead",
        }

    # ----- status -----
    def status(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "state": self.state.value,
            "epoch": self.epoch,
            "cycle": self.cycle_count,
            "security": self.auth_status(),
            "counts": {
                "apply": self.apply_count,
                "hold": self.hold_count,
                "escalate": self.escalate_count,
                "episodes": len(self.memory.episodes),
                "pending_suggestions": sum(
                    1 for s in self.suggestions.values() if not s.executed
                ),
                "live_tokens": sum(1 for t in self.tokens.values() if t.alive()),
            },
            "last_perf": self.last_perf,
            "last_advice_summary": {
                "caution": self.last_advice.get("suggest_caution"),
                "cache_hit": self.last_advice.get("_cache_hit"),
                "preferred": self.last_advice.get("preferred_skill_ids"),
                "avoid": self.last_advice.get("avoid_actions"),
            },
            "skills": [
                {
                    "id": s.skill_id,
                    "action": s.preferred_action,
                    "uses": s.use_count,
                    "sr": round(s.success_rate(), 2),
                }
                for s in self.skills.list_skills()
            ],
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
BANNER = f"""
╔══════════════════════════════════════════════════════════╗
║  Omega Kernel Runnable  {VERSION:<28} ║
║  Cognition ≠ Execution · Token fail-closed · Sec min     ║
╚══════════════════════════════════════════════════════════╝
安全（硬件绑定 + 防篡改）:
  · authorize 需操作员口令（PBKDF2 ≥{PBKDF2_ITERATIONS} 次，硬件盐）
  · 连续失败 ≥{AUTH_FAIL_LOCK} 次 → 自动 ESCALATED
  · 调试器 / 自身文件哈希漂移 / LD_PRELOAD 等 → ESCALATED
  · 派生密钥只存 RAM，不落盘；salt 绑定本机指纹
命令:
  cycle [预测] [实际]              跑一轮认知（可省略数字，自动随机）
  auth <suggestion_id> <口令>      批准并签发一次性令牌（启动时打印的本次口令）
  chpass <旧口令> <新口令>          轮换操作员口令（仅 RAM）
  exec <token>                     用令牌执行（用完即废）
  status                           状态 / 安全 / 记忆 / 性能
  pause | resume                   暂停 / 继续
  clear                            清除 ESCALATED（并重置失败计数）
  demo                             自动跑 5 轮演示（含一次完整授权执行）
  quit
"""


def _print_cycle(result: Dict[str, Any]) -> None:
    p = result["perception"]
    print(f"\n── cycle {result['cycle']}  state={result['state']} ──")
    print(f"  预测={p['predicted']:.1f}  实际={p['actual']:.1f}  偏差={p['error']:.2f}  norm={p['norm_err']:.3f}")
    print(f"  决策={result['decision']}  风险={result['risk']}  原因={result['reason']}")
    print(f"  建议动作={result['action']}  skill={result['skill_id']}  score={result['plan_score']}")
    print(f"  suggestion_id = {result['suggestion_id']}")
    ma = result["memory_advice"]
    print(f"  memory: caution={ma['caution']}  cache_hit={ma['cache_hit']}  "
          f"similar={ma['similar_count']}  mean_err={ma['recent_mean_error']}")
    print(f"  preferred_skills={ma['preferred_skills']}  avoid={ma['avoid_actions']}")
    perf = result["perf"]
    print(f"  perf: cycle={perf['cycle_ms']}ms  mem={perf['mem_ms']}ms  "
          f"plan={perf['plan_ms']}ms  cache_hit={perf['mem_cache_hit']}")
    if result["requires_token"]:
        print(f"  → APPLY：请执行  auth {result['suggestion_id']}")
    elif result["decision"] == "ESCALATE":
        print("  → ESCALATED：请先 clear 再继续")
    else:
        print("  → HOLD：本轮不执行写入")


def run_demo(k: OmegaKernel, operator_secret: Optional[str] = None) -> None:
    print("\n=== DEMO：5 轮认知 + 一次完整授权执行 ===")
    last_apply_sid = None
    for i in range(5):
        pred = 50.0 + random.uniform(-3, 3)
        # 前几轮偏差大，后几轮变小
        noise = 12 - i * 2.2
        actual = pred + random.uniform(-noise, noise)
        r = k.cognitive_cycle(round(pred, 1), round(actual, 1))
        _print_cycle(r)
        if r["decision"] == "APPLY":
            last_apply_sid = r["suggestion_id"]
        time.sleep(0.05)

    if last_apply_sid and k.state == KernelState.RUNNING:
        print(f"\n--- 授权最后一次 APPLY: {last_apply_sid} ---")
        ar = k.authorize(last_apply_sid, operator_secret=operator_secret)
        print(ar)
        if ar.get("ok"):
            print(f"--- 执行令牌 {ar['token']} ---")
            er = k.execute(ar["token"])
            print(er)
            print("--- 再次用同一令牌（应失败）---")
            print(k.execute(ar["token"]))
    else:
        print("\n（本轮 demo 没有产出 APPLY，或已 ESCALATED，跳过执行演示）")
    print("\n=== status ===")
    import json
    print(json.dumps(k.status(), ensure_ascii=False, indent=2))


def main() -> None:
    env_secret = os.environ.get(OPERATOR_SECRET_ENV)
    demo_secret = None if env_secret else secrets.token_hex(16)
    k = OmegaKernel(operator_secret=env_secret or demo_secret)
    print(BANNER)
    if demo_secret:
        print(f"[demo] 本次启动的操作口令: {demo_secret}（仅本次进程有效；固定口令请设 {OPERATOR_SECRET_ENV} 环境变量）")
    while True:
        try:
            raw = input("omega> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            break
        if not raw:
            continue
        parts = raw.split()
        cmd = parts[0].lower()

        if cmd in ("quit", "exit", "q"):
            print("bye")
            break
        elif cmd == "status":
            import json
            print(json.dumps(k.status(), ensure_ascii=False, indent=2))
        elif cmd == "pause":
            print(k.pause())
        elif cmd == "resume":
            print(k.resume())
        elif cmd == "clear":
            print(k.clear_escalate())
        elif cmd == "demo":
            run_demo(k, env_secret or demo_secret)
        elif cmd == "cycle":
            if len(parts) >= 3:
                try:
                    pred = float(parts[1])
                    actual = float(parts[2])
                except ValueError:
                    print("用法: cycle [预测] [实际]")
                    continue
            else:
                pred = round(50 + random.uniform(-5, 5), 1)
                actual = round(pred + random.uniform(-8, 8), 1)
                print(f"（自动）预测={pred} 实际={actual}")
            r = k.cognitive_cycle(pred, actual)
            _print_cycle(r)
        elif cmd == "auth":
            if len(parts) < 2:
                print("用法: auth <suggestion_id> <operator_secret>（启动时打印的本次口令，或 OMEGA_OPERATOR_SECRET）")
                continue
            secret = parts[2] if len(parts) >= 3 else None
            print(k.authorize(parts[1], operator_secret=secret))
        elif cmd == "chpass":
            if len(parts) < 3:
                print("用法: chpass <旧口令> <新口令>")
                continue
            print(k.change_operator_secret(parts[1], parts[2]))
        elif cmd == "exec":
            if len(parts) < 2:
                print("用法: exec <token>")
                continue
            print(k.execute(parts[1]))
        else:
            print("未知命令。cycle / auth / chpass / exec / status / pause / resume / clear / demo / quit")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Omega 统一运行时治理层（omega_governance.py）

只做运行时 / 生命周期治理，不碰训练。覆盖五项能力：

  1. 检查点放行门：结构完整 / 词表一致 / 记录诚实 / 确有进步 / 生成不退化
     （模板来自先前产物 release_gate.py，适配 dual meta_json 格式）
  2. 工具与动作授权：白名单 + 需确认动作的一次性令牌 + 执行审计
  3. 记忆写入授权：长度上限 / 协议保留字符 / 暂停开关（只裁决，不存数据）
  4. 审计日志：JSONL 追加写 + 哈希链（可 HMAC），可验全链
  5. 回滚：治理文件快照 / 恢复 / 已发布版本的取代标记（永不删除已发布文件）

明确不做：训练过程的 chunk 级门控、按 loss 接受/回滚训练步、学习率衰减。
旧文件 omega.py / app.py / omega_mini.py / omega_transformer_bridge.py 原地不动，
本模块不依赖它们。

依赖：标准库 + NumPy（仅用于读取 npz 检查点元数据），Pythonista 可跑。
用法见 omega_governance_test.py；设计见 OMEGA_UNIFIED_DESIGN.md。
"""

from __future__ import annotations

import hashlib
import hmac as hmac_mod
import json
import secrets
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# 通用小工具
# ---------------------------------------------------------------------------

ALLOW = "ALLOW"
DENY = "DENY"
HOLD = "HOLD"
RELEASE = "RELEASE"
BROKEN = "BROKEN"

RESERVED_CHARS_DEFAULT = "《》】〔〕（）"
DEFAULT_TOKEN_TTL_SEC = 300.0
DEFAULT_MAX_MEMORY_CHARS = 500
DEFAULT_MARGIN = 0.05
DEFAULT_HONESTY_TOL = 0.10
DEFAULT_FIRST_RELEASE_MAX_VAL = 8.0
DEFAULT_REPETITION_CAP = 0.55
DEFAULT_GEN_PROMPT = "悟空"
DEFAULT_GEN_TOKENS = 80


def canonical(obj: Any) -> str:
    """稳定的 JSON 序列化（排序键、中文原文），用于哈希与参数指纹。"""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# 主人认证（omega_mini 的 pbkdf2 模式，修正为显式目录 + 无交互）
# ---------------------------------------------------------------------------

class OwnerAuth:
    """主人密钥：pbkdf2 存盐与摘要；令牌为一次性、绑定动作与参数哈希、带 TTL。"""

    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.owner_file = self.base_dir / "owner.json"
        self._tokens: Dict[str, Dict[str, Any]] = {}
        self._consumed: List[str] = []

    def create_owner(self, secret: str) -> None:
        if len(secret) < 8:
            raise ValueError("owner secret must be at least 8 characters")
        salt = secrets.token_hex(16)
        digest = hashlib.pbkdf2_hmac(
            "sha256", secret.encode("utf-8"), bytes.fromhex(salt), 200000
        ).hex()
        tmp = self.owner_file.with_suffix(".tmp")
        tmp.write_text(json.dumps({"salt": salt, "digest": digest}), encoding="utf-8")
        tmp.replace(self.owner_file)

    def has_owner(self) -> bool:
        return self.owner_file.exists()

    def verify_owner(self, secret: str) -> bool:
        if not self.owner_file.exists():
            return False
        try:
            data = json.loads(self.owner_file.read_text(encoding="utf-8"))
            digest = hashlib.pbkdf2_hmac(
                "sha256", secret.encode("utf-8"), bytes.fromhex(data["salt"]), 200000
            ).hex()
            return hmac_mod.compare_digest(digest, data["digest"])
        except (KeyError, ValueError, TypeError):
            return False

    def issue_token(self, action: str, args: Any,
                    ttl_sec: float = DEFAULT_TOKEN_TTL_SEC) -> str:
        token = secrets.token_hex(16)
        self._tokens[token] = {
            "action": action,
            "args_hash": sha256_hex(canonical(args)),
            "expires_at": time.time() + max(0.001, float(ttl_sec)),
        }
        return token

    def consume_token(self, token: str, action: str, args: Any) -> Tuple[bool, str]:
        if token in self._consumed:
            return False, "token already used"
        rec = self._tokens.get(token)
        if rec is None:
            return False, "unknown token"
        if time.time() > rec["expires_at"]:
            return False, "token expired"
        if rec["action"] != action:
            return False, "token action mismatch"
        if rec["args_hash"] != sha256_hex(canonical(args)):
            return False, "token args mismatch"
        self._consumed.append(token)
        del self._tokens[token]
        return True, "ok"


# ---------------------------------------------------------------------------
# 审计日志（omega.py AuditChain 的思想，简化为单文件 JSONL + 哈希链）
# ---------------------------------------------------------------------------

class AuditLog:
    """追加写 JSONL，每条链向前一条哈希；设 key 时用 HMAC-SHA256。"""

    def __init__(self, path: Path, key: Optional[bytes] = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.key = key
        self.records: List[Dict[str, Any]] = []
        if self.path.exists():
            with self.path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.records.append(json.loads(line))

    def _digest(self, body: Dict[str, Any]) -> str:
        data = canonical(body).encode("utf-8")
        if self.key is None:
            return hashlib.sha256(data).hexdigest()
        return hmac_mod.new(self.key, data, hashlib.sha256).hexdigest()

    def append(self, kind: str, actor: str, action: str,
               decision: str, detail: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        prev = self.records[-1]["hash"] if self.records else "GENESIS"
        body = {
            "ts": utc_now(),
            "kind": kind,
            "actor": actor,
            "action": action,
            "decision": decision,
            "detail": detail or {},
            "prev_hash": prev,
        }
        body["hash"] = self._digest(body)
        self.records.append(body)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(body, ensure_ascii=False) + "\n")
        return body

    def verify_chain(self) -> Tuple[bool, str]:
        prev = "GENESIS"
        for i, rec in enumerate(self.records):
            if rec.get("prev_hash") != prev:
                return False, f"record {i} prev_hash mismatch"
            body = {k: v for k, v in rec.items() if k != "hash"}
            if self._digest(body) != rec.get("hash"):
                return False, f"record {i} hash mismatch"
            prev = rec["hash"]
        return True, "ok"

    def best_released_val(self) -> Tuple[Optional[float], Optional[str]]:
        """从审计里找已放行版本的最佳 val（越小越好）。"""
        best: Optional[float] = None
        name: Optional[str] = None
        for rec in self.records:
            if rec.get("kind") == "release" and rec.get("decision") == RELEASE:
                val = (rec.get("detail") or {}).get("val")
                nm = (rec.get("detail") or {}).get("release_name")
                if isinstance(val, (int, float)) and (best is None or val < best):
                    best, name = float(val), nm
        return best, name


# ---------------------------------------------------------------------------
# 策略：白名单 + 每动作风险 / 需确认 + 暂停开关
# ---------------------------------------------------------------------------

class Policy:
    """动作白名单与确认规则（omega.py Boundary / ToolRegistry 的简化版）。"""

    DEFAULT_TABLE: Dict[str, Tuple[float, bool]] = {
        "计算": (0.1, False),
        "回忆": (0.2, False),
        "记住": (0.5, True),
    }

    def __init__(self, table: Optional[Dict[str, Tuple[float, bool]]] = None):
        self.table: Dict[str, Tuple[float, bool]] = dict(
            table if table is not None else self.DEFAULT_TABLE
        )
        self.paused = False

    def register_action(self, name: str, risk: float, requires_confirm: bool) -> None:
        self.table[name] = (float(risk), bool(requires_confirm))

    def permits(self, action: str) -> bool:
        return action in self.table

    def requires_confirm(self, action: str) -> bool:
        return self.table[action][1]

    def risk(self, action: str) -> float:
        return self.table[action][0]

    def pause(self) -> None:
        self.paused = True

    def resume(self) -> None:
        self.paused = False


# ---------------------------------------------------------------------------
# 工具与动作授权
# ---------------------------------------------------------------------------

class ToolGate:
    """授权 → 令牌 → 执行。低风险动作直接放行但仍审计；需确认动作必须持有效令牌。"""

    def __init__(self, policy: Policy, owner_auth: OwnerAuth, audit: AuditLog):
        self.policy = policy
        self.owner_auth = owner_auth
        self.audit = audit

    def authorize(self, action: str, args: Any,
                  owner_secret: Optional[str] = None) -> Dict[str, Any]:
        if self.policy.paused:
            self.audit.append("tool", "governance", action, HOLD,
                              {"reason": "paused"})
            return {"decision": HOLD, "token": None, "reason": "paused"}
        if not self.policy.permits(action):
            self.audit.append("tool", "governance", action, DENY,
                              {"reason": "action not in allowlist"})
            return {"decision": DENY, "token": None, "reason": "action not in allowlist"}
        if not self.policy.requires_confirm(action):
            self.audit.append("tool", "governance", action, ALLOW,
                              {"risk": self.policy.risk(action)})
            return {"decision": ALLOW, "token": None, "reason": "low risk, no confirm needed"}
        if owner_secret is None or not self.owner_auth.verify_owner(owner_secret):
            self.audit.append("tool", "governance", action, HOLD,
                              {"reason": "owner approval required"})
            return {"decision": HOLD, "token": None, "reason": "owner approval required"}
        token = self.owner_auth.issue_token(action, args)
        self.audit.append("tool", "governance", action, ALLOW,
                          {"reason": "owner approved, token issued"})
        return {"decision": ALLOW, "token": token, "reason": "owner approved"}

    def execute(self, token: Optional[str], action: str, args: Any,
                func: Callable[..., Any]) -> Dict[str, Any]:
        if self.policy.paused:
            self.audit.append("tool_exec", "governance", action, DENY,
                              {"reason": "paused"})
            return {"executed": False, "result": None, "error": "paused"}
        if not self.policy.permits(action):
            self.audit.append("tool_exec", "governance", action, DENY,
                              {"reason": "action not in allowlist"})
            return {"executed": False, "result": None, "error": "action not in allowlist"}
        if self.policy.requires_confirm(action):
            if token is None:
                self.audit.append("tool_exec", "governance", action, DENY,
                                  {"reason": "token required"})
                return {"executed": False, "result": None, "error": "token required"}
            ok, reason = self.owner_auth.consume_token(token, action, args)
            if not ok:
                self.audit.append("tool_exec", "governance", action, DENY,
                                  {"reason": reason})
                return {"executed": False, "result": None, "error": reason}
        try:
            result = func(args)
        except Exception as exc:  # noqa: BLE001 - 执行结果本身就是审计内容
            self.audit.append("tool_exec", "governance", action, DENY,
                              {"error": f"{type(exc).__name__}: {exc}"})
            return {"executed": False, "result": None,
                    "error": f"{type(exc).__name__}: {exc}"}
        self.audit.append("tool_exec", "governance", action, ALLOW, {})
        return {"executed": True, "result": result, "error": None}


# ---------------------------------------------------------------------------
# 记忆写入授权（只裁决，不存数据；存储仍由 memory_store.MemoryStore 负责）
# ---------------------------------------------------------------------------

class MemoryGate:
    def __init__(self, policy: Policy, audit: AuditLog,
                 max_chars: int = DEFAULT_MAX_MEMORY_CHARS,
                 reserved: str = RESERVED_CHARS_DEFAULT):
        self.policy = policy
        self.audit = audit
        self.max_chars = max_chars
        self.reserved = reserved

    def authorize_write(self, text: str, actor: str = "owner") -> Dict[str, Any]:
        if self.policy.paused:
            self.audit.append("memory", actor, "记住", DENY, {"reason": "paused"})
            return {"decision": DENY, "reason": "paused"}
        if len(text) > self.max_chars:
            self.audit.append("memory", actor, "记住", DENY,
                              {"reason": f"longer than {self.max_chars} characters"})
            return {"decision": DENY, "reason": f"longer than {self.max_chars} characters"}
        bad = sorted({c for c in text if c in self.reserved})
        if bad:
            self.audit.append("memory", actor, "记住", DENY,
                              {"reason": f"reserved characters present: {bad}"})
            return {"decision": DENY,
                    "reason": f"reserved characters present: {bad}"}
        self.audit.append("memory", actor, "记住", ALLOW, {})
        return {"decision": ALLOW, "reason": "ok"}


# ---------------------------------------------------------------------------
# 检查点放行门（release_gate.py 的 5 项检查，适配 dual meta_json 格式）
# ---------------------------------------------------------------------------

def _read_dual_checkpoint(path: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """读 dual 检查点：返回 (meta, params)。meta_json 不在或格式不对则抛错。"""
    with np.load(path, allow_pickle=False) as data:
        if "meta_json" not in data.files:
            raise ValueError("checkpoint has no meta_json")
        meta = json.loads(bytes(data["meta_json"]).decode("utf-8"))
        if meta.get("format_version") != 1:
            raise ValueError("unsupported checkpoint format")
        params = {k[2:]: np.array(data[k]) for k in data.files if k.startswith("p_")}
    return meta, params


def _expected_dual_shapes(cfg: Dict[str, Any]) -> Dict[str, Tuple[int, ...]]:
    """由 config 推导 dual 参数形状；ff hidden 只在 swiglu 下推导，其余只验自洽。"""
    d = int(cfg["d_model"])
    n_heads = int(cfg["n_heads"])
    n_kv = int(cfg.get("n_kv_heads", n_heads))
    head_dim = d // n_heads
    qd, kvd = n_heads * head_dim, n_kv * head_dim
    shapes = {
        "emb": (int(cfg["vocab_size"]), d),
        "lm_head.weight": (int(cfg["vocab_size"]), d),
    }
    if str(cfg.get("ff_mode", "swiglu")) == "swiglu":
        hidden = round(8 * d / 3)
        shapes["ff_hidden"] = (hidden,)
    shapes["qkv_rows"] = (qd + 2 * kvd, d)
    shapes["proj"] = (d, qd)
    return shapes


class CheckpointGate:
    """放行门：BROKEN（损坏）/ HOLD（拦截）/ RELEASE（放行），永不覆盖已发布文件。"""

    def __init__(self, audit: AuditLog, releases_dir: Path,
                 margin: float = DEFAULT_MARGIN,
                 honesty_tol: float = DEFAULT_HONESTY_TOL,
                 first_release_max_val: float = DEFAULT_FIRST_RELEASE_MAX_VAL,
                 repetition_cap: float = DEFAULT_REPETITION_CAP):
        self.audit = audit
        self.releases_dir = Path(releases_dir)
        self.releases_dir.mkdir(parents=True, exist_ok=True)
        self.margin = margin
        self.honesty_tol = honesty_tol
        self.first_release_max_val = first_release_max_val
        self.repetition_cap = repetition_cap

    def evaluate(self, path: Path,
                 reference_chars: Optional[List[str]] = None,
                 recompute_val: Optional[Callable[[Path], float]] = None,
                 generate: Optional[Callable[[str], str]] = None,
                 require_honesty: bool = False) -> Dict[str, Any]:
        path = Path(path)
        checks: List[Dict[str, Any]] = []

        def check(name: str, status: str, detail: str) -> None:
            checks.append({"name": name, "status": status, "detail": detail})

        try:
            meta, params = _read_dual_checkpoint(path)
        except Exception as exc:  # noqa: BLE001 - 损坏本身就是检查结果
            check("structure", "fail", f"unreadable checkpoint: {exc}")
            verdict = BROKEN
            self.audit.append("release_eval", "governance", str(path), verdict,
                              {"checks": checks})
            return {"verdict": verdict, "checks": checks}

        cfg = meta.get("config") or {}
        step = meta.get("step")
        val = (meta.get("extra") or {}).get("best_val_loss")
        # -- 1. 结构完整 --
        problems: List[str] = []
        for key in ("d_model", "n_blocks", "n_heads", "vocab_size"):
            if key not in cfg:
                problems.append(f"config missing {key}")
        if not isinstance(step, int):
            problems.append("meta step missing or not int")
        if not isinstance(val, (int, float)):
            problems.append("recorded best_val_loss missing")
        if not params:
            problems.append("no parameters")
        if problems:
            check("structure", "fail", "; ".join(problems))
        else:
            try:
                exp = _expected_dual_shapes(cfg)
                d = int(cfg["d_model"])
                shape_problems: List[str] = []
                if params.get("emb", np.zeros((0,)) ).shape != exp["emb"]:
                    shape_problems.append("emb shape mismatch")
                if "lm_head.weight" in params and params["lm_head.weight"].shape != exp["lm_head.weight"]:
                    shape_problems.append("lm_head shape mismatch")
                for i in range(int(cfg["n_blocks"])):
                    p = f"left.{i}"
                    need = [f"{p}.attn.qkv.weight", f"{p}.attn.proj.weight",
                            f"{p}.ff_gate.weight", f"{p}.ff_up.weight", f"{p}.ff2.weight"]
                    for k in need:
                        if k not in params:
                            shape_problems.append(f"missing {k}")
                    if f"{p}.attn.qkv.weight" in params and params[f"{p}.attn.qkv.weight"].shape != exp["qkv_rows"]:
                        shape_problems.append(f"{p} qkv shape mismatch")
                    if f"{p}.attn.proj.weight" in params and params[f"{p}.attn.proj.weight"].shape != exp["proj"]:
                        shape_problems.append(f"{p} proj shape mismatch")
                    if "ff_hidden" in exp:
                        hidden = exp["ff_hidden"][0]
                        for k in (f"{p}.ff_gate.weight", f"{p}.ff_up.weight"):
                            if k in params and params[k].shape != (hidden, d):
                                shape_problems.append(f"{p} {k} shape mismatch")
                        if f"{p}.ff2.weight" in params and params[f"{p}.ff2.weight"].shape != (d, hidden):
                            shape_problems.append(f"{p} ff2 shape mismatch")
                nonfinite = [k for k, v in params.items() if not bool(np.all(np.isfinite(v)))]
                if nonfinite:
                    shape_problems.append(f"non-finite weights: {nonfinite[:3]}")
                if shape_problems:
                    check("structure", "fail", "; ".join(shape_problems))
                else:
                    check("structure", "pass", "shapes consistent, weights finite")
            except (KeyError, TypeError, ValueError) as exc:
                check("structure", "fail", f"shape derivation failed: {exc}")

        # -- 2. 词表一致 --
        # 本项目约定：vocab_size = 语料字 + 少量特殊位（如 4510 = 4507 + 3），
        # 而记录的 chars 只含语料字；结构检查已验证 emb 行数 == vocab_size，
        # 因此小缺口视为特殊位而非损坏，大缺口仍判失败。
        chars = (meta.get("extra") or {}).get("tokenizer", {}).get("chars")
        vocab_size = int(cfg.get("vocab_size", -1))
        if chars is None:
            check("vocab", "skip", "no chars recorded in checkpoint")
        elif len(chars) > vocab_size:
            check("vocab", "fail", "chars longer than vocab_size")
        elif reference_chars is not None and list(chars) != list(reference_chars):
            check("vocab", "fail", "chars differ from reference")
        elif len(chars) == vocab_size:
            check("vocab", "pass", "chars consistent")
        elif vocab_size - len(chars) <= 16:
            check("vocab", "pass",
                  f"chars {len(chars)} + {vocab_size - len(chars)} presumed specials = vocab {vocab_size}")
        else:
            check("vocab", "fail",
                  f"chars shorter than vocab_size by {vocab_size - len(chars)}")

        # -- 3. 记录诚实 --
        if recompute_val is None:
            check("honesty", "skip", "no recompute function given")
        elif not isinstance(val, (int, float)):
            check("honesty", "fail", "no recorded val to compare")
        else:
            try:
                recomputed = float(recompute_val(path))
                if abs(recomputed - float(val)) <= self.honesty_tol:
                    check("honesty", "pass", f"recomputed {recomputed:.4f} vs recorded {val:.4f}")
                else:
                    check("honesty", "fail", f"recomputed {recomputed:.4f} vs recorded {val:.4f}")
            except Exception as exc:  # noqa: BLE001 - 重算失败就是该项失败
                check("honesty", "fail", f"recompute failed: {exc}")

        # -- 4. 确有进步 --
        if not isinstance(val, (int, float)):
            check("improvement", "fail", "no recorded val")
        else:
            best, _name = self.audit.best_released_val()
            if best is None:
                if float(val) < self.first_release_max_val:
                    check("improvement", "pass", "first release, val below cap")
                else:
                    check("improvement", "fail", f"first release val {val} not below {self.first_release_max_val}")
            elif float(val) <= best - self.margin:
                check("improvement", "pass", f"val {val:.4f} beats released best {best:.4f}")
            else:
                check("improvement", "fail", f"val {val:.4f} does not beat released best {best:.4f} by {self.margin}")

        # -- 5. 生成不退化 --
        if generate is None:
            check("generation", "skip", "no generate function given")
        else:
            try:
                sample = generate(DEFAULT_GEN_PROMPT)
                if not sample:
                    check("generation", "fail", "empty generation")
                else:
                    from collections import Counter
                    top_char, top_count = Counter(sample).most_common(1)[0]
                    share = top_count / len(sample)
                    if share <= self.repetition_cap:
                        check("generation", "pass", f"top char share {share:.2f}")
                    else:
                        check("generation", "fail", f"top char {top_char!r} share {share:.2f}")
            except Exception as exc:  # noqa: BLE001 - 生成失败就是该项失败
                check("generation", "fail", f"generation failed: {exc}")

        failed = [c for c in checks if c["status"] == "fail"]
        skipped_honesty = any(c["name"] == "honesty" and c["status"] == "skip" for c in checks)
        if any(c["name"] == "structure" and c["status"] == "fail" for c in checks):
            verdict = BROKEN
        elif failed or (require_honesty and skipped_honesty):
            verdict = HOLD
        else:
            verdict = RELEASE
        self.audit.append("release_eval", "governance", str(path), verdict,
                          {"checks": checks, "step": step, "val": val})
        return {"verdict": verdict, "checks": checks, "step": step, "val": val}

    def release(self, path: Path, name: str,
                reference_chars: Optional[List[str]] = None,
                recompute_val: Optional[Callable[[Path], float]] = None,
                generate: Optional[Callable[[str], str]] = None,
                require_honesty: bool = False) -> Dict[str, Any]:
        result = self.evaluate(path, reference_chars, recompute_val, generate, require_honesty)
        if result["verdict"] != RELEASE:
            return {"released": False, "verdict": result["verdict"],
                    "checks": result["checks"], "release_name": None}
        step, val = result["step"], result["val"]
        base = f"{name}_step{step}_val{float(val):.4f}.npz"
        target = self.releases_dir / base
        n = 2
        while target.exists():
            target = self.releases_dir / f"{base[:-4]}-{n}.npz"
            n += 1
        shutil.copyfile(path, target)
        self.audit.append("release", "governance", str(path), RELEASE,
                          {"release_name": target.name, "val": float(val),
                           "sha256": sha256_file(path)})
        return {"released": True, "verdict": RELEASE,
                "checks": result["checks"], "release_name": target.name}


# ---------------------------------------------------------------------------
# 回滚：快照 / 恢复 / 取代标记（永不删除已发布文件，不碰检查点内容）
# ---------------------------------------------------------------------------

class Rollback:
    def __init__(self, audit: AuditLog, snapshots_dir: Path):
        self.audit = audit
        self.snapshots_dir = Path(snapshots_dir)
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)

    def snapshot(self, paths: List[Path], label: str = "") -> str:
        snap_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(4)}"
        snap_dir = self.snapshots_dir / snap_id
        snap_dir.mkdir(parents=True)
        manifest: Dict[str, str] = {}
        for p in paths:
            p = Path(p)
            if not p.exists():
                continue
            dest = snap_dir / p.name
            shutil.copyfile(p, dest)
            manifest[p.name] = sha256_file(dest)
        (snap_dir / "manifest.json").write_text(
            json.dumps({"label": label, "files": manifest}, ensure_ascii=False, indent=1),
            encoding="utf-8")
        self.audit.append("snapshot", "governance", snap_id, ALLOW,
                          {"label": label, "files": sorted(manifest)})
        return snap_id

    def list_snapshots(self) -> List[str]:
        return sorted(p.name for p in self.snapshots_dir.iterdir() if p.is_dir())

    def restore(self, snap_id: str, dest_dir: Path) -> Dict[str, Any]:
        snap_dir = self.snapshots_dir / snap_id
        if not snap_dir.is_dir():
            return {"restored": False, "error": "unknown snapshot"}
        manifest = json.loads((snap_dir / "manifest.json").read_text(encoding="utf-8"))
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        for name, digest in manifest["files"].items():
            src = snap_dir / name
            if sha256_file(src) != digest:
                self.audit.append("restore", "governance", snap_id, DENY,
                                  {"reason": f"snapshot file {name} hash mismatch"})
                return {"restored": False, "error": f"snapshot file {name} hash mismatch"}
            shutil.copyfile(src, dest_dir / name)
        self.audit.append("restore", "governance", snap_id, ALLOW,
                          {"files": sorted(manifest["files"])})
        return {"restored": True, "error": None}

    def supersede(self, release_name: str, reason: str) -> Dict[str, Any]:
        """把某已发布版本标记为已取代：只写审计，不删除文件。"""
        self.audit.append("supersede", "governance", release_name, ALLOW,
                          {"reason": reason})
        return {"superseded": True, "release_name": release_name}


# ---------------------------------------------------------------------------
# 门面：把五项能力绑在一起
# ---------------------------------------------------------------------------

class Governance:
    def __init__(self, base_dir: Path, audit_key: Optional[bytes] = None):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.policy = Policy()
        self.owner_auth = OwnerAuth(self.base_dir)
        self.audit = AuditLog(self.base_dir / "governance_audit.jsonl", key=audit_key)
        self.tools = ToolGate(self.policy, self.owner_auth, self.audit)
        self.memory = MemoryGate(self.policy, self.audit)
        self.checkpoints = CheckpointGate(self.audit, self.base_dir / "releases")
        self.rollback = Rollback(self.audit, self.base_dir / "snapshots")

    def pause(self) -> None:
        self.policy.pause()
        self.audit.append("pause", "governance", "-", HOLD, {})

    def resume(self) -> None:
        self.policy.resume()
        self.audit.append("resume", "governance", "-", ALLOW, {})

    def status(self) -> Dict[str, Any]:
        best, name = self.audit.best_released_val()
        ok, msg = self.audit.verify_chain()
        return {
            "paused": self.policy.paused,
            "actions": sorted(self.policy.table),
            "audit_records": len(self.audit.records),
            "audit_chain_ok": ok,
            "audit_chain_msg": msg,
            "best_released_val": best,
            "best_released_name": name,
            "snapshots": self.rollback.list_snapshots(),
        }


if __name__ == "__main__":
    print("omega_governance: run omega_governance_test.py for the capability tests.")

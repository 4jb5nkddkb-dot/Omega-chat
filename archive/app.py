"""
app.py —— 真实 OMEGA 接入层（已替换 MockOmega）

内核：omega_baseline_v1.py（Omega.run 执行完整的 观察→意图→预测→执行→归因→决策→记忆 循环）

本模块把 MiniTransformer 接入 OMEGA：
  TrainWorld   —— OMEGA 的 World：observe() 在固定评估集上算 loss；
                  execute("train", k) 真的训练 k 步；execute("measure") 只观测
  TrainPolicy  —— OMEGA 的 Policy：决定做 train 还是 measure，并预测 loss 变化
  训练门控     —— 若 OMEGA 判定 ESCALATE，或 loss 变差，则回滚这一块训练

全部在主线程调用，不使用线程。app_ui.py 通过本模块的函数驱动训练与心跳。
"""

import collections
import math
import os
import secrets
import time

import numpy as np

from omega_baseline_v1 import Omega, Config, Boundary, Intention, Prediction, Decision, clamp
from transformer_core import Adam, encode, text, rng, SEQ_LEN, PARAM_NAMES

BASE = os.path.dirname(os.path.abspath(__file__))
SECRET_PATH = os.path.join(BASE, "omega_secret.txt")   # 请勿分享此文件
STATE_PATH = os.path.join(BASE, "omega_state.json")
CHUNK = 10          # 每个 OMEGA 循环训练的步数
DEFAULT_LR = 0.0003   # OMEGA 门控下的稳定学习率（0.003 时大部分循环会被回滚）
MIN_LR = 1e-4                   # 衰减下限：太小会让训练实际停止
ACCEPT_TOL = 0.0               # 验证 loss 允许的上升量（0 = 严格门控）
REJECT_STREAK_FOR_DECAY = 5   # 连续回滚这么多次，学习率减半

# ---------------- 运行时状态 ----------------
model = None                 # 由 app_ui 设置
LOSS_HISTORY = []            # 被接受的训练循环之后的评估 loss（绘图用）
LR_CUR = DEFAULT_LR
_opt = None

# ---- 训练 / 验证划分 ----
# 原语料有重复段落，先按行去重，再每三行留一行作为验证集。
# 词表不变，因此现有权重仍可加载。
_lines = [l for l in text.split("\n") if l.strip()]
_seen, _uniq = set(), []
for _l in _lines:
    if _l not in _seen:
        _seen.add(_l)
        _uniq.append(_l)
VAL_LINES = [l for i, l in enumerate(_uniq) if i % 3 == 2]
TRAIN_LINES = [l for i, l in enumerate(_uniq) if i % 3 != 2]
TRAIN_IDS = encode("\n".join(TRAIN_LINES) + "\n")
VAL_IDS = encode("\n".join(VAL_LINES) + "\n")
assert len(TRAIN_IDS) > SEQ_LEN + 1 and len(VAL_IDS) > SEQ_LEN + 2, "语料太短，无法划分"

# 固定的验证窗口：每次 observe 都在同样位置算 loss，结果可比
EVAL_WINDOWS = 16   # 验证窗口数：语料变大后需要更多窗口才稳定
EVAL_STARTS = np.linspace(0, len(VAL_IDS) - SEQ_LEN - 2, EVAL_WINDOWS).astype(int)


def set_model(m):
    global model, _opt
    model = m
    _opt = None


def set_lr(lr):
    """改变学习率但保留动量。"""
    global LR_CUR
    LR_CUR = float(lr)
    if _opt is not None:
        _opt.lr = LR_CUR


def _get_opt():
    global _opt
    if _opt is None:
        _opt = Adam(model, lr=LR_CUR)
    return _opt


def eval_loss(m):
    total = 0.0
    for s in EVAL_STARTS:
        x = VAL_IDS[s:s + SEQ_LEN]
        y = VAL_IDS[s + 1:s + SEQ_LEN + 1]
        _, loss, _ = m.forward(x, y)
        total += float(loss)
    return total / len(EVAL_STARTS)


# =========================================================
# World：OMEGA 与 MiniTransformer 之间的接口
# =========================================================

class TrainWorld:
    def __init__(self):
        self.snapshot = None   # 训练前的参数快照，用于回滚

    def observe(self):
        return {"loss": eval_loss(model)}

    def execute(self, action, args):
        if action == "measure":
            return self.observe()
        if action != "train":
            raise ValueError(f"unsupported action: {action}")
        steps = int(args.get("steps", CHUNK))
        self.snapshot = {n: getattr(model, n).copy() for n in PARAM_NAMES}
        opt = _get_opt()
        for _ in range(steps):
            start = rng.integers(0, len(TRAIN_IDS) - SEQ_LEN - 1)
            x = TRAIN_IDS[start:start + SEQ_LEN]
            y = TRAIN_IDS[start + 1:start + SEQ_LEN + 1]
            logits, _, cache = model.forward(x, y)
            grads = model.backward(y, logits, cache)
            opt.step(grads)
        return self.observe()

    def keep(self):
        self.snapshot = None

    def restore(self):
        # 只回滚参数，保留 Adam 动量：清空动量会让下一个循环从大步长重新开始，
        # 反而导致连续被回滚（实测）。
        if self.snapshot is None:
            return
        for name, value in self.snapshot.items():
            getattr(model, name)[...] = value
        self.snapshot = None


WORLD = TrainWorld()


# =========================================================
# Policy：决定动作并预测结果
# =========================================================

class TrainPolicy:
    """预测器：用最近训练循环的 loss 变化（含被回滚的）估计下一次的变化。
    历史不足时使用先验；置信度由历史的离散程度决定。"""
    policy_id = "train-policy"
    policy_version = "2.0"
    MIN_HISTORY = 3

    def __init__(self):
        self.mode = "train"    # "train" 或 "measure"，由调用方设置
        self.k = CHUNK
        self.deltas = collections.deque(maxlen=8)   # 每个训练循环的 loss 变化（每 CHUNK 步）

    def record(self, delta):
        self.deltas.append(float(delta))

    def intend(self, state, goal):
        if self.mode == "train":
            return Intention(action="train", args={"steps": self.k},
                             policy_id=self.policy_id, policy_version=self.policy_version,
                             rationale="advance training by one chunk")
        return Intention(action="measure", args={},
                         policy_id=self.policy_id, policy_version=self.policy_version,
                         rationale="observe only")

    def predict(self, state, intention, goal):
        loss = float(state.get("loss", 0.0))
        if intention.action == "train":
            scale = intention.args.get("steps", CHUNK) / CHUNK
            if len(self.deltas) >= self.MIN_HISTORY:
                ds = list(self.deltas)
                mean = sum(ds) / len(ds)
                std = math.sqrt(sum((d - mean) ** 2 for d in ds) / len(ds))
                conf = clamp(1.0 - std / (abs(mean) + 1e-3) * 0.5, 0.3, 0.9)
            else:
                mean = -0.002          # 先验：每个循环小幅下降
                conf = 0.5
            expected = {"loss": max(0.0, loss + mean * scale)}
        else:
            expected = {"loss": loss}
            conf = 0.95
        return Prediction(expected=expected, confidence=conf,
                          model_id="trend-predictor", model_version="2.0")


POLICY = TrainPolicy()


# =========================================================
# OMEGA 实例
# =========================================================

def _load_secret():
    if os.path.exists(SECRET_PATH):
        with open(SECRET_PATH, encoding="utf-8") as f:
            s = f.read().strip()
        if len(s) >= 8:
            return s
    s = secrets.token_hex(32)
    with open(SECRET_PATH, "w", encoding="utf-8") as f:
        f.write(s)
    return s


OMEGA = None
OWNER = None


def _new_omega():
    # loss 量级约 0.05~0.2，预测误差阈值需要相应缩小
    return Omega(secret=_load_secret(), config=Config(max_error_apply=0.02),
                 boundary=Boundary(frozenset({"train", "measure"})))


def init_omega():
    """启动时调用：创建或恢复 OMEGA 状态。"""
    global OMEGA, OWNER
    OMEGA = _new_omega()
    if os.path.exists(STATE_PATH):
        try:
            OMEGA.load(STATE_PATH)
            print(f"[OMEGA] 已恢复状态，epoch={OMEGA.epoch}")
        except Exception as e:
            print(f"[OMEGA] 状态恢复失败，使用新状态: {type(e).__name__}: {e}")
            OMEGA = _new_omega()
    OWNER = OMEGA.owner()
    OMEGA.set_goal(OWNER, {"loss": 1.0})
    OMEGA.set_limits(OWNER, low={"loss": 0.0}, high={"loss": 10.0})


def save_state():
    if OMEGA is not None:
        OMEGA.save(STATE_PATH)


def omega_summary():
    if OMEGA is None:
        return {}
    return OMEGA.reflect()["summary"]


def clear_escalation():
    """解除 OMEGA 升级状态（需要人工确认后调用）。"""
    if OMEGA is not None:
        OMEGA.deescalate(OWNER)


# =========================================================
# 一个 OMEGA 循环 + 训练门控
# =========================================================

def run_cycle(mode, k=CHUNK):
    """mode = 'train' 或 'measure'。返回一个结果字典，供 UI 显示。"""
    if model is None or OMEGA is None:
        return {"decision": "NONE", "reason": "not_ready", "accepted": False, "loss": None}

    POLICY.mode = mode
    POLICY.k = k
    WORLD.snapshot = None
    try:
        res = OMEGA.run(OWNER, WORLD, POLICY)
    except Exception as e:
        WORLD.restore()
        print(f"[OMEGA] 循环异常，已回滚: {type(e).__name__}: {e}")
        return {"decision": "ERROR", "reason": type(e).__name__, "accepted": False, "loss": None}

    decision = res["decision"]
    dec = decision.value if isinstance(decision, Decision) else str(decision)
    after = res.get("observation", {}).get("loss")
    before = res.get("observation_before", {}).get("loss")

    accepted = True
    if mode == "train":
        # 保留训练结果的条件：OMEGA 没有升级，且验证 loss 没有变差
        accepted = (dec != Decision.ESCALATE.value
                    and after is not None and before is not None
                    and after <= before + ACCEPT_TOL)
        if after is not None and before is not None:
            POLICY.record(after - before)     # 无论保留与否，都进入预测历史
        if accepted:
            WORLD.keep()
            LOSS_HISTORY.append(after)
        else:
            WORLD.restore()
    else:
        WORLD.keep()

    loss = after if after is not None else before
    print(f"[OMEGA] {mode:<7} {dec:<8} reason={res['reason']} "
          f"loss={'-' if loss is None else f'{loss:.4f}'} "
          f"{'保留' if accepted else '回滚'}")
    return {"decision": dec, "reason": res["reason"], "accepted": accepted,
            "loss": loss, "error": res.get("error", 0.0),
            "episode_id": res.get("episode_id")}


def train_job(steps, lr, should_stop=lambda: False):
    """训练任务（生成器）：每个循环 yield 一次进度事件。返回实际保留的训练步数。"""
    set_lr(lr)
    cycles = max(1, steps // CHUNK)
    kept = 0
    streak = 0
    t0 = time.time()
    for i in range(1, cycles + 1):
        if should_stop():
            print("训练已被用户停止")
            break
        r = run_cycle("train", CHUNK)
        if r["accepted"]:
            kept += CHUNK
            streak = 0
        else:
            streak += 1
            if streak >= REJECT_STREAK_FOR_DECAY and LR_CUR > MIN_LR:
                set_lr(max(MIN_LR, LR_CUR * 0.5))
                streak = 0
                print(f"[OMEGA] 连续回滚，学习率降为 {LR_CUR:.6g}")
        if r["decision"] == Decision.ESCALATE.value:
            print("[OMEGA] 已升级，训练暂停。请在设置中解除升级。")
            yield {"kind": "step", "step": i, "total": cycles,
                   "loss": r["loss"] or 0.0, "eta": 0.0, "decision": r["decision"]}
            break
        eta = (time.time() - t0) / i * (cycles - i)
        loss = r["loss"] if r["loss"] is not None else (LOSS_HISTORY[-1] if LOSS_HISTORY else 0.0)
        yield {"kind": "step", "step": i, "total": cycles, "loss": loss,
               "eta": eta, "decision": r["decision"], "accepted": r["accepted"]}
    save_state()
    return kept


def measure_cycle():
    """心跳用：只观测，不训练。"""
    if OMEGA is None or OMEGA.paused:
        return None
    return run_cycle("measure")

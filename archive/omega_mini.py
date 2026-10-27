#!/usr/bin/env python3
"""最小闭环 + 主人权限（设置一次，永久有效）"""

from hashlib import pbkdf2_hmac
from hmac import compare_digest
import secrets, json
from pathlib import Path

OWNER_FILE = Path("omega_owner.json")

def create_owner(secret):
    if len(secret) < 8:
        raise ValueError("至少8位")
    salt = secrets.token_hex(16)
    dig = pbkdf2_hmac("sha256", secret.encode(), bytes.fromhex(salt), 200000).hex()
    OWNER_FILE.write_text(json.dumps({"d": dig, "s": salt}))

def verify_owner(secret):
    if not OWNER_FILE.exists():
        return False
    data = json.loads(OWNER_FILE.read_text())
    dig = pbkdf2_hmac("sha256", secret.encode(), bytes.fromhex(data["s"]), 200000).hex()
    return compare_digest(dig, data["d"])

# 闭环状态
state = {"paused": False, "used": set(), "calibration": {}}

def run_round(token, predicted, confidence, actual):
    if state["paused"]:
        return "ESCALATE（暂停中）"
    if token in state["used"]:
        return "ESCALATE（令牌已用）"
    state["used"].add(token)
    error = actual - predicted
    if confidence < 0.5:
        decision = "HOLD"
    elif abs(error) <= 1.0:
        decision = "APPLY"
        state["calibration"]["last_error"] = error
    else:
        decision = "HOLD"
    print(f"预测={predicted} 实际={actual} 偏差={error:.2f} → {decision}")
    return decision

def main():
    print("=== 最小闭环 + 主人权限 ===")

    # ---------- 主人权限（只设置一次） ----------
    if not OWNER_FILE.exists():
        print("【首次使用】请设置主人密钥（以后都不用再设）")
        a = input("输入密钥（至少8位）：").strip()
        b = input("再输入一次确认：").strip()
        if a != b or len(a) < 8:
            print("设置失败，程序退出")
            return
        create_owner(a)
        print("主人权限已永久保存。\n")
    else:
        secret = input("请输入主人密钥验证：").strip()
        if not verify_owner(secret):
            print("验证失败，程序退出")
            return
        print("验证通过。\n")

    # ---------- 进入闭环 ----------
    print("闭环已启动。命令：")
    print("  授权 <令牌>")
    print("  执行 <令牌> <预测> <置信> <实际值>")
    print("  暂停 / 继续 / 状态 / exit")
    print()

    valid = set()
    while True:
        try:
            cmd = input("mini> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not cmd:
            continue
        if cmd in {"exit", "quit"}:
            print("已退出")
            break

        parts = cmd.split()

        if cmd == "状态":
            print({"paused": state["paused"], "used": list(state["used"]), "calibration": state["calibration"]})
        elif cmd == "暂停":
            state["paused"] = True
            print("已暂停")
        elif cmd == "继续":
            state["paused"] = False
            print("已继续")
        elif parts[0] == "授权" and len(parts) == 2:
            valid.add(parts[1])
            print(f"已授权令牌：{parts[1]}（一次性）")
        elif parts[0] == "执行" and len(parts) == 5:
            token = parts[1]
            if token not in valid:
                print("令牌未授权")
                continue
            try:
                pred, conf, actual = map(float, parts[2:5])
            except ValueError:
                print("数字格式错误")
                continue
            print(run_round(token, pred, conf, actual))
        else:
            print("未知命令")

if __name__ == "__main__":
    main()
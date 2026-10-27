#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""双头运行时整合测试：agent_runtime 的加载入口能识别 dual/modern 格式，
dual SFT 产物经该入口加载后能跑通 run_agent；检查点选择顺序（dual 优先）
用临时目录验证，不依赖镜像里是否放了 dual 产物。

运行：python3 tests/dual_runtime_test.py
全部断言通过则打印 ALL DUAL RUNTIME TESTS PASSED。
只用标准库 + NumPy。dual SFT 产物缺席时跳过加载断言（SKIP），不判失败。
"""
import os
import sys
import tempfile
from pathlib import Path
_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent if _HERE.name in ("tests", "tools", "legacy") else _HERE

sys.path.insert(0, str(ROOT))

import numpy as np

import agent_runtime as ar

MIRROR = ROOT
PRODUCT = Path(os.environ.get(
    "DUAL_SFT_CKPT",
    os.path.expanduser("~/workspace/dual_sft/out/dual_d128_sft.npz")))


def _selection_tests():
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        # 空目录：回落到默认 base 名，架构按缺席文件计为 modern（加载时才报错）。
        p, arch = ar.select_checkpoint(base)
        assert p.name == "modern_ckpt.npz" and arch == "modern", (p, arch)
        (base / "modern_ckpt.npz").touch()
        p, arch = ar.select_checkpoint(base)
        assert p.name == "modern_ckpt.npz" and arch == "modern", (p, arch)
        (base / "modern_ckpt_sft.npz").touch()
        p, arch = ar.select_checkpoint(base)
        assert p.name == "modern_ckpt_sft.npz" and arch == "modern", (p, arch)
        (base / "dual_d128_sft.npz").touch()
        p, arch = ar.select_checkpoint(base)
        assert p.name == "dual_d128_sft.npz" and arch == "dual", (p, arch)
        # 显式非默认检查点名按原样使用，不参与 fallback（缺席文件计为 modern，
        # 加载时才报错；"modern_ckpt.npz" 本身就是 fallback 链的哨兵值）。
        p, arch = ar.select_checkpoint(base, "my_custom.npz")
        assert p.name == "my_custom.npz" and arch == "modern", (p, arch)
    print("selection order (dual first): PASS")


def _selection_detection_tests():
    # 真实文件按结构探测，不再只看文件名：dual 权重即使叫 modern_ckpt.npz
    # 也必须报 dual（2026-10-26 仓库 modern_ckpt.npz 换成 d128 完整检查点后
    # 暴露的问题）；modern 文件仍报 modern。空文件沿用 _selection_tests
    # 的文件名默认值（加载时才报错），此处不覆盖。
    import shutil
    if not PRODUCT.exists():
        print("selection detection (real files): SKIP (no dual product)")
        return
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        shutil.copy(PRODUCT, base / "modern_ckpt.npz")
        p, arch = ar.select_checkpoint(base)
        assert p.name == "modern_ckpt.npz" and arch == "dual", (p, arch)
    modern = MIRROR / "modern_ckpt.npz"
    if modern.exists() and ar.detect_checkpoint_arch(modern) == "modern":
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            shutil.copy(modern, base / "modern_ckpt.npz")
            p, arch = ar.select_checkpoint(base)
            assert p.name == "modern_ckpt.npz" and arch == "modern", (p, arch)
    print("selection detection (real files): PASS")


def _format_tests():
    # 结构未知的文件必须报错，而不是被误认成某一架构。
    with tempfile.TemporaryDirectory() as td:
        junk = Path(td) / "junk.npz"
        np.savez(junk, foo=np.zeros(3))
        try:
            ar.detect_checkpoint_arch(junk)
            raise AssertionError("unknown format was accepted")
        except ValueError:
            pass
    modern = MIRROR / "modern_ckpt_sft.npz"
    if modern.exists():
        assert ar.detect_checkpoint_arch(modern) == "modern"
        print("modern format detection: PASS")
    else:
        print("modern format detection: SKIP (no modern ckpt in mirror)")
    print("unknown format rejection: PASS")


def _dual_load_and_agent_tests():
    if not PRODUCT.exists():
        print(f"dual load + run_agent: SKIP (product not found: {PRODUCT})")
        return
    assert ar.detect_checkpoint_arch(PRODUCT) == "dual"
    model, tok, step, arch = ar.load_checkpoint(PRODUCT)
    assert arch == "dual" and step == 71000, (arch, step)
    assert tok.vocab_size == 4510, tok.vocab_size
    assert all(c in tok.chars for c in "《》】〔〕（）"), "protocol chars incomplete"
    print(f"dual load via agent_runtime entry: PASS (step={step}, vocab={tok.vocab_size})")
    router = ar.ToolRouter({"计算": ar.calc_zh}, tok)
    res = ar.run_agent(model, router, "问：七加八等于几？", ar.contains_verifier("一五"),
                       max_new=64, max_turns=4, max_retries=2, temperature=0.0)
    assert res.ok, res.events
    print("dual run_agent 计算 七加八:", "PASS", "| 答 =", repr(res.text))


def main():
    _selection_tests()
    _selection_detection_tests()
    _format_tests()
    _dual_load_and_agent_tests()
    print("ALL DUAL RUNTIME TESTS PASSED")


if __name__ == "__main__":
    main()

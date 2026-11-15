# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the Omega small-world GUI (Windows, onedir).

Build on windows-latest via .github/workflows/build-windows.yml.
Entry: omega_v010/gui/omega_world_gui.py (tkinter, pure stdlib + numpy).
Model bundling is conditional: set OMEGA_BUNDLE_MODEL=1 and place the
.npz next to this spec (as generalist2_s5100.npz) before running
PyInstaller; otherwise the bundle ships without a model and the GUI
opens its file picker on first run.
The Rust bridge (omega-world-bridge.exe) is built by cargo first and
bundled beside the exe; kernel_autoplay finds it next to the GUI.
"""
import os
from pathlib import Path

HERE = Path(SPECPATH)
GUI = HERE.parent / "gui"
BRIDGE = HERE.parent / "target" / "release" / "omega-world-bridge.exe"
MODEL = HERE / os.environ.get("OMEGA_MODEL_FILE", "generalist2_s5100.npz")
BUNDLE_MODEL = os.environ.get("OMEGA_BUNDLE_MODEL") == "1"
if BUNDLE_MODEL and not MODEL.is_file():
    raise SystemExit(f"OMEGA_BUNDLE_MODEL=1 but model file missing: {MODEL}")

datas = [(str(GUI / "omega.toml"), ".")]
binaries = []
if BRIDGE.is_file():
    binaries.append((str(BRIDGE), "."))
if BUNDLE_MODEL and MODEL.is_file():
    datas.append((str(MODEL), "."))

a = Analysis(
    [str(GUI / "omega_world_gui.py")],
    pathex=[str(GUI)],
    binaries=binaries,
    datas=datas,
    hiddenimports=[],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="OmegaWorld",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    icon=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="OmegaWorld",
)

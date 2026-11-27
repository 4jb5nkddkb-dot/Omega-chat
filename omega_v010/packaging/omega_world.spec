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
# ffmpeg (v0.1.6, bundled per user decision 2026-11-26): CI downloads the
# gyan.dev essentials build (GPLv3) to omega_v010/packaging/ffmpeg.exe
# before running PyInstaller; OMEGA_FFMPEG_EXE overrides the filename.
# video_maker.find_ffmpeg() looks beside the frozen exe, so the binary
# must land in the onedir root (binaries do that).
FFMPEG = HERE / os.environ.get("OMEGA_FFMPEG_EXE", "ffmpeg.exe")
FFMPEG_NOTICE = HERE / "ffmpeg-NOTICE.txt"

datas = [(str(GUI / "omega.toml"), ".")]
if FFMPEG_NOTICE.is_file():
    datas.append((str(FFMPEG_NOTICE), "."))
binaries = []
if BRIDGE.is_file():
    binaries.append((str(BRIDGE), "."))
if FFMPEG.is_file():
    binaries.append((str(FFMPEG), "."))
if BUNDLE_MODEL and MODEL.is_file():
    datas.append((str(MODEL), "."))

a = Analysis(
    [str(GUI / "omega_world_gui.py")],
    pathex=[str(GUI)],
    binaries=binaries,
    datas=datas,
    # PIL is imported inside video_maker functions (deferred imports);
    # list it explicitly so the frozen bundle always carries it.
    hiddenimports=["PIL", "PIL.Image", "PIL.ImageDraw", "PIL.ImageFont",
                   "PIL.ImageEnhance"],
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

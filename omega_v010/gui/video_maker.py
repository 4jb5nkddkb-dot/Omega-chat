#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""视频对话框的渲染器（v0.1.5）：输入人物名字，生成一段几秒的 MP4。

做法是本地渲染，不是扩散模型——离线买断的游戏跑不进文生视频
模型，也不改模型训练。画面是封面图（有就用，没有就画星图）
缓慢推近，配上人物名字、所在地点和一句固定旁白（人物在
world_state.py 里的那句台词），输出 MP4。

依赖：PIL（画帧，只在渲染时 import，没有就明说缺 Pillow）、
ffmpeg（编码 MP4：系统 PATH 里有就用，没有就找游戏目录旁的
ffmpeg.exe；都没有就明说，不静默失败）。视频存在 GUI 目录下
的 videos/ 里，重启还在，界面里能直接打开看。

人物数据只从 world_state.py 取：6 个人物（孙悟空、唐僧、猪八戒、
沙僧、白龙马、观音），名字以外的输入一律不生成，只给提示。
"""
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

# 画面：1280x720，24fps，5 秒。推近幅度 1.00 -> 0.88，够看出在动，
# 又不至于一眼晃到人物名字外面。
WIDTH, HEIGHT = 1280, 720
FPS = 24
DURATION = 5.0

# 人物故乡（视频里的“所在地点”）：队伍人物平时跟着游人走，
# 故乡是他最常在的地方；观音永远在南海。
HOME = {
    "孙悟空": "花果山",
    "唐僧": "取经路",
    "猪八戒": "高老庄",
    "沙僧": "流沙河",
    "白龙马": "取经路",
    "观音": "南海",
}
LOC_EN = {"花果山": "FLOWER FRUIT MT", "水帘洞": "WATER CURTAIN",
          "高老庄": "GAO VILLAGE", "流沙河": "SAND RIVER",
          "取经路": "PILGRIM ROAD", "南海": "SOUTH SEA", "天宫": "HEAVEN"}


class VideoMakerError(Exception):
    """视频生成失败：给界面的提示就是这句话，原样显示。"""


class UnknownCharacterError(VideoMakerError):
    pass


class MissingDependencyError(VideoMakerError):
    pass


def valid_names():
    """小世界里的人物名（顺序按 world_state.py）。"""
    return list(HOME)


def video_dir(base=None):
    """视频保存目录：和游戏进度同一目录下的 videos/。"""
    d = Path(base) if base is not None else HERE / "videos"
    d.mkdir(parents=True, exist_ok=True)
    return d


def list_videos(base=None):
    """已生成的视频（按时间倒序，最新在前）。"""
    d = video_dir(base)
    return sorted(d.glob("*.mp4"), key=lambda p: p.stat().st_mtime,
                  reverse=True)


def open_video(path):
    """用系统默认播放器打开视频。"""
    p = str(path)
    try:
        if sys.platform.startswith("win"):
            os.startfile(p)  # noqa: S606 -- 用户点的自己的视频文件
        elif sys.platform == "darwin":
            subprocess.Popen(["open", p])
        else:
            subprocess.Popen(["xdg-open", p])
    except Exception as e:
        raise VideoMakerError(f"打不开这个视频：{e}")


def find_ffmpeg():
    """找 ffmpeg：环境变量 OMEGA_FFMPEG > 系统 PATH > 游戏目录旁。"""
    env = os.environ.get("OMEGA_FFMPEG")
    if env and Path(env).is_file():
        return env
    found = shutil.which("ffmpeg")
    if found:
        return found
    for cand in (HERE / "ffmpeg.exe", HERE / "ffmpeg"):
        if cand.is_file():
            return str(cand)
    return None


# ---- 字体：Windows 用微软雅黑，Linux 用 fc-match 找中文字体 ----
_FONT_CACHE = {}


def _fc_match(query):
    """fc-match 找字体，返回 (文件, index)，找不到返回 None。"""
    try:
        out = subprocess.run(
            ["fc-match", query, "--format", "%{file}\t%{index}\n"],
            capture_output=True, text=True, timeout=10).stdout.strip()
        if not out or "\t" not in out:
            return None
        f, idx = out.rsplit("\t", 1)
        if Path(f).is_file():
            return f, int(idx)
    except Exception:
        return None
    return None


def find_font(bold=False):
    """找能画中文的字体。Windows 优先微软雅黑，Linux 用 fc-match；
    都没有就回 PIL 默认字体（只能画拉丁字母，中文会画成框框，
    但至少不崩——这种情况只会出现在既没装中文字体又没雅黑的机器上）。"""
    key = "bold" if bold else "regular"
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]
    res = None
    if sys.platform.startswith("win"):
        windir = os.environ.get("SystemRoot", r"C:\Windows")
        msyh = Path(windir) / "Fonts" / "msyh.ttc"
        if msyh.is_file():
            # msyh.ttc 里 index 0 是常规、1 是粗体。
            res = (str(msyh), 1 if bold else 0)
        else:
            msyhbd = Path(windir) / "Fonts" / "msyhbd.ttc"
            if bold and msyhbd.is_file():
                res = (str(msyhbd), 0)
    if res is None:
        res = _fc_match(":lang=zh-CN:bold" if bold else ":lang=zh-CN")
    if res is None:
        # 已知路径兜底：这台开发机上的 Noto Sans CJK。
        for cand in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
                     "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"):
            if Path(cand).is_file():
                res = (cand, 0)
                break
    _FONT_CACHE[key] = res
    return res


def _load_font(size, bold=False):
    """PIL 字体：按 find_font 的结果加载；没有中文字体就用默认字体。"""
    from PIL import ImageFont
    found = find_font(bold=bold)
    if found is not None:
        try:
            return ImageFont.truetype(found[0], size, index=found[1])
        except Exception:
            pass
    return ImageFont.load_default()


def find_cover():
    """找封面图：游戏目录旁的 cover.png > 封面原图 > 开发机上的
    封面文件。都没有就返回 None，渲染改用程序画的星图。"""
    for cand in (HERE / "cover.png",
                 HERE / "omega-world-cover-v0.1.3.png",
                 Path.home() / "workspace" / "omega-world-cover-v0.1.3.png",
                 Path.home() / "workspace" / "omega-world-cover-v0.1.3-square.png"):
        if cand.is_file():
            return cand
    return None


def _cover_base(path):
    """封面裁成 16:9 的基准框（源图坐标）：宽占满，高居中裁。"""
    from PIL import Image
    im = Image.open(path).convert("RGB")
    w, h = im.size
    ch = int(w * 9 / 16)
    if ch <= h:
        top = (h - ch) // 2
        return im, (0, top, w, top + ch)
    # 图太矮（比如正方形封面）：高占满，宽居中裁。
    cw = int(h * 16 / 9)
    left = (w - cw) // 2
    return im, (left, 0, left + cw, h)


def _starfield_base():
    """没封面时的底图：按界面星图的画法，画一张 2560x1440 的星图
    （深底、暖光、双轨道、中心核、星星），推近逻辑和封面同一套。"""
    import random
    from PIL import Image, ImageDraw
    W, H = 2560, 1440
    im = Image.new("RGB", (W, H), (3, 5, 10))
    dr = ImageDraw.Draw(im)
    cx, cy = W // 2, H // 2
    for rx, ry, col in ((900, 520, (20, 16, 5)), (680, 400, (29, 21, 7)),
                        (500, 300, (42, 26, 6))):
        dr.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=col)
    dr.ellipse([cx - 1050, cy - 528, cx + 1050, cy + 528],
               outline=(107, 74, 34), width=3)
    dr.ellipse([cx - 672, cy - 336, cx + 672, cy + 336],
               outline=(42, 90, 102), width=3)
    rng = random.Random(42)
    for _ in range(220):
        x = rng.randint(20, W - 20)
        y = rng.randint(12, H - 12)
        r = rng.choice((1, 2, 3))
        dr.ellipse([x - r, y - r, x + r, y + r], fill=(217, 246, 255))
    dr.ellipse([cx - 200, cy - 200, cx + 200, cy + 200],
               fill=(255, 179, 92), outline=(255, 241, 214), width=4)
    dr.ellipse([cx - 120, cy - 120, cx + 120, cy + 120], fill=(255, 241, 214))
    return im, (0, 0, W, H)


def _frame(im, base_box, t, draw_text):
    """第 t 帧（t 在 0..1 之间）：基准框缓慢推近（1.00 -> 0.88，
    中心左右轻移），缩到 1280x720，再画文字层。"""
    from PIL import Image, ImageDraw
    x0, y0, x1, y1 = base_box
    bw, bh = x1 - x0, y1 - y0
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    z = 1.0 - 0.12 * t
    drift = -16 + 32 * t  # 源图坐标下左右轻移，画面是活的
    cw, ch = bw * z, bh * z
    left = cx + drift - cw / 2
    top = cy - ch / 2
    frame = im.crop((int(left), int(top), int(left + cw), int(top + ch)))
    frame = frame.resize((WIDTH, HEIGHT), Image.LANCZOS)
    draw_text(ImageDraw.Draw(frame, "RGBA"), frame)
    return frame


def render_video(name, out_dir=None, location=None, duration=DURATION,
                 fps=FPS, progress=None):
    """生成一段视频，返回 mp4 路径。

    name 不在小世界里就抛 UnknownCharacterError（界面接住显示，
    不生成空视频）；没 PIL 或没 ffmpeg 就抛 MissingDependencyError，
    提示缺什么、怎么补。progress(i, n) 每 12 帧调一次，供界面
    刷新“正在生成…”的状态。"""
    from PIL import Image  # noqa: F401 -- 没有 Pillow 就在这里明说
    name = (name or "").strip()
    if not name:
        raise UnknownCharacterError("先输入一个人物名字。")
    if name not in HOME:
        raise UnknownCharacterError(
            f"小世界里没有“{name}”。这里的人物有："
            + "、".join(valid_names()) + "。")
    ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        raise MissingDependencyError(
            "生成视频需要 ffmpeg：把它放在游戏目录旁（ffmpeg.exe），"
            "或装到系统 PATH 里，再点生成。")
    loc = location or HOME[name]
    try:
        import world_state as ws
        line = ws.CHARACTERS[name]["line"]
        loc_desc = ws.LOCATIONS.get(loc, "")
    except Exception:
        line = ""
        loc_desc = ""
    cover = find_cover()
    if cover is not None:
        base_im, base_box = _cover_base(cover)
    else:
        base_im, base_box = _starfield_base()

    out = video_dir(out_dir)
    ts = time.strftime("%Y%m%d_%H%M%S")
    target = out / f"视频_{name}_{ts}.mp4"
    n = 2
    while target.exists():
        target = out / f"视频_{name}_{ts}_{n}.mp4"
        n += 1
    total = max(1, int(round(duration * fps)))

    f_name = _load_font(60, bold=True)
    f_loc = _load_font(26)
    f_line = _load_font(30)
    f_top = _load_font(24)
    AMBER = (255, 179, 92)
    CYAN = (127, 246, 255)
    WHITE = (217, 246, 255)

    def draw_text(dr, frame):
        # 顶栏小字 + 底栏：名字 / 地点 / 台词。
        dr.text((48, 36), "OMEGA · 小世界", font=f_top, fill=CYAN)
        dr.rectangle([0, 500, WIDTH, HEIGHT], fill=(0, 0, 0, 170))
        dr.rectangle([0, 500, WIDTH, 503], fill=(255, 179, 92, 255))
        dr.text((64, 522), name, font=f_name, fill=AMBER)
        dr.text((64, 600), f"{loc} · {LOC_EN.get(loc, '')}",
                font=f_loc, fill=CYAN)
        if line:
            dr.text((64, 640), f"“{line}”", font=f_line, fill=WHITE)
        elif loc_desc:
            dr.text((64, 640), loc_desc, font=f_line, fill=WHITE)

    cmd = [ffmpeg, "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{WIDTH}x{HEIGHT}", "-r", str(fps), "-i", "pipe:0",
           "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-crf", "23", "-preset", "veryfast",
           "-movflags", "+faststart", str(target)]
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.PIPE)
    except Exception as e:
        raise MissingDependencyError(f"ffmpeg 启动失败：{e}")
    try:
        for i in range(total):
            frame = _frame(base_im, base_box, i / max(1, total - 1),
                           draw_text)
            try:
                proc.stdin.write(frame.tobytes())
            except BrokenPipeError:
                break
            if progress is not None and (i % 12 == 0 or i == total - 1):
                progress(i + 1, total)
    finally:
        # stdin 关掉才算送完帧；communicate() 会再 flush 已关的 stdin
        # 而报错，所以这里直接读 stderr 再等进程退出。
        try:
            proc.stdin.close()
        except Exception:
            pass
        err = proc.stderr.read()
        rc = proc.wait()
    if rc != 0 or not target.is_file() or target.stat().st_size == 0:
        tail = (err or b"")[-500:].decode("utf-8", "replace")
        raise VideoMakerError(f"视频编码失败（ffmpeg 退出码 "
                             f"{rc}）：{tail}")
    return target


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("name", help="人物名字，如 孙悟空")
    ap.add_argument("--out", default=None)
    ap.add_argument("--duration", type=float, default=DURATION)
    args = ap.parse_args()
    try:
        path = render_video(args.name, out_dir=args.out,
                            duration=args.duration,
                            progress=lambda i, n: print(
                                f"\r帧 {i}/{n}", end="", flush=True))
    except VideoMakerError as e:
        print(f"\n{e}")
        raise SystemExit(1)
    print(f"\n已生成：{path}")


if __name__ == "__main__":
    main()

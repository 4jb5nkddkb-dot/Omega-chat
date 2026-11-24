#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""视频页签的渲染器（v0.1.6）：输入一句提示词，生成一段 5 秒 MP4。

做法是本地模板渲染，不是扩散模型——离线买断的游戏跑不进
文生视频模型，也不改模型训练。渲染器先读提示词，认出里面
的人物、地点、动作/氛围词（白天/夜晚），再按认出的内容选画
面与镜头：地点定背景色调与星图焦点，动作定镜头快慢与方向，
白天/夜晚定画面明暗；画面上出现提示词原文、人物名字、地点
和一句旁白（人物的那句台词，或地点的介绍）。什么都没认出来
也照样生成一段（默认封面推近 + 提示词原文当标题），界面上写
清认出了什么、默认了什么，不拒绝、不崩。

依赖：PIL（画帧，只在渲染时 import，没有就明说缺 Pillow）、
ffmpeg（编码 MP4：系统 PATH 里有就用，没有就找游戏目录旁的
ffmpeg.exe；都没有就明说，不静默失败）。视频存在 GUI 目录下
的 videos/ 里，重启还在，界面里能直接打开看。

人物与地点只从 world_state.py 取：6 个人物（孙悟空、唐僧、
猪八戒、沙僧、白龙马、观音），7 个地点（花果山、水帘洞、
高老庄、流沙河、取经路、南海、天宫）。
"""
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

# 画面：1280x720，24fps，5 秒。
WIDTH, HEIGHT = 1280, 720
FPS = 24
DURATION = 5.0

# 人物故乡（没在提示词里写地点时，视频里的“所在地点”）：
# 队伍人物平时跟着游人走，故乡是他最常在的地方；观音永远在南海。
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

# 地点色调：画面整体往这个颜色偏一点（blend 0.14，看得出是
# 哪个地方，又不至于盖住封面）。星图模式下还在轨道上给地点
# 打一个亮点，镜头就围着那个点推近。
LOC_TINT = {"花果山": (46, 120, 60), "水帘洞": (30, 110, 120),
            "高老庄": (150, 110, 60), "流沙河": (40, 90, 160),
            "取经路": (200, 140, 60), "南海": (60, 160, 180),
            "天宫": (220, 190, 110)}
# 地点在星图外轨道上的角度（度，0=右、90=上），7 个地点错开。
LOC_ANGLE = {"花果山": 210, "水帘洞": 150, "高老庄": 250,
             "流沙河": 290, "取经路": 330, "南海": 30, "天宫": 90}

# 动作词 -> 镜头风格。先列的词组优先匹配（比如“打斗”先于“打”）。
# zoom：推近起止（1.00=整框）；drift：源图坐标下左右轻移起止；
# brightness：画面明暗（白天亮、夜晚暗，默认 1.0 不动）。
ACTIONS = [
    (("打斗", "战斗", "打架", "决斗", "交手", "降妖", "斗法",
      "打妖怪"), "fight", "打斗"),
    (("赶路", "行走", "前行", "出发", "启程", "上路", "西行",
      "走"), "walk", "赶路"),
    (("练功", "修炼", "习武", "练剑", "练武"), "train", "练功"),
    (("讲经", "说法", "念经", "诵经", "讲道"), "chant", "讲经"),
    (("休息", "睡觉", "打盹", "歇息", "乘凉", "打坐"), "rest", "休息"),
]
STYLES = {
    # 练功：稳推近，练功要定。
    "train": {"zoom": (1.00, 0.86), "drift": (-16, 16)},
    # 打斗：推得更快、晃得更开。
    "fight": {"zoom": (1.00, 0.78), "drift": (-48, 48)},
    # 赶路：基本不推，横着平移，像在路上走。
    "walk": {"zoom": (0.82, 0.82), "drift": (-140, 140)},
    # 讲经：慢，稳。
    "chant": {"zoom": (1.00, 0.93), "drift": (-8, 8)},
    # 休息：最慢，几乎不动。
    "rest": {"zoom": (1.00, 0.95), "drift": (-6, 6)},
    # 默认：和原来一样，缓慢推近。
    "default": {"zoom": (1.00, 0.88), "drift": (-16, 16)},
}
DAY_WORDS = ("白天", "日间", "中午", "清晨")
NIGHT_WORDS = ("夜晚", "晚上", "深夜", "夜里", "月夜")


class VideoMakerError(Exception):
    """视频生成失败：给界面的提示就是这句话，原样显示。"""


class MissingDependencyError(VideoMakerError):
    pass


def valid_names():
    """小世界里的人物名（顺序按 world_state.py）。"""
    return list(HOME)


def valid_locations():
    """小世界里的地点名（顺序按 world_state.py）。"""
    return list(LOC_EN)


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


def _starfield_base(focus=None):
    """没封面时的底图：按界面星图的画法，画一张 2560x1440 的星图
    （深底、暖光、双轨道、中心核、星星），推近逻辑和封面同一套。
    focus 是地点在外轨道上的点（源图坐标）：给了就在那打一个
    亮点，镜头围着它推近；没给就围着中心核。"""
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
    if focus is not None:
        fx, fy = focus
        dr.ellipse([fx - 60, fy - 60, fx + 60, fy + 60],
                   outline=(127, 246, 255), width=4)
        dr.ellipse([fx - 26, fy - 26, fx + 26, fy + 26],
                   fill=(255, 179, 92), outline=(255, 241, 214), width=3)
    return im, (0, 0, W, H)


def _loc_focus(location):
    """地点在星图外轨道上的点（2560x1440 源图坐标）。"""
    ang = math.radians(LOC_ANGLE.get(location, 0))
    return (1280 + 1050 * math.cos(ang), 720 + 528 * math.sin(ang))


def _first_hit(prompt, words):
    """words 里哪个词先在提示词里出现（按位置，没出现返回 None）。"""
    best = None
    for w in words:
        i = prompt.find(w)
        if i >= 0 and (best is None or i < best[0]):
            best = (i, w)
    return best[1] if best else None


def parse_prompt(prompt):
    """解析一句提示词，返回认出的内容与渲染要用的全部决定。

    只认小世界里的词：6 个人物、7 个地点、5 组动作词、白天/夜晚。
    什么都没认出来也返回一套默认决定（不拒绝），note 里写清
    认出了什么、默认了什么，界面原样显示。"""
    prompt = (prompt or "").strip()
    info = {"prompt": prompt, "characters": [], "party": False,
            "display_name": "", "location": None, "action": None,
            "action_word": None, "time": None, "recognized": False,
            "note": "", "defaults": "", "title": "", "narration": ""}
    if not prompt:
        return info
    # 人物：按在提示词里出现的顺序排。
    hits = []
    for name in HOME:
        i = prompt.find(name)
        if i >= 0:
            hits.append((i, name))
    hits.sort()
    info["characters"] = [n for _, n in hits]
    info["party"] = "师徒" in prompt
    # 地点：取提示词里先出现的那个。
    info["location"] = _first_hit(prompt, list(LOC_EN))
    # 动作：按 ACTIONS 的顺序找，先列的词组优先。
    for words, key, label in ACTIONS:
        w = _first_hit(prompt, words)
        if w is not None:
            info["action"] = key
            info["action_word"] = w
            info["action_label"] = label
            break
    # 白天 / 夜晚：取先出现的那个。
    day = _first_hit(prompt, DAY_WORDS)
    night = _first_hit(prompt, NIGHT_WORDS)
    if day is not None and (night is None
                            or prompt.find(day) < prompt.find(night)):
        info["time"] = "day"
        info["time_word"] = day
    elif night is not None:
        info["time"] = "night"
        info["time_word"] = night
    # 显示名字：唐僧 + 师徒 = 唐僧师徒；只有“师徒”也当唐僧师徒；
    # 多个人物只取第一个当主角。
    chars = info["characters"]
    if chars:
        if info["party"] and chars[0] == "唐僧":
            info["display_name"] = "唐僧师徒"
        else:
            info["display_name"] = chars[0]
    elif info["party"]:
        info["display_name"] = "唐僧师徒"
        info["characters"] = ["唐僧"]
    # 地点：提示词里没写，就用人物的故乡；都没写就空着（默认封面）。
    if info["location"] is None and chars:
        info["location"] = HOME[chars[0]]
    # 旁白：有人物就用他的台词，没人物有地点就用地点介绍。
    try:
        import world_state as ws
        if chars:
            info["narration"] = ws.CHARACTERS[chars[0]]["line"]
        elif info["location"]:
            info["narration"] = ws.LOCATIONS.get(info["location"], "")
    except Exception:
        pass
    # 认出 / 默认：写给界面看的原话。
    parts = []
    if chars:
        parts.append("、".join(chars))
    if info["party"]:
        parts.append("师徒")
    if _first_hit(prompt, list(LOC_EN)) is not None:
        parts.append(info["location"])
    if info["action_word"]:
        parts.append(info["action_word"])
    if info.get("time_word"):
        parts.append(info["time_word"])
    defaults = []
    if chars and _first_hit(prompt, list(LOC_EN)) is None:
        defaults.append(f"地点默认{info['location']}")
    if not chars and not info["location"]:
        defaults.append("背景默认封面")
    if not info["action"]:
        defaults.append("镜头默认慢推近")
    info["recognized"] = bool(parts)
    if parts:
        info["note"] = "认出：" + "、".join(parts)
    else:
        info["note"] = "没认出人物、地点或动作"
    if defaults:
        info["note"] += "；" + "、".join(defaults)
    info["defaults"] = "、".join(defaults)
    # 文件名标题：名字 + 地点 + 动作；什么都没有就用提示词前 8 个字。
    title = (info["display_name"]
             + (info["location"] or "")
             + (info.get("action_label") or ""))
    if not title:
        title = prompt[:8]
    title = re.sub(r"[/\\:*?\"<>|\s]", "", title)[:16] or "小世界"
    info["title"] = title
    return info


def _wrap(text, font, max_w):
    """按字体宽度断行，最多 2 行，超了截断加省略号。"""
    lines, cur = [], ""
    for ch in text:
        if font.getlength(cur + ch) <= max_w:
            cur += ch
        else:
            lines.append(cur)
            cur = ch
            if len(lines) == 2:
                break
    if len(lines) < 2 and cur:
        lines.append(cur)
    if len(lines) == 2 and len(text) > len("".join(lines)):
        lines[1] = lines[1][:-1] + "…"
    return lines


def _frame(im, base_box, t, draw_text, style, center=None,
           tint=None, brightness=1.0):
    """第 t 帧（t 在 0..1 之间）：按镜头风格推近/平移，缩到
    1280x720，调色调与明暗，再画文字层。center 是推近的中心
    （源图坐标，默认基准框中心）；裁剪框夹在基准框内，不越界。"""
    from PIL import Image, ImageDraw, ImageEnhance
    x0, y0, x1, y1 = base_box
    bw, bh = x1 - x0, y1 - y0
    cx, cy = center if center is not None else ((x0 + x1) / 2,
                                                 (y0 + y1) / 2)
    z0, z1 = style["zoom"]
    d0, d1 = style["drift"]
    z = z0 + (z1 - z0) * t
    drift = d0 + (d1 - d0) * t
    cw, ch = bw * z, bh * z
    left = min(max(cx + drift - cw / 2, x0), x1 - cw)
    top = min(max(cy - ch / 2, y0), y1 - ch)
    frame = im.crop((int(left), int(top), int(left + cw), int(top + ch)))
    frame = frame.resize((WIDTH, HEIGHT), Image.LANCZOS)
    if tint is not None:
        overlay = Image.new("RGB", (WIDTH, HEIGHT), tint)
        frame = Image.blend(frame, overlay, 0.14)
    if brightness != 1.0:
        frame = ImageEnhance.Brightness(frame).enhance(brightness)
    draw_text(ImageDraw.Draw(frame, "RGBA"), frame)
    return frame


def render_video(prompt, out_dir=None, location=None, duration=DURATION,
                 fps=FPS, progress=None):
    """按一句提示词生成一段视频，返回 (mp4 路径, 解析结果)。

    空提示词就抛 VideoMakerError（界面接住显示）；提示词里
    什么都没认出来也照样生成（默认封面推近 + 提示词原文当标题），
    不拒绝。没 PIL 或没 ffmpeg 就抛 MissingDependencyError，
    提示缺什么、怎么补。progress(i, n) 每 12 帧调一次，供界面
    刷新“正在生成…”的状态。location 参数留着给调用方强制指定
    地点（不传就按提示词解析）。"""
    from PIL import Image  # noqa: F401 -- 没有 Pillow 就在这里明说
    prompt = (prompt or "").strip()
    if not prompt:
        raise VideoMakerError("先输入一句提示词，比如“孙悟空在花果山练功”。")
    info = parse_prompt(prompt)
    if location is not None:
        info["location"] = location
    ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        raise MissingDependencyError(
            "生成视频需要 ffmpeg：把它放在游戏目录旁（ffmpeg.exe），"
            "或装到系统 PATH 里，再点生成。")
    style = STYLES[info["action"] or "default"]
    brightness = 1.0
    if info["time"] == "day":
        brightness = 1.08
    elif info["time"] == "night":
        brightness = 0.62
    tint = LOC_TINT.get(info["location"]) if info["location"] else None
    cover = find_cover()
    if cover is not None:
        base_im, base_box = _cover_base(cover)
        center = None  # 封面就围着画面中心推
    else:
        focus = (_loc_focus(info["location"])
                 if info["location"] else None)
        base_im, base_box = _starfield_base(focus)
        center = focus

    out = video_dir(out_dir)
    ts = time.strftime("%Y%m%d_%H%M%S")
    target = out / f"视频_{info['title']}_{ts}.mp4"
    n = 2
    while target.exists():
        target = out / f"视频_{info['title']}_{ts}_{n}.mp4"
        n += 1
    total = max(1, int(round(duration * fps)))

    f_name = _load_font(60, bold=True)
    f_loc = _load_font(26)
    f_line = _load_font(30)
    f_top = _load_font(24)
    f_prompt = _load_font(30, bold=True)
    AMBER = (255, 179, 92)
    CYAN = (127, 246, 255)
    WHITE = (217, 246, 255)
    name = info["display_name"] or "小世界"
    loc = info["location"]
    line = info["narration"]
    prompt_lines = _wrap(prompt, f_prompt, WIDTH - 96)

    def draw_text(dr, frame):
        # 顶栏小字 + 提示词原文（黑底条，盖在画面上方）。
        dr.text((48, 36), "OMEGA · 小世界", font=f_top, fill=CYAN)
        if prompt_lines:
            band_h = 16 + len(prompt_lines) * 40
            dr.rectangle([0, 66, WIDTH, 66 + band_h], fill=(0, 0, 0, 130))
            for i, ln in enumerate(prompt_lines):
                # 黑描边 + 白字，压在亮画面上也看得清。
                dr.text((50, 76 + i * 40), ln, font=f_prompt,
                        fill=(0, 0, 0, 255))
                dr.text((48, 74 + i * 40), ln, font=f_prompt, fill=WHITE)
        # 底栏：名字 / 地点 / 旁白。
        dr.rectangle([0, 500, WIDTH, HEIGHT], fill=(0, 0, 0, 170))
        dr.rectangle([0, 500, WIDTH, 503], fill=(255, 179, 92, 255))
        dr.text((64, 522), name, font=f_name, fill=AMBER)
        if loc:
            dr.text((64, 600), f"{loc} · {LOC_EN.get(loc, '')}",
                    font=f_loc, fill=CYAN)
        else:
            dr.text((64, 600), "小世界 · OMEGA WORLD",
                    font=f_loc, fill=CYAN)
        if line:
            dr.text((64, 640), f"“{line}”", font=f_line, fill=WHITE)

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
                           draw_text, style, center=center, tint=tint,
                           brightness=brightness)
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
    return target, info


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt", help="一句提示词，如 孙悟空在花果山练功")
    ap.add_argument("--out", default=None)
    ap.add_argument("--duration", type=float, default=DURATION)
    args = ap.parse_args()
    try:
        path, info = render_video(
            args.prompt, out_dir=args.out, duration=args.duration,
            progress=lambda i, n: print(f"\r帧 {i}/{n}", end="", flush=True))
    except VideoMakerError as e:
        print(f"\n{e}")
        raise SystemExit(1)
    print(f"\n已生成：{path}")
    print(info["note"])


if __name__ == "__main__":
    main()

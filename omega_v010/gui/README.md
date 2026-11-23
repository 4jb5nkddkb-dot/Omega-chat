# Omega 小世界 GUI — Windows 可玩版

`omega_world_gui.py`，纯标准库 tkinter + NumPy，没有别的依赖。

## 运行

1. 电脑上装 Python 3.10+ 和 NumPy（`pip install numpy`）。
2. 准备一个模型产品 `.npz`：Drive 备份里有 `generalist_s57300.npz`
   （Omega-chat-main 文件夹），下载到这个文件夹里就行；更新的产品
   只要是全能线出的 `.npz` 都能用。
3. 双击 `run_gui.bat`（或 `python omega_world_gui.py`），启动时选
   那个 `.npz` 文件。

## 怎么玩

- **走动**：7 个地点按钮，师徒会跟你走，昼夜每 4 步一翻。
- **查看 / 拾取 / 对话**：可见物品和在场人物都是按钮，点就行。
- **问问题**：4 道自态问题按钮，或自己打字问。
- **抽卡**：卡册页签里 60 张卡（人物 30 / 法宝 15 / 地点 15），
  开包抽卡、集卡；卡包只靠玩得来——每日登录 5 包、走路每 4 步
  1 包、到新地点 1 包、取经完成 3 包，上限攒 10 包；1 包 5 张，
  概率 C 55% / R 28% / SR 13% / SSR 4%，每 10 包保底 1 张 SR 以
  上；重复卡变 10 尘（尘先只攒着）。集卡进度和卡包数存在
  `gacha_progress.json`（和 `player_progress.json` 同目录），
  重启还在；不卖包、不联网。
- **视频**：视频页签里输入人物名字（孙悟空、唐僧、猪八戒、沙僧、
  白龙马、观音），点“生成视频”，就出一段 5 秒 MP4：封面图缓慢
  推近，画面上有名字、所在地点和一句台词。视频是本地渲染的
  （PIL 画帧 + ffmpeg 编码），不改模型、不用联网；存在 `videos/`
  里，重启还在，选一个点“打开播放”就能看。名字不在小世界里就
  明说，不生成空视频。生成视频需要装 Pillow（`pip install
  pillow`）和 ffmpeg（放游戏目录旁的 `ffmpeg.exe` 或装到系统
  PATH 里）；缺了哪个，界面都会明说，不会悄悄失败。封面图在
  游戏目录旁放一个 `cover.png` 就用它做背景，没有就画星图。
- **人格**：下拉换 omega / 紫薇 / 悟空 / 一一 / Dayan，意图框写一句
  想干的事，点应用。
- **内核自动玩 12 步**：点“内核自动玩 12 步”，桥接器把每步的接地
  分交给内核，内核逐步选出下一个动作（observe / explore / exploit /
  stabilize），不再固定轮换；点“停”随时停。每一步的动作、接地分
  和回复都记进 `dialog_log.jsonl`（动作名前带 `[内核]`），桥接器那
  边另有 `run_kernel/` 里的审计记录。自动玩需要界面旁有桥接器
  （`omega-world-bridge`）：没有找到时会自动玩会明说，不启动。
- **接地计分**：每句模型回话都当场判对错（和世界状态对不对得上），
  顶上实时显示“答对 x / y”。对话默认存进 `dialog_log.jsonl`
  （可关；写失败会警告一次并自动暂停，不会打断玩）。
- **在线记忆防复读**：答对的（状态、提问、正确回复）会进一个
  容量 3 的小缓存（`agent_runtime.CognitiveMemoryBuffer`），下一轮
  提示最前面自动拼上这些答对示例，帮模型稳住短句直答的格式，
  压一压它复读原著台词的惯性；答错的不进缓存。**只拼与当前状态
  相同的示例**：状态变了，旧示例不拼（s2100 复测：记忆开 4/7、
  关 7/7，旧答案会污染新状态；修后记忆开 7/7）。缓存只活在内存里，
  关掉就没了，不改模型权重。
- **离线微调数据收集**：`dialog_log.jsonl` 每条除输入输出外，还记
  `verdict`（correct/wrong）、`state`（判卷当时的真实标准状态：
  地点、时间步、昼夜、在场人物、可见物品、随身物品、摘要）以及
  `prompt`（实际送入模型的完整输入）、`base_prompt`（去掉记忆
  示例后的原始提示）、`memory_examples`（当轮生效的示例）。导出
  时筛 `verdict == "correct"` 的条目，用 `base_prompt` + `reply`
  + `state` 就能拼对齐语料。

## 诚实说明

- 内核自动玩每一步的动作都是桥接器里的内核选的（工作台上用 Linux
  桥接器验过 12 步，见 `V010_KERNEL_AUTOPLAY_REPORT.md`）；但 Windows
  版桥接器二进制没有随包带——没在真 Windows 上跑过的二进制不发，
  所以 Windows 上点自动玩会提示没有找到桥接器。手动查看、移动、
  拾取和对话不受影响。
- 内核只选动作，不自己改策略（`omega.toml` 里 `self_modify_enabled
  = false`）；每步只放行白名单动作（observe / explore / exploit /
  stabilize / noop）。
- 模型现在世界问答上经常答错地点（12 步 demo 里 12/12 错），
  答对答错都会原样显示——这就是这个 GUI 要量的东西。
- 文件夹是自包含的：`world_state.py`、`world_anchor.py`、
  `world_loop.py`、`world_self_probe.py`、`agent_runtime.py`、
  `dual_transformer_numpy.py` 都是随包带的版本，和工作台验证过的
  组合一致；不要单独混用别处的同名文件。新增的 `kernel_autoplay.py`
  是纯标准库的桥接器驱动（启动桥接器进程、读写交换文件），`omega.toml`
  是桥接器配置。新增的 `gacha.py` 是 60 张卡池 + 开包经济（纯标准库），
  卡册数据和开包概率都在里面；`gacha_progress.json` 是运行中生成的
  集卡进度，不进仓库。

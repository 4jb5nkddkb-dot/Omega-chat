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
- **人格**：下拉换 omega / 紫薇 / 悟空 / 一一 / Dayan，意图框写一句
  想干的事，点应用。
- **自动玩 12 步**：按内核的四个动作（observe / explore / exploit /
  stabilize）轮着走，看着它自己玩；点“停”随时停。
- **接地计分**：每句模型回话都当场判对错（和世界状态对不对得上），
  顶上实时显示“答对 x / y”。对话默认存进 `dialog_log.jsonl`
  （可关；写失败会警告一次并自动暂停，不会打断玩）。

## 诚实说明

- 自动玩的动作轮换是按内核动作形状排的，还**不是**真正的内核在选
  （真正的桥接器是 `omega-world-bridge`，Linux 二进制在 v0.1.0
  Release 里；Windows 版以后再编——没在真 Windows 上跑过的二进制
  不发）。
- 模型现在世界问答上经常答错地点（12 步 demo 里 12/12 错），
  答对答错都会原样显示——这就是这个 GUI 要量的东西。
- 文件夹是自包含的：`world_state.py`、`world_anchor.py`、
  `world_loop.py`、`world_self_probe.py`、`agent_runtime.py`、
  `dual_transformer_numpy.py` 都是随包带的版本，和工作台验证过的
  组合一致；不要单独混用别处的同名文件。

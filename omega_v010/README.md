# Omega v0.1.0 — 小世界闭环

三件套第一次拼成一个能跑的产品：

- **认知**：omega-cognitive-kernel v0.4.7（Rust，API 已冻结，pin 在 tag v0.4.7）
- **大脑**：全能型小强通用模型（d512，最新产品，见 `out/generalist_s*.npz`）
- **环境**：西游记小世界（`world_state.WorldState`：7 地点、6 人物、6 物品）

一条命令跑起来：

```bash
./run_v010.sh 8        # 8 步闭环
```

两个进程协同：Python sidecar（世界＋模型）与 Rust bridge（内核认知），
用 `run/` 里的三个 JSON 文件握手（`observation.json` / `action.json` /
`outcome.json`）。跑完 `run/` 里留着 `state.json`、`audit.jsonl`、
`trajectory.jsonl`、`checkpoints/`，用发布的 `omega-kernel` 二进制就能看：

```bash
../omega_kernel/bin/omega-kernel-linux-x86_64 --state run/state.json status
../omega_kernel/bin/omega-kernel-linux-x86_64 --state run/state.json audit --summary
```

## 对接是怎么做的（诚实版）

先排除了两条看似能走的路：

1. **发布的 CLI 跑不了这个循环。** `omega-kernel run` 永远用内置的
   模拟观察器（正弦波），`--config` 只能改阈值和开关，读不到外部文件。
   FileObserver / LoggingExecutor 是库接口，不是 CLI 功能。
2. **观察内容本身进不了内核的误差。** 内核的误差 = 预测不确定度 +
   动作惩罚，观察 JSON 只进轨迹存档（可审计），不参与误差计算；
   `Kernel` 的类型参数也只开放 Observer / Executor，Predictor 固定
   用内置的 `ModelBasedPredictor`。

所以 v0.1.0 用的是内核的公开 API 做的**小 Rust 桥接器**
（`src/main.rs`，约 200 行）：

- `FileWorldObserver`：读 sidecar 写的 `observation.json`（世界摘要 +
  模型回话 + 接地误差分）；
- `FileWorldExecutor`：把内核选出的动作（observe / explore /
  exploit / stabilize 四选一）写进 `action.json`；
- 每步之后，bridge 等 sidecar 的 `outcome.json`，把**接地误差**
  换算成奖励，用 `record_reward(action, 1 - grounded_error)` 喂回
  策略记忆——这就是接地进来的地方：答对了（回话与世界状态一致）
  对应策略权重涨，答错了权重跌，下一步决策就跟着变。

sidecar 的接地分就是现成的东西：`world_state.validate()`（叙述与
状态矛盾就打标记）加上 `world_self_probe.score()`（按问题逐项判：
地点、人物、随身、昼夜）。0 分 = 回话和世界对得上，1 分 = 矛盾。

## 安全默认

- `self_modify_enabled = false`（`omega.toml`）。L1 不生成也不提交
  任何补丁；循环靠策略权重、case 记忆和 L2 软调节学习，全部有审计。
  开 self_modify 以后由用户定。
- 桥接器只执行白名单里的四个动作；世界里的移动/拾取/对话都是
  `WorldState` 自己的工具，没有别的副作用。
- 每次补丁前内核自动做 checkpoint，`rollback` 随时可回滚
  （`omega-kernel rollback <id>`）。

## 模型不在仓库里

权重（约 195MB）不进 git：sidecar 默认用工作台最新的
`out/generalist_s*.npz`；对外发布时 README 指向 Drive 备份。

## 还没做的事

- L1 自改策略（关着，等用户开）；
- dashboard（内核自带 `omega-dashboard`，另起）；
- 手机打包（Pythonista 壳另算）；
- 第 202 块后的全面评估与 Drive 备份（训练线照常走）。

## Windows 可玩版（gui/）

`gui/` 是给 Windows 电脑准备的 tkinter 界面：走动、查看、拾取、对话、
问问题、换人格、自动玩 12 步，每句回话当场判对错并计分，对话存进
`dialog_log.jsonl`。纯标准库 tkinter + NumPy，双击 `run_gui.bat`，
启动时选一个模型产品 `.npz`（Drive 备份里的 `s57300` 就行）。
详见 `gui/README.md`。自动玩的动作轮换是按内核动作形状排的，还不是
真正的内核在选——Windows 版桥接器以后再编，没在真 Windows 上跑过的
二进制不发。

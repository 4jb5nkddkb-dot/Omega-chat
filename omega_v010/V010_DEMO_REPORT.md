# Omega v0.1.0 Demo 报告（2026-11-12）

跑的什么：内核 v0.4.7（pin tag）＋ 全能模型 s58200 ＋ 小世界，
`./run_v010.sh 12`，12 步闭环，约 30 秒。`run/` 里留着全部证据
（`state.json`、`audit.jsonl`、`trajectory.jsonl`、两个 log）。

## 数字

- 12 步全跑完，无崩溃、无超时、无握手失败（每步生成约 2 秒）。
- 接地误差：**12/12 错**（`grounded_err=1.00`，reward 全 0）。
  每次回话都提了错地点（花果山 / 水帘洞 / 天宫 / 高老庄乱炖），
  真地点是取经路——这就是已知的世界问答毛病（答错地点），
  现在它在循环里被量出来了，不再是感觉。
- 内核内部误差 `avg_err`：0.068 → 0.551（一直在涨，内核知道情况在变糟）。
- 策略权重：stabilize 1.0 → **0.1（地板）**，observe 在第 6 步被试了一次
  （也错，权重 0.68）。内核确实在根据接地反馈改决策，不是摆设。
- L1 补丁：0 个——`self_modify_enabled=false`，每步审计里记的都是
  `patch_rejected: self_modify_enabled=false`。L2 从未进定点（align 0.000）。
- 四条动作分支全部单独验过：observe（问问题）、explore（移动＋叙述，
  取经路→水帘洞）、exploit-拾取（“你拾起了锦襕袈裟。”）、exploit-对话、
  stabilize（轮换问题）。回放固定：问题每 4 步一轮回，同一问题回话
  逐字相同（贪心解码＋世界没变）。

## 结论（诚实版）

- **循环通了，测量通了，模型没过。** 桥接器、sidecar、审计、回滚
  全部按设计工作；模型在世界问答上 12/12 错，内核的策略权重跟着掉，
  但因为自改关着，damping  stays 1.0，什么也没真正被改掉。
- **世界一次也没变。** 高不确定度下内核只选 stabilize / observe，
  explore 从没赢过——模型越错，内核越不敢动，世界越不变，问题越重复。
  这是设计里的保守性，不是 bug；要让世界动起来，要么等模型答对
  （不确定度下来），要么以后由人开 self_modify。
- 下一步（训练侧）：世界问答是当前最短的板，12/12 的量就是修它的
  靶子；循环侧：开 self_modify 是下一个显式决策，不在这版做。

## 怎么复现

```bash
./run_v010.sh 12
../omega_kernel/bin/omega-kernel-linux-x86_64 --state run/state.json status
../omega_kernel/bin/omega-kernel-linux-x86_64 --state run/state.json audit --summary
```

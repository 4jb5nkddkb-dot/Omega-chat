# Omega-chat

本地小型智能体的练手仓库：纯 NumPy 双头 Transformer（《西游记》语料）+ Omega 门控 + agent 运行时。
本仓库为 source of truth。双头为基准；modern 旧架构归档在 `legacy/`，仍可运行，但不再是主力。

## 布局（标准包布局）

```
src/omega_chat/             包：双头 Transformer + agent 运行时 + 记忆 + 世界状态 + 治理 + 精简内核
  dual_transformer_numpy.py 双头 Transformer（NumPy 参考实现）+ 解码控制
  agent_runtime.py          闭环 agent 运行时：Protocol + ToolRouter + run_agent
  memory_store.py           长期记忆：append-only JSONL 存储 + 记忆工具
  world_state.py            小世界状态机：地点 / 人物 / 物品 / 时间
  omega_governance.py       治理层：ToolGate / MemoryGate / CheckpointGate / 审计
  omega_kernel_runnable.py  精简 Omega 内核（纯标准库，Pythonista 可跑）
tests/                      全部测试（dual_runtime / agent_governance /
                            world_state / omega_governance）
tools/                      make_tool_data.py（协议 SFT 数据生成）、
                            align_check.py（生成复读率 A/B）、train_dual.py
legacy/                     modern 旧架构：modern_transformer_numpy.py、
                            train_modern.py、sft_train.py（modern 配方，
                            不能跑在 dual 基座上）
docs/                       OMEGA_UNIFIED_DESIGN.md（治理层统一设计）
data/                       Xiyouji.txt（训练语料，677,421 字，字表 4,510）
checkpoints/                检查点（见下）
archive/                    旧代码与实验残留（omega.py、app.py、Ui.py、
                            GGUF.py 等）+ 历史 boot 日志，只移动不删除
```

```
checkpoints/
  base/                     dual 参数-only 检查点的 base64 分片
    dual_d128_s69000_params.npz.b64.part00-05
  world4/                   世界 SFT 第四轮产物的 base64 分片
    dual_d128_sft_world4.npz.b64.part00-05
```

暂留根目录（推送通道推不了二进制，下次 PC 端 git 推送时再移入
checkpoints/ 与 archive/；找检查点时先查 checkpoints/ 再查根目录，
功能不受影响）：
```
modern_ckpt.npz             注意：文件名是旧名字，里面现在是 dual 权重
                            （d128，step 69000，best val 4.6361，含优化器状态）
xiyouji_modern_trainer.zip / xiyouji_modern_trainer_1.zip
                            早期训练包存档（应进 archive/）
dual_d128_sft_world9.npz    世界 SFT 第九轮产物（世界循环当前产物，全部验收门通过；
                            应进 checkpoints/，目前仅本地）
```

权重是二进制文件，不走文本推送通道；分片是文本，可直接推送。

## 权重分片拼回

```
cat checkpoints/base/dual_d128_s69000_params.npz.b64.part* > dual_d128_s69000_params.npz.b64
base64 -d dual_d128_s69000_params.npz.b64 > dual_d128_s69000_params.npz
cat checkpoints/world4/dual_d128_sft_world4.npz.b64.part* > dual_d128_sft_world4.npz.b64
base64 -d dual_d128_sft_world4.npz.b64 > dual_d128_sft_world4.npz
```

## 测试

```
python3 tests/dual_runtime_test.py
python3 tests/agent_governance_test.py
python3 tests/world_state_test.py
python3 tests/omega_governance_test.py
python3 src/omega_chat/agent_runtime.py --test
python3 src/omega_chat/memory_store.py --test
```

`--test` 与 `--console` 用 `select_checkpoint` 找检查点：每个名字先查
`checkpoints/`，再查根目录，所以平铺目录也能跑。

## 手机（Pythonista）平铺兼容

仓库是标准包布局，但手机端仍可平铺运行：把 `src/omega_chat/` 下的
文件与 `data/Xiyouji.txt`、检查点放在同一目录直接运行即可。
`agent_runtime.py` 在包外运行时自动回落到同目录 sibling 导入；
`select_checkpoint` 按 `checkpoints/` → 根目录的顺序找检查点；
语料优先读 `data/Xiyouji.txt`，没有则读同目录 `Xiyouji.txt`。
legacy 脚本运行时日志仍写根目录 `boot_log.txt`（不提交）。

## 约定

- 文件名一律用下划线（Python import 的要求）；不要用带空格的文件名
- 工具协议保留字符只用西游记字表内字符：`《》】〔〕（）`（语料里没有半角括号、引号和 ASCII 标记）
- 训练工作台（dual_sft/ 下的各轮训练脚本与产物）不在本仓库里，只推评审过的产物

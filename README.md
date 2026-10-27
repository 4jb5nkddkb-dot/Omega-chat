# Omega-chat

本地小型智能体的练手仓库：纯 NumPy 双头 Transformer（《西游记》语料）+ Omega 门控 + agent 运行时。
本仓库为 source of truth。双头为基准；modern 旧架构归档在 `legacy/`，仍可运行，但不再是主力。

## 布局

```
dual_transformer_numpy.py   双头 Transformer（NumPy 参考实现）+ 解码控制
agent_runtime.py            闭环 agent 运行时：Protocol + ToolRouter + run_agent
memory_store.py             长期记忆：append-only JSONL 存储 + 记忆工具
world_state.py              小世界状态机：地点 / 人物 / 物品 / 时间
omega_governance.py         治理层：ToolGate / MemoryGate / CheckpointGate / 审计
omega_kernel_runnable.py    精简 Omega 内核（纯标准库，Pythonista 可跑）
Xiyouji.txt                 训练语料（677,421 字，字表 4,510）
modern_ckpt.npz             注意：文件名是旧名字，里面现在是 dual 权重
                            （d128，step 69000，best val 4.6361，含优化器状态）
dual_d128_s69000_params.npz.b64.part00-05
                            dual 参数-only 检查点的 base64 分片，拼回方法见下
dual_d128_sft_world4.npz.b64.part00-05
                            世界 SFT 第四轮产物（clean 单段 2000 步，val 4.7059，
                            各轮世界产物中 val 最好），base64 分片，拼回方法见下
xiyouji_modern_trainer*.zip 早期训练包存档
tests/                      全部测试（dual_runtime / agent_governance /
                            world_state / omega_governance）
legacy/                     modern 旧架构：modern_transformer_numpy.py、
                            train_modern.py、sft_train.py（modern 配方，
                            不能跑在 dual 基座上）
tools/                      make_tool_data.py（协议 SFT 数据生成）、
                            align_check.py（生成复读率 A/B）、train_dual.py
docs/                       OMEGA_UNIFIED_DESIGN.md（治理层统一设计）
archive/                    旧代码与实验残留（omega.py、app.py、Ui.py、
                            GGUF.py 等），只移动不删除
```

根目录保持平铺，是因为手机（Pythonista）和 Windows 双端都是同目录平铺运行，
且多个脚本按文件名直接引用 `Xiyouji.txt` / `modern_ckpt.npz`；权重是二进制文件，
也不走文本推送通道，所以权重和语料留在根目录不动。

## 权重分片拼回

```
cat dual_d128_s69000_params.npz.b64.part* > dual_d128_s69000_params.npz.b64
base64 -d dual_d128_s69000_params.npz.b64 > dual_d128_s69000_params.npz
cat dual_d128_sft_world4.npz.b64.part* > dual_d128_sft_world4.npz.b64
base64 -d dual_d128_sft_world4.npz.b64 > dual_d128_sft_world4.npz
```

## 测试

```
python3 tests/dual_runtime_test.py
python3 tests/agent_governance_test.py
python3 tests/world_state_test.py
python3 tests/omega_governance_test.py
python3 agent_runtime.py --test    # 需要 modern_ckpt.npz 与 Xiyouji.txt 在同目录
python3 memory_store.py --test
```

## 约定

- 文件名一律用下划线（Python import 的要求）；不要用带空格的文件名
- 工具协议保留字符只用西游记字表内字符：`《》】〔〕（）`（语料里没有半角括号、引号和 ASCII 标记）
- 训练工作台（dual_sft/ 下的各轮训练脚本与产物）不在本仓库里，只推评审过的产物

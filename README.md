# Omega-chat

本地小型智能体的练手仓库：纯 NumPy Transformer（《西游记》语料）+ Omega 门控 + agent 运行时。
本仓库为 source of truth；手机（Pythonista）和 Windows 双端都是同目录平铺运行，所以文件全部放在根目录。

## 文件地图

### 模型与训练
- `dual_transformer_numpy.py` — 双头 Transformer（NumPy 参考实现），含解码控制（DecodeController）
- `modern_transformer_numpy.py` — 现代架构（GQA + SwiGLU + RMSNorm + RoPE）
- `train_modern.py` — 主力训练器：优化器状态真续训、架构守卫、原子保存；位置参数 `python train_modern.py [增训步数] [d_model] [resume]`
- `GGUF.py` — 现代架构训练 + GGUF 导出（单文件）
- `app.py` — Omega 门控训练接入层（TrainWorld / TrainPolicy / run_cycle）
- `app_ui.py`、`Ui.py` — Pythonista 图形界面
- `omega.py` — Omega 门控内核（observe → predict → authorize → execute）
- `omega_mini.py` — 最小闭环 + 主人权限

### Agent 层
- `agent_runtime.py` — 闭环 agent 运行时：Protocol + ToolRouter（文法约束工具名）+ run_agent（回滚重试）
- `memory_store.py` — 长期记忆：append-only JSONL 存储 + 记忆工具
- `make_tool_data.py` — 工具/记忆协议 SFT 数据生成（输出 `protocol_sft.jsonl`）
- `omega_transformer_bridge.py` — Omega 与 Transformer 的桥接

### 检查与基准
- `check_modern_grads.py` — 现代架构数值梯度检查
- `align_check.py` — 生成复读率 A/B（无控制 vs 解码控制）
- `d5_benchmark_numpy.py` — 单分支 vs 双分支平均基准

### 数据与产物
- `Xiyouji.txt` — 训练语料（677,421 字，字表 4,510）
- `modern_ckpt.npz` — 主检查点（d64；累计 54,950 步，best val 4.6876，见 `boot_log.txt`）
- `boot_log.txt` — 设备运行日志
- `xiyouji_modern_trainer.zip` — 早期训练包存档
- `步数`、`补丁` — 笔记

## 测试

```
python3 dual_transformer_numpy.py --test
python3 check_modern_grads.py
python3 agent_runtime.py --test    # 需要 modern_ckpt.npz 与 Xiyouji.txt 在同目录
python3 memory_store.py --test
```

## 约定
- 文件名一律用下划线（Python import 的要求）；不要用带空格的文件名
- 工具协议保留字符只用西游记字表内字符：`《》】〔〕（）`（语料里没有半角括号、引号和 ASCII 标记）

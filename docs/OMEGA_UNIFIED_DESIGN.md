# Omega 收一：统一运行时治理层设计笔记

## 结论（先说）

新建一个模块 `omega_governance.py`，只做**运行时 / 生命周期治理**，不碰训练。
旧文件（`omega.py`、`app.py`、`omega_mini.py`、`omega_transformer_bridge.py`）**原地不动**，
不删除、不改行为；已验证的 `sft_train.py`、`agent_runtime.py` 也不动。
新层不依赖旧文件（旧文件之间互相引用的缺失模块与本层无关，见下）。

## 现状调查（grep / 结构浏览，未通读 8454 行全文）

| 文件 | 行数 | 实际职责 | 治理相关部分 | 处置 |
|---|---|---|---|---|
| `omega.py` | 8454 | roles-v2 扁平化单核：观察→预测→校准→决策；角色议会（Yiyi/Ziwei/Wukong/Dayan，仅建议）、拓扑 digit-root、共识节点、Vault、异步 reactor、LLM 桥、技能、规划、目标树、记忆引擎 | `Boundary`（动作白名单）、`Owner` + `TokenBook`（一次性令牌、TTL、按 suggestion/epoch 绑定）、`AuditChain`（哈希链、可 HMAC）、`StateMachine`、`ToolRegistry`（白名单/需确认/基础风险）、`_ExecutionMixin` 的 approve→authorize→execute、`seal_record`、`atomic_write` | 只借鉴模式，不复用代码：该核的执行路径围绕 Suggestion/epoch/World 接口，太重；角色议会、拓扑、共识、Vault、规划等与治理无关，全部不搬 |
| `app.py` | 333 | 训练层门控：`TrainWorld` / `TrainPolicy` / `run_cycle`，训练块按验证 loss 接受/回滚，连续回滚则学习率减半 | 快照/恢复（`keep`/`restore`）、接受-拒绝 + 打印审计 | **不搬门控本身**：这是已被实测证伪的模式（2000 步 best 6.0840 后 23 周期全 HOLD，绕开后继续改善），且用户已定方向训练不设门。只借鉴“快照/恢复”与“接受-拒绝留审计”两个思想，改用于治理状态回滚。另外注意：`app.py` 导入 `omega_baseline_v1` 与 `transformer_core`，这两个模块在镜像里不存在，`app.py` 在此已无法导入——它是死代码，不是可复用资产 |
| `omega_mini.py` | 111 | 最小闭环 + 主人权限：pbkdf2 主人密钥（`omega_owner.json`）、一次性令牌、按预测误差的 HOLD/APPLY/ESCALATE、暂停开关 | 主人密钥模式、一次性令牌、暂停开关 | 借鉴模式，修正缺陷：原文件用相对路径（依赖 cwd）、全局可变状态、交互式 CLI、无审计落盘；新层全部改掉 |
| `omega_transformer_bridge.py` | 67 | 角色议会经 `DualTransformerNP` 做前向推理（`ByteTokenizer` 在镜像 dual 文件里存在；`Omega/Config/ReferenceWorld/Owner/SmoothPolicy` 在 `omega.py` 里都存在，`SmoothPolicy` 是 `AdaptivePolicy` 的别名） | 无——纯建议性推理（输出 token + 置信度），不授权任何事 | 不纳入治理层；只作为“角色建议、治理决策”的分工例证 |
| `xiyouji_modern_package/release_gate.py`（先前产物） | — | 检查点放行：5 项检查 + `releases/` 永不覆盖复制 + `release_log.jsonl` 审计 + 退出码 0/1/2 | 5 项检查的完整模板（结构完整、词表一致、记录诚实、确有进步、生成不退化） | 直接作为放行门的设计模板，适配 dual `meta_json` 格式（见下） |

另：`Ui.py` 与 `app_ui.py` 引用 `omega` / `app`，但它们依赖的缺失模块与本层无关，
不在本次收敛范围；已验证路径（dual / memory_store / agent_runtime / align_check）
均不导入任何 Omega 旧文件，因此**不需要兼容垫片**。

## 统一层设计：`omega_governance.py`（标准库 + NumPy，无 pip 依赖，Pythonista 可跑）

分工原则：**治理只做决策与审计，不执行具体业务**。调用方先问治理层“能不能做”，
拿到令牌后再自己执行，执行结果回填审计。训练 chunk 级门控明确不在内。

### 1. 检查点放行门 `CheckpointGate`（模板来自 release_gate.py，适配 dual 格式）

输入检查点路径（dual `meta_json` 格式：`format_version`、`config`、`step`、
`extra.best_val_loss`、`p_*` 参数数组）。五项检查：

1. **结构完整**：`format_version == 1`；`config` 含 `d_model/n_blocks/n_heads/vocab_size`；
   参数名与层数对得上；`emb` 形为 `(vocab, d)`；每块 `qkv/proj/ff` 形状可由
   `d/n_heads/n_kv_heads/ff_mode` 推导；全部权重有限（无 NaN/Inf）。
   任一失败 → `BROKEN`（检查点本身损坏，不可放行）。
2. **词表一致**：`meta` 内字符表长度 == `vocab_size`，且与调用方给定的参考字符表一致。
   未给参考表时此项记为跳过（skip），不算失败。
   特殊位规则（复核修正 2026-10-24）：本项目约定 `vocab_size` = 语料字 + 少量特殊位
   （如 4510 = 4507 + 3），而记录的字符表只含语料字；结构检查已验证 `emb` 行数
   == `vocab_size`，因此缺口 <= 16 视为特殊位而非损坏（审计中注明），缺口 > 16
   或字符表长于 `vocab_size` 仍判失败。真实 d128 检查点（4507/4510）即由此规则放行。
3. **记录诚实**：调用方若提供重算函数（`recompute_val(path) -> float`），要求
   `|重算 - 记录| <= 0.10`；未提供时记为跳过。`require_honesty=True` 时跳过也判 HOLD。
4. **确有进步**：从统一审计日志里找已放行版本的最佳 val，要求新 val 优于它至少
   `margin`（默认 0.05）；首次放行只要求 `val < 8.0`。
5. **生成不退化**：调用方若提供生成函数（`generate(prompt) -> str`），从“悟空”
   温度 0.8 采样 80 字，出现最多的单个字占比 `<= 0.55`；未提供时记为跳过。

裁决：结构失败 → `BROKEN`；其余任一失败（或要求诚实而诚实被跳过）→ `HOLD`；
否则 `RELEASE`。放行时复制到 `releases/<名>_step<步数>_val<loss>.npz`，
目标已存在则加数字后缀——**永不覆盖、永不删除**。每次评估与放行都写审计。

### 2. 工具与动作授权 `ToolGate` + `Policy`

- `Policy`：动作白名单（`Boundary` 的思想）、每动作 `(risk, requires_confirm)`、
  全局暂停开关（`omega_mini` 的暂停思想：暂停时除审计读取外一律 HOLD）。
- `authorize(action, args)`：不在白名单 → `DENY`；需确认的动作 → 发一次性令牌
  （绑定动作 + 参数哈希 + TTL，默认 300 秒，`omega.py` 的 `token_ttl_sec` 同值）；
  低风险动作 → 直接 `ALLOW`，但仍审计。
- `execute(token, callable, args)`：令牌无效 / 过期 / 已用 / 动作或参数不符 →
  拒绝并审计；否则执行可调用对象，成功与异常都审计。
- 默认动作表对齐 `agent_runtime` 的工具名：`计算`（低风险，直接放行）、
  `回忆`（低风险，直接放行）、`记住`（写记忆，需确认）。调用方可注册自己的动作表。

### 3. 记忆写入授权 `MemoryGate`

在 `memory_store` 的既有规则上只做门，不复制存储逻辑：
长度上限（默认 500 字，与 `memory_store` 一致）、协议保留字符
（默认 `《》】〔〕（）`，与 `memory_store.RESERVED_CHARS` 一致）不得出现、
暂停时拒绝。调用方仍用 `MemoryStore.add` 真正写入；门只裁决并审计。
`记住` 动作走 `ToolGate` 的需确认路径时，记忆门做第二道内容检查。

### 4. 审计日志 `AuditLog`

JSONL 追加写，每条含 `ts/kind/actor/action/decision/detail/prev_hash/hash`，
`hash` 为记录体（不含 hash 本身）的 SHA256，设 key 时为 HMAC-SHA256
（`omega.py` 的 `AuditChain` 思想，简化为单文件实现）。
`verify_chain()` 重算全链，可发现任何篡改或缺失。
放行门、工具门、记忆门、回滚的操作全部经过同一本审计，不另起日志文件。

### 5. 回滚 `Rollback`

- `snapshot(paths)`：把治理相关的文件复制到 `snapshots/<id>/` 并写
  `manifest.json`（每个文件的 SHA256）；`restore(snapshot_id)` 先验哈希再拷回，
  全程审计；`list_snapshots()` 列出可用快照。
- 放行回滚：**不删除已发布文件**，只用 `supersede(release_name, reason)` 在审计里
  把某版本标记为已取代；“当前版本”以审计日志里最后一条 RELEASE 为准。
- 本层不做训练回滚：不保存优化器状态、不恢复训练步数、不碰任何检查点内容。

## 明确不做

- 训练过程的 chunk 级门控、按 loss 接受/回滚训练步、学习率衰减循环（`app.py` 模式）。
- 角色议会、LLM 桥、共识、Vault、拓扑、规划、目标树（留在 `omega.py` 里不动）。
- 改动 `sft_train.py`、`agent_runtime.py` 的行为；删除任何仓库文件；推送到 GitHub。

## 文件清单（本次新增）

- `OMEGA_UNIFIED_DESIGN.md`（本文件）
- `omega_governance.py`（统一层实现）
- `omega_governance_test.py`（五项能力的断言测试，`python3 omega_governance_test.py` 运行）

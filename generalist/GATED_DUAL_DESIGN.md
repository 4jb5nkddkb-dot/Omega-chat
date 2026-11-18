# 双半脑门控设计（generalist-v3 → v5）

2026-11-16，用户定：先搭双半脑，门要能决断左还是右输出，最理想是按百分比、
输出和输入都用同一个百分比；百分比动态调整；搭好脑子后直接训 1024。

## v4 改动：每序列一个门（2026-11-17，用户已批准）

v3 的每 token 门已废弃，原因见诊断 `GATE_DIAGNOSIS_S50.md`：门 `g =
sigmoid(emb[token]·W + b)` 只是单个词的查表值，不含位置（RoPE 配置下连
位置项都没有）、不含上下文、不含序列信息；域是整段话的属性，门却只能
按词查表，BCE 监督在共享词上自相矛盾，门在数学上就不可能分化。实测：
step 50 探针 6 域全约 0.51–0.55（分离 0.0155），全词表 4622 个词的 g
都在 [0.4625, 0.5984]，没有一个接近 0 或 1；学到的门功能上就是常数 0.5。

v4 的门：每序列一个数 g（0~1），从序列的平均词向量算出，
`g_seq = sigmoid(w · mean_pool(emb) + b)`，同一序列内所有 token 共用
一个 g；输入分流与输出混合仍用这个 g（广播到 (B, S, 1)）。监督本身
已经是每序列一个目标，门与监督终于在同一粒度上。门参数形状不变
（`gate.weight (1, D)`、`gate.bias (1,)`），recipe 改为 `generalist-v4`，
旧 v3 检查点（step-50 _last）不续跑，新线从头训。

## v5 改动：上下文 hidden-state 门（2026-11-17，用户已批准，选 C）

v4 的 25 步 pilot（2026-11-17 ~10:08 跑完，loss 9.2928 → 6.14）门仍
未分化（分离 0.0047，左 0.5103 vs 世界 0.5057），诊断见
`GATE_DIAGNOSIS_V4_S25.md`：不是设计证伪——池化 Fisher 分离从 s3 的
4.34σ 升到 s25 的 6.45σ，准确率仍 1.000，信号在变强；失败是量级/Adam
预算问题：`||x_pool||` ≈ 0.158，过关要 z 差 0.81，需 `||W||` 5.26
（现有 0.77 的 6.9 倍），25 步 cosine 的 LR 总和只有 0.0133，z 差
最多长 0.066（需要的十二分之一）；满块 300 步上限也恰好卡线
（0.811），加 7:1 配比不平衡与 LM 梯度对抗，直接开满块大概率也够
不着；提 gate-loss-weight 修不了（Adam 对梯度幅值不变）。

v5（用户 2026-11-17 11:14 选 C）：门从共享 stem 的隐状态算。架构：
前 k 块（默认 k=1）为共享 stem，两支共用；门 `g_seq = sigmoid(w ·
mean_pool(h_stem) + b)`，同一序列内所有 token 共用一个 g；输入分流
（`g·h_stem` / `(1-g)·h_stem`）与输出混合仍用这个 g。分支各剩
n_blocks − k 块（d1024/4 块：stem 1 + 每支 3）。门参数形状不变
（`gate.weight (1, D)`、`gate.bias (1,)`），recipe 改为
`generalist-v5`，旧 gated 检查点（v3/v4）参数名已变，不续跑，新线
从头训。

k=1 的取舍：stem 第 1 块已给门完整的因果注意力上下文（每位置看
到全部前缀，池化后即序列级上下文表示）——v4 的词向量池化缺的正是
上下文；第 2 块 stem 只多一轮注意力混合，边际收益小，却要把每支
从 3 块砍到 2 块，域内计算损失是实的。计算量 7 块/token（1 stem +
3 + 3），约 v4 的 7/8，步耗时略降。res_scale 仍按总深度 n_blocks 计
（任一条路径都走 n_blocks 块），初值尺度与 v4 一致。

反向（关键）：门输入来自 stem，gate→stem 的反传路径必须补上——
`dz_seq` 经池化（除以 S）加到 stem 隐状态的梯度上，再与分支输入
的梯度相加后一起走 stem 反向；漏了它，stem 参数收不到门的梯度，
gate.weight 的梯度也会错（v4 的门→x 路径同理，旧的每 token 门根本
没传这条路）。增量解码：池化的 stem 隐状态和与 token 数存在 KV
cache 里随 pasts 传递（stem 隐状态本身因果，池化和精确）；分支
KV 仍在前缀门下建，`kv_cache_parity_test` 对 gated 只比最后位置，
容差 1e-5。

## 门：一个百分比，动态，输入输出同逻辑

- 每序列一个数 g（0~1），由序列的平均 stem 隐状态实时算出（v5）：
  `g_seq = sigmoid(mean_pool(h_stem) @ Wg + bg)`，形状 (B, 1, 1)，
  同一序列内所有 token 共用。不是写死的 30/70——每个序列的百分比
  都不一样，训练中门自己学、百分比跟着变。（v4 是平均词向量，
  已废弃：`||x_pool||` ≈ 0.16 太小，Adam 预算推不动，见
  `GATE_DIAGNOSIS_V4_S25.md`；v3 是每 token 门，见 `GATE_DIAGNOSIS_S50.md`。）
- 输入：`x_left = g * x`，`x_right = (1 - g) * x`（按百分比把输入分给两支）。
- 输出：`merged = g * left_out + (1 - g) * right_out`（同一个 g 混合输出）。
- 起步 g≈0.5（bias=0，两边平分），不预设偏向。
- 增量解码（`forward_cached` / `generate_cached`）：门要对"已看到的全部
  上下文"池化，所以 stem 隐状态和与 token 数存在 KV cache 里随 pasts
  传递；否则每步只看到一个 token，门就退化回每 token 查表。注意：每
  序列门本质上是非因果的（全序列池化），所以全序列前向与增量前向在
  早期的位置上本就不一致——`kv_cache_parity_test` 对 gated 只比最后
  一个位置（增量前缀即全序列），容差 1e-5；`generate` 与
  `generate_cached` 必须逐 token 完全一致（已验证）。stem 的 KV 是精
  确的（stem 在门之前），只有分支 KV 在前缀门下漂移。

## 分工：分开喂，门有监督

- 左支（推理）：经典 / 代码 / 对话 / 常识 / 推理。
- 右支（行动）：世界域（工具记录、自报、探针格式、寄存器、做视频经验）。
- 门的监督：batch 里每条序列带域目标 t（左支域 1.0，世界域 0.0），
  BCE loss，权重 `--gate-loss-weight`（默认 1.0），计入总 loss。
  没有这个监督，门没有理由按域分——之前 average 试点就是同一份数据喂
  两边，只多 1.2%，代价 1.84 倍，那不是半脑。
- 注意：世界域只占配比 15%，右支吃得少。先跑起来看门分不分得开，
  不够再提世界配比（不要一次改两件事）。

## 实现位置

- `dual_transformer_numpy.py`（三份 byte-identical，md5 同步）：
  merge 方法新增 `"gated"`；门参数 `gate.weight (1, D)` + `gate.bias (1,)`；
  `_forward` 算 g、分输入、合输出，返回第 7 项 gcache；
  `loss_and_grads` 加 `gate_targets` / `gate_loss_weight`，门梯度走两路
  （输出混合一路 + 输入分流一路），另加 `gate_values(tokens)` 探针方法
  （返回 (B, S, 1)，同一序列内值相同，现有探针不用改）。
- v4 反向（诊断文档说"loss_and_grads 不用改"是错的，别照抄）：
  门改成池化后，`gate.weight` 的梯度必须用池化后的 x（x_pool，(B,1,D)），
  且 dz_gate 要先在序列维上求和（g 被全序列共享），再乘 x_pool；
  门对输入 x 的梯度经池化反传（除以序列长 S 后广播）也要加回 dx_total——
  旧的每 token 门根本没把这条路传回 dx_total，内置 gradcheck 只抽随机
  emb 项所以一直没发现。新 gradcheck（`gradcheck_gate_seq.py`）专门验
  门参数与输入中真实 token 的 emb 梯度。
- `generalist_pilot_train.py`：`--merge gated`、`--recipe`（v3 用
  `generalist-v3`，v4 用 `generalist-v4`，v5 用 `generalist-v5`）、`--gate-loss-weight`；`make_batch_fn` 多返回第三项
  gate 目标；`train_steps` 接受 3 元 batch（2 元旧 batch 也兼容），
  只在 gated 模型上把 gate 目标传下去。v4 不用改 trainer——gate 目标
  本来就是每序列常数。v5 也不用改 trainer——`n_stem_blocks` 默认 1。
- v5（`n_stem_blocks`，默认 1，写进 config）：参数名 `stem.{i}.`（共享
  stem，k 块）+ `left/right.{i}.`（分支各剩 n_blocks − k 块）；`_forward`
  先跑 stem、池化 stem 隐状态算门、再分输入跑分支；`loss_and_grads`
  加 gate→stem 反传路径（dz 经池化加到 stem 隐状态梯度上，再走 stem
  反向）；`gate_values` 语义不变。旧 gated 检查点参数名不匹配，加载
  即报错（fail-closed，不续跑）。`train_steps` 的 resume 兼容旧检查点
  （`n_stem_blocks` 缺省补 1）。
- 检查点：门参数随 params 一起存（`p_gate.*` + 优化器 `m_/v_`），
  config 带 `merge_method=gated`，老 single 检查点不能直接续跑——新线从头训。

## 规模与代价（d1024，vocab 4622）

- 参数：v4 97,086,129；v5 85,542,235（stem 1 + 每支 3，共 7 块；
  单支 d1024 是 50,909,528；门本身只有 1,025 个参数）。
- 检查点约 1.16 GB（v4，参数 + Adam m/v），v5 约 1.03 GB；
  /tmp 放不下，只写工作台。
- 步耗时：单支实测 85.89 s/step；门控 v4 试点实测 230–238 s/step，
  300 步一块约 19.5 小时；v5 是 v4 的约 7/8 计算量，步耗时略低
  （3 步 pilot 实测后补数）。
- 训练中重启仍会杀进程，`--save-every 25` 的 _last 照常接续。

## 已验证（2026-11-16，v3）

- 门反向数值梯度校验：解析 vs 中心差分，相对误差 1e-6~1e-7，
  有监督/无监督两种情况都过。
- 内置 `numerical_gradient_test`：626 个采样点，四种 merge × 两种 norm 全过。
- 门控检查点存取往返：config 与门参数一致；generate 推理通。
- 试点：`GATED_PILOT_S3.log`，3 步 d1024 门控，测 s/step + 门分流情况。
  探针：训后用 `gate_values` 分别喂推理句与世界句，看平均 g 是否分开
  （推理→近 1，世界→近 0）。

## 已验证（2026-11-17，v4 每序列门）

- 针对性门梯度校验（`gradcheck_gate_seq.py`，float64 中心差分）：
  门参数最差相对误差 8.0e-06（RMS+监督）/ 4.8e-07（LN+监督），与旧门
  同量级；输入中真实 token 的 emb 梯度 1.5e-08~5.7e-08（这条验证了
  门→x 的反传路径，旧门缺失此路径）；无监督情况门梯度小，相对误差
  以绝对误差 ~1e-10 计（有限差分噪声地板）。
- 内置 `numerical_gradient_test`：626 个采样点全过；`--test` 全部冒烟
  测试通过（含更新后的 `kv_cache_parity_test`：gated 只比最后一个位置，
  max_logit_error 5.39e-07；`generate` 与 `generate_cached` 逐 token 一致）。
- v4 试点（3 步，recipe generalist-v4）：见 `GATED_V4_PILOT_S3.log`；
  loss 9.2928 → 8.5744 → 8.3998（238.47 s/step），与 v3 试点同量级。
  探针（`probe_gate_v4_pilot_lean.py`，LEAN 只读 p_*）：3 步后门仍在
  ~0.50（经典 0.5003、代码 0.5011、对话 0.5003、推理 0.5010、常识 0.5006、
  世界 0.5005；左均值 0.5007，分离 0.0001）——未达"左 >0.6 且世界 <0.4"
  的 3 步标准，FAIL。但这不是设计证伪：3 步、warmup LR（1e-4~3e-4）下
  Adam 每步只能把门权重挪 ~lr，g 动不了 0.001；3 步标准本身定得太乐观。
  设计前提已另行验证（`check_pool_separable.py`，pilot 断点 embedding）：
  池化表示线性可分——Fisher 方向上左右分离 4.34 sigma，40 个窗口
  中点阈值准确率 1.000（随机 0.500），信号存在，门能学，只是需要步数。
  结论：不启动完整块；下一步应为 25 步 pilot（约 1.6 小时）+ step-25 探针，
  看分离度是否在长（>0.1 则设计成立，仍 ~0 则另有深层问题）。

## 已验证（2026-11-17，v5 上下文 hidden-state 门）

- 针对性门梯度校验（`gradcheck_gate_seq.py`，float64 中心差分，n_blocks
  改为 2——stem 占 1 块，分支各剩 1 块；另加 stem 参数进抽查）：
  有监督：门参数最差相对误差 2.2e-06（RMS）/ 5.9e-07（LN），与 v4
  同量级；无监督：门梯度本身小（~2e-6），相对误差 1.5e-04/2.5e-04
  以绝对误差 ~4e-10 计（有限差分噪声地板）。输入中真实 token 的 emb
  梯度 2.1e-08~1.6e-07（验证了 gate→stem→x 的反传路径）；stem 参数
  （`stem.0.attn.qkv.weight`）最差相对误差 1.8e-04，绝对误差
  ~4e-10，同样是噪声地板——gate→stem 路径已验。
- 内置 `numerical_gradient_test`：698 个采样点全过（v4 是 626 个；
  gated 改用 n_blocks=2，多了 stem 张量）；`--test` 全部冒烟测试通过
  （含 `kv_cache_parity_test`：gated 最后位置 max_logit_error 4.18e-07，
  v4 是 5.39e-07；`generate` 与 `generate_cached` 逐 token 一致；
  滑动窗口也一致）。三份 `dual_transformer_numpy.py` 同步，md5
  7ac2acac9ba8c25acde18b596a3c69ff。
- 旧 v4 检查点（`gated_v4_pilot_s25.npz`）加载即报错（参数名不匹配），
  fail-closed 符合预期，不续跑。
- v5 试点（3 步，recipe generalist-v5）：见 `GATED_V5_PILOT_S3.log`；
  loss 9.2844 → 8.7999 → 8.6736（206.50 s/step，v4 是 238.47——
  7/8 计算量，快 ~13%）。探针（`probe_gate_v5_pilot_lean.py`，LEAN
  只读 p_*）：3 步后门已在分化——经典 0.5889、代码 0.7329、对话
  0.6686、推理 0.6567、常识 0.5635，左均值 0.6421（已过 0.6）；
  世界 0.5327；分离 0.1094，趋势线（>0.05）PASS——v4 的 3 步分离
  只有 0.0001，25 步也只有 0.0047，v5 的 3 步已超过 v4 25 步的过关
  线 0.1。世界侧尚未压到 0.4 以下（3 步 warmup LR 下，7:1 配比
  的共模上推仍占优，世界只动了一点），需 25 步再看。
  门输入量级已兑现：`||h_pool||` 均值 3.3629（v4 的 `||x_pool||`
  是 0.158，约 21 倍），`||mean_left − mean_world||` = 1.9118
  （v4 是 0.1541，约 12 倍）——诊断的量级论证成立。
  结论：趋势成立；下一步应为 25 步 pilot（约 1.4 小时）+ step-25
  探针，看世界侧是否压到 0.4 以下、分离是否过 0.1。

## 下一步

1. v5 3 步 pilot + 探针（`probe_gate_v5_pilot_lean.py`，趋势线：分离
   > 0.05 即成立，再跑 25 步看是否过线左 >0.6 且世界 <0.4）→ 2. 若
   趋势成立，25 步 pilot（recipe generalist-v5）+ step-25 探针 → 3.
   若 25 步过线，首块 300 步（recipe generalist-v5，从头训；旧 gated
   断点不续跑）→ 4. 验产品、因果探针、旧配置 12 步环，对照 d512 的
   9/12；s5100 仍在产品里，等门控线超 12/12 再换。

## 风险

- 门坍缩（一支赢走全部）：监督目标按域强制分流，起步 50/50，
  坍缩会被 BCE 直接拉回；看 `frac_left` 与分域平均 g 监控。
- 右支数据少（15%）：若行动支学不动，提世界配比或加世界记录。
- 14 小时/块：迭代慢，门设计有问题要早发现——试点 + 早期步数先看门。

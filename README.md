# Omega-chat

- `dual_transformer_numpy.py` — NumPy 手写双分支 Transformer（前向/反向/AdamW/检查点/生成）
- `omega.py` — Omega 闭环控制内核
- `train.py` — **统一训练入口**（Omega 门控训练：每个 chunk 验证 loss 变差或 ESCALATE 就回滚）
- `d5_benchmark_numpy.py` — 单/双分支基准对比
- `pythonista_ui.py` — Pythonista (iOS) 界面
- `Xiyouji.txt` — 训练语料

```
pip install numpy
python train.py test
python train.py train --steps 2000          # 默认用 Xiyouji.txt、字符级分词、余弦学习率
python train.py train --resume --steps 2000 # 续训
python train.py train --no-omega            # 纯训练，不用 Omega 门控
python train.py generate --prompt "悟空"
```

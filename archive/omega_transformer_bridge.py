"""
Omega & Custom Transformer Bridge
================================
Bridge module connecting DualTransformerNP (手写纯 Python/NumPy Transformer) 
with Omega Role Council Engine (Omega 闭环控制内核).
"""

import numpy as np
from dual_transformer_numpy import DualTransformerNP, ByteTokenizer, BOS_ID, EOS_ID
from omega import Omega, Config, ReferenceWorld, SmoothPolicy, Owner

class OmegaTransformerBridge:
    def __init__(self, d_model=32, n_blocks=2, n_heads=4, merge_method="average", max_seq=512):
        """
        初始化纯手写 NumPy Transformer，作为 Omega 的底层智力基座。
        """
        self.tokenizer = ByteTokenizer()
        self.transformer = DualTransformerNP(
            d_model=d_model,
            n_blocks=n_blocks,
            n_heads=n_heads,
            merge_method=merge_method,
            max_seq=max_seq,
            seed=42
        )
        print(f"[Bridge] 成功加载原生自定义 Transformer! 模型参数量: {self.transformer.num_params():,}")

    def role_inference(self, role_name: str, prompt_text: str) -> dict:
        """
        让 Omega 的特定角色（如: 一一、紫微、悟空、大眼）通过自定义 Transformer 进行前向推理。
        """
        tokens = self.tokenizer.encode(prompt_text, add_bos=True)
        if len(tokens) > self.transformer.max_seq:
            tokens = tokens[-self.transformer.max_seq:]
        
        input_tensor = np.array([tokens], dtype=np.int64)

        # 执行原生 Forward Pass 前向传播
        logits = self.transformer.forward(input_tensor)
        
        # 提取最后一个 Token 的特征逻辑值并解码
        next_token_logits = logits[0, -1, :]
        predicted_id = int(np.argmax(next_token_logits))
        decoded_char = self.tokenizer.decode([predicted_id], skip_special=False)

        # 计算响应置信度/残差评分依据
        confidence = float(1.0 / (1.0 + np.exp(-float(np.mean(next_token_logits)))))

        return {
            "role": role_name,
            "engine": "custom_numpy_transformer",
            "prediction_token": decoded_char,
            "confidence": round(confidence, 4),
            "rationale": f"经过原生 Transformer 前向传播计算得出，后续生成倾向: '{decoded_char}'"
        }

if __name__ == "__main__":
    print("=== Omega + 自定义 Transformer 桥接系统集成测试 ===")
    bridge = OmegaTransformerBridge()
    
    # 模拟四个角色的底层矩阵推理
    roles = ["Yiyi (残差评估)", "Ziwei (元认知)", "Wukong (对抗审查)", "Dayan (模式发现)"]
    prompt = "Omega Kernel Step Verification"
    
    for r in roles:
        res = bridge.role_inference(r, prompt)
        print(f"[{res['role']}] 输出倾向: '{res['prediction_token']}' | 置信度: {res['confidence']}")

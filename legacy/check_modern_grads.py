"""Numerical gradient check for ModernTransformerNumpy.loss_and_grads (float64)."""
import sys
import numpy as np
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from modern_transformer_numpy import ModernTransformerNumpy

rng = np.random.default_rng(7)
m = ModernTransformerNumpy(d_model=16, n_blocks=2, n_q_heads=4, n_kv_heads=2,
                             vocab_size=29, seed=3)
# cast to float64 for tight finite differences
m.params = {k: v.astype(np.float64) for k, v in m.params.items()}
x = rng.integers(0, 29, (2, 7))
y = rng.integers(0, 29, (2, 7))
loss, grads = m.loss_and_grads(x, y)
print(f"loss={loss:.6f} params={m.num_params()}")
eps = 1e-6
worst = []
total = 0
for k in sorted(m.params):
    p, g = m.params[k], grads[k]
    it = np.nditer(p, flags=["multi_index"])
    for _ in range(p.size):
        idx = it.multi_index
        orig = p[idx]
        p[idx] = orig + eps
        lp = m.loss(x, y)
        p[idx] = orig - eps
        lm = m.loss(x, y)
        p[idx] = orig
        num = (lp - lm) / (2 * eps)
        ana = g[idx]
        err = abs(num - ana) / max(1.0, abs(num) + abs(ana))
        worst.append((err, k, idx, num, ana))
        total += 1
        it.iternext()
worst.sort(reverse=True)
for err, k, idx, num, ana in worst[:8]:
    print(f"  rel={err:.2e} {k}{idx} num={num:.6f} ana={ana:.6f}")
print(f"checked {total} params, worst rel err = {worst[0][0]:.2e}")
assert worst[0][0] < 1e-5, "GRADCHECK FAILED"
print("GRADCHECK PASS")

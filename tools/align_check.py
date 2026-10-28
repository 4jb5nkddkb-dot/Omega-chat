from pathlib import Path
_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent if _HERE.name in ("tests", "tools", "legacy") else _HERE
import sys
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from omega_chat.agent_runtime import load_modern_model, generate_segment
from omega_chat import dual_transformer_numpy as dtn

model, tok, step = load_modern_model(ROOT / "checkpoints" / "modern_ckpt.npz", ROOT / "data" / "Xiyouji.txt")
print("checkpoint step:", step, "vocab:", tok.vocab_size)
prompt = tok.encode("悟空", add_bos=True)

def rep4(ids):
    g = [tuple(ids[i:i+4]) for i in range(len(ids)-3)]
    return round(1 - len(set(g)) / max(1, len(g)), 3)

plain = generate_segment(model, prompt, 120, temperature=0.0)
print("greedy plain   rep4=", rep4(plain[len(prompt):]), "|", tok.decode(plain[len(prompt):])[:50])
c = dtn.DecodeController(0.0, block_ngram=4, loop_len=8)
blk = generate_segment(model, prompt, 120, temperature=0.0, controller=c)
print("greedy no-rep4 rep4=", rep4(blk[len(prompt):]), "|", tok.decode(blk[len(prompt):])[:50])
c2 = dtn.DecodeController(0.8, target_entropy=1.5, block_ngram=4, loop_len=8)
ctl = generate_segment(model, prompt, 120, temperature=0.8, rng=np.random.default_rng(1), controller=c2)
print("T=0.8 control  rep4=", rep4(ctl[len(prompt):]), "|", tok.decode(ctl[len(prompt):])[:50])
print(c2.summary())

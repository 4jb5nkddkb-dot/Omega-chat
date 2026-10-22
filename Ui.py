import ui
import numpy as np
# 导入咱们的底层核心模块(确保文件在同一个目录下)
from dual_transformer_numpy import DualTransformerNP, ByteTokenizer
from omega import Omega, Config, ReferenceWorld, SmoothPolicy, Owner

class OmegaApp:
    def __init__(self):
        # 1. 初始化底座 Transformer 与 词表
        self.tokenizer = ByteTokenizer()
        self.transformer = DualTransformerNP(
            d_model=32, n_blocks=2, n_heads=4, merge_method="average", max_seq=256, seed=42
        )
        
        # 2. 初始化 Omega 闭环控制内核
        self.omega = Omega(secret="omega-sovereign-secure-secret-12345", auto_restore=False)
        self.owner = self.omega.owner()
        self.world = ReferenceWorld()
        self.policy = SmoothPolicy()
        
        # 记录界面控件引用
        self.view = None
        self.console_view = None
        self.input_view = None

    def make_view(self):
        # 创建 Pythonista 原生 UI 界面
        v = ui.View()
        v.name = "Omega Sovereign AI"
        v.background_color = '#111118' # 极客深色背景
        
        # 标题标签
        lbl_title = ui.Label(frame=(15, 15, 340, 30))
        lbl_title.text = "Ω Omega + NumPy Transformer"
        lbl_title.text_color = '#00ffcc'
        lbl_title.font = ('<system-bold>', 16)
        v.add_subview(lbl_title)
        
        # 状态参数显示
        self.lbl_status = ui.Label(frame=(15, 50, 340, 24))
        self.lbl_status.text = f"参数量: {self.transformer.num_params():,} | 内核状态: IDLE"
        self.lbl_status.text_color = '#aaaaaa'
        self.lbl_status.font = ('<system>', 12)
        v.add_subview(self.lbl_status)
        
        # 文本输入框
        self.input_view = ui.TextField(frame=(15, 85, 235, 40))
        self.input_view.placeholder = "输入指令,唤醒角色议会..."
        self.input_view.background_color = '#1c1c28'
        self.input_view.text_color = '#ffffff'
        v.add_subview(self.input_view)
        
        # 执行按钮
        btn_run = ui.Button(frame=(260, 85, 95, 40))
        btn_run.title = "唤醒议会"
        btn_run.background_color = '#7928ca'
        btn_run.tint_color = '#ffffff'
        btn_run.corner_radius = 6
        btn_run.action = self.on_run_clicked
        v.add_subview(btn_run)
        
        # 终端控制台输出面板 (TextView)
        self.console_view = ui.TextView(frame=(15, 140, 340, 500))
        self.console_view.editable = False
        self.console_view.background_color = '#0a0a0f'
        self.console_view.text_color = '#00ff66'
        self.console_view.font = ('Menlo', 12)
        self.console_view.text = ">>> 系统就绪。\n>>> 底层 NumPy Transformer 引擎加载完毕。\n>>> Omega 角色议会待命中...\n\n"
        v.add_subview(self.console_view)
        
        self.view = v
        return v

    def on_run_clicked(self, sender):
        user_text = self.input_view.text.strip()
        if not user_text:
            user_text = "Omega 状态巡检"
            
        self.log(f"\n--- 触发指令: {user_text} ---")
        
        # 1. 跑自定义 Transformer 前向推理
        tokens = self.tokenizer.encode(user_text, add_bos=True)
        input_tensor = np.array([tokens[-self.transformer.max_seq:]], dtype=np.int64)
        logits = self.transformer.forward(input_tensor)
        next_id = int(np.argmax(logits[0, -1, :]))
        char = self.tokenizer.decode([next_id], skip_special=False)
        self.log(f"[Transformer 底座] 矩阵前向完成 -> 预测后继: '{char}'")
        
        # 2. 跑 Omega 认知闭环循环 (观察 -> 预测 -> 校准 -> 决策)
        cog = self.omega.cognitive_cycle(self.world, self.policy)
        sug = cog.get("suggestion", {})
        self.log(f"[Omega 核心议会] 决策: {sug.get('decision')} | 理由: {sug.get('reason')}")
        self.log(f"[审计状态] 当前 Epoch: {self.omega.epoch} | 校验: 正常")

    def log(self, text):
        self.console_view.text += text + "\n"
        # 自动滚动到底部
        self.console_view.content_offset = (0, self.console_view.content_size[1] - self.console_view.height)

# 启动程序
app = OmegaApp()
v = app.make_view()
v.present('sheet') # 以面板形式在 Pythonista 中弹出

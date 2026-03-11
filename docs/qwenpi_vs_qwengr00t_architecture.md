# QwenPI 与 QwenGR00T 架构详解与对比

## 一、概述

QwenPI 和 QwenGR00T 是 StarVLA 项目中两种视觉-语言-动作（VLA）模型架构。它们共享相同的高层设计：**Qwen VL（视觉-语言模型）骨干网络 + Flow-Matching 扩散动作头**。两者都继承自 `baseframework`，使用相同的 VLM 接口 (`get_vlm_model`)，并通过 flow-matching 方法预测连续的机器人动作。

**核心区别在于 VLM 特征如何传递给动作头。**

### 关键文件路径

| 组件 | 文件路径 |
|------|---------|
| QwenPI 框架 | `starVLA/model/framework/QwenPI.py` |
| QwenGR00T 框架 | `starVLA/model/framework/QwenGR00T.py` |
| QwenPI 动作头 | `starVLA/model/modules/action_model/LayerwiseFM_ActionHeader.py` |
| QwenGR00T 动作头 | `starVLA/model/modules/action_model/GR00T_ActionHeader.py` |
| DiT 实现 | `starVLA/model/modules/action_model/flow_matching_head/cross_attention_dit.py` |
| ActionEncoder | `starVLA/model/modules/action_model/flow_matching_head/action_encoder.py` |
| VLM 接口 | `starVLA/model/modules/vlm/` |
| 基础框架 | `starVLA/model/framework/base_framework.py` |
| 单任务训练 | `starVLA/training/train_starvla.py` |
| 联合训练 | `starVLA/training/train_starvla_cotrain.py` |
| 参数分组工具 | `starVLA/training/trainer_utils/trainer_tools.py` |

---

## 二、架构图

```
QwenGR00T:
  Qwen-VL ──[最后一层隐藏状态]──> FlowmatchingActionHead (DiT)
                                     所有 DiT 层共享相同的 VLM 特征

QwenPI:
  Qwen-VL ──[最后 N 层隐藏状态]──> LayerwiseFlowmatchingActionHead (DiT)
                                      每个 DiT 层接收其对应的 VLM 层特征
```

### 完整架构层次

```
┌─────────────────────────────────────────────────────────┐
│              baseframework（基类）                        │
│  - from_pretrained()   加载预训练权重                      │
│  - unnormalize_actions() 动作反归一化                     │
│  - get_action_stats()  获取动作统计信息                    │
└─────────────────────────────────────────────────────────┘
           ▲                                    ▲
           │                                    │
    ┌──────┴─────────┐              ┌──────────┴─────────┐
    │   Qwen_PI      │              │   Qwen_GR00T       │
    │ (注册为QwenPI)  │              │ (注册为QwenGR00T)   │
    └────────────────┘              └────────────────────┘
           │                                    │
    ┌──────▼──────────┐            ┌───────────▼────────┐
    │ VLM 接口        │            │  VLM 接口           │
    │(get_vlm_model)  │            │ (get_vlm_model)    │
    │ 支持多种VLM:     │            │ 支持多种VLM:        │
    │ Qwen2.5/3/3.5   │            │ Qwen2.5/3/3.5      │
    └─────────────────┘            └────────────────────┘
           │                                    │
    ┌──────▼─────────────────────┐  ┌───────────▼──────┐
    │  逐层 FM 动作头             │  │  单层 FM 动作头   │
    │  (逐层交叉注意力)           │  │  (全局交叉注意力) │
    └────────────────────────────┘  └──────────────────┘
           │                                    │
    ┌──────▼──────────────────────────────────▼──────┐
    │   共享的 DiT + Flow Matching 组件               │
    │  - DiT (cross_attention_dit.py)                │
    │  - ActionEncoder（动作编码器）                   │
    │  - ActionDecoder（MLP 动作解码器）               │
    │  - StateEncoder（状态编码器，可选）              │
    │  - Beta 时间步采样                              │
    └───────────────────────────────────────────────┘
```

---

## 三、核心区别：特征传递策略

这是两者最根本的架构差异。

### QwenGR00T 的方式（`QwenGR00T.py:108`）

```python
last_hidden = qwenvl_outputs.hidden_states[-1]   # 单个张量 [B, L, H]
action_loss = self.action_model(last_hidden, actions, state)
```

- 只提取 VLM 的**最后一层**隐藏状态
- 所有 DiT Transformer 块对**相同的** VLM 特征执行交叉注意力
- DiT 内部的 `forward()` 方法统一处理所有 Transformer 块

### QwenPI 的方式（`QwenPI.py:163-165`）

```python
all_hidden = qwenvl_outputs.hidden_states
expected_layers = len(self.action_model.model.transformer_blocks)
vl_embs_list = list(all_hidden[-expected_layers:])  # N 个张量的列表
action_loss = self.action_model(vl_embs_list, actions, state)
```

- 提取 VLM 的**最后 N 层**隐藏状态（N = DiT 块的数量）
- 每个 DiT 块从其**对应的 VLM 层**接收交叉注意力
- 在 `LayerwiseFM_ActionHeader.py:328-333` 中手动逐层循环：

```python
for layer_idx, layer in enumerate(self.model.transformer_blocks):
    model_output = layer(
        hidden_states=model_output,
        encoder_hidden_states=vl_embs_list[layer_idx],  # 逐层对应！
        temb=temb,
    )
```

---

## 四、动作头详细对比

| 方面 | QwenGR00T (`FlowmatchingActionHead`) | QwenPI (`LayerwiseFlowmatchingActionHead`) |
|------|--------------------------------------|---------------------------------------------|
| **类名** | `FlowmatchingActionHead` | `LayerwiseFlowmatchingActionHead` |
| **VLM 输入** | 单个张量 `[B, L, H]` | N 个张量的列表 `[B, L, H]` |
| **DiT 前向** | 标准 `self.model(...)` 调用 | 手动循环 `transformer_blocks` |
| **DiT 配置** | 预设尺寸（`DiT-B`: 768d, `DiT-L`: 1536d） | 从 VLM 隐藏维度推导（通常 2048d） |
| **默认隐藏维度** | 768 或 1536（独立于 VLM） | 2048（匹配 VLM，或可配置） |
| **DiT 层数** | 从 YAML 配置（通常 16） | = VLM 的 num_hidden_layers 或从 YAML |
| **交叉注意力维度** | 从 YAML `cross_attention_dim` | = VLM 的 hidden_size |

### DiT 维度配置差异

**QwenGR00T** 使用**预设的 DiT 尺寸**（`GR00T_ActionHeader.py:211-214`）：
```python
DiTConfig = {
    "DiT-B": {"input_embedding_dim": 768,  "attention_head_dim": 64, "num_attention_heads": 12},
    "DiT-L": {"input_embedding_dim": 1536, "attention_head_dim": 48, "num_attention_heads": 32},
}
```
DiT 的隐藏维度**独立于** VLM 隐藏维度，通过交叉注意力桥接维度差异。

**QwenPI** 默认**匹配 VLM 维度**（`LayerwiseFM_ActionHeader.py:229-248`）：
```python
vl_hidden_dim = global_config.framework.qwenvl.vl_hidden_dim  # 例如 2048
action_hidden_dim = getattr(action_config, 'action_hidden_dim', vl_hidden_dim)
# 如果 action_hidden_dim == vl_hidden_dim: DiT 维度匹配 VLM（传统模式）
# 如果 action_hidden_dim != vl_hidden_dim: 较小的 DiT + 交叉注意力桥接
```

---

## 五、DiT（扩散 Transformer）内部结构

### 整体 DiT 架构

DiT（`cross_attention_dit.py:190-308`）基于 `diffusers` 库构建。其结构如下：

```
DiT
├── TimestepEncoder          ─ 将离散时间步 → 嵌入向量
├── transformer_blocks[]     ─ N × BasicTransformerBlock
├── norm_out                 ─ 最终 LayerNorm
├── proj_out_1               ─ inner_dim → 2×inner_dim（用于 shift/scale）
└── proj_out_2               ─ inner_dim → output_dim（最终投影）
```

**关键维度**: `inner_dim = num_attention_heads × attention_head_dim`
- QwenGR00T DiT-B: 12 头 × 64 = **768**
- QwenGR00T DiT-L: 32 头 × 48 = **1536**
- QwenPI 默认: 32 头 × 64 = **2048**（匹配 VLM）

### TimestepEncoder（时间步编码器）

将离散整数时间步转换为密集嵌入：

```
timestep (整数)
  → Timesteps(256)           # 正弦投影 → 256 维
  → TimestepEmbedding(256→D) # MLP: 256 → inner_dim
  → temb [B, D]
```

`temb` 是**全局条件信号**——它告诉每一层"当前输入有多嘈杂？"

### BasicTransformerBlock（基本 Transformer 块）

每个块包含 3 个子层：

```
输入: hidden_states [B, T, D], encoder_hidden_states [B, S, D_vlm], temb [B, D]

1. AdaLayerNorm(hidden_states, temb)
   │  LN(x) * (1 + scale) + shift    ← scale/shift 从 temb 推导
   │  这就是时间步调制每一层的方式
   ▼
2. 交叉注意力 (attn1)
   │  Q = 来自动作 tokens [B, T, D]
   │  K, V = 来自 VLM 特征 [B, S, D_vlm]   ← VLM 信息在此注入！
   │  + 残差连接
   ▼
3. FeedForward（GEGLU 激活）
   │  LayerNorm → Linear → GELU-gate → Linear → Dropout
   │  + 残差连接
   ▼
输出: hidden_states [B, T, D]
```

**关键设计选择**（第 124-133 行）：`attn1` 接收 `cross_attention_dim`，这意味着每个块中的注意力是对 VLM 特征的**交叉注意力**，而不是动作 tokens 之间的自注意力。Q 来自动作 tokens，K/V 来自 VLM。

### 交替自注意力（可选）

当 `interleave_self_attention=True` 时（第 232-233 行）：
```python
use_self_attn = idx % 2 == 1 and interleave_self_attention
```
- **偶数层**: 交叉注意力（Q=动作, K/V=VLM）
- **奇数层**: 自注意力（Q=K=V=动作 tokens）

这允许动作 tokens 每隔一层**相互通信**，有助于时序一致性。

### DiT 输出处理（第 301-308 行）

最终输出使用**另一个 AdaLN 风格的调制**：

```python
shift, scale = proj_out_1(SiLU(temb)).chunk(2, dim=1)  # temb → 2D
output = LayerNorm(hidden) * (1 + scale) + shift         # 缩放和偏移
output = proj_out_2(output)                               # 投影到 output_dim
```

### QwenGR00T 与 QwenPI 使用 DiT 的差异

**QwenGR00T** 调用 `self.model(...)` 触发 `DiT.forward()`：
```python
# DiT.forward() — 所有块共享相同的 encoder_hidden_states
for block in self.transformer_blocks:
    hidden = block(hidden, encoder_hidden_states=vl_features, temb=temb)
#                                                ^^^^^^^^^^^
#                                          每层都是相同的
```

**QwenPI** 绕过 `DiT.forward()`，手动循环（`LayerwiseFM_ActionHeader.py:326-333`）：
```python
temb = self.model.timestep_encoder(t_discretized)  # 从 DiT 获取 temb
model_output = sa_embs
for layer_idx, layer in enumerate(self.model.transformer_blocks):
    model_output = layer(
        hidden_states=model_output,
        encoder_hidden_states=vl_embs_list[layer_idx],  # 每层不同！
        temb=temb,
    )
```

注意：QwenPI 跳过了 DiT 的输出处理（`norm_out`, `proj_out_1`, `proj_out_2`）——它直接将 DiT 的原始隐藏状态送入自己的 `action_decoder` MLP。代码第 335 行的注释也承认了这一点：`# TODO miss self att and _process_output, but work well`。

---

## 六、Flow-Matching（流匹配）详解

### 什么是 Flow-Matching？

Flow-matching 是一种**生成建模**技术，学习一个向量场（速度），将噪声分布传输到数据分布。与 DDPM（学习预测噪声）不同，flow-matching 定义了一条从噪声到数据的**直线路径**。

### 直线 ODE

定义从噪声 `x_0` 到干净动作 `x_1` 的路径：

```
x_t = (1 - t) · x_0 + t · x_1      其中 t ∈ [0, 1]
    = (1 - t) · noise + t · actions
```

沿此路径的**速度**（时间导数）：
```
dx/dt = x_1 - x_0 = actions - noise
```

这个速度是**常数**（不依赖于 t）——模型学习预测这个速度。

### 训练：前向传播代码详解

来自 `GR00T_ActionHeader.py:270-318`（两个模型逻辑相同）：

```python
# 1. 采样随机噪声
noise = torch.randn(actions.shape)

# 2. 从 Beta 分布采样时间
t = Beta(1.5, 1.0).sample([B])    # 偏向 t=1（干净数据）
t = (noise_s - t) / noise_s       # 重新缩放：映射到 [0, 0.999]

# 3. 创建含噪轨迹（线性插值）
noisy_trajectory = (1 - t) * noise + t * actions

# 4. 计算真实速度
velocity = actions - noise

# 5. 将 t 离散化用于时间步嵌入
t_discretized = (t * 1000).long()  # [0, 999]

# 6. 编码和预测
action_features = action_encoder(noisy_trajectory, t_discretized)
# ... DiT 前向 ...
pred_velocity = action_decoder(model_output)

# 7. 损失 = 预测速度和真实速度之间的 MSE
loss = ((pred_velocity - velocity) ** 2).mean()
```

### 为什么用 Beta(1.5, 1.0)？

参数为 `alpha=1.5, beta=1.0` 的 Beta 分布是**右偏**的——它采样的 t 值偏向 1（干净数据）。经过变换 `(s - t) / s` 后，实际训练时间偏向 0（更嘈杂）。这意味着模型更多地训练早期去噪步骤，而这些步骤通常更困难。

```
Beta(1.5, 1.0) → 峰值在 t ≈ 0.7-0.8
变换后 → 训练集中在 t ≈ 0.0-0.3（更嘈杂的输入）
```

### 推理：欧拉积分

来自 `GR00T_ActionHeader.py:320-370`：

```python
# 从纯噪声开始
actions = torch.randn(B, T, action_dim)

num_steps = 4  # 只需要 4 步！
dt = 1.0 / 4   # = 0.25

for step in range(4):
    t = step / 4.0                    # t = 0.0, 0.25, 0.5, 0.75
    t_discrete = int(t * 1000)        # t = 0, 250, 500, 750

    pred_velocity = model(actions, t)  # 预测速度场
    actions = actions + dt * pred_velocity  # 欧拉步: x_{t+dt} = x_t + dt·v
```

图示：
```
t=0.0(噪声) ──v₀──> t=0.25 ──v₁──> t=0.5 ──v₂──> t=0.75 ──v₃──> t=1.0(干净动作)
   randn()      +0.25·v₀      +0.25·v₁      +0.25·v₂      +0.25·v₃
```

### 为什么只需 4 步？

使用直线路径的 flow-matching 产生几乎笔直的轨迹。与 DDPM（遵循弯曲的 SDE，需要 50-1000 步）不同，flow-matching 的 ODE 几乎是线性的，因此 **4 个欧拉步**就足以获得良好的近似。这使推理速度比 DDPM 快约 10-50 倍。

### ActionEncoder：融合动作与时间

`ActionEncoder`（`action_encoder.py:57-101`）将 `(actions, timestep)` 映射到 DiT 的隐藏空间：

```
actions [B, T, 7]  ──W1──> a_emb [B, T, D]
                                               ──concat──> [B, T, 2D] ──W2──> swish ──W3──> [B, T, D]
timestep [B]       ──expand──> [B, T] ──sin/cos──> tau_emb [B, T, D]
```

时间步的正弦编码使用标准 Transformer 公式：
```
freq_i = exp(-i · log(10000) / (D/2))
sin(t · freq_i), cos(t · freq_i)     对于 i = 0..D/2-1
```

### 完整数据流总结

```
训练:
  图像 + 文本 ──VLM──> hidden_states
  动作 + 噪声 ──插值──> noisy_trajectory
  noisy_trajectory + t ──ActionEncoder──> action_features
  [state_features | future_tokens | action_features] ──DiT(cross_attn=hidden_states)──> pred
  pred ──ActionDecoder──> pred_velocity
  Loss = MSE(pred_velocity, actions - noise)

推理:
  图像 + 文本 ──VLM──> hidden_states
  x₀ = randn()
  for t in [0, 0.25, 0.5, 0.75]:
      x_t + t ──ActionEncoder──> features
      [state | future_tokens | features] ──DiT──> pred
      pred ──ActionDecoder──> velocity
      x_{t+dt} = x_t + 0.25 · velocity
  返回 x₁（去噪后的动作）
```

---

## 七、训练流程

### 训练脚本

有两个主要训练脚本：
- `starVLA/training/train_starvla.py` — **VLATrainer**: 仅动作训练
- `starVLA/training/train_starvla_cotrain.py` — **VLAMTrainer**: VLA + VLM 联合训练

两个模型（QwenPI 和 QwenGR00T）通过相同的 `build_framework(cfg)` 工厂实例化，由 `cfg.framework.name` 选择。

### 整体训练循环

```
┌─────────────────────────────────────────────────────────────────┐
│                      VLATrainer / VLAMTrainer                   │
│                                                                  │
│  for step in range(max_train_steps):                            │
│    ┌──────────────────────────────────────────────┐             │
│    │ 1. 从 VLA 数据加载器获取 batch                  │             │
│    │    batch = { image, lang, action, [state] }  │             │
│    └──────────────┬───────────────────────────────┘             │
│                   ▼                                              │
│    ┌──────────────────────────────────────────────┐             │
│    │ 2. model.forward(batch) → action_loss        │             │
│    │    （内部: VLM 编码 → 动作头）                  │             │
│    └──────────────┬───────────────────────────────┘             │
│                   ▼                                              │
│    ┌──────────────────────────────────────────────┐             │
│    │ 3. [仅联合训练] VLM 前向 → vlm_loss           │             │
│    │    total_loss = action_loss + vlm_loss * 0.1  │             │
│    └──────────────┬───────────────────────────────┘             │
│                   ▼                                              │
│    ┌──────────────────────────────────────────────┐             │
│    │ 4. 损失尖峰检测 → 梯度裁剪                     │             │
│    │    → optimizer.step() → scheduler.step()      │             │
│    └──────────────────────────────────────────────┘             │
│                                                                  │
│    定期: 保存检查点、验证、日志记录                               │
└─────────────────────────────────────────────────────────────────┘
```

### 差异化学习率

`build_param_lr_groups(model, cfg)` 函数（`trainer_tools.py`）创建独立的优化器参数组：

```yaml
# 典型配置:
trainer:
  learning_rate:
    base: 1e-05           # 未分配参数的默认学习率
    qwen_vl_interface: 1e-05   # VLM 骨干网络（慢速更新）
    action_model: 1e-04        # 动作头（10倍更快）
  freeze_modules: ''           # 留空 = 不冻结
```

含义：
- **VLM**（数十亿参数）以低学习率（1e-5）微调
- **动作头**（数亿参数）以 10 倍更高的学习率（1e-4）训练
- 可通过列出路径冻结模块：`freeze_modules: "qwen_vl_interface.model.visual"`

### 重复扩散步骤

两个模型都沿 batch 维度**重复**数据，然后传递给动作头：

```python
repeated_diffusion_steps = 4  # 默认值，QwenPI 硬编码为 2
actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
vl_features_repeated = vl_features.repeat(repeated_diffusion_steps, 1, 1)
```

这意味着对于一次 VLM 前向传播，动作头在相同的动作目标上训练**多个随机噪声样本**。这摊销了昂贵的 VLM 计算（灵感来自 CogACT）。

| 模型 | 默认重复步数 |
|------|------------|
| QwenGR00T | 4-8（来自配置） |
| QwenPI | 硬编码为 **2**（第 179 行: `repeated_diffusion_steps = 2`） |

QwenPI 使用更少的重复，因为其动作头更大（2048 维匹配 VLM），内存更紧张。

### 优化器与调度器

```python
# AdamW 标准 VLA 超参数
optimizer = AdamW(
    param_groups,
    lr=base_lr,
    betas=(0.9, 0.95),    # 大 Transformer 的标准配置
    weight_decay=1e-8,     # 非常轻的正则化
    eps=1e-8
)

# 带最小学习率下限的余弦衰减
scheduler = cosine_with_min_lr(
    warmup_steps=3000-5000,
    total_steps=100000,
    min_lr=5e-7 到 1e-6     # 永远不低于此值
)
```

---

## 八、联合训练（Co-Training）详解

### 问题背景

当为动作预测微调 VLM（如 Qwen2.5-VL）时，来自动作头的梯度会流回 VLM 骨干网络。随着时间推移，这可能导致**灾难性遗忘**——VLM "忘记"其原始的视觉-语言理解能力，这反而会降低动作预测质量（因为 VLM 特征变得不那么有意义）。

### 解决方案：双损失联合训练

`VLAMTrainer`（`train_starvla_cotrain.py`）同时在两个任务上训练：

```python
def _train_step(self, batch_vla, batch_vlm):
    self.optimizer.zero_grad()

    # ---- 任务 1: VLA（动作预测）----
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output_dict = self.model.forward(batch_vla)     # VLM + 动作头
        action_loss = output_dict["action_loss"]        # Flow-matching MSE
    self.accelerator.backward(action_loss)              # 梯度累积

    # ---- 任务 2: VLM（语言建模）----
    with torch.autocast("cuda", dtype=torch.bfloat16):
        vlm_output = self.model.qwen_vl_interface(**batch_vlm)  # 仅 VLM
        vlm_loss = vlm_output.loss * self.config.trainer.loss_scale.vlm  # × 0.1
    self.accelerator.backward(vlm_loss)                 # 梯度累积

    # ---- 在组合梯度上执行单次优化器步骤 ----
    self.optimizer.step()
    self.lr_scheduler.step()
```

### 损失平衡

来自配置（`loss_scale`）：
```yaml
loss_scale:
  vla: 1.0    # 动作损失权重（隐式，代码中未乘以）
  vlm: 0.1    # VLM 损失权重
```

VLM 损失被缩放到**其原始值的 10%**。原因如下：
1. VLM 交叉熵损失通常在量级上远大于 flow-matching MSE 损失
2. VLM 任务是**正则化器**，不是主要目标——我们想保持能力但不主导梯度
3. 权重太高 → 模型关注语言，动作质量下降
4. 权重太低 → 灾难性遗忘仍然发生

### 梯度流图示

```
                      VLM 骨干网络 (Qwen2.5-VL)
                     ┌──────────────────────────┐
                     │  视觉编码器               │
  图像 ────────────> │  文本编码器               │ <── 指令
                     │  跨模态融合               │
                     └────────┬─────────────────┘
                              │
              ┌───────────────┼────────────────────┐
              │               │                    │
              ▼               ▼                    ▼
        所有层隐藏状态    最后层隐藏状态        VLM logits
        (所有层)         (最后层)             (用于语言)
              │               │                    │
     [仅QwenPI]       [仅QwenGR00T]               │
              │               │                    │
              ▼               ▼                    ▼
        ┌─────────┐   ┌─────────┐          ┌──────────┐
        │ 逐层DiT │   │ 标准DiT │          │ 交叉熵   │
        │         │   │         │          │ 损失     │
        └────┬────┘   └────┬────┘          └────┬─────┘
             │              │                    │
             ▼              ▼                    ▼
        action_loss    action_loss           vlm_loss × 0.1
             │              │                    │
             └──────────────┴────────────────────┘
                              │
                     ∇ = ∇_action + 0.1 · ∇_vlm
                              │
                    optimizer.step() (AdamW)
```

### 两次分离的 backward() 调用

关键实现细节：代码在中间没有 `optimizer.step()` 的情况下调用了 `backward()` **两次**。这是有效的，因为 PyTorch 会**累积**梯度：

```python
self.accelerator.backward(action_loss)   # ∇_action 添加到 .grad
self.accelerator.backward(vlm_loss)      # 0.1 · ∇_vlm 添加到 .grad
# 现在 .grad = ∇_action + 0.1 · ∇_vlm
self.optimizer.step()                    # 使用组合梯度更新
```

这在数学上等价于 `backward(action_loss + 0.1 * vlm_loss)`，但分两步完成以允许不同的 autocast 上下文和潜在的不同损失类型。

### VLM 数据管道

VLM 数据使用标准的视觉问答数据集（OXE 配置第 51-53 行）：
```yaml
vlm_data:
  dataset_py: vlm_datasets
  dataset_use: asv2_conversation_en, asv2_detailed_description_en,
               coco_karpathy_train_567_en, okvqa_en, vqav2_en, ...
  per_device_batch_size: 4   # 小于 VLA batch (16)
```

这些是标准的 VQA/图像描述/定位数据集，维护 VLM 的：
- 图像理解（图像描述、VQA）
- 定位能力（RefCOCO）
- 语言连贯性（对话）

### Batch 大小不对称

```
VLA batch: 16 样本/GPU × N GPUs    （动作预测）
VLM batch:  4 样本/GPU × N GPUs    （语言建模）
```

VLM batch 小 **4 倍**，原因是：
1. VLM 序列更长（文本 + 图像 tokens 最多 2048 个 tokens）
2. VLM 损失已经被 0.1 降权
3. 内存预算主要由 VLA 前向传播占据

### 数据迭代器循环

两个数据加载器独立循环（`_get_next_batch()`，第 304-320 行）：
```python
def _get_next_batch(self):
    try:
        batch_vla = next(self.vla_iter)
    except StopIteration:
        self.vla_iter, self.vla_epoch_count = self._reset_dataloader(...)
        batch_vla = next(self.vla_iter)

    try:
        batch_vlm = next(self.vlm_iter)
    except StopIteration:
        self.vlm_iter, self.vlm_epoch_count = self._reset_dataloader(...)
        batch_vlm = next(self.vlm_iter)
```

由于 VLA 和 VLM 数据集大小不同，它们以不同的速度完成 epoch。`_reset_dataloader` 正确调用 `sampler.set_epoch()` 以确保分布式 shuffling 正确。

### 为什么这种方法有效？

联合训练策略有效地创建了一个**多任务学习**设置：

1. **VLA 任务**教会 VLM 产生对连续动作预测有用的特征
2. **VLM 任务**防止 VLM 特征偏离其预训练分布太远
3. **差异化学习率**（VLM: 1e-5, 动作头: 1e-4）确保 VLM 缓慢适应，而动作头快速适应
4. VLM 的 **0.1 损失权重**确保动作预测仍然是主要目标

这类似于 OpenAI/Google 在为下游任务微调时应用辅助语言损失的方式——这是一种通过梯度混合而非显式正则化项实现的**弹性权重巩固**。

---

## 九、训练超参数对比

| 参数 | QwenGR00T (LIBERO 联合训练) | QwenPI (OXE 联合训练) |
|------|---------------------------|----------------------|
| **框架名** | QwenGR00T | QwenFM |
| **VLA batch** | 16 | 16 |
| **VLM batch** | 4 | 4 |
| **VLM 损失缩放** | 0.1 | 0.1 |
| **基础 LR** | 2.5e-5 | 1e-5 |
| **VLM LR** | 1e-5 | 1e-5 |
| **动作头 LR** | 1e-4 | 1e-4 |
| **预热步数** | 5000 | 5000 |
| **总步数** | 100000 | 100000 |
| **梯度裁剪** | 1.0 | 1.0 |
| **最小 LR** | 1e-6 | 5e-7 |
| **DiT 层数** | 16 | 16 |
| **DiT 隐藏维度** | 768 (DiT-B) | 2048 |
| **重复扩散步数** | 8 | 4 |
| **动作维度** | 7 | 7 |
| **未来动作窗口** | 7 | 15 |

---

## 十、设计理念总结

| 方面 | QwenGR00T | QwenPI |
|------|-----------|--------|
| **来源** | GR00T N1.5 (NVIDIA) | PI_0 (Physical Intelligence) |
| **设计理念** | 简单、轻量 | 深层特征融合 |
| **VLM 使用** | 黑盒（仅使用最终输出） | 白盒（利用中间层） |
| **参数效率** | 较小的 DiT（768d/1536d） | 较大的 DiT（2048d 匹配 VLM） |
| **信息流** | VLM 特征压缩为单一表示 | 多尺度 VLM 特征逐层保留 |
| **权衡** | 更低的计算和内存开销 | 更丰富的梯度流和特征融合潜力 |

**总结**：QwenGR00T 是更简单的方法——它将 VLM 视为黑盒特征提取器，并将最终表示输入标准 DiT 动作头。QwenPI 更复杂——它通过在 VLM 层和 DiT 层之间建立一一对应关系来利用多层 VLM 表示，实现更丰富的梯度流和多尺度特征集成。权衡在于复杂性和参数量 vs. 通过更深的 VLM-动作耦合实现更好动作预测的潜力。

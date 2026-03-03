# 配置文件差异分析报告

## 问题描述
- **qwen3-2b-pi-libero.yaml**: model time 5-7, loss 收敛到 1.5
- **qwenpi_libero.yaml**: model time 1-2, loss 收敛到 1e-3

## 关键差异对比

### 1. 预训练模型 (base_vlm) ⚠️ **最重要差异**

| 配置项 | qwen3-2b-pi-libero.yaml | qwenpi_libero.yaml |
|--------|------------------------|-------------------|
| **模型路径** | `/mnt/cpfs/guchenyang/Pretrained/Qwen3-VL-2B-Instruct` | `/mnt/cpfs/guchenyang/Pretrained/Qwen3-VL-4B-Instruct-Action` |
| **模型大小** | 2B | 4B |
| **模型类型** | Instruct (通用指令) | Instruct-Action (动作专用) |

**影响分析**：
- **4B模型容量更大**：更强的表征能力，能学习更复杂的动作模式
- **Action版本预训练**：Qwen3-VL-4B-Instruct-Action 专门针对动作预测任务进行了预训练，具有更好的动作理解能力
- **2B模型限制**：模型容量较小，可能无法充分学习复杂的动作分布，导致loss难以收敛

### 2. action_model.repeated_diffusion_steps

| 配置项 | qwen3-2b-pi-libero.yaml | qwenpi_libero.yaml |
|--------|------------------------|-------------------|
| **值** | 8 | 15 |

**注意**：在 `QwenPI.py:128` 中，代码硬编码了 `repeated_diffusion_steps = 2`，因此这个配置项可能**未被实际使用**。

### 3. trainer.repeated_diffusion_steps

| 配置项 | qwen3-2b-pi-libero.yaml | qwenpi_libero.yaml |
|--------|------------------------|-------------------|
| **值** | 4 | 4 |

**注意**：虽然配置文件中都是4，但在代码中被硬编码为2（见 `QwenPI.py:128`）。

### 4. Batch Size

| 配置项 | qwen3-2b-pi-libero.yaml | qwenpi_libero.yaml |
|--------|------------------------|-------------------|
| **per_device_batch_size** | 16 | 12 |

**影响分析**：
- 更大的batch size (16) 可能导致：
  - **更稳定的梯度**：有助于训练稳定性
  - **更长的训练时间**：每个batch处理更多样本
  - **更高的内存占用**：可能导致GPU利用率更高，model time增加

### 5. 其他配置项

其他配置项（学习率、优化器、数据等）**完全相同**。

## 根本原因分析

### Model Time 差异 (5-7 vs 1-2)

**主要原因**：
1. **模型大小差异**：
   - 2B模型：参数量少，但可能因为batch size=16导致内存压力大，需要更多时间处理
   - 4B模型：参数量多，但batch size=12，且模型架构可能更优化

2. **Batch Size影响**：
   - batch_size=16 需要处理更多样本，增加计算时间
   - batch_size=12 相对较小，处理更快

3. **预训练模型优化**：
   - Action版本的4B模型可能在架构上针对动作预测进行了优化，推理效率更高

### Loss 收敛差异 (1.5 vs 1e-3)

**主要原因**：
1. **模型容量不足** (最关键)：
   - **2B模型**：参数量有限，可能无法充分学习复杂的动作分布
   - **4B模型**：更大的容量能够学习更精细的动作模式

2. **预训练任务匹配度**：
   - **Qwen3-VL-2B-Instruct**：通用视觉-语言模型，未针对动作预测优化
   - **Qwen3-VL-4B-Instruct-Action**：专门为动作预测任务预训练，具有更好的动作理解能力

3. **特征质量**：
   - Action版本的预训练模型可能学习了更好的动作相关特征表示
   - 这些特征直接有助于动作预测任务的收敛

## 建议解决方案

### 方案1：使用更大的预训练模型（推荐）
```yaml
qwenvl:
  base_vlm: /mnt/cpfs/guchenyang/Pretrained/Qwen3-VL-4B-Instruct-Action
```

### 方案2：调整batch size
如果必须使用2B模型，可以尝试：
```yaml
per_device_batch_size: 8  # 减小batch size，可能提高训练效率
```

### 方案3：检查代码中的硬编码
检查 `QwenPI.py:128` 中的硬编码：
```python
repeated_diffusion_steps = 2 # NO repeat for big action FM
```
这个硬编码可能影响了训练效果。

### 方案4：调整学习率
对于2B模型，可能需要更大的学习率：
```yaml
learning_rate:
  action_model: 2.0e-04  # 从1.0e-04增加到2.0e-04
```

## 结论

**核心问题**：2B模型容量不足 + 非Action版本的预训练，导致无法充分学习动作预测任务。

**最佳解决方案**：使用 `Qwen3-VL-4B-Instruct-Action` 预训练模型，这是专门为动作预测任务优化的模型，具有更好的收敛性和性能。

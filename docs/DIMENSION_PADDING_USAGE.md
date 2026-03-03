# 动作和状态维度填充功能使用说明

## 概述

本功能为 starVLA 数据加载器添加了动作（action）和状态（state）维度填充能力，使得不同具身智能体（embodiment）的数据可以统一到相同的维度空间，便于混合训练。

## 主要特性

- **自动维度填充**：将动作和状态张量填充到指定的目标维度
- **零填充策略**：使用零值填充额外的维度
- **可逆变换**：支持在推理时恢复原始维度
- **统计计算集成**：在计算数据集统计信息时自动应用填充
- **模块化设计**：遵循现有的变换管道架构

## 实现的文件

### 1. 核心变换类
**文件**: `starVLA/dataloader/gr00t_lerobot/transform/state_action.py`

新增了 `StateActionDimensionPadding` 类：
- 继承自 `InvertibleModalityTransform`
- 支持 `apply()` 和 `unapply()` 方法
- 自动跟踪原始维度以支持反向操作

### 2. 统计计算函数
**文件**: `starVLA/dataloader/gr00t_lerobot/datasets.py`

修改了 `calculate_dataset_statistics()` 函数：
- 新增 `target_action_dim` 和 `target_state_dim` 参数
- 在计算统计信息前自动应用填充
- 确保归一化统计信息反映填充后的维度

### 3. 数据配置类
**文件**: `starVLA/dataloader/gr00t_lerobot/data_config.py`

更新了以下配置类：
- `OxeDroidDataConfig`
- `OxeBridgeDataConfig`
- `BridgeV2DataConfig`

每个类都添加了：
- `target_action_dim` 属性
- `target_state_dim` 属性
- 在变换管道中集成了填充变换

## 使用方法

### 方法 1: 在数据配置中设置目标维度

```python
from starVLA.dataloader.gr00t_lerobot.data_config import BridgeV2DataConfig

# 创建配置实例
config = BridgeV2DataConfig()

# 设置目标维度（将动作和状态都填充到 16 维）
config.target_action_dim = 16
config.target_state_dim = 16

# 获取变换管道（会自动包含填充变换）
transform = config.transform()
```

### 方法 2: 直接使用填充变换

```python
from starVLA.dataloader.gr00t_lerobot.transform.state_action import (
    StateActionDimensionPadding,
    StateActionToTensor,
)

# 定义要处理的键
action_keys = ["action.eef_position_delta", "action.gripper_position"]
state_keys = ["state.eef_position", "state.gripper_position"]

# 创建变换管道
transforms = [
    # 1. 先转换为张量
    StateActionToTensor(apply_to=action_keys + state_keys),

    # 2. 应用维度填充（在归一化之前）
    StateActionDimensionPadding(
        apply_to=action_keys + state_keys,
        target_action_dim=16,
        target_state_dim=16,
    ),

    # 3. 然后进行归一化等其他变换
    # ...
]
```

### 方法 3: 计算带填充的数据集统计信息

```python
from pathlib import Path
from starVLA.dataloader.gr00t_lerobot.datasets import calculate_dataset_statistics

# 指定 parquet 文件路径
parquet_paths = list(Path("dataset/data").glob("**/*.parquet"))

# 计算统计信息时应用填充
statistics = calculate_dataset_statistics(
    parquet_paths=parquet_paths,
    target_action_dim=16,  # 动作维度填充到 16
    target_state_dim=16,   # 状态维度填充到 16
)

# 统计信息将反映填充后的维度
# 例如：原本 7 维的动作会变成 16 维，后 9 维的统计值为 0
```

## 完整示例：混合不同具身智能体的数据

### 场景说明
假设我们有两个不同的机器人数据集：
- **Bridge V2**: 动作维度为 7（x, y, z, roll, pitch, yaw, gripper）
- **Droid**: 动作维度为 10（包含额外的关节角度）

我们希望将它们统一到 16 维进行混合训练。

### 实现代码

```python
from pathlib import Path
from starVLA.dataloader.gr00t_lerobot.data_config import (
    BridgeV2DataConfig,
    OxeDroidDataConfig,
)
from starVLA.dataloader.gr00t_lerobot.datasets import (
    LeRobotSingleDataset,
    calculate_dataset_statistics,
)

# 1. 配置 Bridge V2 数据集
bridge_config = BridgeV2DataConfig()
bridge_config.target_action_dim = 16
bridge_config.target_state_dim = 16

# 2. 配置 Droid 数据集
droid_config = OxeDroidDataConfig()
droid_config.target_action_dim = 16
droid_config.target_state_dim = 16

# 3. 计算 Bridge V2 的统计信息（带填充）
bridge_parquet_paths = list(Path("datasets/bridge_v2/data").glob("**/*.parquet"))
bridge_stats = calculate_dataset_statistics(
    parquet_paths=bridge_parquet_paths,
    target_action_dim=16,
    target_state_dim=16,
)

# 4. 计算 Droid 的统计信息（带填充）
droid_parquet_paths = list(Path("datasets/droid/data").glob("**/*.parquet"))
droid_stats = calculate_dataset_statistics(
    parquet_paths=droid_parquet_paths,
    target_action_dim=16,
    target_state_dim=16,
)

# 5. 创建数据集实例
bridge_dataset = LeRobotSingleDataset(
    dataset_path="datasets/bridge_v2",
    modality_configs=bridge_config.modality_config(),
    transform=bridge_config.transform(),
)

droid_dataset = LeRobotSingleDataset(
    dataset_path="datasets/droid",
    modality_configs=droid_config.modality_config(),
    transform=droid_config.transform(),
)

# 6. 现在两个数据集的动作和状态维度都是 16，可以混合训练
print(f"Bridge V2 action shape: {bridge_dataset[0]['action'].shape}")  # (..., 16)
print(f"Droid action shape: {droid_dataset[0]['action'].shape}")      # (..., 16)
```

## 工作原理

### 填充策略

1. **零填充**：在维度末尾添加零值
   - 原始数据：`[1.0, 2.0, 3.0]` (3维)
   - 填充到 16 维：`[1.0, 2.0, 3.0, 0.0, 0.0, ..., 0.0]` (16维)

2. **不截断**：如果当前维度 >= 目标维度，不进行任何操作
   - 原始数据：`[1.0, 2.0, ..., 20.0]` (20维)
   - 目标维度：16
   - 结果：保持 20 维不变

3. **可选填充**：如果 `target_dim = None`，不应用填充

### 变换顺序

**重要**：填充必须在归一化之前进行，以确保统计信息正确。

```
正确的顺序：
1. StateActionToTensor        # 转换为张量
2. StateActionDimensionPadding # 填充维度 ← 在这里
3. StateActionTransform        # 归一化（使用填充后的统计信息）
4. 其他变换...

错误的顺序：
1. StateActionToTensor
2. StateActionTransform        # 归一化 ← 错误！统计信息不匹配
3. StateActionDimensionPadding # 填充
```

### 统计信息的影响

填充后的维度统计信息：
- **均值 (mean)**: 0.0
- **标准差 (std)**: 0.0
- **最小值 (min)**: 0.0
- **最大值 (max)**: 0.0
- **分位数 (q01, q99)**: 0.0

这确保了填充的维度在归一化后仍然保持为 0。

## 推理时的反向操作

### 使用 unapply 恢复原始维度

在推理时，模型输出的动作可能是填充后的维度（如 16 维），需要恢复到原始维度才能发送给机器人。

```python
# 假设模型输出了 16 维的动作
model_output = {
    "action.eef_position_delta": torch.randn(1, 16),  # 填充后的 16 维
}

# 使用 unapply 恢复原始维度
padding_transform = StateActionDimensionPadding(
    apply_to=["action.eef_position_delta"],
    target_action_dim=16,
    target_state_dim=None,
)

# 先 apply 一次以记录原始维度（如果还没有记录）
# 注意：在训练时已经记录了原始维度
original_action = padding_transform.unapply(model_output)

# 现在 original_action["action.eef_position_delta"] 恢复到原始维度（如 3 维）
print(original_action["action.eef_position_delta"].shape)  # torch.Size([1, 3])
```

## 注意事项

### 1. 变换顺序很重要

✅ **正确**：先填充，后归一化
```python
StateActionToTensor(apply_to=keys),
StateActionDimensionPadding(apply_to=keys, target_action_dim=16),
StateActionTransform(apply_to=keys, normalization_modes={...}),
```

❌ **错误**：先归一化，后填充
```python
StateActionToTensor(apply_to=keys),
StateActionTransform(apply_to=keys, normalization_modes={...}),
StateActionDimensionPadding(apply_to=keys, target_action_dim=16),  # 错误！
```

### 2. 统计信息必须匹配

如果使用填充，计算统计信息时也必须使用相同的填充参数：

```python
# 计算统计信息时使用填充
stats = calculate_dataset_statistics(
    parquet_paths=paths,
    target_action_dim=16,  # 必须与配置一致
)

# 数据配置中也使用相同的填充
config.target_action_dim = 16  # 必须与统计信息一致
```

### 3. 不同数据集使用相同的目标维度

混合训练时，所有数据集应使用相同的目标维度：

```python
# 所有数据集都填充到 16 维
bridge_config.target_action_dim = 16
droid_config.target_action_dim = 16
rt1_config.target_action_dim = 16
```

## 常见问题 (FAQ)

### Q1: 为什么需要维度填充？

**A**: 不同的机器人具有不同的动作空间维度。例如：
- 7-DOF 机械臂：7 维动作（位置 + 旋转 + 夹爪）
- 双臂机器人：14 维动作（每个臂 7 维）
- 移动机械臂：10+ 维动作（底盘移动 + 机械臂）

维度填充允许我们将这些不同维度的数据统一到相同的空间，从而可以：
- 混合训练多个数据集
- 使用固定维度的模型架构
- 简化数据处理流程

### Q2: 填充会影响模型性能吗？

**A**: 影响很小，因为：
- 填充的维度全部为 0，统计信息也为 0
- 归一化后填充维度仍然是 0
- 模型可以学习忽略这些零维度
- 在推理时可以通过 `unapply()` 恢复原始维度

### Q3: 可以只填充动作维度而不填充状态维度吗？

**A**: 可以！只需设置其中一个参数：

```python
config.target_action_dim = 16  # 填充动作
config.target_state_dim = None  # 不填充状态
```

### Q4: 如果原始维度已经大于目标维度会怎样？

**A**: 不会进行截断，保持原始维度不变。例如：
- 原始维度：20
- 目标维度：16
- 结果：保持 20 维

### Q5: 填充后的数据可以和未填充的数据混合吗？

**A**: 不建议。应该保持一致性：
- 要么所有数据集都使用填充
- 要么所有数据集都不使用填充

## 最佳实践

### 1. 选择合适的目标维度

建议选择略大于最大原始维度的 2 的幂次：

```python
# 如果最大原始维度是 14
target_dim = 16  # 推荐：2^4

# 如果最大原始维度是 25
target_dim = 32  # 推荐：2^5
```

### 2. 统一配置管理

创建一个配置文件统一管理目标维度：

```python
# config.py
GLOBAL_TARGET_ACTION_DIM = 16
GLOBAL_TARGET_STATE_DIM = 16

# 在各个数据配置中使用
from config import GLOBAL_TARGET_ACTION_DIM, GLOBAL_TARGET_STATE_DIM

bridge_config.target_action_dim = GLOBAL_TARGET_ACTION_DIM
bridge_config.target_state_dim = GLOBAL_TARGET_STATE_DIM
```

### 3. 验证填充效果

在训练前验证填充是否正确应用：

```python
# 加载一个样本
sample = dataset[0]

# 检查维度
print(f"Action shape: {sample['action'].shape}")
print(f"State shape: {sample['state'].shape}")

# 验证填充的维度是否为 0
action_dim = sample['action'].shape[-1]
if action_dim > original_dim:
    padded_values = sample['action'][..., original_dim:]
    assert torch.all(padded_values == 0), "Padded values should be zero"
    print("✓ Padding verified successfully")
```

## 技术细节

### 实现的关键点

1. **填充位置**：在张量的最后一个维度末尾添加零
2. **填充时机**：在归一化之前，确保统计信息正确
3. **可逆性**：通过 `_original_dims` 字典跟踪原始维度
4. **自动识别**：根据键名前缀（`action.` 或 `state.`）自动选择目标维度

### 代码位置

- **变换类**: `starVLA/dataloader/gr00t_lerobot/transform/state_action.py:591`
- **统计函数**: `starVLA/dataloader/gr00t_lerobot/datasets.py:70`
- **数据配置**: `starVLA/dataloader/gr00t_lerobot/data_config.py`

## 总结

维度填充功能为 starVLA 提供了灵活的数据统一能力，主要优势包括：

✅ **简化混合训练**：不同具身智能体的数据可以无缝混合
✅ **保持可逆性**：推理时可以恢复原始维度
✅ **统计信息一致**：填充在归一化前进行，确保正确性
✅ **易于使用**：只需设置两个参数即可启用
✅ **模块化设计**：遵循现有的变换管道架构

### 快速开始

```python
# 1. 设置目标维度
config.target_action_dim = 16
config.target_state_dim = 16

# 2. 计算统计信息时使用相同的参数
stats = calculate_dataset_statistics(
    parquet_paths=paths,
    target_action_dim=16,
    target_state_dim=16,
)

# 3. 开始训练！
```


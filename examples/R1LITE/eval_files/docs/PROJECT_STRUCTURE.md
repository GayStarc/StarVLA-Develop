# R1LITE 项目文件结构说明

本文档说明 eval_files 文件夹的整理结构和各文件的用途。

## 📁 目录结构

```
eval_files/
├── 📄 核心接口文件
│   ├── r1lite_robot_interface.py    ⭐ ROS2机器人接口类（核心）
│   └── r1lite_inference.py          ⭐ 模型推理客户端（核心）
│
├── ⚙️ 配置文件
│   ├── deploy_policy.yml            配置文件：模型路径、服务器地址等
│   ├── requirements.txt             Python依赖包列表
│   └── run_policy_server.sh         启动模型服务器的脚本
│
├── 📖 文档
│   ├── README_ROBOT_INTERFACE.md    详细使用文档和API参考
│   └── PROJECT_STRUCTURE.md         本文档（目录结构说明）
│
├── 📂 examples/                     示例代码
│   ├── example_usage.py             ⭐ 完整推理流程示例（推荐从这里开始）
│   └── example_pose_control.py      姿态控制示例
│
├── 📂 tests/                        测试代码
│   ├── test_robot_interface.py      ⭐ 接口功能测试（推荐先运行）
│   └── test.py                      简单的EE姿态监控
│
└── 📂 tools/                        工具脚本
    ├── data_record_wholebody_go3s.py  数据录制工具
    └── replay_h5.py                   H5文件回放工具
```

## 🎯 快速开始指南

### 1. 核心文件（必须）

这些是实际部署时必须使用的核心文件：

| 文件 | 用途 | 重要性 |
|------|------|--------|
| `r1lite_robot_interface.py` | 与机器人硬件通信的ROS2接口 | ⭐⭐⭐ 必需 |
| `r1lite_inference.py` | 连接模型服务器并执行推理 | ⭐⭐⭐ 必需 |
| `deploy_policy.yml` | 配置模型路径和推理参数 | ⭐⭐⭐ 必需 |
| `run_policy_server.sh` | 启动模型服务器 | ⭐⭐⭐ 必需 |
| `requirements.txt` | Python依赖 | ⭐⭐ 重要 |

### 2. 示例代码（学习用）

用于学习如何使用核心接口：

| 文件 | 用途 | 推荐场景 |
|------|------|----------|
| `examples/example_usage.py` | 完整的推理循环示例 | 第一次使用时参考 |
| `examples/example_pose_control.py` | 姿态控制函数使用示例 | 需要手动控制机器人时参考 |

### 3. 测试代码（验证用）

用于验证系统是否正常工作：

| 文件 | 用途 | 使用时机 |
|------|------|----------|
| `tests/test_robot_interface.py` | 测试接口和显示相机画面 | 部署前测试 |
| `tests/test.py` | 监控机器人末端位姿 | 调试定位问题时使用 |

### 4. 工具脚本（可选）

数据采集和回放工具，不是部署必需：

| 文件 | 用途 | 使用场景 |
|------|------|----------|
| `tools/data_record_wholebody_go3s.py` | 录制ROS2话题数据到H5文件 | 需要收集训练数据时 |
| `tools/replay_h5.py` | 回放H5文件到机器人 | 测试录制的数据 |

## 🚀 使用流程

### 完整部署流程

```bash
# 步骤1: 启动模型服务器（终端1）
cd /path/to/StarVLA-Dev/examples/R1LITE/eval_files
bash run_policy_server.sh

# 步骤2: 测试机器人接口（终端2）
cd /path/to/StarVLA-Dev/examples/R1LITE/eval_files
python3 tests/test_robot_interface.py --duration 10.0

# 步骤3: 运行推理（确认测试通过后）
python3 examples/example_usage.py
```

## 📝 文件依赖关系

```
example_usage.py
    ├── 依赖 → r1lite_robot_interface.py (ROS2通信)
    ├── 依赖 → r1lite_inference.py (模型推理)
    └── 依赖 → deploy_policy.yml (配置)

r1lite_inference.py
    └── 需要 → 模型服务器运行 (run_policy_server.sh)

test_robot_interface.py
    └── 依赖 → r1lite_robot_interface.py
```

## 🔧 配置文件说明

### deploy_policy.yml

```yaml
# 模型检查点路径（必须修改为你的实际路径）
policy_ckpt_path: "/path/to/your/checkpoint.pt"

# 数据集归一化key（根据训练数据设置）
unnorm_key: "r1lite"

# 模型服务器地址
host: "127.0.0.1"
port: 5694

# 推理设置
image_size: [224, 224]      # 输入图像大小
use_ddim: true              # 使用DDIM采样
num_ddim_steps: 4           # DDIM步数
```

### run_policy_server.sh

需要修改的变量：
- `star_vla_python`: Python解释器路径
- `your_ckpt`: 模型检查点路径
- `gpu_id`: 使用的GPU ID
- `port`: 服务器端口（与deploy_policy.yml一致）

## 🎓 核心API速查

### R1LITERobotInterface (机器人接口)

```python
from r1lite_robot_interface import R1LITERobotInterface

robot = R1LITERobotInterface()

# 等待就绪
robot.wait_for_ready(timeout=10.0)

# 获取观测
head_img, left_img, right_img = robot.get_images()
state = robot.get_robot_state()  # 16维状态

# 单独获取各个图像（新增功能）
head_img = robot.get_head_image()
left_img = robot.get_left_image()
right_img = robot.get_right_image()

# 获取gripper状态（新增功能）
left_gripper = robot.get_left_gripper()   # 返回 [0, 1]
right_gripper = robot.get_right_gripper() # 返回 [0, 1]

# 发送delta动作
action = np.zeros(14)  # [left_xyz(3), left_euler(3), left_grip(1),
                       #  right_xyz(3), right_euler(3), right_grip(1)]
robot.send_delta_action(action)

# 姿态控制
left_pos, left_euler, left_grip = robot.get_left_pose_euler()
robot.goto_left_pose_euler(new_pos, new_euler, new_grip)
```

### R1LITEModelClient (模型推理)

```python
from r1lite_inference import create_model_client
import yaml

# 从配置文件创建
with open("deploy_policy.yml", "r") as f:
    config = yaml.safe_load(f)
model = create_model_client(config)

# 重置任务
model.reset("Pick up the red cube")

# 推理步骤
images = [head_img, left_img, right_img]
action = model.step(images, state, instruction)
```

## 🐛 故障排除

### 问题1: 找不到模块
```bash
# 解决方案：设置PYTHONPATH
export PYTHONPATH=/path/to/StarVLA-Dev:${PYTHONPATH}
```

### 问题2: ROS2话题无数据
```bash
# 检查话题列表
ros2 topic list

# 检查话题频率
ros2 topic hz /hdas/camera_head/head/color/image_raw
```

### 问题3: 模型服务器连接失败
```bash
# 检查服务器是否运行
netstat -an | grep 5694

# 检查配置文件中的host和port是否正确
```

## 📚 更多信息

详细的API文档、ROS2话题说明、开发指南等，请参考：
- **[README_ROBOT_INTERFACE.md](README_ROBOT_INTERFACE.md)** - 完整技术文档

## 📌 版本历史

- **2024-02**: 初始整理 - 按功能分类文件夹结构
- **2024-02**: 添加 `get_gripper()` 和单独图像获取功能

## 💡 提示

1. **首次使用**：建议按顺序阅读：
   - PROJECT_STRUCTURE.md (本文档) → 了解结构
   - README_ROBOT_INTERFACE.md → 了解详细API
   - examples/example_usage.py → 查看实际使用

2. **部署前测试**：
   - 先运行 `tests/test_robot_interface.py` 确认硬件连接正常
   - 再运行 `examples/example_usage.py` 进行完整测试

3. **安全第一**：
   - 真机运行前确保急停按钮可用
   - 机器人周围无障碍物
   - 有人监督运行过程

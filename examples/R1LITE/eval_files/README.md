# R1LITE 机器人推理接口

> 用于 R1LITE 双臂机器人真机任务推理的完整接口实现

## 📚 文档导航

### 快速入门
- **[QUICK_START.md](QUICK_START.md)** - ⚡ **5分钟快速启动（推荐从这里开始）**
- **[docs/STARVLA_INFERENCE_GUIDE.md](docs/STARVLA_INFERENCE_GUIDE.md)** - 📘 **StarVLA完整推理指南**

### 详细文档
- **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)** - 📁 项目结构说明
- **[docs/R1LITE.md](docs/R1LITE.md)** - 🤖 R1LITE操作手册
- **[docs/README_ROBOT_INTERFACE.md](docs/README_ROBOT_INTERFACE.md)** - 📖 API技术文档

## 🎯 快速开始

### 方法1: 使用StarVLA推理脚本（推荐）

```bash
# 1. 启动系统（按QUICK_START.md操作）

# 2. 自动模式运行
python starvla_r1lite_inference.py --task "拿起红色方块"

# 3. 手动步进模式（推荐首次使用）
python starvla_r1lite_inference.py --task "拿起红色方块" --manual-step
```

- **自动模式**: 连续执行，适合成熟任务
- **手动模式**: 逐步控制，每步显示详细状态，适合调试和学习

详细说明见:
- 自动模式: [STARVLA_INFERENCE_GUIDE.md](docs/STARVLA_INFERENCE_GUIDE.md)
- 手动模式: [MANUAL_CONTROL_GUIDE.md](MANUAL_CONTROL_GUIDE.md)

### 方法2: 基础推理流程

### 1️⃣ 启动模型服务器

```bash
bash run_policy_server.sh
```

### 2️⃣ 测试机器人接口

```bash
python3 tests/test_robot_interface.py --duration 10.0
```

### 3️⃣ 运行推理

```bash
python3 examples/example_usage.py
```

## 📁 目录结构

```
eval_files/
├── r1lite_robot_interface.py    ⭐ 核心：ROS2机器人接口
├── r1lite_inference.py          ⭐ 核心：模型推理客户端
├── deploy_policy.yml            ⚙️ 配置文件
├── requirements.txt             ⚙️ 依赖列表
├── run_policy_server.sh         ⚙️ 启动脚本
├── examples/                    📂 示例代码
│   ├── example_usage.py         - 完整推理示例
│   └── example_pose_control.py  - 姿态控制示例
├── tests/                       📂 测试代码
│   ├── test_robot_interface.py  - 接口功能测试
│   └── test.py                  - EE姿态监控
└── tools/                       📂 工具脚本
    ├── data_record_wholebody_go3s.py  - 数据录制
    └── replay_h5.py                   - H5回放
```

## 🔥 核心功能

### StarVLA推理 (starvla_r1lite_inference.py)

```bash
# 自动模式 - 连续执行
python starvla_r1lite_inference.py --task "拿起红色方块"

# 手动步进模式 - 逐步控制（新功能）
python starvla_r1lite_inference.py --task "拿起红色方块" --manual-step
```

**手动模式特性**：
- ✅ 每步显示完整的机械臂状态（位置、姿态、夹爪）
- ✅ 每步显示模型输出动作（增量和幅度）
- ✅ 支持单步执行、跳过N步、切换连续模式
- ✅ 适合调试、学习和安全验证

### 机器人接口 (r1lite_robot_interface.py)

```python
from r1lite_robot_interface import R1LITERobotInterface

robot = R1LITERobotInterface()
robot.wait_for_ready(timeout=10.0)

# 获取观测
head_img, left_img, right_img = robot.get_images()
state = robot.get_robot_state()

# 新增：单独获取图像
head_img = robot.get_head_image()
left_img = robot.get_left_image()
right_img = robot.get_right_image()

# 新增：获取gripper状态
left_gripper = robot.get_left_gripper()
right_gripper = robot.get_right_gripper()

# 发送动作
robot.send_delta_action(action)
```

### 模型推理 (r1lite_inference.py)

```python
from r1lite_inference import create_model_client
import yaml

with open("deploy_policy.yml", "r") as f:
    config = yaml.safe_load(f)
model = create_model_client(config)

model.reset("Pick up the red cube")
action = model.step(images, state, instruction)
```

## ⚙️ 配置

编辑 `deploy_policy.yml`:

```yaml
policy_ckpt_path: "/path/to/your/checkpoint.pt"
unnorm_key: "r1lite"
host: "127.0.0.1"
port: 5694
```

编辑 `run_policy_server.sh`:

```bash
star_vla_python=/path/to/your/python
your_ckpt=/path/to/checkpoint.pt
gpu_id=0
```

## 🎓 学习路径

1. **新手入门** → 阅读 [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)
2. **了解API** → 阅读 [README_ROBOT_INTERFACE.md](README_ROBOT_INTERFACE.md)
3. **查看示例** → 运行 `examples/example_usage.py`
4. **深入学习** → 研究核心代码 `r1lite_robot_interface.py` 和 `r1lite_inference.py`

## ⚠️ 安全提示

- 🛑 确保急停按钮可用
- 👀 有人监督运行过程
- 🚫 机器人周围无障碍物
- ✅ 先在仿真环境测试

## 📊 数据格式

**状态** (16维):
```
[left_xyz(3), left_euler(3), left_gripper(1),
 right_xyz(3), right_euler(3), right_gripper(1)]
```

**动作** (14维):
```
[left_delta_xyz(3), left_delta_euler(3), left_gripper(1),
 right_delta_xyz(3), right_delta_euler(3), right_gripper(1)]
```

## 🔗 相关链接

- 完整项目文档: [README_ROBOT_INTERFACE.md](README_ROBOT_INTERFACE.md)
- 项目结构说明: [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)

---

**最后更新**: 2024-02-06

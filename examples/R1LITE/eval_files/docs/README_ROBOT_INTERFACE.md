# R1LITE 双臂机器人推理接口

本目录包含用于 R1LITE 双臂机器人真机任务推理的完整接口实现。

## 文件说明

### 核心文件

1. **r1lite_robot_interface.py** - ROS2 机器人接口类
   - 订阅三个相机话题（头部、左腕、右腕）
   - 订阅机器人状态话题（末端执行器位姿和夹爪状态）
   - 发布动作到机器人控制话题
   - 提供线程安全的观测访问接口

2. **r1lite_inference.py** - 模型推理客户端
   - 连接到 WebSocket 策略服务器
   - 处理图像预处理和状态归一化
   - 支持动作分块和集成
   - 返回 delta 动作

3. **example_usage.py** - 完整的推理示例
   - 展示如何初始化模型和机器人接口
   - 实现完整的推理循环
   - 集成观测获取和动作执行

4. **test_robot_interface.py** - 接口测试脚本
   - 验证机器人接口是否正常工作
   - 打印观测信息
   - 可选地发送测试动作

### 参考文件

- **data_record_wholebody_go3s.py** - 数据记录节点
- **replay_h5.py** - H5 文件回放
- **test.py** - 简单的 EE pose 监控

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                     R1LITE 推理系统                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────┐         ┌──────────────────┐         │
│  │  Model Server    │◄────────┤ R1LITE Model     │         │
│  │  (WebSocket)     │         │ Client           │         │
│  └──────────────────┘         └──────────────────┘         │
│                                        │                     │
│                                        │ action (14-dim)     │
│                                        ▼                     │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         R1LITE Robot Interface (ROS2 Node)           │  │
│  ├──────────────────────────────────────────────────────┤  │
│  │  • Subscribe: 3 cameras + robot state                │  │
│  │  • Publish: EE poses + gripper commands              │  │
│  │  • Delta integration                                 │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │  R1LITE Robot    │
                    │  (Real Hardware) │
                    └──────────────────┘
```

## 数据格式

### 观测 (Observations)

**图像**:
- 头部相机: `/hdas/camera_head/head/color/image_raw` (H×W×3, uint8)
- 左腕相机: `/hdas/camera_wrist_left/color/image_raw` (H×W×3, uint8)
- 右腕相机: `/hdas/camera_wrist_right/color/image_raw` (H×W×3, uint8)

**状态** (16-dim):
```
[left_xyz(3), left_euler(3), left_gripper(1),
 right_xyz(3), right_euler(3), right_gripper(1)]
```

### 动作 (Actions)

**Delta 动作** (14-dim):
```
[left_delta_xyz(3), left_delta_euler(3), left_gripper(1),
 right_delta_xyz(3), right_delta_euler(3), right_gripper(1)]
```

- xyz: 位置增量 (米)
- euler: 欧拉角增量 (弧度, xyz 顺序)
- gripper: 夹爪状态 [0, 1] (0=打开, 1=关闭)

## 使用方法

### 1. 环境准备

确保已安装以下依赖：
```bash
# ROS2 依赖
sudo apt install ros-<distro>-sensor-msgs ros-<distro>-geometry-msgs

# Python 依赖
pip install numpy opencv-python pyyaml h5py
```

### 2. 启动模型服务器

首先启动策略服务器：
```bash
cd /path/to/StarVLA-Dev
bash examples/R1LITE/eval_files/run_policy_server.sh
```

### 3. 测试机器人接口

在发送动作之前，先测试接口是否正常：
```bash
cd examples/R1LITE/eval_files
python3 test_robot_interface.py --duration 10.0
```

如果要测试动作发送（谨慎使用）：
```bash
python3 test_robot_interface.py --send-actions --duration 5.0
```

### 4. 运行推理

编辑 `deploy_policy.yml` 配置文件，然后运行：
```bash
python3 example_usage.py
```

## 配置文件

`deploy_policy.yml` 示例：
```yaml
policy_ckpt_path: "/path/to/checkpoint"
unnorm_key: "r1lite_dataset"
host: "127.0.0.1"
port: 5694
image_size: [224, 224]
use_ddim: true
num_ddim_steps: 4
```

## ROS2 话题

### 订阅话题 (Subscriptions)

| 话题 | 类型 | 说明 |
|------|------|------|
| `/hdas/camera_head/head/color/image_raw` | sensor_msgs/Image | 头部相机图像 |
| `/hdas/camera_wrist_left/color/image_raw` | sensor_msgs/Image | 左腕相机图像 |
| `/hdas/camera_wrist_right/color/image_raw` | sensor_msgs/Image | 右腕相机图像 |
| `/motion_control/pose_ee_arm_left` | geometry_msgs/PoseStamped | 左臂末端位姿 |
| `/motion_control/pose_ee_arm_right` | geometry_msgs/PoseStamped | 右臂末端位姿 |
| `/motion_target/target_position_gripper_left` | sensor_msgs/JointState | 左夹爪状态 |
| `/motion_target/target_position_gripper_right` | sensor_msgs/JointState | 右夹爪状态 |

### 发布话题 (Publications)

| 话题 | 类型 | 说明 |
|------|------|------|
| `/motion_target/target_pose_arm_left` | geometry_msgs/PoseStamped | 左臂目标位姿 |
| `/motion_target/target_pose_arm_right` | geometry_msgs/PoseStamped | 右臂目标位姿 |
| `/motion_target/target_position_gripper_left` | sensor_msgs/JointState | 左夹爪目标位置 |
| `/motion_target/target_position_gripper_right` | sensor_msgs/JointState | 右夹爪目标位置 |

## API 参考

### R1LITERobotInterface

```python
from r1lite_robot_interface import R1LITERobotInterface

# 初始化
robot = R1LITERobotInterface(
    node_name="my_robot_node",
    publish_rate=10.0,
)

# 等待就绪
robot.wait_for_ready(timeout=10.0)

# 获取观测
head_img, left_img, right_img = robot.get_images()
state = robot.get_robot_state()

# 发送动作
action = np.zeros(14)  # delta action
robot.send_delta_action(action)

# 检查就绪状态
if robot.is_ready():
    print("Robot ready!")
```

### R1LITEModelClient

```python
from r1lite_inference import create_model_client

# 从配置创建
model = create_model_client(config)

# 重置任务
model.reset("Pick up the cube")

# 推理步骤
action = model.step(images, state, instruction)
```

## 注意事项

1. **安全第一**: 在真机上运行前，请确保：
   - 机器人周围无障碍物
   - 急停按钮可用
   - 有人监督运行过程

2. **坐标系**: 所有位姿都在 `base_link` 坐标系下

3. **Delta 积分**: 接口内部维护当前位姿状态，自动积分 delta 动作

4. **线程安全**: 所有观测访问都是线程安全的

5. **QoS 设置**: 使用 BEST_EFFORT 可靠性策略以减少延迟

## 故障排除

### 问题: 无法接收图像

**解决方案**:
- 检查相机话题是否发布: `ros2 topic list`
- 检查话题频率: `ros2 topic hz /hdas/camera_head/head/color/image_raw`
- 确认 QoS 设置匹配

### 问题: 机器人不响应动作

**解决方案**:
- 检查控制话题是否有订阅者: `ros2 topic info /motion_target/target_pose_arm_left`
- 确认机器人控制器正在运行
- 检查动作数值是否合理（不要太大）

### 问题: 模型服务器连接失败

**解决方案**:
- 确认服务器已启动: `netstat -an | grep 5694`
- 检查 host 和 port 配置
- 查看服务器日志

## 开发指南

### 自定义任务完成检测

在 `example_usage.py` 中修改 `check_task_done()` 函数：

```python
def check_task_done() -> bool:
    # 示例: 检查夹爪状态
    state = robot.get_robot_state()
    left_gripper = state[6]
    right_gripper = state[13]

    # 如果两个夹爪都关闭，认为任务完成
    return left_gripper > 0.9 and right_gripper > 0.9
```

### 添加额外的传感器

在 `R1LITERobotInterface` 类中添加新的订阅者：

```python
self.sub_force = self.create_subscription(
    WrenchStamped,
    "/force_sensor",
    self._force_callback,
    qos,
)
```

## 许可证

请参考项目根目录的 LICENSE 文件。

## 联系方式

如有问题，请提交 Issue 或联系项目维护者。

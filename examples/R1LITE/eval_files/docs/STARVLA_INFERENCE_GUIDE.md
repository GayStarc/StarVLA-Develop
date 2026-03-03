# StarVLA R1LITE 推理完整操作指南

> **文档版本**: v1.0  
> **最后更新**: 2024-02-06  
> **适用系统**: R1LITE双臂机器人 + StarVLA模型

## 📑 目录

- [系统要求](#系统要求)
- [快速开始](#快速开始)
- [详细操作流程](#详细操作流程)
  - [步骤1: 环境准备](#步骤1-环境准备)
  - [步骤2: 启动机器人系统](#步骤2-启动机器人系统)
  - [步骤3: 启动相机系统](#步骤3-启动相机系统)
  - [步骤4: 验证系统状态](#步骤4-验证系统状态)
  - [步骤5: 启动模型服务器](#步骤5-启动模型服务器)
  - [步骤6: 运行推理](#步骤6-运行推理)
- [高级配置](#高级配置)
- [故障排查](#故障排查)
- [安全注意事项](#安全注意事项)

---

## 系统要求

### 硬件要求

- R1LITE双臂机器人（192.168.1.128）
- 工作站PC（Ubuntu 20.04/22.04）
- NVIDIA GPU（推荐RTX 3090或更高）
- 3个相机：
  - 头顶相机：RealSense D435
  - 左腕相机：GO3S
  - 右腕相机：GO3S

### 软件要求

- ROS2 (Humble或更高版本)
- Python 3.10
- CUDA 11.8+
- 已安装StarVLA环境（conda环境：starVLA）

### 网络要求

- PC和机器人在同一局域网
- 机器人IP: 192.168.1.128
- PC可以SSH登录机器人

---

## 快速开始

> 💡 如果你已经熟悉完整流程，可以使用这个快速启动清单

### 快速启动清单

```bash
# ✅ 1. 启动Bot端机器人控制（终端1 - Bot）
ssh r1lite@192.168.1.128
cd ~/galaxea/install/startup_config/share/startup_config/script/
./robot_startup.sh boot ../sessions.d/ATCStandard/R1LITEBody.d
# 新终端
source ~/galaxea/install/setup.bash
ros2 launch mobiman r1_lite_right_arm_relaxed_ik_launch.py

# ✅ 2. 启动相机（终端2 - PC或Bot）
IMAGE_WIDTH=1280 IMAGE_HEIGHT=720 FRAMERATE=30.0 \
/home/robot/Desktop/wangyinxi/WristInsta360_Project/scripts/wrist_cameras_start_all.sh
# 新终端
/home/robot/Desktop/wangyinxi/WristInsta360_Project/scripts/realsense_head_launch.sh

# ✅ 3. 验证相机（终端3 - PC）
ros2 run image_view image_view --ros-args -r image:=/hdas/camera_head/head/color/image_raw

# ✅ 4. 启动模型服务器（终端4 - PC）
cd /home/robot/guchenyang/Code/StarVLA-Dev/examples/R1LITE/eval_files
bash run_policy_server.sh

# ✅ 5. 运行推理（终端5 - PC）
cd /home/robot/guchenyang/Code/StarVLA-Dev/examples/R1LITE/eval_files
python starvla_r1lite_inference.py --task "拿起红色方块"
```

---

## 详细操作流程

### 步骤1: 环境准备

#### 1.1 检查网络连接

```bash
# 在PC端测试与机器人的连接
ping 192.168.1.128

# 测试SSH连接
ssh r1lite@192.168.1.128
# 密码：（联系管理员获取）
```

#### 1.2 激活StarVLA环境

```bash
# 在PC端激活conda环境
conda activate starVLA

# 验证Python版本
python --version  # 应该是 Python 3.10.x

# 验证必要的包
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import rclpy; print('ROS2: OK')"
```

#### 1.3 确认文件路径

```bash
cd /home/robot/guchenyang/Code/StarVLA-Dev/examples/R1LITE/eval_files

# 检查必要文件
ls -lh r1lite_robot_interface.py
ls -lh r1lite_inference.py
ls -lh starvla_r1lite_inference.py
ls -lh deploy_policy.yml
ls -lh run_policy_server.sh

# 检查模型checkpoint
ls -lh /home/robot/guchenyang/Code/StarVLA-Dev/playground/Finetuned/QwenPI/checkpoints/steps_10000_pytorch_model.pt
```

---

### 步骤2: 启动机器人系统

#### 2.1 登录机器人（Bot端）

打开**终端1**：

```bash
# SSH登录到机器人
ssh r1lite@192.168.1.128

# 确认当前在Bot端
hostname  # 应该显示 r1lite 相关名称
```

#### 2.2 启动机器人全身控制

在Bot端终端1执行：

```bash
# 进入启动脚本目录
cd ~/galaxea/install/startup_config/share/startup_config/script/

# 启动机器人控制系统
./robot_startup.sh boot ../sessions.d/ATCStandard/R1LITEBody.d
```

**预期输出**：
```
[INFO] Starting robot control system...
[INFO] Loading configuration...
[INFO] Robot system initialized successfully
```

> ⏱️ **等待时间**: 约10-15秒

#### 2.3 启动右臂IK控制

打开**终端2**（Bot端）：

```bash
# SSH登录（如果还没有）
ssh r1lite@192.168.1.128

# 设置ROS2环境
source ~/galaxea/install/setup.bash

# 启动右臂IK
ros2 launch mobiman r1_lite_right_arm_relaxed_ik_launch.py
```

**预期输出**：
```
[INFO] [launch]: All log files can be found below...
[INFO] [relaxed_ik]: RelaxedIK node started
[INFO] Right arm IK solver initialized
```

> ✅ **验证**: 应该看到IK节点运行的日志信息

---

### 步骤3: 启动相机系统

#### 3.1 确定相机连接位置

相机可能连接在PC端或Bot端，需要先确认：

```bash
# 在PC端检查
ls /dev/video*

# 在Bot端检查（SSH登录后）
ssh r1lite@192.168.1.128
ls /dev/video*
```

#### 3.2 启动腕部相机（GO3S）

**情况A：相机在PC端**

打开**终端3**（PC端）：

```bash
# 设置相机参数并启动
IMAGE_WIDTH=1280 IMAGE_HEIGHT=720 FRAMERATE=30.0 \
/home/robot/Desktop/wangyinxi/WristInsta360_Project/scripts/wrist_cameras_start_all.sh
```

**情况B：相机在Bot端**

打开**终端3**（Bot端）：

```bash
ssh r1lite@192.168.1.128

IMAGE_WIDTH=1280 IMAGE_HEIGHT=720 FRAMERATE=30.0 \
/home/r1lite/Desktop/WristInsta360_Project/scripts/wrist_cameras_start_all.sh
```

**预期输出**：
```
[INFO] Starting left wrist camera...
[INFO] Starting right wrist camera...
[INFO] Cameras initialized successfully
[INFO] Publishing to topics:
  - /hdas/camera_wrist_left/color/image_raw
  - /hdas/camera_wrist_right/color/image_raw
```

#### 3.3 启动头顶相机（RealSense）

打开**终端4**（根据相机位置选择PC或Bot端）：

**相机在PC端：**
```bash
/home/robot/Desktop/wangyinxi/WristInsta360_Project/scripts/realsense_head_launch.sh
```

**相机在Bot端：**
```bash
ssh r1lite@192.168.1.128
/home/r1lite/Desktop/WristInsta360_Project/scripts/realsense_head_launch.sh
```

**预期输出**：
```
[INFO] RealSense camera node started
[INFO] Publishing to topic: /hdas/camera_head/head/color/image_raw
```

---

### 步骤4: 验证系统状态

#### 4.1 检查ROS2话题

在PC端打开**终端5**：

```bash
# 列出所有话题
ros2 topic list

# 应该包含以下话题：
# /hdas/camera_head/head/color/image_raw
# /hdas/camera_wrist_left/color/image_raw
# /hdas/camera_wrist_right/color/image_raw
# /motion_control/pose_ee_arm_left
# /motion_control/pose_ee_arm_right
# /motion_target/target_position_gripper_left
# /motion_target/target_position_gripper_right
```

#### 4.2 检查话题频率

```bash
# 检查头顶相机频率（应该是 ~30Hz）
ros2 topic hz /hdas/camera_head/head/color/image_raw

# 检查左腕相机频率
ros2 topic hz /hdas/camera_wrist_left/color/image_raw

# 检查右腕相机频率
ros2 topic hz /hdas/camera_wrist_right/color/image_raw
```

> 💡 **提示**: 按Ctrl+C停止检查，继续下一步

#### 4.3 查看相机画面（可选但推荐）

```bash
# 查看头顶相机
ros2 run image_view image_view --ros-args \
    -r image:=/hdas/camera_head/head/color/image_raw
```

在弹出的窗口中应该能看到头顶相机的实时画面。

```bash
# 查看左腕相机（新终端）
ros2 run image_view image_view --ros-args \
    -r image:=/hdas/camera_wrist_left/color/image_raw

# 查看右腕相机（新终端）
ros2 run image_view image_view --ros-args \
    -r image:=/hdas/camera_wrist_right/color/image_raw
```

> ✅ **验证标准**：
> - 3个相机画面都能正常显示
> - 画面清晰无花屏
> - 画面流畅无明显卡顿
> - 左右腕相机对应正确（不要搞反）

#### 4.4 检查机器人末端位姿

```bash
# 查看右臂末端位姿
ros2 topic echo /motion_control/pose_ee_arm_right

# 应该看到类似输出：
# pose:
#   position:
#     x: 0.059
#     y: -0.332
#     z: 0.343
#   orientation:
#     x: 0.0
#     y: 0.0
#     z: 0.0
#     w: 1.0
```

记录这些初始位置值，确认与预期一致。

#### 4.5 运行接口测试（推荐）

```bash
cd /home/robot/guchenyang/Code/StarVLA-Dev/examples/R1LITE/eval_files

# 运行10秒测试，不发送动作
python tests/test_robot_interface.py --duration 10.0

# 如果想测试动作发送（谨慎！）
# python tests/test_robot_interface.py --duration 5.0 --send-actions
```

**预期输出**：
```
[TEST] Waiting for robot interface to be ready...
[TEST] Robot interface ready!
[TEST] Running test for 10.0 seconds...

[Step 0] Observations:
  Head image: (720, 1280, 3)
  Left image: (720, 1280, 3)
  Right image: (720, 1280, 3)
  State shape: (16,)
  Left EE: xyz=[0.059 -0.332 0.343], euler=[...], gripper=0.00
  Right EE: xyz=[0.0585 -0.332 0.343], euler=[...], gripper=0.00
```

> ✅ **验证通过标准**：所有观测数据都能正常获取，无报错

---

### 步骤5: 启动模型服务器

#### 5.1 检查配置文件

在PC端打开**终端6**：

```bash
cd /home/robot/guchenyang/Code/StarVLA-Dev/examples/R1LITE/eval_files

# 查看配置文件
cat deploy_policy.yml
```

确认以下配置项：
- `policy_ckpt_path`: 模型checkpoint路径正确
- `unnorm_key`: 设置为 "r1lite"
- `host`: "127.0.0.1"
- `port`: 5694
- `use_ddim`: true
- `num_ddim_steps`: 4

#### 5.2 编辑服务器启动脚本（如需要）

```bash
# 查看启动脚本
cat run_policy_server.sh

# 如需修改，编辑以下变量：
# - star_vla_python: Python解释器路径
# - your_ckpt: checkpoint路径
# - gpu_id: GPU编号
# - port: 服务器端口
```

#### 5.3 启动模型服务器

```bash
# 进入项目根目录
cd /home/robot/guchenyang/Code/StarVLA-Dev

# 启动服务器
bash examples/R1LITE/eval_files/run_policy_server.sh
```

**预期输出**：
```
Loading checkpoint from: /home/robot/guchenyang/Code/StarVLA-Dev/playground/Finetuned/QwenPI/checkpoints/steps_10000_pytorch_model.pt
[INFO] Model loaded successfully
[INFO] Starting WebSocket server on 127.0.0.1:5694
[INFO] Server is ready to accept connections
```

> ⏱️ **加载时间**: 约30-60秒（取决于模型大小和GPU性能）

> ⚠️ **重要**: 保持此终端运行，不要关闭！

#### 5.4 验证服务器连接

在新终端中测试：

```bash
# 检查端口是否监听
netstat -an | grep 5694
# 应该看到：tcp  0  0  127.0.0.1:5694  0.0.0.0:*  LISTEN

# 或使用
lsof -i :5694
```

---

### 步骤6: 运行推理

#### 6.1 准备工作

在PC端打开**终端7**：

```bash
# 进入eval_files目录
cd /home/robot/guchenyang/Code/StarVLA-Dev/examples/R1LITE/eval_files

# 确认脚本可执行
chmod +x starvla_r1lite_inference.py

# 激活环境
conda activate starVLA
```

#### 6.2 查看帮助信息

```bash
python starvla_r1lite_inference.py --help
```

#### 6.3 运行推理（基本用法）

```bash
# 示例1: 抓取红色方块
python starvla_r1lite_inference.py --task "拿起红色方块"

# 示例2: 使用英文指令
python starvla_r1lite_inference.py --task "Pick up the red cube"

# 示例3: 放置物体
python starvla_r1lite_inference.py --task "把方块放到盘子上"
```

#### 6.4 运行推理（高级选项）

```bash
# 指定最大步数和控制频率
python starvla_r1lite_inference.py \
    --task "拿起香蕉" \
    --max-steps 500 \
    --freq 10.0

# 指定日志目录
python starvla_r1lite_inference.py \
    --task "抓取苹果" \
    --log-dir logs/apple_task_20240206

# 禁用安全检查（谨慎使用！）
python starvla_r1lite_inference.py \
    --task "移动到目标位置" \
    --no-safety-check
```

#### 6.5 监控推理过程

推理运行时，你会看到类似输出：

```
================================================================================
StarVLA R1LITE 推理系统初始化
================================================================================
配置文件: deploy_policy.yml
任务指令: 拿起红色方块
最大步数: 300
控制频率: 10.0 Hz
安全检查: 启用
日志目录: logs/starvla_inference_20240206_150530
================================================================================

[INIT] 初始化ROS2...
[INIT] 加载配置文件: deploy_policy.yml
[INIT] 连接模型服务器...
[INIT] ✓ 模型服务器连接成功 (127.0.0.1:5694)
[INIT] 初始化机器人接口...
[INIT] 等待机器人接口就绪...
[INIT] ✓ 机器人接口就绪

[INIT] 获取机器人初始位置...
[INIT] 左臂位置: [0.059 -0.332 0.343]
[INIT] 左臂姿态: [0.0 0.0 0.0]
[INIT] 左臂夹爪: 0.00
[INIT] 右臂位置: [0.0585 -0.332 0.343]
[INIT] 右臂姿态: [0.0 0.0 0.0]
[INIT] 右臂夹爪: 0.00

[INIT] 重置模型任务: '拿起红色方块'

================================================================================
✓ 初始化完成！准备开始推理...
================================================================================

================================================================================
开始推理：拿起红色方块
================================================================================
提示：按 Ctrl+C 可以随时安全停止

[Step 000] 时间: 0.1s
  推理耗时: 45.2ms
  左臂位置: [0.059, -0.332, 0.343]
  右臂位置: [0.059, -0.332, 0.343]
  左臂动作: [0.0012, -0.0003, 0.0005] 夹爪: 0.00
  右臂动作: [0.0008, -0.0002, 0.0003] 夹爪: 0.00

[Step 010] 时间: 1.2s
  推理耗时: 42.8ms
  左臂位置: [0.061, -0.333, 0.344]
  右臂位置: [0.060, -0.333, 0.344]
  左臂动作: [0.0018, -0.0005, 0.0008] 夹爪: 0.00
  右臂动作: [0.0015, -0.0004, 0.0006] 夹爪: 0.00

...

[Step 150] 时间: 15.8s
  推理耗时: 43.5ms
  左臂位置: [0.125, -0.280, 0.385]
  右臂位置: [0.122, -0.282, 0.383]
  左臂动作: [0.0002, 0.0001, -0.0001] 夹爪: 1.00
  右臂动作: [0.0001, 0.0001, -0.0001] 夹爪: 1.00

[TASK] 检测到动作趋近于零，可能已完成

================================================================================
✓ 任务完成！
================================================================================

================================================================================
推理统计信息
================================================================================
任务指令: 拿起红色方块
总步数: 152
成功步数: 152
失败步数: 0
总耗时: 16.23秒
平均推理时间: 43.8ms
实际控制频率: 9.4 Hz
================================================================================

统计信息已保存到: logs/starvla_inference_20240206_150530/statistics.txt

[CLEANUP] 清理资源...
[CLEANUP] ✓ 机器人接口已关闭
[CLEANUP] ✓ ROS2已关闭
[CLEANUP] 清理完成
```

#### 6.6 紧急停止

如果需要紧急停止推理：

1. **方法1**: 按 `Ctrl+C` （推荐）
   - 脚本会捕获信号并安全停止
   - 会保存统计信息并清理资源

2. **方法2**: 在另一个终端杀死进程
   ```bash
   # 查找进程
   ps aux | grep starvla_r1lite_inference
   
   # 杀死进程（使用SIGTERM，不要使用SIGKILL）
   kill <PID>
   ```

3. **方法3**: 使用机器人急停按钮（最安全）

---

## 高级配置

### 自定义任务完成检测

编辑 `starvla_r1lite_inference.py`，修改 `check_task_completion` 方法：

```python
def check_task_completion(self, step: int, state: np.ndarray, action: np.ndarray) -> bool:
    """自定义任务完成逻辑"""
    
    # 示例1: 检查物体位置
    # target_pos = np.array([0.2, -0.1, 0.4])
    # current_pos = state[0:3]  # 左臂位置
    # distance = np.linalg.norm(current_pos - target_pos)
    # if distance < 0.02:  # 2cm内
    #     return True
    
    # 示例2: 检查夹爪状态
    # left_gripper = state[6]
    # right_gripper = state[13]
    # if left_gripper > 0.9 and step > 50:
    #     return True
    
    # 示例3: 检查动作收敛
    action_magnitude = np.linalg.norm(action[:6])
    if action_magnitude < 0.001 and step > 50:
        return True
    
    return False
```

### 调整安全边界

编辑 `starvla_r1lite_inference.py`，修改 `__init__` 方法中的 `position_limits`：

```python
self.position_limits = {
    'x': (-0.3, 0.5),  # 前后方向
    'y': (-0.6, 0.2),  # 左右方向
    'z': (0.0, 0.8),   # 上下方向
}
```

### 修改控制参数

```python
# 调整控制频率（Hz）
--freq 15.0  # 更高频率，更平滑但计算负担更大

# 调整最大步数
--max-steps 500  # 更长的任务时间
```

### 多任务批处理

创建任务列表文件 `tasks.txt`:
```
拿起红色方块
把方块放到盘子上
拿起蓝色杯子
把杯子放回原位
```

运行批处理脚本：
```bash
#!/bin/bash
while IFS= read -r task; do
    echo "执行任务: $task"
    python starvla_r1lite_inference.py --task "$task" --log-dir "logs/batch_$(date +%Y%m%d_%H%M%S)"
    sleep 5  # 任务间隔
done < tasks.txt
```

---

## 故障排查

### 问题1: 模型服务器连接失败

**症状**：
```
[ERROR] Failed to connect to model server at 127.0.0.1:5694
```

**解决方案**：
1. 检查服务器是否运行：`netstat -an | grep 5694`
2. 检查配置文件中的host和port是否正确
3. 检查防火墙设置
4. 重启模型服务器

### 问题2: 机器人接口未就绪

**症状**：
```
[INIT] ✗ 错误：机器人接口未就绪！
```

**解决方案**：
1. 检查ROS2话题是否发布：`ros2 topic list`
2. 检查相机节点是否运行
3. 检查机器人控制节点是否正常
4. 重新启动相机系统

### 问题3: 安全检查失败

**症状**：
```
[SAFETY] ✗ 安全检查失败: 左臂x轴超出安全范围
```

**解决方案**：
1. 立即停止推理（Ctrl+C或急停按钮）
2. 检查机器人初始位置是否正确
3. 检查安全边界设置是否合理
4. 手动将机器人移动到安全位置
5. 如确认安全，可临时使用 `--no-safety-check`

### 问题4: 图像数据无法获取

**症状**：
```
[Step 000] 等待图像数据...
[Step 001] 等待图像数据...
```

**解决方案**：
1. 检查相机是否正常启动
2. 使用 `ros2 topic hz` 检查图像话题频率
3. 使用 `image_view` 查看图像是否正常
4. 重启相机节点

### 问题5: CUDA内存不足

**症状**：
```
RuntimeError: CUDA out of memory
```

**解决方案**：
1. 关闭其他GPU程序
2. 减小batch size（修改模型配置）
3. 使用更大显存的GPU
4. 清理GPU缓存：
   ```bash
   nvidia-smi
   # 如有必要，重启相关进程
   ```

### 问题6: 推理速度过慢

**症状**：
推理时间 > 100ms，实际频率 < 5Hz

**解决方案**：
1. 检查GPU负载：`nvidia-smi`
2. 减少DDIM步数：修改 `deploy_policy.yml` 中的 `num_ddim_steps`
3. 使用更快的GPU
4. 检查是否有其他程序占用GPU

### 问题7: 机械臂不动

**症状**：
推理正常运行，但机械臂没有动作

**解决方案**：
1. 检查是否有遥操进程在运行（与推理互斥）
   ```bash
   pkill -f mobiman_tabletop_tele_node
   ```
2. 检查动作幅度是否太小
3. 检查机器人控制节点是否正常接收命令
4. 查看ROS2话题：`ros2 topic echo /motion_target/target_pose_arm_right`

---

## 安全注意事项

### ⚠️ 操作前检查清单

运行推理前，**必须**确认以下事项：

- [ ] 机器人周围2米内无人员和障碍物
- [ ] 急停按钮触手可及且功能正常
- [ ] 机器人工作空间内无易碎或危险物品
- [ ] 有至少一人全程监控机器人运行
- [ ] 已测试过接口和相机（运行test_robot_interface.py）
- [ ] 了解如何紧急停止（Ctrl+C或急停按钮）
- [ ] 确认动作幅度在安全范围内

### 🛑 紧急情况处理

**立即按下急停按钮，如果：**
- 机器人动作异常或失控
- 机器人碰到人或物体
- 听到异常声音
- 看到异常运动轨迹
- 任何不确定的情况

**按下急停后：**
1. 保持冷静，不要立即重启
2. 检查机器人和周围环境
3. 确认无损坏和危险
4. 联系技术人员评估
5. 记录事故情况

### 📋 最佳实践

1. **首次运行**：
   - 先在空旷环境测试
   - 使用较小的 `--max-steps` （如50）
   - 仔细观察前几步动作
   - 确认动作符合预期后再继续

2. **日常使用**：
   - 每次启动前检查系统状态
   - 定期检查相机画面质量
   - 记录异常情况
   - 保持工作区域整洁

3. **实验记录**：
   - 使用 `--log-dir` 为每次实验创建独立日志
   - 记录任务描述、初始状态、结果
   - 保存重要的推理视频

---

## 附录

### A. 常用命令速查

```bash
# 检查ROS2话题
ros2 topic list
ros2 topic hz <topic_name>
ros2 topic echo <topic_name>

# 检查GPU状态
nvidia-smi
watch -n 1 nvidia-smi

# 检查网络端口
netstat -an | grep <port>
lsof -i :<port>

# 查看进程
ps aux | grep <keyword>
htop

# SSH隧道（如果需要）
ssh -L 5694:localhost:5694 robot@<pc_ip>
```

### B. 文件结构

```
eval_files/
├── starvla_r1lite_inference.py   # 主推理脚本
├── r1lite_robot_interface.py     # 机器人接口
├── r1lite_inference.py           # 模型推理客户端
├── deploy_policy.yml             # 配置文件
├── run_policy_server.sh          # 服务器启动脚本
├── docs/
│   ├── STARVLA_INFERENCE_GUIDE.md     # 本文档
│   ├── R1LITE.md                       # 操作手册
│   └── README_ROBOT_INTERFACE.md      # API文档
├── examples/
│   ├── example_usage.py          # 基础示例
│   └── example_pose_control.py   # 姿态控制示例
├── tests/
│   ├── test_robot_interface.py   # 接口测试
│   └── test.py                   # 简单测试
└── logs/                          # 推理日志（自动创建）
```

### C. 参考资料

- **StarVLA项目**: [GitHub链接]
- **R1LITE文档**: 参见 `docs/R1LITE.md`
- **API文档**: 参见 `docs/README_ROBOT_INTERFACE.md`
- **项目结构**: 参见 `PROJECT_STRUCTURE.md`

---

## 联系与支持

如遇到问题或需要帮助：

1. 查看本文档的故障排查章节
2. 查看日志文件：`logs/starvla_inference_*/`
3. 提交Issue到GitHub仓库
4. 联系技术支持团队

**文档维护者**: StarVLA Team  
**最后更新**: 2024-02-06  
**文档版本**: v1.0

---

祝使用顺利！🚀

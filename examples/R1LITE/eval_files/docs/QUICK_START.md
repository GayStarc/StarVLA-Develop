# StarVLA R1LITE 快速启动指南

> 💡 5分钟快速上手 - 用于已熟悉系统的用户

## 🚀 一键启动（推荐使用tmux）

### 创建tmux会话

```bash
# 创建新的tmux会话
tmux new -s r1lite_inference

# 或者附加到现有会话
tmux attach -t r1lite_inference
```

### 终端布局

```
┌─────────────────────────┬─────────────────────────┐
│  1. Bot控制             │  2. Bot IK              │
│  (Bot端)                │  (Bot端)                │
├─────────────────────────┼─────────────────────────┤
│  3. 腕部相机            │  4. 头顶相机            │
│  (PC/Bot)               │  (PC/Bot)               │
├─────────────────────────┼─────────────────────────┤
│  5. 模型服务器          │  6. StarVLA推理         │
│  (PC)                   │  (PC)                   │
└─────────────────────────┴─────────────────────────┘
```

## 📋 启动命令清单

### 1️⃣ Bot端 - 机器人控制（2个终端）

**终端1: Bot控制**
```bash
ssh r1lite@192.168.1.128
cd ~/galaxea/install/startup_config/share/startup_config/script/
./robot_startup.sh boot ../sessions.d/ATCStandard/R1LITEBody.d
```

**终端2: 右臂IK**
```bash
ssh r1lite@192.168.1.128
source ~/galaxea/install/setup.bash
ros2 launch mobiman r1_lite_right_arm_relaxed_ik_launch.py
```

### 2️⃣ 相机启动（2个终端）

**选项A: 相机在PC上**

```bash
# 终端3: 腕部相机
IMAGE_WIDTH=1280 IMAGE_HEIGHT=720 FRAMERATE=30.0 \
/home/robot/Desktop/wangyinxi/WristInsta360_Project/scripts/wrist_cameras_start_all.sh

# 终端4: 头顶相机
/home/robot/Desktop/wangyinxi/WristInsta360_Project/scripts/realsense_head_launch.sh
```

**选项B: 相机在Bot上**

```bash
# 终端3: 腕部相机（Bot端）
ssh r1lite@192.168.1.128
IMAGE_WIDTH=1280 IMAGE_HEIGHT=720 FRAMERATE=30.0 \
/home/r1lite/Desktop/WristInsta360_Project/scripts/wrist_cameras_start_all.sh

# 终端4: 头顶相机（Bot端）
ssh r1lite@192.168.1.128
/home/r1lite/Desktop/WristInsta360_Project/scripts/realsense_head_launch.sh
```

### 3️⃣ PC端 - 模型和推理（2个终端）

**终端5: 模型服务器**
```bash
cd /home/robot/guchenyang/Code/StarVLA-Dev/examples/R1LITE/eval_files
bash run_policy_server.sh
# 等待显示: [INFO] Server is ready to accept connections
```

**终端6: 运行推理**
```bash
cd /home/robot/guchenyang/Code/StarVLA-Dev/examples/R1LITE/eval_files
conda activate starVLA

# 基础用法
python starvla_r1lite_inference.py --task "拿起红色方块"

# 完整参数
python starvla_r1lite_inference.py \
    --config deploy_policy.yml \
    --task "Pick up the red cube" \
    --max-steps 300 \
    --freq 10.0 \
    --log-dir logs/my_task
```

## ✅ 快速验证

### 检查系统状态（2分钟）

```bash
# 1. 检查ROS2话题
ros2 topic list | grep -E "camera|motion"

# 2. 检查相机频率（3个相机都应该~30Hz）
ros2 topic hz /hdas/camera_head/head/color/image_raw
ros2 topic hz /hdas/camera_wrist_left/color/image_raw
ros2 topic hz /hdas/camera_wrist_right/color/image_raw

# 3. 检查模型服务器
netstat -an | grep 5694

# 4. 测试接口（可选）
cd /home/robot/guchenyang/Code/StarVLA-Dev/examples/R1LITE/eval_files
python tests/test_robot_interface.py --duration 5.0
```

### 查看相机画面（可选）

```bash
# 在3个新终端中分别运行：
ros2 run image_view image_view --ros-args -r image:=/hdas/camera_head/head/color/image_raw
ros2 run image_view image_view --ros-args -r image:=/hdas/camera_wrist_left/color/image_raw
ros2 run image_view image_view --ros-args -r image:=/hdas/camera_wrist_right/color/image_raw
```

## 🎯 任务示例

```bash
# 抓取任务
python starvla_r1lite_inference.py --task "拿起红色方块"
python starvla_r1lite_inference.py --task "抓住蓝色杯子"
python starvla_r1lite_inference.py --task "Pick up the banana"

# 放置任务
python starvla_r1lite_inference.py --task "把方块放到盘子上"
python starvla_r1lite_inference.py --task "Place the cup on the table"

# 移动任务
python starvla_r1lite_inference.py --task "移动到目标位置"
python starvla_r1lite_inference.py --task "Move to the target location"
```

## 🛑 停止系统

### 正常停止

```bash
# 1. 停止推理（终端6）: 按 Ctrl+C

# 2. 停止模型服务器（终端5）: 按 Ctrl+C

# 3. 停止相机（终端3, 4）: 按 Ctrl+C

# 4. 停止IK（终端2）: 按 Ctrl+C

# 5. 停止Bot控制（终端1）: 按 Ctrl+C
```

### 强制停止（如需要）

```bash
# 在PC端
pkill -f server_policy.py
pkill -f starvla_r1lite_inference.py

# 在Bot端
ssh r1lite@192.168.1.128
pkill -f mobiman
```

## 🔧 常见问题速查

| 问题 | 快速解决 |
|------|----------|
| 相机无图像 | `ls /dev/video*` 检查设备，重启相机节点 |
| 服务器连接失败 | `netstat -an \| grep 5694` 检查端口，重启服务器 |
| 机器人不动 | `pkill -f teleop` 停止遥操，检查控制节点 |
| GPU内存不足 | `nvidia-smi` 查看显存，关闭其他程序 |
| 图像话题无数据 | `ros2 topic hz <topic>` 检查频率，重启相机 |

## 📊 关键指标参考值

| 指标 | 正常范围 | 说明 |
|------|----------|------|
| 相机频率 | 25-30 Hz | 3个相机都应该稳定 |
| 推理时间 | 30-60 ms | GPU性能影响 |
| 控制频率 | 8-10 Hz | 实际执行频率 |
| 服务器启动时间 | 30-60 s | 模型加载时间 |
| 接口就绪时间 | 5-10 s | ROS2连接时间 |

## 🔗 快速链接

- **完整文档**: [STARVLA_INFERENCE_GUIDE.md](docs/STARVLA_INFERENCE_GUIDE.md)
- **操作手册**: [R1LITE.md](docs/R1LITE.md)
- **API文档**: [README_ROBOT_INTERFACE.md](docs/README_ROBOT_INTERFACE.md)
- **项目结构**: [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)

## 💡 tmux快捷键

```bash
# 创建新窗口: Ctrl+b c
# 切换窗口: Ctrl+b 0-9
# 分割窗格: Ctrl+b % (竖直) 或 Ctrl+b " (水平)
# 切换窗格: Ctrl+b 方向键
# 退出会话: Ctrl+b d
# 列出会话: tmux ls
# 附加会话: tmux attach -t <session_name>
```

## ⚠️ 安全提醒

- ✅ 确保机器人周围2米无人
- ✅ 急停按钮触手可及
- ✅ 有人全程监控
- ✅ 首次运行用小步数测试（--max-steps 50）
- ⛔ 不要在真机上直接测试未验证的代码

---

**保持冷静，安全第一！** 🤖✨

**最后更新**: 2024-02-06

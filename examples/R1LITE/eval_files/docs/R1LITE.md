# R1LITE 机器人操作手册

> **设备说明**：`robot` = PC端，`r1lite` = Bot端（机器人）  
> **重要提示**：遥操（teleop）和推理/回放（replay）不能同时运行

## 📑 目录

- [系统架构](#系统架构)
- [一、系统启动](#一系统启动)
  - [1.1 Bot端启动流程](#11-bot端启动流程)
  - [1.2 相机启动](#12-相机启动)
  - [1.3 相机检查与验证](#13-相机检查与验证)
- [二、数据录制](#二数据录制)
  - [2.1 开始录制](#21-开始录制)
  - [2.2 手柄操作](#22-手柄操作)
- [三、数据回放与验证](#三数据回放与验证)
  - [3.1 回放H5数据](#31-回放h5数据)
  - [3.2 检查数据文件](#32-检查数据文件)
- [四、模型推理](#四模型推理)
  - [4.1 Octo模型推理](#41-octo模型推理)
  - [4.2 ConRFT模型推理](#42-conrft模型推理)
  - [4.3 机械臂归位](#43-机械臂归位)
- [五、工具命令](#五工具命令)
  - [5.1 外接硬盘挂载](#51-外接硬盘挂载)
  - [5.2 数据可视化](#52-数据可视化)
- [六、参考信息](#六参考信息)
- [七、故障排查](#七故障排查)

---

## 系统架构

```
┌─────────────────────────────────────────┐
│              PC (robot)                  │
│  - 数据录制与处理                         │
│  - 模型推理服务                           │
│  - 数据可视化                             │
└───────────┬─────────────────────────────┘
            │ ROS2通信
┌───────────┴─────────────────────────────┐
│         Bot (r1lite@192.168.1.128)      │
│  - 机器人控制系统                         │
│  - 双臂运动控制                           │
│  - 末端执行器控制                         │
└─────────────────────────────────────────┘

相机系统:
- 头顶相机: RealSense
- 左腕相机: GO3S
- 右腕相机: GO3S
```

---

## 一、系统启动

### 1.1 Bot端启动流程

**步骤1：启动Bot全身控制 + 右臂IK**

```bash
# SSH登录到Bot
ssh r1lite@192.168.1.128

# 启动机器人控制系统
cd ~/galaxea/install/startup_config/share/startup_config/script/
./robot_startup.sh boot ../sessions.d/ATCStandard/R1LITEBody.d

# 启动右臂IK（新终端）
source ~/galaxea/install/setup.bash
ros2 launch mobiman r1_lite_right_arm_relaxed_ik_launch.py
```

**步骤2：启动遥操（如需遥操控制）**

> ⚠️ **注意**：遥操与推理/回放互斥，只能启动其中之一

```bash
# 在Bot端执行
source ~/galaxea/install/setup.bash
cd ~/galaxea/install/startup_config/share/startup_config/script/
./robot_startup.sh boot ../sessions.d/ATCHostStandard/R1LITET.d/teleop.yaml
```

---

### 1.2 相机启动

#### 1.2.1 启动腕部相机（GO3S）

**情况A：相机连接在Bot上**

```bash
# 在Bot端新终端执行
ssh r1lite@192.168.1.128

IMAGE_WIDTH=1280 IMAGE_HEIGHT=720 FRAMERATE=30.0 \
/home/r1lite/Desktop/WristInsta360_Project/scripts/wrist_cameras_start_all.sh
```

**情况B：相机连接在PC上**

```bash
# 在PC端新终端执行
IMAGE_WIDTH=1280 IMAGE_HEIGHT=720 FRAMERATE=30.0 \
/home/robot/Desktop/wangyinxi/WristInsta360_Project/scripts/wrist_cameras_start_all.sh
```

#### 1.2.2 启动头顶相机（RealSense）

**相机在Bot上：**

```bash
/home/r1lite/Desktop/WristInsta360_Project/scripts/realsense_head_launch.sh
```

**相机在PC上：**

```bash
/home/robot/Desktop/wangyinxi/WristInsta360_Project/scripts/realsense_head_launch.sh
```

---

### 1.3 相机检查与验证

#### 1.3.1 检查USB视频设备端口

```bash
# 列出所有视频设备及其参数
for i in /dev/video*; do 
    echo "--- $i ---"
    v4l2-ctl --device=$i --all | grep -E "Caps|Width/Height|Payload"
done
```

> 💡 **提示**：如果端口不正确，修改脚本开头的端口配置：  
> `/home/robot/Desktop/wangyinxi/WristInsta360_Project/scripts/wrist_cameras_start_all.sh`

#### 1.3.2 查看相机画面

**左腕相机：**

```bash
ros2 run image_view image_view --ros-args \
    -r image:=/hdas/camera_wrist_left/color/image_raw
```

**右腕相机：**

```bash
ros2 run image_view image_view --ros-args \
    -r image:=/hdas/camera_wrist_right/color/image_raw
```

**头顶相机：**

```bash
ros2 run image_view image_view --ros-args \
    -r image:=/hdas/camera_head/head/color/image_raw
```

---

## 二、数据录制

### 2.1 开始录制

> ⚠️ **重要**：必须使用系统Python（`/usr/bin/python3`），不要使用conda环境

#### 选项A：标准录制（相机在Bot上）

```bash
# SSH到Bot端
ssh r1lite@192.168.1.128

# 启动录制
/usr/bin/python3 /home/r1lite/Desktop/WristInsta360_Project/scripts/data_record_wholebody_go3s.py
```

#### 选项B：高清录制（HD兼容模式，推荐）

```bash
# 在PC端执行
/usr/bin/python3 /home/robot/Desktop/wangyinxi/WristInsta360_Project/scripts/data_record_wholebody_go3s_eef_compatible.py
```

#### 选项C：移动录制（相机在Bot上）

```bash
ssh r1lite@192.168.1.128
/usr/bin/python3 /home/r1lite/Desktop/WristInsta360_Project/scripts/data_record_wholebody_go3s_mobile.py
```

### 2.2 手柄操作

- **X键**：开始录制
- **Y键**：停止录制

> 📝 **注意**：按下Y键后，系统会自动转换并生成H5文件，需要等待片刻

---

## 三、数据回放与验证

### 3.1 回放H5数据

**准备工作（停止遥操）：**

```bash
# 在PC端（robot）
tmux kill-ser

# 在Bot端（r1lite）
pkill -f mobiman_tabletop_tele_node
```

**开始回放：**

```bash
# 在PC端执行（自动播放最新的H5文件）
/usr/bin/python3 /home/robot/Desktop/wangyinxi/WristInsta360_Project/scripts/replay_h5_right_arm_eef_euler.py
```

> 💡 回放时机械臂会按照录制的轨迹运动

---

### 3.2 检查数据文件

#### 检查H5文件内部结构

```bash
/usr/bin/python3 /home/robot/Desktop/WristInsta360_Project/scripts/inspect_h5.py
```

#### 查看数据集索引

```bash
cat ~/GalaxeaDataset/dataset.json
```

---

## 四、模型推理

### 4.1 Octo模型推理

**启动推理：**

```bash
conda run -n octo-gpu bash -lc '
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="/usr/local/cuda-11.8/targets/x86_64-linux/lib:/usr/local/cuda-11.8/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda-11.8
export JAX_PLATFORM_NAME=gpu
python /home/robot/Desktop/wangyinxi/octo_ros_infer_r1lite.py \
    --publish_right \
    --allow_missing_grippers \
    --allow_missing_left_pose
'
```

**查看推理日志：**

```bash
tail -f /tmp/octo_infer.log
```

---

### 4.2 ConRFT模型推理

**启动推理：**

```bash
conda run -n conrft bash -lc '
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:/usr/local/cuda-12.8/targets/x86_64-linux/lib:/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH"
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX
export JAX_PLATFORM_NAME=gpu
export HF_ENDPOINT=https://hf-mirror.com
python /home/robot/Desktop/wangyinxi/conrft_octo_ros_infer.py \
    --allow_missing_grippers \
    --conrft_root /home/robot/conrft \
    --checkpoint_path /home/robot/Desktop/wangyinxi/infer/r1lite_pick_banana/checkpoints/bs256_r1lite_pick_banana_2026-01-28_17-42-33_sft/checkpoint_50000
'
```

**查看推理日志：**

```bash
tail -f /tmp/conrft_octo_infer.log
```

---

### 4.3 机械臂归位

```bash
python /home/robot/Desktop/wangyinxi/octo_ros_home.py --publish_right
```

---

## 五、工具命令

### 5.1 外接硬盘挂载

**挂载硬盘：**

```bash
# SSH到Bot端
ssh r1lite@192.168.1.128

# 查看设备列表
lsblk

# 创建挂载点并挂载
sudo mkdir -p /mnt/usb
sudo mount -o uid=$(id -u),gid=$(id -g) /dev/sdb1 /mnt/usb
```

**卸载硬盘：**

```bash
sudo umount -l /mnt/usb
```

---

### 5.2 数据可视化

**使用Rerun可视化H5数据：**

```bash
# 设置Python路径并运行
PYTHONPATH=/home/robot/.local/lib/python3.10/site-packages/rerun_sdk \
python visualize_h5_rerun.py ./0_20260204_232549.h5
```

---

## 六、参考信息

### 6.1 机械臂初始位姿

可用的初始位置参考值：

```python
# 左臂初始位置
self.l_curr_pos = np.array([0.059, -0.332, 0.343])

# 右臂初始位置（选项1）
self.r_curr_pos = np.array([0.0585, -0.332, 0.343])

# 右臂初始位置（选项2）
self.r_curr_pos = np.array([0.0590, -0.3394, 0.3311])
```

### 6.2 查看实时末端位姿

```bash
ros2 topic echo /motion_control/pose_ee_arm_right
```

### 6.3 关键ROS2话题

| 话题 | 说明 |
|------|------|
| `/hdas/camera_head/head/color/image_raw` | 头顶相机图像 |
| `/hdas/camera_wrist_left/color/image_raw` | 左腕相机图像 |
| `/hdas/camera_wrist_right/color/image_raw` | 右腕相机图像 |
| `/motion_control/pose_ee_arm_left` | 左臂末端位姿 |
| `/motion_control/pose_ee_arm_right` | 右臂末端位姿 |
| `/motion_target/target_pose_arm_left` | 左臂目标位姿 |
| `/motion_target/target_pose_arm_right` | 右臂目标位姿 |

---

## 七、故障排查

### 问题1：相机无法启动

**解决方案：**
1. 检查USB连接是否正常
2. 使用 `lsusb` 查看设备是否被识别
3. 检查视频设备端口是否正确（参考1.3.1）
4. 确认脚本中的设备路径配置正确

### 问题2：ROS2话题无数据

**解决方案：**
```bash
# 检查话题列表
ros2 topic list

# 检查特定话题频率
ros2 topic hz /hdas/camera_head/head/color/image_raw

# 查看话题信息
ros2 topic info /hdas/camera_head/head/color/image_raw
```

### 问题3：录制时无法生成H5文件

**解决方案：**
1. 确认使用系统Python（`/usr/bin/python3`）
2. 检查存储空间是否充足
3. 查看脚本输出的错误信息
4. 确认按Y键后等待足够时间

### 问题4：回放时机械臂不动

**解决方案：**
1. 确认已停止遥操进程（`pkill -f mobiman_tabletop_tele_node`）
2. 检查H5文件是否完整（使用 `inspect_h5.py`）
3. 确认机器人控制系统已正常启动

### 问题5：推理失败或无响应

**解决方案：**
1. 检查模型服务器是否正常运行
2. 查看推理日志文件（`/tmp/octo_infer.log` 或 `/tmp/conrft_octo_infer.log`）
3. 确认CUDA环境变量配置正确
4. 检查模型checkpoint路径是否正确

---

## 📞 联系支持

如遇到其他问题，请联系技术支持团队或提交Issue。

**文档版本**：v2.0  
**最后更新**：2024-02-06
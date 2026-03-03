# R1LITE StarVLA 更新日志

## [v2.0] - 2024-02-06

### 🎉 新增功能

#### 1. StarVLA完整推理脚本
- **文件**: `starvla_r1lite_inference.py`
- **功能**:
  - 完整的推理控制器实现
  - 自动初始化和安全检查
  - 实时任务监控
  - 详细日志记录
  - 紧急停止机制
  - 命令行参数支持
- **使用**: `python starvla_r1lite_inference.py --task "拿起红色方块"`

#### 2. 机器人接口增强
- **文件**: `r1lite_robot_interface.py`
- **新增方法**:
  - `get_head_image()` - 单独获取头顶相机图像
  - `get_left_image()` - 单独获取左腕相机图像
  - `get_right_image()` - 单独获取右腕相机图像
  - `get_left_gripper()` - 获取左臂夹爪状态
  - `get_right_gripper()` - 获取右臂夹爪状态

#### 3. 完整操作文档

##### 主要文档
- **[QUICK_START.md](QUICK_START.md)** - 5分钟快速启动指南
- **[docs/STARVLA_INFERENCE_GUIDE.md](docs/STARVLA_INFERENCE_GUIDE.md)** - StarVLA完整推理指南
- **[docs/R1LITE.md](docs/R1LITE.md)** - 重构的R1LITE操作手册

##### 文档特点
- 清晰的步骤分解
- 完整的故障排查
- 安全操作指南
- 快速参考卡片
- 多级文档导航

### 📁 目录结构优化

重新组织了文件结构：

```
eval_files/
├── 核心代码
│   ├── starvla_r1lite_inference.py  (新增) StarVLA推理主脚本
│   ├── r1lite_robot_interface.py    (更新) 机器人接口
│   ├── r1lite_inference.py          模型推理客户端
│
├── 配置文件
│   ├── deploy_policy.yml            部署配置
│   ├── requirements.txt             依赖列表
│   └── run_policy_server.sh         服务器启动脚本
│
├── 文档
│   ├── README.md                    (更新) 主文档
│   ├── QUICK_START.md               (新增) 快速启动
│   ├── CHANGELOG.md                 (新增) 本文件
│   ├── PROJECT_STRUCTURE.md         (新增) 项目结构
│   └── docs/
│       ├── STARVLA_INFERENCE_GUIDE.md  (新增) 推理完整指南
│       ├── R1LITE.md                    (重构) 操作手册
│       └── README_ROBOT_INTERFACE.md    API文档
│
├── 示例代码 (examples/)
│   ├── example_usage.py             基础推理示例
│   └── example_pose_control.py      姿态控制示例
│
├── 测试代码 (tests/)
│   ├── test_robot_interface.py      接口功能测试
│   └── test.py                      EE姿态监控
│
└── 工具脚本 (tools/)
    ├── data_record_wholebody_go3s.py  数据录制
    └── replay_h5.py                    H5文件回放
```

### 🔧 配置更新

#### deploy_policy.yml
- 已配置checkpoint路径
- unnorm_key设置为"r1lite"
- 优化了DDIM参数

#### run_policy_server.sh
- 更新了Python解释器路径
- 配置了GPU设置
- 添加了bf16支持

### 📊 功能对比

| 功能 | 旧版本 | 新版本 |
|------|--------|--------|
| 图像获取 | 仅`get_images()` | 新增单独获取方法 |
| Gripper状态 | 包含在state中 | 新增独立获取方法 |
| 推理脚本 | 基础示例 | 完整生产级脚本 |
| 安全检查 | ❌ | ✅ 位置边界检查 |
| 错误处理 | 基础 | ✅ 完善的异常处理 |
| 日志记录 | 简单打印 | ✅ 结构化日志 |
| 命令行参数 | ❌ | ✅ 完整参数支持 |
| 任务完成检测 | ❌ | ✅ 多种检测方法 |
| 紧急停止 | Ctrl+C | ✅ 安全停止机制 |
| 文档 | 分散 | ✅ 完整文档体系 |

### 🎯 使用示例

#### 基础使用
```bash
python starvla_r1lite_inference.py --task "拿起红色方块"
```

#### 高级使用
```bash
python starvla_r1lite_inference.py \
    --task "Pick up the red cube and place it on the plate" \
    --config deploy_policy.yml \
    --max-steps 500 \
    --freq 10.0 \
    --log-dir logs/pick_and_place
```

#### 获取单独图像
```python
from r1lite_robot_interface import R1LITERobotInterface

robot = R1LITERobotInterface()
robot.wait_for_ready()

# 新增方法
head_img = robot.get_head_image()
left_img = robot.get_left_image()
right_img = robot.get_right_image()
left_grip = robot.get_left_gripper()
right_grip = robot.get_right_gripper()
```

### 🔐 安全改进

- ✅ 添加位置边界安全检查
- ✅ 实现紧急停止信号处理
- ✅ 添加系统状态验证
- ✅ 详细的安全操作指南
- ✅ 预期输出示例

### 📈 性能优化

- 控制频率: 10 Hz (可调整)
- 平均推理时间: 30-60ms (取决于GPU)
- 图像分辨率: 1280x720 @ 30fps
- 动作维度: 14-dim delta actions

### 🐛 已知问题

1. 相机USB端口需要手动配置
2. 遥操和推理不能同时运行（设计限制）
3. 首次启动模型服务器较慢（~60秒）

### 📝 待办事项

- [ ] 添加视觉标注工具
- [ ] 实现数据自动保存
- [ ] 支持多任务批处理
- [ ] 添加性能分析工具
- [ ] 实现自动相机标定

---

## [v1.0] - 2024-01-XX

### 初始版本

- 基础机器人接口
- 简单推理示例
- 基本文档

---

## 升级指南

### 从v1.0升级到v2.0

1. **备份现有代码**
   ```bash
   cp -r eval_files eval_files_backup
   ```

2. **拉取最新代码**
   ```bash
   git pull origin main
   ```

3. **更新配置文件**
   - 检查`deploy_policy.yml`中的checkpoint路径
   - 更新`run_policy_server.sh`中的Python路径

4. **测试新功能**
   ```bash
   # 测试机器人接口
   python tests/test_robot_interface.py --duration 5.0
   
   # 测试推理脚本
   python starvla_r1lite_inference.py --task "test" --max-steps 10
   ```

5. **阅读新文档**
   - [QUICK_START.md](QUICK_START.md) - 快速上手
   - [STARVLA_INFERENCE_GUIDE.md](docs/STARVLA_INFERENCE_GUIDE.md) - 完整指南

---

## 贡献者

- StarVLA Team
- R1LITE Development Team

## 许可证

参见项目根目录的LICENSE文件

---

**最后更新**: 2024-02-06

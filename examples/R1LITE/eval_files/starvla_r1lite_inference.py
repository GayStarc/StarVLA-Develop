#!/usr/bin/env python3
"""
StarVLA R1LITE 推理脚本

完整的StarVLA模型在R1LITE双臂机器人上的推理实现。

功能特性：
- 机器人初始化和安全检查
- 实时图像和状态获取
- StarVLA模型推理
- Delta动作执行
- 手动步进控制（可选）
- 紧急停止机制
- 详细日志记录

使用方法：
    # 自动模式
    python starvla_r1lite_inference.py --task "拿起红色方块"
    
    # 手动步进模式
    python starvla_r1lite_inference.py --task "拿起红色方块" --manual-step

"""

import numpy as np
import yaml
import rclpy
import time
import argparse
import signal
import sys
from pathlib import Path
from typing import Optional, Tuple
from datetime import datetime

from r1lite_inference import R1LITEModelClient, create_model_client
from r1lite_robot_interface import R1LITERobotInterface


class StarVLAR1LITEInference:
    """StarVLA在R1LITE机器人上的推理控制器"""
    
    def __init__(
        self,
        config_path: str,
        task_instruction: str,
        max_steps: int = 300,
        control_freq: float = 10.0,
        enable_safety_check: bool = True,
        log_dir: Optional[str] = None,
        manual_step: bool = False,
    ):
        """
        初始化推理控制器
        
        Args:
            config_path: 配置文件路径（deploy_policy.yml）
            task_instruction: 任务指令描述
            max_steps: 最大推理步数
            control_freq: 控制频率（Hz）
            enable_safety_check: 是否启用安全检查
            log_dir: 日志保存目录
            manual_step: 是否启用手动步进模式
        """
        self.config_path = config_path
        self.task_instruction = task_instruction
        self.max_steps = max_steps
        self.control_freq = control_freq
        self.enable_safety_check = enable_safety_check
        self.manual_step = manual_step
        
        # 创建日志目录
        if log_dir is None:
            log_dir = f"logs/starvla_inference_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化标志
        self.is_running = False
        self.emergency_stop = False
        
        # 机器人初始位置（从ROS2话题获取或使用默认值）
        self.left_arm_init_pos = np.array([0.059, -0.332, 0.343])
        self.right_arm_init_pos = np.array([0.0585, -0.332, 0.343])
        
        # 安全边界（米）
        self.position_limits = {
            'x': (-0.3, 0.5),
            'y': (-0.6, 0.2),
            'z': (0.0, 0.8),
        }
        
        # 统计信息
        self.stats = {
            'total_steps': 0,
            'successful_steps': 0,
            'failed_steps': 0,
            'avg_inference_time': 0.0,
            'start_time': None,
            'end_time': None,
        }
        
        print("=" * 80)
        print("StarVLA R1LITE 推理系统初始化")
        print("=" * 80)
        print(f"配置文件: {self.config_path}")
        print(f"任务指令: {self.task_instruction}")
        print(f"最大步数: {self.max_steps}")
        print(f"控制频率: {self.control_freq} Hz")
        print(f"控制模式: {'手动步进' if self.manual_step else '自动运行'}")
        print(f"安全检查: {'启用' if self.enable_safety_check else '禁用'}")
        print(f"日志目录: {self.log_dir}")
        print("=" * 80)
        
    def initialize(self) -> bool:
        """初始化ROS2和模型"""
        try:
            # 初始化ROS2
            print("\n[INIT] 初始化ROS2...")
            rclpy.init()
            
            # 加载配置文件
            print(f"[INIT] 加载配置文件: {self.config_path}")
            with open(self.config_path, "r") as f:
                self.config = yaml.safe_load(f)
            
            # 创建模型客户端
            print("[INIT] 连接模型服务器...")
            self.model = create_model_client(self.config)
            print(f"[INIT] ✓ 模型服务器连接成功 ({self.config['host']}:{self.config['port']})")
            
            # 创建机器人接口
            print("[INIT] 初始化机器人接口...")
            self.robot = R1LITERobotInterface(
                node_name="starvla_r1lite_inference",
                publish_rate=self.control_freq,
            )
            
            # 等待机器人就绪
            print("[INIT] 等待机器人接口就绪...")
            if not self.robot.wait_for_ready(timeout=15.0):
                print("[INIT] ✗ 错误：机器人接口未就绪！")
                print("[INIT]   请检查：")
                print("[INIT]   1. ROS2话题是否正常发布")
                print("[INIT]   2. 相机节点是否启动")
                print("[INIT]   3. 机器人控制节点是否运行")
                return False
            
            print("[INIT] ✓ 机器人接口就绪")
            
            # 获取初始位置
            print("\n[INIT] 获取机器人初始位置...")
            left_pos, left_euler, left_grip = self.robot.get_left_pose_euler()
            right_pos, right_euler, right_grip = self.robot.get_right_pose_euler()
            
            print(f"[INIT] 左臂位置: {left_pos}")
            print(f"[INIT] 左臂姿态: {left_euler}")
            print(f"[INIT] 左臂夹爪: {left_grip:.2f}")
            print(f"[INIT] 右臂位置: {right_pos}")
            print(f"[INIT] 右臂姿态: {right_euler}")
            print(f"[INIT] 右臂夹爪: {right_grip:.2f}")
            
            # 更新初始位置
            self.left_arm_init_pos = left_pos
            self.right_arm_init_pos = right_pos
            
            # 重置模型
            print(f"\n[INIT] 重置模型任务: '{self.task_instruction}'")
            self.model.reset(self.task_instruction)
            
            # 注册信号处理
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            
            print("\n" + "=" * 80)
            print("✓ 初始化完成！准备开始推理...")
            print("=" * 80)
            
            return True
            
        except Exception as e:
            print(f"\n[INIT] ✗ 初始化失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _signal_handler(self, signum, frame):
        """处理中断信号"""
        print("\n\n[SIGNAL] 收到中断信号，正在安全停止...")
        self.emergency_stop = True
        self.is_running = False
    
    def check_safety(self, state: np.ndarray) -> Tuple[bool, str]:
        """
        安全检查
        
        Args:
            state: 当前机器人状态
            
        Returns:
            (is_safe, message): 是否安全和消息
        """
        if not self.enable_safety_check:
            return True, "安全检查已禁用"
        
        # 检查左臂位置
        left_pos = state[0:3]
        for i, axis in enumerate(['x', 'y', 'z']):
            if not (self.position_limits[axis][0] <= left_pos[i] <= self.position_limits[axis][1]):
                return False, f"左臂{axis}轴超出安全范围: {left_pos[i]:.3f}"
        
        # 检查右臂位置
        right_pos = state[7:10]
        for i, axis in enumerate(['x', 'y', 'z']):
            if not (self.position_limits[axis][0] <= right_pos[i] <= self.position_limits[axis][1]):
                return False, f"右臂{axis}轴超出安全范围: {right_pos[i]:.3f}"
        
        return True, "安全检查通过"
    
    def print_robot_state(self, step: int, state: np.ndarray, action: np.ndarray, inference_time: float):
        """
        打印机器人当前状态和动作信息
        
        Args:
            step: 当前步数
            state: 当前状态
            action: 当前动作
            inference_time: 推理耗时
        """
        print("\n" + "=" * 80)
        print(f"步骤 {step:03d}")
        print("=" * 80)
        
        # 推理信息
        print(f"推理耗时: {inference_time*1000:.1f}ms")
        
        # 左臂状态
        print("\n【左臂状态】")
        print(f"  位置 (xyz):  [{state[0]:+.4f}, {state[1]:+.4f}, {state[2]:+.4f}] 米")
        print(f"  姿态 (rpy):  [{state[3]:+.4f}, {state[4]:+.4f}, {state[5]:+.4f}] 弧度")
        print(f"               [{np.rad2deg(state[3]):+.2f}°, {np.rad2deg(state[4]):+.2f}°, {np.rad2deg(state[5]):+.2f}°]")
        print(f"  夹爪状态:    {state[6]:.2f} {'[闭合]' if state[6] > 0.5 else '[打开]'}")
        
        # 右臂状态
        print("\n【右臂状态】")
        print(f"  位置 (xyz):  [{state[7]:+.4f}, {state[8]:+.4f}, {state[9]:+.4f}] 米")
        print(f"  姿态 (rpy):  [{state[10]:+.4f}, {state[11]:+.4f}, {state[12]:+.4f}] 弧度")
        print(f"               [{np.rad2deg(state[10]):+.2f}°, {np.rad2deg(state[11]):+.2f}°, {np.rad2deg(state[12]):+.2f}°]")
        print(f"  夹爪状态:    {state[13]:.2f} {'[闭合]' if state[13] > 0.5 else '[打开]'}")
        
        # 左臂动作
        print("\n【左臂动作】")
        print(f"  位置增量:    [{action[0]:+.5f}, {action[1]:+.5f}, {action[2]:+.5f}] 米")
        print(f"  姿态增量:    [{action[3]:+.5f}, {action[4]:+.5f}, {action[5]:+.5f}] 弧度")
        print(f"               [{np.rad2deg(action[3]):+.3f}°, {np.rad2deg(action[4]):+.3f}°, {np.rad2deg(action[5]):+.3f}°]")
        print(f"  夹爪目标:    {action[6]:.2f} {'[闭合]' if action[6] > 0.5 else '[打开]'}")
        action_mag_left = np.linalg.norm(action[0:6])
        print(f"  动作幅度:    {action_mag_left:.5f}")
        
        # 右臂动作
        print("\n【右臂动作】")
        print(f"  位置增量:    [{action[7]:+.5f}, {action[8]:+.5f}, {action[9]:+.5f}] 米")
        print(f"  姿态增量:    [{action[10]:+.5f}, {action[11]:+.5f}, {action[12]:+.5f}] 弧度")
        print(f"               [{np.rad2deg(action[10]):+.3f}°, {np.rad2deg(action[11]):+.3f}°, {np.rad2deg(action[12]):+.3f}°]")
        print(f"  夹爪目标:    {action[13]:.2f} {'[闭合]' if action[13] > 0.5 else '[打开]'}")
        action_mag_right = np.linalg.norm(action[7:13])
        print(f"  动作幅度:    {action_mag_right:.5f}")
        
        print("=" * 80)
    
    def wait_for_user_input(self) -> str:
        """
        等待用户输入（手动步进模式）
        
        Returns:
            用户命令：'n'(下一步), 'c'(继续), 'q'(退出), 's'(跳过N步)
        """
        print("\n【控制选项】")
        print("  [Enter/n] 执行下一步")
        print("  [c]       切换到连续模式")
        print("  [s N]     跳过N步（例如: s 10）")
        print("  [q]       退出推理")
        
        while True:
            try:
                user_input = input("\n请输入命令: ").strip().lower()
                
                if user_input == '' or user_input == 'n':
                    return 'next'
                elif user_input == 'c':
                    return 'continue'
                elif user_input == 'q':
                    return 'quit'
                elif user_input.startswith('s '):
                    try:
                        skip_steps = int(user_input.split()[1])
                        if skip_steps > 0:
                            return f'skip:{skip_steps}'
                        else:
                            print("⚠️  跳过步数必须大于0")
                    except (IndexError, ValueError):
                        print("⚠️  格式错误，请使用: s N (例如: s 10)")
                else:
                    print("⚠️  无效命令，请重新输入")
            except KeyboardInterrupt:
                print("\n")
                return 'quit'
    
    def run_inference(self) -> bool:
        """运行推理循环"""
        if not self.is_running:
            print("\n[ERROR] 系统未初始化，请先调用initialize()")
            return False
        
        print("\n" + "=" * 80)
        print(f"开始推理：{self.task_instruction}")
        print("=" * 80)
        if self.manual_step:
            print("【手动步进模式】- 每步需要确认后执行")
        else:
            print("【自动运行模式】- 提示：按 Ctrl+C 可以随时安全停止")
        print()
        
        self.stats['start_time'] = time.time()
        inference_times = []
        continuous_mode = not self.manual_step  # 连续模式标志
        skip_steps = 0  # 跳过步数计数
        
        try:
            for step in range(self.max_steps):
                if self.emergency_stop or not self.is_running:
                    print("\n[STOP] 推理已停止")
                    break
                
                loop_start = time.time()
                
                # 处理ROS2回调
                rclpy.spin_once(self.robot, timeout_sec=0.001)
                
                # 获取观测
                head_img = self.robot.get_head_image()
                left_img = self.robot.get_left_image()
                right_img = self.robot.get_right_image()
                state = self.robot.get_robot_state()
                
                # 检查图像是否就绪
                if head_img is None or left_img is None or right_img is None:
                    print(f"[Step {step:03d}] 等待图像数据...")
                    time.sleep(0.1)
                    continue
                
                images = [head_img, left_img, right_img]
                
                # 安全检查
                is_safe, safety_msg = self.check_safety(state)
                if not is_safe:
                    print(f"\n[SAFETY] ✗ 安全检查失败: {safety_msg}")
                    print("[SAFETY] 停止推理以确保安全")
                    self.emergency_stop = True
                    break
                
                # 模型推理
                inference_start = time.time()
                action = self.model.step(images, state, self.task_instruction)
                inference_time = time.time() - inference_start
                inference_times.append(inference_time)
                
                # 打印详细状态和动作信息
                self.print_robot_state(step, state, action, inference_time)
                
                # 手动步进模式：等待用户确认
                if not continuous_mode:
                    if skip_steps > 0:
                        skip_steps -= 1
                        print(f"\n⏩ 自动执行中... (剩余 {skip_steps} 步)")
                    else:
                        user_command = self.wait_for_user_input()
                        
                        if user_command == 'quit':
                            print("\n[USER] 用户请求退出")
                            break
                        elif user_command == 'continue':
                            print("\n[MODE] 切换到连续模式")
                            continuous_mode = True
                        elif user_command.startswith('skip:'):
                            skip_steps = int(user_command.split(':')[1]) - 1
                            print(f"\n[MODE] 将连续执行 {skip_steps + 1} 步")
                        # 'next' 命令直接继续执行
                
                # 发送动作到机器人
                self.robot.send_delta_action(action)
                
                # 更新统计
                self.stats['total_steps'] = step + 1
                self.stats['successful_steps'] += 1
                
                # 在连续模式下简化输出
                if continuous_mode and step % 10 != 0:
                    # 只在每10步打印一次简要信息
                    pass
                
                # 达到最大步数
                if step >= self.max_steps - 1:
                    print(f"\n[INFO] 达到最大步数 {self.max_steps}")
                    break
                
                # 控制循环频率（仅在连续模式下）
                if continuous_mode:
                    loop_time = time.time() - loop_start
                    sleep_time = max(0, (1.0 / self.control_freq) - loop_time)
                    if sleep_time > 0:
                        time.sleep(sleep_time)
            
            self.stats['end_time'] = time.time()
            if inference_times:
                self.stats['avg_inference_time'] = np.mean(inference_times)
            
            return True
            
        except Exception as e:
            print(f"\n[ERROR] 推理过程出错: {e}")
            import traceback
            traceback.print_exc()
            self.stats['failed_steps'] += 1
            return False
    
    def print_statistics(self):
        """打印统计信息"""
        if self.stats['start_time'] is None:
            print("\n[STATS] 无统计信息（未运行推理）")
            return
        
        total_time = self.stats['end_time'] - self.stats['start_time']
        
        print("\n" + "=" * 80)
        print("推理统计信息")
        print("=" * 80)
        print(f"任务指令: {self.task_instruction}")
        print(f"总步数: {self.stats['total_steps']}")
        print(f"成功步数: {self.stats['successful_steps']}")
        print(f"失败步数: {self.stats['failed_steps']}")
        print(f"总耗时: {total_time:.2f}秒")
        print(f"平均推理时间: {self.stats['avg_inference_time']*1000:.1f}ms")
        print(f"实际控制频率: {self.stats['total_steps']/total_time:.1f} Hz")
        print("=" * 80)
        
        # 保存统计信息到文件
        stats_file = self.log_dir / "statistics.txt"
        with open(stats_file, 'w') as f:
            f.write(f"任务指令: {self.task_instruction}\n")
            f.write(f"总步数: {self.stats['total_steps']}\n")
            f.write(f"成功步数: {self.stats['successful_steps']}\n")
            f.write(f"失败步数: {self.stats['failed_steps']}\n")
            f.write(f"总耗时: {total_time:.2f}秒\n")
            f.write(f"平均推理时间: {self.stats['avg_inference_time']*1000:.1f}ms\n")
            f.write(f"实际控制频率: {self.stats['total_steps']/total_time:.1f} Hz\n")
        
        print(f"\n统计信息已保存到: {stats_file}")
    
    def cleanup(self):
        """清理资源"""
        print("\n[CLEANUP] 清理资源...")
        
        try:
            if hasattr(self, 'robot'):
                self.robot.destroy_node()
                print("[CLEANUP] ✓ 机器人接口已关闭")
        except:
            pass
        
        try:
            rclpy.shutdown()
            print("[CLEANUP] ✓ ROS2已关闭")
        except:
            pass
        
        print("[CLEANUP] 清理完成")
    
    def run(self) -> bool:
        """完整运行流程"""
        try:
            # 初始化
            if not self.initialize():
                print("\n[ERROR] 初始化失败，退出")
                return False
            
            self.is_running = True
            
            # 运行推理
            success = self.run_inference()
            
            # 打印统计信息
            self.print_statistics()
            
            return success
            
        finally:
            self.cleanup()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="StarVLA R1LITE 推理脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 基本使用
  python starvla_r1lite_inference.py --task "拿起红色方块"
  
  # 指定配置文件和最大步数
  python starvla_r1lite_inference.py --config deploy_policy.yml --task "Pick up the red cube" --max-steps 500
  
  # 禁用安全检查（谨慎使用）
  python starvla_r1lite_inference.py --task "放下方块" --no-safety-check
  
  # 指定日志目录
  python starvla_r1lite_inference.py --task "抓取香蕉" --log-dir logs/banana_task
        """
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="deploy_policy.yml",
        help="配置文件路径（默认: deploy_policy.yml）"
    )
    
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="任务指令描述（中文或英文）"
    )
    
    parser.add_argument(
        "--max-steps",
        type=int,
        default=300,
        help="最大推理步数（默认: 300）"
    )
    
    parser.add_argument(
        "--freq",
        type=float,
        default=10.0,
        help="控制频率 Hz（默认: 10.0）"
    )
    
    parser.add_argument(
        "--no-safety-check",
        action="store_true",
        help="禁用安全检查（不推荐）"
    )
    
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="日志保存目录（默认: logs/starvla_inference_YYYYMMDD_HHMMSS）"
    )
    
    parser.add_argument(
        "--manual-step",
        action="store_true",
        help="启用手动步进模式（每步需要确认）"
    )
    
    args = parser.parse_args()
    
    # 创建推理控制器
    controller = StarVLAR1LITEInference(
        config_path=args.config,
        task_instruction=args.task,
        max_steps=args.max_steps,
        control_freq=args.freq,
        enable_safety_check=not args.no_safety_check,
        log_dir=args.log_dir,
        manual_step=args.manual_step,
    )
    
    # 运行推理
    success = controller.run()
    
    # 返回退出码
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

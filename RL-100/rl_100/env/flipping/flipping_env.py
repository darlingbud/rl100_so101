import os
from selectors import SelectorKey
import sys

# 添加项目根目录到 Python 路径
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_current_dir, '..', '..', '..', '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# 添加 flipping 目录到 Python 路径（gello 模块需要）
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

import time
import enum
from math import pi
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
from multiprocessing.managers import SharedMemoryManager

import io
from contextlib import redirect_stdout

from typing import Any, NamedTuple
from dm_env import StepType, specs
from collections import OrderedDict
from gym import spaces
from mjrl.utils.gym_env import GymEnv
import threading
from collections import deque
import queue
import termios
import tty
import select
from copy import deepcopy

from realsense import RealSense
from gello.zmq_core.robot_node import ZMQClientRobot
from gello.env import RobotEnv

def quat_to_rpy(quat):
    """
    将四元数转换为RPY (Roll-Pitch-Yaw)
    这是rpy_to_quat的逆操作
    
    Args:
        quat: 四元数 [x, y, z, w]
    
    Returns:
        rpy: [roll, pitch, yaw] 弧度制
    """
    rot = R.from_quat(quat)
    rpy = rot.as_euler('xyz')
    return rpy

class ExtendedTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    action: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        return getattr(self, attr)
    
is_done = False
is_success = False
MAX_EPISODE_LEN = 2000
NUM_POINTS = 1024
lambda_penalty = 0.05
smooth_panelty = 0.01 #0.001
stdin_thread = None
def stdin_listener():
    """
    在后台线程中监听标准输入，适用于 SSH 环境
    输入数字键（0-9）表示成功，输入字母（a-z）表示失败
    """
    global is_done, is_success
    # 设置 stdin 为非阻塞模式
    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        while True:
            if select.select([sys.stdin], [], [], 0.1)[0]:
                char = sys.stdin.read(1)
                if char:
                    # 数字键（0-9）表示成功
                    if char >= '0' and char <= '9':
                        is_done = True
                        is_success = True
                        print(f"\n[键盘输入] 检测到数字键 '{char}' -> 标记为成功")
                    # 字母键（a-z）表示失败
                    elif char >= 'a' and char <= 'z':
                        is_done = True
                        is_success = False
                        print(f"\n[键盘输入] 检测到字母键 '{char}' -> 标记为失败")
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

def start_stdin_listener():
    """Start the SSH keyboard listener once, when the real flipping env is used."""
    global stdin_thread
    if stdin_thread is not None and stdin_thread.is_alive():
        return
    stdin_thread = threading.Thread(target=stdin_listener, daemon=True)
    stdin_thread.start()
    print("[INFO] 键盘监听已启动（SSH 模式）: 输入数字键(0-9)表示成功，输入字母(a-z)表示失败")

class FlippingEnv:
    def __init__(
        self, 
        dt,
        num_point_cloud=1024,
        max_steps=3000,
        reset_mode='default',  # 'default' 或 'demo'
        demo_path='/home/guoping/project/bimanual-ur/gello_software/demo_data/demo_001.npy',  # demo 文件路径，reset_mode='demo' 时需要
    ):
        start_stdin_listener()
        # self.xarm = XArmWrapper(joints_init=[3.3, -14.1, -99.8, 1.3, 113.5, 4.3])
        # self.xarm_gripper = RobotiqWrapper(robot='xarm')
        # self.franka = FrankaWrapper(joints_init=(0.0845, -0.5603, -0.1064, -2.0000, -0.0475, 1.4359, 0.0))
        # self.franka_gripper = RobotiqWrapper(robot='franka')

        # Reset 模式配置
        self.reset_mode = reset_mode
        self.demo_path = demo_path
        self.demo_data = None
        self.max_steps = max_steps
        if reset_mode == 'demo':
            if demo_path is None:
                raise ValueError("reset_mode='demo' 时必须提供 demo_path")
            if not os.path.exists(demo_path):
                raise FileNotFoundError(f"Demo 文件不存在: {demo_path}")
            self.demo_data = np.load(demo_path, allow_pickle=True)
            print(f"[INFO] 已加载 demo 数据: {demo_path}, 共 {len(self.demo_data)} 帧")

        robot_client = ZMQClientRobot(port=6001, host='127.0.0.1')
        print(f"[INFO] 机器人客户端连接成功, DOFs: {robot_client.num_dofs()}")
        self.robot = RobotEnv(robot_client)

        self.dt = dt
        self.camera = RealSense(num_points=num_point_cloud)
        self.camera.start()
        self.camera_thread = threading.Thread(target=self._get_camera_frame)
        self.camera_thread.start()
        self.camera_queue = queue.Queue(maxsize=1)
        self.use_arm = False
        number_channel = 3
        obs_sensor_dim = 16
        act_dim = 14
        use_point_cloud = True
        self.num_point_cloud = num_point_cloud
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(act_dim,),
            dtype=np.float64
        )
        self.observation_space = spaces.Dict({
            'image': spaces.Box(
                low=0,
                high=1,
                shape=(number_channel, 84, 84),
                dtype=np.float32
            ),
            'depth': spaces.Box(
                low=0,
                high=1,
                shape=(84, 84),
                dtype=np.float32
            ),
            'depth_scale': spaces.Box(
                low=0,
                high=1,
                shape=(1,),
                dtype=np.float32
            ),
            'agent_pos': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(obs_sensor_dim,),
                dtype=np.float32
            ),
            'ee_pose': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(obs_sensor_dim,),
                dtype=np.float32
            ),
        })

        if use_point_cloud:
            self.observation_space['point_cloud'] = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(512, 3),
                dtype=np.float32
            )
        print("FlippingEnv Init Done")

    def _get_camera_frame(self):
        while True:
            frame = self.camera.get_frame(require_pc=True)
            if self.camera_queue.full():
                self.camera_queue.get()
            self.camera_queue.put(frame)

    def get_frame(self):
        while True:
            if self.camera_queue.full():
                frame = self.camera_queue.get()
                assert not self.camera_queue.full()
                return frame
            time.sleep(1 / 300)


    def get_default_reset_joints(self):
        """获取默认的 reset 关节位置"""
        reset_joints_left = np.deg2rad([82, -98, 103, -117, 0, 22.5, 0])   # 左臂 6关节 + 1夹爪
        reset_joints_right = np.deg2rad([-90, -90, -90, -90, 0, 0, 0])     # 右臂 6关节 + 1夹爪
        return np.concatenate([reset_joints_left, reset_joints_right])
    
    def get_demo_initial_qpos(self):
        """获取 demo 的初始关节位置"""
        if self.demo_data is None:
            raise ValueError("Demo 数据未加载，请确保 reset_mode='demo' 且提供了有效的 demo_path")
        return self.demo_data[0]['qpos'].copy()

    def robot_reset(self, max_delta_threshold=1.5, require_confirm=False):
        """
        重置机器人到初始位置
        
        Args:
            max_delta_threshold: 最大允许的关节差距阈值（弧度），超过此值会警告
            require_confirm: 是否在差距过大时要求用户确认
        
        Returns:
            init_robo_obs: 初始观测
        """
        # 根据模式获取目标关节位置
        if self.reset_mode == 'demo':
            reset_joints = self.get_demo_initial_qpos()
            print(f"[INFO] 使用 demo 初始位置进行 reset")
        else:
            reset_joints = self.get_default_reset_joints()
            print(f"[INFO] 使用默认位置进行 reset")
        
        # 获取当前状态
        obs = self.robot.get_obs()
        curr_joints = obs["joint_positions"]
        
        # 检查位置差距
        abs_deltas = np.abs(reset_joints - curr_joints)
        max_delta = abs_deltas.max()
        
        print(f"[INFO] 当前位置与目标位置最大差距: {max_delta:.4f} rad ({np.degrees(max_delta):.2f} deg)")
        
        if max_delta > max_delta_threshold:
            print("[WARNING] 位置差距过大！")
            for i, (delta, target, current) in enumerate(zip(abs_deltas, reset_joints, curr_joints)):
                if delta > 0.5:  # 只打印差距大于 0.5 rad (~28.6度) 的关节
                    print(f"  关节[{i}]: delta={np.degrees(delta):.1f}°, target={np.degrees(target):.1f}°, current={np.degrees(current):.1f}°")
            
            if require_confirm:
                confirm = input("是否继续移动? (y/n): ")
                if confirm.lower() != 'y':
                    print("[INFO] 用户取消")
                    return None
        
        # 计算移动步数
        steps = min(int(max_delta / 0.01), 100)
        steps = max(steps, 10)  # 至少 10 步
        
        print(f"[INFO] 从当前位置移动到 reset 位置 (步数: {steps})...")
        for jnt in np.linspace(curr_joints, reset_joints, steps):
            self.robot.step(jnt)
        print("[INFO] 已到达 Reset 位置")

        init_robo_obs = self.robot.get_obs()

        return init_robo_obs
    

    def robot_step(self, action):
        action_deg = action.copy()
        
        # ========== 安全保护：检查 action 与当前位置的差距 ==========
        # 获取当前末端位置
        current_obs = self.robot.get_obs()
        current_ee_pose = current_obs["ee_pos_rot"]  # [x, y, z, rx, ry, rz] 弧度制欧拉角
        
        # 安全阈值设置
        POS_THRESHOLD = 0.075  # 位置阈值: 5cm
        ROT_THRESHOLD_RAD = np.deg2rad(30.0)  # 旋转阈值: 15度
        
        # 计算位置差 (x, y, z)
        pos_diff = np.abs(action_deg[:3] - current_ee_pose[:3])
        max_pos_diff = np.max(pos_diff)
        
        # 计算旋转差 - 使用相对旋转的角度大小，避免欧拉角直接相减的问题
        rot_current = R.from_euler('xyz', current_ee_pose[3:6])
        rot_target = R.from_euler('xyz', action_deg[3:6])
        rot_relative = rot_current.inv() * rot_target  # 相对旋转
        rot_diff_angle = rot_relative.magnitude()  # 相对旋转的角度大小（弧度）
        # import ipdb;ipdb.set_trace()
        # 检查是否超过安全阈值
        if max_pos_diff > POS_THRESHOLD or rot_diff_angle > ROT_THRESHOLD_RAD:
            print("\n" + "="*60)
            print("[SAFETY ERROR] 检测到异常动作！模型输出与当前位置差距过大！")
            print("="*60)
            print(f"当前末端位置 (xyz, euler): {current_ee_pose[:6]}")
            print(f"目标动作位置 (xyz, euler): {action_deg[:6]}")
            print(f"位置差 (x, y, z): {pos_diff} (最大: {max_pos_diff:.4f}m, 阈值: {POS_THRESHOLD}m)")
            print(f"旋转差: {np.rad2deg(rot_diff_angle):.2f}° (阈值: 15.0°)")
            if max_pos_diff > POS_THRESHOLD:
                print(f"[!] 位置差超出阈值: {max_pos_diff:.4f}m > {POS_THRESHOLD}m")
            if rot_diff_angle > ROT_THRESHOLD_RAD:
                print(f"[!] 旋转差超出阈值: {np.rad2deg(rot_diff_angle):.2f}° > 15.0°")
            print("="*60)
            print("[SAFETY] 程序紧急退出以保护机器人安全！")
            print("="*60 + "\n")
            sys.exit(1)
        # ========== 安全保护结束 ==========

        self.robot.step_cartesian(action_deg)
        robot_obs = self.robot.get_obs()
        q_pos = robot_obs["joint_positions"]
        ee_pose = robot_obs["ee_pos_rot"]
        return q_pos, ee_pose
        

    

    def reset(self):
        global is_done, is_success
        is_done = False
        is_success = False
        print("<RESET>")
        self.env_step = 0
        self.done = False

        print("========== 移动到 Reset 位置 ==========")
        
        init_robo_obs = self.robot_reset()
        q_pos = init_robo_obs["joint_positions"]
        ee_pose = init_robo_obs["ee_pos_rot"]

        time.sleep(5)

        frame = self.get_frame()
        # xarm_pose = self.xarm.get_position()
        # # print('xarm_pose', xarm_pose)
        # franka_pose = self.franka.get_tcp_pose()
        # # print('franka_pose', franka_pose)
        # xarm_q = self.xarm.get_joint()
        # xarm_gripper_state = self.xarm_gripper.get_state()
        # franka_q = self.franka.get_joint()
        # franka_gripper_state = self.franka_gripper.get_state()
        # q_pos = np.concatenate([xarm_q, [xarm_gripper_state], franka_q, [franka_gripper_state]])
        # ee_pose = np.concatenate([xarm_pose, [xarm_gripper_state], franka_pose, [franka_gripper_state]])
        obs = {
            'agent_pos': np.array(q_pos),
            'point_cloud': np.array(frame['point_cloud']),  # (size, num_points, 3)
            'image': np.random.rand(3, 84, 84),  # set fake image
            'depth': np.array(frame['depth']).astype(np.int32),
            'depth_scale': np.array([frame['depth_scale']]).astype(np.float32),
            'ee_pose': np.array(ee_pose),
        }
        
        self.t_start = time.monotonic()
        return obs.copy()

    def step(self, action):
        # import pdb; pdb.set_trace()
        global is_done, is_success
        # start_time = time.time()5        

        self.env_step += 1

        

        if not self.done:
            q_pos, ee_pose = self.robot_step(action)
        else:
            # 当 done 时，仍需获取当前状态来构建观测
            current_obs = self.robot.get_obs()
            q_pos = current_obs["joint_positions"]
            ee_pose = current_obs["ee_pos_rot"]

        frame = self.get_frame()
        reward = 0
        
        obs = {
            'agent_pos': np.array(q_pos),
            'point_cloud': np.array(frame['point_cloud']),  # (size, num_points, 3)
            'image': np.random.rand(3, 84, 84),  # set fake image
            'depth': np.array(frame['depth']).astype(np.int32),
            'depth_scale': np.array([frame['depth_scale']]).astype(np.float32),
            'ee_pose': np.array(ee_pose),
        }
        return_success = deepcopy(is_success)
        if return_success:
            reward = 1
        else:
            reward = 0
        if reward == 1:
            reward -= lambda_penalty * self.env_step / MAX_EPISODE_LEN
        if self.env_step > 1:
            reward -= smooth_panelty * np.linalg.norm(action - self.pre_action)
        self.pre_action = action
        self.done, timeout = self.terminate(is_done)
        if self.done:
            is_done = False
            is_success = False
        # while time.time() - start_time < self.dt:
        #     time.sleep(self.dt / 20)
        return obs, reward, self.done, {'is_success': return_success, 'timeout': timeout}
    
    def terminate(self, is_done):
        if is_done:
            return True, False
        else:
            if self.env_step >= self.max_steps:
                return True, True
            else:
                return False, False
    
    def close(self):
        """关闭环境并释放资源，特别是相机资源"""
        print("[INFO] 正在关闭 FlippingEnv 并释放相机资源...")
        # 停止相机线程
        self._camera_running = False
        if self.camera_thread.is_alive():
            self.camera_thread.join(timeout=2.0)
            if self.camera_thread.is_alive():
                print("[WARNING] 相机线程未能及时停止")
        
        # 停止相机
        try:
            self.camera.stop()
            print("[INFO] 相机已停止")
        except Exception as e:
            print(f"[WARNING] 停止相机时出错: {e}")
        
        # 清空队列
        while not self.camera_queue.empty():
            try:
                self.camera_queue.get_nowait()
            except queue.Empty:
                break
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--reset_mode', type=str, default='default', choices=['default', 'demo'],
                        help='Reset 模式: default 使用默认关节位置, demo 使用 demo 文件第一帧')
    parser.add_argument('--demo_path', type=str, default=None,
                        help='Demo 文件路径 (reset_mode=demo 时需要)')
    args = parser.parse_args()
    
    # 示例: 使用 demo 模式
    # python flipping_env.py --reset_mode demo --demo_path /home/guoping/project/bimanual-ur/gello_software/demo_data/demo_001.npy
    
    env = FlippingEnv(
        dt=0.1,
        reset_mode=args.reset_mode,
        demo_path=args.demo_path
    )
    env.reset()

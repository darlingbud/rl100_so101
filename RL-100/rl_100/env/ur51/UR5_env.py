import os
import sys
import time
import enum
from math import pi
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
from multiprocessing.managers import SharedMemoryManager
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface

# from rl_100.env.ur5.supervisor import JoystickSupervisor
from rl_100.env.ur5.controller import UR5Controller
from rl_100.env.ur5.command import Command
from rl_100.env.ur5.realsense import RealSense
from rl_100.env.ur5.reward import calc_reward
# from rl_100.env.ur5.reward import calc_reward
# from rl_100.env.ur5.calib_extrinsics import X_root_Tpose
# from rl_100.env.ur5.realworld.ur.policy import Policy
from typing import Any, NamedTuple
from dm_env import StepType, specs
from collections import OrderedDict
from gym import spaces
from mjrl.utils.gym_env import GymEnv
import threading
from collections import deque
import queue


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

class UR5Env:
    INIT_TCP_POSE = np.array([0.22947, -0.07107, 0.02121, 1.211, -2.9, 0.])  # [xyz, rotvec] in UR5 base frame
    INIT_QPOS = np.array([135.18, -129.86, 133.90, -93.89, -88.81, 0]) / 180 * np.pi

    def __init__(
        self, 
        robot_ip='192.169.0.10',
        dt=1/30,
        use_camera=True,
        num_point_cloud=512,
        use_point_cloud=True,
    ):
        print("UR5 Env Init")
        self._rtde_r = RTDEReceiveInterface(robot_ip)

        self.shm_manager = SharedMemoryManager()
        self.shm_manager.start()
        self._controller = UR5Controller(self.shm_manager, robot_ip)
        self._controller.start()

        self.dt = dt
        self.use_camera = use_camera

        self._cur_qpos = None
        self._cur_tcp = None  # (xyz, euler)

        self.camera = RealSense()
        self.camera.start()
        self.camera_thread = threading.Thread(target=self._get_camera_frame)
        self.camera_thread.start()
        self.camera_queue = queue.Queue(maxsize=1)
        self.pre_tag_poses = deque(maxlen=10)

        number_channel = 3
        obs_sensor_dim = 6
        act_dim = 2
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
            'agent_pos': spaces.Box(
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
        print("UR5 Env Init End")

    def _get_camera_frame(self):
        while True:
            frame = self.camera.get_frame(require_pc=True)
            tag_poses = self.camera.detect_apriltag(frame['color'])
            if self.camera_queue.full():
                self.camera_queue.get()
            self.camera_queue.put((frame, tag_poses))

    def get_frame(self):
        while True:
            if self.camera_queue.full():
                frame, tag_poses = self.camera_queue.get()
                assert not self.camera_queue.full()
                return frame, tag_poses
            time.sleep(1 / 300)

    def reset(self):
        print("<RESET>")
        self.env_step = 0
        self.done = False

        self._controller.send_action({
            'cmd': Command.MOVEJ.value,
            'action': self.INIT_QPOS
        })

        self._cur_qpos = self.INIT_QPOS.copy()
        self._cur_tcp = self.INIT_TCP_POSE.copy()

        time.sleep(3)
        frame, _ = self.get_frame()
        # frame = self.camera.get_frame(require_pc=True)
        obs = {
            'agent_pos': np.array(self._rtde_r.getActualQ()),
            'agent_xy': np.array(self._rtde_r.getActualTCPPose()[:2]),
            'point_cloud': np.array(frame['point_cloud']),  # (size, num_points, 3)
            'image': np.random.rand(3, 84, 84),  # set fake image
            'depth': np.random.rand(84, 84)  # set fake depth 
        }
        return obs

    def step(self, action):
        self.env_step += 1

        if not self.done:
            if np.linalg.norm(action) > 1:
                action /= np.linalg.norm(action)

            action_scale = 0.01
            self._cur_tcp[0] += action_scale * action[0]
            self._cur_tcp[1] += action_scale * action[1]
            self._controller.send_action({
                'cmd': Command.SERVOL.value,
                'action': self._cur_tcp
            })

        frame, tag_poses = self.get_frame()
        # frame = self.camera.get_frame(require_pc=True)
        # tag_poses = self.camera.detect_apriltag(frame['color'])
        if self.env_step > 1:
            reward, is_success = calc_reward(tag_poses, prev_tag_poses = self.pre_tag_poses[0], static_penalty=0.1)
        else:
            reward, is_success = calc_reward(tag_poses, static_penalty=0.0)
        self.pre_tag_poses.append(tag_poses)

        obs = {
            'agent_pos': np.array(self._rtde_r.getActualQ()),
            'agent_xy': np.array(self._rtde_r.getActualTCPPose()[:2]),
            'point_cloud': np.array(frame['point_cloud']),  # (size, num_points, 3)
            'image': np.random.rand(3, 84, 84),  # set fake image
            'depth': np.random.rand(84, 84)  # set fake depth 
        }

        self.done = self.terminate(is_success)

        return obs, reward, self.done, {'is_success': is_success}
    
    def terminate(self, is_success):
        if is_success:
            return True
        else:
            if self.env_step >= 1000:
                return True
            else:
                return False


if __name__ == '__main__':
    env = UR5Env(
        robot_ip='192.169.0.10',
        dt=1/30,
        multi_process=True,
        control_mode=UR5Env.ControlMode.TCP,
        delta_action=True
    )

    while True:
        time.sleep(1)

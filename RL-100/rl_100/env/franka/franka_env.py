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

from rl_100.env.franka.franka.franka_wrapper import FrankaWrapper
from rl_100.env.franka.leaphand.leap_node import LeapNode
from rl_100.env.franka.realsense import RealSense

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

class FrankaEnv:
    def __init__(
        self, 
        dt=1 / 20,
        num_point_cloud=512
    ):
        # self.franka = FrankaWrapper()
        self.leap_hand = LeapNode()
        self.leap_hand.set_allegro(np.zeros(16))

        self.dt = dt
        self.camera = RealSense(num_points=num_point_cloud)
        self.camera.start()
        self.camera_thread = threading.Thread(target=self._get_camera_frame)
        self.camera_thread.start()
        self.camera_queue = queue.Queue(maxsize=1)
        self.pre_action = None
        self.use_arm = False
        number_channel = 3
        obs_sensor_dim = 16
        act_dim = 16
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
        self.demo_data = []
        print("Franka Env Init Done")

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

    def reset(self):
        print("<RESET>")
        self.env_step = 0
        self.done = False

        self.leap_hand.set_allegro(np.zeros(16))

        time.sleep(1)
        frame = self.get_frame()
        
        leaphand_cur = self.leap_hand.read_pos()
        # franka_cur = self.franka.get_joint()
        q_pos = leaphand_cur #np.concatenate([franka_cur, leaphand_cur])
        obs = {
            'agent_pos': np.array(q_pos),
            'point_cloud': np.array(frame['point_cloud']),  # (size, num_points, 3)
            'image': np.random.rand(3, 84, 84),  # set fake image
            'depth': np.random.rand(84, 84)  # set fake depth 
        }
        self.demo_data = []
        
        self.t_start = time.monotonic()
        return obs.copy()

    def step(self, action):
        # import pdb; pdb.set_trace()
        # if action.shape[-1] == 23:
        #     self.use_arm = action[22] > 0.5
        #     action = action[:22]
        # else:
        #     self.use_arm = True
        start_time = time.monotonic()
        self.env_step += 1
        t_cycle_end = self.t_start + (self.env_step + 1) * self.dt
        t_command_target = t_cycle_end + self.dt

        if not self.done:
            # if action[2] < 0.2:
            #     action[2] = 0.2 # limit z axis 
            # self.franka.franka.schedule_waypoint(action[:6], t_command_target - time.monotonic() + time.time())
            # if self.use_arm:
            #     self.franka.franka.servoL(action[:6], self.dt)
            self.leap_hand.set_allegro(action[:])

        frame = self.get_frame()
        reward = 0
        is_success = False
        
        self.pre_action = action

        leaphand_cur = self.leap_hand.read_pos()
        # franka_cur = self.franka.get_joint()
        q_pos = leaphand_cur #np.concatenate([franka_cur, leaphand_cur])
        obs = {
            'agent_pos': np.array(q_pos),
            'point_cloud': np.array(frame['point_cloud']),  # (size, num_points, 3)
            'image': np.random.rand(3, 84, 84),  # set fake image
            'depth': np.random.rand(84, 84)  # set fake depth 
        }

        self.done, timeout = self.terminate(is_success)
        # self.demo_data.append({'tag_poses': tag_poses,})
        # time.sleep(self.dt - time.monotonic() + start_time)
        return obs, reward, self.done, {'is_success': is_success, 'timeout': timeout, 'tag_pos': self.demo_data,}
    
    def terminate(self, is_success):
        if is_success:
            return True, False
        else:
            if self.env_step >= 500:
                return True, True
            else:
                return False, False

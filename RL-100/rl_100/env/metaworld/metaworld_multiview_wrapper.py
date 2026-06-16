import torch
import gym
import numpy as np
import matplotlib.pyplot as plt
import os
import metaworld
import random
import time

# from natsort import natsorted
from termcolor import cprint
from gym import spaces

TASK_BOUDNS = {
    'default': [-0.5, -1.5, -0.795, 1, -0.4, 100],
}

class MetaWorldMultiViewEnv(gym.Env):
    metadata = {"render.modes": ["rgb_array"], "video.frames_per_second": 10}

    def __init__(self, task_name, device="cuda:0", 
                 rgb_size=128,
                 camera_names=['corner2']
                 ):
        super(MetaWorldMultiViewEnv, self).__init__()

        if '-v2' not in task_name:
            task_name = task_name + '-v2-goal-observable'

        self.env = metaworld.envs.ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[task_name]()
        self.env._freeze_rand_vec = False

        # https://arxiv.org/abs/2212.05698
        # self.env.sim.model.cam_pos[2] = [0.75, 0.075, 0.7]
        self.env.sim.model.cam_pos[2] = [0.6, 0.295, 0.8]
        

        self.env.sim.model.vis.map.znear = 0.1
        self.env.sim.model.vis.map.zfar = 1.5
        
        self.device_id = int(device.split(":")[-1])
        
        self.image_size = rgb_size  
        
        # 多视角相机设置
        self.camera_names = camera_names
        cprint("[MetaWorldEnv] camera_names: {}".format(self.camera_names), "cyan")
        
        self.episode_length = self._max_episode_steps = 200
        self.action_space = self.env.action_space
        self.obs_sensor_dim = self.get_robot_state().shape[0]

        # 为多视角设置观测空间
        obs_space_dict = {}
        
        # 为每个相机添加独立的观测空间
        for cam_name in self.camera_names:
            obs_space_dict[cam_name] = spaces.Box(
                low=0,
                high=255,
                shape=(3, self.image_size, self.image_size),
                dtype=np.float32
            )
        
        # 添加其他观测
        obs_space_dict['agent_pos'] = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_sensor_dim,),
            dtype=np.float32
        )
        obs_space_dict['full_state'] = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(20, ),
            dtype=np.float32
        )
        
        self.observation_space = spaces.Dict(obs_space_dict)

    def get_robot_state(self):
        eef_pos = self.env.get_endeff_pos()
        finger_right, finger_left = (
            self.env._get_site_pos('rightEndEffector'),
            self.env._get_site_pos('leftEndEffector')
        )
        return np.concatenate([eef_pos, finger_right, finger_left])

    def get_rgb(self, camera_names=None):
        """
        获取多视角RGB图像
        Args:
            camera_names: 相机名称列表，如果为None则使用初始化时设置的相机名称
                         可选值包括：'topview', 'corner', 'corner2', 'corner3', 'behindGripper', 'gripperPOV'
        Returns:
            dict: 相机名称为键，图像为值的字典
        """
        if camera_names is None:
            camera_names = self.camera_names
            
        # cam names: ('topview', 'corner', 'corner2', 'corner3', 'behindGripper', 'gripperPOV')
        imgs_dict = {}
        for cam_name in camera_names:
            img = self.env.sim.render(width=self.image_size, height=self.image_size, camera_name=cam_name, device_id=self.device_id)
            imgs_dict[cam_name] = img
        return imgs_dict

    def render_high_res(self, resolution=1024):
        img = self.env.sim.render(width=resolution, height=resolution, camera_name="corner2", device_id=self.device_id)
        return img
    

    def get_multiview_visual_obs(self):
        """
        获取多视角的视觉观测数据
        Returns:
            dict: 相机名称作为顶级键，包含图像和机器人状态的观测字典
        """
        obs_pixels_dict = self.get_rgb()
        robot_state = self.get_robot_state()
        
        # 构建观测字典，相机名称作为顶级键
        obs_dict = {}
        
        # 为每个视角处理图像格式并添加到观测字典
        for cam_name, img in obs_pixels_dict.items():
            if img.shape[0] != 3:
                img = img.transpose(2, 0, 1)
            obs_dict[cam_name] = img

        # 添加机器人状态
        obs_dict['agent_pos'] = robot_state
        
        return obs_dict

    def get_visual_obs(self):
        """保持向后兼容的单视角观测函数"""
        obs_pixels_dict = self.get_rgb()
        obs_pixels = obs_pixels_dict[self.camera_names[0]]  # 使用第一个相机视角
        robot_state = self.get_robot_state()
        
        if obs_pixels.shape[0] != 3:
            obs_pixels = obs_pixels.transpose(2, 0, 1)

        obs_dict = {
            'image': obs_pixels,
            'agent_pos': robot_state,
        }
        return obs_dict
            
            
    def step(self, action: np.array):

        raw_state, reward, done, env_info = self.env.step(action)
        self.cur_step += 1

        # 使用多视角观测
        obs_dict = self.get_multiview_visual_obs()
        obs_dict['full_state'] = raw_state

        done = done or self.cur_step >= self.episode_length
        
        return obs_dict, reward, done, env_info

    def reset(self):
        self.env.reset()
        self.env.reset_model()
        raw_obs = self.env.reset()
        self.cur_step = 0

        # 使用多视角观测
        obs_dict = self.get_multiview_visual_obs()
        obs_dict['full_state'] = raw_obs

        return obs_dict

    def seed(self, seed=None):
        pass

    def set_seed(self, seed=None):
        pass

    def render(self, mode='rgb_array'):
        img_dict = self.get_rgb()
        img = img_dict[self.camera_names[0]]  # 使用第一个相机视角
        return img

    def close(self):
        pass


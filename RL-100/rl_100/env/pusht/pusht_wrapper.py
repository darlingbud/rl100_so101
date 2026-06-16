import numpy as np
import gym
from gym import spaces
import cv2

from rl_100.env.pusht.pusht_image_env import PushTImageEnv

class PushTWrapper(PushTImageEnv):
    def __init__(self, legacy=False, block_cog=None, damping=None, render_size=96, num_points=1024):
        super().__init__(legacy=legacy, block_cog=block_cog, damping=damping, render_size=render_size)
        self.num_points = num_points

        if isinstance(self.observation_space, spaces.Dict):
            obs_spaces = self.observation_space.spaces.copy()
        else:
            obs_spaces = {"state": self.observation_space}
        obs_spaces["point_cloud"] = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.num_points, 6), dtype=np.float32
        )
        self.observation_space = spaces.Dict(obs_spaces)

    def generate_point_cloud(self, image):
        """
        params：
            image: (H, W, 3)
        return：
            point_cloud: shape (num_points, 6)，
                其中 z 固定为 0。
        """
        H, W, _ = image.shape
        total_pixels = H * W

        ys, xs = np.indices((H, W))
        xs = xs.flatten().astype(np.float32)
        ys = ys.flatten().astype(np.float32)
        colors = image.reshape(-1, 3).astype(np.float32)

        if total_pixels >= self.num_points:
            indices = np.random.choice(total_pixels, size=self.num_points, replace=False)
        else:
            indices = np.random.choice(total_pixels, size=self.num_points, replace=True)

        sampled_x = xs[indices]
        sampled_y = ys[indices]
        sampled_z = np.zeros_like(sampled_x, dtype=np.float32)
        coords = np.stack([sampled_x, sampled_y, sampled_z], axis=-1)
        sampled_colors = colors[indices]

        point_cloud = np.concatenate([coords, sampled_colors], axis=-1)
        return point_cloud

    def _get_obs(self):
        img = self._render_frame('rgb_array')
        agent_pos = np.array(self.agent.position)
        img_obs = np.moveaxis(img.astype(np.float32) / 255, -1, 0)
        obs = {
            'image': img_obs,
            'agent_pos': agent_pos
        }
        if self.latest_action is not None:
            action = np.array(self.latest_action)
            coord = (action / self.window_size * self.render_size).astype(np.int32)
            marker_size = 8
            thickness = 1
            cv2.drawMarker(img, tuple(coord), color=(255, 0, 0),
                           markerType=cv2.MARKER_CROSS,
                           markerSize=marker_size, thickness=thickness)
        self.render_cache = img
        obs["point_cloud"] = self.generate_point_cloud(img)
        return obs

if __name__ == "__main__":
    try:
        env = PushTWrapper(legacy=False, render_size=84, num_points=1024)
    except Exception as e:
        raise e
    
    import matplotlib.pyplot as plt

    obs = env.reset()
    print("Observation keys:", list(obs.keys()))
    print("Image shape (channels-first):", obs["image"].shape)
    print("Agent position:", obs["agent_pos"])
    print("Point cloud shape:", obs["point_cloud"].shape)

    point_cloud = obs["point_cloud"]    

    np.save("pusht_point_cloud.npy", point_cloud)
    
    # 提取 x, y, z 坐标及颜色（RGB）
    x = point_cloud[:, 0]
    y = point_cloud[:, 1]
    z = point_cloud[:, 2] 
    colors = point_cloud[:, 3:6] / 255.0  # 归一化颜色至 [0, 1]
    
    # save 3D figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(x, y, z, c=colors, marker='o')
    plt.savefig("/DATA/disk0/jzn/dp3/ft-dp3/RL-100/rl_100/env/pusht/pusht_point_cloud_reset.png")

    # save 2D image
    img = obs["image"].transpose(1, 2, 0) * 255
    img = img.astype(np.uint8)
    cv2.imwrite("/DATA/disk0/jzn/dp3/ft-dp3/RL-100/rl_100/env/pusht/pusht_image_env_reset.png", img)


    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    print("Step 后 point cloud shape:", obs["point_cloud"].shape)

    point_cloud = obs["point_cloud"]    

    np.save("pusht_point_cloud.npy", point_cloud)
    
    x = point_cloud[:, 0]
    y = point_cloud[:, 1]
    z = point_cloud[:, 2] 
    colors = point_cloud[:, 3:6] / 255.0  # 归一化颜色至 [0, 1]
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(x, y, z, c=colors, marker='o')
    plt.savefig("/DATA/disk0/jzn/dp3/ft-dp3/RL-100/rl_100/env/pusht/pusht_point_cloud.png")

    print(env.latest_action)
    img = obs["image"].transpose(1, 2, 0) * 255
    img = img.astype(np.uint8)
    cv2.imwrite("/DATA/disk0/jzn/dp3/ft-dp3/RL-100/rl_100/env/pusht/pusht_image_env_step.png", img)

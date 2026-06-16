from gym import spaces
from rl_100.env.pusht.pusht_env import PushTEnv
import numpy as np
import cv2

class PushTImageEnv(PushTEnv):
    metadata = {"render.modes": ["rgb_array"], "video.frames_per_second": 10}

    def __init__(self,
            legacy=False,
            block_cog=None, 
            damping=None,
            render_size=96):
        super().__init__(
            legacy=legacy, 
            block_cog=block_cog,
            damping=damping,
            render_size=render_size,
            render_action=False)
        ws = self.window_size
        self.observation_space = spaces.Dict({
            'image': spaces.Box(
                low=0,
                high=1,
                shape=(3,render_size,render_size),
                dtype=np.float32
            ),
            'agent_pos': spaces.Box(
                low=0,
                high=ws,
                shape=(2,),
                dtype=np.float32
            )
        })
        self.render_cache = None
    
    def _get_obs(self):
        img = super()._render_frame(mode='rgb_array')

        agent_pos = np.array(self.agent.position)
        img_obs = np.moveaxis(img.astype(np.float32) / 255, -1, 0)
        obs = {
            'image': img_obs,
            'agent_pos': agent_pos
        }

        # draw action
        if self.latest_action is not None:
            action = np.array(self.latest_action)
            coord = (action / 512 * 96).astype(np.int32)
            marker_size = int(8/96*self.render_size)
            thickness = int(1/96*self.render_size)
            cv2.drawMarker(img, coord,
                color=(255,0,0), markerType=cv2.MARKER_CROSS,
                markerSize=marker_size, thickness=thickness)
        self.render_cache = img

        return obs

    def render(self, mode):
        assert mode == 'rgb_array'

        if self.render_cache is None:
            self._get_obs()
        
        return self.render_cache
    
import cv2
import numpy as np

def main():
    # 创建 PushTImageEnv 实例
    env = PushTImageEnv(legacy=False, render_size=96)
    
    # 重置环境，获取初始观测
    obs = env.reset()
    
    print("初始观测 keys:", list(obs.keys()))
    print("Agent 位置:", obs["agent_pos"])
    print("Image shape (channels-first):", obs["image"].shape)
    
    # 使用 render() 获取 reset 后的 RGB 图像，图像格式为 H x W x 3
    img_reset = obs["image"].transpose(1, 2, 0) * 255
    img_reset = img_reset.astype(np.uint8)
    # 保存 reset 后的图像
    cv2.imwrite("pusht_image_env_reset.png", img_reset)
    
    # 随机采样一个动作
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    
    print("Step 后：")
    print("Reward:", reward)
    print("Done:", done)

    # 使用 render() 获取 reset 后的 RGB 图像，图像格式为 H x W x 3
    img = obs["image"].transpose(1, 2, 0) * 255
    img = img.astype(np.uint8)
    cv2.imwrite("pusht_image_env.png", img)

    

if __name__ == "__main__":
    main()

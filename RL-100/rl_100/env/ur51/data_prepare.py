import time
import viser
import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
import torch    
import zarr
from realsense import depth2pc, camera_intrinsics, X_root_camera, point_cloud_downsample
# import pytorch3d.ops as torch3d_ops
import fpsample
from termcolor import cprint
from pathlib import Path
from reward import calc_reward
import os
from tqdm import tqdm
from utils import interpolate_rewards
from copy import deepcopy

PUSHT_DEMO_DIR = Path(os.environ.get("RL100_PUSHT_DEMO_DIR", "data/pushT"))
PUSHT_DATA_DIR = PUSHT_DEMO_DIR.parent
SHOW_REWARD_CURVE = False
SHOW_POINT_CLOUD = False
LOAD_FROM_DATA = True

def compute_return(reward, not_done, gamma: float == 0.99
    ):
        size_ = len(reward)
        return_ = np.zeros((size_, 1))
        pre_return = 0
        for i in tqdm(reversed(range(size_)), desc='Computing the returns'):
            return_[i] = reward[i] + gamma * pre_return * not_done[i]
            pre_return = return_[i]
        return return_
if __name__ == '__main__':
    if not LOAD_FROM_DATA:
        data = {
            'action': list(),
            'reward': list(),
            'is_success': list(),
            'done': list(),
            'agent_pos': list(),
            'point_cloud_origin': list(),
            'point_cloud': list(),
            'timeout': list()
        }

        folder_path = PUSHT_DEMO_DIR
        file_names = [f.name for f in folder_path.iterdir() if f.is_file()]
        for file_name in file_names:
            print('dmeo id:', file_name)
            if not file_name.startswith('demo'):
                continue 
            demo_path = PUSHT_DEMO_DIR / str(file_name)
            demo = np.load(demo_path, allow_pickle=True)

            for i in range(len(demo)):
                print("Frame:", i)
                frame = demo[i]
                t = time.time()
                point_cloud = depth2pc(
                    frame['depth'] * frame['depth_scale'],
                    camera_intrinsics,
                    X_root_camera
                )
                print(time.time() - t)
                reward, _ = calc_reward(frame['tag_poses'])    
                data['action'].append(frame['action'])
                data['reward'].append(reward)
                data['is_success'].append(frame['is_success']) # TODO: None exists in the data, check the data
                data['done'].append(True if data['is_success'] else False) # TODO: all True, check it next time
                data['timeout'].append(True if i == len(demo) - 1 else False)
                data['agent_pos'].append(frame['qpos'])
                # data['point_cloud_origin'].append(point_cloud)
                t = time.time()
                data['point_cloud'].append(point_cloud_downsample(point_cloud, 512))
                print(time.time() - t)
            
            if SHOW_POINT_CLOUD:
                def on_update(frame_idx):
                    frame = demo[frame_idx]
                    point_cloud = data['point_cloud'][frame_idx]
                    value = point_cloud[:, 1] - point_cloud[:, 0]
                    depth_norm = cv2.normalize(value, None, 0, 255, cv2.NORM_MINMAX)
                    depth_8bit = np.uint8(depth_norm)
                    colormap = np.array(cv2.applyColorMap(depth_8bit, cv2.COLORMAP_JET))[:, 0]
                    # print(colormap.shape)

                    server.scene.add_point_cloud(
                        'point_cloud',
                        point_cloud,
                        point_size=0.001,
                        point_shape="circle",
                        colors=colormap
                    )

                    z = -0.005
                    zz = 0.003
                    server.scene.add_point_cloud(
                        'bounding_point',
                        np.array([[0.75, -0.25, z], [0.37, 0.13, z], [-0.13, -0.37, z], [0.25, -0.75, z],
                                [0.75, -0.25, zz], [0.37, 0.13, zz], [-0.13, -0.37, zz], [0.25, -0.75, zz]]),
                        point_size=0.001,
                        point_shape="circle",
                        colors=(102, 192, 255)
                    )

                    for tag_id, tag_pose in enumerate(frame['tag_poses']):
                        server.scene.add_frame(
                            f'tags/tag_{tag_id}',
                            wxyz=R.from_matrix(tag_pose[:3, :3]).as_quat()[[3, 0, 1, 2]],
                            position=tag_pose[:3, 3],
                            axes_length=0.1,
                            axes_radius=0.003
                        )
                server = viser.ViserServer(host='127.0.0.1', port=8080)

                slider = server.gui.add_slider(
                    label='frame',
                    min=0,
                    max=len(demo) - 1,
                    step=1,
                    initial_value=0
                )
                slider.on_update(lambda _: on_update(slider.value))
                while True:
                    time.sleep(1)
            if SHOW_REWARD_CURVE:
                rewards = data['reward']
                plt.figure(figsize=(10, 5))
                plt.plot(rewards, marker='o', linestyle='-', color='b')  # 使用圆圈标记数据点
                plt.title('Reward Curve')
                plt.xlabel('Episode or Step')
                plt.ylabel('Reward')
                plt.grid()
                plt.show()
        np.save(PUSHT_DEMO_DIR / "data.npy", data)

    else:   
        demo_path = PUSHT_DEMO_DIR / "data.npy"
        demo = np.load(demo_path, allow_pickle=True)
        data = demo.item()
        traj_end_index = np.where(data['timeout'])[0] # get the index of the last step of each episode
        save_dir = PUSHT_DATA_DIR / "push_t.zarr"
        if os.path.exists(save_dir):
            cprint('Data already exists at {}'.format(save_dir), 'red')
            cprint("If you want to overwrite, delete the existing directory first.", "red")
            cprint("Do you want to overwrite? (y/n)", "red")
            # user_input = input()
            user_input = 'y'
            if user_input == 'y':
                cprint('Overwriting {}'.format(save_dir), 'red')
                os.system('rm -rf {}'.format(save_dir))
            else:
                cprint('Exiting', 'red')
                exit()
        os.makedirs(save_dir, exist_ok=True)
        # pre process None type reward
        data['reward'] = np.array(data['reward'])
        data['reward'][data['reward'] == 2] = 20

        # 检查 reward 列中为 None 或 NaN 的位置
        data['reward'] = interpolate_rewards(data['reward'])
        data['reward'] = data['reward'] - 1.
        total_count = 0
        img_arrays = []
        point_cloud_arrays = []

        next_img_arrays = []
        next_point_cloud_arrays = []
        next_state_arrays = []   
        next_action_arrays = []

        state_arrays = []
        action_arrays = []
        next_action_arrays = []
        reward_arrays = []
        done_arrays = []
        timeout_arrays = []
        episode_ends_arrays = []
        
        all_total_rewards = []
        # loop over episodes
        episode_id = 0
        img = np.random.rand(3, 84, 84) # DON NOT USE IMAGE
        total_reward = 0.
        total_count_sub = 0
        for total_id, timeout in enumerate(data['timeout']):
            total_reward += data['reward'][total_id]
            img_arrays.append(img)
            point_cloud_arrays.append(deepcopy(data['point_cloud'][total_id]))
            state_arrays.append(deepcopy(data['agent_pos'][total_id]))
            action_arrays.append(deepcopy(data['action'][total_id]))
            reward_arrays.append(deepcopy(data['reward'][total_id]))
            done_arrays.append(deepcopy(data['done'][total_id]))
            timeout_arrays.append(deepcopy(data['timeout'][total_id]))
            total_count_sub += 1

            next_img_arrays.append(img)
            if total_id == len(data['timeout']) - 1:
                next_point_cloud_arrays.append(deepcopy(data['point_cloud'][total_id]))
                next_state_arrays.append(deepcopy(data['agent_pos'][total_id]))
                next_action_arrays.append(deepcopy(data['action'][total_id]))
            else:
                next_point_cloud_arrays.append(deepcopy(data['point_cloud'][total_id+1]))
                next_state_arrays.append(deepcopy(data['agent_pos'][total_id+1]))
                next_action_arrays.append(deepcopy(data['action'][total_id+1]))
            if timeout:
                episode_id += 1
                total_count += total_count_sub
                print('Episode: {}, Episode length: {}, Return: {}'.format(episode_id, total_count_sub, total_reward))

                episode_ends_arrays.append(deepcopy(total_count))
                all_total_rewards.append(deepcopy(total_reward))
                # reset sub arrays
                total_reward = 0.
                total_count_sub = 0
                #-----------------------
        print('Mean reward: {}'.format(np.mean(all_total_rewards)))
        # import pdb; pdb.set_trace()
        # tracemalloc.stop()
        ###############################
        # save data
        ###############################
        # create zarr file
        zarr_root = zarr.group(save_dir)
        zarr_data = zarr_root.create_group('data')
        zarr_meta = zarr_root.create_group('meta')
        # save img, state, action arrays into data, and episode ends arrays into meta
        img_arrays = np.stack(img_arrays, axis=0)
        next_img_arrays = np.stack(next_img_arrays, axis=0)
        if img_arrays.shape[1] == 3: # make channel last
            img_arrays = np.transpose(img_arrays, (0,2,3,1))
            next_img_arrays = np.transpose(next_img_arrays, (0,2,3,1))
        state_arrays = np.stack(state_arrays, axis=0)
        point_cloud_arrays = np.stack(point_cloud_arrays, axis=0)
        action_arrays = np.stack(action_arrays, axis=0)

        next_state_arrays = np.stack(next_state_arrays, axis=0)
        next_point_cloud_arrays = np.stack(next_point_cloud_arrays, axis=0)
        next_action_arrays = np.stack(next_action_arrays, axis=0)
        
        reward_arrays = np.array(reward_arrays).reshape(action_arrays.shape[0], -1)
        done_arrays = np.array(done_arrays).reshape(action_arrays.shape[0], -1)
        timeout_arrays = np.array(timeout_arrays).reshape(action_arrays.shape[0], -1)
        done_arrays = deepcopy(timeout_arrays) # TODO: issues in done arrays, replace it with timeout arrays temporarily
        episode_ends_arrays = np.array(episode_ends_arrays)
        # print(done_arrays)

        # import pdb
        # pdb.set_trace()

        compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
        img_chunk_size = (100, img_arrays.shape[1], img_arrays.shape[2], img_arrays.shape[3])
        state_chunk_size = (100, state_arrays.shape[1])
        point_cloud_chunk_size = (100, point_cloud_arrays.shape[1], point_cloud_arrays.shape[2])
        action_chunk_size = (100, action_arrays.shape[1])
        reward_chunk_size = (100, reward_arrays.shape[1])

        done_chunk_size = (100, done_arrays.shape[1])
        timeout_chunk_size = (100, timeout_arrays.shape[1])
        # compute return for each episode
        not_done_arrays =  1. - (done_arrays | timeout_arrays)
        done_timeout_arrays = done_arrays | timeout_arrays  
        done_indices = np.where(done_timeout_arrays.flatten())[0]
        return_arrays = compute_return(reward_arrays, not_done_arrays, 0.99)
        return_chunk_size = (100, return_arrays.shape[1])

        
        # import pdb
        # pdb.set_trace()
        zarr_data.create_dataset('img', data=img_arrays, chunks=img_chunk_size, dtype='uint8', overwrite=True, compressor=compressor)
        zarr_data.create_dataset('state', data=state_arrays, chunks=state_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
        zarr_data.create_dataset('point_cloud', data=point_cloud_arrays, chunks=point_cloud_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
        zarr_data.create_dataset('action', data=action_arrays, chunks=action_chunk_size, dtype='float32', overwrite=True, compressor=compressor)

        zarr_data.create_dataset('next_img', data=next_img_arrays, chunks=img_chunk_size, dtype='uint8', overwrite=True, compressor=compressor)
        zarr_data.create_dataset('next_state', data=next_state_arrays, chunks=state_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
        zarr_data.create_dataset('next_point_cloud', data=next_point_cloud_arrays, chunks=point_cloud_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
        zarr_data.create_dataset('next_action', data=next_action_arrays, chunks=action_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
        zarr_data.create_dataset('return', data=return_arrays, chunks=timeout_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
        zarr_data.create_dataset('reward', data=reward_arrays, chunks=reward_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
        zarr_data.create_dataset('done', data=done_arrays, chunks=done_chunk_size, dtype='bool', overwrite=True, compressor=compressor)
        zarr_data.create_dataset('timeout', data=timeout_arrays, chunks=timeout_chunk_size, dtype='bool', overwrite=True, compressor=compressor)

        zarr_meta.create_dataset('episode_ends', data=episode_ends_arrays, dtype='int64', overwrite=True, compressor=compressor)
        
        
        # print shape
        cprint(f'img shape: {img_arrays.shape}, range: [{np.min(img_arrays)}, {np.max(img_arrays)}]', 'green')
        cprint(f'point_cloud shape: {point_cloud_arrays.shape}, range: [{np.min(point_cloud_arrays)}, {np.max(point_cloud_arrays)}]', 'green')
        cprint(f'state shape: {state_arrays.shape}, range: [{np.min(state_arrays)}, {np.max(state_arrays)}]', 'green')
        cprint(f'action shape: {action_arrays.shape}, range: [{np.min(action_arrays)}, {np.max(action_arrays)}]', 'green')

        cprint(f'next_img shape: {next_img_arrays.shape}, range: [{np.min(next_img_arrays)}, {np.max(next_img_arrays)}]', 'green')
        cprint(f'next_point_cloud shape: {next_point_cloud_arrays.shape}, range: [{np.min(next_point_cloud_arrays)}, {np.max(next_point_cloud_arrays)}]', 'green')
        cprint(f'next_state shape: {next_state_arrays.shape}, range: [{np.min(next_state_arrays)}, {np.max(next_state_arrays)}]', 'green')
        cprint(f'next_action shape: {next_action_arrays.shape}, range: [{np.min(next_action_arrays)}, {np.max(next_action_arrays)}]', 'green')
        
        cprint(f'reward shape: {reward_arrays.shape}, range: [{np.min(reward_arrays)}, {np.max(reward_arrays)}]', 'green')
        cprint(f'done shape: {done_arrays.shape}, range: [{np.min(done_arrays)}, {np.max(done_arrays)}]', 'green')
        cprint(f'timeout shape: {timeout_arrays.shape}, range: [{np.min(timeout_arrays)}, {np.max(timeout_arrays)}]', 'green')
        cprint(f'return shape: {return_arrays.shape}, range: [{np.min(return_arrays)}, {np.max(return_arrays)}]', 'green')
        cprint(f'Saved zarr file to {save_dir}', 'green')
        
        cprint(f'Saved zarr file to {save_dir}', 'green')
        
        # clean up
        del img_arrays, state_arrays, point_cloud_arrays, action_arrays, reward_arrays, done_arrays, timeout_arrays, episode_ends_arrays
        del zarr_root, zarr_data, zarr_meta
        # del expert_agent

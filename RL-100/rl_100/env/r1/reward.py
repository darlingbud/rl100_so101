import numpy as np

def calc_reward(current_action=None, prev_action=None, smooth_penalty=2):
    if current_action is not None and prev_action is not None:
        reward = - smooth_penalty * np.linalg.norm(np.array(current_action) - np.array(prev_action))**2
    else:
        reward = 0
    return reward

if __name__ == '__main__':
    import time
    from realsense import RealSense

    camera = RealSense()
    camera.start()

    while True:
        frame = camera.get_frame(require_pc=True)
        tag_poses = camera.detect_apriltag(frame['color'])

        reward, is_success = calc_reward(tag_poses)
        print("########## Reward:", reward)
        # print()
        time.sleep(0.1)

import numpy as np


X_root_Tpose = np.array([
    [[0.7071,  0.7071,  0.    ,  0.3402],
     [0.7071, -0.7071,  0.    , -0.4245],
     [0.    ,  0.    , -1.    ,  0.0150],
     [0.    ,  0.    ,  0.    ,  1.    ]],

    [[0.7071,  0.7071,  0.    ,  0.4244],
     [0.7071, -0.7071,  0.    , -0.3397],
     [0.    ,  0.    , -1.    ,  0.0130],
     [0.    ,  0.    ,  0.    ,  1.    ]],

    [[0.7071,  0.7071,  0.    ,  0.4680],
     [0.7071, -0.7071,  0.    , -0.4666],
     [0.    ,  0.    , -1.    ,  0.0150],
     [0.    ,  0.    ,  0.    ,  1.    ]]
])



def calc_reward(tag_poses, weight=5.0, tag_poses_label=None, prev_tag_poses=None, static_thresh=0.01, static_penalty=0.0):
    tag_poses_label = X_root_Tpose if tag_poses_label is None else tag_poses_label

    se3_dists = []
    for tag_pose, tag_pose_label in zip(tag_poses, tag_poses_label):
        if tag_pose is not None:
            rel_translation = tag_pose[:3, 3] - tag_pose_label[:3, 3]
            rel_rotation = tag_pose[:3, :3].T @ tag_pose_label[:3, :3]
            rel_rotation_trace = np.clip(np.trace(rel_rotation), -1, 3)
            rel_angle = np.arccos((rel_rotation_trace - 1) / 2)
            se3_dist = np.linalg.norm(rel_translation) + 0.2 * rel_angle
            # print(np.linalg.norm(rel_translation), 0.3 * rel_angle)
            # print(f"SE3 Dist:, {se3_dist:.4f}", "Tag Pose:", tag_pose[:3, 3])
            se3_dists.append(se3_dist)
    
    if len(se3_dists) == 0:
        return None, None
    else:
        tag_num_in_Tpose = 0
        for tag_pose in tag_poses:
            if tag_pose is not None and tag_pose[2, 3] <= 0.02:
                tag_num_in_Tpose += 1
        is_success = tag_num_in_Tpose >= 2

        if is_success:
            return 20.0, True
        else:
            # filter abnormal value
            if len(se3_dists) == 3:
                median = np.median(se3_dists)
                deviations = [abs(dist - median) for dist in se3_dists]
                max_deviation = max(deviations)
                if max_deviation > 3 * np.median(deviations):
                    outlier_index = deviations.index(max_deviation)
                    se3_dists.pop(outlier_index)
            # return sum(se3_dists) / len(se3_dists), False
            reward = np.exp(-weight * sum(se3_dists) / len(se3_dists)) - 1.
            if prev_tag_poses is not None and static_penalty > 0:
                static_count = 0
                valid_count = 0
                for curr_pose, prev_pose in zip(tag_poses, prev_tag_poses):
                    if curr_pose is not None and prev_pose is not None:
                        # 计算两个位姿平移部分的距离差
                        diff = np.linalg.norm(curr_pose[:3, 3] - prev_pose[:3, 3])
                        valid_count += 1
                        if diff < static_thresh:
                            static_count += 1
                # 如果大部分 tag 位姿都没有移动，则施加惩罚
                if valid_count > 0 and static_count / valid_count > 0.5:
                    reward -= static_penalty
            return reward, False # set the reward to be negative


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

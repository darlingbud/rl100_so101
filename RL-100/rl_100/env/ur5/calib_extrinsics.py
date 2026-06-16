import cv2
import time
import numpy as np
import viser
from scipy.spatial.transform import Rotation as R
import pyrealsense2 as rs
from dt_apriltags import Detector

from realsense import RealSense, camera_intrinsics

# Tpose central position of 3 tags
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


if __name__ == '__main__':
    camera = RealSense()
    camera.start()

    detector = Detector(families="tagStandard41h12")
    server = viser.ViserServer(host='127.0.0.1', port=8080)

    frame = camera.get_frame()
        
    gray_image = cv2.cvtColor(frame['color'], cv2.COLOR_BGR2GRAY)
    detections = detector.detect(
        gray_image,
        estimate_tag_pose=True,
        camera_params=camera_intrinsics,
        tag_size=0.03 * 5 / 9
    )
    X_tag_camera = {}
    label = False
    for detection in detections:
        tag_transform = np.eye(4)
        tag_transform[:3, :3] = detection.pose_R.T
        tag_transform[:3, 3] = -detection.pose_R.T @ detection.pose_t.flatten()
        X_tag_camera[detection.tag_id] = tag_transform

    camera_pose_mean = np.zeros((4, 4))
    for i in range(len(detections)):
        camera_pose = X_root_Tpose[i] @ X_tag_camera[i]
        print(camera_pose)
        camera_pose_mean += camera_pose
    camera_pose_mean /= len(detections)
    print("camera_pose:")
    print(np.array2string(camera_pose_mean, separator=', '))

    server.scene.add_frame(
        f'camera_pose',
        wxyz=R.from_matrix(camera_pose_mean[:3, :3]).as_quat()[[3, 0, 1, 2]],
        position=camera_pose_mean[:3, 3],
        axes_length=0.2,
        axes_radius=0.005
    )
    server.scene.add_point_cloud(
        'pc',
        frame['point_cloud'],
        point_size=0.002,
        point_shape="circle",
        colors=(102, 192, 255)
    )

    while True:
        time.sleep(1)

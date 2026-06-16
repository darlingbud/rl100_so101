import cv2
import time
import numpy as np
import viser
from scipy.spatial.transform import Rotation as R
import pyrealsense2 as rs
from dt_apriltags import Detector
import fpsample


def depth2pc(depth, camera_intrinsics, camera_pose=np.eye(4)):
    height, width = depth.shape
    fx, fy, cx, cy = camera_intrinsics
    z = depth.flatten()

    u, v = np.meshgrid(np.arange(width), np.arange(height))
    u = u.flatten()
    v = v.flatten()

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    points_camera = np.stack((x, y, z), axis=1)[z > 0]

    points_camera_h = np.concatenate((points_camera, np.ones((points_camera.shape[0], 1))), axis=1)  # Nx4
    points_world = (camera_pose @ points_camera_h.T).T[:, :3]

    return points_world


def point_cloud_downsample(point_cloud, num_points):
    x, y, z = point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2]
    bounding_box_mask = (z > 0.075) & (z < 0.20) & (x > 0.3) & (x < 0.75) & (y > -0.05) & (y < 0.2)
    point_cloud = point_cloud[bounding_box_mask]
    # bounding_box_mask = (z > 0.038) & (z < 0.25) & (x > 0.3) & (x < 0.75) & (y > -0.2) & (y < 0.2)
    # point_cloud = point_cloud[bounding_box_mask]

    # FPS sampling
    if len(point_cloud) < num_points:
        point_cloud = np.concatenate([point_cloud] * (num_points // len(point_cloud) + 1), axis=0)
    sample_idx = fpsample.bucket_fps_kdtree_sampling(point_cloud, num_points)
    point_cloud = point_cloud[sample_idx]
    return point_cloud


# camera_intrinsics = (1390.24044429, 1390.24044429, 951.21882917, 531.35245738)
X_root_camera = np.array([
    [-0.99985015,  0.00860291, -0.015022  ,  0.58439754],
    [ 0.01708381,  0.63049465, -0.77600557,  0.46110644],
    [ 0.00279538, -0.77614592, -0.63054714,  0.413612  ],
    [ 0.        ,  0.        ,  0.        ,  1.        ]
])


class RealSense(object):
    def __init__(
            self,
            fps=30,
            enable_depth=True,
            depth_width=640,
            depth_height=480,
            apriltag_families="tagStandard41h12",
            num_points=1024
        ):
        self.enable_depth = enable_depth
        self.depth_width = depth_width
        self.depth_height = depth_height
        self.num_points = num_points

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        if self.enable_depth:
            self.config.enable_stream(rs.stream.depth, depth_width, depth_height, rs.format.z16, fps)
        self.align = rs.align(rs.stream.color)

        self.detector = Detector(families=apriltag_families)

    def start(self):
        profile = self.pipeline.start(self.config)

        # get intrinsics
        frames = self.pipeline.wait_for_frames()
        if self.enable_depth:
            self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
            depth_frame = frames.get_depth_frame()
            depth_intrinsics = depth_frame.get_profile().as_video_stream_profile().get_intrinsics()
            self.depth_intrinsics = (depth_intrinsics.fx, depth_intrinsics.fy, depth_intrinsics.ppx, depth_intrinsics.ppy)

    def stop(self):
        self.pipeline.stop()

    def get_frame(self, require_pc=False):
        while True:
            frames = self.pipeline.wait_for_frames()

            timestamp = frames.get_timestamp() / 1000  # ms -> s
            depth_frame = frames.get_depth_frame()

            if depth_frame:
                break

        depth_image = np.array(depth_frame.get_data())
        # import pdb; pdb.set_trace()
        point_cloud = point_cloud_downsample(depth2pc(
            depth_image * self.depth_scale,
            self.depth_intrinsics,
            X_root_camera
        ), self.num_points) if require_pc else None

        return {
            'timestamp': timestamp,
            'depth': depth_image,
            'depth_scale': self.depth_scale,
            'point_cloud': point_cloud
        }
    
    def detect_apriltag(self, color, tag_size=0.03 * 5 / 9, tag_num=3):
        detections = self.detector.detect(
            cv2.cvtColor(color, cv2.COLOR_BGR2GRAY),
            estimate_tag_pose=True,
            camera_params=self.intrinsics,
            tag_size=tag_size
        )
        # print(f'{len(detections)} tags detected.')

        tag_poses = [None] * tag_num
        for detection in detections:
            X_camera_tag = np.eye(4)
            X_camera_tag[:3, :3] = detection.pose_R
            X_camera_tag[:3, 3] = detection.pose_t.flatten()
            X_root_tag = X_root_camera @ X_camera_tag
            tag_poses[detection.tag_id] = X_root_tag
        
        return tag_poses


if __name__ == '__main__':
    camera = RealSense()
    camera.start()
    
    server = viser.ViserServer(host='127.0.0.1', port=8080)

    while True:
        start_time = time.time()
        frame = camera.get_frame(require_pc=True)

        server.scene.add_frame(
            f'camera_pose',
            wxyz=R.from_matrix(X_root_camera[:3, :3]).as_quat()[[3, 0, 1, 2]],
            position=X_root_camera[:3, 3],
            axes_length=0.2,
            axes_radius=0.006
        )

        server.scene.add_point_cloud(
            'pc',
            frame['point_cloud'],
            point_size=0.0007,
            point_shape="circle",
            colors=(0, 0, 255)
        )
        server.scene.add_point_cloud(
            'bound',
            np.array([
                [0.3, 0.2, 0],
                [0.3, -0.2, 0], 
                [0.75, 0.2, 0],
                [0.75, -0.2, 0]
            ]),
            point_size=0.01,
            point_shape="circle",
            colors=(255, 0, 0)
        )
        print(1 / (time.time() - start_time))
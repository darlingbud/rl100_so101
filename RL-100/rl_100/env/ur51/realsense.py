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
    # bounding box filter
    x, y, z = point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2]
    bounding_box_mask = (z > 0) & (z < 0.2) & (x + y > -0.5) & (x + y < 0.5) & (x - y > 0.2) & (x - y < 1.1)
    point_cloud = point_cloud[bounding_box_mask]

    # table filter
    ro_mask = (point_cloud[:, 2] < 0.03) | (point_cloud[:, 2] > 0.055)
    point_cloud = point_cloud[ro_mask]

    # FPS sampling
    if len(point_cloud) < num_points:
        point_cloud = np.concatenate([point_cloud] * (num_points // len(point_cloud) + 1), axis=0)
    sample_idx = fpsample.bucket_fps_kdtree_sampling(point_cloud, num_points)
    point_cloud = point_cloud[sample_idx]

    return point_cloud


camera_intrinsics = (1350.468, 1350.468, 955.757, 529.578)
# modify after calibrating extrinsics 
X_root_camera = np.array([
    [ 0.98851007,  0.12170388, -0.08856737,  0.40647204],
    [ 0.15032704, -0.82862884,  0.53909247, -0.83822601],
    [-0.0077976 , -0.54631425, -0.83751563,  0.67779899],
    [ 0.        ,  0.        ,  0.        ,  1.        ]
])


class RealSense(object):
    def __init__(
            self,
            color_width=1920,
            color_height=1080,
            fps=30,
            enable_depth=True,
            depth_width=320,
            depth_height=240,
            apriltag_families="tagStandard41h12",
            num_points=1024
        ):
        self.enable_depth = enable_depth
        self.depth_width = depth_width
        self.depth_height = depth_height
        self.num_points = num_points

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, color_width, color_height, rs.format.bgr8, fps)
        if self.enable_depth:
            self.config.enable_stream(rs.stream.depth, depth_width, depth_height, rs.format.z16, fps)
        self.align = rs.align(rs.stream.color)

        self.detector = Detector(families=apriltag_families)

    def start(self):
        profile = self.pipeline.start(self.config)

        # get intrinsics
        self.intrinsics = camera_intrinsics
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        color_intrinsics = color_frame.get_profile().as_video_stream_profile().get_intrinsics()
        self.color_intrinsics = (color_intrinsics.fx, color_intrinsics.fy, color_intrinsics.ppx, color_intrinsics.ppy)
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
            # frames = self.align.process(frames)

            timestamp = frames.get_timestamp() / 1000  # ms -> s
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if color_frame and depth_frame:
                break

        color_image = np.array(color_frame.get_data())
        depth_image = np.array(depth_frame.get_data())

        point_cloud = point_cloud_downsample(depth2pc(
            depth_image * self.depth_scale,
            self.depth_intrinsics,
            X_root_camera
        ), self.num_points) if require_pc else None

        return {
            'timestamp': timestamp,
            'color': color_image,
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
        frame = camera.get_frame(require_pc=True)
        print('timestamp:', frame['timestamp'])

        cv2.imshow('RGB', frame['color'])
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        tag_poses = camera.detect_apriltag(frame['color'])
        for tag_id, tag_pose in enumerate(tag_poses):
            if tag_pose is None:
                continue
            print(f'    tag_{tag_id}: {tag_pose[:3, 3]}')
            server.scene.add_frame(
                f'tags/tag_{tag_id}',
                wxyz=R.from_matrix(tag_pose[:3, :3]).as_quat()[[3, 0, 1, 2]],
                position=tag_pose[:3, 3],
                axes_length=0.1,
                axes_radius=0.003
            )

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
            point_size=0.003,
            point_shape="circle",
            colors=(0, 0, 255)
        )
        z_bounding_lower, z_bounding_upper = 0, 0.2
        z_table_lower, z_table_upper = 0.03, 0.055
        server.scene.add_point_cloud(
            'pc_marker',
            np.array([
                [0.325, 0.175, z_bounding_lower],
                [0.825, -0.325, z_bounding_lower],
                [0.325, -0.825, z_bounding_lower],
                [-0.175, -0.325, z_bounding_lower],
                [0.325, 0.175, z_bounding_upper],
                [0.825, -0.325, z_bounding_upper],
                [0.325, -0.825, z_bounding_upper],
                [-0.175, -0.325, z_bounding_upper],
                [0.325, 0.175, z_table_lower],
                [0.825, -0.325, z_table_lower],
                [0.325, -0.825, z_table_lower],
                [-0.175, -0.325, z_table_lower],
                [0.325, 0.175, z_table_upper],
                [0.825, -0.325, z_table_upper],
                [0.325, -0.825, z_table_upper],
                [-0.175, -0.325, z_table_upper],
            ]),
            point_size=0.003,
            point_shape="circle",
            colors=(255, 0, 0)
        )
        time.sleep(1)

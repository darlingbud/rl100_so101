import cv2
import time
import numpy as np
import viser
from scipy.spatial.transform import Rotation as R
import pyrealsense2 as rs
from dt_apriltags import Detector
from PIL import Image
import fpsample


def depth2pc(depth, camera_intrinsics, camera_pose=np.eye(4), color=None):
    if color is not None:
        color = color.reshape(-1, 3)

    height, width = depth.shape
    fx, fy, cx, cy = camera_intrinsics
    z = depth.flatten()

    u, v = np.meshgrid(np.arange(width), np.arange(height))
    u = u.flatten()
    v = v.flatten()

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    points_camera = np.stack((x, y, z), axis=1)[z > 0]
    if color is not None:
        color = color[z > 0]

    points_camera_h = np.concatenate((points_camera, np.ones((points_camera.shape[0], 1))), axis=1)  # Nx4
    points_world = (camera_pose @ points_camera_h.T).T[:, :3]
    if color is not None:
        return points_world, color
    else:
        return points_world


def point_cloud_downsample(point_cloud, color = None, num_points = None):
    x, y, z = point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2]
    if color is not None:
        bounding_box_mask = (z > 0.07) & (z < 0.3) & (x > 0.3) & (x < 0.75) & (y > -0.2) & (y < 0.2)
    else:
        bounding_box_mask = (z > 0.09) & (z < 0.3) & (x > 0.3) & (x < 0.75) & (y > -0.2) & (y < 0.2)
    point_cloud = point_cloud[bounding_box_mask]
    if color is not None:
        color = color[bounding_box_mask]

    # FPS sampling
    # if len(point_cloud) < num_points:
    #     point_cloud = np.concatenate([point_cloud] * (num_points // len(point_cloud) + 1), axis=0)
    sample_idx = fpsample.bucket_fps_kdtree_sampling(point_cloud, num_points)
    point_cloud = point_cloud[sample_idx]
    if color is not None:
        color = color[sample_idx]
        return point_cloud, color
    else:
        return point_cloud


X_root_camera = np.array([
    [-0.99995623,  0.00573497, -0.00739244,  0.52725206],
    [ 0.00935479,  0.59924596, -0.80051032,  0.48233658],
    [-0.00016101, -0.80054443, -0.59927338,  0.38663279],
    [ 0.        ,  0.        ,  0.        ,  1.        ]
])
depth_intrinsics = (904.5899047851562, 905.5633544921875, 645.1522216796875, 362.4695129394531)


class RealSense(object):
    def __init__(
            self,
            fps=30,
            enable_depth=True,
            color_width=1280,
            color_height=720,
            depth_width=640,
            depth_height=480,
            num_points=5120,
            use_rgb=True,
        ):
        self.enable_depth = enable_depth
        self.depth_width = depth_width
        self.depth_height = depth_height
        self.num_points = num_points
        self.use_rgb = use_rgb  
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        if self.use_rgb:
            self.config.enable_stream(rs.stream.color, color_width, color_height, rs.format.rgb8, fps)
            self.align = rs.align(rs.stream.color)
        if self.enable_depth:
            self.config.enable_stream(rs.stream.depth, depth_width, depth_height, rs.format.z16, fps)

    def start(self):
        profile = self.pipeline.start(self.config)

        # get intrinsics
        frames = self.pipeline.wait_for_frames()
        if self.use_rgb:
            frames = self.align.process(frames)
        if self.enable_depth:
            self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
            depth_frame = frames.get_depth_frame()
            depth_intrinsics = depth_frame.get_profile().as_video_stream_profile().get_intrinsics()
            self.depth_intrinsics = (depth_intrinsics.fx, depth_intrinsics.fy, depth_intrinsics.ppx, depth_intrinsics.ppy)
            # print(self.depth_intrinsics)
            # exit()

    def stop(self):
        self.pipeline.stop()

    def get_frame(self, require_pc=False):
        while True:
            frames = self.pipeline.wait_for_frames()
            if self.use_rgb:
                frames = self.align.process(frames) 

            timestamp = frames.get_timestamp() / 1000  # ms -> s
            if self.use_rgb:
                color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if depth_frame:
                if self.use_rgb:
                    if color_frame:
                        break
                else:
                    break
        if self.use_rgb:
            color_image = np.array(color_frame.get_data())
        depth_image = np.array(depth_frame.get_data())
        if self.use_rgb:
            point_cloud, color_pc = depth2pc(
                depth_image * self.depth_scale,
                self.depth_intrinsics,
                X_root_camera,
                color_image
            ) if require_pc else (None, None)
        else:
            point_cloud = depth2pc(
                depth_image * self.depth_scale,
                self.depth_intrinsics,
                X_root_camera
            ) if require_pc else None
            color_pc = None
            color_image = None

        return {
            'timestamp': timestamp,
            'color_original': color_image,
            'color': color_pc,
            'depth': depth_image,
            'depth_scale': self.depth_scale,
            'point_cloud': point_cloud
        }


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

        hw_border = (frame['color_original'].shape[1] - frame['color_original'].shape[0]) // 2
        rgb_image = frame['color_original'][:, hw_border:-hw_border, :]
        border = 128
        # rgb_image = rgb_image[:-2*border, border:-border, :]
        # rgb_image = Image.fromarray(rgb_image).resize((224, 224), Image.Resampling.LANCZOS)
        # cv2.imshow("RGB", cv2.cvtColor(np.array(rgb_image), cv2.COLOR_RGB2BGR))
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

        point_cloud_ds, color_ds = point_cloud_downsample(frame['point_cloud'], frame['color'], num_points=2560)
        print(color_ds)
        server.scene.add_point_cloud(
            'pc',
            point_cloud_ds,
            point_size=0.002,
            point_shape="circle",
            colors=color_ds
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
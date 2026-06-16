import time
import viser
import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2

from realsense import depth2pc, camera_intrinsics, X_root_camera


if __name__ == '__main__':
    demo_path = "data/pushT/demo_032.npy"
    demo = np.load(demo_path, allow_pickle=True)
    # print(demo)
    import pdb; pdb.set_trace()
    def on_update(frame_idx):
        frame = demo[frame_idx]
        print("Frame:", frame['demo_frame_idx'])
        print("qpos:", frame['qpos'])
        print("action:", frame['action'])

        point_cloud = depth2pc(
            frame['depth'] * frame['depth_scale'],
            camera_intrinsics,
            X_root_camera
        )

        z = frame['depth'][frame['depth'] > 0]
        mask = point_cloud[:, 2] > -0.05
        point_cloud = point_cloud[mask]
        z = z[mask]
        depth_norm = cv2.normalize(z, None, 0, 255, cv2.NORM_MINMAX)
        depth_8bit = np.uint8(depth_norm)
        colormap = np.array(cv2.applyColorMap(depth_8bit, cv2.COLORMAP_JET))[:, 0]
        print(colormap.shape)

        server.scene.add_point_cloud(
            'point_cloud',
            point_cloud,
            point_size=0.0001,
            point_shape="circle",
            colors=colormap
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

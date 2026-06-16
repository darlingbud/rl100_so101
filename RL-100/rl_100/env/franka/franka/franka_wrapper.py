import numpy as np
from multiprocessing.managers import SharedMemoryManager
from scipy.spatial.transform import Rotation as R
from .franka_interpolation_controller import FrankaInterpolationController


class FrankaWrapper:
    def __init__(self):
        self.shm_manager = SharedMemoryManager()
        self.shm_manager.start()
        self.franka = FrankaInterpolationController(
            shm_manager=self.shm_manager,
            robot_ip='172.16.0.1',
            frequency=100,
            Kx_scale=5.0,
            Kxd_scale=2.0,
            joints_init=(0.0248, -0.0877, -0.1910, -2.5318, -0.0573, 2.4074, -2.5648),
            verbose=False,
        )
        self.franka.start()
        
    def get_pose(self):
        state = self.franka.get_state()
        pose = np.eye(4, dtype=np.float32)
        pose[:3, 3] = state['ActualTCPPose'][:3]
        pose[:3, :3] = R.from_rotvec(state['ActualTCPPose'][3:]).as_matrix()
        return pose

    def get_joint(self):
        state = self.franka.get_state()
        return state['ActualQ']
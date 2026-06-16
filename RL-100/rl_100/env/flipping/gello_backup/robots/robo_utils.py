from scipy.spatial.transform import Rotation as R
import numpy as np

def rotvec_to_matrix(rotvec):
    """旋转向量转旋转矩阵"""
    return R.from_rotvec(rotvec).as_matrix()


def matrix_to_rotvec(mat):
    """旋转矩阵转旋转向量"""
    return R.from_matrix(mat).as_rotvec()

def matrix_to_euler_xyz(mat):
    """旋转矩阵转欧拉角xyz"""
    return R.from_matrix(mat).as_euler('xyz')

def robot2world_tcp(robot_tcp, world2rbase):
    cur_ee_rotvec = robot_tcp[3:6]
    cur_ee_rotmat = np.eye(4)
    cur_ee_rotmat[:3, :3] = rotvec_to_matrix(cur_ee_rotvec)
    cur_ee_rotmat[:3, 3] = robot_tcp[:3]
    target_ee_rotmat = world2rbase @ cur_ee_rotmat
    return np.concatenate([target_ee_rotmat[:3, 3], matrix_to_euler_xyz(target_ee_rotmat[:3, :3])])

def world2robot_tcp(world_tcp_euler, r_world2base_inv):
    world_tcp_mat = np.eye(4)
    world_tcp_euler_mat = R.from_euler('xyz', world_tcp_euler[3:6]).as_matrix()
    world_tcp_mat[:3, :3] = world_tcp_euler_mat
    world_tcp_mat[:3, 3] = world_tcp_euler[:3]
    target_ee_euler_mat = r_world2base_inv @ world_tcp_mat
    return np.concatenate([target_ee_euler_mat[:3, 3], matrix_to_rotvec(target_ee_euler_mat[:3, :3])])
import time
from typing import Dict
from scipy.spatial.transform import Rotation as R
from gello.robots.const import L_WORLD2BASE, L_WORLD2BASE_INV, R_WORLD2BASE, R_WORLD2BASE_INV

import numpy as np

from gello.robots.robot import Robot
from gello.robots.robo_utils import robot2world_tcp, world2robot_tcp


class URRobot(Robot):
    """A class representing a UR robot."""

    def __init__(self, robot_ip: str = "192.168.1.10", no_gripper: bool = False, use_world_frame: bool = False):
        import rtde_control
        import rtde_receive

        [print("in ur robot") for _ in range(4)]
        try:
            self.robot = rtde_control.RTDEControlInterface(robot_ip)
        except Exception as e:
            print(e)
            print(robot_ip)
        self.robot_ip = robot_ip
        self.r_inter = rtde_receive.RTDEReceiveInterface(robot_ip)
        if not no_gripper:
            from gello.robots.robotiq_gripper import RobotiqGripper

            self.gripper = RobotiqGripper()
            self.gripper.connect(hostname=robot_ip, port=63352)
            print("gripper connected")
            # gripper.activate()

        [print("connect") for _ in range(4)]

        self._free_drive = False
        self.robot.endFreedriveMode()
        self._use_gripper = not no_gripper
        self._use_world_frame = use_world_frame
        print('use_world_frame', self._use_world_frame)

    def num_dofs(self) -> int:
        """Get the number of joints of the robot.

        Returns:
            int: The number of joints of the robot.
        """
        if self._use_gripper:
            return 7
        return 6

    def _get_gripper_pos(self) -> float:
        import time

        time.sleep(0.001)
        gripper_pos = self.gripper.get_current_position()
        assert 0 <= gripper_pos <= 255, "Gripper position must be between 0 and 255"
        return gripper_pos / 255

    def get_joint_state(self) -> np.ndarray:
        """Get the current state of the leader robot.

        Returns:
            T: The current state of the leader robot.
        """
        robot_joints = self.r_inter.getActualQ()
        if self._use_gripper:
            gripper_pos = self._get_gripper_pos()
            pos = np.append(robot_joints, gripper_pos)
        else:
            pos = robot_joints
        return pos

    def command_joint_state(self, joint_state: np.ndarray) -> None:
        """Command the leader robot to a given state.

        Args:
            joint_state (np.ndarray): The state to command the leader robot to.
        """
        velocity = 0.5
        acceleration = 0.5
        dt = 1.0 / 500  # 2ms
        lookahead_time = 0.2
        gain = 100

        robot_joints = joint_state[:6]
        t_start = self.robot.initPeriod()
        self.robot.servoJ(
            robot_joints, velocity, acceleration, dt, lookahead_time, gain
        )
        if self._use_gripper:
            gripper_pos = joint_state[-1] * 255
            self.gripper.move(gripper_pos, 255, 10)
        self.robot.waitPeriod(t_start)

    def command_cartesian_state(self, cartesian_state: np.ndarray) -> None:
        """Command the robot to a given cartesian state using servoL.

        Args:
            cartesian_state (np.ndarray): [x, y, z, rx, ry, rz, gripper] (7,)
                位置 (米) + 欧拉角 xyz (角度) + 夹爪位置
        """
        if self._use_world_frame:
            if self.robot_ip.endswith('101'):
                tcp_pose = world2robot_tcp(cartesian_state, L_WORLD2BASE_INV)
            else:
                tcp_pose = world2robot_tcp(cartesian_state, R_WORLD2BASE_INV)
        else:
            # 提取位置和欧拉角
            pos = cartesian_state[:3]  # [x, y, z]
            euler_deg = cartesian_state[3:6]  # [rx, ry, rz]
            
            # 欧拉角转 axis-angle (UR 使用的格式)
            rot = R.from_euler('xyz', euler_deg)
            rotvec = rot.as_rotvec()  # axis-angle representation
            
            # 组合成 UR 的 TCP pose 格式
            tcp_pose = np.concatenate([pos, rotvec])  # [x, y, z, rx, ry, rz]
        
        # 使用 servoL 进行笛卡尔空间控制
        acceleration = 0.5
        velocity = 0.5
        dt = 1.0 / 500  # 2ms
        lookahead_time = 0.1
        gain = 300
        
        t_start = self.robot.initPeriod()
        self.robot.servoL(
            tcp_pose.tolist(),
            acceleration,
            velocity,
            dt,
            lookahead_time,
            gain
        )
        
        # 控制夹爪
        if self._use_gripper:
            gripper_pos = cartesian_state[6] * 255
            self.gripper.move(gripper_pos, 255, 10)
        
        self.robot.waitPeriod(t_start)

    def freedrive_enabled(self) -> bool:
        """Check if the robot is in freedrive mode.

        Returns:
            bool: True if the robot is in freedrive mode, False otherwise.
        """
        return self._free_drive

    def set_freedrive_mode(self, enable: bool) -> None:
        """Set the freedrive mode of the robot.

        Args:
            enable (bool): True to enable freedrive mode, False to disable it.
        """
        if enable and not self._free_drive:
            self._free_drive = True
            self.robot.freedriveMode()
        elif not enable and self._free_drive:
            self._free_drive = False
            self.robot.endFreedriveMode()

    def get_observations(self) -> Dict[str, np.ndarray]:
        
        joints = self.get_joint_state()
        
        # 获取真实的 TCP pose (x, y, z, rx, ry, rz) - 使用 axis-angle 表示
        tcp_pose = self.r_inter.getActualTCPPose()  # [x, y, z, rx, ry, rz]
        
        if self._use_world_frame:
            # 转换到 world 系
            if self.robot_ip.endswith('101'):
                pos_rot = robot2world_tcp(tcp_pose, L_WORLD2BASE)
            else:
                pos_rot = robot2world_tcp(tcp_pose, R_WORLD2BASE)
        else:
            # 保持在 robot-base 系，只做格式转换 (axis-angle -> euler)
            pos = np.array(tcp_pose[:3])  # [x, y, z] in meters
            rotvec = np.array(tcp_pose[3:])  # axis-angle representation
            rot = R.from_rotvec(rotvec)
            euler = rot.as_euler('xyz', degrees=False)  # [rx, ry, rz] in radians
            pos_rot = np.concatenate([pos, euler])  # [x, y, z, rx, ry, rz]
        
        gripper_pos = np.array([joints[-1]])
        return {
            "joint_positions": joints,
            "joint_velocities": joints,
            "ee_pos_rot": pos_rot,
            "gripper_position": gripper_pos,
        }


def main():
    robot_ip = "192.168.1.11"
    ur = URRobot(robot_ip, no_gripper=True)
    print(ur)
    ur.set_freedrive_mode(True)
    print(ur.get_observations())


if __name__ == "__main__":
    main()

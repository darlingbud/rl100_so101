import time

from .UR5_env import UR5Env


if __name__ == '__main__':
    env = UR5Env(
        robot_ip='192.169.0.10',
        dt=1/30,
        multi_process=True,
        control_mode=UR5Env.ControlMode.TCP,
        delta_action=True
    )

    while True:
        time.sleep(1)

import numpy as np
from multiprocessing import Process
import threading
import queue
import time
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface

from rl_100.env.ur5.shared_memory.shared_queue import SharedMemoryQueue, Full, Empty
from rl_100.env.ur5.command import Command

class UR5Controller(Process):
    def __init__(self, shm_manager, robot_ip) -> None:
        super().__init__(daemon=True)

        self._queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples={'cmd': Command.SERVOJ.value, 'action': np.zeros(6, dtype=np.float32)},
            buffer_size=256
        )
        self._robot_ip = robot_ip

    def send_action(self, action):
        self._queue.put(action)

    def stop(self):
        self._rtde_c.stopScript()
        self.close()

    def run(self):
        print("Running controller...")
        # robot
        self._rtde_r = RTDEReceiveInterface(self._robot_ip)
        self._rtde_c = RTDEControlInterface(self._robot_ip)

        while True:
            try:
                data = self._queue.get()
                
                if data['cmd'] == Command.SERVOJ.value:  # TODO
                    pass
                    # st_time = time.monotonic()
                    # print("begin controller", self._rtde_r.getActualQ())

                    # action = data['action']
                    # self._gripper_queue.put(action[-1])
                    # print("gripper q:", self._gripper_queue.qsize())
                    # self._rtde_c.servoJ(action[:-1], 0, 0, 0.2, 0.1, 300)

                    # print("end controller", time.monotonic() - st_time, self._rtde_r.getActualQ())
                elif data['cmd'] == Command.SERVOL.value:
                    action = data['action']
                    self._rtde_c.servoL(action, 0, 0, 0.033, 0.1, 200)
                elif data['cmd'] == Command.MOVEJ.value:
                    self._movej(data['action'])
                elif data['cmd'] == Command.STOP.value:
                    self.stop()
            except Empty:
                pass
            except Exception as e:
                print("Error in controller: ", e)

    def _gripper_control(self):
        while True:
            angle = int(self._gripper_queue.get())
            self._gripper.move(angle, 50, 0)

    def _servoj(self, qpos):
        t_start = self._rtde_c.initPeriod()
        self._rtde_c.servoJ(qpos, 0, 0, 0.008, 0.1, 300)
        self._rtde_c.waitPeriod(t_start)

    def _movej(self, qpos):
        self._rtde_c.servoStop()
        self._rtde_c.moveJ(qpos)
        q = self._rtde_r.getActualQ()
        self._rtde_c.servoJ(q, 0, 0, 0.1, 0.1, 300)

    def _stop_script(self):
        self._rtde_c.stopScript()

    def _rescale_action(self, action, action_scale):
        minimum = -1.0
        maximum = 1.0
        scale = 2.0 * action_scale * np.ones_like(action) / (maximum - minimum)
        return -action_scale + (action - minimum) * scale
    
    @property
    def rtde_c(self):
        return self._rtde_c
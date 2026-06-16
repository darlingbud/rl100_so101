import threading
import queue
from command import Command

class Policy:
    def __init__(self, env):
        self.env = env
        self.policy = policy  # TODO

        self.running = True
        self.obs_queue = queue.Queue(maxsize=1)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while self.running:
            try:
                obs = self.obs_queue.get(block=False)
                action = self.policy.predict(obs)  # TODO
                action_scale = 0.01

                self.env._cur_tcp[0] += action_scale * action[0]
                self.env._cur_tcp[1] += action_scale * action[1]
                self._controller.send_action({
                    'cmd': 3,
                    'action': self.env._cur_tcp
                })
            except queue.Empty:
                continue

    def update_obs(self, obs):
        while not self.obs_queue.empty():
            self.obs_queue.get_nowait()
        self.obs_queue.put(obs)

    def stop(self):
        self.running = False
        self.thread.join()

#!/usr/bin/env python
import argparse
import os
import time

import numpy as np

os.environ["MUJOCO_GL"] = "glfw"
os.environ.pop("PYOPENGL_PLATFORM", None)

from mujoco_py import MjViewer


def parse_args():
    parser = argparse.ArgumentParser(description="Open an interactive MuJoCo viewer for RL-100 scenes.")
    parser.add_argument("--suite", choices=["metaworld", "adroit"], default="metaworld")
    parser.add_argument("--task", default="reach", help="MetaWorld: reach/door-unlock/etc. Adroit: door/hammer/pen/relocate.")
    parser.add_argument("--seconds", type=float, default=0.0, help="Auto-close after N seconds. 0 means run until the window is closed.")
    parser.add_argument("--animate", action="store_true", help="Step the simulation instead of showing a static reset state.")
    parser.add_argument("--random-actions", action="store_true", help="Use random actions while animating.")
    parser.add_argument("--fps", type=float, default=60.0)
    return parser.parse_args()


def make_sim(args):
    if args.suite == "metaworld":
        import metaworld

        task_name = args.task
        if "-v2" not in task_name:
            task_name = f"{task_name}-v2-goal-observable"
        env = metaworld.envs.ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[task_name]()
        env._freeze_rand_vec = False
        env.reset()
        return env, env.sim, env.action_space

    from rl_100.env import AdroitEnv

    env = AdroitEnv(env_name=args.task, use_point_cloud=False)
    env.reset()
    return env, env.get_mujoco_sim(), env.action_space


def main():
    args = parse_args()
    env, sim, action_space = make_sim(args)
    viewer = MjViewer(sim)
    if sim.model.ncam > 0:
        viewer.cam.fixedcamid = -1
        viewer.cam.type = 0

    dt = 1.0 / max(args.fps, 1.0)
    start = time.time()
    zero_action = np.zeros(action_space.shape, dtype=action_space.dtype)

    try:
        while True:
            if args.animate:
                action = action_space.sample() if args.random_actions else zero_action
                env.step(action)
            else:
                sim.forward()

            viewer.render()
            time.sleep(dt)

            if args.seconds > 0 and (time.time() - start) >= args.seconds:
                break
    finally:
        if hasattr(env, "close"):
            env.close()


if __name__ == "__main__":
    main()

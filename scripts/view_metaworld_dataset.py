#!/usr/bin/env python
import argparse
import time

import cv2
import numpy as np
import zarr

from scripts.view_metaworld_env import draw_pointcloud_projection, resize_frame


def parse_args():
    parser = argparse.ArgumentParser(description="Play an RL-100 MetaWorld zarr demonstration dataset.")
    parser.add_argument("--zarr", default="RL-100/data/metaworld_reach_expert.zarr")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--record", default=None, help="Optional output video path, e.g. outputs/metaworld_reach_expert_ep0.mp4")
    parser.add_argument("--show-pointcloud", action="store_true", help="Show recorded point-cloud projections next to RGB frames.")
    parser.add_argument("--loop", action="store_true", help="Loop the selected episode until q/esc is pressed.")
    return parser.parse_args()


def episode_slice(episode_ends, episode):
    if episode < 0 or episode >= len(episode_ends):
        raise IndexError(f"episode must be in [0, {len(episode_ends) - 1}], got {episode}")
    start = 0 if episode == 0 else int(episode_ends[episode - 1])
    end = int(episode_ends[episode])
    return start, end


def main():
    args = parse_args()
    root = zarr.open(args.zarr, mode="r")
    data = root["data"]
    episode_ends = root["meta"]["episode_ends"][:]
    start, end = episode_slice(episode_ends, args.episode)

    delay_ms = max(1, int(1000 / args.fps))
    writer = None
    window_name = f"RL-100 MetaWorld dataset episode {args.episode}"

    try:
        while True:
            for local_step, idx in enumerate(range(start, end)):
                frame = data["img"][idx]
                frame = resize_frame(frame, args.width)
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                reward = float(data["reward"][idx, 0])
                done = bool(data["done"][idx, 0])
                action = data["action"][idx]
                text = (
                    f"episode {args.episode} step {local_step}/{end - start - 1} "
                    f"reward {reward:.3f} done {int(done)} | q/esc quit"
                )
                cv2.putText(
                    bgr,
                    text,
                    (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (40, 255, 40),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    bgr,
                    "action " + np.array2string(action, precision=4, suppress_small=False),
                    (10, bgr.shape[0] - 14),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (40, 255, 40),
                    1,
                    cv2.LINE_AA,
                )

                if args.show_pointcloud:
                    pc_view = draw_pointcloud_projection(data["point_cloud"][idx], size=bgr.shape[0])
                    bgr = np.concatenate([bgr, pc_view], axis=1)

                if args.record is not None and writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(args.record, fourcc, args.fps, (bgr.shape[1], bgr.shape[0]))

                if writer is not None:
                    writer.write(bgr)

                cv2.imshow(window_name, bgr)
                key = cv2.waitKey(delay_ms) & 0xFF
                if key in (ord("q"), 27):
                    return
                time.sleep(0.001)

            if not args.loop:
                return
    finally:
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

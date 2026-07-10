#!/usr/bin/env python
import argparse
import time

import cv2
import numpy as np

from rl_100.env.metaworld.metaworld_wrapper import MetaWorldEnv


def parse_args():
    parser = argparse.ArgumentParser(description="Preview an RL-100 MetaWorld environment.")
    parser.add_argument("--task", default="reach", help="MetaWorld task name, e.g. reach, push, pick-place, door-open.")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--rgb-size", type=int, default=128)
    parser.add_argument("--num-points", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--record", default=None, help="Optional output video path, e.g. outputs/metaworld_reach.mp4")
    parser.add_argument("--show-pointcloud", action="store_true", help="Show point-cloud projections next to the RGB render.")
    parser.add_argument("--no-point-crop", action="store_true", help="Disable point-cloud cropping.")
    return parser.parse_args()


def resize_frame(frame, width):
    scale = width / frame.shape[1]
    height = max(1, int(round(frame.shape[0] * scale)))
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_NEAREST)


def draw_pointcloud_projection(point_cloud, size=512):
    point_cloud = np.asarray(point_cloud)
    if point_cloud.ndim == 3:
        point_cloud = point_cloud[-1]

    xyz = point_cloud[:, :3]
    valid = np.isfinite(xyz).all(axis=1) & (np.linalg.norm(xyz, axis=1) > 1e-8)
    xyz = xyz[valid]
    if point_cloud.shape[1] >= 6:
        rgb = np.clip(point_cloud[valid, 3:6], 0, 255).astype(np.uint8)
    else:
        z = xyz[:, 2] if xyz.shape[0] else np.array([0.0])
        z_norm = (z - z.min()) / max(float(z.max() - z.min()), 1e-6)
        rgb = cv2.applyColorMap((z_norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)[:, 0, ::-1]

    canvas = np.full((size, size, 3), 245, dtype=np.uint8)
    if xyz.shape[0] == 0:
        cv2.putText(canvas, "empty point cloud", (24, size // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 40, 40), 2)
        return canvas

    def project(points, dims, origin, extent):
        pts = points[:, dims]
        center = pts.mean(axis=0)
        span = np.maximum(pts.max(axis=0) - pts.min(axis=0), 1e-6)
        scale = 0.78 * min(extent[0] / span[0], extent[1] / span[1])
        uv = (pts - center) * scale
        uv[:, 1] *= -1
        uv += np.array(origin)
        return np.round(uv).astype(np.int32)

    half_h = size // 2
    views = [
        ("top x/y", (0, 1), (size // 2, half_h // 2), (size, half_h)),
        ("front x/z", (0, 2), (size // 2, half_h + half_h // 2), (size, half_h)),
    ]
    cv2.line(canvas, (0, half_h), (size, half_h), (210, 210, 210), 1)
    for label, dims, origin, extent in views:
        uv = project(xyz, dims, origin, extent)
        in_frame = (uv[:, 0] >= 0) & (uv[:, 0] < size) & (uv[:, 1] >= 0) & (uv[:, 1] < size)
        order = np.argsort(xyz[:, 2])
        for idx in order:
            if not in_frame[idx]:
                continue
            color = tuple(int(v) for v in rgb[idx][::-1])
            cv2.circle(canvas, tuple(uv[idx]), 2, color, -1, lineType=cv2.LINE_AA)
        label_y = 24 if origin[1] < half_h else half_h + 24
        cv2.putText(canvas, label, (10, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 1, cv2.LINE_AA)

    cv2.putText(canvas, f"{xyz.shape[0]} pts", (10, size - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1, cv2.LINE_AA)
    return canvas


def main():
    args = parse_args()
    env = MetaWorldEnv(
        task_name=args.task,
        device=args.device,
        use_point_crop=not args.no_point_crop,
        num_points=args.num_points,
        rgb_size=args.rgb_size,
    )
    obs = env.reset()

    delay_ms = max(1, int(1000 / args.fps))
    writer = None
    window_name = f"RL-100 MetaWorld {args.task}"

    try:
        for step in range(args.steps):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)

            frame = env.render(mode="rgb_array")
            frame = resize_frame(frame, args.width)
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            success = float(info.get("success", 0.0))
            cv2.putText(
                bgr,
                f"step {step} reward {float(reward):.3f} success {success:.0f} | q/esc quit, r reset",
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (40, 255, 40),
                1,
                cv2.LINE_AA,
            )

            if args.show_pointcloud:
                pc_view = draw_pointcloud_projection(obs["point_cloud"], size=bgr.shape[0])
                bgr = np.concatenate([bgr, pc_view], axis=1)

            if args.record is not None and writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(args.record, fourcc, args.fps, (bgr.shape[1], bgr.shape[0]))

            if writer is not None:
                writer.write(bgr)

            cv2.imshow(window_name, bgr)
            key = cv2.waitKey(delay_ms) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r") or done:
                obs = env.reset()
            time.sleep(0.001)
    finally:
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        env.close()


if __name__ == "__main__":
    main()

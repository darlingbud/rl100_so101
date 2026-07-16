#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SERVER_URL="${SERVER_URL:-ws://192.168.0.135:8000}"
ROBOT_PORT="${ROBOT_PORT:-/dev/robot_follower}"
FRONT_CAMERA="${FRONT_CAMERA:-0}"
SIDE_CAMERA="${SIDE_CAMERA:-1}"
CONTROL_FPS="${CONTROL_FPS:-10}"
INFERENCE_FPS="${INFERENCE_FPS:-6}"
DURATION="${DURATION:-30}"

echo "LeRobot policy client dry-run test"
echo "  server:         ${SERVER_URL}"
echo "  robot port:     ${ROBOT_PORT}"
echo "  cameras:        front=${FRONT_CAMERA}, side=${SIDE_CAMERA}"
echo "  control target: ${CONTROL_FPS} Hz"
echo "  request target: ${INFERENCE_FPS} Hz"
echo "  duration:       ${DURATION} s"
echo "  motor commands: disabled"

exec "${SCRIPT_DIR}/run_lerobot_policy_client.sh" \
  --url "${SERVER_URL}" \
  --port "${ROBOT_PORT}" \
  --front-camera "${FRONT_CAMERA}" \
  --side-camera "${SIDE_CAMERA}" \
  --control-fps "${CONTROL_FPS}" \
  --inference-fps "${INFERENCE_FPS}" \
  --duration "${DURATION}"

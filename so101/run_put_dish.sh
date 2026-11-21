#!/usr/bin/env bash
# SO-101 follower: run the pretrained ACT policy jliu6718/act-so101-put_dish (wrist camera only).
# Usage:
#   ./so101/run_put_dish.sh calibrate   # once, LeRobot calibration of the follower (interactive)
#   ./so101/run_put_dish.sh camera      # find the wrist camera index, saves test images
#   ./so101/run_put_dish.sh run         # one 15 s rollout
# Stop Station first: it holds the USB ports.
set -euo pipefail

# ---- ADJUST ----
FOLLOWER_PORT=/dev/tty.usbmodem5A7C1215571   # follower bus serial 5A7C121557
CAM_INDEX=0                                   # wrist camera index from "camera" step
DURATION_S=15                                 # training episodes were 5 to 8 s long
MAX_STEP=8                                    # max change per control step (safety clamp, in calibrated units)
# ----------------

ENV=~/lerobot-env/bin
MODEL=~/lerobot-models/act-so101-put_dish
ROBOT_ID=so101_follower_leukerbad

case "${1:-}" in
  calibrate)
    "$ENV/lerobot-calibrate" --robot.type=so101_follower --robot.port="$FOLLOWER_PORT" --robot.id="$ROBOT_ID"
    ;;
  camera)
    "$ENV/lerobot-find-cameras" opencv
    echo "Images saved in ./outputs/captured_images. Pick the wrist view and set CAM_INDEX."
    ;;
  run)
    "$ENV/lerobot-rollout" \
      --robot.type=so101_follower \
      --robot.port="$FOLLOWER_PORT" \
      --robot.id="$ROBOT_ID" \
      --robot.max_relative_target="$MAX_STEP" \
      --robot.cameras="{wrist: {type: opencv, index_or_path: $CAM_INDEX, width: 640, height: 480, fps: 30}}" \
      --policy.path="$MODEL" \
      --policy.device=mps \
      --task="put_dish" \
      --duration="$DURATION_S" \
      --play_sounds=false
    ;;
  *)
    echo "Usage: $0 {calibrate|camera|run}"; exit 1 ;;
esac

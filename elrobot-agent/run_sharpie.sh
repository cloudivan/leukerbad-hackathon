#!/usr/bin/env bash
# Claude API agent: pick the Sharpie and place it on the cardboard (same agent.py as the paper ball run).
# Needs Station (tab 1) and ./start_ik.sh (tab 2) running. DRY_RUN comes from .env unless set here.
# Usage:  ./run_sharpie.sh            (uses .env DRY_RUN)
#         DRY_RUN=1 ./run_sharpie.sh  (solve + safety-check only, nothing moves)
set -euo pipefail
cd "$(dirname "$0")"

# ---- ADJUST ----
CARDBOARD_TOP_Z=0.004   # m, top of the cardboard above the table. Flat sheet ~0.003-0.005. A box: measure it,
                        # but keep it <= 0.05, the arm cannot reach above z 0.10 with a vertical gripper.
CARDBOARD_WHERE="to the side of the sharpie"   # e.g. "on the robot's left (right side in Camera 0)"
# ----------------

FLOOR_Z=0.023   # ADJUST: z_floor at the cardboard (fit 15.11. 15:30, see camera_notes_sharpie.txt)
PLACE_Z=$(python3 -c "print(round($FLOOR_Z + $CARDBOARD_TOP_Z + 0.015, 3))")
TRAVEL_Z=$(python3 -c "print(round(max(0.080, $FLOOR_Z + $CARDBOARD_TOP_Z + 0.045), 3))")

TASK="Pick up the black Sharpie marker lying on the table and place it on the cardboard ${CARDBOARD_WHERE}. \
Close the jaws across the sharpie's narrow width: yaw = 1.5708 if it lies along X (left-right in Camera 0), yaw = 0 if it lies along Y (up-down in Camera 0). \
The table is about 2 cm above z 0 (floor plane in the camera notes): grip it at its middle at z = z_floor + 0.007 (about 0.028). CARDBOARD_TOP_Z = ${CARDBOARD_TOP_Z}. Carry it at z = ${TRAVEL_Z}, \
release it at z = ${PLACE_Z} above the cardboard center, then lift straight up and confirm with Camera 0 that it lies on the cardboard."

echo "Task: $TASK"
CAMERA_ORDER=28ff0ca87978e8f993c2ac64a84d188a,f9a49c847e496cac500305ae1f0a178e CAMERA_NOTES=camera_notes_sharpie.txt exec uv run python agent.py "$TASK"

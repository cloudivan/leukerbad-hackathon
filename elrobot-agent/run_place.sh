#!/usr/bin/env bash
# Claude API agent: pick an object and place it on a target (generic version of run_knife.sh).
# Usage: ./run_place.sh "the red circle (round red badge)" "the brown box"
#        CLAUDE_MODEL=claude-sonnet-5-5 ./run_place.sh ...   (model override for one run)
set -euo pipefail
cd "$(dirname "$0")"
OBJECT=${1:?object description}
TARGET=${2:-the brown box}

# ---- ADJUST ----
TARGET_TOP_Z=${TARGET_TOP_Z:-0.004}   # m, top of the target above the table (flat cardboard ~0.004; a box: measure it, <= 0.04)
FLOOR_Z=0.023        # m, z_floor at the target (floor plane in the camera notes)
GRIP_ABOVE_FLOOR=${GRIP_ABOVE_FLOOR:-0.007}   # m, grip height above the table (thin flat objects ~0.005-0.008)
RELEASE_ABOVE=${RELEASE_ABOVE:-0.030}   # m, open the jaws this far above the target top (object drops the rest)
# ----------------

PLACE_Z=$(python3 -c "print(round($FLOOR_Z + $TARGET_TOP_Z + $RELEASE_ABOVE, 3))")
TRAVEL_Z=$(python3 -c "print(round(max(0.080, $FLOOR_Z + $TARGET_TOP_Z + $RELEASE_ABOVE + 0.020), 3))")

TASK="Pick up ${OBJECT} from the table and place it on ${TARGET}. \
Grip it at z = z_floor + ${GRIP_ABOVE_FLOOR} (the table is about 2 cm above z 0, floor plane in the camera notes). \
TARGET_TOP_Z = ${TARGET_TOP_Z}. Carry it at z = ${TRAVEL_Z}, go straight down over the center of ${TARGET} to z = ${PLACE_Z} and open the jaws there (do not touch the target), \
then lift straight up and confirm with Camera 0 that it lies on ${TARGET}."

echo "Task: $TASK"
CAMERA_ORDER=43640b79a54e71f45e0443ffbb92880a,efd6933351072d2a8004e1cffb19adb3 CAMERA_NOTES=camera_notes_place.txt exec uv run python agent.py "$TASK"

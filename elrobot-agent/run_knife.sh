#!/usr/bin/env bash
# Claude API agent: pick the knife and place it on the brown container (same agent.py as the sharpie run).
# Needs Station + ./start_ik.sh running. DRY_RUN comes from .env.
set -euo pipefail
cd "$(dirname "$0")"

# ---- ADJUST ----
TARGET_TOP_Z=0.004   # m, top of the brown container above the table. Flat cardboard ~0.004. A box: measure it (<= 0.04).
FLOOR_Z=0.023        # m, z_floor at the container (floor plane in the camera notes)
# ----------------

PLACE_Z=$(python3 -c "print(round($FLOOR_Z + $TARGET_TOP_Z + 0.015, 3))")
TRAVEL_Z=$(python3 -c "print(round(max(0.080, $FLOOR_Z + $TARGET_TOP_Z + 0.045), 3))")

TASK="Pick up the knife lying on the table by its handle and place it on the brown container. \
Grip the handle at its middle at z = z_floor + 0.007 (about 0.026; the table is about 2 cm above z 0, floor plane in the camera notes). \
TARGET_TOP_Z = ${TARGET_TOP_Z}. Carry it at z = ${TRAVEL_Z}, release it at z = ${PLACE_Z} above the container center, \
then lift straight up and confirm with Camera 0 that it lies on the container."

echo "Task: $TASK"
CAMERA_ORDER=28ff0ca87978e8f993c2ac64a84d188a,f9a49c847e496cac500305ae1f0a178e CAMERA_NOTES=camera_notes_knife.txt exec uv run python agent.py "$TASK"

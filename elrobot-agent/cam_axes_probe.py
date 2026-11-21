"""Measure camera picture directions with exact commanded moves at hover height.

Poses (all z 0.080, yaw 1.5708): A base (0.17, 0.00), B +X (0.20, 0.00), C +Y (0.17, 0.03).
After each move: photos from every camera to logs/cam_axes_<time>/<pose>_<camera id>.jpg.
Ends at safety.REST_POSE under torque. E-stop: 12 V follower plug.

    uv run python cam_axes_probe.py
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

import safety
from ik_client import IKClient
from safety import Target

HERE = Path(__file__).resolve().parent
load_dotenv(HERE / ".env")
IK_URL = os.environ.get("IK_URL", "http://127.0.0.1:8020").strip()
FOLLOWER_ID = os.environ.get("FOLLOWER_ID", "5B79015365").strip()

Z, YAW = 0.080, 1.5708          # ADJUST: hover height and jaw yaw
POSES = {"A_base": (0.17, 0.00), "B_plusX": (0.20, 0.00), "C_plusY": (0.17, 0.03)}  # ADJUST


def main() -> None:
    ik = IKClient(IK_URL)
    out = HERE / "logs" / f"cam_axes_{time.strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True)
    cams = [c["id"] for c in ik.cameras()]
    log = {}
    try:
        for name, (x, y) in POSES.items():
            r = ik.move_to(FOLLOWER_ID, Target(x, y, Z, YAW), execute=True)
            time.sleep(1.5)   # let the arm settle and the camera stream catch up
            p = ik.follower(FOLLOWER_ID)["pose"]["position"]
            log[name] = {"cmd": [x, y, Z], "meas": [round(p[k], 4) for k in "xyz"], "status": r["motion"]["status"]}
            for cid in cams:
                (out / f"{name}_{cid}.jpg").write_bytes(ik.camera_image(cid))
            print(name, log[name], flush=True)
    finally:
        r = ik.move_to(FOLLOWER_ID, safety.REST_POSE, execute=True)
        print("rest:", r["motion"]["status"])
        (out / "poses.json").write_text(json.dumps(log, indent=1))
        print("saved", out)


if __name__ == "__main__":
    main()

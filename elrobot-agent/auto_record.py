"""Hands-free calibration recorder (torque must already be OFF, nothing is commanded).

Logs the pose at ~10 Hz. Whenever the arm is held still for STILL_S seconds and is more than
MIN_MOVE away from the last snapshot, it saves a snapshot: averaged pose, joints, all camera
images. So: put tips on the floor marker, hold 1.5 s (snapshot), lift 4-5 cm, hold 1.5 s
(snapshot), next marker...

    uv run python auto_record.py            # Ctrl+C to stop
Output: logs/auto_calib_<time>/snapshots.json, s<N>_cam<i>.jpg, stream.jsonl
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

from ik_client import IKClient, IKError

HERE = Path(__file__).resolve().parent
load_dotenv(HERE / ".env")
IK_URL = os.environ.get("IK_URL", "http://127.0.0.1:8020").strip()
FOLLOWER_ID = os.environ.get("FOLLOWER_ID", "5B79015365").strip()

STILL_S = 1.5        # ADJUST: hold time that triggers a snapshot
STILL_TOL = 0.0015   # ADJUST: max xyz wobble (m) that still counts as "still"
MIN_MOVE = 0.012     # ADJUST: min distance from the previous snapshot (m)


def main() -> None:
    ik = IKClient(IK_URL)
    cams = ik.cameras()
    out = HERE / "logs" / f"auto_calib_{time.strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True)
    stream = (out / "stream.jsonl").open("a")
    snaps: list[dict] = []
    window: list[tuple[float, dict, dict]] = []
    last_snap_pos = None
    print(f"Recording to {out}. Hold still {STILL_S}s for a snapshot. Ctrl+C to stop.", flush=True)
    try:
        while True:
            try:
                arm = ik.follower(FOLLOWER_ID)
            except IKError as e:
                print(f"  read failed: {e.code}", flush=True)
                time.sleep(0.5)
                continue
            t = time.time()
            p = arm["pose"]["position"]
            stream.write(json.dumps({"t": t, "pos": p, "torque": arm["torque_enabled"]}) + "\n")
            window.append((t, p, arm))
            window = [w for w in window if t - w[0] <= STILL_S]
            span = t - window[0][0]
            wobble = max(max(w[1][k] for w in window) - min(w[1][k] for w in window) for k in "xyz")
            far = last_snap_pos is None or sum((p[k] - last_snap_pos[k]) ** 2 for k in "xyz") ** 0.5 > MIN_MOVE
            if span >= STILL_S * 0.95 and wobble < STILL_TOL and far:
                avg = {k: round(sum(w[1][k] for w in window) / len(window), 4) for k in "xyz"}
                n = len(snaps) + 1
                imgs = []
                for i, c in enumerate(cams):
                    try:
                        (out / f"s{n}_cam{i}.jpg").write_bytes(ik.camera_image(c["id"]))
                        imgs.append(f"s{n}_cam{i}.jpg")
                    except IKError:
                        pass
                snaps.append({"n": n, "t": t, "pos": avg, "rpy": arm["pose"]["rpy"],
                              "joints": [round(j["angle_rad"], 4) for j in arm["joints"]],
                              "torque": arm["torque_enabled"], "images": imgs})
                (out / "snapshots.json").write_text(json.dumps({"cameras": cams, "snapshots": snaps}, indent=2))
                last_snap_pos = avg
                print(f"SNAP {n}: x {avg['x']:.4f} y {avg['y']:.4f} z {avg['z']:.4f}", flush=True)
                print("\a", end="", flush=True)
            time.sleep(0.1)
    except KeyboardInterrupt:
        print(f"\nStopped. {len(snaps)} snapshots in {out}")


if __name__ == "__main__":
    main()

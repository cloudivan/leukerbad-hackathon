"""Hand-guided calibration: you place the follower, the script records pose + all camera images.

Per point (default 8) two shots:
  A  FLOOR : closed jaw tips touching the table, ideally on a small marker (coin / tape cross).
  B  HOVER : lift the arm STRAIGHT UP about 4-5 cm above the same marker, jaws still vertical.
A gives the real floor z at that (x, y). A vs B gives the wrist-camera parallax
(where the marker moves in the picture when the tips are above it) and camera 0 pixel <-> xy.

    uv run python record_points.py          # 8 points
    uv run python record_points.py 12       # 12 points

Torque is RELEASED at the start: hold the arm before pressing Enter, it will drop.
At the end torque stays OFF (lay the arm down on the table yourself).
Output: logs/calib_<time>/points.json + p<N>_<floor|hover>_cam<i>.jpg, plus a floor-plane fit.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from ik_client import IKClient, IKError

HERE = Path(__file__).resolve().parent
load_dotenv(HERE / ".env")
IK_URL = os.environ.get("IK_URL", "http://127.0.0.1:8020").strip()
FOLLOWER_ID = os.environ.get("FOLLOWER_ID", "5B79015365").strip()   # ADJUST if the serial differs


def read_arm(ik: IKClient) -> dict:
    for attempt in range(5):
        try:
            return ik.follower(FOLLOWER_ID)
        except IKError as e:
            if e.code not in ("stale_frame", "no_frame") or attempt == 4:
                raise
            time.sleep(0.5)
    raise RuntimeError


def shot(ik: IKClient, cams: list[dict], out: Path, tag: str) -> dict:
    """Average 5 pose readings (hand jitter), save every camera image."""
    poses, arm = [], None
    for _ in range(5):
        arm = read_arm(ik)
        poses.append(arm["pose"]["position"])
        time.sleep(0.1)
    avg = {k: round(sum(p[k] for p in poses) / len(poses), 4) for k in ("x", "y", "z")}
    spread = round(max(p["z"] for p in poses) - min(p["z"] for p in poses), 4)
    images = []
    for i, cam in enumerate(cams):
        try:
            path = out / f"{tag}_cam{i}.jpg"
            path.write_bytes(ik.camera_image(cam["id"]))
            images.append(path.name)
        except IKError as e:
            print(f"   cam{i} failed: {e.code}")
    return {"pos": avg, "z_spread": spread, "rpy": arm["pose"]["rpy"],
            "joints": [round(j["angle_rad"], 4) for j in arm["joints"]], "images": images}


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    if not sys.stdin.isatty():
        print("Run this in a real terminal (it waits for Enter).")
        return 1
    ik = IKClient(IK_URL)
    arm = read_arm(ik)
    cams = ik.cameras()
    out = HERE / "logs" / f"calib_{time.strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True)
    print(f"Follower {arm['id']} torque={arm['torque_enabled']}. Cameras: {len(cams)}. Saving to {out}")
    for i, c in enumerate(cams):
        print(f"  cam{i} = {c['id']}")

    if arm["torque_enabled"]:
        input("\nHOLD THE ARM with one hand, then press Enter to release torque (it will go limp)...")
        ik._request("POST", f"/arms/{FOLLOWER_ID}/torque", json={"enabled": False})
        print("Torque OFF.")

    print("\nSpread the points: near/far (x 0.12..0.26) and left/right (y -0.08..0.08).")
    print("Keep the gripper pointing straight down. Keep your hand out of camera 0 if you can.\n")
    points = []
    for k in range(1, n + 1):
        input(f"[{k}/{n}] A FLOOR: jaw tips touching the table on the marker, then Enter ")
        a = shot(ik, cams, out, f"p{k}_floor")
        print(f"   floor  x {a['pos']['x']:.4f}  y {a['pos']['y']:.4f}  z {a['pos']['z']:.4f}  (jitter {a['z_spread']*1000:.1f} mm)")
        input(f"[{k}/{n}] B HOVER: lift straight up ~4-5 cm over the same marker, then Enter ")
        b = shot(ik, cams, out, f"p{k}_hover")
        d = {c: round(b["pos"][c] - a["pos"][c], 4) for c in ("x", "y", "z")}
        print(f"   hover  x {b['pos']['x']:.4f}  y {b['pos']['y']:.4f}  z {b['pos']['z']:.4f}  (dx {d['x']*1000:+.0f} dy {d['y']*1000:+.0f} dz {d['z']*1000:+.0f} mm)")
        points.append({"n": k, "floor": a, "hover": b})
        (out / "points.json").write_text(json.dumps({"cameras": cams, "points": points}, indent=2))

    # Floor plane z = a + b*x + c*y by least squares (3x3 normal equations).
    pts = [(p["floor"]["pos"]["x"], p["floor"]["pos"]["y"], p["floor"]["pos"]["z"]) for p in points]
    if len(pts) >= 3:
        S = [[0.0] * 3 for _ in range(3)]
        r = [0.0] * 3
        for x, y, z in pts:
            v = (1.0, x, y)
            for i in range(3):
                r[i] += v[i] * z
                for j in range(3):
                    S[i][j] += v[i] * v[j]
        # Gaussian elimination
        M = [S[i] + [r[i]] for i in range(3)]
        for i in range(3):
            piv = max(range(i, 3), key=lambda q: abs(M[q][i]))
            M[i], M[piv] = M[piv], M[i]
            for q in range(3):
                if q != i:
                    f = M[q][i] / M[i][i]
                    M[q] = [mq - f * mi for mq, mi in zip(M[q], M[i])]
        a0, bx, cy = (M[i][3] / M[i][i] for i in range(3))
        res = [z - (a0 + bx * x + cy * y) for x, y, z in pts]
        print(f"\nFloor plane: z = {a0:.4f} + {bx:.4f}*x + {cy:.4f}*y   "
              f"(residual max {max(abs(e) for e in res)*1000:.1f} mm)")
        data = json.loads((out / "points.json").read_text())
        data["floor_plane"] = {"a": a0, "bx": bx, "cy": cy, "residuals_mm": [round(e * 1000, 1) for e in res]}
        (out / "points.json").write_text(json.dumps(data, indent=2))

    print(f"\nDone. Torque is still OFF: lay the arm on the table. Data: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

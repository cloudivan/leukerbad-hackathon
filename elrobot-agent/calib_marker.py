"""Overview camera calibration with a RED MARKER object (no clicks, no hidden jaw tips).

The arm picks a small red object (bottle cap, red Lego brick), then for every grid point:
put it down -> open -> lift -> park out of view -> photo -> find the red blob (difference to a
background photo) -> go back -> pick it up again. Pixel of the blob -> commanded x, y -> homography.

    1. Put the red marker on the table at START (default x 0.17, y 0.00, in front of the base).
    2. uv run python calib_marker.py
E-stop: 12 V follower plug. Do not move the camera afterwards.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

import calib_overview as co
import safety
from ik_client import IKClient, IKError
from safety import Target

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------- ADJUST ----
START = (0.17, 0.00)              # where you put the marker before starting
MARKER_H = 0.015                  # m, marker height (bottle cap ~0.012-0.015)
GRID = [(0.13, [-0.10, 0.0, 0.10]),
        (0.17, [-0.12, 0.0, 0.12]),
        (0.21, [-0.12, 0.0, 0.12]),
        (0.25, [-0.10, 0.0, 0.10]),
        (0.28, [-0.06, 0.0, 0.06])]
HOVER_Z = 0.065
PARK = Target(0.10, -0.12, 0.080, 0.0)   # out of the camera's view (agent PARK_XY)
OPEN = 0.75                       # jaw opening for release / approach (1.0 can stall low on tilted jaws)
HELD_MIN = 0.05                   # gripper reading above this after closing = something is held
MIN_BLOB_PX = 60                  # smallest red blob accepted (pixels at 1280x960)
# ----------------------------------------------------------------------------


def retry(fn, *a, tries: int = 4, **kw):
    """Station sometimes reports stale motor state (bus hiccup, 503): wait and retry."""
    for k in range(tries):
        try:
            return fn(*a, **kw)
        except IKError as e:
            if e.code != "stale_frame" or k == tries - 1:
                raise
            time.sleep(0.6)


class Arm:
    def __init__(self) -> None:
        self.ik = IKClient(co.IK_URL)
        self.grip = 0.0

    def state(self) -> dict:
        return retry(self.ik.follower, co.FOLLOWER_ID)

    def steps(self) -> list[int]:
        return [j["steps"] for j in sorted(self.state()["joints"], key=lambda j: j["motor"])]

    def go(self, t: Target, max_jump: int | None = None, speed: int | None = None) -> None:
        plan = retry(self.ik.move_to, co.FOLLOWER_ID, t, execute=False, gripper=self.grip, near_steps=self.steps())
        if max_jump is not None and plan["max_joint_delta"] > max_jump:
            raise RuntimeError(f"joint jump {plan['max_joint_delta']} steps at {t}")
        try:
            retry(self.ik.move_to, co.FOLLOWER_ID, t, execute=True, gripper=self.grip, tilt=plan["tilt"], speed=speed)
        except IKError as e:
            if e.code != "collision" or max_jump is not None:
                raise
            p = self.state()["pose"]["position"]           # path hit the table: up, across high, down
            for via in (Target(p["x"], p["y"], co.SAFE_HIGH_Z, 0.0), Target(t.x, t.y, co.SAFE_HIGH_Z, 0.0), t):
                self.go(via)
        co.wait_still(self.ik)

    def jaws(self, opening: float, at: Target) -> float:
        self.grip = opening
        plan = retry(self.ik.move_to, co.FOLLOWER_ID, at, execute=False, gripper=opening, near_steps=self.steps())
        retry(self.ik.move_to, co.FOLLOWER_ID, at, execute=True, gripper=opening, tilt=plan["tilt"],
              speed=700, accel=30)
        time.sleep(0.6)
        g = float(self.state().get("gripper") or 0.0)
        if opening == 0.0 and g > HELD_MIN:
            self.grip = max(0.0, g - 0.04)                 # steady moderate squeeze, not full stall
        return g

    def down(self, x: float, y: float, z: float) -> None:
        n = max(1, math.ceil((HOVER_Z - z) / 0.010))
        for k in range(1, n + 1):
            self.go(Target(x, y, HOVER_Z - (HOVER_Z - z) * k / n, 0.0), max_jump=150, speed=120)

    def pick(self, x: float, y: float, z: float) -> bool:
        self.go(Target(x, y, HOVER_Z, 0.0))
        self.jaws(OPEN, Target(x, y, HOVER_Z, 0.0))
        self.down(x, y, z)
        held = self.jaws(0.0, Target(x, y, z, 0.0)) > HELD_MIN
        self.go(Target(x, y, HOVER_Z, 0.0))
        return held

    def place(self, x: float, y: float, z: float) -> None:
        self.go(Target(x, y, HOVER_Z, 0.0))
        self.down(x, y, z)
        self.jaws(OPEN, Target(x, y, z, 0.0))
        self.go(Target(x, y, HOVER_Z, 0.0))


def grip_z(x: float, y: float) -> float:
    return co.z_floor(x, y) + max(0.006, MARKER_H / 2) + (0.006 if x >= 0.24 else 0.0)


def red_mask(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (0, 110, 70), (10, 255, 255)) | cv2.inRange(hsv, (165, 110, 70), (180, 255, 255))
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def find_marker(img: np.ndarray, bg: np.ndarray) -> tuple[float, float, int] | None:
    """Red pixels that are new compared with the background photo; largest blob centroid."""
    new = red_mask(img) & cv2.bitwise_not(cv2.dilate(red_mask(bg), np.ones((9, 9), np.uint8)))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(new)
    if n <= 1:
        return None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[i, cv2.CC_STAT_AREA] < MIN_BLOB_PX:
        return None
    return float(cent[i][0]), float(cent[i][1]), int(stats[i, cv2.CC_STAT_AREA])


def photo(arm: Arm) -> np.ndarray:
    time.sleep(0.8)                                         # let the camera stream catch up
    jpg = arm.ik.camera_image(co.OVERVIEW_ID)
    return cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)


def main() -> int:
    arm = Arm()
    out = HERE / "logs" / f"calib_marker_{time.strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True)
    order = [(x, y) for i, (x, ys) in enumerate(GRID) for y in (ys if i % 2 == 0 else ys[::-1])]
    order = [p for p in order if all(
        _plans(arm, Target(p[0], p[1], z, 0.0)) for z in (HOVER_Z, grip_z(*p)))]
    print(f"{len(order)} points plan OK", flush=True)
    points, clicks = [], {}
    p0 = arm.state()["pose"]["position"]
    arm.go(Target(p0["x"], p0["y"], max(HOVER_Z, p0["z"]), 0.0))
    try:
        if not arm.pick(START[0], START[1], grip_z(*START)):
            print("MISSED the marker at START: put it there and run again", flush=True)
            return 1
        arm.go(PARK)
        bg = photo(arm)                                     # table without the marker, arm out of view
        cv2.imwrite(str(out / "background.jpg"), bg)
        for n, (x, y) in enumerate(order, 1):
            z = grip_z(x, y)
            arm.place(x, y, z)
            arm.go(PARK)
            img = photo(arm)
            cv2.imwrite(str(out / f"p{n}.jpg"), img)
            hit = find_marker(img, bg)
            if hit:
                clicks[str(n)] = [hit[0], hit[1]]
                points.append({"n": n, "cmd": [x, y, z], "image": f"p{n}.jpg"})
                print(f"point {n}/{len(order)}  ({x:.3f}, {y:+.3f})  marker at px ({hit[0]:.0f}, {hit[1]:.0f}), "
                      f"{hit[2]} px", flush=True)
            else:
                print(f"point {n}/{len(order)}  ({x:.3f}, {y:+.3f})  marker NOT FOUND", flush=True)
            if not arm.pick(x, y, z):
                print(f"MISSED the marker at ({x:.3f}, {y:+.3f}): stopping, fitting what we have", flush=True)
                break
    finally:
        try:
            arm.go(safety.REST_POSE)
            print("rest: reached", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"rest failed: {e}", flush=True)
        (out / "points.json").write_text(json.dumps(points, indent=1))
        (out / "clicks.json").write_text(json.dumps(clicks))
    if len(points) < 6:
        print(f"only {len(points)} points: not fitted", flush=True)
        return 1
    res = co.fit_and_save(out)
    res_note = "fitted on a red marker lying on the table (calib_marker.py), pixel -> commanded x, y"
    data = json.loads(co.OUT_FILE.read_text()); data["z_note"] = res_note; co.OUT_FILE.write_text(json.dumps(data, indent=1))
    H = np.array(res["H_pixel_to_xy"])
    for p in points:
        q = co.apply_h(H, *clicks[str(p["n"])])
        print(f"  {p['n']:2d} ({p['cmd'][0]:.3f}, {p['cmd'][1]:+.3f})  err {math.hypot(q[0] - p['cmd'][0], q[1] - p['cmd'][1]) * 1000:5.1f} mm")
    print(f"Saved {co.OUT_FILE}: {res['points']} points, RMS {res['rms_mm']} mm, max {res['max_mm']} mm", flush=True)
    return 0


def _plans(arm: Arm, t: Target) -> bool:
    try:
        retry(arm.ik.move_to, co.FOLLOWER_ID, t, execute=False, gripper=0.0)
        return True
    except IKError:
        return False


if __name__ == "__main__":
    sys.exit(main())

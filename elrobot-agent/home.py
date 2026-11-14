"""Operator-only: one direct move of the follower to safety.REST_POSE.

Why this exists: with torque off the limp arm rests in a pose that a vertical
gripper cannot reach in 5 cm Cartesian steps, so the agent's go_rest finds no
path. This script does ONE joint-space move to the rest pose. It ignores the
5 cm step limit on purpose, so it is NOT available to the LLM and always waits
for you at the keyboard.

    uv run python home.py             # solve only, shows what would happen
    uv run python home.py --execute   # asks for Enter, then moves

E-stop: pull the 12 V follower plug (the arm then drops, support it).
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

import safety
from ik_client import IKClient, IKError

HERE = Path(__file__).resolve().parent
load_dotenv(HERE / ".env")
IK_URL = os.environ.get("IK_URL", "http://127.0.0.1:8020").strip()
FOLLOWER_ID = os.environ.get("FOLLOWER_ID", "5B79015365").strip()

SETTLE_TIMEOUT_S = 25.0   # ADJUST: how long to watch the arm after the API stops waiting


def steps_of(arm: dict) -> list[int]:
    return [j["steps"] for j in sorted(arm["joints"], key=lambda j: j["motor"])]


def main() -> int:
    execute = "--execute" in sys.argv[1:]
    ik = IKClient(IK_URL)
    try:
        arm = ik.follower(FOLLOWER_ID)          # refuses anything that is not role=follower
        if not arm["calibrated"]:
            print("Follower is not calibrated. Calibrate by hand (SETUP.md section 3) and freeze the ranges.")
            return 1
        now = steps_of(arm)
        pos = arm["pose"]["position"]
        rest = safety.REST_POSE
        solved = ik.move_to(FOLLOWER_ID, rest, execute=False)   # solve + collision check, no motion
    except IKError as e:
        print(f"Cannot home: {e.short()}")
        return 1

    goal = solved["motor_steps"]
    deltas = [g - n for g, n in zip(goal[:7], now[:7])]
    biggest = max(abs(d) for d in deltas)
    if biggest <= 30:
        print("Already at the rest pose, nothing to do.")
        return 0
    print(f"Follower     : {arm['id']} role={arm['role']} torque={arm['torque_enabled']}")
    print(f"Tool point   : x={pos['x']:.3f} y={pos['y']:.3f} z={pos['z']:.3f}")
    print(f"Rest pose    : x={rest.x:.3f} y={rest.y:.3f} z={rest.z:.3f} yaw={rest.yaw:.2f} (vertical gripper)")
    print(f"Solve        : reachable={solved['reachable']}, no collision reported")
    print(f"Motor steps  : now  {now[:7]}")
    print(f"               goal {goal[:7]}")
    print(f"               move {deltas}   (gripper motor 8 is not touched)")
    print(f"Largest move : {biggest} steps, about {biggest / safety.SPEED:.0f} s at speed {safety.SPEED}, accel {safety.ACCEL}")

    if not execute:
        print("\nSolve only, nothing moved. Add --execute to move.")
        return 0
    if "--yes" in sys.argv[1:]:
        print("\n--yes given: moving without the Enter prompt.")
    else:
        if not sys.stdin.isatty():
            print("\nRefusing: --execute needs a real terminal so you can confirm with Enter (or pass --yes).")
            return 1
        print("\nThis is ONE large joint-space move. The arm swings up and folds over the table.")
        print("Checklist: base clamped, follower on 12 V, card on (Self-controlled), table and cables clear")
        print("of the arm, nobody's hands near it, your hand on the 12 V plug.")
        if input("Press Enter to move, or type anything else + Enter to abort: ").strip():
            print("Aborted, nothing moved.")
            return 1

    try:
        ik.follower(FOLLOWER_ID)                 # role check right before the motion
        done = ik.move_to(FOLLOWER_ID, rest, execute=True)
    except IKError as e:
        print(f"Move failed: {e.short()}")
        return 1
    status = done["motion"]["status"]
    print(f"motion.status = {status} after {done['motion']['duration_ms']} ms")

    if status in ("timeout", "unknown"):
        # The API stopped waiting (8 s limit). Do NOT re-send; just watch until the servos settle.
        print("Watching the arm until it settles (no new command is sent)...")
        last, still, t0 = None, 0, time.time()
        while time.time() - t0 < SETTLE_TIMEOUT_S and still < 4:
            time.sleep(0.5)
            try:
                cur = steps_of(ik.follower(FOLLOWER_ID))
            except IKError as e:
                print(f"  state read failed: {e.short()}")
                continue
            still = still + 1 if last and max(abs(a - b) for a, b in zip(cur, last)) <= 3 else 0
            last = cur
        print("settled" if still >= 4 else "still moving or unreadable after the watch window")

    try:
        arm = ik.follower(FOLLOWER_ID)
        p = arm["pose"]["position"]
        err = max(abs(g - n) for g, n in zip(goal[:7], steps_of(arm)[:7]))
        print(f"Final        : x={p['x']:.3f} y={p['y']:.3f} z={p['z']:.3f} torque={arm['torque_enabled']} "
              f"largest step error {err}")
    except IKError as e:
        print(f"Could not read the final state: {e.short()}")
    print("Arm motors 1..7 hold the rest pose under torque (the gripper motor was not commanded).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

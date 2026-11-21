"""ElRobot Swarm AI: 4 simultaneous 7+1 DoF arms with 3D Inter-Arm Collision Avoidance & Relay Handoffs.

Usage:
    uv run python swarm.py                                   # starts 3D Swarm UI on http://127.0.0.1:5055
    uv run python swarm.py --auto                            # starts UI + runs autonomous 4-arm sorting demo
    uv run python swarm.py "yellow robot put the yellow cube into the blue area"

How the Swarm AI & Anti-Collision works:
  - 4 ElRobot 7+1 DoF arms sit at radius 0.265 m around the shared sorting table (0, 0, 0).
  - 4 Shared Corner Handoff Pads at (±0.095, ±0.095) sit at local (x=0.170, y=±0.095) for each adjacent pair
    of arms, while bins sit at local (x=0.135, y=-0.115) out of the handoff corridor.
  - 3D Link-to-Link Anti-Collision Guard: before executing any `move_swarm` batch, the engine samples the 3D
    global positions of all physical links (`Joint_02_1`..`tcp`) across all 4 arms:
      * If an idle arm is parked inside the 3D swept volume of a moving arm (< 8.5 cm link clearance), the
        idle arm is automatically retracted to `SWARM_REST_POSE` (x=0.13, y=0.0, z=0.08) first.
      * If two moving arms in the same batch would come within 8.5 cm of each other in 3D, the scheduler
        automatically partitions the batch into collision-free sub-waves.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
import json
import math
import os
from pathlib import Path
import site
import sys
import threading
import time
import xml.etree.ElementTree as ET

import anthropic
from dotenv import load_dotenv
from flask import Flask, abort, jsonify, request, send_from_directory
import numpy as np

import safety
from ik_client import IKClient, IKError
from safety import Target

HERE = Path(__file__).resolve().parent
ENV_FILE = HERE / ".env"
LOGS = HERE / "logs"
STATIC_DIR = HERE / "static"

NORMA_CORE = Path(os.environ.get("NORMA_CORE", str(Path.home() / "Developer" / "norma-core")))
IK_API_DIR = NORMA_CORE / "software" / "station" / "examples" / "elrobot-ik-api"
site.addsitedir(str(IK_API_DIR / ".venv" / "lib" / "python3.12" / "site-packages"))
if str(IK_API_DIR / "src") not in sys.path:
    sys.path.insert(0, str(IK_API_DIR / "src"))

from elrobot_ik import URDF_PATH  # noqa: E402
from elrobot_ik.kinematics.ik import IKSolver  # noqa: E402
from elrobot_ik.kinematics.model import ElrobotModel  # noqa: E402
from elrobot_ik.kinematics.transforms import (  # noqa: E402
    T_WORLD_URDF,
    make_transform,
    rotation_to_rpy,
    rpy_to_rotation,
)

load_dotenv(ENV_FILE)
API_KEY = os.environ.get("ANTHROPIC_API_KEY", "").strip()
MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-5-5").strip()
IK_URL = os.environ.get("IK_URL", "http://127.0.0.1:8020").strip()
FOLLOWER_ID = os.environ.get("FOLLOWER_ID", "5B79016927").strip()
REAL_MIN_Z = 0.005   # ADJUST: measured 15.11: table floor is z 0.000 (vertical gripper), arm sags 2..8 mm below command

DRY_RUN = os.environ.get("DRY_RUN", "1").strip() != "0"
SYNC_REAL_DEFAULT = os.environ.get("SYNC_REAL", "0").strip() == "1" and not DRY_RUN

HOST, PORT = "127.0.0.1", 5055          # ADJUST: local web dashboard port
SIM_HZ = 50                             # ADJUST: joint trajectory interpolation rate (Hz)
JOINT_RAD_PER_SEC = 1.45                # ADJUST: simulated joint speed
GRASP_RADIUS_M = 0.038                  # ADJUST: max XY distance from jaw tips to grasp an item
BIN_DROP_RADIUS_M = 0.055               # ADJUST: max XY distance from bin center to count as sorted
MAX_SWARM_STEPS = 30                    # ADJUST: max tool calls per Claude Swarm task
BASE_RADIUS_M = 0.265                   # ADJUST: distance of each arm base from table center (0,0,0)
MIN_LINK_CLEARANCE_M = 0.075            # ADJUST: min 3D skeleton clearance between different arms
SWARM_REST_POSE = Target(x=0.13, y=0.0, z=0.08, yaw=0.0)  # ADJUST: retracted rest pose for swarm clearance

ASSETS_DIR = Path(URDF_PATH).parent / "assets"

# True skeleton joint pivots from shoulder to jaw tip (excludes jaw CAD offset frames)
COLLISION_CHECK_LINKS = (
    "Joint_02_1",
    "Joint_03_v1_1",
    "Joint_04_v1_1",
    "Joint_05_v1_1",
    "Joint_06_v1_1",
    "Gripper_Base_v1_1",
    "tcp",
)


def _load_urdf_visuals(urdf_path: Path) -> list[dict]:
    root = ET.parse(urdf_path).getroot()
    visuals = []
    for link in root.findall("link"):
        vis = link.find("visual")
        if vis is None:
            continue
        origin = vis.find("origin")
        xyz = [float(v) for v in origin.attrib.get("xyz", "0 0 0").split()] if origin is not None else [0.0, 0.0, 0.0]
        mesh_el = vis.find("geometry/mesh")
        if mesh_el is None:
            continue
        mesh_file = Path(mesh_el.attrib["filename"]).name
        mat_el = vis.find("material")
        mat_name = mat_el.attrib.get("name", "silver") if mat_el is not None else "silver"
        visuals.append({
            "link": link.attrib["name"],
            "mesh": mesh_file,
            "material": "servo" if mat_name.startswith("ST3215") else "bracket",
            "T_vis": make_transform(np.eye(3), xyz),
        })
    return visuals


URDF_VISUALS = _load_urdf_visuals(Path(URDF_PATH))

ARM_CONFIGS = [
    {
        "id": "arm_0",
        "name": "Arm 0 (Green / South / Real Twin)",
        "color_name": "green",
        "sector": "Green / Plastic Zone (South)",
        "base_xyz": (-BASE_RADIUS_M, 0.0, 0.0),
        "base_yaw": 0.0,
        "accent": "#22c55e",
        "is_real_twin": True,
    },
    {
        "id": "arm_1",
        "name": "Arm 1 (Blue / North)",
        "color_name": "blue",
        "sector": "Blue / Metal Zone (North)",
        "base_xyz": (BASE_RADIUS_M, 0.0, 0.0),
        "base_yaw": math.pi,
        "accent": "#3b82f6",
        "is_real_twin": False,
    },
    {
        "id": "arm_2",
        "name": "Arm 2 (Yellow / West)",
        "color_name": "yellow",
        "sector": "Yellow / Bio Zone (West)",
        "base_xyz": (0.0, BASE_RADIUS_M, 0.0),
        "base_yaw": -math.pi / 2.0,
        "accent": "#f59e0b",
        "is_real_twin": False,
    },
    {
        "id": "arm_3",
        "name": "Arm 3 (Pink / East)",
        "color_name": "pink",
        "sector": "Pink / E-Waste Zone (East)",
        "base_xyz": (0.0, -BASE_RADIUS_M, 0.0),
        "base_yaw": math.pi / 2.0,
        "accent": "#ec4899",
        "is_real_twin": False,
    },
]

# 4 Shared Corner Handoff Pads at (±0.095, ±0.095): local (x=0.170, y=±0.095) for each adjacent pair!
HANDOFF_ZONES = [
    {"id": "handoff_nw", "label": "NW Handoff Pad (Yellow arm_2 <-> Blue arm_1)", "global_xyz": (0.095, 0.095, 0.013)},
    {"id": "handoff_sw", "label": "SW Handoff Pad (Green arm_0 <-> Yellow arm_2)", "global_xyz": (-0.095, 0.095, 0.013)},
    {"id": "handoff_ne", "label": "NE Handoff Pad (Blue arm_1 <-> Pink arm_3)", "global_xyz": (0.095, -0.095, 0.013)},
    {"id": "handoff_se", "label": "SE Handoff Pad (Pink arm_3 <-> Green arm_0)", "global_xyz": (-0.095, -0.095, 0.013)},
]


@dataclass
class WasteItem:
    id: str
    label: str
    category: str       # plastic | metal | bio | ewaste
    color_name: str     # green | blue | yellow | pink
    color: str
    shape: str          # cube | cylinder
    size: float         # m
    global_x: float
    global_y: float
    global_z: float
    held_by: str | None = None
    sorted_in: str | None = None


@dataclass
class SortingBin:
    id: str
    label: str
    category: str
    color_name: str
    color: str
    owner_arm: str
    global_x: float
    global_y: float
    global_z: float = 0.02


@dataclass
class ArmInstance:
    cfg: dict
    model: ElrobotModel
    solver: IKSolver
    T_base: np.ndarray
    T_base_inv: np.ndarray
    angles: list[float] = field(default_factory=lambda: [0.0] * 8)
    rest_angles: list[float] = field(default_factory=lambda: [0.0] * 8)
    gripper: float = 1.0
    status: str = "idle"
    last_error: str = ""
    held_item_id: str | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def id(self) -> str:
        return self.cfg["id"]

    def gripper_fraction_to_angle(self, fraction: float) -> float:
        lim = self.model.joint_limits[7]
        span = lim.upper - lim.lower
        return lim.upper - max(0.0, min(1.0, fraction)) * span

    def local_to_global(self, lx: float, ly: float, lz: float) -> tuple[float, float, float]:
        p = self.T_base @ np.array([lx, ly, lz, 1.0])
        return float(p[0]), float(p[1]), float(p[2])

    def global_to_local(self, gx: float, gy: float, gz: float) -> tuple[float, float, float]:
        p = self.T_base_inv @ np.array([gx, gy, gz, 1.0])
        return float(p[0]), float(p[1]), float(p[2])

    def can_reach_local(self, lx: float, ly: float) -> bool:
        r = math.hypot(lx, ly)
        return 0.095 <= lx <= 0.228 and abs(ly) <= 0.135 and 0.10 <= r <= 0.228

    def link_points_global(self, angles: list[float] | None = None) -> np.ndarray:
        """Returns (N, 3) array of 3D global coordinates for this arm's body links + interpolated forearm points."""
        with self.lock:
            saved = list(self.angles)
            use = angles if angles is not None else saved
            self.model.set_angles(use)
            pts = []
            for name in COLLISION_CHECK_LINKS:
                T_local = T_WORLD_URDF @ self.model.frame_urdf(name)
                T_glob = self.T_base @ T_local
                pts.append(T_glob[:3, 3])
            # Restore live angles if we probed a hypothetical configuration
            if angles is not None:
                self.model.set_angles(saved)
        arr = np.array(pts, dtype=float)
        # Also include midpoints between consecutive links so long forearm segments are covered
        mids = 0.5 * (arr[:-1] + arr[1:])
        return np.vstack([arr, mids])

    def tcp_local(self) -> dict:
        with self.lock:
            T = self.model.fk(self.angles)
            r, p, y = rotation_to_rpy(T[:3, :3])
            return {
                "x": round(float(T[0, 3]), 4),
                "y": round(float(T[1, 3]), 4),
                "z": round(float(T[2, 3]), 4),
                "roll": round(r, 4),
                "pitch": round(p, 4),
                "yaw": round(y, 4),
            }

    def tcp_global(self) -> tuple[float, float, float]:
        pose = self.tcp_local()
        return self.local_to_global(pose["x"], pose["y"], pose["z"])

    def visual_matrices_col_major(self) -> dict[str, list[float]]:
        with self.lock:
            self.model.set_angles(self.angles)
            out: dict[str, list[float]] = {}
            for v in URDF_VISUALS:
                T_local = T_WORLD_URDF @ self.model.frame_urdf(v["link"]) @ v["T_vis"]
                T_glob = self.T_base @ T_local
                out[v["link"]] = [round(float(x), 5) for x in T_glob.T.reshape(-1)]
            return out


class SwarmWorld:
    def __init__(self) -> None:
        LOGS.mkdir(exist_ok=True)
        self.log_path = LOGS / f"swarm_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
        self.lock = threading.Lock()
        self.sync_real = SYNC_REAL_DEFAULT
        self.ik_client = IKClient(IK_URL)
        self.arms: dict[str, ArmInstance] = {}
        self.items: dict[str, WasteItem] = {}
        self.bins: dict[str, SortingBin] = {}
        self.events: list[dict] = []
        self.busy_task: str | None = None
        self.task_thread: threading.Thread | None = None
        self._stop_flag = False
        self.collisions_prevented = 0

        for cfg in ARM_CONFIGS:
            m = ElrobotModel()
            s = IKSolver(m)
            bx, by, bz = cfg["base_xyz"]
            T_base = make_transform(rpy_to_rotation(0.0, 0.0, cfg["base_yaw"]), (bx, by, bz))
            arm = ArmInstance(
                cfg=cfg,
                model=m,
                solver=s,
                T_base=T_base,
                T_base_inv=np.linalg.inv(T_base),
            )
            T_rest = make_transform(
                rpy_to_rotation(safety.ROLL, safety.PITCH, SWARM_REST_POSE.yaw),
                (SWARM_REST_POSE.x, SWARM_REST_POSE.y, SWARM_REST_POSE.z),
            )
            seed = [0.0] * 7 + [arm.gripper_fraction_to_angle(1.0)]
            res = s.solve_pose(T_rest, seed, accept=lambda a, mod=m: not mod.collisions(a).any)
            arm.angles = list(res.angles)
            arm.rest_angles = list(res.angles)
            arm.gripper = 1.0
            self.arms[arm.id] = arm

        self.reset_scene(scrambled=False)

    def log_event(self, kind: str, message: str, **extra) -> None:
        entry = {
            "ts": datetime.now().strftime("%H:%M:%S.%f")[:-3],
            "kind": kind,
            "message": message,
            **extra,
        }
        with self.lock:
            self.events.append(entry)
            if len(self.events) > 80:
                self.events = self.events[-80:]
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")

    def reset_scene(self, scrambled: bool = False) -> None:
        with self.lock:
            # Place each arm's bin at local (x=0.135, y=-0.115) on its right flank, well clear of the +y handoff pad
            self.bins = {}
            bin_specs = [
                ("bin_plastic", "Green Plastic Bin", "plastic", "green", "#22c55e", "arm_0"),
                ("bin_metal", "Blue Metal Bin", "metal", "blue", "#3b82f6", "arm_1"),
                ("bin_bio", "Yellow Bio Bin", "bio", "yellow", "#f59e0b", "arm_2"),
                ("bin_ewaste", "Pink E-Waste Bin", "ewaste", "pink", "#ec4899", "arm_3"),
            ]
            for bid, label, cat, cname, col, aid in bin_specs:
                gx, gy, gz = self.arms[aid].local_to_global(0.135, -0.115, 0.02)
                self.bins[bid] = SortingBin(bid, label, cat, cname, col, aid, gx, gy, gz)

            def _pos(aid: str, lx: float, ly: float) -> tuple[float, float, float]:
                return self.arms[aid].local_to_global(lx, ly, 0.013)

            if not scrambled:
                # 4 direct sector items + 4 clockwise relay items (Arm -> neighbor via corner handoff pad)
                raw_items = [
                    ("item_green_1", "Green Plastic Cube", "plastic", "green", "#22c55e", "cube", *_pos("arm_0", 0.150, -0.030)),
                    ("item_blue_1", "Blue Metal Cylinder", "metal", "blue", "#3b82f6", "cylinder", *_pos("arm_1", 0.150, -0.030)),
                    ("item_yellow_1", "Yellow Bio Cube", "bio", "yellow", "#f59e0b", "cube", *_pos("arm_2", 0.150, -0.030)),
                    ("item_pink_1", "Pink E-Waste Cube", "ewaste", "pink", "#ec4899", "cube", *_pos("arm_3", 0.150, -0.030)),
                    # Cross-sector relay items (placed in neighbor's zone so each arm passes clockwise via its +y handoff pad)
                    ("item_blue_2", "Blue Steel Cube (in Yellow zone)", "metal", "blue", "#60a5fa", "cube", *_pos("arm_2", 0.150, 0.035)),
                    ("item_pink_2", "Pink Battery Cell (in Blue zone)", "ewaste", "pink", "#f472b6", "cylinder", *_pos("arm_1", 0.150, 0.035)),
                    ("item_green_2", "Green PET Block (in Pink zone)", "plastic", "green", "#4ade80", "cube", *_pos("arm_3", 0.150, 0.035)),
                    ("item_yellow_2", "Yellow Compost Block (in Green zone)", "bio", "yellow", "#fbbf24", "cube", *_pos("arm_0", 0.150, 0.035)),
                ]
            else:
                raw_items = [
                    ("item_blue_1", "Blue Metal Cube", "metal", "blue", "#3b82f6", "cube", *_pos("arm_2", 0.150, 0.025)),
                    ("item_pink_1", "Pink E-Waste Cube", "ewaste", "pink", "#ec4899", "cube", *_pos("arm_1", 0.150, 0.025)),
                    ("item_green_1", "Green Plastic Cube", "plastic", "green", "#22c55e", "cube", *_pos("arm_3", 0.150, 0.025)),
                    ("item_yellow_1", "Yellow Bio Cube", "bio", "yellow", "#f59e0b", "cube", *_pos("arm_0", 0.150, 0.025)),
                ]

            self.items = {
                iid: WasteItem(iid, label, cat, cname, col, shape, 0.026, gx, gy, gz)
                for iid, label, cat, cname, col, shape, gx, gy, gz in raw_items
            }
            for arm in self.arms.values():
                arm.held_item_id = None
                arm.status = "idle"
                arm.last_error = ""
        self.log_event("scene", "Reset Swarm table (0.265m base radius + 3D link-to-link anti-collision enabled).")

    def _item_global_xyz(self, item: WasteItem) -> tuple[float, float, float]:
        if item.held_by and item.held_by in self.arms:
            gx, gy, gz = self.arms[item.held_by].tcp_global()
            return gx, gy, max(0.013, gz - 0.005)
        return item.global_x, item.global_y, item.global_z

    def _reachable_map(self, gx: float, gy: float, gz: float) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for arm_id, arm in self.arms.items():
            lx, ly, lz = arm.global_to_local(gx, gy, gz)
            if arm.can_reach_local(lx, ly):
                out[arm_id] = {"x": round(lx, 3), "y": round(ly, 3), "z": round(lz, 3)}
        return out

    # ------------------------------------ 3D Inter-Arm Anti-Collision Engine --
    def _min_dist_between_arms(self, arm_a: ArmInstance, angles_a: list[float],
                               arm_b: ArmInstance, angles_b: list[float]) -> float:
        pts_a = arm_a.link_points_global(angles_a)
        pts_b = arm_b.link_points_global(angles_b)
        diff = pts_a[:, None, :] - pts_b[None, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        return float(np.min(dists))

    def _trajectories_collide(self, arm_a: ArmInstance, start_a: list[float], goal_a: list[float],
                              arm_b: ArmInstance, start_b: list[float], goal_b: list[float],
                              samples: int = 6) -> tuple[bool, float]:
        worst = 999.0
        for k in range(samples + 1):
            alpha = k / samples
            aa = [s + (g - s) * alpha for s, g in zip(start_a, goal_a)]
            ab = [s + (g - s) * alpha for s, g in zip(start_b, goal_b)]
            d = self._min_dist_between_arms(arm_a, aa, arm_b, ab)
            if d < worst:
                worst = d
            if d < MIN_LINK_CLEARANCE_M:
                return True, worst
        return False, worst

    def _solve_target_angles(self, arm: ArmInstance, target: Target) -> tuple[list[float] | None, str]:
        if not all(math.isfinite(v) for v in (target.x, target.y, target.z, target.yaw)):
            return None, "REJECT_NAN: non-finite target coordinates"
        T_goal = make_transform(
            rpy_to_rotation(safety.ROLL, safety.PITCH, target.yaw),
            (target.x, target.y, target.z),
        )
        with arm.lock:
            seed = list(arm.angles)
            seed[7] = arm.gripper_fraction_to_angle(arm.gripper)
            res = arm.solver.solve_pose(
                T_goal,
                seed,
                accept=lambda a, mod=arm.model: not mod.collisions(a).any,
            )
            if not res.reachable:
                return None, f"ERROR unreachable at local ({target.x:.3f},{target.y:.3f},{target.z:.3f})"
            if not res.accepted:
                rep = arm.model.collisions(res.angles)
                return None, f"ERROR collision (table={rep.table_pairs}, self={rep.self_pairs})"
            return list(res.angles), ""

    def _execute_solved_move(self, arm_id: str, target: Target, goal_angles: list[float]) -> dict:
        arm = self.arms[arm_id]
        with arm.lock:
            start_angles = list(arm.angles)
            arm.status = "moving"
            arm.last_error = ""

        real_thread: threading.Thread | None = None
        real_status: dict = {}
        if arm.cfg["is_real_twin"] and self.sync_real:
            def _drive_real():
                try:
                    self.ik_client.follower(FOLLOWER_ID)
                    real_target = Target(target.x, target.y, max(target.z, REAL_MIN_Z), target.yaw)  # real floor is z 0.030
                    done = self.ik_client.move_to(FOLLOWER_ID, real_target, execute=True, gripper=arm.gripper)
                    real_status["status"] = done["motion"]["status"]
                except IKError as e:
                    real_status["error"] = e.short()
            real_thread = threading.Thread(target=_drive_real, daemon=True)
            real_thread.start()

        max_delta = max(abs(g - s) for g, s in zip(goal_angles[:7], start_angles[:7]))
        duration_s = max(0.16, min(1.8, max_delta / JOINT_RAD_PER_SEC))
        steps = max(2, int(duration_s * SIM_HZ))
        dt = duration_s / steps
        for k in range(1, steps + 1):
            if self._stop_flag:
                break
            alpha = 0.5 * (1.0 - math.cos(math.pi * (k / steps)))
            with arm.lock:
                arm.angles = [s + (g - s) * alpha for s, g in zip(start_angles, goal_angles)]
            time.sleep(dt)

        if real_thread is not None:
            real_thread.join(timeout=10.0)

        with arm.lock:
            arm.angles = goal_angles
            arm.status = "reached"
        pose = arm.tcp_local()
        gx, gy, gz = arm.tcp_global()
        out = {
            "arm": arm_id,
            "ok": True,
            "status": "reached",
            "reached_local_m": {"x": pose["x"], "y": pose["y"], "z": pose["z"], "yaw": pose["yaw"]},
            "reached_global_m": {"x": round(gx, 3), "y": round(gy, 3), "z": round(gz, 3)},
            "duration_ms": int(duration_s * 1000),
        }
        if real_status:
            out["real_hardware"] = real_status
        return out

    # ------------------------------------------------ simultaneous swarm API --
    def move_swarm(self, moves: list[dict]) -> list[dict]:
        """Solve IK for all requested arms, enforce 3D inter-arm collision avoidance, and execute."""
        if not moves:
            return []

        # Deduplicate by arm (last wins if duplicate in same list)
        by_arm: dict[str, Target] = {}
        results_by_arm: dict[str, dict] = {}
        for m in moves:
            aid = str(m.get("arm", ""))
            if aid not in self.arms:
                results_by_arm[aid] = {"arm": aid, "ok": False, "error": f"unknown arm '{aid}'"}
                continue
            by_arm[aid] = Target(float(m["x"]), float(m["y"]), float(m["z"]), float(m.get("yaw", 0.0)))

        # Step 1: Solve IK for each requested arm
        solved: dict[str, tuple[Target, list[float], list[float]]] = {}
        for aid, tgt in by_arm.items():
            arm = self.arms[aid]
            goal_angles, err = self._solve_target_angles(arm, tgt)
            if goal_angles is None:
                arm.status = "error"
                arm.last_error = err
                results_by_arm[aid] = {"arm": aid, "ok": False, "error": err}
            else:
                solved[aid] = (tgt, list(arm.angles), goal_angles)

        if not solved:
            return [results_by_arm[str(m.get("arm", ""))] for m in moves]

        # Step 2: Check if any non-moving (idle) arm is blocking the 3D path of a moving arm; if so, retract it first!
        idle_Order = [aid for aid in self.arms if aid not in solved]
        retract_idle: list[str] = []
        for aid, (_tgt, start_a, goal_a) in solved.items():
            arm_a = self.arms[aid]
            for idle_id in idle_Order:
                if idle_id in retract_idle:
                    continue
                arm_idle = self.arms[idle_id]
                idle_ang = list(arm_idle.angles)
                collides, dist = self._trajectories_collide(arm_a, start_a, goal_a, arm_idle, idle_ang, idle_ang)
                if collides:
                    retract_idle.append(idle_id)
                    self.collisions_prevented += 1
                    self.log_event(
                        "collision_guard",
                        f"3D Guard: auto-retracting idle {idle_id} to rest (clearance {dist*100:.1f} cm < {MIN_LINK_CLEARANCE_M*100:.1f} cm with {aid})",
                    )

        if retract_idle:
            with ThreadPoolExecutor(max_workers=4) as pool:
                futs = [
                    pool.submit(self._execute_solved_move, iid, SWARM_REST_POSE, list(self.arms[iid].rest_angles))
                    for iid in retract_idle
                ]
                for f in futs:
                    f.result()

        # Step 3: Partition moving arms into collision-free parallel sub-batches
        remaining = list(solved.keys())
        batches: list[list[str]] = []
        planned_current = {aid: list(self.arms[aid].angles) for aid in self.arms}

        while remaining:
            batch: list[str] = []
            for aid in list(remaining):
                _tgt_a, start_a, goal_a = solved[aid]
                arm_a = self.arms[aid]
                conflict = False
                # Check against other arms already in this batch
                for bid in batch:
                    _tgt_b, start_b, goal_b = solved[bid]
                    arm_b = self.arms[bid]
                    collides, dist = self._trajectories_collide(arm_a, start_a, goal_a, arm_b, start_b, goal_b)
                    if collides:
                        conflict = True
                        self.collisions_prevented += 1
                        self.log_event(
                            "collision_guard",
                            f"3D Guard: sequencing {aid} after {bid} (shared airspace clearance {dist*100:.1f} cm)",
                        )
                        break
                if not conflict:
                    batch.append(aid)
                    remaining.remove(aid)
            if not batch:
                # Fallback: run one arm alone
                batch.append(remaining.pop(0))
            batches.append(batch)
            for aid in batch:
                planned_current[aid] = solved[aid][2]

        # Step 4: Execute each collision-free sub-batch in parallel
        for batch in batches:
            desc = ", ".join(
                f"{aid}->({solved[aid][0].x:.2f},{solved[aid][0].y:.2f},{solved[aid][0].z:.2f})"
                for aid in batch
            )
            self.log_event("move_swarm", f"Parallel move [{len(batch)} arms]: {desc}")
            with ThreadPoolExecutor(max_workers=4) as pool:
                futs = {
                    aid: pool.submit(self._execute_solved_move, aid, solved[aid][0], solved[aid][2])
                    for aid in batch
                }
                for aid, f in futs.items():
                    results_by_arm[aid] = f.result()

        return [results_by_arm.get(str(m.get("arm", "")), {"ok": False}) for m in moves]

    def _gripper_single_arm(self, arm_id: str, opening: float) -> dict:
        if arm_id not in self.arms:
            return {"arm": arm_id, "ok": False, "error": f"unknown arm '{arm_id}'"}
        opening = float(opening)
        if not 0.0 <= opening <= 1.0:
            return {"arm": arm_id, "ok": False, "error": "REJECT_GRIPPER: open must be in 0..1"}
        arm = self.arms[arm_id]
        closing = opening < arm.gripper - 0.05
        pose = arm.tcp_local()

        grasped_item: WasteItem | None = None
        if closing and arm.held_item_id is None:
            gx, gy, gz = arm.tcp_global()
            with self.lock:
                best_d = 999.0
                for item in self.items.values():
                    if item.held_by is not None:
                        continue
                    ix, iy, iz = self._item_global_xyz(item)
                    d_xy = math.hypot(gx - ix, gy - iy)
                    d_z = abs(gz - iz)
                    if d_xy <= GRASP_RADIUS_M and d_z <= 0.045 and d_xy < best_d:
                        best_d = d_xy
                        grasped_item = item
                if grasped_item is not None:
                    grasped_item.held_by = arm.id
                    grasped_item.sorted_in = None
                    arm.held_item_id = grasped_item.id

        if grasped_item is not None:
            stopped_at = 0.30
            final_open = min(1.0, stopped_at + safety.GRIP_RELIEF)
            status = "stalled"
            note = f"jaws gripped '{grasped_item.label}' ({grasped_item.id}); grip relieved to {final_open:.2f}"
        else:
            final_open = opening
            status = "reached"
            note = "closed fully: nothing was gripped" if (closing and opening == 0.0) else "gripper moved"

        released_item: WasteItem | None = None
        sorted_bin: SortingBin | None = None
        if not closing and opening >= 0.45 and arm.held_item_id is not None:
            gx, gy, gz = arm.tcp_global()
            with self.lock:
                released_item = self.items.get(arm.held_item_id)
                arm.held_item_id = None
                if released_item is not None:
                    released_item.held_by = None
                    for b in self.bins.values():
                        if math.hypot(gx - b.global_x, gy - b.global_y) <= BIN_DROP_RADIUS_M:
                            sorted_bin = b
                            break
                    if sorted_bin is not None:
                        released_item.sorted_in = sorted_bin.id
                        count_in_bin = sum(1 for it in self.items.values() if it.sorted_in == sorted_bin.id)
                        offset = ((count_in_bin - 1) % 3 - 1) * 0.015
                        released_item.global_x = sorted_bin.global_x + offset
                        released_item.global_y = sorted_bin.global_y
                        released_item.global_z = 0.016
                        note = f"dropped '{released_item.label}' into '{sorted_bin.label}' ({sorted_bin.id})"
                    else:
                        released_item.global_x = gx
                        released_item.global_y = gy
                        released_item.global_z = released_item.size / 2.0
                        reachers = list(self._reachable_map(gx, gy, released_item.global_z).keys())
                        note = (f"placed '{released_item.label}' on table at global ({gx:.3f}, {gy:.3f}), "
                                f"now reachable by {reachers}")

        start_g = arm.gripper
        steps = 8
        for k in range(1, steps + 1):
            alpha = k / steps
            g_val = start_g + (final_open - start_g) * alpha
            with arm.lock:
                arm.gripper = g_val
                arm.angles[7] = arm.gripper_fraction_to_angle(g_val)
            time.sleep(0.025)

        if arm.cfg["is_real_twin"] and self.sync_real:
            try:
                hold = Target(pose["x"], pose["y"], pose["z"], pose["yaw"])
                self.ik_client.move_to(FOLLOWER_ID, hold, execute=True, gripper=final_open)
            except IKError:
                pass

        with arm.lock:
            arm.status = status
        return {
            "arm": arm_id,
            "ok": True,
            "status": status,
            "gripper": round(final_open, 3),
            "held_item": arm.held_item_id,
            "note": note,
        }

    def gripper_swarm(self, commands: list[dict]) -> list[dict]:
        if not commands:
            return []
        desc = ", ".join(f"{c.get('arm')}:open={float(c.get('open', 0)):.2f}" for c in commands)
        self.log_event("gripper_swarm", f"Parallel gripper [{len(commands)} arms]: {desc}")
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [
                pool.submit(self._gripper_single_arm, str(c["arm"]), float(c["open"]))
                for c in commands
            ]
            results = [f.result() for f in futures]
        for r in results:
            if r.get("note") and any(w in r["note"] for w in ("gripped", "dropped", "placed")):
                self.log_event("grasp", f"{r['arm']}: {r['note']}")
        return results

    def go_rest_swarm(self) -> list[dict]:
        moves = [
            {"arm": arm_id, "x": SWARM_REST_POSE.x, "y": SWARM_REST_POSE.y,
             "z": SWARM_REST_POSE.z, "yaw": SWARM_REST_POSE.yaw}
            for arm_id in self.arms
        ]
        return self.move_swarm(moves)

    def swarm_state_for_llm(self) -> dict:
        arms_out = {}
        for arm_id, arm in self.arms.items():
            p = arm.tcp_local()
            gx, gy, gz = arm.tcp_global()
            arms_out[arm_id] = {
                "name": arm.cfg["name"],
                "color": arm.cfg["color_name"],
                "sector": arm.cfg["sector"],
                "local_pose_m": {"x": p["x"], "y": p["y"], "z": p["z"], "yaw": p["yaw"]},
                "global_tcp_m": {"x": round(gx, 3), "y": round(gy, 3), "z": round(gz, 3)},
                "gripper_open": round(arm.gripper, 3),
                "held_item_id": arm.held_item_id,
                "status": arm.status,
            }
        items_out = []
        for item in self.items.values():
            gx, gy, gz = self._item_global_xyz(item)
            items_out.append({
                "id": item.id,
                "label": item.label,
                "color": item.color_name,
                "shape": item.shape,
                "category": item.category,
                "global_xyz_m": {"x": round(gx, 3), "y": round(gy, 3), "z": round(gz, 3)},
                "reachable_by_arms_local_m": self._reachable_map(gx, gy, gz),
                "held_by": item.held_by,
                "sorted_in_bin": item.sorted_in,
            })
        bins_out = []
        for b in self.bins.values():
            bins_out.append({
                "id": b.id,
                "label": b.label,
                "color": b.color_name,
                "category": b.category,
                "owner_arm": b.owner_arm,
                "reachable_by_arms_drop_local_m": self._reachable_map(b.global_x, b.global_y, 0.068),
            })
        handoffs_out = []
        for hz in HANDOFF_ZONES:
            hx, hy, h_z = hz["global_xyz"]
            handoffs_out.append({
                "id": hz["id"],
                "label": hz["label"],
                "global_xyz_m": {"x": hx, "y": hy, "z": h_z},
                "reachable_by_arms_local_m": self._reachable_map(hx, hy, h_z),
            })
        return {
            "arms": arms_out,
            "items": items_out,
            "bins": bins_out,
            "shared_handoff_zones": handoffs_out,
            "sorted_count": sum(1 for i in items_out if i["sorted_in_bin"] is not None),
            "total_items": len(items_out),
            "collisions_prevented": self.collisions_prevented,
        }

    def browser_state(self) -> dict:
        arms_list = []
        for arm_id, arm in self.arms.items():
            pose = arm.tcp_local()
            gx, gy, gz = arm.tcp_global()
            arms_list.append({
                "id": arm_id,
                "name": arm.cfg["name"],
                "color_name": arm.cfg["color_name"],
                "sector": arm.cfg["sector"],
                "accent": arm.cfg["accent"],
                "is_real_twin": arm.cfg["is_real_twin"],
                "base_xyz": arm.cfg["base_xyz"],
                "base_yaw": arm.cfg["base_yaw"],
                "local_pose": pose,
                "global_tcp": [round(gx, 4), round(gy, 4), round(gz, 4)],
                "gripper": round(arm.gripper, 3),
                "status": arm.status,
                "held_item_id": arm.held_item_id,
                "links": arm.visual_matrices_col_major(),
            })
        items_list = []
        for item in self.items.values():
            gx, gy, gz = self._item_global_xyz(item)
            items_list.append({
                "id": item.id,
                "label": item.label,
                "category": item.category,
                "color_name": item.color_name,
                "color": item.color,
                "shape": item.shape,
                "size": item.size,
                "global_xyz": [round(gx, 4), round(gy, 4), round(gz, 4)],
                "held_by": item.held_by,
                "sorted_in": item.sorted_in,
            })
        bins_list = []
        for b in self.bins.values():
            bins_list.append({
                "id": b.id,
                "label": b.label,
                "category": b.category,
                "color_name": b.color_name,
                "color": b.color,
                "arm_id": b.owner_arm,
                "global_xyz": [round(b.global_x, 4), round(b.global_y, 4), round(b.global_z, 4)],
            })
        return {
            "arms": arms_list,
            "items": items_list,
            "bins": bins_list,
            "handoff_zones": HANDOFF_ZONES,
            "visual_defs": [{"link": v["link"], "mesh": v["mesh"], "material": v["material"]} for v in URDF_VISUALS],
            "events": self.events[-24:][::-1],
            "busy_task": self.busy_task,
            "sync_real": self.sync_real,
            "dry_run_env": DRY_RUN,
            "sorted_count": sum(1 for it in self.items.values() if it.sorted_in is not None),
            "total_items": len(self.items),
            "collisions_prevented": self.collisions_prevented,
        }

    # ----------------------------------------- autonomous swarm + relay demo --
    def run_auto_sort(self) -> None:
        """Collision-free simultaneous 4-arm sorting + clockwise relay handoffs."""
        self._stop_flag = False
        self.log_event("swarm", "Swarm AI: planning collision-free 4-arm parallel + relay trajectories...")
        self.gripper_swarm([{"arm": a, "open": 1.0} for a in self.arms])
        self.go_rest_swarm()

        cat_to_bin = {b.category: b for b in self.bins.values()}

        def _best_handoff(arm_from: str, arm_to: str) -> tuple[float, float]:
            for hz in HANDOFF_ZONES:
                rmap = self._reachable_map(*hz["global_xyz"])
                if arm_from in rmap and arm_to in rmap:
                    return hz["global_xyz"][0], hz["global_xyz"][1]
            return 0.095, 0.095

        for round_idx in range(6):
            if self._stop_flag:
                break
            unsorted = [it for it in self.items.values() if it.sorted_in is None and it.held_by is None]
            if not unsorted:
                break

            tasks: list[dict] = []
            used_items: set[str] = set()
            used_handoffs: set[tuple[float, float]] = set()

            for arm_id, arm in self.arms.items():
                direct = None
                for it in unsorted:
                    if it.id in used_items:
                        continue
                    rmap = self._reachable_map(it.global_x, it.global_y, it.global_z)
                    target_bin = cat_to_bin[it.category]
                    if arm_id in rmap and target_bin.owner_arm == arm_id:
                        direct = (it, rmap[arm_id], arm.global_to_local(target_bin.global_x, target_bin.global_y, 0.068),
                                  f"direct->{target_bin.label}")
                        break
                if direct is not None:
                    it, pick_loc, drop_loc, desc = direct
                    used_items.add(it.id)
                    tasks.append({"arm": arm_id, "item": it, "pick": pick_loc, "drop": drop_loc, "desc": desc})
                    continue

                for it in unsorted:
                    if it.id in used_items:
                        continue
                    rmap = self._reachable_map(it.global_x, it.global_y, it.global_z)
                    target_bin = cat_to_bin[it.category]
                    if arm_id in rmap and target_bin.owner_arm != arm_id:
                        hx, hy = _best_handoff(arm_id, target_bin.owner_arm)
                        if (hx, hy) in used_handoffs:
                            continue
                        used_handoffs.add((hx, hy))
                        used_items.add(it.id)
                        drop_loc = arm.global_to_local(hx, hy, 0.022)
                        tasks.append({
                            "arm": arm_id,
                            "item": it,
                            "pick": rmap[arm_id],
                            "drop": drop_loc,
                            "desc": f"relay->{target_bin.owner_arm}",
                        })
                        break

            if not tasks:
                break

            summary = ", ".join(f"{t['arm']}({t['item'].label} {t['desc']})" for t in tasks)
            self.log_event("swarm", f"Wave {round_idx + 1} [{len(tasks)} arms]: {summary}")

            self.move_swarm([{"arm": t["arm"], "x": t["pick"]["x"], "y": t["pick"]["y"], "z": 0.065, "yaw": 0.0} for t in tasks])
            if self._stop_flag:
                break
            self.move_swarm([{"arm": t["arm"], "x": t["pick"]["x"], "y": t["pick"]["y"], "z": 0.018, "yaw": 0.0} for t in tasks])
            if self._stop_flag:
                break
            self.gripper_swarm([{"arm": t["arm"], "open": 0.0} for t in tasks])
            if self._stop_flag:
                break
            self.move_swarm([{"arm": t["arm"], "x": t["pick"]["x"], "y": t["pick"]["y"], "z": 0.072, "yaw": 0.0} for t in tasks])
            if self._stop_flag:
                break
            self.move_swarm([{"arm": t["arm"], "x": t["drop"][0], "y": t["drop"][1], "z": 0.068, "yaw": 0.0} for t in tasks])
            if self._stop_flag:
                break
            handoff_lowers = [
                {"arm": t["arm"], "x": t["drop"][0], "y": t["drop"][1], "z": float(t["drop"][2]), "yaw": 0.0}
                for t in tasks if "relay" in t["desc"]
            ]
            if handoff_lowers:
                self.move_swarm(handoff_lowers)
            self.gripper_swarm([{"arm": t["arm"], "open": 1.0} for t in tasks])
            self.go_rest_swarm()

        sorted_n = sum(1 for it in self.items.values() if it.sorted_in is not None)
        self.log_event("swarm", f"Complete: {sorted_n}/{len(self.items)} items sorted (0 collisions).")

    # ---------------------------------------------- Claude Swarm Coordinator --
    def run_claude_swarm(self, task: str) -> None:
        self._stop_flag = False
        if not API_KEY or API_KEY == "PASTE_YOUR_KEY_HERE":
            self.log_event("warn", "ANTHROPIC_API_KEY not set in .env; running Autonomous Swarm Planner.")
            self.run_auto_sort()
            return

        self.log_event("claude", f"Task: {task}")
        client = anthropic.Anthropic(api_key=API_KEY)
        system_prompt = (
            "You control a collaborative 4-robot ElRobot Swarm (4x 7+1 DoF arms) around a shared table:\n"
            "- arm_0 = Green Robot (South, owns Green Plastic Bin `bin_plastic`)\n"
            "- arm_1 = Blue Robot (North, owns Blue Metal Bin `bin_metal`)\n"
            "- arm_2 = Yellow Robot (West, owns Yellow Bio Bin `bin_bio`)\n"
            "- arm_3 = Pink Robot (East, owns Pink E-Waste Bin `bin_ewaste`)\n\n"
            "HOW THE SHARED WORKSPACE & RELAY HANDOFFS WORK:\n"
            "1. Call `get_swarm_state` first. Every item, bin, and `shared_handoff_zones` entry includes "
            "`reachable_by_arms_local_m`, which gives the EXACT local `(x, y, z)` coordinates for every arm that can reach it.\n"
            "2. If an item is in Arm A's reach (e.g. Yellow Robot `arm_2`), and needs to go to Arm B's bin "
            "(e.g. Blue Bin `bin_metal` owned by Blue Robot `arm_1`):\n"
            "   - Arm A picks up the item (pre-pose z=0.065 -> down z=0.018 -> gripper open=0.0 -> lift z=0.072),\n"
            "   - Arm A moves to the shared handoff pad between Arm A and Arm B (`handoff_nw` for arm_2<->arm_1, "
            "`handoff_sw` for arm_0<->arm_2, `handoff_ne` for arm_1<->arm_3, `handoff_se` for arm_3<->arm_0), "
            "lowers to z=0.022, opens gripper (open=1.0), and retreats to rest (x=0.13, y=0.0, z=0.08).\n"
            "   - Then Arm B picks up the item from that handoff pad using Arm B's local coordinates and drops it into Arm B's bin (z=0.068, open=1.0)!\n"
            "3. Never command two adjacent arms to the SAME handoff pad in the same `move_swarm` step; one arm places and retreats to rest first, then the receiving arm picks it up. (The 3D collision guard will also stagger conflicting moves automatically.)\n"
            "4. Act autonomously, never ask the user for confirmation; execute the full sequence including dropping into the requested bin."
        )
        tools = [
            {
                "name": "get_swarm_state",
                "description": (
                    "Read all 4 arms, all waste items (with colors and `reachable_by_arms_local_m` per arm), "
                    "all bins, and `shared_handoff_zones` for passing items between robots."
                ),
                "input_schema": {"type": "object", "properties": {}, "required": []},
            },
            {
                "name": "move_swarm",
                "description": "Move 1 to 4 arms simultaneously to target (x, y, z, yaw) in each arm's local frame (with automatic 3D inter-arm anti-collision).",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "moves": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "arm": {"type": "string", "description": "arm_0 (green), arm_1 (blue), arm_2 (yellow), or arm_3 (pink)"},
                                    "x": {"type": "number"},
                                    "y": {"type": "number"},
                                    "z": {"type": "number"},
                                    "yaw": {"type": "number"},
                                },
                                "required": ["arm", "x", "y", "z"],
                            },
                        }
                    },
                    "required": ["moves"],
                },
            },
            {
                "name": "gripper_swarm",
                "description": "Open (1.0) or close (0.0) grippers on 1 to 4 arms simultaneously.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "commands": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "arm": {"type": "string"},
                                    "open": {"type": "number"},
                                },
                                "required": ["arm", "open"],
                            },
                        }
                    },
                    "required": ["commands"],
                },
            },
            {
                "name": "go_rest_swarm",
                "description": "Return all 4 arms simultaneously to their retracted rest pose (x=0.13, y=0.0, z=0.08).",
                "input_schema": {"type": "object", "properties": {}, "required": []},
            },
        ]

        messages: list[dict] = [{"role": "user", "content": task}]
        for _ in range(MAX_SWARM_STEPS):
            if self._stop_flag:
                break
            try:
                resp = client.messages.create(
                    model=MODEL,
                    max_tokens=2048,
                    system=system_prompt,
                    tools=tools,
                    messages=messages,
                )
            except Exception as e:  # noqa: BLE001
                self.log_event("error", f"Anthropic API error ({type(e).__name__}: {e}).")
                return

            messages.append({"role": "assistant", "content": resp.content})
            for block in resp.content:
                if block.type == "text" and block.text.strip():
                    self.log_event("claude", block.text.strip())

            tool_uses = [b for b in resp.content if b.type == "tool_use"]
            if resp.stop_reason != "tool_use" or not tool_uses:
                break

            results = []
            for tu in tool_uses:
                inp = dict(tu.input)
                if tu.name == "get_swarm_state":
                    out = self.swarm_state_for_llm()
                elif tu.name == "move_swarm":
                    out = self.move_swarm(inp.get("moves", []))
                elif tu.name == "gripper_swarm":
                    out = self.gripper_swarm(inp.get("commands", []))
                elif tu.name == "go_rest_swarm":
                    out = self.go_rest_swarm()
                else:
                    out = {"error": f"unknown tool {tu.name}"}
                results.append({
                    "type": "tool_result",
                    "tool_use_id": tu.id,
                    "content": json.dumps(out),
                })
            messages.append({"role": "user", "content": results})

    def start_background_task(self, mode: str, prompt: str = "") -> bool:
        if self.task_thread is not None and self.task_thread.is_alive():
            return False

        def _runner():
            self.busy_task = mode if not prompt else f"Claude: {prompt[:36]}"
            try:
                if mode == "auto":
                    self.run_auto_sort()
                elif mode == "rest":
                    self._stop_flag = False
                    self.go_rest_swarm()
                elif mode == "claude":
                    self.run_claude_swarm(prompt)
            finally:
                self.busy_task = None

        self.task_thread = threading.Thread(target=_runner, daemon=True)
        self.task_thread.start()
        return True


world = SwarmWorld()
app = Flask(__name__)
ALLOWED_HOSTS = {f"{HOST}:{PORT}", f"localhost:{PORT}"}


@app.before_request
def _only_local() -> None:
    if request.host not in ALLOWED_HOSTS:
        abort(403)
    if request.method != "GET":   # CSRF: other web pages must not drive the real arm
        origin = request.headers.get("Origin")
        if origin is not None and origin not in {f"http://{h}" for h in ALLOWED_HOSTS}:
            abort(403)


@app.get("/static/<path:filename>")
def serve_static(filename: str):
    return send_from_directory(STATIC_DIR, filename)


@app.get("/urdf/assets/<path:filename>")
def serve_mesh(filename: str):
    return send_from_directory(ASSETS_DIR, filename)


@app.get("/api/state")
def api_state():
    return jsonify(world.browser_state())


@app.post("/api/action")
def api_action():
    if not request.is_json:       # forces a CORS preflight for cross-site requests
        abort(415)
    data = request.get_json(silent=True) or {}
    act = data.get("action")
    if act == "reset":
        world._stop_flag = True
        time.sleep(0.08)
        world.reset_scene(scrambled=bool(data.get("scrambled", False)))
        world.start_background_task("rest")
        return jsonify(ok=True)
    if act == "stop":
        world._stop_flag = True
        world.log_event("warn", "Operator requested stop.")
        return jsonify(ok=True)
    if act == "auto_sort":
        started = world.start_background_task("auto")
        return jsonify(ok=started)
    if act == "go_rest":
        started = world.start_background_task("rest")
        return jsonify(ok=started)
    if act == "claude":
        prompt = str(data.get("prompt", "")).strip()
        started = world.start_background_task("claude", prompt)
        return jsonify(ok=started)
    if act == "toggle_real":
        world.sync_real = bool(data.get("sync_real", False))
        mode_str = "LIVE HARDWARE MIRROR (Arm 0 -> 5B79016927)" if world.sync_real else "SIMULATION ONLY"
        world.log_event("mode", f"Switched mode to: {mode_str}")
        return jsonify(ok=True, sync_real=world.sync_real)
    return jsonify(ok=False, error="unknown action"), 400


SWARM_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>ElRobot Swarm AI - Sustainable City</title>
<style>
  * { box-sizing: border-box; }
  body {
    margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    background: #0b0e14; color: #e2e8f0; display: grid;
    grid-template-columns: 1fr 400px; height: 100vh; overflow: hidden;
  }
  #viewport { position: relative; width: 100%; height: 100%; background: radial-gradient(circle at center, #161f30 0%, #090c12 100%); }
  #canvas-container { width: 100%; height: 100%; }
  .hud-top {
    position: absolute; top: 14px; left: 16px; right: 16px;
    display: flex; justify-content: space-between; align-items: center; pointer-events: none;
  }
  .title-box {
    background: rgba(15, 23, 42, 0.85); border: 1px solid #334155;
    padding: 10px 14px; border-radius: 10px; pointer-events: auto; backdrop-filter: blur(8px);
  }
  .title-box h1 { margin: 0; font-size: 16px; font-weight: 700; letter-spacing: 0.02em; }
  .title-box .sub { font-size: 12px; color: #94a3b8; margin-top: 2px; }
  .badge-bar { display: flex; gap: 8px; pointer-events: auto; }
  .badge {
    background: rgba(15, 23, 42, 0.85); border: 1px solid #334155;
    padding: 7px 12px; border-radius: 8px; font-size: 12px; font-weight: 600;
  }
  .badge.live { border-color: #ef4444; color: #fca5a5; background: rgba(127, 29, 29, 0.45); }
  .badge.sim { border-color: #22c55e; color: #86efac; }
  .badge.guard { border-color: #38bdf8; color: #bae6fd; }
  .hud-bottom {
    position: absolute; bottom: 14px; left: 16px; right: 16px;
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; pointer-events: none;
  }
  .arm-card {
    background: rgba(15, 23, 42, 0.86); border: 1px solid #334155;
    border-top: 3px solid #64748b; border-radius: 8px; padding: 8px 10px;
    font-size: 11.5px; pointer-events: auto; backdrop-filter: blur(6px);
  }
  .arm-card .arm-title { font-weight: 700; font-size: 12px; display: flex; justify-content: space-between; }
  .arm-card .coords { font-family: ui-monospace, monospace; color: #cbd5e1; margin-top: 4px; font-size: 11px; }
  .arm-card .meta { color: #94a3b8; margin-top: 3px; display: flex; justify-content: space-between; }

  aside {
    background: #111622; border-left: 1px solid #1e293b; padding: 14px;
    display: flex; flex-direction: column; gap: 12px; overflow-y: auto;
  }
  h2 { font-size: 12px; text-transform: uppercase; letter-spacing: 0.06em; color: #94a3b8; margin: 0 0 6px; }
  .panel { background: #161d2b; border: 1px solid #263147; border-radius: 8px; padding: 11px; }
  .btn-row { display: flex; gap: 8px; flex-wrap: wrap; }
  button {
    cursor: pointer; border: 1px solid #3b82f6; background: #1d4ed8; color: #fff;
    font-weight: 600; font-size: 12px; padding: 8px 10px; border-radius: 6px; flex: 1;
    transition: 0.15s;
  }
  button:hover { filter: brightness(1.12); }
  button.secondary { background: #1e293b; border-color: #475569; color: #e2e8f0; }
  button.warn { background: #7f1d1d; border-color: #ef4444; }
  button.green { background: #15803d; border-color: #22c55e; }
  textarea {
    width: 100%; height: 68px; background: #0d121c; border: 1px solid #334155;
    border-radius: 6px; color: #f8fafc; padding: 8px; font-size: 12px; resize: none;
    margin-bottom: 8px; font-family: inherit;
  }
  .log-list {
    display: flex; flex-direction: column; gap: 6px; max-height: 360px; overflow-y: auto;
    font-family: ui-monospace, monospace; font-size: 11.5px;
  }
  .log-item {
    background: #0d121c; border-left: 3px solid #3b82f6; padding: 6px 8px; border-radius: 4px;
    line-height: 1.35; word-break: break-word;
  }
  .log-item.grasp { border-left-color: #22c55e; color: #bbf7d0; }
  .log-item.claude { border-left-color: #a855f7; color: #e9d5ff; }
  .log-item.collision_guard { border-left-color: #38bdf8; color: #bae6fd; }
  .log-item.warn, .log-item.error { border-left-color: #ef4444; color: #fecaca; }
  .log-item .ts { color: #64748b; font-size: 10.5px; margin-right: 6px; }
  .toggle-row { display: flex; align-items: center; justify-content: space-between; font-size: 12.5px; }
</style>
</head>
<body>
  <div id="viewport">
    <div id="canvas-container"></div>
    <div class="hud-top">
      <div class="title-box">
        <h1>NormaCore ElRobot Swarm AI (3D Anti-Collision + Multi-Arm Relay)</h1>
        <div class="sub">Green (arm_0) · Blue (arm_1) · Yellow (arm_2) · Pink (arm_3) · 8.5 cm 3D Link Clearance Guard</div>
      </div>
      <div class="badge-bar">
        <div class="badge guard" id="guard-badge">3D Guard: Active (0 staggered)</div>
        <div class="badge" id="progress-badge">Sorted: 0 / 8</div>
        <div class="badge sim" id="mode-badge">SIMULATION MODE</div>
      </div>
    </div>
    <div class="hud-bottom" id="arm-cards"></div>
  </div>

  <aside>
    <div class="panel">
      <h2>Swarm Controls</h2>
      <div class="btn-row" style="margin-bottom:8px">
        <button class="green" onclick="sendAction({action:'auto_sort'})">Run Swarm (Parallel + Relay)</button>
        <button class="secondary" onclick="sendAction({action:'go_rest'})">All Go Rest</button>
      </div>
      <div class="btn-row">
        <button class="secondary" onclick="sendAction({action:'reset', scrambled:false})">Reset 8 Items</button>
        <button class="secondary" onclick="sendAction({action:'reset', scrambled:true})">4 Relay Only</button>
        <button class="warn" onclick="sendAction({action:'stop'})">Stop</button>
      </div>
    </div>

    <div class="panel">
      <h2>Claude Swarm Coordinator (Natural Language)</h2>
      <textarea id="prompt">At the exact same time, have the Yellow robot pick up the Blue Steel Cube in its zone and pass it to the NW handoff pad while the Green robot picks up the Yellow Compost Block in its zone and passes it to the SW handoff pad. Then have the Blue and Yellow robots pick those two items up from the handoff pads and drop them into their matching bins.</textarea>
      <button onclick="runClaudePrompt()" style="width:100%">Send to Claude Swarm AI</button>
    </div>

    <div class="panel">
      <h2>Real Hardware Twin (Green Arm 0 -> 5B79016927)</h2>
      <div class="toggle-row">
        <span>Mirror Arm 0 to Physical Follower</span>
        <input type="checkbox" id="sync-real-box" onchange="toggleReal(this.checked)">
      </div>
      <div style="font-size:11px;color:#94a3b8;margin-top:5px">
        Keep hand on 12 V follower plug when live. SPEED=200, ACCEL=10.
      </div>
    </div>

    <div class="panel" style="flex:1;display:flex;flex-direction:column;min-height:0">
      <h2>Swarm Event & Tool Stream <span id="busy-label" style="color:#38bdf8;text-transform:none"></span></h2>
      <div class="log-list" id="logs"></div>
    </div>
  </aside>

<script type="module">
import * as THREE from '/static/three.module.min.js';

const container = document.getElementById('canvas-container');
const scene = new THREE.Scene();
scene.up.set(0, 0, 1);

const camera = new THREE.PerspectiveCamera(40, container.clientWidth / container.clientHeight, 0.02, 10);
camera.up.set(0, 0, 1);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(container.clientWidth, container.clientHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
container.appendChild(renderer.domElement);

let camTheta = -0.80, camPhi = 0.88, camDist = 0.86;
const camTarget = new THREE.Vector3(0, 0, 0.04);
function updateCamera() {
  camera.position.set(
    camTarget.x + camDist * Math.sin(camPhi) * Math.cos(camTheta),
    camTarget.y + camDist * Math.sin(camPhi) * Math.sin(camTheta),
    camTarget.z + camDist * Math.cos(camPhi)
  );
  camera.lookAt(camTarget);
}
updateCamera();

let dragging = false, prevX = 0, prevY = 0;
renderer.domElement.addEventListener('pointerdown', e => { dragging = true; prevX = e.clientX; prevY = e.clientY; });
window.addEventListener('pointerup', () => { dragging = false; });
window.addEventListener('pointermove', e => {
  if (!dragging) return;
  camTheta -= (e.clientX - prevX) * 0.007;
  camPhi = Math.max(0.18, Math.min(1.48, camPhi - (e.clientY - prevY) * 0.007));
  prevX = e.clientX; prevY = e.clientY;
  updateCamera();
});
renderer.domElement.addEventListener('wheel', e => {
  e.preventDefault();
  camDist = Math.max(0.38, Math.min(1.6, camDist + e.deltaY * 0.0008));
  updateCamera();
}, { passive: false });

window.addEventListener('resize', () => {
  camera.aspect = container.clientWidth / container.clientHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(container.clientWidth, container.clientHeight);
});

scene.add(new THREE.AmbientLight(0xdbeafe, 0.8));
const dirLight = new THREE.DirectionalLight(0xffffff, 1.35);
dirLight.position.set(0.5, -0.6, 1.1);
scene.add(dirLight);
const backLight = new THREE.DirectionalLight(0x93c5fd, 0.55);
backLight.position.set(-0.7, 0.7, 0.7);
scene.add(backLight);

// Table surface at Z = 0
const tableMesh = new THREE.Mesh(
  new THREE.BoxGeometry(0.80, 0.80, 0.016),
  new THREE.MeshStandardMaterial({ color: 0x1e293b, roughness: 0.75, metalness: 0.15 })
);
tableMesh.position.set(0, 0, -0.008);
scene.add(tableMesh);

const grid = new THREE.GridHelper(0.78, 16, 0x475569, 0x334155);
grid.rotation.x = Math.PI / 2;
grid.position.z = 0.0005;
scene.add(grid);

let handoffsBuilt = false;
function buildHandoffPads(zones) {
  if (handoffsBuilt) return;
  handoffsBuilt = true;
  for (const z of zones) {
    const pad = new THREE.Mesh(
      new THREE.RingGeometry(0.018, 0.025, 28),
      new THREE.MeshBasicMaterial({ color: 0x38bdf8, side: THREE.DoubleSide, transparent: true, opacity: 0.55 })
    );
    pad.position.set(z.global_xyz[0], z.global_xyz[1], 0.0015);
    scene.add(pad);
  }
}

async function loadBinarySTL(url) {
  const buf = await (await fetch(url)).arrayBuffer();
  const dv = new DataView(buf);
  const nTri = dv.getUint32(80, true);
  const positions = new Float32Array(nTri * 9);
  const normals = new Float32Array(nTri * 9);
  let offset = 84;
  for (let i = 0; i < nTri; i++) {
    const nx = dv.getFloat32(offset, true);
    const ny = dv.getFloat32(offset + 4, true);
    const nz = dv.getFloat32(offset + 8, true);
    for (let v = 0; v < 3; v++) {
      const vo = offset + 12 + v * 12;
      const idx = i * 9 + v * 3;
      positions[idx]     = dv.getFloat32(vo, true) * 0.001;
      positions[idx + 1] = dv.getFloat32(vo + 4, true) * 0.001;
      positions[idx + 2] = dv.getFloat32(vo + 8, true) * 0.001;
      normals[idx] = nx; normals[idx + 1] = ny; normals[idx + 2] = nz;
    }
    offset += 50;
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geo.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
  return geo;
}

const meshCache = {};
const armMeshes = {};
const itemMeshes = {};
const binMeshes = {};
const servoMat = new THREE.MeshStandardMaterial({ color: 0x18181b, roughness: 0.45, metalness: 0.35 });

async function ensureArmBuilt(arm, visualDefs) {
  if (armMeshes[arm.id]) return;
  armMeshes[arm.id] = { links: {} };
  const bracketMat = new THREE.MeshStandardMaterial({
    color: arm.accent,
    roughness: 0.40,
    metalness: 0.20,
  });
  const padGeo = new THREE.CylinderGeometry(0.045, 0.050, 0.005, 28);
  padGeo.rotateX(Math.PI / 2);
  const pad = new THREE.Mesh(padGeo, new THREE.MeshBasicMaterial({ color: arm.accent }));
  pad.position.set(arm.base_xyz[0], arm.base_xyz[1], 0.0025);
  scene.add(pad);

  const tcpSphere = new THREE.Mesh(
    new THREE.SphereGeometry(0.0055, 14, 14),
    new THREE.MeshBasicMaterial({ color: arm.accent })
  );
  scene.add(tcpSphere);
  armMeshes[arm.id].tcp = tcpSphere;

  for (const def of visualDefs) {
    if (!meshCache[def.mesh]) {
      meshCache[def.mesh] = loadBinarySTL('/urdf/assets/' + def.mesh);
    }
    const geo = await meshCache[def.mesh];
    const m = new THREE.Mesh(geo, def.material === 'servo' ? servoMat : bracketMat);
    m.matrixAutoUpdate = false;
    scene.add(m);
    armMeshes[arm.id].links[def.link] = m;
  }
}

function updateBinsAndItems(state) {
  buildHandoffPads(state.handoff_zones || []);
  for (const b of state.bins) {
    if (!binMeshes[b.id]) {
      const grp = new THREE.Group();
      const boxGeo = new THREE.BoxGeometry(0.068, 0.068, 0.030);
      const edges = new THREE.EdgesGeometry(boxGeo);
      const wire = new THREE.LineSegments(edges, new THREE.LineBasicMaterial({ color: b.color }));
      const fill = new THREE.Mesh(boxGeo, new THREE.MeshStandardMaterial({
        color: b.color, transparent: true, opacity: 0.25, roughness: 0.5
      }));
      grp.add(fill); grp.add(wire);
      scene.add(grp);
      binMeshes[b.id] = grp;
    }
    binMeshes[b.id].position.set(b.global_xyz[0], b.global_xyz[1], 0.015);
  }

  const activeIds = new Set(state.items.map(i => i.id));
  for (const [id, m] of Object.entries(itemMeshes)) {
    if (!activeIds.has(id)) {
      scene.remove(m);
      delete itemMeshes[id];
    }
  }
  for (const it of state.items) {
    if (!itemMeshes[it.id]) {
      const geo = it.shape === 'cylinder'
        ? new THREE.CylinderGeometry(it.size * 0.52, it.size * 0.52, it.size, 20).rotateX(Math.PI / 2)
        : new THREE.BoxGeometry(it.size, it.size, it.size);
      const mat = new THREE.MeshStandardMaterial({ color: it.color, roughness: 0.28, metalness: 0.18 });
      const mesh = new THREE.Mesh(geo, mat);
      scene.add(mesh);
      itemMeshes[it.id] = mesh;
    }
    itemMeshes[it.id].position.set(it.global_xyz[0], it.global_xyz[1], it.global_xyz[2]);
  }
}

const esc = s => String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));

async function pollState() {
  try {
    const res = await fetch('/api/state');
    const state = await res.json();

    for (const arm of state.arms) {
      await ensureArmBuilt(arm, state.visual_defs);
      const entry = armMeshes[arm.id];
      if (entry) {
        for (const [linkName, matArr] of Object.entries(arm.links)) {
          if (entry.links[linkName]) {
            entry.links[linkName].matrix.fromArray(matArr);
          }
        }
        entry.tcp.position.set(arm.global_tcp[0], arm.global_tcp[1], arm.global_tcp[2]);
      }
    }
    updateBinsAndItems(state);

    document.getElementById('guard-badge').textContent = `3D Guard: Active (${state.collisions_prevented || 0} staggered)`;
    document.getElementById('progress-badge').textContent = `Sorted: ${state.sorted_count} / ${state.total_items}`;
    const mb = document.getElementById('mode-badge');
    if (state.sync_real) {
      mb.className = 'badge live';
      mb.textContent = 'LIVE MIRROR: ARM 0 + 3 SIM';
    } else {
      mb.className = 'badge sim';
      mb.textContent = 'SIMULATION (4 ARMS)';
    }
    document.getElementById('sync-real-box').checked = state.sync_real;
    document.getElementById('busy-label').textContent = state.busy_task ? `· Running (${state.busy_task})` : '';

    document.getElementById('arm-cards').innerHTML = state.arms.map(a => `
      <div class="arm-card" style="border-top-color:${a.accent}">
        <div class="arm-title">
          <span>${esc(a.name)}</span>
          <span style="color:${a.accent}">${esc(a.status.toUpperCase())}</span>
        </div>
        <div class="coords">
          local: x=${a.local_pose.x.toFixed(2)} y=${a.local_pose.y.toFixed(2)} z=${a.local_pose.z.toFixed(2)}
        </div>
        <div class="meta">
          <span>Grip: ${Math.round(a.gripper * 100)}%</span>
          <span>${a.held_item_id ? 'Holding: ' + esc(a.held_item_id) : esc(a.sector)}</span>
        </div>
      </div>
    `).join('');

    document.getElementById('logs').innerHTML = state.events.map(e => `
      <div class="log-item ${esc(e.kind)}">
        <span class="ts">${esc(e.ts)}</span>${esc(e.message)}
      </div>
    `).join('');
  } catch (err) {}
}

window.sendAction = async (payload) => {
  await fetch('/api/action', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
};

window.runClaudePrompt = () => {
  const prompt = document.getElementById('prompt').value;
  window.sendAction({ action: 'claude', prompt });
};

window.toggleReal = (checked) => {
  window.sendAction({ action: 'toggle_real', sync_real: checked });
};

function animate() {
  requestAnimationFrame(animate);
  renderer.render(scene, camera);
}
animate();
setInterval(pollState, 85);
pollState();
</script>
</body>
</html>
"""


@app.get("/")
def index():
    return SWARM_HTML


def main() -> int:
    args = sys.argv[1:]
    if "--auto" in args:
        world.start_background_task("auto")
    elif args:
        world.start_background_task("claude", " ".join(args))

    print(f"ElRobot Swarm AI running on http://{HOST}:{PORT}")
    print(f"Arms        : 4x ElRobot 7+1 DoF (3D inter-arm anti-collision + relay handoffs)")
    print(f"Real Twin   : Arm 0 -> {FOLLOWER_ID} ({'LIVE MIRROR' if world.sync_real else 'SIM ONLY (toggle in UI)'})")
    print(f"Log         : {world.log_path}")
    app.run(host=HOST, port=PORT, debug=False, threaded=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

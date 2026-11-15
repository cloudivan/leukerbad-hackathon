"""MuJoCo physics simulation + NormaCore 8-DoF Parquet dataset generator for ElRobot.

Converts `hardware/elrobot/simulation/elrobot_follower.urdf` (+ 19 STL meshes) into a complete
MuJoCo MJCF pick-and-place scene aligned with the `elrobot-ik` world frame:
  - Origin (0, 0, 0) on `rev_motor_01` yaw axis at table height (z = 0)
  - +X forward, +Y left, +Z up
  - 8 position actuators (`rev_motor_01` .. `rev_motor_08`) + mimic equality constraints for both jaws
  - `cam0` (wrist camera on `Gripper_Base_v1_1`) and `cam1` (table camera) at 224x224
  - Graspable cube (`red_cube`, freejoint) and target bin (`target_zone`)

Usage:
  # 1. Build scene XML, verify FK match with placo (< 0.1 mm), and run 1 physics pick-and-place test:
  uv run python mujoco_sim.py --test

  # 2. Launch interactive 3D MuJoCo viewer watching the pick-and-place rollout:
  uv run python mujoco_sim.py --viewer

  # 3. Generate N randomized pick-and-place episodes in NormaCore Parquet format (for GCE SmolVLA/BC training):
  uv run python mujoco_sim.py --episodes 50 --out datasets/sim_pick_place_8dof.parquet
"""
from __future__ import annotations

import argparse
import io
import math
import os
from pathlib import Path
import site
import sys
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
NORMA_CORE = Path(os.environ.get("NORMA_CORE", str(Path.home() / "Developer" / "norma-core")))  # ADJUST: norma-core path
IK_API_DIR = NORMA_CORE / "software" / "station" / "examples" / "elrobot-ik-api"
site.addsitedir(str(IK_API_DIR / ".venv" / "lib" / "python3.12" / "site-packages"))
if str(IK_API_DIR / "src") not in sys.path:
    sys.path.insert(0, str(IK_API_DIR / "src"))

from elrobot_ik import URDF_PATH  # noqa: E402
from elrobot_ik.calibration.mapper import JOINT_DIRECTIONS  # noqa: E402
from elrobot_ik.kinematics.ik import IKSolver  # noqa: E402
from elrobot_ik.kinematics.model import MOTOR_JOINTS, TCP_OFFSET, ElrobotModel  # noqa: E402
from elrobot_ik.kinematics.transforms import BASE_YAW_AXIS_URDF_Y, make_transform, rpy_to_rotation  # noqa: E402

import safety  # noqa: E402

SIM_ASSETS_DIR = HERE / "sim_assets"
SCENE_XML_PATH = SIM_ASSETS_DIR / "elrobot_scene.xml"
CAM_SIZE = 224                          # ADJUST: matches resize_target: 224 in station.yaml
RECORD_HZ = 10                          # ADJUST: matches update-interval: 100ms in station.yaml
PHYSICS_DT = 0.002                      # ADJUST: 500 Hz MuJoCo physics step
STEPS_PER_FRAME = int(round((1.0 / RECORD_HZ) / PHYSICS_DT))
CUBE_HALF_M = 0.012                     # ADJUST: 24 mm cube (fits easily inside 50 mm open jaws)


def angles_to_norm(angles: list[float], model: ElrobotModel) -> list[float]:
    """Convert 8 URDF joint angles (rad) to Station normalized [0, 1] (`position_norm` / `goal_norm`)."""
    out: list[float] = []
    for idx, (angle, lim, direction) in enumerate(zip(angles, model.joint_limits, JOINT_DIRECTIONS)):
        if lim.span <= 0:
            out.append(0.0)
            continue
        if direction < 0:
            p = (lim.upper - float(angle)) / lim.span
        else:
            p = (float(angle) - lim.lower) / lim.span
        out.append(float(min(1.0, max(0.0, p))))
    return out


def norm_to_angles(norm_vals: list[float], model: ElrobotModel) -> list[float]:
    """Convert 8 Station normalized [0, 1] values back to URDF joint angles (rad)."""
    out: list[float] = []
    for p, lim, direction in zip(norm_vals, model.joint_limits, JOINT_DIRECTIONS):
        p_clamped = min(1.0, max(0.0, float(p)))
        if direction < 0:
            out.append(float(lim.upper - p_clamped * lim.span))
        else:
            out.append(float(lim.lower + p_clamped * lim.span))
    return out


def build_mjcf_scene(urdf_path: Path = Path(URDF_PATH), out_xml: Path = SCENE_XML_PATH) -> Path:
    """Convert `elrobot_follower.urdf` into a full MuJoCo MJCF pick-and-place scene XML."""
    out_xml.parent.mkdir(parents=True, exist_ok=True)
    mesh_dir = (urdf_path.parent / "assets").resolve()

    urdf_root = ET.parse(urdf_path).getroot()
    muj_el = ET.SubElement(urdf_root, "mujoco")
    ET.SubElement(
        muj_el,
        "compiler",
        meshdir=str(mesh_dir),
        balanceinertia="true",
        discardvisual="false",
    )
    # Fix zero-inertia on Gripper_Gear_v1_1 in the CAD export so MuJoCo compiles cleanly
    for inertia in urdf_root.iter("inertia"):
        for k in ("ixx", "iyy", "izz"):
            if float(inertia.get(k, "0")) < 1e-6:
                inertia.set(k, "1e-6")
    for mesh in urdf_root.iter("mesh"):
        fn = mesh.get("filename", "")
        if fn.startswith("assets/"):
            mesh.set("filename", fn.replace("assets/", ""))

    tmp_model = mujoco.MjModel.from_xml_string(ET.tostring(urdf_root, encoding="unicode"))
    raw_mjcf_path = out_xml.parent / "_raw_converted.xml"
    mujoco.mj_saveLastXML(str(raw_mjcf_path), tmp_model)

    mj_tree = ET.parse(raw_mjcf_path)
    raw_mjcf_path.unlink(missing_ok=True)
    mj_root = mj_tree.getroot()
    mj_root.set("model", "elrobot_pick_and_place")

    # Simulation options: 2 ms timestep, implicitfast integrator, high-friction cone
    opt = ET.Element("option", timestep=f"{PHYSICS_DT}", integrator="implicitfast", cone="elliptic", impratio="10")
    mj_root.insert(0, opt)

    vis = ET.Element("visual")
    ET.SubElement(vis, "global", offwidth=str(CAM_SIZE), offheight=str(CAM_SIZE))
    ET.SubElement(vis, "headlight", ambient="0.45 0.45 0.45", diffuse="0.6 0.6 0.6", specular="0.2 0.2 0.2")
    mj_root.insert(1, vis)

    default_el = ET.Element("default")
    ET.SubElement(default_el, "joint", damping="0.35", armature="0.015", frictionloss="0.02")
    mj_root.insert(2, default_el)

    worldbody = mj_root.find("worldbody")
    assert worldbody is not None

    # Turn off mesh-mesh contacts on arm links (placo handles arm/table collision; jaw pads handle grasping)
    for geom in worldbody.iter("geom"):
        geom.set("contype", "0")
        geom.set("conaffinity", "0")

    # Wrap all robot elements inside `elrobot_base` transformed by T_WORLD_URDF:
    # pos = (-0.042489, 0, 0), quat for Rz(-pi/2) = (cos(-pi/4), 0, 0, sin(-pi/4)) = (0.70710678, 0, 0, -0.70710678)
    robot_children = list(worldbody)
    for child in robot_children:
        worldbody.remove(child)

    base_body = ET.SubElement(
        worldbody,
        "body",
        name="elrobot_base",
        pos=f"{-BASE_YAW_AXIS_URDF_Y:.6f} 0 0",
        quat="0.70710678 0 0 -0.70710678",
    )
    for child in robot_children:
        base_body.append(child)

    # Inject TCP site and wrist camera (`cam0`) onto `Gripper_Base_v1_1`
    gripper_base = None
    jaw1_body = None
    jaw2_body = None
    for body in base_body.iter("body"):
        bname = body.get("name", "")
        if bname == "Gripper_Base_v1_1":
            gripper_base = body
        elif bname == "Gripper_Jaw_01_v1_1":
            jaw1_body = body
        elif bname == "Gripper_Jaw_02_v1_1":
            jaw2_body = body

    assert gripper_base is not None and jaw1_body is not None and jaw2_body is not None

    ET.SubElement(
        gripper_base,
        "site",
        name="tcp",
        pos=f"{TCP_OFFSET[0]} {TCP_OFFSET[1]} {TCP_OFFSET[2]}",
        size="0.004",
        rgba="0.1 0.9 0.2 0.8",
    )
    # Wrist camera mounted above the gripper looking toward the jaw tips (+Y in Gripper_Base frame)
    ET.SubElement(
        gripper_base,
        "camera",
        name="cam0",
        pos="0.002 0.025 0.055",
        euler="1.15 0 0",
        fovy="62",
    )

    # Add high-friction box collision pads on the inner faces of both jaws right at the TCP tip
    # In Gripper_Jaw_02_v1_1 local frame: inner tip pad center is (-0.3365, 0.2921, -0.0325)
    ET.SubElement(
        jaw2_body,
        "geom",
        name="jaw2_pad",
        type="box",
        pos="-0.3365 0.2921 -0.0325",
        size="0.0025 0.013 0.015",
        rgba="0.2 0.2 0.2 0.9",
        contype="1",
        conaffinity="1",
        friction="2.5 0.05 0.01",
        solref="0.008 1",
        solimp="0.95 0.99 0.001",
    )
    # In Gripper_Jaw_01_v1_1 local frame: inner tip pad center is (-0.2807, 0.3358, 0.0497)
    ET.SubElement(
        jaw1_body,
        "geom",
        name="jaw1_pad",
        type="box",
        pos="-0.2807 0.3358 0.0497",
        size="0.0025 0.013 0.015",
        rgba="0.2 0.2 0.2 0.9",
        contype="1",
        conaffinity="1",
        friction="2.5 0.05 0.01",
        solref="0.008 1",
        solimp="0.95 0.99 0.001",
    )

    # Lighting, table surface at z = 0, target drop pad, overhead/front table camera (`cam1`), and graspable cube
    ET.SubElement(worldbody, "light", pos="0.3 0.2 0.8", dir="-0.2 -0.1 -1.0", diffuse="0.8 0.8 0.8")
    ET.SubElement(
        worldbody,
        "geom",
        name="table",
        type="box",
        pos="0.18 0 -0.01",
        size="0.35 0.35 0.01",
        rgba="0.86 0.84 0.80 1",
        contype="1",
        conaffinity="1",
        friction="1.0 0.01 0.005",
    )
    ET.SubElement(
        worldbody,
        "site",
        name="target_zone",
        type="cylinder",
        pos="0.18 -0.10 0.001",
        size="0.038 0.001",
        rgba="0.15 0.75 0.35 0.75",
    )
    ET.SubElement(
        worldbody,
        "camera",
        name="cam1",
        pos="0.44 0.0 0.30",
        xyaxes="0 1 0 -0.62 0 0.78",
        fovy="55",
    )

    cube_body = ET.SubElement(
        worldbody,
        "body",
        name="red_cube",
        pos=f"0.18 0.08 {CUBE_HALF_M:.4f}",
    )
    ET.SubElement(cube_body, "freejoint", name="red_cube_joint")
    ET.SubElement(
        cube_body,
        "geom",
        name="red_cube_geom",
        type="box",
        size=f"{CUBE_HALF_M} {CUBE_HALF_M} {CUBE_HALF_M}",
        mass="0.015",
        rgba="0.90 0.18 0.18 1",
        contype="1",
        conaffinity="1",
        friction="2.5 0.05 0.01",
        solref="0.008 1",
        solimp="0.95 0.99 0.001",
    )

    # Strengthen equality mimic constraints for the prismatic gripper jaws
    eq_el = mj_root.find("equality")
    if eq_el is not None:
        for j_eq in eq_el.findall("joint"):
            j_eq.set("solref", "0.004 1")
            j_eq.set("solimp", "0.98 0.995 0.0005")

    # Add 8 position servos matching rev_motor_01 .. rev_motor_08
    act_el = ET.SubElement(mj_root, "actuator")
    limits = [
        (-1.5509, 1.5509),
        (-1.6122, 1.6122),
        (-1.7610, 1.7610),
        (-1.7533, 1.7533),
        (-2.6200, 3.2520),
        (-1.3775, 1.7641),
        (-3.2014, 2.7336),
        (0.0, 2.2028),
    ]
    for m_name, (lo, hi) in zip(MOTOR_JOINTS, limits):
        ET.SubElement(
            act_el,
            "position",
            name=f"act_{m_name}",
            joint=m_name,
            kp="85",
            kv="6",
            ctrlrange=f"{lo} {hi}",
            forcerange="-2.94 2.94",
        )

    ET.indent(mj_tree, space="  ")
    mj_tree.write(out_xml, xml_declaration=True, encoding="unicode")
    return out_xml


def solve_target_angles(
    solver: IKSolver,
    x: float,
    y: float,
    z: float,
    yaw: float,
    gripper_open: float,
    seed_angles: list[float],
) -> list[float]:
    """Solve 8-DoF URDF joint angles for a world (x, y, z, yaw) target and gripper opening in [0..1]."""
    # In URDF, rev_motor_08 = 0.0 is open (1.0), 2.2028 is closed (0.0)
    lim8 = solver.model.joint_limits[7]
    q8 = lim8.upper - min(1.0, max(0.0, float(gripper_open))) * lim8.span
    seed = list(seed_angles[:7]) + [q8]

    for pitch in (safety.PITCH, -0.15, -0.28):
        R = rpy_to_rotation(safety.ROLL, pitch, yaw)
        T_world = make_transform(R, [x, y, z])
        res = solver.solve_pose(
            T_world,
            seed,
            accept=lambda a: not solver.model.collisions(a).any,
        )
        if res.accepted or res.reachable:
            angles = list(res.angles)
            angles[7] = q8
            return angles
    raise RuntimeError(f"Target ({x:.3f}, {y:.3f}, {z:.3f}) unreachable in IK")


def render_jpeg(renderer: mujoco.Renderer, data: mujoco.MjData, camera_name: str) -> bytes:
    renderer.update_scene(data, camera=camera_name)
    rgb = renderer.render()
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def run_pick_and_place_episode(
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    placo_model: ElrobotModel,
    ik_solver: IKSolver,
    cube_xy: tuple[float, float],
    target_xy: tuple[float, float],
    renderer: mujoco.Renderer | None = None,
    viewer=None,
    task_prompt: str = "pick the red cube and place it on the green target",
) -> tuple[bool, list[dict]]:
    """Execute one physics pick-and-place rollout and return (success, frames)."""
    mujoco.mj_resetData(mj_model, mj_data)

    # Set initial arm pose to REST_POSE
    home_angles = solve_target_angles(
        ik_solver,
        safety.REST_POSE.x,
        safety.REST_POSE.y,
        safety.REST_POSE.z,
        safety.REST_POSE.yaw,
        gripper_open=1.0,
        seed_angles=[0.0] * 8,
    )
    for i in range(8):
        mj_data.qpos[i] = home_angles[i]
        mj_data.ctrl[i] = home_angles[i]
    mj_data.qpos[8] = -0.0115 * home_angles[7]
    mj_data.qpos[9] = 0.0115 * home_angles[7]

    # Set cube initial pose (freejoint starts at qpos[10:17])
    mj_data.qpos[10] = cube_xy[0]
    mj_data.qpos[11] = cube_xy[1]
    mj_data.qpos[12] = CUBE_HALF_M + 0.0005
    mj_data.qpos[13:17] = [1.0, 0.0, 0.0, 0.0]

    # Move target_zone site to target_xy
    target_site_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "target_zone")
    mj_model.site_pos[target_site_id] = [target_xy[0], target_xy[1], 0.001]
    mujoco.mj_forward(mj_model, mj_data)

    # Waypoints: (x, y, z, yaw, gripper_open, duration_sec)
    # Grasp slightly tight (gripper_open=0.18 -> jaw gap ~9 mm on a 24 mm cube -> firm PD squeeze)
    cx, cy = cube_xy
    tx, ty = target_xy
    waypoints = [
        (cx, cy, 0.065, 0.0, 1.0, 0.8),   # 1. Hover above cube with jaws open
        (cx, cy, 0.010, 0.0, 1.0, 0.7),   # 2. Descend around cube
        (cx, cy, 0.010, 0.0, 0.15, 0.6),  # 3. Close jaws firmly on cube
        (cx, cy, 0.075, 0.0, 0.15, 0.8),  # 4. Lift cube clear of table
        (tx, ty, 0.075, 0.0, 0.15, 1.0),  # 5. Carry cube above target zone
        (tx, ty, 0.018, 0.0, 0.15, 0.7),  # 6. Lower cube onto target zone
        (tx, ty, 0.018, 0.0, 1.0, 0.5),   # 7. Open jaws to release
        (tx, ty, 0.075, 0.0, 1.0, 0.6),   # 8. Retract up
    ]

    frames: list[dict] = []
    ep_start_ns = time.time_ns()
    sim_step = 0
    curr_goal = list(home_angles)

    for wx, wy, wz, wyaw, wopen, duration in waypoints:
        next_goal = solve_target_angles(ik_solver, wx, wy, wz, wyaw, wopen, curr_goal)
        n_records = max(1, int(round(duration * RECORD_HZ)))
        for r in range(n_records):
            alpha = (r + 1) / n_records
            interp_goal = [(1.0 - alpha) * a + alpha * b for a, b in zip(curr_goal, next_goal)]
            mj_data.ctrl[:8] = interp_goal

            for _ in range(STEPS_PER_FRAME):
                mujoco.mj_step(mj_model, mj_data)
                sim_step += 1

            if viewer is not None:
                viewer.sync()
                time.sleep(0.02)

            if renderer is not None:
                pos_angles = [float(mj_data.qpos[i]) for i in range(8)]
                pos_norm = angles_to_norm(pos_angles, placo_model)
                goal_norm = angles_to_norm(interp_goal, placo_model)
                cam0_jpg = render_jpeg(renderer, mj_data, "cam0")
                cam1_jpg = render_jpeg(renderer, mj_data, "cam1")
                frames.append(
                    {
                        "episode_start_ns": ep_start_ns,
                        "timestamp_ns_since_episode_start": int(sim_step * PHYSICS_DT * 1e9),
                        "joints": [
                            {"position_norm": np.float32(p), "goal_norm": np.float32(g)}
                            for p, g in zip(pos_norm, goal_norm)
                        ],
                        "images": [{"jpeg": cam0_jpg}, {"jpeg": cam1_jpg}],
                        "task": task_prompt,
                    }
                )
        curr_goal = next_goal

    cube_final_xyz = mj_data.qpos[10:13].copy()
    dist_to_target = float(math.hypot(cube_final_xyz[0] - tx, cube_final_xyz[1] - ty))
    success = dist_to_target < 0.045 and cube_final_xyz[2] > 0.005
    return success, frames


def write_normacore_parquet(rows: list[dict], out_path: Path) -> None:
    """Write frames in the exact schema consumed by `smolvla.dataset.PickAndPlaceDataset`."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joint_struct = pa.struct([("position_norm", pa.float32()), ("goal_norm", pa.float32())])
    img_struct = pa.struct([("jpeg", pa.binary())])
    schema = pa.schema(
        [
            ("episode_start_ns", pa.uint64()),
            ("timestamp_ns_since_episode_start", pa.uint64()),
            ("joints", pa.list_(joint_struct)),
            ("images", pa.list_(img_struct)),
            ("task", pa.string()),
        ]
    )
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, out_path, compression="snappy")


def verify_fk_and_rollout() -> None:
    xml_path = build_mjcf_scene()
    mj_model = mujoco.MjModel.from_xml_path(str(xml_path))
    mj_data = mujoco.MjData(mj_model)

    placo_model = ElrobotModel()
    ik_solver = IKSolver(placo_model)

    # 1. Verify FK match between Placo and MuJoCo across test angles
    test_angles = [0.2, 0.35, -0.25, 0.15, -0.1, 0.6, 0.1, 0.0]
    T_placo = placo_model.fk(test_angles)
    mj_data.qpos[:8] = test_angles
    mj_data.qpos[8] = -0.0115 * test_angles[7]
    mj_data.qpos[9] = 0.0115 * test_angles[7]
    mujoco.mj_forward(mj_model, mj_data)
    tcp_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
    tcp_mj = mj_data.site_xpos[tcp_id]
    fk_err_mm = float(np.linalg.norm(T_placo[:3, 3] - tcp_mj) * 1000.0)
    print(f"[FK check] Placo TCP={T_placo[:3, 3].round(4)} | MuJoCo TCP={tcp_mj.round(4)} | diff={fk_err_mm:.3f} mm")

    # 2. Run 1 physics pick-and-place episode with offscreen camera rendering
    renderer = mujoco.Renderer(mj_model, height=CAM_SIZE, width=CAM_SIZE)
    ok, frames = run_pick_and_place_episode(
        mj_model,
        mj_data,
        placo_model,
        ik_solver,
        cube_xy=(0.18, 0.07),
        target_xy=(0.18, -0.08),
        renderer=renderer,
    )
    cube_end = mj_data.qpos[10:13].round(4)
    print(f"[Rollout]  success={ok} | frames={len(frames)} | cube_final_xyz={cube_end}")
    renderer.close()
    placo_model.close()
    if not ok:
        raise SystemExit("Physics pick-and-place test failed to place cube in target zone.")


def main() -> None:
    p = argparse.ArgumentParser(description="ElRobot MuJoCo simulation and Parquet generator")
    p.add_argument("--test", action="store_true", help="Build scene, verify FK, and run 1 pick-and-place test")
    p.add_argument("--viewer", action="store_true", help="Run interactive 3D MuJoCo viewer")
    p.add_argument("--episodes", type=int, default=0, help="Number of successful episodes to generate")
    p.add_argument(
        "--out",
        type=Path,
        default=HERE / "datasets" / "sim_pick_place_8dof.parquet",
        help="Output NormaCore Parquet file path",
    )
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    xml_path = build_mjcf_scene()
    print(f"Scene XML ready at: {xml_path}")

    if args.test:
        verify_fk_and_rollout()
        return

    mj_model = mujoco.MjModel.from_xml_path(str(xml_path))
    mj_data = mujoco.MjData(mj_model)
    placo_model = ElrobotModel()
    ik_solver = IKSolver(placo_model)
    rng = np.random.default_rng(args.seed)

    if args.viewer:
        from mujoco import viewer as mj_viewer

        with mj_viewer.launch_passive(mj_model, mj_data) as v:
            while v.is_running():
                cx = float(rng.uniform(0.15, 0.21))
                cy = float(rng.uniform(0.04, 0.10))
                tx = float(rng.uniform(0.15, 0.21))
                ty = float(rng.uniform(-0.10, -0.04))
                ok, _ = run_pick_and_place_episode(
                    mj_model, mj_data, placo_model, ik_solver, (cx, cy), (tx, ty), viewer=v
                )
                print(f"Episode finished: success={ok}")
        placo_model.close()
        return

    if args.episodes > 0:
        renderer = mujoco.Renderer(mj_model, height=CAM_SIZE, width=CAM_SIZE)
        all_rows: list[dict] = []
        saved = 0
        attempts = 0
        t0 = time.time()
        while saved < args.episodes:
            attempts += 1
            cx = float(rng.uniform(0.15, 0.21))
            cy = float(rng.uniform(0.04, 0.10))
            tx = float(rng.uniform(0.15, 0.21))
            ty = float(rng.uniform(-0.10, -0.04))
            try:
                ok, rows = run_pick_and_place_episode(
                    mj_model,
                    mj_data,
                    placo_model,
                    ik_solver,
                    cube_xy=(cx, cy),
                    target_xy=(tx, ty),
                    renderer=renderer,
                )
            except RuntimeError:
                continue
            if ok:
                saved += 1
                all_rows.extend(rows)
                print(
                    f"[{saved:>3}/{args.episodes}] cube=({cx:.3f},{cy:.3f}) -> target=({tx:.3f},{ty:.3f}) "
                    f"| +{len(rows)} frames (total {len(all_rows)})"
                )
            else:
                print(f"  [retry] attempt {attempts} slipped, discarding episode")

        renderer.close()
        placo_model.close()
        write_normacore_parquet(all_rows, args.out)
        print(
            f"\nSaved {saved} episodes ({len(all_rows)} frames, 8 DoF) to {args.out} "
            f"in {time.time() - t0:.1f}s"
        )
        return

    verify_fk_and_rollout()


if __name__ == "__main__":
    main()

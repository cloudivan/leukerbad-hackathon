"""Overview camera calibration: pixel (Camera 0) -> table x, y (robot frame) via a homography.

Step 1 (robot): the follower visits a 3x3 grid, jaws closed, tips ~12 mm above the table, and takes an
         overview photo at each point (lift to HOVER_Z between points, so nothing is swept).
Step 2 (you):  a small web page shows each photo; click the CENTER OF THE JAW TIPS. 9 clicks.
Step 3:        the homography is fitted and saved to calib/overview_homography.json, with its error in mm.

    uv run python calib_overview.py            # capture + click page
    uv run python calib_overview.py --click logs/calib_overview_<time>   # redo the clicks only

Clear the grid area of objects first. E-stop: 12 V follower plug. After this, DO NOT move Camera 0.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_file

import safety
from ik_client import IKClient
from safety import Target

HERE = Path(__file__).resolve().parent
load_dotenv(HERE / ".env")
IK_URL = os.environ.get("IK_URL", "http://127.0.0.1:8020").strip()
FOLLOWER_ID = os.environ.get("FOLLOWER_ID", "5B79015365").strip()
OVERVIEW_ID = os.environ.get("OVERVIEW_CAMERA", "43640b79a54e71f45e0443ffbb92880a").strip()
OUT_FILE = HERE / "calib" / "overview_homography.json"

# ---------------------------------------------------------------- ADJUST ----
GRID_X = [0.13, 0.17, 0.21]       # m, inside reach at low height
GRID_Y = [-0.07, 0.0, 0.07]       # m
TIP_ABOVE_FLOOR = 0.012           # m, jaw tips above the table at each grid point
HOVER_Z = 0.080                   # m, travel height between points
PORT = 5070                       # click page: http://127.0.0.1:5070
SCALE = 3                         # photos are shown 3x larger for precise clicks
# -----------------------------------------------------------------------------


def z_floor(x: float, y: float) -> float:
    """Floor plane measured 15.11.2026 15:30 (camera_notes_place.txt)."""
    return 0.0056 + 0.083 * x + 0.011 * y


def capture() -> Path:
    ik = IKClient(IK_URL)
    out = HERE / "logs" / f"calib_overview_{time.strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True)
    points = []
    order = [(x, y) for i, x in enumerate(GRID_X) for y in (GRID_Y if i % 2 == 0 else GRID_Y[::-1])]
    try:
        for n, (x, y) in enumerate(order, 1):
            z = round(z_floor(x, y) + TIP_ABOVE_FLOOR, 4)
            ik.move_to(FOLLOWER_ID, Target(x, y, HOVER_Z, 0.0), execute=True, gripper=0.0)
            ik.move_to(FOLLOWER_ID, Target(x, y, z, 0.0), execute=True, gripper=0.0)
            time.sleep(1.2)                      # settle + let the camera stream catch up
            p = ik.follower(FOLLOWER_ID)["pose"]["position"]
            img = f"p{n}.jpg"
            (out / img).write_bytes(ik.camera_image(OVERVIEW_ID))
            points.append({"n": n, "cmd": [x, y, z], "meas": [round(p["x"], 4), round(p["y"], 4), round(p["z"], 4)],
                           "image": img})
            print(f"point {n}/9  cmd ({x:.3f}, {y:+.3f}, {z:.3f})  meas ({p['x']:.3f}, {p['y']:+.3f}, {p['z']:.3f})",
                  flush=True)
            ik.move_to(FOLLOWER_ID, Target(x, y, HOVER_Z, 0.0), execute=True, gripper=0.0)
    finally:
        r = ik.move_to(FOLLOWER_ID, safety.REST_POSE, execute=True, gripper=0.0)
        print("rest:", r["motion"]["status"], flush=True)
        (out / "points.json").write_text(json.dumps(points, indent=1))
    return out


def fit_homography(px: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """DLT with normalisation. px, xy: (N, 2). Returns H (3x3) mapping pixel -> table xy."""
    def norm(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        c = a.mean(0)
        s = np.sqrt(2) / np.mean(np.linalg.norm(a - c, axis=1))
        T = np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1]])
        return (np.c_[a, np.ones(len(a))] @ T.T)[:, :2], T
    a, Ta = norm(px)
    b, Tb = norm(xy)
    rows = []
    for (u, v), (x, y) in zip(a, b):
        rows.append([-u, -v, -1, 0, 0, 0, u * x, v * x, x])
        rows.append([0, 0, 0, -u, -v, -1, u * y, v * y, y])
    _, _, vt = np.linalg.svd(np.array(rows))
    Hn = vt[-1].reshape(3, 3)
    H = np.linalg.inv(Tb) @ Hn @ Ta
    return H / H[2, 2]


def apply_h(H: np.ndarray, u: float, v: float) -> tuple[float, float]:
    p = H @ np.array([u, v, 1.0])
    return float(p[0] / p[2]), float(p[1] / p[2])


def fit_and_save(folder: Path) -> dict:
    pts = json.loads((folder / "points.json").read_text())
    clicks = json.loads((folder / "clicks.json").read_text())
    use = [p for p in pts if str(p["n"]) in clicks]
    px = np.array([clicks[str(p["n"])] for p in use], float)
    # Fit to the COMMANDED x, y, not the measured pose: the agent later commands what locate returns, so this
    # maps pixel -> "command that puts the jaw tips there". Any repeatable error of the arm's own kinematics
    # (offsets, sag, a slipped joint) is absorbed; the arm does not need to know where it really is.
    xy = np.array([p["cmd"][:2] for p in use], float)
    H = fit_homography(px, xy)
    err = [np.hypot(*(np.array(apply_h(H, *q)) - r)) for q, r in zip(px, xy)]
    res = {"camera_id": OVERVIEW_ID, "H_pixel_to_xy": H.tolist(), "image_size": [299, 224],
           "points": len(use), "rms_mm": round(float(np.sqrt(np.mean(np.square(err)))) * 1000, 1),
           "max_mm": round(float(max(err)) * 1000, 1), "source": str(folder), "time": time.strftime("%Y-%m-%d %H:%M"),
           "z_note": "fitted at tip height ~12 mm above the table, pixel -> commanded x, y"}
    OUT_FILE.parent.mkdir(exist_ok=True)
    OUT_FILE.write_text(json.dumps(res, indent=1))
    return res


PAGE = """<!doctype html><meta charset=utf-8><title>Overview calibration</title>
<style>body{font:15px system-ui;margin:16px;background:#111;color:#eee}img{cursor:crosshair;image-rendering:pixelated;border:1px solid #555}
#msg{margin:8px 0;font-weight:600}button{font:inherit;padding:4px 10px}</style>
<div id=msg></div><img id=im><div><button onclick="skip()">skip this point</button></div>
<script>
let pts=[],i=0,clicks={};const S=%SCALE%;
fetch('points').then(r=>r.json()).then(p=>{pts=p;show()});
function show(){if(i>=pts.length){fetch('done',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(clicks)})
 .then(r=>r.json()).then(r=>{document.getElementById('msg').textContent='Saved. RMS error '+r.rms_mm+' mm, max '+r.max_mm+' mm. You can close this page.';
 document.getElementById('im').style.display='none'});return}
 document.getElementById('msg').textContent='Point '+(i+1)+' of '+pts.length+': click the CENTER of the closed jaw tips.';
 const im=document.getElementById('im');im.src='img/'+pts[i].image;im.onload=()=>{im.width=im.naturalWidth*S}}
document.getElementById('im').onclick=e=>{const r=e.target.getBoundingClientRect();
 clicks[pts[i].n]=[(e.clientX-r.left)/S,(e.clientY-r.top)/S];i++;show()};
function skip(){i++;show()}
</script>"""


def click_page(folder: Path) -> None:
    app = Flask(__name__)
    done = threading.Event()

    @app.get("/")
    def index():
        return PAGE.replace("%SCALE%", str(SCALE))

    @app.get("/points")
    def points():
        return send_file(folder / "points.json")

    @app.get("/img/<name>")
    def img(name: str):
        return send_file(folder / Path(name).name)

    @app.post("/done")
    def finish():
        (folder / "clicks.json").write_text(json.dumps(request.get_json()))
        res = fit_and_save(folder)
        print(f"\nSaved {OUT_FILE}: {res['points']} points, RMS {res['rms_mm']} mm, max {res['max_mm']} mm", flush=True)
        done.set()
        return jsonify(res)

    threading.Thread(target=lambda: app.run(host="127.0.0.1", port=PORT, debug=False, use_reloader=False),
                     daemon=True).start()
    print(f"\nClick page: http://127.0.0.1:{PORT}  (9 clicks, then this script ends)", flush=True)
    done.wait()
    time.sleep(1.0)


def main() -> None:
    if len(sys.argv) >= 3 and sys.argv[1] == "--click":
        folder = Path(sys.argv[2]).resolve()
    else:
        folder = capture()
    click_page(folder)


if __name__ == "__main__":
    main()

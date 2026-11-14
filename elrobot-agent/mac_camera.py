"""Serves the MacBook's built-in camera as JPEG on 127.0.0.1:5060 (third, side-view camera).

Start it from the macOS Terminal app (NOT the terminal inside Claude): macOS only gives
camera access to apps you allowed, and Terminal already has it for Station.

    cd ~/Developer/leukerbad-hackathon/elrobot-agent && uv run python mac_camera.py

    GET /frame.jpg        newest frame (downscaled to MAX_WIDTH)
    GET /info             which camera index is used, its size, and the indexes found
    GET /probe/<i>.jpg    startup snapshot of camera index i (to tell the cameras apart)
    GET /select/<i>       switch to camera index i without restarting

Camera choice: MAC_CAMERA_INDEX in .env, or automatic = the camera with the largest
picture (the FaceTime camera is 1080p, the kit cameras are smaller).
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import cv2
from dotenv import load_dotenv
from flask import Flask, Response, abort, jsonify

HERE = Path(__file__).resolve().parent
load_dotenv(HERE / ".env")

HOST, PORT = "127.0.0.1", 5060          # ADJUST: keep 127.0.0.1
MAX_WIDTH = 1024                        # ADJUST: frames are downscaled to this width
INDEX = os.environ.get("MAC_CAMERA_INDEX", "auto").strip()   # ADJUST: "auto" or 0, 1, 2 ...

app = Flask(__name__)
_lock = threading.Lock()
_latest: tuple[float, bytes, int, int] | None = None   # (time, jpeg, width, height)
_chosen = {"index": None, "width": 0, "height": 0}


def _open(index: int) -> cv2.VideoCapture | None:
    cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        return None
    for _ in range(10):                 # let exposure settle, make sure frames arrive
        ok, _frame = cap.read()
        if ok:
            return cap
        time.sleep(0.05)
    cap.release()
    return None


_probes: dict[int, bytes] = {}          # one snapshot per camera index, taken at startup
_cap: cv2.VideoCapture | None = None
_switch_to: int | None = None           # set by /select/<i>, applied by the capture loop


def _thumb(frame) -> bytes:
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (640, int(h * 640 / w)), interpolation=cv2.INTER_AREA)
    return cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()


def pick_camera() -> cv2.VideoCapture:
    """Probe every camera once (snapshot for /probe/<i>.jpg), then open the chosen one."""
    sizes: dict[int, int] = {}
    for i in range(5):
        cap = _open(i)
        if cap is None:
            continue
        ok, frame = cap.read()
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if ok:
            _probes[i] = _thumb(frame)
            sizes[i] = w * h
            print(f"  camera index {i}: {w}x{h}")
    if not sizes:
        raise SystemExit("No camera could be opened. Allow camera access for this terminal app in "
                         "System Settings > Privacy & Security > Camera, then start again.")
    if INDEX != "auto" and int(INDEX) in sizes:
        choice = int(INDEX)
    else:
        choice = max(sizes, key=lambda i: (sizes[i], i))     # largest picture; on a tie the highest index
    cap = _open(choice)
    if cap is None:
        raise SystemExit(f"Camera index {choice} could not be reopened.")
    _chosen["index"] = choice
    return cap


def capture_loop(cap: cv2.VideoCapture) -> None:
    global _latest, _switch_to
    while True:
        if _switch_to is not None:
            target, _switch_to = _switch_to, None
            cap.release()
            new = _open(target)
            if new is None:
                print(f"camera index {target} could not be opened, going back to {_chosen['index']}")
                new = _open(_chosen["index"])
            else:
                _chosen["index"] = target
                print(f"switched to camera index {target}")
            cap = new
            with _lock:
                _latest = None
            if cap is None:
                time.sleep(1.0)
                cap = _open(_chosen["index"]) or cap
                continue
        if cap is None:
            time.sleep(1.0)
            cap = _open(_chosen["index"])
            continue
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.2)
            continue
        h, w = frame.shape[:2]
        if w > MAX_WIDTH:
            frame = cv2.resize(frame, (MAX_WIDTH, int(h * MAX_WIDTH / w)), interpolation=cv2.INTER_AREA)
            h, w = frame.shape[:2]
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            with _lock:
                _latest = (time.time(), buf.tobytes(), w, h)
            _chosen["width"], _chosen["height"] = w, h
        time.sleep(0.1)                 # about 10 fps is plenty


@app.get("/probe/<int:index>.jpg")
def probe(index: int):
    """Startup snapshot of one camera index, to find out which index is which camera."""
    if index not in _probes:
        abort(404)
    return Response(_probes[index], mimetype="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/select/<int:index>")
def select(index: int):
    """Switch the served camera at runtime (no restart needed)."""
    global _switch_to
    if index not in _probes:
        abort(404)
    _switch_to = index
    return jsonify(switching_to=index)


@app.get("/frame.jpg")
def frame():
    with _lock:
        latest = _latest
    if latest is None or time.time() - latest[0] > 3.0:
        abort(503)
    return Response(latest[1], mimetype="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/info")
def info():
    with _lock:
        age = None if _latest is None else round(time.time() - _latest[0], 2)
    return jsonify(index=_chosen["index"], width=_chosen["width"], height=_chosen["height"], age_s=age,
                   available=sorted(_probes))


if __name__ == "__main__":
    print("Opening the MacBook camera...")
    camera = pick_camera()
    print(f"Using camera index {_chosen['index']}. Serving on http://{HOST}:{PORT}/frame.jpg  (Ctrl+C stops)")
    threading.Thread(target=capture_loop, args=(camera,), daemon=True).start()
    app.run(host=HOST, port=PORT, debug=False, threaded=True)

"""Camera re-alignment: compare the live overview view with the 20:15 reference photo.

    uv run python cam_align.py           # one check: writes photos/align_check.jpg and prints the offset
    uv run python cam_align.py --save    # if aligned well enough: save a calibration mapped through the
                                         # live->reference table homography (no arm motion needed)

The reference view belongs to the 21-point calibration (logs/calib_overview_20261115_200926, RMS 4.2 mm).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import httpx
import numpy as np

HERE = Path(__file__).resolve().parent
REF = HERE / "photos" / "20261115_201516_cam0_43640b79a54e71f45e0443ffbb92880a.jpg"   # ADJUST: reference view
REF_CALIB = HERE / "logs" / "calib_overview_20261115_200926"                           # its calibration photos
OUT_FILE = HERE / "calib" / "overview_homography.json"
CAM = "28ff0ca87978e8f993c2ac64a84d188a"
IK = "http://127.0.0.1:8020/api/v1"


def live() -> np.ndarray:
    jpg = httpx.get(f"{IK}/cameras/{CAM}/image", timeout=5).content
    return cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)


def table_mask(img: np.ndarray) -> np.ndarray:
    """Only the table (lower part of the frame, without the robot on the right) carries static features."""
    h, w = img.shape[:2]
    m = np.zeros((h, w), np.uint8)
    m[int(h * 0.42):, : int(w * 0.70)] = 255
    return m


def match(cur: np.ndarray, ref: np.ndarray) -> tuple[np.ndarray | None, int]:
    orb = cv2.ORB_create(4000)
    g1, g2 = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY), cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    k1, d1 = orb.detectAndCompute(g1, table_mask(cur))
    k2, d2 = orb.detectAndCompute(g2, table_mask(ref))
    if d1 is None or d2 is None:
        return None, 0
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d1, d2, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < 12:
        return None, len(good)
    a = np.float32([k1[m.queryIdx].pt for m in good])
    b = np.float32([k2[m.trainIdx].pt for m in good])
    H, inl = cv2.findHomography(a, b, cv2.RANSAC, 4.0)
    return H, int(inl.sum()) if inl is not None else 0


def main() -> int:
    ref = cv2.imread(str(REF))
    cur = live()
    if cur.shape != ref.shape:
        cur = cv2.resize(cur, (ref.shape[1], ref.shape[0]))
    H, inliers = match(cur, ref)
    blend = cv2.addWeighted(cur, 0.5, ref, 0.5, 0)
    h, w = ref.shape[:2]
    if H is None:
        msg = f"NO MATCH ({inliers} features): the view is too different, move the camera closer to the reference"
    else:
        grid = np.float32([[x, y] for x in np.linspace(80, w * 0.65, 5) for y in np.linspace(h * 0.5, h - 60, 4)])
        moved = cv2.perspectiveTransform(grid[None], H)[0]
        shift = np.linalg.norm(moved - grid, axis=1)
        mean = moved.mean(0) - grid.mean(0)
        for p, q in zip(grid, moved):
            cv2.arrowedLine(blend, tuple(map(int, p)), tuple(map(int, q)), (0, 0, 255), 3, tipLength=0.3)
        msg = (f"inliers {inliers}, table points shift mean {shift.mean():.0f} px, max {shift.max():.0f} px; "
               f"move the picture {'left' if mean[0] > 0 else 'right'} {abs(mean[0]):.0f} px and "
               f"{'up' if mean[1] > 0 else 'down'} {abs(mean[1]):.0f} px")
    cv2.putText(blend, msg[:90], (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    out = HERE / "photos" / "align_check.jpg"
    cv2.imwrite(str(out), blend)
    print(msg)
    print(f"overlay: {out}")
    if "--save" in sys.argv and H is not None:
        cal = json.loads((REF_CALIB / "calib_result.json").read_text()) if (REF_CALIB / "calib_result.json").exists() else None
        import calib_overview as co
        base = co.fit_and_save(REF_CALIB)          # the 21-point reference calibration (pixel_ref -> x, y)
        Href = np.array(base["H_pixel_to_xy"])
        Hnew = Href @ H                             # pixel_live -> pixel_ref -> x, y
        Hnew = Hnew / Hnew[2, 2]
        base.update({"H_pixel_to_xy": Hnew.tolist(), "time": time.strftime("%Y-%m-%d %H:%M"),
                     "note": f"20:14 calibration mapped through a live->reference table homography ({inliers} inliers)"})
        OUT_FILE.write_text(json.dumps(base, indent=1))
        print(f"saved {OUT_FILE} (reference RMS {base['rms_mm']} mm + realignment)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

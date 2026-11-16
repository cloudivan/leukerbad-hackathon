"""Standard demo tests: run each task with the agent (Opus), record the overview camera as MP4, write a results table.

    uv run python demo_run.py            # all tests
    uv run python demo_run.py 2          # only test 2

Output: demo_runs/<time>/test<N>_<name>.mp4, results.md, results.csv. E-stop: 12 V follower plug.
"""
from __future__ import annotations

import csv
import glob
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import httpx
import numpy as np

HERE = Path(__file__).resolve().parent
IK = os.environ.get("IK_URL", "http://127.0.0.1:8020").strip() + "/api/v1"
CAM = "28ff0ca87978e8f993c2ac64a84d188a"      # overview camera
FPS = 5                                        # Station delivers ~5 fps (frame-skip 5)
MODEL = os.environ.get("CLAUDE_MODEL", "claude-opus-5-5")

# ---------------------------------------------------------------- ADJUST ----
TESTS = [
    {"name": "battery_to_half_cup",
     "object": "the AA battery (small cylinder, about 5 cm long, 1.4 cm thick), lying among other clutter on the table; "
               "if it lies flat grip it across its narrow width at its middle at z_floor + 0.008, turning the yaw to match",
     "target": "the half cup (a short cut-off cup with a black circle drawn inside it; drop over the black circle)",
     "env": {"TARGET_TOP_Z": "0.040",         # ADJUST: half cup rim height (4 cm). Keep the cup within ~0.25 m of the base
             "GRIP_ABOVE_FLOOR": "0.008"}},
    {"name": "paper_ball_to_brown_box",
     "object": "the white paper ball (crumpled paper), lying among other clutter on the table",
     "target": "the brown box (flat brown cardboard)",
     "env": {"TARGET_TOP_Z": "0.004", "GRIP_ABOVE_FLOOR": "0.012", "EMPTY_JAWS": "0"}},
]
# ----------------------------------------------------------------------------


class Recorder:
    """Polls the overview camera and writes an MP4 (OpenCV, mp4v)."""

    def __init__(self, path: Path) -> None:
        self.path, self.stop_evt, self.frames = path, threading.Event(), 0
        self.t = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        writer = None
        with httpx.Client(timeout=3.0) as c:
            nxt = time.time()
            while not self.stop_evt.is_set():
                try:
                    r = c.get(f"{IK}/cameras/{CAM}/image")
                    img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
                except Exception:
                    img = None
                if img is not None:
                    if writer is None:
                        h, w = img.shape[:2]
                        writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
                        size = (w, h)
                    if (img.shape[1], img.shape[0]) != size:
                        img = cv2.resize(img, size)
                    cv2.putText(img, datetime.now().strftime("%H:%M:%S"), (10, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    writer.write(img)
                    self.frames += 1
                nxt += 1.0 / FPS
                time.sleep(max(0.0, nxt - time.time()))
        if writer is not None:
            writer.release()

    def __enter__(self):
        self.t.start()
        return self

    def __exit__(self, *exc):
        self.stop_evt.set()
        self.t.join(timeout=5)


def summarize(log: Path) -> dict:
    L = [json.loads(x) for x in log.open()]
    t0, t1 = datetime.fromisoformat(L[0]["ts"]), datetime.fromisoformat(L[-1]["ts"])
    tools = [d for d in L if d.get("kind") == "tool"]
    res = " ".join(str(d.get("result", "")) for d in tools)
    final = next((d["text"] for d in reversed(L) if d.get("kind") == "assistant"), "")
    return {"seconds": round((t1 - t0).total_seconds()), "tool_calls": len(tools),
            "picks": sum(d["tool"] == "pick" for d in tools), "places": sum(d["tool"] == "place" for d in tools),
            "held": res.count("HELD"), "missed": res.count("MISSED"), "final": final.replace("\n", " ")[:400]}


def main() -> int:
    only = {int(a) for a in sys.argv[1:] if a.isdigit()}
    out = HERE / "demo_runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True)
    rows = []
    for i, t in enumerate(TESTS, 1):
        if only and i not in only:
            continue
        print(f"\n=== Test {i}: {t['name']} ({MODEL}) ===", flush=True)
        before = set(glob.glob(str(HERE / "logs" / "session_*.jsonl")))
        env = {**os.environ, **t["env"], "CLAUDE_MODEL": MODEL}
        video = out / f"test{i}_{t['name']}.mp4"
        with Recorder(video) as rec:
            time.sleep(1.0)                          # a second of the scene before the arm moves
            p = subprocess.run([str(HERE / "run_place.sh"), t["object"], t["target"]], env=env,
                               stdin=subprocess.DEVNULL, cwd=HERE)
            time.sleep(1.5)
        new = sorted(set(glob.glob(str(HERE / "logs" / "session_*.jsonl"))) - before)
        s = summarize(Path(new[-1])) if new else {}
        row = {"test": i, "name": t["name"], "model": MODEL, "exit": p.returncode, "frames": rec.frames,
               "video": video.name, **s}
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    with (out / "results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["test"])
        w.writeheader()
        w.writerows(rows)
    md = ["| # | Test | Model | Time | Tool calls | Picks | HELD | MISSED | Agent's verdict |",
          "|---|---|---|---|---|---|---|---|---|"]
    md += [f"| {r['test']} | {r['name']} | {r['model']} | {r.get('seconds')} s | {r.get('tool_calls')} | "
           f"{r.get('picks')} | {r.get('held')} | {r.get('missed')} | {r.get('final', '')[:160]} |" for r in rows]
    (out / "results.md").write_text("\n".join(md) + "\n")
    print(f"\nResults: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

# Leukerbad Robotics Hackathon: ElRobot setup

Setup notes, config and a small helper tool for running a **NormaCore ElRobot** leader + follower arm pair on a MacBook with **NormaCore Station**.

Leukerbad, Switzerland, 14 to 16 November 2026. Track: "Sustainable City".

## What is in here

| File | What it is |
|---|---|
| `SETUP.md` | Step-by-step guide: wiring, calibration, teleop, cameras, troubleshooting |
| `ELROBOT.md` | Kit inventory, joint map with URDF limits, electrical notes, software options |
| `HARDWARE.md` | Other hackathon hardware (reference only) |
| `station.yaml` | Station config (ST3215 servo bus, USB cameras, inference queue) |
| `overlay/calib_overlay.js` | "Calibration coach": a live panel on top of the Station web page |
| `CLAUDE.md` | Instructions for the Claude Code co-pilot used in this project |

## What you need

- ElRobot leader arm (pistol handle, 7.4 V servos) and follower arm (gripper + camera, 12 V servos)
- 2 driver boards, 2 USB data cables, USB hub, 2 USB cameras
- **5 V power supply for the leader, 12 V power supply for the follower**
- 4 clamps (2 per arm)
- macOS on Apple Silicon, Python 3 (only for the overlay)
- NormaCore Station: `station-macos-arm64.dmg` from the releases of [norma-core/norma-core](https://github.com/norma-core/norma-core) (tested with v0.1.0-beta.10)

## Safety first

- **Never plug 12 V into the leader.** It burns the 7.4 V servos. Read the OUTPUT line on the power supply label.
- Clamp both bases to the table before powering anything.
- Keep one hand on the follower's 12 V plug while it moves. Pulling it is the emergency stop.
- Give the cables slack and keep the arm clear of the table.

## How to use it

### 1. Start Station

Install the app from the dmg, then run the binary from this folder (do not open the GUI app, it starts a second Station on the same ports):

```bash
cd leukerbad-hackathon
"/Applications/NormaCore Station.app/Contents/Resources/bin/station" --web --tcp
```

- Web UI: http://localhost:8889
- TCP for the Python client: `localhost:8888`
- Station reads `station.yaml` from the current folder and writes its recordings to `station_data/`.

### 2. Wire and power the arms

1. Mac -> USB hub -> both driver boards and both cameras. Turn on every used hub port.
2. 5 V supply -> leader board. 12 V supply -> follower board.
3. Open http://localhost:8889. You should see **2 bus cards**, each with motors **M1..M8, STATUS OK**, plus 2 cameras.

Each arm shows up under its board's serial number. Ours:

| Arm | Serial | Port |
|---|---|---|
| Leader | `5B61034477` | `/dev/cu.usbmodem5B610344771` |
| Follower | `5B79015365` | `/dev/cu.usbmodem5B790153651` |

Your serials will differ. To find out which is which, move one arm by hand and watch which card's POS values change.

### 3. Calibrate both arms

Calibration teaches Station the min and max position of every joint. Do it once per arm.

1. Bus card dropdown on "(Self-controlled)", so the arm is limp.
2. Put the arm in a middle pose, every joint about mid-travel.
3. Click **Calibrate**.
4. Move the arm by hand until every joint M1..M8 (including the trigger or gripper) has touched both of its ends. Order does not matter.
5. Click **Save**.

Station also has **Auto Calibrate**, which drives each joint until it stalls. It works, but it does not know about the table or the cables, so stay at the plug and stop it if something collides. A collision gets saved as a joint limit.

### 4. Calibration coach (optional)

The coach is a read-only panel on top of the Station page. It shows which joint is moving right now, how far each joint has travelled, and whether Station has recorded a usable range for all 8 joints ("OK TO SAVE" or "DO NOT SAVE YET"). It sends nothing to the robot.

1. Put your own serials into `ARMS` at the top of `overlay/calib_overlay.js`.
2. Serve the file:
   ```bash
   cd overlay && python3 -m http.server 8891 --bind 127.0.0.1
   ```
3. Open http://localhost:8889, open the browser console (DevTools) and paste:
   ```js
   document.head.appendChild(Object.assign(document.createElement('script'),{src:'http://127.0.0.1:8891/calib_overlay.js?'+Date.now()}))
   ```

If the browser asks you to type "allow pasting" first, do that once and paste again. After a Save, press **Reset** in the coach, because saving shifts the position values.

### 5. Teleop: follower mirrors leader

1. Put both arms in roughly the same pose. Compare the **%** column of both cards: the follower is driven to the leader's percentages, so they should be close on every joint before you start.
2. Hand on the 12 V plug.
3. On the **follower** card, open the dropdown and pick the **leader's serial**. The follower now follows the leader.
4. Move the leader slowly, small motions first.
5. To stop, set the dropdown back to "(Self-controlled)".

### 6. Cameras

On the bus card, pick the gripper camera as MAIN and the table camera as AUX. Station records motors and cameras together and can export to Parquet for training.

## If something fails

See the tables in `SETUP.md`. The most common ones:

| Symptom | Fix |
|---|---|
| No bus card | Hub port switched off, or a charge-only USB cable |
| Bus card but "No motors connected" | Arm power supply not plugged in |
| RANGE does not change while you move a joint | Restart Station, then calibrate again |
| % above 100 or below 0 | That joint's range was recorded too small, calibrate again |
| Follower jumps when mirroring starts | Poses or calibration do not match, see step 5.1 |

## Status

Both arms are detected and calibrated (leader by hand, follower with Auto Calibrate). Teleop and camera recording are the next steps.

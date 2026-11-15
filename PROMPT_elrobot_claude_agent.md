# Prompt for Claude Code: ElRobot + Claude API agent

Paste everything below the line into Claude Code, started in `~/Developer/leukerbad-hackathon`.

---

You are setting up an LLM-controlled ElRobot follower arm on my MacBook (Apple silicon) for a hackathon demo. Read `CLAUDE.md`, `SETUP.md` and `ELROBOT.md` in this folder first and follow their safety rules. Never use the em dash character (U+2014) in any file or message.

## Goal
Claude (via the Anthropic API) looks at the camera, reasons, and moves the ElRobot follower through the official NormaCore layer `software/station/examples/elrobot-ik-api` (IK, collision check, HTTP API on port 8020). I only want to paste my API key into one file and run one command.

## Hard rules
1. During setup, DO NOT move the robot. Only solve-only requests (`"execute": false`). Any request with `"execute": true`, `/gripper` or `/torque` needs my explicit "go" in the chat first.
2. Never call `POST /torque {"enabled": false}` (the arm falls). Do not expose torque to the LLM at all.
3. Never touch calibration, EEPROM or Station settings. NEVER suggest "Auto Calibrate" (it collided with the table here). Calibration is manual, see `SETUP.md` section 3.
4. Do not start a second Station. It is already running in another terminal: `station --web --tcp` (web UI http://localhost:8889, WebSocket ws://localhost:8889/api). If http://localhost:8889 is not reachable, stop and tell me.
5. Drive only the follower. Known bus serials: follower `5B79015365`, leader `5B61034477`. Always verify with `GET /api/v1/arms` (`role` field) and refuse to move an arm whose role is not `follower`.
6. Bind every server you start to 127.0.0.1 (hackathon WiFi is open).

## Step 1: get and build the IK API
- `brew install uv` if `uv` is missing. Python must be 3.12 (placo has no 3.13 wheels): `uv python install 3.12`.
- Full clone (NOT sparse, the build copies `hardware/elrobot/simulation/` and `protobufs/` from the repo root):
  `git clone -b dev https://github.com/norma-core/norma-core.git ~/Developer/norma-core`
- `cd ~/Developer/norma-core/software/station/examples/elrobot-ik-api && uv sync && uv run pytest`. All tests must pass (no hardware needed). Read its `README.md` and `AGENTS.md` fully.

## Step 2: project folder `~/Developer/leukerbad-hackathon/elrobot-agent/`
Create:

1. `.env` (gitignored) with exactly:
   ```
   ANTHROPIC_API_KEY=PASTE_YOUR_KEY_HERE
   CLAUDE_MODEL=claude-sonnet-5-5
   IK_URL=http://127.0.0.1:8020
   FOLLOWER_ID=5B79015365
   DRY_RUN=1
   ```
   Also `.env.example` with the same keys and an empty key. `.gitignore` with `.env`, `photos/`, `.venv/`.

2. `start_ik.sh`: starts the IK API from the norma-core checkout with
   `STATION_URL=ws://localhost:8889/api IK_HOST=127.0.0.1 IK_PORT=8020 MOTION_SPEED=200 MOTION_ACCEL=10 uv run elrobot-ik`.
   Check first that port 8020 is free and Station answers on 8889.

3. `safety.py`: one place for all limits, clearly marked `# ADJUST`:
   - Workspace box for the tool point in the world frame (meters, origin on the base motor axis at table height, X forward, Y left, Z up): x 0.10..0.25, y -0.15..0.15, z 0.03..0.20.
   - Max step per move: 0.05 m in position (Euclidean, measured from the current pose from `GET /arms`).
   - Speed 200, accel 10 (fixed, the LLM cannot raise them).
   - Allowed orientations: vertical gripper only (`roll -1.5708, pitch 0`), LLM chooses `yaw` in -1.5708..1.5708.
   - `REST_POSE`: x 0.15, y 0.0, z 0.12, vertical gripper, yaw 0. Verify it with a solve-only request during setup.
   - Function `check_target(current, target) -> (ok, reason)`. Rejections go back to the LLM as tool results, like my YuMi server did with REJECT_BOX / REJECT_STEP.

4. `ik_client.py`: thin httpx wrapper for `/health`, `/arms`, `/arms/{id}/move-to`, `/arms/{id}/gripper`, `/cameras`, `/cameras/{id}/image`. Map the API errors (`unreachable`, `collision`, `not_calibrated`, `stale_frame`, ...) to short readable strings.

5. `agent.py`: Anthropic Python SDK tool-use loop. Tools for the LLM:
   - `get_state`: follower pose (position, yaw), gripper opening, last motion status.
   - `look(camera)`: grabs a JPEG from the IK API, SAVES it to `photos/YYYYMMDD_HHMMSS_<camera>.jpg`, prints the full path in the terminal, and returns the image to Claude as an image content block. Also overwrite `photos/latest_<camera>.jpg` every time.
   - `move_to(x, y, z, yaw)`: always solve first (`execute: false`), run `check_target`, then execute only if `DRY_RUN=0`. In dry run, return the solve result and say "dry run, not executed".
   - `gripper(open)`: 0..1; after a `stalled` close, automatically re-send about 0.05 wider than the stopped opening (see AGENTS.md).
   - `go_rest`: move to `REST_POSE` through the same checks.
   Behaviour:
   - One command at a time, wait for each HTTP call to return (the newest command wins in the API).
   - System prompt: short AGENTS.md rules (pre-pose 0.03 to 0.05 above the object, straight down, lift before any sideways move, accuracy only 1 to 2 cm, `stalled` on an arm move means stop and look). Task context: "Sustainable City" demo (example task: pick a small object and put it into a bin).
   - CLI: `python agent.py "pick the red cube and put it in the bin"`, plus an interactive mode without an argument. Max 25 tool calls per task.
   - Before the first real execute in a session, print the planned target and wait for me to press Enter.
   - Ctrl+C or any exception: try `go_rest` once (only if DRY_RUN=0), never release torque, then exit.
   - Log every tool call and result to `logs/session_<timestamp>.jsonl`.

6. `viewer.py` (optional, small Flask app on 127.0.0.1:5050): shows `photos/latest_*.jpg`, refreshing every second, plus the last 10 tool calls from the newest log. That is where I watch what Claude sees.

7. `README.md`: the run order below, where the photos are, and how to switch to the MCP route: `claude mcp add --transport http elrobot http://127.0.0.1:8020/mcp`.

Use `uv` for the agent folder as well (`pyproject.toml` with anthropic, httpx, python-dotenv, flask). Complete, runnable code, no pseudo code.

## Step 3: verify without moving (report each result to me)
1. `./start_ik.sh` in a new terminal tab, then `curl -s 127.0.0.1:8020/api/v1/health`.
2. `curl -s 127.0.0.1:8020/api/v1/arms`: show me role, `calibrated`, `torque_enabled` and pose for both arms.
   - If the follower answers `calibrated: false` or 409 `not_calibrated`: STOP. Show `context.motors`. The API needs every motor's range to be FROZEN in Station. Tell me to redo the manual calibration from `SETUP.md` section 3 and freeze the ranges in the Station UI. Do not try to fix it in code.
3. `GET /cameras`, then save one image of each camera to `photos/` and give me the full path.
4. Solve-only `move-to` for `REST_POSE` and for 3 corners of the workspace box. Show me `reachable`, `collision`, `motor_steps`. Shrink the box in `safety.py` if a corner is unreachable, and tell me.
5. Run `python agent.py "look at the table and describe what you see, then plan how you would touch the closest object"` with `DRY_RUN=1` (only works once I pasted the key; if the key is missing, stop and tell me exactly which file and line).

## Step 4: first real motion (only after my "go")
Checklist to show me first: base clamped, follower on 12 V, leader on 5 V, follower bus card in Station set to "(Self-controlled)" (not mirroring), workspace clear, my hand on the 12 V plug. Then set `DRY_RUN=0` and do exactly one `go_rest` move. Report `motion.status`.

## Final report
Short list: what works, what failed, the exact commands to start everything (3 terminal tabs: Station, IK API, agent), the photo folder path, and the one value I still have to fill in.

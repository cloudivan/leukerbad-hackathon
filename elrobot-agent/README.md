# elrobot-agent: Claude drives the ElRobot follower

Claude (Anthropic API) looks through the Station cameras, reasons, and moves the follower through the official
NormaCore `elrobot-ik` API (IK + collision check, HTTP on 127.0.0.1:8020).

## One value to fill in
`elrobot-agent/.env`, line 1: `ANTHROPIC_API_KEY=...`

## Run order (3 terminal tabs)

Tab 1, Station (if it is not running yet; never start a second one):
```bash
cd ~/Developer/leukerbad-hackathon && "/Applications/NormaCore Station.app/Contents/Resources/bin/station" --web --tcp
```

Tab 2, IK API:
```bash
cd ~/Developer/leukerbad-hackathon/elrobot-agent && ./start_ik.sh
```

Tab 3, agent:
```bash
cd ~/Developer/leukerbad-hackathon/elrobot-agent && uv run python agent.py "pick the red cube and put it in the bin"
```
Without an argument it starts in interactive mode.

Optional tab 4, viewer (what Claude sees + last 10 tool calls) at http://127.0.0.1:5050:
```bash
cd ~/Developer/leukerbad-hackathon/elrobot-agent && uv run python viewer.py
```

## Dry run vs live
- `.env` `DRY_RUN=1` (default): every move is solved and safety-checked, nothing moves.
- `DRY_RUN=0`: the arm moves. Before the first motion of a session the agent prints the target and waits for Enter.

Checklist before `DRY_RUN=0`:
1. Follower base clamped to the table.
2. Follower on the 12 V PSU, leader on the 5 V PSU.
3. Follower bus card in Station set to "(Self-controlled)" (not mirroring the leader).
4. Follower calibrated BY HAND (SETUP.md section 3) and every range frozen. Never "Auto Calibrate".
5. Workspace clear, hand on the 12 V plug (e-stop).

## Files
| File | Purpose |
|---|---|
| `.env` | key, model, IK URL, follower id, DRY_RUN (gitignored) |
| `start_ik.sh` | starts the IK API on 127.0.0.1:8020 at speed 200 / accel 10 |
| `safety.py` | settings only (`# ADJUST`): speed/accel, vertical gripper, grip relief, `REST_POSE`. No motion limits since 14.11.2026 |
| `ik_client.py` | httpx wrapper for the IK API, readable error strings, no torque method |
| `agent.py` | Claude tool-use loop: `get_state`, `look`, `move_to`, `gripper`, `go_rest` |
| `viewer.py` | Flask viewer on 127.0.0.1:5050 |

## Where things land
- Photos: `elrobot-agent/photos/YYYYMMDD_HHMMSS_<camera>.jpg`, newest per camera also as `photos/latest_<camera>.jpg`.
- Logs: `elrobot-agent/logs/session_<timestamp>.jsonl` (every tool call and result).

## Safety behaviour
- Only the arm with `role == follower` and id `FOLLOWER_ID` is driven; the role is re-checked before every motion.
- Every move: solve only by the IK service. No box, step, yaw or arm-holding limit in this project, only the IK rules (unreachable / collision / calibration / motor errors go back to Claude as
  tool results), then execute.
- `go_rest` is one IK-solved move to `REST_POSE`.
- Max 40 tool calls per task.
- Ctrl+C or a crash: one `go_rest` attempt (only if the session already moved the arm). Torque is never released
  and is not available to the model. Hard stop = pull the 12 V follower plug (the arm then drops, support it).
- Model requests use the server-side refusal fallback (beta). If the API does not accept it, the agent prints a
  note and continues without.

## If it fails
| Symptom | Cause | Fix |
|---|---|---|
| `API key missing` | placeholder in `.env` | paste the key in `.env` line 1 |
| `ik_api_down` | IK API not running | `./start_ik.sh` in tab 2 |
| `not_calibrated` | a motor range is not frozen in Station | redo SETUP.md section 3, freeze all ranges |
| `stale_frame` / `station_disconnected` | Station stopped or follower bus dropped | check tab 1 and the hub port LED |
| `unreachable` at the edge | vertical gripper reaches about 0.24 m at z 0.03, less when higher | go lower or closer |

## Alternative: MCP route (Claude Code drives the arm directly, no safety.py in between)
```bash
claude mcp add --transport http elrobot http://127.0.0.1:8020/mcp
```
The MCP server also exposes `set_torque`. Do not let an agent call it with `enabled: false`: the arm falls.

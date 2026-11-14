# Hardware Inventory: Leukerbad Robotics Hackathon

Status: 13.11.2026, from photos. `(?)` = identification not confirmed yet.

## 1. Arms

| Hardware | What it is | Open questions |
|---|---|---|
| Franka Research 3 (FR3, 3 kg payload, 855 mm reach) + Franka Hand + enabling device | 7-DoF torque-controlled research arm (FCI capable) | FCI unlocked in Desk? Controller/firmware version? PC with real-time kernel for libfranka? Fallback: Desk web UI or `franky` |
| Flexiv Rizon 4 (48 VDC) on aluminium profile + controller box | 7-DoF force-controlled arm | Gripper available? Flexiv RDK enabled (license/firmware)? Controller IP? |
| Universal Robots arm (UR3/UR5 size (?)) + Schunk gripper, on cart with battery charger | 6-DoF cobot | CB3 or e-Series? Gripper interface (I/O or URCap)? Battery-powered mobile setup? Fastest path: `ur_rtde` |
| SO-ARM101 Pro kits (4 boxes) + ~3 assembled leader/follower pairs (Feetech servos, wrist cam, clamps, spare servo) | LeRobot teleop and imitation learning arms | Pairs calibrated? Servo IDs and baud rate? Training GPU? |

## 2. Hands
- Wuji dexterous hand (case labelled WUJI (?)): DoF, SDK, flange adapter for which arm?
- Second "Dexterous Hand" box (Leadtron (?)): brand, interface (CAN/RS485/USB), Python SDK?

## 3. Mobile and humanoid
- UniX AI dual-arm humanoid on wheeled base (lift column, stereo head, two hands): SDK/ROS2 access, teleop included, battery state, allowed to use without supervision?
- Round mobile base with touchscreen and webcam, telepresence style (?): brand, SDK, ROS2, lidar/mapping?
- Waveshare RaspRover 4WD (RPi 4B/5 host + ESP32, JSON over serial): easy Python control.

## 4. Compute and sensors
- NVIDIA Jetson Orin Nano Dev Kit
- ~5x D-Robotics RDK X5 (BPU for on-device vision) + cases
- 2x Raspberry Pi 5
- 2x Logitech C270 webcams
- Silver finned edge PC (Omni... (?))
- Skylandx MetaCam Lite (handheld lidar/SLAM 3D scanner)
- HoverAir camera drone (?)
- Grey felt case: possibly XR headset (?) -> would enable teleop
- 2x black boxes with Ethernet and blue connectors (motor/robot controllers (?))
- 3x EIII aluminium boxes (power supplies or batteries (?))
- White EIII control box (key switch, PC/peripheral ports, e-stop)
- Separate e-stop
- UN3481 box (lithium batteries)

## Cross-cutting questions for the organisers
1. Which robots can our team book, and for how long?
2. Network: dedicated router/switch for robots? IP ranges?
3. GPU available for training (LeRobot ACT/diffusion)?
4. Who holds enabling devices and e-stops for Franka, Flexiv, UR?
5. Do the hands mount on any arm, or are they standalone?

## Project options (fastest to most ambitious)
1. **SO-101 imitation learning** (LeRobot, ACT, ~50 demos): pick and place in a few hours. Lowest risk.
2. **Vision + LLM pick and place on UR or Franka**: webcam, detection, "pick the red cup" by voice/text.
3. **Hand-tracking teleop**: MediaPipe or headset hand pose to Franka/Flexiv + dexterous hand. Close to my thesis pipeline. Medium risk.
4. **Mobile manipulation**: RaspRover or round base fetches, arm picks. High wow factor, high risk.
5. **MetaCam scan into digital twin** + any arm. Good for pitch, weak live demo.

## Recommendation
- Option 1 as safe baseline (one person, Saturday morning).
- Option 2 or 3 as headline demo on whichever big arm has a working Python API by Saturday noon.
- Gate check before choosing: read joint state from a laptop in one call.

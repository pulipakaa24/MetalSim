# TRON1 wheel-foot: realistic sim and classical controller

Robot: LimX TRON1, wheel-foot variant, serial `WF_TRON1A_445` (the motion controller reports
`WF_TRON1B_445`), software `robot-tron1-r-3.5.10`, 8 EtherCAT actuators (abad, hip, knee, wheel
per side). Researched and built 2026-09-23.

## What runs the robot today

* **Remote Control Mode** (R1 + Right on the remote): LimX's closed controller on the robot's own
  Intel i3 (10.192.1.2), driven by the remote or `request_twist` over `ws://10.192.1.2:5000`. It is
  not documented; LimX states its bipeds walk with sim-to-real RL. The Jetson is not involved.
* **Developer Mode** (R1 + Left, persists across reboots): no preset controller. Your controller
  talks to the drives through `limxsdk-lowlevel` (closed `.so`, Fast-DDS based, Linux amd64 /
  aarch64 / Windows, no macOS) at about 1 kHz: `RobotCmd{mode, q, dq, tau, Kp, Kd}` out,
  `RobotState{q, dq, tau (estimated)}` and `ImuData{acc, gyro, quat}` in. It can run off-board.

Read from this robot (EC-Master config, `GET :8080/get_config_ecm`): torque per amp 2.55-2.63 N m/A
for the legs and 1.2 for the wheels, current limit 3500 (35 A if the unit is 10 mA, which gives
89-92 / 42 N m, just above the 80 / 40 N m software ceilings), `qvel_alpha: 1` (no velocity
filtering on the master).

## Code

| Module | What it is |
|---|---|
| `metalsim/tron1/sdk.py` | limxsdk message types, same fields and motor order |
| `metalsim/tron1/realism.py` | every hardware parameter with its source (`[robot]`, `[limx]`, `[limx-dr]`, `[est]`) |
| `metalsim/tron1/sim.py` | `Tron1Sim`: LimX's MJCF + transport latency, drive model, sensors |
| `metalsim/tron1/classical.py` | model-based controller (no learning) |
| `metalsim/tron1/limx_policy.py` | LimX's open-source RL policy, ported, as a baseline |
| `metalsim/tron1/evaluate.py` | batch evaluation; results in `docs/measurements/tron1_eval.json` |
| `metalsim/tron1/scene.py` | collision terrain from a scanned space (Spatial Station) |
| `metalsim/tron1/view.py` | live viewer, keyboard driving |
| `tests/test_tron1.py` | 10 tests |

Live viewer: `.venv/bin/mjpython -m metalsim.tron1.view [--controller classical|limx]
[--params ideal|nominal|random] [--scene assets/tron1/scenes/spatial_station.npz]`. Arrows drive,
space stops, O / P shove forward / sideways, R resets and reloads the controller code.

## Realism layer (what LimX's own MuJoCo sim does not have)

* Latency: sensor messages delayed, commands delayed with jitter and packet loss; drives hold the
  newest command.
* Drives: mode-0 law on the quantized encoder angle and its 1 kHz finite-difference velocity;
  gain errors, motor strength, torque-speed envelope, current-loop lag.
* Joints: Coulomb friction, viscous damping, rotor armature.
* Sensors: encoder quantization, torque *estimate* with a torque-constant error and noise; IMU
  white noise, bias, bias random walk, mounting misalignment, AHRS tilt wander and yaw drift.
* Body and ground: sensor-pack payload, COM offset, mass and inertia scaling, ground friction,
  tyre softness and width.

Three levels: `ideal` (= LimX's tron1-mujoco-sim), `nominal` (best guess of this robot), `random`
(one robot drawn per episode, ranges at least LimX's training ranges).

Findings that changed the model:

* **Tyre width.** LimX's MJCF has a 1 cm wide wheel disc; the URDF they train on
  (tron1-rl-isaacgym) has a 5 cm one. On the 1 cm disc the robot has almost no roll stiffness and
  the classical controller falls when turning in place. The sim uses 5 cm (random: 3-5 cm);
  `ideal` keeps LimX's 1 cm.
* **Command latency.** LimX trains with 0-20 ms of action delay, but their policy falls in this
  sim from 15 ms of command delay on (8/10 at 15 ms, 10/10 at 20 ms, 0/10 at 10 ms or less).
  Since that policy runs on the real robot, the random range is 0-10 ms. The classical
  controller has no falls up to 30 ms.
* **Command scaling.** LimX's deploy script scales joystick commands differently from training
  (`0.5 x (1.5, 1.0, 0.5)` against training's `(2.0, 2.0, 0.25)`); the port uses training's.

## Classical controller

1. Leg IK: for a ride height, hip/knee put the axles under the whole-robot COM with the base
   level (`Tron1Model.stance`). Per-leg IK levels roll and makes a sideways step (wheels shift
   toward the side the robot tips to).
2. Gravity feedforward with the load split between the legs by the measured tilt (fixed
   half-and-half splitting let the loaded leg sag and roll further), plus a slow integral.
   Turning torque is also fed forward to the legs; balance torque is not (the pendulum leans so
   the ground force passes through the COM; feeding it made the hips deflect 20x more).
3. Balance: wheel-inverted-pendulum LQR with parameters computed from LimX's masses at the
   current stance; position and velocity from wheel encoders, pitch from the IMU plus the leg
   kinematics. A slow balance-point estimator (about 2 s) removes the steady offset a COM or IMU
   error causes, which otherwise turned into drift.
4. Heading: PD on heading from the wheel encoders with gyro damping.

## Results (measured in this sim)

Evaluation: 100 episodes x 20 s per cell; a new velocity command every 2.5 s (vx in [-1, 1] m/s,
yaw rate in [-0.6, 0.6] rad/s, 25 % standing), two shoves of 0.2-0.5 m/s in random directions.

| Controller / sim | Falls | vx RMSE | yaw-rate RMSE | pitch p95 | roll p95 |
|---|---|---|---|---|---|
| classical / ideal (1 cm tyre) | 21 / 100 | 0.27 m/s | 0.29 rad/s | 21 deg | 15 deg |
| classical / nominal | **0 / 100** | 0.27 | **0.09** | 21 | 10 |
| classical / random | **13 / 100** | 0.31 | **0.09** | 23 | 15 |
| LimX RL / ideal | 0 / 100 | 0.28 | 0.17 | 13 | 3 |
| LimX RL / nominal | 0 / 100 | 0.27 | 0.17 | 15 | 3 |
| LimX RL / random | 51 / 100 | 0.41 | 0.20 | 21 | 10 |

Standing still for 55 s (12 robots each):

| | Drift, median (max) | Heading change, median (max) | Wheel torque above 5 Hz |
|---|---|---|---|
| LimX RL / nominal | 1.8 m (2.1) | 76 deg (84) | 0.03 N m |
| classical / nominal | 0.03 m (0.21) | 0.2 deg (0.4) | 0.04 N m |
| classical / random | 0.14 m (0.26) | 0.9 deg (2.2) | 0.07 N m |

LimX's policy only regulates velocity; it drifts and arcs while "standing", as reported on the
real robot with the stock controller (whose internals are not public).

Where each controller breaks, from switching on one randomized group at a time:

* LimX RL falls mostly on low ground friction and heavy payloads (a sensor pack is a payload).
* Classical: its remaining falls are hard sideways shoves (0.5 m/s, or two in quick succession)
  on robots with 2-3 kg of payload, close to the tipping limit of a 0.3 m track. The sideways
  step helps (8 -> 4 falls of 48) but is sensitive: raising its roll-rate gain from 0.05 to
  0.1 makes every robot fall. A whole-body controller (QP) would be the principled replacement.

## Unknowns that measurements would settle

Real command latency, joint friction, motor torque-speed envelope, IMU noise and bias, and the
robot's actual mass and COM with the sensor pack. The cheapest checks: weigh the robot; hang it
in Developer Mode and log a damping-only joint sweep and a free wheel spin-down.

## Spatial Station scene

`~/Downloads/spatial_station.ply` is a colored mesh (18.9 M vertices, 37.4 M faces, metres, +y up);
`Spatial-Station.spz` is a Gaussian splat (8.8 M splats), visual only, not used yet.
`assets/tron1/scenes/spatial_station.npz`: 562 x 596 height field at 5 cm, a flat floor (plane
fit residual 1.1 cm) plus obstacles 4 cm-1.3 m above the local floor; unscanned space is walled.
Open question: one corner's floor is 0.3-0.5 m lower in the scan while its ceiling is not; that
is either a real sunken area or glossy-floor artefacts. `floor="scan"` keeps it if it is real.

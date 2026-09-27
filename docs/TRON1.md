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
2. Gravity feedforward with the load split between the legs by the measured tilt, plus a slow
   leg integral; the turning torque is fed forward to the legs, the balance torque is not (the
   pendulum leans so the ground force passes through the COM).
3. Balance: wheel-inverted-pendulum LQR (or manual gains, `manual_K`) on position error, lean,
   speed error and lean rate. Position and speed come from the wheel encoders (rolling without
   slip), lean from the IMU plus the leg kinematics. The speed term is the position D term.
4. Drive / park: while commanded (or slowing down) only speed is tracked and the position
   anchor follows the robot. It parks once |v| < 5 cm/s for 0.3 s, or 1 s after the command ends
   regardless (a robot with a COM offset creeps and would never count as settled). Parked, a
   capped outer loop (1/s, at most 0.15 m/s) nudges the speed target toward the spot, with a
   2 cm deadband. Heading: yaw-rate control while turning, heading hold from the wheel encoders
   once parked.

Superseded along the way (measured, kept here so they are not retried blindly): a balance-point
integrator (made the standing sway a 7 s oscillation), a heavier LQR position weight (sway 23 ->
6 cm but 18/48 falls against 6/48), a torque-capped position term (14-16/48), holding position
while driving (the robot lagged 0.6 m behind the moving target after a reversal and overshot).

Sensor pack: a generic enclosure box (200 x 160 x 80 mm, on the base top) with a MID-360-sized
cylinder (65 mm x 60 mm) on top, both colliding (a hit counts as a fall). Mass and COM are
independent of the geometry: nominal 1.5 kg at (0, 0, 0.08) m in base_Link, all estimated.
`Tron1Sim.set_pack` changes the real pack live; `ClassicalController.set_pack_belief` changes
what the controller assumes (its model, stance table, LQR and feedforward).

Tuning panel: `metalsim.tron1.view` opens http://127.0.0.1:8777 with a slider for every gain
(`metalsim/tron1/tuner.py`), live telemetry, the gains in use, the balance poles and a 12 s plot;
"Save" writes `runs/tron1_gains.json`, `--gains` loads it.

## Results (measured in this sim)

Evaluation: 100 episodes x 20 s per cell; a new velocity command every 2.5 s (vx in [-1, 1] m/s,
yaw rate in [-0.6, 0.6] rad/s, 25 % standing), two shoves of 0.2-0.5 m/s in random directions.
Tyres: smooth fit of the real tread (below); `ideal` keeps LimX's 1 cm disc.

| Controller / sim | Falls | vx RMSE | yaw-rate RMSE | pitch p95 | roll p95 |
|---|---|---|---|---|---|
| classical / ideal (1 cm disc) | 21 / 100 | 0.28 m/s | 0.32 rad/s | 18 deg | 13 deg |
| classical / nominal | **0 / 100** | 0.25 | **0.08** | 29 | 14 |
| classical / random | **26 / 100** | 0.36 | **0.10** | 34 | 16 |
| LimX RL / ideal | 0 / 100 | 0.28 | 0.17 | 13 | 3 |
| LimX RL / nominal | 0 / 100 | 0.27 | 0.17 | 15 | 5 |
| LimX RL / random | 63 / 100 | 0.39 | 0.22 | 25 | 12 |

Standing still for 55 s (12 robots each):

| | Drift, median (max) | Heading change, median (max) | Wheel torque above 5 Hz |
|---|---|---|---|
| LimX RL / nominal | 1.09 m (1.80) | 88 deg (93) | 0.04 N m |
| classical / nominal | 0.05 m (0.10) | 0.4 deg (0.8) | 0.04 N m |
| classical / random | 0.09 m (0.18) | 1.5 deg (3.9) | 0.08 N m |

Remaining standing sway of the classical controller (nominal, ~10 cm peak to peak, speed
3 cm/s rms) is not the position target (a deadband halves direction reversals but not the
speed): with joint/wheel friction removed speed jitter falls from 3.1 to 1.2 cm/s, and without
the AHRS tilt wander the excursion falls from 10.6 to 6.4 cm. Both are estimated parameters.
Wheel friction compensation (`wheel_friction_comp`) takes 3.1 to 1.9-2.5 cm/s but needs the
real friction; off by default.

Tyres: LimX's wheel mesh has a 50 mm tread with a rounded crown, 129.8 mm radius at the centre
and 115 mm at the edges (LimX's collision models: a 127 mm cylinder, 10 mm wide in the MJCF,
50 mm in the training URDF). Colliding with the mesh's convex hull made the wheel a 54-sided
polygon (up to 6.7 deg between points) and raised jitter ~16x, so the default is a smooth
ellipsoid (129.8 x 55 x 129.8 mm semi-axes) that matches the tread profile within ~2 mm. The
rounded tread is harder for the classical controller than the flat cylinder (random falls
13 -> 26 / 100).

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

## Running on the robot (`metalsim/tron1/limx_bridge.py`)

Deployed to the Jetson with `scripts/tron1_jetson_deploy.sh` (`~/tron1_ctrl`, its own Python
3.12 venv; MuJoCo 3.14.0, limxsdk 4.1.1). On the Jetson: controller step 0.97 ms mean (budget 2 ms
at 500 Hz). Listen-only SDK probe of this robot (2026-09-23): joint state ~2 kHz, IMU and remote
500 Hz each, calibration code 0, IMU quaternion (w, x, y, z) and accelerometer +9.75 m/s^2 up at rest
(same conventions as the sim). Quirks: limxsdk needs `ROBOT_TYPE` set or it aborts; it returns 7
motor names for 8 motors (two glued together), so the order check compares the concatenation.

Modes: DAMP (Kp 0 / Kd 1) -> L1 + Y -> STAND (legs to the stance over 3 s, wheels limp; for a
hung robot) -> L1 + A -> BALANCE (only after the stand-up; wheel torque ramps in over 0.5 s).
L1 + X -> DAMP from anywhere. Faults -> DAMP: state/IMU older than 50 ms, control tick later than
20 ms, tilt > 35 deg (BALANCE) / 60 deg (STAND), EtherCAT/IMU diagnostic errors, not calibrated.
The same Bridge code runs the whole hung-start procedure against the sim (a rope that only pulls
when taut, lowered to the ground) in `simulate_procedure`, tested in `tests/test_tron1.py`.

With the gains saved from the panel (`runs/tron1_gains.json`): 0/60 falls nominal and 7/60
randomized (defaults: 0/60 and 16/60), and with extra command delay 0/24 falls at 10 ms, 2/24 at
20 ms, 24/24 at 30 ms.

## Position hold on the stock controller (`metalsim/tron1/stock_hold.py`)

The stock controller takes only velocity commands (`request_twist {x, y, z}` over the websocket,
as LimX's own `tron1-ss` navigation stack sends at 30 Hz). The low-level SDK stream is published
in controller mode too (probe 2026-09-23, stock controller in its damping state: state ~2 kHz,
IMU 500 Hz, wheel angles included), so an outer loop on the Jetson can hold position:
wheel odometry (forward speed r (mean wheel rate + pitch rate), heading from the IMU yaw) ->
`vx = -1.5 e - 0.4 v`, `wz = -1.5 e_psi - 0.2 yaw_rate` (2 cm / 1 deg deadband, 0.5 m/s and
0.4 rad/s caps) at 30 Hz; silent while the remote's sticks are moved. No integral term.

Sim check, LimX's open RL policy as the stand-in for the stock controller, 30 ms link delay, 55 s
standing, 12 robots per row (drift median (max) over robots that did not fall):

| | Falls | Drift | Heading change |
|---|---|---|---|
| policy alone / nominal | 0/12 | 1.16 m (1.77) | 82 deg (88) |
| + hold / nominal | 0/12 | 0.04 m (0.12) | 2.0 deg (2.8) |
| policy alone / random | 6/12 | 6.3 m (14.5) | 49 deg (87) |
| + hold / random | 5/12 | 0.32 m (3.2) | 2.6 deg (4.4) |

Limits: wheel odometry breaks when the policy spins the tyres (random robot 0: odometry says
5.3 m forward, the robot went sideways); heading from the wheel difference failed the same way
(140 deg), hence IMU yaw. Lateral error is not corrected (a wheeled base cannot move sideways).
Some random robots need more than 0.25 m/s of command just to cancel the policy's creep.
Hardware: dry run on the Jetson reads the stream (static robot: odometry < 1 mm); not yet sent.

## LimX's open policy + hold in Developer Mode (`--controller limx`)

Stock-mode hold findings (2026-09-24, `runs/probe_vx.npz`, ramp probe): the stock controller
ignores `request_twist` |vx| up to ~0.15 (forward) / 0.2 (backward) m/s and tracks above; stops
going backward rebound (+0.3 m/s, pitch 2 -> 7.5 deg); at zero command it creeps 3-7 cm/s forward
and turns ~0.9 deg/s. The deadzone is not documented or configurable, and LimX's open wheel-foot
training has none (`min_norm` unused), so it sits in the closed stock command path.

So in Developer Mode the bridge runs LimX's open policy (`metalsim/tron1/policy_hold.py`) with a
continuous PD hold feeding its velocity command; the remote keeps the bridge's stick deadzone.
Through the bridge, hung start then 55 s standing (12 robots, sim nominal):

| | Drift | Heading change |
|---|---|---|
| policy alone | 2.07 m (3.32) | 109 deg (160) |
| policy + hold, wheel cap 40 N m | 0.12 m (0.18) | <= 1 deg |
| policy + hold, wheel cap 20 N m | 0.62 m (3.54) | 4 deg (115) |

The 20 N m first-session cap starves the policy (trained to 40), so `--controller limx` defaults
to 40. Controller step on the Jetson: 0.25 ms mean, p99 0.56 ms.

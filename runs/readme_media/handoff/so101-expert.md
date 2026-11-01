# Hand-off: so101-expert (2026-10-31, 17:45 to 18:10; continued 18:10 to 19:25 by a second agent)

## Status

Done. Recommendation: **ready to film in the README look** once the film tool can run this controller (Next 2); the
caveats below go into the film's notes.

- **MetalSim (the film backend), 7v4_real, 100 seeds: 98/100 success, 98 up at 6 s, time to lift 2.46 s median
  (mean 2.66).** CPU: 100 / 100 / 2.46 (2.61). Same outcome on 98/100 seeds; time to lift within 0.1 s on 81 of the
  98 both lift. Both MetalSim failures (10032, 10064) run out of time after a grasp slip and a retry.

- **Deployment-style expert written:** `metalsim/render/film/so101_expert.py` `DeployExpert`. It keeps so101Sim's
  waypoints, IK and phases, and streams absolute goal positions at LeRobot's 30 Hz. The port is bitwise equal to
  so101Sim's IK step (`check_port.py`).
- **On `7v4_real`, CPU, 100 seeds: 100/100 success, 100/100 still up at 6 s, time to lift 2.46 s median (2.14-5.52).**
  The old incremental expert on the same model and harness: 0/100 (so101-servo).
- **No parameter was tuned.** The first full run gave this result. The later re-run only added logging of the
  transition reasons, and its summary is identical.
- **MetalSim eval:** job `so101-expert-metalsim` (render), re-queued at 18:10 after the first agent's entry died with
  it; it ran 19:19:40-19:19:48. Log: `scratch/so101-expert/logs/metalsim_job.log`; output
  `scratch/so101-expert/metalsim/deploy_7v4_real/`.
- Previews rendered and looked at (by both agents).
- **Smoothness and sag against real arms:** done (Results). The servo model sags 5-6 times more than real followers
  at the shoulder at the same gravity torque; elbow and wrist flex agree within about 30 %.

## Changed

- `metalsim/render/film/so101_expert.py` (new): `DeployExpert`, `LinePlanner`, constants with their sources.
- `runs/readme_media/scratch/so101-expert/` (new, git-ignored):
  - scripts: `model_numbers.py`, `limits.py` (+ `limits.json`), `check_port.py`, `eval_cpu.py`, `eval_metalsim.py`
    (+ `--cpu-dry`), `run_metalsim.sh`, `run_ablations.sh`, `sag.py`, `sim_lag.py`, `real_sag.py`, `contacts.py`;
  - outputs: `cpu/<run>/{episodes.json,smooth.json,traj.npz}`, `metalsim/dry3/`, `logs/`.
  - added by the continuing agent: `real_sag_model.py` (+ `real_sag_model.json`), `real_sag_bins.py`; outputs
    `metalsim/deploy_7v4_real/`, `metalsim/dry100/`; logs `metalsim_job.log`, `dry100.log`, `real_sag.log`,
    `real_sag_model.log`, `real_sag_bins.log`, `sim_lag.log`; check sheets `chk_*.jpg`.
- `runs/readme_media/scratch/ik_preview/` (new files only):
  - `render_deploy.py`;
  - `expert_deploy_7v4_real.mp4` (seeds 10010, 10000, 10051 = a typical outcome: time to lift 2.46 s = the
    median), `expert_old_vs_deploy_7v4_real.mp4` (seed 10010), `expert_deploy_7v4_real_retry10001.mp4` (a retry);
  - strips `deploy_strip.jpg`, `deploy_10051_first3s.jpg`, `old_vs_deploy_strip.jpg`, `deploy_10010_f200.jpg`,
    `deploy_10051_lift_consecutive.jpg`.
- Not touched: README.md, the plan doc, `tasks.py`, `metalsim/render/film/so101.py`, every tracked so101Sim file.

## Verified

### LeRobot evidence (`~/lerobot` at bf31dd79; all read this session)

- **Absolute goal positions every tick.** `src/lerobot/robots/so_follower/so_follower.py:205-230` `send_action` writes
  the action's `<motor>.pos` values with `bus.sync_write("Goal_Position", goal_pos)`. They are absolute positions,
  not deltas.
- **`max_relative_target` is off by default:** `config_so_follower.py:36`, `... | None = None`. When it is set,
  `send_action` reads Present_Position and clamps with `robots/utils.py:93` `ensure_safe_goal_position`.
- **Registers:**
  - P 16, I 0, D 32: `config_so_follower.py:45-47`;
  - gripper `Max_Torque_Limit` 500 (50 %): `so_follower.py:169`.
- **Control rate:**
  - `lerobot-record`: `configs/dataset.py:33` `DatasetRecordConfig.fps = 30`;
  - `lerobot-rollout` (policy deployment): `rollout/configs.py:247` `fps = 30.0`, with `interpolation_multiplier = 1`
    (`:264`);
  - the SO-101 recording example in `docs/source/il_robots.mdx:231`: `FPS = 30`;
  - `lerobot-teleoperate` (no recording): `scripts/lerobot_teleoperate.py:140` `fps = 60`.
  - **30 Hz used.**
  - `record_loop` (`scripts/lerobot_record.py:225-330`) per tick: `get_observation` (Present_Position), then the
    teleop / policy action, then `send_action`.
- **LeRobot's own scripted move.** `rollout/strategies/core.py:173-196` `return_to_initial_position` interpolates
  absolute targets linearly from the present position to the goal, over 3 s at 50 fps, open loop, with no feedback
  and no gravity term. It is the precedent for streaming a planned trajectory of absolute targets.
- **LeRobot's IK pipeline** (`robots/so_follower/robot_kinematic_processor.py:274-372` `InverseKinematicsEEToJoints`,
  `model/kinematics.py:98-143`):
  - IK of the desired end-effector pose, from a guess (current joints or the previous solution), written as absolute
    `<motor>.pos` goals;
  - its error is computed on the kinematic model (placo FK of the guess), never from a measured end-effector, so it
    does not compensate sag. That rests on reading the code; placo itself was not read.
  - `EEBoundsAndSafety` (`:192-251`) rate-limits the commanded end-effector (`max_ee_step_m` 0.05), not relative to
    the measured state.
- **No gravity compensation on the SO arms.** `grep -rni gravity src/lerobot` finds only the Unitree G1's
  (`unitree_g1/config_unitree_g1.py:69` `gravity_compensation = False`). So the expert has none.
- **Units of the datasets used below:** degrees since `6600b60e` ("always use degrees", 2026-02-13); the
  `so_follower` robot type exists since `d791a431` (2026-01-12).

### Controller design (`metalsim/render/film/so101_expert.py`)

Kept from so101Sim (`rl/native/scripted_expert.py` at d4c2990), and checked:

- the waypoints: cube + 10 cm (approach), cube + 8 mm (descend, grasp), (grasp xy, 0.20 m) (lift);
- the IK: damped least squares on the TCP, λ 1e-4, five arm joints, each iteration clipped at DQ_MAX 0.06, wrist-roll
  yaw alignment in approach and descend;
- the phases and their tests: approach → descend at e < 2 cm and yaw < 0.12 rad; descend → grasp at e < 12 mm or
  70 ticks (1.4 s); grasp ≥ 15 ticks (0.3 s); lift, with a retry if the cube is below 7 cm after 25 ticks;
- the TCP and jaw calibration from the `pickup` keyframe, stored as constants (MetalSim's scene has no keyframe).

Check: `check_port.py` (so101Sim venv, MuJoCo 3.10): the constants equal `Expert.__init__`'s, and 4000 IK iterations
over 100 seeds × 20 random poses × align on/off equal `Expert._ik_step` **bitwise** (max diff 0).

Changed, each parameter with its source:

| parameter | value | source |
|---|---|---|
| command | absolute goal per joint, zero-order hold | `send_action` (above) |
| control rate | 30 Hz: tick k on physics step round(k·200/30) (30 / 35 ms) | `fps = 30` (record, rollout) |
| waypoint goal | so101Sim's IK iterated to convergence (≤ 200 iterations, step < 1e-5 rad) on a kinematic copy at the commanded configuration, warm-started from the last goal, re-solved every tick for the perceived cube | the original re-targets every tick; LeRobot's IK step |
| trajectory | straight line in joint space, synchronised trapezoid, re-targeted online (`LinePlanner`) | standard practice; LeRobot's linear interpolation |
| VMAX (rad/s) | 2.72 / 1.85 / 1.53 / 2.26 / 2.67 | `limits.py`: (R / 2b) |
| AMAX (rad/s²) | 21.4 / 14.2 / 14.8 / 28.4 / 34.9 | `limits.py`: (R / 2M) |
| gripper | absolute steps: open 1.4, close -0.15; 50 % torque in the servo model | so101Sim's values; LeRobot `Max_Torque_Limit` 500 |
| phase test after the move | original tolerance on the measured TCP, **or** T_SETTLE = 0.31 s after the command arrived | 7v4_real step response: 2 % settle at the shoulder, 310 ms (so101-servo `step_tables.md`) |
| arrived | command within one encoder step (1.53 mrad) of the goal | STS3215 4096 steps / turn |
| grasp end | after ≥ 0.3 s (the original's), when \|q̇_gripper\| < 0.05 rad/s; timeout 0.88 s | V_HYST of `smoothness.py`; timeout = 1.55 rad / (5.445 / 2 rad/s, 50 % duty) + 0.31 s |
| lift retry | cube < 7 cm after ≥ 0.5 s **and** 0.31 s after the move arrived | the original's test, plus "the move is done" |
| gravity compensation | none | LeRobot has none for the SO arms |
| perception | cube pose from the sim state each tick | as the original (a real script would use a camera) |

**How the limits were derived** (`limits.py` → `limits.json`):

- **Inputs, from 7v4_real:** forcerange 1.912 N·m, Coulomb c 0.0515 N·m, damping b 0.3417 N·m·s/rad.
- **Gravity and inertia:** gravity g_max and inertia M_max (armature included) per joint, along the straight
  segments start → approach → descend → lift. The IK chain is solved for each of 100 seeds; on the lift segment the
  box hangs at the TCP at its heaviest (1.4 × mass, 0.134 kg).
- **Values:** g_max 0 / 0.60 / 0.81 / 0.32 / 0.03 N·m; M_max 0.044 / 0.044 / 0.035 / 0.027 / 0.026 kg·m².
- **Rule:**
  - R = 1.912 − c − g_max is the torque left over gravity;
  - half of R goes to speed (VMAX = R / 2b), half to acceleration (AMAX = R / 2M);
  - at the trapezoid's corner the servo just reaches its duty limit, so the plan is one the servo can follow.
- **The split is not arbitrary:** it is the time-optimal trapezoid under that torque budget for a move of
  D = M·R / b² = 0.32-0.69 rad. The task's descend and lift moves are 0.3-0.55 rad per joint (median); approach moves
  are about 1.2 rad.
- **Firmware caps:** none of the limits reaches them (5.29 rad/s, 39.0 rad/s²).
- **Cross-check against real arms (independent):** real SO-101 teleop joint speeds are p99 1.1-2.6 rad/s, max 2-4
  (datasets below). The deploy expert's are p99 1.3-2.7, max 2.0-3.1.
- **Not derived:** the coupling torques between joints during synchronised motion are ignored.

### Results

Seeds 10000-10099. CPU: so101Sim venv, MuJoCo 3.10, `eval_cpu.py` (so101Sim's `SO101LiftEnv` for the scene, reset and
success test). Table: `python3 table.py` (all rows read this session). Smoothness from the state at every 5 ms physics
step (so101-servo `smoothness.py`: RMS of finite-difference acceleration and jerk; velocity sign reversals per second
with a 0.05 rad/s hysteresis; velocity RMS above 5 Hz = the oscillation energy). Joints pan / lift / elbow / wflex /
wroll / gripper.

| run | success | up at 6 s | time to lift median (mean) s | retries | RMS acc rad/s² | RMS jerk ×10³ rad/s³ | reversals /s | vel RMS > 5 Hz rad/s |
|---|---|---|---|---|---|---|---|---|
| **deploy, 7v4_real, CPU** | **100** | **100** | **2.46 (2.61)** | 6 | 2.1 / 6.6 / 5.3 / 5.8 / 3.3 / 7.6 | 0.33 / 1.03 / 1.00 / 0.78 / 0.12 / 0.89 | 0.36 / 0.79 / 0.50 / 0.55 / 0.29 / 0.93 | 0.018 / 0.042 / 0.035 / 0.037 / 0.011 / 0.079 |
| **deploy, 7v4_real, MetalSim** | **98** | **98** | **2.46 (2.66)** | 12 | 2.1 / 7.2 / 5.5 / 6.3 / 3.6 / 7.6 | 0.33 / 1.33 / 1.09 / 1.04 / 0.16 / 0.47 | 0.39 / 0.81 / 0.53 / 0.59 / 0.32 / 0.60 | 0.017 / 0.046 / 0.035 / 0.039 / 0.013 / 0.087 |
| deploy, 7v4_real, MetalSim harness on CPU MuJoCo 3.14 (`--cpu-dry 100`) | 97 | 97 | 2.46 (2.64) | 10 | | | | |
| deploy, 12v_real, CPU | 100 | 100 | 2.33 (2.37) | 1 | 1.8 / 4.6 / 3.9 / 4.2 / 2.8 / 6.4 | 0.24 / 0.27 / 0.57 / 0.25 / 0.08 / 0.62 | 0.35 / 0.76 / 0.39 / 0.45 / 0.00 / 0.74 | 0.015 / 0.018 / 0.022 / 0.014 / 0.010 / 0.071 |
| deploy, old 7v4 (kp 998), CPU | 91 | 90 | 2.26 (2.60) | 33 | 21 / 28 / 27 / 34 / 34 / 12 | 4.4 / 6.2 / 5.7 / 7.4 / 7.1 / 1.3 | 10.5 / 4.3 / 4.4 / 6.7 / 7.3 / 1.9 | 0.16 / 0.21 / 0.21 / 0.24 / 0.25 / 0.15 |
| ablation: no firmware acceleration limit | 99 | 99 | 2.40 (2.60) | 9 | 2.1 / 6.7 / 5.3 / 5.9 / 3.6 / 7.8 | 0.34 / 1.09 / 1.00 / 0.78 / 0.20 / 0.86 | 0.39 / 0.85 / 0.56 / 0.57 / 0.30 / 0.94 | 0.017 / 0.043 / 0.034 / 0.036 / 0.011 / 0.084 |
| ablation: host latency 10 ms | 99 | 99 | 2.46 (2.63) | 8 | 2.1 / 6.8 / 5.3 / 5.9 / 3.5 / 7.6 | 0.34 / 1.12 / 1.00 / 0.81 / 0.13 / 0.89 | 0.36 / 0.80 / 0.50 / 0.56 / 0.30 / 0.94 | 0.018 / 0.044 / 0.035 / 0.038 / 0.011 / 0.079 |
| ablation: BAM stall 3.41 N·m | 100 | 100 | 2.24 (2.27) | 1 | 2.2 / 4.9 / 4.2 / 4.4 / 3.0 / 7.6 | 0.36 / 0.31 / 0.60 / 0.28 / 0.09 / 0.61 | 0.41 / 0.83 / 0.44 / 0.55 / 0.01 / 0.64 | 0.019 / 0.021 / 0.025 / 0.016 / 0.013 / 0.084 |
| ablation: 50 Hz stream | 99 | 99 | 2.44 (2.56) | 7 | 2.1 / 6.6 / 5.2 / 5.7 / 3.3 / 7.5 | 0.32 / 1.05 / 0.98 / 0.80 / 0.12 / 0.85 | 0.37 / 0.81 / 0.53 / 0.56 / 0.29 / 0.93 | 0.017 / 0.042 / 0.034 / 0.037 / 0.010 / 0.079 |
| old incremental expert, 7v4_real, CPU fresh kin (so101-servo) | 0 | 0 | - | - | 4.9 / 8.5 / 11.6 / 8.5 / 2.9 / 4.3 | 1.43 / 2.50 / 3.36 / 2.48 / 0.44 / 0.27 | 0.21 / 0.29 / 0.36 / 0.88 / 0.23 / 0.18 | 0.027 / 0.050 / 0.066 / 0.049 / 0.025 / 0.034 |
| original PPO, 7v4_real, CPU (so101-servo, arm joints) | 6 | | | | 35-77 | | 2.1-5.9 | 0.28-0.54 |

- **Robust to every inferred piece of 7v4_real:** 99-100/100 under each ablation; no parameter was tuned.
- **Against the failing old expert on the same model** (arm joints, deploy / old): RMS acceleration 0.4-0.8× (wrist
  roll 1.15×), jerk 0.2-0.4×, > 5 Hz velocity 0.4-0.8×, all while doing the whole task. Reversals are higher
  (0.29-0.79 against 0.21-0.88 /s, 1.2-2.7× at pan, lift, elbow): the old expert barely moves after its first droop,
  and the deploy expert's reversals are the task's own direction changes (≤ 0.8 /s, about 5 in 6 s).
- **Against PPO on 7v4_real:** 5-35 times lower RMS acceleration, 3-20 times fewer reversals.
- On the old kp-998 servo the same controller rings (10 reversals/s at the pan): the servo model, not the expert, set
  the old smoothness.
- **Velocity energy above 5 Hz is 0.03-0.6 % of the variance** per joint (`osc_frac`).
- **Peak accelerations** (77-404 rad/s², arm) are single 5 ms steps at 1.0-1.5 s in descend and grasp: contact
  transients of the jaws on the box. p99.9 of |acceleration| is 20-43 rad/s², the firmware cap's order (39).
- **Phase transitions (CPU, 100 episodes, 106 approaches with the retries):** approach → descend on the measured TCP
  79, on the 0.31 s settle timeout 27; descend → grasp measured 99, timeout 7; grasp → lift "jaws stopped" 104,
  grasp timeout 2; lift → retry 6 (seeds 10001, 10024, 10032, 10042, 10069, 10093), all recovered and lifted.
- Time to lift p10 / p50 / p90: 2.24 / 2.46 / 2.80 s.
- **Box tilt in hand at 6 s:** median 29°, p90 43°, max 51°. Not the servo model: the old kp-998 servo gives 42°
  median, 12v_real 40°, the BAM stall 41°. In the frames the box tips while the jaw closes on it on the table
  (seed 10000, 1.6-2.1 s), and hangs at that angle. `contacts.py` (seeds 10000, 10051): in the lift hold the box
  touches only `gripper` and `moving_jaw_so101_v1`, deepest penetration 0.8-1.2 mm, gripper at its 0.986 N·m limit.
  The scene's jaw pads are condim 6 with torsional friction 5e-3 (`so101Sim/mujoco_menagerie/robotstudio_so101/
  so101.xml:24,27`); the box is condim 3 (`scene_box_rl.xml:24`). Whether the box touches the pads or the jaw mesh
  geoms was not checked. **Not established as a defect.**

### MetalSim against CPU

Three runs of 100 seeds: so101Sim's env on CPU MuJoCo 3.10 (`eval_cpu.py`); the film scene
(`assets/so101/scene_box_rl.xml`, `reset_draw`) on MetalSim (mujoco_warp 93e5ce18, MuJoCo 3.14; `eval_metalsim.py`);
the same film-scene harness on CPU MuJoCo 3.14 C (`--cpu-dry 100`, main venv, run with `warp` blocked from import so
nothing touched the GPU; `metalsim/dry100/`, `logs/dry100.log`).

| | CPU 3.10, so101Sim env | MetalSim | CPU 3.14, film scene |
|---|---|---|---|
| success / up at 6 s | 100 / 100 | 98 / 98 | 97 / 97 |
| time to lift median (mean) | 2.46 (2.61) | 2.46 (2.66) | 2.46 (2.64) |
| retries | 6 | 12 | 10 |
| failures | - | 10032, 10064: retry, out of time | 10032 (retry, out of time), 10043 (grasp timeout at 5.63 s), 10095 (NaN, below) |

- Seed agreement: MetalSim vs CPU 3.10 98/100; MetalSim vs CPU 3.14 film scene 97/100, time to lift within 0.1 s on
  85 of 96, median |Δ| 0.00 s. **The 100 → 98 is the film scene / reset / MuJoCo version, not the GPU**: the CPU run of
  the film-scene harness gives 97.
- Retries are the sensitive part: the six CPU 3.10 retry seeds retry on MetalSim too; MetalSim adds 10047, 10064,
  10074, 10088. The grasp slips on some seeds and the outcome then depends on small differences.
- MetalSim solver flags: `LS_ITERATIONS` in 98 worlds; no capacity overflow. `LS_ITERATIONS` is a solver-convergence
  warning (`metalsim/physics/batch.py:436-442` `overflow_flags` docstring); so101-servo saw the same on this scene
  (all 100).
- Smoothness, MetalSim against CPU 3.10, arm joints: RMS acceleration, reversals and > 5 Hz velocity within 10 %;
  RMS jerk up to 1.33× (lift 1.33 against 1.03, wflex 1.04 against 0.78 ×10³ rad/s³; the extra retries add moves);
  gripper jerk and reversals lower (0.47 against 0.89; 0.60 against 0.93). Steady-state sag the same (lift phase:
  32 / 126 / 46 mrad at lift / elbow / wflex, against 33 / 126 / 46).
- **CPU 3.14 instability, seed 10095:** box speed 5.3 m/s at 2.10 s, then MuJoCo's "Nan, Inf or huge value in QVEL
  at DOF 6" (the box free joint) at 2.805 s. On MetalSim and CPU 3.10 the same seed lifts cleanly (box speed max
  0.51 / 0.57 m/s; every run's p99 0.64-0.73). Not investigated; it is off the film path.
- Preview seeds, time to lift CPU 3.10 / MetalSim / CPU 3.14: 10000 2.68 / 2.70 / 2.72; 10010 2.78 / 2.78 / 2.78;
  10051 2.46 / 2.34 / 2.46; 10001 (retry) 4.86 / 4.72 / 4.76.
- Power during the job (the harness's own reading at 19:19:46): AC, charging, 74 %, powermode 2. No timing figure
  was taken.

### Steady-state sag against real arms

**Sim.** `sag.py`: a joint is steady when its command has not changed for 0.31 s, the firmware profile has landed
and |q̇| < 0.05 rad/s; error e = target − q. Deploy, 7v4_real, CPU, median |e| (mean signed), mrad:

| phase | pan | lift | elbow | wflex | wroll |
|---|---|---|---|---|---|
| free (approach, descend; n 44-159) | 4 (−0.5) | 49 (−37) | 82 (−82) | 25 (−26) | 8 (+8) |
| lift (box in hand; n 60-64 k) | 4 (+4) | 33 (−36) | 126 (−125) | 46 (−46) | 9 (+9) |

**Real.** Two LeRobot v3.0 teleop datasets, robot_type `so_follower` (checked in each `meta/info.json`):
`hungho77/so101_left_arm_pick_red_cube_into_pot` (50 episodes, 30 fps) and `IlyaGusev/so101_pick_gum` (104 of 155
episodes, the first data file, 20 fps). Parquets and info.json in this session's scratchpad (`hfdata/`, `hfmeta/`).
`real_sag.py`: steady when action and state each moved < 0.2° per frame over 0.3 s; e = action − state, which also
holds the leader / follower calibration mismatch.

- Raw median |e|, mrad: pan 7-14, lift 12-20, elbow 51-61, wflex 5, wroll 5-8 (mean signed: lift −18 / −24,
  elbow −41 / −50).
- Fontaine's lag metric mean |action[t] − state[t+1]|: real 1.94 / 2.53° (arm mean); sim deploy 3.05°, per joint
  0.85 / 3.86 / 5.75 / 3.15 / 1.66° (`sim_lag.py`). The sim lags more at lift and elbow.
- Joint speed p99, real: 1.1-2.6 rad/s, max 2.1-4.0; sim deploy p99 1.3-2.7, max 2.0-3.1.

**Pose-controlled** (`real_sag_model.py`, `real_sag_bins.py`): each steady real sample is put into the MuJoCo model
(arm alone, qvel 0) and its gravity torque g = qfrc_bias computed. A P-only servo holds still at kp·e = g ± c, so
7v4_real predicts e = g / 5.262 within ± 9.8 mrad. Joint map: sign +1, offset 0 (so101Sim `rl/real/so101_real.py:43`
`DEFAULT_MAP`; the owner's calibrated `rl/real/joint_map.json` also has signs +1 and offsets −0.7 to 8.0°). Rest-pose
samples (shoulder_lift < −85°, arm folded on its stop) dropped. Median e per quintile of g, mrad (model in brackets):

| joint | hungho77 | IlyaGusev |
|---|---|---|
| shoulder_lift, heaviest quintile (g ≈ −0.74 N·m) | −28 (−140) | −23 (−141) |
| shoulder_lift, g ≈ −0.44 to −0.52 | −12 to −16 (−83 to −99) | −22 (−103) |
| elbow_flex, g −0.31 to −0.45 | −50 to −67 (−59 to −85) | −47 to −80 (−59 to −82) |
| wrist_flex, heaviest quintile (g ≈ −0.1) | −13 (−19) | −12 (−17) |

- Method check on the sim's own free-phase steady samples: median |e| 49 / 82 / 25 mrad against |g/kp| 42 / 83 / 18.
  The model formula reproduces the sim.
- **Elbow and wrist flex: the real arms sag about as 7v4_real predicts** (within about 30 %, with an unknown
  calibration offset in each sample).
- **Shoulder lift: the real followers sag 5-6 times less** than 7v4_real at the same gravity torque. Fitted slope of
  e on g: 0.017-0.027 rad/N·m, i.e. kp_eff 37-59 N·m/rad, against the model's 5.26.
- Possible causes, none tested: the TCP resting on the table or object in the extended (heaviest) poses of a pick
  task; a shoulder offset in the joint map that moves g; firmware behaviour not in the model (dead zone, a minimum
  startup force: registers `Minimum_Startup_Force`, `CW_Dead_Zone`, `CCW_Dead_Zone` exist in
  `~/lerobot/src/lerobot/motors/feetech/tables.py:62-64`, values on the datasets' arms unknown); an error in the
  firmware gain for this joint. The elbow agreeing argues against a single wrong gain for all joints.
- The fitted slopes at elbow and wrist flex mean little: g spans only 0.1-0.15 N·m there (r 0.1-0.3). The
  quintile medians above are the comparison. `real_sag_model.py` raw output: `logs/real_sag_model.log`.
- **For the expert:** its sim shoulder sags 33-49 mrad (2-3°) where real arms show 12-28 mrad. That is a servo-model
  question for so101-servo; the expert does not compensate sag (LeRobot has no gravity compensation) and needs none
  to succeed.
- lerobot#3400 (so101-servo `research_others.md:119`): 63 ticks (97 mrad) of settle error at the shoulder_pan with
  P 16, approach-direction dependent. Teleop data here give pan 7-14 mrad median. Not reconciled.

### Previews (looked at by the continuing agent)

`ffprobe`: `expert_deploy_7v4_real.mp4` 1920×480, 50 fps, 300 frames, 6.000 s; `expert_old_vs_deploy_7v4_real.mp4`
1280×480, same timing; `expert_deploy_7v4_real_retry10001.mp4` 640×480, same timing. Frame period from `d.time`
(0.0200 s, `logs/render.log`); each deploy panel's outcome equals `eval_cpu.py`'s (render.log).

Sheets read with the Read tool (`scratch/so101-expert/`):
- `chk_deploy_1hz.jpg`: all three seeds at 0.5, 1.5 … 5.5 s. Approach, grasp, lift; box up from 2.5-3.5 s on; held
  to 6 s; box tilted 25-40° in hand. Labels and times correct.
- `chk_oldvs_1hz.jpg`: old expert droops forward and stays in approach to 6 s; deploy lifts at 2.78 s.
- `chk_10000_first3s.jpg`: seed 10000 at 6 fps, 0-3 s. The arm starts straight out, folds smoothly toward the box,
  descends, the jaw closes and tips the box on the table (1.6-2.1 s), the box comes up at 2.6 s. No visible jitter or
  whip.
- `chk_retry10001.jpg`: the first grasp slips and the box drops back upright (2.8 s); the arm re-approaches, grasps
  and lifts at 4.86 s.

### Tests

`.venv/bin/python -m pytest -q tests/test_film.py tests/test_no_machine_paths.py`: 5 passed, 1 skipped. No machine
paths in `so101_expert.py` (grep).

## Open

- **Shoulder sag 5-6 times the real arms'** (above). Causes not tested. It sets how far the arm droops in a film, not
  whether the expert succeeds (12v_real and BAM stall, both stiffer, also succeed).
- **Box tilt in hand, 29° median**, on every servo model; not established as a defect or as realistic.
- **CPU MuJoCo 3.14 NaN on seed 10095** with the film scene (MetalSim against CPU). Not investigated.
- **Retry sensitivity:** 6 (CPU 3.10) to 12 (MetalSim) retries per 100; which seeds retry depends on the backend.
- **Not modelled by the expert:** camera perception (the cube pose comes from the sim state each tick); joint
  coupling torques in the limit derivation.
- The film tool cannot run this controller yet: `SO101Lift` in `metalsim/render/film/so101.py` drives the PPO policy
  (not my file).
- Seen, not by this stream: `git -C ~/so101Sim status` shows the tracked `rl/tools/jobctl/.dashboard_url` deleted
  (19:25). This stream only read so101Sim files.
- `train_real.py` (PID 15482) was no longer running at 19:18; not touched by this stream.

## Recommendation: ready to film in the README look

- It does the task on the film backend (98/100 MetalSim, 100/100 CPU) with nothing tuned. It is driven the way LeRobot
  drives a real SO-101 (absolute goals at 30 Hz, no gravity compensation), on a servo model with LeRobot's registers.
- Its motion is smooth from the state: arm RMS acceleration 2-7 rad/s², ≤ 0.8 reversals/s, 0.03-0.6 % of velocity
  variance above 5 Hz; joint speeds match real teleop data (p99 1.3-2.7 rad/s against 1.1-2.6).
- The previews look like a real arm: no whip, no jitter; grasps can slip and are retried.
- Say in the film notes: scripted (IK waypoints), not learned; cube pose from the simulator, not a camera; the
  servo model's shoulder sags more than real followers' (2-3° against 0.7-1.6°); the box hangs tilted in the jaws.
- Pick seeds whose MetalSim outcome is known: `metalsim/deploy_7v4_real/episodes.json`.

## Next

1. **Owner review** of `runs/readme_media/scratch/ik_preview/expert_deploy_7v4_real.mp4` and
   `expert_old_vs_deploy_7v4_real.mp4` (real time, MuJoCo's renderer, not the README look).
2. **Film it in the README look** (owner of `so101.py` / `tasks.py`): give `SO101Lift` a controller option that runs
   `DeployExpert` at 30 Hz with the servo layer per physics step. `eval_metalsim.py` is the working pattern: substeps
   1, control tick on physics step round(k·200/30), CPU `MjData` copy + `mj_forward` per tick for the expert, the
   held command through `servo_layer` / `CtrlPlanFeeder`; reset as `SO101Lift.reset` already does. Check the film's
   outcome per seed against `metalsim/deploy_7v4_real/episodes.json`. Good picks: 10051 (typical, lifts at 2.34 s on
   MetalSim), 10010 (2.78 s on all three runs), 10001 (a slip and a retry, 4.72 s).
3. **so101-servo follow-up:** settle the shoulder sag gap on hardware: hold the follower at three extended poses with
   the TCP in free air, log Goal_Position − Present_Position, and compare with g / 5.262 (`real_sag_model.py` gives g
   for any pose).
4. Optional: is the 29° box tilt realistic? Check which geoms touch the box (pads or jaw meshes) and compare with a
   real 4×4×6 cm box grasped by the SO-101 jaws.
5. Optional: reproduce the CPU 3.14 NaN alone (`eval_metalsim.py` `--cpu-dry` on seed 10095 needs the harness to take
   a seed list; it takes a count from 10000 now) and find the contact at 2.10 s.

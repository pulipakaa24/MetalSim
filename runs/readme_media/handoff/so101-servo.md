# Hand-off: so101-servo (2026-10-31, 17:00 to 17:30)

## Status

Done; no RL retraining (brief).

- **Servo options.** New `7v4_real` and `12v_real` model the STS3215 as LeRobot drives it. The old options are
  unchanged: their CPU results reproduce so101-smooth's exactly (expert 20, original 77, isaac_scaled 77,
  isaac_unscaled_raw 86, expert 12 V 95).
- **Ringing.** The new model removes it: 0 % overshoot against 33-89 %. Peak accelerations on a 0.5 rad step fall
  from 77-180 rad/s² to 22-30.
- **Scripted expert on `7v4_real`: 0/100** on CPU and on MetalSim, with the same outcome on all 100 seeds. The expert
  sets each target to the measured position + ≤ 0.06 rad. At LeRobot's P 16 that is ≤ 0.32 N·m; gravity at the
  shoulder needs 0.39-0.57, so the arm sinks.
- **The old CPU-vs-MetalSim gap (20 vs 36)** comes from the evaluation harness: the expert reads CPU kinematics one
  physics step stale.
- **Shared module.** The coordinator's mid-task request is covered: `metalsim/physics/actuation.py`. See "Reusable"
  below.

## Changed

- `metalsim/physics/actuation.py` (new, shared):
  - `voltage_servo()`: a voltage servo as MuJoCo's native position actuator;
  - `SetpointParams` / `SetpointLayer`: command delay, then firmware speed and acceleration limits per physics step;
  - `CtrlPlanFeeder`: per-substep ctrl inside BatchSim's step graph, through `add_substep_hook`.
- `metalsim/render/film/so101.py`:
  - real-servo constants with sources: `STS_STEP`, `BAM_M1`, `LEROBOT`, `HOST_LATENCY`, `_REAL_BASE`;
  - `SERVOS["7v4_real"]` and `SERVOS["12v_real"]`;
  - `apply_servo(m, servo, **over)`: the real options and ablation overrides; the old options are unchanged;
  - `servo_layer(applied, n)`;
  - `SO101Lift` runs the layer and feeder for the real servos, and takes a new `adapter_kw["servo_over"]`;
  - new `SO101_SERVO_SECTION`, appended to `SO101_NOTES`.
- `docs/renders/SO101_LIFT.md`: new section "## Servo model under LeRobot's settings (7v4_real)", before
  "Reproduce". It is identical to `SO101_SERVO_SECTION` (checked).
- `runs/readme_media/scratch/so101-servo/` (git-ignored):
  - scripts:
    - `servo_cpu.py`: attaches the layer to so101Sim's env through a `mujoco` proxy; no so101Sim file is touched;
    - `stepresp.py` (+ `step_probe.py`);
    - `eval_cpu.py`: so101-smooth's script plus the layer, `--over`, `--fresh-kin` and `--cmdbase`;
    - `eval_metalsim.py`: plus a per-substep state recorder;
    - `smoothness.py`, `table.py`, `gpu_feeder_check.py`, `speed_check.py`, `phases.py`, `diag_expert.py`,
      `expert_variants.py`, `run_cpu_evals.sh`, `run_metalsim.sh`;
  - outputs:
    - `stepresp.json`, `step_tables.md`, `table.md`, `section.md`;
    - `cpu/<run>/{reference.json,reference.npz,metrics.json,smooth.json}`;
    - `metalsim/expert_{7v4,7v4_real}/{metrics.json,smooth.json,metalsim.npz}`;
    - `logs/`;
    - `research_others.md`: the research helper's survey.
- `runs/readme_media/scratch/ik_preview/`:
  - `render_expert.py`, rewritten: `servo:seed` panels, servo layer, labels, fps from `d.time`;
  - `expert_7v4_real.mp4` and `expert_old_vs_real.mp4`;
  - strips `old_vs_real_strip.jpg`, `real_strip.jpg`, `real_first_second.jpg`, `old_first_second.jpg`.
- Not touched: README.md, the plan doc, `tasks.py`, so101Sim (all tracked files), `metalsim/tron1/*`.

## Verified

### Sources

All files were read this session.

- **LeRobot** (`~/lerobot` at bf31dd79):
  - `src/lerobot/robots/so_follower/config_so_follower.py:45-47`: P 16, I 0, D 32; `max_relative_target` None;
  - `so_follower.py:159-171` `configure`: gripper Max_Torque_Limit 500, Protection_Current 250, Overload_Torque 25;
  - `motors/feetech/feetech.py:209-217` `configure_motors`: Return_Delay 0, Maximum_Acceleration 254, Acceleration
    254;
  - `motors/feetech/tables.py:41-101`: register addresses;
  - `git show 07e87163`: the first SO-100 calibration wrote "Lower the acceleration of the motors (in [0,254])",
    Acceleration 10, before position moves.
- **Rhoban BAM** (clone b9e1111):
  - `bam/feetech/actuator.py`: error_gain 0.166, max_pwm 0.97, and the internal target rate-limited to max_velocity;
  - `bam/params/feetech_sts3215_7_4V/m1.json`: kt 1.178, R 2.479, error_gain_ratio 1.005, armature 0.0261,
    friction_base 0.0515, friction_viscous 0.0599, max_velocity 5.288, command_delay 1.05 ms;
  - `bam/feetech/record.py:65-66`: identified with P = args.kp and D = 0, `all_record.py` kps [4, 8, 16, 32];
  - `bam/to_mujoco.py`: the export formula.
- **Gregory119** (`scratch/so101-74v/upstream/`): `gymnasium_env_README.md` and `calc_actuator_model.py`.
  - It reads Gp 16 as volts per degree with R = 1 Ω assumed.
  - It ignores the duty scaling, the clip position (it clips total torque), the firmware profile, D and latency.
- **Feetech datasheet** (`scratch/so101-74v/upstream/sts3215_200620a.txt`):
  - 19.5 kg·cm and 52 rpm at 7.4 V;
  - "Acceleration start stop function", "Mode 0 ... Support acceleration movement";
  - load is the duty cycle, 1000 = 100 %;
  - overload protection: 2 s, 20 %.
- **Menagerie, SO-ARM100, ManiSkill, isaac_so_arm101, leisaac, Fontaine blog, lerobot#3400, memory tables:** see
  `scratch/so101-servo/research_others.md` (helper agent; permalinks inside).
  - Menagerie's `robotstudio_so101` actuator is unchanged since its 2025-12-19 add (b0655057).
  - Upstream SO-ARM100 overrides the force limit to ±3.35.
  - Every other simulator config is hand-tuned.
- **MuJoCo:** 3.14 and 3.10 both have actuator `delay` / `nsample` / `interp` (`mjspec.h:756-758`), and MuJoCo Warp
  has `actuator_delay`. They are not used: the delay is host-side so that it sits in the same layer as the profile.
- **In-repo precedent:**
  - `metalsim/tron1/sim.py` (`_drive`, delay queue) and `tron1/realism.py` (names `cmd_delay`, `current_lag`,
    `speed_no_load`);
  - Isaac Lab `actuators/actuator_pd.py`: `DCMotor._clip_effort` (four-quadrant line) and `DelayedPDActuator`.

### The derived model

`7v4_real`:

- `kp = (1.912 / 0.97) × 16 × 0.166 × 1.0049 = 5.262 N·m/rad`, `kv = 0`;
- forcerange 1.912 N·m, gripper 0.986 N·m (50 % duty);
- damping `(1.912 − 0.0515) / 5.445 = 0.3417`;
- frictionloss 0.0515, armature 0.0261;
- internal target ≤ 5.288 rad/s and ≤ 254 × 100 × 2π/4096 = 38.96 rad/s²;
- delay 1.047 + 3.0 ms.

`12v_real` uses 30 kg·cm and 4.717 rad/s, which gives kp 8.095, force 2.942, damping 0.613.

Inferred, not sourced:

- that Acceleration acts with Goal_Velocity 0, and the shape of its profile (trapezoidal toward a moving goal);
- the 3 ms host latency;
- the same firmware gain and friction at 12 V.

### Step response

`stepresp.py`, so101Sim venv (MuJoCo 3.10). The protocol reproduces the orchestrator's old table: 53/53/33 % against
54/57/33, and 89/73/52 against 93/72/52 (`step_probe.py`). Full tables are in `step_tables.md` and the doc section.
At 7.4 V, against the old model:

- overshoot 89/72/52 % → 0/0/0;
- settle 315/240/190 → 310/280/255 ms;
- 90 % reached at 50 → 275 ms;
- sag 1 → 121/96/21 mrad;
- 0.5 rad step: peak speed 3.4-4.3 → 2.8-3.2 rad/s, peak acceleration 77-117 → 22-30 rad/s².

Evidence check:

- lerobot#3400 measures 63 steps (97 mrad) of settle error at P 16;
- the firmware acceleration cap is 39 rad/s².

### Ablation

Step response and the PPO task runs (CPU, 100 seeds); table in the doc section.

- The firmware gain removes the ringing.
- (d), the back-EMF line outside the clip, removes the remaining 33-49 % overshoot and keeps speeds physical: without
  it the PPO policy reaches 16-18 rad/s.
- (b), the profile, halves large-step acceleration and turns PPO's 23-32 reversals/s into 2-6. PPO success goes from
  86 on gains alone to 11.
- (a) and (c) are invisible at these time scales; host latency at 10 ms gives 9 against 6.
- D 32 at a 10 ms loop and BAM's kt/R change the step by ≤ 1 % overshoot and ≤ 60 ms.

### Smoothness

Every 5 ms physics step, CPU, 100 seeds, per joint (`table.md`; doc section).

| controller | old 7.4 V: success, arm reversals /s, >5 Hz rad/s | 7v4_real: success, arm reversals /s, >5 Hz rad/s |
|---|---|---|
| expert | 20; 7.8-13; 0.42-0.72 | 0; 0.19-0.82; 0.02-0.07 |
| original PPO | 77; 11-27; 0.48-0.55 | 6; 2.1-5.9; 0.28-0.54 |
| isaac_scaled | 77; 7.1-29; 0.38-0.56 | 15; 2.5-5.8; 0.27-0.51 |
| isaac_unscaled_raw | 86; 3.9-29; 0.37-0.58 | 4; 2.4-5.0; 0.26-0.48 |

On 7v4_real, RMS acceleration is 2.7-11 rad/s² for the expert and 35-77 rad/s² for the PPO policies.

The PPO policies exceed the 5.45 rad/s no-load speed on 8-14 % of steps. `speed_check.py` shows the cause: wrist roll
alone stops at 5.44; the coupled joints reach 5.6-6.0 with gravity and contacts off; gravity and inertial coupling
back-drive a joint past its line.

### Expert, 100 seeds (10000-10099), 7v4_real

| run | success |
|---|---|
| CPU, so101Sim's loop | 0 |
| CPU with `mj_forward` first (`--fresh-kin`) | 0 |
| MetalSim | 0, same outcome as CPU fresh-kin on 100/100 seeds |

- MetalSim: 15 worlds hit the ITERATIONS cap, all 100 the LS_ITERATIONS cap; no overflow.
- `12v_real`: 0 on CPU.
- `phases.py`: 80 runs stay in approach, with the TCP z minimum at 0.020 m (median), against 0.073 on the old model.
- Diagnostic `--cmdbase` (targets built on the previous command, anti-windup 0.363 rad; not so101Sim's controller):
  3/100. It reaches lift in 45, but the jaws are still at 1.0 rad when the 0.3 s grasp phase ends.

### The CPU-vs-MetalSim gap on the old 7.4 V model

| run | success |
|---|---|
| CPU (stale kinematics) | 20 |
| CPU `--fresh-kin` | 38 |
| MetalSim (this run) | 34 |

- Seed agreement with MetalSim rises from 56 (so101-smooth) to 66.
- **Mechanism:** after `mj_step`, `env.data`'s xpos, site_xpos and Jacobian are those computed before the last
  integration. This rests on a code reading: MuJoCo's `mj_step` runs forward and then integrates. The size of the
  effect is measured.
- **The fix does not close the gap on 7v4_real:** both backends give 0 there, because the expert's command style
  cannot hold the arm.

### GPU path

`gpu_feeder_check.py`, in the GPU queue as `so101-servo-check`:

- MetalSim against CPU MuJoCo 3.14 with the same layer: max |Δq| 8.2e-7 rad over 40 control steps of bang-bang
  commands;
- negative control (first target held for all 4 substeps): up to 8.8e-2 rad.

Jobs: `so101-servo-metalsim` (twice; the first failed on a kernel annotation bug, now fixed) and `so101-servo-check`.
Logs are in `scratch/so101-servo/logs/`.

### Live state

- **mujoco_warp: 93e5ce18, not the pinned 8eb51fe.**
  - Another stream fast-forwarded it at 17:08 (reflog). The GPU holder `rr_smoke_*` now expects 93e5ce18.
  - Its diff from 8eb51fe touches only heightfield collision paths (`collision_convex.py`, `collision_gjk.py`
    `HFIELD_*` flags).
  - The SO-101 scene has nhfield 0, so my MetalSim numbers should equal 8eb51fe's. That rests on reading the diff
    only.
- Power during the jobs: AC, charging, 59 % at 17:20 and 66 % at 17:26; powermode 2. No timing figures were taken.

### Tests and paths

- `.venv/bin/python -m pytest -q tests/test_film.py tests/test_no_machine_paths.py`: 5 passed, 1 skipped.
- No machine paths in `so101.py`, `actuation.py` or the doc section. Scratch scripts use absolute paths; they are
  git-ignored.

### Frames looked at

Read tool:

- `ik_preview/old_vs_real_strip.jpg`: every 50th frame, both panels;
- `real_strip.jpg`: seeds 10010 and 10000, every 50th frame;
- `real_first_second.jpg` and `old_first_second.jpg`: frames 0-66, every 6th.

What they show:

- **Old model:** the arm whips from horizontal to upright in about 0.25 s and tips the box at 0.38 s. It then grasps
  and holds the box up, having lifted at 3.36 s.
- **7v4_real:** the arm droops smoothly forward over about 0.4 s, and the gripper settles near the table beside the
  box. It then creeps up into a folded pose and stays in approach to 6 s, on both seeds.
- No visible jitter on 7v4_real.
- Labels and times read correctly.
- Both videos: ffprobe 1280×480, 50/1 fps, 300 frames, 6.000 s.

## Open

- **Acceleration register (b).** It is the most consequential inferred piece: it moves PPO success from 24 to 6 and
  sets the policies' smoothness. Whether it acts in position mode with Goal_Velocity 0, and its profile shape, need a
  bench test or Feetech's official memory table. The helper found no official text, only third-party tables and an
  image-only PDF.
- **Stall torque.** Datasheet 1.91 against BAM 3.41 N·m. Fontaine's replay fit (1.97 N·m, kp 8.8) sides with the
  datasheet on force and lies between the two on kp.
- **Not modelled:**
  - D 32: the loop period is unknown; bounded as small;
  - overload / current protection on the gripper: the sources disagree on the register semantics;
  - the dead zone, encoder quantization, backlash;
  - LeRobot's 30 fps rate.
- **Host latency 3 ms** is inference.
- **so101Sim's own model should change too:**
  - its scene (Menagerie `sts3215` class) has Gregory's kp 998 / kv 2.731;
  - it needs the `7v4_real` actuator values above, plus the setpoint layer (command delay + firmware profile) in its
    `step`, per physics step;
  - so101Sim's expert also needs a command base that leads the measured position (P-only sag) and a grasp phase that
    waits for the jaws.
- **README cells and films** still show the old model; nothing here re-filmed them.
- **Reusable for the G1 adapters:**
  - `SetpointLayer` (fractional command delay + speed/acceleration-limited setpoint);
  - `CtrlPlanFeeder` (host plan → per-substep ctrl in a captured BatchSim graph);
  - `voltage_servo` (voltage servo, back-EMF outside the clip).
- **Not reusable as is:** TRON1-style `current_lag` on torque. It needs per-step state (a native `dyntype` or a Warp
  hook) and is not in the module. `metalsim/tron1/sim.py` was not switched to the module (not mine).

## Next

1. **Owner review.** `runs/readme_media/scratch/ik_preview/expert_old_vs_real.mp4` and `expert_7v4_real.mp4`
   (real time, 6 s, MuJoCo renderer).
2. **Retraining (next increment, if the owner accepts 7v4_real).**
   - Train so101Sim's lift teacher on `7v4_real`. The env needs the layer: `scratch/so101-servo/servo_cpu.attach(env,
     "7v4_real")` patches so101Sim's env in-process.
   - Start from scratch or from run 2.
   - The smoothness penalties are optional now that the firmware profile filters 25 Hz flipping.
   - Judge on `smoothness.joint_motion`, not on command metrics.
   - Evaluate with `eval_cpu.py <pt> 7v4_real` and `eval_metalsim.py <pt> 7v4_real` (GPU queue). Compare against the
     expert-on-7v4_real smoothness row.
3. **Optional, owner's call.** An expert that works on the real servo, as a new controller in MetalSim, not by
   editing so101Sim:
   - targets built on the previous command;
   - a grasp phase that waits for gripper closure;
   - `expert_variants.py` is the starting point.
4. **Optional.** Settle the Acceleration-register question on hardware: step the follower with Acceleration 0 against
   254 and log Present_Position at 1 kHz.
5. **Commit** with the other SO-101 streams' files. `actuation.py` is new shared code; it is safe to commit, since
   nothing else imports it yet.

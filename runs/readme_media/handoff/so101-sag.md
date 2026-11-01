# Hand-off: so101-sag (2026-10-31)

## Status

**Done.** There is a sourced fix, as a patch file: `runs/readme_media/scratch/so101-sag/so101-sag.patch`. It is not
applied. It works on CPU MuJoCo only; the MetalSim (GPU) path is in Next 1.

### Cause

`7v4_real` leaves out the STS3215's load-dependent gearbox friction, and its P-loop gain is half of BAM's.

**What the real followers do.** They hold a gravity load with a static friction band that widens with the load:

| joint | stop after moving with gravity | stop after moving against gravity |
|---|---|---|
| shoulder_lift | 5-28 mrad | 26-98 mrad |
| elbow_flex | 7-18 mrad | 55-82 mrad |

Which side of the band a joint ends on depends on the task:
- **Shoulder:** 78-79 % of steady samples come right after lowering, so it sits at the low edge.
- **Elbow:** 71-82 % come right after rising, so it sits at the high edge.

**What 7v4_real does.** Its band is ±9.8 mrad around g / 5.26.
- At the shoulder it sags 5-6 times too much.
- At the elbow it agreed only by coincidence: the with-gravity error there is 62 mrad.

### Fix (`7v4_bam`)

Rhoban BAM's m6 identification of the STS3215 7.4 V, used throughout:
- kp 10.58 N·m/rad;
- load-dependent Stribeck friction, written to `dof_frictionloss` every physics step, as BAM's own MuJoCo loop does;
- BAM's stiff friction constraint.

Nothing was fitted to the sag numbers.

**Effect when the datasets' own commands are replayed on the sim follower** (median |sim − real| sag per load bin, mrad):

| joint | 7v4_real | 7v4_bam |
|---|---|---|
| shoulder_lift | 93 | 17 |
| elbow_flex | 9 | 12 |
| wrist_flex | 1.9 | 1.1 |

Split by approach direction, the elbow goes from 62 / 9 mrad (with / against gravity) to 14 / 15.

**Deploy expert, 100 seeds on CPU:**

| servo | success | time to lift, median |
|---|---|---|
| 7v4_real | 100/100 | 2.46 s |
| 7v4_bam | 100/100 | 2.19 s |

The motion is as smooth or smoother on 7v4_bam.

## Changed

All new and git-ignored, under `runs/readme_media/scratch/so101-sag/`. No tracked file was touched.

- **Patch:**
  - `so101-sag.patch`: against `metalsim/physics/actuation.py` (sha1 542b6b1) and `metalsim/render/film/so101.py`
    (working tree, sha1 05b8c0e). Snapshots are in `patch/base/`, patched copies in `patch/new/`.
  - `patched.py`: imports the patched copies in place of the tree's, for testing.
- **Scripts:**
  - `real_dir.py`: real sag split by approach direction and by TCP height.
  - `bam_m6.py`: BAM's friction budget and motor constants.
  - `static_band.py`: analytic hold band of each candidate model against the real data.
  - `sag_servo.py`: experiment variants on top of the unpatched `apply_servo`.
  - `hold_test.py`: approach-and-hold test and creep check.
  - `replay.py`: replays the real commands on the sim follower.
  - `compare.py`: one analysis applied to both the real data and the replays.
  - `nogravity.py`: pan and wrist roll band.
  - `check_budget.py`: our budget against BAM's own code.
  - `eval_cpu_sag.py`: so101-expert's `eval_cpu.py`, plus the variants, the friction hook and `--patched`.
  - `table.py`.
- **Outputs:**
  - `real_dir.json`, `static_band.json`, `hold_test.json`, `compare.json`, `compare_m346.json`,
    `compare_move{0.15,0.6}.json`, `table.md`;
  - `replay/<variant>/` and `cpu/<variant>/{episodes,smooth}.json,traj.npz`;
  - `logs/*.log`.
- **Data:** `hfdata/{hungho77,ilya}/` holds copies of so101-expert's two LeRobot parquets plus their `info.json`:
  - hungho77: sha1 632990d;
  - ilya: sha1 ccce762.
- This hand-off.

## Verified

Interpreters:
- so101Sim venv `~/so101Sim/urlab_bridge/.venv/bin/python` (MuJoCo 3.10), abbreviated PY;
- main venv `.venv/bin/python` (MuJoCo 3.14).

All commands run from `runs/readme_media/scratch/so101-sag/`, with `H=hfdata`.

### Sources read this session

**Rhoban BAM** (clone at b9e1111, `<session tmp>/so101-servo-src/bam`, from so101-servo):
- `bam/params/feetech_sts3215_7_4V/m1..m6.json`. m6 values:
  - kt 1.2753, R 2.7534, error_gain_ratio 1.1617, armature 0.02156;
  - max_velocity 5.096, command_delay 4.98 ms;
  - friction_base 0.0533, friction_stribeck 4.3e-5;
  - load_friction_external 0.2436, load_friction_motor 0.0449;
  - load_friction_external_stribeck 0.1423, load_friction_motor_stribeck 0.00012;
  - quad terms 0.0034 / 0.0076;
  - dtheta_stribeck 0.371, alpha 9.93, friction_viscous 0.0282.
  - m3 / m4 / m5 also carry load friction: 0.21 × |τm − τe| (m3, m4) and 0.24 / 0.15 directional (m5).
- `bam/model.py:125-208` `compute_frictions` (the budget) and `:256-281` (the M1-M6 flags).
- `docs/theory/models.rst` (the equations; "directional efficiency/backdrivability asymmetry"). Paper: arXiv
  2410.08650 (`docs/index.rst:91`).
- `bam/feetech/actuator.py:33-44, 146-155`: vin 7.4, error_gain 0.166, max_pwm 0.97; duty = err × P × 0.166 × ratio.
- `bam/actuator.py:317-342`: τ = kt V / R − kt² / R · ω.
- `bam/mujoco.py:170-203` (`MujocoController.update`). Per step:
  - τe = −qfrc_bias + qfrc_constraint − own friction force;
  - τm = qfrc_actuator;
  - the result is written to `dof_frictionloss`.
- `bam/mjlab.py:252-263`: stiff `solref_friction` (−5e4, −2e2) and `solimp` (0.99 0.9999 0.001 0.5 2), "MuJoCo Warp
  has no noslip solver … a statically-held joint creeps".
- `bam/mjlab.py:595-760`: the GPU version of the same update.
- `bam/testbench_mujoco.py:92-93`: one noslip iteration on CPU.

**LeRobot** (`~/lerobot` at bf31dd79):
- `so_follower.py:159-171`: P 16, I 0, D 32 on every motor.
- `so_follower.py:205-230`: `send_action` writes the action as Goal_Position through `bus.sync_write`, which
  normalizes it with the follower's own calibration.
- `motors_bus.py:874-877, 904-907`: degrees are measured about the middle of each arm's own range.
- `lerobot_record.py:304, 321-323, 357, 362`: observe, then the action, then send; the frame records the action that
  was sent.
- `docs/source/so101.mdx:35`: "The follower arm uses 6x STS3215 motors with 1/345 gearing".

**Others:**
- lerobot#3400, via so101-servo `research_others.md:119-127`: the P 16 settle point "depends on approach direction".
- `so101-servo/research_others.md:160-163`: dead zone default 1 step (third-party table).

### Hypotheses

**(a) Friction: confirmed. It is the main cause, together with kp.**

- *Real data, measured* (`PY -I real_dir.py ~/so101Sim real_dir.json hungho77=$H/hungho77/file-000.parquet@30 IlyaGusev=$H/ilya/file-000.parquet@20`,
  `logs/real_dir.log`). Sag after moving with gravity against after moving against it, mrad:

  | joint | with gravity | against gravity |
  |---|---|---|
  | shoulder_lift | 5-28 | 26-98 |
  | elbow_flex | 7-18 | 55-82 |
  | wrist_flex, ‖g‖ 0.1 | −0.8 | 15 |

  - With 7v4_real the gap between the two would be at most 2c / kp = 19.6 mrad.
  - Share of steady samples after moving with gravity: shoulder 78 / 79 %, elbow 29 / 18 %.
  - Robust to the motion threshold 0.15-0.6 °/frame (`logs/compare_move_sensitivity.log`).

- *Static band, sourced* (`.venv/bin/python -I static_band.py`, `logs/static_band.log`). Each candidate's hold band is
  compared with the real with / against medians.

  | model | median edge error |
  |---|---|
  | 7v4_real | 41 mrad |
  | BAM m1 (kp only) | 21 mrad |
  | BAM m3 / m4 / m5 / m6 | 8.4 / 8.1 / 7.9 / 7.8 mrad |
  | 7v4_real kp + m6 friction | 37 mrad |

  - The BAM motor constants at P 16: kp 9.38 (m1) to 11.95 (m3).
  - **Both BAM's kp and its load friction are needed.**

- *Replay of the real commands on the sim follower, experiment* (`replay.py`, `compare.py`, `logs/compare.log`).
  - Each episode starts at the real follower's state. The actions are streamed at the dataset fps through the servo
    layer, with the friction hook. Contacts are off.
  - The table below is the median |sim − real| per ‖g‖ quintile, both datasets pooled (`table.md`).

  | servo | lift all | lift with | lift against | elbow all | elbow with | elbow against | wflex all |
  |---|---|---|---|---|---|---|---|
  | 7v4_real | 93 | 97 | 34 | 9.3 | 62 | 8.9 | 1.9 |
  | bam_m1 (BAM kp, constant Coulomb) | 44 | 46 | 3.6 | 20 | 28 | 24 | 1.2 |
  | **bam_m6 = `7v4_bam`** | **17** | **17** | **9.8** | **12** | **14** | **15** | **1.1** |
  | 7v4_real + m6 friction | 61 | 63 | 56 | 27 | 42 | 24 | 2.4 |

  - bam_m3 / bam_m4 give the shoulder 17 and the elbow 14-15 (`logs/compare_m346.log`).
  - Heaviest shoulder bin, sim ÷ real:
    - 7v4_real 150.6 / 28.4 = 5.3× (hungho77) and 148.7 / 22.2 = 6.7× (IlyaGusev);
    - 7v4_bam 45.8 / 28.4 = 1.6× and 54.1 / 22.2 = 2.4×.

- *Creep, experiment* (`PY -I hold_test.py ~/so101Sim`, `logs/hold_test.log`).
  - With MuJoCo's default soft friction constraint, the m6 shoulder creeps 11 mrad between 0.5 s and 3 s
    (qd 2.5e-3 rad/s).
  - With BAM's stiff solref / solimp it creeps under 1 mrad.
  - The patch therefore stiffens the constraint.

- *Code check* (`.venv/bin/python -I check_budget.py`): our budget equals BAM's own `Model.compute_frictions` exactly
  (max diff 0.0) for m1-m6 on 20 000 random inputs. colorama was stubbed and nothing was installed.

**(b) The real-data measurement.**

- **The calibration offset: refuted.** e = action − state is the follower's own Goal_Position − Present_Position. The
  leader/follower calibration never enters it (LeRobot lines above). The datasets' state sits on the 360/4095° lattice
  (p99 residual 0.000 step). The opposite claim in so101-expert's `real_sag.py` docstring is wrong.
- **The elbow's "agreement" was an artefact of approach direction.** The elbow mostly stops after rising, at the high
  edge of its band.
- **Shoulder servo variant: refuted.** All six follower motors are STS3215 at 1/345 (lerobot `so101.mdx:35`). The
  supply voltage of the dataset arms is not recorded.
- **Table contact: refuted as the cause.** Shoulder sag at ‖g‖ 0.74 is 25.3 mrad with the TCP below 3 cm and 28.4 above
  6 cm (hungho77). The free-air replay reproduces the gap without any contact.

**(c) Firmware.**

- **Refuted as the cause of the 5-6×.**
  - I is 0, and D 32 acts on velocity only, so neither changes a static hold.
  - BAM's P-loop gain at P 16 puts the real elbow's band centre right: (with + against) / 2 is 34-45 mrad in both the
    real data and bam_m6.
  - A 1-step dead zone is ±1.5 mrad.
- **Open, small:** without gravity the real band is wider than BAM's.
  - Steady median |e| at pan and wrist roll: real 4.6-14 mrad, sim bam_m6 3.4-4.1, 7v4_real 4.1-6.5
    (`nogravity.py`, `logs/nogravity.log`).

**(d) The sim model.**

- **Open, as the residual only.**
  - The real shoulder's band centre is about 1.3× lower than bam_m6's, with a similar width.
  - The real elbow's band is about 1.7× wider than bam_m6's, with the same centre.
- This is consistent with:
  - more friction on the real units than on BAM's one bench servo;
  - and a shoulder-only offset: the MJCF's distal mass or COM (CAD values, not measured), or that unit's gain.
- Not testable from teleop data.
- Armature and joint damping do not act at rest.

### The patch

What it changes:
- `actuation.py`: adds `bam_friction_budget` (a copy of BAM's) and `BamFriction` (CPU per-step update plus `stiffen`).
- `so101.py`:
  - adds `BAM_M6` and `SERVOS["7v4_bam"]`;
  - `apply_servo` honours `error_gain_ratio` and `friction`;
  - adds `friction_hook(applied, m)`;
  - `SO101Lift` raises `NotImplementedError` for a servo with friction, since there is no GPU update yet.
- 7v4_real, 12v_real and the old options are unchanged.

Checks:
- **It applies:** `git apply --check` passes on the snapshot (and reproduces `patch/new/` byte for byte) and on the
  current tree. The tree was not modified.
- **7v4_real is unchanged:** `PY eval_cpu_sag.py ~/so101Sim 7v4_real cpu/patched_7v4_real 100 --patched`
  equals the unpatched run bitwise (max |Δq| 0.0). Both equal so101-expert's `cpu/deploy_7v4_real`: 100/100,
  2.46 (2.6136), 6 retries.
- **7v4_bam matches the experiment variant:** `--patched 7v4_bam` against `bam_m6` gives 100/100 identical outcomes and
  times to lift, median |Δq| 2e-16 and max 0.019 rad (float divergence in contact).
- **Tests:** `pytest tests/test_film.py` with the patched modules loaded: 4 passed, 1 skipped. The patch has no
  machine paths (grep).

### 100-seed CPU eval of the deploy expert

`PY eval_cpu_sag.py ~/so101Sim <variant> cpu/<variant> 100`, seeds 10000-10099, `logs/eval_cpu.log`. Joints
are pan / lift / elbow / wflex / wroll / grip.

| servo | success | up at 6 s | time to lift med (mean) s | retries | RMS acc rad/s² | RMS jerk ×10³ | reversals /s | vel RMS > 5 Hz |
|---|---|---|---|---|---|---|---|---|
| 7v4_real | 100 | 100 | 2.46 (2.61) | 6 | 2.1/6.6/5.3/5.8/3.3/7.6 | 0.33/1.03/1.00/0.78/0.12/0.89 | 0.36/0.79/0.50/0.55/0.29/0.93 | 0.018/0.042/0.035/0.037/0.011/0.079 |
| **7v4_bam** | **100** | **100** | **2.19 (2.23)** | 1 | 2.2/5.0/4.4/4.5/3.2/7.6 | 0.44/0.32/0.58/0.29/0.09/0.58 | 0.43/0.71/0.18/0.37/0.01/0.17 | 0.019/0.024/0.031/0.021/0.018/0.083 |
| bam_m1 | 100 | 100 | 2.24 (2.27) | 1 | 2.1/4.9/4.2/4.4/3.0/7.6 | 0.33/0.31/0.61/0.27/0.09/0.61 | 0.40/0.84/0.44/0.53/0.00/0.62 | 0.019/0.021/0.025/0.016/0.013/0.084 |
| 7v4_real + m6f | 99 | 98 | 2.40 (2.57) | 10 | 1.7/6.1/4.9/5.5/3.4/6.7 | 0.30/0.92/0.84/0.67/0.11/0.49 | 0.36/0.68/0.21/0.48/0.23/0.23 | 0.012/0.037/0.032/0.032/0.013/0.071 |

Smoothness on 7v4_bam:
- Smoother at lift, elbow and wrist flex: jerk 3× lower at lift.
- The pan's jerk is 0.44 against 0.33, and wrist roll's > 5 Hz velocity is 0.018 against 0.011.

Steady-state sag in the expert's lift phase (box in hand), pan / lift / elbow / wflex / wroll, mrad:

| servo | sag |
|---|---|
| 7v4_real | 4 / 33 / 126 / 46 / 9 |
| 7v4_bam | 4 / 6 / 46 / 11 / 1 |

No frames were rendered or looked at in this increment.

## Open

1. **No GPU path.** The friction update runs on CPU MuJoCo only, so the film (MetalSim) cannot use `7v4_bam` until
   Next 1 is done; `SO101Lift` refuses it.
2. **Residual against real arms** (Verified (c) and (d)). Not fitted, on purpose:
   - the shoulder still sags 1.6-2.4× more than the real arms (median of all steady samples, heavy bins);
   - the elbow band is narrower than real;
   - the no-load band is about 1.5-3× narrower than real.
3. **Stall torque.**
   - `7v4_bam` takes BAM's 3.32 N·m (1.74× the datasheet's 1.91).
   - The sag data constrain the small-signal kp only. They reject 5.26 at every joint: the 7v4_real + m6f row.
   - The gripper's 50 % cap becomes 1.71 N·m instead of 0.986 N·m.
   - The effect on grasp realism and box tilt was not looked at.
4. **No 12 V version.** BAM has no 12 V STS3215 identification.
5. **Base file is moving.** `so101.py` is being edited by so101-film (its sha1 changed during this session). Re-check
   with `git apply --check` before applying.
6. **Docs not updated.** The servo doc section (`SO101_SERVO_SECTION`, `docs/renders/SO101_LIFT.md`) and so101-expert's
   sag paragraph still describe 7v4_real.

## Next

1. **GPU friction update** (needed before the film can use `7v4_bam`). Own `metalsim/physics/actuation.py`, and
   `so101.py` with so101-film's agreement.
   1. Apply `so101-sag.patch`.
   2. Build `BatchSim` with `per_world_fields` including `"dof_frictionloss"`.
   3. Add a substep hook: a Warp kernel over (world, arm dof). It scans `d.efc.type/id/force` for `FRICTION_DOF` rows
      below `d.nefc[w]`, then computes τe = −qfrc_bias + qfrc_constraint − own and τm = qfrc_actuator − emf·qvel, the
      budget (`bam_friction_budget`), and writes `m.dof_frictionloss[w, dof]`. Reference: `bam/mjlab.py`
      `BamActuator.compute` and `_dof_friction_force`. The mujoco_warp fields exist (`upstream/mujoco_warp/.../types.py`:
      `dof_frictionloss` batchable; `efc.type/id/force`, `nefc`).
   4. Check against CPU `BamFriction` the way so101-servo's `gpu_feeder_check.py` does: max |Δq| over 40 control
      steps. Go through the GPU queue.
   5. Run so101-expert's `eval_metalsim.py` on 100 seeds and compare with `cpu/patched_7v4_bam`.
   6. Then remove the `NotImplementedError`.
2. **Bench measurement** that would settle the residual and the stall torque. Use one real follower with LeRobot's
   registers.
   - For shoulder_lift at 3 poses, TCP in free air, approach each pose 5 times from above and 5 times from below at
     ≤ 0.5 rad/s. Hold 2 s. Log Goal_Position, Present_Position and Present_Load (duty ‰) at 50 Hz.
   - What this separates:
     - duty / error gives the firmware gain;
     - Present_Load × full torque against the model's g gives the friction band and the mass model.
   - Weigh the links.
   - For the stall, block the joint and step the goal by 0.5 rad; read the force at a known lever with a scale. This
     settles 1.91 against 3.3 N·m.
   - `compare.py` and `real_dir.py` take any LeRobot parquet.
3. **Before filming on `7v4_bam`:** render the deploy expert on it (`scratch/ik_preview/render_deploy.py` pattern) and
   look at the grasp with the stronger gripper and at the box tilt.

# Hand-off: so101-bam-gpu (2026-10-31)

## Status

**Done.** `7v4_bam`'s load-dependent servo friction now runs on MetalSim:
- a Warp kernel over (world, arm dof), after every physics substep inside BatchSim's step graph;
- it writes the per-world `dof_frictionloss`.

| check | result |
|---|---|
| against CPU `BamFriction`, 8 seeds × 40 control steps | max \|Δq\| 1.9e-6 rad: float32 level |
| deploy expert, 100 seeds, MetalSim | 100/100 success, median time to lift 2.18 s, 1 retry |
| same seeds, so101-sag's CPU reference | identical on every seed |
| cost per physics step against `7v4_real` | 0.98-0.99, inside the measurement noise |

**A small defect found and fixed:**
- The first friction budget after a reset used the motor torque from a forward pass at ctrl 0, a spurious full-clip
  torque.
- The CPU reference and so101Sim's env share it.
- `SO101Lift` now powers the servos on holding before that pass.
- The fix moves joints by at most 0.74 mrad and changes no outcome.

**`so101.py` is NOT changed.** so101-film owns it. The change is a patch for the orchestrator to apply: see "Apply"
below.

## Changed

- `metalsim/physics/actuation.py` (untracked, shared):
  - the so101-sag patch's `actuation.py` hunks are applied: byte-identical to `so101-sag/patch/new/actuation.py`;
  - plus `BamFrictionWarp` and `_bam_kernels` (the `bam_update` and `bam_rest` kernels), and docstrings;
  - sha1 542b6b1 → 53c36c5.
  - Additive only: `voltage_servo`, `SetpointLayer` and `CtrlPlanFeeder` are unchanged.
- `tests/test_bam_friction.py` (new):
  - 6 kernel tests on Warp's CPU device, one per BAM model m1-m6;
  - a per-world guard test;
  - a Metal test (BatchSim against CPU MuJoCo), skipped without Metal.
- `runs/readme_media/scratch/so101-bam-gpu/` (git-ignored):
  - **Patches:**
    - `so101-bam-gpu.patch`: the full `so101.py` change against the live file (sha1 88e7922; the sag hunks plus the
      GPU wiring);
    - `so101-bam-gpu-wiring-only.patch`: the wiring alone, on top of the sag hunks;
    - `base/so101.py`: the snapshot the patches were made from;
    - `so101_patched.py`: the patched result;
    - `patched.py`: a loader that imports `so101_patched.py` in place of the tree's file.
  - **Scripts:**
    - `gpu_bam_check.py`: the GPU-vs-CPU check, plus `--cpu-only`;
    - `eval_metalsim_bam.py`: so101-expert's `eval_metalsim.py` with friction on both backends, plus an actuator-force
      record, `--over` and `--legacy-ctrl0`;
    - `bench_bam.py`: cost ratios;
    - `mjw_cpu_preflight.py` and `mjw_cpu_variants.py`: MuJoCo Warp on Warp's CPU device against MuJoCo C;
    - `compare_runs.py` (writes `table.md`);
    - `stall_use.py`;
    - `gpu_job.sh`.
  - **Outputs:**
    - `check/*.json`;
    - `metalsim/{deploy_7v4_bam, deploy_7v4_bam_legacy0, deploy_7v4_real_rerun}/`;
    - `dry/{dry100_7v4_bam, dry100_7v4_bam_legacy0, dry100_7v4_bam_ds_stall, dry100_7v4_real}/`;
    - `bench/ratios.json`, `table.md`, `logs/`.

**Shared-library patch: none needed.**
- BatchSim already has the substep hook with a reset hook: `add_substep_hook(fn, reset_fn)`,
  `metalsim/physics/batch.py:377-385`. Hooks run after every `mjw.step` (`:362-366`).
- It already has per-world model fields: `per_world_fields` (`:86`, `:244`).

### Apply (orchestrator, after so101-film is done)

`git apply runs/readme_media/scratch/so101-bam-gpu/so101-bam-gpu.patch`

- `git apply --check` passed on live `so101.py` sha1 88e7922, and again at 20:44 on 8562b1a.
- The file moved three times during this session (05b8c0e → 6335337 → 88e7922 → 8562b1a): so101-film's docs and
  comments, far from these hunks. The patch's context lines are from 88e7922.
- The patch already contains the so101-sag patch's `so101.py` hunks. Do not apply `so101-sag.patch` as well; its
  `actuation.py` part is already in the tree and would now fail.
- **If the full patch stops applying**, use the two-step route instead:

  `git apply --include=metalsim/render/film/so101.py runs/readme_media/scratch/so101-sag/so101-sag.patch && git apply runs/readme_media/scratch/so101-bam-gpu/so101-bam-gpu-wiring-only.patch`

  Both steps passed `--check` on 88e7922; on 8562b1a both passed as a `patch --dry-run` on a copy.
- After applying, run `.venv/bin/python -m pytest -q tests/test_bam_friction.py -k "not batchsim" tests/test_film.py
  tests/test_no_machine_paths.py`. The Metal test goes through the GPU queue.

### What the `so101.py` patch does

1. The so101-sag hunks: `BAM_M6`, `SERVOS["7v4_bam"]`, `apply_servo` friction, `friction_hook`.
2. `friction_hook(applied, m, sim=None)`: given a BatchSim, it returns `BamFrictionWarp`.
3. `SO101Lift`:
   - adds `"dof_frictionloss"` to `per_world_fields` when the servo has friction;
   - registers the hook;
   - in `reset()`: writes ctrl at the start pose, runs `sim.forward()`, then `friction.update()`.
4. Removes the `NotImplementedError` (sag Next 1.6).
5. Docstrings and the BAM comment block say the GPU path exists.

## Verified

Main venv `.venv/bin/python` (MuJoCo 3.14). mujoco_warp 93e5ce18, Warp 1.18.0.dev20260917.

- GPU job `scripts/gpu_run.sh so101-bam-gpu render 25 -- bash runs/readme_media/scratch/so101-bam-gpu/gpu_job.sh`,
  20:39:43-20:40:31, every step rc 0 (`logs/gpu_job.log`).
- Power, read live at the job's start and end: AC, charging, 81 %, powermode 2.

### Sources read

**Rhoban BAM** (clone at b9e1111, `<session tmp>/so101-servo-src/bam`):
- `bam/model.py:125-208` `compute_frictions`: the quadratic term at `:183-203` is gated on sign(τe) ≠ sign(τm).
- `bam/mjlab.py`:
  - `:601-638` `_dof_friction_force`: rows below `nefc` with type FRICTION_DOF, scatter-added by `efc.id`;
  - `:640-749` `compute`: τe = −qfrc_bias + qfrc_constraint − own; τm = the previous solve's qfrc_actuator;
  - `:516-530`: the quadratic term without the sign gate;
  - `:252-263`: the stiff solref / solimp.
- `bam/mujoco.py` `MujocoController.update`: the CPU loop.

**MuJoCo Warp** (`upstream/mujoco_warp/mujoco_warp/_src`):
- `types.py`:
  - `:579-599` `ConstraintType.FRICTION_DOF` = `mjCNSTR_FRICTION_DOF`;
  - `:2144-2184` `efc.type/id/force`, shape (nworld, njmax), int32 / float32;
  - `:2358` `nefc` (nworld,);
  - `:1686` `dof_frictionloss` is batchable;
  - `:2427-2441` the `qfrc_*` and `actuator_force` fields.
- `constraint.py`:
  - `:1886-1888`: friction rows read the per-world `dof_frictionloss` and are emitted only when it is > 0. The BAM
    budget is always ≥ `friction_base` 0.053, so the rows never drop out.
  - `:116-123`: the direct-form solref (BAM's stiff values) is honoured.
- `forward.py:1841-1852`: `step` = forward, then the integrator. After a step, the forces come from the step's
  forward pass and `qvel` is integrated, as in MuJoCo C's `mj_step`. So "after substep k" equals CPU's "before step
  k+1".
- `io.py:1586`: sparse nnz sizing counts `dof_frictionloss > 0` at `put_data`.

**This repo:**
- `metalsim/physics/batch.py:362-385` (hooks), `:86` and `:244` (per-world fields).
- `tests/test_physics.py:90-113`: the SO-101 MuJoCo Warp vs C trajectory tolerance is 2e-2.
- so101Sim `rl/native/so101_lift.py:101-122`: `reset` runs `mj_forward` at ctrl 0.

### The kernel

`metalsim/physics/actuation.py`:
- `:214-292` `_bam_kernels`, the kernel `bam_update`, follows `BamFriction.update` (`:128-139`) and
  `bam_friction_budget` (`:64-94`) line for line:
  - own-row scan bounded by min(nefc, njmax);
  - τe and τm;
  - flag bits for the m1-m6 terms;
  - numpy's `sign` semantics (`wp.sign(0)` is 1).
- `:149-211` `BamFrictionWarp`:
  - raises if `dof_frictionloss` is not per-world;
  - the reset hook writes the budget at rest to the reset worlds (`BamFriction.reset`);
  - `update()` launches the kernel once after a host state write.

### Tests

- **CPU, outside the queue:** `.venv/bin/python -m pytest -q tests/test_bam_friction.py -k "not batchsim"`: 7 passed.
  - m1-m6 at rtol 2e-6 against float64, on random and edge-case inputs: τ = 0, \|τe\| = \|τm\| of both signs, stale
    rows past `nefc`, `nefc` > `njmax`.
  - Mutation check: ignoring `nefc` fails 4 models; dropping the sign gate (BAM mjlab's form) fails m6.
- **In the GPU job:** all 8 passed, the Metal BatchSim test included (`logs/pytest_gpu.log`).
- `tests/test_film.py` and `tests/test_no_machine_paths.py`, with `so101_patched.py` loaded: 12 passed, 1 skipped.
- No machine paths in the patch, `actuation.py` or the test (grep).

**Accidental Metal use (disclosure).** Once, plain `pytest tests/test_bam_friction.py` was run outside the queue, and
the Metal test was collected.
- It failed on a numpy assignment before any BatchSim was built: the error was reproduced on CPU, at the line
  `m.actuator_biasprm[:6, 1:3] = (…, 0.0)`, since fixed.
- So the only Metal activity was device enumeration. After that, every run outside the queue used
  `-k "not batchsim"`.

### Pre-flight without the GPU (MuJoCo Warp on Warp's CPU device)

`mjw_cpu_variants.py`, film scene, 4 worlds. Output in `check/mjw_cpu_variants_*.json`.

| variant | max \|Δq\| over 60 steps | first step with shoulder_lift \|Δq\| > 1e-6 |
|---|---|---|
| `7v4_real` | 2.6e-7 | — |
| `7v4_bam`, friction frozen | 1.6e-7 | — |
| `7v4_bam` + update, ctrl set *after* the CPU side's first `mj_forward` | 6.8e-4 | step 0 |
| same, with the solver at 100 iterations, at tolerance 0, or with the soft solref | 6.8e-4 (unchanged) | step 0 |
| `7v4_bam` + update, ctrl before the forward on both sides (`--ctrl-first`) | **8.0e-8** | — |
| same, over 400 steps | **3.0e-6** | step 167 |

**Cause:** the budget reads the last forward pass's `qfrc_actuator`. At ctrl 0 that is a full-clip torque toward 0. One
step of a different budget leaves a lasting offset inside the friction band. This is an experiment-backed diagnosis,
consistent with the BAM lines above.

### Step 2: GPU vs CPU check

`gpu_bam_check.py`, `check/gpu_bam_check.json`, `logs/check.log`. Setup:
- **GPU side:** `SO101Lift` from `so101_patched.py`, servo `7v4_bam`, `CtrlPlanFeeder`, 4 substeps per control step.
- **CPU side:** MuJoCo 3.14, the same scene, `reset_draw`, setpoint layer and power-on ctrl, with `BamFriction.update`
  before every `mj_step`.
- 8 worlds, seeds 10000-10007; 40 control steps.

Max \|Δq\| per joint, pan / lift / elbow / wflex / wroll / grip:

| schedule | GPU vs CPU | negative control: friction frozen | negative control: update one step late |
|---|---|---|---|
| steps (gpu_feeder_check's) | 9.8e-8 / 2.5e-7 / 2.1e-7 / **1.9e-6** / 2.4e-8 / 2.9e-7 | up to 1.4e-1 | up to 6.0e-2 |
| slow (ramps and holds) | ≤ 4.3e-7 | up to 3.1e-2 | up to 9.1e-4 |

- The single 1.9e-6 is wrist_flex in one world (seed 10004). Every other world is ≤ 4.4e-7.
- **Kernel alone,** on the GPU's own arrays after each step, against numpy float64: relative error ≤ 1.8e-7.
- **Budget difference GPU vs CPU** at control-step ends: ≤ 1.1e-6 N·m on the slow schedule.
  - On the steps schedule it reaches 1.4e-3 N·m at shoulder_pan and 2.1e-4 at lift; the budget there ranges up to
    5.9 N·m while joints hit their limits (τe includes the limit force).
  - Not traced further; q still agrees to 1e-7.
- Solver flags: `LS_ITERATIONS` in 8/8 worlds; no capacity overflow.

**Conclusion:** float32-level agreement.

### Step 3: deploy expert, 100 seeds (10000-10099)

Run: `eval_metalsim_bam.py 7v4_bam metalsim/deploy_7v4_bam`. Full table: `table.md`.

| run | success | up at 6 s | time to lift median (mean) s | retries | RMS jerk ×10³ (pan/lift/elbow/wflex/wroll/grip) | reversals /s | vel RMS > 5 Hz | lift-phase sag mrad (pan/lift/elbow/wflex/wroll) |
|---|---|---|---|---|---|---|---|---|
| CPU 3.10 so101Sim env (so101-sag `cpu/patched_7v4_bam`) | 100 | 100 | 2.19 (2.23) | 1 | 0.44/0.32/0.58/0.29/0.09/0.59 | 0.43/0.71/0.18/0.37/0.01/0.17 | 0.019/0.024/0.031/0.021/0.018/0.083 | 4/6/46/11/1 |
| **MetalSim** | **100** | **100** | **2.18 (2.23)** | **1** | 0.46/0.34/0.60/0.29/0.09/0.57 | 0.44/0.72/0.18/0.38/0.01/0.17 | 0.019/0.023/0.031/0.021/0.018/0.083 | 4/6/46/11/1 |
| CPU 3.14 film scene (`--cpu-dry 100`) | 100 | 100 | 2.18 (2.23) | 1 | 0.46/0.34/0.60/0.28/0.09/0.58 | 0.43/0.70/0.17/0.36/0.01/0.17 | 0.019/0.023/0.031/0.021/0.018/0.084 | 4/6/46/11/1 |
| `7v4_real` MetalSim (so101-expert), for scale | 98 | 98 | 2.46 (2.66) | 12 | 0.33/1.33/1.09/1.04/0.16/0.47 | 0.39/0.81/0.53/0.59/0.32/0.60 | 0.017/0.046/0.035/0.039/0.013/0.087 | 5/32/126/46/9 |

RMS acceleration, MetalSim against the reference, rad/s²: 2.3/5.0/4.5/4.5/3.2/7.6 against 2.2/5.0/4.4/4.5/3.2/7.6.

**Seeds that differ: none.** MetalSim against the CPU reference:
- same outcome on 100/100 seeds;
- time to lift within 0.04 s on every seed, identical on 71;
- the only retry seed is 10024 on both.

The same holds against the CPU 3.14 dry run (identical time to lift on 86 seeds), and with the reference's own ctrl-0
first budget (`--legacy-ctrl0`: 100/100 identical to MetalSim fixed).

For comparison, `7v4_real` agreed on 98/100 seeds, with 84/98 within 0.1 s. The friction makes the expert's outcome
far less sensitive to the backend.

**Harness checks:**
- My copy reproduces so101-expert's `7v4_real` CPU dry run bitwise.
- On MetalSim it gives the same summary (98, 2.46 (2.66), 12 retries, the same failures and retry seeds) with
  \|Δq\| ≤ 1.9e-3 rad. MetalSim runs are not bitwise repeatable with deterministic mode off.

**Solver flags in the 7v4_bam eval:** `ITERATIONS` in 2 worlds and `LS_ITERATIONS` in 100; `7v4_real` has
`LS_ITERATIONS` only. No capacity overflow.

### Step 4: cost per physics step, ratio only

`bench_bam.py`, `bench/ratios.json`. Setup: substeps 4, 7 interleaved rounds of 200 physics steps.

| ratio | 100 worlds | 1024 worlds |
|---|---|---|
| `7v4_bam` / `7v4_real` | 0.99 | 0.98 |
| update alone (bam / bam with friction frozen) | 1.02 | 1.00 |

The spread between rounds is 6-22 %, so the kernel's cost is below what this measurement resolves. Not a timing
machine.

### Torque use (for the stall question)

`stall_use.py`; logs `logs/stall_use_{dry,metalsim}.log`.

On `7v4_bam`, MetalSim and CPU agree:
- shoulder_lift and elbow_flex exceed the datasheet's 1.912 N·m on about 3 % of physics steps, in all 100 episodes:
  - peaks 2.30 and 2.36 N·m;
  - shoulder_lift mostly in approach, elbow_flex in lift;
- pan, wrist flex and wrist roll stay below it;
- the gripper is at its own 1.714 N·m cap on 87 % of steps.

No frames were rendered or looked at in this increment.

## Open

1. **Stall torque, BAM 3.32 against the datasheet's 1.91 N·m. Learned:**
   - The choice matters for this task's timing, not its success.
   - CPU ablation (`dry/dry100_7v4_bam_ds_stall`): BAM's kp 10.58 and damping 0.619 kept, the clip set to the
     datasheet (1.912, gripper 0.986). Overrides: `stall_Nm` 1.912, `P` 27.82, `noload_rad_s` 3.004.
     - Success stays 100/100.
     - Median time to lift rises 2.18 → 2.52 s. It rises on every seed, by 0.22-2.64 s (median +0.32 s). Retries
       rise 1 → 6.
     - Lift-phase sag is unchanged (4/5/47/12/1 mrad).
     - shoulder_lift and elbow_flex then sit at the clip on 5.4 % and 3.8 % of steps; the gripper on 94 %.
   - **Not learned:** which value the real servo has. The sag data constrain only the small-signal kp (so101-sag).
     The bench stall test in so101-sag Next 2 would settle it.
2. **BAM's own GPU port differs from its `model.py`** in the m6 quadratic term (`mjlab.py:516-530` against
   `model.py:183-203`). This code follows `model.py`, the code the parameters were identified with. The term's size
   is ≤ 0.0076 × τ²; its effect was not measured.
3. **The ctrl-0 first budget is also in so101Sim's env** (`so101_lift.py:104-122`) and in so101-sag's
   `eval_cpu_sag.py`. Fixed only in MetalSim's wiring and my harnesses; the effect there is ≤ 0.74 mrad and no
   outcome changes.
   - An RL loop that resets worlds inside a graph gets the budget at rest for the first step, as CPU `reset()` does.
     It does not get the holding-torque forward pass.
4. **`ITERATIONS` solver flag with the update:** 2/100 eval worlds and 13 % of bench worlds; never on `7v4_real` or
   frozen friction. Not investigated. q still agrees with CPU to float32 level in the check.
5. **The 1.4e-3 N·m budget difference** during joint-limit contact (steps schedule) was not traced.
6. **Docs not updated** (not mine): `docs/renders/SO101_LIFT.md` and `SO101_SERVO_SECTION` still describe
   `7v4_real` only.
7. **No 12 V BAM servo** (so101-sag Open 4).

## Next

1. **Orchestrator:** after so101-film finishes, run "Apply" above, then the listed tests. In the GPU queue:
   `scripts/gpu_run.sh <name> check 5 -- .venv/bin/python -m pytest -q tests/test_bam_friction.py`.
2. **Film on `7v4_bam`** (so101-sag Next 3):
   - Render the deploy expert with `adapter_kw={"servo": "7v4_bam", "controller": "deploy_expert"}`.
   - Look at the grasp with the 1.71 N·m gripper and at the box tilt.
   - Expected MetalSim outcomes per seed: `scratch/so101-bam-gpu/metalsim/deploy_7v4_bam/episodes.json` (for example
     seed 10051 lifts at 2.26 s; 10024 retries and lifts at 5.34 s).
3. **Stall torque on the bench** (so101-sag Next 2): block the joint, step the goal by 0.5 rad, read the force at a
   known lever. If it comes out near 1.91, use the ablation's overrides (Open 1) as the servo.

# Parity matrix against Isaac Sim / Isaac Lab (2026-09-23)

Every row names the Isaac behaviour, what this stack does, the **test that confirms or disproves
the equivalence** (file::function, with the assertion it makes), the measured result, and a verdict.

Verdicts: **confirmed** (a test asserts equivalence within a stated tolerance), **equivalent by
construction** (the same numbers/formulas are applied and a test checks they were applied, but the
reference implementation cannot run here), **partial** (a documented difference remains),
**disproved** (a test shows a difference), **not testable here** (needs Isaac Sim, which does not
run on Apple Silicon). All figures are **measured** on an Apple M4 Max (40-core GPU, 18.4 TFLOPS
FP32, 64 GB) unless marked **reported** (NVIDIA's published number, not re-measured).

The user's standing instruction for this document: honesty over inflated progress. Where a
comparison was not on the same asset, or a setting favours us, the row says so.

## 1. The benchmarked scene: Isaac-Velocity-{Flat,Rough}-G1-v0 on Isaac's own asset

Isaac Lab publishes step / step+inference / step+train throughput for this task at 4096 envs
(`benchmark_non_rl.py` protocol: uniform random actions in [-1, 1], 100 frames, wall time per
`env.step`). Our task is `metalsim.learn.g1_velocity` built from `assets/isaac/G1/g1_minimal.usd`,
the file Isaac Lab loads for the task (44 rigid bodies, 37 revolute joints, three convex colliders),
loaded through `metalsim.scene.usd_to_mjcf`.

### 1.1 Asset and physics model

| Isaac | ours | test (assertion) | result | verdict |
|---|---|---|---|---|
| `g1_minimal.usd` kinematics: joint frames (`localPos0/1`, `localRot0/1`, axis token) define each joint's anchor, axis and zero pose | USD→MjSpec: child body placed at F0·inv(F1), hinge axis from `localRot1`·axis | `tests/test_g1_parity.py::test_joint_frames_and_axes_match_usd` (poses our model at the USD's authored configuration, asserts every joint's world anchor and axis equal the USD joint frames to 1e-5) | 37 joints, worst anchor error 5.6e-7 m, worst axis error 2.4e-6 | confirmed |
| `PhysicsMassAPI` mass, diagonal inertia, centre of mass per link (unauthored COM = link origin) | copied per body (`explicitinertial`), COM sentinel handled | `tests/test_g1_parity.py::test_mass_properties_match_usd` (44 bodies, mass rel 1e-6, inertia rtol 1e-5, COM 1e-6 m; total 32.24 kg) | pass | confirmed |
| Collision set of `g1_minimal`: torso and both `ankle_roll` convex meshes | same three mesh geoms + ground | `tests/test_g1_parity.py::test_collision_set_matches_usd` | pass | confirmed |
| `ImplicitActuatorCfg` groups (G1_CFG: legs 150/200 Nm/rad, 5 Nms/rad, 300 Nm; feet 20, 2, 20 Nm; arms/hands 40, 10, 300 Nm; armature 0.01 / 0.001) override the USD drives | USD drives not converted (`drives=False`); MuJoCo affine actuators τ = kp(q*−q) − kd·q̇ clipped to ±effort; `dof_armature` set | `tests/test_g1_parity.py::test_actuator_gains_match_isaac_implicit_pd` (parses `G1_CFG` from Isaac's source with `ast`, asserts gain, bias, forcerange and armature per joint, all 37) | pass | confirmed (PhysX applies the same PD law inside its solver; MuJoCo applies it as an actuator force inside `implicitfast`, which also treats the damping implicitly) |
| Initial state: pos z 0.74, joint defaults (hip pitch −0.20, knee 0.42, ankle pitch −0.23, shoulders, elbows 0.87, fingers) | keyframe `init`; `qpos0` left at the USD zero (overwriting it would move every joint's zero) | `tests/test_g1_parity.py::test_scene_settings_match_isaac` (asserts the keyframe and that the same table appears in Isaac's source) | pass | confirmed |
| Gravity 9.81 (USD authors magnitude 0, which PhysX reads as default) | loader maps 0 → 9.81 | same test | pass | confirmed (an earlier build copied the 0 literally; found by the standing check) |
| Physics 200 Hz, decimation 4, 20 s episodes | timestep 0.005, 4 substeps per control step, `max_t` 1000 | `tests/test_g1_task_terms.py::test_observation_layout_and_action_mapping` | pass | confirmed |
| Solver: PhysX TGS, 4 position / 0 velocity iterations (Isaac Lab default), penetration-based contacts, friction 0.8 (static) / 0.6 (dynamic) | MuJoCo Newton, 10 iterations, line-search 20, pyramidal cone, `implicitfast`, eulerdamp off (MuJoCo Warp's own G1 benchmark settings), soft contacts (solref 0.02/1) | `tests/test_g1_parity.py::test_warp_matches_mujoco_c_under_pd_hold` (batched Metal vs MuJoCo C, identical state, 0.5 s incl. landing: max joint diff 1.2e-2 rad, median 3.6e-5, pelvis height 1 mm; both fall onto the torso by 2 s) | pass | **partial**: different contact/solver formulations by design (the plan accepts MuJoCo physics); what is confirmed is that our GPU physics equals our CPU reference, not PhysX |
| Zero-action behaviour: under Isaac's 20 Nm/rad ankle gains a PD hold at the default pose cannot resist the forward COM offset, the robot pitches onto its torso | same (pitches forward, torso contact at ~1.5 s) | same test, second half | pass | consistent with the published training curves (episodes start short); not a numeric Isaac comparison |

### 1.2 Task definition (observations, actions, rewards, terminations, commands, terrain)

| Isaac (`velocity_env_cfg.py`, `g1/rough_env_cfg.py`) | ours (`g1_velocity.py` kernels) | test | result | verdict |
|---|---|---|---|---|
| Policy obs: base lin vel (±0.1 noise), base ang vel (±0.2), projected gravity (±0.05), command (3), joint pos rel (±0.01), joint vel rel (±1.5), last action, height scan (rough, ±0.1, clipped ±1) | same order, same uniform noise ranges, 12 + 3·37 (+187) dims | `tests/test_g1_task_terms.py::test_observation_layout_and_action_mapping` (checks layout, noise bounds, projected gravity) | pass | confirmed |
| Action: `JointPositionAction`, scale 0.5, default offset, all 37 joints | ctrl = default + 0.5·a | same test (`ctrl == default + 0.5 a`) | pass | confirmed |
| Rewards (G1 rough weights): track_lin_vel_xy_yaw_frame_exp 1.0 (std 0.5), track_ang_vel_z_world_exp 2.0, feet_air_time_positive_biped 0.25 (thr 0.4), feet_slide −0.1, dof_pos_limits (ankles) −1.0, joint_deviation_l1 hips −0.1 / arms −0.1 / fingers −0.05 / torso −0.1, termination −200, flat_orientation −1.0, action_rate −0.005, dof_acc (hips, knees) −1.25e-7, dof_torques (hips, knees, ankles) −1.5e-7, ang_vel_xy −0.05, lin_vel_z 0 | same terms and weights, each term × dt as Isaac's manager does | `tests/test_g1_task_terms.py::test_reward_terms_match_isaac_formulas` (recomputes tracking, orientation, deviation and action-rate terms in numpy from Isaac's formulas on the live state; rtol 1e-4) | pass for the recomputed terms | confirmed for 5 of 13 terms by recomputation; the contact-based terms (air time, slide, torque, acceleration, limits) use MuJoCo touch sensors / `cvel` / `qfrc_actuator` and are equivalent by construction |
| Contact sensing: Isaac `ContactSensor` net force on feet/torso, history 3 | MuJoCo touch sensors on sites enclosing the colliders (threshold 1 N) | `tests/test_sensors_state.py::test_imu_and_contact_sensors_as_tensors` (touch ≈ m g at rest) | pass | partial (no 3-step force history; only the current step is used, which is what the reward terms read) |
| Terminations: time out; `illegal_contact` torso force > 1 N | same | `tests/test_g1_parity.py::test_warp_matches_mujoco_c_under_pd_hold` (torso contact ends the episode) | pass | confirmed |
| Commands: lin x ∈ [0, 1], lin y ∈ [−1, 1], heading command (stiffness 0.5), resample 10 s, 2 % standing | same | `g1_commands` kernel (not unit-tested separately; the term test reads its output) | – | equivalent by construction |
| Reset: base xy ±0.5 m, yaw ±π, joint pos scale (1, 1), zero velocities | same | `tests/test_g1_parity.py` (the Warp-vs-C test has to override this randomization to compare) | – | equivalent by construction |
| Rough terrain: `ROUGH_TERRAINS_CFG` 10 rows × 20 cols of 8 m cells, curriculum by row, sub-terrain mix (stairs up/down, boxes, random rough, slopes) | re-implemented from the config on a 0.1 m heightfield (`metalsim.learn.terrain`) | `tests/test_terrain.py::test_terrain_layout`; physics on it: `::test_hfield_mesh_contacts_match_mujoco_c` (xfail) | layout pass; contacts pass on the patched kernel (§1.6) | partial for the layout (same mix/ranges, not the same random heights); curriculum implemented (below) |
| Terrain curriculum `terrain_levels_vel`: envs start at random levels ≤ 5, move up a level after walking more than half a cell (4 m), down after walking less than half the commanded distance, random level at the top | same rule in the reward/termination kernel at episode end, origins re-drawn from the (rows × types) table; `max_init_terrain_level` 5 | rough smoke test: levels move in 55 of 64 envs over 122 random-action steps (down, as a falling policy should) | – | equivalent by construction |
| Height scanner: `RayCaster` 1.6 × 1.0 m grid at 0.1 m (187 rays), yaw-aligned, `body_z − hit_z − 0.5` | Warp kernel interpolating the heightfield triangles on the sim queue (capturable; a heightfield needs no ray tracing; `heightscan.metal` remains for arbitrary meshes) | `tests/test_terrain.py::test_height_scan_matches_mj_ray` (vs MuJoCo `mj_ray` on the same heightfield, 1496 rays) | median 0.0 m, 99th pct 0.0 m, max 0.85 m on cell diagonals where MuJoCo's triangulation differs | confirmed |
| Events (`EventCfg` as the G1 config leaves it): `physics_material` with degenerate ranges (static 0.8, dynamic 0.6, no randomization), `add_base_mass`, `base_com` and `push_robot` removed for the G1, `base_external_force_torque` with zero ranges, `reset_robot_joints` scale (1, 1), `reset_base` xy ±0.5 m / yaw ±π | friction 0.8/0.6 fixed; reset_base as Isaac; the removed/zero-range events have no effect to reproduce | read from `assets/isaac/g1_velocity_env_cfg.py` + `g1_rough_env_cfg.py` (`test_scene_settings_match_isaac` checks the init table) | – | equivalent (the earlier "partial" was over-cautious: none of Isaac's G1 events randomizes anything) |

### 1.3 Learner (rsl_rl PPO, `g1_rsl_rl_ppo_cfg.py`)

| Isaac | ours (`metalsim.learn.ppo_warp` with `g1_ppo_config`) | test | verdict |
|---|---|---|---|
| 24 steps/env, 5 epochs, 4 minibatches, lr 1e-3 adaptive (desired KL 0.01, ×/÷1.5), γ 0.99, λ 0.95, clip 0.2, entropy 0.008, grad norm 1.0, ELU MLP 512-256-128 (rough) / 256-128-128 (flat), init std 1.0 | same values; rollout policy evaluated in Warp with shared weights | `tests/test_warp_policy.py::test_parity_with_torch_and_shared_weights` (Warp MLP == torch to 1e-5; log-probs to 1e-4) | confirmed |
| Clipped value loss (`use_clipped_value_loss`) | implemented: max of clipped and unclipped value error, clip 0.2 (`PPOWarpConfig.clip_value`) | cartpole run with it on reaches the same 295/300 (`runs/bench12.log`) | confirmed |
| Rollout with no per-step host launches | whole rollout as replays of one captured graph | `tests/test_warp_policy.py::test_rollout_without_torch_launches` (runtime counters: 0 host syncs, 0 host ops) | confirmed |

### 1.4 Throughput on the G1 task (Isaac's protocol, 4096 envs)

| measurement (4096 envs unless stated; uncontended pass 2026-09-24, synchronized) | this stack (M4 Max, 18.4 TFLOPS) | Isaac Lab (RTX 4090, 82.6 TFLOPS), reported | raw ratio | per-TFLOPS ratio |
|---|---|---|---|---|
| G1 flat, physics only (4 substeps of MuJoCo Warp, graph replay) | 68,660 env-steps/s | – | – | – |
| G1 flat, step only (physics + reset/kinematics + obs + reward, one graph per step) | 55,176 env-steps/s (cost-split run 55,960; 45,958 before the post-reset forward pass was replaced by kinematics only) | 94,000 | 0.59× | 2.6× |
| G1 flat, step + inference (Warp MLP 256-128-128 inside the rollout graph) | 55,397 env-steps/s | 88,000 | 0.63× | 2.8× |
| G1 flat, full PPO loop (24-step rollout + 5 epochs × 4 minibatches on MPS, clipped value loss) | 47,809 env-steps/s (training log over 20 iterations: 51,742) | 82,000 | 0.58× | 2.6× |
| G1 rough (patched heightfield kernel, 187-ray height scan), physics only | 61,679 env-steps/s | – | – | – |
| G1 rough, step only | 55,423 env-steps/s (cost-split run 56,911) | 94,000 | 0.59× | 2.6× |
| G1 rough, step + inference (512-256-128) | 47,821 env-steps/s | 88,000 | 0.54× | 2.4× |
| G1 rough, full PPO loop | 39,764 env-steps/s (training log over 20 iterations: 42,173) | 82,000 | 0.48× | 2.2× |
| G1 flat, 2048 / 1024 envs: step only | 46,793 / 35,369 env-steps/s | – | – | – |
| G1 rough, 2048 envs: step only | 46,292 env-steps/s | – | – | – |

Per-step cost split at 4096 (flat): physics 59.7 ms, full env step 73.2 ms (the per-step
`reset_data` + kinematics for resets and the obs/reward kernels add 23 %; a full forward pass after
reset had cost another 14 ms and was unnecessary, since the next step recomputes everything the
observation does not read), PPO update 282 ms per 24-step rollout (12 % of the loop). Rough: physics
66.8 ms, step 72.0 ms, update 416 ms (the bigger network and 310-dim observations).

All rows above were measured on 2026-09-24 with nothing else on the GPU (`runs/bench12.log`; the
clean pass before the reset change is `runs/bench_clean.log`).

**Measured Isaac Lab reference on the same class of hardware we can rent** (GCP g2-standard-16, NVIDIA L4
24 GB, driver 580.178, Isaac Sim 5.1.0 + Isaac Lab v2.3.2, `benchmark_non_rl.py`, 100 frames, headless;
`runs/parity/isaac_bench/`): Cartpole-Direct 4096 envs **400,717** mean effective env-steps/s (NVIDIA
publishes 620K on an L40, 1.10M on a 4090); G1 rough 4096 envs **39,135** (L40 72K, 4090 94K). The L4's
FP32 peak is 30.3 TFLOPS, so per TFLOPS the L4 runs G1 rough at 1,292 env-steps/s versus this stack's
3,012 on the M4 Max (55,423 / 18.4); raw, the M4 Max is 1.4× the L4 on G1 rough and 0.15× on the
contact-free cartpole. The G1 flat and camera-cartpole reference runs are being re-run (the first
attempts died on a Kit single-instance lock). Scene setup on the VM: with Isaac Lab's debug
visualization on, building 4096 G1 envs did not finish in 65 minutes on the 2.2 GHz Xeon (USD marker
spawning); with it off, setup plus the benchmark took about a minute.

Notes that bear on reading these numbers honestly:
- Same asset, same actuator table, same task terms; **different physics engine** (MuJoCo Warp Newton
  solver vs PhysX TGS). MuJoCo Warp on Metal runs its full iteration budget every step (no early
  exit: `graph_conditional` is unavailable), so the budget is fixed at MuJoCo Warp's own G1 benchmark
  setting (10 Newton, 20 line-search). The line-search cap is hit in most worlds most steps
  (`overflow_flags()['LS_ITERATIONS']`); that is a cost cap, not an error.
- Constraint capacity `njmax` 256 per world (MuJoCo Warp's default 64 overflows and produces NaN).
- Isaac's step includes PhysX contact reporting for its sensors and the height-scan raycast on the
  terrain mesh; ours includes MuJoCo touch sensors and the Metal height scan.
- 4096 envs needed a change to the Warp Metal backend: MuJoCo Warp allocates temporaries every
  eager step and the backend deferred their release until the next host synchronize, so long loops
  failed with `kIOGPUCommandBufferCallbackErrorOutOfMemory`; frees are now released per completed
  command buffer. (`WP_METAL_INFLIGHT` and `WP_METAL_ICB_BATCH` knobs were added while diagnosing
  and default to no-ops.)
- "Synchronized" means the wall time of 110 steps with a device sync at the end. Isaac's script times
  each `env.step` call, which on PhysX includes `fetchResults` (a GPU wait); our per-call mean is
  absorbed by the Metal queue for the first ~64 calls and would read ~2× higher (95K), so it is not
  used.
- One graph replay per env step; the eager (per-kernel launch) path measures 32.8K at 4096.
- The earlier rough-terrain failure at 4096 envs (`kIOGPUCommandBufferCallbackErrorOutOfMemory`) was
  MuJoCo Warp launching its heightfield kernel with one thread per contact slot (`dim=naconmax`,
  1,048,576 at 4096 worlds with its heightfield default of 256 per world) and the kernel's large
  per-thread state; the task now sets 32 contact slots per world (3 colliders × ≤ 4 kept contacts),
  which is also the flat setting.


**Engine comparison at the training setting** (measured 2026-09-24, `scripts/diagnostics/g1_engine_bench.py`,
`runs/g1_engine_bench.log`; 4096 envs, graph-captured, synchronized, idle GPU; the rows above use the
5 ms MuJoCo Warp setting of Isaac's task file, the 2.5 ms row is what the G1 actually trains at, §1.5):

| engine / setting | physics only | full env step | rollout + inference | full PPO loop |
|---|---|---|---|---|
| MuJoCo Warp, 2.5 ms (training setting) | 35,458 | 30,660 | 28,901 | **27,662** env-steps/s |
| MuJoCo Warp, 5 ms (the rows above) | 67,914 | 55,741 | 55,217 | 50,907 |
| Newton XPBD, 4 it. at 1.25 ms (training setting) | 146,857 | 146,822 | 135,708 | **112,609** (4.1× MuJoCo Warp; 1.4× Isaac's published 4090 PhysX number, on a 4.5× smaller GPU) |
| Newton XPBD, 8 it. at 2.5 ms | 161,610 | 162,219 | 147,609 | 120,413 |

Nothing is excluded from the PPO-loop column (observations, rewards, resets, inference, update).
Newton's observation/reward work is almost free (full step ≈ physics only); MuJoCo Warp spends
~18 ms per step in its reset and kinematics calls. The Newton rows predate the final actuator kernel
(the final recipe's training run logged 107,355 env-steps/s including the anomaly monitor); a
re-benchmark with the final recipe is queued. Caveat for the Isaac comparison: PhysX vs XPBD is a
different solver; Isaac Lab 3.0 itself moves to Newton, so the like-for-like number will be Isaac
Lab 3.0's, which is not measured here.

### 1.5 Learning on the G1 task

**Two defects found by running Isaac's full 1,500 iterations, both now fixed.**

1. *Physics* (§1.6 below and `scripts/diagnostics/g1_drive_stability.py`): Isaac's kp 200 drives are explicit
   stiffness in MuJoCo and blow up at the 5 ms step; the 5 ms run logged 190 blown-up episodes per window by the
   end, the same run at **2.5 ms** logged 1 in 1,500 iterations. `physics_dt=0.0025` is now the default for
   training on this asset (2× physics cost per control step).
2. *Reward port bug*: Isaac Lab's `RewardManager` multiplies every term by the step dt, **including the
   termination penalty** (−200 × 0.02 = −4 per fall). Our kernel added −200 unscaled, so an episode that
   survived 400 steps of small penalties scored below one that fell at once, and PPO learned to fall: both the
   5 ms and the 2.5 ms runs peaked at 600 / 383 steps around iteration 750 and then collapsed to ~130–150
   while the return fell to −290 / −309. Fixed in `g1_reward_done` (verified against
   `isaaclab/managers/reward_manager.py` line 150 at v2.3.2).

Runs on record: 5 ms + unscaled penalty `runs/g1_flat_ppo_1500.log` (peak 597 steps at it 750, 190 blow-ups);
2.5 ms + unscaled penalty `runs/g1_flat_ppo_1500_dt25.log` (peak 383 at it 750, 1 blow-up, same collapse);
2.5 ms + corrected penalty `runs/g1_flat_ppo_1500_dt25_fixed.log`: **no blow-ups; episode length 41 → 62 (it 250)
→ 115 (500) → 599 (750) → 764 (it 1000) → 732 → 558 (it 1500)**, so the identical config now keeps the robot
up for 11–15 s of the 20 s horizon through the second half of training instead of collapsing. The
return is still negative and falls as episodes lengthen (−5 → −18 → −28 → −26): the per-step penalties
(joint deviation, action rate, torques, orientation) outweigh the tracking terms, i.e. the policy learns
to stay up, not to walk on command. Whether that is a remaining reward-term mismatch or a physics
effect is exactly what Isaac's own rsl_rl log of the same task (per-term reward means, recorded on
the L4) will show; that comparison is pending. Isaac's own training of the
same task on the L4 (rsl_rl, same config) is recorded on the VM for the side-by-side comparison.

Rough, 2048 envs, Isaac's rough PPO config (512-256-128), 150 iterations on the patched kernel
(`runs/g1_rough_ppo_150.log`): no blow-ups (the guard counted none), episode length 40 → 50.5 control
steps, return −203 → −204, 25.7K env-steps/s end to end while another job shared the GPU. Slower
than flat: the adaptive schedule keeps the learning rate at 1e-5 to 4e-4 because the 310-dim
observation (187 height-scan values) makes the KL estimate exceed 0.02 in most updates; Isaac's
rough runner is configured for 3,000 iterations, so 150 is a smoke test of stability, not a result.

**Per-term comparison with Isaac's own rsl_rl run** (Isaac Sim 5.1 / Isaac Lab 2.3.2 on the L4,
`runs/parity/isaac_train_g1_flat_terms.txt`; ours from `scripts/diagnostics/g1_reward_terms.py`, mean
action, 1024 envs × 1000 steps, in Isaac's units: per-second average of weight × term × dt over the
episode; measured 2026-09-24):

| | Isaac it 100 | Isaac it 200 | Isaac it 300 | MetalSim it 300 | MetalSim it 1000 | MetalSim it 1500 |
|---|---|---|---|---|---|---|
| mean episode length (steps) | 200 | 981 | 1000 | 61 | 412 | 279 |
| mean return | −6.6 | +6.6 | +19.2 | −3.7 | −2.6 | −4.5 |
| track_lin_vel_xy_exp | 0.060 | 0.782 | 0.897 | 0.018 | 0.112 | 0.090 |
| track_ang_vel_z_exp | 0.019 | 0.208 | 0.504 | 0.033 | 0.196 | 0.120 |
| feet_air_time | 0.003 | 0.021 | 0.028 | 0.002 | 0.013 | 0.011 |
| feet_slide | −0.017 | −0.046 | −0.022 | −0.008 | −0.043 | −0.028 |
| joint deviation (hip + arms + fingers + torso) | −0.047 | −0.197 | −0.162 | −0.025 | −0.224 | −0.203 |
| flat_orientation_l2 | −0.013 | −0.010 | −0.009 | −0.004 | −0.021 | −0.016 |
| action_rate_l2 | −0.083 | −0.321 | −0.230 | −0.001 | −0.026 | −0.019 |
| termination_penalty | −0.200 | 0.000 | 0.000 | −0.200 | −0.140 | −0.177 |
| time-outs (fraction of episodes) | 0.00 | 0.99 | 1.00 | 0.00 | 0.30 | 0.11 |

Isaac's action_rate term is dominated by its exploration noise (std 0.72–1.0 on 37 joints under
stochastic actions; ours above is the mean action), so it is not comparable; every other row is. The
picture: Isaac's policy is tracking the command by iteration 200 (0.78 of the maximum 1.0) with no
falls, and by iteration 300 it is at 0.90; ours never learns to track (0.11 at best) and keeps falling
(termination −0.14 to −0.20 per second means 70–89 % of episodes end in a fall), while its joint
deviation and slide costs at iteration 1000 are as large as Isaac's at 200 (it moves as much, to no
effect). This is a factor-7 gap in tracking and a learning-speed gap of more than 5× (Isaac's
iteration 100 ≈ ours at 300). Whether the learner or the simulation is responsible is being decided by
a learner differential: the real rsl_rl library trained on our task through a VecEnv adapter, with
Isaac's exact configuration (`metalsim/learn/train_g1_rslrl.py`, in progress), while the Newton
backend answers the physics half (§2.1).

**Same task on Newton XPBD** (`G1VelocityTask(engine="newton")`, 4 it. at 1.25 ms, same PPO config and
seed, 1500 iterations, `runs/g1_flat_newton_ppo.log`, `scripts/diagnostics/compare_training_logs.py`;
measured 2026-09-24; return and length averaged over ±5 iterations):

| iteration | MuJoCo Warp return / length | Newton return / length |
|---|---|---|
| 100 | −5.2 / 50 | −5.4 / 45 |
| 300 | −5.1 / 65 | −5.9 / 102 |
| 500 | −6.4 / 118 | −11.3 / 303 |
| 750 | −18.6 / 634 | −16.5 / 508 |
| 1000 | −24.3 / 773 | −19.3 / 586 |
| 1500 | −25.4 / 544 | −23.6 / 628 |

The curve has the same shape on both engines (episodes lengthen to 500–770 steps while the return
falls; neither tracks), so the learning gap against Isaac is **not an engine effect**; Newton reaches
each milestone 4–6× sooner in wall-clock (episode length 200 at 6.3 min vs 37 min). This leaves the
learner as the prime suspect (the learning-rate trace: ours decays 2.6e-4 → 2.3e-5 over the run
while the per-iteration KL stays at 0.011–0.017 and never drops below rsl_rl's raise threshold of
0.005; Isaac's action std falls 1.0 → 0.64 by iteration 600); the rsl_rl-on-our-task differential is
running. Also found on the way: the reference path's feet-slide term used MuJoCo's `cvel` (spatial
velocity at the subtree COM) as the foot velocity instead of the foot body's own linear velocity
(Isaac's `body_lin_vel_w`); the fix on both engines is in progress. Newton preflight: termination −4
and the PPO probe pass; the random-policy return (−8.3, bound −8) and the 3σ stress check (37 of 1024
worlds blow up in 400 steps at the fast settings; 8 it. at 0.625 ms passes but costs the speed gain)
fail, so the fast Newton settings are validated for the 1σ training regime only.

### 1.6 Rough terrain: MuJoCo Warp's heightfield contacts, found defective and patched

`tests/test_terrain.py::test_hfield_mesh_contacts_match_mujoco_c`: G1 on Isaac's rough-terrain
layout under a PD hold, MuJoCo Warp vs MuJoCo C on the same model. **Before the patch** (xfail at
commit `066db94`): heightfield-mesh contacts returned an inverted normal (z = −1.000) with a 2.05 m
penetration and a witness point 1.1 m from the foot; one of four worlds was launched (pelvis 2.31 m
vs 0.74 m in C); under random actions joint velocities reached 1e13 rad/s and the acceleration and
torque reward terms exploded (rough PPO run, iteration 3: return −8.7e25).

Localization: the defect reproduced on Warp's CPU device (so MuJoCo Warp's algorithm, not the Metal
backend), was independent of `ccd_iterations`/`ccd_tolerance` (50→500, 1e-6→1e-8), did not occur
for spheres (five placements on a block terrain match C exactly), and had a minimal repro: a box
mesh pressed 3 cm into a flat heightfield yields a sideways normal (0.43, −0.90, 0.09). The kernel
tests each terrain prism (a triangular column down to the heightfield base, 1.9 m here) against the
convex geom with single-witness-point GJK/EPA and keeps the minimum-distance result; MuJoCo C's
`mjc_ConvexHField` produces face-contact manifolds through its multi-contact CCD, which MuJoCo Warp
does not support for HFIELD–MESH pairs (its own warning).

**Patch** (`patches/mujoco_warp-hfield-plane-contacts.patch`, branch `metalsim` of
https://github.com/pulipakaa24/mujoco_warp, gated by `HFIELD_PLANE_CONTACTS`): per prism, the
contact is the deepest vertex of the convex geom below the prism's top-triangle plane whose
footprint lies in that triangle's column; the normal is the triangle normal. **After**: the test
passes (min normal z 0.41, max penetration 3.8 cm against C's own 4.4 cm for the same initial
placement, pelvis heights within 3 cm of C over 0.5 s), and 400 control steps of Isaac-scale random
actions (unit-std Gaussian × 0.5) at 1024 envs stay bounded exactly like flat ground (|q̇| ≤ 56
rad/s, no non-finite states). At 3× that amplitude MuJoCo C itself
blows up on both flat and rough ground (|q̇| 8.8e6 rad/s flat, 1.3e5 rough; touch forces of 1e8 N),
so that regime is a MuJoCo-vs-PhysX robustness difference under violent position targets with
Isaac's gains, not a MetalSim defect; it is listed in §3.

Difference that remains: one contact per (triangle, geom) with MuJoCo Warp's 4-contact selection
per pair versus C's full manifold, and the plane contact measures depth to the top plane of a steep
triangle rather than to the prism's nearest face, so contacts at step risers are stiffer than C's.
Rough-terrain numbers below are therefore **measured on the patched kernel** and labelled as such.

### 1.7 Fidelity protocol against Isaac Sim 5.1 on an L4 (recorded 2026-09-24, PhysX + RTX)

`metalsim/parity/isaac_side/record_g1.py` runs three open-loop protocols in Isaac-Velocity-Flat-G1-v0
(no reset randomization, no terminations, command (0.5, 0, 0)) and records joint state, root pose,
torques, contact forces and camera frames (1024×576, camera at (3.2, −2.4, 1.4) looking at
(0.3, 0, 0.6), vertical FOV 42°, one sun DistantLight 3000 + uniform dome 400) under the RTX real-time
renderer ("rt", every 5 control steps) and the RTX path tracer ("pt", 32 spp, every 25).
`metalsim.parity.record_g1` replays the same protocols on the same asset at 2.5 ms (decimation 8) from
Isaac's recorded initial state and action sequence and renders tier 2 (16 spp × 4 passes) and tier 0
from the same camera and lights. `metalsim.parity.compare` writes `runs/parity/report_{rt,pt}/report.json`.
All numbers below are **measured**; Isaac's side ran PhysX at its task default (5 ms, 4 substeps).

**Physics** (env 0; joint RMSE over the 37 joints; divergence = first step with any joint > 0.1 rad off):

| protocol | joint RMSE 0.5 s / 1 s / 2 s / end | max | divergence | root z end Isaac / MetalSim (RMSE) | orientation err mean / max | contact peak Isaac / MetalSim | torque RMS Isaac / MetalSim | peak joint speed |
|---|---|---|---|---|---|---|---|---|
| A_hold (zero action, 3 s) | 0.005 / 0.006 / 0.009 / 0.027 rad | 0.045 | 1.34 s | 0.054 / 0.054 m (0.013) | 0.015 / 0.109 rad | 681 / 3866 N | 5.30 / 5.25 Nm | 6.9 / 6.3 rad/s |
| B_random (3σ random targets, 5 s) | 0.116 / 0.125 / 0.101 / 0.034 rad | 0.209 | step 0 | 0.143 / 0.150 m (0.067) | 2.216 / 2.974 rad | 2890 / 1753 N | 30.1 / 28.5 Nm | 31.9 / 47.2 rad/s |
| C_drop (zero action from 1.0 m, 3 s) | 0.014 / 0.015 / 0.006 / 0.024 rad | 0.064 | none | 0.054 / 0.054 m (0.035) | 0.058 / 0.237 rad | 1030 / 3499 N | 6.17 / 6.19 Nm | 10.6 / 7.2 rad/s |

Reading: under Isaac's own gains the G1 cannot hold its default pose in either engine (both pitch
forward and end on the torso at pelvis z 0.054 m, cf. §2.1); the hold and the drop agree to a few
hundredths of a radian throughout, the same final resting state, and torque RMS within 1 %. The
random-target protocol is chaotic and the two robots fall in different directions (mean orientation
error 2.2 rad), so its per-joint numbers measure divergence of a chaotic trajectory, not a model
difference; the peak joint speed (47 vs 32 rad/s) and the contact peaks are the informative rows.
Two differences are systematic: (i) MetalSim's contact-force peaks in the falls are 3.4–5.7× Isaac's
(3866 vs 681 N, 3499 vs 1030 N) with equal mean totals (319 vs 304 N, 332 vs 289 N): MuJoCo's soft
contact at τ 20 ms resolves the impact into a shorter, higher spike; (ii) Isaac's finger joints move
at t = 0 (±0.085 rad, 0.34 rad/s, decaying within 0.5 s; hip yaw −0.10 rad in the drop) while ours stay
at their targets — a start-up transient on the PhysX side (hypothesis: finger self-collision at the
default pose; not verified). The 0.1 rad divergence step of A_hold (1.34 s) is at the moment of the
torso impact, where these transients and the contact model meet.

**Rendering** (Isaac RTX vs MetalSim, same state per frame; brightness-matched = MetalSim scaled to
Isaac's mean, because the engines' light units differ):

| Isaac renderer | MetalSim tier | frames | whole frame, raw PSNR / SSIM / LPIPS / FLIP | whole frame, brightness-matched | robot pixels only (states agree, IoU > 0.7), brightness-matched PSNR / SSIM / FLIP, LPIPS on the crop | silhouette IoU | robot depth RMSE |
|---|---|---|---|---|---|---|---|
| RTX real-time | tier 2 (16 spp × 4) | 110 | 12.7 dB / 0.62 / 0.69 / 0.60 | 17.6 dB / 0.67 / 0.65 / 0.41 | 13.7 dB / 0.48 / 0.024 / 0.041 (n = 51) | 0.83 | 2.6 cm |
| RTX real-time | tier 0 (raster) | 110 | 12.6 / 0.68 / 0.67 / 0.63 | 12.4 / 0.68 / 0.67 / 0.64 | 11.5 dB / 0.35 / 0.028 / 0.061 | 0.83 | 2.6 cm |
| RTX path tracer, 32 spp | tier 2 | 22 | 12.0 / 0.61 / 0.70 / 0.62 | 17.3 / 0.67 / 0.66 / 0.42 | 13.8 dB / 0.49 / 0.026 / 0.040 (n = 12) | 0.84 | 2.9 cm |
| RTX path tracer, 32 spp | tier 0 | 22 | 12.7 / 0.68 / 0.68 / 0.62 | 12.6 / 0.68 / 0.68 / 0.62 | 11.8 dB / 0.38 / 0.029 / 0.058 | 0.84 | 2.9 cm |

How to read these. The **whole-frame** numbers are dominated by a scene mismatch, not by the
renderers: Isaac Lab's plane terrain ignores `visual_material`, so Isaac's frames show its default
blue grid ground and the neighbouring envs on the horizon, while MetalSim renders the grey plane the
script asked both for (see the gallery composites). Brightness matching helps tier 2 by 5 dB because
the engines' light units differ (MetalSim tier 2 is 1.3× brighter on the frame, 1.08× on the robot).
The **robot-only** columns mask everything outside the union silhouette and average over the robot's
pixels on frames where the physics states agree, so they compare the shading of the same asset with
the same materials, sun and dome: tier 2 is 2 dB / +0.13 SSIM closer to RTX than tier 0, and Isaac's
real-time and path-traced frames are equally far from ours (they are within 0.2 dB of each other),
i.e. the gap is in the material and light model (MetalSim's plates render darker, its sun shadow is
hard-edged; no denoiser, no area lights), not in noise. 13.7 dB / 0.48 SSIM on the robot is a **large**
remaining gap and is reported as such; it is measured, not estimated. Silhouettes agree (IoU 0.83 on
agreeing states; 0.48–0.52 over all frames because B_random diverges) and the robot's z-depth agrees to
2.6–2.9 cm RMSE; the ground plane depth agrees to 7 mm, of which ~4 mm is tier 2's own depth noise.
Depth conventions: both engines write z-depth; Isaac writes inf for the sky and clips at the 100 m far
plane, MetalSim writes 0 on a miss (`compare.robot_mask`). A re-recording with the grey ground bound
on the Isaac side (`stage6.sh`, one env, no horizon robots) is queued to replace the whole-frame rows.

## 2. Platform capabilities (the workstreams), with the tests behind them

| capability | Isaac | ours | test (assertion) | result | verdict |
|---|---|---|---|---|---|
| GPU physics equals the CPU reference | PhysX GPU == PhysX CPU (NVIDIA's claim) | MuJoCo Warp on Metal == MuJoCo C | `tests/test_physics.py::test_one_step_parity_vs_mujoco_c` (SO-101, Panda, Go1, G1 one step at MuJoCo Warp tolerances), `::test_trajectory_parity_so101_arm_joints` (200 steps, max err < 2e-2, no NEFC overflow, all worlds identical) | pass; SO-101 trajectory 4.5e-7 | confirmed |
| Zero host synchronization per step | tensors stay on device; one sync per rollout | runtime counters as the conformance instrument | `tests/test_physics.py::test_step_does_not_block_host` (0 syncs, 0 host ops, host time < 25 % of GPU time and independent of batch size); `tests/test_render_tier0.py::test_gpu_path_with_batchsim_no_host_sync`; `tests/test_sensors_rt.py::test_gpu_path_ordering_with_sim` | pass | confirmed |
| Zero-copy tensors between physics/render and the learner | PhysX views as torch tensors | Warp Metal buffers aliased as MPS tensors, event ordering across queues | `tests/test_interop.py` (11 tests: alias, lifetime, 2-D/vec dtypes, wait/signal both directions, ping-pong) | pass | confirmed |
| Per-world physics randomization | PhysX per-env mass/friction | per-world model fields + `recompute_constants` | `tests/test_physics.py::test_per_world_model_fields` | pass | confirmed |
| USD scene layer | USD + UsdPhysics/UsdShade/UsdSemantics; URDF/MJCF importers | MJCF/URDF→USD, USD→MjSpec (lossless + generic UsdPhysics) | `tests/test_scene_usd.py` (3 tests: import fields, round trip to MuJoCo, URDF), the G1 rows above (a real Isaac asset) | pass | confirmed for the subset used; MaterialX graphs, Hydra viewing not implemented |
| Renderer tier 0 (raster + PBR + shadows) vs a reference rasterizer | Isaac RTX rasterizer | native Metal raster reading physics buffers | `tests/test_render_tier0.py::test_silhouette_and_depth_and_seg_parity_primitives` (vs `mujoco.Renderer`: IoU > 0.95, depth median < 1 cm, seg IoU > 0.85), `::test_texture_pattern_parity` (corr > 0.97) | IoU 0.994–0.997, depth 0.1 mm, corr 0.996 | confirmed vs MuJoCo; **not testable vs Isaac RTX** here |
| Renderer tier 1 (hybrid RT: soft shadows, AO, reflections) | RTX real-time | fragment-stage Metal ray queries | fidelity benchmark (`metalsim.bench.render_fidelity`): PSNR 19.7 / FLIP 0.30 vs MuJoCo; rollout: `tests/test_render_tier2.py` sibling path (`CartpoleRGBEnv(tier=1)`) | measured | partial (no denoiser / MetalFX upscale yet) |
| Renderer tier 2 (path tracer) | RTX path tracer (Isaac "PathTracing" mode) | progressive Metal RT path tracer, NEE, GGX+Lambert, Russian roulette | `tests/test_render_tier2.py::test_lambertian_plane_analytic` (radiance 0.4000 exact), `::test_white_furnace_uniform_sky` (0.498 vs 0.5), `::test_monte_carlo_convergence` (noise ratio 4.05 for 16× spp; ideal 4.0), `::test_direct_light_matches_tier0` (PSNR 43.7–45.1 dB, depth 1 mm, seg IoU 1.0), `::test_gpu_path_from_batchsim_matches_host`, `::test_cartpole_rgb_rollout_tier2` (full env rollout at tier 2) | pass | confirmed radiometrically; not compared to Isaac's path tracer (not testable here); no denoiser |
| Camera-based RL: Isaac-Cartpole-RGB-Camera-Direct-v0 with the skrl agent Isaac Lab ships (image-only 100×100, NatureCNN + 512 ELU, 64-step rollouts, 4 epochs, 32 minibatches, lr 1e-4 KL-adaptive 0.008, value clip, per-image mean subtraction, running value scaler) | same config on `metalsim.learn.ppo` over the tier-0 renderer, 1024 envs (`metalsim.learn.train_cartpole_rgb`) | `runs/camera_cartpole_tier0.log`: 8M steps, return 8 → 85 (episode length 27 → 133 of 300), still rising; 7.5K env-steps/s incl. training, 76 % of it the CNN update on MPS | learns from pixels; Isaac publishes throughput only (32K on a 4090, 21K on an L40; measured 14.2K rollout-only on the L4), no curve to match; tier 1 and tier 2 runs pending | confirmed (learning), partial (throughput) |
| Tier 1 / tier 2 full rollouts (throughput) | Isaac Cartpole-RGB: 50K steps/s at 1024 envs on a 4090 (rasterized RTX, reported); **measured on the L4** (Isaac Sim 5.1, `benchmark_non_rl.py --enable_cameras`, 1024 envs, 100 frames, 2026-09-24): mean 14,233, max 15,570 env-steps/s | Cartpole-RGB 100×100, 1024 envs, physics + render + reward/reset, one run each, uncontended | `metalsim.learn.cartpole_rgb` (`python -m metalsim.learn.cartpole_rgb 1024`) | tier 0 47,911; tier 1 43,368; tier 2 (1 spp, 1 bounce) 36,896; tier 2 (1 spp, 2 bounces) 35,160; tier 2 (4 spp, 2 bounces) 19,894 env-steps/s (uncontended pass 2026-09-24) | measured; the tier-2 rollout is a full path-traced observation stream at 70 % of the raster rate (1 spp, noisy) |
| Ray-traced sensors (lidar, depth) | RTX lidar | Metal RT acceleration structures refit from physics | `tests/test_sensors_rt.py::test_lidar_vs_mujoco_ray` (median < 2 mm, 95th < 2 cm, hit pattern within 3 %), `::test_raycast_depth_vs_raster_depth` (99.5 % agree, median < 1 mm) | pass | confirmed vs MuJoCo `mj_ray`; **not testable vs Isaac RTX lidar** |
| Lidar-based RL (no Isaac Lab benchmark task exists; MetalSim task) | – | `metalsim.learn.lidar_nav`: 64-beam planar Metal RT lidar, goal navigation among 12 random boxes, vectorized PPO | `tests/test_lidar_nav.py` (scan == `mj_ray`, obs/reward invariants); run `runs/lidar_nav_ppo_300.log`: 1024 envs, 300 iterations, 7.37 M env-steps, 21.2 K env-steps/s including the update; success 1.00 → 0.99 while time-to-goal falls 218 → 69 control steps and contacts/step 0.012 → 0.014 | measured 2026-09-24 | confirmed (learns; no Isaac reference to compare against) |
| IMU / contact / joint sensors as tensors | Isaac sensors | MuJoCo sensors evaluated per step | `tests/test_sensors_state.py::test_imu_and_contact_sensors_as_tensors` (accelerometer 9.81 at rest, ~0 in free fall; touch ≈ m g) | pass | confirmed |
| Height scan on terrain | `RayCaster` | Metal ray queries | `tests/test_terrain.py::test_height_scan_matches_mj_ray` | pass | confirmed |
| Replicator: randomizers, annotators, writers | Omniverse Replicator | GPU randomizers, 2-D/3-D boxes, semantic seg, COCO/KITTI/basic writers | `tests/test_replicator.py::test_randomize_annotate_write` (boxes equal the segmentation's extents, files written) | pass | confirmed for the implemented subset |
| Warp rollout policy | – | MLP policy in Warp with shared torch weights | `tests/test_warp_policy.py` (2 tests) | pass | confirmed |
| Training anomaly monitor (task-agnostic) | rsl_rl logs KL, value loss, entropy, per-term rewards | `metalsim.learn.monitor.AnomalyMonitor` on every log point: KL vs target band, value explained variance, action-std collapse/explosion, LR pinned at bounds, terminal-reward dominance and return-vs-length direction (survival penalties), non-finite rows; physics invariants from the sim buffers (joint speed vs actuator limits, joint-limit violation, contact penetration, contact force vs weight, energy jumps, capacity overflows) | first probe on the corrected G1 config flagged two real items on its own: joints pushed 0.17 rad past their limits and contacts penetrating 5–7 cm during falls (MuJoCo's soft limits and soft contacts vs PhysX's hard ones) | in use; findings feed §3 |
| RL reproduces a published Isaac Lab result | Cartpole-Direct with rsl_rl config learns | same config on the Warp path reaches 295/300 in 45 iterations | `metalsim.learn.ppo_warp` run (docs/PHASES.md) | measured | confirmed (equivalent MJCF cartpole, not Isaac's USD) |

## 2.1 Contact model: what MuJoCo can do, and Newton XPBD on Metal (measured 2026-09-24)

MuJoCo's contacts are soft by construction (regularized convex constraints with a time constant
`solref[0]`, penetration is the state of that spring-damper), not because of missed collisions; `margin`/
`gap` already give speculative pre-impact contacts. G1 drop test at 2.5 ms (`scripts/diagnostics/
g1_contact_stiffness.py`): default τ 20 ms → 2.97 cm impact penetration, 0.06 cm at rest, 4.8× weight
peak; τ 5 ms (the 2·dt floor) + impedance 0.99 → 0.85 cm / 0.00 cm / 14×; + speculative contact
(margin = gap = 1 cm) → 0.00 cm / 0.00 cm / 21×. PhysX on Apple Silicon is not an option (GPU PhysX is
CUDA; the CPU SDK has no supported Apple build).

NVIDIA's Newton engine (Warp-based; Isaac Lab 3.0's physics layer) runs on the MetalSim Warp fork
unmodified: its **XPBD** solver (same substepped position-based family as PhysX TGS), Featherstone and
Semi-Implicit run on Metal; VBD fails to compile; the GJK/MPR narrow phase needed Warp fixed-size arrays,
added to the Metal codegen in fork commit 786cdae (see below). Newton's UsdPhysics
importer loads `g1_minimal.usd` directly (44 bodies, 44 joints, 43 DoF, drives included). A box rests
with 0.3 mm penetration under XPBD.

Newton XPBD on Metal, after the roadblocks were worked (2026-09-24, all numbers measured on the M4
Max, `metalsim/physics/newton_backend.py`, `scripts/diagnostics/newton_xpbd_*.py`,
`newton_mesh_narrowphase.py`):

* **Drives.** Three causes, none Metal-specific (Metal and CPU agree to 1e-6): XPBD never writes
  `State.joint_q` (read joints with `newton.eval_ik`); `ModelBuilder.joint_target_q` is in the coordinate
  layout (7 root slots) while the gains are in the DOF layout (6), so DOF-indexed targets shift every G1
  target by one joint; and Newton's default joint relaxation (linear 0.7, angular 0.4) transmits joint
  torque wrongly (one-joint pendulum: +43 % response to a joint torque, −18 % to gravity, at any
  iteration count; equal factors 0.4/0.4 are exact, angular > 0.4 diverges on the G1). XPBD's own
  compliance drive also has an effective stiffness that is not `ke` (72–1240 Nm/rad for `ke` 200 over
  1–16 iterations), so Isaac's PD law is applied every substep through `Control.joint_f`
  (`ActuatorPD`: implicit damping, effort clip, armature added isotropically to the child inertia).
  Error against the exact static PD equilibrium with the pelvis fixed: XPBD drive 16 it. 0.03 rad legs /
  0.14 ankles / 0.17 arms and hands; `ActuatorPD` 1 it. 0.006 / 0.002 / 0.001; 4 it. ≤ 0.001 rad.
* **Standing.** Under Isaac's gains MuJoCo C itself pitches forward and lands on its torso at ~1.4 s
  (the 20 Nm/rad ankles cannot hold the default pose), so the criterion is "tracks MuJoCo C". Pelvis
  height at 0.25 / 0.5 / 0.75 / 1.0 / 1.25 s: MuJoCo C 0.710 / 0.708 / 0.692 / 0.619 / 0.329 m; XPBD
  4 it. at 1.25 ms 0.704 / 0.701 / 0.683 / 0.604 / 0.250 (worst joint difference at 0.5 s ≤ 0.022 rad);
  8 it. at 2.5 ms 0.697 / 0.689 / 0.677 / 0.591 / 0.214 (≤ 0.044 rad). Newton rests about 1 cm lower
  than MuJoCo (0.69–0.70 vs 0.71 m); cause not found.
* **Mesh colliders on Metal.** Warp fork commit 786cdae adds fixed-size arrays to the Metal codegen
  (a per-thread slice of GPU scratch), so the GJK/MPR narrow phase compiles: Warp's fixed-array tests
  pass 20/20 on CPU and Metal including graph capture; a box as convex hull or triangle mesh gives the
  same 4 contacts as the box primitive (−1.000 mm, 0.043 mm resting penetration, Metal = CPU to
  0.001 mm); the G1 runs on its own convex-mesh colliders with the same trajectory as the box stand-ins.
* **Effort limits and armature.** Peak leg torque is 61–63 Nm in the hold and 173–188 Nm on a drop
  against the 300 Nm cap; only the 20 Nm ankles saturate (0.5–1.5 % of steps during landing). Armature is
  essential (every setting goes NaN without it) and must be added on all three axes of the link inertia
  (axis-only makes 32 links' inertia invalid and Newton inflates them up to 7×). Featherstone: NaN at
  2.5 ms, no result at 1.25 ms within 400 s.

Physics-only throughput, graph-captured, idle GPU, in 2.5 ms-equivalent steps/s
(`scripts/diagnostics/newton_xpbd_throughput.py`):

| envs | MuJoCo Warp | XPBD 4 it., 1.25 ms | XPBD 8 it., 2.5 ms | XPBD 4 it., 2.5 ms | same, mesh colliders |
|---|---|---|---|---|---|
| 256 | 119,448 | 399,828 (3.3×) | 466,555 (3.9×) | 795,504 (6.7×) | 579,442 |
| 1024 | 224,309 | 903,596 (4.0×) | 1,055,494 (4.7×) | 1,833,701 (8.2×) | 1,564,338 |
| 4096 | 313,780 | 1,269,644 (4.0×) | 1,425,129 (4.5×) | 2,543,061 (8.1×) | 2,372,911 |

At 4096 envs the settings that track MuJoCo C give 4.0–4.5× MuJoCo Warp's physics rate (about
159K–178K physics-limited env-steps/s at 50 Hz control against MuJoCo Warp's 39K); the 4 it. / 2.5 ms
setting is the minimal stable one and lands softer, falling 0.1–0.2 s earlier. Graph replay matches
eager stepping to 4e-7 m. These exclude observation, reward, reset and PPO (23 % of the MuJoCo step);
the full-step comparison and a PPO run on the Newton path are in progress.

G1 drop test from 1.0 m, same protocol as `g1_contact_stiffness.py` (impact / resting penetration):
XPBD 2 it. 2.5 ms 1.09 / 0.07 cm; 4 it. 0.69 / 0.03; 8 it. 0.36 / 0.01; 4 it. 1.25 ms 0.24 / 0.01;
8 it. 1.25 ms 0.16 / 0.00; MuJoCo C default 2.97 / 0.06. Isaac's PhysX recording of the same drop
(§1.7) is the reference.

Still open on the Newton path (integration into the task code is done, §1.5: `engine="newton"` runs
the same observation/reward/termination kernels, 9 invariant tests in `tests/test_g1_newton_engine.py`,
flat terrain only): 3σ stability at the fast settings, the monitor's penetration/energy/overflow
checks (no such fields on Newton), rough terrain (needs a heightfield), `ActuatorPD`
is MetalSim code rather than Newton's (its damping is capped at what one step can remove on very light
links), and two upstream issues to raise (relaxation defaults, biased compliance drive).

## 3. What is disproved, missing, or cannot be tested here

- **Physics engine**: PhysX is not reproduced; MuJoCo's soft-contact Newton solver is used, as the
  plan chose. No row above claims contact-level equality with PhysX.
- **Rendering vs Isaac RTX** (tiers 0–2) and **RTX lidar**: Isaac Sim does not run on Apple Silicon,
  so the plan's "shared USD scene" image comparison is not testable here; MuJoCo's rasterizer and
  `mj_ray` are the references used instead. Tier 2 has no denoiser; tier 1 has no MetalFX pass.
- **Cartpole comparisons** (docs/PHASES.md): MJCF equivalents with the same reward/reset/camera
  spec, not Isaac's USD, and a contact-free scene; they exercise the pipeline, not the physics. They
  are kept as pipeline numbers, not as evidence of physics parity.
- **SO-101 lift** (docs/PHASES.md): a reconstructed scene with no published benchmark; the 3M-step
  run's success rate was 0. It is a pipeline demonstration, not a parity claim.
- **MaterialX, Hydra/usdview, ROS 2 bridge, deformables** (3 MuJoCo Warp flex tests fail on Metal):
  not implemented.
- **Exact terrain heights** (re-implemented generator) and **contact force history**: documented differences.
- **Soft joint limits and soft contacts**: the anomaly monitor measures joint ranges exceeded by up to 0.17 rad and contact penetrations of 5–7 cm in falls under MuJoCo's default limit/contact stiffness (solref 0.02); PhysX keeps both near zero. Stiffer `solref` on limits and contacts is possible at the 2.5 ms step (minimum 2·dt = 5 ms time constant) and is quantified by the drop test of the fidelity protocol (§1.7 once recorded).

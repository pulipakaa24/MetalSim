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
`env.step`). Our task is `orchard.learn.g1_velocity` built from `assets/isaac/G1/g1_minimal.usd`,
the file Isaac Lab loads for the task (44 rigid bodies, 37 revolute joints, three convex colliders),
loaded through `orchard.scene.usd_to_mjcf`.

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
| Rough terrain: `ROUGH_TERRAINS_CFG` 10 rows × 20 cols of 8 m cells, curriculum by row, sub-terrain mix (stairs up/down, boxes, random rough, slopes) | re-implemented from the config on a 0.1 m heightfield (`orchard.learn.terrain`) | `tests/test_terrain.py::test_terrain_layout`; physics on it: `::test_hfield_mesh_contacts_match_mujoco_c` (xfail) | layout pass; **contacts fail** (§1.6) | partial for the layout; **disproved** for the physics on the current MuJoCo Warp branch |
| Height scanner: `RayCaster` 1.6 × 1.0 m grid at 0.1 m (187 rays), yaw-aligned, `body_z − hit_z − 0.5` | Warp kernel interpolating the heightfield triangles on the sim queue (capturable; a heightfield needs no ray tracing; `heightscan.metal` remains for arbitrary meshes) | `tests/test_terrain.py::test_height_scan_matches_mj_ray` (vs MuJoCo `mj_ray` on the same heightfield, 1496 rays) | median 0.0 m, 99th pct 0.0 m, max 0.85 m on cell diagonals where MuJoCo's triangulation differs | confirmed |
| Events not reproduced | startup friction randomization (fixed 0.8/0.6), external force/torque event (zero range in the G1 config) | – | – | partial (documented) |

### 1.3 Learner (rsl_rl PPO, `g1_rsl_rl_ppo_cfg.py`)

| Isaac | ours (`orchard.learn.ppo_warp` with `g1_ppo_config`) | test | verdict |
|---|---|---|---|
| 24 steps/env, 5 epochs, 4 minibatches, lr 1e-3 adaptive (desired KL 0.01, ×/÷1.5), γ 0.99, λ 0.95, clip 0.2, entropy 0.008, grad norm 1.0, ELU MLP 512-256-128 (rough) / 256-128-128 (flat), init std 1.0 | same values; rollout policy evaluated in Warp with shared weights | `tests/test_warp_policy.py::test_parity_with_torch_and_shared_weights` (Warp MLP == torch to 1e-5; log-probs to 1e-4) | confirmed |
| Clipped value loss | not implemented (plain MSE) | – | partial |
| Rollout with no per-step host launches | whole rollout as replays of one captured graph | `tests/test_warp_policy.py::test_rollout_without_torch_launches` (runtime counters: 0 host syncs, 0 host ops) | confirmed |

### 1.4 Throughput on the G1 task (Isaac's protocol, 4096 envs)

| measurement (4096 envs unless stated) | this stack (M4 Max, 18.4 TFLOPS), synchronized | Isaac Lab (RTX 4090, 82.6 TFLOPS), reported | raw ratio | per-TFLOPS ratio |
|---|---|---|---|---|
| G1 flat, physics only (4 substeps of MuJoCo Warp, graph replay) | 67,222 env-steps/s | – | – | – |
| G1 flat, step only (physics + reset/forward + obs + reward, one graph per step) | 45,677 env-steps/s (second run 44,991) | 94,000 (rough; flat not published separately) | 0.49× | 2.2× |
| G1 flat, step + inference (Warp MLP 256-128-128 inside the rollout graph) | 45,954 env-steps/s | 88,000 | 0.52× | 2.3× |
| G1 flat, full PPO loop (24-step rollout + 5 epochs × 4 minibatches on MPS) | 41,340 env-steps/s (training log over 15 iterations: 43,300) | 82,000 | 0.50× | 2.3× |
| G1 flat, 2048 envs: step only / + inference / full loop | 38,903 / 39,142 / 32,418 env-steps/s | – | – | – |
| G1 flat, 1024 envs: step only | 29,119 env-steps/s | – | – | – |
| G1 rough, 2048 envs: step only (+ height scan) | 29,358 env-steps/s, **but see the rough-terrain row below: the physics is defective, so this is a cost figure only** | 94,000 | – | – |
| G1 rough, 4096 envs | fails (`kIOGPUCommandBufferCallbackErrorOutOfMemory` in the heightfield CCD kernels, any queue setting) | 94,000 | – | – |

Per-step cost split at 4096 (flat): physics 60.9 ms, full env step 89.7 ms (the per-step
`reset_data` + `forward` for resets and the obs/reward kernels add 47 %), PPO update 239 ms per
24-step rollout (10 % of the loop).

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

### 1.5 Learning on the G1 task

Flat, 4096 envs, Isaac's PPO config unchanged (`runs/g1_flat_ppo_300b.log`): 300 iterations (29.5M env-steps, 11.6 min at 42.4K env-steps/s including
updates). Mean episode length rose from 41 control steps (0.8 s) at iteration 25 to 47 (it 50), 59 (100), 77 (150),
135 (200), 209 (250) and 343 (it 300, 6.9 s of the 20 s horizon); mean return went from −203 to −233
because the penalty terms (joint deviation, action rate, torques, flat orientation) still outweigh the
tracking terms while the robot learns to stay up, and the −200 termination is still paid by most
episodes. So the identical config learns to survive on this stack within 300 iterations but has not
learned to track velocity commands; Isaac Lab's G1 flat runner is configured for 1,500 iterations.
One transient blow-up was recorded at iteration 296 (a world's acceleration/torque terms produced a
return of −1e26 for one rollout; the run recovered at the next iteration), which points at the same
contact-impulse spikes seen on flat ground (peak touch forces of 6–7 kN under random actions).

The reward is dominated by the −200 termination penalty at the start (the robot falls onto its
torso within ~40 control steps under a random policy, as the physics rows predict). NVIDIA
publishes no G1 learning curve in the benchmark, so this row shows that the identical config
learns on this stack (episode length and tracking reward rise), not a curve match.

An earlier run without the non-finite guard crashed at iteration 168 (a world's state became NaN
inside the rollout and reached the actor); the task now ends and resets such episodes and counts
them (`blown_up_episodes`), which is reported alongside the curve.

### 1.6 Rough terrain: disproved on the current MuJoCo Warp branch

`tests/test_terrain.py::test_hfield_mesh_contacts_match_mujoco_c` (marked `xfail(strict)` so the
defect is tracked): G1 on Isaac's rough-terrain layout under a PD hold, MuJoCo Warp vs MuJoCo C on
the same model. Result: heightfield-mesh contacts return an inverted normal (z = −1.000) and a
5.1 cm penetration on the first step; one of four worlds is launched (pelvis 2.31 m vs 0.74 m in
C). MuJoCo Warp also warns that HFIELD-MESH pairs get at most one contact (no multi-contact CCD
support). Under random actions joint velocities reach 1e13 rad/s and the acceleration/torque reward
terms explode (observed in the rough PPO run, iteration 3: return −8.7e25). Rough-terrain
throughput and learning are therefore **not claimed**; the flat task is the comparison. The height
scanner and terrain generator themselves pass their tests (§1.2).

## 2. Platform capabilities (the workstreams), with the tests behind them

| capability | Isaac | ours | test (assertion) | result | verdict |
|---|---|---|---|---|---|
| GPU physics equals the CPU reference | PhysX GPU == PhysX CPU (NVIDIA's claim) | MuJoCo Warp on Metal == MuJoCo C | `tests/test_physics.py::test_one_step_parity_vs_mujoco_c` (SO-101, Panda, Go1, G1 one step at MuJoCo Warp tolerances), `::test_trajectory_parity_so101_arm_joints` (200 steps, max err < 2e-2, no NEFC overflow, all worlds identical) | pass; SO-101 trajectory 4.5e-7 | confirmed |
| Zero host synchronization per step | tensors stay on device; one sync per rollout | runtime counters as the conformance instrument | `tests/test_physics.py::test_step_does_not_block_host` (0 syncs, 0 host ops, host time < 25 % of GPU time and independent of batch size); `tests/test_render_tier0.py::test_gpu_path_with_batchsim_no_host_sync`; `tests/test_sensors_rt.py::test_gpu_path_ordering_with_sim` | pass | confirmed |
| Zero-copy tensors between physics/render and the learner | PhysX views as torch tensors | Warp Metal buffers aliased as MPS tensors, event ordering across queues | `tests/test_interop.py` (11 tests: alias, lifetime, 2-D/vec dtypes, wait/signal both directions, ping-pong) | pass | confirmed |
| Per-world physics randomization | PhysX per-env mass/friction | per-world model fields + `recompute_constants` | `tests/test_physics.py::test_per_world_model_fields` | pass | confirmed |
| USD scene layer | USD + UsdPhysics/UsdShade/UsdSemantics; URDF/MJCF importers | MJCF/URDF→USD, USD→MjSpec (lossless + generic UsdPhysics) | `tests/test_scene_usd.py` (3 tests: import fields, round trip to MuJoCo, URDF), the G1 rows above (a real Isaac asset) | pass | confirmed for the subset used; MaterialX graphs, Hydra viewing not implemented |
| Renderer tier 0 (raster + PBR + shadows) vs a reference rasterizer | Isaac RTX rasterizer | native Metal raster reading physics buffers | `tests/test_render_tier0.py::test_silhouette_and_depth_and_seg_parity_primitives` (vs `mujoco.Renderer`: IoU > 0.95, depth median < 1 cm, seg IoU > 0.85), `::test_texture_pattern_parity` (corr > 0.97) | IoU 0.994–0.997, depth 0.1 mm, corr 0.996 | confirmed vs MuJoCo; **not testable vs Isaac RTX** here |
| Renderer tier 1 (hybrid RT: soft shadows, AO, reflections) | RTX real-time | fragment-stage Metal ray queries | fidelity benchmark (`orchard.bench.render_fidelity`): PSNR 19.7 / FLIP 0.30 vs MuJoCo; rollout: `tests/test_render_tier2.py` sibling path (`CartpoleRGBEnv(tier=1)`) | measured | partial (no denoiser / MetalFX upscale yet) |
| Renderer tier 2 (path tracer) | RTX path tracer (Isaac "PathTracing" mode) | progressive Metal RT path tracer, NEE, GGX+Lambert, Russian roulette | `tests/test_render_tier2.py::test_lambertian_plane_analytic` (radiance 0.4000 exact), `::test_white_furnace_uniform_sky` (0.498 vs 0.5), `::test_monte_carlo_convergence` (noise ratio 4.05 for 16× spp; ideal 4.0), `::test_direct_light_matches_tier0` (PSNR 43.7–45.1 dB, depth 1 mm, seg IoU 1.0), `::test_gpu_path_from_batchsim_matches_host`, `::test_cartpole_rgb_rollout_tier2` (full env rollout at tier 2) | pass | confirmed radiometrically; not compared to Isaac's path tracer (not testable here); no denoiser |
| Tier 1 / tier 2 full rollouts (throughput) | Isaac Cartpole-RGB: 50K steps/s at 1024 envs on a 4090 (rasterized RTX, reported) | Cartpole-RGB 100×100, 1024 envs, physics + render + reward/reset, one run each, uncontended | `orchard.learn.cartpole_rgb` (`python -m orchard.learn.cartpole_rgb 1024`) | tier 0 47,135; tier 1 42,634; tier 2 (1 spp, 1 bounce) 36,401; tier 2 (1 spp, 2 bounces) 34,762; tier 2 (4 spp, 2 bounces) 19,782 env-steps/s | measured; the tier-2 rollout is a full path-traced observation stream at 70 % of the raster rate (1 spp, noisy) |
| Ray-traced sensors (lidar, depth) | RTX lidar | Metal RT acceleration structures refit from physics | `tests/test_sensors_rt.py::test_lidar_vs_mujoco_ray` (median < 2 mm, 95th < 2 cm, hit pattern within 3 %), `::test_raycast_depth_vs_raster_depth` (99.5 % agree, median < 1 mm) | pass | confirmed vs MuJoCo `mj_ray`; **not testable vs Isaac RTX lidar** |
| IMU / contact / joint sensors as tensors | Isaac sensors | MuJoCo sensors evaluated per step | `tests/test_sensors_state.py::test_imu_and_contact_sensors_as_tensors` (accelerometer 9.81 at rest, ~0 in free fall; touch ≈ m g) | pass | confirmed |
| Height scan on terrain | `RayCaster` | Metal ray queries | `tests/test_terrain.py::test_height_scan_matches_mj_ray` | pass | confirmed |
| Replicator: randomizers, annotators, writers | Omniverse Replicator | GPU randomizers, 2-D/3-D boxes, semantic seg, COCO/KITTI/basic writers | `tests/test_replicator.py::test_randomize_annotate_write` (boxes equal the segmentation's extents, files written) | pass | confirmed for the implemented subset |
| Warp rollout policy | – | MLP policy in Warp with shared torch weights | `tests/test_warp_policy.py` (2 tests) | pass | confirmed |
| RL reproduces a published Isaac Lab result | Cartpole-Direct with rsl_rl config learns | same config on the Warp path reaches 295/300 in 45 iterations | `orchard.learn.ppo_warp` run (docs/PHASES.md) | measured | confirmed (equivalent MJCF cartpole, not Isaac's USD) |

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
- **Terrain curriculum and exact terrain heights**, **contact force history**, **clipped value loss**,
  **event randomizations** listed in §1.2: documented differences.

# Changelog

What landed between 2026-09-22 and 2026-09-25, one line per commit, grouped by area and in
commit order within each area (`git log` for the full messages). Dates are the commit dates (month-day).

## Install and packaging

- 09-25 GPU queue dashboard: `scripts/gpu_dashboard.py` (`--serve`, `--watch`, `--html`, `/json`), `scripts/gpu_dashboard.sh` launcher; queue records command lines, grants and releases with exit codes (`runs/gpu_history.jsonl`)
- 09-25 `99c331f` MIT licence (LICENSE, pyproject metadata), THIRD_PARTY_NOTICES.md with the IP audit and `licenses/`, source headers on the three ported files, README rewritten as the front page (what was built in each fork + branch, heads and commits); `METALSIM_CHANGES.md` pushed to every fork branch
- 09-23 `f3d5859` Rename to MetalSim (package metalsim); Warp fork setup script and patches; heightfield defect localized
- 09-23 `be1e347` Point the Warp setup at the pulipakaa24/warp fork (branch metalsim)
- 09-23 `c2fbe01` Fetch NVIDIA Isaac assets instead of committing them; tron1 rename leftovers
- 09-24 `b00c8af` Newton fork setup: scripts/setup_newton.sh installs pulipakaa24/newton@metalsim into .venv-newtonfork

## Physics (MuJoCo Warp on Metal) and interop

- 09-26 Elliptic-cone premium cut further (fork branch `metalsim-elliptic2` `b0150ac`, pending merge): MuJoCo C's incremental Newton structure (quadratic part kept across iterations, cone term rebuilt per iteration) on Metal, mode 2 = per-entry deltas + cone term into `htot` and the plain register Cholesky; G1 task ellip10 55.4 K → 64.5 K env-steps/s (1.42× → 1.21× pyramidal), Go2 +7 %, humanoid +4 %, SO-101 +4 %; per-iteration constraint-state characterisation on walking and falling states (`scripts/diagnostics/competitors/g1_state_*.py`); rejected variants archived behind flags (`docs/DECISIONS.md`, research note §8)
- 09-25 Elliptic follow-ups: `tau10_impact_hardlimits_ellip{1,10,100}` presets (+ field test), confirming training run `g1_flat_flatcfg_ellip10` queued, PhysX-checkpoint falls characterised over 3 seeds (+7–10 %, within one seed sd; `scripts/diagnostics/competitors/g1_falls*.py`), `tests/test_lift_ctrl_ordering.py` (the SO-101 env was already event-ordered), the two pre-existing fork test failures root-caused (heightfield primitives fix on fork branch `metalsim-hfield-test`; tendon-frictionloss test depends on a MuJoCo nightly)
- 09-25 Elliptic friction cones on Metal: reproduced the handoff (2.4–7× slower than pyramidal), fixed in the fork worktree (`c301880`, branch `metalsim-elliptic`, pending merge: per-world cone-list Hessian term, full-width sparse assembly groups, every previous form behind `MJW_JTCJ_MODE` / `MJW_JTDAJ_ELLIPTIC_LANES`), audited every other non-CUDA fallback, re-ran the G1 fidelity protocol and an SO-101 creep protocol with elliptic cones; upstream PR draft `scripts/diagnostics/mjwarp_upstream/DRAFT_elliptic_launch.md`; `docs/research/elliptic_cones_2026-09-25.md`; scripts `scripts/diagnostics/competitors/elliptic_*`, `so101_creep.py`
- 09-25 Shared fork checkouts fast-forwarded to the documented heads (mujoco_warp 1791414 -> 07a51a6, warp 9050cb5 -> 4127c48) after the throughput A/B showed them identical for the G1 (flex changes gated on nflex > 0); the local branches had never been advanced
- 09-22 `b773ae7` orchard: project skeleton and SO-101 lift scene
- 09-22 `7d64c48` interop: zero-copy MPS tensors over Warp Metal arrays and cross-queue event ordering
- 09-22 `688a13c` physics: BatchSim on MuJoCo Warp with graph replay and MPS-aliased state; render: native Metal tier-0 renderer
- 09-22 `3f6c5e7` physics: size constraint buffers (njmax=512) and expose overflow flags; tools: race finder; scene: MJCF->USD importer
- 09-22 `1a2741a` physics: per-world model fields (physics DR) with recompute_constants; lift env randomizes box mass/friction per world; URDF->USD test; IMU/contact sensor tensors test
- 09-22 `1359a52` G1 velocity task on Isaac's g1_minimal.usd with test-cited parity matrix; tier 2 path tracer; Metal backend memory fix
- 09-23 `4e6d663` MuJoCo Warp heightfield plane-contact patch; heightfield contact test now passes (was xfail)
- 09-24 `927d792` Uncontended benchmark pass 2026-09-24; rough terrain at 4096 envs (nconmax 32); docs updated from the clean pass
- 09-24 `eaa4ce3` MuJoCo Warp fork: plane_convex now reproduces MuJoCo C's contact set (fork metalsim 284dcd1)
- 09-24 `6d2accb` Contact/limit tuning module and tuned parity replays against Isaac's PhysX recordings
- 09-24 `11a07c3` plane_convex patch: add the MJW_PLANE_CONVEX=legacy switch (fork metalsim a8e6485); regression test skips under it
- 09-25 Throughput "regression" settled (`docs/research/throughput_regression_2026-09-25.md`): the flex merge in the MuJoCo Warp fork costs the G1 nothing (interleaved A/B ±0.4 %); the 67.6 K → 54.2 K env step is the task's PhysX-parity contact preset (solver only); README / PARITY §1.4 carry both settings; `g1_tp_variants.py` / `g1_step_profile.py` take `contact_cfg` / `solver_cfg`; `g1_tp_generality.py` prints which fast paths a scene takes (Go2, SO-101, Panda at 4096: elliptic cones fall off the fused Hessian)

## Rendering

- 09-22 `0656f1f` render: model light list, headlight/ambient model, tiled shadow maps, radiance calibration; fidelity benchmark (PSNR/FLIP)
- 09-22 `62857e0` render: tier 1 hybrid ray tracing (RT soft shadows, AO, mirror reflections from the fragment stage), skybox colour, Isaac-style lidar spec; bench suite
- 09-24 `c5aca1f` Parity scene: hide collision geometry in renders (Isaac draws visuals only; the foot collider plates were visible); post-queue video re-render
- 09-26 `302bab5` Tier 2: HDR equirectangular environment map (textured USD DomeLight) importance-sampled with MIS (`set_environment`, `env_sampling_table`, `metalsim/render/hdr.py` RGBE reader), oriented as Kit/RTX renders a dome on a z-up stage; USD import records DomeLights (`usd_dome` custom text, `set_environment_from_model`); `Randomizer.environment` per-episode map by key; `set_fovy`, `set_materials`, opt-in `firefly_clamp` (biased); parity presets bit-for-bit unchanged without a map; Poly Haven CC0 maps fetched (not committed); gallery `g1_hdri_tier2.png`; handoff `docs/HANDOFF_tier2_hdri_envmap.md` ported

## Sensors

- 09-22 `fbc6c58` sensors: Metal hardware ray tracing (per-mesh primitive AS, per-frame instance AS refit from physics), lidar and ray-cast depth kernels, tests vs mj_ray and raster depth
- 09-24 `e6d458d` Contact sensor (Isaac Lab ContactSensor equivalent) on MuJoCo Warp as in-graph Warp kernels
- 09-24 `76c2708` Lidar extras (beam divergence, multi-return, reflectance table) and radar-lite on Metal RT
- 09-24 `82fc341` Sensor benches: record GPU queue status at start; 7 interleaved A/B repeats for the contact sensor; measured logs

## Scene layer (USD / MJCF), terrain, replicator

- 09-22 `3c921be` scene: USD -> MuJoCo loader (lossless and generic UsdPhysics paths) with round-trip tests; learn: contiguous NCHW rollout buffer (2.3x faster updates), train_lift script
- 09-22 `46768d0` replicator: randomizers, annotators (2D/3D boxes, semantic seg, camera params), COCO/KITTI/basic writers with test; learn: Warp-rollout PPO with rsl_rl KL schedule (learning issue under investigation)
- 09-24 `293d38a` USD loader: instance proxies and MDL OmniPBR materials (G1 visual meshes render); kinematics-only reset (+20% step rate); clipped value loss; terrain curriculum; cartpole tier-2 video; G1 still
- 09-24 `4b23a3a` Rough terrain: port Isaac Lab v2.3.2's TerrainGenerator exactly (ROUGH_TERRAINS_CFG)

## G1 velocity task and PPO learning

- 09-24 `8954ad7` Blow-up guard by magnitude, non-finite rows dropped from the PPO update, per-iteration callback with periodic checkpoints; hero cartpole video
- 09-24 `40af1c8` G1 1,500-iteration run recorded (learns to stand for 12 s by it 750, then degrades); G1 tier-2 policy video; camera-cartpole PPO brought to Isaac's skrl config
- 09-24 `a6f6eca` G1 blow-up root cause: explicit actuator stiffness at 5 ms with Isaac's kp 200 (0/8 blow-ups at 2.5 ms); physics_dt option; camera PPO gets Isaac's image centering and skrl value scaler
- 09-24 `18e0c88` G1 reward: termination penalty scaled by dt as Isaac's RewardManager does (was -200 unscaled, which made falling optimal)
- 09-24 `a8b7706` Task-agnostic anomaly monitor (learning signals + physics invariants) hooked into training; G1 preflight; monitor findings recorded
- 09-24 `95abfd9` Corrected G1 run recorded (stays up 11-15 s by it 1000, return still negative); monitor trend window 100 iterations
- 09-24 `8818c4f` G1 learning: per-term reward comparison against Isaac's rsl_rl run (PARITY 1.5); reward-term diagnostic
- 09-24 `af3cd6b` G1 feet_slide: Isaac's quantity (foot body world linear velocity) on both engines
- 09-24 `f0f2115` Learner differential: rsl_rl 3.1.2 on MetalSim G1 flat; PPOWarp rollout fixes
- 09-24 `72a19c6` rsl_rl 3.1.2 on MetalSim G1 flat, 1500 iterations (Isaac's G1FlatPPORunnerCfg, seed 0, 4096 envs): log, per-term JSONL, curves script
- 09-24 `5f208f0` G1 task: flat terrain uses Isaac's G1FlatEnvCfg reward set; all 13 reward terms exposed and asserted against Isaac's formulas
- 09-24 `5011a82` Fixed PPO demonstrated on G1 flat: +32.8 return / 995 length at iteration 1000, tracking rsl_rl on the same physics (PARITY 1.5, GAPS, STATUS)
- 09-24 `00ecdd0` G1 task: Isaac ContactSensor for the flat set's contact terms; Isaac's rough-terrain seed and env-column assignment
- 09-24 `f5bdd60` G1 flat: ContactSensor cost at 4096 envs, 5 interleaved A/B repeats: +0.62 ms per control step (+0.4 %), 29,479 vs 29,611 env-steps/s
- 09-24 `930bd28` rsl_rl 3.1.2 on the ported G1FlatEnvCfg task (ContactSensor contact terms), 400 iterations, 4096 envs, seed 0
- 09-25 `7952b89` Like-for-like training result: MuJoCo Warp + fixed PPO on Isaac's flat config reaches +28.4 at iteration 1000 vs Isaac's +27.3 (PARITY 1.5, STATUS, GAPS)
- 09-25 `afe3f27` G1 training CLI: --seed
- 09-25 `f0e1b6b` Per-term reward breakdown at it 300/1000 of the MuJoCo Warp flat-cfg fixed-PPO checkpoints (like-for-like with the Newton rows)

## Other learning tasks (lift, cartpole, camera cartpole, lidar navigation)

- 09-22 `836c518` learn: GPU SO-101 lift task and MPS PPO; render: levels of detail, GPU-writable DR params; docs and gallery
- 09-22 `7d2ee8b` learn: Warp rollout policy with graph-replayed rollouts (zero torch launches per step); Isaac Cartpole-RGB task; physics launch_step for outer captures
- 09-22 `72bad94` learn: cartpole learns with Isaac Lab's rsl_rl config on the Warp rollout path (collision-free cartpole MJCF)
- 09-24 `b8ac7a7` Camera-based RL row: Isaac's camera cartpole agent config learns from tier-0 pixels
- 09-24 `48aa2bb` Lidar-based RL task (planar 64-beam Metal lidar, goal navigation among random obstacles) with scan-vs-mj_ray test
- 09-24 `617cdde` Camera-RL update: research on MPS conv backward and alternatives, update profiler and measurements (torch eager/fp16/bf16/compile, MLX)
- 09-24 `2796675` Camera-RL update on MPS: fast_update path (compiled loss and grad clip, fused Adam, per-rollout image means) and Metal kernels for conv1 weight grad and conv2 input grad; update 7.2 s -> 3.9 s
- 09-25 `ccbfb9c` Camera-cartpole training with the fast update path: 11.4K env-steps/s incl. training (was 7.5K), learning curve matches the reference run
- 09-25 `c4c1ae8` Camera-RL update: conv1 forward Metal kernel, minibatch gather fused into the compiled preprocessing, conv2/conv3 weight-grad kernels (measured, off by default); update 3.68 -> 3.20 s
- 09-25 `303ee69` Camera-RL round two: gather fused in graph, conv1 forward kernel, conv2/conv3 wgrad measured (off), MLX whole-update experiment (no gain in fp32); training 12.7K env-steps/s incl. training, curve unchanged
- 09-25 `8975b42` Camera-RL: 12.7K env-steps/s incl. training after the gather fusion and conv1 forward kernel; MLX not faster in fp32. GPU queue: release checks the job pid

## Parity against Isaac Sim / Isaac Lab (benchmarks, fidelity protocol, side-by-side, transfer)

- 09-22 `50efb27` bench: uncontended suite and Isaac comparisons recorded
- 09-24 `66fdb3e` Fidelity protocol: Isaac-side recorder and policy player, MetalSim replay, metrics (PSNR/SSIM/LPIPS/FLIP, depth, joint RMSE, divergence time, contact stats)
- 09-24 `22aa664` Isaac Lab measured on an L4: Cartpole-Direct 400.7K, G1 rough 39.1K env-steps/s (reference rows)
- 09-24 `016d64d` Side-by-side video composer (Isaac RTX vs MetalSim tier 2, same policy checkpoint and state)
- 09-24 `6368d87` Policy export for Isaac playback; VM stage 5 (play checkpoints from both trainings in Isaac)
- 09-24 `b7df6d8` Isaac-side recorder/player: no debug markers, plain grey ground, 20 m env spacing (fair frames)
- 09-24 `0a711c9` Fidelity protocol results vs Isaac Sim 5.1 (PARITY 1.7): physics rows, whole-frame and robot-only render metrics, gallery composites; lidar RL run; Isaac-side grey ground + stage 6 re-record
- 09-24 `3c4c552` Docs: Isaac's full rsl_rl run and cross-simulator playback results; L4 flat G1 benchmark; stage-video runner
- 09-24 `55502c0` Fidelity: grey-ground re-recording replaces the whole-frame rendering rows (tier 2 vs RTX 18.0 dB / 0.876 SSIM); gallery composites refreshed; VM stopped
- 09-24 `b9c55a0` Fidelity replay: full-extent ground plane; re-render runner
- 09-24 `a0f5984` Gallery: side-by-side training-stage videos (Isaac Sim | MetalSim) for Isaac's and MetalSim's checkpoints; cross-simulator transfer table
- 09-24 `fcc4d97` MuJoCo Warp transfer of Isaac's checkpoints before / after the plane_convex contact-set fix (mujoco_warp f2716b4 vs 284dcd1)
- 09-24 `eea5e0f` Transfer of Isaac checkpoints (500/1000/1499, 8 s) on MuJoCo Warp: old vs fixed plane_convex collider and each contact tuning; tuning throughput logs; --contact_tuning in g1_reward_terms and mjwarp:<tuning> in newton_transfer
- 09-25 `ee7131d` Gallery: round-two side-by-side videos (fixed-PPO policies in Isaac Sim and MetalSim); transfer table both directions

## Newton XPBD engine (experimental; archived 2026-09-25)

- 09-24 `c57d519` Diagnostics: MuJoCo contact stiffness sweep on the G1 drop test; Newton XPBD on Metal with Isaac's G1 USD (feasibility)
- 09-24 `82072b4` Newton XPBD status: drives not engaging yet; throughput figure qualified
- 09-24 `947f51e` Newton XPBD drives: root causes and a faithful PD actuator for the G1
- 09-24 `97f036b` Diagnostics: Isaac actuator model under Newton XPBD (effort limits, armature) on the G1 hold and a 1 m drop
- 09-24 `170caf6` Diagnostics: G1 drop-test penetration under Newton XPBD on Metal (MuJoCo C protocol)
- 09-24 `66db331` Diagnostics: G1 physics throughput, Newton XPBD with working drives vs MuJoCo Warp (graph replay)
- 09-24 `806b96f` Diagnostics: Newton mesh/convex narrow phase (GJK/MPR) on Metal vs CPU
- 09-24 `f769f40` Newton backend: optional native convex-mesh colliders (GJK/MPR on Metal); docstring with the measured recipe
- 09-24 `40acf8f` Newton backend: module docstring with the measured drive recipe
- 09-24 `6000ac1` Newton XPBD results in the parity and gap docs; Warp fixed-size-array patch; fidelity depth metric masks sky and separates the robot silhouette; Isaac-side player accepts MetalSim exports
- 09-24 `ac55b0e` G1 task: engine switch (mjwarp default | newton) with a BatchSim-compatible NewtonSim
- 09-24 `ee6752d` Tests: G1 task invariants on both engines; Newton engine invariants
- 09-24 `ef74f40` Newton engine: gravity-implicit actuator on collider-free subtrees, MuJoCo cvel semantics, 2.5 ms qacc; preflight/bench engine flags
- 09-24 `ded4803` G1 flat PPO on Newton XPBD: 1500 iterations, same config and seed as the MuJoCo Warp reference
- 09-24 `9e33cfe` Newton upstream issue drafts: XPBD joint relaxation defaults, iteration-dependent drive stiffness
- 09-24 `9842062` Diagnostics: Newton XPBD joint-constraint drift on the G1 vs solver setting
- 09-24 `f69b0d4` Newton engine: torso contact history over the last 15 ms of substeps (Isaac's illegal_contact window)
- 09-24 `c65aa09` Newton engine: rough terrain (heightfield collider, height-scan body pose)
- 09-24 `a969f72` g1_engine_bench: --rough (either engine)
- 09-24 `72bf1d4` Newton engine: solver/actuator knobs for the 3-sigma stability sweep
- 09-24 `137045e` Diagnostics: transfer of Isaac's checkpoints to the Newton engine (side_by_side protocol); Newton column in g1_learning_curves
- 09-24 `839b334` Newton engine: keep revolute limits inside +-(pi - 0.15) (XPBD angle wrap at pi); drift-remedy evaluation; Featherstone option
- 09-24 `67fb536` Newton engine: optional recentred revolute zeros (moves XPBD's +-pi angle wrap away from the limits)
- 09-24 `9c9d15d` Newton upstream issue draft: XPBD revolute angle wraps at +-pi (limit ranges or overshoot past pi explode)
- 09-24 `22aee1f` Newton drift remedies evaluated on CPU (projection, Featherstone, recentring, 0.625 ms); throughput script
- 09-24 `d21fc28` newton_remedy_throughput: drop Featherstone from the 4096-env timing (1 env-step/s at 16 envs, measured)
- 09-24 `c2eb2d0` G1 engine benchmark, final Newton recipe: cost split at 4096 envs (power source at start and end in the log)
- 09-24 `1c96e36` Mark the 21:19 final-engine benchmark as contended (not a result)
- 09-24 `c8ce7ff` G1 engine benchmark, final Newton recipe (clean: MuJoCo Warp control at start and end within 5 % of 27,749)
- 09-24 `6e2314a` Isaac fidelity protocol on the Newton engine (A_hold, B_random, C_drop vs Isaac's parity_out2/rt recording)
- 09-24 `929a278` Newton fork item 1 (hinge angle wrap, fork f844a4e6): G1 check harness and measurements
- 09-24 `4594676` Newton engine: limit_margin pass-through (--newton_limit_margin none, sweep ':nolim'); install stamp in sweep logs
- 09-24 `a244580` G1 flat PPO (fixed learner) on Newton XPBD, 1000 iterations, 4 it 1.25 ms, seed 0
- 09-24 `1d4324f` g1_reward_terms: --engine newton (--newton_it, --newton_dt, --newton_limit_margin)
- 09-24 `921f944` Transfer of Isaac's PhysX checkpoints to the Newton engine (side_by_side protocol, 4 envs, 400 steps)
- 09-24 `65af37e` Newton fork items 2, 3, 5, 6 (fork fda6658a, 9f626ce2): diagnostics and measurements
- 09-24 `7f4ba2e` NewtonSim: switches for the Newton fork's solver-level drive (drive='solver', joint_coloring, relaxation_angular)
- 09-24 `72fe5ad` Newton transfer over-travel diagnostics: stance gap, fork drive, subtree joint inertia (opt-in)
- 09-24 `3a9350f` Push-off diagnostic (Isaac ckpt 1000): forward ground-reaction impulses per touch-down, MuJoCo Warp vs Newton
- 09-24 `7010e03` newton_transfer: rlin=/rang= relaxation tokens
- 09-24 `877b2e5` G1 flat fixed-PPO on Newton XPBD 4 it 0.625 ms, 1000 iterations, seed 0 (pinned install)
- 09-24 `193a7ca` Per-term reward breakdown at it 300/1000: MuJoCo Warp vs Newton fixed-PPO policies, each on its own engine
- 09-24 `1e3b6f1` Newton 3-sigma stability sweep: blow-ups of 1024 vs physics throughput
- 09-24 `bb792c2` Newton fork: pd drive damping fix (9b0901d3) and throughput refactor (242eeda7): diagnostics and measurements
- 09-24 `8b8348f` NewtonSim drive='solver': probe joint_drive_mode, read the fork's public solver.joint_drive_force (total drive torque)
- 09-24 `0fb77ef` 3-sigma winner fails transfer: implicit leg stiffness makes Isaac's policies fall (functional check)
- 09-25 `65afe8e` Transfer of Isaac's checkpoints on the Newton fork (fixed solver drive vs ActuatorPD, 1.25 / 0.625 ms, relaxation 0.5/0.4, no clamp)
- 09-25 `d8eac84` Newton fork final (90e23324): GPU throughput and 3-sigma
- 09-25 `d02c172` G1 flat fixed-PPO on the Newton fork with the fixed solver PD drive, 4 it 0.625 ms, 1000 it, seed 0 (throughput provisional: fork build ~30 % slower on GPU)
- 09-25 `da43f77` Per-term reward breakdown at it 300/1000, Newton fork solver-drive (0.625 ms) fixed-PPO policy
- 09-25 `1e0ca7a` Newton fork: 3-sigma check and throughput with the fixed solver PD drive (vs ActuatorPD), GPU, 1024 / 4096 envs
- 09-25 `d87b721` Swing-leg diagnostic (Isaac ckpt 1000): swing duration, clearance, touch-down velocity, step length, hip/knee trajectories, MuJoCo Warp vs Newton 0.625 ms
- 09-25 `2cbe45a` Newton track archived: remaining logs (job-chain status, install watch, partial held benchmark, PhysX compare outputs)
- 09-25 `611e80f` Newton upstream: issues #4313-#4315 and PRs #4316-#4318 filed (bodies, reproducers, regression logs)
- 09-25 `c22b6ac` Newton upstream: three issues and three pull requests filed (FILED.md); gap table

## Tooling (GPU queue)

- 09-24 `7603b80` GPU arbitration: priority queue lock (timing > render > train) with atomic acquisition, stale-holder cleanup and a run wrapper
- 09-24 `c5e2633` gpu_run.sh: record the job's own pid in the lock and kill the job if the wrapper dies (a cut-off wrapper caused a double grant)
- 09-25 `7cd03a7` GPU queue: low-priority class

## Documentation (status, parity matrix, gap ledger, gallery)

- 09-22 `0460afb` docs: status against the plan, acceptance tests, judgement calls
- 09-22 `fa2df43` gallery: tier 1, lidar, learning curves, Isaac comparison
- 09-24 `aaabb98` Contact-model section: MuJoCo stiffness limits and Newton XPBD on Metal (throughput table)
- 09-24 `91f402c` Gap ledger: closed, open, and Newton-track issues
- 09-24 `fb091f9` Gap ledger: Newton-subsumed vs engine-independent items; camera-RL MPS update profile
- 09-24 `0c4f405` Docs: status and gap table refreshed with the fidelity, Newton, lidar and learning-gap measurements; Isaac L4 camera rollout rate
- 09-24 `793d447` Docs: Newton engine switch results (throughput at the training setting, 1500-iteration curve on both engines, preflight), feet-slide term defect, gap table
- 09-24 `1aba0ef` Docs: feet-slide term fix and Newton constraint-drift finding
- 09-24 `3b0910d` Docs: learner differential verdict (rsl_rl on MetalSim learns like Isaac; three PPO rollout bugs fixed), flat-vs-rough reward config mismatch
- 09-24 `8a6399b` Research: PyTorch-on-Metal interop survey (2026-09-22) saved under docs/research
- 09-24 `f2e1a41` Gap table: Newton constraint drift quantified, rough terrain on Newton closed, upstream issue drafts
- 09-24 `597181d` Gap table: XPBD angle-wrap defect and workaround, drift remedies evaluated
- 09-24 `eddfa2d` Gap table: Newton learning parity pending the fixed learner, 3σ cause, Newton fidelity vs PhysX unmeasured, Featherstone verdict
- 09-24 `f3aa07f` Docs: clean Newton benchmark with control row; PhysX fidelity protocol on Newton (PARITY 1.4, 1.7); rough-terrain Newton cost as a new gap
- 09-24 `c8602af` Docs: exact Isaac terrain port (grid-exact, wall ramps remain) in the parity matrix and gap table
- 09-24 `cab5afb` Docs: contact sensor, lidar extras and radar-lite rows; MuJoCo Warp plane_convex contact-set defect as a new gap
- 09-24 `5fe8462` Gap table: camera-RL update profile, chosen route, MLX as a compounding next step
- 09-24 `517e613` Gap table: Newton fork (pulipakaa24/newton, branch metalsim) fixes the angle wrap at source; relaxation fix in progress
- 09-24 `5482890` Docs: fixed PPO on Newton learns tracking far slower (+7.8 vs +32.6 at it 1000); MuJoCo Warp stays the default
- 09-24 `aac922d` Gap ledger: engine position after the measurements (MuJoCo Warp is the parity engine; XPBD is a throughput option)
- 09-24 `6728a7c` Gap table: Newton fork fixes relaxation, drives, contact forces; rest-height explained; coloring as the drift remedy pending throughput
- 09-24 `7a34245` Docs: Isaac checkpoint transfer on Newton over-travels 20-48 % (drift changes the gait); MuJoCo Warp within 5 %
- 09-24 `c212a59` Docs: Isaac flat config ported; rsl_rl on the ported task within 3.2 return of Isaac at it 399; remaining term gaps are contact quantities
- 09-24 `7a3c08b` Gap table: Newton over-travel is a stride-length effect independent of the drift (unexplained); fork drive light-joint fault
- 09-24 `7aa7084` Gap table: fork drive light-joint fix measured; halving the substep beats joint colouring; fork throughput regression under investigation
- 09-25 `564c8d3` Gap table: Newton fork final state (six solver defects fixed at source, costs and remaining items); transfer with the in-solver drive
- 09-25 `ec95eab` Docs: camera-RL update 1.86x faster with Metal gradient kernels; 11.4K env-steps/s incl. training
- 09-25 `8ee39fe` Docs: Newton learning at all settings vs MuJoCo Warp and Isaac (flat weights), 3σ closed with the fork drive, push-off diagnostic, collider A/B on transfer
- 09-25 `4bcf8e0` Docs: Newton XPBD archived as an experimental option; MuJoCo Warp is the parity engine
- 09-25 `3652e1b` Docs: like-for-like per-term rows and the swing-leg localisation of Newton's long stride; Newton validation closed

# Gap ledger (2026-09-24, evening)

Status of every gap against Isaac Sim / Isaac Lab found so far: closed, still open, and newly opened by
the Newton-engine evaluation. Evidence for each row is in `PARITY.md`.

## Closed

| gap | how it was closed | evidence |
|---|---|---|
| Heightfield contacts (MuJoCo Warp returned inverted normals, launched worlds) | per-triangle plane-contact patch in the MetalSim MuJoCo Warp fork | `test_hfield_mesh_contacts_match_mujoco_c` passes; rough runs at 4096 envs |
| Drive blow-ups (Isaac's kp 200 explicit at 5 ms) | 2.5 ms physics step (`physics_dt`) | 0/8 blow-ups in C at 2.5 ms; 1 in 1,500 iterations on Metal vs 190 |
| Termination penalty 50× too large (reward port bug) | scaled by dt like every Isaac term | pre-flight checks −4 per fall; G1 now stands 11–15 s by it 1000 |
| 4096-env rough out-of-memory | 32 contact slots per world (kernel launched per slot) | rough 55.4K env-steps/s at 4096 |
| Eager-loop GPU memory exhaustion | frees released per completed command buffer (Warp fork) | 2048/4096 runs pass |
| USD loader defects (gravity 0, COM −inf, qpos0, instanced visual meshes, MDL materials) | loader fixes | joint frames 5.6e-7 m, masses exact, G1 renders with its materials |
| Task fidelity: terrain curriculum, clipped value loss, event randomizations | implemented / verified equivalent | rows in PARITY §1.2–1.3 |
| No measured Isaac reference on rentable hardware | Isaac Sim 5.1 + Isaac Lab 2.3.2 on an L4 | Cartpole-Direct 400.7K, G1 rough 39.1K env-steps/s (M4 Max G1 rough: 55.4K) |
| Install path | public forks of Warp and MuJoCo Warp with setup scripts; **fresh-clone install verified 2026-09-25** (ee0af6b, dcaa79a): README followed literally on a new venv → 113 passed / 14 skipped (optional packages) in 2.5 min; full patch series for both forks under `patches/`, optional extras `newton`, `parity`, `rslrl`; `docs/GUIDE.md` (install, first runs, bring your own robot, pre-flight and monitor flags, fidelity protocol, GPU queue, throughput-vs-fidelity knobs); `CHANGELOG.md` | remaining: forks not upstreamed (Warp, MuJoCo Warp); first-run kernel compile and PyTorch download times not measured cold |
| Silent RL failures | task-agnostic anomaly monitor + pre-flight | first probe flagged joint-limit and penetration issues on its own |

## Closed on 2026-09-25 (detail rows kept below under "Ledger of 2026-09-25 rows")

| gap | how it was closed | evidence |
|---|---|---|
| Learning parity on G1 (flat) | three PPO rollout bugs fixed (repeated noise, phantom boundary action, time-outs as falls); Isaac's flat config ported exactly | +28.4 / 26.3 / 26.0 return at iteration 1000 over three seeds vs Isaac's +27.3; full episodes at the same iteration; PARITY §1.5 |
| Learning parity on G1 (rough) | exact rough config, Isaac's terrain generator, seed and column assignment; height-scan ray order fixed | ours +16.4 / level 6.1 vs Isaac's +14.1 / 5.9 at iteration 1500 (one seed each; two more seeds running); PARITY §1.5 |
| Cross-simulator transfer | – | Isaac's policies walk within 5 % of their PhysX distance here (3 % with the tuned contacts); ours walk in PhysX once they track; PARITY §1.5 |
| Contact "impact peak" gap (3–5×) | shown to be a reporting artefact (Isaac's sensor reports the last 5 ms of each 20 ms step) | impulses per event match Isaac within 1–4 % by momentum change; PARITY §1.7 correction |
| Soft joint limits | hard-limit preset | limit excursion 0.030 → 0.001 rad at no cost |
| Foot contact set (`plane_convex`) | fork fix, C-exact set, switchable | landing contact sets 61 → 0 of 691 differ from MuJoCo C |
| Feet-slide reward term | per-foot velocity kernel on both engines | vs `mj_objectVelocity` to 2e-3 m/s |
| Contact sensing | Isaac ContactSensor semantics as Warp kernels | vs MuJoCo C to 1e-4 N; +0.3 % step cost |
| Sensor extras (divergence, multi-return, intensity, radar-lite) | `lidar_ext` kernel, `radar.py`, default scan unchanged | tests per feature; costs per option |
| Exact terrain heights | verbatim port of Isaac Lab v2.3.2's generator | grid-exact vs Isaac's own code; 99.99 % within 1e-5 m of its ray cast |
| Vertical-wall terrain (collision) | exact boxes windowed per robot + exact scan | Isaac's checkpoints' wall-cell falls 47 → 14 of 96 (final checkpoint 0 / 32); −6 % throughput |
| Rendering vs RTX | MDL-defined BRDFs, USD light units, sun disk, ACES + sRGB, OIDN on Metal | robot vs RTX path tracer 13.9 → 28.8 dB (RTX's own modes 25.9 dB apart); whole frame 45.6 dB |
| Throughput | 43-dof Cholesky fast path, sparse L'DL, fused Hessian update (forks) | G1 full PPO loop 27.9 → 56.6 K env-steps/s; above Isaac 2.3.2 + PhysX on the L4 (45.9 K) |
| Camera-RL throughput | compiled update, Metal gradient kernels, fused gather | 7.5 → 12.7 K env-steps/s incl. training; the rest of the 2.5× to a 4090 is hardware |
| Camera-RL physical render default | verified over 8 M steps (a NaN in a first attempt was an out-of-memory from a run that executed without the GPU lock, not the renderer); non-finite samples discarded in the path tracer, 1024-frame finiteness test | mean return over the last 20 iterations 85.3 vs 86.3 for tier 0; 8,788 env-steps/s incl. training |
| Replicator | annotators, BasicWriter layout, event terms | 24 exact-match tests |
| MaterialX | load-time flattening of four surface types | zero render cost; unmapped lobes listed |
| Deformables (PhysX 5.1) | MuJoCo Warp flex 237/237 on Metal with five contact fixes; implicit damping; XPBD cloth/rope | cloth 2 mm / 3.3 cm, rope 1.077 vs 1.026 s, cube bounce 0.140 vs 0.135 m, penetration 2.6 vs 1.6 mm (fitted damping, stated) |
| Install path | fresh-clone install verified, full patch series, user guide, changelog | 113 passed / 14 skipped from a fresh clone |
| Newton XPBD (archived as experimental) | six solver defects fixed in the fork; three upstream issues and PRs filed | see the Newton section |

## Open (2026-09-25 evening)

| gap | status | owner / next |
|---|---|---|
| Contact preset | **picked**: `tau10_impact_hardlimits` (PARITY §1.7): transfer of Isaac's policies within noise (3.3 cm), penetration 2.97 → 1.64 cm, limits 0.030 → 0.001 rad, slide −0.044 → −0.032, chatter near default, 1.02× cost; feet_slide convention fixed to Isaac's COM velocity; impulse / 20 ms-force columns in `compare.py` | being made the G1 default; a flat training run on the preset (queued) confirms learning is unharmed |
| Feet air time vs Isaac | **verified as a policy difference** (PARITY §1.7): Isaac's own checkpoint gets 0.037 air time here with impact-only stiffening (its log 0.045, recorded under stochastic training randomization), and our iteration-1000 policy reaches 0.048; the tuning table's 0.016–0.022 was an iteration-400 policy | one convention fix: feet_slide to the foot's centre-of-mass velocity like Isaac (5–15 % effect); rank contact presets on several policies incl. Isaac's checkpoint |
| Isaac Lab 3.0-EA reference (both backends) | **landed** (PARITY §1.8): environment on the VM, Isaac's exact MuJoCo-Warp settings captured, runtime benchmarks and flat training on both backends, fidelity protocols on both; rough Newton training finishing (fetch blocked on a gcloud credential refresh) | port 3.0's extra task events (push, mass randomization, reset velocities) for a like-for-like 3.0 training comparison; adopt Isaac's MuJoCo-Warp solver settings as a preset and compare against our hard-limit preset (Isaac's limits are very soft) |
| Deformables vs Isaac Lab 3.0 | **resolved** (PARITY §2): flex + implicit damping matches PhysX 6's FEM soft cube with PhysX's own parameters; PhysX's own mesh sweep confirms locking and converges to the XPBD `physical` rope; cloth XPBD within 7 mm top height (flex membrane 11 mm, closer on settle time, 0.25 ms step, kept). The flex cloth that fell through the box was MuJoCo's 50-contacts-per-body-flex-pair cap (`mjMAXCONPAIR`; MuJoCo C does the same): fork option `FLEX_MAXCONPAIR` (64a1ea5; default 50 for parity, 400 for cloths above ~20×20), filed as UPSTREAM item 7 | remaining: full Metal VBD replay + throughput (queued, low), Newton cable pinning (NaN), physx_cloth self-collision Newton VBD on Metal replays 3.0's Newton recording within 1.2 mm over 5 s for cloth, cubes and rods up to 4 cells across (8-cell rod: 1.9 mm at 2 s, float32 divergence after); Metal VBD throughput 0.8–12 K env-steps/s (PARITY §2). |
| Newton 1.5.2 (Isaac Lab 3.0's pin) needs mujoco-warp 3.11 | our fork is 3.14 + fixes; the 3.0 Newton reference runs stock 3.11 without them | port the fixes to 3.11 or move the reference to the version that takes 3.14 |
| Hardware validation | none; the endpoint that outranks both simulators | user's Tron1 run; then a hardware protocol (drop, step response, walking distance) as the top-ranked criterion |
| Breadth | one asset (G1) validated end to end | Tron1 next |
| Multi-seed rough | seeds 1 and 2 queued | – |
| Fast-factorization defaults | **closed** (109984b): `metal_register_cholesky_max=48`, `m_dense_max=32` are the global defaults; verified on seven scenes (physics at noise level, throughput ±1 %); 196 tests pass | – |
| Camera-RL residual | 12.7 K vs 32 K (hardware); untried: Apple's Metal Performance Primitives conv op | low |
| Robot appearance residual | exposure is a fitted constant 9 % off the documented formula; RTX real-time shortcuts and NVIDIA denoisers not reproducible | low |
| Renders of the rough task model | **fixed and worse than thought** (56b11c2, d2b9bf4): heightfields were never drawn at all and the box slots were drawn as 2 mm cubes; the renderer now draws MuJoCo's exact heightfield surface and each env's slot boxes (depth along a stair riser matches `mj_ray` to 0.01 mm; gallery `g1_rough_boxes_tier2.png`) | **new cost gap**: tier 0 rasterizes the full 4.8 M-triangle heightfield per env (2.9 s per 1024-env frame; tier 2 188 ms); needs GPU tile culling before rough-terrain camera RL; the true boxes show only inside the physics window |
| Intermittent `test_ppo_warp_rollout` noise test under GPU load | unverified (the contact-sensor one was a real race, fixed) | reproduce loaded vs idle |
| Upstreaming the Warp and MuJoCo Warp forks | not proposed; flex commits and patches 0008–0016 carry a co-author trailer the MuJoCo Warp CLA rejects (rewrite on a fresh branch before filing); `UPSTREAM.md` lists the six flex fixes | after the 3.0 reference lands |
| Newton upstream PRs | #4316–#4318 open, blocked on the EasyCLA signature | user |
| Isaac features not covered | MPM / particles: deferred by the user until the current efforts land; Kamino (Newton's beta-1 maximal-coordinate Proximal-ADMM solver for closed kinematic loops: hard loop closure, contacts, friction, PD drives; MuJoCo's route is soft `connect`/`weld` equality constraints): a low-priority agent is checking whether Kamino runs on the Metal fork and whether MuJoCo Warp's soft closure is adequate at our step sizes (`docs/research/closed_loops_2026-09-25.md`); kinematic node targets and per-element stress for deformables, skeleton / occlusion annotators, texture-swap and scatter randomizers, ROS 2 bridge (no ROS 2 on the target): not started |

## Ledger of 2026-09-25 rows (kept for traceability)

| gap | status | next step |
|---|---|---|
| Vertical-wall terrain (inverted stairs, boxes) | **closed on the collision side** (0cfb188): exact boxes windowed around each robot + exact height scan on those cells, −6 % throughput; Isaac's checkpoints' wall-cell falls 47 / 96 → 14 / 96 (final checkpoint 0 / 32); finer heightfields made it worse, whole-terrain boxes/meshes run out of GPU memory at 4096 envs | remaining falls (earlier Isaac checkpoints in the inverted-stairs pit, 6 / 16) are physics: PhysX step and soft contacts (contact-fidelity work); task-model renders show only the heightfield on box cells |
| Rough-terrain like-for-like training | **closed** (0b45075, PARITY §1.5): ours +16.4 / 992 / level 6.1 at iteration 1500 vs Isaac's +14.1 / 966 / 5.9; curriculum climbed 50–100 iterations later; height-scan ray order defect found and fixed (Isaac's order default) | throughput 23.3 K vs Isaac's 36.5 K on the L4 (rough physics +2.3 % over flat; solver dominates) |
| `tests/test_deformable.py` failed on the main tree without the flex build | **closed**: flex branches merged into `metalsim` on both forks; the tests skip with a reason when the flex support is absent | – |
| Fast-factorization settings are G1-only (`metal_register_cholesky_max=48`, `m_dense_max=0` set in `G1VelocityTask`); `BatchSimOptions` and `wp.config` defaults unchanged because other scenes are unverified | open, low | verify on the cartpole, lift, Tron1 and lidar scenes, then make them defaults |
| Two tests were intermittent under GPU load | contact-sensor one **fixed** (a real race: the next step could overwrite `net_forces_w` before torch's clone read it; the sim now waits for torch after the clone, exact equality kept, 3/3); the PPO rollout-noise one unverified | reproduce the rollout-noise test on a loaded GPU |
| MuJoCo Warp `plane_convex` contact set | **closed, kept** (fork a8e6485 / 284dcd1, `MJW_PLANE_CONVEX=legacy` restores the old set): landing steps with a different contact set than MuJoCo C 61 → 0 of 691, forces within 1e-4; its transfer loss came from the soft default contacts, with the tuned contacts it is within 3 % of Isaac | – |
| Feet-slide reward term used MuJoCo `cvel` at the subtree COM instead of the foot body's own velocity (Isaac's `body_lin_vel_w`) | **closed** (commit af3cd6b) | per-foot velocity kernel on both engines, tested against `mj_objectVelocity` to 2e-3 m/s; the old term overstated sliding ~2.5× under random actions (−0.30 vs −0.12 per step) — every training run so far, including the reference, used the old term |
| Contact model: soft contacts and soft joint limits | **the impact-peak gap was a reporting artefact** (PARITY §1.7 correction): impulses per event match Isaac's within 1–4 % by momentum change; Isaac's sensor reports only the last 5 ms of each 20 ms step and misses 42–45 % of its own torso-impact impulse; the tables' "drop peak" was the torso impact after the fall, the landing itself reads 917 vs 1030 N per foot. Hard joint limits still fix limit excursions (0.030 → 0.001 rad) at no cost; stiff contacts (τ 5 ms; Isaac Lab 3.0's own mapping solref (5 ms, 1.375)) exceed PhysX's bounded force and double contact chatter | re-rank the tuning presets by impulse and 20 ms mean force, feet slide, transfer and stability (not sampled peaks); **done**: final pick `tau10_impact_hardlimits` (impact-only stiffening, τ 10 ms, hard limits) is the G1 task default (`contact_cfg="recommended"`, 1.02× cost; every other preset selectable, DECISIONS ledger); confirming flat training run +27.85 / 1000 vs +28.4 / 26.3 / 26.0 with default contacts, inside seed spread (PARITY §1.5); air time is policy stage (research note) |
| Learning parity on G1 | **closed** (like-for-like: fixed PPO on Isaac's flat config, MuJoCo Warp: +28.4 / 1000 at iteration 1000 vs Isaac's +27.3 / 991; full episodes at the same iteration; one seed each; PARITY §1.5) | remaining: slower rise between iterations 200 and 500 tied to feet slide / air time (contact tuning in progress); three seeds now confirm (28.4 / 26.3 / 26.0 vs Isaac 27.3) |
| G1 task carried the rough task's reward weights/ranges while Isaac's reference trained the flat config | **closed** (5f208f0): `reward_cfg="flat"` ports G1FlatEnvCfg exactly (13 terms tested against Isaac's formulas); rsl_rl on the ported task: full episodes by it 150, return 20.5 vs Isaac's 23.6 at it 399, linear tracking 0.934 vs 0.922 | remaining term gaps are contact-model quantities: feet_slide 3.4× Isaac's, feet_air_time half (with the contact-fidelity work) |
| Rendering vs Isaac RTX; denoiser | **closed as far as physics allows** (b0cf9b1, PARITY §1.7): robot vs RTX path tracer 13.9 → 28.8 dB (RTX's own two modes are 25.9 dB apart), whole frame 45.6 dB on kinematic replay, sky pixel matched without per-image gain; OIDN 2.5 on Metal as the hero-frame denoiser (+22 ms/frame); parity presets only, old renderer default and bit-identical | remaining: exposure is a fitted constant (documented formula kept, 9 % off); RTX real-time's shortcuts and NVIDIA's denoisers are not reproducible; MetalFX not used |
| Contact sensing (touch sites, no force history) | **closed** (e6d458d): `metalsim/sensors/contact.py` = Isaac's ContactSensor semantics (net force vectors, substep history, filtered force matrix, air/contact time and first-contact flags with Isaac's formulas) as Warp kernels inside the step graph; matches MuJoCo C's per-body sums to 1.3e-4 N on shared contact sets; +0.3 % step cost at 4096 envs | hook into the G1 task (two lines, handed to the task owner); Newton-side exact forces with the fork agent |
| Sensors: beam divergence, multi-return, radar | **closed with stated scope** (76c2708): `lidar_ext` kernel with beam divergence (up to 32 sub-rays, uniform/Gaussian), multi-return with echo merging (NVIDIA does not document its rule; ours: hits closer than `min_echo_sep` merge power-weighted, strongest `max_returns` kept), per-material reflectance intensity; default scan bit-identical to before; radar-lite (`metalsim/sensors/radar.py`: range/azimuth/elevation/radial velocity to 4.7e-7 m/s vs MuJoCo C, Lambertian RCS proxy). Cost per 64-beam scan at 1024 envs: 2 returns 2.5×, 4–8 sub-rays 4–8× the 23.5 µs default | not modelled: intra-scan motion, fire timing, NVIDIA's intensity formula, radar beam pattern / Doppler binning / CFAR; lidar-based RL demonstrated earlier |
| Exact terrain heights | **closed at grid points** (4b23a3a): Isaac Lab v2.3.2's generator ported (`metalsim/learn/isaac_terrain.py`) and verified against Isaac's own code run here (`tests/isaac_terrain_ref.py`): env origins exact, all 2.4 M grid heights equal the top surface of Isaac's mesh, 99.99 % of grid points within 1e-5 m of Isaac's ray cast; throughput unchanged (55.2–55.7 K). The old generator differed by up to 3.4 m (different column layout, difficulty, border, orientation) | remaining: between grid points a 0.1 m heightfield turns Isaac's vertical walls into ramps (mean 8 mm, p99 14 cm, max 0.36 m off-grid); CUDA box heights reproduced from torch's Philox, not yet confirmed on NVIDIA hardware; task should use seed 42 and Isaac's env→column assignment (handed to the task owner) |
| Throughput | **doubled** (2026-09-25, PARITY §1.4): G1 flat full PPO loop 27.9 K → 56.6 K env-steps/s at 4096 envs (rough 23.7 K → 41.9 K), physics unchanged to float noise; cause was the G1's 43-dof dense Cholesky falling off the Warp fork's register fast path (limit 40), plus sparse L'DL, a one-world-per-thread factorization and a fused Hessian update; the earlier "23 % reset/obs overhead" was a measurement artefact (6.5 ms of 128) | now above Isaac 2.3.2 + PhysX on the L4 (45.9 K) and 0.69× the published 4090 number; remaining: the 43×43 Cholesky itself, the sparse factorization, the Warp MLP inference (~6 ms/step) |
| Camera-RL throughput | **12,688 env-steps/s incl. training** (was 7,483; Isaac's published 32 K on a 4090 = 2.5× gap, a hardware gap: the 4090 has ~5× the fp32 throughput). Update 7.2 → 3.22 s: `fast_update` (compiled loss and clip, fused Adam, gather fused into the compiled graph) + Metal kernels for conv1's weight gradient, conv2's input gradient and conv1's forward (`metalsim/learn/metal_conv.py`, 18 tests); learning curve identical to the reference (return 86.3 vs 86.3 over the last 20 iterations). MLX whole-update measured **not faster in fp32** (4.22 s; 3.30 s best case) and 1.15× in fp16 (different numerics) | closed as far as it goes: remaining headroom to the ~1.7–2.0 s update floor gives ~17–19 K, then the 1.8 s rollout caps at ~36 K; untried: Apple's Metal Performance Primitives conv op for conv1 forward / conv3 input gradient (estimated) |
| Replicator | **closed for the listed annotators, writer and Isaac Lab event terms** (d1ab426, PARITY §2; 24 exact-match tests; Warp kernels 2–90× cheaper than the torch versions, which are kept) | not done: per-pixel occlusion, skeleton data, collider offsets, rigid-body scale, texture swap, scatter randomizers (effort estimates in `docs/research/replicator_2026-09-25.md`); MuJoCo has one friction coefficient, so dynamic friction / restitution randomization has no effect |
| MaterialX | **closed for common surface types** (a2013aa, PARITY §2): standard_surface / open_pbr_surface / UsdPreviewSurface / gltf_pbr flattened at load time, zero render cost | not mapped: coat, sheen, transmission, subsurface, thin film, anisotropy, normal maps, low-level BSDF graphs |
| Deformables on Metal | **PhysX 5.1 comparison measured** (PARITY §2): XPBD cloth matches PhysX's drape to 2 mm rest height and 3.3 cm height map; flex matches the rope's period (1.006 vs 1.026 s) but not its damping; soft cube too lively (bounce 0.262 vs 0.135 m). Flex branch merged into both forks with rigid results unchanged (Warp f194006, MuJoCo Warp 8fbf965; 7 + 16 patches; `tests/test_deformable.py` skips without the flex build) | **rope and cube closed by fitting** (PhysX 5.1): rope XPBD 1.077 s / decrement 0.63 vs 1.026 / 0.59; soft cube with implicit flex damping and direct contact stiffness/damping (the time-constant form is floored at 2× the timestep): bounce 0.140 vs 0.135 m, settle 0.51 vs 0.55 s, peak penetration 2.6 vs 1.6 mm (was 35.7), 2.9 K env-steps/s at 4096 envs. Rope presets: `physical` (Euler-Bernoulli, period 1.30 s; default) and `physx_ref` (mesh-locked fit). Locking evidence: NVIDIA's docs and the FEM literature cited; locally flex's rod is 14× / 7× too stiff with one / two cells across and the cube gets livelier as its mesh refines (0.122 / 0.140 / 0.216 m at 2 / 4 / 8 cells); the 3.0 recorder sweeps 2 / 4 / 8 cells on both backends to test it on Isaac's solvers | remaining: our cube compresses less on impact (lowest centroid 0.095 vs 0.079 m); the matching elasticity damping is 20× PhysX's stated value (PhysX's decay is numerical); the volume formulation choice and the Isaac Lab 3.0 recordings are pending; `UPSTREAM.md` fixes not filed and the fork's flex commits/patches 0008–0016 need their co-author trailer removed on a fresh branch before filing |
| Upstreaming the forks | Newton: issues and PRs filed; Warp and MuJoCo Warp forks (Metal backend fixes, heightfield contacts, plane-convex contact set, flex contact fixes, GPU sorts) not yet proposed upstream | – |

## Newton XPBD: archived as an experimental option (2026-09-25)

Decision after the measurements: MuJoCo Warp is the parity engine (see "Engine position" below);
Newton XPBD stays in the tree (`engine="newton"`, its tests, the fork with six solver fixes) as a
validated 1.6× throughput option with a stated fidelity cost (loose joints, 14–23 % over-travel of
PhysX-trained policies, harder impacts) and is not used for parity claims. The three solver defects
found are being filed upstream (`scripts/diagnostics/newton_upstream/FILED.md`). No further GPU or
agent time is spent on it unless MuJoCo Warp's contact work fails to close the impact-peak gap.

## Ledger of the Newton XPBD evaluation (status 2026-09-24, measured unless noted)

| issue | status | evidence / next step |
|---|---|---|
| Joint drives not engaging under XPBD | **closed** | three causes: `State.joint_q` never written by XPBD (use `eval_ik`), targets indexed by DOF instead of coordinate layout (shifted every G1 target by one joint), and Newton's default joint relaxation 0.7/0.4 transmitting torque wrongly (+43 % / −18 % on a pendulum; 0.4/0.4 exact). `scripts/diagnostics/newton_xpbd_drives.py` |
| XPBD's compliance drive is not a PD with stiffness `ke` (72–1240 Nm/rad for `ke` 200 over 1–16 it.) | **closed by replacement** | Isaac's PD law computed each substep and applied through `Control.joint_f` (`ActuatorPD`); ≤ 0.001 rad from the exact static equilibrium at 4 it. Upstream issue to raise |
| No effort limits, armature, joint friction, velocity limits in XPBD | **closed for the G1** | `ActuatorPD` clips to the effort limit (only the 20 Nm ankles saturate, 0.5–1.5 % of landing steps) and adds armature isotropically to the child inertia (required: NaN without it; axis-only armature invalidates 32 links). Joint friction / velocity limits not needed by the G1 task |
| GJK/MPR narrow phase needs fixed-size arrays; Metal codegen lacked them | **closed** | Warp fork 786cdae: 20/20 fixed-array tests on Metal incl. graph capture; mesh = primitive contacts to 0.001 mm; G1 on its own convex meshes, < 3 % throughput cost |
| Stability / cost of stiff drives under XPBD | **closed** | tracks MuJoCo C within 0.022 rad (4 it., 1.25 ms) or 0.044 rad (8 it., 2.5 ms) at 4.0–4.5× MuJoCo Warp's physics rate at 4096 envs |
| Newton's XPBD defects, upstream | **filed** (2026-09-25, `scripts/diagnostics/newton_upstream/FILED.md`): issues newton-physics/newton#4313 (angle wrap), #4314 (inconsistent joint relaxation), #4315 (iteration-dependent drive stiffness); pull requests #4316 (angle wrap, plus a second bug found on the way: swing angles below ~0.02 rad read at half their value), #4317 (consistent relaxation, default 0.7 → 0.5 with the reasoning), #4318 (opt-in PD joint drives, stacked on the other two); each with tests that fail on main and pass on the branch, no regressions in ~1,500 XPBD-related tests | blocked on the Linux Foundation EasyCLA signature (account owner) and maintainer CI approval |
| Newton rests the G1 lower than MuJoCo | **explained** (fork item 5): with identical colliders, masses and joint angles (to 1e-4 rad) the quasi-static stand is 4.7 mm lower because relaxed hard joint rows keep a steady anchor gap under load, summed over six leg joints; converged joints (16 it., 0.625 ms, or joint coloring 0.8) reduce it to ~1 mm, the rest being MuJoCo's soft-contact sink | closes with the drift remedy |
| VBD solver fails to compile on Metal | **closed** (2026-09-25, `docs/research/newton_vbd_metal_2026-09-25.md`): Newton 1.5.2 (Isaac Lab 3.0's pin) runs VBD on the Metal fork unchanged; Newton main failed only on a missing `nextafterf` in Metal, added to the Warp fork (9050cb54, patch 0007) with a bit-exact test; Newton's VBD / cloth / collision suites 89 of 94 on Metal, same as CPU; soft cube within 0.13 mm of CPU over 200 steps; cloth 441 particles 3.7 K env-steps/s at 4096 worlds, soft cube 0.57 K (compute-bound elasticity kernel) | VBD throughput is low; Newton's coupled solvers (proxy, ADMM) also run graph-captured on Metal within 3.7 mm / 0.08 mm of CPU |
| State layout: maximal coordinates, no MuJoCo sensors, approximate contact forces | **closed** | `G1VelocityTask(engine="newton")`: same kernels, body state vs MuJoCo to 1e-5, foot/torso contact from collider forces, 9 invariant tests; full PPO loop 112.6 K env-steps/s at 4096 envs (4.1× MuJoCo Warp) |
| Learning parity on Newton | **close, not equal**: fork in-solver drive at 4 it. / 0.625 ms reaches +24.4 at iteration 1000 vs MuJoCo Warp's +28.5 and Isaac's +27.3 (all Isaac's flat weights), 7–10 behind at 300–750, at 1.6× MuJoCo Warp's training-loop rate; pinned 1.25 ms +7.8, pinned 0.625 ms +20.5 | MuJoCo Warp stays the default; Newton is a validated faster option with the stated gap; the unexplained 72 %-longer stride is the lead |
| `ActuatorPD` is MetalSim code, not Newton's | **superseded on the fork** by the solver's own PD drive (equal static accuracy, 3σ 1 vs 17 blow-ups); the builder's isotropic armature and the solver's `joint_armature_inertia` must not both be applied | switch NewtonSim to the fork drive once its throughput is measured |
| 3σ stress check at the fast Newton settings | **closed with the fork's in-solver drive**: 0 blown of 1024 at 4 it. / 1.25 ms and 0.625 ms (peak joint speed 85–89 rad/s) vs 19 blown / 386 rad/s with ActuatorPD (`runs/newton_fork_solver_sigma3.log`, GPU, queued) | – |
| Newton monitor coverage: penetration / energy / overflow checks have no source fields | open, low | derive from contact depth and body energy |
| Newton rough terrain: no speed advantage (heightfield collision costs ~6× Newton's flat physics; rough full PPO loop 26.1 K vs MuJoCo Warp's 24.8 K env-steps/s) | **new, open** | profile Newton's heightfield narrow phase; a plane-per-cell contact like the MuJoCo Warp patch, or a coarser broad phase |
| Newton rough terrain | **closed** (c65aa09) | Isaac's generated terrain as a Newton heightfield with MuJoCo's geometry; 60 dropped spheres rest on MuJoCo's surface to 0.1 mm median; scanner torso pose matches MuJoCo; throughput in the queued benchmark |
| Newton's joint constraints do not fully converge at the fast settings (feet vs joint-angle kinematics at the 4 it./1.25 ms training setting: 15 mm median / 55 mm p99; off-axis joint rotation 2.9 mrad median / 31 mrad p99, i.e. below the ±10 mrad observation noise at the median but not at the tail) | open, medium; quantified (9842062) | observations already come from body state so the policy never sees the anchor gap; the cheapest remedy is a shorter step, not more iterations: 4 it. at 0.625 ms drifts half as much as 8 it. at 1.25 ms at equal solver work (3.4 vs 6.1 mm median) |
| XPBD mishandles revolute joints whose angle can cross ±π (the G1's elbow-pitch limit is 3.42 rad; a crossing read as a ~2π violation fired a huge impulse: passive energy ×925 in 2 s; also behind most 3σ blow-ups) | **fixed at source** in the Newton fork (https://github.com/pulipakaa24/newton branch `metalsim`, f844a4e6; MetalSim 929a278): one-DoF joints measure their angle within π of a stateless reference (middle of the limit range, else the drive target); 4 new solver tests fail upstream and pass on the fork; third upstream draft ready | measured on CPU: reproducer peak joint speed 746,531 → 3.0 rad/s; G1 passive energy with the original 3.42 rad limit ×925 → ×0.60; 3σ blow-ups 36 → 11 of 1024 (remainder are fallen robots at ~900 rad/s). The limit clamp in `newton_backend.py` becomes unnecessary once the fork is adopted (`limit_margin=None`) |
| Drift remedies evaluated | measured (fork item 4): 4 it. / 1.25 ms 14.8 / 51.8 mm (p50 / p99); **4 it. / 0.625 ms 3.2 / 17.7 mm at 47.2 K env-steps/s on the fork**; joint colouring 0.8: 3.4 / 20.1 mm but only 39.5 K and worse 3σ (30 blown vs 4); +4 joint-only passes 5.7 / 26.3; projection and Featherstone unusable | conclusion: halving the substep is the remedy; the fork drive gives 12.7 / 39.9 mm at 1.25 ms on its own |
| Newton fidelity against PhysX | **partly met**: open-loop protocol rows comparable to MuJoCo Warp (PARITY §1.7); transfer of Isaac's checkpoints over-travels by 37–48 % at 4 it. / 1.25 ms and 20–35 % at 0.625 ms with ActuatorPD, **14–23 % at 0.625 ms with the fork's in-solver drive** (MuJoCo Warp within 5 % of Isaac); every robot stays upright | the long stride is unexplained (push-off diagnostic queued); the fork's 0.625 ms learning run with the in-solver drive decides whether Newton can be a fidelity-grade option |
| Featherstone (reduced coordinates, would remove the drift by construction) | open, not viable as measured | finite only at 0.3125 ms with 12 cm contact sinking, no contact-force reporting, ~1 env-step/s on Metal |
| Newton contact forces approximate | **closed on the fork** (9f626ce2): `SolverXPBD(body_contact_forces=True)` gives exact per-body force vectors (total and normal-only) from the iteration impulses: within 0.02 % / 0.14 % of m·g on a box stack, feet sum 1.0006× weight at rest; maps to `ContactSensor.net_forces_w` | wire into the Newton path's contact terms |
| Newton over-travel: a stride-length effect independent of the drift (strides ~75 % longer at similar cadence; friction, leg damping, push-off impulse and the anchor gap ruled out) | **localised, archived**: the swing-leg joint trajectories themselves differ (hip-pitch travel 0.121 vs 0.061 rad, swing 0.123 vs 0.108 s, foot lands 0.065 vs 0.031 m ahead), i.e. drive / dynamics rather than kinematics; closed-loop test, commanded targets not separated | not pursued further (Newton archived) |
| Newton fork, final state (https://github.com/pulipakaa24/newton branch `metalsim` at 90e23324; 13 new solver tests fail upstream, pass on the fork) | **six solver defects fixed at source**: angle wrap (G1 passive energy ×925 → ×0.60 with the original limits), relaxation math (1.36 / 0.78 → 1.001 / 0.999 of exact, new defaults 0.5/0.4), in-solver PD drive (stiffness 199.6–200.3 Nm/rad at 1–16 it.; hand/arm/leg step responses 1.03 / 1.07 / 1.01 of MuJoCo C; 3σ blow-ups 0 vs 17 with ActuatorPD), exact per-body contact forces (0.02 % of weight), rest height explained (anchor separation under load, 0.6–0.8 mm with converged joints), drift remedies measured | costs: the in-solver drive runs at 98.6 K vs 141 K env-steps/s (4 it. / 1.25 ms, physics-only) and 49.1 K vs 71 K at 0.625 ms, and its 1-iteration statics are worse (arms 0.034 rad; 0.008 at 4 it.). Still open on the fork: ankles read low against MuJoCo under every drive (step-response ratio 0.67–0.83, not drive-related), an ultralight three-link chain alternates ±33 % without colouring, hinge limit ranges ≥ 2π unhandled, nothing proposed upstream yet |
| Isaac Lab 3.0's rigid/deformable coupling is a lagged two-solver exchange (`SolverCoupledProxy`: MuJoCo Warp owns the robot, VBD owns the particles, robot links appear in VBD as proxy bodies with MuJoCo's effective mass, contact forces fed back on the next pass; "lagged" mode, 1 proxy iteration, 2 substeps, 4 for cables; ADMM and hand-written couplers also ship) | **verified from source** (Newton 1.5.2, Isaac Lab v3.0.0-EA) | our coupling design should mirror this exchange, not PhysX's per-iteration one and not MuJoCo Warp flex's single solve |
| Newton 1.5.2's MuJoCo solver requires mujoco-warp 3.11; our fork (3.14 + contact fixes) and the 1.7.0.dev fork do not build against it | **new, open** | the 3.0 reference environment `.venv-newton152` (`scripts/setup_newton152.sh`) runs stock mujoco-warp 3.11 without our contact fixes; either port the fixes to 3.11 or move the Newton reference to the version that takes 3.14 |
| PhysX deformable port | **scoped and prototyped** (`docs/research/physx_deformables_port_2026-09-25.md`): PhysX 5 is BSD-3 throughout; particle cloth (PBD, 16 substeps, 8 partition copies blended back) was removed in PhysX 5.9, so Isaac Sim 6 has only FEM surface cloth; FEM soft body is co-rotational XPBD on voxel tets with per-iteration rigid exchange; none of our backends is algorithmically the same. Prototype `metalsim/physics/physx_cloth.py` (PhysX 5.6.1 particle-cloth step, batched, graph-captured, 7 tests): 41.3 K env-steps/s at 4096 worlds vs XPBDSim's 285 K on the same cloth, softer (2.9 % vs 0.9 % stretch) | low: the recorded 5.1 cloth's spring set comes from omni.physx's closed cooker (Isaac writes the springs onto the USD prim; dump those attributes when recording); self-collision, mesh/heightfield contacts and two-way coupling not ported; estimates PBD cloth 5–6 days, FEM 6–9 |
| Newton API churn (alpha) | low | pin the version (1.7.0.dev) |
| Importer copies the USD's gravity 0 | low | set gravity explicitly (done in `g1_builder`) |

## Reference version (corrected 2026-09-25)

Isaac Lab 3.0 is **released** as Early Access (v3.0.0-EA, 2026-09-16; Isaac Sim 6.1, Python 3.12,
PyTorch 2.11, Warp 1.16, Newton 1.5.2), after 3.0.0-beta (March) and beta2 (June) on Isaac Sim 6.0.
Earlier text in these documents that described Isaac Lab 3.0 and its Newton backend as future was
wrong. What the release provides (from its release notes and the Newton-integration docs): a
multi-backend architecture selected per task (`physics=isaacsim_physx`, `physics=ovphysx`,
`physics=newton_mjwarp`), Newton with MuJoCo-Warp as the primary solver plus VBD, MPM, Kamino and
coupled rigid/deformable/MPM workflows, "expanded support for deformables, cables, and particle
systems"; the Newton integration is marked under active development with a limited set of classic RL
and flat-terrain locomotion examples and cross-backend policy transfer (Newton ↔ PhysX) used as its
own validation. Isaac Sim 6.0 removed the old PhysX cloth API and introduced `DeformablePrim`
(surface/mesh and volume/tet). Consequences: (1) every reference in PARITY.md so far is against
Isaac Sim 5.1 + Isaac Lab 2.3.2 (PhysX) and is labelled as such; (2) an Isaac Sim 6.1 + Isaac Lab
3.0-EA environment is being built on the VM to re-record the G1 references on both backends, so
MetalSim's MuJoCo Warp is compared like for like against Isaac's own MuJoCo-Warp on NVIDIA hardware
(same solver, different hardware) and against PhysX; (3) for deformables the current-release
formulation under Newton is VBD with coupling, so making VBD run on the Metal Warp fork is the
priority over porting PhysX's solvers; PhysX's `DeformablePrim` remains the PhysX-backend reference.

## What "fidelity" is measured against (grounding note, 2026-09-25)

PhysX is not established as the most realistic engine; no engine is. The robotics literature leans
the other way for articulated robots: the ICRA 2015 comparison (Erez, Tassa, Todorov; MuJoCo's own
authors) found MuJoCo the most accurate and fastest for robotics-type contact, with PhysX/Bullet/Havok
suited to gaming-type contact; ETH's SimBenchmark rates engines on speed-accuracy curves and notes
MuJoCo's soft contact cannot control elasticity and has consistent slip; a 2023 comparative study
reports Isaac Sim's PhysX trading contact-dynamics accuracy for scalability; and humanoid sim-to-real
work (Humanoid-Gym, PolySim) uses MuJoCo as the validation step for Isaac-trained policies because its
dynamics are closer to the real robot. NVIDIA itself now ships MuJoCo Warp as Newton's primary solver
in Isaac Lab 3.0. So PhysX parity in these documents is a *reference* goal (matching what Isaac users
get, on hardware we cannot run), not a realism endpoint. The realism endpoint is the real robot: a
hardware protocol (drop / step-response / walking-distance on the user's Tron1) is the reference that
outranks both simulators, and where MuJoCo's soft contacts and slip deviate from PhysX they also
deviate from reality in the same direction, so tuning them toward PhysX's hard contacts is not wasted.

## Engine position (2026-09-24, after the measurements)

MuJoCo Warp is the parity engine, not a stopgap. Measured: it matches the PhysX open-loop protocol
as closely as Newton XPBD on the hold and the drop (PARITY §1.7); Isaac's PhysX-trained policies
walk the same distance on it within 0.1 m over 8 s at every training stage (§1.5); with the learner
fixed it trains at least as well as Isaac's own run. Its remaining differences from PhysX (3–5×
contact-impact peaks at equal mean force from MuJoCo's soft contacts; soft joint limits; the
`plane_convex` contact-set defect) are tunable or fixable within MuJoCo Warp. Isaac Lab 3.0 itself
moves to Newton with **MuJoCo Warp as the default solver**, so this is also the like-for-like
reference engine going forward. Newton XPBD is a throughput option (4× on flat ground, none on
rough) with fidelity costs measured above (joint drift, slower tracking-reward learning); it stays
optional unless the fork's drift and drive fixes close those.

## What a Newton XPBD replacement would and would not change

Subsumed (would not need to be carried over): the MuJoCo Warp heightfield patch (Newton has native
heightfields, subject to the narrow-phase port), the 2.5 ms step chosen for MuJoCo's explicit drive
stiffness (XPBD drives are constraint-based; their stable settings are their own question), the
per-world contact-slot cap (MuJoCo Warp launch geometry), MuJoCo's soft contact / soft limit tuning
(XPBD contacts and limits are hard), and any further optimization of the MuJoCo Warp step (note: the
"23 % reset/obs overhead" quoted here earlier was a measurement artefact; see PARITY §1.4).

Unchanged by the engine (worth doing regardless): the USD scene layer (visual meshes, materials,
cameras, lights), all three renderer tiers and the rendering-fidelity comparison against Isaac RTX,
ray-traced sensors (lidar, depth, height scan) and the lidar RL task, the replicator, the learner
(Warp rollout policy, PPO, clipped value loss, anomaly monitor, pre-flight), the reward/termination
term logic (needs new state sources under Newton but not new logic), the terrain generator and
curriculum, the camera-RL pipeline and its MPS update bottleneck, the Isaac reference measurements
and the fidelity protocol, documentation and videos. The Warp-backend memory fix applies to any Warp
engine.

Engine-coupled and would need redoing: every kernel that reads MuJoCo state (qpos/qvel/xpos/xmat,
sensordata touch, qacc, qfrc_actuator, cvel) in the G1 task, the height scanner's body-pose input,
the renderer's geom-pose input (from body poses + shape transforms), the reset path
(reset_data/kinematics), the parity tests that assert MuJoCo model fields, and the benchmark harness.

Camera-RL throughput, measured 2026-09-24 on the M4 Max: the skrl update (4 epochs × 32 minibatches
over 65,536 100×100 images) takes 5.9 s on MPS versus 1.4 s for the 64-step rollout, so the pipeline
is update-bound at ~8K env-steps/s; minibatch size makes no difference (compute-bound conv
backward), fp16 autocast gains 10 %, channels-last is 4× slower (5.99 s, avoid). This is a PyTorch-MPS convolution limit,
independent of the physics engine; the options are a Warp/Metal implementation of the encoder's
backward pass or a smaller encoder than Isaac's.

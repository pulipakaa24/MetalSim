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
| Install path | public forks of Warp and MuJoCo Warp with setup scripts | fresh clone builds |
| Silent RL failures | task-agnostic anomaly monitor + pre-flight | first probe flagged joint-limit and penetration issues on its own |

## Still open (MuJoCo Warp path)

| gap | status | next step |
|---|---|---|
| MuJoCo Warp `plane_convex` drops foot-mesh vertices more than 1 mm shallower than the deepest one (`collision_primitive.py:94`), so a tilted landing foot gets 2 contacts where MuJoCo C finds 4; transient force differences up to ~1 kN during landings (61 of 691 in-contact steps on a G1 landing) | **new, open** (found by the contact-sensor verification) | widen the threshold or take the k deepest vertices in the fork, verify against C on the landing test; also a candidate cause of the contact-peak gap vs PhysX in PARITY §1.7 |
| Feet-slide reward term used MuJoCo `cvel` at the subtree COM instead of the foot body's own velocity (Isaac's `body_lin_vel_w`) | **closed** (commit af3cd6b) | per-foot velocity kernel on both engines, tested against `mj_objectVelocity` to 2e-3 m/s; the old term overstated sliding ~2.5× under random actions (−0.30 vs −0.12 per step) — every training run so far, including the reference, used the old term |
| Contact model: soft contacts and soft joint limits (5–7 cm impact penetration, 0.17 rad past limits) | measured against PhysX on the L4 (PARITY §1.7): the 1 m drop lands in the same state (root z RMSE 3.5 cm, joints within 0.06 rad) but MetalSim's contact-force peaks are 3.4–5.7× Isaac's at equal mean force; Newton XPBD gives 0.16–0.69 cm impact penetration (§2.1) | subsumed by the Newton path; on MuJoCo, τ 5 ms + impedance 0.99 + speculative contact is the tunable |
| Learning parity on G1 | **closed for the learner** (fixed PPO: return +32.8, episode length 995 at iteration 1000 on MuJoCo Warp, tracking rsl_rl's +35.5 within a few units; Isaac +27.3 under its own weights). Root cause: the real rsl_rl on our physics learns like Isaac (full episodes by it 200, tracking 0.91 vs 0.92 at it 1000; return +39.6 vs Isaac's +27.2 under our weights); our PPO had three rollout bugs (repeated exploration noise every rollout, phantom action at rollout boundaries, time-outs treated as falls), fixed in f0f2115 (PARITY §1.5) | demonstrate the fixed PPO with a 1000-iteration run; ablate the three fixes; explain rsl_rl's ~50-iteration later takeoff |
| G1 task carried the rough task's reward weights/ranges while Isaac's reference trained the flat config | **closed** (5f208f0): `reward_cfg="flat"` ports G1FlatEnvCfg exactly (13 terms tested against Isaac's formulas); rsl_rl on the ported task: full episodes by it 150, return 20.5 vs Isaac's 23.6 at it 399, linear tracking 0.934 vs 0.922 | remaining term gaps are contact-model quantities: feet_slide 3.4× Isaac's, feet_air_time half (with the contact-fidelity work) |
| Rendering vs Isaac RTX; denoiser; MetalFX; MaterialX | measured (PARITY §1.7): robot-only, states agreeing, tier 2 vs RTX 13.7 dB PSNR / 0.48 SSIM (tier 0: 11.5 / 0.35); silhouettes IoU 0.83, robot depth 2.6 cm RMSE; Isaac's RT and PT frames equally far from ours, so the gap is materials/lights (darker plates, hard sun shadow), not noise | re-record with a grey ground (stage 6, queued) for whole-frame rows; then material/light model work: area sun, OmniPBR parameter audit, denoiser |
| Contact sensing (touch sites, no force history) | **closed** (e6d458d): `metalsim/sensors/contact.py` = Isaac's ContactSensor semantics (net force vectors, substep history, filtered force matrix, air/contact time and first-contact flags with Isaac's formulas) as Warp kernels inside the step graph; matches MuJoCo C's per-body sums to 1.3e-4 N on shared contact sets; +0.3 % step cost at 4096 envs | hook into the G1 task (two lines, handed to the task owner); Newton-side exact forces with the fork agent |
| Sensors: beam divergence, multi-return, radar | **closed with stated scope** (76c2708): `lidar_ext` kernel with beam divergence (up to 32 sub-rays, uniform/Gaussian), multi-return with echo merging (NVIDIA does not document its rule; ours: hits closer than `min_echo_sep` merge power-weighted, strongest `max_returns` kept), per-material reflectance intensity; default scan bit-identical to before; radar-lite (`metalsim/sensors/radar.py`: range/azimuth/elevation/radial velocity to 4.7e-7 m/s vs MuJoCo C, Lambertian RCS proxy). Cost per 64-beam scan at 1024 envs: 2 returns 2.5×, 4–8 sub-rays 4–8× the 23.5 µs default | not modelled: intra-scan motion, fire timing, NVIDIA's intensity formula, radar beam pattern / Doppler binning / CFAR; lidar-based RL demonstrated earlier |
| Exact terrain heights | **closed at grid points** (4b23a3a): Isaac Lab v2.3.2's generator ported (`metalsim/learn/isaac_terrain.py`) and verified against Isaac's own code run here (`tests/isaac_terrain_ref.py`): env origins exact, all 2.4 M grid heights equal the top surface of Isaac's mesh, 99.99 % of grid points within 1e-5 m of Isaac's ray cast; throughput unchanged (55.2–55.7 K). The old generator differed by up to 3.4 m (different column layout, difficulty, border, orientation) | remaining: between grid points a 0.1 m heightfield turns Isaac's vertical walls into ramps (mean 8 mm, p99 14 cm, max 0.36 m off-grid); CUDA box heights reproduced from torch's Philox, not yet confirmed on NVIDIA hardware; task should use seed 42 and Isaac's env→column assignment (handed to the task owner) |
| Throughput | MuJoCo Warp at the 2.5 ms training setting: 27.7 K full-PPO env-steps/s (0.34× the 4090's published 82 K); Newton XPBD: 112.6 K (1.4×), PARITY §1.4 | subsumed by the Newton path |
| Camera-RL throughput | 8.1 K env-steps/s incl. training vs Isaac's 32 K (4090); learns. Profiled 2026-09-24 (`docs/research/metal_cnn_update_2026-09-24.md`, `runs/camera_update/`): update 6.67 s, of which conv backward 48 % (conv1 weight gradient at 1.6 TFLOP/s and conv2 input gradient at 2.5 vs 8–11 elsewhere; PyTorch's Metal backend has no native conv2d training kernels), image preprocessing 21 %, clip + Adam 10 %. Measured alternatives: torch.compile + fused Adam 5.29 s; MLX whole-update 4.44 s fp32 / 3.79 s fp16 | in progress: torch.compile + preprocessing fix + custom Metal kernels for the two slow gradients (est. 3.9–4.2 s, ~12 K env-steps/s). **Possible next step, compounding:** move the whole update into MLX (fused preprocessing + optimizer) around those kernels, est. 3.2–3.6 s; fp16 as a separately reported configuration. Arithmetic floor on the M4 Max ≈ 1.7–2.0 s; Isaac's 32 K is a hardware gap |
| Deformables on Metal; upstreaming the forks | open | – |

## Opened by the Newton XPBD evaluation (status 2026-09-24, measured unless noted)

| issue | status | evidence / next step |
|---|---|---|
| Joint drives not engaging under XPBD | **closed** | three causes: `State.joint_q` never written by XPBD (use `eval_ik`), targets indexed by DOF instead of coordinate layout (shifted every G1 target by one joint), and Newton's default joint relaxation 0.7/0.4 transmitting torque wrongly (+43 % / −18 % on a pendulum; 0.4/0.4 exact). `scripts/diagnostics/newton_xpbd_drives.py` |
| XPBD's compliance drive is not a PD with stiffness `ke` (72–1240 Nm/rad for `ke` 200 over 1–16 it.) | **closed by replacement** | Isaac's PD law computed each substep and applied through `Control.joint_f` (`ActuatorPD`); ≤ 0.001 rad from the exact static equilibrium at 4 it. Upstream issue to raise |
| No effort limits, armature, joint friction, velocity limits in XPBD | **closed for the G1** | `ActuatorPD` clips to the effort limit (only the 20 Nm ankles saturate, 0.5–1.5 % of landing steps) and adds armature isotropically to the child inertia (required: NaN without it; axis-only armature invalidates 32 links). Joint friction / velocity limits not needed by the G1 task |
| GJK/MPR narrow phase needs fixed-size arrays; Metal codegen lacked them | **closed** | Warp fork 786cdae: 20/20 fixed-array tests on Metal incl. graph capture; mesh = primitive contacts to 0.001 mm; G1 on its own convex meshes, < 3 % throughput cost |
| Stability / cost of stiff drives under XPBD | **closed** | tracks MuJoCo C within 0.022 rad (4 it., 1.25 ms) or 0.044 rad (8 it., 2.5 ms) at 4.0–4.5× MuJoCo Warp's physics rate at 4096 envs |
| Newton's default relaxation and biased drive | **fixed at source** in the fork (fda6658a): both parts of a positional row now scale by the linear relaxation (pendulum torque 1.362 → 1.001, gravity 0.776 → 0.999 of exact); new defaults 0.5/0.4 chosen by measurement (consistent 0.7 injects energy on the floating G1); `joint_legacy_relaxation=True` restores the old behaviour. New `SolverXPBD(joint_drive_mode="pd")`: pendulum stiffness 200.1 / 200.2 / 200.1 Nm/rad at 1 / 4 / 16 it. for `ke` 200 (upstream 72 / 298 / 533); fixed-base G1 ≤ 0.0008 rad from 4 it.; floating G1 3σ blow-ups 1 vs 17 of 1024 with ActuatorPD. ActuatorPD is now unnecessary on the fork (kept) | throughput of the fork drive pending; file the three upstream issues |
| Newton rests the G1 lower than MuJoCo | **explained** (fork item 5): with identical colliders, masses and joint angles (to 1e-4 rad) the quasi-static stand is 4.7 mm lower because relaxed hard joint rows keep a steady anchor gap under load, summed over six leg joints; converged joints (16 it., 0.625 ms, or joint coloring 0.8) reduce it to ~1 mm, the rest being MuJoCo's soft-contact sink | closes with the drift remedy |
| VBD solver fails to compile on Metal | open, low | not needed for rigid robots |
| State layout: maximal coordinates, no MuJoCo sensors, approximate contact forces | **closed** | `G1VelocityTask(engine="newton")`: same kernels, body state vs MuJoCo to 1e-5, foot/torso contact from collider forces, 9 invariant tests; full PPO loop 112.6 K env-steps/s at 4096 envs (4.1× MuJoCo Warp) |
| Learning parity on Newton | **measured, not met at the training setting**: fixed PPO on Newton (4 it. / 1.25 ms) reaches +7.8 return at iteration 1000 vs MuJoCo Warp's +32.6 with the same learner; survival learned equally fast, velocity tracking much slower; 4.2× throughput does not compensate | MuJoCo Warp stays the default. Queued: Newton at 4 it. / 0.625 ms (the PhysX-closest setting) for 1000 iterations, and the fork build (angle wrap fixed, no clamp); the fork's drift and drive fixes are the likely levers |
| `ActuatorPD` is MetalSim code, not Newton's | **superseded on the fork** by the solver's own PD drive (equal static accuracy, 3σ 1 vs 17 blow-ups); the builder's isotropic armature and the solver's `joint_armature_inertia` must not both be applied | switch NewtonSim to the fork drive once its throughput is measured |
| 3σ stress check fails at the fast Newton settings (37 of 1024 worlds in 400 steps at 4 it./1.25 ms) | open, medium; cause partly found | most blow-ups went through the ±π angle-wrap defect (a knee 0.33 rad past its limit read −3.41 rad and hit 878 rad/s); with the limit clamp 4–7 of 256 remain, all robots already fallen and moving at 20–48 m/s: a general limit of XPBD at few iterations. Sweep of iterations × step × relaxation × implicit leg stiffness queued (`newton_stability_sweep.py`) |
| Newton monitor coverage: penetration / energy / overflow checks have no source fields | open, low | derive from contact depth and body energy |
| Newton rough terrain: no speed advantage (heightfield collision costs ~6× Newton's flat physics; rough full PPO loop 26.1 K vs MuJoCo Warp's 24.8 K env-steps/s) | **new, open** | profile Newton's heightfield narrow phase; a plane-per-cell contact like the MuJoCo Warp patch, or a coarser broad phase |
| Newton rough terrain | **closed** (c65aa09) | Isaac's generated terrain as a Newton heightfield with MuJoCo's geometry; 60 dropped spheres rest on MuJoCo's surface to 0.1 mm median; scanner torso pose matches MuJoCo; throughput in the queued benchmark |
| Newton's joint constraints do not fully converge at the fast settings (feet vs joint-angle kinematics at the 4 it./1.25 ms training setting: 15 mm median / 55 mm p99; off-axis joint rotation 2.9 mrad median / 31 mrad p99, i.e. below the ±10 mrad observation noise at the median but not at the tail) | open, medium; quantified (9842062) | observations already come from body state so the policy never sees the anchor gap; the cheapest remedy is a shorter step, not more iterations: 4 it. at 0.625 ms drifts half as much as 8 it. at 1.25 ms at equal solver work (3.4 vs 6.1 mm median) |
| XPBD mishandles revolute joints whose angle can cross ±π (the G1's elbow-pitch limit is 3.42 rad; a crossing read as a ~2π violation fired a huge impulse: passive energy ×925 in 2 s; also behind most 3σ blow-ups) | **fixed at source** in the Newton fork (https://github.com/pulipakaa24/newton branch `metalsim`, f844a4e6; MetalSim 929a278): one-DoF joints measure their angle within π of a stateless reference (middle of the limit range, else the drive target); 4 new solver tests fail upstream and pass on the fork; third upstream draft ready | measured on CPU: reproducer peak joint speed 746,531 → 3.0 rad/s; G1 passive energy with the original 3.42 rad limit ×925 → ×0.60; 3σ blow-ups 36 → 11 of 1024 (remainder are fallen robots at ~900 rad/s). The limit clamp in `newton_backend.py` becomes unnecessary once the fork is adopted (`limit_margin=None`) |
| Drift remedies evaluated | measured (fork item 4): 4 it. / 1.25 ms 14.8 / 51.8 mm (p50 / p99); **4 it. / 0.625 ms 3.2 / 17.7 mm at 47.2 K env-steps/s on the fork**; joint colouring 0.8: 3.4 / 20.1 mm but only 39.5 K and worse 3σ (30 blown vs 4); +4 joint-only passes 5.7 / 26.3; projection and Featherstone unusable | conclusion: halving the substep is the remedy; the fork drive gives 12.7 / 39.9 mm at 1.25 ms on its own |
| Newton fidelity against PhysX | **partly met**: open-loop protocol rows comparable to MuJoCo Warp (PARITY §1.7); transfer of Isaac's checkpoints over-travels by 37–48 % at 4 it. / 1.25 ms and 20–35 % at 0.625 ms with ActuatorPD, **14–23 % at 0.625 ms with the fork's in-solver drive** (MuJoCo Warp within 5 % of Isaac); every robot stays upright | the long stride is unexplained (push-off diagnostic queued); the fork's 0.625 ms learning run with the in-solver drive decides whether Newton can be a fidelity-grade option |
| Featherstone (reduced coordinates, would remove the drift by construction) | open, not viable as measured | finite only at 0.3125 ms with 12 cm contact sinking, no contact-force reporting, ~1 env-step/s on Metal |
| Newton contact forces approximate | **closed on the fork** (9f626ce2): `SolverXPBD(body_contact_forces=True)` gives exact per-body force vectors (total and normal-only) from the iteration impulses: within 0.02 % / 0.14 % of m·g on a box stack, feet sum 1.0006× weight at rest; maps to `ContactSensor.net_forces_w` | wire into the Newton path's contact terms |
| Newton over-travel is not the joint drift: with Isaac's checkpoint 1000 Newton's strides are ~75 % longer at similar cadence (0.140 vs 0.079 m; base speed 0.575 vs 0.344 m/s) and halving the substep cuts the foot-vs-kinematics gap 4× without changing the stride; joint colouring alone leaves ~40 % over-travel; friction (μ 1.0) and leg damping ruled out | **new, open, unexplained** | candidates untested: the contact model during push-off, swing-leg behaviour; diagnostic queued (push-off ground-reaction impulse per step on each engine) |
| Newton fork, final state (https://github.com/pulipakaa24/newton branch `metalsim` at 90e23324; 13 new solver tests fail upstream, pass on the fork) | **six solver defects fixed at source**: angle wrap (G1 passive energy ×925 → ×0.60 with the original limits), relaxation math (1.36 / 0.78 → 1.001 / 0.999 of exact, new defaults 0.5/0.4), in-solver PD drive (stiffness 199.6–200.3 Nm/rad at 1–16 it.; hand/arm/leg step responses 1.03 / 1.07 / 1.01 of MuJoCo C; 3σ blow-ups 0 vs 17 with ActuatorPD), exact per-body contact forces (0.02 % of weight), rest height explained (anchor separation under load, 0.6–0.8 mm with converged joints), drift remedies measured | costs: the in-solver drive runs at 98.6 K vs 141 K env-steps/s (4 it. / 1.25 ms, physics-only) and 49.1 K vs 71 K at 0.625 ms, and its 1-iteration statics are worse (arms 0.034 rad; 0.008 at 4 it.). Still open on the fork: ankles read low against MuJoCo under every drive (step-response ratio 0.67–0.83, not drive-related), an ultralight three-link chain alternates ±33 % without colouring, hinge limit ranges ≥ 2π unhandled, nothing proposed upstream yet |
| Newton API churn (alpha) | low | pin the version (1.7.0.dev) |
| Importer copies the USD's gravity 0 | low | set gravity explicitly (done in `g1_builder`) |

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
(XPBD contacts and limits are hard), and any further optimization of the MuJoCo Warp step (the 23 %
reset/obs overhead is mostly task kernels and stays, but the physics share changes).

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

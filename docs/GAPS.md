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
| Feet-slide reward term used MuJoCo `cvel` at the subtree COM instead of the foot body's own velocity (Isaac's `body_lin_vel_w`) | **closed** (commit af3cd6b) | per-foot velocity kernel on both engines, tested against `mj_objectVelocity` to 2e-3 m/s; the old term overstated sliding ~2.5× under random actions (−0.30 vs −0.12 per step) — every training run so far, including the reference, used the old term |
| Contact model: soft contacts and soft joint limits (5–7 cm impact penetration, 0.17 rad past limits) | measured against PhysX on the L4 (PARITY §1.7): the 1 m drop lands in the same state (root z RMSE 3.5 cm, joints within 0.06 rad) but MetalSim's contact-force peaks are 3.4–5.7× Isaac's at equal mean force; Newton XPBD gives 0.16–0.69 cm impact penetration (§2.1) | subsumed by the Newton path; on MuJoCo, τ 5 ms + impedance 0.99 + speculative contact is the tunable |
| Learning parity on G1 | **closed for the learner** (fixed PPO: return +32.8, episode length 995 at iteration 1000 on MuJoCo Warp, tracking rsl_rl's +35.5 within a few units; Isaac +27.3 under its own weights). Root cause: the real rsl_rl on our physics learns like Isaac (full episodes by it 200, tracking 0.91 vs 0.92 at it 1000; return +39.6 vs Isaac's +27.2 under our weights); our PPO had three rollout bugs (repeated exploration noise every rollout, phantom action at rollout boundaries, time-outs treated as falls), fixed in f0f2115 (PARITY §1.5) | demonstrate the fixed PPO with a 1000-iteration run; ablate the three fixes; explain rsl_rl's ~50-iteration later takeoff |
| G1 task carries the rough task's reward weights/ranges while Isaac's reference trained the flat config (7 weight/range differences, termination over 1 vs 3 substeps) | **new, open** | port `G1FlatEnvCfg`'s weights as a `terrain="flat"` reward set, re-run the per-term comparison; the rsl_rl flat-weights diagnostic run is prepared on branch `exp/g1-isaac-flat-weights` |
| Rendering vs Isaac RTX; denoiser; MetalFX; MaterialX | measured (PARITY §1.7): robot-only, states agreeing, tier 2 vs RTX 13.7 dB PSNR / 0.48 SSIM (tier 0: 11.5 / 0.35); silhouettes IoU 0.83, robot depth 2.6 cm RMSE; Isaac's RT and PT frames equally far from ours, so the gap is materials/lights (darker plates, hard sun shadow), not noise | re-record with a grey ground (stage 6, queued) for whole-frame rows; then material/light model work: area sun, OmniPBR parameter audit, denoiser |
| Contact sensing (touch sites, no force history) | open | contact-force reduction per body from the contact buffer |
| Sensors: beam divergence, multi-return, radar | open; lidar-based RL now demonstrated (`metalsim.learn.lidar_nav`, 21.2 K env-steps/s, time-to-goal 218 → 69 steps) | – |
| Exact terrain heights | re-implemented generator | port Isaac's generator functions |
| Throughput | MuJoCo Warp at the 2.5 ms training setting: 27.7 K full-PPO env-steps/s (0.34× the 4090's published 82 K); Newton XPBD: 112.6 K (1.4×), PARITY §1.4 | subsumed by the Newton path |
| Camera-RL throughput | 7.5K env-steps/s incl. training vs Isaac's 32K (4090); learns | CNN update on MPS is 76 % of the time |
| Deformables on Metal; upstreaming the forks | open | – |

## Opened by the Newton XPBD evaluation (status 2026-09-24, measured unless noted)

| issue | status | evidence / next step |
|---|---|---|
| Joint drives not engaging under XPBD | **closed** | three causes: `State.joint_q` never written by XPBD (use `eval_ik`), targets indexed by DOF instead of coordinate layout (shifted every G1 target by one joint), and Newton's default joint relaxation 0.7/0.4 transmitting torque wrongly (+43 % / −18 % on a pendulum; 0.4/0.4 exact). `scripts/diagnostics/newton_xpbd_drives.py` |
| XPBD's compliance drive is not a PD with stiffness `ke` (72–1240 Nm/rad for `ke` 200 over 1–16 it.) | **closed by replacement** | Isaac's PD law computed each substep and applied through `Control.joint_f` (`ActuatorPD`); ≤ 0.001 rad from the exact static equilibrium at 4 it. Upstream issue to raise |
| No effort limits, armature, joint friction, velocity limits in XPBD | **closed for the G1** | `ActuatorPD` clips to the effort limit (only the 20 Nm ankles saturate, 0.5–1.5 % of landing steps) and adds armature isotropically to the child inertia (required: NaN without it; axis-only armature invalidates 32 links). Joint friction / velocity limits not needed by the G1 task |
| GJK/MPR narrow phase needs fixed-size arrays; Metal codegen lacked them | **closed** | Warp fork 786cdae: 20/20 fixed-array tests on Metal incl. graph capture; mesh = primitive contacts to 0.001 mm; G1 on its own convex meshes, < 3 % throughput cost |
| Stability / cost of stiff drives under XPBD | **closed** | tracks MuJoCo C within 0.022 rad (4 it., 1.25 ms) or 0.044 rad (8 it., 2.5 ms) at 4.0–4.5× MuJoCo Warp's physics rate at 4096 envs |
| Newton's default relaxation and biased drive | **drafted for upstream** (9e33cfe) | `scripts/diagnostics/newton_upstream/`: two issues with CPU-only reproducers (pendulum responds to torque at 1.43× and to gravity at 0.815× under the 0.7/0.4 defaults; the compliance drive's stiffness is 72–1240 Nm/rad for `ke` 200 over 1–16 it.); ready to file |
| Newton rests the G1 ~1 cm lower than MuJoCo (0.69–0.70 vs 0.71 m) | open | cause not found |
| VBD solver fails to compile on Metal | open, low | not needed for rigid robots |
| State layout: maximal coordinates, no MuJoCo sensors, approximate contact forces | **closed** | `G1VelocityTask(engine="newton")`: same kernels, body state vs MuJoCo to 1e-5, foot/torso contact from collider forces, 9 invariant tests; full PPO loop 112.6 K env-steps/s at 4096 envs (4.1× MuJoCo Warp) |
| Learning parity on Newton | **not yet measured with the fixed learner** | the only Newton training run (1500 it.) used the pre-fix PPO and matched MuJoCo Warp's pre-fix curve, which showed the Isaac gap was not an engine effect; the fixed-PPO run on Newton (1000 it., same config/seed, with the angle-wrap clamp) is queued and is one of three results required before Newton becomes the default |
| `ActuatorPD` is MetalSim code, not Newton's; damping capped at one step's removal on very light links | open, low | document; revisit if hand joints misbehave |
| 3σ stress check fails at the fast Newton settings (37 of 1024 worlds in 400 steps at 4 it./1.25 ms) | open, medium; cause partly found | most blow-ups went through the ±π angle-wrap defect (a knee 0.33 rad past its limit read −3.41 rad and hit 878 rad/s); with the limit clamp 4–7 of 256 remain, all robots already fallen and moving at 20–48 m/s: a general limit of XPBD at few iterations. Sweep of iterations × step × relaxation × implicit leg stiffness queued (`newton_stability_sweep.py`) |
| Newton monitor coverage: penetration / energy / overflow checks have no source fields | open, low | derive from contact depth and body energy |
| Newton rough terrain | **closed** (c65aa09) | Isaac's generated terrain as a Newton heightfield with MuJoCo's geometry; 60 dropped spheres rest on MuJoCo's surface to 0.1 mm median; scanner torso pose matches MuJoCo; throughput in the queued benchmark |
| Newton's joint constraints do not fully converge at the fast settings (feet vs joint-angle kinematics at the 4 it./1.25 ms training setting: 15 mm median / 55 mm p99; off-axis joint rotation 2.9 mrad median / 31 mrad p99, i.e. below the ±10 mrad observation noise at the median but not at the tail) | open, medium; quantified (9842062) | observations already come from body state so the policy never sees the anchor gap; the cheapest remedy is a shorter step, not more iterations: 4 it. at 0.625 ms drifts half as much as 8 it. at 1.25 ms at equal solver work (3.4 vs 6.1 mm median) |
| XPBD mishandles revolute joints whose angle can cross ±π (the G1's elbow-pitch limit is 3.42 rad; a crossing reads as a ~2π violation and the solver fires a huge impulse: passive kinetic energy ×925 in 2 s, one elbow at 1005 rad/s; the same path caused the 3σ blow-ups via limit overshoot) | **closed by workaround** (839b334, 67fb536); third upstream draft (9c9d15d) | revolute limits kept inside ±(π − 0.15) in the Newton model (elbow 3.42 → 2.99 rad; MuJoCo path unchanged); passive energy now only decays; optional joint re-centring moves the wrap point away from the limits. The first Newton training run predates the clamp |
| Drift remedies evaluated | measured on CPU (22aee1f) | joint projection removes the drift but pushes contact-held feet 67 cm through the ground (unusable); Featherstone is finite only at 0.3125 ms with 12 cm contact sinking and ~1 env-step/s (not viable); **4 it. at 0.625 ms** is the real remedy (3.3 mm median drift, 0.06 cm drop penetration), its 4096-env throughput queued |
| **Newton fidelity against PhysX: unmeasured** | open, required for the default switch | MuJoCo Warp went through the L4 fidelity protocol (PARITY §1.7) and the cross-simulator transfer test; Newton has been checked only against MuJoCo C (hold within 0.02–0.04 rad, drop penetration 0.2–0.7 cm). Queued on Newton: the same A_hold / B_random / C_drop protocol rows against the Isaac recording, and the transfer of Isaac's checkpoints (one preliminary checkpoint walked 4.47 m on Newton vs 3.19 m in Isaac and 3.01 m on MuJoCo Warp, on a busy GPU: to be re-measured) |
| Featherstone (reduced coordinates, would remove the drift by construction) | open, not viable as measured | finite only at 0.3125 ms with 12 cm contact sinking, no contact-force reporting, ~1 env-step/s on Metal |
| Newton API churn (alpha) | low | pin the version (1.7.0.dev) |
| Importer copies the USD's gravity 0 | low | set gravity explicitly (done in `g1_builder`) |

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

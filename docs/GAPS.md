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
| Contact model: soft contacts and soft joint limits (5–7 cm impact penetration, 0.17 rad past limits) | tunable to 0.85 cm / 0.00 cm at rest with τ 5 ms + impedance 0.99, 0.00 cm with speculative contact, at higher impact peaks | apply and quantify with the drop-test row of the fidelity protocol against the L4 recordings |
| Learning parity on G1 | stands but does not track; return still negative | compare per-term rewards with Isaac's own rsl_rl log (recording on the VM) |
| Rendering vs Isaac RTX; denoiser; MetalFX; MaterialX | fidelity recordings in progress on the VM; no denoiser/MetalFX/MaterialX | run `metalsim.parity.compare` when the recordings land |
| Contact sensing (touch sites, no force history) | open | contact-force reduction per body from the contact buffer |
| Sensors: beam divergence, multi-return, radar | open | – |
| Exact terrain heights | re-implemented generator | port Isaac's generator functions |
| Throughput | 0.59× a 4090 raw (2.6× per TFLOPS); 1.4× the L4 on G1 rough | reset/obs overhead 23 % of the step |
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
| Newton's default relaxation and biased drive | open, upstream | to be reported to newton-physics/newton |
| Newton rests the G1 ~1 cm lower than MuJoCo (0.69–0.70 vs 0.71 m) | open | cause not found |
| VBD solver fails to compile on Metal | open, low | not needed for rigid robots |
| Featherstone on the G1: NaN at 2.5 ms, no result at 1.25 ms in 400 s | open, low | XPBD is the path |
| State layout: maximal coordinates, no MuJoCo sensors, approximate contact forces | **in progress** | engine switch in the G1 task (obs/reward/termination from body state + `eval_ik`), then a PPO run against `runs/g1_flat_ppo_1500_dt25_fixed.log` |
| Learning parity on Newton | open | no policy trained on Newton yet (run in progress) |
| `ActuatorPD` is MetalSim code, not Newton's; damping capped at one step's removal on very light links | open, low | document; revisit if hand joints misbehave |
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

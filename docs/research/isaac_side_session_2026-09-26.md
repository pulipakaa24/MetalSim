# Isaac-side recording session, 2026-09-26: PhysX penetration, Kit dome azimuth, PhysX four-bar

One VM session on `aditya-isaac` (GCP project `mcmc-193822`, us-west1-a, NVIDIA L4, driver 580). The VM was
started 19:13:31 PDT and stopped 19:42:42 PDT: **29 min of VM time**, then confirmed `TERMINATED`. Nothing ran on
the local GPU. The analysis in this note is CPU numpy on the fetched artifacts
(`metalsim/parity/isaac_side/analyze_session_0926.py`). "Measured" = recorded in this session on the L4. Scope is
stated on each number. One seed, one asset per protocol.

| # | script (committed) | environment | artifacts |
|---|---|---|---|
| 1 | `metalsim/parity/isaac_side/record_penetration.py` | Isaac Sim 5.1 + Isaac Lab 2.3.2 (`~/env_isaaclab`), GPU and CPU PhysX | `runs/parity3/isaac/penetration_il2/`, `penetration_il2_cpu/` |
| 1b | `metalsim/parity/isaac_side/isaaclab3/record_penetration.py` | Isaac Sim 6.1 + Isaac Lab 3.0.0-EA (`~/env_isaaclab3`), `physics=isaacsim_physx` | `runs/parity3/isaac/penetration_il3/` |
| 2 | `metalsim/parity/isaac_side/record_dome_azimuth.py` | Kit 107.3.3 (Isaac Sim 5.1) and Kit 110.3.0 (Isaac Sim 6.1), RTX path tracing | `runs/parity3/isaac/dome_il2/`, `dome_il3/` (`dome_il3_try1/`: first 6.1 attempt, see §2) |
| 3 | `metalsim/parity/isaac_side/record_fourbar.py` | Isaac Sim 5.1 + Isaac Lab 2.3.2, PhysX GPU and CPU | `runs/parity3/isaac/fourbar_il2/<run>/` (+ `summary.jsonl`) |

Each Isaac process ran on its own (Kit's single-instance lock). The VM logs are copied next to each artifact.

## 1. PhysX per-contact penetration during the G1 1 m drop

**Protocol.** This is `record_g1.py`'s C_drop: Isaac-Velocity-Flat-G1(-v0), one env, no reset randomization, zero
actions, root placed at z = 1.0 m, one settle step, then 150 control steps × decimation 4, so 601 physics steps at
5 ms. The recorder unrolls `env.step` (`action_manager.process_action` once per control step, then
`apply_action`, `scene.write_data_to_sim`, `sim.step` and `scene.update` per physics step) so it can read contacts
after **every physics step**. The recording reproduces the earlier ones: final root z is 0.053803 m (5.1 GPU),
0.053800 m (5.1 CPU) and 0.053800 m (6.1), against 0.0538 m in the earlier C_drop recordings.

**APIs used (cited from the installed sources on the VM):**
- `omni.physics.tensors` `RigidContactView.get_contact_data(dt)` returns `(forces, points, normals, separations,
  counts, start_indices)`. Its docstring: "Gets the detailed contact data between sensors and filter prims per patch.
  This includes the contact forces, contact points, contact normals, and separation distances". Source:
  `omni.physics.tensors-107.3.26/omni/physics/tensors/impl/api.py:5877`. The view is created with
  `SimulationManager.get_physics_sim_view().create_rigid_contact_view(sensor_paths, filter_patterns=[[ground
  collider]]*n, max_contact_data_count=16*n)`. The sensors are the 44 robot bodies that carry
  `PhysxContactReportAPI`, which is the Isaac Lab ContactSensor's body set. The filter is
  `/World/ground/terrain/GroundPlane/CollisionPlane`. Isaac Lab's `ContactSensor` calls the same function when
  `track_contact_points=True` but unpacks only the points (`isaaclab/sensors/contact_sensor/contact_sensor.py:399`).
  It does not expose separation, and `track_pose` is the body pose, not contact data. On 6.1 the view comes from
  `isaaclab_physx.physics.PhysxManager.get_physics_sim_view()`.
- `omni.physx` `get_physx_simulation_interface().get_full_contact_report()` returns headers plus `ContactData` with
  `.separation` and `.impulse` ("Contact separation value", `omni/physx/bindings/_physx.pyi:546`). Under the GPU
  pipeline it returned **0 rows**. On CPU PhysX it returned 5,730 rows, and its per-step minimum separation is
  **identical (to 1e-7 m) to the tensor API's**. That cross-validates the tensor numbers.
- PhysX sign convention: separation < 0 is penetration. Contacts appear once the shapes are within contactOffset, so
  positive separations show up before touchdown. The foot colliders (`*_ankle_roll_link/Cube`, a Mesh) author
  `restOffset = contactOffset = -inf`, which means PhysX's automatic defaults (rest offset 0). Contacts are generated
  before the solve, so step k's separation is the pose at the end of step k−1.

**Results (measured, G1 C_drop, one run each):**

| run | first contact | peak penetration, feet (left / right) | at | settled (last 0.5 s, per-step min) | torso peak |
|---|---|---|---|---|---|
| Isaac Sim 5.1 PhysX, GPU (the reference config) | step 47 (0.235 s), +13.3 mm | **0.512 / 0.431 mm** | step 49 (0.245 s) | −0.013 mm (feet −0.006 / −0.007) | 0.104 mm at 1.40 s |
| Isaac Sim 5.1 PhysX, CPU | step 47 | 0.404 / 0.494 mm | step 49 | −0.005 mm | 0.114 mm |
| Isaac Sim 6.1 PhysX (Isaac Lab 3.0-EA), GPU | step 47, +18.3 mm | **2.077 / 2.025 mm** | step 49 | −0.008 mm | 0.033 mm |

Time profile (5.1 GPU, minimum separation over all contacts per step, eight contacts, four per foot box):

| step | 47 | 48 | 49 | 50 | 51 | 52 | 55 | 60 | 61–100 mean | 401–601 |
|---|---|---|---|---|---|---|---|---|---|---|
| sep [mm] | +13.3 | +1.82 | **−0.51** | −0.48 | −0.26 | −0.23 | −0.13 | −0.08 | −0.029 | −0.013 |
| Σ normal force [N] | 0 | 6497 | 1376 | 1472 | 1278 | 1134 | 903 | 659 | 325 | 316 |

On 6.1 the sequence is +18.3, +6.86, **−2.08**, −0.82, −0.28 mm. PhysX already applies 6.5 kN at step 48, while the
feet are still 1.8 mm (5.1) or 6.9 mm (6.1) above the ground: that is the speculative contact inside contactOffset.
The robot then falls over. The torso touches down at 1.40 s, and the pelvis rests at 5.38 cm.

**Summary.** PhysX's peak foot penetration on landing from the 1 m drop is **0.51 mm on Isaac Sim 5.1 GPU PhysX**
(0.49 mm on CPU PhysX, 2.08 mm on Isaac Sim 6.1 PhysX). It occurs one step after first contact and decays below
0.1 mm within 55 ms. At rest it is 0.005–0.013 mm, that is resting separation ≈ restOffset 0. MetalSim's
1.55–1.64 cm peak (GAPS, our side, not re-measured here) is therefore **~30× PhysX 5.1's** (~8× 6.1's). The
reference is a sub-millimetre, near-rigid contact, not a soft one. Scope: one drop, one asset (Isaac's G1 minimal
USD, box foot colliders), 5 ms step, Isaac's G1 solver settings (8/4 iterations, TGS). The 5.1 → 6.1 difference
(0.5 → 2.1 mm) is measured but not explained. The contact offset differs: the first report comes at 13 mm vs
18 mm.

## 2. Kit/RTX DomeLight latlong mapping (azimuth verified)

**Recipe** (`record_dome_azimuth.py`, all generated on the VM):
- A 1024×512 RGBE `.hdr` latlong map, black except three 24-px squares of radiance 200: RED at (u, v) = (0.10, 0.25),
  GREEN at (0.35, 0.25), BLUE at (0.60, 0.10). Here u = column/width (0 = left of the file) and v = row/height
  (0 = top row). With three non-collinear markers, the pole, the azimuth zero and the handedness are all determined.
- A `UsdLux.DomeLight` (`texture:format = latlong`, intensity 1, exposure 0), a white diffuse ground (z = 0) and a
  white diffuse sphere (r 0.25 m at (0, 0, 0.25)) on a z-up stage.
- Six 90° cube-face cameras (256², at (5, 5, 2)) see the dome background directly. Every marker pixel is
  back-projected through the authored camera pose, weighted by pixel solid angle.
- A top-down camera at (0, 0, 6) looks along −z, with image up = +y and image right = +x. Its frame shows the
  sphere's coloured shadows as a lighting check.
- Path tracing (`/rtx/rendermode = PathTracing`, 64 spp) through `omni.replicator.core` render products and the
  `rgb` annotator.
- Dome variants: un-rotated, `rotateZ +90°`, `rotateX +90°`, then un-rotated again. The frames are PNGs in the
  artifact directories. Recipe and results are in `dome_results.json`.

**Measured** (Kit 107.3.3 / Isaac Sim 5.1; Kit 110.3.0 / Isaac Sim 6.1 gives the same directions within 0.6°):

| marker (u, v) | elevation measured / expected for pole +z | atan2(x, −y) measured | our convention predicts | Kit − ours |
|---|---|---|---|---|
| GREEN (0.35, 0.25) | 44.95° / 45° | −35.85° | +54.0° | **−89.85°** |
| BLUE (0.60, 0.10) | 71.88° / 72° | −125.77° | −36.0° | **−89.77°** |
| RED (0.10, 0.25) | 44.7° / 45° | 50.26° | 144.0° | −93.7° (the marker straddles two cube faces; the 6.1 run gives −93.1°) |
| with `rotateZ +90°`: GREEN / BLUE | 44.93° / 71.84° | 53.68° / −35.87° | 144° / 54° | −90.32° / −89.87° |

- **Pole axis: +z.** The top row of the file is +z. The elevations match v → 90° − 180°·v within 0.3° for fully
  visible markers.
- **Handedness: same as ours.** Increasing u turns the azimuth clockwise seen from above, at the same rate
  (GREEN → BLUE: Δu = 0.25 gives −89.9°). There is no mirroring.
- **Azimuth zero: off by −90° from our adopted convention.** Kit's mapping, measured, is
  **u = 0.5 − (atan2(x, −y) + π/2 − yaw)/2π**, equivalently **u = −(atan2(y, x) − yaw)/2π (mod 1)**, with
  v = acos(z)/π. So u = 0 (the left edge of the file) faces **+x**, u = 0.25 faces −y, the image centre u = 0.5
  faces **−x**, and u = 0.75 faces +y. Our adopted convention puts the centre at −y (rendering_vs_rtx §6.1). It is
  Kit's mapping rotated by +90° about z, so MetalSim reproduces Kit with `set_environment(yaw=-π/2)` (−90°), or
  with the constant folded into the formula.
- **`rotateZ` on the dome prim** rotates the map by +yaw about +z, in the sense of our formula's `yaw` term
  (measured +90.0° ± 0.3°).
- **`rotateX +90°`** is applied as an ordinary rigid rotation. BLUE moved from (−0.252, 0.182, 0.950) to
  (−0.252, −0.950, 0.182) = R_x(90°)·d exactly. So Kit honours authored dome rotations, and the texture is
  *already* z-up. Authoring USD's `OrientToStageUpAxis` +90° x-rotation for a z-up stage (as the OpenUSD +Y-pole
  convention expects) tips the map over in Kit. This agrees with NVIDIA's `usd-exchange-samples` note that
  "Kit/RTX treats the DomeLight environment as Z-up". Importers should not add the rotation for Kit-authored
  stages.
- **Lighting check (top-down PT frame `none_top.png`).** The magenta shadow (GREEN blocked) lies toward +x+y. The
  cyan shadow (RED blocked) lies toward −x+y. The short yellow shadow (BLUE blocked, high elevation) lies toward
  +x−y. Each is opposite the marker direction measured from the background, so the illumination uses the same
  mapping as the visible background.
- The first 6.1 attempt (`dome_il3_try1/`) returned black frames for the first variant: the texture was not yet
  loaded after a 465 s first-launch shader compile. The script now renders three warm-up frames and repeats the
  un-rotated variant at the end. The repeat matches the first variant exactly on 5.1 and on 6.1.

## 3. PhysX four-bar (closed loop via `excludeFromArticulation`)

**Build** (`record_fourbar.py`). The mechanism is `mechanisms.fourbar_geometry()` from
`scripts/diagnostics/closed_loops/`: the same pivots, rod masses, COMs and inertias (uniform rods, r = 1 cm), and
the same start (crank −20°, at rest, undamped, gravity −z, no colliders). It is authored as USD and spawned with
Isaac Lab's `UsdFileCfg`:
- `world –FixedJoint– ground` is the articulation root link (fixed base).
- `ground –revolute crank– crank –revolute coupler– coupler` and `ground –revolute rocker– rocker` form the
  articulation tree.
- The loop is `coupler –revolute loop0– rocker` at C, with `physics:excludeFromArticulation = true` (UsdPhysics).
  PhysX 5 closes loops "by adding rigid-body Joints between articulation links" (Articulations docs), and this
  joint is solved as a maximal-coordinate constraint.
- Joint frames are world-aligned at the start pose. Link damping, sleep and stabilization thresholds are 0, and the
  joints have no friction.
- `ImplicitActuatorCfg` has stiffness = damping = 0, so the joints are passive.

The runs cover 5 s at 2.5 ms and 5 ms through Isaac Lab's `SimulationContext` + `Articulation`, reading
`body_pos_w`, `body_quat_w`, `joint_pos` and `joint_vel` per step. The loop-closure gap is |coupler tip − rocker
tip| from the body poses. The in-tree joints are exact by construction; the recorded `max |y|` is 2e-8 m. Articulation
iterations are 8 position / 4 velocity (Isaac Lab's G1 setting) with TGS, plus variants. The comparison is against
the exact planar DAE reference `runs/closed_loops/ref/fourbar_none.npz`, using the same metric as
`runs/closed_loops/c_summary.md` (max body-angle error over [0, T]). Energy is checked two independent ways
(central differences of the body angles, and exact rates from the recorded joint velocities), and they agree within
1 %.

**Measured** (one initial state; `runs/parity3/isaac/fourbar_il2/summary.jsonl`):

| run | dt | max angle err 1 s / 2 s / 5 s [rad] | energy drift over 5 s | closure gap max / mean [mm] |
|---|---|---|---|---|
| PhysX GPU, TGS 8/4 | 2.5 ms | 0.135 / 0.494 / 2.64 | **+9.1 %** (joint-vel estimate +10.0 %) | **0.154 / 0.036** |
| PhysX GPU, TGS 32/4 | 2.5 ms | 0.135 / 0.494 / 2.64 | +9.1 % | 0.143 / 0.035 |
| PhysX GPU, PGS 8/4 | 2.5 ms | 0.135 / 0.494 / 2.64 | +9.1 % | 0.162 / 0.037 |
| PhysX CPU, TGS 8/4 | 2.5 ms | 0.135 / 0.495 / 2.64 | +9.0 % | 0.040 / 0.008 |
| PhysX GPU, TGS 8/4 | 5 ms | 0.270 / 1.02 / 3.39 | **+20.0 %** | **0.747 / 0.149** |
| PhysX GPU, TGS 32/4 | 5 ms | 0.270 / 1.02 / 3.39 | +20.0 % | 0.699 / 0.143 |
| PhysX CPU, TGS 8/4 | 5 ms | 0.269 / 1.01 / 3.38 | +19.7 % | 0.196 / 0.031 |
| MuJoCo C, solref 0.02 (for context, `c_summary.md`) | 2.5 ms | 0.020–0.032 / 0.10–0.20 / 0.56–1.08 | −1.1 to −2.5 % | 1.07–1.16 / 0.45–0.47 |

The energy gain is monotonic: 1.146 → 1.250 J at 2.5 ms. It is first order in dt (it doubles at 5 ms) and does not
depend on iteration count or on the solver type. Wall time was 5.4 s per 2,000 steps (GPU) and 2.2 s (CPU). A
diagnostic run without the loop joint (`--no_loop`: an open crank–coupler double pendulum plus a rocker pendulum;
`*_noloop/`) *lost* energy (1.15 → −0.17 J over 5 s at 2.5 ms). So PhysX's articulation integration is not
energy-conserving on this chaotic open chain either, and the loop constraint pushes the drift the other way. This
was measured but not investigated further. It is not a reference trajectory.

**Summary.** PhysX closes the four-bar loop tightly: **0.15 mm max / 0.04 mm mean at 2.5 ms, 0.75 / 0.15 mm at
5 ms** (GPU; CPU PhysX 4× tighter). That is 3–8× tighter than MuJoCo's soft `connect` (0.36–1.16 mm at 2.5 ms)
and close to Kamino at α 0.5 (0.044 mm). But PhysX **gains energy** on the undamped free swing (+9 % in 5 s at
2.5 ms, +20 % at 5 ms, where MuJoCo loses 1–5 %), and it follows the exact trajectory less closely than MuJoCo
(0.135 rad by 1 s vs 0.02–0.08 rad). Iterations (8 → 32) and TGS/PGS change nothing measurable. So the loop
reference from Isaac is "hard closure, energy-injecting at robot time steps". Matching PhysX here means matching
closure; the trajectory itself is a poorer reference than the exact DAE. Scope: one passive swing, one mechanism,
Isaac Sim 5.1 PhysX only (not run on 6.1).

## 4. Open items this session leaves

- Penetration: the local MetalSim comparison of the same per-step profile (peak/settle definition as above) is the
  next step. The 5.1 → 6.1 peak difference (0.5 → 2.1 mm) is unexplained.
- Dome: apply `yaw = −90°` (or fold it into the formula) in MetalSim's USD DomeLight import, and do not add a
  +90° x-rotation for Kit stages. PARITY §1.7's caveat and the GAPS row can then be closed (owner: rendering).
- Four-bar: the PhysX energy gain versus the loop-joint formulation, and the four-bar on Isaac Sim 6.1 PhysX, were
  not recorded.

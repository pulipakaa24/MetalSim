# Newton VBD and Newton's coupled rigid/deformable solvers on Metal (the Isaac Lab 3.0 deformable path)

2026-09-25. Isaac Lab 3.0 EA (Isaac Sim 6.1) runs deformables on the Newton backend with Newton's VBD solver,
coupled to MuJoCo Warp rigid bodies through Newton's experimental coupled-solver framework. This note records
(1) what kept VBD from running on the Metal Warp fork and the fix, (2) VBD and the coupled solvers on Metal
against the CPU device, with throughput, and (3) from source, how Isaac Lab 3.0 couples rigid and deformable
bodies under Newton. Companion: `physx_deformables_port_2026-09-25.md` (PhysX-backend formulations).
**Measured** = run here (M4 Max; Metal through `scripts/gpu_run.sh` at kind `low`, `gpu_lock.py status` at the top of
every log); **source** = read in the cited file.

Sources: [N152] `upstream/newton-1.5.2` = newton-physics/newton tag `v1.5.2` (cf5378db, 2026-09-09), the version
Isaac Lab 3.0 EA pins; [NF] `upstream/newton` = MetalSim fork branch `metalsim` (90e23324, 1.7.0.dev); [W]
`upstream/warp-innate` = Warp fork branch `metalsim`; [IL3] isaac-sim/IsaacLab tag `v3.0.0-EA` and `develop`
(24fd3182), read from a sparse clone. Logs and data: `runs/newton_vbd/`. Scripts:
`scripts/diagnostics/newton_vbd/` (`vbd_probe.py`, `coupled_probe.py`, `run_newton_tests.py`,
`profile_launches.py`), environment `scripts/setup_newton152.sh` (`.venv-newton152`).

## 1. Summary

* **Newton VBD runs on Metal.** Newton 1.5.2's SolverVBD (cloth and tetrahedral soft bodies) compiles and runs on
  the Metal Warp fork unchanged. Newton main / the 1.7.0.dev fork failed to build one module,
  `rigid_vbd_kernels`, with a link error: `Undefined symbol(s) for architecture 'air64': 'nextafterf'`. The Metal
  standard library has no `nextafterf`, and clang lowers `__builtin_nextafterf` to a call to it. Newton's
  interval arithmetic for rigid–soft Divide-and-Truncate (`newton/_src/solvers/vbd/interval_arithmetic.py`,
  Newton e42568d9, after 1.5.2) calls it from the non-CUDA branch of a native snippet, and Metal takes that
  branch. **Fixed in the Warp fork** (9050cb54; `patches/warp/0007-Metal-nextafterf-*`, and merged into the fork's `metalsim` as f194006 by c041d10): `metal_crt.h` now has a bit-exact IEEE
  `nextafterf`, computed on bit patterns because Apple GPUs flush float32 subnormals in comparisons, and maps
  `__builtin_nextafterf` onto it. The new test `test_metal.test_nextafterf` matches `numpy.nextafter` bit for bit
  on Metal and on the CPU. `warp.tests.test_metal` passes 16/16. No Newton change was needed.
* The earlier ledger row "VBD fails to compile on Metal" (GAPS, Newton XPBD ledger) is closed for both Newton
  versions. The error was invisible in Warp's exception because the fork's Metal compile-log filter drops lines
  after a warning and keeps only 4,096 bytes. Recompiling the cached `.metal` source through a 30-line Objective-C
  harness (`scripts/diagnostics/newton_vbd/mtlc.m`: `MTLDevice newLibraryWithSource`, the fork's compile options) printed the full log (`runs/newton_vbd/mtlc_fork_rigid.log`).
* **Metal matches the CPU device** to float32 sensitivity. Soft cube, 200 steps: at most 1.3e-4 m. Cloth draped
  over a box edge: at most 0.18 m, against 0.16 m when the CPU is compared with itself from a start perturbed by
  1e-7 m. Median vertex error at step 199: 1.2e-2 m (Metal) vs 1.4e-2 m (perturbed CPU).
* **Newton's test suites** (1.5.2) give the same results on Metal as on the CPU:
  * VBD/cloth/soft body/cloth collision: 89 of 94 pass on both devices. The other 5 error on a missing LFS asset
    (`bunny.usd`). The one extra Metal "failure" is a CUDA-only test (it asserts `is_cuda`) that the runner
    registered on Metal.
  * Coupled-solver and ADMM suites: 117 on the CPU and 118 on Metal (includes `metal_0` tagged variants). Both
    devices have the same single error, `test_compacted_joint_targets_use_local_layout`, which is not
    Metal-specific.
* **Newton's coupled MuJoCo Warp + VBD solvers run on Metal** with Newton 1.5.2 and mujoco-warp 3.11.0: the proxy
  and ADMM examples, graph-captured. Final states match the CPU to 3.7 mm (proxy scene, rigid bodies resting on
  cloth) and 8e-5 m (ADMM). Graph capture needs one eager frame first. SolverVBD sizes its contact buffers on the
  first step, and allocating inside a capture needs a CUDA memory pool, which Metal lacks. Newton's own error
  message prescribes the eager frame.
* **Version constraint.** Newton 1.5.2's `SolverMuJoCo` requires mujoco-warp ~=3.11. Against MetalSim's
  MuJoCo Warp fork (3.14) it fails to build `convert_mjw_contacts_to_newton_kernel`: `contact_force_fn` overload
  mismatch. The Newton fork (1.7.0.dev, which declares ~=3.12) fails the same way. The coupled reference
  therefore runs in `.venv-newton152` with stock mujoco-warp 3.11.0, i.e. without MetalSim's MuJoCo Warp fixes.
* **Throughput** (graph replay, env-steps/s at 1024 / 4096 worlds): VBD cloth (441 particles) 3.6K / 3.7K; VBD
  soft cube (125 nodes) 0.69K / 0.57K. The cube's `solve_elasticity` kernel is compute-bound from ~256 worlds.
  For scale, the PhysX-style PBD cloth prototype gets 57.7K / 41.3K on the same cloth, and MetalSim `XPBDSim`
  320K / 285K (companion doc §6). Isaac Lab publishes no deformable throughput.

## 2. Versions and environments

| | Newton | mujoco / mujoco-warp | Warp | where |
|---|---|---|---|---|
| Isaac Lab 3.0 EA | 1.5.2 (`pyproject.toml:407`, uv override; line 93 is the loose `newton[sim]>=1.2.0`) | ~=3.11 (Newton's `[sim]` extra) | 1.16 | [IL3] |
| Isaac Lab `develop` | 1.6.0 (`pyproject.toml:91`) | | | [IL3] |
| `.venv-newton152` (this work) | 1.5.2, editable `upstream/newton-1.5.2` | 3.11.0 / 3.11.0 (stock) | fork 1.18.0.dev (`upstream/warp-innate` 9050cb54) | `scripts/setup_newton152.sh` |
| `.venv-newtonfork` | 1.7.0.dev fork (or 1.5.2 via `PYTHONPATH`) | 3.14.0 / MetalSim fork 3.14.0 | same | `scripts/setup_newton.sh` |

On particle-only scenes the Newton 1.5.2 and 1.7.0.dev VBD produce **bitwise-identical** CPU trajectories (cloth
and cube, 200 steps). The 1.7.0.dev differences are in the rigid (AVBD) path and in new options
(`rigid_compliant_alm`, rigid–soft Divide-and-Truncate, soft self-contact through `CollisionPipeline`). The
coupled-solver API is unchanged: the `coupled/` diffs are renames and plumbing (subagent diff of both trees).

## 3. VBD on Metal: measurements

Scenes (`vbd_probe.py`):
* **Cloth:** the PhysX protocol's cloth geometry. 21×21 particles of 0.02 kg on a 1 m grid, released flat at
  0.5 m over a static 0.4 m box on a plane. `tri_ke = tri_ka` 1e4, bending `edge_ke` 1e-2,
  `soft_contact_ke` 1e4, μ 0.6.
* **Soft cube:** 0.2 m, 4 cells per side, 125 nodes, E 1e5, ν 0.4, density 1000 (the protocol's material),
  `k_mu`/`k_lambda` from E and ν, dropped from a centre height of 0.5 m.
* Both: 200 Hz steps, 10 substeps × 10 VBD iterations, tile solve on (off made no difference).

| check | cloth | soft cube |
|---|---|---|
| Metal vs CPU, max vertex error at step 10 / 50 / 100 / 199 | 0 / 2.3e-3 / 0.18 / 0.10 m | 7e-15 / 4.3e-6 / 9.6e-6 / 1.3e-4 m |
| CPU vs CPU started 1e-7 m apart (float32 control), same steps | 1.1e-6 / 2.1e-3 / 0.16 / 0.16 m | – |
| final mean height, CPU / Metal | 0.27540 / 0.27545 m | 0.12382 / 0.12382 m |
| Newton 1.5.2 vs 1.7.0.dev (CPU) | identical | identical |
| graph capture on Metal | works (with an even number of substeps per captured step) | works |

The cloth is chaotic where it folds over the box edge. Metal's divergence has the same size and timing as the
float32 control, so it is not a Metal defect. First-step compile on Metal: 13–16 s for the cloth modules
(cached afterwards).

Throughput (graph replay, env-steps/s; one env-step = 5 ms = 10 substeps × 10 iterations; `metal_tp.log`):

| worlds | 1 | 64 | 256 | 1024 | 4096 |
|---|---|---|---|---|---|
| VBD cloth (441 particles) | 38 | 1,792 | 3,487 | 3,580 | 3,669 |
| VBD soft cube (125 nodes) | 16 | 493 | 1,736 | 689 | 571 |

Per-kernel profile (`profile_launches.py`, eager, synchronising after every launch; `metal_profile.log`):
* `solve_elasticity` (40 launches per substep) dominates.
* Cube: 33 ms per substep at 64 worlds, 36 ms at 256, 153 ms at 1024. The kernel is launch/latency-bound up to
  ~256 worlds and compute-bound after, which is why the cube's env-steps/s peak at 256.
* Cloth: 23 → 30 ms from 256 to 1024 worlds.
* On the CPU the cube's cost is linear in worlds (20 ms per world-step at 16–256 worlds).

## 4. Newton's coupled solvers (from source) and on Metal

**Mechanism (source, [N152] `newton/_src/solvers/coupled/`, exported as `newton.solvers.experimental.coupled`)**
* `CouplingInterface` (`interface.py:108`) is a set of hooks: effective mass, gravity, rewinding proxy
  bodies/particles, harvesting proxy wrenches or particle forces, and preparing proxy contacts. MuJoCo, VBD,
  Featherstone, XPBD, SemiImplicit, Kamino and ImplicitMPM implement it.
  * `SolverMuJoCo` overrides only gravity and the (block) effective mass, taken from `body_invweight0`
    (`mujoco/solver_mujoco.py:4139-4208`).
  * `SolverVBD` (docstring `vbd/solver_vbd.py:95-103`: "VBD for particles and Augmented VBD (AVBD) for rigid
    bodies") harvests explicit contact forces rather than momentum changes (`:938`, `:1039`).
* All schemes share one `Model`/`State`. Each entry (solver, owned bodies/particles/joints/shapes, substeps)
  steps a compacted `ModelView` (`solver_coupled.py:326-360`, `model_view.py:60-77`).
* Three schemes:

| scheme | exchange | schedule | direction |
|---|---|---|---|
| `SolverCoupled` (`solver_coupled.py:308`) | none; owned state reconciled | each entry steps independently | none |
| `SolverCoupledProxy` (`solver_coupled_proxy.py:215`) | source bodies appear as proxy bodies in the destination solver, with mass = source effective mass × `mass_scale`; the destination's contact wrenches are harvested and applied to the source on the next pass | per `step()`: `iterations` passes of add lagged feedback → step source → sync proxy poses (`"lagged"`: begin pose + end velocity; `"staggered"`: end pose) → destination collision → step destination → harvest → relax (fixed/Aitken) (`:1316-1569`) | two-way, lagged one pass; at most 2 entries |
| `SolverCoupledADMM` (`solver_coupled_admm.py:424`) | interface constraints: cross-entry joints, body–particle attachments, rigid–rigid / rigid–particle / particle–particle contacts | internal collision, then `iterations` (default 5) of force accumulation → step all entries → dual update (`:2824-2885`) | symmetric two-way |

* A legacy pattern also exists: `SolverVBD(integrate_with_external_rigid_solver=True)` gives one-way coupling
  (the robot is kinematic for VBD); it is used by `examples/cloth/example_cloth_franka.py`.

**Isaac Lab 3.0 (source, [IL3])**
* The coupled Franka soft/cloth/cable lift tasks (`isaaclab_tasks/core/lift/config/franka_soft/*_env_cfg.py`,
  EA `franka_soft_env_cfg.py:143-186`) use `SolverCoupledProxy` through
  `isaaclab_contrib/coupling/coupler.py` (`NewtonCouplerManager`, `:383-400`):
  * rigid entry: `MJWarpSolverCfg(cone="elliptic", integrator="implicitfast")`, owning the whole robot;
  * soft entry: `VBDSolverCfg(iterations=10)` with all particles and static shapes;
  * proxies rigid → soft for the hand and fingers only, `collide_interval=1`, full-surface rigid–soft contact,
    proxy iterations 1;
  * 2 substeps (4 for the cable).
* Isaac Lab's documentation (`docs/source/concepts/coupled_solvers.rst`) calls the proxy "the established path for
  Isaac Lab's coupled MJWarp–VBD tasks" and recommends `mode="lagged"`. ADMM is available through the same manager.
* EA also ships a hand-rolled one-/two-way coupler (`isaaclab_contrib/custom_coupling/coupled_mjwarp_vbd_manager.py`)
  as a separate example task.
* A deformable scene without articulations uses one `SolverVBD` (`isaaclab_newton/physics/vbd_manager.py:62-72`).
* Unclear: runtime behaviour of the Isaac Lab tasks, and whether `develop` still ships `custom_coupling`.

**Consequence for MetalSim's coupling design:** the reference to mirror is a lagged, two-way impulse exchange
between two solvers inside each substep (proxy bodies with the rigid solver's effective mass, contact wrenches
fed back on the next pass). It is not one monolithic solve (MuJoCo Warp flex, today's path) and not PhysX TGS's
per-iteration exchange. Newton's own implementation runs on Metal (below), so the most faithful route is to run it
directly. The price is SolverMuJoCo on stock mujoco-warp 3.11, i.e. without MetalSim's MuJoCo Warp contact and
heightfield fixes, until Newton's `SolverMuJoCo` is ported to the 3.14 API or our fixes are backported to 3.11.

**On Metal (measured, `metal_coupled.log`; Newton 1.5.2 + mujoco-warp 3.11.0; 60 frames at 60 fps; one eager frame
before capture on every device)**

| example | scene | CPU frames/s | Metal frames/s | Metal vs CPU at frame 60 | `test_final` |
|---|---|---|---|---|---|
| `example_mujoco_vbd_coupled_solver` (proxy, the Isaac Lab path) | 6 MuJoCo bodies (boxes + pendulum chain) on a VBD cloth + soft bodies, 1,115 particles, 8 substeps | 2.8 | 10.2 | bodies 3.7 mm max (0.87 mm median); particles 3.7 mm max, 5.6e-5 m median | pass / pass |
| `example_mujoco_vbd_admm_solver` (ADMM) | 3 bodies, 144 particles | 43.2 | 20.1 | bodies 8.2e-5 m; particles 1.2e-5 m | pass / pass |

Without the eager frame both fail on Metal at capture ("SolverVBD body-body contact state buffer needs to grow from 0
to 1000 during graph capture, but allocation during capture is not enabled on this device").

## 5. Decisions (archived options, DECISIONS.md style; not yet copied into `docs/DECISIONS.md`)

| date | decision | options considered (with numbers) | chosen and why | how to re-enable the others |
|---|---|---|---|---|
| 2026-09-25 | Where to fix VBD's `nextafterf` on Metal | (a) Warp fork: `nextafterf` + `__builtin_nextafterf` in `metal_crt.h`; (b) Newton fork: a `__METAL_VERSION__` branch in `interval_arithmetic.py`; (c) disable interval arithmetic | (a): fixes every native snippet that uses the C name, keeps Newton upstream code unmodified, bit-exact vs numpy (test) | (b)/(c) not implemented; (a) is one header block (revert 9050cb54) |
| 2026-09-25 | `nextafterf` implementation | float compares (first version: +-1-ULP subnormal neighbours wrong on Metal because comparisons flush subnormals) vs integer ordering on the bit pattern | bit pattern (bit-exact on Metal and CPU) | – |
| 2026-09-25 | Newton version for the Isaac Lab 3.0 reference | 1.5.2 (EA pin) vs 1.7.0.dev fork (identical VBD particle trajectories; fork's SolverMuJoCo also incompatible with our MuJoCo Warp) | 1.5.2 in `.venv-newton152` | `PYTHONPATH=upstream/newton` or `.venv-newtonfork` |
| 2026-09-25 | MuJoCo Warp under Newton's SolverMuJoCo | stock 3.11.0 (runs, coupled examples pass on Metal) vs MetalSim fork 3.14 (build error in `convert_mjw_contacts_to_newton_kernel`) | stock 3.11.0 for the Newton reference | port SolverMuJoCo to 3.14 or backport fork fixes (open) |
| 2026-09-25 | Capture of coupled VBD scenes on Metal | allocation during capture (needs a memory pool; Metal has none) vs one eager frame before capture (Newton's own recommendation) | eager frame | `coupled_probe.py --no-warmup-before-capture` |

## 6. Open items

* The ledger row "VBD solver fails to compile on Metal" (GAPS.md, Newton XPBD ledger) can be closed: fork
  9050cb54, this note. `docs/*.md` is not edited here.
* Improve the fork's Metal compile-error report. Keep link-stage lines ("Undefined symbol"), which lack `: error:`,
  and do not truncate at 4 KB. This needs a native rebuild of `libwarp`.
* VBD soft-body throughput: `solve_elasticity` is compute-bound at 1024+ worlds (0.57–0.69K env-steps/s at 10×10
  substeps/iterations). Profile inside the kernel (tet Hessians per vertex) before tuning.
* PhysX protocol comparison of VBD cloth/cube (the other agent's recordings), with Isaac Lab 3.0's own Newton
  parameters once an Isaac Lab 3.0 reference is recorded on the L4.

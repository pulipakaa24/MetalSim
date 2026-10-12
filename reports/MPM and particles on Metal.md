# Port Newton's implicit MPM, then accelerate it

*Research report, dated 2026-10-09. Scope: metalsim (Apple-Silicon Metal) against Isaac Lab 3.0-EA / Isaac Sim 6.1 / Newton 1.5.2, with references recorded on an NVIDIA L4 machine.*

For Isaac parity, metalsim's MPM has to be **Newton's `SolverImplicitMPM`**, not an MLS-MPM. That is the only MPM that Isaac Lab 3.0-EA exposes. PhysX removed its experimental MPM and FLIP solvers in 5.4.0, and today it offers only position-based (PBD) particles, which Isaac Lab does not wire into any asset or task. Newton's solver is a different algorithm from the Taichi/Genesis family. It solves one implicit, nonsmooth Drucker-Prager rheology and frictional-contact problem per step, on a trilinear (Q1) velocity grid with piecewise-constant (P0) stress. A Genesis-style explicit MLS-MPM could match it statistically at best, never step for step. The most important finding is that **Newton's solver already runs on the M4 Max through the in-repo Warp Metal fork after a one-line patch**. The patch makes `warp.fem` integrate in float32 instead of float64. With it, the solver matched the CPU to about 1e-8 and was bitwise repeatable across Metal runs. A faithful reference therefore exists on the Mac before any custom kernel is written. On the performance side, measured Metal microbenchmarks show that particle-to-grid transfer (P2G) should use **threadgroup fixed-point int32 tiles**: 0.33–0.48 ms per million particles, 6–8.7× faster than device float atomics, and deterministic. The plan, then, is to record a small set of Newton/Isaac Lab references on the L4, set up the Newton-on-Metal port as the parity oracle, and replace its hot paths with native Metal kernels one at a time. Each replacement must pass a tiered benchmark suite whose tolerances are tied to measured noise floors and to published granular scaling laws.

## Isaac's only MPM is Newton ImplicitMPM; PhysX offers PBD particles alone

### PhysX particles: maintained, but PBD-only and changing between versions

PhysX still ships and maintains `PxPBDParticleSystem` for fluids, granular material and diffuse particles. However, the **experimental `PxFLIPParticleSystem` and `PxMPMParticleSystem` were removed in v5.4.0**. PhysX 5.9.0 then removed particle cloth, particle rigids, attachments, particle volumes and `PxParticleSolverType`, which leaves PBD as the only particle solver ([PhysX CHANGELOG](https://github.com/NVIDIA-Omniverse/PhysX/blob/main/physx/CHANGELOG.md)). The PBD solver has kept changing since then. Version 5.10 extended mass-splitting to particle contacts. Version 5.11 changed velocity iterations to "solve particles only for velocity, not position errors" and made lowered iteration counts take effect ([PhysX CHANGELOG](https://github.com/NVIDIA-Omniverse/PhysX/blob/main/physx/CHANGELOG.md)). So **a PBD reference recorded on Isaac Sim 5.1 (PhysX 5.6.1) is not a valid reference for 6.1**.

The USD schemas (`PhysxParticleSystem`, `PhysxPBDMaterialAPI`, `PhysxParticleSetAPI` and others) are all still labelled "draft API". NVIDIA's dev guide says particles are **GPU-only** and that particles in different particle systems cannot collide ([omni_physics particles](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/particles/particles.html)). Isaac Lab 3.0-EA has no PhysX particle spawner, asset or task. Its deformables concept page sends fluids and granular media to the separate MPM path ([IsaacLab v3.0.0-EA](https://github.com/isaac-sim/IsaacLab/tree/v3.0.0-EA)).

PBD is a position-based, SPH-like constraint model, not continuum plasticity. **It is at most a cross-model sanity reference for metalsim's MPM, and a low-priority one.**

### Newton MPM: a first-class but experimental path in Isaac Lab 3.0-EA

The v3.0.0-EA release (2026-09-16) is "built for Isaac Sim 6.1, Python 3.12, PyTorch 2.11, NVIDIA Warp 1.16, and Newton 1.5.2". It advertises MPM and coupled-solver workflows, and GA is targeted for the end of October 2026 ([release notes](https://github.com/isaac-sim/IsaacLab/releases/tag/v3.0.0-EA)). Isaac Lab exposes the solver through the following pieces:

- **Solver config.** `MPMSolverCfg` (`solver_type="implicit_mpm"`) mirrors every field of Newton's `SolverImplicitMPM.Config` and adds `project_outside_colliders`.
- **Asset.** `MPMObject` provides `particle_pos_w` and `particle_vel_w` readback.
- **Spawners.** `MPMGridCfg`, `MPMPointsCfg` and `MPMParticleMaterialCfg`.
- **Demos.** Four of them: granular, two-way coupling, snowball smash and teapot fill.
- **Contrib RL tasks.** Two, which couple MuJoCo-Warp to MPM through Newton's `SolverCoupledProxy`.

The concepts page states plainly that "MPM support and rigid-MPM coupling are experimental" ([using_mpm.rst](https://github.com/isaac-sim/IsaacLab/blob/v3.0.0-EA/docs/source/concepts/using_mpm.rst)).

Two parity traps follow from this wiring. First, **Isaac Lab's `MPMParticleMaterialCfg` defaults differ from Newton's registry defaults**: friction 0.68 and yield_pressure 1e12 in Isaac Lab, against 0.5 and 1e15 in Newton ([mpm_cfg.py](https://github.com/isaac-sim/IsaacLab/blob/v3.0.0-EA/source/isaaclab_newton/isaaclab_newton/sim/spawners/mpm/mpm_cfg.py)). Second, each Newton substep runs at `SimulationCfg.dt / NewtonCfg.num_substeps`. A metalsim scene that copies "the defaults" can therefore silently diverge from Isaac.

The flagship task, `IsaacContrib-Franka-Pour`, sets a useful bar for robot-scale fidelity. It uses `sim.dt = 1/120` with decimation 4 and an MPM entry at **voxel 0.015, `solver="jacobi"`, `max_iterations=24`, `collider_basis="pic27"` and `separate_worlds=True`**. The fill is 735 particles per environment, the suggested scale is 2048 environments, and coupling is staggered with `mass_scale=1000` ([pour_env_cfg.py](https://github.com/isaac-sim/IsaacLab/blob/v3.0.0-EA/source/isaaclab_tasks/isaaclab_tasks/contrib/franka_pour/pour_env_cfg.py)).

### What the solver actually computes

The `SolverImplicitMPM` docstring describes "an implicit Material Point Method ... roughly following" Daviet & Bertails-Descoubes 2016 ([DOI 10.1145/2897824.2925877](https://doi.org/10.1145/2897824.2925877)). It is "less versatile than traditional explicit MPM but offers unconditional stability with respect to the time step" ([solver_implicit_mpm.py v1.5.2](https://github.com/newton-physics/newton/blob/v1.5.2/newton/_src/solvers/implicit_mpm/solver_implicit_mpm.py)).

Each `step` performs **one implicit solve with no internal substepping**, in this order:

1. Bin particles into cells with a `PicQuadrature`.
2. Rasterize collider SDFs onto collider-basis nodes.
3. Run an APIC P2G to get the unconstrained velocity.
4. Assemble the collider Delassus operator, the Hencky-strain elastic compliance, and a strain operator B in BSR form.
5. Run a warm-started rheology/contact solve to `tolerance=1e-4` or `max_iterations=250`, using colored Gauss-Seidel by default for Q1.
6. Gather velocities back to particles (G2P) and advect.

The yield surface is a Drucker-Prager cone with cohesion (`yield_stress`), a tensile cutoff and a compressive cap (`yield_pressure`). It uses a piecewise-linear anisotropic flow rule; Cam-Clay sits behind a development toggle that is off ([rheology_solver_kernels.py](https://github.com/newton-physics/newton/blob/v1.5.2/newton/_src/solvers/implicit_mpm/rheology_solver_kernels.py)). The default Young's modulus of 1e15 Pa makes the default material effectively **rigid-plastic**. Sand, snow, mud and viscous fluid are all parameter points in this one family; for example, Newton's multi-material example uses mud with yield_stress 300 and viscosity 100 ([newton/examples/mpm](https://github.com/newton-physics/newton/tree/v1.5.2/newton/examples/mpm)).

The velocity/strain basis menu (Q1/B2/B3 × P0/P1d/Q1/pic) and the hardening, dilatancy and viscosity options line up with NVIDIA's SIGGRAPH 2026 "Mixed Material Point Methods" paper. That paper reports being "integrated as a first-party module in the Newton physics engine" ([project page](https://research.nvidia.com/labs/prl/mixed_mpm/index.html)). **The 2026 paper, more than the 2016 one, is the effective spec.**

Rigid coupling is impulse-based. The MPM solve includes finite-mass bodies' Delassus operators, and `collect_collider_impulses()` returns the reactions, which the rigid solver applies with a one-frame lag. In Newton's example, MuJoCo runs 4 substeps per 1/100 s frame and MPM runs one step per frame ([example_mpm_twoway_coupling.py](https://github.com/newton-physics/newton/blob/v1.5.2/newton/examples/mpm/example_mpm_twoway_coupling.py)).

A version-skew caveat: Newton 1.6.0 (2026-09-10) adds USD import of MPM materials, `extract_particle_surface()` and a water dam-break example. **Isaac Lab 3.0-EA pins 1.5.2, so references must be recorded on 1.5.2** ([Newton CHANGELOG](https://github.com/newton-physics/newton/blob/main/CHANGELOG.md)).

## Only one candidate solver is both the Isaac reference and runnable on Metal today

| Solver | Time integration and discretization | Materials | Rigid coupling | Runs on Apple Metal today? |
|---|---|---|---|---|
| **Newton `SolverImplicitMPM`** (Isaac Lab 3.0-EA's MPM) | Implicit, one nonsmooth solve per step; Q1 velocity, P0 stress, APIC, sparse NanoVDB or dense grid | Drucker-Prager + cap + cohesion + viscosity + dilatancy + hardening; Hencky elasticity optional | Two-way, implicit within the MPM step, lagged to the rigid solver | **Yes, via the Warp Metal fork with a float32-accumulate patch** (measured, small scene) |
| Genesis MPM | Explicit MLS-MPM, quadratic B-spline, dense grid | Elastic, ElastoPlastic, Snow, Sand (Klár DP), Liquid, Muscle | Two-way, explicit SDF velocity projection; legacy coupler only | Compiles via Quadrants Metal; nondeterministic P2G; open Metal miscompile issues; no MPM measurement |
| Taichi mpm88/99/128/3d | Explicit MLS-MPM, quadratic B-spline, slip walls 3 cells deep | Fluid, jelly, snow (fixed-corotated) | None | Taichi has a Metal backend; no published Apple MPM numbers |
| Warp examples | `example_apic_fluid` (APIC fluid with a pressure solve) only; no `example_mpm` upstream | Fluid | None | The fork supports the needed features; untested |

### Genesis and Taichi are the explicit MLS-MPM lineage

Genesis's solver uses quadratic B-spline weights on a 3×3×3 stencil. Its P2G term is `−dt·V_p·4·inv_dx²·τ + m_p·C_p`, and it scatters into a dense grid with float atomics. Its default dt is 1e-2 with 1 substep, even though its own warning threshold at grid density 64 is about 3.1e-4 s ([mpm_solver.py](https://raw.githubusercontent.com/Genesis-Embodied-AI/Genesis/main/genesis/engine/solvers/mpm_solver.py), [solvers.py](https://raw.githubusercontent.com/Genesis-Embodied-AI/Genesis/main/genesis/options/solvers.py)). Its Sand model is the Klár 2016 Hencky-strain Drucker-Prager return map ([Klár et al.](https://www.math.ucla.edu/~cffjiang/research/sand/paper.pdf)). Rigid coupling exchanges an explicit `−mΔv/dt` force per grid node. **The newer SAP coupler has no MPM support at all** ([legacy_coupler.py](https://raw.githubusercontent.com/Genesis-Embodied-AI/Genesis/main/genesis/engine/couplers/legacy_coupler.py), [sap_coupler.py](https://raw.githubusercontent.com/Genesis-Embodied-AI/Genesis/main/genesis/engine/couplers/sap_coupler.py)).

On Metal, Genesis compiles through Quadrants, but open issues hit exactly the patterns MPM uses:

- atomics inside divergent control flow being lowered to threadgroup barriers ([quadrants#925](https://github.com/Genesis-Embodied-AI/quadrants/issues/925));
- fields larger than 2/4 GiB being mis-addressed through 32-bit offsets ([quadrants#963](https://github.com/Genesis-Embodied-AI/quadrants/issues/963)).

The Taichi examples are the canonical minimal MLS-MPM, as in [mpm99.py](https://raw.githubusercontent.com/taichi-dev/taichi/master/python/taichi/examples/simulation/mpm99.py). Their transfers come from the MLS-MPM paper ([DOI 10.1145/3197517.3201293](https://doi.org/10.1145/3197517.3201293)) and APIC ([DOI 10.1145/2766996](https://doi.org/10.1145/2766996)).

### Explicit and implicit solvers have different cost models

An explicit MLS-MPM is the easier kernel to write, but it is the wrong parity target. It steps at the elastic CFL limit, dt ≈ 0.3·Δx/√(E/ρ). With ρ = 1000 kg/m³ and Δx = 1 cm, this works out (my estimate) to:

- about **67 substeps per 20 ms control step at E = 1e5 Pa**;
- about **670 substeps at E = 1e7 Pa**.

Newton instead steps once per 1/60–1/120 s frame. Explicit MPM is many cheap, atomic-heavy passes; implicit MPM is a few heavy, iterative passes. Implicit wins for stiff and rigid-plastic media, which is Newton's default. Explicit wins only for soft materials on coarse grids. The two families cannot share trajectories. **Comparing them is restricted to aggregate observables** such as pile angle, runout and penetration force.

### Newton on Metal: measured working after a one-line patch

The Metal Warp fork (branch `metalsim`) supports what ImplicitMPM needs:

- native float32 and int32 device atomics;
- forward-only tile ops within 32 KB of threadgroup memory;
- host-built NanoVDB volumes;
- BSR matrices with a host-side topology path.

It has **no float64**. It also does not support `wp.config.deterministic` or CUDA conditional graphs ([metal.rst, local](/Users/aditya/robosim/upstream/warp-innate/docs/user_guide/metal.rst)). `fem.integrate` defaults to `accumulate_dtype=wp.float64` ([integrate.py, local](/Users/aditya/robosim/upstream/warp-innate/warp/_src/fem/integrate.py)), and Newton never overrides it. As a result, **unmodified `SolverImplicitMPM.step` fails on `metal:0`** with "Kernel 'integrate_fraction_…' uses builtins that are not supported on Metal yet: float64".

The probe scene was 1,000 particles at voxel 0.1 and dt 1/60 on a ground plane, run with Newton `90e23324` on the fork. With `fem.integrate` defaulted to float32, 120 frames ran on Metal for both dense and sparse grids. Results:

- **Dense grid, CPU vs Metal:** final minimum z 0.0026439887 on the CPU against 0.0026439829 on Metal. They differ at float32 round-off.
- **Repeatability:** two Metal runs of each grid type gave identical `particle_q` MD5 hashes.
- **Dense vs sparse on Metal:** minimum z **0.0017** (sparse) against 0.0026 (dense), and mean z 0.0342 against 0.0311.

The probe scripts were in this session's scratchpad (`mpm_metal_probe.py` and `mpm_metal_probe_f32.py`, not in the repo). The last result matters for parity: **grid type is part of the physics spec**, so references must be recorded with the grid type metalsim will run.

Determinism is structurally favourable:

- **P2G.** `warp.fem` P2G is a node-centric gather over a stable radix sort, with no float atomics.
- **G2P.** It is a single add per particle under `integration_scheme="pic"`. It becomes order-dependent only under GIMP ([integrate.py, local](/Users/aditya/robosim/upstream/warp-innate/warp/_src/fem/integrate.py)).

Two problems remain:

- **Performance.** The port rebuilds NanoVDB on the host, takes host paths for BSR topology, uses `block_dim = 1` fallbacks on non-CUDA code paths, and drives the iteration loop from the host.
- **Precision.** The CUDA reference accumulates in float64 and Metal cannot, so even a perfect port will not be bitwise-equal to Isaac. The size of that float32-vs-float64 drift over long rollouts has not been measured.

## Record seven reference sets on the L4, starting with noise floors

### How to record each path

Recording Newton directly is the simplest path. The sequence is:

1. Create a `ModelBuilder` and call `SolverImplicitMPM.register_custom_attributes(builder)`.
2. Add particles with `add_particle_grid(...)`, which takes per-particle `mpm:*` attributes.
3. Call `finalize()`, then construct `SolverImplicitMPM(model, config)`.
4. Each frame, call `solver.step(s0, s1, None, None, dt)`, then `project_outside` for granular scenes, then swap the states.

Positions and velocities are read from `state.particle_q` and `particle_qd`. Stress, plastic J and elastic strain live under `state.mpm.*` ([example_mpm_granular.py](https://github.com/newton-physics/newton/blob/v1.5.2/newton/examples/mpm/example_mpm_granular.py)).

Capture details:

- The examples graph-capture only on CUDA with `grid_type="fixed"`.
- Rebuildable sparse grids need `max_active_cell_count > 0`, an uncaptured warm-up step, and a post-replay `check_sparse_grid_rebuild_status()`, because overflow is reported asynchronously ([solver_implicit_mpm.py](https://github.com/newton-physics/newton/blob/v1.5.2/newton/_src/solvers/implicit_mpm/solver_implicit_mpm.py)).

The Isaac Lab path proves "what Isaac users get". Build a `SimulationContext` with `NewtonCfg(solver_cfg=MPMSolverCfg(...))` and an `MPMObjectCfg` spawned with `MPMGridCfg(particle_placement="cell_center", jitter=0)`. Read `obj.data.particle_pos_w` and `particle_vel_w` each step. The repo's existing `record_deformables_il3.py` already reads `ProxyArray`s this way and can be cloned. Its usage string must change, because **3.0-EA removed `--headless` in favour of `--viz none`** ([release notes](https://github.com/isaac-sim/IsaacLab/releases/tag/v3.0.0-EA)).

PhysX PBD needs Kit with GPU dynamics and USD authored through `omni.physx.scripts.particleUtils` ([particleUtils.py](https://github.com/NVIDIA-Omniverse/PhysX/blob/main/omni/extensions/runtime/source/omni.physx/python/scripts/particleUtils.py)). The 6.1 readback route is unverified. It could be a tensor view, USD `points` writeback, or ovphysx buffers, so a smoke test must come first.

No source gives L4-specific MPM problems or any published Newton MPM throughput. **The L4 timings metalsim records will be the first numbers for this solver.**

### Pin and store the full spec

Every reference should store three files:

- `init.npz`, holding the exact initial particle order;
- `traj.npz`, holding float32 `pos[T,N,3]` and `vel[T,N,3]`, plus `stress` and `Jp` every k steps;
- `meta.json`, holding every config field, both material-default sources, grid type, solver, iteration cap, tolerance, warm-start mode, the **iterations and residual actually reached per step**, and the versions of Newton, Warp, CUDA driver and GPU.

Two control recordings turn the comparison from guesswork into measurement:

- **Run each scene twice on the L4.** This gives the CUDA run-to-run noise floor; atomics, GS coloring and tile reductions might make it nonzero, and this has not been tested.
- **Run each scene once on the L4 with the same float32-accumulate patch used on Metal.** The gap between that run and the stock float64 run isolates the precision effect from everything else metalsim does.

## Validate with a tiered suite tied to noise floors and scaling laws

Granular flow is chaotic, and nobody publishes a formal protocol for benchmarking two MPM codes against each other. The literature compares fitted scaling laws and end-state observables, not particle trajectories ([arXiv 1602.07881](https://arxiv.org/html/1602.07881v1)). The suite below therefore uses three kinds of check: exact checks where physics allows them, short-horizon trajectory checks where chaos has not yet amplified errors, and statistical checks over seeds for everything else.

### The quantitative anchors

**Column collapse is the best-documented granular benchmark.** Normalized runout (L_f − L_i)/L_i is linear in the aspect ratio a for small a, then follows a^(2/3) in 2D or a^(1/2) axisymmetrically. The published fits are:

- Lube 2005 (2D): 1.2a for a < 2.3, then 1.9a^(2/3).
- Lajeunesse 2005 (2D): a for a < 3, then a^(2/3).

Fits from different labs disagree by about 20% ([arXiv 1602.07881](https://arxiv.org/html/1602.07881v1)). Continuum MPM with a µ(I) model reproduces the "jointed power-law scaling" ([Dunatunga & Kamrin, MIT DSpace](https://dspace.mit.edu/handle/1721.1/107838?show=full)). Rate-independent continuum models tend to over-predict runout at high aspect ratio, so a ≥ 5 should be read with that bias in mind.

**Silo discharge** should follow Beverloo, W = C·ρ_b·√g·(D − kd)^(5/2), with C ≈ 0.58 (range 0.55–0.65) and k ≈ 1.5 ([Thermopedia](https://thermopedia.com/content/821/)). In MPM, the grain-size term kd becomes a grid-scale annulus, so the D^(5/2) exponent and the fitted C are the comparable quantities.

**Resistive force theory (RFT)** is the robotics-relevant anchor. The generic coefficients are:

| A00 | A10 | B11 | B01 | B-11 | C11 | C01 | C-11 | D10 |
|---|---|---|---|---|---|---|---|---|
| 0.206 | 0.169 | 0.212 | 0.358 | 0.055 | −0.124 | 0.253 | 0.007 | 0.088 |

All are in N/cm³, scaled by a medium factor ζ: 0.194 for loose 0.3 mm glass and 0.371 when close-packed. The paper's own accuracy is about 10% on peak leg force and about 20% for the scaling routine ([Li, Zhang & Goldman, arXiv 1303.7065](https://arxiv.org/html/1303.7065)).

**Dam-break fronts** in Martin & Moyce run at v/√(gH) of 1.48 and 1.69, against the Ritter limit of 2 ([arXiv 1308.0115](https://arxiv.org/pdf/1308.0115)).

**ManiSkill2** supplies crisp success criteria for MPM manipulation. Its Pour task requires the liquid level within 4 mm of the line and fewer than 100 particles spilled. Its throughput was about 80–84 FPS for 16 environments on an RTX Titan, limited by kernel launches ([arXiv 2302.04659](https://arxiv.org/pdf/2302.04659)).

### Throughput metrics that cannot be gamed

Compare speed only at **matched fidelity**: the same voxel size, particles per cell, grid type, solver, iteration cap and tolerance. Report, for each:

- particle-steps per second;
- ms per step per million particles, broken down into sort, P2G, assembly, rheology iterations and G2P;
- environment-steps per second and real-time factor for batched tasks;
- **the mean and maximum rheology iterations reached, and the final residual**.

The last item closes a loophole. A port that "wins" by hitting `max_iterations` at a worse residual has lost fidelity, not gained speed. Conversely, a port that reaches the same tolerance in fewer iterations has genuinely gained. Configurations like Franka-Pour's fixed 24 Jacobi iterations probably hit the cap every step, so there the solver type and cap must match exactly.

## Fixed-point threadgroup P2G is the fastest and deterministic path on the M4 Max

### Measured: Apple's atomics invert the CUDA playbook

Microbenchmarks on this M4 Max (in the session scratchpad at `bench/p2g.swift`) found three constraints that the published CUDA playbook does not anticipate:

- **`threadgroup atomic_float` does not compile** under MSL 3.0 through 4.0.
- **64-bit device `fetch_add` and compare-exchange do not compile either.** Only 64-bit max compiles, even though Apple's feature table says "the full set of 64-bit atomic operations" is available from Apple9 ([Metal feature set tables](https://developer.apple.com/metal/limits)).
- **Float atomics exist only in device memory.** Threadgroup memory has native 32-bit integer atomics only.

The benchmark used quadratic B-spline P2G with 4 channels and 108 scattered adds per particle at 16.8M particles:

| P2G variant (M4 Max, measured) | ms per M particles | Deterministic |
|---|---|---|
| Non-atomic floor (wrong results) | 0.17 | n/a |
| Device `atomic_float`, unsorted | 2.85 | No |
| Device `atomic_float`, sorted | 2.30 | No |
| Sorted + SIMD-group segmented reduction + device float atomics | 0.50 | No |
| Threadgroup float tile via CAS loop | 5.54 | No |
| **Threadgroup int32 fixed-point tile (5.5 KB) + int32 device flush** | **0.33–0.48** | **Yes** |
| Same + SIMD-group reduction first | 0.42 | Yes |

Atomics account for about 93% of naive P2G time. At a 2^16 scale, fixed-point accuracy was 3.3e-4 absolute on node values up to about 12, about 3e-5 relative, with a range of ±32768 per channel.

The literature arrives at the same structure. Gao 2018 got **23×** over atomic-only scattering by sorting by cell, reducing within each warp and accumulating in a block ([Gao 2018](https://people.csail.mit.edu/kuiwu/GPU_MPM/GPU_MPM.pdf)). Fei 2021 found that CAS-emulated shared-memory float atomics were the V100 bottleneck. Dropping them gave 3–5×, and re-sorting only every 10–30 steps saved another 8–17% ([arXiv 2111.00699](https://arxiv.org/abs/2111.00699)). Apple's guidance describes the same reduction hierarchy: SIMD group, then threadgroup, then device ([WWDC22 10159](https://developer.apple.com/videos/play/wwdc2022/10159)). Gao's design must not be ported literally, because on Apple GPUs threadgroup float accumulation becomes a CAS loop that is slower than doing nothing.

### Supporting numbers for the rest of the step

- **Sorting.** The repo's stable radix sort runs at about **3.1 Gkeys/s** ([SPLAT.md, local](/Users/aditya/robosim/metalsim/render/SPLAT.md)). A full 32-bit sort of 1M particles therefore costs about as much as P2G itself (estimate: 0.33 ms). Counting sorts on block keys of 16 bits or fewer, done rarely, are the right choice.
- **Grid clearing.** Clearing a 34 MB dense grid took 0.037 ms (measured). Dense per-environment grids are effectively free to reset.
- **Saturation.** P2G cost per particle was flat from 2.1M to 16.8M particles. A single dispatch of about 0.25M particles nearly fills the 40-core GPU, so small environments must be batched into one dispatch.

For explicit MLS-MPM, a full step should cost an estimated **0.8–1.2 ms per million particles**. That is the same order as Wang 2020's 1.2–1.4 ms/M on a Quadro P6000 ([Wang 2020](https://yzhu.io/publication/mpmgpu2020siggraph/paper.pdf)). Only P2G has been measured; the rest of that figure is an estimate.

### Implicit iteration cost scales with grid nodes, not particles (inference)

Newton assembles the strain operator B as a BSR matrix over grid nodes. The P2G and G2P transfers therefore run once per step, and the 24–250 rheology iterations operate on strain and velocity nodes. At the 8–27 particles per voxel used in the shipped examples ([newton/examples/mpm](https://github.com/newton-physics/newton/tree/v1.5.2/newton/examples/mpm)), each iteration touches roughly an order of magnitude fewer entities than a particle pass would. This is an inference from the code structure, not a measurement.

If it holds, the iteration loop is short, dependent and launch-bound. Its cost on Metal is dominated by dispatch count and host round-trips for convergence checks, not by atomics. The fork already reports that **launch fusion paid off on Metal: fusing 5 of 9 per-iteration launches gave +3.9% on a 4096-environment MuJoCo Warp workload** ([throughput note, local](/Users/aditya/robosim/docs/research/throughput_2026-09-26.md)). The performance plan should therefore attack the iteration loop's encoding, and the transfer kernels second.

## Recommendations

### Which solver to match

Match **Newton 1.5.2 `SolverImplicitMPM`**, and pin the reference discretization to:

- `velocity_basis="Q1"`, `strain_basis="P0"`, `transfer_scheme="apic"`, `integration_scheme="pic"`;
- dense/fixed grids for parity scenes;
- the solver named by each scene (`gs` by default; `jacobi`, 24 iterations for Franka-Pour).

Use `integration_scheme="pic"` because GIMP's G2P is order-dependent. Treat the Mixed MPM 2026 paper as the spec for any extension. Defer B2/B3 bases, GIMP and separate-world sparse grids until the core passes.

Do not build an explicit MLS-MPM for parity. Build one later only if metalsim wants soft elastic or Genesis/ManiSkill-style scenes, and validate it statistically. Record PhysX PBD only if metalsim adds PBD fluids.

### Isaac references to record on the L4

| ID | Path | Scene and parameters | Purpose |
|---|---|---|---|
| R0 | Newton 1.5.2 direct, ×2 runs, plus a float32-accumulate run | R1 scene | CUDA noise floor; float64-vs-float32 precision floor |
| R1 | Newton 1.5.2 direct | Granular column collapse on a ground plane (µ_ground 0.5), µ_material 0.68, yield_pressure 1e12, E 1e15, ν 0.3, density 1000, voxel 0.05, 3 ppc, `cell_center` lattice with no jitter, `grid_type="fixed"`, gs, 250 iterations, tol 1e-4, dt 1/60, 3 s; quasi-2D slab with a ∈ {0.5, 1, 2, 3, 5, 8}; ≥10 jitter seeds for the statistical tier | Core trajectory and statistical parity |
| R2 | Isaac Lab `MPMObject` + `MPMSolverCfg` | Same as R1 at a = 2, `--viz none` | Proves the Isaac wrapper adds nothing |
| R3 | Newton direct | `example_mpm_multi_material` (snow: yield_pressure 2e4, tensile 0.2, µ 0.1, hardening 10, dilatancy 1; mud: yield_stress 300, viscosity 100, tensile 1, µ 0), `example_mpm_snow_ball`, `example_mpm_viscous` (240 fps, voxel 0.005, viscosity 50), `example_mpm_beam_twist` | Material coverage |
| R4 | Newton direct | `example_mpm_twoway_coupling` (voxel 0.05, fixed grid, 50 iterations, MuJoCo 4 substeps) | Rigid two-way coupling |
| R5 | Isaac Lab contrib | Franka-Pour and UR10-Particle-Push with logged open-loop action sequences and reset states | Robot-task parity |
| R6 | Newton direct | `example_mpm_anymal` (voxel 0.03, 3 ppc, tol 1e-6, 50 iterations, kinematic colliders) | Locomotion sinkage |
| R7 (optional) | Isaac Sim 6.1 Kit, PhysX PBD | Granular block via `particleUtils`; smoke-test the readback first | Cross-model sanity only |

Example parameters are from the Newton examples ([newton/examples/mpm](https://github.com/newton-physics/newton/tree/v1.5.2/newton/examples/mpm)) and the Isaac Lab tasks ([pour_env_cfg.py](https://github.com/isaac-sim/IsaacLab/blob/v3.0.0-EA/source/isaaclab_tasks/isaaclab_tasks/contrib/franka_pour/pour_env_cfg.py)).

### Benchmark suite and pass criteria

The tolerances below are proposals, derived from the cited accuracy of each anchor and from the measured noise floors. Re-base them once R0 is in.

| Tier | Test | Pass criterion |
|---|---|---|
| T0 Determinism | Metal run twice; mass conservation | Bitwise-identical `particle_q`; total mass exact |
| T0 Elastic analytic | Cantilever δ = PL³/3EI; bar wave speed √(E/ρ); `beam_twist` | Within 1% of analytic, and Metal–Newton gap within 1% |
| T1 Short-horizon trajectory | R1 a = 2 and a 1–3k-particle scene, first 10–60 steps | Per-particle RMS position error ≤ max(3 × R0 precision floor, solver tolerance × characteristic length); centre-of-mass and KE(t) within 1% |
| T2 Column collapse | R1 sweep, ≥10 seeds | Mean runout and final height within 10% and within the combined 2σ of Newton at every a; fitted exponent within ±0.05; within 20% of Lube 2005 / Lajeunesse 2005 for a ≤ 3 |
| T2 Repose / silo | Pile at φ = 20°, 30°, 40°; flat-bottom silo sweeping D | Repose angle within 2° of Newton; Beverloo exponent 2.5 ± 0.2 and fitted C within 10% of Newton |
| T2 RFT | 3.81 × 2.54 × 0.64 cm plate at 1 cm/s, 7 × 7 sweep of (β, γ), depths 2.5–7.6 cm | α_z and α_x within 5–10% of Newton; within 20% of ζ·generic |
| T2 Fluid / snow / mud | R3 scenes; viscous dam-break front Z(T) | Front position and heap profile (Chamfer or h(x) RMS) within 10% of Newton |
| T3 Robot tasks | R4–R6 with replayed actions | Box sinking depth and impulses within 10%; Franka-Pour poured fraction and spill count within 5 percentage points; ANYmal per-stance sinkage and base height within 10% |
| P Throughput | Fixed scenes at matched fidelity | Report the metrics listed above; no regression in T0–T3 allowed in exchange for speed |

### Staged build and performance plan

| Stage | Work | Gate |
|---|---|---|
| S0 | Record R0–R6 on the L4. Upstream a Metal default of `accumulate_dtype=float32` in the Warp fork (or a Newton pass-through). Run all reference scenes with Newton on `metal:0`. | T0 passes; T1/T2 measured, which sizes the float32 penalty before any custom code |
| S1 | Build the harness for the suite above in `metalsim/parity`. Freeze Newton-on-Metal as the oracle. | Oracle passes T1–T2 against the L4, or the gap is explained by R0's float32 run |
| S2 | Remove host round-trips: fixed/dense grids (no host NanoVDB rebuild), padded or preallocated BSR topology, convergence checks every k iterations, whole step encoded in one command buffer, Metal-tuned block sizes in place of the `is_cuda` gates. | Bitwise equal to the S1 oracle for any k that is a check-frequency change only; otherwise T1 |
| S3 | Native Metal kernels, one at a time behind A/B switches: counting sort by block (re-sorted when particles leave a free zone, every 10–30 steps per Fei 2021); P2G via a threadgroup int32 fixed-point tile (Q1 = 8 nodes per particle), with per-channel scale chosen so quantization stays ≤ 0.1× tolerance; G2P gather; colored-GS rheology with one dispatch per color and fused local solves. | Each kernel passes T0, T1 and T2 against the oracle before merging |
| S4 | Batched environments: env-major particle buffers, dense per-environment tiles, one dispatch over all environments, active-block lists. Coupling to metalsim's rigid solver using Newton's lagged impulse scheme. | T3; throughput at Franka-Pour scale (735 particles × 2048 environments) |
| S5 (optional) | Explicit MLS-MPM for soft materials; G2P2G for small environments; half-precision storage for render-only particles. | Statistical T2 only; never on the Isaac parity path |

## Conclusion

The question has narrowed from "which MPM algorithm should metalsim implement?" to "how fast can metalsim make Newton's algorithm run while keeping it numerically equivalent?" The reference is fixed: a nonsmooth implicit Drucker-Prager solver whose answer depends on grid type, solver choice, iteration cap and coloring. It already runs on Metal, so parity can be measured before it is engineered. The real risk is no longer porting but precision. Metal's lack of float64 accumulation, and the iterative solver's sensitivity to it, cannot be eliminated, only measured. That is why recording an L4 run with the same float32 patch is the single most valuable reference to collect.

The performance work also inverts the CUDA intuition in two places. On Apple GPUs, integer threadgroup atomics beat float atomics by an order of magnitude, so the deterministic fixed-point P2G path is also the fastest one. And because Newton's iterations run on grid nodes rather than particles, the main speed lever is launch and synchronization structure, not scatter throughput. Neither Isaac nor anyone else publishes MPM throughput for this solver or for Apple Silicon. The L4 and M4 Max numbers metalsim records under matched-fidelity rules will be the first published numbers for either, which makes the residual-and-iteration reporting rule as important as the speed itself.

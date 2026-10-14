# Why a box sinks through the sand bed: bearing physics, the implicit MPM contact path, and the coupling lag (2026-10-13)

Observation (batch_01, both MetalSim's `MPMCoupling` and Isaac's `SolverCoupledProxy` path, `mode="staggered"`): a
2 kg, 0.1 m box dropped from 0.15 m (impact 1.7 m/s) onto a 0.3 x 0.3 x 0.08 m bed (1,500 kg/m^3, `mpm:friction`
0.7, rigid, cohesionless; voxel 0.02, 50 iterations) sinks to ~1 cm above the floor. Research by a sub-agent (source
reading of the vendored solver + literature; full citations in its report, summarised here) and my own check of the
coupling algebra.

## Expected physics

* **What `mpm:friction` is.** The solver's yield surface is a Drucker-Prager cone sqrt(J2) <= mu p (stress stored as a
  6-vector with r_N = tr(sigma)/sqrt(6), |r_T| = sqrt(J2); `YieldParamVec.from_values` stores mu p_max against
  sqrt(3/2) p_max: `implicit_mpm/rheology_solver_kernels.py:68-74, 100`; Warp `fem/space/dof_mapper.py:220-226`).
  Matched to Mohr-Coulomb, mu = 0.7 is phi = 30.3 deg (triaxial-compression / outer cone), 35 deg (naive tan), 44 deg
  (plane strain, non-dilatant). A 3D footing sits between: take phi ~ 30 deg as the conservative value.
* **Static bearing capacity** (B = 0.1 m square on the surface, gamma = 14.7 kN/m^3): Vesic 10.3 kPa (phi 30.3),
  21 kPa (35), Meyerhof / Hansen similar; the box loads 1.96 kPa, a safety factor of 5-10. A thin layer on a rough rigid
  base (H/B = 0.8) raises it further (Mandel & Salencon 1972, Meyerhof & Chaplin 1953). Static failure needs
  phi < ~19 deg, i.e. mu < ~0.3-0.4. With a rigid-plastic material below capacity the static settlement is ~0 (in a
  grid method: within about half a voxel plus a particle spacing, ~1-2 mm here).
* **Dynamic penetration** of the 1.7 m/s impact (2.94 J): energy balance against a flat-punch resistance growing with
  depth gives 2.2 cm (phi 30), 1.2 cm (35); Uehara et al. 2003's sphere law (an upper bound for a flat face) gives
  2.8-3.1 cm; Katsuragi & Durian 2007's force law behaves alike. Expected: **bottom-out 1-2.5 cm, then no creep**.

So 7 cm is not a sand-strength result unless the effective friction were below ~0.3.

## The coupling lag explains a steady drift

The MPM solve stops the proxy body against the sand, but only the harvested wrench goes back to MuJoCo, and it acts in
the *next* rigid step (Isaac's exchange). With rigid sand and a converged contact solve (MPM brings the proxy from its
synced velocity to rest, impulse = -M v_synced):

* `mode="staggered"` (MPM sees the rigid step's end velocity v_n): v_{n+1} = v_n - g dt + (-v_n) = **-g dt** at every
  step, whatever the sand's strength. The resting box therefore moves down g dt^2 per step: 0.68 mm per step,
  8.2 cm/s at dt = 1/120 s. Over the 60-120 steps of the batch-01 runs: 4-8 cm, i.e. through the bed. This matches the
  observation on both paths (Isaac's Franka-Pour uses staggered, but its proxies are cups held by the gripper, where a
  drift of g dt^2 relative to the sand is masked by the arm).
* `mode="lagged"` with the velocity rewind (Newton's default for `SolverCoupledProxy.Proxy`; MetalSim implements it):
  MPM sees v_n - dt F_{n-1}/M; the recurrence oscillates (period ~6 steps) with zero mean drift if the contact solve
  converges; an under-converged solve (fraction alpha of the stopping impulse) drifts at -(1 - alpha)/alpha g dt.

Already visible in batch_01's data (`runs/mpm_main/box_on_sand/box_w2b_summary.json`, staggered, dt 1/120): once the
box sits in the sand it descends 0.80 mm per step between steps 30 and 40 and 0.75 mm per step between 40 and 60 while
the harvested sand force is the box's weight (19.4-20.0 N against 19.6 N): full support and still a steady drift of
about g dt^2 = 0.68 mm per step, as derived. batch_03 tests the lagged mode against it (below). If it holds, MetalSim's default exchange mode for robots
standing on sand must be `"lagged"` (staggered stays available for Isaac's Franka-Pour settings).

## Solver-side knobs (source, `metalsim/physics/mpm/implicit_mpm/`)

* Collider contact is velocity-level only (no positional correction): existing penetration is never pushed back by the
  grid solve; `project_outside` moves particles only past `projection_threshold` (0.01 voxel = 0.2 mm)
  (`contact_solver_kernels.py:68-87, 218-284`, `solver_implicit_mpm.py:414-424`).
* Collider nodes are active within 0.5/degree voxel of the surface (S2: 0.25 voxel = 5 mm); every node inside the box is
  active with the nearest face's normal (`rasterized_collisions.py:183-240, 492, 732-734`).
* Finite-mass colliders enter through J / IJtm with mass splitting over all contact nodes (`contact_solver_kernels.py:
  37-43`, `rasterized_collisions.py:520-571, 796-887`): with ~500 active S2 nodes the box looks light per node, so
  convergence in 50 iterations relies on the impulse warm start.
* `critical_fraction` 0: the sand resists any compression; nothing pushes over-dense regions apart, so particle
  clustering can accumulate under a sustained load.
* Upstream issue newton#4277 (fixed by #4291, Sep 2026, after 1.5.2): `_symmetric_part_transposed_op` multiplies by
  `b * 0.5` (`rheology_solver_kernels.py:135`). For a cohesionless rigid cone the cone is scale-invariant, so it should
  not weaken this case; it halves elastic response and the yield caps. Isaac Lab 3.0-EA ships it; MetalSim keeps
  Isaac's behaviour (noted in GAPS).
* Related upstream reports: newton#2697 (particles penetrate *moving* kinematic walls), #3954, #3473; no issue about
  bearing / sinking. Isaac Lab's MPM docs call the coupling experimental and suggest smaller dt / substeps.
* `harvest="last_substep"` with `mpm_substeps > 1` divides the last substep's impulse by the coupled dt: Isaac's
  convention (MPM.md 8i), under-reporting support by the substep count; MetalSim offers `harvest="mean"`.

## Probe design for batch_03 (expected outcome if the physics is right)

| # | change from baseline (drop from 0.15 m, staggered, 1 substep, voxel 0.02, 50 it.) | expected |
|---|---|---|
| 1 | baseline | bottom-out 1-2.5 cm, creep < 1 mm/s after 0.5 s |
| 2 | `mode="lagged"` | as 1; if 1 drifts and 2 does not, the lag is the cause |
| 3 | staggered, dt = 1/240 | as 1; a lag artefact drifts at g dt, i.e. half as fast |
| 4 | box resting on the surface, lagged | settlement < 2 mm; sand force / weight 1.00 +- 0.05 (mean) |
| 5 | as 4, staggered | drift g dt^2 per step if the derivation is right |
| 6 | as 4, 250 iterations, tolerance 1e-6 | as 4 (a difference = convergence-limited) |
| 7 | as 4, collider basis Q1 / pic27 | as 4 within half a voxel |
| 8 | as 4, voxel 0.01 | as 4; settlement shrinks with the voxel |
| 9 | as 4, mu 0.3 | bearing failure: sinks |
| 10 | as 4, mu 1.2 | settlement ~0 |

Pass bands: drop-test bottom-out 0.5-3 cm with creep < 1 mm/s after 0.5 s; rest-test settlement < 2 mm; failure
threshold between mu 0.3 and 0.45. Record per run: box height and velocity per step, sand force / weight, packing
under the box, particles inside the box.

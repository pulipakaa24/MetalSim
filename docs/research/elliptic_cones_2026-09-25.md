# Elliptic friction cones on Metal: reproduction, fix, audit, fidelity re-check (2026-09-25)

Scope: the handoff `docs/HANDOFF_elliptic_cone_perf.md` (another machine's session) found elliptic friction
cones 2.9–3.9× slower than pyramidal on Metal and attributed it to the launch sizing of
`_update_gradient_JTCJ_dense` in `mujoco_warp/_src/solver.py`. This note reproduces it on this machine,
fixes it in a worktree of the MuJoCo Warp fork (`upstream/mujoco_warp-ellip`, branch `metalsim-elliptic`,
based on `07a51a6`), audits the fork for every other non-CUDA fallback and capacity-sized launch, and
re-runs the fidelity protocol with elliptic cones now that they are cheap. Every number is labelled
**measured** (this machine, M4 Max 40-core GPU, 64 GB, through `scripts/gpu_run.sh ... timing` on an idle
device unless stated), **published** (from a document with a citation) or **estimated** (with the
reasoning). Scripts: `scripts/diagnostics/competitors/` (`elliptic_bench.sh`, `elliptic_check.{py,sh}`,
`elliptic_cpu_check.py`, `elliptic_cpu_time.py`, `elliptic_fidelity_g1.sh`, `elliptic_presets.py`,
`so101_creep.py`); logs `runs/competitors/elliptic_*.log`, `ellip_*.log`, `so101_creep_*`.

Benchmark protocol (the handoff's, `metalsim_step.py ROBOT 4096 matched`): Menagerie scene, joint-space PD
to random targets around the `home` keyframe resampled every 20 steps, 4 ms step, one substep, 4096 worlds,
Newton 10 iterations / 20 line-search iterations, 1000 timed steps after 100 warm-up, graph replay,
synchronized; `CONE=pyramidal|elliptic` overrides the model's cone, `JAC=dense|sparse` the constraint
Jacobian layout, `NJMAX` the row capacity (MetalSim's default 512). Physics steps/s = 4096 × 1000 / wall time.

## 1. Reproduction (measured, fork `07a51a6` unpatched, this machine)

| model (nv, Jacobian layout) | pyramidal steps/s | elliptic steps/s | elliptic cost | handoff (other machine) |
|---|---|---|---|---|
| Go2 (18, dense; the model's own cone is elliptic, impratio 100) | 923,850 | 240,281 | 3.84× | 1,023,772 / 262,371, 3.9× |
| G1 Menagerie (35, `auto` → sparse) | 218,295 | 77,263 | 2.83× | 229,447 / 80,024, 2.9× |
| G1 Menagerie, forced dense (`JAC=dense`, the layout the G1 task uses) | 196,519 | 60,731 | 3.24× | – |
| SO-101 + box (12, dense; Menagerie's own cone elliptic, impratio 10) | 1,065,380 | 441,361 | 2.41× | – |
| DeepMind humanoid (27, dense; `mujoco_warp/test_data/humanoid`) | 958,995 | 143,607 | 6.68× | – |

The throughput agent's independent measurement on the installed fork agrees
(`docs/research/throughput_regression_2026-09-25.md`: Go2 932,628 vs 239,935, 3.9×; SO-101 lift task
292,171 vs 122,534 env-steps/s, 2.4×; Panda 28,403 vs 9,923 env-steps/s, 2.9×).

Per-kernel GPU time (measured, eager mode with `WP_METAL_PROFILE=1`, sum of dispatch times per step,
`metalsim_profile.py ... kernels`, `CONE=elliptic`):

| model | step total ms | largest kernel | ms (share) | launches/step |
|---|---|---|---|---|
| Go2 | 24.13 | `_update_gradient_JTCJ_dense` | 19.11 (79.2 %) | 11 |
| G1 sparse | 38.94 | `_JTDACJ_sparse` (elliptic, one lane per group) | 16.81 (43.2 %); `_update_gradient_init_h_sparse` 5.51 (14.2 %) | 11 |
| G1 dense | 87.99 | `_update_gradient_JTCJ_dense` | 69.67 (79.2 %) | 11 |
| SO-101 | 14.57 | `_update_gradient_JTCJ_dense` | 8.70 (59.7 %) | 11 |
| humanoid | 44.98 | `_update_gradient_JTCJ_dense` | 41.13 (91.4 %) | 11 |

So the handoff's diagnosis holds for every dense-Jacobian model (the cone-term launch is 60–91 % of the
step), but **not for the Menagerie G1**: with nv 35 MuJoCo Warp's `auto` rule picks the sparse Jacobian,
which never launches `_update_gradient_JTCJ_dense`. Its 2.8× comes from a second non-CUDA fallback in the
same file: the sparse Hessian assembly `_JTDACJ_sparse` broadcasts the cone curvature terms from lane 0
through a shared tile (`tile_scatter_masked`, CUDA cooperative groups), and the Metal device patch
(`8ce5bb0`) ran that kernel with **one lane per constraint group** off CUDA (`threads_per_group = 1`), so the
whole per-group Hessian block was computed serially by one thread.

## 2. Fixes (fork worktree `upstream/mujoco_warp-ellip`, branch `metalsim-elliptic`)

All behind environment variables read at import (defaults in bold), so every option stays available:

- `MJW_JTCJ_MODE` (dense Jacobian, Newton, elliptic cones):
  - `capacity`: upstream's non-CUDA fallback, `dim_block = naconmax` (33.6 M threads per launch at 4096 Go2
    worlds), kept for A/B.
  - `contact`: upstream's CUDA form on any device that reports a core count (Warp's Metal backend exposes
    `Device.sm_count` = GPU cores, 40 here): `dim_block = ceil(sm_count · k · 256 / ndof_tri)` with
    `MJW_JTCJ_SM_FACTOR` = k (upstream's 6), each thread striding over contact slots. Handoff fix 1.
  - `world`: one thread per (world, Hessian entry) scanning the world's constraint rows for CONE-state
    elliptic contacts, no atomics, launch sized by nworld; deterministic summation order.
  - **`world2`**: `world` in two passes: one thread per world lists its cone contacts with their curvature
    terms (`SolverContext.cone_count / cone_efcid / cone_terms`, at most njmax // 3 contacts per world,
    45 MB at njmax 512 × 4096 worlds), then one thread per (world, entry) applies the list. Same arithmetic
    and order as `world`.
- `MJW_JTDAJ_ELLIPTIC_LANES` (sparse Jacobian, elliptic): **32** = full thread groups with the cone terms
  evaluated per lane (a handful of loads per lane instead of the shared-tile broadcast); 1 = the fork's
  previous one-lane groups.
- `MJW_JTDAJ_GROUPS_PER_WORLD` (sparse Jacobian): constraint groups processed in parallel per world; **0** =
  Warp's occupancy query, which is (1, 1) off CUDA and gives one group per world.
- `MJW_CCD_GRID_WAVES` (`collision_convex._ccd_grid_size`): off-CUDA CCD grid as k · cores · 256 threads
  striding over the candidate pairs; **0** = the previous capacity-sized launch (`naconmax` threads).

### 2.1 Throughput after the fix (measured, 4096 worlds, physics steps/s)

| model | pyramidal | elliptic before | `contact` (k = 6) | `world` | `world2` | best elliptic cost | speed-up |
|---|---|---|---|---|---|---|---|
| Go2 | 923,850 | 240,281 | 538,057 (k 2: 483,040; k 16: 551,979) | 553,510 | **580,420** | 1.59× | 2.42× |
| G1 sparse (`auto`) | 218,295 | 77,263 | – (lanes fix only) | 164,231 (lanes 1: 76,672; groups 4: 167,131) | – | **1.33×** | 2.13× |
| G1 dense | 196,519 | 60,731 | 113,340 | 126,568 | **144,348** | 1.36× | 2.38× |
| SO-101 + box | 1,065,380 | 441,361 | – | 837,584 | 816,526 | 1.27× (world) / 1.30× | 1.90× |
| DeepMind humanoid | 958,995 | 143,607 | – | 608,507 | **640,464** | 1.50× | 4.46× |

MetalSim's own G1 task (43 dofs, dense, 2.5 ms × 8 substeps per control step, `bench_contact_tuning.py`,
3 interleaved repeats; env-steps/s): `tau10_impact_hardlimits` 78,038 / 77,913 (patched / base); + elliptic
impratio 1: 11,029 base (7.07×) → 51,981 `world` → **55,379 `world2` (1.41×)**; impratio 10: 54,841 (1.42×);
impratio 100: 51,062 (1.53×).

Pyramidal rows re-measured in the same jobs are within 1 % (Go2 927,201 / 928,544 / 931,028; G1 216,796 –
220,102), which is the noise band of this protocol. Per-kernel time of the cone term after the fix (eager
profile): Go2 `world` 1.54 ms (19.6 % of 7.86), `world2` 1.26 ms (14.0 % of 8.97); G1 dense `world` 5.02 ms
(21.6 % of 23.21), `world2` 1.90 ms (9.2 % of 20.72); SO-101 `world` 0.97 ms (11.0 % of 8.83); humanoid
`world` 0.87 ms (18.3 % of 4.72). G1 sparse after the lanes fix: `_JTDACJ_sparse` 1.71 ms (7.2 % of 23.77).

What remains of the elliptic premium is the solver structure, not a launch: with elliptic cones MuJoCo Warp
rebuilds H = M + JᵀDJ (the tiled `_update_gradient_JTDAJ_dense_tiled`: Go2 0.94–1.09 ms, G1 dense 3.65 ms)
and refactors it every iteration, where pyramidal cones use the fork's fused incremental Hessian update +
register Cholesky (`ccfaffb`); on the sparse path `_update_gradient_init_h_sparse` (5.5 ms on the G1) has the
same role. Next to it: `_update_constraint_efc` and the line search are launched over (nworld, njmax) = 2.1 M
threads at njmax 512 (Go2 `NJMAX=64` with `world`: 608,499 steps/s, +9.9 %; the G1 task rejected njmax 128 as
+1.8 % in DECISIONS, so this is a per-scene MetalSim option, `BatchSimOptions(njmax=)`).

CPU device (measured, `elliptic_cpu_time.py`, Go2 elliptic, 32 worlds, 20 steps): `capacity` 12.8–13.4
ms/step, `world` 5.8–6.0, `world2` 5.1. The "fall back for CPU" branch was slow on the CPU too.

### 2.2 Options evaluated and not adopted (estimated unless stated)

- **Fold the cone term into the tiled JᵀDJ build** (handoff fix 3). The cone Hessian of a contact is
  Zᵀ C Z with Z its condim scaled Jacobian rows and C a condim × condim matrix
  (C = dm [w wᵀ + β u uᵀ + γ Eₜ], w = e₀ − (μ/t) u); inside `_update_gradient_JTDAJ_dense_tiled` it would be
  one (8 × 8)(8 × nv_pad) and one (nv_pad × 8)(8 × nv_pad) tile matmul per contact, i.e. about half a
  16-row tile iteration per contact. Measured tile cost: the G1 dense tiled kernel runs 3.65 ms / 11 launches
  for ~4 tile iterations per world ≈ 83 µs per iteration; 16 cone contacts per world ≈ 8 iterations ≈ 0.67 ms
  per launch against `world2`'s 0.17 ms. Estimated slower on the G1 and at best equal on the Go2 (5
  contacts); not implemented.
- **Fold into the fused register-Cholesky launch** (`_update_gradient_h_incremental_cholesky`): that kernel
  exists only for the incremental path (pyramidal); elliptic rebuilds H, so there is nothing to fuse into
  until the next item exists. Not implemented.
- **Extend the incremental Hessian update to cone rows** (handoff fix 4). Jaref changes every iteration for
  every row, so every CONE-state contact's curvature changes every iteration: the cone term must be
  recomputed for all cone contacts each iteration regardless (that is `world2`, 1.3–1.9 ms), and the saving
  would be the JᵀDJ rebuild (Go2 1.0 of 9.0 ms, G1 dense 3.65 of 20.7 ms) minus the incremental machinery's
  own launches (state tracking, rank-1 updates for flipped quadratic rows, subtract-old / add-new cone blocks).
  Estimated ≤ 10 % on the Go2 and ≤ 15 % on the G1 dense, with a second code path whose rounding differs from
  the rebuild; not implemented, kept as the next step if elliptic cones become a default anywhere.
- **`contact` mode** (upstream's form with the SM count): 3–20 % slower than `world2` on Go2 / G1 dense
  (measured above), has a tuning factor and atomics; kept as `MJW_JTCJ_MODE=contact`.
- **Upstream `main`'s cone math**: upstream commit `bc8ed60` (#1660, after v3.14.0) rewrote the same
  contraction in a scale-invariant form (projections normalised by t, `mu·n/t` instead of `mu·n/t³`), tested by
  `test_elliptic_hessian_scale_invariance`. The fork keeps v3.14.0's form so that physics stays within float
  noise of the unpatched fork (the two differ only when |Jaref| of a cone contact is below ~1e-4, where the
  old form's `t³` clamp at `MJ_MINVAL` bites); adopting the new form is a separate, upstream-tracking change.

### 2.3 Physics unchanged (measured)

- CPU oracle (`elliptic_cpu_check.py`, Warp CPU device vs `mj_step`, 2 worlds, PD to random targets, elliptic
  cones): Go2 max |dq| at steps 1 / 2 / 5 / 10 / 20 / 40 = 1.8e-7 / 2.3e-7 / 1.9–3.0e-7 / 6.7–6.9e-7 /
  1.8–1.9e-6 / 2.1–2.4e-6 rad for `capacity`, `contact`, `world`, `world2` and the unpatched fork alike
  (the 3.0 vs 1.9e-7 and 2.4 vs 2.1e-6 spread is the summation order: atomics vs one thread per entry);
  G1 sparse 6.7e-8 … 4.3e-7 identical for every mode (the CPU runs one lane per group, so the lanes change
  is exercised on Metal only); SO-101 2.4e-8 … 1.3e-7; humanoid 1.6e-7 … 2.9e-4 at 20 steps (the falling
  humanoid is chaotic; same for base and patched).
- Metal, graph replay, 512 worlds × 200 steps, elliptic cones (`elliptic_check.py`; max |Δqpos| over worlds
  and dofs at step 25 / 200, and the share of worlds more than 1e-3 rad apart; the floor is two instances of
  the same configuration in one process, MuJoCo Warp's contact order being nondeterministic):

  | model | floor (base vs base) | `world2` vs base | `world` vs base | `contact` vs base |
  |---|---|---|---|---|
  | Go2 | 1.9e-6 / 5.3e-5, 0 % | 2.0e-6 / 2.9e-6, 0 % | 1.7e-6 / 9.4e-4, 0 % | 1.9e-6 / 9.4e-4, 0 % |
  | G1 sparse (`auto`) | 6.6e-6 / 0.63, 14.5 % | 4.6e-6 / 0.46, 15.4 % | 5.6e-6 / 0.61, 16.0 % | 8.3e-6 / 0.46, 14.6 % |
  | G1 dense | 2.0e-5 / 0.63, 17.2 % | 5.3e-6 / 0.62, 14.3 % | – | – |

  The G1 rows grow to 0.5 rad in 14–17 % of the worlds for the base configuration against itself: the
  PD-to-random-targets G1 falls and its trajectories are chaotic, so the floor itself diverges; every
  patched form sits inside that floor (and the Go2's 1e-6 band). Logs `runs/competitors/ellip_check_*_b.log`,
  `ellip_check_base2.log`, snapshots `ellip_check_<robot>_<label>.npz`, offline comparer
  `elliptic_check_compare.py`.
- A first round of these checks (`ellip_check_base.log` / `_world.log` / `_contact.log`) showed the base
  configuration 1.4e-2 rad from itself with 100 % of worlds apart at step 25, which turned out to be a
  defect of the check script, not of the fork: it wrote `ctrl` through the torch view of the shared Metal
  buffer (`sim.t.ctrl.copy_` + `torch.mps.synchronize()`), which runs on torch's command queue and can land
  while the previous step's graph is still executing on Warp's queue; the slow base configuration hit it,
  the fast patched one rarely did. The scripts now write `ctrl` and the reset state through Warp after
  `sim.synchronize()`. `metalsim_step.py` (the throughput protocol) has the same pattern and is unaffected
  in its timing; `metalsim/learn/so101_lift.py` writes `ctrl` through the torch view before `sim.step()`
  as well (line 211) and should be checked by its owner (a stale-`ctrl` race would change actions, not
  timing).
- MuJoCo C oracle on Metal (4 worlds, PD hold + fixed random targets, elliptic; max |dq| vs `mj_step` at
  0.004 / 0.016 / 0.048 / 0.1 / 0.2 / 0.4 s, race-free re-run `ellip_oracle_*.log`): Go2 base 1.89e-7 /
  4.13e-7 / 4.98e-5 / 8.28e-5 / 7.64e-5 / 9.19e-5 rad, `world2` 1.89e-7 / 4.13e-7 / 4.99e-5 / 8.28e-5 / 7.66e-5 /
  9.20e-5; G1 sparse base 5.91e-8 / 1.23e-7 / 1.94e-7 / 6.23e-7 / 2.07e-6 / 3.46e-6, `world2` (lanes 32) 5.91e-8
  / 1.23e-7 / 1.81e-7 / 6.29e-7 / 2.13e-6 / 3.50e-6; G1 dense base 5.91e-8 / 1.23e-7 / 1.96e-7 / 6.47e-7 /
  1.80e-6 / 3.44e-6, `world2` 5.91e-8 / 1.23e-7 / 1.81e-7 / 6.29e-7 / 2.13e-6 / 3.50e-6: the same envelope
  against MuJoCo C, to the third digit at the early steps.
- Fork test suite on Metal (`pytest` in the worktree, default `world2`, `ellip_pytest_full.log`): **1450
  passed, 39 skipped, 2 failed** in 7.0 min; both failures (`collision_driver_test::test_hfield_maxconpair`,
  `io_test::test_put_data_nefc_zero_dense`, `AssertionError: 1 != 0`) fail identically on the unpatched
  `07a51a6` (`ellip_pytest_hf_base.log`, `ellip_pytest_io_base.log`), so they predate this work (the flex
  merge `8fbf965`; the handoff's "316 tests" was a subset count from an earlier tree). CPU device (`--cpu`,
  `solver_test`, `forward_test`, `constraint_test`): 242 passed, 10 skipped in the worktree; the upstream-PR
  worktree on `main`: 246 passed, 10 skipped.

## 3. Audit: non-CUDA fallbacks and capacity-sized launches in the fork (`mujoco_warp/_src`)

Representative set at 4096 worlds: G1 task (43 dofs, dense, pyramidal, the RL path), Go2 (elliptic), SO-101
(elliptic + impratio 10), DeepMind humanoid. "Matters" = measured or estimated share of a step.

| site | what happens off CUDA | Metal on the slow path? | matters at 4096? | fix / cost |
|---|---|---|---|---|
| `solver._update_gradient` cone term (`dim_block = naconmax`) | 33.6 M-thread launch, > 99 % idle, atomics | yes | **60–91 % of an elliptic dense step** (measured, §1) | `world2`: 2.4× step speed-up, physics within float noise (§2) |
| `solver._JTDACJ_sparse` elliptic (`threads_per_group = 1`) | one lane per constraint group | yes | **43 % of the G1 sparse elliptic step** (measured) | per-lane cone terms, 32 lanes: 2.1× (§2) |
| `solver._jtdaj_groups_per_world` (`get_suggested_block_size` = (1, 1) off CUDA) | one constraint group per world at a time | yes | G1 sparse elliptic: groups 4 → +1.8 % (measured, within noise) | knob kept, default unchanged |
| `solver._update_gradient_init_h_sparse` (nworld, nv_pad, nv_pad) | same on CUDA; 9.4 M threads per iteration | no (both) | 14 % of the G1 sparse elliptic step (measured); pyramidal skips it (incremental) | estimated fix: write only the upper triangle from the CSR list, ~2× on this kernel (7 % of the step); not done (the G1 task is dense) |
| `solver._update_constraint_efc`, `_linesearch_jv_fused`, `_linesearch_zero_jv` (nworld, njmax) | capacity-sized on every device; 2.1 M threads per iteration at njmax 512 | no (both) | Go2 elliptic: 1.5–1.8 ms of 8–9 (measured); njmax 64 → +9.9 % (measured) | `BatchSimOptions(njmax=)` per scene (DECISIONS row of the G1: +1.8 % for 128) |
| `solver` `graph_conditional` (`capture_while` CUDA only) | all `opt.iterations` iterations launched, converged worlds exit per kernel | yes | none at these settings: 500–3400 of 4096 worlds hit the iteration cap every step, so the batch would run all iterations on CUDA too (measured `overflow` counts) | – |
| `solver._cholesky_factorize_solve` (`_DENSE_CHOL_MAX_OFF_CUDA`, block 32) | fork's Metal path (register Cholesky) | no (fork-optimised) | 4–17 % of the step (measured) | already decided (DECISIONS 09-25) |
| `collision_convex._ccd_grid_size` (`naconmax` off CUDA) | 196,608-thread grid striding over `ncollision` pairs | yes | G1 pyramidal: waves 2 → −7.7 %, waves 8 → +1.0 % (measured): the idle threads are not the cost, the GJK/EPA work is (2.7 ms, 12 % of the G1 step) | knob `MJW_CCD_GRID_WAVES` kept, default 0 |
| `collision_convex` hfield kernel `dim=naconmax` | capacity-sized on every device | no (both) | rough terrain only; the fork's per-triangle patch (`f2716b4`) is a different kernel | estimated < 2 % (one launch per step, early exit) |
| `collision_primitive._primitive_narrowphase` `dim=naconmax` | capacity-sized on every device | no (both) | 0.05–0.19 ms per step (measured, < 1 %) | – |
| `collision_driver` SAP broadphase (`wp.utils.segmented_sort_pairs`, `array_scan`: host-side on Metal) | host sorts | yes, when `opt.broadphase` is SAP | none: every model here uses the default NXN broadphase (`_nxn_broadphase` in the profiles) | flex has its device sorts (`FLEX_DEVICE_SORT`); SAP would need the same |
| `constraint._efc_contact_init` / `_efc_contact_jac_*` `dim=naconmax` or (naconmax, nmaxdim) | capacity-sized on every device, once per step | no (both) | 0.09–0.13 ms per step (measured, < 1.5 %) | – |
| `sensor` collision / touch / contact sensors `dim=naconmax` | capacity-sized, once per step | no (both) | only with those sensors; estimated < 1 % (one early-exit launch) | – |
| `smooth._cfrc_ext_contact` `dim=naconmax`, `smooth` adhesion (nacttrnbody, naconmax, nv) | capacity-sized | no (both) | only with cfrc sensors / adhesion actuators; estimated < 1 % | – |
| `smooth._factor_i_sparse` / `_solve_LD_sparse` | fork's one-world-per-thread Metal path | no (fork-optimised) | 3–5 % of the G1 step (measured) | already decided (`ad22120`) |
| `island.py`, `sleep.py` (nworld, njmax) / `naconmax` launches | capacity-sized | no (both) | islands / sleeping are off in every task here | – |
| `collision_flex` (`_device_sort`, `dim=naconmax` launches) | device sorts on Metal (fork) | no (fork-optimised) | flex scenes only | already decided (`361f11f`) |
| `warp_util.check_toolkit_driver` | CUDA-only warning | – | none | – |

Reading: two sites were on a Metal-specific slow path and mattered (both fixed here); the remaining
capacity-sized launches are the same on CUDA and cost under 2 % each, except the (nworld, njmax) solver
launches, which are a MetalSim-side capacity choice (`njmax`), and the sparse-path Hessian init, which only
matters when a model takes the sparse Jacobian with elliptic cones.

## 4. Fidelity re-check with cheap elliptic cones

### 4.1 G1 (Isaac-Velocity-Flat-G1 asset, PhysX reference)

Protocol: PARITY §1.7 / DECISIONS 2026-09-25 (`metalsim.parity.record_g1` A_hold / B_random / C_drop vs
Isaac 5.1's `parity_out2/rt` recording, `compare.py`, momentum-derived impulses, Isaac's checkpoints played
on the task for transfer, `air_time_rollout.py` for feet_air_time / feet_slide of Isaac's checkpoint 1000,
`bench_contact_tuning.py` for cost). Presets: the adopted `tau10_impact_hardlimits` plus elliptic cones at
impratio 1 / 10 / 100 on top of it (`elliptic_presets.py` registers `tau10_impact_elliptic[_imp10|_imp100]_hardlimits`),
and the earlier `elliptic_hardlimits` for continuity with the DECISIONS row.

| preset (all on `tau10_impact_hardlimits`) | 20 ms mean force hold / drop-torso / landing N (Isaac 2190 / 2200 / 1996) | impulse ratio ours / Isaac hold / drop-torso / landing | drop penetration max / mean cm | limit excursion drop rad | hold joint RMSE end / max rad | transfer x m, Isaac it 500 / 1000 / 1499 (Isaac 3.19 / 3.02 / 3.11) | Isaac ckpt 1000 here: feet_slide COM / feet_air_time / falls (Isaac −0.0127 / 0.0446) | worlds at the iteration cap of 4096 | cost (env-steps/s) |
|---|---|---|---|---|---|---|---|---|---|
| pyramidal (adopted) | 2689 / 3729 / 2707 | 0.990 / 0.973 / 0.994 | 1.64 / 0.07 | 0.001 | 0.017 / 0.030 | 3.25 / 2.99 / 3.05 | −0.0144 / 0.0368 / 142 | 0 | 1.00× (78,038) |
| + elliptic, impratio 1 | 2536 / 2607 / 2699 | 0.989 / 0.971 / 0.994 | 1.33 / 0.06 | 0.002 | 0.003 / 0.034 | 3.24 / 2.97 / 3.05 | −0.0132 / 0.0376 / 171 | 0 | 1.41× (55,379) |
| + elliptic, impratio 10 | **2339 / 2379 / 2691** | 0.988 / 0.970 / 0.994 | 1.55 / 0.07 | 0.002 | 0.013 / 0.038 | 3.25 / 2.97 / 3.04 | −0.0131 / 0.0375 / 168 | 60–177 | 1.42× (54,841) |
| + elliptic, impratio 100 | 2328 / 2347 / 2689 | 0.988 / 0.970 / 0.994 | 1.44 / 0.07 | 0.002 | 0.016 / 0.038 | 3.24 / 2.96 / 3.03 | −0.0131 / 0.0377 / 160 | 3600–4080 | 1.53× (51,062) |
| `elliptic_hardlimits` (soft default contacts + elliptic, the earlier DECISIONS option (g)) | 2670 / 2818 / 2429 | 0.987 / 0.968 / 0.993 | 2.47 / 0.16 | 0.002 | 0.011 / 0.039 | 3.17 / 2.86 / 2.91 (earlier run) | – | 0 | 1.58× vs default (50,631; base fork 11,004) |

Sources: `runs/parity/tuning/report_<preset>/report.json` (`compare.py`, momentum-derived impulses and
20 ms mean forces from `contact_momentum`), `runs/competitors/ellip_fid_{rec,transfer,air,cost_world2}.log`,
`runs/contact_research/air_time/isaac1000_<preset>.json` (mean action, seed 1, ~1140 episodes; seed noise
±0.0006 slide, ±0.0005 air from the earlier batch). Isaac's PhysX numbers are the §1.7 references.

Reading. Impulses per event are within 1–3 % for every preset (as before: the contact model conserves the
same momentum). What elliptic cones change, on top of the adopted impact-only stiffening, is the force
profile of the torso impacts: the drop-torso 20 ms mean force goes from 1.70× PhysX's (3729 N) to 1.18×
(impratio 1) and **1.08× (impratio 10)**, the hold-torso from 1.23× to 1.07×, penetration 1.64 → 1.33–1.55 cm;
the landing is unchanged (1.35×), joint-limit behaviour unchanged, and Isaac's checkpoints travel the same
distance (within the 4-env spread). Foot sliding of Isaac's checkpoint moves from −0.0144 to −0.0131
(Isaac −0.0127; noise ±0.0006), i.e. the "no slide gain" of the earlier row was true for the soft default
contacts it was measured on and is a 9 % gain on the adopted preset; air time +0.0008 (noise ±0.0005, so
marginal); falls of the PhysX-trained checkpoint 142 → 160–171 in the single-seed run (+13–20 %; over 3 seeds 149 ± 13 → 160–164 ± 11–22, +7–10 %, §7.2: the one metric that moves away).
Impratio 100 converges in fewer than 10 Newton iterations in only ~10 % of the worlds (the 1.53× cost), so it
is not usable at this budget; impratio 10 leaves 60–177 worlds at the cap (pyramidal: 0).

### 4.2 SO-101 grasp: in-hand creep (measured, MuJoCo C, `so101_creep.py`)

MuJoCo's modelling guide recommends elliptic cones with a large impratio (and the Newton solver) when
contact slip is a problem (published: mujoco.readthedocs.io/en/stable/modeling.html, "Contact slip"); the
handoff cites a so101Sim / YAM grasp study with 1–2 mm elliptic vs 1–2.8 cm pyramidal creep. Our scene is the
lift task's (`assets/so101/scene_box_rl.xml`: 4 × 4 × 6 cm box, friction 1, box solref 0.01 / 1, the arm's
Menagerie servos, 5 ms step, Newton 10 / 20, **the scene's own setting is elliptic with impratio 10**).
Protocol (ours): gripper opened at the home pose, the box placed between the jaw tips aligned with the
gripper frame, jaws closed by the force-limited servo (ctrl −0.17 rad), 1.5 s settle, then 3 s of (a) static
hold, (b) wrist_roll ±0.6 rad at 1 Hz, (c) shoulder_lift ±0.25 rad at 1 Hz; creep = displacement of the box
centre in the gripper-site frame from the settled state (mm); one grasp pose.

| setting | hold creep mm / 3 s | wrist_roll | shoulder_lift | box shift during closing (x, mm) |
|---|---|---|---|---|
| pyramidal | **0.84** | 1.35 | 2.83 | +4.8 |
| elliptic, impratio 1 | 4.70 | 4.91 | 6.17 | +1.3 |
| elliptic, impratio 10 (the scene's own) | 5.24 | 5.73 | 8.04 | −12.9 |
| elliptic, impratio 100 | 1.48 | 2.08 | 4.19 | −17.4 |
| pyramidal, 50 Newton iterations | 0.84 | 1.35 | 2.83 | +4.8 |
| elliptic 100, 50 iterations | 1.48 | 2.08 | 4.19 | −17.4 |
| pyramidal, contact impedance 0.99 | 2.30 | 2.44 | 4.23 | +0.9 |
| elliptic 10, impedance 0.99 | 2.78 | 3.68 | 5.87 | −16.9 |
| elliptic 100, impedance 0.99 | diverges (box ejected) | diverges | 4.99 | −18.2 |

Reading: the creep is linear in time (0.25–1.3 mm per 0.5 s) and independent of the iteration count, i.e. it
is the soft-constraint steady state, not solver convergence; in this scene the pyramidal cone creeps
least, elliptic + impratio 100 is second (1.5 mm), and the scene's own impratio 10 creeps 6× more than
pyramidal. The real gripper does not creep (rigid Coulomb friction) and PhysX's TGS friction holds a static
grasp with drift only from its own iteration count, so on this metric pyramidal is the more faithful setting
for the lift task, and the published recommendation does not transfer to this scene as configured. Caveats:
one grasp pose, one box; the study the handoff cites used a different gripper and contact parameters.
MuJoCo Warp on Metal, same protocol, 64 identical worlds, fork `c301880` (measured, `runs/competitors/so101_creep_warp.log`):
pyramidal 0.85 / 1.35 / 2.82 mm, elliptic impratio 1: 4.75 / 4.96 / 6.21, impratio 10: 5.31 / 5.79 / 8.09,
impratio 100: 1.55 / 1.81 / 4.18, elliptic 10 + impedance 0.99: 2.96 / 3.60 / 6.05 (hold / wrist_roll /
shoulder_lift; MuJoCo C above), i.e. the Metal elliptic path reproduces MuJoCo C's creep to 0.1–0.3 mm; the
one exception is pyramidal + impedance 0.99 (1.54 vs 2.30 mm), where the box settled 7 mm further along the
jaw in Warp than in C (the stiff setting amplifies the contact-set difference), so that row is not used. A
first Metal run of this protocol had the same torch-side `ctrl` race as the check script (§2.3) and showed the
box dropped under the shoulder_lift load with pyramidal cones (`so101_creep_warp_racy.log`, kept as the
record of the defect); the race-free run matches C.

### 4.3 Recommendation per model class (cost next to each; no task default changed by this agent)

- **G1 locomotion (PhysX reference): elliptic cones with impratio 10 on top of `tau10_impact_hardlimits`**,
  1.42× the physics cost (55 K vs 78 K env-steps/s physics-only at 4096; on the full PPO loop, where physics is
  ~60 % of the step, estimated ~1.25×). Evidence: drop-torso and hold 20 ms forces at 1.07–1.08× PhysX's instead
  of 1.23–1.70×, penetration −5 %, slide 9 % closer, everything else unchanged, except +18 % falls of a
  PhysX-trained checkpoint (a policy trained on the preset is the check, as for the adopted preset). PhysX's own
  cone is exact (not a pyramid), so this is also the structurally faithful choice; Isaac Lab 3.0's MuJoCo-Warp
  G1 settings use pyramidal for cost, not fidelity (`runs/parity3/isaac/newton_mjwarp_settings.json`). The
  owner decides; the earlier "no slide gain, 3.84×" rejection no longer holds.
- **Go2-class quadrupeds** whose Menagerie model declares elliptic cones (impratio 100): keep the model's cone
  (fidelity to the model author's tuning) at 1.59× instead of 3.84×; check the iteration-cap count at impratio
  100 with the task's budget (the G1 needed > 10 iterations at impratio 100).
- **SO-101 grasping: pyramidal** on the creep metric (0.84 vs 1.5–5.2 mm per 3 s; the scene's own elliptic
  impratio 10 is the worst of the four); if elliptic is wanted for structure, impratio 100 at 1.27–1.30×. The
  published recommendation (elliptic + large impratio for slip) did not reproduce in this scene as configured.
- **DeepMind humanoid / other Menagerie models**: the model's own cone; elliptic now 1.50×.

## 5. Upstream

`scripts/diagnostics/mjwarp_upstream/DRAFT_elliptic_launch.md`: draft PR text for the world-major kernel
as the off-CUDA path (branch `elliptic-jtcj-offcuda` on the fork, one commit `ce17a73` on upstream `main`
`cc97eea`, using `main`'s scale-invariant cone math; CPU tests 246 passed / 10 skipped; not opened).

## 6. Status and what is left open

Done: reproduction, fixes (fork commits `c301880` + `4d53712` on `metalsim-elliptic`, pushed to the fork,
**not merged** into `metalsim`: the main session's approval is needed), Metal and CPU physics checks, fork
test suite, audit, G1 fidelity re-check, SO-101 creep protocol, upstream PR draft (branch
`elliptic-jtcj-offcuda`, pushed, not opened). Open:

- Merging `metalsim-elliptic` into the fork's `metalsim` (fast-forward from `07a51a6`) and fast-forwarding the
  shared checkout; then re-running the G1 headline throughput (pyramidal path untouched: the only shared code
  change is `_JTDACJ_sparse`'s factory signature and the `SolverContext` fields, allocated only for elliptic
  dense Newton; G1 pyramidal measured within noise in every job here).
- Task defaults: the recommendation in §4.3 (G1: elliptic impratio 10 on the adopted preset, 1.42×; SO-101:
  pyramidal) is the owner's decision; a policy trained on the elliptic preset is the check for the +18 % falls
  of the PhysX-trained checkpoint.
- The remaining elliptic premium (H rebuilt every iteration, §2.2) and the sparse-path `init_h_sparse` (§3)
  are the next throughput items if elliptic becomes a default.
- The two pre-existing fork test failures are root-caused in §7 (one fixed on a fork branch, one a MuJoCo
  version dependence).
- `metalsim/learn/so101_lift.py`: **no race** (correction of the first version of §2.3): its `step()` signals
  the learner event from torch after the `ctrl` write and the sim waits on it before its graph
  (`_learner_done`), and `sim.after(vs)` makes torch wait for the step's completion before the next write.
  My check scripts lacked exactly that pair. `tests/test_lift_ctrl_ordering.py` pins the order of the four
  calls per step and checks the event-ordered loop against host-synchronised stepping (§7).
- Upstream `main`'s scale-invariant cone math (`bc8ed60`) is not in the fork (v3.14.0 form kept for
  float-noise parity with the unpatched fork); adopting it is a separate upstream-tracking change.

## 7. Follow-ups requested by the coordinator (2026-09-25 late evening)

### 7.1 Confirming training run (G1 default decision, learning side)

Preset `tau10_impact_hardlimits_ellip10` (and `_ellip1`, `_ellip100`) added to `metalsim/physics/contact_tuning.py`
(the adopted preset plus elliptic cone and impratio; `tests/test_contact_tuning.py::test_elliptic_presets_set_cone_and_impratio`
checks the MjModel / MjSpec fields and the G1 builder's model: 3 passed). Queued (train class) as
`runs/il3/run_train2.sh flat flat tau10_impact_hardlimits_ellip10 none 1000 0 g1_flat_flatcfg_ellip10 com origin`,
i.e. the queued COM run's command with the contact preset swapped (2.3.2 flat config, base_velocity com,
feet_slide_velocity origin, seed 0, 1000 iterations, same PPO): log `runs/il3/g1_flat_flatcfg_ellip10.log`,
stdout `runs/il3/g1_flat_flatcfg_ellip10.stdout`. Result (measured, finished 2026-09-25 23:30, exit 0, policy `runs/il3/ckpt/g1_flat_flatcfg_ellip10.pt`):

| iteration | elliptic impratio 10 (this run): return / length | COM run, default contacts | recommended preset | origin seeds 0 / 1 / 2 (default contacts) | Isaac PhysX |
|---|---|---|---|---|---|
| 100 | −5.1 / 83 | −5.4 / 99 | – | −5.3 / 91 | −6.6 / 200 |
| 200 | −5.9 / 926 | 0.1 / 999.5 | −2.1 / 956 | 0.7 / 999 | 6.6 / 981 |
| 300 | 9.8 / 999.5 | 10.9 / 969 | – | 12.1 / 10.8 / 9.8 | 19.2 / 1000 |
| 500 | 18.5 / 989 | 19.2 / 991 | 19.9 / 1000 | 21.0 / 18.4 / 19.6 | 25.3 / 996 |
| 750 | 23.8 / 972 (700: 23.85 / 1000) | 24.6 / 1000 | – | 26.3 / 1000 | 26.8 / 988 |
| 1000 | **26.31 / 989.9 (n = 40)** | 27.11 / 1000 | 27.85 / 1000 | 28.4 / 26.3 / 26.0 (26.9 ± 1.3) | 27.3 / 991 |
| env-steps/s in the loop | **29.3 K** | 50.6 K | 42.2 K | 25.7 K (older tree) | – |

No blow-up or non-finite reset; anomaly monitor 2501 log points (COM run 3408, recommended 2356): the same
"terminal-step reward dominates" and 0.15–0.18 rad soft-limit flags as the recommended run, no penetration
flags (the default-contact runs have them). Learning is inside the seed spread of the pyramidal runs (0.6
below the three-seed mean, above two of the three origin seeds, 1.5 below the recommended run's single seed),
so the cone changes nothing measurable in learning at 1000 iterations; its cost in the PPO loop is **1.44×**
the recommended preset (29.3 vs 42.2 K env-steps/s; 1.42× physics-only).

**Recommendation to the owner (one paragraph).** For the G1 velocity task the most PhysX-faithful contact
setting we have measured is `tau10_impact_hardlimits_ellip10` (elliptic cones, impratio 10, on the adopted
impact-only stiffening with hard limits): against Isaac's PhysX recordings it brings the drop-torso and hold
20 ms mean forces to 1.08× / 1.07× of PhysX's (pyramidal: 1.70× / 1.23×), penetration 1.64 → 1.55 cm, feet_slide
of Isaac's checkpoint −0.0144 → −0.0131 (Isaac −0.0127), with impulses, limit excursions, transfer distances
and air time unchanged, and the cone itself is the exact cone PhysX uses (Isaac Lab 3.0's own MuJoCo-Warp G1
settings pick pyramidal for cost, not fidelity). Against it: falls of a PhysX-trained checkpoint +7–10 %
(3 seeds, within one seed sd, all genuine topples under forward commands, cause not established), and the
cost, **1.44× in the full PPO loop** (29.3 vs 42.2 K env-steps/s) with learning at +26.3 vs +27.85 at iteration
1000, inside the ±1.3 seed spread. So: if the task's purpose is PhysX parity of the contact physics (the
project's rule), adopt `tau10_impact_hardlimits_ellip10` as the G1 default and pay the 1.44×; if the loop
throughput headline matters more for this task, keep `tau10_impact_hardlimits` and state elliptic as the
measured fidelity option. I did not change the default. For the SO-101 grasp the answer is the other way
(pyramidal creeps least, §4.2); for Menagerie models that declare elliptic cones, keep the model's own.

### 7.2 The +18 % falls of the PhysX-trained checkpoint under elliptic cones

Setup of the number: Isaac's PhysX-trained flat checkpoint at iteration 1000 (`runs/contact_research/
isaac_model_1000_metalsim.pt`), played with the mean action on the task's own commands and resets, 1024 envs ×
1000 control steps (20 s), seed 1, one run per preset; a fall is the task's termination (torso touch force
> 1 N over the 15 ms history, the only non-time-out termination), so every fall is a torso-to-ground contact.
Counts: pyramidal 142 falls in 1122 episodes, elliptic impratio 1 / 10 / 100: 171 / 168 / 160 in ~1140.
Characterisation (`scripts/diagnostics/competitors/g1_falls.py`, `g1_falls_batch.sh`, `g1_falls_table.py`;
`runs/competitors/g1_falls/`): 3 seeds × 4 presets for checkpoint 1000, checkpoints 500 and 1499 for pyramidal
vs impratio 10, plus `default` and `tau5_imp99_hardlimits` for scale; per fall: time into the episode, pelvis
pitch / roll 40 ms before the torso contact (forward / backward / sideways), the command in force, and how
many distinct envs fell. Results (measured, `runs/competitors/g1_falls/`, table by `g1_falls_table.py`):

| checkpoint | preset | falls per seed (episodes) | mean ± sd | envs that fell (twice or more) | fall time 1–5 s / 5–15 s / ≥ 15 s | median s | forward / backward / sideways 40 ms before | mean \|cmd_xy\| at fall (n at < 0.1) |
|---|---|---|---|---|---|---|---|---|
| 1000 | `default` (soft contacts) | 349 (1283) | 349 | 90 (77) | 260 / 86 / 3 | 3.3 | 235 / 33 / 81 | 0.38 (31) |
| 1000 | `tau5_imp99_hardlimits` | 134 (1114) | 134 | 44 (28) | 96 / 38 / 0 | 3.5 | 118 / 10 / 6 | 0.41 (13) |
| 1000 | **`tau10_impact_hardlimits`** (adopted) | 152 / 134 / 160 | **149 ± 13** | 43 (30) / 37 (27) / 48 (31) | 347 / 94 / 5 | 3.2 | 345 / 33 / 68 | 0.38 (21) |
| 1000 | + elliptic impratio 1 | 164 / 148 / 169 | 160 ± 11 | 50 (33) / 34 (29) / 51 (33) | 374 / 102 / 5 | 3.2 | 404 / 21 / 56 | 0.39 (27) |
| 1000 | **+ elliptic impratio 10** | 165 / 146 / 175 | **162 ± 15** | 47 (31) / 33 (29) / 52 (35) | 378 / 104 / 4 | 3.0 | 407 / 20 / 59 | 0.39 (27) |
| 1000 | + elliptic impratio 100 | 159 / 144 / 188 | 164 ± 22 | 46 (32) / 34 (28) / 53 (38) | 389 / 99 / 3 | 3.0 | 415 / 22 / 54 | 0.40 (26) |
| 500 | `tau10_impact_hardlimits` | 240 (1197) | 240 | 67 (31) | 186 / 33 / 2 (+ 19 in the first second) | 2.3 | 156 / 57 / 27 | 0.58 (8) |
| 500 | + elliptic impratio 10 | 278 (1233) | 278 | 69 (38) | 225 / 33 / 0 (+ 20) | 2.3 | 216 / 38 / 24 | 0.62 (4) |
| 1499 | `tau10_impact_hardlimits` | 205 (1177) | 205 | 52 (39) | 166 / 39 / 0 | 3.0 | 60 / 65 / 80 | 0.32 (51) |
| 1499 | + elliptic impratio 10 | 222 (1190) | 222 | 56 (40) | 179 / 42 / 1 | 2.8 | 72 / 71 / 79 | 0.32 (57) |

Reading:
- **How solid**: over three seeds the elliptic excess on checkpoint 1000 is +11 to +15 falls on 149
  (**+7 to +10 %**, not the +18 % of the single-seed run), which is about one seed standard deviation
  (11–22 falls) and the same for impratio 1, 10 and 100. The direction is consistent across the three
  checkpoints (500: +16 %, 1000: +9 %, 1499: +8 %, one seed each), so it is probably a real but small effect;
  a 3-seed sd of 13–15 means it would need ~6 seeds per arm to be shown at 2σ. For scale, the contact
  stiffness moves the same count from 349 (soft default) to 134–149, i.e. the cone effect is a tenth of the
  stiffness effect.
- **What kind of fall**: none is a start-up failure (0 falls in the first second for checkpoint 1000; 19–20
  for checkpoint 500 under both cones), the median fall is 3.0–3.5 s into a 20 s episode, and every fall is
  a genuine topple (pelvis tilted > 0.3 rad 40 ms before the torso contact, "upright" count 0). Falls are
  concentrated: 33–53 of 1024 envs fall, and 27–38 of those fall twice or more, so the episodes that fall
  are particular command / reset draws that this policy cannot handle in our contact model, not a diffuse
  rate. At checkpoint 1000 the falls are forward under a forward command (mean |cmd_xy| 0.38–0.40 m/s,
  only ~6 % at a standing command); elliptic cones add forward falls (345 → 404–415 over 3 seeds) and remove
  a few sideways ones (68 → 54–59); checkpoint 1499 falls in all directions equally.
- **Why a PhysX-trained policy falls slightly more in the more PhysX-like cone**: not established. The
  candidates are (i) the torso-contact termination is ours, and the elliptic cone changes the impact force
  profile (§4.1) which is exactly what decides whether a stumble ends in a > 1 N torso touch; (ii) the
  policy was trained with PhysX's patch-friction model at 200 Hz on 5 ms steps and the flat stiffness
  (Isaac Lab's contact offsets), so the "more PhysX-like" friction cone is combined with a contact stiffness
  that is not PhysX's, and the combination was never seen in training; (iii) plain policy sensitivity
  (checkpoint 500 falls 60–90 % more than 1000 under every setting). What is established is that slide,
  air time, transfer distance and the force profile move toward PhysX under elliptic cones while the fall
  count moves 7–10 % away, within one seed sd. The confirming training run (§7.1) is the right arbiter for
  the default: a policy trained on the elliptic preset does not have caveat (ii).

### 7.3 `so101_lift.py` ctrl ordering

Finding corrected: the env is already event-ordered (see §6). Added `tests/test_lift_ctrl_ordering.py`
(needs Metal; 2 passed, `runs/competitors/lift_ordering_test4.log`): (a) per `step()`, the learner-event /
physics calls are exactly `signal(learner event)` → `sim.wait(learner event, same value)` → physics →
`sim.after(v)` (then the env's second commit before the post-reset forward), three steps in a row, and the sim's
`ctrl` equals what `step()` computed; (b) 30 random-action steps with no host sync give the same `qpos` (1e-4) as
stepping with a host sync after every step. (a) fails if `_learner_done()` or `after()` is dropped; (b) fails
if a stale `ctrl` reached the physics. No change to `metalsim/learn/so101_lift.py`.

### 7.4 The two pre-existing fork test failures (both fail on the CPU device too, so not Metal-specific)

- `collision_driver_test::test_hfield_maxconpair` (a 2 × 2 m box resting 1 mm into a 0.2 × 0.2 m heightfield,
  expects 4 contacts, gets 0): passes at `8ce5bb0` (Metal device patch), fails from `f2716b4` (the fork's
  per-triangle heightfield plane contacts, MetalSim's rough-terrain fix). Root cause: that path tests every
  vertex of a mesh of ≤ 256 vertices against each triangle's column (the G1 feet), but for primitives it only
  has the single support point along the triangle normal, and a box's bottom corner lies outside every
  triangle's footprint of a heightfield smaller than the box, so no triangle claims a contact; upstream's
  GJK/EPA against the prism clips the box instead. Fix on fork branch `metalsim-hfield-test` (`b630530`, from
  `4d53712`): primitives take upstream's GJK/EPA path, meshes keep the plane path;
  `HFIELD_PLANE_CONTACTS_PRIMITIVES = True` restores the previous form (archived). CPU: `-k hfield` 3 passed
  (was 1 failed). MetalSim's `tests/test_terrain.py::test_hfield_mesh_contacts_match_mujoco_c` (the G1 feet,
  mesh path unchanged) and `test_plane_convex_contacts.py` against that branch on Metal: 24 passed; the fork's
  own `collision_driver_test -k hfield` on Metal: 3 passed (`runs/competitors/hfield_gate_tests{,2}.log`).
  Not merged (coordinator's call; the G1 rough task uses mesh feet on the heightfield, so its contacts are
  unchanged; scenes with primitive geoms on heightfields go back to upstream's behaviour).
- `io_test::test_put_data_nefc_zero_dense` (`AssertionError: 1 != 0` at `self.assertEqual(mjd.nefc, 0)`): the
  assertion is on MuJoCo C's own `MjData` before MuJoCo Warp is involved. In the installed MuJoCo 3.14.0 the
  fixture's tendon with `frictionloss="0.5"` instantiates one `mjCNSTR_FRICTION_TENDON` row (nefc = 1); the test
  (added with the v3.14.0 bump, `88af9cc`) expects none, and fails identically on upstream `main` `cc97eea` with
  3.14.0. Upstream's lock file pins a MuJoCo nightly from py.mujoco.org (`3.13.1.dev984848064`), where the
  fixture evidently has nefc = 0; so this is a MuJoCo-version dependence of the test, not a fork or Metal
  defect. No change; expected to fail with a released MuJoCo 3.14.0.
- Aside noticed while bisecting: every pre-existing MetalSim commit on the fork's `metalsim` branch
  (`f2716b4` … `07a51a6`) carries a `Co-Authored-By: Claude` trailer, so that branch's history is not
  CLA-clean as a whole; anything for upstream has to be re-committed on a clean branch (as the PR branch
  `elliptic-jtcj-offcuda` is: one clean commit on upstream `main`; my two `metalsim` commits carry no trailer).

## 8. Cutting the elliptic premium (2026-09-26, coordinator's follow-up; fork branches `metalsim-elliptic2*`)

Goal: make elliptic cones as cheap as they can be before the G1 default decision. Baseline at fork `fe6fe71`
(= shared checkout `b630530`): G1 task elliptic impratio 10 at 1.42× physics / 1.44× loop vs pyramidal.
Protocol for the numbers below: `metalsim_step.py ROBOT 4096 matched` through the timing queue (PD to random
targets, Newton 10 / 20, 1000 timed steps); `g1task` = the task's own model (Isaac's G1 asset, 43 dofs, its PD
servos, 2.5 ms × 8 substeps per call, `recommended` or `tau10_impact_hardlimits_ellip10`), physics steps/s.
Per-kernel profiles are eager-mode GPU time per physics step (`WP_METAL_PROFILE=1`).

### 8.1 What MuJoCo C does (engine_solver.c, 3.3.7, functions `FactorizeHessian`, `HessianIncremental`, `HessianCone`, `mj_solCGNewton`)

`FactorizeHessian` builds H = M + JᵀDJ with D = efc_D on QUADRATIC rows and 0 elsewhere, factorizes it into
`L`, then, "if (ctx->ncone)", calls `HessianCone`. `HessianIncremental` (called every Newton iteration after
`CGupdateConstraint`) keeps `L` up to date with `mju_cholUpdate` rank-1 updates for each row whose QUADRATIC
state flipped (add or subtract J_i·√D_i), falls back to `FactorizeHessian(flg_recompute=1)` if a rank-1 update
loses rank, and then again calls `HessianCone`. `HessianCone` copies `L` into `Lcone` and, for every contact in
`mjCNSTRSTATE_CONE`, factorizes the contact's local dim×dim cone Hessian (`con->H`, from
`mj_constraintUpdate` with `flg_HessianCone`), forms LTJ = Lᵀ_local·J_contact and applies `dim` rank-1
`mju_cholUpdate`s to `Lcone`. So in C the quadratic part is incremental across iterations, and the cone part is
re-added from scratch every iteration (it depends on Jaref), as `dim` rank-1 factor updates per cone contact
on a fresh copy of L.

### 8.2 How often the state changes (measured on the CPU device, `g1_state_changes.py`)

States: 64 worlds of the G1 task model from MuJoCo C (PD hold at the standing keyframe with ±0.05 rad target
noise resampled every 20 control steps; the robot stands for ~0.5 s and lies on its torso from ~1.5 s, so the
later snapshots are contact-rich but not walking; the policy-rollout snapshots are `g1_state_dump.py`,
pending in the render queue). Per Newton iteration (mean over worlds still iterating), elliptic impratio 10:

| snapshot | rows / world | CONE rows / world by iteration | QUADRATIC flips / world by iteration | worlds with 0 flips (of 64) | solver_niter mean (max) |
|---|---|---|---|---|---|
| t = 10 (standing) | 26 | 0 | 1.7, 0 | 12, 64 | 1.8 (2) |
| t = 60 (falling) | 9.5 | 5.2, 6.0, 5.0, 5.3, 6.0, 6.2, 5.7, 7.8, 6.0 | 1.3, 1.3, 2.2, 1.2, 0.8, 0.4, 0.9, 1.2, 3.0 | 18, 35, 33, 48, 59, 62, 61, 63, 62 | 5.2 (10) |
| t = 100 (on the torso) | 20 | 2.9, 5.0, 6.1, 6.7, 9.0 | 3.5, 3.5, 1.3, 1.0, 0.7 | 4, 27, 50, 57, 62 | 4.0 (8) |
| t = 200 | 20 | 4.0, 4.4, 5.5, 5.8, 5.2 | 4.5, 3.1, 1.3, 1.2, 1.5 | 4, 24, 48, 58, 62 | 3.8 (7) |
| t = 300 | 20 | 2.9, 4.4, 4.7, 4.5, 5.0 | 3.8, 3.0, 1.5, 1.5, 1.0 | 8, 31, 49, 57, 62 | 3.6 (7) |
| all 9 snapshots | | | 2.2 (it 1), 1.6 (it 2) | | **2.76** |
| same states, pyramidal | 26–34 | – | 2.8 (it 1), 2.7 (it 2) | | **3.07** |

Policy states (`g1_state_dump.py`: the elliptic-trained checkpoint `g1_flat_flatcfg_ellip10.pt` walking on
the task, 256 envs, snapshots at control steps 2–399, first 64 worlds analysed; the task's njmax 112):

| snapshot | rows / world (elliptic rows) | CONE rows / world by iteration | QUAD flips / world by iteration | solver_niter mean (max), elliptic | same states, pyramidal: rows, flips it 1–3, niter |
|---|---|---|---|---|---|
| t = 10 | 12.9 (8.8) | 1.4, 2.9, 4.0, 5.2, 4.1, 6.0, 7.5, 6.8 | 3.8, 1.4, 1.2, 1.5, 3.4, 2.5 | 3.48 (9) | 15.8, 3.8 / 2.2 / 2.6, 3.27 (9) |
| t = 60 | 12.8 (8.9) | 1.9, 2.5, 3.5, 3.8, 3.0, 4.3, 6.0, 5.0, 7.5 | 3.3, 1.6, 1.6, 1.8, 1.7, 2.1 | 4.14 (10) | 15.8, 3.3 / 2.1 / 2.4, 4.03 (9) |
| t = 200 | 12.7 (8.7) | 2.4, 4.1, 4.0, 3.8, 3.5, 3.0 | 3.0, 1.5, 1.9, 1.5, 1.4, 1.7 | 4.03 (10) | 15.6, 3.0 / 1.9 / 2.7, 3.62 (8) |
| t = 399 | 12.9 (8.9) | 1.7, 2.6, 3.4, 4.7, 5.1, 7.0, 7.5, 9.0 | 3.1, 1.7, 1.9, 1.6, 1.8, 1.0 | 3.67 (9) | 15.8, 2.8 / 2.3 / 2.5, 3.61 (7) |
| all 9 snapshots | | 15–45 % of the elliptic rows are CONE at iteration 1 | 2.8, 1.5, 1.7, 1.7, 2.0, 1.8 | **3.72** | flips 2.9, 2.1, 2.4, 2.2, 2.4; niter **3.41** |

Reading: 1–5 rows flip QUADRATIC state per world-iteration (max 12–13), the same order as pyramidal; 2–9
rows per world (1–3 contacts, i.e. 15–45 % of the contact rows while walking) sit in the CONE state through
the iterations, and those rows' curvature changes every iteration whatever the flips. On the walking states
elliptic needs 9 % more Newton iterations than pyramidal (3.72 vs 3.41; max 10 vs 9, a few worlds hit the cap
where pyramidal does not; on the falling MuJoCo C states it needs fewer, 2.76 vs 3.07). The extra iterations
run only on the worlds still iterating (the tail: 10 of 64 worlds beyond iteration 5), so they are a small
share of the premium; a warm start in force space (item 4) would at best remove that share and cannot be
made without changing the converged result at the tolerance (the solver converges to the same fixed point
from either start), so it was not built.

### 8.3 Implementations and measurements

| variant (flag) | G1 task ellip10, physics steps/s | Go2 | humanoid | SO-101 |
|---|---|---|---|---|
| pyramidal (`recommended`, reference) | 3,268,888 | 923,850 | 958,995 | 1,065,380 |
| baseline `b630530` (full JᵀDJ rebuild + cone term + Cholesky every iteration) | 2,061,253 (1.59×) | 580,782 | 640,936 | 816,143 |
| mode 1: incremental h + cone term added inside the fused Cholesky launch (`MJW_ELLIPTIC_INCREMENTAL=1`) | 1,738,302 (1.88×) | 606,498 | 613,305 | 820,336 |
| **mode 2: per-entry deltas + cone term → htot, plain register Cholesky (`=2`, default)** | **2,187,925 (1.49×)** | **623,230** | **666,355** | **847,292** |
| mode 2 + zero-row skip in the cone kernel (`MJW_CONE_SKIP_ZERO=1`, archived: the 3 test loads cost more than the skipped ones) | 2,182,409 (−0.3 %) | 607,832 (−2.5 %) | 660,338 (−0.9 %) | 838,682 (−1.0 %) |
| line search: secondary rows skip their loads (`MJW_LS_SKIP_SECONDARY`, on top of mode 1) | 1,739,221 (0 %) | 607,770 (+0.2 %) | 614,622 (+0.2 %) | 820,999 (+0.1 %) |

Per-kernel (G1 task, ms of GPU time per physics step, 4096 worlds): pyramidal 10.89 total (fused
incremental Cholesky 3.38 ×10 + initial Cholesky 1.18, constraint update 1.17, line search 0.54, M
factor 0.95); elliptic baseline 16.86 (Cholesky 6.01 ×11, cone term 2.39, tiled JᵀDJ 1.41, constraint update
1.42, line search 1.14); mode 1 18.98 (fused + cone 7.07 ×10 + 2.10, cone term 2.37); mode 2 15.64
(Cholesky 4.36 ×10 + 1.25, cone term + deltas 2.49, constraint update 1.30, line search 1.19).

In the task's own protocol (`bench_contact_tuning.py`: 4096 worlds, standing start, 3 s, 8 substeps of 2.5 ms,
physics only, 3 interleaved repeats): `recommended` 78,349 env-steps/s, `tau10_impact_hardlimits_ellip10`
**64,519 (1.21×)** with mode 2, against 55,379 (1.42×) at `b630530` and 11,029 (7.07×) before the launch fix:
the elliptic premium on the G1 task is now 21 % physics-only (estimated ~15 % in the full PPO loop, where
physics is ~60 % of the step; the earlier 1.44× loop measurement was with the 1.42× physics).

Reading:
- The "rebuild every iteration" structure was not the cost it looked like: the tiled JᵀDJ rebuild is
  0.13 ms per launch (1.4 ms per step, 8 % of the elliptic step), because the G1's rows fit one or two
  16-row tiles. Mode 1 loses because adding the cone buffer inside the 32-lane fused kernel costs more
  (one extra 48 × 48 tile load per world, no skipping) than the rebuild it removes. Mode 2 wins 6 % by
  moving the flipped-row deltas into the 946-thread-per-world cone kernel and letting the plain Cholesky
  skip worlds with neither cone rows nor flips.
- What remains of the premium (mode 2 vs pyramidal, 4.75 ms of 15.64): the cone term itself 2.5 ms
  (16 %), the Cholesky that worlds with cone rows must run every iteration (+1.0 ms; pyramidal's fused
  kernel skips worlds with no flips, the "stable-state fast path", which is exact only when every active
  row is quadratic), the elliptic line search (+0.65 ms: a quad precompute pass per launch and the cone
  cost evaluations), and the constraint update (+0.13). MuJoCo C's `Lcone` rank-dim updates would replace
  the full refactorization for cone worlds (≈ 9 rank-1 updates of 43² vs 43³/3: ~35 % less work on the
  Cholesky share) but need a rank-1 update of the register-resident factor across iterations, which is
  the Cholesky kernel itself (out of scope here); estimated ≤ 1 ms of the 4.75.
- Fold into the tiled JᵀDJ build (item 2): with the rebuild at 0.13 ms/launch and the cone term at 0.22
  ms/launch as a separate per-entry pass, folding the cone term into the 16-row tile loop (two extra
  small tile matmuls per cone contact per world) is estimated at ≥ 0.5 ms/launch on the G1 (§2.2); not built.
- Line search (item 3): the elliptic kernel has the same launch count as pyramidal (10 per step); its
  extra cost is the per-launch quad precompute and the cone evaluations on primary rows; skipping the
  secondary rows' contact loads measured within noise (kept as `MJW_LS_SKIP_SECONDARY`, on the
  `metalsim-elliptic2-ls` branch, not merged). MuJoCo's exact-solution shortcut (quadratic-only fast
  exit) is the stable-state fast path already in the fork; it cannot apply while any row is in the CONE
  state (non-quadratic cost along the ray).
- Solver conditioning (item 4): §8.2 (fewer iterations than pyramidal on the same states).

### 8.4 Physics unchanged (measured)

- CPU device (`elliptic_cpu_check.py`, fused path enabled with `MJW_FUSE_H_CHOLESKY_CPU=1`), max |dq| vs
  `mj_step` over 40 steps: Go2 1.8e-7 … 2.3e-6 rad, SO-101 2.4e-8 … 8.3e-6, humanoid 1.6e-7 … 1.9e-6 at
  10 steps (chaotic fall after), G1 Menagerie (sparse, unchanged path) 6.7e-8 … 4.3e-7: the same envelope for
  the rebuild (`=0`), mode 1, mode 2 and mode 2 + zero-row skip (modes 1 and 2 give the same digits).
- Fork CPU tests (`solver_test`, `forward_test`, `constraint_test`, `--cpu`, fused path on): 242 passed, 10 skipped.
- Metal, graph replay, 512 worlds × 200 steps vs the unpatched snapshots (`elliptic_check.py`; floor = two
  instances of the same configuration): mode 1: Go2 1.5e-6 / 2.5e-6 / 5.3e-5 rad at steps 25 / 100 / 200 (floor
  2.0e-6 / 2.8e-6 / 5.3e-5), G1 sparse 5.5e-6 / 2.9e-2 / 0.61 with 0 / 4.9 / 16.2 % of worlds apart (floor
  3.9e-6 / 3.4e-2 / 0.63, 0 / 5.3 / 15.8 %: the chaotic falling G1), G1 dense 5.4e-6 / 3.7e-2 / 0.46, 0 / 6.1 /
  15.0 % (floor 5.5e-6 / 3.4e-2 / 0.62, 0 / 5.3 / 14.3 %); MuJoCo C oracle identical to three digits (Go2
  3.08e-7 … 9.22e-5; G1 5.91e-8 … 3.48e-6). G1 task model (256 worlds × 100 control steps of 8 substeps,
  ellip10): within its own floor (the PD-to-random-targets G1 falls; 26.6 % of worlds apart at step 100 for
  patched-vs-patched and patched-vs-base alike; the base-vs-base floor 16 %). Mode 2 (`e2_check_sz*`): Go2
  1.8e-6 / 2.2e-6 / 9.4e-4 vs base (floor 2.0e-6 / 3.2e-6 / 5.2e-5; 0 % of worlds apart), G1 sparse 5.6e-6 /
  2.8e-2 / 0.61 with 0 / 5.9 / 15.4 % (floor 6.6e-6 / 3.4e-2 / 0.61, 0 / 5.5 / 15.0 %), G1 dense 2.4e-6 / 3.4e-2 /
  0.62 with 0 / 5.3 / 16.2 % (floor 2.0e-5 / 3.4e-2 / 0.34, 0 / 4.1 / 15.6 %), G1 task 0.8 / 28.1 % (floor 0.8 /
  26.2 %); oracles identical to three digits (Go2 1.89e-7 … 9.21e-5, G1 5.91e-8 … 3.46e-6; the task-model
  oracle now steps MuJoCo C 8 times per call: 9.9e-8 at 4 ms, 1.6e-2 at 16 ms as the robot falls).
- **Fork test suite on Metal with mode 2 (`e2_pytest_sz`): 1451 passed, 39 skipped, 1 failed** — the
  pre-existing MuJoCo-nightly `test_put_data_nefc_zero_dense` (§7.4); `test_hfield_maxconpair` passes since
  `b630530`.

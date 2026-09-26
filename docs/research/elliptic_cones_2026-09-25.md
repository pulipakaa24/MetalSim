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
marginal); falls of the PhysX-trained checkpoint 142 → 160–171 (+13–20 %, the one metric that moves away).
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
stdout `runs/il3/g1_flat_flatcfg_ellip10.stdout`. Comparison targets: +28.4 (default contacts), +27.85
(`recommended`), and `runs/il3/g1_flat_flatcfg_com.log` (queued ahead of it). Not finished at the time of writing.

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
many distinct envs fell. **Results: pending (queued as `g1_falls1` / `g1_falls2`, render class, behind a
training job); filled in below when they land.**

### 7.3 `so101_lift.py` ctrl ordering

Finding corrected: the env is already event-ordered (see §6). Added `tests/test_lift_ctrl_ordering.py`
(needs Metal; queued as `lift_ordering_test`): (a) per `step()`, the calls are exactly `signal(learner event)`
→ `sim.wait(learner event, same value)` → physics → `sim.after(v)`, three steps in a row, and the sim's `ctrl`
equals what `step()` computed; (b) 30 random-action steps with no host sync give the same `qpos` (1e-4) as
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
  mesh path unchanged) and `test_plane_convex_contacts.py` against that branch: queued (`hfield_gate_tests2`).
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

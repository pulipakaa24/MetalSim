# Throughput without fidelity cost, running log (2026-09-26)

Standing task: find and land throughput improvements anywhere in the stack that change no result (physics states
within the measured run-to-run floor, rewards / observations / renders bit-identical, learning untouched); anything
that changes a result is reported as an option, not landed. Every number is labelled measured / estimated with its
scope. GPU numbers go through the timing queue (`scripts/gpu_run.sh ... timing`, idle GPU logged at both ends);
logs in `runs/tp26/`. Fork work in the worktrees `upstream/mujoco_warp-tp` and `upstream/warp-innate-tp` (branches
`metalsim-tp`), imported through `PYTHONPATH` inside the jobs (the import path is printed in every log); the shared
checkouts are untouched. Starting point: MetalSim `05cc873`, MuJoCo Warp fork `fe6fe71`, Warp fork `4127c48`;
G1 flat task, 4096 envs, 2.5 ms x 8 substeps, `contact_cfg="recommended"` (the task default): full env step
54.1-54.3 K env-steps/s, `solver.solve` 5.7 ms per substep (`docs/research/throughput_regression_2026-09-25.md`).

## 1. Where the preset's 19.5 % goes: Newton iterations and re-factorizations (measured, CPU device)

`scripts/diagnostics/g1_solver_iters.py` hooks `_solver_iteration` on Warp's CPU device (the iteration path is the
Metal one up to float noise) and counts, per Newton iteration over 64 worlds x 64 substeps of random actions from
the init pose (`runs/tp26/solver_iters_cpu.log`):

| preset | solver_niter mean / p90 / max | at the cap | re-factorizations per world-substep (iterations 1-10, initial excluded) | flips per re-factorizing world | worlds with a state change but no quadratic flip |
|---|---|---|---|---|---|
| `default` (MuJoCo contacts and limits) | 2.41 / 4 / 7 | 0 | **1.41** | 3.03 (iteration 1: many with 5-16; later 1-2) | 0 |
| `hardlimits` | 2.82 / 5 / 9 | 0 | 1.81 | 2.54 | 0 |
| `recommended` (= tau10_impact_hardlimits) | 2.95 / 5 / 10 | 10 of 4096 (0.2 %) | **1.93** | 2.25 | 0 |
| `isaaclab3_every_substep_cap20` (cap 20, tol 1e-6) | 2.10 / 4 / 8 | 0 | 1.10 | 2.58 | 0 |

Reading: MuJoCo Warp re-factorizes the full 43 x 43 Hessian (`_update_gradient_h_incremental_cholesky`: delta
update of H, then the register Cholesky and the solve) for every world with any constraint-state change in an
iteration; the preset raises those from 1.41 to 1.93 per world-substep and the mean iteration count from 2.4 to
2.95, on top of the initial factorization every substep. There is never a state change without a quadratic flip,
so reusing a stored factor ("solve only") has no case to serve. (Scope: 64 worlds from the init pose; the 4096-env
random-action state on the GPU has more contacts, nefc mean 14.3 vs 9.8 here.)

Precedent: MuJoCo C 3.14 `engine_solver.c` `HessianIncremental` maintains the factor with rank-1 Cholesky
updates / downdates (`mju_cholUpdate`) per changed constraint and refactorizes only when an update loses rank.
Numerics of the same scheme in float32 on the real H sequence (`scripts/diagnostics/g1_chol_update_numerics.py`,
16 worlds x 48 substeps, recommended preset, `runs/tp26/chol_numerics_budget.log`): search-direction error against
float64, relative, over all iteration solves: refactorization (what MuJoCo Warp does) median 8.4e-8, p99 1.1e-5,
max 2.8e-5; rank-1 updates with at most U updates since the last factorization: U = 1: p99 5.9e-5, max 2.3e-4;
U = 2: 8.0e-5; U = 4: 1.25e-4; U = 8: 1.9e-4; unbounded: 2.3e-4, max 4.8e-4. Even a single float32 update from a
fresh factor carries a 5-10x larger error tail than the refactorization (MuJoCo C runs this in float64, where it
is irrelevant), so the rank-1 path is **not a free-fidelity change at the source** and is reported as an option
(section 6), not implemented.

## 2. Cholesky-side kernels: the register solve landed (+5.2 %), the L'DL lanes rejected

### 2a. Register triangular solve for the Newton Hessian (Warp fork `metalsim-tp` 27e63fd + a5b65da)

`tile_cholesky_solve` with a vector right-hand side ran Warp's cooperative scalar path on Metal: two threadgroup
barriers per row per sweep (172 barriers for the G1's 43 dofs) with the factor and the vector in threadgroup
memory. The fork's register Cholesky already keeps the factor's columns in the 32 lanes' registers; the new
`metal_register_cholesky_solve` (`warp/native/tile_solve.h`) uses the same layout: forward substitution reduces each
row's dot product with `simd_sum`, back substitution broadcasts `x_i` with `simd_shuffle`, no barriers inside the
sweeps. It applies to every `tile_cholesky_solve` of size <= `metal_register_cholesky_max` (48) in a block of <= 32
lanes: the initial and every per-iteration Newton solve of the G1 (pyramidal and elliptic paths alike).
Correctness (`scripts/diagnostics/metal_register_solve_check.py`, Metal, 256 matrices each of n = 8, 31, 32, 43, 48,
upper and lower, block_dim 32 and 16): register vs cooperative path 1.3-3.4e-7 relative (summation order only),
register vs float64 4.0e-7 at n = 48 (cooperative 4.6e-7). Knob: `warp.config.metal_register_solve` /
`WP_METAL_REGISTER_SOLVE=0` (part of the module hash). **Measured** (`runs/tp26/regsolve.wrapper.log`,
`regsolve2.wrapper.log`, interleaved I S I S at 4096, recommended preset, bound capacities): solve alone at n = 43
0.516 -> 0.335 ms per 4096 (n = 32: 0.160 -> 0.106, n = 48: 0.799 -> 0.521), factor + solve 1.146 -> 1.039;
G1 full env step 73.4-73.5 -> **69.8-69.9 ms (55,752-55,795 -> 58,587-58,718 env-steps/s, +5.2 %)**, physics
only 83.3 -> 85.5 K. A compact two-column register layout tried alongside (`metal_compact_register_cholesky`,
bitwise the same factor) measured 5-8 % slower on factor + solve and is archived off (DECISIONS).

### 2b. Lane-parallel sparse L'DL factorization and solve of M (MuJoCo Warp fork `metalsim-tp` 7245794 / 778cf9e): rejected

The fork's one-world-per-thread `_factor_i_sparse_serial` / `_solve_LD_sparse_serial` (2026-09-25, chosen because
MuJoCo Warp's level-per-launch form costs a full drain per level) run 2242 dependent FMAs per world for the G1's
factorization, twice per substep (M, and M - dt D for the implicit integrator), plus two solves: ~1.25 ms of the
8.1 ms substep in the eager per-kernel profile (`runs/mjw_tp/profile8.log`, physics segment: 0.51 and 0.23 ms per
launch, each including ~0.06 ms of eager dispatch overhead). The level schedule (`io.py`) keys a level by the
*target row's* depth, so within a level every update targets a distinct row of that depth and reads deeper rows
only. `_factor_i_sparse_lanes` gives one lane per (target row, element) pair of a level, each applying its row's
updates in list order on the world's factor in threadgroup memory: the per-element arithmetic and its order are
exactly the serial kernel's, the critical path drops from 2242 to 309 dependent FMAs (levels 0-5 are the six
root dofs receiving 37-42 updates each) and one world occupies one SIMD group instead of one lane.
`_solve_LD_sparse_lanes`: forward substitution one lane per target row (list order kept), backward substitution
over a level's updates in parallel (distinct targets). Schedule tables are built once in `put_model`
(`qLD_updates_byrow`, `qLD_lane_pairs`, `qLD_lane_rows` and offsets; a row's pairs never straddle a 32-lane block
because the diagonal lane reads the raw L[k, i] before storing the scaled one and relies on lockstep).
Correctness (`scripts/diagnostics/ldl_lanes_check.py`, Metal, 64 worlds of random poses): factor and solve
**bitwise equal** to the serial kernels on the G1 task (nv 43, 434 nonzeros, 14 levels), the menagerie Go2 (18,
all-sparse layout) and the DeepMind humanoid (27); solve residual against the dense M 3.3e-6 / 8.2e-7 / 5.3e-6.
**Measured slower and archived** (`runs/tp26/regsolve*.wrapper.log`): the first form (factor in a threadgroup
tile) 0.80 vs 0.45 ms per 4096 factorizations and 0.31 vs 0.22 per solve, G1 step 83.6 vs 75.2 ms: Warp's shared-tile
element accesses synchronize per access; rewritten on device memory with one barrier per level (fork 778cf9e, still
bitwise) 0.57 / 0.27 ms, step 72.7 vs 69.9 ms with the register solve. The per-level barriers and the six root
levels with one to six busy lanes cost more than the 7x shorter dependency chain saves. Kept behind
`MJW_METAL_LDL_LANES=1` (default off, fork 2026-09-26); the serial one-world-per-thread kernels stay.

## 3. Profile under the preset (measured, `runs/tp26/prof.wrapper.log`, 04:04, uncontended: granted 04:03:59, released 04:04:45, next job granted a second later)

Register Cholesky cost split at 4096 matrices, graph replay (`scripts/diagnostics/metal_cholesky_parts.py`, installed Warp):

| n | load/store only | factor | solve (stored factor) | factor + solve |
|---|---|---|---|---|
| 32 | 0.072 ms | 0.172 | 0.161 | 0.265 |
| 33 | 0.110 | 0.207 | 0.225 | 0.307 |
| 40 | 0.183 | 0.538 | 0.420 | 0.821 |
| **43 (G1)** | 0.245 | **0.974** | **0.518** | **1.117** |
| 48 | 0.299 | 1.931 | 0.779 | 2.606 |

32 -> 43 costs 5.7x for 2.4x the flops (n^3) and 33 -> 48 9x for 3x: the second column per lane (86 registers at
n = 43) is spilling, the cost is latency per world, not arithmetic. The solve is 0.5 ms on its own (two barriers
per row per sweep in threadgroup memory).

G1 step under `recommended` (graph mode, 20 replays): physics 69.8 ms, whole step 72.9 ms (56,159 env-steps/s
with deterministic actions), `solver.solve` 5.71 ms per substep (105 dispatches), `mjw.step` 9.36 ms (202). Eager
per-kernel attribution of one substep's solve (sum of per-dispatch GPU times, each carrying ~50-60 us of
command-buffer overhead): **recommended 10.97 ms vs null 11.01 ms, the same**, with the fused incremental-H +
Cholesky at 4.66 vs 4.15 ms (x10), the initial Cholesky 2.73 vs 2.81, line search 0.77 vs 0.72,
`_update_constraint_efc` 0.72 vs 0.88. The graph-mode difference (5.71 vs 4.18 ms) is therefore not kernel
work: in graph mode an iteration whose worlds are all converged costs only its dispatches, while an iteration
with a few active worlds costs the full latency of one world's factor + solve per launch (the eager profile hides
this behind its per-dispatch overhead). Under the preset, iterations 5-9 still have 9, 4.5, 1.8, 0.7 and 0.2 % of
worlds active (section 1) where the default setting has 0.9, 0.2 and 0 %: five more launches at full
per-world latency. So the levers are (a) the per-world latency of the fused factor + solve (register spill,
barrier solve) and (b) the dispatch chain per iteration (8 launches), not the amount of arithmetic.
`runs/tp26/itercost.wrapper.log` (marginal graph-mode cost per iteration, k = 0..10) quantifies both.

## 4. Landed: provable capacity bounds for the flat G1 task (measured, `runs/tp26/cap.wrapper.log`, 04:09-04:11)

`metalsim/physics/capacity.py` derives, from the model alone, the largest contact and constraint-row counts any
world can hold: the pairs MuJoCo Warp tests (`nxn_geom_pair_filtered`, contact pairs only) x the largest count each
narrowphase routine writes per pair type (plane-mesh 4, plane-box 8, capsule pairs 2, GJK/EPA pairs 1, or 4 with
MULTICCD since `multicontact` returns a 4 x 3 witness matrix), rows per contact from condim and the cone, plus
limit, equality and friction-loss rows; it refuses heightfields, SDFs and flex (their per-pair counts are capped by
the kernels with an overflow flag, not bounded). G1 flat: 6 pairs x 4 contacts = 24 contacts, 96 + 37 limit rows =
133 -> `njmax` 144, `nconmax` 24 (the task used 512 / 128). `BatchSimOptions(njmax="auto", nconmax="auto")` applies
it; `BatchSim.check_overflow()` raises on any capacity overflow flag and the PPO loop calls it at every log point
(and `metalsim/learn/monitor.py` now checks MuJoCo Warp's real flag names: it looked for "NCON" / "NACON", which
do not exist, so capacity overflows were invisible to the anomaly monitor). Tests: `tests/test_capacity.py` (the
bound holds over 60 random-action steps at 64 envs; a deliberately small capacity raises; unbounded models refuse).

| setting (4096, recommended preset, `--quick`, interleaved D A D A) | physics only | full env step |
|---|---|---|
| task capacities njmax 512 (256 padded) / nconmax 128 | 81,534 / 81,520 | 54,385 / 54,340 (75.3 ms) |
| bound njmax 144 / nconmax 24 | 83,400 / 83,339 | **55,741 / 55,646 (73.5 ms), +2.5 %** |
| Isaac Lab 3.0 preset (`isaaclab3_every_substep_cap20`), task capacities | 71,803 | 56,301 (72.8 ms) |
| Isaac Lab 3.0 preset, bound | 74,480 | **58,287 (70.3 ms), +3.6 %** |

The G1 flat task now uses the bound by default (`g1_velocity.py`; heightfield / box terrains keep their explicit
capacities). Result-neutral by construction: no overflow is possible below the bound, so the rows, their order and
every kernel's arithmetic are unchanged (the earlier +1.8 % row in DECISIONS was the same idea measured under
default contacts without the guard).

## 4c. Rough task capacities: not bounded, guard now in place (finding, not a throughput change)

The rough task (`boxes_local`: 96 per-world box slots around the robot, refreshed every substep) runs with
njmax 256 / nconmax 128 per world. Its model has 288 box-mesh pairs (GJK/EPA with MULTICCD: up to 4 contacts each),
3 heightfield-mesh pairs (kernel-capped) and 3 mesh-mesh pairs: the model bound is above 1,150 contacts and 4,600 rows
per world, so the 128 / 256 capacities are a practical choice (2.6 contacts per world observed), not a guarantee. An
overflow would have been silent until today (`monitor.py` looked for flag names that do not exist); it now raises at
every PPO log point (`BatchSim.check_overflow()`), and `capacity.bounds` refuses the model (heightfield geom) rather
than pretend. Recommendation for the task owner: keep the guard, and size `nconmax` from the largest observed
`nacon` over a training run plus margin.

## 5. Rough terrain under the preset (measured, `runs/tp26/rough.wrapper.log`, 04:11-04:12; first measurement)

| rough (boxes_local, 4096, `--quick`, interleaved R N R N) | physics only | full env step | Newton cap hit |
|---|---|---|---|
| `recommended` (task default) | 63,922 / 63,969 | **45,268 / 45,217 (90.5 ms)** | 23-25 worlds |
| `contact_cfg=None` (the README's rough setting) | 64,355 / 64,370 | 53,843 / 53,988 (76.0 ms) | none |

The preset costs 19 % of the rough step as on flat, all of it in the solve (physics-only, standing robots, is
equal). The rough step profile's "outside physics 24.9 ms" is the state artifact already documented on 2026-09-25
(the physics segment replays without resets, so rough robots come to rest; the full step resets them into
contact-rich states), not task work; the profile's collision-only capture then failed with a Metal out-of-memory
at naconmax 524288 (a diagnostic-script limitation, noted, not pursued).

## 4b. Marginal cost per Newton iteration (measured, `runs/tp26/itercost2.wrapper.log`, `scripts/diagnostics/g1_solve_iteration_cost.py`)

`solver.solve` captured with the iteration cap k = 0..10 on one contact-rich state (20 random-action steps; nefc
mean 7.8 / max 35 under the preset), 20 graph replays each, 4096 envs, bound capacities; installed forks vs the
worktree (register solve; L'DL lanes off):

| k | converged before k (recommended) | marginal ms, recommended: installed / worktree | converged (null) | marginal ms, null: installed / worktree |
|---|---|---|---|---|
| 0 (init: Jaref, Ma, rows, JTDAJ, first factor + solve) | – | 1.62 / 1.50 | – | 1.64 / 1.50 |
| 1 | 0 | 1.43 / 1.27 | 0 | 1.38 / 1.25 |
| 2 | 1006 | 1.01 / 0.93 | 1384 | 0.82 / 0.76 |
| 3 | 2131 | 0.69 / 0.65 | 2613 | 0.46 / 0.42 |
| 4 | 2882 | 0.50 / 0.42 | 3522 | 0.25 / 0.24 |
| 5 | 3405 | 0.39 / 0.36 | 3963 | 0.22 / 0.20 |
| 6 | 3749 | 0.28 / 0.26 | 4083 | 0.10 / 0.13 |
| 7-10 | 3946-4087 | 0.26, 0.22, 0.22, 0.21 / 0.24, 0.22, 0.21, 0.18 | 4096 | 0.07-0.11 / 0.07-0.10 |
| total | | 6.82 / 6.24 | | 5.25 / 4.85 |

Reading: an iteration with no active world costs its 9 launches, 0.07-0.11 ms (about 10 us per dispatch in graph
replay); one with a few active worlds (9-30 of 4096, iterations 8-10 under the preset) costs 0.21 ms: the
dispatches plus the per-world latency of the line search and the fused factor + solve; iteration 1 (all worlds
active, all refactorizing) costs 1.43 ms, of which the 4096 factorizations are ~1.0 ms. Under the preset the
solve is 6.8 vs 5.2 ms here because iterations 4-10 keep 0.2-1.2 K worlds active. The register solve saves 0.58 /
0.40 ms per substep (8 %). Next levers, in this order: (a) the 9 launches per iteration (fusing the five small
per-iteration kernels into one launch: 0.4 ms per substep at 10 us a dispatch, section 2c), (b) the per-world
latency of the n = 43 factorization (the 32 -> 43 cost jump, 0.17 -> 0.97 ms per 4096, is 2.4x the flop ratio;
not register spill: the compact layout did not help; unresolved), (c) the line-search kernel's latency.

### 2d. Rolled register Cholesky (rejected)

Hypothesis from 4b (b): the 33 -> 48 cost growth is instruction fetch of the ~n^2 unrolled code. Test: the same lane
layout and operations with runtime loops (`metal_rolled_cholesky`, `WP_METAL_ROLLED_CHOLESKY`), bitwise the same
factor and solve (`runs/tp26/rolled.wrapper.log`): factor 3.36 vs 0.98 ms at n = 43, 0.66 vs 0.19 at 32, 5.18 vs 1.94
at 48. Rejected: thread-memory arrays cost far more than the unrolled code; the register form is the right one and
its n = 43 cost is unexplained by these two hypotheses (register pressure limiting resident SIMD groups remains the
candidate; a 64-lane / two-SIMD-group form with one threadgroup exchange per column is the untested option).

### 2c. Fused per-iteration update launch (MuJoCo Warp fork `metalsim-tp`, in measurement)

`_update_constraint_gradient_fused`: the five launches between the line search and the Hessian update (zero the
change counters, `_update_constraint_efc`, `qfrc_constraint = J^T force`, `_update_gradient_zero_grad_dot`,
`_update_gradient_grad`) as one 32-lanes-per-world launch on Metal (dense Jacobian, pyramidal cones, the incremental
path). Row forces / states and the per-dof sums keep each kernel's arithmetic and order (bitwise); `grad_dot` becomes
a SIMD-group sum of per-lane partials instead of atomic adds in arbitrary order (float noise, as the atomics were).
9 -> 5 launches per iteration. Knob `MJW_METAL_FUSE_UPDATE=0`. **Measured** (`runs/tp26/fuse.wrapper.log`, interleaved F U F U,
4096, recommended, bound capacities, register solve on in both): full env step 69.9-70.1 -> **67.4 ms
(58,439-58,589 -> 60,762-60,803 env-steps/s, +3.9 %)**, physics only 85.5-85.9 -> 90.2-90.3 K; the solve per substep
6.24 -> 5.92 ms (iterations 6-10 now 0.15-0.23 ms each, from 0.18-0.26). Landed (fork dd42c23). State-difference
protocol: `runs/tp26/check*.wrapper.log`.

## 7. The learning loop outside physics (measured, `runs/tp26/loop.wrapper.log`, `loop2.wrapper.log`, 4096 envs)

Warp policy `act()` (actor + critic + sampling) per step: flat (obs 123, 256-128-128) 3.03 ms, rough (obs 310,
512-256-128) 12.41 ms (14 % of the rough step). The committed layer kernel maps one thread per (env, output),
outputs fastest (mapping A). Bit-identical alternatives (every output keeps its sequential dot product; only the
thread-to-work mapping changes; `scripts/diagnostics/warp_policy_cost.py`, all bitwise equal to A):

| actor forward per step | A (env, output) | B (output, env) | C (env, 4 outputs) | **D (4 outputs, env)** |
|---|---|---|---|---|
| flat | 1.551 ms | 1.813 | 1.468 | **0.855 (1.8x)** |
| rough | 6.227 ms | 6.779 | 15.406 | **3.039 (2.0x)** |

D landed as `mlp_layer4` (`metalsim/learn/warp_policy.py`, `METALSIM_MLP_MAPPING=A` restores the original;
`tests/test_warp_policy.py::test_layer_mapping_bitwise`). **Measured** on the full loop (`runs/tp26/fuse.wrapper.log`,
installed forks, interleaved D A D A): rollout + inference 74.2 -> 72.7-72.9 ms per step (55,191-55,197 ->
56,201-56,313 env-steps/s), full PPO loop 50,912-50,925 -> **51,877-51,918 (+1.9 %)**; the step itself unchanged
(73.6-73.7 ms). Rough (12.4 -> ~6 ms of act() per step) not re-measured.

PPO update (torch on MPS, 5 epochs x 4 minibatches of 24,576 rows, 167 K parameters): 151 ms per iteration, 8 % of
the loop. Removing the adaptive-KL `.item()` per minibatch (a host sync) saves 18 ms, `clip_grad_norm_` 15 ms
(no foreach on MPS), the GAE recursion is 0.8 ms, forward + backward 101 ms, Adam ~0. Both removable pieces change
results (the lr schedule, the clipping), so they are options (each ~1 % of the loop), not changes; the
forward/backward is compute (torch.compile changes numerics; not tried).

## 5b. Physics identity evidence for the landed fork changes (measured, `runs/tp26/check.wrapper.log`, `check_base.npz` / `check_wt.npz`)

`scripts/diagnostics/g1_tp_check.py` (512 envs, 50 control steps, deterministic actions, the task default preset,
bound capacities): installed forks vs the worktree (register solve + fused per-iteration launches), graph replay:

| control step | qpos max | qpos p99 | qpos median | obs max | reward max |
|---|---|---|---|---|---|
| 1 | 1.96e-6 rad | 1.2e-7 | 0 | 5.3e-4 | 6.3e-7 |
| 2 | 9.6e-6 | 1.1e-6 | 1.9e-9 | 1.1e-3 | 2.0e-6 |
| 5 | 3.4e-3 | 3.0e-6 | 3.0e-8 | 0.84 | 1.3e-3 |
| 50 | 1.8 (chaotic, as two instances of one configuration: 9.2 rad in `runs/mjw_tp/final_checks.log`) | 2.4e-2 | 8.2e-8 | 9.4 | 4.1 |

The register solve changes the summation order of every Newton solve in every world, so the one-step difference
(max 2e-6 rad, p99 1.2e-7) is above the atomics-only floor of two identical instances (max 1.3-1.5e-7) but is
float32 noise (1e-7 relative) and far below the protocol's p99 criterion (~1e-3). MuJoCo C protocol (4 worlds, PD
hold + fixed random targets, |dq| max / median): installed 4.08e-6 / 1.3e-8 at 0.02 s, 4.41e-6 / 2.2e-8 at 0.04 s,
0.156 / 0.015 at 0.08 s, 0.244 / 0.061 at 0.5 s, 0.466 / 0.12 at 1 s; worktree 4.08e-6, 4.41e-6, 0.156, 0.237,
0.456 (the same envelope; the divergence after 0.08 s is the chaotic random-target protocol, identical for every
configuration measured on 2026-09-25). Capacity / overflow unchanged (nefc max 35 of 144, ITERATIONS cap hits 31-36
of 512, LS 476-481 in both). The fused launches alone (`runs/tp26/check2.wrapper.log`, fused vs unfused within the worktree): step 1 qpos max
1.38e-6 (the two configurations' own two-instance floors: 1.51e-6 and 1.85e-6), step 5 1.08e-3 (floors 3.4e-3 /
1.08e-3), step 50 1.80 (floors 1.63 / 1.80), MuJoCo C protocol identical at 0.02-0.24 s (4.08e-6, 4.41e-6, 0.156,
0.256) and within the envelope after: the fusion is inside the run-to-run floor at every step.

## 8. Elliptic cones: the cone term as rank-1 updates of the stored factor (coordinator item; in measurement)

With the elliptic incremental path (fork head 9b4e96a, mode 2) the remaining elliptic premium on the G1 task is a
per-entry cone term plus a full register Cholesky of htot = h + cone every iteration for worlds with CONE rows.
MuJoCo C (`engine_solver.c` `HessianConeUpdate`) copies L into Lcone and applies dim rank-1 updates per cone contact
with v = L_con' Z (L_con the Cholesky of the contact's local curvature). Implemented on `metalsim-tp` (MuJoCo Warp
2d37... `MJW_ELLIPTIC_CONE_UPDATE=1`, Warp `tile_cholesky_update_inplace` with a register form: lanes own rows, L[k,k]
and x[k] broadcast by SIMD shuffle, no barriers): `_cone_vectors` builds the vectors per world (worlds above
`MJW_CONE_UPDATE_KMAX` = 6 cone rows keep the per-entry path), the factor of h is stored in `ctx.hfactor` and reused
across iterations without a quadratic flip. Cone statistics on the G1 task with `tau10_impact_hardlimits_ellip10`
(CPU device, `runs/tp26/solver_iters_ellip10.log`, 64 worlds x 64 substeps): 87 % of solving world-iterations have
cone rows, 65 % of those with no quadratic flip (a factor reuse + updates instead of a refactorization); rows per
cone world: 3-4 in 40 %, 5-6 in 30 %, 9-12 in 25 %, 13-24 in 5 %.

CPU-device numerics (`scripts/diagnostics/cone_update_cpu_check.py`, 16 worlds x 24 substeps, 1112 iteration solves):
search-direction error against the float64 solve of h + cone term: mode 2 (per-entry + refactor) median 5.4e-6,
p99 1.4e-3, max 7.1e-3; mode 3 (rank-1 updates) median 2.2e-6, p99 1.7e-3, first version max 0.42 in one solve
(the local Cholesky's absolute 1e-15 pivot floor in float32: a near-zero pivot blows its column up), replaced by a
relative tolerance (1e-6 of the largest local diagonal) that drops the direction: median 2.1e-6, p99 2.0e-3, max 2.1e-2
(worlds with 1-3 vectors p99 6.2e-3; the per-entry path's own tail is 1.4e-3 / 7.1e-3: these cone Hessians are
ill-conditioned in float32 on either path). Trajectories mode 3 vs mode 2: |dq| max 1.2e-7 / 3.6e-6 / 3.4e-6 rad
after 1 / 2 / 3 control steps (mode 2 vs itself on the CPU: 0; the Metal run-to-run floor at step 2 is 8e-6). Metal check and throughput on the G1
task (ellip10), Go2, humanoid, SO-101: `runs/tp26/cone.wrapper.log`.

## 6. Options that change results (reported, not landed)

- Rank-1 Cholesky updates of the Newton Hessian factor (MuJoCo C's scheme) in float32: numbers in section 1.
  Estimated gain if it were acceptable: the per-iteration re-factorization (about 1.9 ms of the 4.2 ms solve under
  default contacts, more under the preset) would shrink by roughly the ratio of a rank-1 update (O(n^2), 1-3 per
  refactorizing world) to the factorization (O(n^3 / 3)), i.e. most of it; not measured.

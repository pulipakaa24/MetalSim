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

A third form, one lane per single-child chain of the dof tree per level (the G1: 12 chains in 4 levels, dependent
chain 105 updates instead of 391; the solve keeps the serial summation order bitwise, the factor's cross-chain
updates are atomic as upstream's CUDA `_qLD_acc`; fork cdc4fd4 / e068734, `MJW_METAL_LDL_CHAINS=1`), correct to
6e-7 relative (`runs/tp26/chains2.wrapper.log`), is also slower: factor 0.65 ms, solve 0.37, G1 step 72.9 vs
67.2 ms. Three parallel forms lose to the serial kernel: the per-world work is memory-latency bound (391
dependent updates at ~1 us each), and the SIMD-group forms add barriers while most lanes idle. Left as the
residual (section 9).

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
133 -> `njmax` 144, `nconmax` 24 (the flat task ran with 256 / 32: its flat branch takes the heightfield
capacities; the code says 512 / 128 for other terrains). `BatchSimOptions(njmax="auto", nconmax="auto")` applies
it; `BatchSim.check_overflow()` raises on any capacity overflow flag and the PPO loop calls it at every log point
(and `metalsim/learn/monitor.py` now checks MuJoCo Warp's real flag names: it looked for "NCON" / "NACON", which
do not exist, so capacity overflows were invisible to the anomaly monitor). Tests: `tests/test_capacity.py` (the
bound holds over 60 random-action steps at 64 envs; a deliberately small capacity raises; unbounded models refuse).

| setting (4096, recommended preset, `--quick`, interleaved D A D A) | physics only | full env step |
|---|---|---|
| task capacities njmax 256 / nconmax 32 (the flat task's values: its flat branch takes the heightfield capacities) | 81,534 / 81,520 | 54,385 / 54,340 (75.3 ms) |
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
after 1 / 2 / 3 control steps (mode 2 vs itself on the CPU: 0; the Metal run-to-run floor at step 2 is 8e-6).
Final CPU numbers against a fresh mode-2 reference (`runs/tp26/coneupd_mode3.log`, `coneupd_k12.log`, the first
reference file was stale): KMAX 6: median 2.3e-6, p99 4.5e-3, max 2.4e-2, trajectories 1.2e-7 / 3.6e-6 / 3.5e-6 rad;
KMAX 12: p99 8.0e-3, max 9.4e-2 (7-12 vectors p99 3.2e-2); KMAX 0 and a reuse-disabled variant are bitwise
mode 2 over 160 iteration records (the kernels are exact re-implementations of the per-entry path). Kernel form
after the first Metal run: a single matrix tile per world (the two-tile form asked 39 KB of threadgroup memory,
above Metal's 32 KB) prepared by `_cone_update_prepare` (h + cone for worlds above KMAX, h for a refactorization,
the stored factor for a reuse); the stable-state skip mirrors mode 2's.

**Measured on Metal and rejected** (`runs/tp26/cone2.wrapper.log`, 07:36-07:41, interleaved 0 1 0 1, 4096 worlds,
elliptic cones, physics steps/s): G1 task ellip10 2,606-2,607 K -> 2,180 K (-16 %), Go2 644-646 K -> 615 K (-5 %),
DeepMind humanoid 704-705 K -> 533 K (-24 %), SO-101 866-869 K -> 774-775 K (-11 %). Per-kernel profile of the G1
task with the update path (15.85 ms GPU per physics step): the factor / update / solve launch 4.25 ms over 10
iterations (0.42 ms each, no cheaper than a refactorization: the k x 43-column sqrt / divide chain of the rank-1
recurrence has the factorization's latency on one SIMD group), the prepare pass 2.36 ms (the stored factor
travels through htot), `_cone_vectors` 0.56. Physics within the floors: Go2 (elliptic_check, 512 worlds, 200
steps) 3.8e-7 rad at step 1 growing to 9.4e-4 at step 200, the reference's own two-instance value; G1 task
(g1_tp_check, ellip10, 512 envs) step 1 8.2e-7 (floors 6.3e-7 / 2.1e-6), step 50 0.90 (floors 0.85 / 0.82), MuJoCo C
protocol identical to 0.5 s. Archived off (`MJW_ELLIPTIC_CONE_UPDATE=1`); MuJoCo C's scheme pays in double on a
CPU where a rank-1 update is 40x cheaper than a refactorization, not on a 32-lane register factorization whose
cost is latency rather than flops. Metal check and throughput on the G1
task (ellip10), Go2, humanoid, SO-101: `runs/tp26/cone.wrapper.log`.

## 9. Before / after, everything landed today (measured, `runs/tp26/final.wrapper.log` and the interleaved A/Bs above)

All at 4096 envs, 2.5 ms x 8, `contact_cfg="recommended"`, `g1_tp_variants.py` (full: physics only, full env step,
rollout + inference, PPO update, full PPO loop), timing queue, idle GPU logged. "Before" is the day's start
(`runs/mjw_tp/cc.wrapper.log` R1 / R2, 2026-09-25 evening, the same code as this morning: MetalSim 98a41c1,
forks fe6fe71 / 4127c48, task capacities njmax 256 / nconmax 32, policy mapping A). "After" is the worktrees
`metalsim-tp` (register solve, fused per-iteration launches; L'DL lanes / chains, compact layout, rolled form
and the cone update off) with the capacity bound and the policy mapping D (F1 / F2, 06:40).

| | physics only | full env step | rollout + inference | PPO update | full PPO loop |
|---|---|---|---|---|---|
| before (R1 / R2) | 80,982 / 81,008 (50.6 ms) | 54,244 / 54,094 (75.5-75.7 ms) | 54,163 / 53,830 | 149 / 159 ms | 50,055 / 49,509 |
| after (F1 / F2) | **90,395 / 90,404 (45.3 ms)** | **60,775 / 60,946 (67.2-67.4 ms)** | **61,397 / 61,687 (66.4-66.7 ms)** | 143 / 144 ms | **56,351 / 56,590** |
| gain | +11.6 % | **+12.4 %** | +14 % | – | **+13.3 %** |

Attribution from the interleaved A/Bs (env step): capacity bound +2.5 % (75.3 -> 73.5 ms), register solve +5.2 %
(73.5 -> 69.9), fused launches +3.9 % (69.9 -> 67.4); rollout: policy mapping D +2.0 % (74.2 -> 72.8 ms per step
before the fork changes). The same job's B runs, the installed forks (now 9b4e96a) with njmax 512 / nconmax 128
forced, measure 84.0-84.6 ms per step (48,441-48,767; loop 48,060-48,147): the capacity-sized launches cost 12 %
between 256 / 32 and 512 / 128, which is why the bound (144 / 24) and the overflow guard matter beyond the 2.5 %.

Physics identity: section 5b (installed vs worktree within the run-to-run floor; the MuJoCo C protocol identical
to 0.24 s), the fused launches alone within their own floors, the capacity bound exact by construction, the policy
mapping bitwise. Learning untouched (no reward, observation, reset or update change).

## 10. Residual list (estimated; what would still pay and what it would cost)

1. The register Cholesky at n = 43: 0.97 ms per 4096 factorizations against 0.17 at n = 32 (5.7x for 2.4x the
   flops); 2 full launches + the partial later iterations are ~3 ms of the 5.9 ms solve per substep, ~35 % of the
   step. Not register spill (compact layout) nor instruction fetch (rolled form). Untested: a 64-lane form (two
   SIMD groups per world, one threadgroup exchange per column). Estimated 5-10 % of the step if it halved.
2. The sparse L'DL of M (2 factorizations + 2 solves per substep, 1.3 ms, 16 % of the step): three parallel forms
   lost to the serial one-world-per-thread kernel (sections 2b, 9); the per-world chain is memory-latency bound.
   Untested: a dense-tile factorization of the 6 x 6 root block plus per-branch serial chains in one thread each
   with the root updates deferred (no barriers); estimated <= 5 %.
3. Iterations with few active worlds (5-10 under the preset) cost 0.15-0.23 ms each (5 launches + one world's
   line search and factor + solve latency): ~1 ms of the 5.9 ms solve; only a shorter per-world latency (1) or
   fewer launches (fusing the line search's mul_m, ~0.1 ms per substep, 1 %) reach it.
4. Line search latency (20 bracketing iterations x 3 evaluations over <= 35 rows per world): not measured in
   isolation; estimated 1-2 % of the step.
5. The PPO update (8 % of the loop): the adaptive-KL `.item()` (18 ms) and `clip_grad_norm_` (15 ms) change
   results (options, section 7); the forward/backward (101 ms) is torch MPS compute.
6. Rough terrain: act() 12.4 -> ~6 ms per step with the policy mapping (7 % of the rough step, estimated from
   the isolated kernel numbers, not re-measured on the rough loop); the rough capacities (256 / 128) are not
   model-bounded (section 4c), so no bound can be applied there.
7. Elliptic cones: the rank-1 cone update (section 8) measured 5-24 % slower on every model and is archived;
   the remaining elliptic premium (per-entry cone term 2.5 ms, refactorization 1.0 ms of 15.6 ms per G1 step) has
   no cheaper exact form on this hardware that was found today.

## 6. Options that change results (reported, not landed)

- Rank-1 Cholesky updates of the Newton Hessian factor (MuJoCo C's scheme) in float32: numbers in section 1.
  Estimated gain if it were acceptable: the per-iteration re-factorization (about 1.9 ms of the 4.2 ms solve under
  default contacts, more under the preset) would shrink by roughly the ratio of a rank-1 update (O(n^2), 1-3 per
  refactorizing world) to the factorization (O(n^3 / 3)), i.e. most of it; not measured.

## 11. Round 2 (coordinator list, 2026-09-26 morning)

### 11.1 Persistent per-world iteration loop (estimate, not built yet)

Upstream MuJoCo Warp (and Newton through it) exits early on CUDA with `wp.capture_while` (a conditional graph node
re-launching `_solver_iteration` while `nsolving > 0`); Warp's Metal backend has no conditional node and MetalSim
sets `graph_conditional = False`, so all 10 iterations are launched and converged worlds exit per thread. A
per-world persistent loop is expressible on Metal: one 32-lane kernel per world running `while not done:` over
the fused iteration body (shared tiles inside data-dependent loops are freed per scope by the fork's arena, and
MuJoCo Warp already loops tile ops in the JTDAJ kernel). What it can and cannot save, from the measured marginal
costs (4b): an iteration with no active world costs 0.07-0.10 ms (its 5 launches after the fusion), one with a few
active worlds 0.15-0.23 ms, so the launch share of the tail (iterations 5-10 under the preset) is ~0.07 ms x 6 =
0.4 ms per substep, ~3.4 ms per step (5 %); the other ~0.1 ms per iteration is the slowest world's own line
search + factor + solve latency, which a persistent loop pays just the same (it cannot finish before its slowest
world's remaining iterations). Cost: porting the pyramidal line search (bracketing over 20 iterations with three
tile reductions each), `mul_m`, the fused update and the fused H + Cholesky into one kernel body, with the
Cholesky's 86 registers plus the line-search state resident together (occupancy risk). Estimated gain <= 5 % of
the step for a large port; deferred behind the per-world-latency items (11.2, 11.3), which also shrink the tail.
An alternative general mechanism is a GPU-side edit of the replayed indirect command buffer (Metal lets a kernel
re-encode a `compute_command`'s dispatch size), i.e. a real early exit for every iteration kernel without
porting anything; it needs the Warp backend to expose the ICB commands to a kernel (not built; estimate the same
5 % plus the empty-iteration dispatches of the default setting).

### 11.2 The 43-dof register Cholesky: 64-lane form (built, in measurement) and the block split (already measured)

64-lane form (Warp fork `metalsim-tp` 2834cf31, `WP_METAL_CHOL64`; MuJoCo Warp `MJW_METAL_CHOL_LANES=64`): two
SIMD groups per world, one column per lane (43 registers instead of 86), the owner's unscaled column and pivot
exchanged through the tile's own column storage with one threadgroup barrier per column, every lane forming the
same `column[i] * (1/d)` products the 32-lane form broadcasts (bitwise expected), the register solve run by group
0 with group 1 idling at the barriers. **Measured** (`runs/tp26/chol64.wrapper.log`): bitwise equal to the 32-lane
form at n = 16..48 (factor and solve); cost per 4096, 32 vs 64 lanes: n = 43 factor 0.989 vs 0.998 ms, solve 0.326 vs
0.239, factor + solve 1.030 vs 1.054; n = 48: 2.158 vs 2.052; n = 32: 0.200 vs 0.419. No gain at the G1's size: the
43 threadgroup barriers cost what the halved per-lane work saves. Archived (DECISIONS). The n = 43 factorization
stays the largest residual; what is left untried is a two-column-per-lane form whose second column starts at row
32 *and* whose shuffle loop is split per column (the compact layout kept the two-column loop), and a two-world-
per-threadgroup form to raise resident SIMD groups per core.

Block split (31 body/leg + 12 finger dofs): the finger rows of H couple to their arm chain, the torso and the six
root dofs, i.e. a full 12 x ~16 off-diagonal block, not low rank; the exact form is the Schur complement, measured
on 2026-09-25 (core 19 + arms 24, tile ops): 1.35 vs 1.11 ms per 4096 solves, rejected (DECISIONS 2026-09-25,
`scripts/diagnostics/metal_schur_cost.py`). Not repeated.

### 11.3 Sparse L'DL of M: the per-model unrolled register form (built, in measurement)

The estimate that motivated it: the serial kernel's 0.45 ms is 391 dependent updates at ~1 us each (device
memory round trips per element on one thread with ~3 SIMD groups resident per core); the same dependency chain
with the factor in registers costs a shuffle-latency per update (~10 ns) if every register index is a compile-time
constant. Design (MuJoCo Warp fork `metalsim-tp` f4276b0, `MJW_METAL_LDL_UNROLLED`, default on when it measures):
lane j of a 32-lane world holds position j of every row of the factor (43 registers), the model's update list is
emitted as straight-line native code (a `wp.func_native` snippet generated per model layout: 4,087 lines for the
factor, 1,785 for the solve), each update two `simd_shuffle`s (L[k,i], L[k,k]), one divide, one predicated FMA,
no barriers, no threadgroup memory, no dynamic indexing; the solve keeps x replicated in every lane. The same
operations in the same order as the serial kernels: bitwise on the CPU scalar branch (Metal:
`runs/tp26/unrolled.wrapper.log`). Estimated 0.45 -> ~0.15 ms per factorization and 0.21 -> ~0.05 per solve,
about 7 ms of the 67 ms step.

### 11.4 Rough loop with everything landed (measured, `runs/tp26/rough2.wrapper.log`, 08:33)

Rough terrain (boxes_local, recommended preset, task capacities 256 / 128, the installed forks now at the merged
heads, policy mapping D), full `g1_tp_variants`, two runs: physics only 69,458 / 69,499 (58.9-59.0 ms), full env
step **49,013 / 49,100 (83.4-83.6 ms)**, rollout + inference 51,380 / 51,420 (79.7 ms per step), PPO update 296-300 ms,
full PPO loop **44,449 / 44,504**. Against the morning's first rough measurement under the preset (section 5:
physics 63,922-63,969, step 45,217-45,268 at 90.5 ms): physics +8.7 %, step +8.4 %; the loop had not been measured
under the preset before (the README's 41.9 K rough loop is the default-contact setting). Rough gains less than flat
because its physics is 71 % of the step (collision against 96 box slots per world, the 187-ray height scan) and
its capacities are not model-boundable. The flat loop in the same job: 56,079 (step 60,540), as in section 9.

### 11.5 Camera-RL path (measured, `runs/tp26/camera.wrapper.log`, 08:37-08:40)

Isaac-Cartpole-RGB (1024 envs, tier 0, Isaac's skrl configuration), 458,752 env-steps of training per run
(34-39 s), landed forks vs the day's starting forks (07a51a6 / f194006a through `PYTHONPATH`), interleaved: after
13,369 / 12,931 env-steps/s including training, before 12,655 / 11,802. The 4-7 % difference is inside the spread
of two runs of the same configuration at this length (13,369 vs 12,931; 12,655 vs 11,802) and the physics is
0.6-0.8 % of this step (the update is 66-70 %, the policy's torch CNN inference 29-34 %; the Warp MLP mapping does not
apply here), so nothing is claimed for the camera path. No render-side code changed today, so the renderer's
SHA identity for a given state is unaffected by construction; end-to-end frames differ only through the
float-noise state differences documented in 5b.

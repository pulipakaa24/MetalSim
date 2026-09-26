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

## 4b. A/B measurements pending

- `runs/tp26/regsolve.wrapper.log` (done, section 2 below) and `regsolve2.wrapper.log` (rerun after the fixes).
- `runs/tp26/itercost.wrapper.log`: marginal graph-mode cost per Newton iteration.
- `runs/tp26/check.wrapper.log`: state-difference protocol, installed vs worktree.
- `runs/tp26/tests*.wrapper.log`: fork test modules and MetalSim tests.

## 6. Options that change results (reported, not landed)

- Rank-1 Cholesky updates of the Newton Hessian factor (MuJoCo C's scheme) in float32: numbers in section 1.
  Estimated gain if it were acceptable: the per-iteration re-factorization (about 1.9 ms of the 4.2 ms solve under
  default contacts, more under the preset) would shrink by roughly the ratio of a rank-1 update (O(n^2), 1-3 per
  refactorizing world) to the factorization (O(n^3 / 3)), i.e. most of it; not measured.

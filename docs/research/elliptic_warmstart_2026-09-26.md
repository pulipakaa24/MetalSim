# Cutting the elliptic-cone premium without changing results: fused elliptic iteration, ICB early exit, line-search noise floor, warm start (2026-09-26)

Scope: the G1 default is elliptic friction cones, impratio 10, Newton cap 20 (`contact_cfg="recommended"`), measured
at 1.68× the full-PPO-loop cost of the pyramidal preset (`recommended_pyramidal`, cap 10): loop 34.3 K vs 57.6 K
env-steps/s at 4096 envs (`runs/tp26/ellip.wrapper.log`). This note is the work on the owner's question "can that
premium be cut without changing the converged result", reprioritised mid-way by the independent review
(`elliptic_cost_review_2026-09-26.md`). Every number is **measured** (this machine, M4 Max 40-core GPU, 64 GB; GPU
runs through `scripts/gpu_run.sh ... timing` on an idle device, importing frozen per-job worktrees) unless marked
**estimated**. Raw logs: `runs/ellip_ws/`. Fork branch `metalsim-ellip-ws` (MuJoCo Warp `pulipakaa24/mujoco_warp`,
on 0a9de8e; Warp `warp-innate` on 9abceff9). Scripts added: `scripts/diagnostics/competitors/g1_newton_iters.py`
(MuJoCo C and Warp-CPU Newton statistics on dumped states), `scripts/diagnostics/ellip_fused_cpu_check.py` (fused vs
unfused, bitwise, CPU), `scripts/diagnostics/ls_noise_floor_check.py` (line-search floor, CPU),
`scripts/diagnostics/competitors/cap_probe_variant.py` (cap probe for a solver variant); `g1_solve_iteration_cost.py`
gained `"state":"rollout"` and the warm-start-buffer restore.

## 0. Summary

| item | result | changes the converged result? | status |
|---|---|---|---|
| 1. Fused per-iteration launch on the elliptic path (`_update_constraint_gradient_fused` with the CONE rule, the cone list on lane 0, the flipped-row deltas + cone term into `htot` by the same lanes; 128 lanes per world) | launches per Newton iteration **11 → 5**; G1 task ellip10 cap 20 at 4096: physics only 76.0 → 57.0 ms, env step 112.4 → 100.5, rollout + inference 113.6 → 102.3 ms per step, **loop 34.3 K → 37.7 K env-steps/s** (one run each, §3.4; the two-repeat A/B is §6) | no: bitwise the unfused launches on the CPU device over 8 substeps (qacc, qfrc_constraint, efc_force, efc_state, solver_niter, qpos, qvel); on Metal inside the unfused configuration's two-instance floor with identical medians | **landed**, default on Metal (`MJW_METAL_FUSE_UPDATE=0`, `MJW_METAL_FUSE_HTOT=0`, `MJW_METAL_FUSE_LANES`) |
| 2. Exact per-world early exit of the Newton loop in graph replay (Warp fork: ICB indirect execution ranges; MuJoCo Warp: `_solve_done` zeroes the remaining iterations' range lengths) | elliptic cap 20 physics only 55.4 → 47.7 ms, env step 106 → 103; pyramidal cap 10 physics 43.8 → 39.7 ms; empty iterations 0.06–0.15 → ~0 ms | no: skipped iterations had no active world (bitwise by construction; medians identical to the floor's on Metal) | **landed**, default on (`MJW_METAL_ICB_EARLY_EXIT=0`) |
| 3. Line-search derivative noise floor (Genesis PR #3382's rule) | CPU: line-search iterations per call 5.97 → 2.72, the launch's longest search 14.1 → 11.2 (median 20 → 10), Newton iterations unchanged; Metal: env step −2.2 %, loop 37.5 K → 38.1 K | at the floor on walking states (CPU 1-ulp floor and the 4096-env cap probe), **1.3× the floor's p99 on lying robots** (cap probe, independent of the cap) | **archived as an option** (`MJW_LS_NOISE_FLOOR=8`), §5 |
| 4. Warm start in cone space | characterised (§2): MuJoCo C's better-of-two start is taken in 0.7 % of solves; the extrapolated start 2·qacc_prev − qacc_prevprev cuts Newton iterations 3.69 → 2.73 (elliptic) and 2.93 → 1.98 (pyramidal) on walking states, env step −4 %, but 5.8× more worlds reach the cap on random-action states and its state differences sit systematically 1.5–2× above the floor's median | tolerance level | **archived** behind `MJW_WARMSTART_EXTRAP=1` (deprioritised by the review; the cap probe is §2.4) |

## 1. What the elliptic iteration launches (before / after)

Per Newton iteration on the incremental mode-2 path (dense Jacobian, G1 task), counted with a `wp.launch` hook on the
CPU device (`ellip_fused_cpu_check.py`, 64 worlds, cap 20, 8 substeps):

| path | launches per iteration | per substep at cap 20 |
|---|---|---|
| elliptic, before (`mul_m`, line search, zero counters, `_update_constraint_efc`, qfrc, zero grad_dot, grad, cone list, htot per entry, Cholesky, `_solve_done`) | **11** | 255 |
| elliptic, fused rows + cone list (`MJW_METAL_FUSE_HTOT=0`) | 6 | 195 |
| **elliptic, fused rows + cone list + htot (default)** | **5** | 183 |
| pyramidal (fused since dd42c23; its kernel signature grew here, arithmetic unchanged) | 5 (from 9 unfused) | 160 at cap 10 |

Bitwise identity on the CPU device (Warp's CPU atomics run in order, so the comparison is exact): over 8 substeps of 64
G1-task worlds from the walking snapshot t = 60, max |Δ| of qacc, qfrc_constraint, efc_force, qpos, qvel = 0, state
rows differing 0, worlds with a different solver_niter 0, for elliptic fused vs unfused, elliptic fused + htot vs
unfused, and pyramidal fused vs unfused (`runs/ellip_ws/fused_cpu_check.log`). What is not bitwise is what the
pyramidal fusion already changed: `grad_dot` is a SIMD-group sum instead of atomics (order noise), and the order of
`quad_changed_ids` (atomics in both).

## 2. Warm start: characterisation (measured before the review arrived; item 4 is archived)

States: `runs/competitors/g1_states/ellip10.npz` (the elliptic-trained policy walking, 256 envs, 9 snapshots at
control steps 2–399, `g1_state_dump.py`); every snapshot stepped 8 substeps (2.5 ms) from its saved `qacc_warmstart`,
Newton cap 100. `scripts/diagnostics/competitors/g1_newton_iters.py`, `runs/ellip_ws/*.json`.

### 2.1 MuJoCo C's warm start vs the fork's

MuJoCo C (`engine_forward.c` `warmstart()`, lines 1072–1131 of the 2026-09-25 head 3991e54; the same function exists
in 3.14.0) starts the solve from the **better of `qacc_warmstart` and `qacc_smooth` by cost**; upstream MuJoCo Warp
(`solver._solve_init_dof`) starts from `qacc_warmstart` unconditionally. C also exits with zero iterations when the
duality-gap certificate already holds (`engine_solver.c:2857–2875`). Measured in C on the dumped states (256 worlds ×
9 snapshots × 8 substeps = 18,432 solves per cone):

| | elliptic impratio 10 | pyramidal (impratio 1) |
|---|---|---|
| Newton iterations: mean / p50 / p90 / p99 / max | 3.75 / 3 / 7 / 10 / 19 | 2.92 / 3 / 5 / 7 / 11 |
| solves ≥ 10 iterations / ≥ 20 | 1.5 % / 0 | 0.05 % / 0 |
| C chose `qacc_smooth` over the warm start | **0.7 %** of solves (iterations when it did: mean 3.8) | 0.6 % |
| zero-iteration exits | 0.04 % | 0.04 % |
| line-search evaluations per Newton iteration (`mjSolverStat.neval`), mean / by iteration 1, 2, 3, 4 | **4.93** / 4.09, 4.64, 5.48, 6.06 (p90 at iterations 3–4: 14) | 3.10 / 3.56, 2.90, 2.88, 2.89 |
| Newton iterations with the line search at its budget (neval ≥ 20) | 1.85 % | 0 |

So neither of C's two start rules is a lever: the better-of-two picks the smooth start in 0.7 % of solves and the
zero-iteration exit fires in 0.04 %. The elliptic premium in iterations is 3.75 vs 2.92 (+28 %) with a tail to 19,
and the line search takes 1.6× the evaluations with a p90 of 14 at iterations 3–4 (C's `ls_iterations` is 20). At the
start of the solve 25 % of the elliptic rows are in the CONE state (2.07 of 8.13 per world, Warp CPU; the review's C
`iterations = 0` probe reports the state after `warmstart()`'s last constraint update at `qacc_smooth`, not the start).
Iterations grow with the cone rows at the start: 0 rows 2.4, 6 rows 3.8, 15 rows 6.6 (C, `niter_by_cone_rows_at_start`).

### 2.2 Warm-start variants on the fork's CPU device (128 worlds, the same protocol; `runs/ellip_ws/warp_*.json`)

| start of the solve | elliptic: mean / p90 / p99 / max, ≥ 10 | later substeps (2–8) mean | pyramidal: mean / p99 / max |
|---|---|---|---|
| `qacc_warmstart` (the fork, MuJoCo Warp upstream) | 3.69 / 6 / 10 / 13, 1.1 % | 3.68 | 2.93 / 7 / 10 |
| `qacc_smooth` (warm start disabled) | 3.83 / 6 / 8 / 13, 0.23 % | 3.83 | – |
| **2 qacc_prev − qacc_prevprev** (linear extrapolation from the second substep on) | **2.73 / 5 / 9 / 12, 0.53 %** | **2.58** | **1.98 / 6 / 9** (later substeps 1.82) |

The extrapolation cuts the elliptic iteration count by 26 % (30 % on substeps 2–8) and pyramidal's by 32 %; the
fork's implementation (`MJW_WARMSTART_EXTRAP=1`, §2.3) reproduces the script's manual extrapolation to the digit
(2.7258 / 1.9773). The smooth start shortens the tail but costs iterations on average (MuJoCo C's better-of-two rule
would take it in 0.7 % of solves).

### 2.3 Implementation (archived, off)

`_solve_init_dof(extrap)` starts from `2 * qacc_warmstart - qacc_ws_prev` when `qacc_warmstart` still equals what the
last forward wrote (`qacc_ws_last`; a reset or a user write of `qacc_warmstart` falls back to the plain start), and
records `qacc_ws_prev = qacc_warmstart`; the forward's copy becomes `_save_warmstart` writing both arrays. Two Data
fields (`qacc_ws_prev`, `qacc_ws_last`, warp-only, at the end of the dataclass; `types_test` field order and docstring
tests pass). No extra launch per substep.

### 2.4 Metal measurements (`runs/ellip_ws/check.wrapper.log`, `time.wrapper.log`, `iter.wrapper.log`)

- State difference (`g1_tp_check.py`, 512 envs, 50 control steps, deterministic random actions, task default) vs the
  unfused reference: the extrapolated start's **median** |Δqpos| is 3.7e-9 at step 2 where every floor's median is
  0.0, and 1.5–2× the floors' medians from step 5 on (6.0e-8 vs 2.4–3.0e-8, 1.2e-7 vs 8.9e-8, 3.2e-7 vs 2.0–2.4e-7,
  6.2e-7 vs 3.6–4.2e-7); the fused variants' medians equal the floors'. Worlds at the iteration cap of 20 over the run:
  **56–58 of 512 vs 9–12** for every other variant: on falling / impacting robots the extrapolation overshoots and the
  solve then needs more than 20 iterations.
- Throughput at 4096 (`--quick`, on top of the fused launch without the htot fusion): physics only 71.0 → 67.4 ms,
  env step 109.1 → 104.3 ms (−4.4 %).
- Marginal cost per iteration in graph mode (`g1_solve_iteration_cost.py`, random-action state): 12.63 → 12.56 ms per
  substep (converged before iteration 3: 1468 → 1591 worlds; before 5: 2001 → 2227).
- Cap probe (`cap_probe_variant.py`, 4096 envs, states from the elliptic-trained policy and from random actions,
  extrapolated start at caps 10 / 20 / 40 / 100 vs the plain solver at cap 100, floor = a second plain cap-100 run):
  (`runs/ellip_ws/probe_extrap_{policy,random}.json`, `probe_b.wrapper.log`) — walking states (pelvis z 0.69, no
  falls): floor p99 |Δqvel| max / mean over the 10 checkpoints 2.64e-3 / 1.42e-3, worlds > 0.01 per step 11.7; the plain
  solver at cap 20 2.50e-3 / 1.35e-3, 12.8 (at the floor); **extrapolated at cap 20: 4.43e-3 / 2.27e-3, 18.9** (1.6–1.7×
  the floor), at cap 40 / 100 the same (3.96e-3 / 2.31e-3, 19.2; 4.04e-3 / 2.12e-3, 18.2), Newton iterations 2.90 vs
  3.62, max 15 vs 16. Lying robots (random actions, 3,281 of 4,096 below z 0.4): floor 1.44e-3 / 9.6e-4, 5.0 worlds;
  plain cap 20 1.60e-3 / 1.03e-3, 6.7 (4.1 worlds at the cap per step); **extrapolated cap 20: 3.21e-3 / 2.72e-3, 16.8**
  (9.2 at the cap), cap 40 / 100 2.98e-3 / 2.49e-3, 13.5 (max 35–39 iterations vs the plain solver's 32). The
  extrapolated start converges to a different tolerance-level fixed point on every state class (the difference does
  not vanish with the cap), so it fails the bar.

Decision: archived off. It changes states at the tolerance level in every world (medians above the floor's), puts
5–6× more worlds at the cap on the states that need the cap, and the review shows the tail is inherent to impratio 10.
A better-of-two guard (MuJoCo C's `warmstart()` rule applied to the extrapolated vs plain start) would cost a launch
per substep and was not built. Re-enable: `MJW_WARMSTART_EXTRAP=1`.

## 3. Item 1: the fused elliptic iteration

### 3.1 What was fused

`_update_constraint_gradient_fused(nv, elliptic, fuse_htot)` (one launch of `MJW_METAL_FUSE_LANES` lanes per world):
zero the change counters; `_update_constraint_efc`'s rows including the CONE rule (a CONE-state row counts as a state
change, as before; the early returns of the elliptic rows are kept: nothing written, nothing counted); the
stable-state fast path (`state_changed_count == 0` returns before any Hessian work, with `cone_count` set to 0);
`qfrc_constraint = Jᵀ force` and `grad` per dof, `grad_dot` as a SIMD sum; then lane 0 lists the CONE-state contacts
with their curvature terms (`_update_gradient_JTCJ_cone_list` verbatim), and with `fuse_htot` every lane applies the
flipped-row deltas to `h` and writes `h + cone term` into `htot`, one upper-triangle entry per lane iteration with the
per-entry arithmetic and order of `_update_gradient_JTCJ_dense_world2_htot`. The register Cholesky and `_solve_done`
stay separate launches. MuJoCo C's structure is unchanged (the cone term is rebuilt every iteration, as `HessianCone`
does); only the launch shape changed.

### 3.2 Physics identity

CPU device: bitwise (§1). Metal (`runs/ellip_ws/check.wrapper.log`, 512 envs, 50 control steps, the unfused
configuration as the reference file; the floor is two graph-replay instances of the same configuration): fused vs
unfused |Δqpos| max / median at steps 1, 2, 5, 10, 20, 50: 1.13e-6 / 0, 2.30e-5 / 0, 2.06e-4 / 2.8e-8, 3.40e-2 /
8.9e-8, 4.54e-2 / 2.0e-7, 0.60 / 1.2e-7 against the unfused floor 3.67e-6 / 0, 2.30e-5 / 0, 2.34e-2 / 2.4e-8,
3.40e-2 / 8.9e-8, 6.52e-2 / 2.4e-7, 0.92 / 1.2e-7 (the falling random-action G1 is chaotic from step 5, so the maxima
are the floor's and the medians are the measure). Fused + htot: 1.42e-6 / 0, 5.11e-5 / 0, 2.34e-2 / 3.0e-8, 3.40e-2 /
8.9e-8, 6.52e-2 / 2.1e-7, 1.41 / 1.2e-7 (its own floor 3.49e-6, 1.62e-3, 2.34e-2, 2.66e-2, 6.52e-2, 1.41). Worlds at
the iteration cap and at the line-search cap over the run: 9–12 / 476–483 in every configuration. Fork suite on Metal
from the frozen worktree 5d0d165 (`runs/ellip_ws/suite.wrapper.log`): **1451 passed, 1 failed (the pre-existing
`test_put_data_nefc_zero_dense`), 39 skipped** = baseline; the same count on the final fork head fbf7da8 (fused
launch at 128 lanes, ICB early exit, line-search knob) with default knobs and with `MJW_LS_NOISE_FLOOR=8`
(`runs/ellip_ws/suite3.wrapper.log`; the intermediate 433c305 had two test-only failures, the direct `_solve_done`
launches in `solver_test.py` with the old argument list, fixed in fbf7da8). CPU (`--cpu`): `solver_test` + `forward_test`
+ `types_test` + `io_test` 313 passed, 12 skipped, the same pre-existing failure; `solver_test` + `forward_test` with the
fused kernels forced on the CPU device (`MJW_METAL_FUSE_UPDATE_CPU=1 MJW_FUSE_H_CHOLESKY_CPU=1`) and with
`MJW_WARMSTART_EXTRAP=1`: 150 passed, 8 skipped each. Fork heads: MuJoCo Warp `metalsim-ellip-ws` 063ff98 (code
fbf7da8), Warp `metalsim-ellip-ws` c200d46b (code 6bceb39d), both pushed.

### 3.3 Throughput (4096 envs, G1 task, elliptic ellip10 cap 20, `g1_tp_variants.py --quick`, two interleaved repeats)

| variant (frozen worktree) | launches / iteration | physics only ms (standing state) | full env step ms (random actions) |
|---|---|---|---|
| U unfused (5d0d165, `MJW_METAL_FUSE_UPDATE=0`) | 11 | 76.0 / 76.0 (53.9 K env-steps/s) | 113.1 / 113.1 (36.2 K) |
| F fused, per-entry htot launch kept | 6 | 71.2 / 70.8 (57.5–57.8 K) | 109.4 / 108.8 (37.4–37.6 K) |
| H32 fused + htot, 32 lanes | 5 | 55.6 / 55.4 (73.6–73.9 K) | 106.2 / 106.1 (38.6 K) |
| H64 (ca431fc, `MJW_METAL_FUSE_LANES=64`) | 5 | 55.6 / 55.6 | 101.3 / 101.2 (40.4 K) |
| **H128 (default)** | 5 | 57.0 / 56.9 (71.8–72.0 K) | **101.0 / 100.8 (40.6 K)** |

Why the htot fusion is worth 15 ms of the standing-state physics step: the per-entry launch is a (nworld × 946)-thread
grid, 20× per substep, running for every not-yet-converged world even when nothing changed (it rewrote `htot` for
worlds whose Cholesky is then skipped); fused, a world with no state change does nothing after the row pass. Why 128
lanes: at full activity the 32-lane entry loop (30 entries per lane, each with 2·nrows Jacobian loads per cone contact)
is slower than the 946-thread grid (marginal cost of iteration 1: 1.80 vs 1.63 ms), and the tail is faster; 128 lanes
(7.4 entries per lane) recover most of the full-activity cost (marginal per substep, random-action state: U 12.97, F
12.63, H32 13.64, H64 12.80, H128 12.19 ms; `iter*.log`).

### 3.4 The rollout regime (`runs/ellip_ws/prof.wrapper.log`, one run per variant, frozen ca431fc)

The first full A/B with H32 (`ab.wrapper.log`, a00c21f) moved the loop from 34.2–34.4 K to only 34.1–34.5 K
env-steps/s although the env step had gained 6 %: the rollout's states are not the random-action states. Measured on
the state after one PPO rollout of the untrained policy (nefc mean 3.0, 0.55 contacts per world: light contact, most
worlds converge in 2–5 iterations): solve per substep U 10.35, F 9.70, H32 9.55, **H128 8.72 ms**; marginal cost of
iterations 16–20 U 0.27–0.19, F 0.27–0.15, H32 0.08–0.05, H128 0.19–0.06 ms. Full `g1_tp_variants` in the same job:

| variant | physics only | env step | rollout + inference ms / step | PPO update | **loop env-steps/s** |
|---|---|---|---|---|---|
| U unfused | 75.8 ms | 112.4 | 113.6 | 142 | 34,276 |
| F fused, per-entry htot | 70.9 | 108.9 | 109.7 | 143 | 35,420 |
| **H128** | 57.0 | 100.5 | **102.3** | 150 | **37,742** |

## 4. Item 2: exact early exit of the Newton loop in graph replay (Warp fork `metalsim-ellip-ws` 6bceb39d, MuJoCo Warp 21be81b)

Mechanism. Warp's Metal graph replay encodes the recorded dispatches 1:1 into an indirect command buffer and executes
them with `executeCommandsInBuffer:withRange:` (`metal.mm` `graph_build_icb` / `graph_launch`). Metal's
`executeCommandsInBuffer:indirectBuffer:indirectBufferOffset:` takes the range (`MTLIndirectCommandBufferExecutionRange`
{location, length}) from a GPU buffer at execution time. New capture-time API `wp_metal_capture_range_begin(range_host)`
/ `_end` (Python `warp._src.context.metal_capture_range_begin(ranges, index)` / `_end`): the launches recorded between
them become a segment replayed through an indirect range whose 16 bytes (uint32 location, length, full length, pad)
live in a Warp array; the backend fills location and lengths at the range's end (unified memory) and resolves the
`MTLBuffer` at begin (the array is freed at the end of `solve()` and retained by the graph, as any capture-time
allocation). MuJoCo Warp (`_solve`, `MJW_METAL_ICB_EARLY_EXIT`, on when the Warp fork has the API and a Metal capture
is open, off under `graph_conditional`): every Newton iteration is one segment; `_solve_init_efc` restores every
length from its full length each substep; in `_solve_done` the world whose atomic brings `nsolving` to zero writes 0
into the lengths of iterations i+1 … cap−1. The skipped iterations had no active world, so the result is bitwise
the fixed-count loop by construction; without the fork's Warp (or with `WP_METAL_ICB=0`) everything launches as before.

Metal (`runs/ellip_ws/icb2.wrapper.log`, frozen 12cc1e2 + Warp 6bceb39d, htot fusion at 32 lanes, two interleaved
repeats): state difference on vs off at 512 envs: medians identical to the floors' at every step (0, 0, 2.9e-8, 8.9e-8,
2.2e-7, 4.0e-7, 2.7e-7, 1.3e-7), worlds at the caps identical (9–11 / 479–481). Physics only **55.4 → 47.7 / 47.6 ms**
(the standing state: most of the 20 iterations are empty), env step 106.0 / 105.6 → **102.7 / 102.9 ms** (−3.0 %).
Marginal cost of iterations 18–20 in graph mode (random-action state, all worlds converged before 18): 0.089 / 0.032
/ 0.076 → −0.036 / −0.014 / 0.043 ms (noise around zero); the solve total on that state 13.54 vs 13.53 ms per substep
because only its last three iterations are empty. Pyramidal cap 10 gains its empty iterations too: physics only 43.8
→ 39.7 ms in the final A/B (§6). Landed, default on.

## 5. Item 3: line-search derivative noise floor (MuJoCo Warp 12cc1e2, `MJW_LS_NOISE_FLOOR`)

The convergence test of `_linesearch_iterative_kernel` is |P′(α)| < gtol with gtol = max(tol·ls_tol·‖s‖·scale, 1e-6).
The derivative is a float32 sum over the rows whose rounding noise is ~eps × its gross magnitude Σ|q1_i| + |α| Σ 2q2_i;
when gtol is below it the search cannot terminate on the derivative and bisects to its budget (the review's finding
(B): 10 % of calls, 18 % at Newton iteration 1). Genesis PR #3382 (merged 2026-09-22) floors the tolerance at that
noise and steps to kinks instead of midpoints; only the floor is ported here: `gtol_at(α) = max(gtol, K·eps·(q1_abs +
|α|·p0[2]))` with `q1_abs` the Cauchy–Schwarz bound the fork already forms for the alpha floor
(`sqrt(2 Σcost_i Σ2q2_i) + |gauss q1|`), K = `MJW_LS_NOISE_FLOOR` in units of float32 eps. `MJW_LS_STATS=1` records
each world's line-search iteration count (`ctx.ls_iters`).

CPU (`ls_noise_floor_check.py`, walking states, 256 worlds × 9 snapshots × 8 substeps, cap 100; reference ls_iterations
100 with the floor off; `runs/ellip_ws/ls_floor_*.log`):

| setting | Newton iterations mean / max | control steps with a search at its budget | LS iterations per call mean; launch max mean (median) | \|Δqvel\| vs reference after 8 substeps: median / p99 / max, worlds > 1e-3 of 2,304 |
|---|---|---|---|---|
| elliptic: reference ls 100 | 3.657 / 17 | 0 | 6.40; 18.9 (25) | 0 |
| floor: ls 200 | 3.657 / 17 | 0 | 6.40; 18.9 (25) | 0 |
| floor: 1-ulp qvel nudge | 3.657 / 17 | 0 | 6.41; 18.9 (25) | 1.5e-5 / 3.7e-4 / 5.2e-3, 11 |
| **ls 20, floor off (the task)** | 3.657 / 17 | **82.9 %** | 5.97; **14.1 (20)** | 2.4e-7 / 3.0e-4 / **0.43**, 6 |
| ls 20, floor 8 eps | 3.657 / 17 | 26.5 % | **2.72; 11.2 (10)** | 4.5e-6 / 4.6e-4 / 8.6e-3, 12 |
| ls 20, floor 32 eps | 3.658 / 17 | 6.1 % | 2.48; 10.2 (9) | 4.8e-6 / 4.6e-4 / 7.4e-3, 11 |
| ls 20, floor 2 eps | 3.657 / 17 | 50.9 % | 3.20; 12.0 (12) | 3.3e-6 / 3.8e-4 / 4.9e-3, 11 |
| pyramidal: 1-ulp nudge floor | 2.924 / 11 | 0 | 6.03; 17.3 (24) | 6.3e-6 / 3.1e-4 / 9.1e-2, 2 |
| pyramidal: ls 20, floor off | 2.924 / 11 | 79.1 % | 5.61; 13.4 (20) | 1.2e-7 / 2.0e-4 / 1.8e-2, 2 |
| pyramidal: ls 20, floor 8 eps | 2.925 / 11 | 22.3 % | 2.10; 10.1 (7) | 4.8e-7 / 2.8e-4 / 9.1e-2, 5 |

Reading: the floor does not change the Newton iteration count, halves the line-search iterations per call and cuts the
launch's longest search (the kernel's latency) from 14.1 to 11.2 (median 20 → 10); the state after 8 substeps is at
the 1-ulp floor (p99 4.6e-4 vs 3.7e-4, the same 11–12 chaotic worlds; the task's own setting has a 0.43 rad/s
outlier the floored search does not). Metal (`runs/ellip_ws/ls.wrapper.log`, frozen 12cc1e2, early exit on, 32 lanes): state difference vs the floor-off
reference at 512 envs, medians at steps 5 / 10 / 20 / 30: 3.7e-8 / 1.0e-7 / 2.1e-7 / 4.0e-7 (8 eps) and 3.5e-8 /
9.7e-8 / 2.1e-7 / 3.6e-7 (32 eps) against the floors' 2.6–3.0e-8 / 8.9e-8 / 2.1–2.2e-7 / 3.6–4.0e-7: 1.0–1.4× the
floor's median (a termination change is float noise in every world, unlike the fusion's exact identity); worlds at the
line-search cap over 50 steps 478 → 403 (8 eps) → 311 (32 eps) of 512, worlds at the iteration cap 9–12 in all.
Throughput at 4096 (`--quick`, two repeats): env step 103.2 / 102.7 → 100.9 / 100.8 (8 eps) → 100.9 / 100.7 ms (32 eps),
physics only 47.7 → 47.4; graph-mode solve per substep 13.56 → 13.24 ms (iteration 1: 1.78 → 1.72, iteration 2: 1.60
→ 1.51). Final A/B (§6): loop 37.3–37.7 K → 38.1–38.2 K env-steps/s (+1.7 %).

Cap probe (`cap_probe_variant.py`, `VARIANT=lsfloor`, 4096 envs, `probe_lsfloor8_*.json`): walking states — floor p99
|Δqvel| max / mean 2.91e-3 / 1.50e-3, worlds > 0.01 per step 14.8; plain cap 20 2.55e-3 / 1.43e-3, 14.3; **floor 8 eps
at cap 20: 2.85e-3 / 1.58e-3, 14.8** (at the floor; Newton iterations 3.62, max 16, unchanged); at caps 40 / 100 the
same. Lying robots (random actions) — floor 1.48e-3 / 9.85e-4, 5.3; plain cap 20 1.48e-3 / 1.04e-3, 6.5; **floor 8 eps
at cap 20: 1.95e-3 / 1.31e-3, 9.5** (worlds > 0.1: 6.3 vs 4.0), at caps 40 / 100 2.44e-3 / 1.34e-3, 9.4 and 2.22e-3 /
1.32e-3, 8.5: **1.3× the floor's p99 and 1.5–1.8× its count of moved worlds, independent of the cap**. A small real
change on robots lying on many contacts (searches that stop at the derivative's noise end at a different kink).
Decision: **archived as an option, off by default** (`MJW_LS_NOISE_FLOOR=8` for the +1.7 % loop, +2.2 % env step; the
2-eps setting, which on the CPU still removes 40 % of the budget-bound searches, was not probed on Metal).

## 6. Final A/B (job_ellip.sh protocol: `g1_tp_variants.py 4096`, two interleaved repeats P E P E, frozen worktrees)

`runs/ellip_ws/final.wrapper.log` (12:03–12:06), frozen MuJoCo Warp 433c305 (fused elliptic iteration at 128 lanes,
ICB early exit; line-search floor by env var) + Warp 6bceb39d, MetalSim b7080fb, env-steps/s at 4096 envs (ms):

| preset, cap | physics only | full env step | rollout + inference | PPO update | **full PPO loop** | loop vs pyramidal |
|---|---|---|---|---|---|---|
| recommended_pyramidal, cap 10 (P1 / P2) | 103,167 / 102,710 (39.7 / 39.9 ms) | 62,710 / 62,683 (65.3) | 62,717 / 62,480 (65.3 / 65.6 per step) | 143 / 154 ms | **57,472 / 56,899** | 1.00× |
| recommended = elliptic ellip10, cap 20 (E0) | 85,888 / 85,692 (47.7 / 47.8) | 41,779 / 42,010 (98.0 / 97.5) | 39,461 / 39,874 (103.8 / 102.7) | 144 / 143 | **37,303 / 37,693** | **1.53×** |
| the same with the line-search floor 8 eps (E8) | 85,870 / 86,266 (47.7 / 47.5) | 42,820 / 43,094 (95.7 / 95.0) | 40,413 / 40,473 (101.4 / 101.2) | 150 / 144 | **38,072 / 38,207** | **1.50×** |
| before this work (`runs/tp26/ellip.wrapper.log`, 10:52): pyramidal / elliptic | 93,308–93,822 / 54,017–54,046 (43.9 / 75.8) | 62,069–62,539 / 36,371–36,513 (66.0 / 112.6) | 62,993–63,249 / 36,042–36,251 | 144–155 / 148 | 57,510–57,678 / **34,191–34,375** | **1.68×** |

The elliptic cap stays 20 (the cap probe of `elliptic_cones_2026-09-25.md` §8.5 is unchanged by launch-form work; the
iteration counts are the same to the digit in every check here). Pyramidal's loop is unchanged (57.5 K) although its
physics-only step gained the empty iterations of cap 10 (43.9 → 39.7 ms): the loop is bound elsewhere for it.

## 7. What remains of the premium and why

Loop premium 1.68× → 1.53× (1.50× with the line-search floor): of the 0.68 excess, 0.15–0.18 is removed, i.e. 22–26 %
of the premium, inside the review's 20–30 % estimate for the result-neutral levers. What remains (1.50–1.53×, 63–66 ms
of rollout + inference per step against pyramidal's 65 ms) is the review's "inherent" part, measured here:

- +28 % Newton iterations with a tail to 17–19 on walking states (MuJoCo C 3.75 vs 2.92, Warp 3.66 vs 2.92, §2.1–2.2)
  and 20–34 on fallen robots (§8.5 of the cones note): the cap of 20 is needed, and the iterations 11–20 now cost only
  when a world is still active (the early exit removes the empty ones; the active-tail iterations cost their launches
  plus the slowest world's line search + factor + solve, 0.2–0.3 ms each in graph mode).
- The line search: 1.6× the evaluations per Newton iteration (C: 4.93 vs 3.10) and, in the fork, launches whose slowest
  world ran to the budget in half the launches; the noise floor brings the launch's longest search from 14.1 to 11.2
  iterations, not to pyramidal's ~10 (its own launch max is 13.4 → 10.1 with the floor). Kink candidates (the second
  half of Genesis #3382) were not ported.
- Per-iteration work: the cone term (rebuilt every iteration as in MuJoCo C's `HessianCone`, now inside the fused
  launch at ~0.1 ms more than the per-entry grid at full activity even at 128 lanes), the refactorization of every
  world with a cone row (the stable-state fast path cannot apply to CONE rows), and the elliptic line-search kernel's
  quad precompute. The rank-1 alternative (`MJW_ELLIPTIC_CONE_UPDATE`) measured slower on Metal (DECISIONS 2026-09-26).
- Not physics: the rollout + inference step is 102–104 ms against an env step of 95–98 ms, and pyramidal's rollout is
  65.3 ms for a 65.3 ms env step; the 5–7 ms gap under elliptic (also present before this work, 113.6 vs 112.4) is
  unexplained here.

Options not landed, with their measured numbers: the extrapolated warm start (§2: −26 % iterations, −4 % env step,
tolerance-level state change, more worlds at the cap; `MJW_WARMSTART_EXTRAP=1`); MuJoCo C's better-of-two start
(0.7 % of solves, not built); a persistent per-world tail kernel (`throughput_2026-09-26.md` §11.1, not built).

## Sources

- MuJoCo C: `scratch/deformable/mujoco_main/src/engine/engine_forward.c` (`warmstart()` 1072–1131, `mj_fwdConstraint`
  1234–1296, `qacc_warmstart` written in `mj_advance` 1488), `engine_solver.c` (`mj_solPrimal` 2786–2935: the zero-
  iteration certificate 2857–2875, the main loop 2880–2935; `PrimalSearch` 1970–2145), head 3991e54 (2026-09-25).
- Warp fork `upstream/warp-innate` `warp/native/metal.mm` (`Graph`, `graph_build_icb`, `graph_launch`, 92–130 /
  1400–1470 at 9abceff9), Apple `MTLComputeCommandEncoder executeCommandsInBuffer:indirectBuffer:indirectBufferOffset:`.
- Genesis PR #3382 (genesis-world, merged 2026-09-22): line-search gradient tolerance floored at the rounding noise of
  the derivative terms, kink candidates instead of midpoints; "bit-identical" on its convex run.
- `docs/research/elliptic_cost_review_2026-09-26.md` (the independent review), `elliptic_cones_2026-09-25.md` §8,
  `throughput_2026-09-26.md` §2c / §4b / §11.1.

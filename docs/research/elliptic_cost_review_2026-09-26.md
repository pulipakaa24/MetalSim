# Elliptic-cone premium: independent review (2026-09-26)

Scope: can the G1 task's elliptic-cone premium (impratio 10, Newton cap 20) be cut further **without changing the
converged result**, and what did the implementation work miss? Research only: no fork code edited, no GPU work.
Fork read: `upstream/mujoco_warp` branch `metalsim` at `0a9de8e` (`mujoco_warp/_src/solver.py`); MuJoCo C reference:
the installed `mujoco` 3.14.0 wheel for probes and `scratch/deformable/mujoco_main/src/engine/engine_{solver,forward}.c`
(a MuJoCo main checkout) for source reading. Labels: **measured** (with scope), **published** (cited), **estimated**
(with the reasoning).

CPU probes (all new, `scripts/diagnostics/competitors/elliptic_review/`, logs `runs/competitors/elliptic_review/`) run
on the dumped task states `runs/competitors/g1_states/ellip10.{mjb,npz}` (`g1_state_dump.py`: the ellip10-trained
policy walking on flat ground, 256 worlds x 9 snapshots, each taken at a control-step boundary with the new action
applied). Each probe runs that control step's 8 substeps of 2.5 ms, so there are 2,304 control steps and 18,432 Newton
solves per configuration. The Newton cap is 100 unless stated. Engines: MuJoCo C float64 and MuJoCo Warp float32 on
Warp's CPU device, imported read-only from the shared checkout. The second state set is
`g1_states/c_ellip10` (MuJoCo C, 64 worlds standing, falling and lying).

## 0. Summary

| question | answer |
|---|---|
| 1. where the 31.9 ms goes | Iterations 11-20 cost ~15 ms, about half. They are 0.19 % of the world-iterations (measured, CPU), so this is dispatches plus the latency of the slowest 1-17 worlds. Extra dispatches in iterations 1-10 (11-12 launches per elliptic iteration vs 5-6 on the fused pyramidal path) cost ~4 ms. Real work and latency in iterations 1-10 cost ~13 ms: +26 % world-iterations, the cone Hessian term, refactorization of every cone world, and 1.6x line-search evaluations. The loop premium is larger than the physics-only one (48.5 ms per control step vs 31.9), inferred to be physics on the rollout's harder states. |
| 2. why ~2x the cap | **Inherent, and caused by impratio 10, not by the cone.** MuJoCo C float64 on the same walking states: elliptic imp10 mean 3.77 iterations, 0.61 % of solves > 10, max 17 (at ls_iterations >= 50). Pyramidal: 2.92, 0.01 %, max 11. **Elliptic imp1: 3.07, 0.02 %, max 11, i.e. pyramidal-like.** MuJoCo Warp float32 matches C (3.67 / 0.42 % / 17). The tail depends on the start point: a cold start removes 85 % of the > 10 solves on walking states but is 1.7x worse on lying states. It is not a line-search artefact: in the tail iterations 0 % of line searches hit their cap. |
| 3. prior art | No cheaper exact-cone formulation keeps MuJoCo's converged result. PhysX TGS projects each anchor's 2-D friction impulse onto a disk inside a fixed-iteration PGS/TGS loop: an exact cone, a different (unconverged) solution. MuJoCo C does the same per-iteration cone work (`HessianCone`). Genesis (Sept 2026) made two elliptic changes: a lane-parallel cone-block accumulation (bit-identical, +39-78 %; our `world2` / mode 2 already does this) and a line search that stops at the derivative's rounding noise and steps to kinks instead of midpoints (0-4 % on convex scenes, bit-identical on its Go2 convex run). The second is transferable (§3). |
| 4. result-neutral settings | Raising `tolerance` is not free: MuJoCo Warp already clamps it to 1e-6 (`io.py:390`), and 1e-7 / 1e-6 give the same iterations. `ls_iterations`: 20 and 15 stay within the 1-ulp floor, 10 does not. `noslip` is unsupported in MuJoCo Warp and 0 in the model. The cap of 20 cannot become a per-world exit on Metal (no conditional graph node). On walking states **6.2 of the 20 launched iterations per substep are empty** (measured distribution, 4096 worlds). 11-12 have 1-47 active worlds and carry the full per-world latency. |
| 5. verdict | Removable without changing results: **~20-30 % of the 0.68 premium**. Fusing the elliptic per-iteration launches saves ~6.5-9.6 ms per control step. A GPU-side exact early exit through ICB indirect execution ranges saves 0.5-2.4 ms. A noise-floored line-search tolerance saves ~1-3 ms. Together the loop goes from 1.68x to ~1.49-1.56x (estimated). A persistent per-world tail kernel (large port) could reach ~1.42-1.52x. The rest (~1.45x) is inherent to impratio 10 elliptic cones on a latency-bound GPU: more Newton iterations, a long tail, and non-quadratic line searches. MuJoCo C shows all three. |

Two findings neither line of work had:

- **(A) The MuJoCo C oracle is unsafe at ls_iterations 20 under ellip10.** On these states MuJoCo C (float64,
  ls_iterations 20) blows up in 6 of 2,304 control steps (|qvel| up to 2,144 rad/s within one substep). It differs from
  MuJoCo Warp by > 1e-3 rad/s in 252 of them. With ls_iterations 50 or 100: 0 blow-ups and 8 differing, the same as
  pyramidal's 10. So C's line search, at its budget, accepts a destructive step. Warp's line search only accepts an
  improving alpha, so Warp is safe. Every C-oracle comparison under elliptic impratio >= 10 should run C with
  `ls_iterations >= 50` (measured, `c_blowup.log`, `c_ls100.log`).
- **(B) Warp's line search hits its 20-iteration budget in ~10 % of calls, on both cone types.** 18 % of calls at Newton
  iteration 1, 0 % in the tail. Iterations 20-100 of those searches change the state below the 1-ulp floor. This is
  the fp32 "tolerance below the rounding of its derivative" pathology that Genesis fixed in PR #3382. It costs
  throughput in the full early iterations, not tail latency.

## 1. Where the 31.9 ms per step goes (4096 envs, 8 substeps, physics-only protocol)

Measured anchors:

- `runs/tp26/ellip.wrapper.log`: pyramidal cap 10 physics 43.8 ms, elliptic imp10 cap 20 physics 75.8 ms per control
  step. Rollout 64.9 vs 113.3 ms per step, loop 57.6 K vs 34.3 K env-steps/s (71.0 vs 119.5 ms per env-step batch).
- `runs/competitors/e2_loop_cap.log` (fork `9b4e96a`): elliptic physics 64.8 K (cap 10) and 52.2 K (cap 20)
  env-steps/s, i.e. 63.2 and 78.5 ms. **Cap 10 to 20 costs 15.3 ms, 0.19 ms per added iteration-substep.** Pyramidal
  cap 10 to 20: 49.1 to 56.7 ms, 0.094 ms per added (empty) iteration.
- Dispatch cost in a dependent graph-replay chain: 8.1 µs per removed dispatch (the pyramidal fusion: -2.6 ms per
  step for 4 launches x 80 iterations, `throughput_2026-09-26.md` §2c). An empty 9-launch iteration costs 0.07-0.11 ms
  (§4b).
- Launches per Newton iteration, counted in `solver.py`. Elliptic, mode 2: `mul_m` (1-2), line search,
  `_zero_change_counters`, `_update_constraint_efc`, qfrc, `zero_grad_dot`, `grad`, `_update_gradient_JTCJ_cone_list`,
  `_update_gradient_JTCJ_dense_world2_htot`, `_update_gradient_cholesky`, `_solve_done`: 11-12. Pyramidal
  (`_update_constraint_gradient_fused` and the fused h + Cholesky; `_fuse_update` excludes elliptic): 5-6.
- Work distribution (measured, CPU, walking states; `warp_cpu_iters.log`). Elliptic: 3.67 world-iterations per solve,
  of which **0.0068 (0.19 %) lie beyond iteration 10**. Pyramidal: 2.92. Expected active worlds per substep at 4096 for
  elliptic iterations 1..20: 4096, 3977, 2700, 1679, 1088, 680, 403, 221, 109, 47, 17.1, 7.8, 2.0, 0.4, 0.2, 0.2, 0.2,
  0, 0, 0. Pyramidal: ..., 29.6, 8.2, 2.0 at iterations 8-10, then 0.2, 0.

Reconciled budget (per control step; rounding ±1-2 ms):

| component | ms | status |
|---|---|---|
| (a) iterations 11-20 (the cap raise) | ~15 | measured as a block (15.3 at `9b4e96a`). Split: ~7-8 ms dispatch floor (80 iteration-substeps x 11-12 launches x 8-10 µs, estimated) and ~7 ms latency of the slowest active world (inferred as the remainder). The work itself is 0.19 % of the world-iterations, i.e. nil. |
| (b) extra dispatches in iterations 1-10 | ~4 | estimated: 6 more launches x 80 x 8.1 µs. Consistent with the gap growing 2.6 ms when the pyramidal fusion landed. |
| (c) work and latency in iterations 1-10 | ~13 | the remainder (measured gap 14.1 ms at cap 10 on `9b4e96a`, plus the fusion growth, minus (b)). By the eager split of §8.3 (cone term + deltas 2.49, extra Cholesky 1.0, extra line search 0.65, constraint update 0.13 ms per substep, eager), scaled to graph mode: cone Hessian term ~6-7, refactorizing every cone world ~2-3, line search ~2-3 (quad precompute plus 1.6x evaluations, see §2), plus the extra 26 % of world-iterations spread over these. The split inside (c) is inferred, not measured. |
| total | ~32 | vs the measured 31.9 |

Answers to the sub-questions. (a) The extra cap iterations are almost entirely latency and dispatch: 6.2 of the 20
are empty and 5-6 have fewer than 50 active worlds. (b) The cone Hessian term is ~6-7 ms (inferred) and cannot be
avoided: it depends on Jaref and changes every iteration, and MuJoCo C recomputes it every iteration too
(`HessianIncremental` then `HessianCone` on a fresh `Lcone`). (c) The line search is ~2-3 ms of in-cap work plus its
share of the tail latency (inferred). (d) The rollout's physics gap is 48.5 ms, not 31.9. The rollout uses an
untrained policy whose robots flail and fall, and the cap probe shows that regime has more worlds past 10 (546 per
step at cap 10) and a longer tail (max 20-34). The difference is inferred to be physics on harder states: the obs /
reward / reset kernels are the same in both. **What is missing is a graph-mode marginal-iteration profile under
ellip10** (`scripts/diagnostics/g1_solve_iteration_cost.py` with `contact_cfg="recommended"`, one timing job of ~1
min). It would turn (a)'s split and (c) into measurements, and it is the natural before/after for the fusion work.

## 2. Why elliptic needs about twice the cap

**MuJoCo C (measured, float64, walking states, 8 substeps, cap 100; `c_iters_substeps.log`, `c_ls100.log`):**

| configuration | mean iterations | p99 | max | solves > 10 | LS evaluations per iteration |
|---|---|---|---|---|---|
| elliptic imp10 (task), ls 20 | 3.75 | 10 | 19 | 0.68 % | 4.93 |
| elliptic imp10, ls 50 / 100 | 3.77 | – | 17 | 0.61 % | 4.96 |
| elliptic imp1 | 3.07 | 8 | 11 | 0.02 % | 3.39 |
| pyramidal imp1 (task's pyramidal) | 2.92 | 7 | 11 | 0.01 % | 3.10 |
| pyramidal imp10 | 3.99 | 9 | 14 | 0.18 % | 3.48 |
| elliptic imp10, tolerance 1e-6 | 3.64 | 10 | 18 | 0.47 % | 4.78 |

**MuJoCo Warp, CPU device, float32 (measured, same states; `warp_cpu_iters.log`):** elliptic imp10 3.67 / max 17 /
0.42 % > 10; elliptic imp1 2.98 / 11 / 0.02 %; pyramidal 2.92 / 11 / 0.01 %; pyramidal imp10 3.98 / 14 / 0.20 %.
Warp's effective tolerance is 1e-6 (clamped in `put_model`), and its numbers match C's tol-1e-6 row.

Per control step at 4096 worlds, 0.42-0.61 % of solves > 10 means 1-(1-p)^8, i.e. 3.3-4.8 %, or 137-200 worlds. The
Metal cap probe counted 270 at cap 10 on its own walking rollout: the same order.

Reading:

- **Inherent to the formulation at impratio 10.** MuJoCo C in float64 needs the same iterations and the same tail
  (max 17). No Metal or float32 trick removes it.
- **The driver is impratio, not the cone.** Elliptic at imp1 is within 5 % of pyramidal on the mean and the tail.
  Pyramidal at imp10 also grows (mean +37 %, tail 0.18 %). Newton is affine-invariant, so conditioning in the linear
  sense does not explain it. What does: impratio 10 makes D of the frictional rows 10x that of the normal row (so the
  primal cone is 1/√10 as wide), and the cost's curvature jumps by that ratio when a contact crosses the cone boundary
  (QUADRATIC to CONE). The semismooth Newton then crosses the boundary in several damped steps. §8.2 of the elliptic
  note attributed this to "the line search takes small steps there". The C evaluation counts (4.96 vs 3.10 per
  iteration) agree that line searches are longer. The tail itself, though, is extra Newton iterations, not exhausted
  line searches: see (B) below, where 0 % of line searches hit their cap in iterations 11+.
- **Warm start: part of the tail comes from the start point** (`warmstart_probe.log`, `cold_cap.log`). On the walking
  states, cold start (qacc_smooth) vs warm start: mean 3.87 vs 3.67, solves > 10 0.065 % vs 0.42 %, max 13 vs 17
  (Warp; C similar). On the lying states (`c_ellip10`), cold start is much worse: mean 4.69 vs 2.78, max 14 vs 11. For
  pyramidal and elliptic imp1 a cold start is better everywhere on walking states (2.11 vs 2.92, 2.29 vs 2.98): the
  previous substep's qacc is a poor start for this model under PD control. MuJoCo C picks the better of the warm start
  and qacc_smooth by cost (`engine_forward.c` `warmstart()`, lines 1071-1131). MuJoCo Warp takes the warm start as is
  (`_solve_init_dof`). But C's rule does not remove the tail (C warm, which already uses that rule: 0.61 % > 10), so
  cost-based selection is not the fix. A cone-space warm start (the parallel agent's item) is plausible but unproven.
  The targets it has to beat on this harness are: walking ≥ 11 at 0.065 % (cold) against 0.42 % (warm), and lying mean
  2.78 (warm) against 4.69 (cold).
- **A start change changes results only at the tolerance level.** The problem is strictly convex in qacc (M > 0), so
  the minimizer is unique. Cold vs warm start after 8 substeps (Warp CPU, elliptic imp10): p99 |Δqvel| 3.5e-4, max 0.29,
  3 of 2,304 control steps > 1e-3. The 1-ulp qvel nudge floor is 2.6e-4 / 6.6e-4 / 0. That is above the strict CPU floor
  in 0.1 % of steps, the same class as cap 12-15 (below), and inferred to lie inside the Metal run-to-run floor (cap
  probe floor: p99 0.0026 rad/s, 13 worlds > 0.01 per step).

## 3. Prior art

- **MuJoCo C** (source read). `mj_solPrimal` runs the same semismooth Newton. Per iteration it calls
  `PrimalUpdateConstraint(flg_HessianCone)`, then `HessianIncremental` (rank-1 `mju_cholUpdate` for flipped quadratic
  rows) and `HessianCone` (copy L to Lcone, dim rank-1 updates per CONE contact). It exits when improvement, gradient
  or Newton decrement is below the tolerance, or when alpha = 0. Warm start: best of warm start / smooth. Line search:
  1-D Newton with bracketing on the piecewise cost (published, Computation chapter). `noslip` is a PGS-style
  post-process over the friction rows that removes residual slip and changes the solution (MuJoCo docs, `noslip_iterations`).
  MuJoCo Warp does not implement it (MJX-Warp: "All except PGS, noslip"). The model has `noslip_iterations = 0`.
- **Todorov 2014** (ICRA, "Convex and analytically-invertible dynamics with contacts and constraints"). This is the
  convex primal formulation both engines solve. The elliptic cone cost is C1 but not C2 at the cone boundary, which is
  why Newton needs state-dependent damping. The paper offers no cheaper exact solver, only the choice of CG / Newton /
  PGS on the same problem.
- **MJX (JAX)**: published support for PYRAMIDAL and ELLIPTIC with CG / Newton. Its advice is to lower `iterations` /
  `ls_iterations` "just low enough that the simulation remains stable", i.e. accept non-convergence. It uses a
  `while_loop` on the accelerator (early exit on convergence), which Metal graph replay lacks.
- **MuJoCo Warp on CUDA**: `wp.capture_while(nsolving, ...)` (`solver.py:4884`) re-launches the iteration while any
  world is unconverged. On Metal MetalSim sets `graph_conditional = False`, so all `opt.iterations` launch.
- **Genesis** (published, GitHub, Sept 2026). PR #3380 (merged 2026-09-20): the elliptic cone block accumulated by
  the 32 lanes instead of one thread, "bit-identical", +39 % Go2 / +78 % arm at 2048 envs on an RTX 5090. That is the
  same fix as our `MJW_JTDAJ_ELLIPTIC_LANES=32` / `world2` (done 2026-09-25). PR #3382 (merged 2026-09-22): the line
  search takes the next kink (a friction block switching stick / saturate, a normal row switching on / off) instead of
  the midpoint, and floors the gradient tolerance at the rounding noise of the derivative terms. The old line search
  "spent its whole budget on a tolerance below the rounding of its derivative". Step rate +0-4 % on 2048-env convex
  scenes, "the go2 convex run is bit-identical to main". PR #3373 (closed): the same idea cut an aloha-clutter grasp
  from 12-54 to 11-22 Newton iterations at the stick / slip transitions of its Signorini mode. Genesis's own measured
  elliptic premium in our competitor run: Go2 2.23x, G1 1.72x (`competitors_2026-09-25.md`, pre-#3380).
- **PhysX 5.6.1 TGS** (source `DyTGSContactPrep.cpp` ~1745). Per friction anchor, `totalImpulse = sqrt(f0² + f1²)`
  is clamped to `frictionScale·μ·normalImpulse` and both axes are rescaled: an exact circular cone applied as an O(1)
  projection inside a fixed number of TGS position / velocity iterations (patch friction, two anchors per patch;
  published: "TGS applies friction throughout all position and all velocity iterations"). It costs nothing extra
  because the solver never converges a Newton system. It is a different solution (unconverged, patch-anchored), so it
  is not a way to get MuJoCo's converged result cheaply.
- **Other exact-cone solvers.** Cone-projected Gauss-Seidel (PGS with disk projection, MuJoCo's own PGS
  `projectCone`), APGD / SOCP interior point (Tasora & Anitescu; Acary & Brémond), Drake's SAP (Castro et al. 2022,
  convex unconstrained, Newton with exact line search), and the ADMM-type solvers of Carpentier et al. (2024, "From
  compliant to rigid contact simulation"). They either solve a different regularized problem (SAP, ADMM: different
  converged forces) or converge to MuJoCo's problem more slowly per unit of work (PGS / APGD: linear convergence;
  interior point: 20-50 factorizations). None beats a 3-4-iteration semismooth Newton on MuJoCo's own problem. No known
  cheaper exact-cone formulation keeps MuJoCo's converged result.

## 4. Settings and exits that cut iterations without changing results (measured on CPU unless stated)

`warp_cpu_lssweep.log`: elliptic imp10, walking states. The reference is `ls_iterations` 100 at cap 100. The floor is
the 1-ulp qvel nudge (p99 2.6e-4, max 6.6e-4, 0 control steps > 1e-3):

| setting | mean iterations / max | control steps with an LS at its budget | p99 / max \|Δqvel\| after 8 substeps, steps > 1e-3 of 2,304 |
|---|---|---|---|
| ls_iterations 200 | 3.669 / 17 | 0 % | 0 / 0, 0 |
| ls_iterations 50 | 3.669 / 17 | 0 % | 0 / 0, 0 |
| ls_iterations 30 | 3.669 / 17 | 9.2 % | 1.6e-6 / 2.9e-4, 0 |
| **ls_iterations 20 (task)** | 3.669 / 17 | **83.5 %** | 2.2e-4 / 6.7e-4, 0 |
| ls_iterations 15 | 3.669 / 17 | 88.8 % | 2.3e-4 / 6.7e-4, 0 |
| ls_iterations 10 | 3.666 / 14 | 95.7 % | 0.12 / 0.97, 41 |
| ls_tolerance 0.1 | 3.669 / 17 | 52.8 % | 2.3e-4 / 0.63, 1 |
| tolerance 1e-6 / 1e-7 (Warp clamps 1e-8 to 1e-6) | identical to the task | – | identical |
| cap 10 | 3.662 / 10 | – | 5.3e-4 / 3.0, 19 |
| cap 12 | – / 12 | – | 2.2e-4 / 0.09, 1 |
| cap 15 | – / 15 | – | 0 / 1.3e-3, 1 |
| cap 20 | 3.669 / 17 | – | at the floor, 0 |

- **tolerance**: already at MuJoCo Warp's float32 clamp of 1e-6 (`io.py:387-390`). A looser tolerance changes the
  result. Not a lever.
- **ls_iterations / ls_tolerance**: (B) above. Per call (`ls_by_iter.log`), 9.7 % of elliptic line searches hit 20
  (pyramidal 10.3 %, elliptic imp1 9.9 %): 18-19 % at Newton iteration 1, 7-8 % at iteration 2, 0 % from iteration
  11. So it is neither elliptic-specific nor tail latency. It is throughput in the full early launches. Lowering the
  budget is only safe down to ~15. The better fix is Genesis's: stop once the derivative is below the rounding noise of
  its own sum. The fork already computes that bound for alpha (`noise_floor`, `solver.py:1131-1135`) but not for
  `gtol`, which is `max(tol·ls_tol·‖s‖·scale, 1e-6)`. Kink candidates (the closed-form alpha where each elliptic
  contact enters or leaves the cone along the ray) would also shorten the searches that do converge. Estimated 1-3 ms
  per control step on the elliptic path (less on pyramidal). The expected state change is float-noise level (ls 20 /
  30 / 50 already all sit inside the floor), to be verified with the usual oracle and Metal floors.
- **noslip_iterations**: must stay 0 (it changes the solution). MuJoCo Warp does not implement it.
- **Cap vs per-world exit.** The iterations do not need a lower cap. They need to not launch when empty. Measured
  expectation at 4096 worlds (walking): iterations 1-12 are always non-empty, 13 is non-empty with probability 0.86, 14
  0.36, 15-17 0.2 each, 18-20 ~0. That is **6.2 empty iterations per substep, 50 per control step**. Iterations 11-14
  carry 17, 8, 2 and 0.4 active worlds and pay the full chain latency. On the rollout's flailing states fewer are
  empty (tail 20-34). An exact early exit on Metal is possible without a conditional graph node. The Warp fork already
  replays graphs through an ICB with `executeCommandsInBuffer:withRange:` (`warp/native/metal.mm:1464`). Metal also
  has `executeCommands(in:indirectBuffer:indirectBufferOffset:)`, where the range comes from a GPU buffer (published,
  Apple docs, compute encoder, iOS 13+ / macOS). Encoding each Newton iteration as its own ICB range and letting
  `_solve_done` write the next range's length (0 once `nsolving == 0`) implements `capture_while` bitwise. Estimated
  saving: 50 empty iterations x (11-12 launches now, ~6 after fusion) x 8.1 µs = 2.4-4.8 ms per control step on
  walking states, 0.5-2 ms on the rollout's states. Effort ~2-3 days in the Warp Metal backend plus MuJoCo Warp's
  loop. It also removes pyramidal's 1-2 empty iterations.

## 5. Verdict

Loop premium now: 119.5 / 71.0 ms per env-step batch = 1.68x. The excess of 0.68 is 48.5 ms per control step, all of
it physics (the PPO update is 144-155 vs 148 ms).

| mechanism | changes results? | estimated saving per control step | effort | status |
|---|---|---|---|---|
| Fuse the elliptic per-iteration launches: efc + qfrc + grad into one kernel (as `_update_constraint_gradient_fused`), cone list + htot folded into the Cholesky launch, 11-12 to ~5-6 | no (bitwise, apart from the SIMD-sum order of `grad_dot`, as in the pyramidal fusion) | 6.5-9.6 ms (5-6 dispatches x 160 iteration-substeps x 8-10 µs) | 1-2 days | the parallel agent's item 1; confirmed as the largest result-neutral lever |
| Exact early exit via ICB indirect execution ranges | no (bitwise) | 0.5-2.4 ms after fusion (state-dependent) | 2-3 days (Warp backend) | new |
| Noise-floored line-search `gtol` (plus optional kink candidates, Genesis #3382) | float-noise level (to verify) | 1-3 ms | 1 day | new |
| Persistent per-world tail kernel (iterations ≥ ~8 in one launch per active world, §11.1 of the throughput note) | no | 3-5 ms beyond the above (removes the tail's dispatches, keeps its per-world latency) | large port, occupancy risk | deferred in §11.1; the elliptic tail makes it worth more than on pyramidal |
| Cone-space warm start | tolerance level (like cap 12-15) | pays only through a lower cap or an early exit. If it matched the cold-start tail on walking states without the lying-state penalty, cap 15 would become possible: ~4-5 ms, pending a cap-probe re-validation on every state class | – | parallel agent's item 2. Evidence neutral to positive (§2). The random-action states need 20-34 iterations with today's start. |
| lower `tolerance`, `ls_iterations` < 15, cap < 20, noslip | yes | – | – | rejected |

**Removable without changing results: ~9-14 ms per control step** (fusion + early exit + line-search floor), loop
1.68x to **~1.49-1.56x**, i.e. **~20-30 % of the premium**. With the persistent tail kernel, ~12-19 ms and
~1.42-1.52x (~30-40 %). All estimated from measured per-dispatch costs and measured iteration distributions; the
fusion job's before/after is the first check.

**Inherent (~1.45x on the loop):** +26-29 % Newton iterations (C 3.77 vs 2.92, Warp 3.67 vs 2.92), a tail to 17
iterations on walking states and 20-34 on fallen ones (C shows the same), 1.6x line-search evaluations per iteration
(C: 4.96 vs 3.10), and the per-iteration cone Hessian and refactorization of cone worlds. MuJoCo C does the same
per-iteration cone work; its rank-1 form measured slower on Metal (DECISIONS 2026-09-26). The root cause is
impratio 10: elliptic at impratio 1 costs about what pyramidal does. That is a fidelity choice (PhysX impact forces
within 8 %) and stays.

Recommended next measurements (GPU, owner's queue):

1. The ellip10 graph-mode marginal-iteration profile (§1).
2. The fusion A/B.
3. A CPU-first prototype of the noise-floored `gtol`, checked on this harness (`warp_cpu_lssweep.py`) before any
   Metal run.
4. Run MuJoCo C oracles under ellip10 with `ls_iterations >= 50`.

## Sources

- Code: `upstream/mujoco_warp/mujoco_warp/_src/solver.py` (`0a9de8e`; lines cited above), `io.py:387-390`;
  `upstream/warp-innate/warp/native/metal.mm:1420-1470`; `scratch/deformable/mujoco_main/src/engine/engine_solver.c`
  (`mj_solPrimal`, `HessianIncremental`, `HessianCone`), `engine_forward.c` (`warmstart`, lines 1071-1131);
  `upstream/PhysX-5.6.1/physx/source/lowleveldynamics/src/DyTGSContactPrep.cpp` (~1730-1760).
- MuJoCo docs, Computation chapter: https://mujoco.readthedocs.io/en/stable/computation/index.html. MJX feature table:
  https://mujoco.readthedocs.io/en/stable/mjx.html
- E. Todorov, "Convex and analytically-invertible dynamics with contacts and constraints: theory and implementation
  in MuJoCo", ICRA 2014.
- Genesis PRs: https://github.com/Genesis-Embodied-AI/genesis-world/pull/3380,
  https://github.com/Genesis-Embodied-AI/genesis-world/pull/3382,
  https://github.com/Genesis-Embodied-AI/genesis-world/pull/3373
- PhysX rigid body dynamics (friction, TGS): https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/docs/RigidBodyDynamics.html
- Apple Metal, `executeCommands(in:indirectBuffer:indirectBufferOffset:)`:
  https://developer.apple.com/documentation/metal/mtlcomputecommandencoder/executecommands(in:indirectbuffer:indirectbufferoffset:)
- A. M. Castro et al., "An unconstrained convex formulation of compliant contact", IEEE T-RO 2022 (SAP). J.
  Carpentier et al., "From compliant to rigid contact simulation: a unified and efficient approach", RSS 2024
  (arXiv 2405.17020).
- Project measurements: `runs/tp26/ellip.wrapper.log`, `runs/competitors/e2_loop_cap.log`,
  `runs/competitors/cap_probe/*.json`, `docs/research/elliptic_cones_2026-09-25.md` §8,
  `docs/research/throughput_2026-09-26.md` §2c / §4b / §11.1; new probes in `runs/competitors/elliptic_review/`.

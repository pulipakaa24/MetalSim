# G1 throughput "regression" 2026-09-25: fork commit or machine? (measured on the 16-inch M4 Max)

Scope: the report from a session on another M4 Max (14-inch, 64 GB) that MetalSim `fbaab25` with the forks at their
`metalsim` heads measures the G1 flat task (4096 envs, 2.5 ms, decimation 8) at physics-only 84.1 K, full env step
55.2 K and full PPO loop 50.2 K env-steps/s against the README's 81.0 K / 67.6 K / 56.6 K, with `g1_step_profile.py`
putting the difference in `solver.solve` (5.50 ms per substep there against 4.23 ms in `runs/mjw_tp/final_checks.log`).
Candidates named: the flex merge `8fbf965` in the MuJoCo Warp fork (the headline was taken at `ccfaffb` / `1791414`), or
the machine. This note settles it with interleaved back-to-back runs on the machine the headline was measured on.
Every number below is measured on this machine through the timing queue (`scripts/gpu_run.sh ... timing`), on AC
power, with the wrapper's per-process GPU log idle (0 %, no process at or above 1 %) at the start and end of every
job; logs in `runs/mjw_tp/ab/` and `runs/mjw_tp/*.wrapper.log`, job scripts `runs/mjw_tp/job_ab1.sh`, `job_ab2.sh`,
`job_cc.sh`, `job_gen.sh`.

## 1. What is actually installed here (a finding in itself)

The shared checkout `upstream/mujoco_warp` (the editable install the queue's jobs import) is at **`1791414`**, the
local `metalsim` branch, 11 commits behind `fork/metalsim` (`07a51a6`): the flex merge and its follow-ups were
pushed from the `upstream/mujoco_warp-merge` worktree and the shared checkout's branch was never fast-forwarded. So
"the fork head" on this machine has been the headline's own commit all along, and the other machine (a fresh clone
at `07a51a6`) was the first to run the merge. The A/B below therefore compares the installed `1791414` against a
worktree at `07a51a6` (`upstream/mujoco_warp-07a51a6`, imported through `PYTHONPATH`, import path verified in every
run). Warp: the shared checkout `upstream/warp-innate` is at `9050cb54`, 3 commits behind `fork/metalsim`
(`f194006a`; those commits touch only `segmented_sort_pairs` in `warp/_src/utils.py`, a flex-only path). The
other machine's Warp commit `4127c48` is not an object in this repository's fork remote, so it could not be
measured here; the Warp delta since the headline (`b9557cb` → `9050cb54`) is one commit adding `nextafterf` to the
Metal kernel CRT (no kernel used by the G1 changes).

Code delta `1791414..07a51a6` outside the flex files: `constraint.py` adds `_equality_flex_reserve` (launched only
under `m.nflex > 0`), `io.py` adds a CPU-side `_flex_elemorder` at `put_model`, `types.py` one Model field. The G1
has no flex, so no launch, default or block size changes for it; the measurement confirms it.

## 2. A/B: MuJoCo Warp fork `1791414` (A) vs `07a51a6` (B), interleaved A B A B

MetalSim `626f54c`, Warp `9050cb54`, same process settings, `scripts/diagnostics/g1_tp_variants.py 4096`
(`runs/mjw_tp/ab1.wrapper.log`, 19:39–19:41) and `g1_step_profile.py 4096 0.0025 --reps 20`
(`runs/mjw_tp/ab2.wrapper.log`, 19:41). `pmset -g batt`: AC power, 100 %; `pmset -g therm`: no thermal or
performance warning level recorded, before every run.

| run | physics only | full env step | rollout + inference | PPO update | full PPO loop | nefc max / mean | nacon per world | overflow |
|---|---|---|---|---|---|---|---|---|
| A1 `1791414` | 81,077 (50.5 ms) | 54,272 (75.5 ms) | 54,225 | 152 ms | 50,022 | 43 / 14.4 | 2.77 | ITERATIONS 26, LS 3887 |
| B1 `07a51a6` | 81,014 (50.6 ms) | 54,035 (75.8 ms) | 54,023 | 149 ms | 49,937 | 44 / 14.4 | 2.75 | ITERATIONS 16, LS 3863 |
| A2 `1791414` | 81,287 (50.4 ms) | 54,077 (75.7 ms) | 53,836 | 147 ms | 49,823 | 43 / 14.2 | 2.70 | ITERATIONS 12, LS 3877 |
| B2 `07a51a6` | 81,041 (50.5 ms) | 54,169 (75.6 ms) | 53,990 | 144 ms | 50,022 | 44 / 14.5 | 2.75 | ITERATIONS 22, LS 3890 |
| spread A vs B | 0.3 % | 0.4 % | 0.7 % | 5 % (update, MPS) | 0.4 % | | | |

Step profile (20 replays, ms per control step; `solver.solve` per substep):

| run | physics segment | whole step graph | solver.solve / substep | mjw.step / substep | dispatches (step / substep) |
|---|---|---|---|---|---|
| A1 `1791414` | 70.07 | 72.98 (56,127 env-steps/s) | 5.69 | 9.41 | 1653 / 202 |
| B1 `07a51a6` | 69.76 | 73.02 (56,094) | 5.74 | 9.40 | 1653 / 202 |
| A2 `1791414` | 69.78 | 72.89 (56,194) | 5.72 | 9.41 | 1653 / 202 |
| B2 `07a51a6` | 69.83 | 73.13 (56,007) | 5.72 | 9.32 | 1653 / 202 |
| headline (`runs/mjw_tp/final_checks.log`, 10:01 the same day, fork `1791414`) | 54.24 | 58.52 (69,999) | 4.23 | 8.13 | 1653 / 202 |
| other machine (`runs/competitors/metalsim_g1_step_profile.log`, `07a51a6`) | 68.16 | 71.29 (57,457) | 5.50 | 9.13 | 1653 / 202 |

**Verdict on the fork: not a code regression in `1791414..07a51a6`.** A and B agree within the run-to-run spread on
every column, with the same kernels and dispatch counts. The other machine's `solver.solve` (5.50 ms) is if anything
faster than this machine's today (5.7 ms), so the "14-inch, possibly power-limited" explanation is also wrong: the
two machines agree once the same code runs on both.

**But the headline is not reproducible today at the headline's own fork commit**: physics-only matches (81.0–81.3 K
vs 80.5–81.0 K) while the full env step is 54.1–54.3 K against 67.6 K, the loop 49.8–50.0 K against 55.8–56.3 K, and
`solver.solve` is 5.7 ms per substep against 4.23. Physics-only is measured from the reset pose (robots standing);
the step and the profile are measured under random actions (contact-rich). The cost that moved is state dependent
and solver-side, with the same launches: more worlds are unconverged at each of the 10 Newton iterations (the
per-thread `ctx.done` exit does less), and `ITERATIONS` overflow (worlds hitting the Newton cap) now appears in
12–26 of 4096 worlds where the headline runs had none. That is the signature of a harder constraint problem, not of
slower kernels: the MetalSim task, not the fork, changed between the headline session (MetalSim `d867143` +
`df0fda1`) and now (`626f54c`).

## 3. Mechanism: the task's contact preset (MetalSim `e3ba79f`)

Candidates in `git diff df0fda1..626f54c -- metalsim/`: `e3ba79f` "G1 task: contact_cfg (default `recommended` =
`tau10_impact_hardlimits`, the 2026-09-25 PhysX-parity decision)" makes every `G1VelocityTask` apply contact
solref 10 ms / solimp 0.9→0.999 over 5 mm and hard joint limits (solref 5 ms, solimp 0.99→0.999 over 1 mm) instead
of MuJoCo's 20 ms / 0.9–0.95 defaults; `ba0762a` / `13a57a4` (Isaac Lab 3.0 presets, only with `solver_cfg`),
`aa88f11` (reward term on the foot COM), `109984b` (factorization defaults, the same layout for the G1), and render-only
commits. The decision row for the preset (DECISIONS.md, 2026-09-25 "contact / limit preset, final") lists its cost
as "1.02×", measured as 117.5 ms per step at 4096 envs, i.e. before the throughput work (when the 43×43 dense
Cholesky was two thirds of physics and the solver's state-dependent work a small share). After the fast paths the
fixed cost is gone and the state-dependent part is what is left.

A/B at MetalSim HEAD, installed forks, interleaved R D R D (`runs/mjw_tp/cc.wrapper.log`): `contact_cfg="recommended"`
(R, the task default since `e3ba79f`) against `contact_cfg=null` (D, the model's own solref/solimp: the headline
setting). Results in §3.1 (filled from the run).

### 3.1 Results (`runs/mjw_tp/cc.wrapper.log`, 19:56–19:58, MetalSim `98a41c1`, forks `9050cb54` / `1791414`, AC power, idle GPU)

| run | physics only | full env step | rollout + inference | PPO update | full PPO loop | nefc mean | nacon / world | overflow |
|---|---|---|---|---|---|---|---|---|
| R1 `contact_cfg="recommended"` (task default) | 80,982 (50.6 ms) | 54,244 (75.5 ms) | 54,163 | 149 ms | 50,055 | 14.3 | 2.71 | ITERATIONS 24, LS 3885 |
| D1 `contact_cfg=null` (headline setting) | 80,600 (50.8 ms) | **67,209** (60.9 ms) | 61,366 | 149 ms | **56,157** | 15.0 | 2.85 | LS 3847 |
| R2 recommended | 81,008 (50.6 ms) | 54,094 (75.7 ms) | 53,830 | 159 ms | 49,509 | 14.3 | 2.74 | ITERATIONS 28, LS 3880 |
| D2 null | 78,023 (52.5 ms) | **67,102** (61.0 ms) | 61,176 | 144 ms | **56,154** | 15.1 | 2.87 | LS 3859 |
| headline 2026-09-25 10:00 (`runs/mjw_tp/costsplit_final.log`, two runs) | 80,469–80,885 | 67,606–67,626 | 61,059–61,510 | 149–153 ms | 55,769–56,250 | 15.1 | 2.88–2.89 | LS 3819–3838 |

Step profile, same job: recommended: physics 69.92 ms, whole step 73.14 ms (56,006 env-steps/s), `solver.solve`
**5.71 ms** per substep, `mjw.step` 9.39 ms; null: physics 54.08 ms, whole step 58.30 ms (70,251), `solver.solve`
**4.18 ms**, `mjw.step` 8.12 ms; headline: 54.24 / 58.52 (69,999) / 4.23 / 8.13. Same 1653 / 202 dispatches.

**Verdict: the README's 67.6 K / 56.6 K reproduce today to within 0.1 % (67.1–67.2 K / 56.15 K) with the contact
setting they were measured under; the task's current default costs 19.5 % of the env step and 11 % of the PPO
loop (54.2 K / 50.0 K), all of it in `solver.solve` (+1.5 ms per substep, 4.18 → 5.71, × 8 substeps = 12 ms of the
14.7 ms step difference; the rest is the rollout's share of the same physics).** Physics-only (standing robots,
no limit contact) is unchanged, which is why the other machine's physics-only "matched" the README while its step did
not. The other machine's numbers (55.2 K / 50.2 K / 5.50 ms) are this same setting on the same code and are within
2–4 % of this machine's: no hardware or power effect is visible, and nothing in the forks moved.

Why the preset costs 20 % now when its decision row says 1.02×: the 1.02× was measured at 117.5 ms per step, before
the throughput work, when the dense 43×43 Cholesky (fixed cost per launch, independent of convergence) was two
thirds of physics; the solver's convergence-dependent share was a few percent of that. The fast paths removed
the fixed cost, so the harder problem the preset poses (hard limits with solref 5 ms = 2 dt at the 2.5 ms step,
impedance 0.99–0.999 over 1 mm; contacts stiff on impact) now shows directly: worlds stay unconverged for more of
the 10 Newton iterations (the per-thread `ctx.done` exit does less work per launch), and 12–28 of 4096 worlds hit
the Newton cap (`ITERATIONS` overflow) every measured step where the headline setting had none. The cap is a cost cap
(PARITY §1.4 note), but a world leaving the solver at the cap is a world whose constraint forces are less converged:
the preset's owner should know that its solve is not only slower but also not fully converged in ~0.5 % of worlds at
10 iterations (a fidelity observation, not acted on here: the preset is a PhysX-parity decision and fidelity
decides). Which part of the preset carries the cost (hard limits vs the 10 ms impact contacts) is in §3.2.

### 3.2 Decomposition (`runs/mjw_tp/cc2.wrapper.log`)

(filled below)

## 4. Generality: where the G1 fast paths apply (task D)

The three Metal fast paths and their conditions (`scripts/diagnostics/g1_tp_generality.py` prints them per scene from
the built model): **register Cholesky** (Warp fork: one SIMD group, no workspaces) for Newton Hessians with
nv ≤ `metal_register_cholesky_max` (48); MuJoCo Warp launches the single-tile Cholesky off CUDA for nv ≤ 64
(`MJW_METAL_DENSE_CHOL_MAX`), so 49–64 dofs take the cooperative tile path (the G1's pre-step-1 state, physics
1.4× slower) and above 64 the blocked path (−18 % measured on the G1 when forced); **M layout**: the dense tile for
trees up to `m_dense_max` (32) dofs, the one-thread-per-world sparse L'DL above; **fused incremental Hessian +
Cholesky**: Newton solver, non-elliptic cone (MuJoCo Warp's incremental Hessian is pyramidal-only:
`_use_incremental`), dense constraint Jacobian (`jacobian=auto` goes sparse above 32 dofs), nv ≤ 64.

Measured 2026-09-25 19:41–19:48 on this machine (`runs/mjw_tp/gen.wrapper.log`, installed fork `1791414`, 4096
worlds; Go2 by the competitor protocol `scripts/diagnostics/competitors/metalsim_step.py go2 4096 matched`,
4 ms, PD, 10 / 20 iterations, random targets; SO-101 lift and Panda by `fast_factorization_scenes.throughput`,
their own `BatchSimOptions`, fixed random ctrl, 3 builds × 3 × 50 steps, median; kernel shares from the eager
`WP_METAL_PROFILE=1` sum of per-dispatch times, an attribution not a timing):

| model (scene default cone) | nv / substeps | register Cholesky | M | fused Hessian | pyramidal | elliptic | elliptic / pyramidal |
|---|---|---|---|---|---|---|---|
| G1 task (pyramidal, forced dense) | 43 / 8 | yes | sparse L'DL | yes | 81 K env-steps/s physics, 54 K step (§2) | not run (the elliptic agent's worktree) | – |
| Go2, menagerie scene (elliptic) | 18 / 1 | yes | dense tile | pyramidal only | 932,628 physics steps/s | 239,935 | 3.9× |
| SO-101 lift scene (elliptic) | 12 / 4 | yes | dense tile | pyramidal only | 292,171 env-steps/s (1.17 M physics steps/s) | 122,534 (490 K) | 2.4× |
| Panda, menagerie scene (pyramidal; 100 / 50 iterations) | 9 / 10 | yes | dense tile | pyramidal only | 28,403 env-steps/s (284 K physics steps/s) | 9,923 (99 K) | 2.9× |

Per-kernel attribution (ms of GPU time per env step, eager; Go2 from `runs/mjw_tp/gen2.wrapper.log`): Go2 pyramidal 7.99 ms (`_update_constraint_efc` 1.9, fused Hessian + Cholesky 0.6, M factor 0.6, line search 0.55); Go2 elliptic 23.96 ms, of which **`_update_gradient_JTCJ_dense` 19.0 ms (79 %)**; SO-101 pyramidal 32.7 ms total, of which
`_update_constraint_efc` 8.5, line search 3.4, the fused `_update_gradient_h_incremental_cholesky` 3.3 (Newton
Cholesky 13 %, Hessian assembly 13 %); SO-101 elliptic 52.1 ms, of which **`_update_gradient_JTCJ_dense` 34.5 ms**
(66 %), `_update_constraint_efc` 5.1, then the full JTDAJ + Cholesky every iteration (2.0 + 1.3). Panda pyramidal
218 ms per env step at 100 Newton iterations (`_update_constraint_efc` 133 ms, i.e. the iteration budget, not the
factorization: Cholesky 8 %); Panda elliptic 649 ms with `_update_gradient_JTCJ_dense` at 454 ms (70 %).

Reading: every scene in the repository (2–18 dofs) and the G1 (43) is on the register Cholesky and the intended M
layout, so the two factorization paths generalize as far as the thresholds reach; the models that fall off them are
not in the repository yet (a humanoid with hands or a dual-arm above 48 dofs takes the cooperative Cholesky, above
64 the blocked one; a model with `jacobian=auto` and 33–64 dofs, e.g. the menagerie G1 at nv 35 under the competitor
protocol, goes sparse and loses the fused path). **The path that does not generalize is the fused / incremental
Hessian: it is pyramidal-only, and three of the four scenes above default to elliptic cones** (Go2, Go1, the SO-101
lift scene; MuJoCo's own recommendation against slip for manipulation and quadrupeds). Off CUDA, MuJoCo Warp sizes
the elliptic Hessian term's grid at `dim_block = d.naconmax` (the "CPU fallback" in `_update_gradient`), so every
Newton iteration launches `naconmax × nv(nv+1)/2` threads that mostly exit on `conid >= nacon`: a capacity-sized
launch per iteration, exactly the cost class the G1 work removed elsewhere. Estimated gain from extending the fast
paths to elliptic cones: if the JTCJ term cost what the pyramidal JTDAJ assembly costs on the same scene (SO-101:
34.5 → ~2 ms per env step), SO-101 elliptic would go from 52 to ~20 ms of GPU time per step, i.e. from 123 K to an
estimated 250–280 K env-steps/s (2.0–2.3×), and Panda elliptic from 9.9 K to an estimated 25 K; Go2 elliptic
(240 K, JTCJ 19.0 of 24.0 ms) to an estimated 700–850 K physics steps/s. That is the elliptic-cone work already under way in the fork
worktree `upstream/mujoco_warp-ellip` (`docs/HANDOFF_elliptic_cone_perf.md`), so it is reported here, not done.
Whether a task uses elliptic cones is a fidelity choice (slip), not a throughput knob: the SO-101 lift scene's
2.4× should not be taken by switching its cone.

## 5. Ledger updates, contamination check, what is left open

- README row "G1 flat throughput" and PARITY §1.4: today's numbers under both contact settings, with the setting and
  time of day; the headline row keeps its numbers (its setting is now stated). GAPS: one row (throughput under the
  preset, kept: fidelity first; the elliptic Hessian as the non-generalizing path). DECISIONS: one row (publish the
  task default; the flex merge may be installed; how to re-measure the headline setting). CHANGELOG: one line.
- Contamination: every timing window here (`ab1`, `ab2`, `gen`, `gen2`, `cc`, `cc2` wrapper logs) started at 0 %
  device utilisation with no process at or above 1 % except WindowServer / Terminal / a Chrome helper at 1.0–2.3 %
  (the queue dashboard tab), and ended the same; the wrapper's "end: device utilisation 95–100 %" readings are the
  job's own last command buffers draining (the `ab1` and `gen2` jobs read 0 % at the end with the same code). The
  `ellip_bench_world` job was granted 2 s after `mjw_generality` released (`runs/gpu_history.jsonl` 19:48:02 →
  19:48:04), no overlap. No row added to `runs/CONTAMINATION_2026-09-25.txt`.
- Tools: `scripts/diagnostics/g1_tp_variants.py` and `g1_step_profile.py` take `contact_cfg` / `solver_cfg` in
  `MJW_TP_VARIANT` (null = the headline setting); `scripts/diagnostics/g1_tp_generality.py` prints the fast-path flags
  of any scene and measures it. Worktree `upstream/mujoco_warp-07a51a6` (read-only, for `PYTHONPATH` A/Bs).
- For the main session: the shared checkout `upstream/mujoco_warp` is 11 commits behind `fork/metalsim`; the A/B
  says fast-forwarding it to `07a51a6` costs the G1 nothing. Not done here (shared checkout, rule 5). Warp
  `upstream/warp-innate` is 3 commits behind for the same reason (flex-only). No fork fix branch was needed
  (`metalsim-tp-regress` not created: there is nothing to fix in the fork).
- Open: (1) the Newton cap hits under the preset (12–28 worlds of 4096 per step): whether those worlds' forces are
  acceptably converged is a fidelity question for the preset's owner (a cheap probe: `iterations` 20 vs 10 under
  the preset on the state-difference protocol, `g1_tp_check.py`); (2) rough terrain was not re-measured under the
  preset (the README's rough 41.9 K is the default-contact number); (3) the elliptic-cone Hessian grid
  (§4) is the one place a fast path is missing for models in the repository, owned by the elliptic-cone session;
  (4) the other machine's Warp `4127c48` is not in this repository's remote, so that commit remains unmeasured here
  (its measured numbers match this machine's, so nothing suggests it matters).

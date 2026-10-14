# Deterministic BatchSim on the G1 task: research before the fix (2026-10-14)

Status: research and experiment design. The experiment is batch_18 (`scratch/b18/batch_18.sh`). Its results go at the
end of this file. Rounds batch_13 to batch_17 tested hypotheses from code reading by toggling switches (the
solver loop's early exit, the fused update, capture reuse, ICB replay). That was confounded: each toggle changed the
solver trajectory and so whether a world reached the iteration cap, which is the event this note identifies as the
trigger (section 3). Toggles are not used as evidence of causation below.

## 1. Prior art: is this already solved?

- **MuJoCo Warp issue #1635 / PR #1636** (wagner-austin, opened 2026-09-02, open, unreviewed;
  https://github.com/google-deepmind/mujoco_warp/issues/1635, https://github.com/google-deepmind/mujoco_warp/pull/1636).
  - Under Warp's `wp.config.deterministic`, the convex narrowphase silently loses contacts. The cause is the same one
    MetalSim found on 2026-10-13: EPA keeps its polytope in global scratch, so "the counting pass reads back values it
    was never allowed to write" and the two passes take different branches.
  - Their fix is the same split as MetalSim's `metalsim-det-ccd`: stage per-collision results, then write contacts in a
    second kernel. They also split the heightfield kernel, which MetalSim does not; MetalSim refuses heightfield GJK/EPA
    in deterministic mode.
  - They note that under graph capture Warp's overflow checks are off, so failures are silent. They measured 64.8x on
    humanoid at 256 worlds captured.
  - The PR author argues that MuJoCo Warp's own `opt.deterministic` track is the better primary path.
- **MuJoCo Warp native determinism (`opt.deterministic`)**:
  - Sources:
    - #562 umbrella: https://github.com/google-deepmind/mujoco_warp/issues/562
    - #1422 (johnnynunez, open): https://github.com/google-deepmind/mujoco_warp/pull/1422
    - #1687 (Taylor Howell, the determinism API), and #1775 (huster-wgm, "Rebase determinism API onto main and fix
      deterministic arithmetic", open): https://github.com/google-deepmind/mujoco_warp/pull/1775
    - Newton #4631 / #4632 (set the option before the model constants are computed):
      https://github.com/newton-physics/newton/pull/4632
  - Approach, in MuJoCo Warp's own kernels rather than Warp's two-pass mode:
    - contacts sorted into a stable permutation after the narrowphase (int64 keys, world first);
    - constraint rows allocated by count / scan / emit instead of atomic slot grabbing;
    - deterministic gathers instead of atomic `qfrc_constraint` / JTDAJ accumulation;
    - an order-fixed reduction for the Newton termination test.
  - #1422 tried Warp's built-in mode on unmodified MuJoCo Warp. It found a scatter-buffer overflow, about 44 GB of
    scratch at 256 worlds (compare our 3.5 GB plan at 1,024 worlds), and lost graph capture inside the conditional
    while body ("Conditional body graph contains an unsupported operation (memory allocation)").
  - Cost reported: about 1.8x per step at 256 worlds (−31 % to −81 % throughput, 1 to 8,192 worlds).
  - Status: open. #1775 still has placeholder ordering hooks, and "ALL ... does not yet promise deterministic full
    simulation". As of 2026-10-10 a commenter on #562 reports bit-identical replicas with the #1533 head.
  - The official answer is still "run on CPU for determinism" (MJWarp FAQ: https://mujoco.readthedocs.io/en/latest/mjwarp/).
- **Warp** (deterministic mode, #1443 / PR #1355, Warp 1.15;
  https://github.com/NVIDIA/warp/issues/1443,
  https://nvidia.github.io/warp/latest/user_guide/execution_and_performance/deterministic_execution.html).
  - Documented limits: "Side effects in the counting pass" (the counting pass suppresses array writes, non-counter
    atomics and printf; a kernel must not decide its counter calls from data it writes in the same launch); overflow
    checks are disabled during capture and replay; deterministic kernels are not supported inside `capture_while` /
    `capture_if` bodies.
  - Upstream has no runtime check for a phase mismatch. MetalSim's tripwire (Warp fork `metalsim`, 2026-10-13) is
    MetalSim's own.
  - I found no Warp issue on a deterministic counter that misbehaves only under capture.
- **Early-exit iterative solvers elsewhere**:
  - PhysX's GPU solver (TGS / PGS) runs fixed position and velocity iteration counts per step. It has no convergence
    early exit, and run-to-run determinism additionally needs `PxSceneFlag::eENABLE_ENHANCED_DETERMINISM`. These are
    general PhysX docs (O3DE, Unity notes; https://www.docs.o3de.org/docs/user-guide/interactivity/physics/nvidia-physx/),
    not checked against the GPU pipeline's source.
  - MuJoCo CPU stops on tolerance and is deterministic, because the stopping test is evaluated in a fixed order.
  - Jolt is deterministic with a fixed iteration count (https://github.com/jrouwe/JoltPhysics/discussions/617).
  - MJX: I found nothing primary on early exit and determinism. Genesis: not found. Both unverified.
  - The general lesson: an early exit is not itself a determinism problem when the convergence test and the iteration
    counter are computed deterministically per world. The problem below is how Warp's two-pass counters interact with
    a counter call that depends on data the same kernel writes.

Conclusion: the CCD part of the problem was found independently upstream and fixed the same way (#1636, unmerged). The
solver iteration count / early exit interaction with Warp's two-pass counters (sections 2 and 3) is not covered by any
prior art I found. Upstream's direction for MuJoCo Warp determinism is its own kernels (`opt.deterministic`), not
Warp's mode. That is a design question for #29 beyond this bug (section 5).

## 2. What flag bit 2 means (Warp fork `metalsim`, `warp/native/metal_deterministic.h`, `warp/_src/metal_deterministic.py`)

A kernel with consumed-return counters (`slot = atomic_add(counter, i, n)` with the result used) runs as two passes:
- **phase 0**: global array stores, non-counter atomics (`WP_DET_SIDE_EFFECT_IF_ACTIVE`; integer `+=` on an array is
  one of them, `warp/_src/deterministic.py`) and printf suppressed. Each counter call records (destination, count) at
  `thread * records_per_thread + call` and returns 0.
- Then sort, scan and phase 1, where each call returns its slot. The writeback then publishes the totals and zeroes
  every thread's two call cursors (phase 0 and phase 1).

The cursors and other zeroed state live in one shared zero pool. Every deterministic launch reuses it from offset 0 and
must leave it zero.

The sticky flag (`state.overflow`, one int32) gets:
- **bit 1** (overflow): a call past `records_per_thread`, or a record past the plan's capacity.
- **bit 2** (phase mismatch), set three ways:
  - (a) in phase 1, `counter_add` sees a call number at or past the thread's phase-0 count (`call >= cursors[idx]`);
  - (b) in phase 1, the record exists but with a different destination or count;
  - (c) after phase 1, `det_counter_writeback` finds a thread whose phase-0 and phase-1 call counts differ.

So bit 2 alone cannot say which of (a), (b) or (c) happened, or whether the phase-0 count was a real count or a stale
cursor. batch_18 uses a diagnostic Warp build (`upstream/warp-diag`, not pushed) that sets distinct bits:
- 2: (a) with a phase-0 count of 0 (no phase-0 record at all);
- 16: (a) with a nonzero phase-0 count;
- 8: (b);
- 4: (c).

Uncaptured launches are checked after each launch. In a capture the flag is read only by the host after a replay
(BatchSim.check_overflow), or per launch by MetalSim's probes.

## 3. `_solve_done` and the candidate mechanisms

`_solve_done` (MuJoCo Warp fork, `solver.py`, Newton path; one thread per world, launched once per solver iteration):

```
if ctx_done_in[w]: return
solver_niter_out[w] += 1                         # integer atomic: suppressed in phase 0
done = <tolerance tests on inputs>
if done or solver_niter_out[w] == opt_iterations:   # read back
  ...
  ctx_done_out[w] = True                         # store: suppressed in phase 0
  was_solving = atomic_add(nsolving_out, 0, -1)  # consumed slot counter (assigned even with the early exit off)
  if EARLY_EXIT and was_solving == 1: zero the remaining iterations' ICB range lengths
```

batch_17's captured per-launch probe named `_solve_done` (flag 2, 2–4 hits) as the first and only tripping launch, at
64 worlds with the early exit and the fused update on, with each of them off, and in every tripping configuration.

Candidate mechanisms:
- **M1, read-back of a suppressed increment.** In phase 0 the read-back sees the old count, so a world that is not
  converged but reaches the cap in this iteration makes the counter call in phase 1 only. This is the documented
  "side effects in the counting pass" limitation.
  - Predicts: kind 2 (no phase-0 record), in `_solve_done`, only in steps where some world reaches the cap.
  - Predicts: independent of capture (eager runs trip too when the cap is reached) and of the probe's inserted kernels.
  - Predicts: removed by keeping the count in a local.
- **M2, stale cursors.** A deterministic launch whose passes ran only partly, for example cut by an indirect-command-
  buffer range, a mid-graph recorded memset or a host operation, leaves nonzero cursors in the shared zero pool. The
  next deterministic launch then starts from them.
  - Predicts: kinds 16 / 4 / 8 (a nonzero phase-0 count), possibly in kernels other than `_solve_done`.
  - Predicts: capture-only, not tied to cap steps, not removed by the M1 fix.
- **M3, plan or pool aliasing under capture.** Plans recreated after the cache clear (more than 64), or a pool that
  grows mid-capture so that recorded launches see different buffers.
  - Predicts: capture-only, kinds 8 / 16, not tied to cap steps.
- **M4, replay hazard.** A missing barrier between a deterministic launch's passes in ICB replay.
  - Predicts: varying kinds and kernels from run to run, capture-only.
- **The 1,024-world probe run that stayed clean (batch_17):** under M1 it reached no cap in its shorter horizon (22
  frames against 50). Under M2–M4 the probe's extra kernels changed something.
  - Distinguish by recording the number of worlds at the cap in every step.

## 4. Experiment design (batch_18)

Every run records, per physics step, the deterministic launches that set the flag (diagnostic bits) and the number of
worlds that reached the iteration cap (`OverflowType.ITERATIONS` in `d.overflow`, cleared per step).
`scripts/diagnostics/det_step_probe.py`; the diagnostic Warp build is first on `PYTHONPATH`.

1. Stack scene, 64 worlds, iteration cap 2 (every world reaches it every step), deterministic, **uncaptured**, current
   MuJoCo Warp pin. M1 predicts a trip in `_solve_done`, kind 2, every step; M2–M4 predict none.
2. The same **captured**. M1 predicts the same kernel, kind and steps.
3. G1, 64 and 1,024 worlds, captured, everything on (as batch_16 failed it). M1 predicts trips only in steps with cap
   worlds, `_solve_done`, kind 2.
4. 1–3 with the M1 fix (MuJoCo Warp copy `upstream/mjw-solvedone`: count kept in a local; the counter consumed only
   with the early exit). M1-only predicts all clean. Then G1 deterministic at 64 and 1,024 worlds, captured, everything
   on, the configuration batch_16 failed, with BatchSim.check_overflow.

Decision rule (in the batch, no stop to report):
- If 1 trips in `_solve_done` with kind 2 only in cap steps, and every run of 4 is clean: M1 is confirmed and the fix
  works. Then commit the fix to the MuJoCo Warp fork (branch `metalsim-det-solvedone`), run
  tests/test_batch_deterministic.py (new `test_iteration_cap_repeats`), time G1 and the coupled box in deterministic
  mode against default mode, move the MuJoCo Warp pin, and update #29 and this note.
- If 1 is clean but 2 trips (capture-only): M2–M4. Classify by kind and kernel from 2, keep the fix uncommitted, and
  record.
- If 4 still trips: record the kernel and kind of the remaining trip for the next round.

## 5. Beyond the bug (for #29's scope)

Upstream MuJoCo Warp is building determinism into its own kernels (sorted contacts, count-scan-emit rows, deterministic
gathers) rather than relying on Warp's two-pass mode, and reports about 1.8x cost at 256 worlds. MetalSim's
`BatchSimOptions(deterministic=True)` uses Warp's mode: +3–4 % on the coupled box when it works. It needs every MuJoCo
Warp kernel that consumes a counter to be phase-safe, and the tripwire is how we find the ones that aren't.

Once #1775 lands, the trade to weigh is to switch to `opt.deterministic` (upstream-maintained, slower) or to keep Warp's
mode (fast, our tripwire, our fixes). Our fixes (CCD split, `_solve_done`) are worth offering upstream either way: the
CCD one duplicates #1636.

## 6. Results (batch_18, 16-inch)

The runs do not match any single mechanism's prediction: see the per-run lines.

- `pre_stack_eager` (stack, 64 worlds, capture 0, cap 2): steps with cap worlds 0/20, steps with trips 0, trips only in cap steps True, kernels {}, flag values {}
- `pre_stack_capt` (stack, 64 worlds, capture 1, cap 2): steps with cap worlds 0/40, steps with trips 0, trips only in cap steps True, kernels {}, flag values {}
- `pre_g1_64` (g1, 64 worlds, capture 1, cap 20): steps with cap worlds 60/60, steps with trips 60, trips only in cap steps True, kernels {'_solve_done__locals__kernel': 237}, flag values {'6': 237}
- `pre_g1_1024`: no result
- `fix_stack_eager` (stack, 64 worlds, capture 0, cap 2): steps with cap worlds 0/20, steps with trips 0, trips only in cap steps True, kernels {}, flag values {}
- `fix_stack_capt` (stack, 64 worlds, capture 1, cap 2): steps with cap worlds 0/40, steps with trips 0, trips only in cap steps True, kernels {}, flag values {}
- `fix_g1_64` (g1, 64 worlds, capture 1, cap 20): steps with cap worlds 60/60, steps with trips 0, trips only in cap steps True, kernels {}, flag values {}
- `fix_g1_1024`: no result
- G1 deterministic, captured, everything on, with the fix (det_cost_g1.py, check_overflow): exit code 64 worlds 0, 1,024 worlds 1

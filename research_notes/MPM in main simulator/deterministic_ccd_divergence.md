# Deterministic BatchSim: stacked-boxes divergence (2/256 worlds from step 139) — root cause and fix (2026-10-13)

Suspects first (source reading, before any run), then the batch_05 evidence, the root cause and the fix (end of file).

* Box-box pairs go through MuJoCo Warp's CCD kernel (GJK -> EPA -> multicontact; collision_driver.py:78, 866-870;
  collision_convex.py:885-1003); plane-box is the primitive path. The single-box and panda scenes never launch CCD (they
  repeat). Box-box contact starts at about step 138.
* Warp's RUN_TO_RUN on Metal makes consumed int32 counters two-phase: phase 0 runs the kernel with global stores
  suppressed (except counter targets and zeroed scratch) and counter_add returning 0, then a sort / scan, then phase 1.
  Nothing checks that phase 1 makes the same counter calls as phase 0 (metal_deterministic.py:225-246;
  metal_deterministic.h:175-194, 259-271).
* ccd_kernel has two consumed counters (nccd_in for EPA scratch rows, nacon_out in write_contact). In phase 0 every
  thread gets ccdid 0 and runs EPA on scratch row 0 with its polytope stores suppressed, reading uninitialised or stale
  wp.empty memory, so its contact count (and nacon reservations) can differ from phase 1. Extra phase-1 calls read stale
  prefixes from a shared scratch pool filled in scheduling order; fewer calls leave reserved slots unwritten. Neither
  sets the overflow flag. Fits: same outlier hashes, scheduling-dependent worlds (all = 0 mod 16).
* Lesser suspects: single-threadgroup in-kernel finalize (WP_METAL_DET_TG_FINALIZE=0), small-launch counter resolve
  (WP_METAL_DET_SMALL_RECORDS=0). BatchSim.check_determinism does not cover `module="unique"` kernels (name prefix).
* Diagnostics: mjDSBL_NATIVECCD A/B (box-box via primitive box_box); env A/Bs; contact audit with a sentinel worldid
  at steps 137-140; per-launch hashing; a phase-mismatch tripwire in the Warp fork.
* Fixes: MuJoCo Warp fork — split CCD into a counter-free solve kernel (scratch indexed by collision id) and a
  write_contact kernel; Warp fork — permanent phase-mismatch tripwire; BatchSim — check unique modules via the graph's
  module execs.
* Newton's SolverMuJoCo uses the same module-option scheme; with use_mujoco_contacts=True (its default) it would hit the
  same CCD path; Newton's determinism tests cover only its own collision pipeline.

## Evidence (batch_05, 16-inch, runs/mpm_main/batch_05)

* Native CCD off (`mjDSBL_NATIVECCD`: box-box through MuJoCo Warp's primitive `box_box`, no GJK/EPA, no `nccd`
  counter): stack, 256 worlds, 200 steps, two runs in one process: **no divergence** (`in_process_first_diff [null]`).
  The contact audit with CCD off: every world's state identical through step 150 (2,048 contacts per step).
* Warp-side switches (single-threadgroup finalize off, small-record counter resolve off): still diverge at step 139 in
  2 worlds (different worlds per run: 144/208, 32/48). The finalize and resolve paths are not the cause.
* Contact audit, CCD on (per-world box-box contact counts from `d.contact`, steps 130-150, two runs): steps 130-138
  identical. At step 139 both runs have 1,024 contacts and no holes (no slot left unwritten), but the box-box count
  histogram is `[1, 254, 1]`: one world has 0 box-box contacts, one has 2, the other 254 have 1 — in both runs, in
  different worlds per run (all multiples of 16). From then on worlds diverge (2 at 139, 7 by 146). The physics gives
  every world the same single box-box contact at step 139; a contact moved from one world's slot to another's.

## Root cause

MuJoCo Warp's `ccd_kernel` consumes two slot counters: `ccdid = atomic_add(nccd, ...)` picks the EPA / multicontact
scratch row and `write_contact`'s `atomic_add(nacon, ...)` picks the contact slot. Under Warp's RUN_TO_RUN mode a
kernel with consumed counters runs twice: phase 0 with every global array store suppressed and each counter call
returning 0 (it only records how many calls each thread makes), then a sort / scan, then phase 1 with real slots.
In phase 0 every thread gets `ccdid = 0` and runs EPA and multicontact on scratch row 0 with their polytope stores
dropped, so they read whatever row 0 holds (stale data from the previous launch, in scheduling order). The contact
count `ncollision` they reach decides how many `write_contact` calls they record. In phase 1 the same thread runs on its
own row with real stores and can reach a different count. Slots are then assigned for phase 0's counts but used by
phase 1's calls: a thread that now writes 2 contacts where it reserved 1 writes into the next thread's (next world's)
slot, which that thread, having reserved 1 and now writing 0, leaves to it. Total and holes are unchanged and nothing
sets the overflow flag — exactly the `[1, 254, 1]` histogram. Which threads mismatch depends on what row 0 held, so
on scheduling: hence different worlds per run. Native CCD off removes the EPA scratch (and its counter) and the
divergence goes away.

Nothing in Warp checks that phase 1 makes the calls phase 0 recorded; any deterministic kernel whose counter calls
depend on data it writes and reads back in the same launch has the same hazard. In MuJoCo Warp that is the two CCD
kernels (`ccd_kernel`, `ccd_hfield_kernel`); the primitive narrowphase and the constraint kernels decide their calls
from inputs only.

## Fix (MuJoCo Warp fork, branch `metalsim-det-ccd`; `patches/mujoco_warp/0017-deterministic-ccd-split.diff`)

In deterministic mode `convex_narrowphase` runs non-heightfield CCD as two kernels (`ccd_det_kernels_builder`):

1. a solve kernel with no slot counter: GJK, EPA and multicontact as before, the scratch row being the thread id (the
   launch width, which is also the grid stride, is capped at `naccdmax`, so rows are private and reused across a
   thread's collisions); it stages each collision's contact count, distances, positions and frame. Without a consumed
   counter the kernel is single phase and its stores are real.
2. a write kernel over the same collisions that recomputes margin / gap / material and calls `write_contact` once per
   staged contact; its calls depend only on the staged count, so phase 0 and phase 1 agree.

The default (non-deterministic) path is untouched, so default-mode results stay bitwise identical. On CPU, the split
produces the same contacts as the single kernel bit for bit at 40 sampled steps of the stacked boxes (8 worlds, 135
box-box contacts; `probes/ccd_split_equiv.py`, runs/mpm_main/batch_06/ccd_split_equiv_cpu.log). The heightfield kernel
is not split: in deterministic mode a primitive against a heightfield raises in MuJoCo Warp, and BatchSim refuses
meshes of more than 256 vertices that can touch a heightfield (both take GJK/EPA there); meshes of at most 256 vertices
take the plane path, which uses no scratch and repeats. `BatchSim.check_determinism` now also covers MuJoCo Warp's
`module="unique"` kernel modules (CCD, solver, smooth builders), found by their kernels' source module.

## Tripwire (Warp fork, branch `metalsim-det-tripwire`; `patches/warp/det-phase-mismatch-tripwire.diff`)

Upstream Warp documents the hazard as a limitation ("Side effects in the counting pass" in
docs/user_guide/execution_and_performance/deterministic_execution.rst: counter calls must not depend on scratch writes
made earlier in the launch) but checks nothing, on CUDA or in the Metal port: phase 1 (CUDA: `record_slots`; Metal:
the record at `thread * records_per_thread + call`) simply returns whatever slot the record holds. On Metal both phases
already count each thread's calls (`cursors[0:dim]` phase 0, `cursors[dim:2 dim]` phase 1, cleared by
`det_counter_writeback`), so the check is cheap:

* in phase 1, `counter_add` compares the call with phase 0's record of it: a call number beyond phase 0's count, or a
  different destination or count, sets bit 2 of the sticky deterministic flag (bit 1 stays the overflow);
* `det_counter_writeback` compares each thread's two counts before clearing them, catching fewer calls in phase 1;
* outside a graph capture the launch raises a RuntimeError naming the kernel (as for an overflow);
  `BatchSim.check_overflow` reads the flag after captured replays and raises "phase mismatch".

Warp's unit tests get a kernel that breaks the rule both ways (more and fewer calls in phase 1) and one that keeps it.
The CUDA path is unchanged (nothing here to test it on). batch_06 runs MuJoCo Warp's single CCD kernel
(`MJW_CCD_DET_SPLIT=0`) at 256 worlds expecting the error, and every other deterministic run with the tripwire on, so a
false positive in any MuJoCo Warp kernel would show.

## Validation (batch_06, 16-inch, runs/mpm_main/batch_06)

With both fork branches first on `PYTHONPATH`: Warp's Metal deterministic tests 15 OK (the tripwire tests included); stacked boxes at 4 worlds take the split kernels (`ccd_det_kernels_builder` modules, RUN_TO_RUN); at 256 worlds and 200 steps, two runs in one process and three processes (two cold kernel caches, one warm) are bitwise identical; the contact audit gives every world the same box-box contact count at every step 130-160 and no state difference; tests/test_batch_deterministic.py 7 passed. Tripwire A/B: the single CCD kernel (`MJW_CCD_DET_SPLIT=0`) at 256 worlds now stops with "Warp deterministic-mode counter phase mismatch" from `BatchSim.check_overflow` instead of diverging silently.

Pinned on 2026-10-13 (batch_08): MuJoCo Warp `4d05cb7` (code `cfa290b`), Warp `52c09cdd` (code `9346cf72`, rebased onto the coupled-speed pin `31cf79d7`). Newton's `SolverMuJoCo(deterministic=True)` with MuJoCo contacts takes the same CCD path and gets the fix once its MuJoCo Warp includes it.

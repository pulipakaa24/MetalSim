# Deterministic BatchSim: stacked-boxes divergence (2/256 worlds from step 139) — suspects (2026-10-13)

Source reading (MuJoCo Warp fork, Warp fork, generated MSL in the cold kernel cache); nothing run yet.

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

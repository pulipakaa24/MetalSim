Opt-in run-to-run bitwise repeatability for `BatchSim`: `BatchSimOptions(deterministic=True)` runs MuJoCo Warp's kernels in Warp's RUN_TO_RUN mode. The default mode is unchanged and stays non-repeatable.

- BatchSim sets the mode before MuJoCo Warp's first launch, and only one mode is allowed per process. It sets the record bound for `smooth`, checks after capture that every replayed MuJoCo Warp module was built deterministic (now including the `module="unique"` kernel modules), and reports the deterministic overflow and phase-mismatch flags in `check_overflow`. Heightfield contacts that go through GJK/EPA are refused in this mode.
- Why two stacked boxes diverged (2 of 256 worlds from step 139): MuJoCo Warp's CCD kernel picked its EPA scratch row with a slot counter. Warp's counting pass suppresses stores, so it ran EPA on stale scratch and could count a different number of contacts than the execution pass then wrote, and a box-box contact landed in another world's slot. The full analysis is in research_notes/MPM in main simulator/deterministic_ccd_divergence.md.
- MuJoCo Warp pin: in deterministic mode CCD now runs as a solve kernel without a slot counter, followed by a contact-write kernel (fork branch `metalsim-det-ccd`; on CPU it gives the same contacts as the single kernel, bit for bit).
- Warp pin: a tripwire, rebased onto coupled-speed's `31cf79d7`, raises "Deterministic counter phase mismatch" when a kernel's slot-counter calls differ between the counting and the execution pass. Upstream Warp documents this as a limitation but does not check it at run time.
- Validation (batch_06): two stacked boxes at 256 worlds are bitwise identical within a process and across processes (cold and warm kernel cache), and every world keeps its contact. `tests/test_batch_deterministic.py` passes (7). The old single kernel now raises the phase-mismatch error. batch_08 ran Warp's and Newton's determinism tests and the rigid, render and MPM suites with the new pins before moving them (runs/mpm_main/batch_08/STATUS.txt).
- **G1 (batch_13-19, research_notes/MPM in main simulator/deterministic_g1_capture.md):** deterministic runs tripped Warp's counter tripwire whenever a world reached the solver's iteration cap: MuJoCo Warp's `_solve_done` read the iteration count back after incrementing it in place (an increment Warp's counting pass suppresses), so the early exit's slot counter was called in the execution pass only. Fixed in the MuJoCo Warp pin `8eb51fe` (count kept in a local); test `test_iteration_cap_repeats`. The earlier suspects (the early exit, the fused update) only changed whether some world reached the cap. In deterministic mode the pin also takes the unfused Newton update: the fused kernel reads back its own change counter within the launch, and its deterministic plan at 1,024 worlds under capture asked for more than 2^31 scratch elements (batch_18); unfused costs nothing measurable (batch_17).
- Cost of deterministic mode (16-inch, median of 3 interleaved repetitions):

| scene | worlds | default | deterministic | cost |
  |---|---|---|---|---|
  | G1 flat env step (batch_19b) | 64 | 7.66 ms | 13.39 ms | +74.8 % |
  | G1 flat env step (batch_19b) | 1,024 | 15.46 ms | 32.59 ms | +110.8 % |
  | coupled box step (batch_17, same session) | 64 | 23.87 ms | 24.32 ms | +1.9 % |
  | coupled box step (batch_10 default, batch_17 deterministic) | 1,024 | 228.78 ms | 231.50 ms | +1.2 % |
unfused @ 64 worlds: first tripping launch _solve_done__locals__kernel (module _solve_done__locals__kernel_629f5f1d, dim 64, flag 2); 2 of 3773 captured launches set it; kernels {'_solve_done__locals__kernel': 2}
no_early_exit @ 64 worlds: first tripping launch _solve_done__locals__kernel (module _solve_done__locals__kernel_6aa55336, dim 64, flag 2); 4 of 2373 captured launches set it; kernels {'_solve_done__locals__kernel': 4}
both_on @ 64 worlds: first tripping launch _solve_done__locals__kernel (module _solve_done__locals__kernel_629f5f1d, dim 64, flag 2); 4 of 2373 captured launches set it; kernels {'_solve_done__locals__kernel': 4}
- Cost of switching 1 and 2 off in deterministic mode (coupled box, deterministic, captured, median of 3 interleaved repetitions, batch_16):

  | worlds | early exit + fused (as now) | both off | change |
  |---|---|---|---|
  | 64 | 24.96 ms | 53.96 ms | +116 % |
  | 1,024 | 237.13 ms | 289.60 ms | +22 % |

  Which switch carries the cost is in batch_17's decomposition (#29 comment). Switching them off is neither cheap nor sufficient, so the fix will be in the kernels (deterministic passes contained in the early-exit ranges, a counter-free fused update) unless the decomposition shows otherwise.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

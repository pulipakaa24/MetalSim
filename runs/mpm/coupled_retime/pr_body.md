## What

This PR brings the MPM docs in line with the measured record and re-times the coupled scenes against the L4.

**Docs**
- **STATUS / GAPS MPM rows.**
  - All six floor scenes are inside the L4's perturbation ensemble (multi_material 0.80, ANYmal 0.94; MPM.md 8e). They no longer say "1.15x outside".
  - The pins are now the current ones: Warp 95f6f121, Newton 3fd1f052, MuJoCo Warp f1b35e2.
  - The mesh-SDF barrier fix was checked on GitHub in both pinned Newton 1.7 commits (a4be3604, 9f317390), and every pinned fork commit is pushed. Both stale to-dos are gone.
  - Franka-Pour status and its seed study are included.
- **MPM.md 8d, the 2026-10-12 02:32 GPU hang.** It is most likely not Newton's cylinder-on-plane test.
  - The test passed in its 3 other runs and in 20 isolated repetitions.
  - The 3rd 25M-grain particle build of the night was granted at 02:32:49, and the build killed at 02:20 kept running on the GPU with no owner.
  - All three 10-12 wedges followed that 25M-instance build, which went past Metal's 2^24 instance limit without extended limits. It is fixed now (two-level grains), and 25M passes on both Macs.
  - The evidence is strong but circumstantial, and the text says so. The queue-history excerpt and the test's watchdog log are under `runs/rigid17/out/`.
- **MPM.md, superseded "Open:" lines** (sections 3, 4, 5, 7, 1b) are marked resolved with pointers. The history is kept.
- **MPM.md 8i** points to the seed study. **PARITY §1b.5:** the Isaac seeds are waiting for L4 time.
- **RENDER_COVERAGE:** the particle-rendering rows now point to PRs #4, #9, #14, #15 (T0 and T2 render particles; T1 and RT do not).

**Timing: new MPM.md section 8j, data in `runs/mpm/coupled_retime/`**

All runs were on the 16-inch, as exclusive timing jobs with the GPU idle. Three repetitions were interleaved, using the L4's 2 s protocol. ms/frame, `metal_fast_math="off"`:

| scene | L4 | Metal | Metal / L4 |
|---|---|---|---|
| two-way, eager | 73.0 | 79.6 | 1.09x |
| two-way, captured | 31.0 | 54.4 | **1.76x** |
| ANYmal, eager | 58.9 | 39.6 | 0.67x |
| ANYmal, captured | 24.2 | 31.2 | **1.29x** |

**Where the time goes**
- **Two-way.** The frame cost grows during the run on Metal only. Early in the run, frames 6-65, captured two-way takes 36.9 ms.
  - The growth is one kernel, `_bsr_mv_listed_rows`, the rigidity product in the rheology loop: 0.6 ms/frame early, 19.3 ms/frame late.
  - The likely cause is that the long body rows of IJtm are summed by one thread each. CUDA uses Warp's tiled `bsr_mv` for them.
- **ANYmal.** Gauss-Seidel takes 2.6x the L4's time (latency-bound), and many small MuJoCo Warp kernels take 3-11x.

**What would close it.** A row-tiled IJtm product on Metal, which should bring two-way to about 1.2x the L4. It would need a floor re-check. Not started.

## STATUS
Done. MPM is **not** marked done, because two items are open:
1. Isaac Franka-Pour seeds 1-4: waiting for L4 time.
2. Captured coupled-scene speed, as above.

Hand-off notes: `runs/mpm/coupled_retime/NOTES.md`.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

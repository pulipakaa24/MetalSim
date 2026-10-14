## 8k. Coupled scenes: closing the gap (2026-10-13; Warp `31cf79d7`, Newton `addd10cb`)

Research: `research_notes/coupled_speed/NOTE.md`. Logs, JSON and the batch scripts: `runs/mpm/coupled_speed/`
(batch_02 to batch_08, bf1/, hv1/, any1/, b08/). Every timing ran on the 16-inch as an exclusive `timing` job on an
idle GPU, with three interleaved repetitions unless noted. ms/frame figures use the L4's protocol (2 s, median).

**1. The two-way growth was the rigidity product IJtm** (section 8j, item 1). `probes/rigidity_mv_probe.py` times
the two in-loop products separately (eager, synchronized):
- IJtm has 12 rows (two per body), and the longest row grows to about 1,170 blocks as the boxes sink.
- Untiled, it takes 29-40 ms per 50 calls. J (655,360 rows, at most 2 blocks each) takes about 9-12 ms, either way.

Warp's `bsr_mv` sums such rows with its tiled kernel on CUDA only (`nnz * block_size > 2 * 64 * rows`; the L4's nsys
profile shows one tiled and one untiled product per iteration). Metal summed every row serially in one thread.
- Warp `31cf79d7`: `warp.config.metal_bsr_mv_tiles` (default on, `WP_METAL_BSR_MV_TILES=0` off) makes CUDA's choice
  on Metal. Metal's `tile_sum` is CUDA's shuffle tree, so the summation order is the L4's.
- Newton `addd10cb` (first of three commits): `_ListedRowsMv` sends a matrix that `bsr_mv` would tile through `bsr_mv`
  itself.
- First launches: `probes/bsr_mv_tiles_check.py`, under a watchdog.
- IJtm: 29-40 -> 12.1-12.9 ms per 50 calls.
- Late two-way GPU kernel sum per frame: 55.3 -> 36.4 ms.

**2. The ordered body-force sum** (two-way example, 3.0 ms per frame). Every body scanned all 655,360 grid nodes, in
every substep.
- Newton `addd10cb` (third commit): compact the nodes in contact with a body first (flag, integer scan, scatter;
  capture-safe), then the 256 fixed lanes run over that list.
- 1.05 -> 0.38 ms per call (`probes/body_force_compact_check.py`, exact compaction, 5e-7 from float64).
- `NEWTON_METAL_BODY_FORCE_COMPACT=0` restores the full scan.

**3. The proxy-wrench harvest** (`SolverImplicitMPM.coupling_harvest_proxy_wrenches`, used by Franka-Pour's
`SolverCoupledProxy` and section 11's MPMCoupling). With one world, every proxy body scanned every active collider
node.
- Newton `addd10cb` (second commit): the same compaction, per world range.
- `probes/harvest_compact_check.py`: one world, 16 or 64 bodies, 1.6 -> 0.33 ms per call.
- `NEWTON_METAL_HARVEST_COMPACT=0` restores the full scan.

**Results** (ms/frame, Metal / L4):

| scene | before (8j) | IJtm tiled | + body-force compaction | L4 |
|---|---|---|---|---|
| two-way, captured | 54.4-54.8 (1.76-1.77x) | 40.0-40.4 (1.29-1.30x) | **37.6 (1.21x)** | 31.0 |
| two-way, eager | 78.8-79.6 (1.08-1.09x) | 61.5-62.1 (0.84-0.85x) | **61.3 (0.84x)** | 73.0 |
| ANYmal, captured (control) | 29.9-31.2 (1.23-1.29x) | 29.4 (1.21x) | unchanged | 24.2 |

MPMCoupling (`probes/multibody_sand.py`, 1 world, about 73k particles), harvest compaction off -> on:
- The harvest alone: 15.6-15.8 -> 1.5-1.7 ms per call.
- Coupled step: 47.7 -> 32.3 ms (4 bodies), 49.3 -> 34.2 ms (16 bodies), 80.7 -> 65.2 ms (64 bodies).

Franka-Pour at 2,048 envs is unchanged: 1.80 s per policy step with the tiled choice on or off, and with the harvest
compaction on or off.

**Results: bitwise and parity.**
- Pure-MPM floor scenes (granular, multi_material, plate drag, column) and ANYmal are bitwise unchanged; they use no
  `bsr_mv` and no proxy harvest.
- Two-way and Franka-Pour change rounding, because the summation order changes:
  - for IJtm, it becomes CUDA's order;
  - for the two compactions, it becomes another fixed order. The L4 sums these with float atomics, so there is no CUDA
    order to match.
- All of them repeat bitwise run to run: two-way recordings, Franka-Pour 64 envs × 30 steps, MPMCoupling 30 steps.
- Two-way stays inside the L4's floor ensemble (worst-frame ratio 0.83, as before).
- Each switch off restores the previous behaviour bitwise:
  - Franka-Pour with the tiled choice off equals the trees without the change (batch_06);
  - with the harvest compaction off, it equals the trees without that change (batch_hv1).
- On the pinned commits: `tests/test_mpm.py` 15 passed, 1 xfailed; `tests/test_mpm_main.py` 15 passed. The vendored
  `implicit_mpm` equals `addd10cb` (`scripts/vendor_mpm152.py --check`).

**ANYmal: where the 1.2x is** (`probes/anymal_probe.py`, per frame, captured robot graph + eager sand step):
- Robot 8.3 ms, sand 22.2 ms.
- The colour launch with 1 / 2 / 4 / 8 threadgroups per GPU core gives bitwise equal states; 2, the current choice, is
  already the fastest (1 per core: sand 26.2 ms).
- Newton's reordered Gauss-Seidel (`solver="gs-soa"`) is bitwise equal to `"gs"` but slower (sand 26.0 ms).
- On the L4, the nsys kernel time for the same frame is about 10.5 ms (MPM about 9.1, robot about 1.4) against a
  24.2 ms frame, so most of the L4's frame is host launch cost of its eager sand step.
- What remains on Metal:
  - the MuJoCo Warp robot graph: 8.3 ms against about 1.4 ms of L4 kernel time; that is the rigid stack's own work;
  - the latency-bound Gauss-Seidel colour passes (sections 7, 8).
- The "captured" comparison is like for like: the example captures the sand step only for the `"fixed"` grid and only
  on CUDA, and the default grid is `"sparse"`. So both GPUs capture only the robot.

**Left on two-way** (late profile, `bf1/profile/joined_late_bf.md`):
- Gauss-Seidel: 7.0 vs 3.2 ms.
- Warp's tiled `bsr_mv`: 1.33 vs 0.12 ms over 51 launches. Only 12 rows x 3 x 64 lanes run, so the GPU sits mostly
  idle. A larger tile would cut it but would leave CUDA's summation order.
- Friction / impulse kernels: 1.5-3.4x.

# Colored Gauss-Seidel throughput for the implicit-MPM stress solve on Metal

Scope: how to make each iteration of Newton 1.5.2's `SolverImplicitMPM` colored GS / nonsmooth projected GS stress solve cheaper on an Apple M4 Max without changing results. Baseline profile from the brief: ~250 iterations per step; each iteration is 8 color dispatches (~20 us each, a few thousand strain nodes per color), a contact solve (~25 us), and a residual every 5 iterations. Dispatch latency is ~3-4.5 us.

Primary sources used: the Mixed MPM paper PDF (Daviet, TOG 45(4) art. 151, SIGGRAPH 2026, [project page](https://research.nvidia.com/labs/prl/mixed_mpm/), [PDF](https://research.nvidia.com/labs/prl/mixed_mpm/files/mixed_mpm.pdf)); the local Newton checkout at `/Users/aditya/robosim/upstream/newton` (commit `90e23324`, 2026-09-24), files `newton/_src/solvers/implicit_mpm/{solve_rheology.py, rheology_solver_kernels.py, solver_implicit_mpm.py, implicit_mpm_solver_kernels.py}`. Line numbers below refer to that checkout. Web sources are cited inline.

## What NVIDIA's own implementation and paper do (Newton implicit_mpm on CUDA), and published CUDA numbers

### Takeaway
On CUDA, Newton uses the same structure you have: one launch per color, 8 colors for Q1 velocity with per-voxel strain nodes, and a grid-stride launch of `2 x SM count` blocks of 64 threads. The iteration loop runs in a CUDA conditional graph (`wp.capture_while`), checking the residual every 5 iterations. Newton has no fused or persistent multi-color kernel. Its only throughput variants are an entry-major SoA reordered-layout GS (`gauss-seidel-soa`, bitwise-equivalent in algorithm) and a "batched" GS that merges colors with mass-split Jacobi inside each batch (a different algorithm). The paper says plainly that small problems do not saturate the GPU within one color.

### Cited Findings
- **Paper, solver structure.** Algorithm 1 alternates a Jacobi impulse (contact) iteration with a colored GS stress iteration, after a stress and impulse warm start, "until increment below threshold or iterations exceeded". — [Mixed MPM PDF, §5, Alg. 1](https://research.nvidia.com/labs/prl/mixed_mpm/files/mixed_mpm.pdf)
- **Paper, contact solve.** The contact solve is a Jacobi iteration with mass splitting and the isotropic local contact solver of Daviet 2020. "For Q1 collision nodes, G is the identity matrix, meaning that all contacts are independent and the Jacobi iteration is exact." — [Mixed MPM PDF, §5.1, Alg. 2](https://research.nvidia.com/labs/prl/mixed_mpm/files/mixed_mpm.pdf)
- **Paper, why GS for stress.** Mass-split Jacobi "is also an option", but stresses "need to propagate much farther", so GS's faster convergence is "generally appealing". A "grid-based coloring scheme" keeps the solve deterministic. "If determinism is not a requirement, atomic-based versions would also be possible and possibly faster; we have not explored those." — [Mixed MPM PDF, §5.2](https://research.nvidia.com/labs/prl/mixed_mpm/files/mixed_mpm.pdf)
- **Paper, color count.** "For Q1 velocities and discontinuous stresses, we need 8 colors (all strain nodes belonging to the same voxel are solved sequentially). For B2 or B3 velocities, we need 64 colors, so that colored Gauss–Seidel becomes less competitive to Jacobi — unless the grid is large enough for each color to saturate the GPU by itself." — [Mixed MPM PDF, §5.2](https://research.nvidia.com/labs/prl/mixed_mpm/files/mixed_mpm.pdf)
- **Paper, choice of W_i.** The choice of the local operator W_i "affects the convergence, but not the final result" of the iterative Jacobi or GS. They use a per-node eigen-decomposition of the 6x6 Delassus diagonal block, done once at solver initialization (Alg. 3). — [Mixed MPM PDF, §5.2, Alg. 3](https://research.nvidia.com/labs/prl/mixed_mpm/files/mixed_mpm.pdf)
- **Paper, memory traffic.** The 6x3 B blocks are compressed to a single 3-vector per velocity-stress pair, which "greatly reduces the amount of data that need to be read from global memory in the Gauss–Seidel iterations". — [Mixed MPM PDF, §6 Implementation details](https://research.nvidia.com/labs/prl/mixed_mpm/files/mixed_mpm.pdf)
- **Paper, warm start and termination.** Stresses are warm-started through G2P/P2G transfers, which works better than grid-to-grid interpolation. Termination uses the l2 and l-infinity norms of the plastic strain delta, "typically 10^-3 and 250" (tolerance, max iterations). — [Mixed MPM PDF, §6](https://research.nvidia.com/labs/prl/mixed_mpm/files/mixed_mpm.pdf)
- **Paper, local solver cost.** Root-finding iterations: mean 6, 99th percentile 12, max 19, over 10^6 random local problems. The root-finding "usually is not the bottleneck". An isotropic variant that skips root-finding costs "only slightly lower" per iteration and underperforms overall. — [Mixed MPM PDF, §6.1, Table 3](https://research.nvidia.com/labs/prl/mixed_mpm/files/mixed_mpm.pdf)
- **Paper, small-problem saturation.** On a Q1-dP1 instance with 38k velocity and 90k strain nodes: "As this instance is too small to saturate our GPU for a single Gauss–Seidel color, we also see good performance of a Jacobi variant, though it lags in l2 norm minimization." — [Mixed MPM PDF, §6.1, Fig. 16](https://research.nvidia.com/labs/prl/mixed_mpm/files/mixed_mpm.pdf)
- **Paper, hardware and timings.** Tests ran on an Intel i9-10980XE with an NVIDIA RTX PRO 6000 Blackwell Max-Q. Selected rows of Table 4 (n_sigma = strain nodes, n_GS = GS iterations, n_s = substeps per frame, t_f = wall seconds per frame) — [Mixed MPM PDF, §6, Table 4](https://research.nvidia.com/labs/prl/mixed_mpm/files/mixed_mpm.pdf):

  | Scene | n_sigma | n_GS | n_s | t_f |
  |---|---|---|---|---|
  | Rigid Ball | 25k | 239 | 4 | 0.5 |
  | Dam Break | 99k | 115 | 2 | 0.1 |
  | Viscous | 44k | 215 | 2 | 0.2 |
  | Press | 276k | 247 | 4 | 0.8 |
  | Avalanche | 1.9M | 149 | 8 | 5.4 |
  | Sand city | 5.1M | 234 | 2 | 4.0 |

- **Paper, discretization cost.** Frame time is 106 ms for Q1-P0, 180 ms for Q1-dP1 and 13.3 s for B2-dP1, "consistent with 8× more Gauss–Seidel colors and 8× larger stencils". B2 with Jacobi ran at 3.1 s per frame. — [Mixed MPM PDF, §6.2](https://research.nvidia.com/labs/prl/mixed_mpm/files/mixed_mpm.pdf)
- **Paper, future work.** The author calls slow asymptotic convergence of colored GS / Jacobi a limitation. Directions named: progressive force blending, sampling-accelerated convergence (Lan et al. 2025), multigrid (Wang et al. 2020a), AVBD-style primal-dual. "Low-level optimization of GPU kernels" is listed as open. — [Mixed MPM PDF, §7–8](https://research.nvidia.com/labs/prl/mixed_mpm/files/mixed_mpm.pdf)
- **Newton code, coloring.** `_compute_coloring` (`solver_implicit_mpm.py:3788`) colors by `(i mod s, j mod s, k mod s)` with `s = round(cbrt(velocity_nodes_per_element))`: s = 2 gives 8 colors for Q1 and s = 3 gives 27. Coloring is per cell for pic/DG strain and per node for Q1 strain. A `radix_sort_pairs` on the color key follows, so inside a color the blocks keep their original index order (`implicit_mpm_solver_kernels.py:1154-1215`). A color "block" is the range of strain nodes of one cell, solved sequentially by one thread (`fill_uniform_color_block_indices`, `fill_dynamic_color_block_indices`). — local source
- **Newton code, launch shape.** `_GaussSeidelSolver` (`solve_rheology.py:681-775`) launches with `color_block_count = device.sm_count * 2` on CUDA (1 otherwise) and `block_dim = 64`. The kernel grid-strides over color blocks (`for color_offset in range(color_beg, color_end, launch_dim)`, `rheology_solver_kernels.py:809`). `solve()` issues one launch per color (`set_param_at_index(0, color)`). The kernels are compiled with `fast_math: True` (`rheology_solver_kernels.py` `make_gs_solve_kernel`). — local source
- **Newton code, SoA variant.** `_ReorderedGaussSeidelSolver` ("gauss-seidel-soa", `solve_rheology.py:777-984`) reorders the BSR strain matrix into "a flat, entry-major SoA layout" in flat color order, with zero-padded columns and values. Its docstring says the kernel "statically unrolls the velocity gather loop for coalesced memory access, giving significant speedups on higher-order bases at the cost of increased memory usage". The per-node math is the same, so the update is the same algorithm. Whether it is bitwise-equal depends on the compiler, since zero-padded entries add `+0*x` terms. — local source
- **Newton code, batched variant.** `_BatchedGaussSeidelSolver` ("gauss-seidel-batched", `solve_rheology.py:987-1313`) merges colors into batches (default 9 batches when there are 27 colors, otherwise 16, `_resolve_batched_gs_batching`). Inside a batch it uses "a mass-split Delassus diagonal and atomic velocity scatter", with GS ordering between batches. This changes the algorithm and is non-deterministic because of atomics. — local source
- **Newton code, loop control.** With `use_graph`, the loop is `wp.capture_while(condition, ...)`, a CUDA conditional graph. Each body runs `solve_granularity = 5` × (contact solve, then all colors), then `eval_residual()` (an `evaluate_strain_residual` kernel plus a tiled sum and max reduction) and a 1-thread `update_condition` kernel that sets the condition flag (`solve_rheology.py:1786-1848`). Without graphs, it checks every `solve_granularity = 25` iterations with a host read-back (`solve_rheology.py:1866-1885`). The stopping iteration therefore depends on the check period: it is a multiple of 5 in graph mode and of 25 in host mode. — local source
- **Newton code, residual.** `evaluate_strain_residual` (`rheology_solver_kernels.py:1477`) computes, per node, `|cw_mul(world_to_local(delta_stress[i], R_i), D_i)|^2`. `delta_stress` is the GS kernel's own output from the latest sweep, so the residual is a pure function of per-node data that the color kernel already has in registers. — local source
- **Newton code, local solve.** The local sliding root-finders loop at most 24 Newton steps and break when `delta < _SLIDING_NEWTON_TOL * alpha_max` (`rheology_solver_kernels.py:396-408, 491-500`). Early exit is already present. — local source
- **Published CUDA per-iteration timings.** Newton's code and the paper give none. The paper gives only per-frame totals (Table 4). An earlier note in this repo also found no published Newton ImplicitMPM per-iteration timings (`/Users/aditya/robosim/docs/research/mpm_2026-10-09_notes/performance.md:440`). — local notes

### Inferences
- Upper bounds per GS iteration on the RTX PRO 6000 Max-Q, computed as `t_f / (n_s × n_GS)`; these also include P2G, G2P, assembly and contact:
  - Rigid Ball (25k strain nodes): ≤ 0.52 ms
  - Dam Break (99k): ≤ 0.43 ms
  - Viscous (44k): ≤ 0.47 ms
  - Press (276k): ≤ 0.81 ms
  - Avalanche (1.9M): ≤ 4.5 ms
  - Sand city (5.1M): ≤ 8.5 ms

  This assumes n_GS is the average per substep, which Table 4's caption suggests but does not state. Against those bounds, your ~190 us per iteration (8×20 + 25 + residual/5) for a problem of tens of thousands of strain nodes is not obviously behind NVIDIA's own numbers. The small sizes are latency- and launch-bound on both platforms.
- With a few thousand strain nodes per color, one color is a few thousand threads, against a capacity of up to 40 cores × 384-3072 threads per core ([metal-benchmarks](https://github.com/philipturner/metal-benchmarks)). Each color dispatch is therefore occupancy-starved and latency-bound: ~20 us is roughly the per-node critical path (gather 8 velocity nodes, compliance rows, rotation, local solve, scatter) plus dispatch and tail. This matches the paper's "too small to saturate our GPU for a single Gauss–Seidel color". The levers are (a) a shorter per-thread dependency chain and better memory locality, (b) fewer dispatch boundaries, and (c) more independent work per dispatch, which changes the algorithm unless the extra work is something independent that already exists, such as running the residual alongside contact.
- **Bitwise-safe freedom within a color:** nodes (blocks) of one color are independent. Any permutation of blocks inside a color, and any mapping of blocks to threads, threadgroups or SIMD groups, gives bitwise-identical results, provided each block's own arithmetic is unchanged (same instruction sequence and FMA contraction). Two constraints remain. The color order must stay the same, and the nodes inside a block (one cell's strain nodes) must stay sequential and in order.

### Gaps
- No published CUDA per-color or per-iteration kernel times for Newton's GS, and no published comparison of `gauss-seidel` vs `gauss-seidel-soa` vs `gauss-seidel-batched` timings. The SoA docstring's "significant speedups on higher-order bases" is unquantified.
- I could not confirm whether Table 4's n_GS is per substep or per frame.

## Literature and practice: GPU colored GS and nonlinear GS, plus GPU performance techniques (fused colors, persistent kernels, reordering, layout, precision, fewer colors, overlap)

### Takeaway
The graphics literature (Vivace, Chebyshev/PD course notes, Mixed MPM) treats multi-color GS as one launch per color and gains speed by changing the algorithm: randomized or fewer colors, Jacobi plus Chebyshev, mass-splitting, atomics, multigrid. I found no published method that fuses the colors of a colored GS into one launch while keeping results bitwise-equal, apart from generic temporal-blocking / overlapped-tiling ideas from stencil codes, which do not address colored GS. The levers that keep results equal are layout and locality, removing dispatches that are not color boundaries (residual, condition, contact), and per-thread latency work.

### Cited Findings
- **Vivace (Fratarcangeli, Tibaldo, Pellacini, TOG 35(6) 2016).** A "parallel randomized Gauss-Seidel" for PBD and Projective Dynamics. Each frame the equations are partitioned by a randomized graph coloring, and each partition (no shared unknowns) is solved in parallel on the GPU. Reported faster and better scaling than the state of the art at small time budgets. — [Chalmers record](https://research.chalmers.se/en/publication/507089)
- **Fratarcangeli 2015, GPU PBD.** Uses greedy graph coloring of constraints, with each partition solved in parallel on the GPU. — [Chalmers](https://research.chalmers.se/en/publication/219708)
- **Fratarcangeli & Pellacini, CGF 34 (2015), "Scalable Partitioning for Parallel PBD".** Reduces partition count so that "the performance hit caused by the GPU kernel calls is significantly decreased". This is evidence that launch-per-color overhead was the bottleneck in GPU colored GS, and that their remedy (fewer partitions) changes the ordering. — [Chalmers](https://research.chalmers.se/en/publication/217734)
- **Wang 2015, Chebyshev semi-iterative acceleration of PD/PBD (TOG 34(6)).** Chebyshev speeds convergence "by at least one order of magnitude" with direct, Jacobi or GS global steps. On the GPU, "Chebyshev + Jacobi outperforms any other combination". — [SIGGRAPH Asia 2015 listing](https://sa2015.siggraph.org/en/attendees/technical-papers/event/techpapers/283.html), [SIGGRAPH history](https://history.siggraph.org/?p=210344)
- **SIGGRAPH 2018 course (Fratarcangeli, Wang, Yang), "Parallel iterative solvers for real-time elastic deformations".** Covers multi-color GS, Chebyshev, and Jacobi+Chebyshev descent on GPU. — [Chalmers](https://research.chalmers.se/en/publication/508809)
- **Mixed MPM's own comparison.** "GS, Ours" vs "Jacobi, Ours" vs "GS, FB-SOC" vs "GS, Isotropic", plotted as residual against iteration count and wall time (Fig. 16). The full-W_i Fischer-Burmeister local solver's per-iteration cost "is too high to make up for the convergence gains". — [Mixed MPM PDF, Fig. 16, §6.1](https://research.nvidia.com/labs/prl/mixed_mpm/files/mixed_mpm.pdf)
- **Temporal blocking / overlapped tiling (stencils).** Overlapped tiling widens tile halos in proportion to the number of fused steps and the stencil radius, so several steps run per global-memory round trip. Halo exchange between thread blocks in device-tiled schemes uses a CUDA grid-level barrier. PERKS (a persistent-kernel model for iterative memory-bound GPU codes) keeps the working set on chip across iterations. The 2305.07390 paper also proposes EBISU, low occupancy with large tiles, with up to 2.53x over prior work. — [Revisiting Temporal Blocking (arXiv 2305.07390)](https://arxiv.org/pdf/2305.07390), [AN5D (arXiv 2001.01473)](https://arxiv.org/pdf/2001.01473), [PERKS (arXiv 2204.02064)](https://arxiv.org/pdf/2204.02064)
- **Search gap.** Searches found no red-black / multi-color GS-specific temporal blocking result. The available sources cover Jacobi-type stencils, where update order is free. — (search result; see Gaps)
- **Fei et al. 2021.** "Revisiting integration in the material point method" (TOG 40(4)) is about integration and particle position correction, not the solver. The same group's GPU MPM work is "Principles towards Real-Time Simulation of MPM on Modern GPUs" (arXiv 2111.00699), reporting 1.7×–8.6× single-GPU speedups for explicit MPM. Neither addresses colored GS. — [SIGGRAPH history](https://history.siggraph.org/?p=121224), [arXiv 2111.00699](https://arxiv.org/pdf/2111.00699)
- **Apple GPU per-core resources and caches** (relevant to layout):
  - ~208 KB register file and ~60 KB threadgroup memory per core
  - 8 KB L1 data cache and 128 B global cache line
  - L2 of 512K-1M on Max-class chips (M1/M2 figures)
  - Large SLC (48 MB on M1 Max)
  - "ALU utilization maxes out at 24 simds/core"
  - "The Apple GPU architecture only supports 32-bit atomics on pointer values"

  — [philipturner/metal-benchmarks](https://github.com/philipturner/metal-benchmarks)
- **Concurrent dispatch on Apple GPUs.** "The M1 Max has 32 GPU cores, but can perform up to 96 compute commands simultaneously", so several independent small dispatches can overlap on the GPU. — [metal-benchmarks](https://github.com/philipturner/metal-benchmarks)
- **Metal dispatch types.** `MTLDispatchTypeConcurrent` runs encoded dispatches in a pass concurrently. The developer must synchronize memory between dependent dispatches ("you're responsible for synchronizing the memory operations"), using `memoryBarrierWithScope:` / `memoryBarrier(resources:)`. — [Apple docs: MTLDispatchType.concurrent](https://developer.apple.com/documentation/metal/mtldispatchtype/concurrent.md), [mlx-native CommandEncoder docs](https://docs.rs/mlx-native/0.9.5/mlx_native/struct.CommandEncoder.html)
- **Launch-count reduction measured locally.** In this repo on this machine, fusing 5 of 9 launches per solver iteration gave +3.9% env-steps/s for the MuJoCo Warp fork. — local note `/Users/aditya/robosim/docs/research/mpm_2026-10-09_notes/performance.md:216` (citing `docs/research/throughput_2026-09-26.md` §2c)

### Inferences

Bitwise-preserving levers, ordered by expected value and confidence:

1. **Fold the residual into the color kernels.**
   - The residual is a pure per-node function of `delta_stress`, `R_i` and `D_i`, which the GS kernel already holds (see the `evaluate_strain_residual` finding above).
   - On residual iterations, use a specialized kernel variant (function constant): each thread writes `residual[i]` as it finishes its node, and each threadgroup can also write a partial sum and max.
   - That removes the `evaluate_strain_residual` dispatch. A "last-threadgroup-done" reduction (an atomic counter incremented after a device fence; the last threadgroup sums the partials and sets the stop flag) also removes the reduction and `update_condition` dispatches. This pattern does not spin, so it needs no forward-progress guarantee.
   - The node updates are untouched, so results are bitwise-equal. Only the residual's float summation order can differ, and that matters only if the stop decision flips at the threshold. Newton's own tiled sum order differs between CUDA and Metal anyway.
   - Caveat: the per-node residual expression must compile identically, with the same FMA contraction under `fast_math`, or residuals can differ in the last bits.
2. **Overlap independent work instead of serializing it.**
   - The residual reduction for iteration k reads only `delta_stress`, which the next contact solve does not write. In a concurrent-dispatch encoder, the residual/reduction and the next iteration's contact solve can share a barrier interval.
   - The catch is the stop decision. If the GPU stops at k, the speculative contact solve must not have changed state. It needs double-buffered `r` and `u` (write to scratch, commit on continue), which costs more than it saves unless contact is long.
   - Simpler: the contact solve's per-node prologue (collider velocity, w_i) and the stress kernel's static per-node data do not depend on the iterate and could be precomputed once per step.
3. **Reorder within colors for locality and SIMD coherence (bitwise-safe).**
   - Order blocks within each color in Morton/Z order, so the 32 lanes of a SIMD group gather velocity nodes from neighbouring cache lines. Newton's radix sort keeps index order, which is already row-major and fairly local; Morton order is better in 3D.
   - Pack all static per-node data (B 3-vectors, compliance rows, D_i, R_i, yield params, volume) into one contiguous per-color stream in solve order, like Newton's SoA "reordered" layout. Each thread then reads its record with unit stride across the SIMD group.
   - Optionally group nodes by expected local-solve path (isotropic, root-finding, inactive or empty), so lanes in a SIMD group take the same branch. This changes no arithmetic, only which nodes share a SIMD group. The paper's root-finding count (mean 6, max 19) and the fact that anisotropy appears mostly at material boundaries suggest a mixed SIMD group runs at its slowest lane's path.
4. **Shorten the per-thread critical path.**
   - Each color is latency-bound with low occupancy. Prefetch all 8 velocity gathers and the static record before the 6x6 math (independent loads issued back to back).
   - Keep the 5x5 rotation in registers.
   - Avoid dynamic loops over BSR offsets: statically unroll the 8 velocity entries, as Newton's SoA kernel does.
   - These are instruction-scheduling changes. They stay bitwise-equal only if the floating-point operation order and contraction are preserved, so verify against the reference with exact compares.
5. **Fewer dispatch boundaries per iteration (needs measurement).**
   - Encode all 8 colors and the contact solve of one iteration, or a whole 5-iteration body, in one compute encoder, with barriers (serial dispatch, or concurrent plus `memoryBarrier`) rather than separate encoders.
   - Whether an intra-encoder barrier costs much less than your measured 3-4.5 us dispatch latency on M4 Max is unknown; I found no published number. This is the first thing to microbenchmark.
6. **Persistent multi-threadgroup fused solve with a device-wide barrier.** This is the only route to fewer than 9 barriers' worth of launch latency per iteration, and it carries forward-progress risk (next section). It is bitwise-equal if each node's arithmetic and the color order are preserved.
7. **Overlapped tiling across colors (bitwise-equal, complex).**
   - A threadgroup could own a spatial tile plus a halo and run several colors in threadgroup memory, recomputing halo nodes redundantly. This mirrors temporal-blocking overlapped tiling.
   - For colored GS, the halo must cover every node whose value at color c depends on earlier colors in the same sweep. With a 2x2x2 parity coloring and an 8-velocity-node stencil, the dependency cone grows by about one cell per color, so the halo grows with the colors fused.
   - The redundant halo work dominates for small tiles. With only a few thousand nodes per color in total, this probably costs more than it saves; I found no published GS implementation of it.

Non-bitwise levers (they change the algorithm or ordering):
- Fewer colors or batching (Newton's `gauss-seidel-batched`, Fratarcangeli's partition reduction)
- Atomic-based (uncolored) GS, which the paper mentions as unexplored
- Jacobi plus Chebyshev (Wang 2015), which wins on GPU in PD/PBD
- Randomized coloring (Vivace)
- Block-Jacobi across threadgroups with GS inside a threadgroup tile (domain decomposition)
- Mixed precision (fp16 storage of B or compliance)
- Multigrid

All of these change the iterates. Some may cut wall time to a given residual, but none gives bitwise parity with Newton.

### Gaps
- No published Apple-GPU (or CUDA) measurement of colored GS per-color dispatch time vs fused alternatives.
- No measured cost of `memoryBarrier` within a concurrent compute encoder on M-series GPUs, compared with separate dispatches or encoders.
- Could not retrieve the full text of Vivace, Wang 2015 or Li et al. 2022 ("Energetically consistent inelasticity") within the tool budget. Their GPU solver details and numbers beyond the abstracts are unverified. I found no source tying Li et al. 2022 to colored GS performance.
- No published red-black / multicolor GS temporal-blocking scheme for GPUs was found. The halo analysis above is my own reasoning.

## Device-wide barriers on Apple GPUs: feasibility and risks

### Takeaway
Metal documents no forward-progress, co-residency or fair-scheduling guarantee between threadgroups. An Apple engineer explicitly called inter-threadgroup spin-waiting "undefined behavior". An atomics-based grid barrier over an occupancy-limited persistent grid may work in practice but can hang the GPU. If tried, it must be guarded: a bounded spin with a fallback, and a grid no larger than what is provably resident. Your single-threadgroup result shows the work needs most of the 40 cores, which is exactly the regime where a grid barrier is riskiest.

### Cited Findings
- **Apple engineer on spin-waiting.** In Apple Developer Forums thread 672532, a Graphics and Games engineer said a kernel that spin-waited on a flag set by the previous threadgroup relied on undefined behavior. Metal does not guarantee that threadgroups start in grid order. It does not guarantee that concurrently running threadgroups are "fairly schedule[d] so that both make forward progress". `volatile` memory is not guaranteed atomic. The engineer recommended a conventional multi-pass scan instead. The kernel worked on some GPUs and hung on others (IOAF code 5 on DTK and M1). — [Apple Developer Forums 672532](https://developer.apple.com/forums/thread/672532)
- **Open forum question, June 2026.** A user asks whether Metal gives forward-progress guarantees for persistent-thread sync. They report that device-scope atomics allow global counters and persistent-thread coordination, but they find no documented guarantee on threadgroup co-residency, global forward progress, or "occupancy-bounded synchronization safety". In their experiments, "synchronization schemes that rely on all threadgroups making progress eventually can become unreliable, while strictly local producer/consumer handoff patterns appear much more robust". There were 0 replies when fetched (Oct 2026). — [Apple Developer Forums 831017](https://developer.apple.com/forums/thread/831017)
- **Progress models in other APIs.** "Vulkan and Metal say almost nothing about relative forward progress guarantees between workgroups." — [Sorensen et al., "Specifying and Testing GPU Workgroup Progress Models", OOPSLA 2021](https://multicore.doc.ic.ac.uk/publications/gpu-oopsla-21.html)
- **Sorensen's thesis.** A general global barrier can deadlock through starvation. A safe barrier can be built over an "occupancy-bound" subset of workgroups discovered at runtime, relying on per-GPU scheduling behaviour. The guarantees were studied on other vendors' GPUs, not Apple's. — [Sorensen PhD thesis, Device-wide Barrier Synchronisation on GPUs](https://users.soe.ucsc.edu/~tsorensen/publication/phdthesis/)
- **Commentary on Metal 3.2.** Efficient GPU work queues need forward progress plus global sync. Chaining small kernels gives the same sync on older Metal "but the performance would be very bad". This is a community comment, not Apple documentation. — [techboards Metal 3.2 thread](https://techboards.net/threads/metal-3-2-new-features.4732)
- **Apple GPU atomics.** 32-bit atomics only on pointer values, plus a non-returning 64-bit min/max. — [metal-benchmarks](https://github.com/philipturner/metal-benchmarks)

### Inferences
- **Why it can deadlock.** An atomic counter barrier (arrive with `atomic_fetch_add`, spin on a generation counter, with device fences) is correct only if every participating threadgroup is resident at the same time and keeps being scheduled. On macOS the GPU is shared with WindowServer and other clients, and nothing in Metal promises that 40 or more threadgroups are co-resident or fairly scheduled. A non-resident threadgroup would deadlock the barrier until the GPU watchdog kills the command buffer.
- **Mitigations, if experimenting.**
  1. Launch at most one threadgroup per core (40) or fewer, with modest register and threadgroup-memory use so residency is likely.
  2. Use an occupancy-discovery prologue (Sorensen-style): threadgroups that arrive within a short window get a "participant" ticket, and late ones exit. Participants then grid-stride over all nodes, so correctness does not depend on how many joined.
  3. Bound every spin with an iteration cap that sets an error flag, and fall back to the per-color dispatch path for the step. Validate bitwise against the dispatch path.
  4. Make all flags real `atomic<uint>` with `memory_order_relaxed` plus `threadgroup_barrier(mem_flags::mem_device)` / `atomic_thread_fence` as MSL allows. Do not use `volatile`.
- **Payoff.** Per iteration this saves about (9 to 10 dispatches × ~3-4.5 us) minus (9 to 10 barriers × barrier cost). Grid barrier cost on Apple is unmeasured in any source found. If it is near 1-2 us, the saving is roughly 15-30 us per ~190 us iteration (about 10-15%). Results stay bitwise-equal because the arithmetic per node and the color order are unchanged.
- **Lower-risk alternative.** Use producer/consumer "local handoff" between spatial tiles instead of a global barrier: a tile's color c may start once its neighbouring tiles have finished color c-1. This is a dataflow (wavefront) schedule. The forum poster found local handoffs more robust, but they still assume the producer is eventually scheduled, so the same residency caveat applies.

### Gaps
- No Apple documentation or measured result for a grid-wide barrier's latency on M-series GPUs.
- Apple has not answered the 2026 forum thread.
- I found no source on whether Metal 4's new command encoders change the cost of intra-encoder barriers or dispatches.

## Convergence-side wins that keep results equal, and non-bitwise options

### Takeaway
Bitwise results require the same sequence of sweeps and the same stopping iteration. Warm-starting is already used. Checking the residual less often, or with a different cadence than Newton, changes the stopping iteration and therefore the result. Only the cost of each check can be cut (fusing it, as above), and work after convergence can be skipped. Anything that reduces iteration count (early exit at a different granularity, tolerance changes, acceleration) is a non-bitwise option.

### Cited Findings
- **Newton's stop cadence.** In graph mode it evaluates the residual after every block of 5 iterations and stops when `res_l2 < tol^2 * scale^2` and `res_linf < tol` or `cur_it > max_iterations` (`update_condition`, `solve_rheology.py:240-258`). In host mode the period is 25 (`solve_granularity`). — local source
- **Paper's termination rule.** l2 and l-infinity norms of the plastic strain delta, typically tolerance 10^-3 and 250 max iterations. — [Mixed MPM PDF, §6](https://research.nvidia.com/labs/prl/mixed_mpm/files/mixed_mpm.pdf)
- **Warm start.** Stress and impulses are warm-started, and warm-starting via G2P/P2G works better than grid-to-grid. — [Mixed MPM PDF, Alg. 1, §6](https://research.nvidia.com/labs/prl/mixed_mpm/files/mixed_mpm.pdf)
- **Iteration counts in practice.** Average n_GS in Table 4 ranges from 92 to 247. Several scenes sit near the 250 cap (Rigid Ball 239, Press 247, Sand city 234), so many steps run to max iterations. — [Mixed MPM PDF, Table 4](https://research.nvidia.com/labs/prl/mixed_mpm/files/mixed_mpm.pdf)

### Inferences

Bitwise-equal options:
- **Keep the check period at 5** (to match Newton's graph path), but make each check nearly free by fusing it into the color kernels, as described in the literature section.
- **Skip checks that cannot succeed.** If a step will run to the 250-iteration cap anyway, you cannot know that in advance. But it is exact to skip the condition evaluation on iterations that are not multiples of 5, which you already do.
- **After the stop flag is set on the GPU,** make the remaining pre-encoded dispatches of the block early-return through a device flag read at kernel entry. That costs only dispatch latency, not work. If you encode a fixed 250-iteration schedule, this is how you get "early exit" bitwise-equal to Newton's `capture_while`.
- **Precompute iteration-invariant per-node quantities once per step:** the Delassus diagonal and rotation (already done in Newton's `compute_diagonal_factorization`), packed static records, and contact-node constants.

Non-bitwise options (each changes the iterates or the stopping point):
- A different residual cadence (checking every 10 or 25) changes when the solve stops
- Adaptive tolerance
- Chebyshev or Anderson acceleration over sweeps
- Over-relaxation (SOR)
- Symmetric GS (forward then backward color order)
- Fewer colors or batched GS
- Jacobi-with-mass-splitting for stress
- Atomic-based uncolored GS
- Multigrid
- Progressive force blending, sampling acceleration or AVBD-style primal-dual (paper's future work)
- Isotropic local solver (paper: underperforms overall)
- fp16 storage

### Gaps
- No source quantifies how often Newton's 5-iteration check stops early vs hits the cap on the scenes in this repo; that needs a local histogram.
- No source quantifies how much the l-infinity criterion, rather than l2, drives the iteration count. The paper notes that anisotropic boundary nodes penalize l-infinity convergence (§6.1), which suggests a few boundary nodes set the iteration count.

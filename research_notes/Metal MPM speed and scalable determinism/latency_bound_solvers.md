# Speeding up latency-bound iterative solvers and sparse-grid MPM on GPUs without changing results

Scope: techniques for an implicit MPM in the style of Newton/Warp `SolverImplicitMPM` (sparse grid, P2G/G2P, colored Gauss-Seidel rheology solve, convergence check every 5 iterations, mesh/SDF colliders), targeting Apple Metal. For each technique: reported speedup, the hardware it was measured on, whether it applies to Metal, and whether it keeps results bitwise identical.

Primary-source reading done for this note: Newton source (local checkout `upstream/newton-pr-review3`, commit 5a11beef, 2026-09-26), full text of Gao et al. 2018, Wang et al. 2020, Fei et al. 2021, Vivace 2016 and AsyncTaichi 2020 (PDFs extracted locally). Everything else comes from abstracts or search snippets, and is marked as such where it matters.

## Q1. Fusing colored Gauss-Seidel sweeps (persistent / single-dispatch solvers, block-local coloring, on-chip GS, grid-wide barriers without cooperative launch)

### Takeaway
In Newton, the colored GS is already a persistent-style grid-stride kernel with one launch per color, and CUDA conditional graphs remove the CPU from the loop. On Metal, the cheap bitwise-safe win is to encode every color dispatch and the convergence test into one compute encoder, with GPU-side predication. A true single-dispatch solver with a grid-wide spin barrier would also be bitwise-identical, because nodes within a color write disjoint data. But Metal gives no inter-threadgroup forward-progress guarantee, so it risks deadlock. Block-local or on-chip coloring in threadgroup memory changes the update order, and therefore the results. I found no published measured speedup for fused multi-color GS that transfers directly.

### Cited Findings
**How Newton does it today (source code)**
- Newton's `_GaussSeidelSolver` launches `solve_local` once per color (`for color in range(self.color_count): ... launch()`), and `apply_stress_gs` once per color for the initial guess. — [Newton solve_rheology.py](https://github.com/newton-physics/newton/tree/main/newton/_src/solvers/implicit_mpm) (read locally, lines ~680–775)
- On CUDA, each per-color launch is a fixed "persistent-style" grid: `color_block_count = device.sm_count * 2`, `block_dim = 64`, `max_blocks = color_block_count`. Each thread grid-strides over the color's blocks (`for color_offset in range(color_beg, color_end, launch_dim)`) and walks a contiguous range of strain nodes sequentially. On CPU the count is 1 block. Launches are pre-recorded (`record_cmd=True`) and re-launched with `set_param_at_index(0, color)`. — [Newton solve_rheology.py / rheology_solver_kernels.py](https://github.com/newton-physics/newton/tree/main/newton/_src/solvers/implicit_mpm)
- Coloring is geometric, not graph-based: `color = (x mod s)*s*s + (y mod s)*s + (z mod s)` over cells or nodes, giving at most s³ colors (`node_color` / `cell_color` kernels). Within a color, the velocity scatter is a plain `velocities[u_i] += delta_u` with no atomics, because the coloring makes writes exclusive. — [Newton implicit_mpm_solver_kernels.py](https://github.com/newton-physics/newton/tree/main/newton/_src/solvers/implicit_mpm)
- The iteration loop with graphs: `solve_granularity = 5`. Each batch runs 5 × (contact_solver.solve + rheology_solver.solve), then `eval_residual`, then a 1-thread `update_condition` kernel, all inside `wp.capture_while(condition, ...)`, which is a CUDA conditional graph node. So the host never reads back the residual. Without graphs, granularity comes from the solver (25 for GS) and the residual is read back with `.numpy()` each batch. — [Newton solve_rheology.py](https://github.com/newton-physics/newton/tree/main/newton/_src/solvers/implicit_mpm)
- Newton also ships `_ReorderedGaussSeidelSolver`. It reorders the BSR strain matrix into an "entry-major SoA layout" and "statically unrolls the velocity gather loop for coalesced memory access", and its docstring claims "significant speedups on higher-order bases at the cost of increased memory usage". It gives no number. All setup is done "without host synchronization" (it reads `color_offsets[-1]` on device). — [Newton solve_rheology.py docstring](https://github.com/newton-physics/newton/tree/main/newton/_src/solvers/implicit_mpm)

**GPU colored Gauss-Seidel literature**
- Vivace (Fratarcangeli, Tibaldo, Pellacini, SIGGRAPH Asia 2016): a randomized parallel graph coloring (Grable–Panconesi), re-run every frame, with one parallel step per color plus Chebyshev-style over-relaxation (`q ← ω(q̂ − q_prev) + q_prev`). "The number of parallel steps to perform a full iteration is equal to the number of colors", so it is "desirable to have as small a number of colors as possible, to lower the number of kernel executions." Reported: self-colliding cloth with 324K constraints at 5 ms/frame, noodles with 156K constraints at 4 ms/frame, Armadillo with 55K tets at 15 ms/frame. It converges faster and is more stable than Jacobi and Chebyshev-Jacobi at small time budgets. A greedy 2Δ−1 coloring is too many colors for Δ=35 meshes. — [Vivace paper PDF](https://publications.lib.chalmers.se/records/fulltext/245496/local_245496.pdf)
- NVIDIA PBD patent (position-based FEM): a solver that keeps all DOFs in GPU shared memory is reported as about 50% faster than the main-memory version and ALU-bound. The same patent reports that red-black/multi-color GS "streaming in the 8 or more different element colorings degraded performance by more than 75%" on the GPU, and that Jacobi in PBD converges about 2× slower than GS. This is a patent summary, not a peer-reviewed benchmark. — [US 9990754 patent](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/9990754)
- Graph coloring for GPU PBD (Fratarcangeli & Pellacini 2015): the GPU kernel-call "performance hit... is significantly decreased" by keeping partitions few and balanced. Coloring yields "an imbalanced amount of work for each kernel" because color sizes differ. — [Chalmers PBD GPU paper](https://research.chalmers.se/en/publication/219708); [review, PMC10642556](https://pmc.ncbi.nlm.nih.gov/articles/PMC10642556)
- Parallel Block Neo-Hookean XPBD (MIG 2022): minimal coloring is bounded below by maximum valence. Clustering constraints into supernodes "significantly reduces the number of partitions". This changes the solver's update order, and therefore the results. — [Parallel Block XPBD](https://profs.etsmtl.ca/sandrews/publication/xpbd_mig2022)
- Vertex Block Descent (Chen et al., SIGGRAPH 2024): vertex-level block GS for implicit Euler. The abstract claims a fixed compute budget can be met by capping iterations. I could not retrieve its per-iteration GPU timings. — [VBD (Roblox)](https://about.roblox.com/publications/vertex-block-descent)
- Multi-color GS in GPU sparse linear algebra: a Sandia study reports that coloring quality reduces multi-colored symmetric GS solve time by up to 33% vs cuSPARSE. A GPU graph-coloring paper reports a 4.1× average speedup (up to 8.9×) over serial coloring. Both are for general sparse matrices on NVIDIA GPUs. — [Sandia](https://sandia.gov/ccr/publications/details/parallel-graph-coloring-for-manycore-architectures-2016-07-18); [arXiv 1606.06025](https://arxiv.org/pdf/1606.06025)

**Grid-wide barriers without cooperative launch**
- Xiao & Feng (IPDPS 2010) lock-free inter-block barrier: blocks wait on flags and a master block releases them, avoiding a kernel relaunch. It is reported 3–7× faster than other GPU barriers. It requires persistent threads, meaning only as many blocks as can be co-resident. — [Xiao & Feng TR](https://eprints.cs.vt.edu/archive/00001087/01/TR_GPU_synchronization.pdf)
- Without an occupancy guarantee, such barriers deadlock: blocks spinning at the barrier "never release their resources, so pending blocks never get a chance to execute". Sorensen et al. (OOPSLA 2016) fix this with an occupancy-discovery protocol that dynamically finds a safe number of workgroups. — [Pitt thesis](http://d-scholarship.pitt.edu/34943/1/thesis.pdf); [Sorensen et al. OOPSLA 2016](https://www.doc.ic.ac.uk/~afd/homepages/papers/pdfs/2016/OOPSLA.pdf)
- Vulkan and Metal give essentially no specified inter-workgroup forward-progress guarantee, so the progress model must be chosen and tested empirically. — [Sorensen et al., Specifying and Testing GPU Workgroup Progress Models, arXiv 2109.06132](https://arxiv.org/html/2109.06132v1)
- CUDA cooperative launch (`grid.sync()`) is the sanctioned version. Even it has a cost: cooperative dispatches "must wait until enough resources become available", which can mean "significant scheduling delay". — [Independent Forward Progress of Work-groups (UCR)](https://www.cs.ucr.edu/~mchow009/publications/ductu2020independent.pdf)
- Fei et al. 2021 use exactly this pattern across GPUs: a spin-lock barrier via system atomics over NVLink instead of CPU-mediated sync, to avoid pipeline bubbles. — [Fei et al. arXiv 2111.00699](https://arxiv.org/pdf/2111.00699)

### Inferences
- **Bitwise status of fusing all colors into one dispatch with a grid barrier: preserved.** Newton's per-color work writes disjoint velocity entries with plain `+=`, and each thread's strain-node loop is sequential. So which thread runs which block, and in what wall-clock order, cannot change any value, provided the barrier keeps "all of color c finished before any of color c+1 starts" and the per-node arithmetic is unchanged. This holds only if the Metal kernel is compiled with the same FP contraction settings (FMA fusion) as the per-color kernel, which is worth checking when merging code paths.
- **Metal risk:** with no forward-progress guarantee, a persistent kernel with a spin barrier needs (a) a threadgroup count ≤ co-resident capacity, found empirically per chip (occupancy discovery, à la Sorensen), and (b) tolerance for preemption. A grid of about 2× core count × small threadgroups (Newton's CUDA choice is 2 × SM count × 64 threads) is the natural starting size. This is engineering inference; I found no Apple statement guaranteeing co-residency.
- **Bitwise status of block-local / on-chip GS (a threadgroup sweeps all colors of its tile in threadgroup memory): changed.** Nodes near tile boundaries would see neighbor updates in a different order than the global color schedule. This turns global colored GS into block GS (domain decomposition), which is a different iteration. Supernode/cluster coloring (Parallel Block XPBD) and Vivace's randomized re-coloring also change the update order. The NVIDIA patent's +50% for shared-memory solving is evidence of the upside, but it is not bitwise-compatible with the reference.
- **Cheap, bitwise-safe Metal analogue of Newton's conditional graph:** encode the whole loop (5 × [contact + s³ color dispatches] + residual + condition) into one compute encoder with buffer memory barriers. Make subsequent batches use `dispatchThreadgroups(indirectBuffer:)`, whose argument the condition kernel sets to 0 threadgroups once converged. Unconverged batches run exactly as in the reference, and converged ones become empty dispatches. The iteration count and arithmetic are identical, but the empty dispatches still pay per-dispatch overhead. This is my design inference from the Apple indirect-dispatch docs (see Q3), not a measured result.
- **Keeping the convergence check every 5 iterations is required for bitwise parity.** Checking at a different cadence changes the iteration at which the loop stops, and therefore the final state.

### Gaps
- No published measurement of a single-dispatch, multi-color GS against one-dispatch-per-color on any GPU, Metal included.
- I could not locate PhysX / Macklin descriptions of how their GPU solvers launch colors (one kernel per partition vs persistent). The searches returned only academic analogues.
- VBD's GPU timing tables and Vivace's per-iteration GPU timing and hardware model were not retrievable from the text I extracted.
- I did not check Newton's default stencil size, which gives 8 or 27 colors.

## Q2. GPU MPM performance literature: layouts, sorting, launch strategy, measured gains

### Takeaway
The big measured wins in the MPM literature come from three places: transfer kernels (warp-level reduction plus sorting by cell), kernel fusion (G2P2G, merging per-particle work), and removing CPU-GPU syncs and reallocations. Gains run from 1.7× to more than 10×, all measured on NVIDIA hardware. Almost all of them rely on floating-point atomics or reordered summations, so they are not bitwise-stable relative to a fixed-order reference unless the reference already uses that exact order.

### Cited Findings
**Gao et al. 2018, "GPU Optimization of Material Point Methods" (SIGGRAPH Asia; TITAN Xp and Quadro P6000; CPU baseline 18-core Xeon Gold 6140)** — [paper PDF](https://people.csail.mit.edu/kuiwu/GPU_MPM/GPU_MPM.pdf)
- GSPGrid: a GPU port of SPGrid with 4×4×4 blocks. Particles are sorted to cells by histogram sort rather than radix sort ("observed performance speedup approaches an order of magnitude"). "Delayed ordering" reorders only selected particle attributes, which keeps reordering from being a bottleneck.
- P2G is a scatter with warp-intrinsic segmented reduction (ballot/shfl) over particles sharing a cell, then shared-memory accumulation, then bulk global write. They show scatter beats gather. A naive atomic-only scatter is "an order of magnitude slower".
- P2G/G2P are more than 16× faster than CPU FLIP and about 8× and 13× faster than CPU MLS. The implicit solve is a matrix-free Krylov solver where the matvec is G2P-Implicit followed by P2G-Implicit. For the stress kernel, recomputing the SVD on the fly is faster than storing it. Their fast 3×3 SVD runs at 0.37 ns per SVD on a Titan Xp vs 2.3 ns for AVX512 on the CPU. — [3x3_SVD_CUDA](https://github.com/littlemine/3x3_SVD_CUDA)
- Implicit example: 4.2M particles on a 256³ grid at an average 10.5 s per 48 Hz frame on a Quadro P6000.
- They argue GVDB's texture storage prevents scatter because textures do not support atomics. — same PDF

**Wang et al. 2020, "A Massively Parallel and Scalable Multi-GPU MPM" ("Claymore", SIGGRAPH; Quadro P6000, 1–4 and 8 GPUs)** — [paper PDF](https://yzhu.io/publication/mpmgpu2020siggraph/paper.pdf); [code](https://github.com/penn-graphics-research/claymore)
- G2P2G fuses G2P of step k with P2G of step k+1. It "eliminates two transfer kernel launches and two particle data accesses for each time step" and reduces particle storage. It requires an off-by-two staggered block mapping and a CFL bound, using max grid velocity, so particles stay inside the 2×2×2 shared-memory block window. The paper notes the same fusion could apply to implicit MPM, "where the transfer process can take up to 90% of the wall time".
- The AoSoA particle layout uses bins inside block-buckets. The whole method is 2× faster than Hu et al. 2019 (Taichi) on a single GPU, and more than 100× per step vs a CPU code on 5–40M particles. Weak and strong scaling across 4 GPUs is near-linear.
- An independent measurement on Wang's CUDA code: "AOSOA+G2P2G is 2.1× faster than G2P2G" alone, and Wang et al. is 2.12× faster than AsyncTaichi on MLS-MPM. — [AsyncTaichi arXiv 2012.08141](https://arxiv.org/pdf/2012.08141)

**Fei, Huang, Gao 2021, "Principles towards Real-Time Simulation of MPM on Modern GPUs" (arXiv; V100 ×4 with NVLink)** — [arXiv 2111.00699](https://arxiv.org/pdf/2111.00699)
- Overall: 1.7–8.6× faster on one GPU and 2.5–14.8× on four GPUs vs Wang et al. 2020. 14.8× is for a real-time 55K-particle case and 2.5× for 12M particles offline. Real-time demos: 1.33M particles at 68.5 FPS and 143K particles at 55.9 FPS on 4× V100.
- Principles for small, latency-bound problems: avoid memory reallocation (a GPU vector growing 4× when half full), minimize the number of kernels per step, minimize CPU-GPU syncs (each sync exposes the accumulated launch gap as a "pipeline bubble"), and tune block size against register/shared-memory use.
- Merge kernels: all per-particle work (SVD, plasticity, stress) goes into the P2G phase. Merging also hides the "tail effect" of tiny kernels.
- A "free zone" (3×3×3 neighbour blocks with −½dx shift) reduces rebuild-mapping frequency. At 55.3K particles, rebuild-mapping was the bottleneck and the free zone alone gave 3.7×.
- Avoiding full particle re-sort (an in-warp 10-bit radix sort only) gave 8–17% across resolutions.
- Dropping shared-memory float atomics, which are emulated with CAS loops on SM_70, in favour of warp-reduced native global atomics gave 3–5× on transfers. Spilling B-spline weights to shared memory to raise occupancy is another technique they use.

**Hu et al. Taichi (2019) and AsyncTaichi (2020)**
- Taichi's sparse data structures yield "competitive performance" on stencils and particle scattering, and the thesis claims 4.55× higher performance with 1/10 lines of code on sparse computations and 235M MPM particles on one GPU. I did not see the baseline for 4.55×. — [Taichi 2019](https://yuanming.taichi.graphics/publication/2019-taichi/); [Hu thesis](https://yuanming.taichi.graphics/publication/2021-taichi-thesis/taichi-thesis.pdf)
- AsyncTaichi: an asynchronous engine keeps a window of pending kernels and applies sparse-aware passes (list-generation removal, activation elimination), task fusion and dead-store elimination. Results: 4.02× fewer kernel launches with 1.87× geometric-mean speedup on GPU benchmarks, and 5× fewer launches on MGPCG. — [AsyncTaichi arXiv 2012.08141](https://arxiv.org/pdf/2012.08141)

**2024–2026 work**
- Convex MPM–rigid coupling (arXiv 2503.05046, 2025) sorts once per time step and reuses the ordering across substeps with a 10-bit-key radix sort. It cites kernel launches and host-device sync as the cost of per-substep sorting at interactive (small) sizes. — [arXiv 2503.05046](https://arxiv.org/html/2503.05046v2)
- Portable MPM for compressible flows (arXiv 2404.17057, 2024) skipped kernel fusion. Its preliminary tests suggested fusion "would not have changed the execution time's order of magnitude" at its (large) scale. — [arXiv 2404.17057](https://arxiv.org/pdf/2404.17057)
- GeoWarp (arXiv 2507.09435) is an implicit MPM in NVIDIA Warp using autodiff Jacobians, and ADMM frictional contact for implicit MPM appears in arXiv 2602.02038. Neither reported launch-strategy numbers in what I retrieved. — [GeoWarp](https://arxiv.org/html/2507.09435v2); [arXiv 2602.02038](https://scour.ing/@matmat/p/https://arxiv.org/abs/2602.02038)
- Newton changelog: an opt-in isolated multi-world implicit MPM with "capacity-bounded rebuildable sparse grids" and a `check_sparse_grid_rebuild_status()` status check, which fits the "no reallocation, no host sync" principle. — [Newton changelog (safetycli mirror)](https://data.safetycli.com/packages/pypi/newton/changelog)

### Inferences
- **Bitwise classification of MPM techniques:**
  - *Preserving:* register spilling, occupancy tuning, block-size tuning; preallocation / capacity-bounded grids (no reallocation); removing host syncs; re-launch caching. Merging per-particle elementwise kernels (SVD, plasticity, stress into P2G) is preserving only if the arithmetic expression and FMA contraction stay the same.
  - *Changing summation order:* warp-level segmented reduction vs per-particle atomics; shared-memory pre-accumulation; any float `atomicAdd` P2G (nondeterministic run-to-run); particle sorting or re-sorting frequency changes (the free zone, in-warp-only sorting, sort-once-per-step), which changes which particles land in which warp and so the order of contributions; AoSoA re-binning.
  - *Changing the algorithm:* G2P2G (merges two steps and moves the partition update, and the CFL estimate switches from particle to grid max velocity); Taichi dead-store elimination is semantics-preserving, but fusion may reassociate.
- A deterministic Metal P2G must fix the contribution order: either gather per node over a sorted particle list in fixed order, or use fixed-order segmented reductions. That makes particle sorting part of the bitwise contract.
- The Newton GS rheology solve has no atomics (exclusive writes by coloring). So for the GS phase, only launch-structure techniques matter for bitwise parity. P2G, G2P and collider rasterization are where order-sensitive choices live.

### Gaps
- No Apple Silicon / Metal GPU MPM performance papers found.
- Per-kernel timing breakdowns for Newton's implicit MPM (fraction in GS vs P2G vs collider) were not found in any public source. They should be measured.
- I did not find Daviet's own implicit-MPM papers (the formulation behind Newton's solver) with performance numbers. Search did not surface them.
- Gao 2018 GPU implicit (Krylov) per-iteration timings exist in its figures but were not legible in the extracted text.

## Q3. Reducing dispatch count: merging kernels, indirect dispatch, mega-kernels, graph-level optimization; numbers on Metal

### Takeaway
Measured per-dispatch overhead on Apple GPUs via WebGPU is tens of microseconds: 32–71 µs on Metal vs 24–36 µs on Vulkan, and fusion helped Metal about 2× in that study. There is no Apple-published native-Metal per-dispatch number. The standard tools are indirect dispatch and indirect command buffers (GPU-sized work), one long-lived compute encoder with targeted memory barriers, and compiler-level fusion (AsyncTaichi: about 4× fewer launches, 1.87× speedup). Pure launch-restructuring keeps results bitwise; fusion that reassociates arithmetic does not.

### Cited Findings
- WebGPU dispatch overhead study (2026; four vendors including Apple): WebGPU API overhead is 32–71 µs per dispatch on Metal vs 24–36 µs on Vulkan, and about 95 µs per op including Python. Implementation choice within Metal changes overhead by 2.2×. Naive single-op benchmarks overestimate dispatch cost about 20×. A search summary of the paper attributes a 2× Metal speedup from fusing dispatches (vs 1.17× on Vulkan) to the 71 µs Metal overhead. The abstract itself only states a 53% fusion throughput gain on Vulkan, and "CUDA fusion provides no benefit". The M-series chip model was not named in the abstract. — [arXiv 2604.02344](https://arxiv.org/abs/2604.02344)
- Metal indirect dispatch: `dispatchThreadgroups(indirectBuffer:threadsPerThreadgroup:)` reads threadgroup counts from an `MTLBuffer` so a shader can compute them "in the GPU timeline". The argument must match `MTLDispatchThreadgroupsIndirectArguments` and be 4-byte aligned, and Metal 4 requires residency via `MTLResidencySet`. — [Apple docs: MTL4ComputeCommandEncoder dispatchThreadgroups(indirectBuffer:)](https://developer.apple.com/documentation/metal/mtl4computecommandencoder/dispatchthreadgroups(indirectbuffer:threadsperthreadgroup:).md)
- Metal indirect command buffers can be encoded on the CPU or the GPU. `executeCommands(buffer:indirectBuffer:)` lets the GPU choose the executed range. You cannot use a barrier to wait for all commands within an ICB to complete. — [Apple: Indirect command encoding](https://developer.apple.com/documentation/metal/indirect-command-encoding.md); [executeCommands(buffer:indirectBuffer:)](https://developer.apple.com/documentation/metal/mtl4computecommandencoder/executecommands(buffer:indirectbuffer:).md)
- Apple's Metal Best Practices guide says indirect buffers "eliminate unnecessary data transfers between the CPU and the GPU, which results in reduced processor idle time". It gives no numbers. — [Metal Best Practices: Indirect Buffers](https://developer.apple.com/library/archive/documentation/3DDrawing/Conceptual/MTLBestPracticesGuide/IndirectBuffers.html)
- Practitioner pattern (mlx-native crate docs): keep one compute encoder open across many dispatches to avoid per-dispatch encoder overhead. With `MTLDispatchTypeConcurrent`, dispatches may overlap unless separated by `memoryBarrier(scope: .buffers)`, so place barriers only on true read-after-write edges. No barrier-cost numbers are given. — [mlx-native CommandEncoder docs](https://docs.rs/mlx-native/0.9.5/mlx_native/struct.CommandEncoder.html)
- Apple Silicon LLM work finds dispatch, not compute, dominates small FFT ops on MPS ("single-digit microseconds" each, multiplied by pipeline steps). — [arXiv 2605.05699](https://arxiv.org/pdf/2605.05699)
- A proposal (not a measurement) estimates 0.8–1.5 ms of host encoding per step for about 720 dispatches on Metal, and suggests ICB replay to bring that to under 0.03 ms. — [GitHub issue anthony-chaudhary/fak #12587](https://github.com/anthony-chaudhary/fak/issues/12587)
- Graph-level: AsyncTaichi gives 4.02× fewer launches and 1.87× speedup. Fei et al. list "minimizing the number of CUDA kernels executed within a single time step" and "minimizing synchronization between GPU and CPU" as principles. Newton uses conditional CUDA graphs (`wp.capture_while`) for the GS loop. — [AsyncTaichi](https://arxiv.org/pdf/2012.08141); [Fei et al.](https://arxiv.org/pdf/2111.00699); [Newton source](https://github.com/newton-physics/newton/tree/main/newton/_src/solvers/implicit_mpm)

### Inferences
- **Dispatch budget arithmetic for the Newton GS:** per iteration there is 1 contact dispatch plus up to s³ color dispatches (s=2 gives 8, s=3 gives 27). Every 5 iterations add residual and condition kernels, which are themselves multi-launch reductions (`_tiled_sum_kernel` loops). At 100 iterations with 27 colors that is about 2,800 dispatches. At even 5–10 µs of GPU-side gap per dispatch, that is 14–28 ms of pure overhead. That makes dispatch structure the first-order cost for small and medium scenes on Metal. The per-dispatch figure is an assumption, to be measured with Metal counters.
- **Bitwise-preserving dispatch reductions on Metal, roughly in order of effort:**
  1. One compute encoder per solve with serial dispatch, or concurrent dispatch plus barriers.
  2. Pre-encoded or GPU-sized indirect dispatches, or an ICB replayed each step.
  3. GPU-side convergence predication via zero-sized indirect dispatches.
  4. Merging the apply-stress and solve kernels of the same color only where Newton already does it in one kernel.
  5. A persistent single-dispatch kernel with a spin barrier, carrying the forward-progress risk from Q1.

  None of these changes per-node arithmetic or the color ordering.
- **Non-bitwise options:** fusing reductions differently, such as a single-pass residual norm instead of the multi-pass tiled sum, reassociates the sum. That changes the residual in the last bits and could flip a convergence decision at the threshold, changing the iteration count. A bitwise-safe residual must reproduce the tiled reduction tree exactly.

### Gaps
- No Apple-published or WWDC-sourced measurement of native Metal compute dispatch latency (GPU-side gap between back-to-back dependent dispatches in one encoder) was found. The WebGPU numbers include API layers above Metal.
- No measured cost for `memoryBarrier` between dispatches on Apple GPUs.
- Whether Apple GPUs keep all threadgroups of a small grid co-resident (needed for a spin barrier) is undocumented.

## Q4. Which techniques keep floating-point evaluation order unchanged (bitwise) vs change summation order

### Takeaway
Techniques that only change when or how work is launched keep results bitwise, as long as the per-element arithmetic and the color/iteration order stay fixed. Techniques that change who sums what, or in what order, change results. The second group covers atomics, warp/shared reductions, re-sorting, block-local or re-colored GS, cluster coloring and different residual reduction trees.

### Cited Findings
- The Newton GS writes velocities with plain `+=` under exclusive coloring, with no atomics. Each thread processes a contiguous strain-node range sequentially. — [Newton rheology_solver_kernels.py](https://github.com/newton-physics/newton/tree/main/newton/_src/solvers/implicit_mpm)
- Gauss-Seidel results depend on update order. Vivace randomizes ordering deliberately because randomization "removes this error faster", which shows that ordering is a numerical choice, not a neutral implementation detail. — [Vivace PDF](https://publications.lib.chalmers.se/records/fulltext/245496/local_245496.pdf)
- Gao 2018 and Fei 2021 P2G use float atomics, with warp-segmented pre-reduction, to global or shared memory. Fei shows shared float atomics are emulated by CAS loops on SM_70. — [Gao 2018](https://people.csail.mit.edu/kuiwu/GPU_MPM/GPU_MPM.pdf); [Fei 2021](https://arxiv.org/pdf/2111.00699)
- Wang 2020 G2P2G changes the dt estimate from particle max velocity to grid max velocity ("less than 1%" difference in dt), so it is not bitwise-equivalent to the unfused pipeline. — [Wang 2020 PDF](https://yzhu.io/publication/mpmgpu2020siggraph/paper.pdf)

### Inferences
**Classification for the target pipeline:**

| Technique | Bitwise vs Newton reference? |
|---|---|
| Single encoder / fewer encoders; ICB; indirect dispatch; pre-recorded launches | Yes |
| GPU-side convergence flag at the same 5-iteration cadence | Yes, if the residual reduction tree is identical |
| Persistent kernel with grid barrier executing colors in the same order | Yes (exclusive writes), but deadlock risk on Metal |
| Merging apply_stress + solve within a color | Yes, if the same expression and FMA contraction |
| Occupancy, register spill, threadgroup size, preallocated capacity | Yes |
| SoA/entry-major reordering of strain matrix (Newton `_ReorderedGaussSeidelSolver`) | Only if the per-node gather/accumulate order is kept; not verified |
| Block-local GS in threadgroup memory / domain-decomposed colors | No (different iteration) |
| Re-coloring (randomized, cluster/supernode, different stencil size) | No |
| Changing convergence-check cadence | No (different stop iteration) |
| P2G with float atomics | No (run-to-run nondeterministic) |
| Warp/shared reductions, sorted vs unsorted particles, sort frequency | No (summation order) |
| G2P2G fusion | No (algorithmic reorder plus dt change) |
| Single-pass vs multi-pass residual norm | No (reassociation) |

- For FMA: Metal's default fast-math (`-ffast-math`, on by default in the Metal compiler) can contract or reassociate differently from Warp/NVCC. Bitwise parity across CUDA→Metal is therefore a separate concern from bitwise parity across Metal launch strategies. This is general knowledge and was not sourced in this pass, so verify it against the Metal compiler docs.

### Gaps
- No source directly benchmarks "deterministic vs atomic P2G" cost on any GPU. The cost of fixed-order gather P2G vs atomic scatter on Apple GPUs is unknown.
- I did not verify whether Newton's reordered-GS SoA path preserves the per-node accumulation order of the baseline GS kernel. That needs a code diff of the two kernels.

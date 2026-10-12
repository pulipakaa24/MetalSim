# Apple M4 Max (Metal) vs NVIDIA L4 (Ada, sm_89, CUDA): microarchitecture for simulation kernels and fair compute-adjusted comparison

Scope note: sources below are a mix of vendor specs (labelled), third-party aggregators (labelled), and measured microbenchmarks. Many measured Apple figures are for M1/M2-generation GPUs (Apple GPU family 7/8); M3/M4 (family 9) changed the on-chip memory design ("Dynamic Caching") and no independent M4 microbenchmark study of caches/registers was found. Treat Apple per-core numbers as "previous generation, likely similar or better on M4".

## Q1. Peak FP32, bandwidth, caches, register file, occupancy, SIMD width, clock

### Takeaway
On paper the L4 has roughly 1.8x the FP32 peak of a 40-core M4 Max (30.3 vs ~16-17 TFLOPS), but the M4 Max has about 1.8x the DRAM bandwidth (546 vs 300 GB/s). That gives it about 3x the bytes per FLOP. The L4 has a much larger L2 (48 MB vs a few MB on Apple), and Apple compensates with a large system-level cache (SLC) behind it. Both use 32-wide SIMD and have similar per-core register files (Ada 256 KB, Apple ~208 KB on family 7/8). Apple family 9 (M3/M4) allocates registers dynamically.

### Cited Findings
**NVIDIA L4 (AD104, sm_89)**
- FP32 30.3 TFLOPS, 24 GB GDDR6, 300 GB/s, 72 W TDP, 1-slot low-profile PCIe (vendor datasheet; INT8/FP8 tensor figures are quoted with sparsity) — [NVIDIA L4 datasheet via Cisco](https://bxin.cisco.com/c/dam/en/us/products/collateral/servers-unified-computing/ucs-c-series-rack-servers/nvidia-l4-gpu.pdf)
- Boost clock 2040 MHz, base 795 MHz, 48 MB L2, 192-bit GDDR6 bus, 128 KB L1 per SM, 7,424 shading units (aggregator). The same page says "60 SMs" in one place, but 7,424 / 128 = 58 SMs, so 58 enabled SMs is the internally consistent figure — [fpsbench L4 vs L40S](https://fpsbench.com/compare/gpu/nvidia-l4-vs-nvidia-l40s)
- Check: 7,424 FP32 lanes x 2 FLOP (FMA) x 2.04 GHz = 30.3 TFLOPS, which matches the datasheet (own arithmetic from the figures above).
- Compute capability 8.9 limits: 1,536 resident threads per SM, 48 warps per SM, 24 blocks per SM, 64K 32-bit registers per SM (= 256 KB), max 255 registers per thread, 128 KB unified L1/shared per SM with shared memory up to 100 KB per SM (99 KB per block), warp size 32 — [CUDA Programming Guide, compute capabilities](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html)
- Ada keeps a large private L1 per SM, as Ampere did — [Chips and Cheese, Microbenchmarking RTX 4090](https://chipsandcheese.com/p/microbenchmarking-nvidias-rtx-4090); 128 KB L1 per SM — [Wikipedia: Ada Lovelace](https://en.wikipedia.org/wiki/Ada_Lovelace_(microarchitecture))

**Apple M4 Max (40-core GPU)**
- Up to 546 GB/s unified memory bandwidth, up to 128 GB, 40 GPU cores, TSMC N3E. The 40-core GPU comes only with the 16-core CPU bin. Apple claims the GPU is up to 1.9x faster than M1 Max (vendor claim) — [MacRumors on Apple's announcement](https://macrumors.com/2024/10/30/apple-announces-m4-max-chip); [Notebookcheck](https://www.notebookcheck.net/Apple-M4-Max-official-with-incremental-upgrades-over-the-M3-Max.911098.0.html)
- 5,120 ALUs (640 EUs). Peak FP32 is 17.2 TFLOPS at 1,680 MHz per EatYourBytes, ~16.2 TFLOPS per NanoReview, and 17.04 TFLOPS with 1,578 MHz per LowEndMac/Wikipedia. These aggregators disagree on clock, and 5,120 x 2 x 1.578 GHz = 16.2 TFLOPS, so the 17.04/1,578 pairing is internally inconsistent — [EatYourBytes](https://www.eatyourbytes.com/applecpu-detail/apple-m4-max-40-core-gpu/); [LowEndMac](https://lowendmac.com/1234/apple-silicon-m4-max-chip-specs/); [NanoReview](https://nanoreview.net/en/gpu-compare/apple-m4-max-gpu-40-core-vs-apple-m3-max-gpu-40-core)
- Base M4 (10-core GPU) measured: ~2.9 FP32 TFLOPS peak achieved, ~100 GB/s STREAM-style (Metal port of the GPU STREAM kernels), close to theoretical — [Hübner et al., "Apple vs. Oranges", arXiv:2502.05317](https://arxiv.org/abs/2502.05317v2)

**Apple GPU per-core microarchitecture (measured on family 7/8 = M1/M2, Philip Turner)** — [philipturner/metal-benchmarks](https://github.com/philipturner/metal-benchmarks)
- Register file ~208 KB per core (Ada 256 KB). Threadgroup memory ~60 KB per core. Instruction cache 12 KB. L1 data cache 8 KB per core (Ada 28-120 KB L1/SM as listed there).
- L2: M1 ~768 KB, M1 Max 512 KB, M2 ~1.5 MB, M2 Max ~1 MB. SLC: M1 8 MB, M1 Pro 24 MB, M1 Max 48 MB, M1 Ultra 96 MB. 128 B cache line.
- Bandwidth per core-cycle: on-core 64 B, on-GPU ~32 B, SLC ~15.4-19.8 B, RAM ~7.7-9.9 B.
- SIMD width 32. 4 schedulers and 128 ALUs per core. Max 384-3,072 threads per core (Ampere/Ada 256-1,536 listed). ALU utilization peaks at ~24 simds per core.
- 256 FP32 FLOP per core-cycle (same as Ampere/Ada per SM). Clocks: M1 Max 1.296 GHz, M2 1.398 GHz.
- FFMA32 latency ~2.2 cycles (adjusted). RECIP32 6.5, RSQRT32 ~9, IMUL32 ~4, IADD64 4 cycles. 64-bit integer multiply is slow (8 throughput cycles).
- Up to 96 concurrent commands on the 32-core M1 Max (NVIDIA up to 128).
- Chips and Cheese (M2 Pro) contradicts Turner on L2 size: they measure a 3 MB L2 on M2 Pro (Turner lists ~512 KB), roughly 3x M1's — [Chips and Cheese, M2 Pro iGPU](https://chipsandcheese.com/p/a-brief-look-at-apples-m2-pro-igpu)
- Chips and Cheese (M2 Pro measured): L1 8 KB, plus a separate 24 KB texture cache. L2 bandwidth >1 TB/s (M1 just under 400 GB/s). SLC latency 234 ns. DRAM latency >342 ns at 128 MB and >400 ns at larger sizes (likely TLB). vkpeak FP32 6.5 TFLOPS. Local (threadgroup) memory latency is slightly higher than L1, slower than recent AMD/NVIDIA — [Chips and Cheese](https://chipsandcheese.com/p/a-brief-look-at-apples-m2-pro-igpu)

**Apple family 9 (M3/M4) "Dynamic Caching" (vendor description)** — [Apple Tech Talk 111375, "Explore GPU advancements in M3 and A17 Pro"](https://developer.apple.com/videos/play/tech-talks/111375/)
- The register file is now a cache. Registers are allocated and freed dynamically over the shader's lifetime instead of reserving the peak at launch.
- Registers, threadgroup, tile, stack and buffer data share "fewer larger" on-chip caches, so unused threadgroup memory no longer wastes storage.
- Hardware adjusts occupancy at runtime to prevent spilling. Apple says developers need to tune occupancy "a lot less often".
- FP32, FP16 and integer pipelines can issue in parallel from different SIMD-groups, for "up to 2x ALU performance" vs prior Apple GPUs (vendor). This needs occupancy to realize.

### Inferences
- Ratios (theoretical): FLOPS L4/M4Max ~1.8-1.9x. DRAM bandwidth M4Max/L4 ~1.82x. Bytes per FLOP: M4 Max ~546/16.2 ≈ 34 B per kFLOP vs L4 ~300/30.3 ≈ 10 B per kFLOP, about 3.4x more bandwidth per FLOP on Apple. Implicit MPM (P2G/G2P scatter-gather, sparse-grid stencils, Jacobi/GS sweeps, BVH traversal) is mostly bandwidth-, latency- or atomic-bound, not FLOP-bound. A FLOPS-normalized comparison would therefore bias in favor of Apple. Normalize by DRAM bandwidth, or report both.
- Ada's FP32 peak counts both datapaths per SM partition (128 FP32 lanes per SM). Real kernels with integer address math share one of those paths, so effective FP32 on index-heavy simulation kernels is often well below 30 TFLOPS. This is a known Ampere/Ada property, not sourced here, so verify with Nsight.
- Working-set fit: L4's 48 MB L2 can hold a medium sparse grid plus particle attributes entirely on chip. Apple has only a few MB of GPU L2, then a shared SLC. M4 Max SLC size was not found (M1 Max: 48 MB). So for working sets of roughly 4-48 MB, L4 may serve from L2 where M4 Max goes to SLC (~234 ns latency on M2 Pro). For working sets over ~48 MB, both go to DRAM, and Apple's 1.8x bandwidth wins.
- The small Apple L1 (8 KB/core vs up to 128 KB/SM) hurts gather-heavy kernels like BVH traversal and G2P neighborhood reads unless data is staged in threadgroup memory. M3/M4's unified on-chip cache may soften this, but it is unmeasured.
- L4 is a 72 W card. Under sustained FP32 load it plausibly runs below its 2,040 MHz boost. This is unverified, so measure clocks with nvidia-smi during runs.

### Gaps
- No independent measurement of M4 Max GPU clock, L2 size, SLC size, or per-core register/threadgroup capacity was found. The Chips and Cheese M4 GPU article (if one exists) did not come up.
- No measured L4-specific cache latency/bandwidth study was found. RTX 4090 (same Ada SM) is the closest proxy.
- The M4 Max 32-core bin bandwidth (reportedly 410 GB/s) was not verified here.

## Q2. Kernel dispatch latency and back-to-back dependent dispatch cost (Metal vs CUDA graphs)

### Takeaway
CUDA has well-documented dispatch costs: a dependent kernel-to-kernel gap of about 1-3 µs on streams, and CUDA graph repeat launch of about 2.5 µs + ~1 ns per node on Ampere+ straight-line graphs, with device-side per-node savings up to 60 ns. Metal has no comparably rigorous published microbenchmark. Anecdotes show much larger costs when each dispatch is a separate command buffer (~120 µs for commit + wait on M2 Max; 32-71 µs per dispatch via wgpu/Metal). Metal per-dispatch cost within one encoder is unpublished and must be measured. A solver with many short dependent dispatches is the most likely place for M4 Max to lose badly.

### Cited Findings
- CUDA graphs on Ampere+: repeat-launch CPU cost for straight-line graphs dropped from ~2 µs + 200 ns/node to ~2.5 µs + ~1 ns/node. Pre-Ampere is still O(n). A graph with 4 root nodes costs 6 µs for a 1-node launch. Device-side inter-kernel latency optimization saves up to 60 ns/node (vendor measurements) — [NVIDIA blog, constant-time launch for straight-line CUDA graphs](https://developer.nvidia.com/blog/constant-time-launch-for-straight-line-cuda-graphs-and-other-performance-enhancements)
- Stream (non-graph) back-to-back dependent kernels: user-measured switch gaps of ~1-3 µs, e.g. 2.88 µs in Nsight Systems (user report; hardware unstated). A reply suggests CUDA graphs and/or Programmatic Dependent Launch can eliminate most of the gap — [NVIDIA forum: successive kernels switch latency](https://forums.developer.nvidia.com/t/kernel-switch-latency-successive-kernels-switch-latency/309504)
- Native CUDA launch overhead measured 7.4 ± 9.2 µs (sequential, hardware likely RTX 5090). Cited typical CUDA launch latency is 3-10 µs, with CUDA Graphs <1 µs per kernel. WebGPU via wgpu-native on Metal (Apple M2) is 71.1 µs per dispatch (48.3 single-op), vs 23.8-35.8 µs for Vulkan on RTX 5090. The paper has no native-Metal baseline — [arXiv:2604.02344, WebGPU dispatch overhead across vendors](https://arxiv.org/html/2604.02344)
- Native Metal anecdote: an empty kernel with commit + `waitUntilCompleted` took 0.12 ms on M2 Max, vs 0.001 ms for CUDA on an RTX 3080. These are not like-for-like, since the Metal figure includes a full CPU-GPU round trip — [Apple Developer Forums 767853](https://developer.apple.com/forums/thread/767853)
- A bitonic sort with hundreds of dispatches was dominated by per-dispatch overhead from Swift. The same kernel ran 25x faster when driven from Objective-C, which points to host-binding overhead rather than GPU cost — [Apple Developer Forums 750363](https://developer.apple.com/forums/thread/750363)
- Fusing an MLP block from 7 dispatches to 3 on M2 (wgpu/Metal) gave a 2.01x speedup (5.74 to 2.85 ms). Fewer dispatches helped more on Metal than on Vulkan — [search summary of arXiv:2608.08730](https://arxiv.org/pdf/2608.08730)
- Metal GPU-driven alternatives: indirect dispatch (`dispatchThreadgroups(indirectBuffer:)`, threadgroup counts computed on GPU) and indirect command buffers (ICBs). ICBs are reusable across submissions, unlike single-use MTLCommandBuffers. Metal 4's `executeCommands(buffer:indirectBuffer:)` reads the executed range from GPU memory, which enables GPU-driven command selection. Caveat: barriers cannot be used to wait for all commands inside an ICB — [Apple docs, MTL4ComputeCommandEncoder executeCommands](https://developer.apple.com/documentation/metal/mtl4computecommandencoder/executecommands(buffer:indirectbuffer:).md); [Apple, Indirect command encoding](https://developer.apple.com/documentation/metal/indirect-command-encoding.md); [Apple Metal Best Practices: indirect buffers](https://developer.apple.com/library/archive/documentation/3DDrawing/Conceptual/MTLBestPracticesGuide/IndirectBuffers.html)

### Inferences
- The honest comparison for a solver with N dependent short dispatches per step is GPU-timeline cost per dependent dispatch. On CUDA that means a captured graph (and/or PDL); on Metal it means a single compute encoder, ideally `MTLDispatchTypeConcurrent` with explicit `memoryBarrier(scope: .buffers)` between dependent dispatches, all in one command buffer, timed with `gpuStartTime`/`gpuEndTime`. Comparing CUDA graphs against Metal one-command-buffer-per-dispatch overstates the gap by around 100x.
- A colored Gauss-Seidel solve with, say, 8 colors x 50 iterations = 400 dependent dispatches per step would cost ~0.4 ms of pure gap on CUDA at ~1 µs/dispatch (stream) and much less with graphs. The Metal equivalent is unknown. If each barrier+dispatch costs even 5-10 µs, that becomes 2-4 ms per step, which can dominate small and medium scenes. This is an illustrative calculation, not a measurement.
- Metal has no equivalent of CUDA conditional graph nodes (device-side while/if) or dynamic parallelism. Convergence-based early exit needs either a CPU readback (a full round trip) or a fixed iteration count with indirect dispatch of zero threadgroups once converged. The latter avoids the readback but still pays a per-dispatch cost for no-op dispatches.

### Gaps
- No rigorous published native-Metal microbenchmark was found for (a) per-dispatch cost within one encoder, (b) the cost of `memoryBarrier` between dependent dispatches, or (c) serial vs concurrent dispatch type, on any M-series chip. Philip Turner's metal-benchmarks does not include dispatch latency. This should be measured in-house.
- No measured CUDA graph node-to-node device latency was found specifically for L4/Ada. The NVIDIA blog figures are for Ampere+ generally.

## Q3. Atomics (float atomic add, contention), threadgroup memory bandwidth, SIMD shuffles

### Takeaway
Apple GPUs support 32-bit atomics, including float add on device memory, which PyTorch MPS uses in production. On family 7/8 they lack general 64-bit atomics (only a non-returning u64 min/max from M2 onward), and half-precision atomics are absent. Measured global atomic latency on M2 Pro is ~59 ns, similar to RDNA2. Local atomics are behind recent discrete GPUs. No contended float-atomic throughput comparison vs Ada was found. Apple SIMD shuffle bandwidth per core is 2x Ada's per SM (Turner).

### Cited Findings
- Apple supports only 32-bit atomics on pointer values. M2 added a single non-returning UInt64 min/max. A 32-bit buffer-atomic workaround costs "2.5x bandwidth / 5x latency" — [philipturner/metal-benchmarks](https://github.com/philipturner/metal-benchmarks); 64-bit atomics workaround notes — [philipturner/ue5-nanite-macos AtomicsWorkaround](https://github.com/philipturner/ue5-nanite-macos/blob/main/AtomicsWorkaround)
- PyTorch's Metal backend uses `atomic_fetch_add_explicit` (relaxed) on `metal::atomic<float>` in device memory. As of Metal 3.2 there are no half-precision atomics, so it uses a 32-bit CAS loop instead — [PyTorch c10/metal/atomic.h mirror](https://generationalwealth.duckdns.org/venv/lib/python3.13/site-packages/torch/include/c10/metal/atomic.h) (non-canonical mirror of the PyTorch source)
- M2 Pro global atomic latency is 58.57 ns, comparable to RDNA 2. Local (threadgroup) atomics beat GTX 980 Ti but trail recent discrete GPUs. Local memory latency is slightly higher than the 8 KB L1 — [Chips and Cheese, M2 Pro iGPU](https://chipsandcheese.com/p/a-brief-look-at-apples-m2-pro-igpu)
- SIMD shuffle bandwidth is 256 B/cycle per Apple core vs 128 B/cycle per Ampere/Ada SM. SHUFFLE_ROTATE32 has ~2 cycles throughput. Random shuffles are far costlier (~28-32 cycles) — [philipturner/metal-benchmarks](https://github.com/philipturner/metal-benchmarks)
- Threadgroup memory ~60 KB per Apple core (family 7/8) vs up to 100 KB shared per SM on sm_89 — [metal-benchmarks](https://github.com/philipturner/metal-benchmarks); [CUDA compute capabilities](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html)

### Inferences
- For MPM P2G, the standard mitigation on both platforms is to sort or bin particles by cell/block, reduce in threadgroup memory or with SIMD reductions, then do one device atomic per grid node per block. This matters more on Apple, where device atomics go through a small L1 and a modest L2. Apple's faster shuffles make warp-level segmented reductions relatively cheap.
- 64-bit integer atomics are a common route to deterministic fixed-point scatter on CUDA (`atomicAdd` on `unsigned long long`). They are not generally available on Apple family 7/8. Deterministic accumulation on Metal therefore needs 32-bit fixed point (range/precision risk), split hi/lo words with carry handling, or a sort-then-segmented-reduce design.
- Float atomics are non-deterministic in order on both platforms. Whether Apple's float atomic add is performed at L2 or in the core was not established.

### Gaps
- No published contended float atomic-add throughput (ops/s) for any Apple GPU vs Ada was found. Atomic throughput in Turner's tables is absent.
- No measured Apple threadgroup memory bandwidth in B/clk per core was found beyond the "on-core data 64 B/cycle" figure. Ada shared memory bandwidth (commonly 128 B/clk/SM) was not sourced here.
- Whether M4 (family 9) adds 64-bit atomics was not verified.

## Q4. Global barrier / forward progress on Metal for persistent-thread kernels

### Takeaway
Metal has no grid-wide sync primitive (no equivalent of CUDA cooperative launch `grid.sync()`) and no documented guarantee of threadgroup co-residency or inter-threadgroup forward progress. Independent research found Apple GPUs exhibiting forward-progress failures. Practitioners avoid spin-waiting on other threadgroups. They use kernel boundaries (dispatch + barrier) as the global sync, tree/reduce-then-scan instead of decoupled look-back, or designs that stay correct under adversarial scheduling.

### Cited Findings
- Tyler Sorensen's group found that "Apple and ARM exhibit failures of forward progress". Decoupled look-back "depends on a forward progress guarantee" and can hang under an adversarial scheduler. Metal lacks a device-scope barrier, so there is "simply no way to run decoupled look-back on Metal" unless the payload fits in 32 bits. Workarounds: Elias Naur's variant, where waiting workgroups do scalar progress instead of spinning, and falling back to tree reduction (decoupled look-back is ~1.5x faster where it works) — [Raph Levien, "Prefix sum on portable compute shaders" (2021)](https://raphlinus.github.io/gpu/2021/11/17/prefix-sum-portable.html)
- A June 2026 developer question found no documented guarantee of threadgroup co-residency, global forward progress, or occupancy-bounded sync safety on Apple GPUs. In their experiments, schemes assuming all threadgroups eventually progress "can be unreliable", while strictly local producer/consumer handoffs were more robust. There is no Apple reply (0 replies) — [Apple Developer Forums 831017](https://developer.apple.com/forums/thread/831017)
- On CUDA, forward progress across CTAs is also not guaranteed for ordinary launches. Grid-wide sync is supported only through cooperative launch, which bounds the grid to resident capacity — [TVM discuss: global barrier on CUDA](https://discuss.tvm.apache.org/t/global-barrier-implementation-on-cuda/1232) (cooperative-launch detail is general CUDA knowledge, not from this page)

### Inferences
- A persistent-thread design for the strain solve (all colors/iterations inside one kernel with grid barriers) is supported on L4 via cooperative groups but is unsafe on Metal. The Metal design must pay per-dispatch barrier cost (Q2) or restructure so iterations are local to a threadgroup, e.g. block-Jacobi/Schwarz with local GS sweeps inside threadgroup memory and fewer global exchanges. That reduces dependent dispatches by the number of inner sweeps.
- A heuristic "occupancy-bounded" global barrier, launching no more threadgroups than fit concurrently, is risky on M3/M4. Dynamic Caching changes occupancy at runtime based on shader behavior, so the number of co-resident threadgroups is not statically knowable. This is an inference from Apple's tech talk.

### Gaps
- The Metal Shading Language specification's exact wording on forward progress was not retrieved.
- No published measurement of how many threadgroups are co-resident on M4 Max was found.

## Q5. Why Metal compute underperforms CUDA at equal FLOPS, and documented gap closures

### Takeaway
The documented reasons are host/dispatch overhead (Q2), a missing device-side control flow and grid-sync toolbox (no conditional graph nodes, dynamic parallelism, or cooperative launch), limited atomics (no general 64-bit, no fp16), a small L1/L2 vs Ada's 128 KB L1 + 48 MB L2, higher DRAM latency (>342 ns on M2 Pro), and pre-M3 static register allocation limiting occupancy. Documented gap closures come from kernel fusion, batching dispatches into one command buffer, avoiding Swift-level overhead, and exploiting Apple's bandwidth advantage. A clean "equal-FLOPS" CUDA-vs-Metal simulation case study was not found.

### Cited Findings
- Dispatch fusion: 7 to 3 dispatches gave 2.01x on M2 (wgpu/Metal) — [arXiv:2608.08730](https://arxiv.org/pdf/2608.08730); fusing an rfft/irfft pipeline into one Metal kernel removed most overhead — [arXiv:2605.05699](https://arxiv.org/pdf/2605.05699)
- Host binding: 25x speedup moving a dispatch-heavy bitonic sort from Swift to Objective-C — [Apple Developer Forums 750363](https://developer.apple.com/forums/thread/750363)
- Pre-M3 occupancy limited by peak register use. M3/M4 dynamic register allocation targets exactly the case of branchy kernels with one register-heavy path (vendor) — [Apple Tech Talk 111375](https://developer.apple.com/videos/play/tech-talks/111375/)
- Bandwidth-bound workloads: on FluidX3D (LBM, bandwidth bound), M2 Pro was close to an R9 390 and well ahead of AMD Phoenix, while recent discrete GDDR6 GPUs scored higher — [Chips and Cheese](https://chipsandcheese.com/p/a-brief-look-at-apples-m2-pro-igpu)
- HPC characterization: M1-M4 GPUs reach >200 GFLOPS/W. FP64 support on the GPU is limited (no native FP64). Grace-Hopper beats Apple on both bandwidth and compute — [Hübner et al., arXiv:2502.05317](https://arxiv.org/abs/2502.05317v2)
- A 2026 study showed GPU work on M4 Pro leaves residual state in the shared SLC that slows the next CPU traversal. This is relevant when CPU and GPU interleave in one sim step — [arXiv:2606.27098 (via EmergentMind)](https://www.emergentmind.com/papers/2606.27098)
- Smallest measured Apple GPU load draws ~800 mW (one simd) — [metal-benchmarks](https://github.com/philipturner/metal-benchmarks)

### Inferences (how to make a fair compute-adjusted comparison)
- Report several normalizations, not one: (1) wall-clock per sim step; (2) per peak FP32 (L4 30.3 vs M4 Max ~16.2-17.2 TFLOPS, factor ~1.8x); (3) per DRAM bandwidth (300 vs 546 GB/s, factor ~0.55x); (4) per watt, using measured package/board power (powermetrics on Mac, nvidia-smi on L4); (5) optionally per dollar-hour. Because the FLOPS and bandwidth ratios point in opposite directions (~1.8x each way), MPM results normalized only by FLOPS will make Apple look ~3.3x better than when normalized by bandwidth. State the roofline position of each kernel to justify which normalization applies.
- Decompose step time into (a) kernel-busy time and (b) inter-dispatch gaps, using GPU timestamps (Metal counter sample buffers or `gpuStartTime`/`gpuEndTime`; Nsight Systems on CUDA). Compare (a) per kernel against bandwidth/FLOP roofs and (b) as dispatch count x per-dispatch cost. This separates "Apple silicon is slower" from "the Metal API usage pattern is slower".
- Use the best idiom on each side: CUDA graphs (or PDL) on L4; one command buffer with a single concurrent compute encoder and explicit barriers on Metal. Use the same algorithm, precision (FP32), and determinism mode. Deterministic fixed-point atomics are 64-bit on CUDA but must be 32-bit or emulated on Metal, which can skew the comparison if not matched.
- Expected direction (inference): M4 Max should be relatively strong on large, DRAM-bound P2G/G2P and grid passes (1.8x bandwidth), and on SIMD-reduction-heavy code (2x shuffle bandwidth). It should be relatively weak on many tiny dependent dispatches (solver iterations), on working sets of ~4-48 MB that fit in L4's L2 but not Apple's L2, on cache-reuse-heavy BVH queries (8 KB L1), on contended device atomics, and on anything needing 64-bit atomics or grid-wide sync.
- Check L4 sustained clocks under the 72 W cap and Apple GPU clocks under sustained load (MacBook vs Mac Studio thermals). Both can deviate from the nominal peaks used in the normalization.

### Gaps
- No published case study porting a physics simulator (MPM/FEM/LBM) between CUDA on Ada and Metal on M3/M4 with per-kernel breakdown was found.
- Compiler quality (Metal's AIR to Apple GPU backend vs NVCC/ptxas) has no quantitative published comparison. Register spill behavior on Apple is not exposed the way `-Xptxas -v` exposes it on CUDA. Xcode's GPU profiler reports occupancy limiters, but no published numbers were found.
- Asahi Linux reverse-engineering docs were not consulted in this pass (budget). They may hold ISA-level details on atomics and barriers for family 7/8.

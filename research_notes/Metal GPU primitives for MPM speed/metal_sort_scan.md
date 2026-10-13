# GPU parallel primitives on Apple Silicon / Metal: stable radix sort, prefix scan, run-length encode

Scope: what to port into a Warp fork's Metal backend to replace the host versions of `warp.utils.radix_sort_pairs`, `runlength_encode` and `array_scan` (~130 ms per step today for 0.5-2M-element sorts, plus a GPU drain per call). The target is an M4 Max with 40 cores, 32-wide SIMD groups and 32 KB threadgroup memory.

Reference point in Warp itself, read from the local Warp source checkout at `scratch/deformable/patchtest/fresh/w/warp/native/`:
- The CUDA path calls `cub::DeviceRadixSort::SortPairs` with `cub::DoubleBuffer` and `begin_bit`/`end_bit`, which is why Warp's API requires keys and values buffers of size 2n (`sort.cu`).
- `scan.cu` calls `cub::DeviceScan::InclusiveSum`/`ExclusiveSum`.
- `runlength_encode.cu` calls `cub::DeviceRunLengthEncode::Encode`.
- Key types are int32, uint32, float32, int64, uint64 and float64. The value size is 4 or 8 bytes (`SortPayload<4>`/`<8>`). Scans cover int32, int64, float32 and float64 (`warp/_src/utils.py`).
- These are local code facts, not web sources.

## Q1. Which sort algorithms work on Apple GPUs, given no forward-progress guarantee between threadgroups?

### Takeaway
Use a multi-pass reduce-then-scan LSD radix sort (DeviceRadixSort / FidelityFX / Ha et al.-style): one upsweep histogram, one spine scan and one scatter per digit pass. It needs no inter-threadgroup spin-waiting, so it is correct on Apple GPUs by construction.
- **Onesweep** (and any chained scan with decoupled look-back) relies on a forward-progress guarantee that Apple GPUs lack. Without a fallback it hangs or runs slower than reduce-then-scan.
- **Decoupled Fallback** (Smith, Levien and Owens, SPAA 2025) makes chained scans safe on M1 Max and M3 and gets near-memcpy scan speed. It is a later optimization, not the first thing to port.

### Cited Findings
- **Apple's own position.** An Apple engineer said a chained (look-back) prefix-sum kernel that hung on M1 "relies on some undefined behavior." The assumption it relies on is that "for any pair of threadgroups that run concurrently … the GPU will fairly schedule them so that both make forward progress concurrently," which Apple has not documented because Metal runs on many GPUs that schedule differently. The engineer recommended "a more traditional prefix sum algorithm." — [Apple Developer Forums thread 672532](https://developer.apple.com/forums/thread/672532)
- **Sorensen et al. data.** "Apple and ARM exhibit failures of forward progress, so it cannot be specified in the Vulkan core." — [Raph Levien, "Prefix sum on portable compute shaders" (2021)](https://raphlinus.github.io/gpu/2021/11/17/prefix-sum-portable.html)
- **Levien's 2021 view on Metal.** He concluded that "there is simply no way to run decoupled look-back on Metal" unless the payload fits in a 32-bit word. His reason was that `threadgroup_barrier(mem_flags::mem_device)` has threadgroup scope, not device scope. — [Levien 2021](https://raphlinus.github.io/gpu/2021/11/17/prefix-sum-portable.html)
  - This predates Metal 3.2's device-scope `atomic_thread_fence`. forge-sort (below) states it requires Metal 3.2 for exactly that. — [forge-sort docs](https://docs.rs/forge-sort)
- **GPUSorting's own split.**
  - DeviceRadixSort "uses reduce-then-scan for the inter-threadblock prefix sum" and is the recommended choice when portability matters.
  - OneSweep uses chained-scan with decoupled lookback, depends on forward thread-progress guarantees, and the author says it "likely fails on … Apple hardware."
  - Shaders are wave-size agnostic (4 to 128), tested on 4, 16, 32 and 64. MIT license (checked via the GitHub API / LICENSE file).
  - Source: [b0nes164/GPUSorting](https://github.com/b0nes164/GPUSorting)
- **Linebender sorting survey.**
  - Single-pass approaches "perform poorly on devices without a forward progress guarantee," with Apple Silicon singled out.
  - Onesweep hardcodes warp size 32 and is hard to port. FidelityFX needs a subgroup size of at least 16.
  - Correct warp-level multi-split needs a subgroup barrier (`simdgroup_barrier` in Metal).
  - Source: [linebender.org/wiki/gpu/sorting](https://linebender.org/wiki/gpu/sorting/)
- **Onesweep (Adinets & Merrill 2022).**
  - LSD radix sort with a single-pass prefix sum. Each digit-binning iteration costs about 2n global reads and writes, against about 3n for two-pass reduce-then-scan sorts.
  - 29.4 GKey/s on 256M random 32-bit keys on an A100, about 1.5x CUB's previous LSD sort.
  - Source: [arXiv 2206.01784](https://arxiv.org/abs/2206.01784)
- **Decoupled Fallback: mechanism.**
  - A workgroup that waits on a predecessor tile spins at most a set number of times (4 in the paper), then recomputes that tile's reduction itself (work stealing).
  - Atomic compare-and-swap makes sure only one workgroup posts the result. Without a guarantee, a plain chained scan "at best" runs slower than reduce-then-scan and "frequently triggers" timeout/recovery.
  - Source: paper LaTeX in [b0nes164/Decoupled-Fallback-Paper](https://github.com/b0nes164/Decoupled-Fallback-Paper); DOI [10.1145/3694906.3743326](https://dl.acm.org/doi/10.1145/3694906.3743326)
- **Decoupled Fallback: measured inclusive prefix sum, 2^25 32-bit elements, Dawn/WebGPU, 10^9 elements/s.** — same paper, Table "results"

  | Device | Memcpy | Decoupled Fallback (DF) | Reduce-then-scan (RTS) | DF / RTS |
  |---|---|---|---|---|
  | M1 Max (32 cores, occupancy 160, subgroup 32) | 37.42 | 36.85 | 25.75 | 143% |
  | M3 (10 cores) | 10.82 | 10.87 | 7.46 | 145.7% |

  - The paper also notes that "a scheduler can be fair yet still lack FPG," using the M3 as the example.
- **Decoupled Fallback: tile state.**
  - It avoids 64-bit atomics by splitting each reduction into 16-bit pieces, each carried in its own 32-bit atomic with a 2-bit state copy. A message counts as received only when all parts agree.
  - This supports 32- and 64-bit payloads using only relaxed 32-bit atomics.
  - Tuning used: workgroup 256, 16 elements per thread (tile 4096), about 1 KB of shared memory.
  - Source: [Decoupled-Fallback-Paper](https://github.com/b0nes164/Decoupled-Fallback-Paper)
- **GPUPrefixSums repo.** It reports that Decoupled Fallback "shows promising results on Apple M GPU's" (9/2024), says its included code is out of date, and says active development moved to Vello. — [b0nes164/GPUPrefixSums](https://github.com/b0nes164/GPUPrefixSums)

### Inferences
- **For the first port, take a reduce-then-scan LSD sort.**
  - It costs about 3n traffic per pass against about 2n for Onesweep. At 2M elements on an M4 Max, the absolute cost is a few milliseconds at most (see Q2), against 130 ms today, so the extra pass is worth the guaranteed correctness.
- **Decoupled Fallback (or Onesweep with fallback) is a later 1.3-1.5x optimization.** Scope it only if the sort becomes the bottleneck again.
- **Do not port plain decoupled look-back (CUB-style) to Metal.** A hang on Apple GPUs is a GPU timeout, not just a slowdown.

### Gaps
- I found no published measurement of Onesweep or Decoupled Fallback running natively in MSL (as opposed to Dawn/WebGPU) on M4-class GPUs.
- I found no Apple documentation, as of 2026, that adds forward-progress guarantees for M3/M4 (Apple GPU family 9). The M3 result above shows fair scheduling in practice but no guarantee.

## Q2. Existing Metal implementations and measured throughput

### Takeaway
None of the open, permissively licensed options is a drop-in Metal stable radix sort-pairs, so there is nothing ready to copy.
- **MetalRadixSort** (Swift/Metal, stable (u64, u32) pairs, single encoder, no readback) is the closest match to Warp's need. It is AGPL-3.0, which is likely incompatible with a permissive Warp fork.
- **forge-sort**, which it derives from, is fast but its LSD passes were unstable (racing atomics), and it is also AGPL.
- **MLX** has a stable merge sort, not a radix sort.
- **MPS** has no sort primitive.
- **Recommendation:** port b0nes164's MIT DeviceRadixSort (HLSL, reduce-then-scan, 8-bit digits) to MSL, using MetalRadixSort/forge-sort only as design references for Apple-specific tricks.

### Cited Findings
- **MetalRadixSort (mnmly), created May 2026.**
  - "Stable GPU radix sort for (UInt64 key, UInt32 value) pairs on Metal. Single command encoder, no CPU readback." Equal keys come out in input order "since 0.3.0." Up to about 13 dispatches.
  - `beginBit`/`endBit` mirror `cub::DeviceRadixSort::SortPairs`, byte-granular.
  - License AGPL-3.0-only. Built for gsplat's tile-key sort.
  - M5 Max, 1M random (u64, u32) pairs: about 4.1 ms full 64-bit (8 passes plus a gather), about 3.6 ms for `endBit=50`, about 2.1 ms for `endBit=32`.
  - Source: [github.com/mnmly/MetalRadixSort README](https://github.com/mnmly/MetalRadixSort)
- **MetalRadixSort architecture (STATUS.md).**
  - An MSD scatter on the top byte into 256 buckets, then 7 LSD passes per bucket in threadgroup memory.
  - Ranks come from a deterministic "split-and-flag" bit-sort: 9 one-bit passes per 2048-element sub-tile, using a SIMD-group prefix plus a serial cross-SIMD-group combine, with no atomics.
  - A per-(tile, digit) offset table replaces the global atomic in the MSD scatter. Stability cost about 10-15% (3.5 ms to 4.1 ms).
  - Tests check the exact value permutation against a CPU stable sort.
  - Source: [MetalRadixSort STATUS.md](https://github.com/mnmly/MetalRadixSort/blob/main/STATUS.md)
- **forge-sort was not stable.** Its `sort_inner_fused` used a racing `atomic_fetch_add` for the within-SIMD-group rank. That made each LSD pass non-stable and "~1% wrong on random u64 keys"; forge-sort's tests only used a 3-element input. — [MetalRadixSort STATUS.md](https://github.com/mnmly/MetalRadixSort/blob/main/STATUS.md)
- **forge-sort 0.2.3 (Rust/Metal, AGPL-3.0 or commercial).**
  - MSD 8-bit histogram and atomic scatter to 256 buckets, then 3 fused LSD passes per bucket. 4 dispatches in one encoder.
  - Per-SIMD-group atomic histograms with 8 parallel accumulators. Requires Metal 3.2 (device-scope `atomic_thread_fence`).
  - Self-reported on an M4 Pro (20-core GPU), 16M elements, Mk/s:

    | Operation | Mk/s |
    |---|---|
    | u32 sort | 2,802 |
    | u32 sort_pairs | 882 |
    | u64 sort | 918 |
    | u64 sort_pairs | 450 |
    | i64 sort_pairs | 424 |

  - 1M u32 sort: about 779-836 Mk/s. 64-bit is "roughly 3x slower."
  - The docs.rs build had failed.
  - Source: [docs.rs/forge-sort](https://docs.rs/forge-sort)
  - Stability caveat: see the previous finding.
- **MLX sort.**
  - `mlx/backend/metal/kernels/sort.h` is a block merge sort "based on GPU merge sort algorithm" from CUB/CCCL: odd-even sort in registers, merge-path merges in threadgroup memory, and a multi-block scheme (`mb_block_sort` / `mb_block_partition` / `mb_block_merge`) for large arrays.
  - Ties take the left element and the comparisons are strict, so it behaves as a stable sort; the code does not state stability explicitly. It is not a radix sort. MIT.
  - Source: [mlx sort.h](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/sort.h)
- **MPS has no sort or scan primitive.** An Apple engineer answered a request for Reduce, Scan or Radix sort in MPS by asking developers to file a radar. — [Apple forum 105886](https://developer.apple.com/forums/thread/105886); forge-sort docs also state MPS lacks a sort ([docs.rs/forge-sort](https://docs.rs/forge-sort))
- **Levien's WebGPU sort experiment (compute-shader-101 PR #31, Apache-2.0 repo).**
  - FidelityFX-style sort with Onesweep-inspired warp-local multi-split (WLMS), with subgroup ops emulated in shared memory: "approximately 1G element/s on M1 Max."
  - The linebender wiki reports about 3G elements/s for a Metal port with real subgroup ballots, on 16M elements. A stronger barrier cost about 2%. 4-bit digits; 8-bit "not yet yielded sustained gains" in WebGPU.
  - Sources: [compute-shader-101 PR #31](https://github.com/googlefonts/compute-shader-101/pull/31); [linebender wiki](https://linebender.org/wiki/gpu/sorting/)
- **wgpu_sort (BSD-2).** WGSL re-implementation of Fuchsia's Vulkan radix sort, 32-bit key-value only. It guesses subgroup size by trial. No Apple numbers given; 1M pairs in 0.317 ms on an RTX A5000. — [KeKsBoTer/wgpu_sort](https://github.com/KeKsBoTer/wgpu_sort)
- **FidelityFX Parallel Sort (MIT).** 4-bit digits (ranking via two 2-bit LSD passes); needs subgroup size of at least 16. — [linebender wiki](https://linebender.org/wiki/gpu/sorting/); license from the GitHub API for GPUOpen-Effects/FidelityFX-ParallelSort
- **Kieber-Emmons.** "Memory Bandwidth Optimized Parallel Radix Sort in Metal for Apple M1 and Beyond" (Medium) builds radix sort on his Metal scan/reduce primitives. The page returned 403, so I could not get its numbers or code. — [medium.com/p/4f4590cfd5d3](https://medium.com/p/4f4590cfd5d3)
- **Gaussian-splatting Metal projects** (msplat and others) mostly use per-tile bitonic sorts, not a global radix sort. — [msplat on PyPI](https://pypi.org/project/msplat/)

### Inferences
- **Expected speed at Warp's sizes.**
  - MetalRadixSort's about 4 ms for 1M full 64-bit pairs on an M5 Max suggests about 6-10 ms for 2M int64 pairs at all 64 bits on an M4 Max.
  - Passing a tight `end_bit` would cut that a lot (Warp's BVH and hash-grid keys rarely use all 64 bits).
  - A 32-bit-key sort of 2M pairs should be about 1-3 ms.
  - Either is roughly a 15-100x improvement over 130 ms. These are extrapolations, not measurements.
- **Licensing.** Warp is permissively licensed (Apache-2.0, to my knowledge; verify). The AGPL code (MetalRadixSort, forge-sort) should be used only as a reference for ideas, not copied. The MIT/Apache sources are GPUSorting, FidelityFX, MLX and compute-shader-101.
- **Bit-identical results.** A stable LSD radix sort over the same bit range gives exactly one possible output permutation. That should match Warp's host radix sort bit for bit, provided float and signed keys use the same order-preserving bit transform (sign-flip for ints, flip-all-if-negative for floats) and NaN and -0.0 handling matches the host. Verify -0.0 against +0.0 explicitly.

### Gaps
- I found no published keys/s numbers for MLX argsort, PyTorch-MPS `torch.sort` (which uses MPSGraph), or Taichi/Genesis Metal sorts on M-series chips.
- I could not check whether wgpu_sort or Fuchsia's sort runs correctly on Metal; Fuchsia's sort design was not verified here.
- I found no independent M4 Max measurements of any Metal radix sort; all numbers are self-reported.

## Q3. Scan and run-length encode on Metal

### Takeaway
- **Scan:** build a three-kernel reduce-then-scan:
  1. A per-threadgroup reduction using `simd_sum` / `simd_prefix_exclusive_sum` plus a small threadgroup-memory spine.
  2. A single-threadgroup spine scan.
  3. A downsweep scan that adds the spine offset.
- **Speed:** on an M1 Max this reached about 69% of memcpy (25.75 against 37.42 G elements/s). That is far beyond what Warp needs at 2M elements, where a scan should take tens of microseconds.
- **RLE:** compute flags, scan, then scatter, the same as CUB-style `DeviceRunLengthEncode`. The run count is written to a device buffer, not read back.

### Cited Findings
- **Apple SIMD costs (Apple 7/8, cycles).**

  | Operation | Throughput | Adjusted latency |
  |---|---|---|
  | `PREFIX_ESUM<F32>` | 11.57 | 21.93 |
  | `PREFIX_ISUM<I32>` | 13.01 | 24.78 |
  | `BALLOT` | about 2 | not given |
  | Broadcast / rotate shuffle | about 2 | not given |
  | Random shuffle | 28-32 | not given |

  - Other figures: SIMD shuffle bandwidth 256 B/cycle per core, about 60 KB of shared memory per core, about 208 KB register file per core, 128 B global cache line.
  - Apple GPUs support only 32-bit atomics on device pointers; M2 added non-returning u64 min/max.
  - Source: [philipturner/metal-benchmarks](https://github.com/philipturner/metal-benchmarks)
- **Reduce-then-scan and chained scan on Apple** are covered in Q1 (M1 Max RTS 25.75 G elements/s, DF 36.85). — [Decoupled-Fallback-Paper](https://github.com/b0nes164/Decoupled-Fallback-Paper)
- **Workgroup raking (Merrill & Grimshaw).** Split the input among a fixed number of workgroups c, with c at least the occupancy. This caps the spine at a fixed size, so recursion depth is at most 2 and there are only 3 launches. Classic recursive reduce-then-scan instead needs 2⌈log_t n⌉−1 launches. — [Decoupled-Fallback-Paper LaTeX](https://github.com/b0nes164/Decoupled-Fallback-Paper)
- **Chained scans fuse compaction ("in-situ compaction").** That is relevant to RLE and stream compaction if a fallback scan is adopted later. — same source
- **GPUPrefixSums** catalogs the block-level and device-level scans (Kogge-Stone, Brent-Kung, raking reduce-scan, warp-sized-radix variants, reduce-then-scan, chained scan, Decoupled Fallback) in D3D12/HLSL, CUDA, Unity and WGPU. Its license file reads "NOASSERTION" via the API; check the LICENSE file. — [b0nes164/GPUPrefixSums](https://github.com/b0nes164/GPUPrefixSums)

### Inferences
- **Fixed launch shape.** Use the raking layout: a fixed number of threadgroups, for example a few times 40 cores, each scanning a contiguous chunk serially in tiles. The spine is then at most a few hundred entries and one threadgroup scans it. This keeps a fixed three-dispatch structure for any n, which suits capture.
- **Integer scans are exact by construction.** Integer addition is associative, so int32/int64 results match the host bit for bit. int64 needs 64-bit adds in registers; Metal supports `long` arithmetic. SIMD prefix intrinsics for 64-bit types should be verified or emulated with shuffles.
- **Float scans will not match a sequential host scan.**
  - A parallel float scan reassociates additions, so float32/float64 results will differ from a sequential host scan in the last bits. Warp's CUB path has the same property.
  - If bit-identity with the current host path is a hard requirement for float scans, the options are:
    - accept tolerance-based tests;
    - run float scans as one serial thread on the GPU, which is slow but exact for small n;
    - do the scan in fixed point.
  - float64 on Apple GPUs has no native hardware and would need emulation.
- **RLE recipe (sorted input).**
  1. `flag[i] = (i==0 || in[i]!=in[i-1])`.
  2. Exclusive scan of the flags gives `run_id`. Also write `num_runs = scan[n-1]+flag[n-1]` to a device buffer.
  3. Where `flag[i]`, write `run_values[run_id]=in[i]` and `run_start[run_id]=i`.
  4. Second kernel: `run_length[j] = start[j+1]-start[j]`, with `n` for the last run.
  - All of this is integer, deterministic and identical to the host.
- **Segmented reductions.** Use the same flags with a segmented scan (a head-flag operator in the SIMD scan, or scan then subtract at segment heads for integer sums).

### Gaps
- No public M4 measurements of a native MSL scan.
- I did not find whether `simd_prefix_exclusive_sum` supports 64-bit integer types in MSL 3.x/4; check the MSL specification.

## Q4. Practical details: 64-bit keys, passes, digit size, occupancy, threadgroup atomics, capture-friendliness

### Takeaway
- **Digits and passes:** 8-bit digits and 256 bins, so 4 passes for 32-bit keys and 8 for 64-bit, minus passes skipped via `begin_bit`/`end_bit`.
- **Ranking:** use deterministic SIMD-ballot or split-and-flag ranking, never racing threadgroup or device atomics in the ranking/scatter step.
- **Encoding:** encode every pass into one compute encoder with fixed dispatch counts, with n read from a device buffer if capture needs variable n.

### Cited Findings
- **Digit sizes elsewhere.**
  - 8-bit: Onesweep, DeviceRadixSort, Hybrid Radix Sort, Splatter.
  - 4-bit: FidelityFX, and Levien's WebGPU hybrid, where 8-bit had not yet paid off without real subgroups.
  - Radix sorts typically need 4 or 8 passes, against dozens for bitonic.
  - Source: [linebender wiki](https://linebender.org/wiki/gpu/sorting/)
- **64-bit cost.**
  - forge-sort: 64-bit keys are "roughly 3x slower due to halved tile size and extra passes"; u64 sort_pairs 450 Mk/s against u32 882 Mk/s at 16M on an M4 Pro. — [docs.rs/forge-sort](https://docs.rs/forge-sort)
  - MetalRadixSort: skipping high bytes with `endBit` cuts time by about 12-50%. — [MetalRadixSort](https://github.com/mnmly/MetalRadixSort)
- **Atomics and stability.**
  - Racing `atomic_fetch_add` for ranks produces correct counts but non-deterministic order, which breaks LSD stability.
  - The stable fix is a per-(tile, digit) prefix offset table plus deterministic in-tile ranking.
  - Source: [MetalRadixSort STATUS.md](https://github.com/mnmly/MetalRadixSort/blob/main/STATUS.md)
- **Atomics for histograms.** forge-sort reduces contention with per-SIMD-group atomic histograms, 8 parallel accumulators each. — [docs.rs/forge-sort](https://docs.rs/forge-sort)
  - Histogram counts are order-independent, so atomics there do not affect stability; only rank assignment must be deterministic.
- **Warp-level multi-split** (Onesweep-style ranking with ballots) needs `simdgroup_barrier` for correctness in Metal. — [linebender wiki](https://linebender.org/wiki/gpu/sorting/)
- **Capture-friendly design.** MetalRadixSort encodes its whole sort into one compute encoder, with up to about 13 dispatches, no encoder boundaries and no CPU readback, so it schedules "alongside the rest of your frame work." — [MetalRadixSort](https://github.com/mnmly/MetalRadixSort)
  - wgpu_sort supports indirect dispatch. — [wgpu_sort](https://github.com/KeKsBoTer/wgpu_sort)
- **Apple threadgroup memory and occupancy** (about 60 KB per core, 384-3072 threads per core): see [metal-benchmarks](https://github.com/philipturner/metal-benchmarks). I found no published numbers on threadgroup-atomic throughput.

### Inferences
- **Concrete plan for Warp's `radix_sort_pairs` on Metal, porting GPUSorting DeviceRadixSort (MIT) to MSL.**
  - **Per pass, three dispatches:**
    1. Upsweep: each threadgroup builds a 256-bin histogram of its tile. Use per-SIMD-group sub-histograms in threadgroup memory, or 8 KB of `atomic_uint` threadgroup counters; order-independent, so this is safe. Write the counts into a `[digit][tile]` global array.
    2. Scan: one threadgroup per digit, or a single threadgroup, does an exclusive scan of the `[digit][tile]` table.
    3. Downsweep/scatter: each threadgroup reloads its tile, ranks keys stably within the tile, and writes each key and value to `global_offset[digit][tile] + local_rank`.
       - Ranking can use SIMD-ballot WLMS with `simd_ballot` plus popcount over 8 bits, or a split-and-flag bit-sort.
       - Stability across tiles comes from tile order in the scanned table. Stability within a tile comes from deterministic ranking.
  - **Tile size:** for example 256-512 threads with 8-16 keys per thread, giving 2K-8K keys per tile. 64-bit keys need roughly half the keys per thread to fit in 32 KB of threadgroup memory.
  - **Scheduling inside one encoder:**
    - Ping-pong between the two halves of Warp's 2n buffers, matching CUB's `DoubleBuffer` semantics. The result must end up in the first half, or Warp must expose which half holds it, as CUB does via `d_keys.Current()`. Warp currently copies back, so check `sort.cu`.
    - Insert `memoryBarrierWithScope:MTLBarrierScopeBuffers` between dependent dispatches, or rely on the encoder's default hazard tracking.
  - **Pass count:** about 3 × number_of_passes dispatches in total, 24 for full 64-bit.
- **Pass count from `end_bit`.** Warp's callers often know the key bit width, for example a grouped BVH's `(group << 32) | morton` key. Plumbing `end_bit` through and rounding up to 8-bit digits avoids wasted passes. If the bit range is not a multiple of 8, the last digit can be narrower.
- **Fixed dispatch sizes for capture.** Size the grid for `max_n` and have kernels read `n` from a device buffer, early-exiting tiles past n. That keeps graph and capture replay valid even when `n` changes. Warp's `count` argument is host-side today, so allocate the grid by array capacity.
- **Signed and float keys.** Apply the standard order-preserving bit transform on load and invert it on store, as CUB does. Then int64/float64 keys sort through the same unsigned 64-bit path.
- **Decoupled Fallback later.** Fusing the upsweep into a Onesweep-style single pass with Decoupled Fallback would cut traffic from about 3n to about 2n per pass, a 1.3-1.5x gain on M1 Max-class hardware per the scan results. Consider it only after the reduce-then-scan version is correct and profiled.

### Gaps
- No measured data on Apple threadgroup-memory atomic throughput or contention (`atomic_fetch_add` on threadgroup `atomic_uint`). This needs a microbenchmark on the M4 Max.
- No authoritative source found on whether M4 (Apple GPU family 9) changed scheduling or forward-progress behavior compared with M1/M3.
- Whether MSL `simd_ballot` / `simd_prefix_*` support 64-bit operands is not verified here.

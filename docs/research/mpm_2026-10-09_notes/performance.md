# GPU performance techniques for MPM and how they transfer to Apple Silicon Metal (M4 Max, 40-core GPU)

State as of 2026-10-09. Every literature number names its GPU. "Measured here" means a microbenchmark run for these notes on
this machine (Apple M4 Max, macOS 26.7.1, runtime-compiled MSL, default language version plus checks under MSL 3.0–4.0).
Benchmark source: `/private/tmp/claude-501/-Users-aditya-robosim/3a553738-d88f-4d8d-ac36-4ecff3ac4bc1/scratchpad/bench/p2g.swift`
(scratchpad, not in the repo). "Estimate" marks anything inferred and not measured.

**Correction to the task premise, measured here:** on this M4 Max, `atomic_float` add is available in **device** memory
only. `threadgroup atomic_float` does **not** compile under MSL 3.0, 3.1, 3.2 or 4.0 ("no matching function for call to
atomic_fetch_add_explicit ... cannot pass pointer to address space 'threadgroup'"). Threadgroup memory has native 32-bit
**integer** atomics only. 64-bit device atomics compile for `atomic_max_explicit` on `atomic_ulong` but **not** for
`fetch_add` or compare-exchange (MSL 4.0). That contradicts Apple's feature-set table, which says "the full set of 64-bit
atomic operations is supported on all platforms starting with Apple9" ([Metal feature set tables](https://developer.apple.com/metal/limits)).
Either the table is ahead of the shipping compiler or the full set needs something these tests didn't use. This changes the
P2G plan (see Q1 and Q3).

---

## Q1. The P2G scatter bottleneck: atomics, sorting, SIMD-group reduction, threadgroup accumulation, G2P2G, sparse structures

### Takeaway
In the literature, P2G dominates explicit MPM. Gao 2018 made P2G 23× faster than atomic-only scattering (cell sort, warp
reduction, shared-memory block accumulation). Wang 2020 fused G2P2G and used AoSoA for 2–3× more. Fei 2021 found that on
V100 the CAS-emulated shared-memory float atomics were the bottleneck and dropped them for 3–5×; it also sorted only at
"rebuild mapping" steps, every 10–30 steps.

On the M4 Max the same split holds, measured here. Float atomics are native in device memory but CAS-emulated in
threadgroup memory, while integer atomics are native in both. The fastest P2G measured is a **threadgroup tile of
fixed-point int32 atomics, flushed with int32 device atomics: 0.33–0.48 ms per million particles (16.8M particles,
quadratic B-spline, 4 channels)**. That is 6–8.7× faster than plain device float atomics and bitwise deterministic. Next
best is a **SIMD-group segmented reduction plus device float atomics** at 0.50 ms/M (4.6× over sorted plain atomics).

### Cited findings — literature

**Gao et al. 2018, "GPU Optimization of Material Point Methods" (SIGGRAPH Asia / TOG 37(6))** — [PDF](https://people.csail.mit.edu/kuiwu/GPU_MPM/GPU_MPM.pdf)
- GSPGrid: a GPU port of SPGrid with 4×4×4-cell blocks and 16 channels. Each CUDA block handles at most 512 particles
  (8 particles per cell × 64 cells).
- P2G method: particles sorted by cell; warp-level reduction (shuffle) of same-node writes inside each warp; shared-memory
  accumulation for cross-warp conflicts; bulk write to global memory. "Crossing-warp conflicts only infrequently happen
  such that the negative impact caused by atomic-adds to the shared memory is almost negligible" (on their hardware).
- Measured: their P2G-MLS is **23× faster than atomic-only scattering and 15× faster than GVDB gathering**. Hardware: about
  7M uniformly sampled particles at Δx = 1/128, NVIDIA TITAN Xp (Fig. 6, read off the plot: the axis tops out at 180 ms,
  which is the scale of the naive atomic and GVDB P2G times).
- Against SIMD CPU code (18-core Xeon Gold 6140): P2G and G2P are more than 16× faster than CPU FLIP, and about 8× and 13×
  faster than CPU MLS.
- With atomic-only scattering or GVDB gathering, "P2G occupies more than 90% of the solver time"; with their method,
  about 40%.
- Sorting: they replace radix sort with a **histogram (counting) sort** keyed on (hashed block index, cell within block).
  Spatial hashing turns sparse 64-bit SPGrid offsets into consecutive block ids. "The observed performance speedup
  approaches an order of magnitude."
- Delayed reordering: only positions are reordered each step; the other attributes are written in sorted order as a side
  effect of the kernels. This "can reduce sorting time dramatically while all other kernels are barely impacted" (Fig. 10,
  7M particles).
- Particle density benchmark (Fig. 7): about 3.5M particles at PPC 4–16, a whole explicit step of roughly 2–6 ms on
  Quadro P6000 (read off the bar chart).
- Production scenes, implicit, P6000 (Table 1): 9.0M particles on a 512³ grid at dt = 1e-4 took 20.95 s per 48 Hz frame.
  The paper's teaser: 4.2M particles, 256³ grid, 10.5 s per frame.
- Memory: 2.4 GB for 7M particles, 900K cells, 17K blocks; 70 MB for 95K particles.

**Wang et al. 2020, "A Massively Parallel and Scalable Multi-GPU MPM" (SIGGRAPH / TOG 39(4), claymore)** — [paper](https://yzhu.io/publication/mpmgpu2020siggraph/paper.pdf), [project](https://yzhu.io/publication/mpmgpu2020siggraph/)
- G2P2G: fuses G2P, the particle update and the next step's P2G into one kernel. Removes two transfer kernels; particle
  velocity becomes kernel-local. Needs double-buffered particles and an "off-by-two" staggered block mapping so particles
  touch only 2×2×2 grid blocks (4×4×4 nodes each) in shared memory.
- Binning instead of warp reduction: they **leave particles unsorted at the cell level** and pseudo-color particles into
  bins so lanes in a warp hit different nodes. The paper explains the choice: the Gao warp reduction "imposes restrictions
  on the order of the particles, which cannot be satisfied in the context of the G2P2G pipeline".
- Ablation, RTX 2080, ms (Table 4, sum of transfer kernels):

  | scene | particles | Hu 2019 ref | G2P2G | speedup | AoSoA + G2P2G | speedup |
  |---|---|---|---|---|---|---|
  | dragons | 775,196 | 3.98 | 2.91 | 1.37× | 1.33 | 2.99× |
  | bomb falling | 3,193,038 | 16.95 | 12.25 | 1.38× | 7.00 | 2.42× |
  | cube (uniform, ordered) | 262,144 | 0.99 | 1.10 | **0.9×** | 0.74 | 1.34× |

  G2P2G alone is about 1.4×, and **slower on an already ordered uniform cube**: fusion "may lessen the latency-hiding
  capability".
- Full step against Hu 2019 Taichi, ms per step (Table 2): dragons (775K) 4.3 → 2.0 on P6000 and 3.0 → 1.5 on RTX 2080;
  bomb falling (984K) 6.4 → 3.4 on P6000. About 2× overall, 2.5× with Taichi's initial sort disabled.
- Scaling, single Quadro P6000, ms per step (Table 3): 5M particles 5.97; 10M 11.91; 20M 27.38; 40M 50.15. That is
  **about 1.2–1.4 ms per million particles per step**, 110–120× an 8086K CPU (SIMD code by Fang et al. 2019).
- Bytes per particle: G2P2G 16 B for fluid and 52 B for solid; the conventional pipeline 68 B and 100 B.

**Fei, Huang, Gao 2021, "Principles towards Real-Time Simulation of MPM on Modern GPUs" (arXiv 2111.00699)** — [arXiv](https://arxiv.org/abs/2111.00699)
- 1.7–8.6× over Wang 2020 on one GPU and 2.5–14.8× on four. Real time on 4× V100 (NVLink): 1.33M-particle snow at 68.5 fps
  and 143K-particle fountain at 55.9 fps.
- **Avoid non-native atomics.** On sm_70, a float atomicAdd to shared memory compiles to an ATOMS.CAS loop, while the
  global float add is a single RED.E.ADD.F32 instruction. Dropping shared-memory accumulation and writing **straight to
  global memory with native atomics** after the warp reduction gave **3–5×** (Fig. 10). It also raised occupancy and
  allowed 3×3×3 neighbour blocks.
- Warp reduction is kept: particles sorted to cell granularity, then warp-level segmented reduction ("more robust towards
  extreme scenarios" than the Wang/Hu shuffle-to-avoid-conflicts approach).
- Sort frequency: a "free zone" of (10Δx)³ per particle block means block mapping and hashing are rebuilt only when
  particles leave it. Observed **every 10–30 steps** (36 steps per frame). Between rebuilds there is an **in-warp radix sort
  on 10 bits** before the reduction. Skipping the full sort gave 8–17%; a full re-sort every step costs **+11.3–24.8%**.
  The free zone alone gave **3.7×** at 55.3K particles, where rebuild-mapping was the bottleneck.
- They use G2P2G only below about 100K particles per GPU, and kernel merging otherwise. Fast-math intrinsics: negligible
  error (stress error ≤ 0.12 dyne/cm² on 3.1M sand particles).

**Hu et al. 2019 Taichi (sparse SNodes)** — numbers via Wang 2020 Table 2 above (P6000: 4.3 ms per step for 775K
particles). Taichi's MPM was the reference "state of the art" in 2020. [Wang 2020](https://yzhu.io/publication/mpmgpu2020siggraph/paper.pdf)

**QuanTaichi (Hu et al. 2021)** — [project](https://yuanming.taichi.graphics/publication/2021-quantaichi/), [repo](https://github.com/taichi-dev/quantaichi)
- 1.7× compression per MPM particle; 235M MPM particles on one GPU. Speed is described only qualitatively ("often run as
  fast as the full-precision simulators"); no MPM speedup figure was found.
- Follow-up "Automatic Quantization for Physics-Based Simulation": 2.21× and 2.15× memory compression for MLS-MPM (position,
  velocity, F and C quantized); CUDA MPM 2D/3D only 1.04× slower on average with dithering. [arXiv 2207.04658](https://arxiv.org/pdf/2207.04658)

### Cited findings — measured here on the M4 Max (P2G only)

Setup:
- Particles: 8 per cell, jittered. Quadratic B-spline weights over 3×3×3 nodes; 4 channels (mass and APIC momentum
  m·v + A·dpos, with a 3×3 affine matrix A per particle): 108 scattered adds per particle.
- Particle record: 80 B. Grid: dense, (n+3)³ nodes × float4.
- Timing: GPU timestamps, median of 30. Grid clear is a blit fill, not counted. Correctness: largest per-node difference
  against the float-atomic result; total mass checked.
- Ordering: "sorted" = block-major (4×4×4 cells), then sorted by the B-spline base node within each block (the key P2G
  reductions need). The first run, in plain cell order, is noted where it differs.

| variant | 0.26M (32³) ms | 2.1M (64³) ms | 16.8M (128³) ms | **ms per M particles** (16.8M) | effective G adds/s |
|---|---|---|---|---|---|
| V0 non-atomic racy RMW (wrong results; memory and ALU floor) | 0.054 | 0.33 | 2.88 | 0.17 | 629 |
| V1 device `atomic_float`, unsorted particles | 1.04 | 5.98 | 47.9 | 2.85 | 37.8 |
| V2 device `atomic_float`, sorted | 0.76 | 4.78 | 38.6 | 2.30 | 46.9 |
| V3 sorted + SIMD-group segmented reduction (`simd_shuffle_down`, `simd_ballot` run ends) + device float atomics from run heads (Gao 2018 / Fei 2021 style) | 0.17 | 1.05 | 8.32 | **0.50** | 218 |
| V4 one threadgroup per block, threadgroup float tile via CAS loop on `atomic_uint`, flush with device float atomics | 1.73 | 11.7 | 92.9 | 5.54 | 19.5 |
| V5 fixed-point int32 device atomics, sorted (2^16 scale) | 0.61 | 4.76 | 38.3 | 2.28 | 47.3 |
| V5u same, unsorted | 0.76 | 5.98 | 47.9 | 2.85 | 37.9 |
| V6 one threadgroup (512 thr) per block, **threadgroup int32 tile (7³ nodes × 4 ch = 5.5 KB) + int32 device flush** | 0.13 | 1.02 | 8.09 | **0.48** (0.33 in plain cell order: 5.56 ms) | 224 (326) |
| V7 V6 + SIMD-group reduction before the threadgroup atomics | 0.12 | 0.89 | 6.98 | **0.42** | 259 |

- Determinism: float-atomic results differ bitwise between two runs (run on unsorted particles). Fixed-point results
  (V5, V6, V7) were bitwise identical across runs.
- Fixed-point accuracy at a 2^16 scale: the largest absolute difference from float atomics was 3.3e-4, on node values up to
  about 12 (about 3e-5 relative). Range is ±32768 per channel.
- Atomic share: plain device float atomics (V2) cost 2.30 ms/M against a 0.17 ms/M non-atomic floor, so atomics are
  about 93% of P2G time.
- Sorting gain: on plain device atomics it is only 1.24× (V1 → V2). The big gains come from reducing before the atomic
  (V3: 4.6×) or moving the atomic into threadgroup memory as integers (V6/V7: 4.8–7×).
- Ordering inside a threadgroup matters for threadgroup integer atomics. Base-node order puts same-address writes in the
  same SIMD group, which serializes them: V6 went 0.33 → 0.48 ms/M. Adding a SIMD reduction (V7) recovers part of that:
  0.42 ms/M.
- **CAS-emulated threadgroup float atomics (V4) are the slowest option, 2.4× slower than plain device float atomics.** This
  is the Apple equivalent of Fei 2021's sm_70 finding.
- Apple's own guidance (WWDC22 "Scale compute workloads across Apple GPUs"): global atomics contend and adding cores "leads
  to more contention". Recommended pattern: reduce in the SIMD group, then one threadgroup atomic per SIMD group, then one
  device atomic per threadgroup. Rule of thumb: 1K–2K concurrent threads per GPU core for good occupancy; up to 1024
  threads per threadgroup. No atomic cost numbers are given. — [WWDC22 10159](https://developer.apple.com/videos/play/wwdc2022/10159)
  (The fetched summary's example code used `threadgroup atomic_float`, which does not compile on this machine; see the
  correction at the top.)

### Inferences
- Recommended P2G for Metal: **V7/V6-style**. One threadgroup per 4×4×4 block, a threadgroup int32 fixed-point tile, an
  optional SIMD segmented reduction first, and an int32 device flush.
  - It is the fastest variant measured, deterministic, and fits easily in 32 KB (5.5 KB at 4 channels; 7³ × 10 channels,
    e.g. with an MLS stress term or an extra field, is about 13.7 KB).
  - Its cost is a fixed-point scale per channel. Choose it from the expected per-node maximum (estimate: mass is bounded
    by PPC·m·max weight, momentum by mass·v_max), or use two int32 words (hi/lo with carry) where range matters (estimate:
    about 2× atomic cost).
  - 64-bit integer add would be ideal, but it does not compile here (see the correction at the top).
- If float grids are needed (e.g. for gradients or simpler code), **V3** (sorted, SIMD segmented reduction, device
  `atomic_float`) is 0.50 ms/M and non-deterministic.
- Do not port Gao-style shared-memory float accumulation literally. On Apple it becomes a CAS loop, which is slower than
  doing nothing.
- Per-step estimate on the M4 Max (estimate, not measured):
  - P2G about 0.4–0.5 ms/M.
  - G2P is a 27-node gather with no atomics, about the V0 floor plus ALU: about 0.2–0.3 ms/M.
  - Stress, SVD and plasticity: about 0.1–0.3 ms/M. Gao 2018 reports GPU SVD at about 0.37 ns per particle on P6000.
  - Grid update: negligible. Clearing a 34 MB dense grid took 0.037 ms (measured).
  - Sort amortized every 10–30 steps (Fei 2021).
  - Total about **0.8–1.2 ms per million particles per step**. That is the same order as Wang 2020 on Quadro P6000
    (1.2–1.4 ms/M/step) and RTX 2080 (about 1.9 ms/M for 775K particles in the full pipeline).
- G2P2G: worth adding later for small environments (Fei 2021 used it below 100K particles per GPU). It cuts one particle
  read and write per step and launch count. The measured gain is only about 1.4× (Wang 2020), and it conflicts with
  cell-sorted warp reduction. Do V6/V7 first.

### Gaps
- No full-step Metal MPM measurement yet: only P2G was measured. G2P, stress and the full step above are estimates.
- Fig. 6/7 values in Gao 2018 were read off bar charts in extracted text; exact per-kernel ms were not tabulated in the text.
- Fei 2021 per-step ms on V100 (Fig. 1 bar values) were not extracted.
- Unresolved: why `fetch_add` on `atomic_ulong` fails to compile despite the Apple9 feature-set entry (possibly needs a
  newer SDK or other syntax).

---

## Q2. Grid choices: dense vs block-sparse, grid reset cost, batched multi-env layouts

### Takeaway
Use **dense fixed grids per environment** for bounded robot-learning workspaces. Clearing is negligible on Apple's
unified memory (34 MB in 0.037 ms), and dense grids remove the hashing and rebuild work that dominated small scenes in
Fei 2021. Reserve block-sparse (4×4×4 blocks, hashed, rebuilt every 10–30 steps) for 1–10M-particle open scenes. For
batching, put all environments in **one dispatch** over a combined, env-major buffer. A 262K-particle environment almost
saturates the GPU alone (0.48 vs 0.46 ms/M at 16.8M for V6); 10K–50K-particle environments do not.

### Cited findings
- Gao 2018: sparse GSPGrid. Memory is set to 60% of the uniform grid. A hash table of 64 entries per active block, keyed by
  64-bit block offsets, is used only for sorting and topology at the start of each step. [Gao 2018](https://people.csail.mit.edu/kuiwu/GPU_MPM/GPU_MPM.pdf)
- Fei 2021: "One can push to use more neighboring blocks than we do, and the extreme would end up with a dense background
  grid, where the rebuild mapping can be removed entirely." At 55.3K particles, "rebuild-mapping itself is the bottleneck".
  Untouched blocks are marked and skipped. [Fei 2021](https://arxiv.org/abs/2111.00699)
- Wang 2020: grid blocks of 4×4×4, spatially hashed. Particle blocks are allocated by count with "a lightweight exclusive
  scan" to map block id to bin id. [Wang 2020](https://yzhu.io/publication/mpmgpu2020siggraph/paper.pdf)
- Newton `SolverImplicitMPM` (Warp, current main) — [source](https://raw.githubusercontent.com/newton-physics/newton/main/newton/_src/solvers/implicit_mpm/solver_implicit_mpm.py)
  - `grid_type` is one of `"sparse"` (default, NanoVDB, rebuilt in place within a capacity), `"dense"` or `"fixed"`;
    default `voxel_size` 0.1.
  - Multi-world: "Multi-world models retain the shared FEM topology by default", i.e. one combined grid.
    `separate_worlds=True` gives each world its own FEM environment, with particles stored contiguously by world.
  - Rebuildable sparse grids: B2/B3 velocity bases reserve 64 nodes per active cell against 8 for Q1. Outer CUDA-graph
    capture needs a fixed or capacity-bounded sparse grid.
- ManiSkill2: custom Warp + CUDA MPM. One soft-body env with rendering ran at about 17–18 FPS; **16 parallel envs on one
  RTX Titan about 80–84 FPS total**. The authors say "a single CPU processor not able to submit CUDA kernels fast enough to
  keep up with the GPU", and vectorized simulation was left as future work. ManiSkill3 reuses the same soft-body
  environments. [ManiSkill2](https://arxiv.org/pdf/2302.04659), [ManiSkill3](https://arxiv.org/pdf/2410.00425)
- Measured here: clearing a dense 131³ × float4 grid (34 MB) took 0.037 ms; 67³ (4 MB) 0.006 ms; 35³ 0.003 ms. P2G per
  particle was about flat from 2.1M to 16.8M particles. At 0.26M it was the same or slightly worse (V6: 0.51 vs 0.48
  ms/M), so one dispatch of about 0.25M particles fills most of the 40-core GPU.
- Local context, measured in this repo on this machine: launch-count reduction paid off on Metal. Fusing 5 of 9 launches
  per solver iteration gave +3.9% env-steps/s for the 4096-env MuJoCo Warp fork. A SIMD-group sum of per-lane partials
  replaced atomic adds there. — `/Users/aditya/robosim/docs/research/throughput_2026-09-26.md` §2c

### Inferences
- Batched layout (recommended):
  - Particle buffer sorted env-major, then block, then cell. Grid as one buffer of per-env dense tiles: env offset plus
    local (x, y, z). With 64³ cells × float4 per env, that is 4 MB, so 1024 envs = 4 GB (fine in 64 GB unified memory).
  - Threadgroup→block maps are built once from a per-env block list. Every kernel is one dispatch over all envs, which
    avoids the ManiSkill2 per-env launch bottleneck.
  - Per-env domains are bounded, so there is no hashing.
  - Example: 512 envs × 50K particles = 25.6M particles, about 21–31 ms per step (estimate from the ~0.8–1.2 ms/M
    figure). Smaller envs: 4096 × 10K = 41M, about 33–49 ms (estimate).
- Skip empty blocks: dense storage plus an "active block" list (from the sort histogram) bounds grid update work by
  occupancy, without hashing.
- Single large scenes (1–10M particles):
  - A dense grid works up to about 256³ (268 MB at float4, plus extra channels). Beyond that, or with sparse occupancy, use
    4×4×4 blocks with a hash table built during the (infrequent) sort.
  - The 64 GB unified memory makes Gao's memory budgeting (2.4 GB for 7M particles) a non-issue.

### Gaps
- No published Genesis-on-Metal MPM numbers were found (the Genesis docs list a Metal backend, but no MPM throughput on
  Apple). [Genesis docs](https://genesis-world.readthedocs.io/en/v0.4.3/_sources/user_guide/advanced_topics/multi_gpu.md.txt)
- No ManiSkill3 batched-MPM throughput numbers were found.
- The combined-grid vs per-env-grid tradeoff was not measured on Metal.

---

## Q3. Precision: float32, half storage, mixed precision, fixed-point atomics for determinism

### Takeaway
Keep float32 compute everywhere: there is no float64 on Apple GPUs. Fixed-point int32 atomics are the best P2G path on
Metal (Q1), not a penalty, and give bitwise-reproducible P2G. Quantized or half particle storage saves about 1.7–2.2× memory
at about 1.04× slowdown (CUDA), but it is a memory optimization, not a speed one, for the particle counts here.

### Cited findings
- Measured here: fixed-point int32 P2G was bitwise identical across runs, and float atomics were not. Overhead:
  - Device-atomic path: none. Int32 device atomics ran at the same speed as float atomics (2.28 vs 2.30 ms/M sorted;
    2.85 vs 2.85 unsorted).
  - Threadgroup path: fixed point is 8–13× faster than the float equivalent, because threadgroup float atomics are
    CAS-emulated (V6 0.48 vs V4 5.54 ms/M).
- Accuracy at a 2^16 scale: 3.3e-4 absolute on values up to about 12 (measured here). Range limit ±32768 per channel.
- Fei 2021: fast-math intrinsics give negligible stress error (≤ 0.12 dyne/cm² over 60 frames, 3.1M sand particles), with
  no visible difference. [Fei 2021](https://arxiv.org/abs/2111.00699)
- QuanTaichi: 1.7× per-particle compression for MPM, "often run as fast as" full precision. [QuanTaichi](https://yuanming.taichi.graphics/publication/2021-quantaichi/)
- Automatic Quantization (2022): 2.15–2.21× MLS-MPM memory compression meeting a precision target; 1.04× slower on CUDA;
  dithering costs up to 8× on CPU. [arXiv 2207.04658](https://arxiv.org/pdf/2207.04658)
- PyTorch's Metal atomic header (secondary copy): "as of Metal 3.2" half-precision atomics are unsupported and emulated with
  CAS on 32-bit atomics. [c10/metal/atomic.h copy](https://generationalwealth.duckdns.org/venv/lib/python3.13/site-packages/torch/include/c10/metal/atomic.h)
- Philip Turner (M1/M2): Apple GPUs "only support 32-bit atomics on pointer values"; M2 (Apple8) added non-returning UInt64
  min/max. [metal-benchmarks](https://github.com/philipturner/metal-benchmarks) This is consistent with the M4 Max compile
  tests here (64-bit max OK; 64-bit add and CAS not available).

### Inferences
- Determinism recipe:
  - Fixed-point int32 grid accumulation with per-channel scale.
  - Deterministic stable sort: the repo's splat sort is already bit-identical and stable
    (`/Users/aditya/robosim/metalsim/render/SPLAT.md` "Determinism").
  - No float atomics anywhere else (SIMD-group reductions have a fixed order).
  - G2P and particle updates are gathers, so they are deterministic already.
  - Result: bitwise-reproducible MPM steps at no speed cost (measured for P2G).
- Range safety: for stiff materials, momentum plus the stress term (MLS: −dt·V·P·Fᵀ·D⁻¹·dpos) can be large. Options are a
  scale per channel per env, or splitting stress into a separate channel (estimate).
- Half storage for v, C and F is plausible for rendering-only particle sets. For dynamics, the literature reports only
  memory gains; there is no measured energy-drift data for half-precision MPM state (gap).

### Gaps
- No published measurements of energy drift or accuracy for mixed-precision (half-storage) MPM on any GPU were found.
- No published Metal-specific fixed-point MPM overhead figures were found; the only numbers are those measured here.
- WebGPU MPM demos reportedly use fixed-point integer atomics (WGSL has no float atomics), but the Codrops article
  returned 403 and this could not be verified.

---

## Q4. Metal-specific facts: atomics, SIMD-group ops, threadgroup memory, ICBs, sort performance

### Takeaway
On the M4 Max:
- Device float atomics are native but slow under the 27-node scatter: about 47 G adds/s sorted (measured).
- Threadgroup integer atomics are fast: 224–326 G effective adds/s in P2G (measured).
- SIMD-group shuffles cost about 2 cycles and `simd_sum` about 14.5 cycles (M1/M2, Turner).
- Threadgroup memory is limited to 32 KB per threadgroup.
- The repo's own Metal radix sort runs at about 3 G key-value pairs/s (5 passes of 8 bits).
- forge-sort self-reports 4.1 Gkeys/s on an M4 Pro. For comparison, Onesweep reports 29.4 Gkeys/s on an A100.

### Cited findings
- Feature tables:
  - Floating-point atomics: Apple7+.
  - SIMD-scoped reduction: Apple7+. SIMD-scoped permute: Apple6+. SIMD shift and fill: Apple8+.
  - 64-bit atomics: Apple9 ("full set ... starting with Apple9"; Apple8 min/max only, macOS).
  - Maximum threadgroup memory 32 KB for Apple families.

  — [Metal feature set tables](https://developer.apple.com/metal/limits) (PDF text extracted)
- Compile tests here (M4 Max):
  - Compiles: device `atomic_float` add; threadgroup `atomic_uint` add; device `atomic_ulong` max.
  - Does not compile (MSL 4.0): threadgroup `atomic_float` (MSL 3.0–4.0), device `atomic_ulong`/`atomic<long>` add,
    `atomic_ulong` CAS and load/store, threadgroup `atomic_ulong`.
- Philip Turner metal-benchmarks (Apple7/8 = M1/M2): [repo](https://github.com/philipturner/metal-benchmarks)
  - Threadgroup memory about 60 KB per core; register file about 208 KB per core; L1 data 8 KB per core.
  - Shuffle bandwidth 256 B/cycle per core.
  - Cycles: ballot/broadcast/shuffle about 2.0; SUM<F32> 14.54; PREFIX_ESUM<F32> 11.57; SHUFFLE_RANDOM 28–32;
    simdgroup 8×8 F32 matmul about 18.
  - Global cache line 128 B.
  - No atomic-throughput numbers in the repo.
- Apple WWDC22 10159: global atomics serialize under contention; reduce SIMD, then threadgroup, then device; 1K–2K threads
  per core for occupancy; consider indirect command buffers for GPU-driven encoding; replace `waitUntilCompleted` with
  `MTLSharedEvent`. [WWDC22](https://developer.apple.com/videos/play/wwdc2022/10159)
- An Apple developer forum thread (2026) reports that relaxed threadgroup integer atomics on M5 gave "a meaningful speedup
  over a scan-based fallback" in histogram/radix-sort kernels (integer, not float). [forum 841598](https://developer.apple.com/forums/thread/841598)
- Measured here: device atomic throughput, effective float adds/s in the P2G pattern on the M4 Max:
  - About 38 G/s unsorted and 47 G/s sorted.
  - The non-atomic floor of the same kernel is about 630 G/s, so device atomics cost about 13× the non-atomic RMW.
  - For comparison (different workload and hardware): Gao 2018 measured atomic-only P2G at 23× slower than their optimized
    P2G on TITAN Xp. On the M4 Max the optimized-vs-naive ratio is 5.7–8.7× (V1 vs V3/V6/V7).
- Sort on this M4 Max, measured in-repo (`/Users/aditya/robosim/metalsim/render/SPLAT.md` §5):
  - Stable 8-bit LSD radix sort, 5 passes, 32-bit keys + 32-bit values, simd-ballot multisplit ranking.
  - 20.9M keys in 6.79 ms (≈3.1 Gkeys/s); 133.9M keys in 46.47 ms (≈2.9 Gkeys/s); 11.4M keys in 3.7 ms (≈3.1 Gkeys/s);
    0.44M keys in 0.2 ms (≈2.2 Gkeys/s, launch-bound).
  - A single-pass Onesweep with a local reorder before the scatter is listed there as the next optimization (not yet done).
- forge-sort (Rust, Metal, self-reported, M4 Pro): 3,432 Mkeys/s at 1M and 4,131 Mkeys/s at 16M u32 keys with zero-copy
  input; 836 and 2,784 Mkeys/s via plain `sort_u32`. MSD+LSD hybrid in 4 dispatches. Notes that MPS has no sort primitive.
  [forge-sort](https://docs.rs/forge-sort)
- Onesweep (NVIDIA A100): 29.4 Gkeys/s for 256M 32-bit keys, about 1.5× CUB. [Onesweep](https://arxiv.org/abs/2206.01784v1)

### Inferences
- MPM sort cost on the M4 Max (estimate from the in-repo sort rate):
  - A full 32-bit cell-key sort of 1M particles takes about 0.33 ms (5 passes), which is comparable to the P2G cost itself.
    So follow Gao and Fei: sort rarely.
  - Better, a **counting sort** with per-env/per-block keys of ≤ 16 bits (for example, a 64³ env has 4096 blocks × 64 cells
    = 18 bits; block id alone 12 bits) is 1–2 passes, about 0.1 ms/M (estimate).
  - Full re-sort only when particles leave a block's free zone (Fei: every 10–30 steps).
  - In between: an in-SIMD-group 32-lane sort or ballot-based match on the base-node key before reduction (Fei's 10-bit
    in-warp radix sort maps directly to `simd_ballot` / `simd_prefix_exclusive_sum` multisplit, which the repo's splat sort
    already uses).
- Command encoding: with dense per-env grids and fixed capacities, a whole MPM substep is a fixed sequence of about 3–5
  compute dispatches, encoded once per step in one command buffer. Indirect command buffers or indirect dispatch help only
  when particle counts change (sparse grids); this is optional.

### Gaps
- No public microbenchmark of Apple GPU atomic throughput against NVIDIA was found. Only the P2G-pattern measurements here
  exist.
- No published MPS sort exists (forge-sort says MPS has no sort primitive). No Metal port of FidelityFX Parallel Sort with
  published numbers was found.
- No Metal Onesweep numbers.

---

## Q5. Existing MPM implementations on Metal / Apple Silicon and their throughput

### Takeaway
I found **no published particle-step throughput for any MPM on Apple Silicon GPUs**: not Taichi Metal, Genesis on Mac,
MLX, Apple sample code or WebGPU demos on Macs. The P2G numbers measured here (0.33–0.5 ms per million particles on the M4
Max) appear to be the only Metal MPM kernel timings available.

### Cited findings
- Taichi has a Metal backend (since v0.5.0; initially dense SNodes and 32-bit types only). Its official benchmark repo
  reports RTX 3080 numbers only and is "not particularly tuned". [taichi_benchmark](https://github.com/taichi-dev/taichi_benchmark)
- Genesis lists a Metal backend; no MPM-on-Metal throughput was found. [Genesis docs](https://genesis-world.readthedocs.io/en/v0.4.3/_sources/user_guide/advanced_topics/multi_gpu.md.txt)
- WebGPU MLS-MPM:
  - Codrops (2025): "real-time simulation of 100,000 particles on an iGPU" on an AMD Ryzen 7 5825U, not Apple. [Codrops](https://tympanus.net/codrops/2025/02/26/webgpu-fluid-simulations-high-performance-real-time-rendering/)
  - A three.js forum MLS-MPM installation was tested on a MacBook Pro, with no fps given. [three.js forum](https://discourse.threejs.org/t/mls-mpm-simulation-with-webgpu-and-threejs/77686)
  - luma.gl's experimental 2D WebGPU MLS-MPM is capped at 65,536 particles. [luma.gl](https://luma.gl/docs/api-reference/experimental/mls-mpm-fluid-simulation)
  - WebGPUMPM (GitHub) claims 200K particles in real time (Chrome Canary era). [WebGPUMPM](https://github.com/chetan-parthiban/WebGPUMPM)
- MLX: no MPM implementation found.

### Inferences
- Since no Metal MPM baseline exists, the plan should treat this repo's own kernels as the reference. The P2G
  microbenchmark here is a starting point for a full-step benchmark.

### Gaps
- Taichi `mpm88` / `mpm3d` on `ti.metal` on the M4 Max could be run locally for a baseline; not done here.
- No Apple-published MPM sample code was found.

---

## Q6. Implicit MPM (Newton ImplicitMPM) vs explicit MLS-MPM: costs and when implicit wins

### Takeaway
Newton's `SolverImplicitMPM` follows Daviet & Bertails-Descoubes 2016 (semi-implicit, nonsmooth granular rheology). By
default it uses trilinear (Q1) velocity nodes (8 per particle rather than 27), a P0 strain basis, APIC transfer, a sparse
NanoVDB grid, and a Gauss-Seidel rheology solve capped at 250 iterations with tolerance 1e-4. Implicit wins when the
explicit elastic CFL limit (Δt ∝ Δx/√(E/ρ)) forces many substeps per control step (stiff solids, fine grids), or for rigid
or plastic granular flow where the yield-surface projection is the physics. Explicit MLS-MPM wins for soft materials at
coarse Δx, where only a few substeps per control step are needed and each step costs about one P2G + G2P.

### Cited findings
- Newton `SolverImplicitMPM` — [source](https://raw.githubusercontent.com/newton-physics/newton/main/newton/_src/solvers/implicit_mpm/solver_implicit_mpm.py)
  - References DOI 10.1145/2897824.2925877 (Daviet & Bertails-Descoubes, SIGGRAPH 2016).
  - Config: `max_iterations` 250, `tolerance` 1e-4, `transfer_scheme` "apic", `velocity_basis` "Q1" (or "B2"/"B3"),
    `strain_basis` "P0", `collider_basis` "S2", `grid_type` "sparse".
  - Built on `warp.fem` / `warp.sparse`. Colliders are rasterized to the grid.
- Isaac Lab wrapper: the `"auto"` solver picks Gauss-Seidel for Q1 and batched GS for B2/B3. Jacobi, CG/CR and GMRES are
  available, and a tuple runs a warm-start sequence (e.g. `("cr","gs")`). Colliders are resolved at grid level, with an
  optional `project_outside` Coulomb push-out. [Isaac Lab mpm_manager_cfg](https://isaac-sim.github.io/IsaacLab/release/3.0.0/_modules/isaaclab_newton/physics/mpm_manager_cfg.html)
- Newton v1.4.0: inactive MPM particles no longer take part in grid transfers. [changelog](https://data.safetycli.com/packages/pypi/newton/changelog)
- Gao 2018 (implicit Krylov, matrix-free) — [Gao 2018](https://people.csail.mit.edu/kuiwu/GPU_MPM/GPU_MPM.pdf)
  - "each iteration consists of a call to the P2G kernel and a call to the G2P kernel".
  - Solver time split, P6000: G2P 11–13%, gradient and SVD 13–37%, P2G 40–45%, others 7–28%.
  - Explicit at Δt = 1e-4 vs implicit at Δt = 1e-3 for colliding dragons: "the implicit solver spends more time per step
    ... but also can use larger time step."
  - Production implicit runs spent 66–87% of frame time in the solver (Table 1, e.g. 13.94 of 20.95 s).
- A 2025 review: explicit MPM suits GPU parallelism; implicit "remains computationally expensive, especially when coupling
  MPM with rigid bodies for frictional contact". [arXiv 2503.05046](https://arxiv.org/html/2503.05046v2)
- Yu, Du et al. (convex MPM-rigid coupling, quasi-Newton): up to 500× faster than prior convex formulations. [arXiv 2503.05046](https://arxiv.org/html/2503.05046v2)

### Inferences
- Cost model (estimate): one implicit Krylov iteration costs about one P2G + one G2P (Gao 2018). With the Metal P2G at
  about 0.4–0.5 ms/M and G2P at about 0.2–0.3 ms/M, an iteration is about 0.6–0.8 ms/M. Implicit is cheaper only when
  (iterations × 0.7) < (explicit substeps saved × 1.0 ms/M).
- Example CFL, dt ≈ 0.3·Δx/c, with c = √(E/ρ) and ρ = 1000 kg/m³ (estimate):
  - E = 1e5 Pa: c = 10 m/s; at Δx = 1 cm, dt = 3e-4 s, about 67 explicit substeps per 20 ms control step.
  - E = 1e7 Pa: c = 100 m/s, about 670 substeps.
  - Implicit at 1–2 substeps × 20–50 iterations ≈ 20–100 P2G/G2P pairs.

  So implicit wins for stiff solids (E ≳ 1e6–1e7), and explicit wins for soft or granular materials at coarse Δx in
  short-horizon RL.
- Newton's Q1 basis halves or better the scatter work per particle (8 vs 27 nodes). Its Gauss-Seidel on the grid needs
  coloring (the code has `compute_color_offsets`) and is less parallel than Jacobi/CG on a GPU. On Metal, colored GS fits
  one threadgroup-local pass per color (estimate). An explicit MLS-MPM with fixed-point P2G is the simpler first target;
  matching Newton's implicit solver is a later parity item.
- Nonsmooth granular (Drucker–Prager) rheology is cheap per node (a local projection), so iteration cost is dominated by
  the P2G/G2P-like operators. The fixed-point threadgroup tile helps implicit iterations just as it helps explicit steps.

### Gaps
- No published Newton ImplicitMPM timings (per step, per particle, per iteration count) were found.
- The Daviet & Bertails-Descoubes 2016 paper text and the HAL preprint (hal-05070887) could not be retrieved (bot
  protection), so their GS iteration counts and timings are missing.
- Typical iteration counts actually reached for robot-manipulation scenes in Newton are unknown.

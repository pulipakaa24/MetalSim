# Scalable run-to-run deterministic float atomics and reductions on GPUs (batched robot sim, thousands of worlds)

Researched 2026-10-10. Versions are cited inline. "Local measurement" means a probe I ran on this machine (Apple M4 Max, 40-core GPU, macOS Darwin 25.6, Metal runtime compile through Swift `MTLDevice.makeLibrary(source:)`). The probe sources are in the session scratchpad, not in the repo.

## Q1. NVIDIA Warp deterministic mode: mechanism, coverage, overhead, version

### Takeaway
Warp added `wp.config.deterministic` / `wp.DeterministicMode` in **1.15.0 (2026-07-07)**, not 1.10–1.14. It is a compile-time rewrite with two paths. Accumulation atomics whose return value is unused become **scatter-to-buffer, then a CUB radix sort by (dest, thread id), then reduce-by-key**. Atomics whose return value is consumed (counters) become a **two-pass count, sort, segmented-scan, re-execute**. Under high contention the sort path is *faster* than native float atomics (0.13x time at 1M writes into 1 output). Under low contention it is 10–24x slower, because the sort dominates (RTX 4090 numbers).

### Cited Findings
- **Version and API.** Introduced in Warp 1.15.0 (2026-07-07): "Add a deterministic execution mode for supported atomic operations ... `wp.config.deterministic` or per module via the `"deterministic"` module option, using `wp.DeterministicMode.RUN_TO_RUN` for bit-exact repeatable results on the same GPU, or `wp.DeterministicMode.GPU_TO_GPU` to also target consistency across GPU architectures." It covers accumulation into array elements and counters that hand out slots, and it extends to autodiff backward reductions. Also new: `wp.config.deterministic_max_records` (GH-1443). — [Warp CHANGELOG 1.15.0](https://github.com/NVIDIA/warp/blob/main/CHANGELOG.md)
- 1.16.0 (2026-08-03) fixed "kernel launches using stale deterministic launch metadata after switching a module's `deterministic` mode, which could execute consumed-return counter atomics twice or crash" (GH-1637). The 1.17.0 (2026-08-31) and 1.18.0 (2026-10-05) changelogs have no further determinism entries. — [Warp CHANGELOG](https://github.com/NVIDIA/warp/blob/main/CHANGELOG.md)
- The original implementation PR is NVIDIA/warp#1355 by mmacklin. — [Newton issue #2479 comment by eric-heiden](https://github.com/newton-physics/newton/issues/2479)
- **Modes.** `NOT_GUARANTEED` is the default. `RUN_TO_RUN` is "bit-exact reproducibility across repeated runs on the same GPU architecture". `GPU_TO_GPU` is the "stronger reduction path intended to preserve identical results across GPU architectures". — [Warp design doc deterministic-execution.md](https://github.com/NVIDIA/warp/blob/main/design/deterministic-execution.md)
- **Ops covered:** `atomic_add`, `atomic_sub`, `atomic_min`, `atomic_max`.
  - Float add/sub/min/max with an unused return value go through scatter-sort-reduce. This includes vec/mat/quat/transform leaves.
  - Integer atomics with an unused return value are left as native atomics ("already deterministic").
  - Bitwise and/or/xor are allowed only as bare statements.
  - Not covered: `atomic_cas`, `atomic_exch`, and tile atomics (`tile_atomic_add`).
  - The counter path (return value consumed) supports **only `atomic_add` on `int32` counters**.
  — [Warp design doc](https://github.com/NVIDIA/warp/blob/main/design/deterministic-execution.md)
- **Pattern A mechanism (scatter-sort-reduce).**
  - Each thread writes a `(sort_key, value)` record, where the key packs `(dest_index << 32 | thread_id)`, into a temporary buffer.
  - After the kernel, CUB radix sort orders the records.
  - RUN_TO_RUN then uses CUB `DeviceReduce::ReduceByKey`. GPU_TO_GPU uses Warp's own sequential segmented kernel, so the left-to-right order is explicit. Composite types always use the segmented kernel.
  - Buffers have a minimum capacity of 1024 records per target. Capacity is a static per-thread lower bound from codegen, or `deterministic_max_records` if that is larger. Records beyond capacity are **truncated** and set an overflow flag. Overflow is not checked under CUDA graph capture.
  - Under graph capture the **full fixed-capacity buffer is sorted**, padded with sentinel keys, because CUB needs a host-side count.
  - Limit: 2^31−1 records.
  — [Warp design doc](https://github.com/NVIDIA/warp/blob/main/design/deterministic-execution.md)
- **Pattern B mechanism (counters).** "The kernel runs twice":
  - Phase 0 records `(counter_flat_index, deterministic_record_order, value)` and suppresses ordinary global stores.
  - Warp then snapshots the counter base, sorts, and runs a segmented prefix scan.
  - Phase 1 re-executes the kernel, and each counter atomic returns the precomputed deterministic offset.
  - Finally the counter totals are written back.

  The design doc says this "mirrors the well-established count-scan-write pattern" that Warp's marching cubes already uses. — [Warp design doc](https://github.com/NVIDIA/warp/blob/main/design/deterministic-execution.md)
- **Alternatives Warp rejected, and why:**
  - "Fixed-point integer accumulation | Loses precision/range; not general for all float types"
  - "Per-thread output arrays | O(threads * output_size) memory; doesn't scale"
  - "Serialized atomics (mutex) | Prohibitively slow on GPU"
  - "Kahan compensated summation | ... does not guarantee determinism"

  — [Warp design doc](https://github.com/NVIDIA/warp/blob/main/design/deterministic-execution.md)
- **Measured overhead, Pattern A.** RTX 4090, CUDA 12.4, median CUDA-event ms, graph replay of zeroing plus the kernel. — [Warp docs, deterministic_execution.rst "Performance"](https://github.com/NVIDIA/warp/blob/main/docs/user_guide/execution_and_performance/deterministic_execution.rst)

  | writes | outputs | native | RUN_TO_RUN | GPU_TO_GPU | R2R ratio |
  |---|---|---|---|---|---|
  | 65,536 | 1 | 0.099 | 0.075 | 7.66 | 0.76x |
  | 262,144 | 1 | 0.364 | 0.096 | 31.0 | 0.26x |
  | 1,048,576 | 1 | 1.46 | 0.193 | 172 | 0.13x |
  | 65,536 | 65,536 | 0.0058 | 0.079 | 0.076 | 13.6x |
  | 262,144 | 65,536 | 0.0081 | 0.192 | 0.187 | 23.9x |
  | 1,048,576 | 65,536 | 0.020 | 0.201 | 0.228 | 10.0x |

- **Measured overhead, Pattern B** (slot counter, direct launches): 65,536 writes went from 0.020 to 0.129 ms (6.3x), 262,144 from 0.024 to 0.144 ms (6.0x), and 1,048,576 from 0.022 to 0.165 ms (7.6x). The docs note that "Application wall time can be higher because this direct-launch path still allocates temporary prefix-scan buffers." — [Warp docs](https://github.com/NVIDIA/warp/blob/main/docs/user_guide/execution_and_performance/deterministic_execution.rst)
- The Warp docs also show a manual alternative: "give each output element one thread and visit the inputs in a fixed order" (a gather). They call it O(num_bins × num_values) and suitable for debugging or a small number of bins, and they recommend an explicit count-scan-write for slot allocation. — [Warp docs](https://github.com/NVIDIA/warp/blob/main/docs/user_guide/execution_and_performance/deterministic_execution.rst)
- **Known failure modes in practice:**
  - Phase 0 store suppression made MuJoCo Warp's convex narrowphase **silently produce zero contacts** under RUN_TO_RUN (warp-lang 1.15/1.16, mujoco-warp 3.11). Raising `deterministic_max_records` turned the loud overflow error into the silent failure. — [mujoco_warp #1635](https://github.com/google-deepmind/mujoco_warp/issues/1635)
  - "One reduction family per array" blocks kernels that mix `atomic_max` and `atomic_add` on disjoint ranges of the same array. — [mujoco_warp #1590](https://github.com/google-deepmind/mujoco_warp/issues/1590)
  - Scatter-buffer overflow occurs in data-dependent loops. — [Newton #4116](https://github.com/newton-physics/newton/issues/4116)
  - Mixing determinism settings with a **warm kernel cache** silently reused non-deterministic modules. — [mujoco_warp #1590](https://github.com/google-deepmind/mujoco_warp/issues/1590)

### Inferences
- The sort-based path costs roughly a constant 0.08–0.2 ms per deterministic kernel launch on a 4090 at these sizes, mostly radix sort over records plus temporary buffers. It wins only where native atomics serialize on a hot address. For a batched simulator with many deterministic scatter kernels per step, the per-launch cost adds up. That matches the 30–80% throughput hit MuJoCo Warp saw with a blanket rewrite (Q2).
- Warp's own guidance and MJW experience both point to the same conclusion: hand-written **gathers / count-scan-emit** for hot paths, with sort-reduce only as a fallback.

### Gaps
- I found no published Warp overhead numbers on other GPUs (H100/Blackwell) or for realistic multi-kernel physics steps.
- `asv/benchmarks/deterministic.py` exists in the repo, but I did not find published results from it.

## Q2. MuJoCo Warp, Newton, Isaac Lab / PhysX determinism guarantees and cost

### Takeaway
- **MuJoCo Warp:** the released version (3.15.0, 2026-10-05) is *not* GPU-deterministic. The official answer is "run on CPU".
  - An opt-in `opt.deterministic` is in open PRs. #1422/#1425 use contact sort, count-scan-emit row allocation and gathers. #1687 is the official API; #1775 integrates the two.
  - Measured cost of #1425: about **−36% throughput, roughly flat in world count** (from 1 to 8192 worlds). The earlier version cost −31% to −81%.
- **Newton:** per-solver `deterministic` flags since 1.4.0. The collision pipeline uses a sort-based `deterministic` since 1.2.0, and hydroelastic contacts use fixed-point accumulation since 1.5.0.
- **Isaac Lab 3.0 / PhysX:** rigid bodies and articulations are deterministic on the same hardware and version. Enhanced determinism costs performance but publishes no numbers. Non-rigid bodies are not guaranteed.

### Cited Findings
- **MJW docs:** "Is MJWarp on GPU deterministic? No. There may be ordering or *small* numerical differences ... characteristic of non-deterministic atomic operations on GPU. Set device to CPU with `wp.set_device("cpu")` for deterministic results." — [MuJoCo Warp docs](https://mujoco.readthedocs.io/en/latest/mjwarp/index.html)
- MJW tracking issue #562 (opened 2025-07-31) is still open. A maintainer reply on 2025-12-10 said "for now, the requirement for deterministic results is running mjwarp on cpu." — [mujoco_warp #562](https://github.com/google-deepmind/mujoco_warp/issues/562)
- **Warp's global switch does not work on MJW.**
  - With warp 1.16.0 and mujoco_warp 3.11.0, codegen fails in `_sensor_tactile` (mixed max/add). — [#562 comment, 2026-08-27](https://github.com/google-deepmind/mujoco_warp/issues/562)
  - The convex narrowphase silently drops contacts. A maintainer said "mujoco warp currently does not support warp's determinism functionality" (2026-09-10). — [#1635](https://github.com/google-deepmind/mujoco_warp/issues/1635)
- The MJW release notes for v3.13.0, v3.14.0 and v3.15.0 contain no mention of determinism (checked via `gh release view`). The `types.py` on main has no `deterministic` field yet. — [mujoco_warp releases](https://github.com/google-deepmind/mujoco_warp/releases)
- **PR #1422** (open, 2026-06-11), "Deterministic simulation":
  - contacts are "sorted into a stable permutation after narrowphase";
  - constraint row allocation is moved "from atomic slot grabbing into a count-scan-emit pattern";
  - the sparse solver's atomic accumulations (qfrc_constraint and JTDAJ) are "replaced with deterministic gathers using a per-solve column index built once per step";
  - with the flag off, kernels compile to the original atomic code via `wp.static` dispatch;
  - deterministic mode "costs roughly 30-80% of step throughput depending on world count."

  — [mujoco_warp PR #1422](https://github.com/google-deepmind/mujoco_warp/pull/1422)
- **PR #1425** (open): the JTDAJ gather uses a "two-pointer merge-intersection", and the contact world-boundary scan uses one thread per contact instead of O(nworld·nacon). Incremental Newton keeps change flags per efcid "rather than a nondeterministically-ordered atomic-append list". Throughput on three_humanoids, graph captured, RTX PRO 6000 Blackwell, in steps/s: — [mujoco_warp PR #1425](https://github.com/google-deepmind/mujoco_warp/pull/1425)

  | nworld | det=off | det=on (#1422) | det=on (#1425) | overhead |
  |---|---|---|---|---|
  | 1 | 1519 | 1042 | 1307 | −31% → −14% |
  | 256 | 205k | 100k | 134k | −51% → −34% |
  | 1024 | 664k | 199k | 426k | −70% → −36% |
  | 8192 | 1395k | 259k | 895k | −81% → −36% |

  "The overhead is now roughly flat in world count."
- **PR #1533** (open) extends determinism to the full pipeline with "children gathers over a precomputed body-children CSR walked level by level". The sparse LDL "gather[s] per row slot in ascending source order". Energy "reduces serially per world". The PR says "the gathers serialize work the atomic path spreads across threads" and gives no throughput numbers. — [mujoco_warp PR #1533](https://github.com/google-deepmind/mujoco_warp/pull/1533)
- **PR #1687** ("determinism api", by maintainer thowell, open since 2026-09-22) adds the `DeterminismType` IntFlag on `m.opt.deterministic`: `NONE=0`, `CONTACTS`, `CONSTRAINT`, `ATOMICS`, `ISLANDS`, `ALL=15`. **PR #1775** (2026-10-10) integrates it with the #1422 ordering and reports that "deferred reductions cannot substitute for writes consumed by later substitution levels". — [PR #1687](https://github.com/google-deepmind/mujoco_warp/pull/1687), [PR #1775](https://github.com/google-deepmind/mujoco_warp/pull/1775)
- **Evidence of batch-size dependence** (MJW 3.12.0, Warp 1.17.0, RTX PRO 6000):
  - With `opt.deterministic=False`, nine identical replica worlds "diverge at the first substep". The first-step max qacc difference is 0.00198, and after 80 substeps the max qpos difference is 0.086.
  - With #1533, replicas are bit-identical, and 1-, 9- and 144-world runs give bit-identical trajectories.
  - The determinism option must be set before the model-constant precompute (`body_invweight0` and similar), otherwise those constants already differ per world.

  — [#562 comment 2026-10-10](https://github.com/google-deepmind/mujoco_warp/issues/562); [Newton #4631](https://github.com/newton-physics/newton/issues/4631)
- **Newton:**
  - 1.2.0 (2026-05-12) added a "`deterministic` flag to `CollisionPipeline` and `NarrowPhase` for GPU-thread-scheduling-independent contact ordering via radix sort and deterministic fingerprint tiebreaking in contact reduction".
  - 1.4.0 (2026-07-16) added a "`deterministic` constructor argument to `SolverXPBD`, `SolverSemiImplicit`, `SolverFeatherstone`, `SolverVBD`, and `SolverMuJoCo` ... the default inherits `wp.config.deterministic`" (#3300).
  - 1.5.0 (2026-08-11) made hydroelastic contacts deterministic, with "overflow-safe fixed-point pressure accumulation" (#3661).
  - The latest release is v1.6.1 (2026-10-05), with v1.7.0rc1 on 2026-10-08.

  — [Newton CHANGELOG](https://github.com/newton-physics/newton/blob/main/CHANGELOG.md)
- Newton's `CollisionPipeline` docstring says `deterministic` will "Sort contacts after the narrow phase so that results are independent of GPU thread scheduling ... Adds a radix sort + gather pass." With deterministic reduction, `max_triangle_pairs` is "limited to 2**20 - 1" because reducer indices are packed into `CONTACT_ID_BITS`. The contact sort runs over the full preallocated buffer, with sentinel keys, so that CUDA graph capture works. — [newton/_src/sim/collide.py](https://github.com/newton-physics/newton/blob/main/newton/_src/sim/collide.py), [contact_sort.py](https://github.com/newton-physics/newton/blob/main/newton/_src/geometry/contact_sort.py)
- In Newton #3561, a developer (Gilles) is quoted: "MPM as a standalone solver should be deterministic by default (at least it used to, I have not run determinism tests recently)". — [Newton #3561](https://github.com/newton-physics/newton/issues/3561)
- **Isaac Lab 3.0 docs:**
  - "Same hardware and Isaac Sim version: the simulation produces identical results for scenes with rigid bodies and articulations." Results "can vary across different hardware". "PhysX does not guarantee determinism for any scene with non-rigid bodies, such as cloth or soft bodies."
  - "enhanced determinism ... is best-effort on OvPhysX and not verified end to end."
  - The `--deterministic` flag sets `warp.config.deterministic=RUN_TO_RUN`.
  - On the Newton backend, determinism is supported for MJWarp with `disable_sensors=True`, XPBD and Featherstone. An unsupported solver raises an error.
  - "Deterministic contact ordering adds sorting work and allocates buffers sized for the configured maximum contact count ... overhead therefore grow[s] with contact capacity."
  - Runtime domain randomization of materials can break GPU determinism.

  — [Isaac Lab 3.0 Reproducibility and Determinism](https://isaac-sim.github.io/IsaacLab/release/3.0.0/source/features/reproducibility.html)
- In Isaac Lab 2.x, `enable_enhanced_determinism` is described as "Enable/disable improved determinism at the expense of performance. Defaults to False", with a recommendation to enable it only for large steps (under 30 Hz) and a warning that it can give incorrect contact-sensor forces. — [Isaac Lab 2.x simulation_cfg source](https://isaac-sim.github.io/IsaacLab/v2.3.0/_modules/isaaclab/sim/simulation_cfg.html)
- **PhysX 5.4.1 SDK:**
  - "limited deterministic simulation": results are identical given the same scene, the "same actors inserted in the same order", the same time-stepping, and the same PhysX release on the same platform.
  - Unrelated actors can change results.
  - `PxSceneFlag::eENABLE_ENHANCED_DETERMINISM` makes an island's simulation "identical regardless of any other islands in the scene", and "sacrifices some performance". No numbers are given.

  — [PhysX 5.4.1 Rigid Body Dynamics](https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/docs/RigidBodyDynamics.html)

### Inferences
- A blanket "make every atomic deterministic" approach (Warp global mode) does not compose well with a full physics pipeline. The working CUDA reference designs instead combine three things:
  1. canonical sort of the contacts / append buffers;
  2. count-scan-emit for row/slot allocation;
  3. per-destination **gathers** over a precomputed CSR / transpose index, accumulated in ascending source index.
- With that combination, overhead becomes roughly constant in world count (about −36%). This is the most relevant precedent for replacing "serialize atomics into one SIMD-group" on Metal.
- Batch-size invariance (results independent of nworld and neighbouring worlds) is a stronger property than run-to-run. Gathers keyed per world give it for free. Global-index sort keys also give it if the key starts with the world id.

### Gaps
- I could not find published PhysX GPU enhanced-determinism overhead numbers, or a statement of whether it applies to GPU dynamics. The PhysX doc text did not say.
- I could not verify how MJW #1687/#1775 perform; they are still open and unbenchmarked in their descriptions.

## Q3. Other techniques and their cost at scale

### Takeaway
There are five families:
1. **Fixed-point integer atomics.** Deterministic for free, because integer add is associative. Used by AMBER SPFP, Tinker-HP, Newton hydroelastic. Needs 64-bit integer add to have enough range, and that is unavailable on Metal (Q4).
2. **Fixed-order tree reduction.** CUB RUN_TO_RUN uses fixed hierarchical trees with no atomics, at only slightly more cost than atomic-based reduction.
3. **Binned / reproducible accumulators** (ReproBLAS, CCCL GPU_TO_GPU RFA). +20–30% at large N on GPU, about 4x on CPU. Order-independent, so they could even be used with atomics if the bins were integers.
4. **Sort then segmented reduce.** Warp: about 0.08–0.2 ms per launch on a 4090.
5. **Count-scan-write instead of append counters.** Warp: about 6–8x a native atomic counter for an isolated kernel, still about 0.15 ms.

### Cited Findings
- **CCCL determinism levels** (CUB single-phase API with execution environment, **CCCL 3.1**):
  - `not_guaranteed` uses atomics for the block partials and the final value, in a single kernel.
  - `run_to_run` (the default) uses "a fixed hierarchical tree instead of atomics": per-thread, then warp shuffles, then shared memory, then a second kernel.
  - `gpu_to_gpu` uses the Reproducible Floating-point Accumulator (RFA) with 3 exponent bins by default, "increasing execution time by 20% to 30% for large problem sizes" (H200, per the figure).
  - Atomics are the root cause: "unordered execution across threads results in a different order of operations between runs".

  — [NVIDIA blog: Controlling Floating-Point Determinism in NVIDIA CCCL](https://developer.nvidia.com/blog/controlling-floating-point-determinism-in-nvidia-cccl/)
- **AMBER SPFP:** "combination of single and fixed precision", with "fixed scaled integer precision for all accumulations". "integer fixed-precision accumulation ensures that its simulations are self-consistent". DPFP gives "reproducible results thanks to the integer accumulation." Citation: Le Grand, Götz, Walker, "SPFP: Speed without compromise", Comp. Phys. Comm. 184 (2013) 374–380, doi:10.1016/j.cpc.2012.09.022. — [AMBER GPU logistics page](https://ambermd.org/GPULogistics.php)
- **Tinker-HP** (GPU, inspired by Walker/Götz) uses 64-bit fixed-point force accumulation with 34 fractional and 30 integer bits. "each single-precision pairwise contribution is small enough to be efficiently captured by 64 bits fixed point", which offers "deterministic operations". — [Tinker-HP GPU paper, arXiv:2011.01207](https://arxiv.org/pdf/2011.01207)
- **ReproBLAS** (Ahrens, Nguyen, Demmel, binned summation):
  - single pass, with "7n floating point additions to sum n words" using a 6-word accumulator;
  - a 4096-element double dot product runs "4x slowdown compared to ... MKL" on a Core i7-2600;
  - the error bound can be "up to 10^-8 times smaller" than conventional summation.

  — [Efficient Reproducible Floating Point Summation and BLAS, UCB EECS-2015-229](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2015/EECS-2015-229.pdf)
- **GROMACS:**
  - Developers attribute GPU non-reproducibility to "atomic operations (on floating point values) in force accumulation of the short-range GPU kernels". Deterministic accumulation was "on our todo list".
  - `-reprod` does not make GPU nonbonded runs bit-reproducible.
  - A 2018 developer reply says a bitwise reproducible version "would be slow, for no known value to users".

  — [gmx-users 2018-10](https://mailman-1.sys.kth.se/pipermail/gromacs.org_gmx-users/2018-October/122667.html), [gmx-users 2019-03](https://mailman-1.sys.kth.se/pipermail/gromacs.org_gmx-users/2019-March/124696.html)
- **LAMMPS KOKKOS** "uses thread-private arrays by default" (on CPU threads) to avoid write conflicts. I found no source on GPU determinism. — [LAMMPS Speed_kokkos docs](https://doc.lammps.org/Speed_kokkos.html)
- **PyTorch** lists `torch.index_add()` and `Tensor.scatter_add_()` on CUDA as switching to deterministic implementations under `torch.use_deterministic_algorithms(True)`. `Tensor.put_(accumulate=True)` on CUDA raises an error instead. — [PyTorch use_deterministic_algorithms (2.14)](https://docs.pytorch.org/docs/2.14/generated/torch.use_deterministic_algorithms.html)
- **Newton** chose fixed-point for hydroelastic pressure accumulation: "overflow-safe fixed-point pressure accumulation" (1.5.0). — [Newton CHANGELOG](https://github.com/newton-physics/newton/blob/main/CHANGELOG.md)

### Inferences
- Approximate cost ranking for a scatter with moderate contention (per-world forces, particle-to-grid), cheapest first:
  1. fixed-point integer atomics: about native speed, if the hardware has the needed integer width;
  2. gather over a precomputed CSR/transpose: no atomics; cost is index-build plus imbalance;
  3. per-thread or per-threadgroup partials plus a fixed-order tree: CUB says slightly slower than atomics;
  4. sort plus segmented reduce: radix sort over records, about 0.1–0.2 ms per launch at 64k–1M records on a 4090;
  5. ReproBLAS/RFA binned: +20–30% for large reductions, but a heavier accumulator state per destination.
- For MPM particle-to-grid (P2G) specifically, sorting particles by cell (which MPM codes do anyway for locality) plus a per-cell gather or segmented reduce is the natural deterministic form. Fixed-point P2G is the cheapest if 32-bit range suffices.
- **Fixed-point with only 32-bit integer atomics** is risky for forces/momenta. Range × resolution is about 2^31. At a 2^-16 resolution that leaves ±32768 of range, and per-contribution rounding is a quantization of physics, not just reordering. Options:
  - a split hi/lo two-word scheme with carry, which is not atomic across words unless done by a deterministic post-pass;
  - per-quantity scaling chosen from known bounds.

### Gaps
- I did not retrieve the SPFP paper's speed numbers or exact bit layout. The AMBER page does not state "64-bit"; Tinker-HP does.
- I found no current GROMACS docs on a deterministic GPU mode, and no LAMMPS KOKKOS GPU determinism statement.
- I found no large-N GPU benchmark comparing fixed-point atomics against float atomics.

## Q4. Apple GPU (Metal) specifics: 64-bit atomics, float atomics, simdgroup reduction ordering

### Takeaway
On Apple9 (M3/M4), MSL has 64-bit atomics only as **`atomic_ulong` min/max with no return value, on device memory**. There is no 64-bit add, CAS or fetch-with-return, so **64-bit fixed-point atomic accumulation (the SPFP approach) is not available natively**. Device `atomic_float` add exists. There are no threadgroup `atomic_float` or threadgroup 64-bit atomics. Apple does not document that `simd_sum` uses a fixed combining order.

### Cited Findings
- **Local probe** (Apple M4 Max, `supportsFamily(.apple9)=true`; Metal runtime compile with MSL 3.1, 3.2 and 4.0):
  - **OK:** device `atomic_max_explicit` / `atomic_min_explicit` on `atomic_ulong` with no return value; device `atomic_fetch_add_explicit` on `atomic_float`.
  - **FAIL** ("no matching function"): `atomic_fetch_add_explicit` on `atomic_ulong` (MSL 3.1, 3.2 and 4.0); `atomic_fetch_max_explicit` with a returned value; `atomic_compare_exchange_weak_explicit` on `atomic_ulong`; threadgroup `atomic_float`; threadgroup `atomic_ulong` max.
  - **Not a type:** `atomic_long`.

  — local measurement (Swift, `MTLDevice.makeLibrary(source:)`, macOS Darwin 25.6, 2026-10-10)
- Dawn's WebGPU backend describes Apple's 64-bit atomics as "only max/min on u64 ... do not yield the value, and they are store-only operations to storage (not workgroup) memory". It enables them on Apple9, and on Apple8+Mac2 (M2). — [Dawn commit "Atomic 64 bit for MSL"](https://dawn.googlesource.com/dawn.git/+/b620b2fe514a5ee2e6a2d924fd998c514019da87%5E%21/)
- Community testing found M2 Max passing a ulong min/max test while M1 Max did not. — [philipturner ue5-nanite-macos AtomicsWorkaround](https://github.com/philipturner/ue5-nanite-macos/blob/main/AtomicsWorkaround)
- An Apple developer forum post reports a compile failure for 64-bit CAS despite reading the Feature Set Tables as listing it. — [Apple Developer Forums thread 848400](https://developer.apple.com/forums/thread/848400)
- Apple family 9 corresponds to A17 / M3 / M4. — [MTLGPUFamily docs](https://developer.apple.com/tutorials/data/documentation/metal/mtlgpufamily.md)
- **simdgroup reductions:** I found no Apple documentation guaranteeing a combining order for `simd_sum` or similar. A PyTorch PR reports that a shuffle-based reduction misbehaves on M1/M2 when not all simdgroup lanes are active, but works on M4, so behaviour differs between generations. — [pytorch/pytorch #150566](https://github.com/pytorch/pytorch/pull/150566)

### Inferences
- **Deterministic options on Metal:**
  - (a) 32-bit integer fixed-point atomics. `atomic_uint`/`atomic_int` add is available everywhere and deterministic, but range and resolution are limited.
  - (b) Two-word fixed-point: atomic add of a 32-bit low part into a *per-contribution* staging array, then a deterministic pass. This degenerates into the gather/sort approach.
  - (c) Gathers over a precomputed CSR or sorted particle order.
  - (d) A custom radix sort plus segmented reduce. There is no CUB on Metal, so it would need porting (MPS has no general segmented reduce-by-key).
  - (e) Count-scan-write compaction with a Metal prefix scan.
- `atomic_ulong` max/min could encode an order-independent "winner" selection, for example a deterministic tie-break by packing (value bits, index) into 64 bits. That gives deterministic argmax/argmin, but not sums.
- `simd_sum` is plausibly deterministic on a fixed chip and OS. Without a documented order, though, bitwise run-to-run on the same device is likely but unproven, and cross-generation is unknown. An explicit `simd_shuffle_xor` butterfly in fixed order removes the doubt. Note that a butterfly gives every lane the same value only if all lanes compute the same sequence, which is true for XOR butterflies.

### Gaps
- I did not read the current Metal Feature Set Tables PDF rows directly. The 64-bit atomic findings rest on the local compile probe plus Dawn. I did not probe M1/M2.
- There is no Apple statement on `simd_sum` ordering or determinism.

## Q5. Are float atomics run-to-run deterministic in practice at small vs large sizes?

### Takeaway
No, not even at small sizes on a hot address. Both CUDA (vendor docs, MD codes, MJW replicas) and Apple Metal (local probe) show variation once a single address receives contributions from more than a few concurrently running threads. With low contention (few contributions per address, spread across distant threadgroups), the result *appeared* stable in my Metal probe. That is scheduling luck, not a guarantee.

### Cited Findings
- **Vendor statements:**
  - CCCL: atomics' "unordered execution across threads results in a different order of operations between runs". — [NVIDIA CCCL blog](https://developer.nvidia.com/blog/controlling-floating-point-determinism-in-nvidia-cccl/)
  - The Warp design doc says the same: "threads execute in unpredictable order ... different execution orderings produce different rounding, yielding different results each run". — [Warp design doc](https://github.com/NVIDIA/warp/blob/main/design/deterministic-execution.md)
- **CUDA physics evidence:** in MJW 3.12.0 / Warp 1.17.0 on an RTX PRO 6000, nine identical replica worlds diverge at the first substep (qacc difference 0.00198). Batch size also changes a world's trajectory. — [mujoco_warp #562 comment](https://github.com/google-deepmind/mujoco_warp/issues/562)
- **CUDA MD evidence:** GROMACS GPU runs "will naturally diverge as we use atomic operations for the final step in force reduction/accumulation". — [gmx-users 2018-10](https://mailman-1.sys.kth.se/pipermail/gromacs.org_gmx-users/2018-October/122670.html)
- **Metal evidence** (local measurement, M4 Max, `atomic_fetch_add_explicit` on device `atomic_float`, `out[i % nout] += v[i]`, values uniform in [−1,1], threadgroup size 256, 30 repeats each):

  | N | nout | distinct results / 30 | max abs diff vs run 0 |
  |---|---|---|---|
  | 1,024 | 1 | 14 | 6.1e-5 |
  | 32,768 | 1 | 25 | 1.1e-3 |
  | 1,048,576 | 1 | 30 | 3.0e-2 |
  | 1,048,576 | 1,024 | 23 | 1.5e-5 |
  | 4,194,304 | 65,536 | 1 | 0 |
  | 4,194,304 | 1,048,576 | 1 | 0 |

  In the last two rows each address received 64 or 4 contributions from threads 65,536 or more indices apart. The contributions probably ran at well-separated times, so the order was effectively fixed by dispatch order. This is unguaranteed and would change with contention or GPU load.
- **Warp's high-contention benchmark** shows native float atomics serializing: 1.46 ms for 1M writes into 1 address on a 4090, against 0.193 ms for sort-reduce. — [Warp docs](https://github.com/NVIDIA/warp/blob/main/docs/user_guide/execution_and_performance/deterministic_execution.rst)

### Inferences
- "Small launches look deterministic" is false for contended addresses: 1,024 threads into one address gave 14 distinct results out of 30 runs on Metal.
- For a sim at thousands of worlds, per-world accumulators (per-body force, per-dof) typically get tens to hundreds of contributors that run concurrently in one threadgroup or adjacent ones. That is the varying regime, so explicit determinism is required.

### Gaps
- I found no published systematic CUDA study of float-atomic variation versus launch size or contention. The CUDA evidence is anecdotal or physics-level; only the Metal table here is a controlled sweep, run on one device.

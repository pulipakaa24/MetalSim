# Parallel kernel compilation at startup

Status 2026-10-14:
- Done: designed, implemented, and tested on the CPU.
- Pending: the Metal measurements. The batch `_metalsim_parallel_compile/batch.sh` (in the Warp worktree) produces
  them and adds them under "Measured" below.
- Measured so far on Metal (the batch's build step, 2026-10-14 20:14): this M4 Max (macOS 26) reports
  `maximumConcurrentCompilationTaskCount` = **16 with and without `shouldMaximizeConcurrentCompilation`**. The flag
  buys nothing here, so it is off by default (section 3).
- 2026-10-15: the first G1 cold matrix stopped on the byte-identity guard. Two
  `mujoco_warp._src.forward` `.metal` files differed between the lazy run and the one-worker preload. The cause
  turned out to be in Warp's codegen, not in preloading (section 3, "Codegen nondeterminism found by the guard").
  The matrices now run with a fixed `PYTHONHASHSEED`.
- Rebased 2026-10-14 onto MetalSim main `83ce2ebc` (after #29 deterministic BatchSim and #66): the preload runs inside
  `BatchSim.det_scope`, so it works in deterministic mode (section 3).

**Problem.** Warp compiles a module lazily, at its first launch, one module at a time, on the Python thread.
- **Cold start.** A cold start of the G1 velocity task loads 47 modules, and their load timers add up to 29.7 s of the
  ~45 s start (`runs/competitors/metalsim_g1_flat_4096.log`). The largest are `ccd_kernel` 2.86 s, `smooth` 1.99 s
  and `sensor` 1.95 s.
- **Warm start.** A warm start still spends ~8.4 s in 43 "cached" loads, ~195 ms each
  (`runs/fma_rigid_parity/tp_integ/warm_I_rough.log`).

## 1. Prior art: solved elsewhere?

Only partly.
- **Warp** has a parallel loader. It only works for modules you can name in advance.
- **MuJoCo Warp, Newton and Isaac Lab** don't ship a warm-up that finds the modules for you.
- **Warp's own fix** for that is an open issue.

### Warp: `max_workers` / `load_module_max_workers`

**When it was added.** Warp 1.11.0 (2026-01-02). The CHANGELOG entry reads: "Add support for parallel module
compilation and loading via `max_workers` parameter in `wp.load_module()` and `wp.force_load()`. Defaults to serial
loading; set `wp.config.load_module_max_workers` to enable parallel loading (GH-1086)".
- Request: [NVIDIA/warp#1086](https://github.com/NVIDIA/warp/issues/1086). It is a child of #944, "Investigate
  compile time improvements".
- Merge commit: [dd454af](https://github.com/NVIDIA/warp/commit/dd454af). It added a `ThreadPoolExecutor` in
  `force_load`, a lock around `Adjoint.build`, and per-thread build directories.
- Docs:
  - [config](https://warp.readthedocs.io/en/latest/api_reference/_generated/warp.config.load_module_max_workers.html):
    0 means serial; `None` means `min(cpu_count, 4)`.
  - [Reducing compilation and startup time](https://nvidia.github.io/warp/latest/user_guide/execution_and_performance/reducing_compilation_and_startup_time.html).
    It warns: "Parallel loading offers little benefit when one large module dominates or when module loads are
    short."

**Why the default is 0.** No reason is stated anywhere: not in the issue, the commit, the docs or the changelog. Our
reading:
1. **The config alone does nothing for lazy loads.** The config value only matters when `force_load` or
   `load_module` is called with a module list. MuJoCo Warp's
   [PR #989](https://github.com/google-deepmind/mujoco_warp/pull/989) set only the config and saw no gain
   (95.91 s vs 95.87 s).
2. **The parallel path had bugs for months.**
   - Shared-`@wp.func` codegen races: [#1474](https://github.com/NVIDIA/warp/issues/1474) and
     [#1532](https://github.com/NVIDIA/warp/issues/1532). Fixed in 1.15.0 by serializing Python codegen under
     `_codegen_lock`.
   - CPU/LLVM cold-start races that failed 1 in 12 to 6 in 20 fresh processes:
     [#1705](https://github.com/NVIDIA/warp/issues/1705), fixed in 1.17.0.
   - Build errors silently dropped by the parallel path: [#1749](https://github.com/NVIDIA/warp/issues/1749), fixed
     in 1.17.0.

   Our fork (1.18.0.dev) contains all of these fixes.

**Non-CUDA devices.** `force_load` submits `Module.load` for every device in the list, CPU included. Our Metal backend
serialized it anyway:
- `wp_metal_load_library` (MSL to `MTLLibrary`) and `wp_metal_get_kernel` (pipeline creation) both ran entirely
  inside `WP_METAL_LOCK()`, a process-wide recursive mutex (`warp/native/metal.mm`).
- So the native compile, which is the dominant cost, could never overlap with another one. Only the cheap Python side
  could overlap with the thread holding the lock.

**GIL.** Native calls go through `ctypes.CDLL`, which releases the GIL. Python codegen holds the GIL and
`_codegen_lock` (an `RLock`), so it stays serial by design. It is Amdahl's serial part.

**Still open upstream.** [NVIDIA/warp#1977](https://github.com/NVIDIA/warp/issues/1977) (milestone 1.20.0) proposes
deferring compilation to graph launch through APIC, so that all of a graph's kernels compile in parallel.
- Its motivation is ours: Newton's MuJoCo ANYmal example takes "about a minute" cold, compiling modules "on demand,
  and serially".
- It notes that `max_workers` "requires the user to know in advance which ones will be needed".

### MuJoCo Warp

- **No compile-everything helper.** `testspeed` reports "Total JIT time", which is the first `fn(m, d)` inside
  `wp.ScopedCapture()`, so lazy compilation during the first capture
  ([cli.py](https://github.com/google-deepmind/mujoco_warp/blob/main/mujoco_warp/_src/cli.py),
  [testspeed.py](https://github.com/google-deepmind/mujoco_warp/blob/main/mujoco_warp/testspeed.py)).
- **[#813 "Optimize GJK JIT time"](https://github.com/google-deepmind/mujoco_warp/issues/813).**
  - Each `ccd_kernel_builder` module took 10-12 s cold on CUDA.
  - Precompiled headers plus `--Ofast-compile=min` cut the cold start from 32 s to 22 s.
  - Parallel loading was "the" fix: about 5x in a synthetic test.
  - The follow-up (#989) was closed because lazy loads stay serial.
- **[PR #1676](https://github.com/google-deepmind/mujoco_warp/pull/1676)** (merged 2026-10-08) avoids recompiling
  `solver.py` for a `block_dim` mismatch. It gives no numbers.
- **Model-specialized kernels** (the `@cache_kernel` factories in `warp_util.py`; modules named
  `..._locals__kernel_<hash>`) are created by the Python step code itself. A generator call is cached by argument
  sizes, so the same Python code always yields the same `Kernel` objects.

### Newton and Isaac Lab

- **No usage found.** We found no use of `load_module`, `force_load` or `load_module_max_workers`. A code search was
  not possible, so this means "not found", not "absent".
- **Startup work elsewhere.** [newton#3881](https://github.com/newton-physics/newton/issues/3881) works on warm
  startup and says GPU work is "concentrated in first-step graph capture". Isaac Sim's Newton docs say the first run
  "may take up to a minute" ([docs](https://docs.isaacsim.omniverse.nvidia.com/6.0.1/physics/newton_physics.html)).

### Apple Metal

**Thread safety.** No Apple page says "newLibraryWithSource is thread-safe" in so many words. The evidence that it is:
- `MTLDevice` is declared `Sendable` ([MTLDevice](https://developer.apple.com/documentation/metal/mtldevice)).
- An Apple engineer wrote: "You should be able to safely alloc resources from a device"
  ([forum 93346](https://developer.apple.com/forums/thread/93346)).
- Chromium compiles libraries and pipelines off the main thread
  ([metal_api_proxy.mm](https://chromium.googlesource.com/chromium/src.git/+/49cea54a/components/viz/common/gpu/metal_api_proxy.mm)).

**Concurrency is capped by the device.**
- [`shouldMaximizeConcurrentCompilation`](https://developer.apple.com/documentation/metal/mtldevice/shouldmaximizeconcurrentcompilation)
  (macOS 13.3+, default NO) means the device "uses additional CPU threads for compilation tasks".
- [`maximumConcurrentCompilationTaskCount`](https://developer.apple.com/documentation/metal/mtldevice/maximumconcurrentcompilationtaskcount)
  reports the cap.
- WWDC23 [session 10127](https://developer.apple.com/videos/play/wwdc2023/10127/): "When you set it to Yes, Metal
  compiler will make the best use of your CPU cores."
- Blender Cycles sizes its compile pool as `max(2, maximumConcurrentCompilationTaskCount - 1)`.
- The default count is not documented. **Measured on this M4 Max: 16, with or without the flag**. So on this Mac the device
  setting does not limit 8 workers; any limit comes from the compiler service's own CPU use.

**Disk cache.** Apple's system shader cache (`$(getconf DARWIN_USER_CACHE_DIR)/com.apple.metal`) also serves
`newLibraryWithSource`. ggml measured 16 s cold vs 0.4 s warm
([report](https://github.com/leehack/stable-diffusion-native/issues/8)). It cannot be redirected per process, so
"cold" below means a cold *Warp* cache. The first run of each matrix is reported separately.

### Our fork's build path (read for thread safety)

**`Module.load` steps:**
1. Hash and codegen, under `_codegen_lock`.
2. Write to a per-thread build directory (`<module>_p<pid>_t<tid>`), published with `os.replace`.
3. On Metal: read the `.metal` file, then call `wp_metal_load_library`.
4. Store the `ModuleExec` in `module.execs`, a dict assignment under the GIL.

Upstream exercises this path in `warp/tests/test_module_parallel_load.py`.

**What Metal caches.** The Metal source **is** the cache artifact. A warm start skips codegen but still compiles the
MSL with `newLibraryWithSource`, which Apple's cache serves when it hits. So warm starts gain from parallelism too.

**Module size.** Every Metal module is 1.6-2.6 MB of MSL, mostly the prelude (measured in section 2). That is why even
a one-kernel module costs ~350 ms cold.

## 2. Measurements available before the Metal runs (this Mac, CPU only)

### Python share of a Metal cold load

Script: `scripts/diagnostics/compile_cost_cpu.py codegen`, on the G1 flat BatchSim (4 substeps; step, forward and
reset). Two runs: 4.99 s and 5.04 s.
- **Recording.** The step code records 43 module variants and 75 kernels in 0.145 s of Python (no launches).
- **Serial Python work: 5.0 s.** Module hashing takes 0.08 s and Metal-target code generation 4.9-5.0 s.
- **Generated source sizes:**

  | module | MSL size | codegen time |
  |---|---|---|
  | `sensor` | 2.56 MB | 400 ms |
  | `ccd_kernel` | 2.46 MB | 385 ms |
  | `smooth` | 2.54 MB | 361 ms |
  | small modules | ~1.65 MB | ~80 ms |
- **The rest is native.** The rest of the 27.3 s cold BatchSim load (see section 4) is native compile and I/O, about
  22.3 s. Pipeline creation at the first launch adds to that, outside the timers.

### Amdahl check on the CPU target

Script: `scripts/diagnostics/compile_cost_cpu.py scaling --workers N`. It compiles the same 43 G1 modules with LLVM
through `wp.preload_launches`, using fresh caches.

| workers | load [s] | model S + P/N (fitted: S = 2.80 s, P = 10.18 s) |
|---|---|---|
| 1 | 12.98 | 12.98 |
| 2 | 7.41 | 7.89 |
| 4 | 5.51 | 5.35 |
| 8 | 4.07 (4.02 on a rerun) | 4.07 |

The model holds within 7 %. Five toy modules took 4.4 s serially and 0.96 s on 4 threads.

## 3. Design

### Warp fork (branch `metalsim-parallel-compile`; no kernel code change)

1. **`metal.mm`: compile outside the lock.** `wp_metal_load_library` and `wp_metal_get_kernel` hold the runtime mutex
   only for two things:
   - reading device state (the device handle and the address-table slot);
   - publishing results (the error string and the library's pipeline list).

   `newLibraryWithSource`, `newFunctionWithName:constantValues:` and `newComputePipelineStateWithDescriptor` now run
   outside it. A new function, `wp_metal_set_maximize_concurrent_compilation(ordinal, on)`, returns
   `maximumConcurrentCompilationTaskCount`.
2. **`wp.preload_launches(device, max_workers, kernel_hooks=True)`**, a context manager.
   - **Inside the block.** `wp.launch`, and everything built on it, records the kernel, its overload and the resolved
     `block_dim` instead of launching. This applies to the calling thread only, and to that device only.
   - **On exit:**
     - The distinct (module, device, block_dim) variants load on `max_workers` threads, through the same
       `Module.load`.
     - As each module finishes, each recorded kernel's hooks are looked up. On Metal that creates the pipelines,
       which is compiler work too.
     - `shouldMaximizeConcurrentCompilation` is **not** set by default. It is opt-in through
       `warp.config.metal_maximize_concurrent_compilation` (`WP_METAL_MAXIMIZE_CONCURRENT_COMPILATION=1`). This Mac
       reports 16 concurrent compile tasks either way, and the flag is process-wide and sticky, so turning it on has
       a cost (risk 4) and no measured benefit here. It is kept for devices whose default count is lower. The
       benchmark matrix no longer has a with/without-flag variant.
   - **Failures are collected, not raised.** The real launch replays the module's recorded build error, so error
     behaviour is unchanged.
   - **`LaunchPreload.add(kernel, device, block_dim)`** records kernels known from elsewhere.

### MetalSim (branch `parallel-compile`; opt-in)

Turn it on with the `preload_workers` option, or with `METALSIM_PRELOAD_WORKERS=N`.

- **`BatchSim(..., BatchSimOptions(preload_workers=N))`.**
  - **What it does.** Before the step, forward and reset graphs are captured, `metalsim.physics.warmup.preload_capture`
    runs the same three launch bodies once. The pass runs under `preload_launches`, inside a `wp.ScopedCapture` whose
    graph is thrown away. Nothing executes: memory operations land in the discarded graph, and launches are only
    recorded.
  - **Why it finds the specialized kernels.** MuJoCo Warp's generator functions run exactly as they do in the real
    capture, so the `__locals__` modules exist and are recorded. This is the cheapest way to make them exist first:
    ~0.15 s of Python and no change to MuJoCo Warp.
  - **Why it is safe.** The body already runs under capture. `BatchSim` already re-captures and discards graphs
    whenever `add_substep_hook` is called.
  - **Deterministic mode (#29).** The pass runs in `_capture_graphs`, which `_capture` calls inside `det_scope`. So the
    generated MuJoCo Warp kernels are created, hashed and loaded under the same `RUN_TO_RUN` config as in the real
    capture. That matters: a module's hash is fixed by the config at its first load, so loading a generated kernel
    outside `det_scope` would leave a non-deterministic executable that the capture then reuses.
    `check_determinism()` still runs after the capture and would flag that. The worker threads read the same
    process-global `warp.config` that `det_scope` sets.
- **`MPMCoupling` (`MPMConfig.preload_workers`).**
  - **Why not the same approach.** The first coupled step is eager: the solver sizes its buffers from device results.
    So it cannot run under a capture, and recording without executing would size the buffers wrong.
  - **The manifest instead.** `warmup.CompileManifest` writes the named modules (and launched kernel keys) that the
    first eager step loaded to `<kernel cache parent>/metalsim_compile_manifests/<config hash>.json`. The file lives
    next to Warp's versioned cache, not inside it, so it survives a cleared cache or a Warp upgrade. The next run
    loads those modules on N threads before its first step.
  - **Not covered.** Kernels generated during that step (warp.fem `dyn.*`) still compile lazily. The rigid half is
    covered by `BatchSim`'s preload.
  - **MuJoCo Warp modules are never in the manifest.** This covers `_end_kinematics`' kernels too. Reason: in
    deterministic mode they must load inside `BatchSim.det_scope` (see above), and the manifest preload runs outside
    it.
- **Default off.**
  - Turning it on by default is a decision for after the Metal numbers and the parity suites with
    `METALSIM_PRELOAD_WORKERS=8`. The batch runs both.
  - The native lock narrowing is always on. Warp's Metal suite and MetalSim's parity suites cover it.

### Correctness checks

These run in the batch. Any failure stops it before any commit.
- **Kernel cache bytes.** Every cold run compares its kernel cache with the serial run's, file by file (`.metal`,
  `.metal.meta`). One differing byte stops the batch.
- **`box_sand` results.** The result hash (box pose, velocity, sand wrench and every particle) must equal the serial
  run's. This scene is bitwise repeatable in default mode (`tests/test_mpm_main.py`).
- **G1 results.** Default-mode G1 is not run-to-run repeatable (`docs/GAPS.md`), so its hashes are only reported.
  Deterministic-mode runs (`BatchSimOptions(deterministic=True)`), lazy vs 8 threads, must match bit for bit.
- **`tests/test_preload_compile.py`.** A Go1 BatchSim in deterministic mode: same state, same cache bytes, no extra
  compiles. Also a manifest round trip.
- **Warp unit tests.** `warp/tests/test_preload_launches.py` (9 tests, CPU-verified here), `test_module_parallel_load`,
  `test_metal` and `test_metal_deterministic`.
- **MetalSim parity suites.** `test_physics`, `test_g1_parity`, `test_batch_deterministic` and `test_mpm_main`, once
  with preloading off and once with `METALSIM_PRELOAD_WORKERS=8`.

### Codegen nondeterminism found by the guard (2026-10-15)

**What the guard saw.** The G1 cold matrix compared the kernel cache of `cold_w1_r0` (preload, one worker) with
`cold_w0_r0` (lazy). Of 86 files, 2 differed: `wp_mujoco_warp._src.forward_2c0e669.metal` and `..._ad6f1d2.metal`.

**What the difference is.**
- The module hashes are the same, the kernel set is the same, and the file sizes are the same.
- The diff is 24 lines in one kernel: the two `wp::copy` lines before a dynamic loop are swapped, so `var_482` and
  `var_483` trade roles consistently through the loop's `assign` and the `where` after it. Same program, different
  text.
- The *prime* run (also lazy, serial) is byte-identical to the one-worker preload, and differs from the other lazy
  run.

**Cause.** Warp's `Adjoint._promote_mutated_iter_constants` (codegen.py) collects the names a dynamic loop assigns
into a Python `set` and emits a `copy` for each in set order. Set order of strings follows Python's per-process hash
randomization. So any two processes may number those variables differently, with or without preloading.

**Check.** `scripts/diagnostics/codegen_determinism_cpu.py` runs on the CPU in about 1 min. It records or runs the
G1 BatchSim launch code in four processes: lazy and preload, each under `PYTHONHASHSEED` 1 and 2. It then compares
the Metal source of all 43 MuJoCo Warp module variants.

| Comparison | Variants compared | Different Metal source |
|---|---|---|
| lazy vs preload, same seed (seed 1; seed 2) | 43 | none |
| lazy vs lazy, two seeds | 43 | only `mujoco_warp._src.forward` |
| preload vs preload, two seeds | 43 | only `mujoco_warp._src.forward` |

So preloading records the same modules, with the same module hashes and the same source, as lazy loading. The
difference is pre-existing process-to-process nondeterminism.

**What changed.**
- The benchmark and `tests/test_preload_compile.py` now run every process with `PYTHONHASHSEED=0`, so the byte
  comparison isolates what preloading changes.
- The G1 results from the first attempt are discarded and the matrix reruns.

**Follow-up (not in this change).** Iterating the names in source order, `dict.fromkeys` over the same walk, would
make Warp's codegen deterministic across processes. It changes only variable numbering in generated source, not
module hashes or semantics. It belongs on its own fork branch: it touches every cached module with such a loop.

## 4. Expected gain (to be replaced by the measurement)

### G1 BatchSim, cold Warp cache

- **Serial part S = 5.0 s**: Python codegen, measured in section 2.
- **Parallel part P = 22.3 s**: 27.3 s minus S. The 27.3 s is the 29.7 s total minus the task's own three modules:
  `__main__` 1.16 s, `metalsim.sensors.contact` 0.74 s and `metalsim.learn.warp_policy` 0.46 s.
- **T(N) = S + P/N**, assuming Apple's compiler service runs N compiles at once:

  | N | T(N) |
  |---|---|
  | 2 | 16.2 s |
  | 4 | 10.6 s |
  | 8 | 7.8 s |

- **Critical path.** The largest single module (`ccd_kernel`: 0.39 s codegen + ~2.5 s compile) stays under T(8).
- **Result at N = 8:** about 19.5 s saved, so the ~45 s cold start falls to ~25 s (1.8x). Pipeline creation is also
  parallelized, and is not counted in this figure.
- **Device cap.** It does not limit N <= 8 here: the device reports 16 concurrent compile tasks.

### G1, warm Warp cache

- **Where the time goes.** 43 cached loads at ~195 ms each make 8.4 s. Most of that is the native compile, served by
  Apple's cache, plus the 2 MB file reads and the regex scans.
- **Estimate.** The Python share is estimated at ~1.2 s, so T(8) ≈ 1.2 + 7.2/8 ≈ 2.1 s. The ~14 s warm start falls to
  ~8 s.

### Coupled MPM

- **Basis.** The closest logged coupled cold start is Newton's two-way example
  (`runs/mpm/metal_floors_integrated/F_twoway.log`): 30 named modules take 30.3 s, and 60 generated modules take
  14.2 s.
- **Estimate.** With `BatchSim` plus the manifest, and assuming G1's 18 % Python share, the named part would go from
  30.3 s to ~8.6 s at N = 8 (5.5 + 24.8/8). The generated part stays serial at 14.2 s. Total: 44.6 s → ~23 s.
- `box_sand` has fewer rigid modules; the batch measures it.

## 5. Risks

1. **The native lock change is not opt-in.** Every Metal module load now compiles outside the runtime mutex.
   - Device state is still read and written under the lock.
   - Concurrent *failing* compiles can overwrite each other's error string, because Warp's error buffer is
     process-global (already so for CPU and CUDA loads). This affects messages only.
2. **The preload pass assumes capture-safe code.**
   - A graph captured inside the pass holds no launches. So code that captures and caches a graph on its first call
     must not make that first call inside the pass. MuJoCo Warp doesn't do this, and newton_compat builds its chunk
     graph only outside a capture.
   - Lazily allocated state whose initialization is recorded only into the discarded graph would also break. None is
     known in `mjw.step`, `forward` or `reset_data`, and `BatchSim` already re-captures and discards. The
     deterministic and `box_sand` bitwise checks guard against this.
3. **Byte identity relies on order-independent codegen.** Upstream fixed the known shared-function races. The batch
   compares every file of every cold run.
4. **CPU contention.**
   - 8 Python threads plus the compiler service's processes run on the M4 Max at once.
   - `shouldMaximizeConcurrentCompilation` is process-wide and stays on. It could slow other compiler work in the
     same process, for example a renderer compiling shaders. It is now off by default, since it measured no
     difference on this Mac (16 tasks either way).
5. **Stale manifest.** A stale manifest loads extra modules (costs time) or misses some (they compile lazily, as
   before). It never gives wrong results.
6. **Apple's shader cache.** Apple's cache is shared, so cold-Warp numbers after the first run include warm
   Apple-cache hits. The prime run shows the true first compile. Confirmed in the first G1 attempt: prime 21.1 s wall
   (18.9 s build) against 8.6 s for the next lazy cold-Warp run.
7. **Peak memory.** Startup peak memory rises slightly: up to 8 compiles in flight, plus one throwaway graph.

## 6. Measured

Measured on the 16-inch M4 Max by `scripts/diagnostics/startup_compile_bench.py matrix` (medians; process wall = interpreter start to the end of the steps; speed-up against the serial run of the same cache state; raw records in `runs/parallel_compile/`).

### g1 (4096 worlds)

| cache | workers | runs | process wall [s] | build [s] | first step [s] | modules compiled | sum of module load timers [s] | speed-up vs serial |
|---|---|---|---|---|---|---|---|---|
| cold_prime | w0 | 1 | 15.9 | 14.4 | 0.1 | 43 | 13.8 | 1.00x |
| cold | w0 | 2 | 8.2 | 6.7 | 0.1 | 43 | 6.1 | 1.00x |
| cold | w1 | 2 | 8.3 | 6.8 | 0.1 | 43 | 6.2 | 0.99x |
| cold | w2 | 2 | 8.1 | 6.6 | 0.1 | 43 | 12.1 | 1.01x |
| cold | w4 | 2 | 8.0 | 6.5 | 0.1 | 43 | 23.7 | 1.03x |
| cold | w8 | 2 | 7.9 | 6.4 | 0.1 | 43 | 43.1 | 1.04x |
| warm | w0 | 2 | 3.8 | 2.3 | 0.1 | 0 | 1.8 | 1.00x |
| warm | w2 | 2 | 3.7 | 2.2 | 0.1 | 0 | 3.3 | 1.04x |
| warm | w4 | 2 | 3.6 | 2.1 | 0.1 | 0 | 6.2 | 1.05x |
| warm | w8 | 2 | 3.6 | 2.1 | 0.1 | 0 | 11.6 | 1.06x |
| cold_det | w0 | 1 | 39.8 | 37.6 | 0.2 | 47 | 27.3 | 1.00x |
| cold_det | w8 | 1 | 9.5 | 7.3 | 0.2 | 47 | 50.2 | 4.19x |

Kernel cache files compared with the serial build: 9 runs, 782 files, 0 differ.
Distinct result hashes: default mode: 19, deterministic: 1

### box_sand (2 worlds)

| cache | workers | runs | process wall [s] | build [s] | first step [s] | modules compiled | sum of module load timers [s] | speed-up vs serial |
|---|---|---|---|---|---|---|---|---|
| cold_prime | w0 | 1 | 16.2 | 8.5 | 6.6 | 113 | 12.9 | 1.00x |
| cold | w0 | 2 | 16.2 | 8.5 | 6.5 | 113 | 12.8 | 1.00x |
| cold | w1 | 2 | 16.1 | 8.5 | 6.5 | 113 | 12.8 | 1.00x |
| cold | w2 | 2 | 16.0 | 8.3 | 6.5 | 113 | 18.6 | 1.01x |
| cold | w4 | 2 | 15.8 | 8.2 | 6.5 | 113 | 29.4 | 1.02x |
| cold | w8 | 2 | 15.8 | 8.2 | 6.4 | 113 | 47.5 | 1.02x |
| warm | w0 | 2 | 7.9 | 4.0 | 2.8 | 0 | 4.6 | 1.00x |
| warm | w2 | 2 | 7.7 | 3.8 | 2.7 | 0 | 6.4 | 1.03x |
| warm | w4 | 2 | 7.6 | 3.8 | 2.7 | 0 | 9.5 | 1.04x |
| warm | w8 | 2 | 7.6 | 3.8 | 2.7 | 0 | 15.2 | 1.04x |

Kernel cache files compared with the serial build: 8 runs, 1808 files, 0 differ.
Distinct result hashes: default mode: 1


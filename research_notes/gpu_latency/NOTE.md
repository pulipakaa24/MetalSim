# Making the M4 Max GPU busy in the coupled MPM scenes (research before measuring, 2026-10-13)

**Observation (user).** At 100 % "Device Utilization" the package power stays modest.
- Device utilization only counts time the GPU has work queued.
- A stream of short, dependent, latency-bound dispatches keeps it at 100 % while most ALUs idle.

**Scope** (MPM.md 8k, "left"):
- (1) the Gauss-Seidel (GS) colour passes: ANYmal's sand step 22.2 ms per frame;
- (2) Warp's tiled `bsr_mv` on Metal: 1.33 ms vs the L4's 0.12 ms;
- (3) the MuJoCo Warp robot graph: 8.3 ms per frame vs about 1.4 ms of L4 kernel time.

**Constraint.** Results must stay bitwise where the math is unchanged.

## What is known (code and earlier measurements)

- **One queue, serial encoders.** Warp's Metal backend has one `MTLCommandQueue` per device, and every dispatch goes
  into a default (serial) compute encoder (`warp/native/metal.mm`: `computeCommandEncoder`, `dev.queue`).
  - A serial encoder runs each dispatch after the previous one finishes.
  - Graph replays are indirect-command-buffer batches (MPM.md 8, "4. Graph replay") inside the same serial ordering.
  - So independent small kernels never overlap. The GPU runs one short kernel at a time, and between kernels it pays
    the dispatch cost: about 1.4-2.5 us per dependent dispatch in a replay (MPM.md 3, 8.3).
- **GS colour pass** (`rheology_solver_kernels.py:992`, `gs_solve_kernel_impl`):
  - One launch per colour per iteration, `2 x cores` threadgroups of 64 threads.
  - Each thread walks colour blocks and, inside a block, its strain nodes **in order**. That order is the algorithm:
    nodes of a block can share velocity nodes.
  - Per node: a gather over its strain-matrix row (velocities), the local flow-rule solve, then a scatter. These form
    one long dependent chain of global loads and math.
  - The pass therefore lasts as long as the longest chain of the colour, plus the dispatch.
  - Measured (batch_any1):
    - 1 / 2 / 4 / 8 threadgroups per core give bitwise equal results; 2 is already the fastest, so the thread count is
      not the limit.
    - Newton's reordered layout (`gs-soa`) is bitwise equal and slower.
  - Isolated per-pass time: about 21 us on Metal vs 8 us on the L4 (MPM.md 8j profile).
- **Tiled `bsr_mv`** (`warp/_src/sparse.py`, `make_bsr_mv_tiled_kernel`, default tile 64):
  - Thread (row, subrow, lane) sums the scalar columns `lane, lane + 64, ...` of its row in order, then `tile_sum`
    over the 64 lanes (CUDA: warp shuffle tree plus a fold of the 2 warp partials; Metal: the same tree,
    `metal_reduce::block_reduce`).
  - For two-way's IJtm that is 12 rows x 3 subrows = 36 threadgroups of 64, with about 55 terms per lane.
  - Each term is `x[columns[block] * 3 + c]`, a dependent pair of loads (column index, then x), so each lane is a
    serial chain of about 55 load-dependent steps. About 26 us per launch, with 4 % of the GPU's threads in use.
  - The CUDA order is fixed by the lane decomposition (64 strided partial sums, then the tree). Splitting a lane's chain
    across threads would change the rounding.
- **Robot graph** (ANYmal, `SolverMuJoCo.step` x 4 substeps per frame; MuJoCo Warp 3.11 in `.venv-newton152`):
  - Hundreds of small kernels per substep (kinematics, com / crb tree passes per tree level, factorization, collision,
    constraints, a solver loop with line search).
  - On the L4 these kernels take about 1.4 ms per frame; on Metal the robot part takes 8.3 ms.
  - The isolated per-kernel profile shows 3-11x per kernel. That includes command-buffer start cost, so in-graph
    numbers are needed.

## Prior art (external)

- **Concurrent dispatch with tracked barriers: verified in source** (read 2026-10-13 at the commits below).
  - Apple: `MTLDispatchType.concurrent` lets dispatches in one encoder overlap. The caller must order accesses to a
    shared resource with memory barriers.
  - **MLX** (ml-explore/mlx `243bdf09c0cccbb85ca6a686dab3cc0102cf1f7e`):
    - `mlx/backend/metal/device.cpp:596` creates every compute encoder with `computeCommandEncoder(MTL::DispatchTypeConcurrent)`.
    - `set_input_array` (`:361`) adds each input buffer to `next_inputs_` and sets `needs_barrier_` if that buffer is
      in `prev_outputs_` (`:371`).
    - `register_output_array` (`:385`) adds each output buffer to `next_outputs_` and sets `needs_barrier_` if it is
      in `prev_inputs_` (`:394`). So write-after-read is covered too. Outputs are also passed through
      `set_input_array` first (`set_output_array`).
    - `maybeInsertBarrier` (`:409`) runs before every dispatch. When `needs_barrier_` is set it calls
      `memoryBarrier(MTL::BarrierScopeBuffers)` (`:411`) and starts a new epoch (the previous sets become this
      dispatch's sets). Otherwise the sets accumulate.
    - `barrier()` (`:441`) forces a barrier.
    - `device.h:32-40, 92` `ConcurrentContext` / `start_concurrent()` marks a group of dispatches as mutually
      independent: their outputs go to `concurrent_outputs_` and join the tracked outputs when the group ends.
    - Granularity: whole buffers. Each op declares its inputs and outputs explicitly.
  - **llama.cpp / ggml-metal** (ggml-org/llama.cpp `091b6a7fc561b78da90a0ed2d524ec995458c657`):
    - `ggml/src/ggml-metal/ggml-metal-device.m:841` creates the encoder with
      `computeCommandEncoderWithDispatchType: MTLDispatchTypeConcurrent`, and `:894` issues
      `memoryBarrierWithScope:MTLBarrierScopeBuffers`.
    - `ggml-metal-context.m:142` turns concurrency on unless `GGML_METAL_CONCURRENCY_DISABLE` is set.
    - `ggml-metal-ops.cpp:159-184` (`ggml_metal_op_concurrency_reset/check/add`) and `:240` decide per node.
    - `ggml-metal-common.cpp:276-327` (`ggml_mem_ranges_check`) tracks byte ranges per buffer, typed src / dst. Two
      sources may overlap (`:286`); any overlap involving a destination needs a barrier. Granularity: byte ranges.
  - Dawn and Skia stay on serial encoders (their own comments).
  - **Applicability to Warp:**
    - Warp's launch knows each array argument, but no per-argument read/write flag is passed to the runtime today.
    - Two ways forward:
      1. Conservative: every array argument counts as an output (MLX's rule with outputs = all buffers). A barrier is
         placed when a dispatch shares any buffer with the current epoch. Kernels on disjoint buffers overlap.
      2. Precise: derive the per-argument write set from codegen (stores and atomics on an argument) and pass it to
         the launch. MLX's input/output split then lets readers overlap. Whether Warp's adjoint already records this
         must be checked.
    - Same kernels and same dependencies, so results stay bitwise if the hazard rule is complete. Risk: a missed hazard
      is a silent race. It needs every MPM, rigid and MuJoCo Warp bitwise test with concurrency on, plus a kill switch
      (`GGML_METAL_CONCURRENCY_DISABLE`-style).
- **Graph capture.** MuJoCo Warp's docs: `mjw.step` is a collection of kernel launches; capture it and replay it with
  `wp.capture_launch` "for significantly faster" stepping. Optimized for throughput, not latency. No kernel-count or
  fusion data was found.
- **Forward progress.** Grid-wide barriers and persistent kernels stay excluded on Metal (no inter-threadgroup progress
  guarantee: `research_notes/Metal MPM speed and scalable determinism/latency_bound_solvers.md` Q1/Q4; GPU-hang
  memory).

## Candidate levers (bitwise unless stated)

1. **Robot / sand overlap in ANYmal.**
   - The sand sees the robot as a kinematic collider (`setup_collider(body_mass=0)`), and the robot never sees the
     sand: one-way coupling.
   - So robot(k+1) does not depend on sand(k), and both could be in flight together if Warp could submit two
     independent streams or a concurrent encoder.
   - Upper bound on gain: min(robot, sand) = 8.3 ms of 30.5.
   - Needs Metal stream concurrency in the Warp fork (a second queue, or concurrent encoders with events).
   - Bitwise (no shared writes).
2. **Concurrent encoder with conservative hazards** (Warp fork). Applies to all small-kernel graphs (robot graph, MPM
   setup kernels). Gain unknown until measured: it needs the share of time that is dispatch serialization rather than
   kernel execution.
3. **Tiled `bsr_mv`: memory-level parallelism inside each lane, same order.**
   - Unroll the lane loop by 4-8 with all index and value loads issued first, then the accumulations in the original
     order. Same operations in the same order (bitwise), and 4-8x more loads in flight per thread.
   - A Metal-only kernel variant in the Warp fork.
   - Also: one threadgroup per (row, subrow) is fixed by the tree, so the order cannot be widened. Unrolling is the
     order-preserving lever.
4. **GS pass:** the same unroll / prefetch idea for the per-node gather, inside Newton's kernel (fork). Bitwise only if
   the arithmetic order is unchanged, so it must be checked. Measure first how much of a pass is chain latency vs
   dispatch.
5. **Fewer dispatches in the robot graph** (fusion in MuJoCo Warp): changes code paths. That is upstream MuJoCo Warp
   work, out of scope for MPM. Report it if the measurement shows it dominates.

## Measurement plan (batch_lat1)

1. **GS structure and in-replay pass time** (`probes/gs_structure_probe.py`, ANYmal):
   - colours; blocks and strain nodes per colour; nodes per block (min / median / max);
   - microseconds per colour pass in a 40-sweep graph replay;
   - the replay cost of an empty dispatch (dispatch floor).
   - Gives chain latency per pass = pass time - floor.
2. **Robot graph by subtraction** (`probes/robot_graph_probe.py`, ANYmal): replay graphs of one substep's parts, each
   200 times:
   - Newton's whole `SolverMuJoCo.step`;
   - `mujoco_warp.step` alone;
   - its stages: `fwd_position`, `fwd_velocity`, `fwd_actuation`, `fwd_acceleration`, `solver.solve`, the integrator.
   - Dispatches per replay from the Warp fork's counters, so microseconds per dispatch show whether the stage is
     dispatch-bound.
3. **Tiled `bsr_mv` unroll** (`probes/tiled_mv_unroll_probe.py`): Warp's tiled kernel vs unrolled-by-4 and by-8 copies
   (same order) on two-way's IJtm shape. Bitwise check and microseconds per launch.
4. **GPU busy-ness:**
   - `ioreg` device utilization sampled at 5 Hz during two-way and ANYmal timing runs (`probes/gpu_busy_sampler.py`);
   - kernel-busy fraction = calibrated sum of isolated kernel GPU time / wall time per frame (from the per-kernel
     profiles).
   - Package power needs `sudo powermetrics`. The batch prints the command for the user to run alongside, if they want.

# Phase log

Machine: Apple M4 Max, 40-core GPU, 64 GB, macOS 26.7, Command Line Tools only (no Xcode.app).
Python 3.12 venv at `.venv`. Labels: **measured** (this machine), **reported**, **estimate**.

## Phase 0: Measure (2026-09-22) — done

Reproduced mjbatch-metal's baselines with its own scripts (measured):

| workload | published (mjbatch-metal) | reproduced here |
|---|---|---|
| CPU MuJoCo + batched render, cartpole-class, 64 px, N=1024 | 52,600 | 49,595 steps/s |
| same, N=4096 | 53,100 | 53,673 steps/s |
| CPU MuJoCo + batched render, SO-101 lift scene, 128 px, N=64 | 16,957 | 18,040 steps/s |

The SO-101 lift scene is a reconstruction (`assets/so101/scene_box_rl.xml`: Menagerie
robotstudio_so101 + box + `base_cam`); the original lives on another machine.

Not done: Metal System Trace and powermetrics profiling (need Xcode Instruments / sudo).

## Phase 1: GPU physics + interop — in progress

### GPU physics (WS2), measured

MuJoCo Warp 3.14 (metal patch) on the innate-inc Warp fork, `mjwarp-testspeed`, 200 steps,
`--overflow_behavior continue` (the SO-101 scene hits `ls_iterations=20` in a few worlds):

| scene | N | realtime factor | steps/s (= factor / dt) |
|---|---|---|---|
| SO-101 lift (dt 0.005, nv 12, 50 geoms) | 64 | 227× | ~45,000 |
| | 256 | 812× | ~162,000 |
| | 1024 | 2,655× | ~531,000 |
| | 4096 | 5,319× | ~1,064,000 |
| mjwarp humanoid benchmark (dt 0.005, nv 27) | 1024 | 2,084× | ~417,000 |
| | 4096 | 4,711× | ~942,000 |

For scale (reported): MJX humanoid on an A100 at batch 8192 is 950K steps/s; MuJoCo Playground
CartpoleBalance on an A100 is 719K. The plan estimated "low hundreds of thousands" for the
Apple GPU; the measurement is several times higher for these scenes.

MuJoCo Warp test suite on `metal:0`: 1447 passed, 3 failed (2 flex-constraint parity tests,
1 known upstream), 39 skipped. The flex failures are deformable-body tests and are logged for
follow-up (deformables are a later workstream).

### Interop (WS3), measured

- Added to the Warp Metal runtime (fork branch `orchard-interop`): device/queue/buffer
  handles, foreign event wait and signal.
- `orchard.interop`: zero-copy MPS tensors over Warp arrays (DLPack `kDLMetal` and
  `at::from_blob`), event wait/signal on torch's command buffer.
- `tests/test_interop.py`: 11 tests pass, including proofs that waits hold the GPU and
  double-buffered producer/consumer ordering in both directions with no host sync in the loop.
- Ping-pong round trip (1M-element ops, both directions each step): 0.46 ms/step host time.

Lesson recorded: Warp's `zeros()`/memsets are asynchronous on its queue; any torch write to an
aliased buffer must be ordered after them (event or sync), otherwise the memset lands later.

### Remaining for Phase 1

- Batched environment on MuJoCo Warp with graph-captured substeps, obs/action tensors aliased
  to MPS, event-ordered handoff (`orchard.physics`).
- Trajectory parity test vs CPU MuJoCo on SO-101 and Menagerie robots at MJWarp tolerances.
- "Zero host copies per step" conformance: without Instruments, verify by host-time scaling
  (host time per step must not grow with GPU work).

### Graph replay through indirect command buffers (WS2), measured

Warp's Metal graph replay re-encoded every recorded dispatch on the CPU (~4 µs each; a MuJoCo
Warp step of the SO-101 scene is 201 dispatches), so small batches were host-bound. The fork now
encodes a captured graph once into an `MTLIndirectCommandBuffer` (barrier between commands) and
replays it with one call. SO-101 lift, 4 substeps per step, `tests/test_physics.py`:

| N | host time / step before | after | wall / step (GPU-bound) | physics steps/s |
|---|---|---|---|---|
| 64 | 3.4 ms | 0.08 ms | 5.0 ms | 51,000 |
| 2048 | 18.6 ms (queue back-pressure) | 0.02 ms | 16.7 ms | 491,000 |

Runtime counters per step: 804 dispatches, 2 command buffers, 0 host ops, 0 host waits.
Remaining floor at small N: ~6 µs of GPU time per kernel dispatch (Metal's per-dispatch cost
with barriers), i.e. ~1.2 ms per physics substep regardless of batch size. That is a per-kernel
launch floor, the Metal analogue of CUDA kernel launch latency; reducing it needs fewer, fused
kernels in MuJoCo Warp itself (upstream work), not this stack.

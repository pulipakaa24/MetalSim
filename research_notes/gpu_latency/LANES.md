# Ordering lanes: running independent GPU work concurrently on Metal (design, 2026-10-13)

**Goal.** Let independent work run at the same time on the M4 Max without changing a single result. The first target
is ANYmal on sand: the robot affects the sand but not the reverse, so robot step k+1 does not depend on sand step k.
- batch_lat1 measured the robot graph at 6.6 ms and the sand step at 22.2 ms per frame.
- Ideal overlap: 29.4 -> about 22-23 ms per frame. The L4 takes 24.2.

## How Metal orders work today in the Warp fork

From `warp/native/metal.mm` at pin `31cf79d7`:
- **One `MTLCommandQueue` per device**, holding untracked buffers (kernels use raw GPU addresses).
- **One event chain.** Every command buffer waits for `dev.event` to reach the previous buffer's value
  (`get_encoder`) and signals the next value when committed (`flush`). That chain is the only reason consecutive
  command buffers don't overlap: "Buffers are untracked ..., so consecutive command buffers would overlap on the GPU;
  each one waits for the previous one through this event."
- Encoders are serial, so dispatches inside a buffer run in order.
- **Interop already exists:**
  - `wp_metal_event_handle` / `wp_metal_event_value`: the device's chain;
  - `wp_metal_signal_event(event, value)`: signal an external event after the work so far;
  - `wp_metal_wait_event(event, value)`: later work waits for an external event.
- **Host waits are global.** `synchronize()` waits for every in-flight buffer.
- **Frees are tied to completion order.**
  - Deferred frees ride on the next committed buffer and are released when it completes (`pending_frees`). That is
    only correct because buffers on the one chain complete in commit order.
  - The args ring wraps with a full synchronize.
- Python's `wp.Stream` / `wp.Event` are CUDA-only (`warp/_src/context.py`, `Stream.__init__`: "is not a CUDA device").

## Metal semantics this relies on

- Command buffers on one queue may overlap on the GPU when nothing orders them (untracked resources). Within a buffer,
  serial compute encoders run dispatches in order.
- An `MTLEvent` (same device) orders command buffers: `encodeSignalEvent:value:` / `encodeWaitForEvent:value:` (Apple,
  "Synchronizing events within a single device"). The wait is done by the GPU's command scheduler, not by a spinning
  kernel.
  - A wait on a value that is never signalled leaves that command buffer pending. That is a hang hazard for the queue,
    so every cross-lane wait must be paired with a signal that is already committed.
  - The prototype uses no cross-lane waits at all.
- Prior art for concurrency on Metal (NOTE.md, verified in source): MLX and llama.cpp use concurrent dispatch inside one
  encoder with tracked barriers. That is the finer-grained option 2. Lanes are coarser: independent chains of command
  buffers, the Metal analogue of two CUDA streams with event dependencies (Warp's `wp.Stream`, `wp.Event`,
  `stream.wait_event`).

## Design (Warp fork, `metal.mm`; patch `runs/mpm/latency/warp_lanes.diff`)

- `Device::lanes`: per lane, its own `MTLEvent` and value. `dev.event` / `dev.event_value` stay the current lane's
  chain, so every existing code path (eager launches, graph replay, memcpy, the interop calls) works per lane unchanged.
- `wp_metal_select_lane(ordinal, lane)`:
  - commits the current lane's open buffer;
  - parks its chain and makes `lane` current (a new lane gets a new `MTLEvent` at 0);
  - refused during capture.
- Cross-lane order: `wp_metal_event_handle` / `wp_metal_event_value` read on lane A, then `wp_metal_wait_event` on lane
  B. That is the existing interop.
- **Frees:** once a second lane has been used (`multi_lane`), deferred frees are released only at a full synchronize.
  Buffers on different lanes complete out of commit order.
- **Host waits stay global** (correct for any readback). Consequence: in the prototype, B's robot work is submitted
  after A's sand step has done its host reads (grid build). It is submitted from a hook at the start of the solver loop,
  so it overlaps the GPU-gated solve, which has no host reads.
- **Bitwise:** the same kernels run, in the same order within each lane, on data no other lane touches.

## Prototype (`probes/lanes_overlap_probe.py`, `runs/mpm/latency/batch_lanes1.sh`)

- Two independent ANYmal instances, A (sand only) and B (robot only). The modes are seq (lane 0 only) and conc (B on
  lane 1 during A's solve).
- It reports ms per frame and checks that both scenes' per-frame state hashes are equal between modes.
- `tests/test_mpm.py` also runs on the patched build (lane 0 only: the default path must be unchanged).
- If it gains, the real integration in one scene is:
  1. Hand the robot's body poses to the sand through a copy written at the end of robot step k (lane 1). The sand
     reads the copy, and robot step k+1 then writes its own state while the sand runs.
  2. Check that the MPM step writes nothing the robot reads.
  3. Put the robot graph on lane 1 one frame ahead, with an event wait so that sand step k+1 waits for robot step k+1's
     copy.

## Option 2 (later)

Concurrent dispatch inside an encoder, with buffer or range hazard tracking (MLX / ggml pattern). It needs per-argument
read/write sets in Warp's launch; see NOTE.md.

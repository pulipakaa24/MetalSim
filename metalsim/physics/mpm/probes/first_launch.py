"""First launches of changed MPM kernels: WORLDS worlds, eager steps, every step's GPU completion awaited from a
worker thread with a TIMEOUT-second limit (exit at once on timeout, without waiting for the GPU). The first step
(module compilation on the host) gets COMPILE_TIMEOUT.   tiny_step.py [WORLDS] [STEPS] [TIMEOUT] [MODE]"""
import os, sys, threading, time
W = int(sys.argv[1]) if len(sys.argv) > 1 else 1
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 5
TIMEOUT = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0
MODE = sys.argv[4] if len(sys.argv) > 4 else "eager"
COMPILE_TIMEOUT = 300.0
HERE = os.path.dirname(os.path.abspath(__file__))
sys.argv = [HERE + "/pour_bitwise.py", "record", "--worlds", str(W), "--mode", MODE]
sys.path.insert(0, HERE)
import numpy as np
import pour_bitwise as pb
sp, wp = pb.sp, pb.wp


# Host-side module builds (kernels compiled on first use) do not count against the GPU completion timeout
from warp._src.context import Module
LOAD_S = [0.0]
_orig_load = Module.load
IN_LOAD = [0]
def _timed_load(self, *a, **k):
    t0 = time.perf_counter()
    IN_LOAD[0] += 1
    try:
        return _orig_load(self, *a, **k)
    finally:
        IN_LOAD[0] -= 1
        LOAD_S[0] += time.perf_counter() - t0
Module.load = _timed_load


def guarded(fn, what, timeout):
    done = threading.Event(); err = []
    def work():
        try:
            fn(); wp.synchronize_device()
        except BaseException as e:  # noqa: BLE001
            err.append(e)
        done.set()
    threading.Thread(target=work, daemon=True).start()
    t0 = time.perf_counter()
    load0 = LOAD_S[0]
    while not done.wait(0.1):
        if IN_LOAD[0]:
            continue  # a module build on the host is in progress
        waited = time.perf_counter() - t0 - (LOAD_S[0] - load0)
        if waited > timeout:
            print(f"TIMEOUT: {what} did not complete within {timeout} s (excluding {LOAD_S[0] - load0:.1f} s of "
                  "module builds); exiting without waiting", flush=True)
            os._exit(3)
    if err:
        raise err[0]
    return time.perf_counter() - t0


os.environ["NEWTON_METAL_LISTED_ROWS_MV"] = "0"; os.environ["NEWTON_METAL_RIGIDITY_COMPACT"] = "0"
BOUNDED = wp.config.metal_bounded_launch
wp.config.metal_bounded_launch = False
with wp.ScopedDevice(pb.DEV):
    model, solver, src, bpw = pb.make_solver()
    state = model.state()
    rng = np.random.default_rng(1)
    phase = wp.array((rng.random(W) * 1.5 - 0.6).astype(np.float32), dtype=float)
    body_idx = wp.array(np.arange(W, dtype=np.int32) * bpw + src, dtype=int)
    base = wp.array(np.asarray(model.body_q.numpy())[body_idx.numpy(), :3].astype(np.float32), dtype=wp.vec3)
    t = [0.0]
    def step():
        wp.launch(sp.drive_source_cup, dim=W, inputs=[t[0], phase, body_idx, base, state.body_q, state.body_qd])
        solver.step(state, state, None, None, sp.MPM_DT)
        t[0] += sp.MPM_DT
    # step 0 compiles every module (whole modules, so the changed kernels too) with the changed paths off
    guarded(step, "warm-up step (changed paths off)", COMPILE_TIMEOUT)
    os.environ["NEWTON_METAL_LISTED_ROWS_MV"] = "1"; os.environ["NEWTON_METAL_RIGIDITY_COMPACT"] = "1"
    wp.config.metal_bounded_launch = BOUNDED
    print("warm-up done", flush=True)
    for i in range(STEPS):
        dt = guarded(step, f"step {i}", TIMEOUT)
        print(f"step {i} completed in {dt*1e3:.1f} ms", flush=True)
    print("finite", bool(np.isfinite(state.particle_q.numpy()).all()), "OK", flush=True)

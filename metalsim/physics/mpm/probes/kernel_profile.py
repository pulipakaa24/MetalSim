"""Per-kernel GPU time of an MPM scene on Metal, and the dispatch structure of the timed configuration.

Two passes over one process (both after the same warm-up):

1. ``--mode profile`` (run with WP_METAL_PROFILE=1): eager stepping with the CUDA stopping rule, every dispatch in its
   own command buffer; per kernel name the summed GPU time (GPUEndTime - GPUStartTime) and launch count per frame.
   Each dispatch then runs alone on an idle GPU: its "isolated" time is its critical path plus the command-buffer
   start/finish cost, which ``--calib`` measures with an empty one-thread kernel (subtract it per launch).
2. ``--mode timing``: the timing-class configuration (chunk graph + GPU exit, or the example's whole-step capture
   with ``--set capture=true``): wall ms/frame and Metal runtime counters per frame (dispatches encoded, command
   buffers committed, host waits).

    WP_METAL_PROFILE=1 PYTHONPATH=. \
        .venv-newton152/bin/python metalsim/physics/mpm/probes/kernel_profile.py example:mpm_granular --mode profile
"""
import argparse, ctypes, json, os, re, time

ap = argparse.ArgumentParser()
ap.add_argument("scene")
ap.add_argument("--mode", choices=["profile", "timing"], default="profile")
ap.add_argument("--device", default="metal:0")
ap.add_argument("--frames", type=int, default=6)
ap.add_argument("--warmup", type=int, default=3)
ap.add_argument("--set", action="append", default=[])
ap.add_argument("--json")
ap.add_argument("--calib", type=int, default=2000, help="empty-kernel command buffers for the overhead calibration")
args = ap.parse_args()

import numpy as np
import warp as wp

wp.config.quiet = True
wp.init()
from warp._src.context import runtime
from metalsim.physics.mpm import newton_compat, scenes

profiling = os.environ.get("WP_METAL_PROFILE") is not None
if args.mode == "profile" and not profiling:
    raise SystemExit("--mode profile needs WP_METAL_PROFILE=1")
if args.mode == "timing" and profiling:
    raise SystemExit("--mode timing must run without WP_METAL_PROFILE")

core = runtime.core
core.wp_metal_counters.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_uint64), ctypes.c_int]
core.wp_metal_counters.restype = ctypes.c_int


def counters():
    out = (ctypes.c_uint64 * 4)()
    core.wp_metal_counters(wp.get_device(args.device).metal_ordinal, out, 4)
    return np.array(list(out), dtype=np.int64)  # n_sync, n_flush, n_host_ops, n_dispatch


def report():
    rows = []
    for line in core.wp_metal_profile_report().decode().splitlines():
        m = re.match(r"\s*([0-9.]+) ms\s+(\d+)\s+(.*)$", line)
        if m:
            rows.append((m.group(3), float(m.group(1)), int(m.group(2))))
    return rows


@wp.kernel
def _empty(a: wp.array[float]):
    i = wp.tid()
    if i < 0:
        a[0] = 1.0


timing = args.mode == "timing"
newton_compat.install(accumulate="float32", loop="cuda", log=True, chunk_graph=timing, gpu_exit=timing)
kw = {k: json.loads(v) for k, v in (s.split("=", 1) for s in args.set)}
result = {"scene": args.scene, "mode": args.mode, "set": kw, "frames": args.frames}
with wp.ScopedDevice(args.device):
    sc = scenes.make_scene(args.scene, **kw)
    for _ in range(args.warmup):
        sc.step()
    wp.synchronize_device()
    newton_compat.drain_log()
    if profiling:
        report()  # clears the warm-up totals
    c0 = counters()
    walls = []
    for _ in range(args.frames):
        wp.synchronize_device()
        t0 = time.perf_counter()
        sc.step()
        wp.synchronize_device()
        walls.append((time.perf_counter() - t0) * 1e3)
    c1 = counters()
    log = newton_compat.drain_log()
    n = args.frames
    result.update(
        particles=int(sc.model.particle_count),
        wall_ms=walls, wall_median_ms=float(np.median(walls)),
        iterations=log["iterations"].tolist(),
        iterations_per_frame=float(log["iterations"].sum() / n) if log["iterations"].size else None,
        per_frame_counters=dict(zip(["host_waits", "command_buffers", "host_ops", "dispatches_encoded"],
                                    ((c1 - c0) / n).tolist())),
    )
    if profiling:
        rows = report()
        result["kernels"] = [{"kernel": k, "ms_per_frame": ms / n, "launches_per_frame": c / n} for k, ms, c in rows]
        result["kernel_total_ms_per_frame"] = sum(ms for _, ms, _ in rows) / n
        result["launches_per_frame"] = sum(c for _, _, c in rows) / n
        # calibration: an empty one-thread kernel and a 5120-thread kernel, each in its own command buffer
        a = wp.zeros(1, dtype=float)
        for dim in (1, 5120):
            for _ in range(50):
                wp.launch(_empty, dim=dim, inputs=[a])
            wp.synchronize_device()
            report()
            for _ in range(args.calib):
                wp.launch(_empty, dim=dim, inputs=[a])
            wp.synchronize_device()
            cal = [r for r in report() if "_empty" in r[0]]
            result[f"empty_cb_us_dim{dim}"] = cal[0][1] * 1e3 / cal[0][2] if cal else None

print(json.dumps({k: v for k, v in result.items() if k != "kernels"}, indent=1))
if "kernels" in result:
    for r in result["kernels"][:40]:
        print(f"{r['ms_per_frame']:9.3f} ms {r['launches_per_frame']:8.1f}  {r['kernel'][:110]}")
if args.json:
    os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
    with open(args.json, "w") as fh:
        json.dump(result, fh, indent=1)

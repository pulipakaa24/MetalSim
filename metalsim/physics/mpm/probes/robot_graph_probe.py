"""ANYmal's robot side on Metal by subtraction (research_notes/gpu_latency): one substep's parts captured as separate
graphs and replayed back to back, with the Warp fork's runtime counters per replay.

After --warmup frames of example_mpm_anymal (the example's own capture), captures and replays (--reps times, one
synchronize at the end, so the replays queue like the example's):
  robot_graph     the example's robot graph (4 substeps: SolverMuJoCo.step each)
  newton_step     one SolverMuJoCo.step (Newton <-> MuJoCo Warp conversions + mujoco_warp.step)
  mjw_step        mujoco_warp.step alone
  fwd_position, fwd_velocity, fwd_actuation, fwd_acceleration, solve, integrate   its stages
Reports us per replay and dispatches per replay (counter delta / reps). Mutates the simulation (timing only).

    PYTHONPATH=<warp>:<newton>:<metalsim> python robot_graph_probe.py --json out.json
"""
import argparse
import ctypes
import json
import os
import time

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="metal:0")
ap.add_argument("--warmup", type=int, default=5)
ap.add_argument("--reps", type=int, default=200)
ap.add_argument("--json")
args = ap.parse_args()

import numpy as np
import warp as wp

wp.config.quiet = True
wp.init()
from warp._src.context import runtime

from metalsim.physics.mpm import newton_compat, scenes

core = runtime.core
core.wp_metal_counters.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_uint64), ctypes.c_int]
core.wp_metal_counters.restype = ctypes.c_int


def counters():
    out = (ctypes.c_uint64 * 4)()
    core.wp_metal_counters(wp.get_device(args.device).metal_ordinal, out, 4)
    return np.array(list(out), dtype=np.int64)  # host waits, command buffers, host ops, dispatches encoded


newton_compat.install(accumulate="float32", loop="cuda", log=False, chunk_graph=True, gpu_exit=True)
res = {"reps": args.reps, "parts": {}}
with wp.ScopedDevice(args.device):
    sc = scenes.make_scene("example:mpm_anymal", capture=True)
    ex = sc.example
    for _ in range(args.warmup):
        sc.step()
    wp.synchronize_device()
    solver = ex.solver
    mjw = solver._mujoco_warp
    m, d = solver.mjw_model, solver.mjw_data
    from mujoco_warp._src import forward as F  # noqa: E402
    from mujoco_warp._src import solver as S  # noqa: E402
    from mujoco_warp._src.types import IntegratorType  # noqa: E402

    integ = int(m.opt.integrator)
    if integ == int(IntegratorType.EULER):
        integrate = lambda: F.euler(m, d)  # noqa: E731
    elif integ in (int(IntegratorType.IMPLICITFAST), int(IntegratorType.IMPLICIT)):
        integrate = lambda: F.implicit(m, d)  # noqa: E731
    else:
        integrate = lambda: F.rungekutta4(m, d)  # noqa: E731
    parts = {
        "newton_step": lambda: solver.step(ex.state_0, ex.state_1, ex.control, None, ex.sim_dt),
        "mjw_step": lambda: mjw.step(m, d),
        "fwd_position": lambda: F.fwd_position(m, d, factorize=False),
        "fwd_velocity": lambda: F.fwd_velocity(m, d),
        "fwd_actuation": lambda: F.fwd_actuation(m, d),
        "fwd_acceleration": lambda: F.fwd_acceleration(m, d, factorize=True),
        "solve": lambda: S.solve(m, d),
        "integrate": integrate,
    }
    graphs = {"robot_graph": ex.graph}
    for name, fn in parts.items():
        try:
            fn()
            wp.synchronize_device()
            with wp.ScopedCapture(force_module_load=False) as cap:
                fn()
            graphs[name] = cap.graph
        except Exception as e:  # noqa: BLE001
            res["parts"][name] = {"error": repr(e)[:300]}
    for name, g in graphs.items():
        if g is None:
            res["parts"][name] = {"error": "no graph"}
            continue
        for _ in range(5):
            wp.capture_launch(g)
        wp.synchronize_device()
        best = None
        for _trial in range(3):
            c0 = counters()
            t0 = time.perf_counter()
            for _ in range(args.reps):
                wp.capture_launch(g)
            wp.synchronize_device()
            dt = (time.perf_counter() - t0) * 1e6 / args.reps
            c = (counters() - c0) / args.reps
            if best is None or dt < best[0]:
                best = (dt, c)
        res["parts"][name] = {"us_per_replay": best[0], "dispatches_per_replay": float(best[1][3]),
                              "command_buffers_per_replay": float(best[1][1]), "host_waits_per_replay": float(best[1][0]),
                              "us_per_dispatch": best[0] / best[1][3] if best[1][3] else None}
        print(f"{name:18s} {best[0]:9.1f} us/replay  {best[1][3]:7.1f} dispatches  "
              f"{(best[0] / best[1][3]) if best[1][3] else float('nan'):6.2f} us/dispatch", flush=True)
    stages = [p for p in ("fwd_position", "fwd_velocity", "fwd_actuation", "fwd_acceleration", "solve", "integrate")
              if "us_per_replay" in res["parts"].get(p, {})]
    res["stage_sum_us"] = float(sum(res["parts"][p]["us_per_replay"] for p in stages))
    print("stage sum", round(res["stage_sum_us"], 1), "us; mjw_step", res["parts"].get("mjw_step", {}).get("us_per_replay"))
if args.json:
    os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
    json.dump(res, open(args.json, "w"), indent=1)

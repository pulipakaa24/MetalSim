"""Per-control-step cost of BatchSim's capacity-overflow latch (one dim-N launch after every substep) in the G1 flat task
at its default capacities: the same task timed with the latch hook attached (A) and detached (B), interleaved A/B
repeats, synchronized env-steps/s over benchmark_step's protocol (random actions, full env.step graph).

    bash scripts/gpu_run.sh overflow_latch_cost timing 10 -- .venv/bin/python scripts/diagnostics/overflow_latch_cost.py --envs 4096
"""
import argparse
import os
import sys
import time

import numpy as np
import warp as wp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))   # this checkout's metalsim
from metalsim.learn.g1_velocity import G1VelocityTask, benchmark_step  # noqa: E402


def rate(task, frames):
    benchmark_step(task, num_frames=10, warmup=0)                    # warm-up / capture
    t0 = time.perf_counter(); benchmark_step(task, num_frames=frames, warmup=0); task.sim.synchronize()
    return frames * task.n / (time.perf_counter() - t0)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--envs", type=int, default=4096); ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--repeats", type=int, default=5)
    a = ap.parse_args(); wp.config.quiet = True
    task = G1VelocityTask(a.envs, terrain="flat", seed=0, physics_dt=0.0025)
    sim = task.sim
    print("njmax", sim.d.njmax, "naconmax / world", sim.d.naconmax // a.envs, "cone", sim.mj_model.opt.cone, "decimation", task.decimation)
    full = list(sim._substep_hooks)
    without = [h for h in full if getattr(h, "__func__", None) is not type(sim)._latch_overflow]
    assert len(without) == len(full) - 1

    def attach(on):
        sim._substep_hooks[:] = full if on else without
        with wp.ScopedDevice(task.device):
            sim._capture()
    res = {"latch": [], "none": []}
    for r in range(a.repeats):
        for on in (True, False):
            attach(on)
            res["latch" if on else "none"].append(rate(task, a.frames))
    for k, v in res.items():
        print(f"{k:5s}: {np.median(v):,.0f} env-steps/s (median of {len(v)}: {', '.join(f'{x:,.0f}' for x in v)}); "
              f"{1e3 * a.envs / np.median(v):.2f} ms per control step")
    d = 1e3 * a.envs / np.median(res["latch"]) - 1e3 * a.envs / np.median(res["none"])
    print(f"N={a.envs}: latch costs {d:+.3f} ms per control step ({100 * (np.median(res['none']) / np.median(res['latch']) - 1):+.2f} %)")


if __name__ == "__main__":
    main()

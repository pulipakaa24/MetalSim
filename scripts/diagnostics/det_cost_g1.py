"""Cost of BatchSimOptions(deterministic=True) on the G1 flat-terrain velocity task (one process per mode: the mode is
process-wide). env step = actions, physics (decimation substeps), observations / rewards / resets, one captured graph
per step (``metalsim.learn.g1_velocity.benchmark_step``, mean wall time per step over --frames after --warmup).

    python -P det_cost_g1.py --worlds 64 --deterministic 1 --json out.json
"""
import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", type=int, default=64)
    ap.add_argument("--deterministic", type=int, default=0)
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--json")
    a = ap.parse_args()
    import warp as wp

    wp.config.quiet = True
    from metalsim.learn.g1_velocity import G1VelocityTask, benchmark_step

    t0 = time.perf_counter()
    task = G1VelocityTask(a.worlds, terrain="flat", batch_options={"deterministic": bool(a.deterministic)})
    build_s = time.perf_counter() - t0
    r = benchmark_step(task, num_frames=a.frames, warmup=a.warmup)
    task.sim.check_overflow()
    res = {"scene": "g1_flat", "worlds": a.worlds, "deterministic": a.deterministic, "build_s": build_s, **r}
    print(json.dumps(res))
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()

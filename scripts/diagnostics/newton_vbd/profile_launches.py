"""Per-kernel wall time of a Newton VBD step (eager, synchronizing after every launch): which kernel scales badly.

    PYTHONPATH=upstream/newton-1.5.2 .venv-newtonfork/bin/python scripts/diagnostics/newton_vbd/profile_launches.py \
        --device metal:0 --scene cube --worlds 256 1024
"""
import argparse, collections, json, os, sys, time

import warp as wp

sys.path.insert(0, os.path.dirname(__file__))
import vbd_probe  # noqa: E402
import newton  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="metal:0")
    ap.add_argument("--scene", default="cube")
    ap.add_argument("--worlds", type=int, nargs="+", default=[256, 1024])
    ap.add_argument("--steps", type=int, default=2)
    a = ap.parse_args()
    wp.config.quiet = True
    orig = wp.launch
    for nw in a.worlds:
        with wp.ScopedDevice(a.device):
            m = vbd_probe.cloth_model(worlds=nw) if a.scene == "cloth" else vbd_probe.cube_model(worlds=nw)
            solver = newton.solvers.SolverVBD(m, iterations=10)
            s0, s1, ctrl = m.state(), m.state(), m.control()
            pipe = newton.CollisionPipeline(m)
            contacts = pipe.contacts()
            dt = 1.0 / 2000.0

            def sub():
                nonlocal s0, s1
                s0.clear_forces()
                pipe.collide(s0, contacts)
                solver.step(s0, s1, ctrl, contacts, dt)
                s0, s1 = s1, s0
            sub(); wp.synchronize_device()             # compile
            times = collections.Counter(); counts = collections.Counter()

            def timed(kernel, *args, **kw):
                wp.synchronize_device()
                t = time.perf_counter()
                r = orig(kernel, *args, **kw)
                wp.synchronize_device()
                key = getattr(kernel, "key", str(kernel))
                times[key] += time.perf_counter() - t
                counts[key] += 1
                return r
            wp.launch = timed
            try:
                t0 = time.perf_counter()
                for _ in range(a.steps * 10):
                    sub()
                wp.synchronize_device()
                total = time.perf_counter() - t0
            finally:
                wp.launch = orig
            top = [(k, round(1000 * v / (a.steps * 10), 3), counts[k] // (a.steps * 10)) for k, v in times.most_common(8)]
            print(json.dumps({"scene": a.scene, "worlds": nw, "ms_per_substep": round(1000 * total / (a.steps * 10), 2),
                              "top_kernels_ms_per_substep_launches": top}), flush=True)


if __name__ == "__main__":
    main()

"""Isaac Lab's MPM -> rigid coupling harvest (``SolverImplicitMPM.coupling_harvest_proxy_wrenches``, called by Newton's
``SolverCoupledProxy`` for every proxy body each coupled step) on the Franka-Pour scene at N worlds: the float-atomic
kernel against the fixed-order per-world sum, for agreement, run-to-run repeatability and cost.

    PYTHONPATH=<warp>:<newton>:. python metalsim/physics/mpm/probes/harvest_check.py --worlds 2048 --json out.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--worlds", type=int, default=64)
ap.add_argument("--steps", type=int, default=200, help="MPM steps before harvesting (flowing media in some worlds)")
ap.add_argument("--repeats", type=int, default=20)
ap.add_argument("--json")
args = ap.parse_args()

HERE = os.path.dirname(os.path.abspath(__file__))
sys.argv = [os.path.join(HERE, "pour_bitwise.py"), "record", "--worlds", str(args.worlds), "--mode", "gpuexit",
            "--steps", "0"]
sys.path.insert(0, HERE)
import pour_bitwise as pb  # noqa: E402

wp = pb.wp
newton = pb.sp.newton


def main():
    res = {"worlds": args.worlds}
    with wp.ScopedDevice(pb.DEV):
        model, solver, src, bpw = pb.make_solver()
        state = model.state()
        rng = np.random.default_rng(1)
        phase = wp.array((rng.random(args.worlds) * 1.5 - 0.6).astype(np.float32), dtype=float)
        body_idx = wp.array(np.arange(args.worlds, dtype=np.int32) * bpw + src, dtype=int)
        base = wp.array(np.asarray(model.body_q.numpy())[body_idx.numpy(), :3].astype(np.float32), dtype=wp.vec3)
        t = 0.0
        for _ in range(args.steps):
            wp.launch(pb.sp.drive_source_cup, dim=args.worlds,
                      inputs=[t, phase, body_idx, base, state.body_q, state.body_qd])
            solver.step(state, state, None, None, pb.sp.MPM_DT)
            t += pb.sp.MPM_DT
        wp.synchronize_device()

        # proxies: every dynamic or kinematic cup of every world (Isaac's CouplerProxyMappingCfg: source and target cups)
        flags = model.body_flags.numpy()
        proxy = int(newton.BodyFlags.PROXY)
        flags[:] = flags | proxy
        model.body_flags.assign(flags)
        local_to_global = wp.array(np.arange(model.body_count, dtype=np.int32), dtype=int)
        out = wp.zeros(model.body_count, dtype=wp.spatial_vector)

        def harvest(ordered: bool):
            solver._ordered_proxy_harvest = ordered
            solver.coupling_harvest_proxy_wrenches(local_to_global, out, body_qd_before=None, state=state,
                                                   state_out=state, contacts=None, dt=pb.sp.MPM_DT)

        for name, ordered in (("atomic", False), ("ordered", True)):
            harvest(ordered)
            wp.synchronize_device()
            runs = []
            for _ in range(args.repeats):
                harvest(ordered)
                runs.append(out.numpy().copy())
            ts = []
            for _ in range(10):
                wp.synchronize_device()
                t0 = time.perf_counter()
                harvest(ordered)
                wp.synchronize_device()
                ts.append((time.perf_counter() - t0) * 1e3)
            res[name] = {
                "ms_median": float(np.median(ts)),
                "distinct_results": len({r.tobytes() for r in runs}),
                "nonzero_bodies": int(np.count_nonzero(np.abs(runs[0]).sum(axis=1))),
            }
            res[name + "_result"] = runs[0]
        a, o = res.pop("atomic_result"), res.pop("ordered_result")
        scale = np.abs(a).max()
        res["max_abs_diff_atomic_vs_ordered"] = float(np.abs(a - o).max())
        res["max_rel_diff"] = float(np.abs(a - o).max() / scale) if scale > 0 else 0.0
        res["collider_nodes"] = int(solver._scratchpad.collider_node_count)
    print(json.dumps(res, indent=1))
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()

"""Default-mode G1: does MuJoCo Warp's fused Newton update (MetalSim fork, Metal) change convergence?
(research_notes/MPM in main simulator/deterministic_g1_capture.md section 9; batch_22)

One arm per process (MJW_METAL_FUSE_UPDATE is read at import): G1 flat velocity task, captured BatchSim in DEFAULT
mode, random actions from torch seed 0, per step: worlds that hit the solver iteration cap (OverflowType.ITERATIONS,
cleared per step), d.solver_niter per world (the last substep), qpos per world (saved to npz). Also whether the fused
kernel's module actually ran (else the comparison is vacuous).

    python -P fused_cap_check.py --worlds 64 --steps 60 --out runs/.../A      (writes A.json, A.npz)
"""
import argparse
import json
import pathlib
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", type=int, default=64)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", required=True, help="the MetalSim checkout")
    a = ap.parse_args()
    sys.path.insert(0, a.root)
    import os

    import torch
    import warp as wp

    wp.config.quiet = True
    import mujoco_warp as mjw
    from warp._src import context

    from metalsim.interop import torch_bridge as tb
    from metalsim.learn.g1_velocity import G1VelocityTask

    dev = wp.get_device("metal:0")
    task = G1VelocityTask(a.worlds, terrain="flat", batch_options={"deterministic": False, "capture": True})
    sim = task.sim
    cap = int(sim.m.opt.iterations)
    action = wp.zeros((a.worlds, task.act_dim), dtype=float, device=dev)
    t_action = tb.mps_tensor(action)
    torch.manual_seed(0)
    task.reset_all()
    ITER = int(mjw.OverflowType.ITERATIONS)
    sim.synchronize()
    sim.d.overflow.zero_()
    cap_worlds, niter, qpos = [], [], []
    for _ in range(a.steps):
        t_action.uniform_(-1.0, 1.0)
        v = sim.event.next_value(); tb.signal_event(sim.event, v); sim.wait(sim.event, v)
        with wp.ScopedDevice(dev):
            task.launch_apply_action(action)
        sim.step()
        sim.synchronize()
        ov = sim.d.overflow.numpy()
        cap_worlds.append(int(((ov & ITER) != 0).sum()))
        sim.d.overflow.zero_()
        niter.append(sim.d.solver_niter.numpy().astype(np.int32).copy())
        qpos.append(sim.d.qpos.numpy().astype(np.float64).copy())
    fused_ran = any("_update_constraint_gradient_fused" in n and m.execs for n, m in list(context.user_modules.items()))
    niter = np.stack(niter)
    res = {"worlds": a.worlds, "steps": a.steps, "iteration_cap": cap,
           "MJW_METAL_FUSE_UPDATE": os.environ.get("MJW_METAL_FUSE_UPDATE", "1"), "fused_kernel_ran": bool(fused_ran),
           "steps_with_cap_worlds": int(sum(c > 0 for c in cap_worlds)), "cap_world_steps": int(sum(cap_worlds)),
           "mean_niter": float(niter.mean()), "niter_hist": np.bincount(niter.ravel(), minlength=cap + 1).tolist(),
           "mujoco_warp": mjw.__file__}
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(out) + ".npz", qpos=np.stack(qpos), niter=niter, cap_worlds=np.array(cap_worlds))
    (pathlib.Path(str(out) + ".json")).write_text(json.dumps(res, indent=1))
    print("RESULT " + json.dumps(res))


if __name__ == "__main__":
    main()

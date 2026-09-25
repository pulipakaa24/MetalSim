"""Physics-only throughput of the G1 under each contact/limit tuning (metalsim.physics.contact_tuning),
MuJoCo Warp, 4096 worlds, 2.5 ms step, 8 substeps per control step (the replay/training setting),
10 Newton / 20 line-search iterations, nconmax 32, njmax 256. Each timing starts from the standing
keyframe with the default-pose PD targets and simulates 3 s (150 control steps: the stand, the forward
fall and the torso contact, as protocol A_hold), graph replay, synchronized; settings interleaved,
3 repeats, median and min reported. Prints the GPU queue status first.

    scripts/gpu_run.sh contact_tuning_bench timing 15 -- .venv/bin/python scripts/diagnostics/bench_contact_tuning.py default tau5_imp99 ...
    PYTHONPATH=<worktree of the fork before the plane_convex fix> ... (old collider)
"""
import datetime
import subprocess
import sys
import time

import numpy as np
import torch
import warp as wp

import mujoco_warp
from metalsim.learn.g1_velocity import build_g1_model
from metalsim.physics import contact_tuning
from metalsim.physics.batch import BatchSim, BatchSimOptions

N = 4096
DT = 0.0025
CALLS = 150
REPS = 3


def main():
    print("gpu_lock status at start", datetime.datetime.now().isoformat(timespec="seconds"), flush=True)
    print(subprocess.run(["python3", "scripts/gpu_lock.py", "status"], capture_output=True, text=True).stdout, flush=True)
    print("mujoco_warp from", mujoco_warp.__file__, flush=True)
    wp.config.quiet = True
    names = sys.argv[1:] or ["default", "tau5_imp99", "tau5_imp99_hardlimits"]
    sims = {}
    for nm in names:
        m, _ = build_g1_model("flat", physics_dt=DT)
        contact_tuning.apply(m, nm)
        sim = BatchSim(m, N, options=BatchSimOptions(substeps=8, njmax=256, nconmax=32, solver_iterations=10, ls_iterations=20))
        q = np.tile(m.key_qpos[0], (N, 1)).astype(np.float32)
        q[:, 0] = (np.arange(N) % 64) * 2.5; q[:, 1] = (np.arange(N) // 64) * 2.5
        ctrl = torch.as_tensor(np.tile(m.key_qpos[0][7:], (N, 1)).astype(np.float32))
        sims[nm] = (sim, q, ctrl, m)
    res = {nm: [] for nm in names}
    for rep in range(REPS):
        for nm in names:
            sim, q, ctrl, m = sims[nm]
            sim.set_state(q, np.zeros((N, m.nv), np.float32)); sim.t.ctrl.copy_(ctrl); torch.mps.synchronize()
            sim.synchronize(); t0 = time.perf_counter()
            for _ in range(CALLS):
                sim.step()
            sim.synchronize()
            res[nm].append((time.perf_counter() - t0) / CALLS)
            ov = sim.overflow_flags()
            if ov:
                print(f"  {nm}: overflow {ov}", flush=True)
    print(f"G1 MuJoCo Warp physics only, {N} worlds, {DT * 1e3} ms x 8 substeps per control step, 3 s from standing, {REPS} interleaved repeats")
    base = np.median(res[names[0]])
    for nm in names:
        v = np.array(res[nm]); med = np.median(v)
        print(f"  {nm:28s} {med * 1e3:8.2f} ms/control step (min {v.min() * 1e3:.2f})  {N / med:9.0f} env-steps/s  "
              f"{N * 8 / med:10.0f} 2.5ms-steps/s  ({med / base:.3f}x {names[0]})  runs {np.round(v * 1e3, 2).tolist()}", flush=True)


if __name__ == "__main__":
    main()

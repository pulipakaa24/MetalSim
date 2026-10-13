"""How often does the Franka-Pour sparse grid's occupancy stay the same from one MPM step to the next?

Runs pour_bitwise's solver (eager) and, per step, compares the set of (world, cell) keys of the particles with the
previous step's (the rebuildable grid's topology depends only on that set). Reports the steps whose whole set is
unchanged and, per step, how many worlds changed.   pour_occupancy.py WORLDS STEPS
"""
import json
import os
import sys

import numpy as np

W = int(sys.argv[1]); STEPS = int(sys.argv[2])
HERE = os.path.dirname(os.path.abspath(__file__))
sys.argv = [os.path.join(HERE, "pour_bitwise.py"), "record", "--worlds", str(W), "--mode", "eager", "--steps", "1"]
sys.path.insert(0, HERE)
import pour_bitwise as pb  # noqa: E402  (parses argv; does not run main)

sp, wp = pb.sp, pb.wp
with wp.ScopedDevice(pb.DEV):
    model, solver, src, bpw = pb.make_solver()
    state = model.state()
    rng = np.random.default_rng(1)
    phase = wp.array((rng.random(W) * 1.5).astype(np.float32), dtype=float)
    body_idx = wp.array(np.arange(W, dtype=np.int32) * bpw + src, dtype=int)
    base = wp.array(np.asarray(model.body_q.numpy())[body_idx.numpy(), :3].astype(np.float32), dtype=wp.vec3)
    pw = model.particle_world.numpy().astype(np.int64)
    prev = None
    same_all = 0
    per_world_same = np.zeros(W)
    changed_worlds = []
    t = 0.0
    for _ in range(STEPS):
        wp.launch(sp.drive_source_cup, dim=W, inputs=[t, phase, body_idx, base, state.body_q, state.body_qd])
        solver.step(state, state, None, None, sp.MPM_DT)
        t += sp.MPM_DT
        c = np.floor(state.particle_q.numpy() / sp.VOXEL).astype(np.int64)
        cur = np.unique((pw << 42) ^ ((c[:, 0] & 0x3FFF) << 28) ^ ((c[:, 1] & 0x3FFF) << 14) ^ (c[:, 2] & 0x3FFF))
        if prev is not None:
            same_all += int(cur.shape == prev.shape and np.array_equal(cur, prev))
            wc, wpv = cur >> 42, prev >> 42
            changed = 0
            for w in range(W):
                a, b = cur[wc == w], prev[wpv == w]
                eq = a.shape == b.shape and np.array_equal(a, b)
                per_world_same[w] += eq
                changed += not eq
            changed_worlds.append(changed)
        prev = cur
    n = len(changed_worlds)
    print(json.dumps({"worlds": W, "steps": n, "steps_all_same": same_all,
                      "per_world_same_frac_mean": float(per_world_same.mean() / max(n, 1)),
                      "changed_worlds_per_step_mean": float(np.mean(changed_worlds)),
                      "changed_worlds_per_step_min": int(np.min(changed_worlds))}))

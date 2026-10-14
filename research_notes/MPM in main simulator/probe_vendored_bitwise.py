"""Is MetalSim's vendored MPM solver bitwise the Newton 1.5.2 fork's, in either environment?

Runs a small column collapse (Isaac Lab lattice, aspect 1, 0.25 s) with the solver ``metalsim.physics.mpm.solver``
picks (``METALSIM_MPM_SOLVER`` = newton / vendored) and prints a hash of the particle positions and velocities
after every frame, plus the iteration counts. Run it in ``.venv-newton152`` with both solvers and in ``.venv`` (Newton
1.7, vendored only) and compare the hash lists:

    PYTHONPATH=<worktree> METALSIM_MPM_SOLVER=newton   .venv-newton152/bin/python probe_vendored_bitwise.py OUT.json
    PYTHONPATH=<worktree> METALSIM_MPM_SOLVER=vendored .venv-newton152/bin/python probe_vendored_bitwise.py OUT.json
    PYTHONPATH=<worktree>                              .venv/bin/python           probe_vendored_bitwise.py OUT.json
"""
import hashlib
import json
import sys

import numpy as np
import warp as wp

wp.config.quiet = True

from metalsim.physics.mpm import newton_compat, scenes  # noqa: E402

out = sys.argv[1]
scene = sys.argv[2] if len(sys.argv) > 2 else "column"
newton_compat.install(accumulate="float32", loop="cuda", rigid_deterministic=None, device_args=None)
import newton  # noqa: E402

res = {"newton": newton.__version__, "solver": scenes.SolverImplicitMPM.__module__, "scene": scene, "frames": []}
with wp.ScopedDevice("metal:0"):
    kw = dict(aspect=1.0, seconds=0.25) if scene == "column" else {}
    sc = scenes.make_scene(scene, **kw)
    for f in range(sc.frames):
        sc.step()
        q = sc.state.particle_q.numpy()
        v = sc.state.particle_qd.numpy()
        h = hashlib.sha256(q.tobytes() + v.tobytes()).hexdigest()[:16]
        res["frames"].append(h)
    res["final_q"] = q.tolist() if q.shape[0] < 20000 else None
res["iterations"] = [int(i) for i in newton_compat.drain_log()["iterations"]]
json.dump(res, open(out, "w"))
print(res["newton"], res["solver"], len(res["frames"]), res["frames"][0], res["frames"][-1], "iters", sum(res["iterations"]))

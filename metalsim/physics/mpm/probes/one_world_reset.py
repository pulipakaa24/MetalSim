"""One-world masked reset of the MPM material (MPMCoupling._solver_reset's single-world path) against the solver's
own full reset: per-step hashes of the robot state, sand wrench and particles over 15 steps, a reset, 15 steps.

    python one_world_reset.py [--json out.json]

``selected``: ``mpm.reset([True])`` (the device-mask path); ``full``: the same with ``SolverImplicitMPM.reset(world_mask
=None)`` for the solver part; ``unselected``: ``mpm.reset([False])``; ``none``: no reset. Expected: selected == full,
unselected == none, selected != none. Exits non-zero otherwise.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4]))

import numpy as np  # noqa: E402


def trajectory(mode, steps=15):
    import newton

    from metalsim.physics.mpm.probes import scale_coupled

    b = argparse.Namespace(worlds=1, rigid_only=False, voxel=0.02, iterations=50, solver="auto", cells_per_world=512,
                           capture=1)
    sim, mpm = scale_coupled.build(b)
    if mode == "full":
        flags = int(newton.StateFlags.PARTICLE_Q) | int(newton.StateFlags.PARTICLE_QD)
        mpm._solver_reset = lambda wm: mpm.solver.reset(mpm.state, world_mask=None, flags=flags)
    out = []

    def hx():
        h = hashlib.sha256()
        for arr in (sim.d.qpos, sim.d.qvel, mpm.coupling_f, mpm.particle_q, mpm.particle_qd):
            h.update(arr.numpy().tobytes())
        return h.hexdigest()[:16]

    for k in range(2 * steps):
        if k == steps and mode != "none":
            mpm.reset(np.array([mode != "unselected"]))
        mpm.step()
        out.append(hx())
    mpm.check_status()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    a = ap.parse_args()
    import warp as wp

    wp.config.quiet = True
    with wp.ScopedDevice("metal:0"):
        t = {m: trajectory(m) for m in ("selected", "full", "unselected", "none")}
    res = {"selected_equals_full": t["selected"] == t["full"], "unselected_equals_none": t["unselected"] == t["none"],
           "reset_changes_state": t["selected"][-1] != t["none"][-1]}
    res["ok"] = all(res.values())
    print(json.dumps(res))
    if a.json:
        json.dump({**res, "hashes": t}, open(a.json, "w"), indent=1)
    sys.exit(0 if res["ok"] else 1)


if __name__ == "__main__":
    main()

"""Check that MuJoCo Warp's deterministic-mode CCD split (solve kernel + write kernel, fork branch
metalsim-det-ccd) produces the same contacts as the single CCD kernel.

    python -P -m metalsim.physics.mpm.probes.ccd_split_equiv [--device cpu] [--worlds 8] [--steps 200] [--json out]

Steps the stacked-boxes scene (rigid_det_diag.py) with the default path, and at sampled steps runs collision on the
same Data twice: once with the single kernel, once with the split path forced on (no Warp deterministic mode needed,
so this runs on CPU as well). The contact sets, sorted by (world, geoms, position), must match bit for bit. Exits
non-zero on a mismatch or when no box-box contact was seen.
"""
from __future__ import annotations

import argparse
import json
import sys

import mujoco
import numpy as np
import warp as wp

from metalsim.physics.mpm.probes.rigid_det_diag import model_of


def contacts(d):
    n = int(d.nacon.numpy()[0])
    cols = {k: getattr(d.contact, k).numpy()[:n] for k in ("worldid", "geom", "pos", "dist", "frame", "dim",
                                                           "includemargin", "friction", "solref", "solimp")}
    key = np.lexsort((cols["pos"][:, 2], cols["pos"][:, 1], cols["pos"][:, 0], cols["geom"][:, 1], cols["geom"][:, 0],
                      cols["worldid"]))
    return {k: v[key] for k, v in cols.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--worlds", type=int, default=8)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--json")
    a = ap.parse_args()

    import mujoco_warp as mjw
    from mujoco_warp._src import collision_convex as cc

    assert hasattr(cc, "ccd_det_kernels_builder"), f"{cc.__file__} has no deterministic CCD split"
    wp.init()
    model = model_of("stack")
    mjd = mujoco.MjData(model)
    mujoco.mj_forward(model, mjd)
    box = int(mujoco.mjtGeom.mjGEOM_BOX)
    with wp.ScopedDevice(a.device):
        m = mjw.put_model(model)
        d = mjw.put_data(model, mjd, nworld=a.worlds, nconmax=64, njmax=256)
        # per-world spread so worlds differ
        qvel = d.qvel.numpy()
        qvel[:, :] += np.linspace(0.0, 0.05, a.worlds)[:, None]
        d.qvel.assign(qvel)
        checked, bb_seen, mismatches = 0, 0, []
        orig = cc._deterministic_mode
        for step in range(a.steps):
            if step % a.every == 0:
                cc._deterministic_mode = lambda: False
                mjw.collision(m, d)
                ref = contacts(d)
                cc._deterministic_mode = lambda: True
                mjw.collision(m, d)
                got = contacts(d)
                cc._deterministic_mode = orig
                checked += 1
                geoms = ref["geom"]
                bb = int(np.sum((model.geom_type[geoms[:, 0]] == box) & (model.geom_type[geoms[:, 1]] == box)))
                bb_seen += bb
                for k in ref:
                    if ref[k].shape != got[k].shape or not np.array_equal(ref[k], got[k]):
                        mismatches.append({"step": step, "field": k, "n_ref": len(ref[k]), "n_got": len(got[k])})
                        break
            mjw.step(m, d)
    out = {"device": a.device, "worlds": a.worlds, "steps": a.steps, "checked": checked, "box_box_contacts_seen": bb_seen,
           "mismatches": mismatches[:10], "n_mismatch": len(mismatches)}
    print(json.dumps(out, indent=1))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)
    sys.exit(0 if (not mismatches and bb_seen > 0) else 1)


if __name__ == "__main__":
    main()

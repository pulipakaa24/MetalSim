"""Dump the particle-to-grid inputs of the first MPM step (PIC quadrature: cell particle lists; the velocity test
space restriction: node -> element lists) and its outputs (node velocity, inverse mass), to find where two devices'
summation orders differ.

    python dump_p2g.py --device cuda:0|metal:0 --out DIR [--scene example:mpm_multi_material]
"""
import argparse
import json
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--device", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--scene", default="example:mpm_multi_material")
ap.add_argument("--set", action="append", default=[])
ap.add_argument("--fuse-fp-off", action="store_true", help="compile every module with fuse_fp=False (any device)")
args = ap.parse_args()

import numpy as np
import warp as wp

wp.config.quiet = True
wp.init()
if args.fuse_fp_off:
    from warp._src.context import Module as _Module

    _resolve = _Module.resolve_options

    def _resolve_nofma(self, *a, **kw):
        o = _resolve(self, *a, **kw)
        o["fuse_fp"] = False
        return o

    _Module.resolve_options = _resolve_nofma
here = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(here, "newton_compat.py")):
    sys.path.insert(0, here)
    import newton_compat, scenes  # noqa: E401
else:
    from metalsim.physics.mpm import newton_compat, scenes
from newton._src.solvers.implicit_mpm import solver_implicit_mpm as sim

newton_compat.install(accumulate="float32", loop="cuda", log=True)


def collect(obj, prefix, out, depth=0, seen=None):
    seen = seen if seen is not None else set()
    if id(obj) in seen or depth > 5:
        return
    seen.add(id(obj))
    if isinstance(obj, wp.array):
        try:
            out[prefix] = obj.numpy().copy()
        except Exception as e:  # noqa: BLE001
            out[prefix + "!err"] = np.array(str(e))
        return
    if isinstance(obj, (int, float, bool, str, type(None))) or callable(obj) and not hasattr(obj, "__dict__"):
        return
    d = getattr(obj, "__dict__", None)
    if d is None:
        return
    for k, v in list(d.items()):
        if k.startswith("__"):
            continue
        if isinstance(v, (list, tuple)):
            for i, x in enumerate(v):
                collect(x, f"{prefix}.{k}[{i}]", out, depth + 1, seen)
        else:
            collect(v, f"{prefix}.{k}", out, depth + 1, seen)


orig = sim.SolverImplicitMPM._compute_unconstrained_velocity
done = [False]


def hooked(self, state_in, dt, pic, scratch, inv_cell_volume):
    r = orig(self, state_in, dt, pic, scratch, inv_cell_volume)
    if not done[0]:
        done[0] = True
        out = {}
        collect(pic, "pic", out)
        collect(scratch.velocity_test, "vtest", out)
        collect(self._mpm_model, "mpm_model", out)
        out["inv_mass"] = scratch.inv_mass_matrix.numpy().copy()
        out["vel"] = scratch.velocity_field.dof_values.numpy().copy()
        np.savez_compressed(os.path.join(args.out, "p2g.npz"), **out)
    return r


sim.SolverImplicitMPM._compute_unconstrained_velocity = hooked
os.makedirs(args.out, exist_ok=True)
kw = {k: json.loads(v) for k, v in (s.split("=", 1) for s in args.set)}
with wp.ScopedDevice(args.device):
    sc = scenes.make_scene(args.scene, **kw)
    sc.step()
print("done", args.out)

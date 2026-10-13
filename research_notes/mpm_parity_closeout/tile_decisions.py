"""Log warp.fem's dispatched-assembly tiling decisions (integrate.py _generate_auxiliary_kernels) for the first steps
of a scene: per call the quadrature length, restriction sizes and the chosen tile size, so the L4's and Metal's
choices can be compared call by call.

    python tile_decisions.py --device cuda:0|metal:0 --scene example:mpm_multi_material --steps 3 --out FILE.json
"""
import argparse
import json
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--device", required=True)
ap.add_argument("--scene", required=True)
ap.add_argument("--set", action="append", default=[])
ap.add_argument("--steps", type=int, default=3)
ap.add_argument("--out", required=True)
args = ap.parse_args()

import warp as wp

wp.config.quiet = True
wp.init()
here = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(here, "newton_compat.py")):
    sys.path.insert(0, here)
    import newton_compat, scenes  # noqa: E401
else:
    from metalsim.physics.mpm import newton_compat, scenes
from warp._src.fem import integrate as itg

newton_compat.install(accumulate="float32", loop="cuda", log=True)
LOG = []
orig = itg._generate_auxiliary_kernels


def hooked(quadrature, test, trial, accumulate_dtype, device, kernel_options=None):
    r = orig(quadrature, test, trial, accumulate_dtype, device, kernel_options)
    if r:
        rec = {"linear": trial is None, "tile": int(r[0][1]), "qp_eval": int(quadrature.evaluation_point_count()),
               "elements": int(test.domain.element_count())}
        if trial is None:
            rec["node_elements"] = int(test.space_restriction.total_node_element_count())
            rec["nodes"] = int(test.space_restriction.node_count())
        LOG.append(rec)
    return r


itg._generate_auxiliary_kernels = hooked
kw = {}
for s in args.set:
    k, v = s.split("=", 1)
    kw[k] = json.loads(v)
with wp.ScopedDevice(args.device):
    sc = scenes.make_scene(args.scene, **kw)
    for i in range(args.steps):
        LOG.append({"step": i + 1})
        sc.step()
json.dump(LOG, open(args.out, "w"))
print(args.scene, "calls", sum(1 for r in LOG if "tile" in r), "tiled", sum(1 for r in LOG if r.get("tile", 1) > 1))

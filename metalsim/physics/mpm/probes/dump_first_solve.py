"""Dump every array reachable from the rheology and contact solvers before and after the first strain solve, plus the
particle state after the first frame, so two devices (L4 vs Metal) can be compared stage by stage.

    python dump_first_solve.py --device cuda:0|metal:0 --out DIR [--scene column --set aspect=2] [--solves 1]

Runs standalone on the L4 (next to newton_compat.py / scenes.py) and from the repo on the Mac.
"""
import argparse, json, os, sys

ap = argparse.ArgumentParser()
ap.add_argument("--device", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--scene", default="column")
ap.add_argument("--set", action="append", default=[])
ap.add_argument("--solves", type=int, default=1)
args = ap.parse_args()

import numpy as np
import warp as wp

wp.config.quiet = True
wp.init()
here = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(here, "newton_compat.py")):
    sys.path.insert(0, here)
    import newton_compat, scenes  # noqa: E401
else:
    from metalsim.physics.mpm import newton_compat, scenes
from newton._src.solvers.implicit_mpm import solve_rheology as sr

newton_compat.install(accumulate="float32", loop="cuda", log=True)
inner = sr._run_solver_loop
count = [0]


def collect(obj, prefix, out, depth=0, seen=None):
    seen = seen if seen is not None else set()
    if id(obj) in seen or depth > 4:
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


def hooked(rheology_solver, contact_solver, *rest):
    n = count[0]
    count[0] += 1
    if n < args.solves:
        pre = {}
        collect(rheology_solver, "rheo", pre)
        collect(contact_solver, "contact", pre)
        np.savez_compressed(os.path.join(args.out, f"solve{n}_pre.npz"), **pre)
    r = inner(rheology_solver, contact_solver, *rest)
    if n < args.solves:
        post = {}
        collect(rheology_solver, "rheo", post)
        collect(contact_solver, "contact", post)
        np.savez_compressed(os.path.join(args.out, f"solve{n}_post.npz"), **post)
    return r


sr._run_solver_loop = hooked
os.makedirs(args.out, exist_ok=True)
kw = {k: json.loads(v) for k, v in (s.split("=", 1) for s in args.set)}
with wp.ScopedDevice(args.device):
    sc = scenes.make_scene(args.scene, **kw)
    q0 = sc.state.particle_q.numpy().copy()
    while count[0] < args.solves:
        sc.step()
    np.savez_compressed(os.path.join(args.out, "frame.npz"), q0=q0, q=sc.state.particle_q.numpy(),
                        qd=sc.state.particle_qd.numpy())
print(json.dumps({k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in newton_compat.drain_log().items()}))

"""The two rigidity products inside the rheology loop, timed separately, and the shapes of their matrices.

Newton's coupled solve applies the rigidity operator in every iteration as two products: ``delta_body_qd = -IJtm @
delta_impulse`` (IJtm: one block row per rigid body DOF block, one block per collider node of that body) and
``collider_velocity += J @ delta_body_qd`` (J: one block row per grid node). On Metal both go through
``_ListedRowsMv`` (Newton fork, ``solve_rheology.py``). This probe runs the scene with the solver loop eager (no chunk
graph: every launch is host-visible), wraps ``_ListedRowsMv.launch`` with device synchronizations and records, per
frame and per matrix: launches, synchronized ms, rows, stored blocks (device ``offsets[nrow]``), the longest row, and
whether the product took the tiled ``bsr_mv`` path (``full``, Newton forks with the tiled path only).

    PYTHONPATH=<warp>:<newton>:<metalsim> .venv-newton152/bin/python \
        metalsim/physics/mpm/probes/rigidity_mv_probe.py example:mpm_twoway_coupling --frames 150 --json out.json
"""
import argparse
import collections
import json
import os
import time

ap = argparse.ArgumentParser()
ap.add_argument("scene")
ap.add_argument("--device", default="metal:0")
ap.add_argument("--frames", type=int, default=150)
ap.add_argument("--set", action="append", default=[])
ap.add_argument("--json")
args = ap.parse_args()

import numpy as np
import warp as wp

wp.config.quiet = True
wp.init()
from metalsim.physics.mpm import newton_compat, scenes

newton_compat.install(accumulate="float32", loop="cuda", log=False, chunk_graph=False, gpu_exit=False)
from newton._src.solvers.implicit_mpm import solve_rheology as sr  # noqa: E402

frame_stats = collections.defaultdict(lambda: collections.defaultdict(lambda: {"launches": 0, "ms": 0.0}))
shapes = {}
cur = {"frame": -1}
_init, _launch = sr._ListedRowsMv.__init__, sr._ListedRowsMv.launch


def _label(mv):
    # IJtm has the few body rows, J one row per grid node
    return "IJtm" if mv.A.nrow < mv.A.ncol else "J"


def init(self, A, *a, **kw):
    _init(self, A, *a, **kw)
    lab = _label(self)
    off = A.offsets.numpy()[: A.nrow + 1]
    lengths = np.diff(off)
    shapes.setdefault(cur["frame"], {})[lab] = {
        "nrow": int(A.nrow), "ncol": int(A.ncol), "stored_blocks": int(off[-1]), "host_nnz": int(A.nnz),
        "nonempty_rows": int((lengths > 0).sum()), "max_row_blocks": int(lengths.max()) if lengths.size else 0,
        "full_tiled_path": bool(getattr(self, "full", False)),
    }


def launch(self):
    if wp.get_device(args.device).is_capturing:
        return _launch(self)
    wp.synchronize_device()
    t0 = time.perf_counter()
    _launch(self)
    wp.synchronize_device()
    s = frame_stats[cur["frame"]][_label(self)]
    s["launches"] += 1
    s["ms"] += (time.perf_counter() - t0) * 1e3


sr._ListedRowsMv.__init__, sr._ListedRowsMv.launch = init, launch
kw = {k: json.loads(v) for k, v in (s.split("=", 1) for s in args.set)}
walls = []
with wp.ScopedDevice(args.device):
    sc = scenes.make_scene(args.scene, **kw)
    for f in range(args.frames):
        cur["frame"] = f
        wp.synchronize_device()
        t0 = time.perf_counter()
        sc.step()
        wp.synchronize_device()
        walls.append((time.perf_counter() - t0) * 1e3)
        st = frame_stats[f]
        sh = shapes.get(f, {})
        if f % 10 == 0 or f == args.frames - 1:
            print(f"frame {f:4d} wall {walls[-1]:7.1f} ms | " + " | ".join(
                f"{k}: {v['launches']} x, {v['ms']:.2f} ms, rows {sh.get(k, {}).get('nrow')}, blocks "
                f"{sh.get(k, {}).get('stored_blocks')}, longest row {sh.get(k, {}).get('max_row_blocks')}, "
                f"tiled {sh.get(k, {}).get('full_tiled_path')}" for k, v in sorted(st.items())), flush=True)
out = {"scene": args.scene, "frames": args.frames, "wall_ms": walls,
       "per_frame": {f: dict(v) for f, v in frame_stats.items()}, "shapes": shapes,
       "warp": wp.__file__, "newton": sr.__file__}
if args.json:
    os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
    with open(args.json, "w") as fh:
        json.dump(out, fh, indent=1)

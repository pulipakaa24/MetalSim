"""Run-to-run repeatability of the coupled main simulator: box_on_sand twice in one process, report the first step at
which box state, sand wrench or particles differ.

    .venv/bin/python repeat_check.py [--worlds 2] [--steps 60] [--capture 1] [--runs 2]
"""
import argparse
import importlib.util
import pathlib

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--worlds", type=int, default=2)
ap.add_argument("--steps", type=int, default=60)
ap.add_argument("--capture", type=int, default=1)
ap.add_argument("--runs", type=int, default=2)
ap.add_argument("--mass", default="effective")
a = ap.parse_args()
spec = importlib.util.spec_from_file_location("box_on_sand", pathlib.Path(__file__).with_name("box_on_sand.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class A:
    worlds, steps, voxel, cells, capture, mass, mode, mpm_substeps, particle_stride = (
        a.worlds, a.steps, 0.02, "auto", a.capture, a.mass, "staggered", 1, 1000)


import warp as wp  # noqa: E402

wp.config.quiet = True
with wp.ScopedDevice("metal:0"):
    runs = [mod.run_metalsim(A) for _ in range(a.runs)]


def first(x, y):
    d = [i for i in range(len(x)) if not np.array_equal(x[i], y[i])]
    return d[0] if d else None


r0 = runs[0]
for k, r in enumerate(runs[1:], 1):
    print(f"run 0 vs {k}: pos {first(r0['pos'], r['pos'])} qvel {first(r0['qvel'], r['qvel'])} "
          f"wrench {first(r0['wrench'], r['wrench'])} particles {first(r0['phash'], r['phash'])}")

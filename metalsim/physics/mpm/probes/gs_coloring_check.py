"""Is the Gauss-Seidel coloring of the implicit MPM a valid coloring with separate worlds?

Builds an N-world sand bed (``separate_worlds=True``, Gauss-Seidel, P0 strain), runs one step and checks, for every
color, that no two strain nodes solved concurrently write the same velocity node (strain-matrix columns), and that
the color blocks cover every strain node once. With environment-first strain partitions, Newton 1.5.2 builds the
blocks from cell indices (``fill_uniform_color_block_indices``: ``cell * nodes_per_cell``) while the solver indexes
strain nodes in partition (environment-major) order, so the blocks are another permutation and colors overlap.

    python gs_coloring_check.py [--device cpu] [--worlds 2]
"""
import argparse
import sys

import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4]))
import numpy as np
import warp as wp

wp.config.quiet = True
ap = argparse.ArgumentParser()
ap.add_argument("--device", default="cpu")
ap.add_argument("--worlds", type=int, default=2)
ap.add_argument("--cells", default="-1")
ap.add_argument("--grid", default="sparse")
a = ap.parse_args()

import newton  # noqa: E402

from metalsim.physics.mpm import newton_compat  # noqa: E402
from metalsim.physics.mpm.coupled import GridEmitter, MPMMaterial  # noqa: E402
from metalsim.physics.mpm.solver import solver_class  # noqa: E402

newton_compat.install(accumulate="float32", loop="cuda", log=False, rigid_deterministic=None, device_args=None)
S = solver_class()
captured = {}
sr = sys.modules[S.__module__.rsplit(".", 1)[0] + ".solve_rheology"]
orig = sr._run_solver_loop


def spy(rheology_solver, *args, **kw):
    if "rheo" not in captured:
        r = rheology_solver.rheology
        captured["rheo"] = dict(color_offsets=r.color_offsets.numpy().copy(), color_blocks=r.color_blocks.numpy().copy(),
                                offs=r.strain_mat.offsets.numpy().copy(), cols=r.strain_mat.columns.numpy().copy(),
                                n=r.stress.shape[0], env=None if r.strain_environment_offsets is None
                                else r.strain_environment_offsets.numpy().copy())
    return orig(rheology_solver, *args, **kw)


sr._run_solver_loop = spy
with wp.ScopedDevice(a.device):
    b = newton.ModelBuilder()
    S.register_custom_attributes(b)
    for _ in range(a.worlds):
        b.begin_world()
        b.add_shape_plane(width=0.0, length=0.0, body=-1, cfg=newton.ModelBuilder.ShapeConfig(mu=0.5, density=0.0))
        GridEmitter(lower=(-0.15, -0.15, 0.0), upper=(0.15, 0.15, 0.08),
                    material=MPMMaterial(density=1500.0, friction=0.7)).emit(b, 0.02)
        b.end_world()
    m = b.finalize()
    cfg = S.Config(voxel_size=0.02, max_iterations=5, solver="gauss-seidel", separate_worlds=a.worlds > 1,
                   grid_type=a.grid)
    ppw = m.particle_count // a.worlds
    cfg.max_active_cell_count = ((ppw + 511) // 512) * 512 * a.worlds if a.cells == "auto" else int(a.cells)
    s = S(m, cfg)
    st = m.state()
    s.step(st, st, None, None, 1.0 / 120.0)

c = captured["rheo"]
co, cb, offs, cols, n = c["color_offsets"], c["color_blocks"], c["offs"], c["cols"], c["n"]
covered = np.zeros(n, int)
bad_colors = 0
conflicts = 0
for k in range(len(co) - 1):
    seen = {}
    clash = 0
    for blk in range(co[k], co[k + 1]):
        beg, end = cb[0, blk], cb[1, blk]
        covered[beg:end] += 1
        for t in range(beg, end):
            for v in cols[offs[t]:offs[t + 1]]:
                if v in seen and seen[v] != blk:
                    clash += 1
                seen[v] = blk
    conflicts += clash
    bad_colors += clash > 0
active = n if c["env"] is None else int(c["env"][-1])
print(f"worlds {a.worlds}: strain nodes {n} (active {active}), colors {len(co) - 1}, blocks {co[-1]}; "
      f"strain nodes covered once {int((covered[:active] == 1).sum())}/{active}, "
      f"velocity-node write conflicts within a color {conflicts} in {bad_colors} colors")
print("VALID" if conflicts == 0 and (covered[:active] == 1).all() else "INVALID")

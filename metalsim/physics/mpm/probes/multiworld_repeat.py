"""Run-to-run repeatability of the implicit MPM with separate worlds (no rigid side): a sand bed on a floor in N
worlds, stepped twice from the same initial state in one process; prints the first step whose particle state
differs, per solver configuration.

    python multiworld_repeat.py [--worlds 2] [--steps 10]
"""
import argparse
import hashlib
import itertools

import warp as wp

wp.config.quiet = True
ap = argparse.ArgumentParser()
ap.add_argument("--worlds", type=int, default=2)
ap.add_argument("--steps", type=int, default=10)
ap.add_argument("--solvers", default="auto,jacobi,gauss-seidel")
ap.add_argument("--bases", default="S2,pic27")
ap.add_argument("--cells", default="auto,-1")
ap.add_argument("--offset", type=float, default=0.0, help="world k's material moved by k * offset along x [m]")
a = ap.parse_args()

import newton  # noqa: E402

from metalsim.physics.mpm import newton_compat  # noqa: E402
from metalsim.physics.mpm.coupled import GridEmitter, MPMMaterial  # noqa: E402
from metalsim.physics.mpm.solver import solver_class  # noqa: E402

newton_compat.install(accumulate="float32", loop="cuda", log=False, chunk_graph=True, gpu_exit=True,
                      rigid_deterministic=None, device_args=None)
S = solver_class()
print("newton", newton.__version__, S.__module__)


def build():
    b = newton.ModelBuilder()
    S.register_custom_attributes(b)
    for w in range(a.worlds):
        b.begin_world()
        b.add_shape_plane(width=0.0, length=0.0, body=-1, cfg=newton.ModelBuilder.ShapeConfig(mu=0.5, density=0.0))
        x = w * a.offset
        GridEmitter(lower=(x - 0.15, -0.15, 0.0), upper=(x + 0.15, 0.15, 0.08),
                    material=MPMMaterial(density=1500.0, friction=0.7)).emit(b, 0.02)
        b.end_world()
    return b.finalize()


def run(solver, basis, cells):
    m = build()
    cfg = S.Config(voxel_size=0.02, max_iterations=50, solver=solver, collider_basis=basis, separate_worlds=True)
    ppw = m.particle_count // a.worlds
    cfg.max_active_cell_count = ((ppw + 511) // 512) * 512 * a.worlds if cells == "auto" else int(cells)
    s = S(m, cfg)
    st = m.state()
    out = []
    for _ in range(a.steps):
        s.step(st, st, None, None, 1.0 / 120.0)
        q = st.particle_q.numpy().reshape(a.worlds, -1, 3)
        out.append((hashlib.sha256(q.tobytes() + st.particle_qd.numpy().tobytes()).hexdigest()[:16],
                    [hashlib.sha256(q[w].tobytes()).hexdigest()[:16] for w in range(a.worlds)]))
    return out


with wp.ScopedDevice("metal:0"):
    for solver, basis, cells in itertools.product(a.solvers.split(","), a.bases.split(","), a.cells.split(",")):
        h = [run(solver, basis, cells) for _ in range(2)]
        d = [i for i in range(a.steps) if h[0][i][0] != h[1][i][0]]
        if d:
            i = d[0]
            pw = [(h[0][i][1][w] != h[1][i][1][w]) for w in range(a.worlds)]
            print("  worlds differing at that step:", [w for w in range(a.worlds) if pw[w]], flush=True)
        print(f"solver {solver:13s} basis {basis:6s} cells {cells:5s}: first differing step {d[0] if d else None}",
              flush=True)

"""Why test_coupled_contact_filter_keeps_source_order can fail: is SolverCoupled's module built in RUN_TO_RUN mode?

Prints, around newton_compat.install(rigid_deterministic=True): wp.config.deterministic, the module options Warp
holds for the coupled-solver modules (by module name), whether the compiled module's hash changed, and the first
filtered point ids of three launches of the filter kernel (source order expected: 2, 4, 5, ...).
    PYTHONPATH=<warp>:<newton>:<metalsim> python coupled_det_diag.py
"""
import numpy as np
import warp as wp

wp.config.quiet = True
wp.init()
import newton  # noqa: E402
from warp._src import context as ctx  # noqa: E402

from metalsim.physics.mpm import newton_compat  # noqa: E402

sc = __import__("newton._src.solvers.coupled.solver_coupled", fromlist=["x"])
mode = wp.DeterministicMode
print("warp", wp.__file__, "newton", newton.__file__)
print("config.deterministic before:", wp.config.deterministic)


def show(tag):
    for name in ("newton._src.solvers.coupled.solver_coupled", "newton._src.solvers.coupled.solver_coupled_proxy"):
        m = ctx.get_module(name)
        print(f"  [{tag}] {name}: options.deterministic={m.options.get('deterministic')}")
    k = sc._filter_rigid_contacts_global_shape_ids_kernel
    print(f"  [{tag}] kernel module: {k.module.name} options.deterministic={k.module.options.get('deterministic')}")


show("before install")
newton_compat.install(accumulate="float32", loop="cuda", rigid_deterministic=True)
print("config.deterministic after:", wp.config.deterministic, "state:", newton_compat._STATE.get("coupled_deterministic"))
show("after install")

n, shapes = 1 << 18, 64
rng = np.random.default_rng(0)
s0 = rng.integers(0, shapes, n).astype(np.int32)
s1 = rng.integers(0, shapes, n).astype(np.int32)
flags = np.where(np.arange(shapes) % 3 == 0, 0, int(newton.ShapeFlags.COLLIDE_SHAPES)).astype(np.int32)
keep = (flags[s0] != 0) & (flags[s1] != 0)
with wp.ScopedDevice("metal:0"):
    i32 = lambda a: wp.array(a, dtype=wp.int32)  # noqa: E731
    v3 = lambda: wp.array(rng.random((n, 3)).astype(np.float32), dtype=wp.vec3)  # noqa: E731
    f32 = lambda: wp.array(rng.random(n).astype(np.float32), dtype=wp.float32)  # noqa: E731
    src = [i32(np.ones(1)), i32(np.array([n])), i32(s0), i32(s1), i32(np.arange(n)), v3(), v3(), v3(), v3(),
           v3(), f32(), f32(), i32(np.arange(n)), i32(flags), int(newton.ShapeFlags.COLLIDE_SHAPES)]
    for r in range(3):
        dst_count = wp.zeros(1, dtype=wp.int32)
        dst = [dst_count, *(wp.zeros(n, dtype=wp.int32) for _ in range(3)), *(wp.zeros(n, dtype=wp.vec3) for _ in range(5)),
               wp.zeros(n, dtype=wp.float32), wp.zeros(n, dtype=wp.float32), wp.zeros(n, dtype=wp.int32),
               wp.full(n, -1, dtype=wp.int32)]
        wp.launch(sc._filter_rigid_contacts_global_shape_ids_kernel, dim=n, inputs=src + dst)
        ids = dst[3].numpy()[: int(dst_count.numpy()[0])]
        ok = np.array_equal(ids, np.nonzero(keep)[0])
        print(f"launch {r}: first ids {ids[:6].tolist()} source order {ok}")
    show("after launches")

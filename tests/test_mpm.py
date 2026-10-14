"""MPM on Metal: Newton 1.5.2's SolverImplicitMPM (Isaac Lab 3.0-EA's MPM) with metalsim's compat layer and
accelerations. Every acceleration must reproduce the oracle (Newton's own kernels, CUDA stopping rule) bit for bit.

Needs the Newton 1.5.2 environment that scripts/setup_newton152.sh builds (.venv-newton152: the Newton fork's
metalsim-1.5.2 branch and the Warp fork's metalsim branch, pinned there):
    scripts/gpu_run.sh test_mpm check 15 -- .venv-newton152/bin/python -m pytest tests/test_mpm.py
An environment that imports stock Newton 1.5.2 / Warp fails test_environment_has_the_forks (and the branch-only tests),
so a mis-built .venv-newton152 cannot pass with those tests silently skipped (on 2026-10-13 it imported stock Newton for
hours unnoticed). A stock environment built on purpose (the setup script's stock-oracle recipe) sets
METALSIM_NEWTON152_STOCK=1; the branch-only tests are then skipped.
"""
import inspect
import os

import numpy as np
import pytest
import warp as wp

newton = pytest.importorskip("newton")
pytestmark = [
    pytest.mark.skipif(not newton.__version__.startswith("1.5.2"), reason="needs Newton 1.5.2 (Isaac Lab 3.0-EA's pin)"),
    pytest.mark.skipif(not wp.is_metal_available(), reason="needs Metal"),
]

from metalsim.physics.mpm import compare, newton_compat, scenes  # noqa: E402
from newton._src.solvers.implicit_mpm import solve_rheology as sr  # noqa: E402

DEV = "metal:0"
# the forks' API that the branch-only tests use; stock Newton 1.5.2 / Warp have none of it
FORK_NEWTON = hasattr(sr, "_color_block_count") and hasattr(sr, "active_node_bound")
FORK_WARP = "dim_limit" in inspect.signature(wp.launch).parameters
STOCK = os.environ.get("METALSIM_NEWTON152_STOCK") == "1"     # a stock environment on purpose
SMALL = dict(aspect=1.0, seconds=0.25)


def _run(frames=None, **kw):
    """Run a small column collapse; return (positions per frame, iterations per solve)."""
    with wp.ScopedDevice(DEV):
        sc = scenes.make_scene("column", **{**SMALL, **kw})
        out = []
        for _ in range(frames or sc.frames):
            sc.step()
            out.append(sc.state.particle_q.numpy().copy())
    return np.stack(out), newton_compat.drain_log()["iterations"]


@pytest.fixture(autouse=True)
def _compat():
    newton_compat.install(accumulate="float32", loop="cuda")
    newton_compat.drain_log()
    yield
    newton_compat.install(accumulate="float32", loop="cuda")
    newton_compat.drain_log()


def test_cuda_stopping_rule_jacobi_cap_24():
    """A Jacobi solve capped at 24 iterations runs 25 (Isaac's conditional-graph rule), not 0 (Newton's host loop)."""
    _, iters = _run(frames=3, solver="jacobi", max_iterations=24, tolerance=1e-12)
    assert iters.tolist() == [25, 25, 25]


def test_iterations_are_multiples_of_five():
    """Converged solves stop on the 5-iteration boundary the CUDA graph checks."""
    _, iters = _run(frames=10)
    assert (iters % 5 == 0).all() and iters.max() <= 250


def test_metal_deterministic():
    """Two Metal runs are bitwise identical."""
    a, ia = _run()
    b, ib = _run()
    assert np.array_equal(a, b) and np.array_equal(ia, ib)


def test_chunk_graph_bitwise():
    """Replaying the 5-iteration chunk as a recorded graph changes nothing."""
    a, ia = _run()
    newton_compat.install(accumulate="float32", loop="cuda", chunk_graph=True)
    b, ib = _run()
    assert np.array_equal(a, b) and np.array_equal(ia, ib)


def test_gpu_exit_bitwise():
    """Deciding the stop on the GPU (indirect execution range zeroed by a gate kernel) changes nothing: same
    positions, same iteration counts as the host-checked chunk graph."""
    newton_compat.install(accumulate="float32", loop="cuda", chunk_graph=True)
    a, ia = _run()
    newton_compat.install(accumulate="float32", loop="cuda", chunk_graph=True, gpu_exit=True)
    b, ib = _run()
    assert np.array_equal(a, b) and np.array_equal(ia, ib)


def test_gpu_exit_unlogged_bitwise():
    """Without the per-solve log the GPU-decided stop reads nothing back (no host wait per solve) and changes
    nothing: same positions as the logged run."""
    newton_compat.install(accumulate="float32", loop="cuda", chunk_graph=True, gpu_exit=True)
    a, _ = _run()
    newton_compat.install(accumulate="float32", loop="cuda", chunk_graph=True, gpu_exit=True, log=False)
    b, ib = _run()
    assert np.array_equal(a, b) and ib.size == 0


def test_gpu_exit_jacobi_cap_24():
    """The GPU-decided stop follows the CUDA rule at a cap that is not a multiple of 5 (25 iterations at 24)."""
    newton_compat.install(accumulate="float32", loop="cuda", chunk_graph=True, gpu_exit=True)
    _, iters = _run(frames=3, solver="jacobi", max_iterations=24, tolerance=1e-12)
    assert iters.tolist() == [25, 25, 25]


@pytest.mark.xfail(reason="with FMA contraction (Warp fuse_fp on Metal, branch metalsim-fp-contract) the lane-shuffled "
                          "sums of simd_gs contract differently from Newton's serial loop; simd_gs is opt-in and gave no "
                          "speedup", strict=False)
def test_simd_sweep_bitwise():
    """The SIMD-cooperative Gauss-Seidel sweep reproduces Newton's sweep bit for bit."""
    a, ia = _run()
    newton_compat.install(accumulate="float32", loop="cuda", simd=True)
    b, ib = _run()
    assert np.array_equal(a, b) and np.array_equal(ia, ib)


def test_fused_solve_bitwise():
    """The one-dispatch persistent-threadgroup solve reproduces Newton's loop bit for bit."""
    a, ia = _run()
    newton_compat.install(accumulate="float32", loop="cuda", fused=True)
    b, ib = _run()
    assert np.array_equal(a, b) and np.array_equal(ia, ib)


@pytest.mark.skipif(STOCK and not FORK_NEWTON, reason="stock Newton 1.5.2 (METALSIM_NEWTON152_STOCK=1)")
def test_color_blocks_on_all_cores_bitwise():
    """Spreading each color over all GPU cores (the Newton branch) reproduces the one-block oracle bit for bit."""
    fn = sr._color_block_count
    try:
        sr._color_block_count = lambda device: 1
        a, ia = _run()
    finally:
        sr._color_block_count = fn
    b, ib = _run()
    assert np.array_equal(a, b) and np.array_equal(ia, ib)


def test_column_observables_from_lattice():
    """Initial lattice matches Isaac Lab's cell_center emitter; observables are finite and the column spreads."""
    with wp.ScopedDevice(DEV):
        sc = scenes.make_scene("column", **SMALL)
        q0 = sc.state.particle_q.numpy()
        for _ in range(sc.frames):
            sc.step()
        q1 = sc.state.particle_q.numpy()
    # Isaac Lab's _emit_grid: resolution = ceil(ppc * extent / voxel) with float32 corners, so 3 * 0.2 / 0.05 is
    # exactly 12 (float64 would give 12.000000000000002 -> 13); the L4 recording through Isaac Lab confirmed 12.
    res_x = int(np.ceil(3.0 * np.float32(0.2) / 0.05))
    assert res_x == 12
    spacing = 0.2 / res_x
    assert np.isclose(q0[:, 0].min(), 0.5 * spacing, atol=1e-6)
    assert len(np.unique(np.round(q0[:, 0], 6))) == res_x
    run = {"meta": {"info": sc.info}, "traj": {"pos": q1[None], "frame": np.array([sc.frames])},
           "init": {"radius": sc.model.particle_radius.numpy()}}
    obs = compare.column_observables(run)
    assert obs["L_f"] > obs["L_i"] and np.isfinite(obs["H_f"]) and 0 < obs["H_f"] <= obs["H_i"] + spacing


# ---------------------------------------------------------------------------------------------------------------------
# Isaac Lab Franka-Pour scale (separate worlds, rebuildable sparse grid with Isaac's 1,024-cell reservation per world):
# the scaling changes of the Warp fork (bounded launches, branch metalsim-mpm-scale) and the Newton branch must leave
# the trajectory bit for bit unchanged. Each run is its own process (probes/pour_bitwise.py), so the switch
# WP_METAL_BOUNDED_LAUNCH applies to everything it gates.

import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402

_PROBES = os.path.join(os.path.dirname(__file__), "..", "metalsim", "physics", "mpm", "probes")
_scale_branches = pytest.mark.skipif(
    STOCK and not (FORK_WARP and FORK_NEWTON), reason="stock Newton 1.5.2 / Warp (METALSIM_NEWTON152_STOCK=1)"
)


def test_environment_has_the_forks():
    """The environment imports the Newton fork (metalsim-1.5.2) and the Warp fork (metalsim), as
    scripts/setup_newton152.sh installs them; fails, not skips, otherwise (unless METALSIM_NEWTON152_STOCK=1)."""
    if STOCK:
        pytest.skip("stock Newton 1.5.2 environment (METALSIM_NEWTON152_STOCK=1)")
    assert FORK_NEWTON, (f"{newton.__file__} is stock Newton 1.5.2, not the fork's metalsim-1.5.2 branch: "
                         "run scripts/setup_newton152.sh (or set METALSIM_NEWTON152_STOCK=1 for a stock environment)")
    assert FORK_WARP, f"{wp.__file__} is not the Warp fork (no wp.launch(dim_limit=)): run scripts/setup_warp.sh"



def _probe(script, *args, env=None, check=True):
    out = subprocess.run([sys.executable, os.path.join(_PROBES, script), *args], capture_output=True, text=True,
                         env={**os.environ, **(env or {})}, timeout=1800)
    if check:
        assert out.returncode == 0, out.stderr[-3000:]
    return out


def _pour(tmp_path, name, *args, env=None):
    path = tmp_path / f"{name}.json"
    _probe("pour_bitwise.py", "record", "--worlds", "8", "--steps", "160", "--every", "20", *args, "--json", str(path),
           env=env)
    return json.loads(path.read_text())


@_scale_branches
def test_pour_bounded_launches_bitwise(tmp_path):
    """Isaac Lab Franka-Pour MPM: bounded launches (active prefix only) give the trajectory of full launches."""
    ref = tmp_path / "full.npz"
    _pour(tmp_path, "full", "--mode", "gpuexit", "--out", str(ref), env={"WP_METAL_BOUNDED_LAUNCH": "0"})
    res = _pour(tmp_path, "bounded", "--mode", "gpuexit", "--compare", str(ref))
    assert res["finite"] and res["compare"]["bitwise_equal"], res["compare"]


@_scale_branches
def test_pour_whole_step_capture_bitwise(tmp_path):
    """The whole Franka-Pour MPM step records into one Metal graph and replays bitwise like eager stepping."""
    ref = tmp_path / "eager.npz"
    _pour(tmp_path, "eager", "--mode", "gpuexit", "--out", str(ref))
    res = _pour(tmp_path, "capture", "--mode", "capture", "--compare", str(ref))
    assert res["compare"]["bitwise_equal"], res["compare"]


@_scale_branches
def test_pour_capacity_overflow_raises(tmp_path):
    """An undersized sparse-grid reservation reports the overflow (warning, then RuntimeError) instead of crashing."""
    out = _probe("pour_bitwise.py", "record", "--worlds", "16", "--steps", "700", "--cells-per-world", "64",
                 "--check-status", "--json", str(tmp_path / "o.json"), check=False)
    assert out.returncode == 1, out.stderr[-3000:]
    assert "sparse grid rebuild capacity was exceeded" in out.stderr
    assert "Warning: Implicit MPM sparse grid rebuild failed" in out.stdout + out.stderr


@_scale_branches
def test_proxy_wrench_harvest_repeatable(tmp_path):
    """Isaac Lab's coupler harvest (proxy body wrenches) sums in a fixed order on Metal: one result on every call."""
    path = tmp_path / "h.json"
    _probe("harvest_check.py", "--worlds", "16", "--steps", "120", "--repeats", "10", "--json", str(path))
    res = json.loads(path.read_text())
    assert res["ordered"]["distinct_results"] == 1
    assert res["ordered"]["nonzero_bodies"] > 0 and res["max_rel_diff"] < 1e-5


def test_coupled_contact_filter_keeps_source_order():
    """With ``install(rigid_deterministic=True)``, SolverCoupled's per-entry contact filter (an atomic-counter
    compaction) hands out the filtered slots in source order, every run: the entry solver then sees the same contact
    list, which Isaac Lab's Franka-Pour needs to repeat bitwise (it differed run to run from the first step)."""
    sc = pytest.importorskip("newton._src.solvers.coupled.solver_coupled")
    newton_compat.install(accumulate="float32", loop="cuda", rigid_deterministic=True)
    n, shapes = 1 << 18, 64
    rng = np.random.default_rng(0)
    s0 = rng.integers(0, shapes, n).astype(np.int32)
    s1 = rng.integers(0, shapes, n).astype(np.int32)
    flags = np.where(np.arange(shapes) % 3 == 0, 0, int(newton.ShapeFlags.COLLIDE_SHAPES)).astype(np.int32)
    keep = (flags[s0] != 0) & (flags[s1] != 0)
    with wp.ScopedDevice(DEV):
        i32 = lambda a: wp.array(a, dtype=wp.int32)  # noqa: E731
        v3 = lambda: wp.array(rng.random((n, 3)).astype(np.float32), dtype=wp.vec3)  # noqa: E731
        f32 = lambda: wp.array(rng.random(n).astype(np.float32), dtype=wp.float32)  # noqa: E731
        src = [i32(np.ones(1)), i32(np.array([n])), i32(s0), i32(s1), i32(np.arange(n)), v3(), v3(), v3(), v3(),
               v3(), f32(), f32(), i32(np.arange(n)), i32(flags), int(newton.ShapeFlags.COLLIDE_SHAPES)]
        for _ in range(3):
            dst_count = wp.zeros(1, dtype=wp.int32)
            dst = [dst_count, *(wp.zeros(n, dtype=wp.int32) for _ in range(3)), *(wp.zeros(n, dtype=wp.vec3) for _ in range(5)),
                   wp.zeros(n, dtype=wp.float32), wp.zeros(n, dtype=wp.float32), wp.zeros(n, dtype=wp.int32),
                   wp.full(n, -1, dtype=wp.int32)]
            wp.launch(sc._filter_rigid_contacts_global_shape_ids_kernel, dim=n, inputs=src + dst)
            count = int(dst_count.numpy()[0])
            assert count == int(keep.sum())
            np.testing.assert_array_equal(dst[3].numpy()[:count], np.nonzero(keep)[0])  # point ids: source order

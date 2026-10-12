"""Where the time of one Gauss-Seidel color pass goes on Metal, at sustained (hot) GPU clocks.

Hooks one strain solve after warm-up (like loop_breakdown.py) and times, per color pass averaged over all colors,
probe-local copies of Newton's colored GS kernel (same Newton functions via wp.static, so the same code):
    full          Newton's kernel body (gather B v, local Drucker-Prager solve, stress update, scatter B^T)
    index         the color/block index walk only
    gather        local strain only (tau = rhs + B v)
    gather_solve  gather + local solve, no stress update or scatter
    solve         local solve from a saved local strain
    scatter       velocity update from a saved stress delta
plus Newton's own launch (``newton``), an empty kernel with the same launch shape (``empty``), the ``full`` copy at
other launch shapes (results do not depend on the shape: blocks of one color touch disjoint nodes), and ``*_fast``
variants compiled with Metal fast division and sqrt (metal::fast::divide / fast::sqrt, the class of math CUDA uses
for these fast_math kernels: rounding-level change, NOT bitwise).

Every measurement replays a recorded graph of whole sweeps back to back for --seconds, so the GPU is at its
sustained clock (latency_clock.swift: the M4 Max clocks down ~5x when lightly loaded); per-pass times include the
in-graph dependent-dispatch cost (~1.1 us, dispatch_modes.swift). Mutates the solve it measures (timing only).

    PYTHONPATH=. .venv-newton152/bin/python \
        metalsim/physics/mpm/probes/gs_phases.py example:mpm_granular
"""
import argparse, json, os, re, time

ap = argparse.ArgumentParser()
ap.add_argument("scene")
ap.add_argument("--device", default="metal:0")
ap.add_argument("--warmup", type=int, default=3)
ap.add_argument("--seconds", type=float, default=1.0)
ap.add_argument("--set", action="append", default=[])
ap.add_argument("--json")
ap.add_argument("--no-fast", action="store_true",
                help="skip the *_fast variants (Warp trees whose fast_math already lowers / and sqrt approximately)")
args = ap.parse_args()

import numpy as np
import warp as wp

wp.config.quiet = True
wp.init()
from warp._src.context import runtime
from metalsim.physics.mpm import newton_compat, scenes
from newton._src.solvers.implicit_mpm import rheology_solver_kernels as rk
from newton._src.solvers.implicit_mpm import solve_rheology as sr

# ---- fast-math rewrite of probe modules, applied while _FAST["on"] (module loads happen at first launch) ----
_FAST = {"on": False, "rewrites": 0}
_load = runtime.core.wp_metal_load_library
_HELPER = (
    "\ntemplate <typename T> inline T _wp_fdiv(T a, T b) { return a / b; }\n"
    "template <> inline float _wp_fdiv<float>(float a, float b) { return metal::fast::divide(a, b); }\n"
)


def _load_library(ordinal, source):
    if _FAST["on"]:
        s = source.decode()
        n0 = len(s)
        s = s.replace("#include <metal_stdlib>\n", "#include <metal_stdlib>\n" + _HELPER, 1)
        s, a = re.subn(r"return a/b;", "return _wp_fdiv(a, b);", s)
        s, b = re.subn(r"ret\[i\] = a\[i\] / b\[i\];", "ret[i] = _wp_fdiv(a[i], b[i]);", s)
        s, c = re.subn(r"inline float sqrtf\(float x\) \{ return metal::sqrt\(x\); \}",
                       "inline float sqrtf(float x) { return metal::fast::sqrt(x); }", s)
        if not (a and b and c) or len(s) <= n0:
            raise RuntimeError(f"fast-math rewrite did not apply ({a}, {b}, {c})")
        _FAST["rewrites"] += 1
        source = s.encode()
    return _load(ordinal, source)


runtime.core.wp_metal_load_library = _load_library

vec6, mat66, mat55, mat13 = rk.vec6, rk.mat66, rk.mat55, rk.mat13
YP = rk.YieldParamVec


def make_kernels(hv, hd, hc, svnc, hr, variant=0):
    gather = rk.make_compute_local_strain(hc, svnc)
    solve = rk.make_solve_local_stress(hv, hd, hr)
    scatter = rk.make_apply_stress_delta(svnc)

    def full(color: int, launch_dim: int, color_offsets: wp.array[int], color_blocks: wp.array2d[int],
             yield_params: wp.array[YP], strain_node_volume: wp.array[float], compliance_mat_offsets: wp.array[int],
             compliance_mat_columns: wp.array[int], compliance_mat_values: wp.array[mat66],
             strain_mat_offsets: wp.array[int], strain_mat_columns: wp.array[int],
             strain_mat_values: wp.array[mat13], delassus_diagonal: wp.array[vec6],
             delassus_rotation: wp.array[mat55], inv_mass_matrix: wp.array[float], local_strain_rhs: wp.array[vec6],
             velocities: wp.array[wp.vec3], local_stress: wp.array[vec6], delta_correction: wp.array[vec6],
             saved: wp.array[vec6], mode: int):
        i = wp.tid()
        color_beg = color_offsets[color] + i
        color_end = color_offsets[color + 1]
        for color_offset in range(color_beg, color_end, launch_dim):
            beg, end = color_blocks[0, color_offset], color_blocks[1, color_offset]
            for tau_i in range(beg, end):
                if mode == 100 + wp.static(variant):  # never true; makes each variant's module source distinct
                    delta_correction[tau_i] = vec6(1.0)
                if mode == 1:  # index walk only
                    delta_correction[tau_i] = vec6(float(tau_i))
                    continue
                if strain_mat_offsets[tau_i] == strain_mat_offsets[tau_i + 1]:
                    delta_correction[tau_i] = vec6(0.0)
                    continue
                if mode == 5:  # scatter only, from a saved delta
                    wp.static(scatter)(tau_i, saved[tau_i], strain_mat_offsets, strain_mat_columns,
                                       strain_mat_values, inv_mass_matrix, velocities)
                    continue
                if mode == 4:  # solve only, from a saved local strain
                    local_strain = saved[tau_i]
                else:
                    local_strain = wp.static(gather)(tau_i, compliance_mat_offsets, compliance_mat_columns,
                                                     compliance_mat_values, strain_mat_offsets, strain_mat_columns,
                                                     strain_mat_values, local_strain_rhs, velocities, local_stress)
                if mode == 2:  # gather only
                    delta_correction[tau_i] = local_strain
                    continue
                delta_stress = wp.static(solve)(tau_i, local_strain, yield_params, strain_node_volume,
                                                delassus_diagonal, delassus_rotation, local_stress)
                if mode == 3 or mode == 4:  # gather + solve / solve only
                    delta_correction[tau_i] = delta_stress
                    continue
                local_stress[tau_i] += delta_stress
                delta_correction[tau_i] = delta_stress
                wp.static(scatter)(tau_i, delta_stress, strain_mat_offsets, strain_mat_columns, strain_mat_values,
                                   inv_mass_matrix, velocities)

    def empty(color: int, launch_dim: int, color_offsets: wp.array[int], out: wp.array[vec6]):
        i = wp.tid()
        if i < 0:
            out[0] = vec6(float(color + launch_dim + color_offsets[0]))

    return wp.kernel(full, module="unique"), wp.kernel(empty, module="unique")


@wp.kernel
def walk_small(color: int, launch_dim: int, color_offsets: wp.array[int], color_blocks: wp.array2d[int],
               out: wp.array[vec6]):
    """The index walk of the GS kernel (mode 1) as a small kernel with few arguments."""
    i = wp.tid()
    color_beg = color_offsets[color] + i
    color_end = color_offsets[color + 1]
    for color_offset in range(color_beg, color_end, launch_dim):
        beg, end = color_blocks[0, color_offset], color_blocks[1, color_offset]
        for tau_i in range(beg, end):
            out[tau_i] = vec6(float(tau_i))


@wp.kernel
def walk_small_load(color: int, launch_dim: int, color_offsets: wp.array[int], color_blocks: wp.array2d[int],
                    offsets: wp.array[int], out: wp.array[vec6]):
    """walk_small plus the strain-matrix row offsets read (one more dependent load level)."""
    i = wp.tid()
    color_beg = color_offsets[color] + i
    color_end = color_offsets[color + 1]
    for color_offset in range(color_beg, color_end, launch_dim):
        beg, end = color_blocks[0, color_offset], color_blocks[1, color_offset]
        for tau_i in range(beg, end):
            out[tau_i] = vec6(float(offsets[tau_i + 1] - offsets[tau_i]))


MODES = {"full": 0, "index": 1, "gather": 2, "gather_solve": 3, "solve": 4, "scatter": 5}
result = {}


def probe(rs, cs):
    rh = rs.rheology
    dev = rs.device
    flags = (rh.has_viscosity, rh.has_dilatancy, rh.compliance_mat.nnz > 0, rh.strain_velocity_node_count,
             not rk._ISOTROPIC_LOCAL_LHS)
    k_ref, k_empty = make_kernels(*flags)
    k_fast = None
    if not args.no_fast:
        k_fast, _ = make_kernels(*flags, variant=1)
        _FAST["on"] = True  # only the fast copy's module is compiled while the rewrite is on
        try:
            k_fast.module.load(dev, block_dim=64)
        finally:
            _FAST["on"] = False
    offs = rh.color_offsets.numpy()
    ncol = len(offs) - 1
    sizes = np.diff(offs)
    blk = rh.color_blocks.numpy()
    nodes_per_color = [int((blk[1, offs[c]:offs[c + 1]] - blk[0, offs[c]:offs[c + 1]]).sum()) for c in range(ncol)]
    smat_nnz = rh.strain_mat.offsets.numpy()
    result.update(colors=ncol, color_blocks=sizes.tolist(), strain_nodes_per_color=nodes_per_color,
                  strain_nodes=int(rh.stress.shape[0]), velocity_nodes=int(rs.momentum.velocity.shape[0]),
                  strain_nnz_per_node=float(np.diff(smat_nnz).mean()), flags=[int(f) for f in flags],
                  sm_count=dev.sm_count)
    vel0 = wp.clone(rs.momentum.velocity)
    stress0 = wp.clone(rh.stress)
    saved_strain = wp.zeros_like(rh.stress)
    saved_delta = wp.zeros_like(rh.stress)
    scratch = wp.zeros_like(rh.stress)

    def inputs(c, launch_dim, saved, mode):
        return [c, launch_dim, rh.color_offsets, rh.color_blocks, rh.yield_params, rh.strain_node_volume,
                rh.compliance_mat.offsets, rh.compliance_mat.columns, rh.compliance_mat.values, rh.strain_mat.offsets,
                rh.strain_mat.columns, rh.strain_mat.values.view(dtype=mat13), rs.delassus_operator.delassus_diagonal,
                rs.delassus_operator.delassus_rotation, rs.momentum.inv_volume, rh.elastic_strain_delta,
                rs.momentum.velocity, rh.stress, scratch, saved, mode]

    def sweep_fn(kernel, mode, block_dim=64, blocks=None, saved=None):
        blocks = blocks or 2 * dev.sm_count
        launch_dim = blocks * block_dim

        def fn():
            for c in range(ncol):
                if kernel is k_empty:
                    wp.launch(k_empty, dim=launch_dim, inputs=[c, launch_dim, rh.color_offsets, scratch],
                              block_dim=block_dim, max_blocks=blocks)
                else:
                    wp.launch(kernel, dim=launch_dim, inputs=inputs(c, launch_dim, saved if saved is not None else scratch, mode),
                              block_dim=block_dim, max_blocks=blocks)
        return fn

    def newton_sweep():
        rs.solve()

    def timed(fn):
        """us per color pass at sustained clocks: graph of K sweeps replayed back to back for --seconds."""
        wp.copy(rs.momentum.velocity, vel0)
        wp.copy(rh.stress, stress0)
        fn()  # loads modules
        wp.synchronize_device()
        k = 40
        with wp.ScopedCapture(force_module_load=False) as cap:
            for _ in range(k):
                fn()
        per = []
        t_end = time.perf_counter() + args.seconds
        while time.perf_counter() < t_end:
            wp.copy(rs.momentum.velocity, vel0)
            wp.copy(rh.stress, stress0)
            wp.synchronize_device()
            t0 = time.perf_counter()
            wp.capture_launch(cap.graph)
            wp.synchronize_device()
            per.append((time.perf_counter() - t0) * 1e6 / (k * ncol))
        per = per[len(per) // 2:]
        return float(np.median(per))

    # saved inputs for the solve-only / scatter-only modes: one gather pass and one gather+solve pass
    for c in range(ncol):
        wp.launch(k_ref, dim=2 * dev.sm_count * 64, inputs=inputs(c, 2 * dev.sm_count * 64, scratch, 2), block_dim=64,
                  max_blocks=2 * dev.sm_count)
    wp.copy(saved_strain, scratch)
    for c in range(ncol):
        wp.launch(k_ref, dim=2 * dev.sm_count * 64, inputs=inputs(c, 2 * dev.sm_count * 64, scratch, 3), block_dim=64,
                  max_blocks=2 * dev.sm_count)
    wp.copy(saved_delta, scratch)
    wp.synchronize_device()

    t = {}
    t["warm (discard)"] = timed(newton_sweep)
    t["newton"] = timed(newton_sweep)
    t["empty"] = timed(sweep_fn(k_empty, 0))
    ld = 2 * dev.sm_count * 64

    def small(kernel, extra):
        def fn():
            for c in range(ncol):
                wp.launch(kernel, dim=ld, inputs=[c, ld, rh.color_offsets, rh.color_blocks, *extra, scratch],
                          block_dim=64, max_blocks=2 * dev.sm_count)
        return fn

    # Newton's own kernel with every color empty (color_offsets all zero): its fixed per-dispatch cost
    zeros = wp.zeros_like(rh.color_offsets)
    launch = rs.solve_local_launch

    def newton_zero():
        launch.set_param_at_index(2, zeros)
        for c in range(ncol):
            launch.set_param_at_index(0, c)
            launch.launch()
        launch.set_param_at_index(2, rh.color_offsets)

    t["newton_zero_work"] = timed(newton_zero)
    t["full_zero_work"] = timed(lambda: [wp.launch(k_ref, dim=ld, inputs=[c, ld, zeros] + inputs(c, ld, scratch, 0)[3:],
                                                   block_dim=64, max_blocks=2 * dev.sm_count) for c in range(ncol)])
    t["index_small"] = timed(small(walk_small, []))
    t["index_small_offsets"] = timed(small(walk_small_load, [rh.strain_mat.offsets]))
    for name, mode in MODES.items():
        saved = saved_strain if name == "solve" else saved_delta if name == "scatter" else None
        t[name] = timed(sweep_fn(k_ref, mode, saved=saved))
        if name in ("full", "gather_solve", "solve") and k_fast is not None:
            t[name + "_fast"] = timed(sweep_fn(k_fast, mode, saved=saved))
    for bd, nb in ((32, 2 * dev.sm_count), (32, 4 * dev.sm_count), (64, 4 * dev.sm_count), (128, dev.sm_count),
                   (64, dev.sm_count), (32, 8 * dev.sm_count)):
        t[f"full_bd{bd}_blocks{nb}"] = timed(sweep_fn(k_ref, 0, block_dim=bd, blocks=nb))
    t.pop("warm (discard)")
    t["contact_solve_per_iteration"] = timed(lambda: [cs.solve() for _ in range(ncol)]) * 1.0  # per call (ncol calls)
    result["us_per_color_pass"] = t
    result["fast_rewrites"] = _FAST["rewrites"]
    wp.copy(rs.momentum.velocity, vel0)
    wp.copy(rh.stress, stress0)


cuda_loop = sr._run_solver_loop


def hook(rheology_solver, contact_solver, *rest):
    if result.get("armed") and "us_per_color_pass" not in result and rheology_solver.name == "Gauss-Seidel":
        probe(rheology_solver, contact_solver)
    return cuda_loop(rheology_solver, contact_solver, *rest)


newton_compat.install(accumulate="float32", loop="cuda", log=False)
cuda_loop = sr._run_solver_loop
sr._run_solver_loop = hook
kw = {k: json.loads(v) for k, v in (s.split("=", 1) for s in args.set)}
with wp.ScopedDevice(args.device):
    sc = scenes.make_scene(args.scene, **kw)
    for _ in range(args.warmup):
        sc.step()
    result["armed"] = True
    sc.step()
result.pop("armed")
result["scene"] = args.scene
print(json.dumps({k: v for k, v in result.items() if k not in ("color_blocks",)}, indent=1))
if args.json:
    os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
    json.dump(result, open(args.json, "w"), indent=1)

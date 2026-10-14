"""Run Newton 1.5.2's ``SolverImplicitMPM`` on Metal with the semantics it has on CUDA.

Isaac Lab 3.0-EA's MPM is Newton 1.5.2's implicit MPM (docs/research/mpm_2026-10-09.md). Two things stop that
solver from meaning the same thing on Metal as on the L4:

1. ``warp.fem.integrate`` accumulates in float64 by default and Metal has no float64. ``install(accumulate="float32")``
   makes float32 the default accumulation type. On CUDA the same switch gives the precision-control recording (R0).
2. The rheology iteration loop has two implementations with different stopping rules. With conditional CUDA graphs
   (the L4, i.e. Isaac) it checks convergence every 5 iterations with the ``update_condition`` kernel, counting from
   1, and stops once converged or past ``max_iterations``. Without them (Metal, CPU) it runs
   ``max_iterations // solve_granularity`` host batches of 25 (Gauss-Seidel) or 50 (Jacobi) iterations. A Jacobi
   solve capped at 24 iterations (Isaac Lab's Franka-Pour task) therefore runs 25 iterations on CUDA and none at
   all on Metal. ``install(loop="cuda")`` replaces the loop with a host loop that launches the same kernels in the
   same order as the graph body and decides with the same ``update_condition`` kernel, so both devices stop at the
   same iteration.

3. The coupled examples' rigid side (MuJoCo Warp through ``SolverMuJoCo``) sums with float atomics across more than
   one SIMD-group (the solver's reductions over 36 dofs in ``example_mpm_twoway_coupling``). On CUDA the warps of
   such small launches tend to reach the atomics in launch order (repeated ANYmal runs on the L4 were byte-identical;
   twoway repeatability on the L4 is unmeasured); on Metal the SIMD-groups arrive in varying order and two identical
   runs drift apart (twoway: frame 54). ``install(rigid_deterministic=True)`` makes ``SolverMuJoCo`` on Metal
   default to ``wp.DeterministicMode.RUN_TO_RUN`` and ``newton.CollisionPipeline`` to ``deterministic=True``.
   It also builds ``SolverCoupled``'s kernels in RUN_TO_RUN mode: the coupled solver compacts each entry's contacts
   with an atomic counter, so without it the entry solver saw them in thread-arrival order. That made Isaac Lab's
   Franka-Pour scene differ run to run from its first step (about 2e-7), even with everything above. With it, the
   scene repeats bitwise at 64 and 2,048 worlds, eager and captured, at the same cost per step.
   The Warp fork (branch ``metalsim-metal-det-sort``, merged into ``metalsim-integrated``) implements RUN_TO_RUN
   on Metal as on CUDA, scalably: float accumulation atomics add into exact per-element integer accumulators (order
   cannot change the bits), and counters that hand out slots run in two phases with a GPU sort and scan, so slots
   follow thread order. The Newton fork (branch ``metalsim-rigid-det-sort``, merged into ``metalsim-integrated``)
   fixes the collision pipeline's contact-reduction hash table, whose 64-bit compare-and-swap was not atomic on Metal
   (the integrated Warp branch now refuses such atomics on Metal instead of running them as a plain
   read-modify-write). Repeatable bitwise at 1 to 2048 worlds. Set the module option
   ``metal_ordered_atomics`` to get the earlier ordered launches (CUDA's warp-then-lane order in one SIMD-group).

4. Newton marks its rheology kernels ``fast_math``. On CUDA (NVRTC ``--use_fast_math``) every float32 division in them
   is ``div.approx.ftz.f32`` (SASS ``MUFU.RCP`` + ``FMUL.FTZ``) and every sqrt ``sqrt.approx.ftz.f32`` (``MUFU.SQRT``),
   measured on the L4. ``install(metal_fast_math=...)`` picks the Metal form (``wp.config.metal_fast_math``, Warp fork
   ``metalsim`` from ``91b8a859``):``"cuda"`` reproduces both instructions bit for bit (integer emulation of NVIDIA's
   interpolator, research_notes/.../cuda_approx_math.md), ``"relaxed"`` uses Metal's own fast divide / sqrt (Apple's
   approximations, not NVIDIA's), ``"off"`` compiles them correctly rounded. Default ``"off"`` (MPM.md section 8, 1b):
   against the L4's 16-run floor ensemble ``"cuda"`` does not do better than ``"off"`` (anymal 1.01 vs 0.99 x the
   largest floor, multi_material 1.20 vs 1.15 x) and costs 1.2-2.2x ``"off"``'s ms/frame (exact MUFU results are
   about 3x a correctly rounded division on Apple GPUs); ``"relaxed"`` is 0.64-0.94x ``"off"``'s time but leaves the
   floor on twoway, anymal and the plate.

Every solve's iteration count and final residuals are appended to ``SOLVE_LOG`` (the parity rule: a speed number is
only comparable at the residual it reached).
"""
from __future__ import annotations

import functools
import importlib

import numpy as np
import warp as wp
import warp.fem as fem

SOLVE_LOG: list[dict] = []
METAL_FAST_MATH_MODES = ("off", "cuda", "relaxed")
METAL_FAST_MATH_DEFAULT = "off"
_STATE: dict = {"installed": False, "run_solver_loop": {}}
VENDORED_MODULE = "metalsim.physics.mpm.implicit_mpm.solve_rheology"


def newton_is_152() -> bool:
    """True when the environment's Newton is the 1.5.2 line (``.venv-newton152``): its own implicit MPM is the solver
    MetalSim's parity work is about. On the Newton 1.7 line (``.venv``) Newton's implicit MPM is a later algorithm and
    MetalSim uses its vendored copy of the 1.5.2 fork's solver instead (``metalsim/physics/mpm/implicit_mpm``)."""
    try:
        import newton  # noqa: PLC0415
    except ImportError:
        return False
    return str(newton.__version__).startswith("1.5.")


def rheology_modules() -> list:
    """The ``solve_rheology`` modules ``install`` patches: Newton's own when Newton is 1.5.2, and MetalSim's vendored
    copy of the same file (``metalsim.physics.mpm.implicit_mpm``) in every environment. Newton 1.7's own implicit MPM
    is never patched (its loop has another signature, and MetalSim does not run it)."""
    mods = []
    if newton_is_152():
        mods.append(importlib.import_module("newton._src.solvers.implicit_mpm.solve_rheology"))
    mods.append(importlib.import_module(VENDORED_MODULE))
    return mods


def install(accumulate: str = "float32", loop: str = "cuda", log: bool = True, chunk_graph: bool = False,
            fused: bool = False, fused_max_color_blocks: int = 1 << 30, simd: bool = False,
            gpu_exit: bool = False, rigid_deterministic: bool | None = True, device_args: bool | None = True,
            metal_fast_math: str | None = METAL_FAST_MATH_DEFAULT) -> None:
    """Patch Newton's implicit MPM in this process (and MetalSim's vendored copy of it, :func:`rheology_modules`).

    Args:
        accumulate: ``"float32"`` (required on Metal) or ``"native"`` (Warp's default, float64; CUDA/CPU only).
        loop: ``"cuda"`` (the conditional-graph stopping rule, run from the host; works on every device) or
            ``"native"`` (Newton's own loop for the device).
        log: append ``{iterations, res_l2, res_linf, solver}`` per solve to ``SOLVE_LOG``.
        chunk_graph: on Metal, record the 5-iteration chunk once per solve and replay it (``loop="cuda"`` only).
        fused: on Metal, run eligible solves (plain Gauss-Seidel, nodal contact, no rigid coupling, one environment,
            at most ``fused_max_color_blocks`` blocks in a color) as one persistent-threadgroup dispatch
            (:mod:`metalsim.physics.mpm.fused_gs`).
        simd: on Metal, run eligible Gauss-Seidel sweeps with 8 SIMD lanes per strain node
            (:mod:`metalsim.physics.mpm.simd_gs`; bit-identical to Newton's sweep).
        gpu_exit: with ``chunk_graph`` on Metal, decide the stop on the GPU: the recorded chunk (including
            ``update_condition``) is an indirect-command-buffer execution range whose length a gate kernel zeroes once
            the condition is 0; the host submits the maximum number of chunk replays without reading anything back
            and reads the iteration count once at the end. Same kernels, same order, same stopping iteration.
        rigid_deterministic: on Metal, construct ``SolverMuJoCo`` with ``deterministic=wp.DeterministicMode.RUN_TO_RUN``
            unless the caller (or ``wp.config.deterministic``) chose a mode, and ``newton.CollisionPipeline`` with
            ``deterministic=True`` unless the caller chose (point 3 above). No effect on CUDA/CPU.
            ``None`` leaves Newton's rigid solvers alone (MetalSim's main simulator: the rigid side is ``BatchSim``).
        device_args: set ``wp.config.metal_device_args`` (Warp branch ``metalsim-integrated``; read when a module is
            built): Metal kernels read their arguments from device instead of constant memory, which cuts the fixed
            cost of each dispatch (Newton's MPM examples: 3-6 ms per frame, bitwise the same). Warp's default is
            constant memory: MuJoCo Warp's G1 physics step is slower with device arguments, and the Metal compiler
            service fails on its ``_tendon_dot`` kernel built that way (models with tendons, e.g. the Franka panda).
            ``None`` leaves the setting alone. No effect on CUDA/CPU.
        metal_fast_math: how ``fast_math`` modules (Newton's rheology solve) compile on Metal (point 4 above):
            ``"off"`` (correctly rounded division and sqrt), ``"cuda"`` (CUDA's ``--use_fast_math`` results bit for bit)
            or ``"relaxed"`` (Metal's fast divide / sqrt). Sets ``wp.config.metal_fast_math``, read when a module is
            built, so call ``install`` before the solver's kernels load. ``None`` leaves the setting alone. With a
            Warp tree that predates the setting (fast_math always relaxed on Metal), ``"off"`` makes every module
            resolve ``fast_math`` to False and ``"cuda"`` raises. No effect on CUDA/CPU kernels.
    """
    modules = rheology_modules()
    if not _STATE["installed"]:
        _STATE["fem_integrate"] = fem.integrate
        _STATE["installed"] = True
    for sr in modules:
        _STATE["run_solver_loop"].setdefault(sr.__name__, sr._run_solver_loop)

    orig_integrate = _STATE["fem_integrate"]
    if accumulate == "float32":

        @functools.wraps(orig_integrate)
        def integrate_f32(*args, **kwargs):
            kwargs.setdefault("accumulate_dtype", wp.float32)
            return orig_integrate(*args, **kwargs)

        fem.integrate = integrate_f32
    elif accumulate == "native":
        fem.integrate = orig_integrate
    else:
        raise ValueError(f"accumulate must be 'float32' or 'native', got {accumulate!r}")

    if loop not in ("cuda", "native"):
        raise ValueError(f"loop must be 'cuda' or 'native', got {loop!r}")
    for sr in modules:
        if loop == "cuda":
            sr._run_solver_loop = functools.partial(_run_solver_loop_cuda_semantics, sr, log)
        else:
            orig_loop = _STATE["run_solver_loop"][sr.__name__]
            sr._run_solver_loop = functools.partial(_run_native_logged, sr, orig_loop, log) if log else orig_loop
    if rigid_deterministic is not None:
        _install_rigid_determinism(rigid_deterministic)
    _install_metal_fast_math(metal_fast_math)
    if device_args is not None and hasattr(wp.config, "metal_device_args"):
        _STATE.setdefault("metal_device_args", wp.config.metal_device_args)
        wp.config.metal_device_args = bool(device_args)
    _STATE.update(accumulate=accumulate, loop=loop, chunk_graph=chunk_graph, fused=fused,
                  fused_max_color_blocks=fused_max_color_blocks, simd=simd, gpu_exit=gpu_exit,
                  rigid_deterministic=rigid_deterministic,
                  metal_fast_math=getattr(wp.config, "metal_fast_math", _STATE.get("fast_math_fallback")))


def _install_metal_fast_math(mode: str | None) -> None:
    """Select the Metal form of ``fast_math`` modules (``install(metal_fast_math=...)``)."""
    if mode is None:
        return
    if mode not in METAL_FAST_MATH_MODES:
        raise ValueError(f"metal_fast_math must be one of {METAL_FAST_MATH_MODES}, got {mode!r}")
    from warp._src.context import Module  # noqa: PLC0415

    if "resolve_options" in _STATE:  # undo an earlier fallback
        Module.resolve_options = _STATE.pop("resolve_options")
        _STATE.pop("fast_math_fallback", None)
    if hasattr(wp.config, "metal_fast_math"):
        _STATE.setdefault("metal_fast_math_orig", wp.config.metal_fast_math)
        wp.config.metal_fast_math = mode
        return
    # Warp without the setting: fast_math is honoured on Metal as relaxed mode (integrated fork) or ignored
    if mode == "cuda":
        raise RuntimeError("metal_fast_math='cuda' needs the Warp fork's metalsim branch from 91b8a859 on (wp.config.metal_fast_math)")
    if mode == "off" and wp.is_metal_available() and not wp.is_cuda_available():
        orig = Module.resolve_options
        _STATE["resolve_options"] = orig

        def resolve_no_fast_math(self, *a, **kw):
            o = orig(self, *a, **kw)
            o["fast_math"] = False
            return o

        Module.resolve_options = resolve_no_fast_math
    _STATE["fast_math_fallback"] = mode


def _install_rigid_determinism(enabled: bool) -> None:
    """Default ``SolverMuJoCo(deterministic=...)`` to RUN_TO_RUN and ``CollisionPipeline(deterministic=...)`` to True
    on Metal devices (see the module docstring, point 3)."""
    mode = getattr(wp, "DeterministicMode", None)
    if mode is None:  # Warp without deterministic mode (before 1.15)
        return
    import newton  # noqa: PLC0415
    from newton.solvers import SolverMuJoCo  # noqa: PLC0415

    if "solver_mujoco_init" not in _STATE:
        _STATE["solver_mujoco_init"] = SolverMuJoCo.__init__
        _STATE["collision_pipeline_init"] = newton.CollisionPipeline.__init__
    orig_init = _STATE["solver_mujoco_init"]
    orig_pipeline_init = _STATE["collision_pipeline_init"]
    _set_coupled_determinism(enabled, mode)
    if not enabled:
        SolverMuJoCo.__init__ = orig_init
        newton.CollisionPipeline.__init__ = orig_pipeline_init
        return

    @functools.wraps(orig_pipeline_init)
    def pipeline_init(self, model, *args, deterministic=None, **kwargs):
        if deterministic is None:
            deterministic = bool(getattr(model.device, "is_metal", False))
        orig_pipeline_init(self, model, *args, deterministic=deterministic, **kwargs)

    newton.CollisionPipeline.__init__ = pipeline_init

    @functools.wraps(orig_init)
    def init(self, model, *args, deterministic=None, **kwargs):
        if (deterministic is None and getattr(model.device, "is_metal", False)
                and wp.config.deterministic == mode.NOT_GUARANTEED):
            deterministic = mode.RUN_TO_RUN
        orig_init(self, model, *args, deterministic=deterministic, **kwargs)

    SolverMuJoCo.__init__ = init


def _coupled_modules():
    """Newton's coupled-solver kernel modules (``SolverCoupled``, ``SolverCoupledProxy``), the ones imported."""
    modules = []
    for name in ("newton._src.solvers.coupled.solver_coupled", "newton._src.solvers.coupled.solver_coupled_proxy"):
        try:
            modules.append(importlib.import_module(name))
        except ImportError:
            pass
    return modules


def _set_coupled_determinism(enabled: bool, mode) -> None:
    """On Metal, build ``SolverCoupled``'s kernels in RUN_TO_RUN mode (``install(rigid_deterministic=True)``).

    The coupled solver compacts each entry's rigid and soft contacts with an atomic counter
    (``_filter_*_contacts_global_shape_ids_kernel``), so the entry solver (MuJoCo Warp) saw them in thread-arrival
    order: Isaac Lab's Franka-Pour scene differed run to run from its first step even with the rigid side
    deterministic. Warp's deterministic mode hands out the slots in thread order and orders the module's float sums.
    Not on CUDA/CPU (Newton as shipped there), and not when ``wp.config.deterministic`` already chose a mode.
    """
    if not wp.is_metal_available() or wp.config.deterministic != mode.NOT_GUARANTEED:
        return
    want = mode.RUN_TO_RUN if enabled else mode.NOT_GUARANTEED
    if not enabled and not _STATE.get("coupled_deterministic"):
        return
    for module in _coupled_modules():
        wp.set_module_options({"deterministic": want}, module=module)
    _STATE["coupled_deterministic"] = enabled


def uninstall() -> None:
    if not _STATE["installed"]:
        return
    fem.integrate = _STATE["fem_integrate"]
    for sr in rheology_modules():
        if sr.__name__ in _STATE["run_solver_loop"]:
            sr._run_solver_loop = _STATE["run_solver_loop"][sr.__name__]
    if _STATE.get("rigid_deterministic") is not None:
        _install_rigid_determinism(False)
    if "metal_device_args" in _STATE:
        wp.config.metal_device_args = _STATE.pop("metal_device_args")
    if "metal_fast_math_orig" in _STATE:
        wp.config.metal_fast_math = _STATE.pop("metal_fast_math_orig")
    if "resolve_options" in _STATE:
        from warp._src.context import Module  # noqa: PLC0415

        Module.resolve_options = _STATE.pop("resolve_options")
        _STATE.pop("fast_math_fallback", None)
    _STATE.update(installed=False, accumulate=None, loop=None, rigid_deterministic=False)


def settings() -> dict:
    return {"accumulate": _STATE.get("accumulate"), "loop": _STATE.get("loop"),
            "chunk_graph": _STATE.get("chunk_graph", False), "fused": _STATE.get("fused", False),
            "fused_max_color_blocks": _STATE.get("fused_max_color_blocks"), "simd": _STATE.get("simd", False),
            "gpu_exit": _STATE.get("gpu_exit", False), "rigid_deterministic": _STATE.get("rigid_deterministic", False),
            "metal_fast_math": _STATE.get("metal_fast_math")}


# Mirrors solve_rheology._run_solver_loop's graph branch (Newton 1.5.2, solve_rheology.py:1791-1858).
_GRAPH_GRANULARITY = 5


@wp.kernel
def _close_range_when_stopped(condition: wp.array[int], ranges: wp.array[wp.uint32]):
    """After ``update_condition``: once the loop condition is 0, zero the chunk range's length so every later replay
    of the chunk graph executes nothing (indirect execution range: location, length, full length, pad)."""
    if condition[0] == 0:
        ranges[1] = wp.uint32(0)


@wp.kernel
def _reset_ranges(n: int, ranges: wp.array[wp.uint32]):
    """Start of a captured solve: every chunk's execution range back to its full length (a previous replay may have
    zeroed some)."""
    for j in range(n):
        ranges[4 * j + 1] = ranges[4 * j + 2]


@wp.kernel
def _close_later_ranges(condition: wp.array[int], chunk: int, n: int, ranges: wp.array[wp.uint32]):
    """After chunk ``chunk``'s ``update_condition`` inside a captured solve: once the loop condition is 0, zero the
    lengths of the remaining chunks' execution ranges."""
    if condition[0] == 0:
        for j in range(chunk + 1, n):
            ranges[4 * j + 1] = wp.uint32(0)


# Arrays referenced by recorded graphs (range buffers of captured solves) must outlive the graphs.
_GRAPH_KEEPALIVE: list = []


def _range_api():
    """The Warp fork's indirect-execution-range capture API (``metal_capture_range_begin`` / ``_end``), or None."""
    import warp._src.context as ctx  # noqa: PLC0415

    return ctx if hasattr(ctx, "metal_capture_range_begin") else None


def _max_chunks(max_iterations: int) -> int:
    """Chunks the CUDA rule can run: the iteration counter starts at 1 and the loop stops once it exceeds
    ``max_iterations`` (or converges), checking every ``_GRAPH_GRANULARITY`` iterations."""
    return (max(int(max_iterations), 1) - 1) // _GRAPH_GRANULARITY + 1


def _residual_norms(sr, rheology_solver, residual_np, l2_tolerance_scale):
    host_scale = (
        l2_tolerance_scale
        if rheology_solver.rheology.strain_environment_offsets is None
        else l2_tolerance_scale.numpy()
    )
    return sr._nonlinear_solver_result_norms(residual_np, host_scale)


def _run_solver_loop_cuda_semantics(
    sr, log, rheology_solver, contact_solver, max_iterations, tolerance, l2_tolerance_scale, use_graph, verbose,
    temporary_store,
):
    device = rheology_solver.device
    iteration_and_condition = fem.borrow_temporary(temporary_store, shape=(2,), dtype=int)
    iteration_and_condition.fill_(1)
    iteration = iteration_and_condition[:1]
    condition = iteration_and_condition[1:]
    batched = rheology_solver.rheology.strain_environment_offsets is not None

    out = {}
    if _STATE.get("simd"):
        from metalsim.physics.mpm import simd_gs

        if simd_gs.eligible(rheology_solver):
            rheology_solver.solve = simd_gs.SimdSweep(rheology_solver)

    def chunk():
        for _k in range(_GRAPH_GRANULARITY):
            contact_solver.solve()
            rheology_solver.solve()
        residual = rheology_solver.eval_residual()
        out["residual"] = residual
        if not batched:
            wp.launch(
                sr.update_condition,
                dim=1,
                inputs=[tolerance * tolerance, l2_tolerance_scale * l2_tolerance_scale, _GRAPH_GRANULARITY,
                        max_iterations, residual, iteration, condition],
                device=device,
            )
        else:
            wp.launch(
                sr.update_batched_condition,
                dim=1,
                inputs=[tolerance * tolerance, l2_tolerance_scale, _GRAPH_GRANULARITY, max_iterations, residual,
                        iteration, condition],
                device=device,
            )

    use_fused = False
    if _STATE.get("fused") and not batched:
        from metalsim.physics.mpm import fused_gs

        use_fused = fused_gs.eligible(rheology_solver, contact_solver, device, _STATE["fused_max_color_blocks"])
    if use_fused:
        residual_out = fem.borrow_temporary(temporary_store, shape=(2,), dtype=float)
        fused_gs.run(rheology_solver, contact_solver, max_iterations, tolerance, l2_tolerance_scale,
                     iteration_and_condition, residual_out)
        it_cond = iteration_and_condition.numpy()
        r = residual_out.numpy()
        out["residual"] = wp.array(r.reshape(2, 1), dtype=float, device="cpu")
        residual_out.release()
    elif (_STATE.get("gpu_exit") and getattr(device, "is_metal", False) and device.is_capturing
          and _range_api() is not None):
        # Inside an outer capture (e.g. a whole simulation step): unroll the maximum number of chunks, each recorded as
        # its own indirect execution range; once update_condition stops the loop, the gate zeroes the later ranges.
        # Same kernels, same order, same stopping iteration; nothing is read back.
        ctx = _range_api()
        n = _max_chunks(max_iterations)
        ranges = wp.empty(4 * n, dtype=wp.uint32, device=device)
        _GRAPH_KEEPALIVE.append(ranges)
        # (iteration, condition) = (1, 1) every replay: fill_ above was recorded into the graph
        wp.launch(_reset_ranges, dim=1, inputs=[n, ranges], device=device)
        for j in range(n):
            ctx.metal_capture_range_begin(ranges, j, device=device)
            chunk()
            wp.launch(_close_later_ranges, dim=1, inputs=[condition, j, n, ranges], device=device)
            ctx.metal_capture_range_end(device=device)
        it_cond = None
    elif _STATE.get("chunk_graph") and getattr(device, "is_metal", False) and not device.is_capturing:
        # Record the chunk once per solve and replay it: same kernels, same order, same stopping rule.
        ctx = _range_api()
        gpu_exit = _STATE.get("gpu_exit") and ctx is not None
        ranges = wp.empty(4, dtype=wp.uint32, device=device) if gpu_exit else None
        with wp.ScopedCapture(device=device, force_module_load=False) as capture:
            gated = gpu_exit and ctx.metal_capture_range_begin(ranges, 0, device=device)
            chunk()
            if gated:
                wp.launch(_close_range_when_stopped, dim=1, inputs=[condition, ranges], device=device)
                ctx.metal_capture_range_end(device=device)
        graph = capture.graph
        if gated:
            # the range's fields are written at capture end; replays past the stop execute an empty range
            for _ in range(_max_chunks(max_iterations)):
                wp.capture_launch(graph)
            # the iteration count is only read for the log: without it nothing waits for the solve, and the host
            # goes on encoding the rest of the step while the GPU iterates (the counter is a stream-ordered
            # temporary; the graph's buffers are freed only once no GPU work can use them)
            it_cond = iteration_and_condition.numpy() if (log or verbose) else None
            if it_cond is None:
                iteration_and_condition.release()
                return None
        else:
            while True:
                wp.capture_launch(graph)
                it_cond = iteration_and_condition.numpy()
                if it_cond[1] == 0:
                    break
    else:
        while True:
            chunk()
            it_cond = iteration_and_condition.numpy()
            if it_cond[1] == 0:
                break
    residual = out["residual"]

    if (log or verbose) and it_cond is not None:
        res_l2, res_linf = _residual_norms(sr, rheology_solver, residual.numpy(), l2_tolerance_scale)
        n_iter = int(it_cond[0]) - 1
        if log:
            SOLVE_LOG.append({"iterations": n_iter, "res_l2": res_l2, "res_linf": res_linf,
                              "solver": rheology_solver.name})
        if verbose:
            print(f"{rheology_solver.name} terminated after {n_iter} iterations with residuals {res_l2}, {res_linf}")
    if it_cond is None:
        _GRAPH_KEEPALIVE.append(iteration_and_condition)  # referenced by the recorded graph on every replay
    else:
        iteration_and_condition.release()
    return None


def _run_native_logged(sr, orig_loop, log, rheology_solver, contact_solver, max_iterations, tolerance,
                       l2_tolerance_scale, use_graph, verbose, temporary_store):
    """Newton's own loop; the residual is re-evaluated afterwards (read-only) for the log.

    The iteration count is not observable from outside the native loop, so it is logged as -1.
    """
    graph = orig_loop(rheology_solver, contact_solver, max_iterations, tolerance, l2_tolerance_scale, use_graph,
                      verbose, temporary_store)
    res_l2, res_linf = _residual_norms(sr, rheology_solver, rheology_solver.eval_residual().numpy(),
                                       l2_tolerance_scale)
    SOLVE_LOG.append({"iterations": -1, "res_l2": res_l2, "res_linf": res_linf, "solver": rheology_solver.name})
    return graph


def drain_log() -> dict:
    """Return and clear the per-solve log as arrays."""
    out = {
        "iterations": np.array([e["iterations"] for e in SOLVE_LOG], dtype=np.int32),
        "res_l2": np.array([e["res_l2"] for e in SOLVE_LOG], dtype=np.float64),
        "res_linf": np.array([e["res_linf"] for e in SOLVE_LOG], dtype=np.float64),
    }
    SOLVE_LOG.clear()
    return out

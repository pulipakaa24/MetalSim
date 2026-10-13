"""Franka-Pour MPM at N worlds (scale_pour.py's ``--part mpm`` scene and solver config): bitwise recordings, per-kernel
GPU profile and step timing, for checking that a scaling change leaves the trajectory bit for bit unchanged.

    record   fresh solver from t = 0 (per-world phase offsets, as scale_pour), ``--steps`` steps; SHA-1 of particle_q,
             particle_qd, particle_qd_grad (APIC), the solver's warm-start stress, every ``--every`` steps, plus the
             final arrays (npz). ``--compare other.npz`` checks hashes and np.array_equal of the final arrays.
    profile  WP_METAL_PROFILE=1: summed GPU time per kernel name over ``--steps`` steps after ``--warmup``.
    time     wall ms per step (synchronized) and pipelined, after ``--warmup`` steps (scale_pour's start time 0.6 s).

    PYTHONPATH=<warp>:<newton>:. .venv-newton152/bin/python metalsim/physics/mpm/probes/pour_bitwise.py \
        record --worlds 64 --mode gpuexit --out runs/x.npz
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import sys
import time

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("what", choices=["record", "profile", "time"])
ap.add_argument("--worlds", type=int, default=64)
ap.add_argument("--cells-per-world", type=int, default=0)
ap.add_argument("--max-leaf", type=int, default=-1, help="SolverImplicitMPM.Config.max_leaf_node_count (-1: automatic)")
ap.add_argument("--max-lower", type=int, default=-1, help="max_lower_node_count (-1: automatic)")
ap.add_argument("--max-upper", type=int, default=-1, help="max_upper_node_count (-1: automatic)")
ap.add_argument("--mode", default="gpuexit", choices=["eager", "gpuexit", "capture"])
ap.add_argument("--steps", type=int, default=320)
ap.add_argument("--every", type=int, default=10)
ap.add_argument("--warmup", type=int, default=0, help="time/profile: steps before measuring (default: start time)")
ap.add_argument("--start-time", type=float, default=0.6)
ap.add_argument("--device", default="metal:0")
ap.add_argument("--out")
ap.add_argument("--compare")
ap.add_argument("--json")
ap.add_argument("--no-tstore", action="store_true",
                help="construct the solver without a fem.TemporaryStore (Isaac Lab passes TemporaryStore(); mpm_manager.py)")
ap.add_argument("--check-status", action="store_true", help="call check_sparse_grid_rebuild_status every step")
args = ap.parse_args()
for _k in ("out", "compare", "json"):  # relative paths: against $POUR_OUT if set (wrappers that change directory)
    _v = getattr(args, _k)
    if _v and not os.path.isabs(_v) and os.environ.get("POUR_OUT"):
        setattr(args, _k, os.path.join(os.environ["POUR_OUT"], _v))

HERE = os.path.dirname(os.path.abspath(__file__))
sys.argv = [os.path.join(HERE, "scale_pour.py"), "--part", "mpm", "--device", args.device, "--worlds", str(args.worlds)]
sys.path.insert(0, HERE)
import scale_pour as sp  # noqa: E402  (parses the argv above; wp.init)

wp = sp.wp
DEV = sp.DEV
IS_METAL = sp.IS_METAL


def make_solver():
    from newton.solvers import SolverImplicitMPM

    from metalsim.physics.mpm import newton_compat

    newton_compat.install(accumulate="float32" if IS_METAL else "native", loop="cuda" if IS_METAL else "native",
                          log=False, chunk_graph=args.mode != "eager" and IS_METAL,
                          gpu_exit=args.mode != "eager" and IS_METAL)
    if IS_METAL and not _environment_grids_supported():
        sp.allow_metal_environment_grids()
    model, src, tgt, bpw, _ = sp.build(args.worlds, robot=False, media=True, source_mass=50.0)
    cfg = SolverImplicitMPM.Config()
    cfg.voxel_size = sp.VOXEL
    cfg.grid_type = "sparse"
    cfg.grid_padding = 0
    per_world = args.cells_per_world or ((735 + sp.CELL_ALIGN - 1) // sp.CELL_ALIGN) * sp.CELL_ALIGN
    cfg.max_active_cell_count = per_world * args.worlds
    cfg.strain_basis = "P0"
    cfg.transfer_scheme = "apic"
    cfg.max_iterations = 24
    cfg.tolerance = 1.0e-4
    cfg.warmstart_mode = "auto"
    cfg.velocity_basis = "Q1"
    cfg.collider_basis = "pic27"
    cfg.collider_velocity_mode = "forward"
    cfg.solver = "jacobi"
    cfg.separate_worlds = True
    cfg.max_leaf_node_count = args.max_leaf
    cfg.max_lower_node_count = args.max_lower
    cfg.max_upper_node_count = args.max_upper
    import warp.fem as fem

    # Isaac Lab: NewtonMPMManager._create_solver(..., temporary_store=TemporaryStore()) (also for coupler entries)
    solver = SolverImplicitMPM(model, cfg, temporary_store=None if args.no_tstore else fem.TemporaryStore())
    return model, solver, src, bpw


def _environment_grids_supported():
    """True when this Warp builds environment Nanogrids on Metal without the probe-local relaxation."""
    import inspect

    import warp._src.fem.geometry.nanogrid as ng

    return "if not device.is_cuda:" not in inspect.getsource(ng._normalize_environment_voxels)


def main():
    with wp.ScopedDevice(DEV):
        model, solver, src, bpw = make_solver()
        state = model.state()
        rng = np.random.default_rng(1)
        phase = wp.array((rng.random(args.worlds) * 1.5 - args.start_time).astype(np.float32), dtype=float)
        body_idx = wp.array(np.arange(args.worlds, dtype=np.int32) * bpw + src, dtype=int)
        base = wp.array(np.asarray(model.body_q.numpy())[body_idx.numpy(), :3].astype(np.float32), dtype=wp.vec3)
        tdev = wp.zeros(1, dtype=float)

        @wp.kernel
        def advance(tdev: wp.array[float], dt: float):
            tdev[0] = tdev[0] + dt

        @wp.kernel
        def drive_dev(tdev: wp.array[float], phase: wp.array[float], body: wp.array[int], base: wp.array[wp.vec3],
                      body_q: wp.array[wp.transform], body_qd: wp.array[wp.spatial_vector]):
            w = wp.tid()
            tt = wp.max(tdev[0] + phase[w], 0.0)
            s1 = wp.clamp(tt / 0.4, 0.0, 1.0)
            s2 = wp.clamp((tt - 0.4) / 0.4, 0.0, 1.0)
            s3 = wp.clamp((tt - 0.8) / 0.8, 0.0, 1.0)
            v1 = wp.where(tt < 0.4, 0.12 / 0.4, 0.0)
            v2 = wp.where(tt >= 0.4 and tt < 0.8, -0.12 / 0.4, 0.0)
            om = wp.where(tt >= 0.8 and tt < 1.6, 2.443461 / 0.8, 0.0)
            p = base[w] + wp.vec3(0.0, -0.12 * s2, 0.12 * s1)
            q = wp.quat_from_axis_angle(wp.vec3(-1.0, 0.0, 0.0), 2.443461 * s3)
            b = body[w]
            body_q[b] = wp.transform(p, q)
            body_qd[b] = wp.spatial_vector(wp.vec3(0.0, v2, v1), wp.vec3(-om, 0.0, 0.0))

        def body():
            wp.launch(drive_dev, dim=args.worlds, inputs=[tdev, phase, body_idx, base, state.body_q, state.body_qd])
            solver.step(state, state, None, None, sp.MPM_DT)
            wp.launch(advance, dim=1, inputs=[tdev, sp.MPM_DT])

        graph = {"g": None}
        res = {"worlds": args.worlds, "mode": args.mode, "cells_per_world": args.cells_per_world,
               "temporary_store": not args.no_tstore,
               "warp": wp.__file__, "newton": sp.newton.__file__}

        def step():
            if args.mode == "capture":
                if graph["g"] is None:
                    body()  # first step eager (scratchpad, module loads)
                    with wp.ScopedCapture(device=DEV, force_module_load=False) as cap:
                        body()  # recorded only
                    graph["g"] = cap.graph
                    wp.capture_launch(graph["g"])
                    return 2
                wp.capture_launch(graph["g"])
                return 1
            body()
            if args.check_status:
                solver.check_sparse_grid_rebuild_status()
            return 1

        def arrays():
            out = {"q": state.particle_q, "qd": state.particle_qd}
            for name in ("particle_qd_grad",):
                a = getattr(state, name, None)
                if a is not None:
                    out[name] = a
            mpm = getattr(state, "mpm", None)
            if mpm is not None:
                for name in dir(mpm):
                    a = getattr(mpm, name, None)
                    if isinstance(a, wp.array) and a.device == DEV:
                        out["mpm." + name] = a
            return {k: v.numpy().copy() for k, v in out.items()}

        if args.what == "record":
            hashes = []
            n = 0
            t0 = time.perf_counter()
            while n < args.steps:
                k = step()
                n += k
                if n % args.every == 0 or n >= args.steps or k > 1:
                    arrs = arrays()
                    h = hashlib.sha1()
                    for name in sorted(arrs):
                        h.update(name.encode())
                        h.update(np.ascontiguousarray(arrs[name]).tobytes())
                    hashes.append((n, h.hexdigest()))
            wp.synchronize_device()
            res["record_s"] = time.perf_counter() - t0
            final = arrays()
            res["hashes"] = hashes
            res["finite"] = bool(np.isfinite(final["q"]).all())
            with contextlib.suppress(Exception):
                solver.check_sparse_grid_rebuild_status()
                res["grid_status"] = "ok"
            if args.out:
                np.savez_compressed(args.out, hashes=np.array([h for _, h in hashes]),
                                    steps=np.array([s for s, _ in hashes]), **final)
            if args.compare:
                ref = np.load(args.compare)
                ref_h = dict(zip(ref["steps"].tolist(), ref["hashes"].tolist()))
                common = [(st_, h) for st_, h in hashes if st_ in ref_h]
                cmp = {"hash_steps_equal": int(sum(ref_h[st_] == h for st_, h in common)),
                       "hash_steps": len(common), "arrays": {}}
                for name, a in final.items():
                    if name in ref.files:
                        b = ref[name]
                        cmp["arrays"][name] = bool(a.shape == b.shape and np.array_equal(a.view(np.uint8),
                                                                                         b.view(np.uint8)))
                    else:
                        cmp["arrays"][name] = "missing in reference"
                cmp["bitwise_equal"] = (cmp["hash_steps_equal"] == cmp["hash_steps"] == len(ref["hashes"])
                                        and all(v is True for v in cmp["arrays"].values()))
                res["compare"] = cmp
        else:
            nwarm = args.warmup or int(round(args.start_time / sp.MPM_DT))
            n = 0
            while n < nwarm:
                n += step()
            wp.synchronize_device()
            if args.what == "profile":
                from warp._src.context import runtime

                core = runtime.core
                core.wp_metal_profile_report.restype = __import__("ctypes").c_char_p
                core.wp_metal_profile_report()
                for _ in range(args.steps):
                    step()
                wp.synchronize_device()
                rows = []
                for line in core.wp_metal_profile_report().decode().splitlines():
                    import re

                    m = re.match(r"\s*([0-9.]+) ms\s+(\d+)\s+(.*)$", line)
                    if m:
                        rows.append((float(m.group(1)) / args.steps, int(m.group(2)) / args.steps, m.group(3)))
                rows.sort(key=lambda r: -r[0])
                res["kernel_total_ms"] = sum(r[0] for r in rows)
                res["kernels"] = rows
                for r in rows[:45]:
                    print(f"{r[0]:9.3f} ms {r[1]:7.1f}  {r[2][:120]}")
            else:
                ts = []
                for _ in range(args.steps):
                    wp.synchronize_device()
                    t0 = time.perf_counter()
                    step()
                    wp.synchronize_device()
                    ts.append((time.perf_counter() - t0) * 1e3)
                t0 = time.perf_counter()
                for _ in range(args.steps):
                    step()
                wp.synchronize_device()
                res["pipelined_ms"] = (time.perf_counter() - t0) * 1e3 / args.steps
                res["step_ms_median"] = float(np.median(ts))
                res["step_ms_min"] = float(np.min(ts))
                res["step_ms"] = ts
        try:
            sc = solver._scratchpad
            res["velocity_nodes"] = int(sc.velocity_node_count)
            res["collider_nodes"] = int(sc.collider_node_count)
        except Exception as e:  # noqa: BLE001
            res["grid_info_error"] = str(e)[:200]
    print(json.dumps({k: v for k, v in res.items() if k not in ("kernels", "hashes", "step_ms")}, indent=1,
                     default=str))
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(res, fh, indent=1, default=str)


if __name__ == "__main__":
    main()

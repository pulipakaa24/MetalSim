"""Scaled benchmark: Isaac Lab 3.0-EA's Franka-Pour MPM task (IsaacContrib-Franka-Pour) mirrored with plain Newton at
many worlds, on any Warp device (``--device metal:0`` or ``--device cuda:0``).

What is mirrored (upstream/IsaacLab3/source/isaaclab_tasks/isaaclab_tasks/contrib/franka_pour/):
  * media (media.py, pour_env_cfg.py:97-157): source cup 42 x 42 x 110 mm cavity, fill 0.70, 5 mm lattice
    (voxel 0.015, 3 particles per cell, cell_center placement, jitter 0.3 x spacing) -> 7 x 7 x 15 = 735 particles
    per world; density 1500, friction 0.7, yield_pressure 1e12, Isaac Lab material defaults otherwise.
  * MPM solver (pour_env_cfg.py:633-648): voxel 0.015, sparse rebuildable grid (max_active_cell_count = 1024 per
    world, :276-292), grid_padding 0, strain P0, APIC, Jacobi, max_iterations 24, tolerance 1e-4, warmstart auto,
    velocity Q1, collider pic27, collider velocity forward, separate_worlds; dt = (1/120) / 2 outer / 2 MPM = 1/480 s.
  * colliders per world: hollow source cup (5 boxes, proxy body seen by MPM at 50 g x mass_scale 1000 = 50 kg),
    hollow target cup (kinematic), spill floor (static).
  * rigid side (pour_env_cfg.py:613-629): Franka (MuJoCo Menagerie panda.xml, whose collision meshes are the task's
    link*_c proxies), table, both cups; SolverMuJoCo(use_mujoco_contacts=False, integrator="implicitfast",
    njmax=2560, nconmax=400) fed by Newton's CollisionPipeline; dt = (1/120) / 2 / 3 = 1/720 s.

Per policy step Isaac runs (decimation 4) x (num_substeps 2) = 8 coupled steps, i.e. 16 MPM steps, 24 MuJoCo steps,
and 4 collision-pipeline calls. The source cup is driven kinematically in ``--part mpm`` (lift, carry, tilt to
140 deg with a per-world phase offset, so worlds are in different flow states at any time, as in RL).

Parts (each run is one process, one world count; a driver loops):
  mpm      SolverImplicitMPM alone. --grid sparse (Isaac's layout; on Metal this needs a probe-local relaxation of
           fem.Nanogrid's CUDA-only environment-grid check, applied automatically and recorded in the JSON) |
           fixed (dense, worlds co-located) | shared (one sparse grid, separate_worlds=False).
           --cells-per-world overrides Isaac's capacity rule (particles rounded up to 512 = 1024 cells per world). --mode eager|chunk|gpuexit|capture (chunk/gpuexit: metalsim's Metal loop
           options; capture: whole step captured, solver loop GPU-gated on Metal, conditional graph on CUDA).
           --sections adds Newton's own section timers (synchronizing).
           --det on sets wp.config.deterministic = RUN_TO_RUN for every module (off = unchanged default).
           Standalone (no metalsim package): newton_compat.py next to this file; PANDA_XML env var for panda.xml.
  rigid    Franka + table + cups, SolverMuJoCo + CollisionPipeline. --det off|on (wp.DeterministicMode.RUN_TO_RUN
           for MuJoCo Warp's modules), --pipeline-det (CollisionPipeline(deterministic=True)), --mode eager|capture.
  ordered  micro-benchmark of an atomic-add kernel, normal vs deterministic module, vs launch size.
  bodyforce  MPM collider-impulse -> body-wrench reduction at N worlds: float atomics (Isaac's coupler kernel),
           the chunked fixed-order sum of example_mpm_twoway_coupling (Newton fork 53a903d7), and a per-world
           segmented fixed-order sum (proposed replacement).

    PYTHONPATH=. .venv-newton152/bin/python \
        metalsim/physics/mpm/probes/scale_pour.py --part mpm --worlds 64 --device metal:0 --json out.json
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import re
import resource
import sys
import time
from collections import Counter, defaultdict

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--part", choices=["mpm", "rigid", "ordered", "bodyforce"], required=True)
ap.add_argument("--worlds", type=int, default=1)
ap.add_argument("--device", default="metal:0")
ap.add_argument("--steps", type=int, default=20)
ap.add_argument("--warmup", type=int, default=6)
ap.add_argument("--mode", default="eager", choices=["eager", "chunk", "gpuexit", "capture"])
ap.add_argument("--sections", action="store_true", help="mpm: Newton section timers (extra synchronized steps)")
ap.add_argument("--count", action="store_true", help="count host readbacks / syncs / launches in one eager step")
ap.add_argument("--det", default="off", choices=["off", "on"])
ap.add_argument("--pipeline-det", action="store_true")
ap.add_argument("--max-triangle-pairs", type=int, default=None,
                help="rigid: CollisionPipeline max_triangle_pairs (Newton default 1e6 overflows from ~512 worlds here)")
ap.add_argument("--repeat-check", action="store_true",
                help="run the trajectory twice and compare bitwise (rigid --mode capture: also as a replayed graph)")
ap.add_argument("--accumulate", default=None, help="float32 (Metal) | native (CUDA default float64)")
ap.add_argument("--phase-spread", type=float, default=1.5, help="mpm: per-world motion phase spread [s]")
ap.add_argument("--start-time", type=float, default=0.6, help="mpm: trajectory time at which timing starts [s]")
ap.add_argument("--grid", default="sparse", choices=["sparse", "fixed", "shared"],
                help="mpm: sparse = Isaac (separate_worlds, rebuildable sparse); fixed = separate_worlds, dense fixed "
                     "grid, worlds co-located; shared = one rebuildable sparse grid, separate_worlds=False")
ap.add_argument("--fixed-padding", type=int, default=12, help="mpm --grid fixed: padding voxels")
ap.add_argument("--cells-per-world", type=int, default=0,
                help="mpm: active-cell capacity per world (0 = Isaac's rule: particles rounded up to 512 = 1024)")
ap.add_argument("--no-tstore", action="store_true",
                help="mpm: no fem.TemporaryStore (Isaac Lab passes TemporaryStore(): NewtonMPMManager._create_solver, "
                     "also for coupler entries; without it every step reallocates its capacity-sized temporaries)")
ap.add_argument("--json")
ap.add_argument("--kprof", default=None, choices=["collide", "substep", "tick", "captured"],
                help="rigid: per-kernel profile of K ticks of this phase (Metal: run with WP_METAL_PROFILE=1; CUDA: "
                     "brackets the phase with cuProfilerStart/Stop for nsys --capture-range=cudaProfilerApi), then exit")
ap.add_argument("--kprof-ticks", type=int, default=4)
ap.add_argument("--dump", default=None, help="rigid --repeat-check: save final body_q / contact counts of each run (npz)")
args = ap.parse_args()

import warp as wp  # noqa: E402

wp.config.quiet = True
wp.init()
DEV = wp.get_device(args.device)
IS_METAL = bool(getattr(DEV, "is_metal", False))

import newton  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
PANDA_XML = os.environ.get("PANDA_XML") or os.path.join(REPO, "upstream/mujoco_menagerie/franka_emika_panda/panda.xml")

# ----------------------------------------------------------------------------------------------- task constants
VOXEL = 0.015
PPC = 3.0
FILL_RES = 0.005
FILL_LEVEL = 0.70
COLLIDER_MARGIN = 0.004
SOURCE_POS = (0.5, 0.0, 0.0)
TARGET_POS = (0.5, -0.18, 0.0)
CELL_ALIGN = 512
MPM_DT = (1.0 / 120.0) / 2.0 / 2.0
RIGID_DT = (1.0 / 120.0) / 2.0 / 3.0
ENV_SPACING = 2.5
MATERIAL = {"young_modulus": 1.0e15, "poisson_ratio": 0.3, "viscosity": 0.0, "friction": 0.7, "damping": 0.0,
            "yield_pressure": 1.0e12, "tensile_yield_ratio": 0.0, "yield_stress": 0.0, "hardening": 0.0,
            "dilatancy": 0.0}
DENSITY = 1500.0


class Cup:
    def __init__(self, w, d, depth, wall, bottom):
        self.w, self.d, self.depth, self.wall, self.bottom = w, d, depth, wall, bottom

    @property
    def rim(self):
        return self.bottom + self.depth

    def boxes(self):
        ox, oy, hh = 0.5 * self.w + self.wall, 0.5 * self.d + self.wall, 0.5 * self.rim
        hw = 0.5 * self.wall
        return (((0.0, 0.0, 0.5 * self.bottom), (ox, oy, 0.5 * self.bottom)),
                ((0.5 * self.w + hw, 0.0, hh), (hw, oy, hh)), ((-0.5 * self.w - hw, 0.0, hh), (hw, oy, hh)),
                ((0.0, 0.5 * self.d + hw, hh), (0.5 * self.w, hw, hh)), ((0.0, -0.5 * self.d - hw, hh), (0.5 * self.w, hw, hh)))


SOURCE = Cup(0.042, 0.042, 0.110, 0.007, 0.009)
TARGET = Cup(0.140, 0.140, 0.065, 0.009, 0.009)


def media_bounds():
    """franka_pour/media.py ``_media_grid_bounds`` (fill 0.70 -> 7 x 7 x 15 lattice)."""
    spacing, clearance = FILL_RES, max(FILL_RES, COLLIDER_MARGIN)

    def axis(size):
        usable = size - 2.0 * clearance
        intervals = max(0, math.floor(usable / spacing + 1.0e-9))
        count = intervals + 1
        first = -0.5 * size + clearance + 0.5 * (usable - intervals * spacing)
        lower = first - 0.5 * spacing
        return lower, float(np.nextafter(np.float32(lower + count * spacing), np.float32(lower)))

    lx, ux = axis(SOURCE.w)
    ly, uy = axis(SOURCE.d)
    max_z = max(1, math.floor((SOURCE.depth - 2.0 * clearance) / spacing) + 1)
    cz = max(1, min(max_z, math.floor(FILL_LEVEL * max_z + 0.5)))
    lz = SOURCE.bottom + clearance - 0.5 * spacing
    uz = float(np.nextafter(np.float32(lz + cz * spacing), np.float32(lz)))
    return (lx, ly, lz), (ux, uy, uz)


def media_particles(seed=0):
    """Isaac Lab ``_emit_grid`` cell_center lattice in the cup frame, float32 corners, uniform jitter."""
    lo, hi = media_bounds()
    lower, upper = np.asarray(lo, np.float32), np.asarray(hi, np.float32)
    ext = upper - lower
    res = np.maximum(np.ceil(PPC * ext / VOXEL), 1).astype(np.int32)
    cell = ext / res
    ii, jj, kk = np.meshgrid(np.arange(res[0]), np.arange(res[1]), np.arange(res[2]), indexing="ij")
    pts = lower + 0.5 * cell + np.stack([ii, jj, kk], -1).reshape(-1, 3) * cell
    jitter = 0.3 * (VOXEL / PPC)
    pts = pts + (np.random.default_rng(seed).random(pts.shape) - 0.5) * jitter
    vol = float(np.prod(cell))
    return pts.astype(np.float64), vol, 0.5 * float(np.cbrt(vol)), res


def add_cup(builder, body, cup, cfg):
    for off, half in cup.boxes():
        builder.add_shape_box(body, xform=wp.transform(wp.vec3(*off), wp.quat_identity()), hx=half[0], hy=half[1],
                              hz=half[2], cfg=cfg)


def world_builder(robot: bool, media: bool, source_mass: float):
    """One Franka-Pour world (local frame; table top at z = 0)."""
    from newton.solvers import SolverImplicitMPM

    b = newton.ModelBuilder()
    b.rigid_gap = 0.01  # Isaac Lab NewtonShapeCfg.gap default (newton_manager_cfg.py:97); Newton's own is 0.1
    if media:
        SolverImplicitMPM.register_custom_attributes(b)
    shape_only = newton.ModelBuilder.ShapeConfig(has_particle_collision=False)
    particle_only = newton.ModelBuilder.ShapeConfig(has_shape_collision=False)
    both = newton.ModelBuilder.ShapeConfig()
    both.mu = 2.0
    if robot:
        b.add_mjcf(PANDA_XML, xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
                   enable_self_collisions=True)
        # Isaac's home pose (pour_env_cfg.py _ARM_HOME) and open gripper, as joint state and position targets
        home = [-1.07505691, 0.76868522, 0.53213346, -2.93226814, 2.50670838, 1.40047050, 0.21146376, 0.04, 0.04]
        for i, q in enumerate(home):
            b.joint_q[i] = q
            b.joint_target_q[i] = q
        # table top at z = 0, clear of the robot base (Isaac: SeattleLabTable under the cups)
        b.add_shape_box(-1, xform=wp.transform(wp.vec3(0.6, 0.0, -0.025), wp.quat_identity()), hx=0.4, hy=0.4,
                        hz=0.025, cfg=shape_only)
    src = b.add_body(xform=wp.transform(wp.vec3(*SOURCE_POS), wp.quat_identity()), mass=source_mass,
                     label="SourceCup")
    add_cup(b, src, SOURCE, both)
    tgt = b.add_body(xform=wp.transform(wp.vec3(*TARGET_POS), wp.quat_identity()), mass=0.0, is_kinematic=True,
                     label="TargetCup")
    add_cup(b, tgt, TARGET, both)
    if media:
        # spill floor: particle-only static slab under the table top
        b.add_shape_box(-1, xform=wp.transform(wp.vec3(0.5, 0.0, -0.01), wp.quat_identity()), hx=1.0, hy=1.0,
                        hz=0.01, cfg=particle_only)
        pts, vol, radius, _ = media_particles()
        pts = pts + np.asarray(SOURCE_POS)
        n = pts.shape[0]
        attrs = {f"mpm:{k}": [float(v)] * n for k, v in MATERIAL.items()}
        b.add_particles(pos=[wp.vec3(*p) for p in pts], vel=[wp.vec3(0.0)] * n, mass=[vol * DENSITY] * n,
                        radius=[radius] * n, custom_attributes=attrs)
    return b, src, tgt


def build(worlds: int, robot: bool, media: bool, source_mass: float, colocate: bool = False):
    from newton.solvers import SolverImplicitMPM

    wb, src, tgt = world_builder(robot, media, source_mass)
    scene = newton.ModelBuilder()
    scene.rigid_gap = 0.01
    if media:
        SolverImplicitMPM.register_custom_attributes(scene)
    side = int(math.ceil(math.sqrt(worlds)))
    spacing = 0.0 if colocate else ENV_SPACING
    xforms = [wp.transform(wp.vec3(spacing * (i % side), spacing * (i // side), 0.0), wp.quat_identity())
              for i in range(worlds)]
    t0 = time.perf_counter()
    scene.replicate(wb, worlds, xforms=xforms)
    if robot:  # Isaac: /World/GroundPlane at z = -1.05
        scene.add_ground_plane(height=-1.05, cfg=newton.ModelBuilder.ShapeConfig(has_particle_collision=False))
    model = scene.finalize(device=DEV)
    build_s = time.perf_counter() - t0
    bodies_per_world = wb.body_count
    return model, src, tgt, bodies_per_world, build_s


# ----------------------------------------------------------------------------------------------- instrumentation
class Counters:
    """Counts host readbacks, synchronizations and kernel launches (and ordered Metal launches)."""

    def __init__(self):
        self.c = Counter()
        self.ordered = Counter()
        self.ordered_tasks = Counter()
        self.active = False
        self.sites = Counter()
        self._patch()

    @staticmethod
    def site():
        """Innermost caller frames outside this file and warp's array/context plumbing."""
        import traceback

        fr = [f for f in traceback.extract_stack()[:-2] if "scale_pour" not in f.filename
              and not f.filename.endswith(("warp/_src/types.py", "warp/_src/context.py"))]
        return " <- ".join(f"{os.path.basename(f.filename)}:{f.lineno}:{f.name}" for f in fr[-3:][::-1])

    def _patch(self):
        import warp._src.context as ctx

        me = self
        orig_numpy = wp.array.numpy

        def numpy(self_):
            if me.active and getattr(self_.device, "is_cpu", True) is False:
                me.c["array.numpy (device->host)"] += 1
                me.sites[me.site()] += 1
            return orig_numpy(self_)

        wp.array.numpy = numpy
        for name in ("synchronize", "synchronize_device", "synchronize_stream", "synchronize_event"):
            orig = getattr(wp, name)

            def mk(orig, name):
                def f(*a, **k):
                    if me.active:
                        me.c[f"wp.{name}"] += 1
                        me.sites[me.site()] += 1
                    return orig(*a, **k)
                return f

            setattr(wp, name, mk(orig, name))
        orig_launch = ctx.launch

        def launch(kernel, dim, *a, **k):
            if me.active:
                me.c["wp.launch"] += 1
            return orig_launch(kernel, dim, *a, **k)

        ctx.launch = launch
        wp.launch = launch
        if hasattr(ctx, "invoke_metal"):
            orig_inv = ctx.invoke_metal

            def invoke_metal(kernel, hooks, params, device, block_dim, adjoint=False, **kw):
                if me.active:
                    me.c["metal dispatches"] += 1
                    if not adjoint and getattr(hooks, "metal_forward_ordered", None):
                        me.ordered[kernel.key] += 1
                        try:
                            me.ordered_tasks[kernel.key] += int(params[0].size)
                        except Exception:  # noqa: BLE001
                            pass
                    if kw.get("det_args") is not None:
                        me.c["deterministic kernel dispatches"] += 1
                return orig_inv(kernel, hooks, params, device, block_dim, adjoint, **kw)

            ctx.invoke_metal = invoke_metal

    @contextlib.contextmanager
    def on(self):
        self.active = True
        try:
            yield
        finally:
            self.active = False

    def report(self, steps=1):
        out = {k: v / steps for k, v in self.c.items()}
        if self.sites:
            out["host sync sites"] = {k: v / steps for k, v in self.sites.most_common(12)}
        if self.ordered:
            out["ordered launches"] = sum(self.ordered.values()) / steps
            out["ordered kernels (top)"] = {k: [v / steps, self.ordered_tasks[k] / max(v, 1)]
                                            for k, v in self.ordered.most_common(15)}
        return out


def maxrss_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2**20 if sys.platform == "darwin" else r / 1024


def sync():
    wp.synchronize_device(DEV)


def timed_loop(fn, n):
    ts = []
    for _ in range(n):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        ts.append((time.perf_counter() - t0) * 1e3)
    return ts


def stats(ts):
    a = np.asarray(ts)
    return {"median": float(np.median(a)), "mean": float(a.mean()), "min": float(a.min()), "max": float(a.max())}


RESULT = {"argv": sys.argv[1:], "device": str(DEV), "device_name": getattr(DEV, "name", ""),
          "warp": wp.__version__, "newton": newton.__version__, "worlds": args.worlds}


def finish():
    RESULT["maxrss_mb"] = maxrss_mb()
    print(json.dumps(RESULT, indent=1, default=str))
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(RESULT, fh, indent=1, default=str)


# ----------------------------------------------------------------------------------------------- part: mpm
@wp.kernel
def drive_source_cup(t: float, phase: wp.array[float], body: wp.array[int], base: wp.array[wp.vec3],
                     body_q: wp.array[wp.transform], body_qd: wp.array[wp.spatial_vector]):
    """Scripted pour: lift 0.12 m (0.4 s), carry 0.12 m toward the target (0.4 s), tilt 140 deg about x (0.8 s)."""
    w = wp.tid()
    tt = wp.max(t + phase[w], 0.0)
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


def allow_metal_environment_grids():
    """Warp's fem.Nanogrid environment builder (``from_environment_voxels``, what separate_worlds + sparse uses)
    refuses non-CUDA devices with one device check, although it is built from Warp kernels plus
    ``Volume.allocate_by_voxels`` (GPU on Metal in the fork, 93ed7138). Probe-local patch: relax that check only."""
    import inspect

    import warp._src.fem.geometry.nanogrid as ng

    src = inspect.getsource(ng._normalize_environment_voxels)
    new = src.replace("if not device.is_cuda:", "if not (device.is_cuda or getattr(device, 'is_metal', False)):")
    assert new != src
    ns = {}
    exec(compile(new, ng.__file__, "exec"), ng.__dict__, ns)
    ng._normalize_environment_voxels = ns["_normalize_environment_voxels"]


def run_mpm():
    from newton.solvers import SolverImplicitMPM

    try:
        from metalsim.physics.mpm import newton_compat
    except ImportError:  # standalone copy next to newton_compat.py (on the L4 machine)
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import newton_compat

    if args.det == "on":  # mpm: every module in RUN_TO_RUN (Warp's global default mode); default off as before
        wp.config.deterministic = wp.DeterministicMode.RUN_TO_RUN
        RESULT["det"] = "on (wp.config.deterministic = RUN_TO_RUN, all modules)"
    accumulate = args.accumulate or ("native" if DEV.is_cuda else "float32")
    # CUDA: Newton's own conditional-graph loop is what Isaac runs (bitwise equal to loop="cuda", MPM.md section 4)
    loop = "native" if DEV.is_cuda else "cuda"
    newton_compat.install(accumulate=accumulate, loop=loop, log=False,
                          chunk_graph=args.mode in ("chunk", "gpuexit", "capture") and IS_METAL,
                          gpu_exit=args.mode in ("gpuexit", "capture") and IS_METAL)
    if IS_METAL and args.grid == "sparse":
        import inspect

        import warp._src.fem.geometry.nanogrid as _ng

        if "if not device.is_cuda:" in inspect.getsource(_ng._normalize_environment_voxels):
            allow_metal_environment_grids()  # stock Warp (the metalsim-mpm-scale branch builds them on Metal)
            RESULT["patched"] = "fem.Nanogrid environment builder allowed on Metal (probe-local)"
    with wp.ScopedDevice(DEV):
        model, src, tgt, bpw, build_s = build(args.worlds, robot=False, media=True, source_mass=50.0,
                                              colocate=args.grid == "fixed")
        cfg = SolverImplicitMPM.Config()
        cfg.voxel_size = VOXEL
        cfg.grid_type = "fixed" if args.grid == "fixed" else "sparse"
        cfg.grid_padding = args.fixed_padding if args.grid == "fixed" else 0
        per_world = args.cells_per_world or ((735 + CELL_ALIGN - 1) // CELL_ALIGN) * CELL_ALIGN
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
        cfg.separate_worlds = args.grid != "shared"
        t0 = time.perf_counter()
        import warp.fem as fem

        solver = SolverImplicitMPM(model, cfg, temporary_store=None if args.no_tstore else fem.TemporaryStore())
        RESULT["temporary_store"] = not args.no_tstore
        solver_s = time.perf_counter() - t0
        state = model.state()
        rng = np.random.default_rng(1)
        phase = wp.array((rng.random(args.worlds) * args.phase_spread - args.start_time).astype(np.float32),
                         dtype=float)
        body_idx = wp.array(np.arange(args.worlds, dtype=np.int32) * bpw + src, dtype=int)
        base = wp.array(np.asarray(model.body_q.numpy())[body_idx.numpy(), :3].astype(np.float32), dtype=wp.vec3)
        clock = {"t": 0.0}

        def step(t=None):
            wp.launch(drive_source_cup, dim=args.worlds,
                      inputs=[clock["t"] if t is None else t, phase, body_idx, base, state.body_q, state.body_qd])
            solver.step(state, state, None, None, MPM_DT)

        RESULT.update(part="mpm", mode=args.mode, grid=args.grid, accumulate=accumulate, loop=loop, particles=model.particle_count,
                      bodies=model.body_count, colliders=int(solver._mpm_model.collider.collider_mesh.shape[0]),
                      build_s=build_s, solver_init_s=solver_s, max_active_cell_count=cfg.max_active_cell_count)
        # advance the clock to the start time with the phase offsets (worlds in mixed states), untimed
        t_setup = time.perf_counter()
        nwarm = max(args.warmup, int(round(args.start_time / MPM_DT)) if args.start_time > 0 else args.warmup)
        for _ in range(nwarm):
            step()
            clock["t"] += MPM_DT
        sync()
        RESULT["warmup_steps"] = nwarm
        RESULT["warmup_s"] = time.perf_counter() - t_setup

        graph = None
        if args.mode == "capture":
            # the clock enters as a device scalar so one graph replays every step
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

            tdev.fill_(clock["t"])

            def body():
                wp.launch(drive_dev, dim=args.worlds, inputs=[tdev, phase, body_idx, base, state.body_q, state.body_qd])
                solver.step(state, state, None, None, MPM_DT)
                wp.launch(advance, dim=1, inputs=[tdev, MPM_DT])

            try:
                body()  # warm module loads for the device-clock kernels
                with wp.ScopedCapture(device=DEV, force_module_load=False) as cap:
                    body()
                graph = cap.graph
                RESULT["capture"] = "ok"
            except Exception as e:  # noqa: BLE001
                import traceback

                RESULT["capture"] = f"failed: {type(e).__name__}: {e}"[:400]
                RESULT["capture_traceback"] = [f"{os.path.basename(f.filename)}:{f.lineno}:{f.name}"
                                               for f in traceback.extract_tb(e.__traceback__)][-12:]
                graph = None
                with contextlib.suppress(Exception):
                    wp.capture_end(device=DEV)

        if args.count:
            cnt = Counters()
            with cnt.on():
                step()
            clock["t"] += MPM_DT
            sync()
            RESULT["per_step_counts_eager"] = cnt.report()

        def one():
            if graph is not None:
                wp.capture_launch(graph)
            else:
                step()
                clock["t"] += MPM_DT

        ts = timed_loop(one, args.steps)
        RESULT["step_ms"] = stats(ts)
        RESULT["step_ms_all"] = ts
        # pipelined throughput: no sync between steps
        sync()
        t0 = time.perf_counter()
        for _ in range(args.steps):
            one()
        sync()
        RESULT["step_ms_pipelined"] = (time.perf_counter() - t0) * 1e3 / args.steps
        if args.mode == "capture" and graph is not None and hasattr(solver, "check_sparse_grid_rebuild_status"):
            try:
                solver.check_sparse_grid_rebuild_status()
                RESULT["grid_status"] = "ok"
            except Exception as e:  # noqa: BLE001
                RESULT["grid_status"] = str(e)[:300]

        # iteration count of the last solves (one logged solve, outside the timed loop)
        newton_compat.install(accumulate=accumulate, loop="cuda", log=True,
                              chunk_graph=IS_METAL and args.mode != "eager", gpu_exit=False)
        for _ in range(4):
            step()
            clock["t"] += MPM_DT
        log = newton_compat.drain_log()
        RESULT["iterations_last4"] = log["iterations"].tolist()
        RESULT["res_l2_last4"] = log["res_l2"].tolist()
        newton_compat.install(accumulate=accumulate, loop=loop, log=False,
                              chunk_graph=args.mode in ("chunk", "gpuexit", "capture") and IS_METAL,
                              gpu_exit=args.mode in ("gpuexit", "capture") and IS_METAL)

        if args.sections:
            solver.enable_timers = True
            pat = re.compile(r"^(.*?) took ([0-9.]+) ms")
            tot = defaultdict(float)
            nsec = max(4, args.steps // 2)
            for _ in range(nsec):
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    step()
                clock["t"] += MPM_DT
                for line in buf.getvalue().splitlines():
                    m = pat.match(line.strip())
                    if m:
                        tot[m.group(1).strip()] += float(m.group(2))
            solver.enable_timers = False
            RESULT["sections_ms"] = {k: v / nsec for k, v in sorted(tot.items(), key=lambda kv: -kv[1])}

        # grid occupancy: cells actually touched by particles (host-side, after the timed steps)
        q = state.particle_q.numpy()
        pw = model.particle_world.numpy()
        cells = np.floor(q / VOXEL).astype(np.int64)
        RESULT["occupied_cells_per_world"] = len(np.unique(np.column_stack([pw, cells]), axis=0)) / args.worlds
        try:
            sp = solver._scratchpad
            RESULT["velocity_nodes"] = int(sp.velocity_node_count)
            RESULT["strain_nodes"] = int(sp.strain_node_count) if hasattr(sp, "strain_node_count") else None
            RESULT["collider_nodes"] = int(sp.collider_node_count)
        except Exception as e:  # noqa: BLE001
            RESULT["grid_info_error"] = str(e)[:200]

        if args.repeat_check:
            import gc

            solver = None  # free the timed solver first (2,048 worlds: three live solvers exceed the L4's 24 GB)
            gc.collect()
            RESULT["repeat_check"] = repeat_check_mpm(model, None, cfg)
    finish()


def repeat_check_mpm(model, solver_unused, cfg):
    """Fresh solver twice from the initial state, 40 steps from t=0 with the same phases; bitwise compare."""
    from newton.solvers import SolverImplicitMPM

    import gc

    import warp.fem as fem

    outs = []
    for _ in range(2):
        # with the TemporaryStore Isaac passes (bitwise equal to the store-less solver, MPM.md section 7b)
        s = SolverImplicitMPM(model, cfg, temporary_store=None if args.no_tstore else fem.TemporaryStore())
        st = model.state()
        rng = np.random.default_rng(1)
        phase = wp.array((rng.random(args.worlds) * args.phase_spread).astype(np.float32), dtype=float)
        bpw = model.body_count // args.worlds
        body_idx = wp.array(np.arange(args.worlds, dtype=np.int32) * bpw, dtype=int)
        base = wp.array(np.asarray(model.body_q.numpy())[body_idx.numpy(), :3].astype(np.float32), dtype=wp.vec3)
        t = 0.0
        for _ in range(40):
            wp.launch(drive_source_cup, dim=args.worlds, inputs=[t, phase, body_idx, base, st.body_q, st.body_qd])
            s.step(st, st, None, None, MPM_DT)
            t += MPM_DT
        outs.append(st.particle_q.numpy().copy())
        del s, st
        gc.collect()
    d = np.abs(outs[0] - outs[1])
    return {"bitwise_equal": bool(np.array_equal(outs[0], outs[1])), "max_abs_diff": float(d.max()),
            "worlds_differing": int(np.any(d.reshape(args.worlds, -1) > 0, axis=1).sum())}


# ----------------------------------------------------------------------------------------------- part: rigid
def _pipeline_kwargs():
    return {} if args.max_triangle_pairs is None else {"max_triangle_pairs": args.max_triangle_pairs}


def run_rigid():
    from newton.solvers import SolverMuJoCo

    det = wp.DeterministicMode.RUN_TO_RUN if args.det == "on" else wp.DeterministicMode.NOT_GUARANTEED
    with wp.ScopedDevice(DEV):
        model, src, tgt, bpw, build_s = build(args.worlds, robot=True, media=False, source_mass=0.05)
        t0 = time.perf_counter()
        solver = SolverMuJoCo(model, use_mujoco_contacts=False, integrator="implicitfast", njmax=2560, nconmax=400,
                              deterministic=det)
        solver_s = time.perf_counter() - t0
        pipeline = newton.CollisionPipeline(model, deterministic=args.pipeline_det, **_pipeline_kwargs())
        contacts = pipeline.contacts()
        s0, s1 = model.state(), model.state()
        control = model.control()
        newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
        RESULT.update(part="rigid", det=args.det, pipeline_det=args.pipeline_det, mode=args.mode,
                      max_triangle_pairs=args.max_triangle_pairs,
                      bodies=model.body_count, shapes=model.shape_count, joints=model.joint_count,
                      dofs=model.joint_dof_count, build_s=build_s, solver_init_s=solver_s,
                      rigid_contact_max=int(contacts.rigid_contact_max))
        st = {"a": s0, "b": s1}

        def collide():
            pipeline.collide(st["a"], contacts)

        def substep():
            solver.step(st["a"], st["b"], control, contacts, RIGID_DT)
            st["a"], st["b"] = st["b"], st["a"]

        def tick():  # one coupled step's rigid work: collide once, 3 substeps (even swaps keep buffers stable x2)
            collide()
            for _ in range(3):
                substep()

        for _ in range(args.warmup):
            tick()
        sync()
        if args.kprof:
            kprof(args.kprof, {"collide": collide, "substep": substep, "tick": tick})
            RESULT["rigid_contacts"] = int(contacts.rigid_contact_count.numpy()[0])
            finish()
            return
        if args.count:
            cnt = Counters()
            with cnt.on():
                tick()
            sync()
            RESULT["per_tick_counts_eager"] = cnt.report()
        RESULT["collide_ms"] = stats(timed_loop(collide, args.steps))
        RESULT["substep_ms"] = stats(timed_loop(substep, args.steps))
        RESULT["tick_ms"] = stats(timed_loop(tick, args.steps))
        if args.mode == "capture":
            try:
                tick()  # odd tick count would leave swapped buffers; capture two ticks to return to the same buffers
                with wp.ScopedCapture(device=DEV, force_module_load=False) as cap:
                    tick()
                    tick()
                g = cap.graph
                RESULT["capture"] = "ok"
                ts = timed_loop(lambda: wp.capture_launch(g), args.steps)
                RESULT["tick_ms_captured"] = stats([t / 2 for t in ts])
            except Exception as e:  # noqa: BLE001
                RESULT["capture"] = f"failed: {type(e).__name__}: {e}"[:400]
        nc = contacts.rigid_contact_count.numpy()
        RESULT["rigid_contacts"] = int(nc[0])
        RESULT["nefc_mean"] = float(np.mean(solver.mjw_data.nefc.numpy())) if hasattr(solver, "mjw_data") else None
        RESULT["finite"] = bool(np.isfinite(st["a"].body_q.numpy()).all())
        if args.repeat_check:
            RESULT["repeat_check"] = repeat_check_rigid(model, det)
            if args.mode == "capture":
                RESULT["repeat_check_captured"] = repeat_check_rigid(model, det, captured=True)
    finish()


def kprof(phase, fns):
    """Per-kernel GPU time of ``--kprof-ticks`` repetitions of one phase (see --kprof)."""
    import ctypes

    from warp._src.context import runtime

    K = args.kprof_ticks
    core = runtime.core
    fn = fns.get(phase)
    if phase == "captured":
        fns["tick"]()  # leave the buffers where a two-tick graph returns them
        with wp.ScopedCapture(device=DEV, force_module_load=False) as cap:
            fns["tick"]()
            fns["tick"]()
        g = cap.graph
        wp.capture_launch(g)
        sync()
        fn = lambda: wp.capture_launch(g)  # noqa: E731
        K = max(1, K // 2)
    if IS_METAL:
        core.wp_metal_counters.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_uint64), ctypes.c_int]
        core.wp_metal_counters.restype = ctypes.c_int

        def counters():
            out = (ctypes.c_uint64 * 4)()
            core.wp_metal_counters(DEV.metal_ordinal, out, 4)
            return np.array(list(out), dtype=np.int64)

        profiling = os.environ.get("WP_METAL_PROFILE") is not None

        def report():
            rows = []
            for line in core.wp_metal_profile_report().decode().splitlines():
                m = re.match(r"\s*([0-9.]+) ms\s+(\d+)\s+(.*)$", line)
                if m:
                    rows.append((m.group(3), float(m.group(1)), int(m.group(2))))
            return rows

        if profiling:
            report()
        c0 = counters()
        t0 = time.perf_counter()
        for _ in range(K):
            fn()
        sync()
        wall = (time.perf_counter() - t0) * 1e3 / K
        c1 = counters()
        RESULT["kprof"] = {"phase": phase, "reps": K, "wall_ms_per_rep": wall, "profiling": profiling,
                           "per_rep_counters": dict(zip(["host_waits", "command_buffers", "host_ops", "dispatches"],
                                                        ((c1 - c0) / K).tolist()))}
        if profiling:
            rows = report()
            RESULT["kprof"]["kernels"] = [{"kernel": k, "ms": ms / K, "launches": c / K} for k, ms, c in rows]
            RESULT["kprof"]["kernel_total_ms"] = sum(ms for _, ms, _ in rows) / K
            RESULT["kprof"]["launches"] = sum(c for _, _, c in rows) / K
            for r in RESULT["kprof"]["kernels"][:30]:
                print(f"{r['ms']:9.3f} ms {r['launches']:8.1f}  {r['kernel'][:110]}", file=sys.stderr)
    else:
        cuda = ctypes.CDLL("libcuda.so.1")
        sync()
        cuda.cuProfilerStart()
        t0 = time.perf_counter()
        for _ in range(K):
            fn()
        sync()
        wall = (time.perf_counter() - t0) * 1e3 / K
        cuda.cuProfilerStop()
        RESULT["kprof"] = {"phase": phase, "reps": K, "wall_ms_per_rep": wall}


def repeat_check_rigid(model, det, captured=False):
    """Run the same 60 ticks twice (fresh solver, pipeline and states each time) and compare body_q bitwise.

    ``captured``: 2 eager ticks (module loads), then a graph of 2 ticks replayed 29 times.
    """
    from newton.solvers import SolverMuJoCo

    outs, extra = [], []
    for _ in range(2):
        solver = SolverMuJoCo(model, use_mujoco_contacts=False, integrator="implicitfast", njmax=2560, nconmax=400,
                              deterministic=det)
        pipeline = newton.CollisionPipeline(model, deterministic=args.pipeline_det, **_pipeline_kwargs())
        contacts = pipeline.contacts()
        st = {"a": model.state(), "b": model.state()}
        control = model.control()
        newton.eval_fk(model, model.joint_q, model.joint_qd, st["a"])

        def tick():
            pipeline.collide(st["a"], contacts)
            for _ in range(3):
                solver.step(st["a"], st["b"], control, contacts, RIGID_DT)
                st["a"], st["b"] = st["b"], st["a"]

        if captured:
            for _ in range(2):
                tick()
            with wp.ScopedCapture(device=DEV, force_module_load=False) as cap:
                tick()
                tick()
            for _ in range(29):
                wp.capture_launch(cap.graph)
        else:
            for _ in range(60):
                tick()
        outs.append(st["a"].body_q.numpy().copy())
        extra.append({"body_qd": st["a"].body_qd.numpy().copy(), "joint_q": st["a"].joint_q.numpy().copy(),
                      "contact_count": contacts.rigid_contact_count.numpy().copy(),
                      "nefc": solver.mjw_data.nefc.numpy().copy() if hasattr(solver, "mjw_data") else np.zeros(1)})
    d = np.abs(outs[0] - outs[1])
    import hashlib

    res = {"bitwise_equal": bool(np.array_equal(outs[0], outs[1])), "max_abs_diff": float(d.max()),
           "worlds_differing": int(np.any(d.reshape(args.worlds, -1) > 0, axis=1).sum()),
           "finite": bool(np.isfinite(outs[0]).all()),
           "body_q_sha": [hashlib.sha256(o.tobytes()).hexdigest()[:16] for o in outs],
           "body_qd_sha": [hashlib.sha256(e["body_qd"].tobytes()).hexdigest()[:16] for e in extra],
           "contact_count": [int(e["contact_count"][0]) for e in extra],
           "nefc_sum": [int(e["nefc"].sum()) for e in extra]}
    if args.dump:
        tag = "cap" if captured else "eager"
        np.savez(f"{args.dump}_{tag}.npz", body_q0=outs[0], body_q1=outs[1],
                 **{f"{k}{i}": e[k] for i, e in enumerate(extra) for k in e})
    return res


# ----------------------------------------------------------------------------------------------- part: ordered
def run_ordered():
    def make(mode):
        @wp.kernel(module="unique", module_options={"deterministic": mode})
        def acc(values: wp.array[float], idx: wp.array[int], out: wp.array[float]):
            i = wp.tid()
            wp.atomic_add(out, idx[i], values[i])
        return acc

    k_off = make(wp.DeterministicMode.NOT_GUARANTEED)
    k_on = make(wp.DeterministicMode.RUN_TO_RUN)
    rows = []
    with wp.ScopedDevice(DEV):
        for n in (1 << 10, 1 << 13, 1 << 16, 1 << 19, 1 << 21):
            for outputs in (n, max(1, n // 64)):  # per-thread outputs (no contention) / 64 writers per output
                rng = np.random.default_rng(0)
                vals = wp.array(rng.random(n).astype(np.float32), dtype=float)
                idx = wp.array(rng.integers(0, outputs, n).astype(np.int32), dtype=int)
                out = wp.zeros(outputs, dtype=float)
                r = {"n": n, "outputs": outputs}
                for name, k in (("off", k_off), ("on", k_on)):
                    wp.launch(k, dim=n, inputs=[vals, idx, out])
                    sync()
                    ts = timed_loop(lambda: wp.launch(k, dim=n, inputs=[vals, idx, out]), 15)
                    r[name] = float(np.median(ts))
                rows.append(r)
                print(r, flush=True)
    RESULT.update(part="ordered", rows=rows)
    finish()


# ----------------------------------------------------------------------------------------------- part: bodyforce
@wp.kernel
def harvest_atomic(dt: float, cids: wp.array[int], imp: wp.array[wp.vec3], pos: wp.array[wp.vec3],
                   cbody: wp.array[int], body_q: wp.array[wp.transform], body_com: wp.array[wp.vec3],
                   out: wp.array[wp.spatial_vector]):
    """Isaac's coupler path (_harvest_mpm_proxy_wrenches_kernel, minus the proxy-mapping checks)."""
    i = wp.tid()
    c = cids[i]
    if c < 0:
        return
    b = cbody[c]
    if b < 0:
        return
    f = imp[i] / dt
    r = pos[i] - wp.transform_point(body_q[b], body_com[b])
    wp.atomic_add(out, b, wp.spatial_vector(f, wp.cross(r, f)))


@wp.kernel
def harvest_chunked(dt: float, chunk: int, cids: wp.array[int], imp: wp.array[wp.vec3], pos: wp.array[wp.vec3],
                    cbody: wp.array[int], body_q: wp.array[wp.transform], body_com: wp.array[wp.vec3],
                    partial: wp.array2d[wp.spatial_vector]):
    """example_mpm_twoway_coupling's compute_body_forces_chunked (Newton fork 53a903d7): every body scans every node."""
    b, k = wp.tid()
    begin = k * chunk
    end = wp.min(begin + chunk, cids.shape[0])
    com = wp.transform_point(body_q[b], body_com[b])
    f = wp.spatial_vector()
    for i in range(begin, end):
        c = cids[i]
        if c >= 0:
            if cbody[c] == b:
                fw = imp[i] / dt
                f += wp.spatial_vector(fw, wp.cross(pos[i] - com, fw))
    partial[b, k] = f


@wp.kernel
def sum_chunks(partial: wp.array2d[wp.spatial_vector], out: wp.array[wp.spatial_vector]):
    b = wp.tid()
    f = wp.spatial_vector()
    for k in range(partial.shape[1]):
        f += partial[b, k]
    out[b] = f


@wp.kernel
def harvest_segmented(dt: float, bodies_per_world: int, node_offsets: wp.array[int], cids: wp.array[int],
                      imp: wp.array[wp.vec3], pos: wp.array[wp.vec3], cbody: wp.array[int],
                      body_q: wp.array[wp.transform], body_com: wp.array[wp.vec3], out: wp.array[wp.spatial_vector]):
    """Proposed: collider nodes are environment-first (scratch.collider_environment_offsets), so one thread per
    (world, local body) walks only its world's node range in node order: O(nodes) total, fixed order, no atomics."""
    w, lb = wp.tid()
    b = w * bodies_per_world + lb
    com = wp.transform_point(body_q[b], body_com[b])
    f = wp.spatial_vector()
    for i in range(node_offsets[w], node_offsets[w + 1]):
        c = cids[i]
        if c >= 0:
            if cbody[c] == b:
                fw = imp[i] / dt
                f += wp.spatial_vector(fw, wp.cross(pos[i] - com, fw))
    out[b] = f


def run_bodyforce():
    """Synthetic but task-sized: per world 2 proxy bodies, ``nodes_per_world`` collider nodes (environment-first),
    ~30 % of them touching a body (the source cup's wetted walls)."""
    W = args.worlds
    nodes_per_world = 1600  # measured collider-node count per world in --part mpm is reported there; order of 10^3
    bpw = 2
    rng = np.random.default_rng(0)
    n = W * nodes_per_world
    cid = np.full(n, -1, np.int32)
    touch = rng.random(n) < 0.3
    world_of = np.repeat(np.arange(W), nodes_per_world)
    cid[touch] = (world_of[touch] * bpw + rng.integers(0, bpw, touch.sum())).astype(np.int32)  # collider == body
    with wp.ScopedDevice(DEV):
        cids = wp.array(cid, dtype=int)
        imp = wp.array(rng.standard_normal((n, 3)).astype(np.float32), dtype=wp.vec3)
        pos = wp.array(rng.standard_normal((n, 3)).astype(np.float32), dtype=wp.vec3)
        cbody = wp.array(np.arange(W * bpw, dtype=np.int32), dtype=int)
        body_q = wp.array(np.tile(np.array([0, 0, 0, 0, 0, 0, 1], np.float32), (W * bpw, 1)), dtype=wp.transform)
        body_com = wp.zeros(W * bpw, dtype=wp.vec3)
        out = wp.zeros(W * bpw, dtype=wp.spatial_vector)
        offs = wp.array(np.arange(W + 1, dtype=np.int32) * nodes_per_world, dtype=int)
        chunk = 256
        nchunk = (n + chunk - 1) // chunk
        res = {"nodes": n, "bodies": W * bpw}

        def atomic():
            out.zero_()
            wp.launch(harvest_atomic, dim=n, inputs=[1e-3, cids, imp, pos, cbody, body_q, body_com, out])

        def segmented():
            wp.launch(harvest_segmented, dim=(W, bpw), inputs=[1e-3, bpw, offs, cids, imp, pos, cbody, body_q,
                                                               body_com, out])

        for name, fn in (("atomic", atomic), ("segmented", segmented)):
            fn()
            sync()
            res[name + "_ms"] = float(np.median(timed_loop(fn, 10)))
        segmented()
        ref = out.numpy().copy()
        atomic()
        res["atomic_vs_segmented_maxdiff"] = float(np.abs(out.numpy() - ref).max())
        # chunked: O(bodies x nodes); guard the size (it is the thing being measured)
        work = W * bpw * n
        res["chunked_work_items"] = work
        if work <= 4e10:
            partial = wp.empty((W * bpw, nchunk), dtype=wp.spatial_vector)

            def chunked():
                wp.launch(harvest_chunked, dim=(W * bpw, nchunk), inputs=[1e-3, chunk, cids, imp, pos, cbody, body_q,
                                                                          body_com, partial])
                wp.launch(sum_chunks, dim=W * bpw, inputs=[partial, out])

            chunked()
            sync()
            res["chunked_ms"] = float(np.median(timed_loop(chunked, 3)))
            res["chunked_vs_segmented_maxdiff"] = float(np.abs(out.numpy() - ref).max())
        else:
            res["chunked_ms"] = None
    RESULT.update(part="bodyforce", **res)
    finish()


if __name__ == "__main__":
    {"mpm": run_mpm, "rigid": run_rigid, "ordered": run_ordered, "bodyforce": run_bodyforce}[args.part]()

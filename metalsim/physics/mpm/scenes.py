"""MPM parity scenes, written against Newton 1.5.2's public API only (numpy + warp + newton).

The same file runs on the L4 (Isaac Lab 3.0-EA's Newton 1.5.2, CUDA) and on the Mac (Newton 1.5.2 on the Warp
Metal fork), so both sides build bit-identical initial states. Scene catalogue and the reasons behind each choice:
docs/research/mpm_2026-10-09.md ("Record seven reference sets").

Two kinds of scene:
- ``column``: quasi-2D granular column collapse (Lajeunesse 2005 / Lube 2005 geometry). Particles are placed exactly
  as Isaac Lab 3.0-EA's ``MPMGridCfg(particle_placement="cell_center", jitter=0)`` places them
  (isaaclab_newton/sim/spawners/mpm/mpm.py, ``_emit_grid``), with Isaac Lab's ``MPMParticleMaterialCfg`` defaults.
- ``plate``: resistive-force-theory plate intrusion / drag through a granular bed (force on the plate).
- ``example:<name>``: one of Newton's own ``newton/examples/mpm`` scenes, unmodified, driven without a viewer.
"""
from __future__ import annotations

import importlib
import types
from dataclasses import asdict, dataclass, field

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverImplicitMPM

# Isaac Lab 3.0-EA MPMParticleMaterialCfg defaults (mpm_cfg.py). Newton's own registry defaults differ
# (friction 0.5, yield_pressure 1e15); Isaac users get these.
ISAACLAB_MATERIAL = {
    "young_modulus": 1.0e15,
    "poisson_ratio": 0.3,
    "viscosity": 0.0,
    "friction": 0.68,
    "damping": 0.0,
    "yield_pressure": 1.0e12,
    "tensile_yield_ratio": 0.0,
    "yield_stress": 0.0,
    "hardening": 0.0,
    "dilatancy": 0.0,
}


@dataclass
class ColumnSpec:
    """Quasi-2D column collapse: a column of length ``length`` (x) against a back wall at x = 0, height
    ``aspect * length`` (z), between frictionless side walls ``width`` apart (y), on a frictional floor.

    Runout is measured along +x from the back wall; the plane-strain setting is what the 2D laws (Lajeunesse 2005,
    Lube 2005) describe, and frictionless side walls keep it plane-strain.
    """

    aspect: float = 2.0
    length: float = 0.2
    width: float = 0.1
    voxel_size: float = 0.05
    particles_per_cell: float = 3.0
    density: float = 1000.0
    jitter: float = 0.0  # fraction of a particle cell; 0 = Isaac Lab cell_center lattice
    seed: int = 0  # only used when jitter > 0
    floor_mu: float = 0.5
    back_wall_mu: float = 0.5
    side_wall_mu: float = 0.0
    material: dict = field(default_factory=lambda: dict(ISAACLAB_MATERIAL))
    # solver
    grid_type: str = "dense"
    solver: str = "gs"
    max_iterations: int = 250
    tolerance: float = 1.0e-4
    fps: float = 60.0
    seconds: float = 3.0
    project_outside: bool = True


def _cell_center_lattice(lower, upper, voxel_size, particles_per_cell):
    """Isaac Lab ``_emit_grid`` with ``particle_placement="cell_center"``: positions, per-particle mass factor.

    Corners are float32 as in Isaac Lab (``_as_finite_vector3``), so ``ceil(3 * 0.2 / 0.05)`` is 12; in float64 it
    would round to 13 (verified against Isaac Lab's own emitter on the L4, R2)."""
    lower = np.asarray(lower, dtype=np.float32)
    upper = np.asarray(upper, dtype=np.float32)
    extent = upper - lower
    resolution = np.maximum(np.ceil(particles_per_cell * extent / voxel_size), 1).astype(np.int32)
    cell_size = extent / resolution
    first = lower + 0.5 * cell_size
    return first, resolution, cell_size


def build_column(spec: ColumnSpec):
    """Return ``(model, solver_config, info)`` for a column-collapse scene."""
    builder = newton.ModelBuilder()
    SolverImplicitMPM.register_custom_attributes(builder)

    h = spec.aspect * spec.length
    first, res, cell = _cell_center_lattice((0.0, 0.0, 0.0), (spec.length, spec.width, h), spec.voxel_size,
                                            spec.particles_per_cell)
    cell_volume = float(np.prod(cell))
    radius = 0.5 * float(np.cbrt(cell_volume))
    # jitter as Newton's add_particle_grid applies it (uniform in [-jitter/2, jitter/2]), seeded per scene
    jitter = spec.jitter * float(np.min(cell))
    if jitter > 0.0:
        ii, jj, kk = np.meshgrid(np.arange(res[0]), np.arange(res[1]), np.arange(res[2]), indexing="ij")
        pts = first + np.stack([ii, jj, kk], -1).reshape(-1, 3) * cell
        rng = np.random.default_rng(spec.seed)
        pts = pts + (rng.random(pts.shape) - 0.5) * jitter
        n = pts.shape[0]
        attrs = {f"mpm:{k}": [float(v)] * n for k, v in spec.material.items()}
        builder.add_particles(pos=[wp.vec3(*p) for p in pts], vel=[wp.vec3(0.0)] * n,
                              mass=[cell_volume * spec.density] * n, radius=[radius] * n,
                              custom_attributes=attrs)
    else:
        builder.add_particle_grid(
            pos=wp.vec3(*first.tolist()), rot=wp.quat_identity(), vel=wp.vec3(0.0),
            dim_x=int(res[0]), dim_y=int(res[1]), dim_z=int(res[2]),
            cell_x=float(cell[0]), cell_y=float(cell[1]), cell_z=float(cell[2]),
            mass=cell_volume * spec.density, jitter=0.0, radius_mean=radius,
            custom_attributes={f"mpm:{k}": float(v) for k, v in spec.material.items()},
        )

    # static walls: back wall (x < 0) and the two side walls (y < 0, y > width), tall enough for the column
    wall_h = 0.5 * h + 0.5
    reach = 0.5 * spec.length * (2.0 + 4.0 * spec.aspect) + 1.0  # well past any runout
    t = 0.25
    back = newton.ModelBuilder.ShapeConfig(mu=spec.back_wall_mu, density=0.0)
    side = newton.ModelBuilder.ShapeConfig(mu=spec.side_wall_mu, density=0.0)
    builder.add_shape_box(body=-1, cfg=back, xform=wp.transform(wp.vec3(-t, 0.5 * spec.width, wall_h),
                                                                wp.quat_identity()),
                          hx=t, hy=0.5 * spec.width + 2 * t, hz=wall_h)
    for y in (-t, spec.width + t):
        builder.add_shape_box(body=-1, cfg=side, xform=wp.transform(wp.vec3(0.5 * reach, y, wall_h),
                                                                    wp.quat_identity()),
                              hx=0.5 * reach + t, hy=t, hz=wall_h)
    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=spec.floor_mu))

    model = builder.finalize()
    cfg = SolverImplicitMPM.Config()
    cfg.voxel_size = spec.voxel_size
    cfg.grid_type = spec.grid_type
    cfg.solver = spec.solver
    cfg.max_iterations = spec.max_iterations
    cfg.tolerance = spec.tolerance
    info = {"scene": "column", "spec": asdict(spec), "initial_height": h, "initial_length": spec.length,
            "particle_count": int(model.particle_count)}
    return model, cfg, info


class ColumnScene:
    """Uniform driver: ``step()`` advances one frame; ``state`` is the current state."""

    def __init__(self, spec: ColumnSpec):
        self.spec = spec
        self.model, self.config, self.info = build_column(spec)
        self.solver = SolverImplicitMPM(self.model, config=self.config)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.frame_dt = 1.0 / spec.fps
        self.frames = int(round(spec.seconds * spec.fps))

    @property
    def state(self):
        return self.state_0

    body_state = None

    def step(self):
        self.solver.step(self.state_0, self.state_1, None, None, self.frame_dt)
        if self.spec.project_outside:
            self.solver.project_outside(self.state_1, self.state_1, self.frame_dt)
        self.state_0, self.state_1 = self.state_1, self.state_0


@dataclass
class PlateSpec:
    """Resistive-force-theory benchmark (Li, Zhang & Goldman 2013): a rigid plate moved at constant velocity through a
    walled granular bed; the recorded quantity is the force on the plate.

    ``mode="intrude"``: horizontal plate (length x width in x/y, thin in z) pushed straight down from the surface.
    ``mode="drag"``: vertical plate (thin in x) inserted to ``depth`` (plate centre below the surface) and dragged
    along +x. Particles overlapping the plate's starting volume are not created. The bed uses Isaac Lab's material
    defaults and the cell_center lattice; walls and floor are frictional (mu 0.5); the plate's friction is
    ``plate_mu``.
    """

    mode: str = "intrude"
    bed: tuple = (0.8, 0.4, 0.25)
    plate_length: float = 0.15
    plate_width: float = 0.10
    plate_thickness: float = 0.02
    depth: float = 0.12
    travel: float = 0.15
    speed: float = 0.1
    plate_mu: float = 0.4
    wall_mu: float = 0.5
    voxel_size: float = 0.02
    particles_per_cell: float = 3.0
    density: float = 1000.0
    material: dict = field(default_factory=lambda: dict(ISAACLAB_MATERIAL))
    grid_type: str = "dense"
    solver: str = "gs"
    max_iterations: int = 250
    tolerance: float = 1.0e-4
    fps: float = 60.0
    settle_seconds: float = 0.25


class PlateScene:
    def __init__(self, spec: PlateSpec):
        self.spec = spec
        bx, by, bz = spec.bed
        builder = newton.ModelBuilder()
        SolverImplicitMPM.register_custom_attributes(builder)

        if spec.mode == "intrude":
            half = np.array([0.5 * spec.plate_length, 0.5 * spec.plate_width, 0.5 * spec.plate_thickness])
            start = np.array([0.5 * bx, 0.5 * by, bz + half[2] + 0.5 * spec.voxel_size])
            self.direction = np.array([0.0, 0.0, -1.0])
        elif spec.mode == "drag":
            half = np.array([0.5 * spec.plate_thickness, 0.5 * spec.plate_width, 0.5 * spec.plate_length])
            start = np.array([0.25 * bx, 0.5 * by, bz - spec.depth])
            self.direction = np.array([1.0, 0.0, 0.0])
        else:
            raise ValueError(spec.mode)
        self.start = start

        first, res, cell = _cell_center_lattice((0.0, 0.0, 0.0), spec.bed, spec.voxel_size, spec.particles_per_cell)
        ii, jj, kk = np.meshgrid(np.arange(res[0]), np.arange(res[1]), np.arange(res[2]), indexing="ij")
        pts = first + np.stack([ii, jj, kk], -1).reshape(-1, 3) * cell
        inside = np.all(np.abs(pts - start) < half + 0.5 * cell, axis=1)
        pts = pts[~inside]
        n = pts.shape[0]
        cell_volume = float(np.prod(cell))
        radius = 0.5 * float(np.cbrt(cell_volume))
        builder.add_particles(pos=[wp.vec3(*p) for p in pts], vel=[wp.vec3(0.0)] * n,
                              mass=[cell_volume * spec.density] * n, radius=[radius] * n,
                              custom_attributes={f"mpm:{k}": [float(v)] * n for k, v in spec.material.items()})

        self.body = builder.add_body(xform=wp.transform(wp.vec3(*start), wp.quat_identity()), is_kinematic=True)
        builder.add_shape_box(self.body, hx=float(half[0]), hy=float(half[1]), hz=float(half[2]),
                              cfg=newton.ModelBuilder.ShapeConfig(mu=spec.plate_mu, density=0.0))
        t, wall_h = 0.25, 0.5 * bz + 0.3
        wall = newton.ModelBuilder.ShapeConfig(mu=spec.wall_mu, density=0.0)
        for c, hx, hy in (((-t, 0.5 * by, wall_h), t, 0.5 * by + 2 * t), ((bx + t, 0.5 * by, wall_h), t, 0.5 * by + 2 * t),
                          ((0.5 * bx, -t, wall_h), 0.5 * bx + 2 * t, t), ((0.5 * bx, by + t, wall_h), 0.5 * bx + 2 * t, t)):
            builder.add_shape_box(body=-1, cfg=wall, xform=wp.transform(wp.vec3(*c), wp.quat_identity()), hx=hx, hy=hy,
                                  hz=wall_h)
        builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=spec.wall_mu))
        self.model = builder.finalize()

        cfg = SolverImplicitMPM.Config()
        cfg.voxel_size = spec.voxel_size
        cfg.grid_type = spec.grid_type
        cfg.solver = spec.solver
        cfg.max_iterations = spec.max_iterations
        cfg.tolerance = spec.tolerance
        cfg.collider_velocity_mode = "backward"
        self.config = cfg
        self.solver = SolverImplicitMPM(self.model, config=cfg)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.solver.setup_collider(body_mass=wp.zeros_like(self.model.body_mass), body_q=self.state_0.body_q)
        self.frame_dt = 1.0 / spec.fps
        self.settle_frames = int(round(spec.settle_seconds * spec.fps))
        self.frames = self.settle_frames + int(round(spec.travel / spec.speed * spec.fps))
        self.frame = 0
        self._force = np.zeros(3)
        self.info = {"scene": "plate", "spec": asdict(spec), "particle_count": int(n), "plate_start": start.tolist(),
                     "plate_half_extents": half.tolist()}

    @property
    def state(self):
        return self.state_0

    @property
    def body_state(self):
        return self.state_0

    def step(self):
        self.frame += 1
        moved = max(0, self.frame - self.settle_frames) * self.spec.speed * self.frame_dt
        pos = self.start + moved * self.direction
        q = self.state_0.body_q.numpy()
        q[self.body, :3] = pos
        self.state_0.body_q.assign(q)
        self.solver.step(self.state_0, self.state_1, None, None, self.frame_dt)
        self.solver.project_outside(self.state_1, self.state_1, self.frame_dt)
        imp, _pos, cid = self.solver.collect_collider_impulses(self.state_1)
        body_of = self.solver.collider_body_index.numpy()
        ids = cid.numpy()
        mask = (ids >= 0) & (body_of[np.clip(ids, 0, None)] == self.body)
        self._force = imp.numpy()[mask].sum(0) / self.frame_dt
        self.state_1.body_q.assign(self.state_0.body_q)
        self.state_0, self.state_1 = self.state_1, self.state_0

    def extra(self) -> dict:
        """Per-frame scalars saved by the recorder: plate force [N] and plate displacement [m]."""
        moved = max(0, self.frame - self.settle_frames) * self.spec.speed * self.frame_dt
        return {"plate_force": self._force.astype(np.float64), "plate_travel": np.float64(moved)}


class ExampleScene:
    """One of Newton's ``newton/examples/mpm`` examples, unmodified, without a viewer and without CUDA graph capture
    (the recorder runs every device eagerly so the solver loop's stopping rule can be logged; the graph changes
    launch overhead, not results)."""

    def __init__(self, name: str, seconds: float | None = None, overrides: dict | None = None, capture: bool = False):
        """``capture=True`` keeps the example's own whole-step graph capture (CUDA only; timing runs)."""
        mod = importlib.import_module(f"newton.examples.mpm.example_{name}")
        Example = mod.Example
        if not hasattr(Example, "_metalsim_capture"):
            Example._metalsim_capture = getattr(Example, "capture", None)
        import newton.examples

        parser = Example.create_parser() if hasattr(Example, "create_parser") else newton.examples.create_parser()
        args = parser.parse_args([])
        for k, v in (overrides or {}).items():
            setattr(args, k, v)
        # avoid the example's own outer graph capture
        def no_capture(self):
            self.graph = None
            self.sand_graph = None

        if capture and Example._metalsim_capture is not None:
            Example.capture = Example._metalsim_capture
        else:
            Example.capture = no_capture
        from newton.viewer import ViewerNull

        self.example = Example(ViewerNull(num_frames=1 << 30), args)
        self.args = args
        fps = getattr(args, "fps", None) or getattr(self.example, "fps", 60)
        self.frame_dt = getattr(self.example, "frame_dt", 1.0 / fps)
        self.frames = int(round((seconds or 2.0) / self.frame_dt))
        self.model = self.example.model
        self.solver = getattr(self.example, "mpm_solver", None) or getattr(self.example, "solver", None)
        self.info = {"scene": f"example:{name}", "args": {k: _jsonable(v) for k, v in vars(args).items()},
                     "particle_count": int(self.model.particle_count)}

    @property
    def state(self):
        """State holding the MPM particles (the sand state in the coupled examples)."""
        ex = self.example
        return getattr(ex, "sand_state_0", None) or ex.state_0

    @property
    def body_state(self):
        """State holding rigid bodies, if the example has any."""
        ex = self.example
        return ex.state_0 if getattr(ex.model, "body_count", 0) else None

    def step(self):
        self.example.step()


def _jsonable(v):
    if isinstance(v, (int, float, str, bool)) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return str(v)


def make_scene(name: str, **kw):
    if name == "column":
        return ColumnScene(ColumnSpec(**kw))
    if name == "plate":
        return PlateScene(PlateSpec(**{k: tuple(v) if k == "bed" else v for k, v in kw.items()}))
    if name.startswith("example:"):
        return ExampleScene(name.split(":", 1)[1], **kw)
    raise ValueError(f"unknown scene {name!r}")

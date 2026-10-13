"""MPM material (sand, snow, mud, ...) in MetalSim's main simulator, two-way coupled with a ``BatchSim`` robot.

The granular solver is Isaac Lab 3.0-EA's: Newton 1.5.2's implicit MPM (``SolverImplicitMPM``) with MetalSim's Metal
changes, run from the environment's Newton 1.7 types through MetalSim's vendored copy of the fork's solver package
(:mod:`metalsim.physics.mpm.solver`). The coupling reproduces Isaac Lab's ``SolverCoupledProxy`` exchange with
MuJoCo Warp (research and every source line: ``research_notes/MPM in main simulator/NOTE.md``), with ``BatchSim`` as
the rigid entry:

* the coupled MuJoCo bodies appear in the MPM model as *proxy bodies* (``BodyFlags.PROXY``) carrying the bodies'
  collision geoms; inside the implicit MPM solve they are compliant colliders with a mass (``proxy_mass``):
  ``"effective"`` (Isaac's articulated effective mass from MuJoCo's ``body_invweight0`` x ``mass_scale``), ``"body"``
  (the link's own mass and inertia, Newton's two-way example) or ``"kinematic"`` (zero: an infinitely stiff collider);
* one coupled step = one ``BatchSim`` step (its ``substeps`` MuJoCo steps): the last harvested sand wrench is added
  to ``xfrc_applied`` for the MuJoCo substeps (and the caller's own ``xfrc_applied`` restored afterwards); the proxies
  take the robot's state (``"staggered"``, Isaac's Franka-Pour: end pose and velocity; ``"lagged"``: begin pose, end
  velocity, minus the lagged sand impulse); ``mpm_substeps`` MPM steps; the proxies' wrenches are harvested with the
  solver's own fixed-order harvest (``coupling_harvest_proxy_wrenches``; Isaac's: the last MPM substep's impulses over
  the coupled step's dt, ``harvest="last_substep"``; ``"mean"`` uses every substep's);
* static geoms (bodies welded to the world, e.g. the ground plane) are static colliders.

Usage::

    sim = BatchSim(model, num_envs=64)
    mpm = MPMCoupling(sim, MPMConfig(voxel_size=0.02))
    mpm.add_emitter(GridEmitter(lower=(-0.2, -0.2, 0.0), upper=(0.2, 0.2, 0.1), material=MPMMaterial(friction=0.7)))
    mpm.finalize()                  # builds the MPM model (N worlds) and solver
    for _ in range(T):
        v = mpm.step()              # one coupled step; returns sim.event's value
    renderer.set_particles(mpm.particle_q)   # (N * P, 3) Warp array, world-major
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, fields

import mujoco
import numpy as np
import warp as wp

from metalsim.physics.mpm import colliders as C

# ---------------------------------------------------------------------------------------------------------------------
# Configuration (Isaac Lab 3.0-EA's MPM configuration classes, same names and defaults)
# ---------------------------------------------------------------------------------------------------------------------


@dataclass
class MPMMaterial:
    """``MPMParticleMaterialCfg`` (IL ``sim/spawners/mpm/mpm_cfg.py``): density [kg/m^3] gives particle masses;
    the rest become Newton's ``mpm:*`` per-particle attributes."""

    density: float = 1000.0
    young_modulus: float = 1.0e15
    poisson_ratio: float = 0.3
    viscosity: float = 0.0
    friction: float = 0.68
    damping: float = 0.0
    yield_pressure: float = 1.0e12
    tensile_yield_ratio: float = 0.0
    yield_stress: float = 0.0
    hardening: float = 0.0
    dilatancy: float = 0.0

    def attributes(self) -> dict[str, float]:
        return {f"mpm:{f.name}": float(getattr(self, f.name)) for f in fields(self) if f.name != "density"}


SAND = MPMMaterial(density=1500.0, friction=0.7)          # Isaac Lab's Franka-Pour media (pour_env_cfg.py:152-156)


@dataclass
class GridEmitter:
    """``MPMGridCfg``: particles on a lattice filling the box ``[lower, upper]`` (world frame of every env).
    ``voxel_size`` None = the solver's; ``placement`` ``"cell_center"`` or ``"boundary"``; ``jitter`` [m] as Newton's
    ``add_particle_grid`` applies it."""

    lower: tuple = (0.0, 0.0, 0.0)
    upper: tuple = (0.1, 0.1, 0.1)
    voxel_size: float | None = None
    particles_per_cell: float = 3.0
    placement: str = "cell_center"
    jitter: float = 0.0
    mass: float | None = None
    radius: float | None = None
    velocity: tuple = (0.0, 0.0, 0.0)
    material: MPMMaterial = field(default_factory=MPMMaterial)

    def emit(self, builder, voxel_size: float) -> None:
        """Isaac Lab's ``_emit_grid`` (IL ``sim/spawners/mpm/mpm.py:102-174``) at the identity env pose."""
        lower = np.asarray(self.lower, dtype=np.float32)
        upper = np.asarray(self.upper, dtype=np.float32)
        extent = upper - lower
        if np.any(extent <= 0.0):
            raise ValueError(f"GridEmitter: upper {self.upper} must exceed lower {self.lower}")
        voxel = float(self.voxel_size or voxel_size)
        res = np.maximum(np.ceil(float(self.particles_per_cell) * extent / voxel), 1).astype(np.int32)
        cell = extent / res
        cell_volume = float(np.prod(cell))
        mass = cell_volume * float(self.material.density) if self.mass is None else float(self.mass)
        if self.placement == "boundary":
            first, dims = lower, res + 1
        elif self.placement == "cell_center":
            first, dims = lower + 0.5 * cell, res
        else:
            raise ValueError(f"placement must be 'boundary' or 'cell_center', got {self.placement!r}")
        if self.radius is not None:
            radius = float(self.radius)
        elif self.placement == "cell_center":
            radius = 0.5 * float(np.cbrt(cell_volume))
        else:
            radius = 0.5 * float(np.max(cell))
        builder.add_particle_grid(
            pos=wp.vec3(*first.tolist()), rot=wp.quat_identity(), vel=wp.vec3(*(float(v) for v in self.velocity)),
            dim_x=int(dims[0]), dim_y=int(dims[1]), dim_z=int(dims[2]),
            cell_x=float(cell[0]), cell_y=float(cell[1]), cell_z=float(cell[2]),
            mass=mass, jitter=float(self.jitter), radius_mean=radius, custom_attributes=self.material.attributes())


@dataclass
class PointsEmitter:
    """``MPMPointsCfg``: explicit particles (world frame of every env)."""

    positions: np.ndarray = None
    velocities: np.ndarray | None = None
    mass: float | np.ndarray = 1.0e-3
    radius: float | np.ndarray = 5.0e-3
    material: MPMMaterial = field(default_factory=MPMMaterial)

    def emit(self, builder, voxel_size: float) -> None:
        p = np.asarray(self.positions, np.float32).reshape(-1, 3)
        v = np.zeros_like(p) if self.velocities is None else np.asarray(self.velocities, np.float32).reshape(p.shape)
        n = p.shape[0]
        mass = np.broadcast_to(np.asarray(self.mass, np.float64), (n,)).tolist()
        radius = np.broadcast_to(np.asarray(self.radius, np.float64), (n,)).tolist()
        builder.add_particles(pos=p.tolist(), vel=v.tolist(), mass=mass, radius=radius,
                              custom_attributes=self.material.attributes())


@dataclass
class MPMConfig:
    """``MPMSolverCfg`` defaults (IL ``physics/mpm_manager_cfg.py``) plus the coupling settings.

    ``separate_worlds`` is always on (each ``BatchSim`` world has its own material). ``max_active_cell_count``
    ``"auto"``: Isaac's Franka-Pour capacity rule (particles per world rounded up to 512, times the worlds) with a
    rebuildable sparse grid, which makes the step capturable; ``-1``: Newton's host-sized sparse grid (eager only)."""

    max_iterations: int = 250
    tolerance: float = 1.0e-4
    solver: str = "auto"
    warmstart_mode: str = "auto"
    collider_velocity_mode: str = "forward"
    voxel_size: float = 0.1
    grid_type: str = "sparse"
    grid_padding: int = 0
    max_active_cell_count: int | str = "auto"
    transfer_scheme: str = "apic"
    integration_scheme: str = "pic"
    critical_fraction: float = 0.0
    air_drag: float = 1.0
    collider_basis: str = "S2"
    strain_basis: str = "P0"
    velocity_basis: str = "Q1"
    solver_extra: dict = field(default_factory=dict)        # any other SolverImplicitMPM.Config field
    # coupling
    mpm_substeps: int = 1                 # MPM steps per coupled (BatchSim) step
    mode: str = "staggered"               # "staggered" (Isaac's Franka-Pour) | "lagged"
    proxy_mass: str = "effective"         # "effective" | "body" | "kinematic"
    mass_scale: float = 1.0               # Isaac's proxy mass_scale
    feedback: bool = True                 # False: the sand does not push the robot (one-way, Newton's ANYmal example)
    harvest: str = "last_substep"         # "last_substep" (Isaac) | "mean"
    collider_friction: float | None = None   # sand-collider Coulomb friction; None = each geom's sliding friction
    collider_margin: float = 0.0          # collider thickness [m] (Newton shape margin)
    capture: bool = True                  # one graph per coupled step when the grid allows it
    # compat (newton_compat.install, Franka-Pour's settings)
    chunk_graph: bool = True
    gpu_exit: bool = True
    metal_fast_math: str = "off"

    def solver_config(self, cls, particles_per_world: int, worlds: int):
        cfg = cls.Config()
        for name in ("max_iterations", "tolerance", "solver", "warmstart_mode", "collider_velocity_mode", "voxel_size",
                     "grid_type", "grid_padding", "transfer_scheme", "integration_scheme", "critical_fraction",
                     "air_drag", "collider_basis", "strain_basis", "velocity_basis"):
            setattr(cfg, name, getattr(self, name))
        cells = self.max_active_cell_count
        if cells == "auto":
            cells = ((particles_per_world + 511) // 512) * 512 * worlds if self.grid_type == "sparse" else -1
        cfg.max_active_cell_count = int(cells)
        cfg.separate_worlds = True
        for k, v in self.solver_extra.items():
            if not hasattr(cfg, k):
                raise ValueError(f"SolverImplicitMPM.Config has no field {k!r}")
            setattr(cfg, k, v)
        return cfg


# ---------------------------------------------------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------------------------------------------------


@wp.kernel
def _proxy_state_from_mujoco(xpos: wp.array2d(dtype=wp.vec3), xquat: wp.array2d(dtype=wp.quat),
                             xipos: wp.array2d(dtype=wp.vec3), cvel: wp.array2d(dtype=wp.spatial_vector),
                             subtree_com: wp.array2d(dtype=wp.vec3), body_rootid: wp.array(dtype=int),
                             body_map: wp.array(dtype=int), write_q: int, write_qd: int,
                             body_q: wp.array(dtype=wp.transform), body_qd: wp.array(dtype=wp.spatial_vector)):
    """Proxy ``body_q`` (body frame: MuJoCo's ``xpos`` / ``xquat``, quaternion w-first there) and ``body_qd``
    (Newton: linear velocity of the centre of mass, angular velocity, world frame) from MuJoCo Warp's state.
    MuJoCo's ``cvel`` is (angular, linear) about the root's subtree centre of mass."""
    w, k = wp.tid()
    nb = body_map.shape[0]
    b = body_map[k]
    i = w * nb + k
    if write_q != 0:
        q = xquat[w, b]
        body_q[i] = wp.transform(xpos[w, b], wp.quat(q[1], q[2], q[3], q[0]))
    if write_qd != 0:
        cv = cvel[w, b]
        ang = wp.spatial_top(cv)
        lin = wp.spatial_bottom(cv)
        v_com = lin + wp.cross(ang, xipos[w, b] - subtree_com[w, body_rootid[b]])
        body_qd[i] = wp.spatial_vector(v_com, ang)


@wp.kernel
def _add_coupling_wrench(f: wp.array(dtype=wp.spatial_vector), body_map: wp.array(dtype=int),
                         xfrc: wp.array2d(dtype=wp.spatial_vector)):
    """``xfrc_applied`` (world-frame force and torque about the body's centre of mass, as Newton's ``body_f``) +=
    the proxy's harvested wrench."""
    w, k = wp.tid()
    nb = body_map.shape[0]
    b = body_map[k]
    xfrc[w, b] = xfrc[w, b] + f[w * nb + k]


@wp.kernel
def _accumulate(src: wp.array(dtype=wp.spatial_vector), dst: wp.array(dtype=wp.spatial_vector)):
    i = wp.tid()
    dst[i] = dst[i] + src[i]


@wp.kernel
def _zero_masked_wrench(world_mask: wp.array(dtype=wp.bool), nb: int, f: wp.array(dtype=wp.spatial_vector)):
    i = wp.tid()
    if world_mask[i // nb]:
        f[i] = wp.spatial_vector()


@wp.kernel
def _restore_masked_particles(world_mask: wp.array(dtype=wp.bool), particle_world: wp.array(dtype=wp.int32),
                              q0: wp.array(dtype=wp.vec3), qd0: wp.array(dtype=wp.vec3),
                              q: wp.array(dtype=wp.vec3), qd: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    w = particle_world[i]
    if w >= 0 and world_mask[w]:
        q[i] = q0[i]
        qd[i] = qd0[i]


# ---------------------------------------------------------------------------------------------------------------------
# Effective mass (Isaac's SolverMuJoCo.coupling_eval_effective_mass_block, Newton 1.5.2 mujoco/kernels.py:321-382)
# ---------------------------------------------------------------------------------------------------------------------


def effective_mass(model: mujoco.MjModel, b: int) -> tuple[float, np.ndarray]:
    """Mass and inertia (body frame, about the centre of mass) Isaac's coupler gives MuJoCo body ``b``'s proxy:
    ``m = 1 / body_invweight0[b, 0]`` (r = 0), inertia scaled by ``max(trace(I^-1) / 3 / invweight0[b, 1], 1)``
    (float32, as Newton's kernel)."""
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, model.body_iquat[b])
    R = R.reshape(3, 3)
    inertia = (R @ np.diag(model.body_inertia[b]) @ R.T).astype(np.float32)
    mass = np.float32(model.body_mass[b])
    inv_mass, inv_rot = (np.float32(v) for v in model.body_invweight0[b])
    if inv_mass > 0:
        mass = np.float32(1.0) / inv_mass
    det = np.linalg.det(inertia.astype(np.float64))
    if inv_rot > 0 and abs(det) > 1e-30:
        free_inv_rot = np.float32(np.trace(np.linalg.inv(inertia))) / np.float32(3.0)
        inertia = inertia * max(free_inv_rot / inv_rot, np.float32(1.0))
    elif model.body_mass[b] > 0:
        inertia = inertia * max(mass / np.float32(model.body_mass[b]), np.float32(1.0))
    return float(mass), inertia


# ---------------------------------------------------------------------------------------------------------------------
# Coupler
# ---------------------------------------------------------------------------------------------------------------------


class MPMCoupling:
    """MPM material in every world of a :class:`~metalsim.physics.batch.BatchSim`, two-way coupled with its bodies.

    Call :meth:`add_emitter` (and optionally :meth:`set_colliders`), then :meth:`finalize`, then :meth:`step` instead
    of ``sim.step()``. A scene without an ``MPMCoupling`` runs exactly as before: nothing here touches ``BatchSim``.
    """

    def __init__(self, sim, config: MPMConfig | None = None):
        self.sim = sim
        self.cfg = config or MPMConfig()
        if self.cfg.mode not in ("staggered", "lagged"):
            raise ValueError(f"mode must be 'staggered' or 'lagged', got {self.cfg.mode!r}")
        if self.cfg.proxy_mass not in ("effective", "body", "kinematic"):
            raise ValueError(f"proxy_mass must be 'effective', 'body' or 'kinematic', got {self.cfg.proxy_mass!r}")
        if self.cfg.harvest not in ("last_substep", "mean"):
            raise ValueError(f"harvest must be 'last_substep' or 'mean', got {self.cfg.harvest!r}")
        self.emitters: list = []
        self._geoms = None
        self._bodies = None
        self.finalized = False

    # -- scene -----------------------------------------------------------------------------------------------------

    def add_emitter(self, emitter) -> None:
        """Add material: a :class:`GridEmitter` or :class:`PointsEmitter` (the same particles in every world)."""
        if self.finalized:
            raise RuntimeError("add_emitter after finalize()")
        self.emitters.append(emitter)

    def set_colliders(self, geoms=None, bodies=None) -> None:
        """Which MuJoCo geoms the material collides with (default: every colliding geom) and which bodies are coupled
        proxies (default: every moving body with such a geom). Names or ids."""
        m = self.sim.mj_model

        def ids(seq, obj):
            return None if seq is None else [mujoco.mj_name2id(m, obj, s) if isinstance(s, str) else int(s) for s in seq]

        self._geoms = ids(geoms, mujoco.mjtObj.mjOBJ_GEOM)
        self._bodies = ids(bodies, mujoco.mjtObj.mjOBJ_BODY)

    # -- build -----------------------------------------------------------------------------------------------------

    def finalize(self) -> None:
        import newton

        from metalsim.physics.mpm import newton_compat
        from metalsim.physics.mpm.solver import solver_class

        cfg, sim = self.cfg, self.sim
        if not self.emitters:
            raise ValueError("no MPM material: add_emitter() first")
        # CUDA's stopping rule and float32 accumulation (MPM.md section 1) for the MPM kernels only: MuJoCo Warp's
        # compile options stay as they are (no rigid determinism patch, no global argument-space change)
        newton_compat.install(accumulate="float32", loop="cuda", log=False, chunk_graph=cfg.chunk_graph,
                              gpu_exit=cfg.gpu_exit, rigid_deterministic=None, device_args=None,
                              metal_fast_math=cfg.metal_fast_math)
        Solver = solver_class()
        model = sim.mj_model
        data = mujoco.MjData(model)
        data.qpos[:] = sim.d.qpos.numpy()[0]
        mujoco.mj_forward(model, data)
        self.layout = lay = C.layout(model, self._geoms, self._bodies)
        meshes = C.GeomMeshes(newton, model)
        builder = newton.ModelBuilder()
        Solver.register_custom_attributes(builder)
        n = sim.n
        p_counts = []
        for _w in range(n):
            builder.begin_world()
            C.add_world_colliders(newton, builder, model, data, lay, friction=cfg.collider_friction,
                                  margin=cfg.collider_margin, meshes=meshes)
            p0 = builder.particle_count
            for e in self.emitters:
                e.emit(builder, cfg.voxel_size)
            p_counts.append(builder.particle_count - p0)
            builder.end_world()
        if len(set(p_counts)) != 1:
            raise RuntimeError(f"worlds got different particle counts {sorted(set(p_counts))}")
        self.particles_per_world = p_counts[0]
        with wp.ScopedDevice(sim.device):
            mm = builder.finalize()
            self._set_proxy_masses(newton, mm)
            self.model = mm
            self.solver_cfg = cfg.solver_config(Solver, self.particles_per_world, n)
            self.solver = Solver(mm, config=self.solver_cfg)
            self.state = mm.state()
            nb = lay.nb
            self.nb = nb
            self._body_map = wp.array(lay.body_map, dtype=int)
            self._rootid = wp.array(model.body_rootid, dtype=int)
            self._proxy_ids = wp.array(np.arange(n * nb), dtype=int)
            self.coupling_f = wp.zeros(n * nb, dtype=wp.spatial_vector)    # wrench applied in the next rigid step
            self._harvest_tmp = wp.zeros(n * nb, dtype=wp.spatial_vector)
            self._user_xfrc = wp.zeros_like(sim.d.xfrc_applied)
            self._q0 = wp.clone(self.state.particle_q)
            self._qd0 = wp.clone(self.state.particle_qd)
            self._sync(write_q=True, write_qd=True)
            self.solver.reset(self.state)
        self.dt = sim.dt
        self.capturable = cfg.capture and (cfg.grid_type in ("fixed", "dense") or
                                           int(self.solver_cfg.max_active_cell_count) > 0)
        self._graph = None
        self.finalized = True

    def _set_proxy_masses(self, newton, mm) -> None:
        """Proxy flags, and the mass / inertia the MPM solve gives each proxy collider (``cfg.proxy_mass``)."""
        model, lay, n, nb = self.sim.mj_model, self.layout, self.sim.n, self.layout.nb
        flags = mm.body_flags.numpy()
        flags[:n * nb] |= int(newton.BodyFlags.PROXY)
        mm.body_flags.assign(flags)
        mass = np.zeros(n * nb, np.float32)
        inertia = np.zeros((n * nb, 3, 3), np.float32)
        for k, b in enumerate(lay.body_map):
            if self.cfg.proxy_mass == "kinematic" or model.body_mocapid[b] >= 0:
                m, I = 0.0, np.zeros((3, 3), np.float32)
            elif self.cfg.proxy_mass == "effective":
                m, I = effective_mass(model, int(b))
            else:
                R = np.zeros(9)
                mujoco.mju_quat2Mat(R, model.body_iquat[b])
                R = R.reshape(3, 3)
                m, I = float(model.body_mass[b]), (R @ np.diag(model.body_inertia[b]) @ R.T).astype(np.float32)
            s = np.float32(self.cfg.mass_scale)
            mass[k::nb] = np.float32(m) * s
            inertia[k::nb] = I * s
        inv_mass = np.where(mass > 0, 1.0 / np.maximum(mass, 1e-30), 0.0).astype(np.float32)
        inv_inertia = np.zeros_like(inertia)
        pos = mass > 0
        inv_inertia[pos] = np.linalg.inv(inertia[pos].astype(np.float64)).astype(np.float32)
        mm.body_mass.assign(mass)
        mm.body_inv_mass.assign(inv_mass)
        mm.body_inertia.assign(inertia)
        mm.body_inv_inertia.assign(inv_inertia)
        self.proxy_mass = mass[:nb].copy()

    # -- stepping --------------------------------------------------------------------------------------------------

    def _sync(self, write_q: bool, write_qd: bool) -> None:
        d = self.sim.d
        wp.launch(_proxy_state_from_mujoco, dim=(self.sim.n, self.nb),
                  inputs=[d.xpos, d.xquat, d.xipos, d.cvel, d.subtree_com, self._rootid, self._body_map,
                          int(write_q), int(write_qd)],
                  outputs=[self.state.body_q, self.state.body_qd])

    def _launch_coupled_step(self) -> None:
        sim, cfg, n, nb = self.sim, self.cfg, self.sim.n, self.nb
        d = sim.d
        wp.copy(self._user_xfrc, d.xfrc_applied)
        if cfg.feedback:
            wp.launch(_add_coupling_wrench, dim=(n, nb), inputs=[self.coupling_f, self._body_map],
                      outputs=[d.xfrc_applied])
        if cfg.mode == "lagged":
            self._sync(write_q=True, write_qd=False)            # begin pose
        sim.launch_step()
        wp.copy(d.xfrc_applied, self._user_xfrc)
        if cfg.mode == "staggered":
            self._sync(write_q=True, write_qd=True)             # end pose and velocity
        else:
            self._sync(write_q=False, write_qd=True)            # end velocity, minus the lagged sand impulse
            if cfg.feedback:
                self.solver.coupling_rewind_proxy_body(self._proxy_ids, self.state, self.coupling_f, None, self.dt)
        h = self.dt / cfg.mpm_substeps
        for s in range(cfg.mpm_substeps):
            self.solver.step(self.state, self.state, None, None, h)
            if cfg.harvest == "mean":
                self._harvest(self._harvest_tmp)
                if s == 0:
                    wp.copy(self.coupling_f, self._harvest_tmp)
                else:
                    wp.launch(_accumulate, dim=n * nb, inputs=[self._harvest_tmp], outputs=[self.coupling_f])
        if cfg.harvest == "last_substep":
            self._harvest(self.coupling_f)

    def _harvest(self, out: wp.array) -> None:
        self.solver.coupling_harvest_proxy_wrenches(self._proxy_ids, out, body_qd_before=None, state=self.state,
                                                    state_out=self.state, contacts=None, dt=self.dt)

    def step(self) -> int:
        """One coupled step (one ``BatchSim`` step); returns the ``sim.event`` value signalled at its end."""
        if not self.finalized:
            raise RuntimeError("finalize() first")
        with wp.ScopedDevice(self.sim.device):
            if self.capturable:
                if self._graph is None:
                    with wp.ScopedCapture(device=self.sim.device) as cap:
                        self._launch_coupled_step()
                    self._graph = cap.graph
                wp.capture_launch(self._graph)
            else:
                self._launch_coupled_step()
        return self.sim._signal()

    # -- state -----------------------------------------------------------------------------------------------------

    @property
    def particle_q(self) -> wp.array:
        """Particle positions [m], (N * P, 3) Warp array, world-major (the renderer's ``set_particles``)."""
        return self.state.particle_q

    @property
    def particle_qd(self) -> wp.array:
        """Particle velocities [m/s], (N * P, 3)."""
        return self.state.particle_qd

    def particles(self) -> np.ndarray:
        """(N, P, 3) positions on the host (synchronizes)."""
        return self.state.particle_q.numpy().reshape(self.sim.n, self.particles_per_world, 3)

    def wrenches(self) -> np.ndarray:
        """(N, nb, 6) the sand wrench (force, torque about the COM, world frame) the next step applies to each coupled
        body, in ``layout.body_map`` order (synchronizes)."""
        return self.coupling_f.numpy().reshape(self.sim.n, self.nb, 6)

    def reset(self, mask=None) -> None:
        """Put the material of the worlds in ``mask`` ((N,) bool, None = all) back to its initial state, clear their
        sand wrench and the solver's history (``SolverImplicitMPM.reset`` per world)."""
        import newton

        n = self.sim.n
        m = np.ones(n, bool) if mask is None else np.asarray(mask, bool).reshape(n)
        with wp.ScopedDevice(self.sim.device):
            wm = wp.array(np.append(m, False), dtype=wp.bool)
            wp.launch(_restore_masked_particles, dim=self.state.particle_q.shape[0],
                      inputs=[wm, self.model.particle_world, self._q0, self._qd0],
                      outputs=[self.state.particle_q, self.state.particle_qd])
            wp.launch(_zero_masked_wrench, dim=n * self.nb, inputs=[wm, self.nb], outputs=[self.coupling_f])
            self._sync(write_q=True, write_qd=True)
            self.solver.reset(self.state, world_mask=wm,
                              flags=int(newton.StateFlags.PARTICLE_Q) | int(newton.StateFlags.PARTICLE_QD))

    def check_status(self) -> None:
        """Raise if the rebuildable sparse grid overflowed its capacity (synchronizes)."""
        if hasattr(self.solver, "check_sparse_grid_rebuild_status"):
            self.solver.check_sparse_grid_rebuild_status()

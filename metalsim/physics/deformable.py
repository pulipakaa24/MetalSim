"""Deformable bodies (cloth, cables) on Metal through MuJoCo Warp's flex path.

MuJoCo models deformables as *flexes*: point-mass vertices (one body with three slide joints each)
joined by 1D capsules (cables), 2D triangles (cloth) or 3D tetrahedra (soft volumes). This module
builds flex scenes, steps N worlds of one on ``metal:0`` with a replayed graph, and provides the
helpers the tests and the benchmark use:

    model = scene_model("box")                 # cloth + cable dropping onto a box
    sim = DeformableSim(model, num_envs=1024)
    sim.randomize(seed=0)                      # per-world offsets of each flex (domain randomization)
    sim.step()                                 # graph replay, no host sync
    sim.vertices()                             # (N, nflexvert, 3) zero-copy MPS tensor
    sim.energy()                               # (N, 2) potential / kinetic

Model choices (see docs/research/deformables_2026-09-25.md): cloth and cable stretch is carried by
flex *edge equality constraints* (implicit, stable at the rigid-body timestep), not by the explicit
continuum elasticity (``<elasticity young=...>``), which needs a smaller step for stiff cloth.
The Jacobian is sparse (every flex vertex adds 3 DOFs), and the per-world constraint Jacobian
non-zero budget is sized from the flex structure, not MuJoCo Warp's default ``njmax * nv`` (which
would need gigabytes at thousands of worlds).
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import mujoco
import numpy as np
import warp as wp

import mujoco_warp as mjw

# --------------------------------------------------------------------------------------------------
# Scenes
# --------------------------------------------------------------------------------------------------

_CLOTH = """
    <flexcomp name="cloth" type="grid" count="{n} {n} 1" spacing="{sp} {sp} {sp}" pos="{x} {y} {z}" dim="2"
              radius="{r}" mass="{mass}" rgba="0.2 0.5 0.8 1">
      <edge equality="true" solref="{eq_solref}"/>
      <contact condim="3" solref="0.01 1" friction="{mu}" selfcollide="none"/>
    </flexcomp>"""

_CABLE = """
    <flexcomp name="cable" type="grid" count="{n} 1 1" spacing="{sp} {sp} {sp}" pos="{x} {y} {z}" dim="1"
              radius="{r}" mass="{mass}" rgba="0.9 0.6 0.1 1">
      <edge equality="true" solref="{eq_solref}"/>
      <contact condim="3" solref="0.01 1" friction="{mu}"/>
    </flexcomp>"""


@dataclass
class ClothCfg:
    count: int = 10            # vertices per side
    spacing: float = 0.04      # m
    pos: tuple = (0.0, 0.0, 0.45)
    radius: float = 0.005      # collision radius (thickness / 2)
    mass: float = 0.2          # kg, whole cloth
    friction: float = 0.8
    eq_solref: str = "0.02 1"  # edge-equality time constant / damping ratio


@dataclass
class CableCfg:
    count: int = 12
    spacing: float = 0.03
    pos: tuple = (0.12, 0.28, 0.40)
    radius: float = 0.006
    mass: float = 0.05
    friction: float = 0.8
    eq_solref: str = "0.02 1"


_SOFT = """
    <flexcomp name="soft" type="grid" count="{n} {n} {n}" spacing="{sp} {sp} {sp}" pos="{x} {y} {z}" dim="3"
              radius="{r}" mass="{mass}" dof="{dof}" rgba="0.3 0.8 0.4 1">
      <elasticity young="{E}" poisson="{nu}" damping="{damp}"/>
      <contact condim="3" solref="0.01 1" friction="{mu}" selfcollide="none"/>
    </flexcomp>"""


@dataclass
class SoftCfg:
    """A soft cube (tetrahedral FEM, MuJoCo's continuum elasticity), Isaac's DeformableObject analogue."""
    count: int = 4
    spacing: float = 0.05
    pos: tuple = (0.02, 0.01, 0.45)
    radius: float = 0.003
    mass: float = 0.5
    dof: str = "trilinear"     # 8 nodes (24 DOFs) drive all 64 vertices; "full": 3 DOFs per vertex
    young: float = 5e3         # Pa
    poisson: float = 0.3
    damping: float = 0.01
    friction: float = 0.8


def _soft_xml(c: SoftCfg) -> str:
    return _SOFT.format(n=c.count, sp=c.spacing, x=c.pos[0], y=c.pos[1], z=c.pos[2], r=c.radius, mass=c.mass, dof=c.dof,
                        E=c.young, nu=c.poisson, damp=c.damping, mu=c.friction)


def _cloth_xml(c: ClothCfg) -> str:
    return _CLOTH.format(n=c.count, sp=c.spacing, x=c.pos[0], y=c.pos[1], z=c.pos[2], r=c.radius, mass=c.mass,
                         mu=c.friction, eq_solref=c.eq_solref)


def _cable_xml(c: CableCfg) -> str:
    return _CABLE.format(n=c.count, sp=c.spacing, x=c.pos[0], y=c.pos[1], z=c.pos[2], r=c.radius, mass=c.mass,
                         mu=c.friction, eq_solref=c.eq_solref)


def box_scene_xml(cloth: ClothCfg | None = ClothCfg(), cable: CableCfg | None = CableCfg(), timestep: float = 0.002,
                  iterations: int = 20, ls_iterations: int = 10, solver: str = "CG", energy: bool = True,
                  soft: SoftCfg | None = None) -> str:
    """A cloth and a cable (and optionally a soft cube) dropping onto a 0.3 m box on a plane."""
    flexes = (_cloth_xml(cloth) if cloth else "") + (_cable_xml(cable) if cable else "") + (_soft_xml(soft) if soft else "")
    flag = '<flag energy="enable"/>' if energy else ""
    return f"""
<mujoco model="metalsim_deformable_box">
  <option timestep="{timestep}" solver="{solver}" iterations="{iterations}" ls_iterations="{ls_iterations}"
          jacobian="sparse" cone="pyramidal">{flag}</option>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="2 2 0.1" friction="0.8 0.005 0.0001"/>
    <geom name="box" type="box" size="0.15 0.15 0.15" pos="0 0 0.15" friction="0.8 0.005 0.0001" rgba="0.7 0.3 0.3 1"/>{flexes}
  </worldbody>
</mujoco>"""


def scene_model(kind: str = "box", cloth: ClothCfg | None = None, cable: CableCfg | None = None, **kw) -> mujoco.MjModel:
    """``kind="box"``: cloth + cable onto a box. ``kind="g1"``: cloth + cable dropped onto Isaac's G1 (standing,
    its PD targets held at the initial pose; keyframe 0)."""
    if kind == "box":
        return mujoco.MjModel.from_xml_string(box_scene_xml(cloth or ClothCfg(), cable if cable is not None else CableCfg(), **kw))
    if kind == "g1":
        return g1_scene_model(cloth, cable, **kw)
    raise ValueError(kind)


def g1_scene_model(cloth: ClothCfg | None = None, cable: CableCfg | None = None, energy: bool = True,
                   physics_dt: float = 0.0025, flexes: bool = True) -> mujoco.MjModel:
    """Isaac's G1 (``metalsim.learn.g1_velocity.build_g1_model``) with a cloth over its shoulders and a cable;
    its PD targets held at the initial pose (keyframe 0). ``flexes=False``: the G1 alone (XPBD's rigid side)."""
    from metalsim.learn.g1_velocity import build_g1_model

    cloth = cloth or ClothCfg(count=12, spacing=0.05, pos=(0.0, 0.0, 1.55), mass=0.3)
    cable = cable or CableCfg(count=16, spacing=0.04, pos=(0.35, -0.3, 1.2))
    _, info = build_g1_model(physics_dt=physics_dt)
    spec = info["spec"]
    if flexes:
        child = mujoco.MjSpec.from_string(f"<mujoco><worldbody>{_cloth_xml(cloth)}{_cable_xml(cable)}</worldbody></mujoco>")
        spec.attach(child, frame=spec.worldbody.add_frame(), prefix="d_")
        spec.option.jacobian = mujoco.mjtJacobian.mjJAC_SPARSE      # MuJoCo Warp: dense only up to nv 60
        spec.option.solver = mujoco.mjtSolver.mjSOL_CG               # hundreds of flex DOFs: no Newton factorization
        spec.option.iterations = 30
        spec.option.ls_iterations = 10
    if energy:
        spec.option.enableflags |= mujoco.mjtEnableBit.mjENBL_ENERGY
    m = spec.compile()
    key = m.key_qpos[0]
    ctrl = np.array([key[m.jnt_qposadr[m.actuator_trnid[a, 0]]] for a in range(m.nu)])
    spec.keys[0].ctrl = ctrl.tolist()
    return spec.compile()


# --------------------------------------------------------------------------------------------------
# Sizing
# --------------------------------------------------------------------------------------------------

def default_sizes(model: mujoco.MjModel, contacts_per_vertex: float = 6.0) -> dict:
    """Per-world contact / constraint budgets for a flex scene.

    CCD slots: 128 per world (see below). Contacts: the contact buffer also holds the raw candidates before the per-pair MJ_MAXCONPAIR (50)
    selection, and a cloth draped on a box produces up to 3 per triangle vertex plus box corners (measured:
    384 raw candidates for a 10x10 cloth, 3.4 per vertex), so the default budget is 6 per vertex (plus the
    rigid scene's own);
    rows: every flex edge-equality row, 2*(condim-1) pyramid rows per contact, and the rigid scene's
    limits; Jacobian non-zeros: an edge row touches 6 DOFs, a vertex contact row 3 (a rigid-body contact
    row up to nv of the kinematic chain, bounded here by the rigid DOF count)."""
    nvert = int(model.nflexvert)
    nedge_eq = int(sum(model.flex_edgenum[model.eq_obj1id[e]] for e in range(model.neq) if model.eq_type[e] == mujoco.mjtEq.mjEQ_FLEX))
    rigid_nv = int(model.nv - 3 * nvert)
    ncon = int(contacts_per_vertex * nvert) + 16 + 4 * max(rigid_nv, 0)
    condim = int(max([3] + list(model.geom_condim) + list(model.flex_condim)))
    npyr = 2 * (condim - 1)
    njmax = nedge_eq + npyr * ncon + model.njnt + 16
    nnz = nedge_eq * 6 + npyr * ncon * max(6, rigid_nv + 3) + (model.njnt + 16) * max(1, rigid_nv)
    # CCD (GJK/EPA) slots: each carries ~4.8 KB of EPA polytope workspace at ccd_iterations 35, and MuJoCo Warp
    # defaults to one per contact slot (1.4 GB at 256 worlds of the G1 cloth scene: out of memory on Metal).
    # 128 per world covered a cloth draped on the G1's mesh colliders with no CCD overflow (300 steps, measured).
    return {"nconmax": ncon, "njmax": int(njmax), "njmax_nnz": int(nnz), "nccdmax": min(ncon, 128)}


# --------------------------------------------------------------------------------------------------
# Batched simulation
# --------------------------------------------------------------------------------------------------

class DeformableSim:
    """N worlds of a flex model on one device, stepped by graph replay (Metal) or eagerly."""

    def __init__(self, model: mujoco.MjModel, num_envs: int, device: str = "metal:0", substeps: int = 1,
                 capture: bool = True, keyframe: int | None = None, sizes: dict | None = None):
        self.mj_model = model
        self.n = num_envs
        self.substeps = substeps
        self.device = wp.get_device(device)
        self.is_metal = bool(getattr(self.device, "is_metal", False))
        mjd = mujoco.MjData(model)
        if keyframe is None and model.nkey:
            keyframe = 0
        if keyframe is not None:
            mujoco.mj_resetDataKeyframe(model, mjd, keyframe)
        mujoco.mj_forward(model, mjd)
        self.mjd0 = mjd
        self.sizes = sizes or default_sizes(model)
        with wp.ScopedDevice(self.device):
            self.m = mjw.put_model(model)
            self.m.opt.warn_overflow = 0
            if self.is_metal:
                self.m.opt.graph_conditional = False   # Metal: conditional graph nodes would read back per iteration
            self.d = mjw.put_data(model, mjd, nworld=num_envs, **self.sizes)
            self.graph = None
            if capture and (getattr(self.device, "is_metal", False) or self.device.is_cuda):
                self.capture()
        self._t = {}

    # -- stepping ---------------------------------------------------------------------------------

    def launch(self) -> None:
        for _ in range(self.substeps):
            mjw.step(self.m, self.d)

    def capture(self) -> None:
        with wp.ScopedDevice(self.device):
            with wp.ScopedCapture(device=self.device) as cap:
                self.launch()
            self.graph = cap.graph

    def step(self, eager: bool = False) -> None:
        with wp.ScopedDevice(self.device):
            if self.graph is not None and not eager:
                wp.capture_launch(self.graph)
            else:
                self.launch()

    def synchronize(self) -> None:
        wp.synchronize_device(self.device)

    # -- state ------------------------------------------------------------------------------------

    def tensor(self, name: str):
        """Zero-copy MPS tensor over a Data field (Metal only)."""
        t = self._t.get(name)
        if t is None:
            from metalsim.interop import torch_bridge as tb
            t = self._t[name] = tb.mps_tensor(getattr(self.d, name))
        return t

    def vertices(self):
        return self.tensor("flexvert_xpos")

    def energy(self):
        return self.tensor("energy")

    def nodal_vel(self, flex: int = 0):
        """(N, nvert, 3) zero-copy view of a full-dof flex's vertex velocities (its slide-joint qvel; the vertex
        bodies are axis-aligned, so these are world-frame velocities). Isaac's ``nodal_vel_w``."""
        m = self.mj_model
        if m.flex_interp[flex] != 0:
            raise ValueError("interpolated flex: vertex velocities are not DOFs")
        a, b = self.flex_qpos_slices()[flex]
        dof = m.jnt_dofadr[np.searchsorted(m.jnt_qposadr, a)]
        return self.tensor("qvel")[:, dof: dof + (b - a)].view(self.n, -1, 3)

    def nodal_pos(self, flex: int = 0):
        """(N, nvert, 3) world vertex positions of one flex (view into ``flexvert_xpos``). Isaac's ``nodal_pos_w``."""
        m = self.mj_model
        a = m.flex_vertadr[flex]
        return self.vertices()[:, a: a + m.flex_vertnum[flex]]

    def flex_qpos_slices(self) -> list[tuple[int, int]]:
        """qpos range of each flex's vertex slide joints (flexcomp full dof: 3 per vertex, contiguous)."""
        m = self.mj_model
        out = []
        for f in range(m.nflex):
            if m.flex_interp[f] != 0:   # interpolated: the DOFs are the nodes'
                bodies = m.flex_nodebodyid[m.flex_nodeadr[f]: m.flex_nodeadr[f] + m.flex_nodenum[f]]
            else:
                bodies = m.flex_vertbodyid[m.flex_vertadr[f]: m.flex_vertadr[f] + m.flex_vertnum[f]]
            jnts = np.concatenate([np.arange(m.body_jntadr[b], m.body_jntadr[b] + m.body_jntnum[b]) for b in bodies if b > 0])
            q = m.jnt_qposadr[jnts]
            assert np.all(np.diff(q) == 1), "flex slide joints are not contiguous"
            out.append((int(q[0]), int(q[-1]) + 1))
        return out

    def randomize(self, seed: int = 0, xy: float = 0.05, z: float = 0.02, worlds_equal_first: bool = True) -> np.ndarray:
        """Translate each flex by a per-world random offset (uniform in +-xy, +-xy, 0..z) on the host,
        then ``forward``. World 0 keeps the nominal state if ``worlds_equal_first``. Returns qpos (N, nq)."""
        rng = np.random.default_rng(seed)
        q = np.tile(self.mjd0.qpos, (self.n, 1))
        for a, b in self.flex_qpos_slices():
            off = np.stack([rng.uniform(-xy, xy, self.n), rng.uniform(-xy, xy, self.n), rng.uniform(0, z, self.n)], 1)
            if worlds_equal_first:
                off[0] = 0
            q[:, a:b] += np.tile(off, (1, (b - a) // 3))
        self.set_qpos(q)
        return q

    def set_qpos(self, qpos: np.ndarray, qvel: np.ndarray | None = None) -> None:
        self.synchronize()
        self.d.qpos.assign(np.asarray(qpos, dtype=np.float32).reshape(self.n, -1))
        self.d.qvel.assign(np.zeros((self.n, self.mj_model.nv), np.float32) if qvel is None else np.asarray(qvel, np.float32))
        with wp.ScopedDevice(self.device):
            mjw.forward(self.m, self.d)

    def get_world(self, world: int) -> mujoco.MjData:
        self.synchronize()
        mjd = mujoco.MjData(self.mj_model)
        with wp.ScopedDevice(self.device):
            mjw.get_data_into(mjd, self.mj_model, self.d, world_id=world)
        return mjd

    def overflow(self) -> dict[str, int]:
        self.synchronize()
        ov = self.d.overflow.numpy() if hasattr(self.d, "overflow") else np.zeros(1, np.int32)
        return {f.name: int(((ov & int(f)) != 0).sum()) for f in mjw.OverflowType
                if int(f) not in (0, int(mjw.OverflowType.ALL)) and ((ov & int(f)) != 0).any()}


# --------------------------------------------------------------------------------------------------
# References and measurements
# --------------------------------------------------------------------------------------------------

def mjc_rollout(model: mujoco.MjModel, qpos: np.ndarray, nsteps: int, keyframe: int | None = None) -> dict:
    """MuJoCo C (float64, CPU) from ``qpos`` at rest: per-step flex vertex positions, qpos and energy."""
    d = mujoco.MjData(model)
    if keyframe is None and model.nkey:
        keyframe = 0
    if keyframe is not None:
        mujoco.mj_resetDataKeyframe(model, d, keyframe)
    d.qpos[:] = qpos
    d.qvel[:] = 0
    mujoco.mj_forward(model, d)
    xs, qs, es, ncon = [], [], [], []
    for _ in range(nsteps):
        mujoco.mj_step(model, d)
        xs.append(d.flexvert_xpos.copy()); qs.append(d.qpos.copy()); es.append(d.energy.copy()); ncon.append(d.ncon)
    return {"vert": np.array(xs), "qpos": np.array(qs), "energy": np.array(es), "ncon": np.array(ncon)}


def benchmark(model: mujoco.MjModel, num_envs: int, seconds: float = 1.0, substeps: int = 1, warmup: int = 20,
              device: str = "metal:0") -> dict:
    """Physics steps per second (env-steps/s) by graph replay, synchronized at the end of each timed batch."""
    sim = DeformableSim(model, num_envs, device=device, substeps=substeps)
    sim.randomize(seed=1)
    for _ in range(warmup):
        sim.step()
    sim.synchronize()
    n = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        for _ in range(10):
            sim.step()
        sim.synchronize()
        n += 10
    dt = time.perf_counter() - t0
    e = sim.d.energy.numpy()
    return {"num_envs": num_envs, "steps": n * substeps, "wall_s": dt,
            "env_steps_per_s": num_envs * n * substeps / dt, "sim_s_per_wall_s": n * substeps * model.opt.timestep / dt,
            "finite": bool(np.isfinite(sim.d.qpos.numpy()).all()), "energy_max": float(np.nanmax(e.sum(1))),
            "overflow": sim.overflow(), "nv": model.nv, "nflexvert": model.nflexvert, **sim.sizes}


# ==================================================================================================
# MetalSim-native XPBD cloth / cable (the cheaper formulation)
# ==================================================================================================
#
# Same flex topology as the MuJoCo model (vertices, edges, triangle flaps, vertex masses, radius), solved
# with small-step XPBD (Macklin et al. 2019: one constraint iteration per substep, many substeps):
# distance constraints on every flex edge (stretch/shear) and on the two opposite vertices of every
# interior edge (bending), processed in graph colours (parallel Gauss-Seidel); vertex-vs-rigid-geom contacts
# (plane, sphere, capsule, box) with position-level Coulomb friction, against the geoms of a MuJoCo Warp
# ``Data`` (so rigid bodies that MuJoCo Warp steps move the obstacles). Optional two-way coupling adds
# the contact reaction to the touched body's ``xfrc_applied`` for MuJoCo Warp's next step.
# Every launch has a fixed size, so the whole step (rigid step + cloth substeps) is one captured graph.

_GEOM_PLANE, _GEOM_SPHERE, _GEOM_CAPSULE, _GEOM_BOX, _GEOM_MESH = 0, 2, 3, 6, 7


@wp.kernel
def _xpbd_predict(x: wp.array2d(dtype=wp.vec3), xp: wp.array2d(dtype=wp.vec3), v: wp.array2d(dtype=wp.vec3),
                  inv_mass: wp.array(dtype=float), gravity: wp.vec3, h: float, damping: float):
    w, i = wp.tid()
    xp[w, i] = x[w, i]
    if inv_mass[i] == 0.0:
        return
    vi = (v[w, i] + gravity * h) * (1.0 - damping * h)
    v[w, i] = vi
    x[w, i] = x[w, i] + vi * h


@wp.kernel
def _xpbd_distance(x: wp.array2d(dtype=wp.vec3), xp: wp.array2d(dtype=wp.vec3), inv_mass: wp.array(dtype=float),
                   ci: wp.array(dtype=int), cj: wp.array(dtype=int), rest: wp.array(dtype=float),
                   compliance: wp.array(dtype=float), damping: wp.array(dtype=float), start: int, h: float):
    # XPBD distance constraint with compliance alpha = 1/k and Rayleigh-style damping beta = d (Macklin et al. 2016,
    # eq. 26): gamma = alpha * d / h. With k, d per spring this is PhysX's particle-cloth spring (implicit spring
    # k, d solved per TGS substep, PhysX 5.6.1 particlesystem.cu ps_solveSpringsLaunch).
    w, k0 = wp.tid()
    k = start + k0
    i = ci[k]
    j = cj[k]
    wi = inv_mass[i]
    wj = inv_mass[j]
    wsum = wi + wj
    if wsum == 0.0:
        return
    d = x[w, i] - x[w, j]
    l = wp.length(d)
    if l < 1.0e-9:
        return
    n = d / l
    alpha = compliance[k] / (h * h)
    gamma = compliance[k] * damping[k] / h
    rel = wp.dot(n, (x[w, i] - xp[w, i]) - (x[w, j] - xp[w, j]))
    dl = -((l - rest[k]) + gamma * rel) / ((1.0 + gamma) * wsum + alpha)
    x[w, i] = x[w, i] + n * (dl * wi)
    x[w, j] = x[w, j] - n * (dl * wj)


@wp.kernel
def _xpbd_bend_mid(x: wp.array2d(dtype=wp.vec3), xp: wp.array2d(dtype=wp.vec3), inv_mass: wp.array(dtype=float),
                   bi: wp.array(dtype=int), bj: wp.array(dtype=int), bk: wp.array(dtype=int), rest: wp.array(dtype=wp.vec3),
                   compliance: wp.array(dtype=float), damping: wp.array(dtype=float), start: int, h: float):
    # rod bending as the vector constraint C = x_j - (x_i + x_k)/2 - C0 (linear in the bend angle: for segment l and
    # angle theta |C| = l theta / 2, so an angular stiffness EI/l maps to k = 4 EI / l^3), XPBD with damping as above
    w, t0 = wp.tid()
    t = start + t0
    i = bi[t]
    j = bj[t]
    k = bk[t]
    wi = inv_mass[i]
    wj = inv_mass[j]
    wk = inv_mass[k]
    wsum = wj + 0.25 * (wi + wk)
    if wsum == 0.0:
        return
    C = x[w, j] - 0.5 * (x[w, i] + x[w, k]) - rest[t]
    dC = (x[w, j] - xp[w, j]) - 0.5 * ((x[w, i] - xp[w, i]) + (x[w, k] - xp[w, k]))
    alpha = compliance[t] / (h * h)
    gamma = compliance[t] * damping[t] / h
    dl = -(C + gamma * dC) / ((1.0 + gamma) * wsum + alpha)
    x[w, j] = x[w, j] + dl * wj
    x[w, i] = x[w, i] - dl * (0.5 * wi)
    x[w, k] = x[w, k] - dl * (0.5 * wk)


@wp.func
def _geom_sdf(gtype: int, size: wp.vec3, p: wp.vec3):
    """Signed distance and outward normal of a geom in its local frame."""
    if gtype == 0:      # plane (z up)
        return p[2], wp.vec3(0.0, 0.0, 1.0)
    if gtype == 2:      # sphere
        l = wp.length(p)
        n = wp.vec3(0.0, 0.0, 1.0)
        if l > 1.0e-9:
            n = p / l
        return l - size[0], n
    if gtype == 3:      # capsule along z
        c = wp.vec3(0.0, 0.0, wp.clamp(p[2], -size[1], size[1]))
        r = p - c
        l = wp.length(r)
        n = wp.vec3(1.0, 0.0, 0.0)
        if l > 1.0e-9:
            n = r / l
        return l - size[0], n
    # box
    q = wp.vec3(wp.abs(p[0]) - size[0], wp.abs(p[1]) - size[1], wp.abs(p[2]) - size[2])
    qo = wp.vec3(wp.max(q[0], 0.0), wp.max(q[1], 0.0), wp.max(q[2], 0.0))
    lo = wp.length(qo)
    inside = wp.min(wp.max(q[0], wp.max(q[1], q[2])), 0.0)
    if lo > 0.0:
        n = wp.vec3(wp.sign(p[0]) * qo[0], wp.sign(p[1]) * qo[1], wp.sign(p[2]) * qo[2]) / lo
        return lo, n
    # inside: push out through the nearest face
    n = wp.vec3(0.0, 0.0, wp.sign(p[2]))
    if q[0] >= q[1] and q[0] >= q[2]:
        n = wp.vec3(wp.sign(p[0]), 0.0, 0.0)
    elif q[1] >= q[2]:
        n = wp.vec3(0.0, wp.sign(p[1]), 0.0)
    return inside, n


@wp.kernel
def _xpbd_collide(x: wp.array2d(dtype=wp.vec3), xp: wp.array2d(dtype=wp.vec3), inv_mass: wp.array(dtype=float),
                  mass: wp.array(dtype=float), radius: wp.array(dtype=float), mu_vert: wp.array(dtype=float),
                  geom_ids: wp.array(dtype=int), geom_type: wp.array(dtype=int), geom_size: wp.array2d(dtype=wp.vec3),
                  geom_off: wp.array(dtype=wp.vec3),
                  geom_friction: wp.array2d(dtype=wp.vec3), geom_bodyid: wp.array(dtype=int),
                  geom_xpos: wp.array2d(dtype=wp.vec3), geom_xmat: wp.array2d(dtype=wp.mat33),
                  xipos: wp.array2d(dtype=wp.vec3), h: float, two_way: int, force_scale: float,
                  xfrc: wp.array2d(dtype=wp.spatial_vector)):
    w, i = wp.tid()
    if inv_mass[i] == 0.0:
        return
    p = x[w, i]
    r = radius[i]
    for gk in range(geom_ids.shape[0]):
        g = geom_ids[gk]
        gt = geom_type[g]
        R = geom_xmat[w, g]
        c = geom_xpos[w, g]
        pl = wp.transpose(R) * (p - c) - geom_off[g]
        dist, nl = _geom_sdf(gt, geom_size[w % geom_size.shape[0], g], pl)
        pen = r - dist
        if pen > 0.0:
            n = R * nl
            dx = n * pen
            # friction: cancel tangential motion over the substep up to mu * normal correction (static obstacle frame)
            mv = (p + dx) - xp[w, i]
            mt = mv - n * wp.dot(mv, n)
            lt = wp.length(mt)
            mu = wp.max(mu_vert[i], geom_friction[w % geom_friction.shape[0], g][0])
            if lt > 0.0:
                if lt < mu * pen:
                    dx = dx - mt
                else:
                    dx = dx - mt * (mu * pen / lt)
            p = p + dx
            if two_way != 0:
                b = geom_bodyid[g]
                if b > 0:
                    f = -dx * (mass[i] / (h * h)) * force_scale
                    tq = wp.cross(p - xipos[w, b], f)
                    wp.atomic_add(xfrc, w, b, wp.spatial_vector(f, tq))
    x[w, i] = p


@wp.kernel
def _xpbd_velocity(x: wp.array2d(dtype=wp.vec3), xp: wp.array2d(dtype=wp.vec3), v: wp.array2d(dtype=wp.vec3),
                   inv_mass: wp.array(dtype=float), h: float):
    w, i = wp.tid()
    if inv_mass[i] == 0.0:
        v[w, i] = wp.vec3(0.0)
        return
    v[w, i] = (x[w, i] - xp[w, i]) / h


@wp.kernel
def _xpbd_energy(x: wp.array2d(dtype=wp.vec3), v: wp.array2d(dtype=wp.vec3), mass: wp.array(dtype=float),
                 gravity: wp.vec3, energy: wp.array2d(dtype=float)):
    w, i = wp.tid()
    m = mass[i]
    wp.atomic_add(energy, w, 0, -m * wp.dot(gravity, x[w, i]))
    wp.atomic_add(energy, w, 1, 0.5 * m * wp.dot(v[w, i], v[w, i]))


@wp.kernel
def _zero_xfrc(xfrc: wp.array2d(dtype=wp.spatial_vector)):
    w, b = wp.tid()
    xfrc[w, b] = wp.spatial_vector()


def _color_edges(edges: np.ndarray, nvert: int) -> tuple[np.ndarray, list[int]]:
    """Greedy graph colouring of constraints (no two in a colour share a vertex). Returns the order and the
    colour boundaries."""
    colors = np.full(len(edges), -1)
    used: list[set] = [set() for _ in range(nvert)]
    for k, (i, j) in enumerate(edges):
        c = 0
        while c in used[i] or c in used[j]:
            c += 1
        colors[k] = c
        used[i].add(c); used[j].add(c)
    order = np.argsort(colors, kind="stable")
    bounds = [0] + list(np.cumsum(np.bincount(colors)))
    return order, bounds


@dataclass
class XPBDCfg:
    substeps: int = 10                 # per rigid (MuJoCo) step
    stretch_compliance: float = 1e-7   # m/N  (inverse stiffness of an edge)
    bend_compliance: float = 1e-3      # m/N  (cross-edge distance; 0 disables bending constraints if None)
    stretch_damping: float = 0.0       # N s/m per edge constraint (XPBD damping; PhysX spring_damping)
    bend_damping: float = 0.0          # N s/m per bending constraint
    damping: float = 0.5               # 1/s, velocity damping
    friction: float = 0.8
    two_way: bool = False              # add contact reactions to MuJoCo Warp bodies' xfrc_applied
    bending: bool = True
    rope_bending: str = "distance"     # 1D flexes: "distance" (i, i+2 distance constraints) or "midpoint" (rod bending,
                                       # linear in the bend angle, k = 4 EI / l^3)


class XPBDSim:
    """Flexes of a MuJoCo model simulated by MetalSim's XPBD kernels; the rigid part by MuJoCo Warp.

    ``flex_model``: the full model (flexes included) that defines the cloth/cable topology, masses and
    rest shape. ``rigid_model``: the same scene without the flexes (``None``: the rigid scene is static and
    is taken from ``flex_model``'s geoms at its initial pose, no MuJoCo Warp step is run)."""

    def __init__(self, flex_model: mujoco.MjModel, num_envs: int, rigid_model: mujoco.MjModel | None = None,
                 device: str = "metal:0", cfg: XPBDCfg | None = None, capture: bool = True, keyframe: int | None = None):
        self.cfg = cfg = cfg or XPBDCfg()
        self.n = num_envs
        self.device = wp.get_device(device)
        fm = flex_model
        self.flex_model = fm
        fd = mujoco.MjData(fm)
        mujoco.mj_forward(fm, fd)
        nvert = int(fm.nflexvert)
        self.nvert = nvert
        body = fm.flex_vertbodyid
        mass = fm.body_mass[body].astype(np.float32)
        has_dof = np.array([fm.body_dofnum[b] > 0 for b in body])
        inv_mass = np.where(has_dof & (mass > 0), 1.0 / np.maximum(mass, 1e-12), 0.0).astype(np.float32)
        rad = np.concatenate([np.full(fm.flex_vertnum[f], fm.flex_radius[f]) for f in range(fm.nflex)]).astype(np.float32)
        mu = np.concatenate([np.full(fm.flex_vertnum[f], fm.flex_friction[f, 0]) for f in range(fm.nflex)]).astype(np.float32)
        mu = np.maximum(mu, cfg.friction)
        x0 = fd.flexvert_xpos.astype(np.float32).copy()
        # constraints: all edges (+ cross-edge bending pairs for 2D flexes, i/i+2 pairs for 1D)
        ci, cj, rest, comp, damp = [], [], [], [], []
        trip_i, trip_j, trip_k, trip_rest, trip_comp, trip_damp = [], [], [], [], [], []
        for f in range(fm.nflex):
            va, ea, en = fm.flex_vertadr[f], fm.flex_edgeadr[f], fm.flex_edgenum[f]
            e = fm.flex_edge[ea:ea + en] + va
            ci += list(e[:, 0]); cj += list(e[:, 1])
            rest += list(np.linalg.norm(x0[e[:, 0]] - x0[e[:, 1]], axis=1)); comp += [cfg.stretch_compliance] * en
            damp += [cfg.stretch_damping] * en
            if not cfg.bending:
                continue
            if fm.flex_dim[f] == 2:
                flap = fm.flex_edgeflap[ea:ea + en]
                ok = flap[:, 1] >= 0
                bi, bj = flap[ok, 0] + va, flap[ok, 1] + va
            elif fm.flex_dim[f] == 1 and cfg.rope_bending == "midpoint":
                nb = fm.flex_vertnum[f]
                ti = np.arange(nb - 2) + va
                trip_i += list(ti); trip_j += list(ti + 1); trip_k += list(ti + 2)
                trip_rest += list(x0[ti + 1] - 0.5 * (x0[ti] + x0[ti + 2]))
                trip_comp += [cfg.bend_compliance] * len(ti); trip_damp += [cfg.bend_damping] * len(ti)
                continue
            elif fm.flex_dim[f] == 1:
                nb = fm.flex_vertnum[f]
                bi, bj = np.arange(nb - 2) + va, np.arange(2, nb) + va
            else:
                continue
            ci += list(bi); cj += list(bj)
            rest += list(np.linalg.norm(x0[bi] - x0[bj], axis=1)); comp += [cfg.bend_compliance] * len(bi)
            damp += [cfg.bend_damping] * len(bi)
        edges = np.stack([ci, cj], 1).astype(np.int32)
        order, self.color_bounds = _color_edges(edges, nvert)
        self.ncons = len(edges)
        with wp.ScopedDevice(self.device):
            self.inv_mass = wp.array(inv_mass, dtype=float)
            self.mass = wp.array(mass, dtype=float)
            self.radius = wp.array(rad, dtype=float)
            self.mu = wp.array(mu, dtype=float)
            self.ci = wp.array(edges[order, 0], dtype=int)
            self.cj = wp.array(edges[order, 1], dtype=int)
            self.rest = wp.array(np.asarray(rest, np.float32)[order], dtype=float)
            self.comp = wp.array(np.asarray(comp, np.float32)[order], dtype=float)
            self.damp = wp.array(np.asarray(damp, np.float32)[order], dtype=float)
            # rod-bending triplets, coloured so no two in a colour share a vertex
            self.ntrip = len(trip_i)
            if self.ntrip:
                tcol = np.full(self.ntrip, -1)
                used = [set() for _ in range(nvert)]
                for t, (a, b_, c) in enumerate(zip(trip_i, trip_j, trip_k)):
                    col = 0
                    while col in used[a] or col in used[b_] or col in used[c]:
                        col += 1
                    tcol[t] = col
                    used[a].add(col); used[b_].add(col); used[c].add(col)
                to = np.argsort(tcol, kind="stable")
                self.trip_bounds = [0] + list(np.cumsum(np.bincount(tcol)))
                self.ti = wp.array(np.asarray(trip_i, np.int32)[to], dtype=int)
                self.tj = wp.array(np.asarray(trip_j, np.int32)[to], dtype=int)
                self.tk = wp.array(np.asarray(trip_k, np.int32)[to], dtype=int)
                self.trest = wp.array(np.asarray(trip_rest, np.float32)[to], dtype=wp.vec3)
                self.tcomp = wp.array(np.asarray(trip_comp, np.float32)[to], dtype=float)
                self.tdamp = wp.array(np.asarray(trip_damp, np.float32)[to], dtype=float)
            self.x0 = x0
            self.x = wp.array(np.tile(x0, (num_envs, 1, 1)), dtype=wp.vec3)
            self.xp = wp.zeros_like(self.x)
            self.v = wp.zeros_like(self.x)
            self.energy_arr = wp.zeros((num_envs, 2), dtype=float)
            # rigid side
            self.rigid_model = rigid_model
            rm = rigid_model if rigid_model is not None else fm
            rd = mujoco.MjData(rm)
            if keyframe is None and rm.nkey:
                keyframe = 0
            if keyframe is not None and rm.nkey:
                mujoco.mj_resetDataKeyframe(rm, rd, keyframe)
            mujoco.mj_forward(rm, rd)
            self.m = mjw.put_model(rm)
            self.m.opt.warn_overflow = 0
            if getattr(self.device, "is_metal", False):
                self.m.opt.graph_conditional = False
            self.d = mjw.put_data(rm, rd, nworld=num_envs)
            # colliders: plane, sphere, capsule, box exactly; a mesh by its oriented bounding box (geom_aabb)
            gids = [g for g in range(rm.ngeom) if rm.geom_type[g] in (_GEOM_PLANE, _GEOM_SPHERE, _GEOM_CAPSULE, _GEOM_BOX, _GEOM_MESH)
                    and (rm.geom_contype[g] or rm.geom_conaffinity[g])]
            self.skipped_geoms = [g for g in range(rm.ngeom) if g not in gids and (rm.geom_contype[g] or rm.geom_conaffinity[g])]
            gtype = rm.geom_type.astype(np.int32).copy()
            gsize = rm.geom_size.astype(np.float32).copy()
            goff = np.zeros((rm.ngeom, 3), np.float32)
            mesh = gtype == _GEOM_MESH
            gtype[mesh] = _GEOM_BOX
            gsize[mesh] = rm.geom_aabb[mesh, 3:6]
            goff[mesh] = rm.geom_aabb[mesh, 0:3]
            self.geom_ids = wp.array(np.array(gids, np.int32), dtype=int)
            self.geom_type_x = wp.array(gtype, dtype=int)
            self.geom_off = wp.array(goff, dtype=wp.vec3)
            self.geom_size3 = wp.array(gsize[None], dtype=wp.vec3)
            self.geom_friction3 = wp.array(rm.geom_friction.astype(np.float32)[None], dtype=wp.vec3)
            self.step_rigid = rigid_model is not None and rigid_model.nv > 0
            self.gravity = wp.vec3(*[float(g) for g in fm.opt.gravity])
            self.dt = float(rm.opt.timestep if rigid_model is not None else fm.opt.timestep)
            self.graph = None
            if capture and (getattr(self.device, "is_metal", False) or self.device.is_cuda):
                with wp.ScopedCapture(device=self.device) as cap:
                    self.launch()
                self.graph = cap.graph
        self._t = {}

    def launch(self) -> None:
        cfg = self.cfg
        h = self.dt / cfg.substeps
        n, nv = self.n, self.nvert
        if self.step_rigid:
            mjw.step(self.m, self.d)          # rigid bodies see the reaction accumulated during the previous step
            if cfg.two_way:
                wp.launch(_zero_xfrc, dim=(n, self.d.xfrc_applied.shape[1]), inputs=[self.d.xfrc_applied])
        for _ in range(cfg.substeps):
            wp.launch(_xpbd_predict, dim=(n, nv), inputs=[self.x, self.xp, self.v, self.inv_mass, self.gravity, h, cfg.damping])
            for c in range(len(self.color_bounds) - 1):
                a, b = self.color_bounds[c], self.color_bounds[c + 1]
                wp.launch(_xpbd_distance, dim=(n, b - a),
                          inputs=[self.x, self.xp, self.inv_mass, self.ci, self.cj, self.rest, self.comp, self.damp, a, h])
            if self.ntrip:
                for c in range(len(self.trip_bounds) - 1):
                    a, b = self.trip_bounds[c], self.trip_bounds[c + 1]
                    wp.launch(_xpbd_bend_mid, dim=(n, b - a),
                              inputs=[self.x, self.xp, self.inv_mass, self.ti, self.tj, self.tk, self.trest, self.tcomp,
                                      self.tdamp, a, h])
            wp.launch(_xpbd_collide, dim=(n, nv),
                      inputs=[self.x, self.xp, self.inv_mass, self.mass, self.radius, self.mu, self.geom_ids, self.geom_type_x,
                              self.geom_size3, self.geom_off, self.geom_friction3, self.m.geom_bodyid, self.d.geom_xpos, self.d.geom_xmat,
                              self.d.xipos, h, int(cfg.two_way and self.step_rigid), 1.0 / cfg.substeps, self.d.xfrc_applied])
            wp.launch(_xpbd_velocity, dim=(n, nv), inputs=[self.x, self.xp, self.v, self.inv_mass, h])
        self.energy_arr.zero_()
        wp.launch(_xpbd_energy, dim=(n, nv), inputs=[self.x, self.v, self.mass, self.gravity, self.energy_arr])

    def step(self, eager: bool = False) -> None:
        with wp.ScopedDevice(self.device):
            if self.graph is not None and not eager:
                wp.capture_launch(self.graph)
            else:
                self.launch()

    def synchronize(self) -> None:
        wp.synchronize_device(self.device)

    def set_vertices(self, x: np.ndarray, v: np.ndarray | None = None) -> None:
        self.synchronize()
        self.x.assign(np.asarray(x, np.float32).reshape(self.n, self.nvert, 3))
        self.v.assign(np.zeros((self.n, self.nvert, 3), np.float32) if v is None else np.asarray(v, np.float32))

    def randomize(self, seed: int = 0, xy: float = 0.05, z: float = 0.02, worlds_equal_first: bool = True) -> np.ndarray:
        """Per-world translation of each flex (same distribution as ``DeformableSim.randomize``)."""
        rng = np.random.default_rng(seed)
        fm = self.flex_model
        x = np.tile(self.x0, (self.n, 1, 1))
        for f in range(fm.nflex):
            a, b = fm.flex_vertadr[f], fm.flex_vertadr[f] + fm.flex_vertnum[f]
            off = np.stack([rng.uniform(-xy, xy, self.n), rng.uniform(-xy, xy, self.n), rng.uniform(0, z, self.n)], 1)
            if worlds_equal_first:
                off[0] = 0
            x[:, a:b] += off[:, None, :]
        self.set_vertices(x)
        return x

    def tensor(self, name: str):
        t = self._t.get(name)
        if t is None:
            from metalsim.interop import torch_bridge as tb
            t = self._t[name] = tb.mps_tensor(getattr(self, name))
        return t

    def vertices(self):
        return self.tensor("x")

    def energy(self):
        return self.tensor("energy_arr")

    def edge_strain(self, world: int = 0) -> np.ndarray:
        """Relative length error of the stretch constraints (the flex edges) in one world."""
        self.synchronize()
        x = self.x.numpy()[world]
        i, j, r, c = self.ci.numpy(), self.cj.numpy(), self.rest.numpy(), self.comp.numpy()
        s = c == self.cfg.stretch_compliance
        return (np.linalg.norm(x[i[s]] - x[j[s]], axis=1) - r[s]) / r[s]


def rigid_scene_xml(**kw) -> str:
    """The box scene without its flexes (the obstacles XPBD collides with)."""
    return box_scene_xml(cloth=None, cable=None, **kw)


def benchmark_xpbd(flex_model: mujoco.MjModel, num_envs: int, seconds: float = 1.0, cfg: XPBDCfg | None = None,
                   rigid_model: mujoco.MjModel | None = None, warmup: int = 20, device: str = "metal:0") -> dict:
    sim = XPBDSim(flex_model, num_envs, rigid_model=rigid_model, device=device, cfg=cfg)
    sim.randomize(seed=1)
    for _ in range(warmup):
        sim.step()
    sim.synchronize()
    n = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        for _ in range(10):
            sim.step()
        sim.synchronize()
        n += 10
    dt = time.perf_counter() - t0
    return {"num_envs": num_envs, "steps": n, "wall_s": dt, "env_steps_per_s": num_envs * n / dt,
            "sim_s_per_wall_s": n * sim.dt / dt, "finite": bool(np.isfinite(sim.x.numpy()).all()),
            "substeps": sim.cfg.substeps, "ncons": sim.ncons, "ncolors": len(sim.color_bounds) - 1, "nflexvert": sim.nvert}


def _main():
    import argparse, json
    ap = argparse.ArgumentParser(description="Deformable throughput on Metal (run through scripts/gpu_run.sh)")
    ap.add_argument("--backend", choices=["mjw", "xpbd"], default="mjw")
    ap.add_argument("--scene", choices=["box", "g1"], default="box")
    ap.add_argument("--envs", type=int, nargs="+", default=[64, 256, 1024, 4096])
    ap.add_argument("--cloth", type=int, default=10, help="cloth vertices per side")
    ap.add_argument("--cable", type=int, default=12)
    ap.add_argument("--substeps", type=int, default=10, help="XPBD substeps per step")
    ap.add_argument("--seconds", type=float, default=2.0)
    ap.add_argument("--flex-flags", default="", help="MuJoCo Warp collision_flex module flags, e.g. FLEX_DEVICE_SORT=0,FLEX_FPS_MODE=main")
    a = ap.parse_args()
    wp.config.quiet = True
    if a.flex_flags:
        from mujoco_warp._src import collision_flex as cf
        for kv in a.flex_flags.split(","):
            k, v = kv.split("=")
            cur = getattr(cf, k)
            setattr(cf, k, v if isinstance(cur, str) else bool(int(v)) if isinstance(cur, bool) or cur is None else type(cur)(v))
    if a.scene == "box":
        fm = scene_model("box", cloth=ClothCfg(count=a.cloth), cable=CableCfg(count=a.cable))
        rm = None
    else:
        fm = g1_scene_model(ClothCfg(count=a.cloth, spacing=0.05, pos=(0.0, 0.0, 1.55), mass=0.3),
                            CableCfg(count=a.cable, spacing=0.04, pos=(0.35, -0.3, 1.2)))
        rm = g1_scene_model(flexes=False)
    for n in a.envs:
        if a.backend == "mjw":
            r = benchmark(fm, n, seconds=a.seconds)
        else:
            r = benchmark_xpbd(fm, n, seconds=a.seconds, cfg=XPBDCfg(substeps=a.substeps, two_way=rm is not None), rigid_model=rm)
        r.update(backend=a.backend, scene=a.scene, cloth=a.cloth, cable=a.cable, dt=float(fm.opt.timestep if rm is None else rm.opt.timestep),
                 flex_flags=a.flex_flags)
        print(json.dumps(r), flush=True)


if __name__ == "__main__":
    _main()

"""Isaac Lab 3.0-EA's ``IsaacContrib-Franka-Pour`` scene and physics on Newton 1.5.2 (metalsim fork), any Warp device.

Everything here mirrors what Isaac Lab builds for the task; the source of each value is cited as
``<file>:<line>`` relative to ``upstream/IsaacLab3/source/`` (``IL``) or ``upstream/newton-1.5.2/newton/_src/`` (``NW``),
task files relative to ``IL/isaaclab_tasks/isaaclab_tasks/contrib/franka_pour/``.

Scene (``pour_env_cfg.py`` ``PourSceneCfg``, ``pour_env.py`` ``_add_pour_world_to_builder``), per world:
  * Franka: Isaac's ``Robots/FrankaEmika/franka_panda.usda`` (Isaac 6.1 asset root; converted from MuJoCo Menagerie's
    ``mjx_panda``), composed as the task's spawn hook intends (see ``compose_robot_stage``), imported with Newton's USD
    importer and Isaac Lab's schema resolvers (Newton, PhysX, MuJoCo; ``IL/isaaclab_newton/.../newton_manager.py``
    ``_get_usd_import_schema_resolvers``); actuator overrides of ``PourSceneCfg.robot`` and ``FRANKA_PANDA_CFG``.
  * SeattleLab table (``Props/Mounts/SeattleLabTable/table_instanceable.usd``), kinematic, at (0.5, 0, 0) rotated 90 deg.
  * source cup (``Contrib/MPM/Pour/franka_pour_cups.usda`` variant ``source``), dynamic, 50 g, friction 2.0;
    receiver (variant ``target``), kinematic, friction 0.8.
  * the task's per-world builder hook, verbatim in effect: robot shapes do not collide with particles, finger and
    grasp-proxy contact gains (ke 1e5, kd 500, kf 1e3) in MuJoCo's force-space solref mode, bowls as particle-only
    colliders (margin 4 mm, mu 0.9 / 0.8), the receiver's rigid mesh replaced by an analytic box, the tabletop collider
    lowered by 3 mm with zero margin, a kinematic particle-only spill floor; then the media (``media.py``): a 7 x 7 x 15
    cell-centred lattice (735 particles, 5 mm, jitter 0.3 x spacing) emitted with Newton's ``add_particle_grid`` exactly
    as ``IL/isaaclab_newton/.../sim/spawners/mpm/mpm.py`` ``_emit_grid`` does.
  * a global ground plane at z = -1.05 (``/World/GroundPlane``).

Physics (``pour_env_cfg.py:597-677``, research notes in ``docs/research/franka_pour_2026-10-11.md``): sim dt 1/120,
decimation 4, ``NewtonCfg(num_substeps=2)`` around a ``SolverCoupledProxy`` with an ``arm`` entry
(``SolverMuJoCo(use_mujoco_contacts=False, integrator="implicitfast", njmax=2560, nconmax=400)`` and Isaac's
``MJWarpSolverCfg`` defaults, 3 substeps) and a ``media`` entry (``SolverImplicitMPM``: voxel 0.015, sparse grid,
P0 / APIC / Jacobi 24 / pic27 / separate worlds, 2 substeps, in place), one proxy mapping (both cups into MPM, mass
scale 1000, staggered). One ``sim.step`` = one ``CollisionPipeline.collide`` + 2 coupled steps
(``IL/isaaclab_newton/.../physics/newton_manager.py`` ``_simulate_physics_only``); a policy step = 4 of them, each
preceded by the action terms' ``apply_actions`` (the relative arm action re-reads the joint positions every physics
step). ``PourPhysics.capture_policy_step`` records all of it as one graph.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import numpy as np
import warp as wp

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ASSET_DIR = os.path.join(ROOT, "assets", "isaac", "franka_pour")
FRANKA_USD = os.path.join(ASSET_DIR, "franka", "franka_panda.usda")
CUPS_USD = os.path.join(ASSET_DIR, "franka_pour_cups.usda")
TABLE_USD = os.path.join(ASSET_DIR, "table", "table_instanceable.usd")
RESET_DATASET = os.path.join(ASSET_DIR, "reset_dataset.pt")
FRANKA_PHYSICS_PAYLOAD_SHA256 = "0dc38454f02ea14d9ddd2437995fdc7c4a65634443cacdcc2a04e3de25655e00"  # pour_env_cfg.py:62
RESET_DATASET_CONTENT_SHA256 = "74057f08b1aec203ef1ab4259fea0f1404d580f75a2a15a716a878199282c7b2"  # published 6.1 artifact

# ------------------------------------------------------------------------------------------------ task constants
ARM_JOINTS = tuple(f"panda_joint{i}" for i in range(1, 8))
FINGER_JOINTS = ("panda_finger_joint1", "panda_finger_joint2")
ARM_HOME = (-1.07505691, 0.76868522, 0.53213346, -2.93226814, 2.50670838, 1.40047050, 0.21146376)  # cfg:82-90
GRIPPER_OPEN_POSITION = 0.04                                   # pour_env_cfg.py:91
SOURCE_CUP_POSITION = (0.5, 0.0, 0.0)                          # :92
TARGET_CUP_POSITION = (0.5, -0.18, 0.0)                        # :93
SOURCE_CUP_MASS = 0.05                                         # :94
SOURCE_CUP_GRASP_FRICTION = 2.0                                # :95
TARGET_CUP_FRICTION = 0.8                                      # :96
MEDIA_FILL_LEVEL = 0.70                                        # :97
MEDIA_FILL_RESOLUTION = 0.005                                  # :98
MPM_VOXEL_SIZE = 0.015                                         # :99
MPM_PARTICLES_PER_CELL = 3.0                                   # :100
MEDIA_DENSITY, MEDIA_FRICTION, MEDIA_YIELD_PRESSURE = 1500.0, 0.7, 1.0e12   # :152-156
MEDIA_JITTER_FRACTION = 0.3                                    # media.py:17
# FRANKA_POUR_ARM_DRIVE_STIFFNESS / _DAMPING (pour_env_cfg.py:66-79), per arm joint 1..7
ARM_STIFFNESS = (1000.0, 1000.0, 750.0, 750.0, 300.0, 300.0, 300.0)
ARM_DAMPING = (20.0, 20.0, 4.0, 4.0, 2.0, 2.0, 2.0)
# FRANKA_PANDA_CFG.actuators joint_effort_limit (IL/isaaclab_assets/.../robots/franka.py:55,62,69): the task's
# ``replace(effort_limit_sim=None, ...)`` clears only the deprecated alias, so these stay in force
ARM_EFFORT_LIMIT = (87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0)
FINGER_EFFORT_LIMIT = 200.0
ARM_COLLISION_PROXIES = ("link0_c", "link1_c", "link2_c", "link3_c", "link4_c", "link5_c0", "link5_c1", "link5_c2",
                         "link6_c", "link7_c")             # pour_env_cfg.py:161-174
# FrankaPourResetDatasetEnvCfg fields (pour_env_cfg.py:548-577)
TCP_BODY = "panda_hand"
TCP_OFFSET_POS = (0.0, 0.0, 0.107)
GRASP_CONTACT_KE, GRASP_CONTACT_KD, GRASP_CONTACT_KF = 1.0e5, 5.0e2, 1.0e3
COLLIDER_MARGIN = 0.002
PARTICLE_MAX_VELOCITY = 10.0
MPM_CELL_CAPACITY_ALIGNMENT = 512
# pour_env.py
SOURCE_CUP_PARTICLE_FRICTION = 0.9                             # :67
SEATTLELAB_TABLETOP_COLLISION_INSET = 0.003                    # :62
MJWARP_SOLREF_MODE_FORCE_SPACE = 0                             # :66
# geometry.py
MPM_COLLIDER_MARGIN = 0.004                                    # :70


@dataclass(frozen=True)
class CupGeometry:
    """geometry.py ``CupGeometry``: dimensions of one base-origin rectangular cup [m]."""

    inner_width: float
    inner_depth: float
    cavity_depth: float
    wall_thickness: float
    bottom_thickness: float

    @property
    def rim_height(self) -> float:
        return self.bottom_thickness + self.cavity_depth

    @property
    def outer_half_extents(self) -> tuple[float, float, float]:
        return (0.5 * self.inner_width + self.wall_thickness, 0.5 * self.inner_depth + self.wall_thickness,
                0.5 * self.rim_height)

    @property
    def inner_bounds(self):
        return ((-0.5 * self.inner_width, -0.5 * self.inner_depth, self.bottom_thickness),
                (0.5 * self.inner_width, 0.5 * self.inner_depth, self.rim_height))


SOURCE_CUP_GEOMETRY = CupGeometry(0.042, 0.042, 0.110, 0.007, 0.009)    # geometry.py:66
TARGET_CUP_GEOMETRY = CupGeometry(0.140, 0.140, 0.065, 0.009, 0.009)    # geometry.py:67
CUP_GRASP_HEIGHT = 0.083                                                # geometry.py:68
CUP_GRASP_TCP_QUAT_C = (0.0, 2.0 ** -0.5, 0.0, 2.0 ** -0.5)              # geometry.py:69

# timing (pour_env_cfg.py:597-601, 629, 654, 676)
SIM_DT = 1.0 / 120.0
DECIMATION = 4
NUM_SUBSTEPS = 2
ARM_SUBSTEPS = 3
MEDIA_SUBSTEPS = 2
EPISODE_LENGTH_S = 12.0
ENV_SPACING = 2.5                                              # PourSceneCfg(env_spacing=2.5)


def media_grid_bounds(fill_level: float = MEDIA_FILL_LEVEL):
    """media.py ``_media_grid_bounds`` for the source cup (verbatim arithmetic)."""
    g = SOURCE_CUP_GEOMETRY
    level = float(fill_level)
    if not math.isfinite(level) or not 0.0 < level <= 1.0:
        raise ValueError(f"fill_level must lie in (0, 1], got {fill_level}")
    spacing = float(MEDIA_FILL_RESOLUTION)
    clearance = max(spacing, float(MPM_COLLIDER_MARGIN))

    def horizontal_axis(size):
        usable = float(size) - 2.0 * clearance
        intervals = max(0, math.floor(usable / spacing + 1.0e-9))
        count = intervals + 1
        first = -0.5 * float(size) + clearance + 0.5 * (usable - intervals * spacing)
        lower = first - 0.5 * spacing
        upper = float(np.nextafter(np.float32(lower + count * spacing), np.float32(lower)))
        return lower, upper

    lx, ux = horizontal_axis(g.inner_width)
    ly, uy = horizontal_axis(g.inner_depth)
    max_count_z = max(1, math.floor((float(g.cavity_depth) - 2.0 * clearance) / spacing) + 1)
    count_z = max(1, min(max_count_z, math.floor(level * max_count_z + 0.5)))
    first_z = float(g.bottom_thickness) + clearance
    lz = first_z - 0.5 * spacing
    uz = float(np.nextafter(np.float32(lz + count_z * spacing), np.float32(lz)))
    return (lx, ly, lz), (ux, uy, uz)


def media_lattice(fill_level: float = MEDIA_FILL_LEVEL):
    """``_emit_grid`` (IL/isaaclab_newton/isaaclab_newton/sim/spawners/mpm/mpm.py:120-174) for the task's MPMGridCfg:
    returns (first particle in the cup frame, dims, cell size, particle mass, radius, jitter), all from float32 corners."""
    lo, hi = media_grid_bounds(fill_level)
    lower, upper = np.asarray(lo, np.float32), np.asarray(hi, np.float32)
    extent = upper - lower
    resolution = np.maximum(np.ceil(MPM_PARTICLES_PER_CELL * extent / MPM_VOXEL_SIZE), 1).astype(np.int32)
    cell = extent / resolution
    cell_volume = float(np.prod(cell))
    mass = cell_volume * MEDIA_DENSITY
    first = lower + 0.5 * cell
    radius = 0.5 * float(np.cbrt(cell_volume))
    jitter = MEDIA_JITTER_FRACTION * (MPM_VOXEL_SIZE / MPM_PARTICLES_PER_CELL)
    return first, resolution, cell, mass, radius, jitter


def media_particle_count(fill_level: float = MEDIA_FILL_LEVEL) -> int:
    return int(np.prod(media_lattice(fill_level)[1]))


def media_material_attributes() -> dict[str, float]:
    """MPMParticleMaterialCfg(density=1500, friction=0.7, yield_pressure=1e12) with Isaac Lab's other defaults
    (IL/isaaclab_newton/.../sim/spawners/mpm/mpm_cfg.py:29-59), as ``mpm.py`` ``_material_custom_attributes`` maps them."""
    return {"mpm:young_modulus": 1.0e15, "mpm:poisson_ratio": 0.3, "mpm:viscosity": 0.0,
            "mpm:friction": MEDIA_FRICTION, "mpm:damping": 0.0, "mpm:yield_pressure": MEDIA_YIELD_PRESSURE,
            "mpm:tensile_yield_ratio": 0.0, "mpm:yield_stress": 0.0, "mpm:hardening": 0.0, "mpm:dilatancy": 0.0}


def env_origins(num_envs: int, spacing: float = ENV_SPACING) -> np.ndarray:
    """IL/isaaclab/isaaclab/cloner/clone_plan.py ``grid_transforms`` (the terrain importer's centred grid)."""
    num_rows = int(math.ceil(num_envs / math.sqrt(num_envs)))
    num_cols = int(math.ceil(num_envs / num_rows))
    ii, jj = np.meshgrid(np.arange(num_rows, dtype=np.float32), np.arange(num_cols, dtype=np.float32), indexing="ij")
    ii, jj = ii.flatten()[:num_envs], jj.flatten()[:num_envs]
    x = -(ii - (num_rows - 1) / 2) * spacing
    y = (jj - (num_cols - 1) / 2) * spacing
    return np.stack([x, y, np.zeros_like(x)], -1).astype(np.float32)


# ------------------------------------------------------------------------------------------------ USD composition
def _set_attr(prim, name, value, type_name):
    from pxr import Sdf

    a = prim.GetAttribute(name)
    if not a:
        a = prim.CreateAttribute(name, getattr(Sdf.ValueTypeNames, type_name))
    a.Set(value)


def compose_robot_stage(disable_hand_hull: bool = True):
    """The Franka the task's spawn hook (``pour_env_cfg.py:178-230`` ``spawn_franka_with_arm_collisions``) intends,
    plus the spawner properties of ``PourSceneCfg.robot``, authored on a session layer of Isaac's asset.

    The hook asserts a composition the published asset (Isaac 6.1 and 6.2 asset roots, identical files; its
    ``physics.usda`` is the pinned payload ``0dc38454...``) does not have: arm collision proxies ``link0_c`` ..
    ``link7_c`` (present only under the asset's ``Colliders = convex_hulls`` variant; the default is ``gripper_only``)
    and a second finger mimic ``PhysxMimicJointAPI:linear`` on ``panda_finger_joint2`` (absent; only the
    ``NewtonMimicAPI`` on ``panda_finger_joint1`` exists, which the hook would delete as the duplicate). Composed here:
    ``Colliders = convex_hulls``; the ten proxies non-instanceable with collision enabled (the hook's two edits);
    the asset's ``hand_c`` hull, which the hook does not list, without collision (``disable_hand_hull``; the gripper
    keeps the asset's hand capsule and finger pads); exactly one finger mimic (the asset's ``NewtonMimicAPI``, the
    equality the hook means to keep, in the opposite leader/follower direction, the same constraint in MuJoCo);
    ``newton:selfCollisionEnabled = True`` (the hook); ``MujocoJointCfg(actuatorgravcomp=True)`` on every revolute and
    prismatic joint with ``mjc:gravcomp = 1`` on each one's child body (``IL/isaaclab_newton/.../sim/schemas/schemas.py``
    ``apply_mujoco_joint``); ``Physics = physx`` (the asset default, which sublayers the pinned ``physics.usda`` and adds
    ``physxJoint:armature = 0.1``)."""
    from pxr import Sdf, Usd, UsdPhysics

    stage = Usd.Stage.Open(FRANKA_USD)
    stage.SetEditTarget(stage.GetSessionLayer())
    root = stage.GetDefaultPrim()
    _standardize_pose(root, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))      # ArticulationCfg.InitialStateCfg defaults
    root.GetVariantSet("Colliders").SetVariantSelection("convex_hulls")
    root.GetAttribute("newton:selfCollisionEnabled").Set(True)
    proxies = {}
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if prim.GetName() in ARM_COLLISION_PROXIES and prim.GetParent().GetName() == prim.GetName():
            proxies[prim.GetName()] = prim.GetParent().GetPath()
    if set(proxies) != set(ARM_COLLISION_PROXIES):
        raise RuntimeError(f"Franka asset is missing collision proxies: {sorted(set(ARM_COLLISION_PROXIES) - set(proxies))}")
    for path in proxies.values():
        stage.OverridePrim(path).SetInstanceable(False)
    for path in proxies.values():
        UsdPhysics.CollisionAPI(stage.GetPrimAtPath(path.AppendChild(path.name))).GetCollisionEnabledAttr().Set(True)
    if disable_hand_hull:
        hand = [p for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies())
                if p.GetName() == "hand_c" and p.GetParent().GetName() == "hand_c"]
        for p in hand:
            stage.OverridePrim(p.GetParent().GetPath()).SetInstanceable(False)
            UsdPhysics.CollisionAPI(stage.GetPrimAtPath(p.GetPath())).GetCollisionEnabledAttr().Set(False)
    gravcomp_bodies = []
    for prim in Usd.PrimRange(root):
        if prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint):
            prim.AddAppliedSchema("MjcJointAPI")
            _set_attr(prim, "mjc:actuatorgravcomp", True, "Bool")
            gravcomp_bodies += list(UsdPhysics.Joint(prim).GetBody1Rel().GetTargets())
    for body_path in gravcomp_bodies:
        body = stage.GetPrimAtPath(body_path)
        cur = body.GetAttribute("mjc:gravcomp").Get() if body.GetAttribute("mjc:gravcomp") else None
        if cur is None or cur == 0.0:
            _set_attr(body, "mjc:gravcomp", 1.0, "Float")
    del Sdf
    return stage


def _standardize_pose(prim, translation, orientation_xyzw):
    """``IL/isaaclab/isaaclab/sim/utils/transforms.py`` ``standardize_xform_ops`` as ``create_prim`` calls it from
    ``spawn_from_usd``: ``[translate, orient, scale]`` on the spawned prim, overriding the referenced root's own
    transform (the SeattleLab table's ``xformOp:translate = (0.55, 0, 0)`` is replaced by the init-state pose)."""
    from pxr import Gf, UsdGeom

    x = UsdGeom.Xformable(prim)
    x.ClearXformOpOrder()
    t = x.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
    o = x.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
    s = x.AddScaleOp(UsdGeom.XformOp.PrecisionDouble)
    t.Set(Gf.Vec3d(*map(float, translation)))
    qx, qy, qz, qw = map(float, orientation_xyzw)
    o.Set(Gf.Quatd(qw, qx, qy, qz))
    s.Set(Gf.Vec3d(1.0, 1.0, 1.0))
    x.SetXformOpOrder([t, o, s])


def _cup_stage(kind: str):
    """``_source_cup_asset_cfg`` / ``_target_cup_asset_cfg`` (pour_env_cfg.py:103-135): the cups USD variant as a
    rigid body (source dynamic with MassAPI 0.05 kg, target kinematic) with a bound physics material, at its
    init-state pose."""
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    name = "SourceCup" if kind == "source" else "TargetCup"
    prim = stage.DefinePrim(f"/{name}", "Xform")
    stage.SetDefaultPrim(prim)
    prim.GetReferences().AddReference(CUPS_USD)
    prim.GetVariantSet("cupType").SetVariantSelection(kind)
    _standardize_pose(prim, SOURCE_CUP_POSITION if kind == "source" else TARGET_CUP_POSITION, (0.0, 0.0, 0.0, 1.0))
    rb = UsdPhysics.RigidBodyAPI.Apply(prim)
    rb.CreateRigidBodyEnabledAttr(True)
    rb.CreateKinematicEnabledAttr(kind != "source")
    if kind == "source":
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(SOURCE_CUP_MASS)
    mu = SOURCE_CUP_GRASP_FRICTION if kind == "source" else TARGET_CUP_FRICTION
    mat = UsdShade.Material.Define(stage, f"/{name}/physicsMaterial")
    m = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    m.CreateStaticFrictionAttr(mu)
    m.CreateDynamicFrictionAttr(mu)
    m.CreateRestitutionAttr(0.0)
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat, UsdShade.Tokens.weakerThanDescendants, "physics")
    return stage


def _table_stage():
    """``PourSceneCfg.table`` (pour_env_cfg.py:323-330): SeattleLab table, kinematic rigid body."""
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    prim = stage.DefinePrim("/Table", "Xform")
    stage.SetDefaultPrim(prim)
    prim.GetReferences().AddReference(TABLE_USD)
    _standardize_pose(prim, (0.5, 0.0, 0.0), (0.0, 0.0, 0.707, 0.707))
    rb = UsdPhysics.RigidBodyAPI.Apply(prim)
    rb.CreateRigidBodyEnabledAttr(True)
    rb.CreateKinematicEnabledAttr(True)
    return stage


# ------------------------------------------------------------------------------------------------ model building
def create_builder():
    """``NewtonManager.create_builder`` (newton_manager.py:1217-1249) with the coupler's attribute registration
    (MuJoCo and implicit-MPM custom attributes) and ``NewtonShapeCfg`` defaults (newton_manager_cfg.py:89-120)."""
    import newton
    from newton.solvers import SolverImplicitMPM, SolverMuJoCo

    b = newton.ModelBuilder(up_axis="Z")
    SolverMuJoCo.register_custom_attributes(b)
    SolverImplicitMPM.register_custom_attributes(b)
    cfg = b.default_shape_cfg
    cfg.margin, cfg.gap, cfg.ke, cfg.kd, cfg.mu = 0.0, 0.01, 2.5e3, 100.0, 1.0
    return b


def _resolvers():
    from newton.usd import SchemaResolverMjc, SchemaResolverNewton, SchemaResolverPhysx

    return [SchemaResolverNewton(), SchemaResolverPhysx(), SchemaResolverMjc()]


def _import(stage, label_root: str):
    """``newton_clone_utils._build_source_builder``: one prototype builder per clone source (poses authored on the
    stage, as the spawner leaves them; Newton's ``xform`` argument is not used: with it, a free rigid body's
    orientation came out rotated twice in Newton 1.5.2, e.g. the table at 180 deg instead of 90)."""
    b = create_builder()
    b.add_usd(stage, hide_collision_shapes=True, load_visual_shapes=False,
              schema_resolvers=_resolvers())
    # labels as the cloner leaves them for env 0 (rename_builder_labels); the hook matches suffixes only
    for attr in ("body_label", "shape_label", "joint_label"):
        labels = getattr(b, attr)
        for i, lab in enumerate(labels):
            lab = str(lab)
            if lab.startswith("/"):
                rest = lab[len(lab.split("/")[1]) + 1:]
                labels[i] = f"/World/envs/env_0/{label_root}{rest}"
    return b


@dataclass
class PourModelInfo:
    num_envs: int
    origins: np.ndarray
    bodies_per_world: int
    particles_per_world: int
    robot_bodies: list = field(default_factory=list)       # local body ids within a world
    arm_dofs: list = field(default_factory=list)           # local dof ids (panda_joint1..7)
    finger_dofs: list = field(default_factory=list)
    arm_q: list = field(default_factory=list)              # local joint_q coordinate ids
    finger_q: list = field(default_factory=list)
    hand_body: int = -1
    source_body: int = -1
    target_body: int = -1
    spill_body: int = -1
    table_body: int = -1
    joint_q_per_world: int = 0
    joint_dof_per_world: int = 0
    source_free_q: int = -1                                # first joint_q coordinate of the source cup's free joint
    target_free_q: int = -1
    source_free_dof: int = -1
    target_free_dof: int = -1


def _range_tail(arr, world):
    stop = len(arr)
    start = stop
    while start > 0 and int(arr[start - 1]) == world:
        start -= 1
    return range(start, stop)


def _set_roles(builder, shape, *, rigid, particles, visible):
    """pour_env.py:371-384 ``_set_shape_roles``."""
    from newton import ShapeFlags

    flags = int(builder.shape_flags[shape])
    for flag, on in ((ShapeFlags.COLLIDE_SHAPES, rigid), (ShapeFlags.COLLIDE_PARTICLES, particles),
                     (ShapeFlags.VISIBLE, visible)):
        flags = flags | int(flag) if on else flags & ~int(flag)
    builder.shape_flags[shape] = flags


def _force_space(builder, shape):
    """pour_env.py:94-103, 338-342: Newton's contact gains in MuJoCo's force-space solref mode."""
    builder.shape_material_ke[shape] = GRASP_CONTACT_KE
    builder.shape_material_kd[shape] = GRASP_CONTACT_KD
    attr = builder.custom_attributes.get("mujoco:solref_mode")
    if attr.values is None:
        attr.values = {}
    attr.values[shape] = MJWARP_SOLREF_MODE_FORCE_SPACE


def pour_world_hook(builder, env_id: int, position, quaternion) -> dict:
    """``FrankaPourEnv._add_pour_world_to_builder`` (pour_env.py:141-259), same edits in the same order."""
    import newton
    from newton import ShapeFlags

    builder.particle_max_velocity = PARTICLE_MAX_VELOCITY
    body_ids = _range_tail(builder.body_world, env_id)
    shape_ids = _range_tail(builder.shape_world, env_id)
    body_set = set(body_ids)

    def body_label(b):
        return str(builder.body_label[b])

    # robot shapes do not collide with particles; finger contact material (pour_env.py:386-412)
    for s in shape_ids:
        b = int(builder.shape_body[s])
        if b not in body_set:
            continue
        lab = body_label(b)
        if "/Robot/" in lab or lab.endswith("/Robot"):
            builder.shape_flags[s] &= ~int(ShapeFlags.COLLIDE_PARTICLES)
        if lab.endswith(("/panda_leftfinger", "/panda_rightfinger")):
            builder.shape_material_ke[s] = GRASP_CONTACT_KE
            builder.shape_material_kd[s] = GRASP_CONTACT_KD
            builder.shape_material_kf[s] = GRASP_CONTACT_KF
            builder.shape_material_mu[s] = SOURCE_CUP_GRASP_FRICTION

    def find_body(name):
        m = [b for b in body_ids if body_label(b).rsplit("/", 1)[-1] == name]
        if len(m) != 1:
            raise RuntimeError(f"expected one {name} body in world {env_id}, found {m}")
        return m[0]

    def find_shape(suffix, body):
        m = [s for s in shape_ids if int(builder.shape_body[s]) == body and str(builder.shape_label[s]).endswith(suffix)]
        if len(m) != 1:
            raise RuntimeError(f"expected one shape ending in {suffix} on body {body} in world {env_id}, found {m}")
        return m[0]

    source = find_body("SourceCup")
    target = find_body("TargetCup")
    # _add_kinematic_rigid_object_articulation (pour_env.py:286-310)
    child_joints = [j for _, j in builder.joint_parents.get(target, ())] if hasattr(builder, "joint_parents") else \
        [j for j in range(builder.joint_count) if int(builder.joint_child[j]) == target]
    if not child_joints:
        j = builder.add_joint_free(child=target, label=f"{body_label(target)}/FreeJoint")
        builder.add_articulation([j], label=body_label(target))
    builder.body_flags[target] = int(newton.BodyFlags.KINEMATIC)
    builder.body_mass[target] = 0.0
    builder.body_inv_mass[target] = 0.0
    builder.body_inertia[target] = wp.mat33()
    builder.body_inv_inertia[target] = wp.mat33()
    # grasp proxy (pour_env.py:328-336)
    gp = find_shape("/SourceCup/geometry/grasp_proxy", source)
    _set_roles(builder, gp, rigid=True, particles=False, visible=False)
    builder.shape_margin[gp] = 0.0
    _force_space(builder, gp)
    builder.shape_material_kf[gp] = GRASP_CONTACT_KF
    builder.shape_material_mu[gp] = SOURCE_CUP_GRASP_FRICTION
    # bowls as particle-only colliders (pour_env.py:344-354)
    for suffix, body, mu in (("/SourceCup/geometry/bowl", source, SOURCE_CUP_PARTICLE_FRICTION),
                             ("/TargetCup/geometry/bowl", target, TARGET_CUP_FRICTION)):
        s = find_shape(suffix, body)
        _set_roles(builder, s, rigid=False, particles=True, visible=True)
        builder.shape_margin[s] = MPM_COLLIDER_MARGIN
        builder.shape_material_mu[s] = mu
    rc = find_shape("/TargetCup/geometry/rigid_collider", target)
    _set_roles(builder, rc, rigid=False, particles=False, visible=False)
    hx, hy, hz = TARGET_CUP_GEOMETRY.outer_half_extents
    box = builder.add_shape_box(body=target, xform=wp.transform(wp.vec3(0.0, 0.0, hz), wp.quat_identity()),
                                hx=hx, hy=hy, hz=hz, label=f"/World/envs/env_{env_id}/TargetCup/geometry/rigid_box")
    _set_roles(builder, box, rigid=True, particles=False, visible=False)
    builder.shape_margin[box] = COLLIDER_MARGIN
    builder.shape_material_ke[box] = GRASP_CONTACT_KE
    builder.shape_material_kd[box] = GRASP_CONTACT_KD
    builder.shape_material_kf[box] = GRASP_CONTACT_KF
    builder.shape_material_mu[box] = TARGET_CUP_FRICTION
    # tabletop (pour_env.py:207-231)
    table = [s for s in shape_ids if str(builder.shape_label[s]).endswith("/Table/Collisions/Cube")
             and int(builder.shape_flags[s]) & int(ShapeFlags.COLLIDE_SHAPES)]
    if len(table) != 1:
        raise RuntimeError(f"expected one tabletop collider in world {env_id}, found {table}")
    t = table[0]
    tx = builder.shape_transform[t]
    p = wp.transform_get_translation(tx)
    builder.shape_transform[t] = wp.transform(wp.vec3(p[0], p[1], p[2] + SEATTLELAB_TABLETOP_COLLISION_INSET),
                                              wp.transform_get_rotation(tx))
    builder.shape_margin[t] = 0.0
    _force_space(builder, t)
    # spill floor (pour_env.py:233-259)
    world_xform = wp.transform(wp.vec3(*[float(v) for v in position]), wp.quat(*[float(v) for v in quaternion]))
    spill = builder.add_body(xform=world_xform, mass=0.0, inertia=wp.mat33(), is_kinematic=True, lock_inertia=True,
                             label=f"/World/envs/env_{env_id}/SpillFloor")
    ss = builder.add_shape_plane(body=spill, xform=wp.transform_identity(), width=0.0, length=0.0,
                                 cfg=newton.ModelBuilder.ShapeConfig(mu=0.8, margin=MPM_COLLIDER_MARGIN,
                                                                     has_shape_collision=False,
                                                                     has_particle_collision=True),
                                 label=f"/World/envs/env_{env_id}/SpillFloor/Collision")
    _set_roles(builder, ss, rigid=False, particles=True, visible=False)
    return {"source": source, "target": target, "spill": spill, "table_shape": t}


def emit_media(builder, position, fill_level: float = MEDIA_FILL_LEVEL) -> int:
    """``add_mpm_entry_to_builder`` + ``_emit_grid``: the source-cup lattice at the cup's world pose."""
    first, dims, cell, mass, radius, jitter = media_lattice(fill_level)
    cup = np.asarray(SOURCE_CUP_POSITION, np.float32)
    world = (first + cup + np.asarray(position, np.float32)).astype(np.float32)  # identity orientations
    before = builder.particle_count
    builder.add_particle_grid(pos=wp.vec3(*world.tolist()), rot=wp.quat_identity(), vel=wp.vec3(0.0, 0.0, 0.0),
                              dim_x=int(dims[0]), dim_y=int(dims[1]), dim_z=int(dims[2]), cell_x=float(cell[0]),
                              cell_y=float(cell[1]), cell_z=float(cell[2]), mass=mass, jitter=jitter,
                              radius_mean=radius, custom_attributes=media_material_attributes())
    return builder.particle_count - before


def build_model(num_envs: int, device: str = "metal:0", fill_level: float = MEDIA_FILL_LEVEL,
                disable_hand_hull: bool = True):
    """Isaac's cloner (IL/isaaclab_newton/.../cloner/replicate.py ``_build_newton_builder_from_mapping``): global
    shapes first, then per world ``begin_world``, the prototypes (table, robot, source cup, target cup, in
    ``PourSceneCfg`` order) at the env origin, the task hook, the media hook, ``end_world``; then
    ``_prepare_builder_for_finalize`` (mpm_manager.py:104-124) and ``finalize``."""
    import newton
    from newton import BodyFlags, GeoType

    protos = [
        ("Table", _import(_table_stage(), "Table")),
        ("Robot", _import(compose_robot_stage(disable_hand_hull), "Robot")),
        ("SourceCup", _import(_cup_stage("source"), "SourceCup")),
        ("TargetCup", _import(_cup_stage("target"), "TargetCup")),
    ]
    robot_proto = protos[1][1]
    # Isaac's default joint state (PourSceneCfg.robot.init_state, pour_env_cfg.py:365-366)
    names = [str(lab).rsplit("/", 1)[-1] for lab in robot_proto.joint_label]
    for j, nm in enumerate(names):
        if nm in ARM_JOINTS or nm in FINGER_JOINTS:
            q0 = robot_proto.joint_q_start[j]
            d0 = robot_proto.joint_qd_start[j]
            val = ARM_HOME[ARM_JOINTS.index(nm)] if nm in ARM_JOINTS else GRIPPER_OPEN_POSITION
            robot_proto.joint_q[q0] = val
            robot_proto.joint_target_q[d0] = val

    origins = env_origins(num_envs)
    b = create_builder()
    # /World/GroundPlane (pour_env_cfg.py:331-335): a static plane at z = -1.05
    b.add_ground_plane(height=-1.05)
    hooks = []
    particles_per_world = None
    for w in range(num_envs):
        b.begin_world()
        x = wp.transform(wp.vec3(*origins[w].tolist()), wp.quat_identity())
        for _, proto in protos:
            b.add_builder(proto, xform=x)
        # the cloner renames labels per env after all hooks (newton_clone_utils.py:317-381); do it here directly
        for attr, arr_world in (("body_label", b.body_world), ("shape_label", b.shape_world),
                                ("joint_label", b.joint_world)):
            labels = getattr(b, attr)
            for i in _range_tail(arr_world, w):
                labels[i] = str(labels[i]).replace("/World/envs/env_0/", f"/World/envs/env_{w}/")
        hooks.append(pour_world_hook(b, w, origins[w], (0.0, 0.0, 0.0, 1.0)))
        n = emit_media(b, origins[w], fill_level)
        if particles_per_world is None:
            particles_per_world = n
        elif n != particles_per_world:
            raise RuntimeError("media particle count differs between worlds")
        b.end_world()
    # NewtonMPMManager._prepare_builder_for_finalize (mpm_manager.py:104-124)
    for body_id, flags in enumerate(b.body_flags):
        if int(flags) & int(BodyFlags.KINEMATIC):
            b.body_mass[body_id] = 0.0
            b.body_inv_mass[body_id] = 0.0
            b.body_inertia[body_id] = wp.mat33()
            b.body_inv_inertia[body_id] = wp.mat33()
    for s, t in enumerate(b.shape_type):
        if t == GeoType.CONVEX_MESH:
            b.shape_type[s] = GeoType.MESH
    model = b.finalize(device=device)

    # per-world layout (homogeneous by construction)
    bodies_per_world = (model.body_count) // num_envs
    info = PourModelInfo(num_envs=num_envs, origins=origins, bodies_per_world=bodies_per_world,
                         particles_per_world=int(particles_per_world))
    h0 = hooks[0]
    info.source_body, info.target_body, info.spill_body = h0["source"], h0["target"], h0["spill"]
    labels = [str(x) for x in model.body_label]
    info.robot_bodies = [i for i in range(bodies_per_world) if "/World/envs/env_0/Robot/" in labels[i]]
    info.hand_body = next(i for i in info.robot_bodies if labels[i].endswith("/" + TCP_BODY))
    info.table_body = next(i for i in range(bodies_per_world) if labels[i].endswith("/Table"))
    jl = [str(x) for x in model.joint_label]
    jw = model.joint_world.numpy()
    jq = model.joint_q_start.numpy()
    jd = model.joint_qd_start.numpy()
    w0 = [j for j in range(model.joint_count) if jw[j] == 0]
    info.joint_q_per_world = int(model.joint_coord_count // num_envs)
    info.joint_dof_per_world = int(model.joint_dof_count // num_envs)
    for nm in ARM_JOINTS:
        j = next(j for j in w0 if jl[j].rsplit("/", 1)[-1] == nm)
        info.arm_q.append(int(jq[j]))
        info.arm_dofs.append(int(jd[j]))
    for nm in FINGER_JOINTS:
        j = next(j for j in w0 if jl[j].rsplit("/", 1)[-1] == nm)
        info.finger_q.append(int(jq[j]))
        info.finger_dofs.append(int(jd[j]))
    jc = model.joint_child.numpy()
    for key, body in (("source", info.source_body), ("target", info.target_body)):
        j = next(j for j in w0 if jc[j] == body)
        setattr(info, f"{key}_free_q", int(jq[j]))
        setattr(info, f"{key}_free_dof", int(jd[j]))
    return model, info


def apply_actuator_properties(model, info: PourModelInfo):
    """Isaac Lab's implicit-actuator writes (IL/isaaclab/isaaclab/actuators/actuator_control.py:430-459): per arm joint
    the task's stiffness / damping and ``FRANKA_PANDA_CFG``'s effort limits; fingers keep the USD drive (350 N/m,
    10 N s/m on ``panda_finger_joint1``, none on the mimic follower) with effort limit 200 N. Armature, friction and
    velocity limits are left as imported (the resolved defaults equal the USD values)."""
    n = info.num_envs
    dpw = info.joint_dof_per_world
    ke = model.joint_target_ke.numpy()
    kd = model.joint_target_kd.numpy()
    eff = model.joint_effort_limit.numpy()
    for w in range(n):
        for k, d in enumerate(info.arm_dofs):
            ke[w * dpw + d] = ARM_STIFFNESS[k]
            kd[w * dpw + d] = ARM_DAMPING[k]
            eff[w * dpw + d] = ARM_EFFORT_LIMIT[k]
        for d in info.finger_dofs:
            eff[w * dpw + d] = FINGER_EFFORT_LIMIT
    model.joint_target_ke.assign(ke)
    model.joint_target_kd.assign(kd)
    model.joint_effort_limit.assign(eff)


# MetalSim memory budgets (PourPhysicsConfig.mpm_grid_budget = "measured"): sparse-grid NanoVDB node capacities from
# the peak demand of 2,048-world rollouts with random actions (per world 3.6 leaves, 1.1 lower and 0.01 upper nodes,
# 80 cells; runs/franka_pour/memory/), with about 4x headroom. Newton's automatic capacities reserve a leaf per
# reserved cell and 16x the lower / upper nodes of the first grid, and the vertex grid takes 8x those again: 12.4 GB
# of NanoVDB storage at 2,048 worlds for ~2.3k lower nodes in use. Capacities only bound storage: a grid within them
# is the same grid, so results are bitwise unchanged; an overflow is reported by PourPhysics.check_status (the PPO
# loop calls it once per iteration).
MPM_MEASURED_LEAVES_PER_WORLD = 16
MPM_MEASURED_LOWER_PER_WORLD = 4.5
MPM_MEASURED_UPPER_MIN = 80


def mpm_config(num_envs: int, fill_level: float = MEDIA_FILL_LEVEL, grid_budget: str = "isaac",
               cells_per_world: int | None = None):
    """``MPMSolverCfg`` of the media entry (pour_env_cfg.py:633-648) through ``_make_solver_config``
    (mpm_manager.py:30-67), with the capacity rule ``_resolve_mpm_cell_cap`` (pour_env_cfg.py:276-292).

    ``grid_budget="measured"`` replaces Newton's automatic NanoVDB node capacities by the measured budgets above;
    ``cells_per_world`` replaces Isaac's cell reservation (1,024 per world at 735 particles). Both leave results
    unchanged while the grid stays within them (see the budget note above)."""
    from newton.solvers import SolverImplicitMPM

    per_world = ((media_particle_count(fill_level) + MPM_CELL_CAPACITY_ALIGNMENT - 1)
                 // MPM_CELL_CAPACITY_ALIGNMENT) * MPM_CELL_CAPACITY_ALIGNMENT
    if cells_per_world is not None:
        per_world = int(cells_per_world)
    leaf = lower = upper = -1
    if grid_budget == "measured":
        leaf = min(per_world * num_envs, MPM_MEASURED_LEAVES_PER_WORLD * num_envs)
        lower = min(leaf, max(8, math.ceil(MPM_MEASURED_LOWER_PER_WORLD * num_envs)))
        upper = min(lower, max(MPM_MEASURED_UPPER_MIN, math.ceil(num_envs / 25)))
    elif grid_budget != "isaac":
        raise ValueError(f"grid_budget must be 'isaac' or 'measured', got {grid_budget!r}")
    return SolverImplicitMPM.Config(
        max_iterations=24, tolerance=1.0e-4, solver="jacobi", warmstart_mode="auto", collider_velocity_mode="forward",
        voxel_size=MPM_VOXEL_SIZE, grid_type="sparse", grid_padding=0, max_active_cell_count=per_world * num_envs,
        max_leaf_node_count=leaf, max_lower_node_count=lower, max_upper_node_count=upper, separate_worlds=True,
        transfer_scheme="apic", integration_scheme="pic", critical_fraction=0.0, air_drag=1.0,
        collider_normal_from_sdf_gradient=False, collider_basis="pic27", strain_basis="P0", velocity_basis="Q1")


# MJWarpSolverCfg (IL/isaaclab_newton/.../physics/mjwarp_manager_cfg.py) defaults + the task's overrides, as
# NewtonManager._filter_solver_kwargs passes them to SolverMuJoCo
MJWARP_KWARGS = dict(use_mujoco_contacts=False, integrator="implicitfast", njmax=2560, nconmax=400, iterations=100,
                     ls_iterations=50, solver="newton", cone="pyramidal", impratio=1.0, tolerance=1.0e-6,
                     ccd_iterations=35, update_data_interval=1, use_mujoco_cpu=False, disable_contacts=False,
                     disable_sensors=False)


# ------------------------------------------------------------------------------------------------ physics runtime
@wp.kernel
def apply_pour_actions(joint_q: wp.array[float], arm_delta: wp.array2d[float], gripper_target: wp.array[float],
                       q_per_world: int, dof_per_world: int, arm_q: wp.array[int], arm_dof: wp.array[int],
                       finger_dof: wp.array[int], joint_target_q: wp.array[float]):
    """``ActionManager.apply_action`` for the two terms (IL/isaaclab/isaaclab/envs/mdp/actions/joint_actions.py:231-233,
    binary_joint_actions.py): arm target = processed delta + current joint position, re-read every physics step;
    both finger targets = the filtered gripper command."""
    e = wp.tid()
    for k in range(7):
        joint_target_q[e * dof_per_world + arm_dof[k]] = arm_delta[e, k] + joint_q[e * q_per_world + arm_q[k]]
    for k in range(2):
        joint_target_q[e * dof_per_world + finger_dof[k]] = gripper_target[e]


@dataclass
class PourPhysicsConfig:
    num_envs: int = 64
    device: str = "metal:0"
    fill_level: float = MEDIA_FILL_LEVEL
    rigid_deterministic: bool = False       # Isaac: NewtonCfg.deterministic_mode = "not_guaranteed"
    metal_fast_math: str = "off"            # newton_compat.install default
    # Isaac: NewtonCollisionPipelineCfg.max_triangle_pairs = 1_000_000, which this scene overflows from ~1,250 worlds
    # (contacts dropped in atomic allocation order; newton-physics/newton#4604, fixed upstream by #4605 sizing it from
    # the model). None = max(1e6, 4,096 per world) (8,388,608 at 2,048 worlds), so no contact is dropped.
    max_triangle_pairs: int | None = None
    capture: bool = True                    # replay captured graphs (Isaac: use_cuda_graph=True)
    # "sim": one graph per sim.step (Isaac's boundary, newton_manager.py _simulate_physics_only) with the action
    # kernel launched between replays; "policy": one graph per policy step (4 sim steps + actions). "sim" is the
    # default because it is Isaac's granularity. "policy" works from Warp 41b7bc45 on: before that, the graph held every
    # step's temporaries (70.7 GB at 2,048 worlds, past the working set) and replayed NaN; now 37.4 GB (MPM.md 8f).
    capture_unit: str = "sim"
    disable_hand_hull: bool = True
    # MetalSim memory budgets (bitwise-neutral within capacity; see mpm_config): "isaac" keeps Newton's automatic
    # sparse-grid node capacities, "measured" uses the measured budgets; None keeps Isaac's cell reservation.
    mpm_grid_budget: str = "isaac"
    mpm_cells_per_world: int | None = None


class PourPhysics:
    """The task's Newton physics: model, ``SolverCoupledProxy``, collision pipeline, double-buffered state, and the
    policy-step graph (4 x [apply actions, collide, 2 x coupled step])."""

    def __init__(self, cfg: PourPhysicsConfig):
        import newton
        import warp.fem as fem
        from newton.solvers import SolverImplicitMPM, SolverMuJoCo
        from newton.solvers.experimental.coupled import SolverCoupled, SolverCoupledProxy

        from metalsim.physics.mpm import newton_compat

        self.cfg = cfg
        self.device = wp.get_device(cfg.device)
        is_metal = bool(getattr(self.device, "is_metal", False))
        # Metal: float32 accumulation (no float64), CUDA's stopping rule decided on the GPU, chunk graph
        # (metalsim/physics/mpm/MPM.md sections 1, 7b, 8); CUDA/CPU: Newton as shipped
        newton_compat.install(accumulate="float32" if is_metal else "native", loop="cuda" if is_metal else "native",
                              log=False, chunk_graph=is_metal, gpu_exit=is_metal,
                              rigid_deterministic=cfg.rigid_deterministic, device_args=None,
                              metal_fast_math=cfg.metal_fast_math)
        with wp.ScopedDevice(self.device):
            self.model, self.info = build_model(cfg.num_envs, cfg.device, cfg.fill_level, cfg.disable_hand_hull)
            apply_actuator_properties(self.model, self.info)
            m, info = self.model, self.info
            n = cfg.num_envs
            labels = [str(x) for x in m.body_label]
            body_world = m.body_world.numpy()

            def bodies_of(name):           # CouplerEntryCfg body regex "/World/envs/env_.*/<name>" (+ descendants)
                return [b for b in range(m.body_count) if body_world[b] >= 0
                        and (labels[b].endswith(f"/{name}") or f"/{name}/" in labels[b])]

            arm_bodies = bodies_of("Robot") + bodies_of("Table") + bodies_of("SourceCup") + bodies_of("TargetCup")
            arm_set = set(arm_bodies)
            shape_body = m.shape_body.numpy()
            arm_shapes = [s for s in range(m.shape_count) if int(shape_body[s]) in arm_set or int(shape_body[s]) < 0]
            jp, jc = m.joint_parent.numpy(), m.joint_child.numpy()
            arm_joints = [j for j in range(m.joint_count) if int(jc[j]) in arm_set
                          and (int(jp[j]) < 0 or int(jp[j]) in arm_set)]
            spill = bodies_of("SpillFloor")
            spill_set = set(spill)
            media_shapes = [s for s in range(m.shape_count) if int(shape_body[s]) in spill_set]
            proxies = bodies_of("SourceCup") + bodies_of("TargetCup")
            mj_kwargs = dict(MJWARP_KWARGS)
            if hasattr(wp, "DeterministicMode"):
                mj_kwargs["deterministic"] = (wp.DeterministicMode.RUN_TO_RUN if cfg.rigid_deterministic
                                              else wp.DeterministicMode.NOT_GUARANTEED)
            mpm_cfg = mpm_config(n, cfg.fill_level, grid_budget=cfg.mpm_grid_budget,
                                 cells_per_world=cfg.mpm_cells_per_world)
            self._tstore = fem.TemporaryStore()       # NewtonMPMManager._create_solver (mpm_manager.py:126-133)
            entries = [
                SolverCoupled.Entry(name="arm", solver=lambda view: SolverMuJoCo(view, **mj_kwargs), bodies=arm_bodies,
                                    particles=[], joints=arm_joints, shapes=arm_shapes, substeps=ARM_SUBSTEPS),
                SolverCoupled.Entry(name="media",
                                    solver=lambda view: SolverImplicitMPM(view, mpm_cfg, temporary_store=self._tstore),
                                    bodies=spill, particles=list(range(m.particle_count)), joints=[],
                                    shapes=media_shapes, substeps=MEDIA_SUBSTEPS, in_place=True),
            ]
            coupling = SolverCoupledProxy.Config(
                proxies=[SolverCoupledProxy.Proxy(source="arm", destination="media", bodies=proxies, particles=[],
                                                  mode="staggered", mass_scale=1000.0, collide_interval=None,
                                                  collision_pipeline=None)],
                iterations=1)
            self.solver = SolverCoupledProxy(model=m, entries=entries, coupling=coupling)
            # NewtonCollisionPipelineCfg(soft_contact_max=0) with its defaults (newton_collision_cfg.py:97-172)
            self.pipeline = newton.CollisionPipeline(
                m, broad_phase="explicit", reduce_contacts=True, rigid_contact_max=None,
                max_triangle_pairs=cfg.max_triangle_pairs or max(1_000_000, 4096 * n), soft_contact_max=0,
                soft_contact_margin=0.01,
                enable_rigid_soft_full_surface_contact=False, deterministic=cfg.rigid_deterministic)
            self.contacts = self.pipeline.contacts()
            self.solver.prepare_contacts(self.contacts)
            self.s0, self.s1 = m.state(), m.state()
            self.control = m.control()
            newton.eval_fk(m, m.joint_q, m.joint_qd, self.s0)
            newton.eval_fk(m, m.joint_q, m.joint_qd, self.s1)
            # action inputs written by the task (torch views over these arrays)
            self.arm_delta = wp.zeros((n, 7), dtype=float)
            self.gripper_target = wp.zeros(n, dtype=float)
            self._arm_q = wp.array(np.asarray(info.arm_q, np.int32), dtype=int)
            self._arm_dof = wp.array(np.asarray(info.arm_dofs, np.int32), dtype=int)
            self._finger_dof = wp.array(np.asarray(info.finger_dofs, np.int32), dtype=int)
            self.mpm = self.solver.solver("media")
            self.art_world = m.articulation_world.numpy() if m.articulation_count else np.zeros(0, np.int32)
        self.graph = None
        self.sim_steps = 0

    # -- stepping ------------------------------------------------------------------------------------------------
    def _apply_actions(self):
        info = self.info
        wp.launch(apply_pour_actions, dim=self.cfg.num_envs,
                  inputs=[self.s0.joint_q, self.arm_delta, self.gripper_target, info.joint_q_per_world,
                          info.joint_dof_per_world, self._arm_q, self._arm_dof, self._finger_dof],
                  outputs=[self.control.joint_target_q], device=self.device)

    def sim_step(self):
        """One ``sim.step`` (newton_manager.py ``_simulate_physics_only``): collide once, ``num_substeps`` coupled
        steps with state swaps (even count: the buffers end where they started) and ``clear_forces``."""
        self.pipeline.collide(self.s0, self.contacts)
        dt = SIM_DT / NUM_SUBSTEPS
        for _ in range(NUM_SUBSTEPS):
            self.solver.step(self.s0, self.s1, self.control, self.contacts, dt)
            self.s0, self.s1 = self.s1, self.s0
            self.s0.clear_forces()

    def _policy_step_body(self):
        for _ in range(DECIMATION):
            self._apply_actions()
            self.sim_step()

    def capture(self):
        """Record one policy step (``capture_unit="policy"``) or one ``sim.step`` (``"sim"``, as Isaac captures
        ``_simulate_physics_only``) as a graph, after the first eager step has loaded every module."""
        with wp.ScopedDevice(self.device):
            with wp.ScopedCapture(device=self.device, force_module_load=False) as cap:
                if self.cfg.capture_unit == "policy":
                    self._policy_step_body()
                else:
                    self.sim_step()
        self.graph = cap.graph

    def policy_step(self):
        """Advance one policy step (4 physics steps); queued on the Warp stream, no host synchronization."""
        with wp.ScopedDevice(self.device):
            if self.graph is None:
                self._policy_step_body()
                if self.cfg.capture:
                    self.capture()
            elif self.cfg.capture_unit == "policy":
                wp.capture_launch(self.graph)
            else:
                for _ in range(DECIMATION):
                    self._apply_actions()
                    wp.capture_launch(self.graph)
        self.sim_steps += DECIMATION

    def check_status(self):
        """``NewtonMPMManager._check_solver_status``: raise on a sparse-grid capacity overflow (synchronizes)."""
        if hasattr(self.mpm, "check_sparse_grid_rebuild_status"):
            self.mpm.check_sparse_grid_rebuild_status()

    # -- resets ---------------------------------------------------------------------------------------------------
    def world_mask(self, env_ids) -> wp.array:
        """Newton reset masks carry one trailing slot for global entities (pour_env.py:683-685)."""
        mask = np.zeros(self.cfg.num_envs + 1, dtype=bool)
        mask[np.asarray(env_ids, np.int64)] = True
        return wp.array(mask, dtype=wp.bool, device=self.device)

    def finish_reset(self, env_ids):
        """After the task has written joint state, targets, cup poses and particles of ``env_ids`` into ``s0`` (and the
        particles into ``s1``): ``NewtonMPMManager.reset_solver_state(world_mask, BODY | PARTICLE)`` on both state
        buffers (pour_env.py:662, mpm_manager.py:291-303), then what the next ``sim.step`` does first for dirty worlds:
        the coupled solver's ``reset(flags=0)`` (newton_manager.py:1006) and ``forward()``'s masked ``eval_fk``."""
        import newton

        with wp.ScopedDevice(self.device):
            wm = self.world_mask(env_ids)
            flags = int(newton.StateFlags.BODY) | int(newton.StateFlags.PARTICLE)
            amask = wp.array(np.isin(self.art_world, np.asarray(env_ids)), dtype=wp.bool, device=self.device)
            # the robot's FK runs on the first body read after its joint writes (pour_env.py:634)
            newton.eval_fk(self.model, self.s0.joint_q, self.s0.joint_qd, self.s0, mask=amask)
            self.solver.reset(self.s1, world_mask=wm, flags=flags)
            self.solver.reset(self.s0, world_mask=wm, flags=flags)
            self.solver.reset(self.s0, world_mask=wm, flags=0)
            newton.eval_fk(self.model, self.s0.joint_q, self.s0.joint_qd, self.s0, mask=amask)

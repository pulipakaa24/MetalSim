"""Scene configuration: one small YAML (or the same dataclasses in Python) declares a robot, movable objects, camera
mounts, sensors, a background and the env count; ``metalsim.scene.compose`` builds it and ``python -m metalsim.run``
runs or films it.

Names follow Isaac Lab 3.0 (``upstream/IsaacLab3`` at ``ae37b028``) wherever the concept is the same, so an Isaac
scene config reads across almost field for field:

* ``SceneCfg`` = ``InteractiveSceneCfg`` (``num_envs``, ``env_spacing``) with the entities as fields: ``robot``
  (``ArticulationCfg``), ``objects`` ({name: ``RigidObjectCfg``}), ``cameras`` ({name: ``CameraCfg``}), ``sensors``
  ({name: ``RayCasterCfg``}), ``background`` (ground plane / dome light / lights / splat).
* Spawners: ``MjcfFileCfg`` / ``UrdfFileCfg`` / ``UsdFileCfg`` (``asset_path`` / ``usd_path``), shapes ``CuboidCfg``
  ``SphereCfg`` ``CylinderCfg`` ``CapsuleCfg`` ``MeshFileCfg``, with ``mass_props.mass``,
  ``visual_material.diffuse_color`` (``PreviewSurfaceCfg``) and ``physics_material`` (``RigidBodyMaterialCfg``:
  static_friction / dynamic_friction / restitution); ``GroundPlaneCfg``, ``DomeLightCfg`` (``texture_file`` = HDRI),
  ``DistantLightCfg``.
* ``init_state``: ``pos``, ``rot`` as **(x, y, z, w)** (Isaac Lab 3.0's convention; 2.x used w-first), ``joint_pos``
  {regex: value}.
* ``CameraCfg``: ``prim_path`` "{ENV_REGEX_NS}/Robot/<link>/<name>" mounts the camera on that link (or a fixed camera
  without the Robot part), ``offset`` (``OffsetCfg``: pos, rot xyzw, ``convention`` "ros" | "opengl" | "world"),
  ``width`` / ``height``, ``data_types`` (rgb, distance_to_image_plane, distance_to_camera, semantic_segmentation,
  instance_id_segmentation_fast, normals), ``spawn`` (``PinholeCameraCfg``: focal_length, horizontal_aperture,
  clipping_range). Added here: ``renderer`` (tier 0 / 1 / 2; a splat background composites tier 0 under the splats).
* ``RayCasterCfg`` with ``LidarPatternCfg`` (channels, vertical_fov_range, horizontal_fov_range, horizontal_res) and
  ``max_distance``: metalsim's hardware ray-traced lidar (``metalsim.sensors.raytrace.Lidar``).
* Randomization: ``randomize.pose_range`` / ``velocity_range`` with the keys of Isaac's ``reset_root_state_uniform``
  (x, y, z, roll, pitch, yaw), applied per env at every reset.
* ``background.splat`` (``SplatBackgroundCfg``): a Gaussian-splat scene (``metalsim.render.splat``), its file -> sim
  transform and its collision proxy (``CollisionProxyCfg``: "plane" at a height, a supplied "mesh", the room
  "structure" of an InteriorGS scene, or "none"). Isaac Lab composes NuRec backgrounds the same way: the splat is only
  seen, a ground plane at z = 0 (or the reconstruction's mesh) is what collides (docs/SCENES.md).

``load(path)`` reads YAML / JSON into these classes (``type:`` picks the spawner class, as Isaac's ``class_type``).
"""
from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass, field


# ------------------------------------------------------------------------------------------------ spawners
@dataclass
class MassPropertiesCfg:
    mass: float | None = None
    density: float | None = None


@dataclass
class PreviewSurfaceCfg:
    diffuse_color: tuple = (0.18, 0.18, 0.18)
    roughness: float = 0.5
    metallic: float = 0.0
    texture_file: str | None = None          # metalsim addition (Isaac: MdlFileCfg / texture randomization)


@dataclass
class RigidBodyMaterialCfg:
    static_friction: float = 0.5
    dynamic_friction: float = 0.5
    restitution: float = 0.0


@dataclass
class SpawnerCfg:
    visible: bool = True
    semantic_tags: list = field(default_factory=list)


@dataclass
class ShapeCfg(SpawnerCfg):
    mass_props: MassPropertiesCfg = field(default_factory=MassPropertiesCfg)
    visual_material: PreviewSurfaceCfg | None = None
    physics_material: RigidBodyMaterialCfg = field(default_factory=RigidBodyMaterialCfg)


@dataclass
class CuboidCfg(ShapeCfg):
    size: tuple = (0.1, 0.1, 0.1)            # full extents [m], as Isaac


@dataclass
class SphereCfg(ShapeCfg):
    radius: float = 0.05


@dataclass
class CylinderCfg(ShapeCfg):
    radius: float = 0.05
    height: float = 0.1


@dataclass
class CapsuleCfg(ShapeCfg):
    radius: float = 0.05
    height: float = 0.1                      # cylinder part, as Isaac


@dataclass
class MeshFileCfg(ShapeCfg):
    """A mesh object from an .obj / .stl file (Isaac: UsdFileCfg of a mesh prim)."""

    asset_path: str = ""
    scale: tuple = (1.0, 1.0, 1.0)


@dataclass
class MjcfFileCfg(SpawnerCfg):
    asset_path: str = ""
    keyframe: str | None = "home"            # metalsim: default joint state from this MJCF keyframe when present


@dataclass
class UrdfFileCfg(SpawnerCfg):
    asset_path: str = ""
    fix_base: bool = False


@dataclass
class UsdFileCfg(SpawnerCfg):
    usd_path: str = ""
    visuals: bool = True


@dataclass
class GroundPlaneCfg(SpawnerCfg):
    color: tuple = (0.5, 0.5, 0.5)
    size: tuple = (100.0, 100.0)
    physics_material: RigidBodyMaterialCfg = field(default_factory=RigidBodyMaterialCfg)
    texture_file: str | None = None          # metalsim addition: tiled texture (Poly Haven path or file)
    texture_repeat: float = 1.0
    height: float = 0.0


@dataclass
class DomeLightCfg(SpawnerCfg):
    intensity: float = 1.0
    color: tuple = (1.0, 1.0, 1.0)
    texture_file: str | None = None          # equirectangular HDRI


@dataclass
class DistantLightCfg(SpawnerCfg):
    intensity: float = 1.0
    color: tuple = (1.0, 1.0, 1.0)
    angle: float = 0.53
    direction: tuple = (-0.4, 0.3, -0.85)    # metalsim: direction the light travels (Isaac: the prim's rotation)


# ------------------------------------------------------------------------------------------------ assets
@dataclass
class InitialStateCfg:
    pos: tuple = (0.0, 0.0, 0.0)
    rot: tuple = (0.0, 0.0, 0.0, 1.0)        # (x, y, z, w), Isaac Lab 3.0
    lin_vel: tuple = (0.0, 0.0, 0.0)
    ang_vel: tuple = (0.0, 0.0, 0.0)
    joint_pos: dict = field(default_factory=dict)


@dataclass
class RandomizeCfg:
    """Per-env reset randomization with ``reset_root_state_uniform``'s keys (x, y, z, roll, pitch, yaw): position =
    default + uniform sample, orientation = default x euler(roll, pitch, yaw) sample."""

    pose_range: dict = field(default_factory=dict)
    velocity_range: dict = field(default_factory=dict)


@dataclass
class ArticulationCfg:
    prim_path: str = "{ENV_REGEX_NS}/Robot"
    spawn: SpawnerCfg = field(default_factory=MjcfFileCfg)
    init_state: InitialStateCfg = field(default_factory=InitialStateCfg)
    randomize: RandomizeCfg = field(default_factory=RandomizeCfg)


@dataclass
class RigidObjectCfg:
    prim_path: str = ""
    spawn: SpawnerCfg = field(default_factory=CuboidCfg)
    init_state: InitialStateCfg = field(default_factory=InitialStateCfg)
    randomize: RandomizeCfg = field(default_factory=RandomizeCfg)
    count: int = 1                           # metalsim: replicate as <name>_0 .. <name>_{count-1}
    kinematic: bool = False                  # metalsim: fixed in the world (no free joint)


# ------------------------------------------------------------------------------------------------ sensors
@dataclass
class OffsetCfg:
    pos: tuple = (0.0, 0.0, 0.0)
    rot: tuple = (0.0, 0.0, 0.0, 1.0)        # (x, y, z, w)
    convention: str = "ros"                  # "ros" (+z forward, +y down), "opengl" (-z forward, +y up), "world" (+x fwd, +z up)


@dataclass
class PinholeCameraCfg:
    focal_length: float = 24.0
    horizontal_aperture: float = 20.955
    vertical_aperture: float | None = None
    clipping_range: tuple = (0.01, 1.0e6)


@dataclass
class RendererCfg:
    """metalsim addition: which renderer draws the camera. tier 0 raster, 1 raster + ray-traced shadows / AO /
    reflections, 2 path tracer (rgb only; ``passes`` x ``spp``)."""

    tier: int = 0
    passes: int = 1
    spp: int = 4


@dataclass
class CameraCfg:
    prim_path: str = "{ENV_REGEX_NS}/Camera"
    offset: OffsetCfg = field(default_factory=OffsetCfg)
    width: int = 640
    height: int = 480
    data_types: list = field(default_factory=lambda: ["rgb"])
    spawn: PinholeCameraCfg = field(default_factory=PinholeCameraCfg)
    update_period: float = 0.0
    renderer: RendererCfg = field(default_factory=RendererCfg)


@dataclass
class LidarPatternCfg:
    channels: int = 16
    vertical_fov_range: tuple = (-15.0, 15.0)
    horizontal_fov_range: tuple = (-180.0, 180.0)
    horizontal_res: float = 1.0


@dataclass
class RayCasterCfg:
    prim_path: str = "{ENV_REGEX_NS}/Robot/base"
    offset: OffsetCfg = field(default_factory=lambda: OffsetCfg(convention="world"))
    pattern_cfg: LidarPatternCfg = field(default_factory=LidarPatternCfg)
    max_distance: float = 20.0
    update_period: float = 0.0


# ------------------------------------------------------------------------------------------------ background
@dataclass
class SplatTransformCfg:
    """File -> sim map (``metalsim.render.splat.make_transform``): the file's up axis ("z" | "y" | "-y") onto +z,
    uniform scale, yaw [deg], translation [m]."""

    up: str = "z"
    scale: float = 1.0
    yaw_deg: float = 0.0
    translate: tuple = (0.0, 0.0, 0.0)


@dataclass
class CollisionProxyCfg:
    """What collides where the splat is seen. "plane": a ground plane at ``plane_z`` (Isaac Lab's NuRec recipe);
    "mesh": ``mesh_path`` (.obj / .stl / .usd(a/z) mesh, e.g. a NuRec scene's mesh.usd), in the splat file's frame
    unless ``mesh_in_sim_frame``; "structure": an InteriorGS ``structure.json`` (floor plane + every room's walls
    extruded from its floor polygon, ``wall_height`` high, ``wall_thickness`` thick; doors are closed); "occupancy":
    the InteriorGS ``occupancy.png`` / ``occupancy.json`` next to ``structure_json`` as a height field (occupied cells
    ``wall_height`` tall, furniture included); "none"."""

    type: str = "plane"
    plane_z: float = 0.0
    mesh_path: str | None = None
    mesh_in_sim_frame: bool = False
    structure_json: str | None = None
    wall_height: float = 2.6
    wall_thickness: float = 0.1
    friction: float = 0.8
    visible: bool = False                    # draw the proxy (debugging); never drawn over the splat otherwise


@dataclass
class SplatBackgroundCfg:
    path: str = ""
    transform: SplatTransformCfg = field(default_factory=SplatTransformCfg)
    crop: object = "auto"
    collision: CollisionProxyCfg = field(default_factory=CollisionProxyCfg)
    source: str = ""                         # provenance for docs (where the file came from, its licence)
    licence: str = ""


@dataclass
class BackgroundCfg:
    ground: GroundPlaneCfg | None = field(default_factory=GroundPlaneCfg)
    dome: DomeLightCfg | None = None
    lights: list = field(default_factory=lambda: [DistantLightCfg()])
    splat: SplatBackgroundCfg | None = None


# ------------------------------------------------------------------------------------------------ scene
@dataclass
class PhysicsCfg:
    dt: float = 0.005
    substeps: int = 4                        # physics steps per control step (Isaac: decimation)
    solver_iterations: int | None = None
    ls_iterations: int | None = None
    njmax: int | str | None = 512
    nconmax: int | str | None = None


@dataclass
class PolicyCfg:
    """What drives the robot in ``metalsim.run``: "hold" (position actuators hold the initial joint targets), "zero"
    (zero controls), "random" (uniform in the control range, ``scale``)."""

    type: str = "hold"
    scale: float = 0.3
    seed: int = 0


@dataclass
class SceneCfg:
    """``InteractiveSceneCfg`` plus the entity fields. ``task``: a registered film task (``metalsim.render.film``)
    whose own scene builder is used instead (e.g. franka_pour); then only ``film`` applies."""

    num_envs: int = 1
    env_spacing: float = 3.0
    robot: ArticulationCfg | None = None
    objects: dict = field(default_factory=dict)
    cameras: dict = field(default_factory=dict)
    sensors: dict = field(default_factory=dict)
    background: BackgroundCfg = field(default_factory=BackgroundCfg)
    physics: PhysicsCfg = field(default_factory=PhysicsCfg)
    policy: PolicyCfg = field(default_factory=PolicyCfg)
    task: str | None = None
    film: dict = field(default_factory=dict)  # film overrides: viewer (eye, lookat, ...), seconds, passes, ...
    assets: list = field(default_factory=list)  # [{name, source, url, licence}] for film docs / provenance
    name: str = "scene"
    base_dir: str = ""                       # directory relative paths resolve against (set by load())


# ------------------------------------------------------------------------------------------------ loading
SPAWNERS = {c.__name__: c for c in (CuboidCfg, SphereCfg, CylinderCfg, CapsuleCfg, MeshFileCfg, MjcfFileCfg,
                                     UrdfFileCfg, UsdFileCfg, GroundPlaneCfg, DomeLightCfg, DistantLightCfg)}


def _build(cls, data):
    """dict -> dataclass ``cls`` recursively, by field types; ``type:`` selects a spawner subclass."""
    if data is None or dataclasses.is_dataclass(data):
        return data
    if not isinstance(data, dict):
        raise TypeError(f"{cls.__name__}: expected a mapping, got {data!r}")
    data = dict(data)
    if "type" in data and cls in (SpawnerCfg, ShapeCfg) or (cls is not CollisionProxyCfg and "type" in data
                                                           and data["type"] in SPAWNERS):
        name = data.pop("type")
        if name not in SPAWNERS:
            raise ValueError(f"unknown spawner type {name!r}; known: {sorted(SPAWNERS)}")
        cls = SPAWNERS[name]
    hints = {f.name: f for f in dataclasses.fields(cls)}
    kw = {}
    for k, v in data.items():
        if k not in hints:
            raise ValueError(f"{cls.__name__} has no field {k!r} (fields: {sorted(hints)})")
        kw[k] = _convert(cls, k, hints[k], v)
    return cls(**kw)


_NESTED = {
    ("ArticulationCfg", "spawn"): SpawnerCfg, ("RigidObjectCfg", "spawn"): ShapeCfg,
    ("ArticulationCfg", "init_state"): InitialStateCfg, ("RigidObjectCfg", "init_state"): InitialStateCfg,
    ("ArticulationCfg", "randomize"): RandomizeCfg, ("RigidObjectCfg", "randomize"): RandomizeCfg,
    ("ShapeCfg", "mass_props"): MassPropertiesCfg, ("ShapeCfg", "visual_material"): PreviewSurfaceCfg,
    ("ShapeCfg", "physics_material"): RigidBodyMaterialCfg, ("GroundPlaneCfg", "physics_material"): RigidBodyMaterialCfg,
    ("CameraCfg", "offset"): OffsetCfg, ("CameraCfg", "spawn"): PinholeCameraCfg, ("CameraCfg", "renderer"): RendererCfg,
    ("RayCasterCfg", "offset"): OffsetCfg, ("RayCasterCfg", "pattern_cfg"): LidarPatternCfg,
    ("SceneCfg", "robot"): ArticulationCfg, ("SceneCfg", "background"): BackgroundCfg,
    ("SceneCfg", "physics"): PhysicsCfg, ("SceneCfg", "policy"): PolicyCfg,
    ("BackgroundCfg", "ground"): GroundPlaneCfg, ("BackgroundCfg", "dome"): DomeLightCfg,
    ("BackgroundCfg", "splat"): SplatBackgroundCfg, ("SplatBackgroundCfg", "transform"): SplatTransformCfg,
    ("SplatBackgroundCfg", "collision"): CollisionProxyCfg,
}
_DICT_OF = {("SceneCfg", "objects"): RigidObjectCfg, ("SceneCfg", "cameras"): CameraCfg,
            ("SceneCfg", "sensors"): RayCasterCfg}
_LIST_OF = {("BackgroundCfg", "lights"): DistantLightCfg}


def _owner(cls, name):
    for c in cls.__mro__:
        if (c.__name__, name) in _NESTED:
            return _NESTED[(c.__name__, name)]
    return None


def _convert(cls, name, f, v):
    sub = _owner(cls, name)
    if sub is not None and isinstance(v, dict):
        return _build(sub, v)
    if (cls.__name__, name) in _DICT_OF and isinstance(v, dict):
        return {k: _build(_DICT_OF[(cls.__name__, name)], x) for k, x in v.items()}
    if (cls.__name__, name) in _LIST_OF and isinstance(v, list):
        return [_build(_LIST_OF[(cls.__name__, name)], x) for x in v]
    if isinstance(v, list) and name not in ("data_types", "semantic_tags", "lights", "assets"):
        return tuple(v)
    return v


def from_dict(d: dict, base_dir: str = "") -> SceneCfg:
    cfg = _build(SceneCfg, d)
    cfg.base_dir = base_dir
    return cfg


def load(path: str) -> SceneCfg:
    """Read a scene config (.yaml / .yml / .json)."""
    with open(path) as f:
        text = f.read()
    if path.endswith((".yaml", ".yml")):
        import yaml

        d = yaml.safe_load(text)
    else:
        d = json.loads(text)
    cfg = from_dict(d or {}, os.path.dirname(os.path.abspath(path)))
    if cfg.name == "scene":
        cfg.name = os.path.splitext(os.path.basename(path))[0]
    return cfg


def to_dict(cfg) -> dict:
    return dataclasses.asdict(cfg)

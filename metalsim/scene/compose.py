"""Build a ``SceneCfg`` (``metalsim.scene.cfg``) into a MuJoCo model, a batched simulator and its sensors.

    from metalsim.scene import cfg as C, compose
    scene = compose.compose(C.load("configs/scenes/go2_boxes.yaml"))      # MjSpec -> MjModel -> BatchSim + cameras
    scene.reset()
    for _ in range(100):
        scene.step()                                  # scene.sim.t.ctrl[:] = actions (torch, per env) before, if driving
    imgs = scene.render()                             # {camera: {data_type: (N, H, W, ...) numpy}}
    hits = scene.sense()                              # {lidar: {"range": (N, beams), ...}}

``build_spec(cfg)`` is the CPU part (no GPU): robot (MJCF / URDF / USD), objects as free bodies, ground, lights,
camera and lidar mounts, the splat background's collision proxy. ``ComposedScene`` adds ``BatchSim`` (MuJoCo Warp on
Metal), one renderer per camera (tier 0 / 1 / 2, or the splat renderer compositing tier 0 when the background is a
splat) and the ray-traced lidars. The host paths used by ``render`` / ``sense`` synchronize (tools, films, tests);
training code uses the renderers' GPU paths directly (``scene.cameras[name].renderer.render(scene.sim, v)``).
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field

import numpy as np

from metalsim.scene import cfg as C
from metalsim.scene import proxy as PX

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
POLYHAVEN = os.environ.get("POLYHAVEN", os.path.join(ROOT, "assets", "polyhaven"))

# camera frame conventions (Isaac Lab ``OffsetCfg.convention``) -> MuJoCo's (looks along -z, +y up = "opengl"):
# R_mujoco = R_offset @ CONV[convention]
CONV = {"opengl": np.eye(3),
        "ros": np.diag([1.0, -1.0, -1.0]),                                  # +z forward, +y down
        "world": np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])}   # +x forward, +z up
UP_AXES = {"z": np.eye(3), "y": np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]]),
           "-y": np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]])}         # = metalsim.render.splat.scene.UP_AXES
DATA_TYPES = {"rgb": "rgb", "rgba": "rgb", "distance_to_image_plane": "depth", "depth": "depth",
              "distance_to_camera": "depth", "semantic_segmentation": "seg", "instance_segmentation": "seg",
              "instance_id_segmentation_fast": "seg", "normals": "normal"}


# ------------------------------------------------------------------------------------------------ small math
def quat_xyzw_to_mat(q) -> np.ndarray:
    x, y, z, w = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def mat_to_wxyz(R) -> np.ndarray:
    import mujoco

    q = np.zeros(4); mujoco.mju_mat2Quat(q, np.asarray(R, float).ravel()); return q


def xyzw_to_wxyz(q):
    q = np.asarray(q, float); return np.array([q[3], q[0], q[1], q[2]]) / np.linalg.norm(q)


def euler_xyz_to_wxyz(roll, pitch, yaw) -> np.ndarray:
    """Isaac's ``quat_from_euler_xyz`` (extrinsic x, y, z = R_z(yaw) R_y(pitch) R_x(roll)), w-first."""
    cr, sr, cp, sp, cy, sy = np.cos(roll / 2), np.sin(roll / 2), np.cos(pitch / 2), np.sin(pitch / 2), np.cos(yaw / 2), np.sin(yaw / 2)
    return np.array([cy * cr * cp + sy * sr * sp, cy * sr * cp - sy * cr * sp, cy * cr * sp + sy * sr * cp,
                     sy * cr * cp - cy * sr * sp])


def qmul_wxyz(a, b):
    w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def camera_quat(offset: C.OffsetCfg) -> np.ndarray:
    """MuJoCo camera quaternion (w-first, in the parent body frame) of an Isaac ``OffsetCfg`` rotation."""
    if offset.convention not in CONV:
        raise ValueError(f"camera offset convention {offset.convention!r}: ros | opengl | world")
    return mat_to_wxyz(quat_xyzw_to_mat(offset.rot) @ CONV[offset.convention])


def fovy_deg(cam: C.CameraCfg) -> float:
    """Vertical field of view of a pinhole ``PinholeCameraCfg`` (vertical aperture from the aspect when unset, as
    Isaac Lab does)."""
    s = cam.spawn
    va = s.vertical_aperture or s.horizontal_aperture * cam.height / cam.width
    return float(np.degrees(2 * np.arctan(va / 2 / s.focal_length)))


def splat_transform(t: C.SplatTransformCfg):
    """(A, t) of ``metalsim.render.splat.make_transform`` (same order: up axis, scale, yaw, translate)."""
    c, s = np.cos(np.radians(t.yaw_deg)), np.sin(np.radians(t.yaw_deg))
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return float(t.scale) * Rz @ UP_AXES[t.up], np.asarray(t.translate, float)


def resolve(cfg: C.SceneCfg, p: str | None) -> str | None:
    """A config path: absolute; else relative to the config file, the repo, or the Poly Haven root (hdri/..., tex/...)."""
    if not p:
        return p
    p = os.path.expanduser(p)
    if os.path.isabs(p):
        return p
    for base in (cfg.base_dir, ROOT, POLYHAVEN):
        if base and os.path.exists(os.path.join(base, p)):
            return os.path.join(base, p)
    return os.path.join(cfg.base_dir or ROOT, p)


def have_file(cfg: C.SceneCfg, p: str | None, what: str) -> bool:
    """True if the config file ``p`` exists; else warns (the scene falls back to a plain colour / no dome)."""
    if not p:
        return False
    if os.path.exists(resolve(cfg, p)):
        return True
    import warnings
    warnings.warn(f"{what} {p!r} not found (Poly Haven: scripts/fetch_polyhaven_assets.sh train): left out")
    return False


def mount_body(prim_path: str) -> str | None:
    """Body a sensor is mounted on from an Isaac prim path: "{ENV_REGEX_NS}/Robot/<link>/<name>" -> link;
    "{ENV_REGEX_NS}/<object>/<name>" -> object; "{ENV_REGEX_NS}/<name>" (or "/World/...") -> None (world-fixed)."""
    parts = [x for x in prim_path.replace("{ENV_REGEX_NS}", "").split("/") if x]
    if parts and parts[0] == "World":
        parts = parts[1:]
    if len(parts) >= 2 and parts[0] == "envs" and parts[1].startswith("env_"):
        parts = parts[2:]
    if parts and parts[0] == "Robot":
        return parts[1] if len(parts) >= 3 else None
    return parts[0] if len(parts) >= 2 else None


# ------------------------------------------------------------------------------------------------ spec
@dataclass
class SpecInfo:
    robot_joints: list = field(default_factory=list)      # (name, type) of the robot's joints
    robot_free: str | None = None                          # robot root free joint name
    robot_qpos: dict = field(default_factory=dict)         # joint name -> default value (keyframe / joint_pos)
    robot_ctrl: list | None = None                         # keyframe ctrl
    objects: dict = field(default_factory=dict)            # body name -> (RigidObjectCfg, default pos, default wxyz)
    cameras: dict = field(default_factory=dict)            # camera name -> CameraCfg
    lidars: dict = field(default_factory=dict)             # site name -> RayCasterCfg
    proxy: dict | None = None
    assets: list = field(default_factory=list)             # files the scene read (for provenance)


def _load_robot(cfg: C.SceneCfg, info: SpecInfo):
    import mujoco

    rc = cfg.robot
    sp = rc.spawn
    if isinstance(sp, C.MjcfFileCfg) or isinstance(sp, C.UrdfFileCfg):
        path = resolve(cfg, sp.asset_path)
        spec = mujoco.MjSpec.from_file(path)
        d = os.path.dirname(path)
        # absolute asset directories: the spec is compiled and exported (film twin) from elsewhere
        spec.meshdir = os.path.join(d, spec.meshdir) if spec.meshdir else d
        spec.texturedir = os.path.join(d, spec.texturedir) if spec.texturedir else d
        info.assets.append(path)
    elif isinstance(sp, C.UsdFileCfg):
        from metalsim.scene.usd_to_mjcf import load_usd

        path = resolve(cfg, sp.usd_path)
        spec = load_usd(path, visuals=sp.visuals)
        info.assets.append(path)
    else:
        raise ValueError(f"robot spawn {type(sp).__name__}: MjcfFileCfg | UrdfFileCfg | UsdFileCfg")
    # keyframes stop matching once objects are added: keep the robot's default, then drop them
    key = None
    kname = getattr(sp, "keyframe", None)
    for k in list(spec.keys):
        if key is None and (kname is None or k.name == kname):
            key = k
    for j in spec.joints:
        if j.type == mujoco.mjtJoint.mjJNT_FREE and not j.name:
            j.name = f"{j.parent.name or 'robot'}_freejoint"
    m0 = spec.compile()
    if key is not None and len(key.qpos):
        q = np.asarray(key.qpos, float)
        for j in range(m0.njnt):
            nm = mujoco.mj_id2name(m0, mujoco.mjtObj.mjOBJ_JOINT, j)
            a = m0.jnt_qposadr[j]
            if int(m0.jnt_type[j]) in (2, 3):              # slide, hinge
                info.robot_qpos[nm] = float(q[a])
        if len(key.ctrl):
            info.robot_ctrl = list(key.ctrl)
    for k in list(spec.keys):
        spec.delete(k)
    for j in range(m0.njnt):
        nm = mujoco.mj_id2name(m0, mujoco.mjtObj.mjOBJ_JOINT, j)
        info.robot_joints.append((nm, int(m0.jnt_type[j])))
        if int(m0.jnt_type[j]) == 0 and info.robot_free is None:
            info.robot_free = nm
    for pat, v in (rc.init_state.joint_pos or {}).items():
        for nm, typ in info.robot_joints:
            if typ in (2, 3) and re.fullmatch(pat, nm or ""):
                info.robot_qpos[nm] = float(v)
    # root pose: a fixed base is moved by its body pose, a floating base by its free joint at reset
    root = spec.worldbody.first_body()
    if root is not None and info.robot_free is None:
        root.pos = list(rc.init_state.pos)
        root.quat = list(xyzw_to_wxyz(rc.init_state.rot))
    return spec


def _material(spec, name, vm: C.PreviewSurfaceCfg | None, cfg):
    import mujoco

    if vm is None:
        return None
    m = spec.add_material(name=name, rgba=[*vm.diffuse_color[:3], 1.0])
    m.roughness, m.metallic = float(vm.roughness), float(vm.metallic)
    if have_file(cfg, vm.texture_file, "texture"):
        path = resolve(cfg, vm.texture_file)
        spec.add_texture(name=name + "_tex", type=mujoco.mjtTexture.mjTEXTURE_2D, file=path)
        m.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = name + "_tex"
    return name


def _add_shape(spec, body, gname, sp, cfg):
    import mujoco

    mat = _material(spec, gname + "_mat", sp.visual_material, cfg)
    kw = dict(name=gname, friction=[sp.physics_material.static_friction, 0.005, 0.0001])
    if mat:
        kw["material"] = mat
    if sp.mass_props.mass is not None:
        kw["mass"] = float(sp.mass_props.mass)
    elif sp.mass_props.density is not None:
        kw["density"] = float(sp.mass_props.density)
    if isinstance(sp, C.CuboidCfg):
        kw.update(type=mujoco.mjtGeom.mjGEOM_BOX, size=[s / 2 for s in sp.size])
    elif isinstance(sp, C.SphereCfg):
        kw.update(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[sp.radius, 0, 0])
    elif isinstance(sp, C.CylinderCfg):
        kw.update(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[sp.radius, sp.height / 2, 0])
    elif isinstance(sp, C.CapsuleCfg):
        kw.update(type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[sp.radius, sp.height / 2, 0])
    elif isinstance(sp, C.MeshFileCfg):
        path = resolve(cfg, sp.asset_path)
        spec.add_mesh(name=gname + "_mesh", file=path, scale=list(sp.scale))
        kw.update(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=gname + "_mesh")
    else:
        raise ValueError(f"object spawn {type(sp).__name__}: Cuboid | Sphere | Cylinder | Capsule | MeshFile")
    if not sp.visible:
        kw["group"] = PX.PROXY_GROUP
    if "rgba" not in kw and not mat:
        kw["rgba"] = [0.6, 0.6, 0.6, 1.0]
    return body.add_geom(**kw)


def build_spec(cfg: C.SceneCfg):
    """``SceneCfg`` -> (``mujoco.MjSpec``, ``SpecInfo``); CPU only."""
    import mujoco

    info = SpecInfo()
    spec = _load_robot(cfg, info) if cfg.robot is not None else mujoco.MjSpec()
    if cfg.physics.dt:
        spec.option.timestep = float(cfg.physics.dt)
    bg = cfg.background
    # ground plane
    if bg.ground is not None:
        g = bg.ground
        mat = None
        if have_file(cfg, g.texture_file, "ground texture"):
            path = resolve(cfg, g.texture_file)
            spec.add_texture(name="ground_tex", type=mujoco.mjtTexture.mjTEXTURE_2D, file=path)
            m = spec.add_material(name="ground_mat", rgba=[*g.color[:3], 1.0], texrepeat=[g.texture_repeat] * 2,
                                  texuniform=True)
            m.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "ground_tex"
            mat = "ground_mat"
            info.assets.append(path)
        kw = dict(name="ground", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[g.size[0] / 2, g.size[1] / 2, 0.1],
                  pos=[0, 0, g.height], friction=[g.physics_material.static_friction, 0.005, 0.0001])
        if mat:
            kw["material"] = mat
        else:
            kw["rgba"] = [*g.color[:3], 1.0]
        spec.worldbody.add_geom(**kw)
    # lights
    if bg.lights:
        for l in list(spec.lights):
            spec.delete(l)
        for i, lc in enumerate(bg.lights):
            d = np.asarray(lc.direction, float); d /= np.linalg.norm(d)
            spec.worldbody.add_light(name=f"light_{i}", type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, pos=[0, 0, 5],
                                     dir=list(d), diffuse=list(np.asarray(lc.color) * min(lc.intensity, 1.0)),
                                     castshadow=True)
        spec.visual.headlight.active = 0
    if bg.dome is not None and have_file(cfg, bg.dome.texture_file, "dome HDRI"):
        info.assets.append(resolve(cfg, bg.dome.texture_file))
    # objects
    for name, oc in cfg.objects.items():
        for i in range(max(int(oc.count), 1)):
            bname = name if oc.count <= 1 else f"{name}_{i}"
            pos = np.asarray(oc.init_state.pos, float) + (np.array([0.0, 0.35 * i, 0.0]) if oc.count > 1 else 0.0)
            q = xyzw_to_wxyz(oc.init_state.rot)
            b = spec.worldbody.add_body(name=bname, pos=list(pos), quat=list(q))
            if not oc.kinematic:
                b.add_freejoint(name=bname + "_joint")
            _add_shape(spec, b, bname + "_geom", oc.spawn, cfg)
            info.objects[bname] = (oc, pos, q)
    # cameras
    bodies = {b.name for b in spec.bodies}
    for name, cc in cfg.cameras.items():
        bn = mount_body(cc.prim_path)
        if bn is not None and bn not in bodies:
            raise ValueError(f"camera {name}: no body {bn!r} to mount on (prim_path {cc.prim_path!r})")
        parent = spec.body(bn) if bn else spec.worldbody
        parent.add_camera(name=name, pos=list(cc.offset.pos), quat=list(camera_quat(cc.offset)), fovy=fovy_deg(cc))
        info.cameras[name] = cc
    # lidars: a site whose +x is the beam azimuth 0, +z elevation 90 deg ("world" convention offset)
    for name, sc in cfg.sensors.items():
        bn = mount_body(sc.prim_path) or (sc.prim_path.rstrip("/").split("/")[-1] if "Robot" in sc.prim_path else None)
        parent = spec.body(bn) if bn and bn in bodies else spec.worldbody
        R = quat_xyzw_to_mat(sc.offset.rot)          # the lidar frame itself (Isaac's ray casters: "world" convention)
        parent.add_site(name=name, pos=list(sc.offset.pos), quat=list(mat_to_wxyz(R)), size=[0.01, 0, 0], group=4)
        info.lidars[name] = sc
    # splat background: only its collision proxy goes into the model
    if bg.splat is not None:
        A, t = splat_transform(bg.splat.transform)
        pc = bg.splat.collision
        if pc.mesh_path:
            pc.mesh_path = resolve(cfg, pc.mesh_path)
        if pc.structure_json:
            pc.structure_json = resolve(cfg, pc.structure_json)
        info.proxy = PX.add_proxy(spec, pc, A, t)
        info.assets.append(resolve(cfg, bg.splat.path))
    return spec, info


# ------------------------------------------------------------------------------------------------ runtime
@dataclass
class Camera:
    name: str
    cfg: C.CameraCfg
    renderer: object
    kind: str                       # "tier0" | "tier2" | "splat"
    outputs: tuple


class ComposedScene:
    """The built scene: ``spec`` / ``model``, ``sim`` (BatchSim), ``cameras``, ``lidars``, defaults and resets.
    ``device=None``: no simulator or sensors (the model, its defaults and the reset sampling only; CPU)."""

    def __init__(self, cfg: C.SceneCfg, device: str = "metal:0", renderers: bool = True, log=print):
        import mujoco

        self.cfg, self.log = cfg, log
        t0 = time.perf_counter()
        self.spec, self.info = build_spec(cfg)
        self.model = m = self.spec.compile()
        self.build_s = time.perf_counter() - t0
        self.n = int(cfg.num_envs)
        self.rng = np.random.default_rng(cfg.policy.seed)
        self._defaults()
        self.cameras, self.lidars, self._rt, self.splats = {}, {}, None, None
        self.control_dt = float(m.opt.timestep) * int(cfg.physics.substeps)
        if device is None:                  # model, defaults and reset sampling only (CPU; tests, tools)
            self.sim, self.sim_s = None, 0.0
            return
        from metalsim.physics.batch import BatchSim, BatchSimOptions

        ph = cfg.physics
        opt = BatchSimOptions(substeps=int(ph.substeps), njmax=ph.njmax, nconmax=ph.nconmax,
                              solver_iterations=ph.solver_iterations, ls_iterations=ph.ls_iterations)
        t0 = time.perf_counter()
        self.sim = BatchSim(m, self.n, device=device, options=opt)
        self.sim_s = time.perf_counter() - t0
        if renderers:
            self._build_sensors()
        log(f"[scene] {cfg.name}: {self.n} envs, nbody {m.nbody} ngeom {m.ngeom} nq {m.nq} nu {m.nu}, "
            f"{len(self.cameras)} cameras, {len(self.lidars)} lidars; spec {self.build_s:.2f} s, sim {self.sim_s:.2f} s")
        del mujoco

    # -- defaults and resets ------------------------------------------------------------------------------------
    def _defaults(self):
        import mujoco

        m, info = self.model, self.info
        q = m.qpos0.copy()
        jid = lambda nm: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, nm)
        for nm, v in info.robot_qpos.items():
            j = jid(nm)
            if j >= 0:
                q[m.jnt_qposadr[j]] = v
        self.free = []                    # (qpos adr, dof adr, default pos, default wxyz, RandomizeCfg, init lin/ang vel)
        if info.robot_free:
            rc = self.cfg.robot
            j = jid(info.robot_free)
            self.free.append((m.jnt_qposadr[j], m.jnt_dofadr[j], np.asarray(rc.init_state.pos, float),
                              xyzw_to_wxyz(rc.init_state.rot), rc.randomize, rc.init_state))
        for bname, (oc, pos, quat) in info.objects.items():
            j = jid(bname + "_joint")
            if j >= 0:
                self.free.append((m.jnt_qposadr[j], m.jnt_dofadr[j], pos, quat, oc.randomize, oc.init_state))
        for adr, _, pos, quat, _, _ in self.free:
            q[adr:adr + 3], q[adr + 3:adr + 7] = pos, quat
        self.qpos_default = q
        # controls: keyframe ctrl, else position-type actuators hold their joint's default position
        ctrl = np.zeros(m.nu)
        if info.robot_ctrl is not None and len(info.robot_ctrl) == m.nu:
            ctrl[:] = info.robot_ctrl
        else:
            for a in range(m.nu):
                if int(m.actuator_biastype[a]) == int(mujoco.mjtBias.mjBIAS_AFFINE) and m.actuator_trntype[a] == 0:
                    ctrl[a] = q[m.jnt_qposadr[m.actuator_trnid[a, 0]]]
        self.ctrl_default = ctrl

    def sample_states(self, env_ids):
        """Per-env (qpos, qvel) with Isaac's ``reset_root_state_uniform`` sampling for every randomized root."""
        m = self.model
        k = len(env_ids)
        qpos = np.repeat(self.qpos_default[None], k, 0)
        qvel = np.zeros((k, m.nv))
        for adr, dadr, pos, quat, rnd, st in self.free:
            pr, vr = rnd.pose_range or {}, rnd.velocity_range or {}
            u = lambda key, rr: self.rng.uniform(*rr.get(key, (0.0, 0.0)), size=k)
            dp = np.stack([u("x", pr), u("y", pr), u("z", pr)], 1)
            qpos[:, adr:adr + 3] = pos + dp
            r, p, y = u("roll", pr), u("pitch", pr), u("yaw", pr)
            for i in range(k):
                qpos[i, adr + 3:adr + 7] = qmul_wxyz(quat, euler_xyz_to_wxyz(r[i], p[i], y[i]))
            qvel[:, dadr:dadr + 3] = np.asarray(st.lin_vel) + np.stack([u("x", vr), u("y", vr), u("z", vr)], 1)
            qvel[:, dadr + 3:dadr + 6] = np.asarray(st.ang_vel) + np.stack([u("roll", vr), u("pitch", vr), u("yaw", vr)], 1)
        return qpos, qvel

    def reset(self, env_ids=None):
        ids = list(range(self.n)) if env_ids is None else list(env_ids)
        qpos, qvel = self.sample_states(ids)
        self.sim.set_state(qpos, qvel, worlds=ids)
        self.set_ctrl(None)
        self.sim.synchronize()

    def set_ctrl(self, ctrl=None):
        """Write controls (N, nu) (numpy or torch); None = the scene policy's ("hold" / "zero" / "random")."""
        import torch

        if ctrl is None:
            pol = self.cfg.policy.type
            c = np.repeat(self.ctrl_default[None], self.n, 0)
            if pol == "zero":
                c[:] = 0
            elif pol == "random":
                lo, hi = self.model.actuator_ctrlrange.T
                c = c + self.cfg.policy.scale * self.rng.uniform(-1, 1, c.shape) * (hi - lo) / 2
            ctrl = c
        t = torch.as_tensor(np.asarray(ctrl, np.float32) if not torch.is_tensor(ctrl) else ctrl, device="mps")
        self.sim.t.ctrl[:] = t.reshape(self.n, -1)

    def step(self, ctrl=None) -> int:
        if ctrl is not None or self.cfg.policy.type == "random":
            self.set_ctrl(ctrl)
        return self.sim.step()

    # -- sensors ------------------------------------------------------------------------------------------------
    def _build_sensors(self):
        m, cfg = self.model, self.cfg
        bg = cfg.background
        if bg.splat is not None:
            try:
                from metalsim.render.splat import SplatRenderer, load_scene
            except ImportError as e:
                raise ImportError("splat backgrounds need metalsim.render.splat (the splat renderer, branch "
                                  "splat-renderer)") from e
            A, t = splat_transform(bg.splat.transform)
            self.splats = load_scene(resolve(cfg, bg.splat.path), A, t, crop=bg.splat.crop, log=self.log)
        hdr = None
        if bg.dome is not None and have_file(cfg, bg.dome.texture_file, "dome HDRI"):
            from metalsim.render.hdr import load_hdr
            hdr = load_hdr(resolve(cfg, bg.dome.texture_file)) * float(bg.dome.intensity)
        for name, cc in cfg.cameras.items():
            outs = tuple(dict.fromkeys(DATA_TYPES[d] for d in cc.data_types))
            unknown = [d for d in cc.data_types if d not in DATA_TYPES]
            if unknown:
                raise ValueError(f"camera {name}: unknown data_types {unknown}; known {sorted(DATA_TYPES)}")
            tier = int(cc.renderer.tier)
            if self.splats is not None:
                from metalsim.render.splat import SplatRenderer
                from metalsim.render.tier0 import Tier0Renderer
                mo = tuple(dict.fromkeys(("rgb", "depth") + outs))
                mesh = Tier0Renderer(m, self.n, width=cc.width, height=cc.height, camera=name, outputs=mo, tier=min(tier, 1),
                                     include_planes=False, max_group=3)
                r = SplatRenderer(self.splats, self.n, mesh=mesh, outputs=("rgb", "depth"))
                kind = "splat"
            elif tier == 2:
                from metalsim.render.tier2 import Tier2Renderer, mujoco_scene_kwargs
                if outs != ("rgb",):
                    raise ValueError(f"camera {name}: tier 2 renders rgb only (got {list(cc.data_types)})")
                kw = mujoco_scene_kwargs(m)
                kw.update(headlight=False)
                r = Tier2Renderer(m, self.n, width=cc.width, height=cc.height, camera=name, spp=cc.renderer.spp,
                                  max_bounces=4, **kw)
                if hdr is not None:
                    r.set_environment(hdr)
                kind = "tier2"
            else:
                from metalsim.render.tier0 import Tier0Renderer
                r = Tier0Renderer(m, self.n, width=cc.width, height=cc.height, camera=name, outputs=outs, tier=tier,
                                  far=float(min(cc.spawn.clipping_range[1], 1e4)), near=float(cc.spawn.clipping_range[0]))
                kind = "tier0"
            self.cameras[name] = Camera(name, cc, r, kind, outs)
        if self.info.lidars:
            from metalsim.sensors.raytrace import RayTracer
            maxr = max(sc.max_distance for sc in self.info.lidars.values())
            self._rt = RayTracer(m, self.n, max_range=float(maxr))
            for name, sc in self.info.lidars.items():
                p = sc.pattern_cfg
                el = np.linspace(p.vertical_fov_range[0], p.vertical_fov_range[1], int(p.channels))
                az = np.arange(p.horizontal_fov_range[0], p.horizontal_fov_range[1], p.horizontal_res)
                A_, E_ = np.meshgrid(az, el)
                # name=: the outputs of trace / trace_host are keyed by it (Lidar's default name is "lidar")
                self.lidars[name] = self._rt.make_lidar(name, np.deg2rad(A_.ravel()).astype(np.float32),
                                                         np.deg2rad(E_.ravel()).astype(np.float32), name=name)

    def host_datas(self):
        return [self.sim.get_world(e) for e in range(self.n)]

    def render(self, names=None, datas=None) -> dict:
        """Host path: {camera: {data_type: numpy (N, H, W[, C])}} (synchronizes)."""
        datas = datas or self.host_datas()
        out = {}
        for name, cam in self.cameras.items():
            if names and name not in names:
                continue
            cc = cam.cfg
            if cam.kind == "splat":
                import mujoco
                cid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, name)
                pos = np.stack([d.cam_xpos[cid] for d in datas]); mat = np.stack([d.cam_xmat[cid].reshape(3, 3) for d in datas])
                o = cam.renderer.render_host(pos, cam_mat=mat, mesh_datas=datas)
                if "seg" in cam.outputs or "normal" in cam.outputs:
                    mo = cam.renderer.mesh.out
                    o.update({k: getattr(mo, k).cpu().numpy() for k in ("seg", "normal") if k in cam.outputs})
            elif cam.kind == "tier2":
                o = cam.renderer.render_host(datas, passes=int(cc.renderer.passes))
            else:
                o = cam.renderer.render_host(datas)
            res = {}
            for d in cc.data_types:
                key = DATA_TYPES[d]
                if key not in o:
                    continue
                v = np.asarray(o[key])
                if d == "distance_to_camera":
                    v = planar_to_radial(v, cc.width, cc.height, fovy_deg(cc))
                res[d] = v
            out[name] = res
        return out

    def sense(self, datas=None) -> dict:
        if not self.lidars:
            return {}
        datas = datas or self.host_datas()
        return self._rt.trace_host(datas, list(self.lidars.values()))


def planar_to_radial(depth, width, height, fovy):
    """distance_to_image_plane -> distance_to_camera (Isaac's two depth outputs)."""
    f = (height / 2) / np.tan(np.radians(fovy) / 2)
    u = (np.arange(width) + 0.5 - width / 2) / f
    v = (np.arange(height) + 0.5 - height / 2) / f
    k = np.sqrt(1 + u[None, :] ** 2 + v[:, None] ** 2)
    return depth * k


def compose(cfg: C.SceneCfg, **kw) -> ComposedScene:
    return ComposedScene(cfg, **kw)

"""USD -> MuJoCo (MjSpec) loader (WS1): the physics engine never sees a second scene format.

Two paths:

* **Lossless**: a stage written by ``mjcf_to_usd`` carries the original MJCF in
  ``mjc:source`` customData on the default prim; it is compiled directly.
* **Generic**: any stage using UsdPhysics (Isaac-authored included) is converted prim by prim:
  ``PhysicsRigidBodyAPI`` xforms become bodies (mass/COM/inertia from ``PhysicsMassAPI``),
  UsdPhysics joints become MuJoCo joints (revolute -> hinge, prismatic -> slide, spherical ->
  ball, D6/generic with all axes free -> free; fixed joints merge nothing but attach the child
  rigidly), joint drives become position actuators (stiffness -> kp, damping -> kv, maxForce ->
  forcerange; angular gains converted from per-degree to per-radian), Gprims with
  ``PhysicsCollisionAPI`` become collision geoms (Cube/Sphere/Cylinder/Capsule/Plane/Mesh with
  convex hulls), non-colliding Gprims become visual geoms, ``UsdShade`` materials carry
  diffuseColor/opacity/roughness/metallic, cameras and lights are mapped back. Bodies without a
  joint to their parent are welded (MuJoCo child body with no joint). Unsupported: tendons,
  deformables, articulation force sensors, PhysX-only schemas beyond ``physxJoint:armature`` and
  ``physxJoint:jointFriction`` (both read when present, as are the ``mjc:`` attributes).

Units: metersPerUnit is honoured for positions and sizes; angles in USD are degrees.
"""
from __future__ import annotations

import os
import tempfile

import mujoco
import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade, UsdSemantics


def load_usd(path: str, *, lossless: bool = True, drives: bool = True, visuals: bool = True) -> mujoco.MjSpec:
    stage = Usd.Stage.Open(path)
    root = stage.GetDefaultPrim() or stage.GetPrimAtPath("/World")
    if lossless and root and root.HasCustomDataKey("mjc:source"):
        spec = mujoco.MjSpec.from_string(root.GetCustomDataByKey("mjc:source"))
        base = os.path.dirname(os.path.abspath(path))
        # mesh assets resolve relative to the original MJCF; the importer keeps absolute meshdir
        return spec
    return convert_stage(stage, base_dir=os.path.dirname(os.path.abspath(path)), drives=drives, visuals=visuals)


def _q_from_gf(q) -> np.ndarray:
    return np.array([q.GetReal(), *q.GetImaginary()], float)


def _local_xform(prim, scale_units: float):
    """Translation, wxyz quaternion and scale of a prim's local transform."""
    xf = UsdGeom.Xformable(prim)
    mat = xf.GetLocalTransformation() if xf else Gf.Matrix4d(1.0)
    t = mat.ExtractTranslation()
    rot = mat.ExtractRotationQuat()
    # scale: column norms
    m = np.array(mat).T
    scale = np.linalg.norm(m[:3, :3], axis=0)
    q = np.array([rot.GetReal(), *rot.GetImaginary()], float)
    q /= np.linalg.norm(q) + 1e-12
    return np.array(t, float) * scale_units, q, scale


def _body_of(prim):
    """Nearest ancestor-or-self with RigidBodyAPI, else None (world)."""
    p = prim
    while p and p.IsValid() and not p.IsPseudoRoot():
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            return p
        p = p.GetParent()
    return None


def _world_xform(prim, scale_units):
    """World position, wxyz quaternion and scale of a prim (composes the full USD hierarchy)."""
    pos = np.zeros(3); quat = np.array([1.0, 0, 0, 0]); scale = np.ones(3)
    chain = []
    p = prim
    while p and p.IsValid() and not p.IsPseudoRoot():
        chain.append(p)
        p = p.GetParent()
    for p in reversed(chain):
        t, q, s = _local_xform(p, scale_units)
        R = np.zeros(9); mujoco.mju_quat2Mat(R, quat); R = R.reshape(3, 3)
        pos = pos + R @ (t * scale)
        qn = np.zeros(4); mujoco.mju_mulQuat(qn, quat, q); quat = qn
        scale = scale * s
    return pos, quat, scale


def _accumulate_xform_to(prim, ancestor, scale_units):
    """Transform of ``prim`` relative to ``ancestor`` (position, quaternion, scale). ``ancestor`` need
    not be a USD ancestor: articulations exported from URDF (Isaac's assets) keep all links as
    siblings with world-space transforms, so the relative pose is inv(parent_world) * child_world."""
    pw, qw, sw = _world_xform(prim, scale_units)
    if ancestor is None:
        return pw, qw, sw
    pa, qa, sa = _world_xform(ancestor, scale_units)
    qinv = np.array([qa[0], -qa[1], -qa[2], -qa[3]])
    Ri = np.zeros(9); mujoco.mju_quat2Mat(Ri, qinv); Ri = Ri.reshape(3, 3)
    pos = Ri @ (pw - pa)
    quat = np.zeros(4); mujoco.mju_mulQuat(quat, qinv, qw)
    return pos, quat, sw / np.where(sa == 0, 1, sa)


def convert_stage(stage: Usd.Stage, base_dir: str = ".", drives: bool = True, visuals: bool = True) -> mujoco.MjSpec:
    """``drives=False`` skips turning UsdPhysics drives into actuators (callers that apply their own
    actuator model, as Isaac Lab's ImplicitActuatorCfg does, add them afterwards)."""
    units = float(UsdGeom.GetStageMetersPerUnit(stage) or 1.0)
    spec = mujoco.MjSpec()
    spec.modelname = "usd_import"
    spec.compiler.degree = False
    spec.compiler.boundmass = 1e-6      # floor for bodies without MassAPI; Isaac's assets author 1e-6 for helper links
    spec.compiler.boundinertia = 1e-8
    spec.compiler.balanceinertia = True
    scene = next((p for p in stage.Traverse() if p.IsA(UsdPhysics.Scene)), None)
    if scene is not None:
        ps = UsdPhysics.Scene(scene)
        d = ps.GetGravityDirectionAttr().Get(); mag = ps.GetGravityMagnitudeAttr().Get()
        # UsdPhysics: a zero direction means "use the stage up axis", a negative magnitude means
        # "earth gravity"; PhysX also treats a magnitude of exactly 0 as unauthored (Isaac assets author
        # 0 and rely on the simulator default 9.81), so only a positive magnitude sets the value.
        dv = np.array(d, float) if d is not None else np.zeros(3)
        if np.linalg.norm(dv) == 0:
            dv = np.array([0.0, 0.0, -1.0])
        if mag is not None and mag > 0:
            spec.option.gravity = (dv / np.linalg.norm(dv) * mag * units).tolist()
        else:
            spec.option.gravity = (dv / np.linalg.norm(dv) * 9.81).tolist()
    rp = stage.GetDefaultPrim() or stage.GetPrimAtPath("/World")
    if rp and rp.HasCustomDataKey("mjc:timestep"):
        spec.option.timestep = float(rp.GetCustomDataByKey("mjc:timestep"))

    # -- materials ----------------------------------------------------------------------------
    mat_by_path = {}
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):   # materials may live under instanced prims
        if prim.IsA(UsdShade.Material):
            mat = UsdShade.Material(prim)
            shader = None
            out = mat.GetSurfaceOutput()
            src = out.GetConnectedSource() if out else None
            if not src:   # NVIDIA assets: an MDL surface output (OmniPBR / OmniGlass) instead of UsdPreviewSurface
                out = mat.GetSurfaceOutput("mdl"); src = out.GetConnectedSource() if out else None
            if src:
                shader = UsdShade.Shader(src[0].GetPrim())
            rgba = [0.8, 0.8, 0.8, 1.0]; rough = 0.5; metal = 0.0; tex = None
            specular = 0.5   # MuJoCo material `specular`, read by tier 2 as F0 = 0.08 * specular (OmniPBR specular_level; ior 1.5 -> 0.5)
            mdl = shader.GetPrim().GetAttribute("info:mdl:sourceAsset").Get() if shader else None
            if shader and mdl:
                # OmniPBR parameters (the subset MuJoCo materials can carry): constant colour or diffuse
                # texture, roughness, metallic, opacity
                def _in(name):
                    i = shader.GetInput(name); return i.Get() if i else None
                dc = _in("diffuse_color_constant")
                if dc is not None: rgba[:3] = [float(c) for c in dc]
                tint = _in("diffuse_tint")
                if tint is not None: rgba[:3] = [rgba[k] * float(tint[k]) for k in range(3)]
                dt = _in("diffuse_texture")
                if dt is not None and str(getattr(dt, "path", "")):
                    tex = dt.resolvedPath or os.path.join(base_dir, str(dt.path))
                rr = _in("reflection_roughness_constant"); mt = _in("metallic_constant"); op = _in("opacity_constant")
                if rr is not None: rough = float(rr)
                if mt is not None: metal = float(mt)
                if op is not None: rgba[3] = float(op)
                sl = _in("specular_level")   # OmniPBR: custom_curve_layer weight over a 0.08 -> 1 curve (default 0.5)
                if sl is not None: specular = float(sl)
                ec = _in("emissive_color"); ei = _in("emissive_intensity"); ee = _in("enable_emission")
                emission = float(ei) if (ee and ei is not None) else 0.0
            elif shader:
                emission = 0.0
                inp = shader.GetInput("diffuseColor")
                if inp:
                    csrc = inp.GetConnectedSource()
                    if csrc:
                        tsh = UsdShade.Shader(csrc[0].GetPrim())
                        f = tsh.GetInput("file")
                        if f and f.Get():
                            tex = f.Get().resolvedPath or os.path.join(base_dir, str(f.Get().path))
                        sc = tsh.GetInput("scale")
                        if sc and sc.Get() is not None:
                            rgba = list(sc.Get())
                    elif inp.Get() is not None:
                        rgba[:3] = list(inp.Get())
                op = shader.GetInput("opacity")
                if op and op.Get() is not None:
                    rgba[3] = float(op.Get())
                r = shader.GetInput("roughness"); mt = shader.GetInput("metallic")
                if r and r.Get() is not None: rough = float(r.Get())
                if mt and mt.Get() is not None: metal = float(mt.Get())
                ior = shader.GetInput("ior")   # UsdPreviewSurface metallic workflow: F0 = ((1 - ior) / (1 + ior))^2
                if ior and ior.Get() is not None:
                    specular = float(((1.0 - ior.Get()) / (1.0 + ior.Get())) ** 2 / 0.08)
            else:
                emission = 0.0
            mm = spec.add_material()
            used_mat = set(mat_by_path.values()); base = prim.GetPath().name; name = base; k = 2
            while name in used_mat:                      # instanced assets repeat material names per link
                name = f"{base}_{k}"; k += 1
            mm.name = name                               # MjSpec rejects a repeated name at assignment
            mm.rgba = rgba
            if emission > 0.0:
                mm.emission = min(emission, 10.0)
            mm.roughness = rough
            mm.metallic = metal
            mm.specular = min(max(specular, 0.0), 1.0)
            mm.shininess = max(0.0, 1.0 - rough)
            if prim.HasAttribute("mjc:reflectance"):
                mm.reflectance = float(prim.GetAttribute("mjc:reflectance").Get())
            if prim.HasAttribute("mjc:texrepeat"):
                mm.texrepeat = list(prim.GetAttribute("mjc:texrepeat").Get())
            if prim.HasAttribute("mjc:texuniform"):
                mm.texuniform = bool(prim.GetAttribute("mjc:texuniform").Get())
            if tex and os.path.exists(tex):
                t = spec.add_texture()
                t.name = mm.name + "_tex"; t.type = mujoco.mjtTexture.mjTEXTURE_2D
                t.file = os.path.abspath(tex)
                mm.textures[int(mujoco.mjtTextureRole.mjTEXROLE_RGB)] = t.name
            mat_by_path[prim.GetPath()] = mm.name

    # -- MaterialX materials (.mtlx arcs, outputs:mtlx:surface): flattened into the same table; no-op otherwise
    from metalsim.scene import materialx as _mtlx
    if _mtlx.stage_has_materialx(stage):
        _mtlx.import_materials(stage, spec, mat_by_path, base_dir)

    # -- bodies -----------------------------------------------------------------------------------
    rigid = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    # joints indexed by child body
    joints_by_child = {}
    for p in stage.Traverse():
        if p.IsA(UsdPhysics.Joint):
            j = UsdPhysics.Joint(p)
            b1 = j.GetBody1Rel().GetTargets()
            if b1:
                joints_by_child.setdefault(b1[0], []).append(p)
    # build tree: parent = body0 of the joint, else the nearest rigid ancestor, else world
    parent_of = {}
    for p in rigid:
        js = joints_by_child.get(p.GetPath(), [])
        par = None
        if js:
            b0 = UsdPhysics.Joint(js[0]).GetBody0Rel().GetTargets()
            if b0 and stage.GetPrimAtPath(b0[0]).HasAPI(UsdPhysics.RigidBodyAPI):
                par = b0[0]
        if par is None:
            anc = _body_of(p.GetParent())
            par = anc.GetPath() if anc else None
        parent_of[p.GetPath()] = par
    bodies = {None: spec.worldbody}
    order = []
    remaining = [p.GetPath() for p in rigid]
    while remaining:
        progressed = False
        for path in list(remaining):
            par = parent_of[path]
            if par in bodies:
                prim = stage.GetPrimAtPath(path)
                parent_prim = stage.GetPrimAtPath(par) if par is not None else None
                pos, quat, _ = _accumulate_xform_to(prim, parent_prim, units)
                js_here = joints_by_child.get(path, [])
                if js_here and par is not None:
                    # child pose at the joint's zero configuration: T0 (in parent) * inverse(T1 (in child))
                    j = UsdPhysics.Joint(js_here[0])
                    p0 = np.array(j.GetLocalPos0Attr().Get() or (0, 0, 0), float) * units
                    q0 = _q_from_gf(j.GetLocalRot0Attr().Get() or Gf.Quatf(1, 0, 0, 0))
                    p1 = np.array(j.GetLocalPos1Attr().Get() or (0, 0, 0), float) * units
                    q1 = _q_from_gf(j.GetLocalRot1Attr().Get() or Gf.Quatf(1, 0, 0, 0))
                    q1i = np.array([q1[0], -q1[1], -q1[2], -q1[3]])
                    R0 = np.zeros(9); mujoco.mju_quat2Mat(R0, q0); R0 = R0.reshape(3, 3)
                    R1i = np.zeros(9); mujoco.mju_quat2Mat(R1i, q1i); R1i = R1i.reshape(3, 3)
                    quat = np.zeros(4); mujoco.mju_mulQuat(quat, q0, q1i)
                    pos = p0 - R0 @ (R1i @ p1)
                body = bodies[par].add_body()
                body.name = path.name
                body.pos = pos.tolist(); body.quat = quat.tolist()
                if prim.HasAPI(UsdPhysics.MassAPI):
                    ma = UsdPhysics.MassAPI(prim)
                    mass = ma.GetMassAttr().Get()
                    com = ma.GetCenterOfMassAttr().Get(); di = ma.GetDiagonalInertiaAttr().Get(); pa = ma.GetPrincipalAxesAttr().Get()
                    if mass is not None and mass > 0 and di is not None and max(di) > 0:
                        body.mass = float(mass)
                        # USD's sentinel for "not authored" is (-inf, -inf, -inf): PhysX then uses the body origin
                        com_ok = com is not None and all(np.isfinite(float(c)) for c in com)
                        body.ipos = [float(c) * units for c in com] if com_ok else [0, 0, 0]
                        body.inertia = [float(x) * units * units for x in di]
                        if pa is not None:
                            qa = _q_from_gf(pa)
                            if np.linalg.norm(qa) > 0.5:      # some exporters write (0,0,0,0)
                                body.iquat = (qa / np.linalg.norm(qa)).tolist()
                        body.explicitinertial = True
                bodies[path] = body
                order.append(path)
                remaining.remove(path)
                progressed = True
        if not progressed:
            raise RuntimeError(f"could not order rigid bodies: {remaining}")

    # -- joints and actuators ---------------------------------------------------------------------
    for path in order:
        prim = stage.GetPrimAtPath(path)
        body = bodies[path]
        js = joints_by_child.get(path, [])
        if not js:
            # a rigid body with no joint to its parent: free if it hangs off the world, welded otherwise
            if parent_of[path] is None and not UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr().Get():
                fj = body.add_freejoint(); fj.name = path.name + "_free"
            continue
        for jp in js:
            j = UsdPhysics.Joint(jp)
            if jp.IsA(UsdPhysics.FixedJoint):
                continue
            lp1 = np.array(j.GetLocalPos1Attr().Get() or (0, 0, 0), float) * units
            lr1 = _q_from_gf(j.GetLocalRot1Attr().Get() or Gf.Quatf(1, 0, 0, 0))
            R1 = np.zeros(9); mujoco.mju_quat2Mat(R1, lr1); R1 = R1.reshape(3, 3)
            axis_tok = "X"
            if jp.IsA(UsdPhysics.RevoluteJoint) or jp.IsA(UsdPhysics.PrismaticJoint):
                axis_tok = (UsdPhysics.RevoluteJoint(jp) if jp.IsA(UsdPhysics.RevoluteJoint) else UsdPhysics.PrismaticJoint(jp)).GetAxisAttr().Get() or "X"
            ax = {"X": np.array([1.0, 0, 0]), "Y": np.array([0, 1.0, 0]), "Z": np.array([0, 0, 1.0])}[axis_tok]
            axis_child = R1 @ ax
            jt = body.add_joint()
            jt.name = jp.GetPath().name
            jt.pos = lp1.tolist()
            if jp.IsA(UsdPhysics.RevoluteJoint):
                jt.type = mujoco.mjtJoint.mjJNT_HINGE; jt.axis = axis_child.tolist()
                rj = UsdPhysics.RevoluteJoint(jp)
                lo, hi = rj.GetLowerLimitAttr().Get(), rj.GetUpperLimitAttr().Get()
                if lo is not None and hi is not None and np.isfinite(lo) and np.isfinite(hi) and lo < hi:
                    jt.range = [np.deg2rad(lo), np.deg2rad(hi)]; jt.limited = mujoco.mjtLimited.mjLIMITED_TRUE
            elif jp.IsA(UsdPhysics.PrismaticJoint):
                jt.type = mujoco.mjtJoint.mjJNT_SLIDE; jt.axis = axis_child.tolist()
                pj = UsdPhysics.PrismaticJoint(jp)
                lo, hi = pj.GetLowerLimitAttr().Get(), pj.GetUpperLimitAttr().Get()
                if lo is not None and hi is not None and np.isfinite(lo) and np.isfinite(hi) and lo < hi:
                    jt.range = [lo * units, hi * units]; jt.limited = mujoco.mjtLimited.mjLIMITED_TRUE
            elif jp.IsA(UsdPhysics.SphericalJoint):
                jt.type = mujoco.mjtJoint.mjJNT_BALL
            else:
                # generic D6: treat as free if no limits lock anything (common for floating bases)
                jt.type = mujoco.mjtJoint.mjJNT_FREE
                jt.pos = [0, 0, 0]
            for attr, field in (("mjc:armature", "armature"), ("physxJoint:armature", "armature"),
                                ("mjc:frictionloss", "frictionloss"), ("physxJoint:jointFriction", "frictionloss"),
                                ("mjc:damping", "damping")):
                if jp.HasAttribute(attr) and jp.GetAttribute(attr).Get() is not None:
                    val = float(jp.GetAttribute(attr).Get())
                    cur = getattr(jt, field)
                    if isinstance(cur, np.ndarray):        # damping/stiffness are 3-vectors in MjSpec
                        cur[:] = val
                    else:
                        setattr(jt, field, val)
            # drives -> position actuators
            for inst in ("angular", "linear", "rotX", "rotY", "rotZ", "transX", "transY", "transZ"):
                if not drives or not jp.HasAPI(UsdPhysics.DriveAPI, inst):
                    continue
                drv = UsdPhysics.DriveAPI(jp, inst)
                kp = float(drv.GetStiffnessAttr().Get() or 0.0); kv = float(drv.GetDampingAttr().Get() or 0.0)
                if jt.type == mujoco.mjtJoint.mjJNT_HINGE:
                    kp, kv = kp * 180.0 / np.pi, kv * 180.0 / np.pi
                if kp <= 0 and kv <= 0:
                    continue
                act = spec.add_actuator()
                act.name = jp.GetPath().name
                act.target = jt.name; act.trntype = mujoco.mjtTrn.mjTRN_JOINT
                act.gainprm[0] = kp
                act.biasprm[0] = 0.0; act.biasprm[1] = -kp; act.biasprm[2] = -kv
                act.dyntype = mujoco.mjtDyn.mjDYN_NONE; act.gaintype = mujoco.mjtGain.mjGAIN_FIXED; act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
                mf = drv.GetMaxForceAttr().Get()
                if mf is not None and np.isfinite(mf) and mf > 0:
                    act.forcerange = [-float(mf), float(mf)]; act.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
                if jp.HasAttribute("mjc:ctrlrange"):
                    cr = jp.GetAttribute("mjc:ctrlrange").Get(); act.ctrlrange = [float(cr[0]), float(cr[1])]; act.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
                elif jt.limited == mujoco.mjtLimited.mjLIMITED_TRUE:
                    act.ctrlrange = list(jt.range); act.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE

    # -- geoms, cameras, lights ---------------------------------------------------------------------
    mesh_names = {}
    used_geom_names = set()
    # instance proxies included: Isaac's assets reference their visual meshes as instanceable prims
    # (g1_minimal.usd: 43 visual meshes, 343K faces, materials bound), which stage.Traverse() skips
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        if not visuals and prim.IsA(UsdGeom.Gprim) and not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        purpose = UsdGeom.Imageable(prim).GetPurposeAttr().Get() if prim.IsA(UsdGeom.Imageable) else None
        is_gprim = prim.IsA(UsdGeom.Gprim)
        if not (is_gprim or prim.IsA(UsdGeom.Camera) or prim.IsA(UsdLux.BoundableLightBase) or prim.IsA(UsdLux.NonboundableLightBase)):
            continue
        if prim.GetPath().pathString.startswith("/World/Meshes"):
            continue  # prototypes
        owner = _body_of(prim)
        body = bodies[owner.GetPath()] if owner is not None else spec.worldbody
        pos, quat, scale = _accumulate_xform_to(prim, owner, units)
        if prim.IsA(UsdGeom.Camera):
            cam = body.add_camera(); cam.name = prim.GetPath().name; cam.pos = pos.tolist(); cam.quat = quat.tolist()
            c = UsdGeom.Camera(prim)
            fl = c.GetFocalLengthAttr().Get(); ha = c.GetHorizontalApertureAttr().Get(); va = c.GetVerticalApertureAttr().Get()
            if prim.HasAttribute("mjc:resolution"):
                res = prim.GetAttribute("mjc:resolution").Get(); cam.resolution = [int(res[0]), int(res[1])]
            if fl and va:
                if prim.HasAttribute("mjc:resolution"):
                    cam.sensor_size = [ha / 1000.0, va / 1000.0]; cam.focal_length = [fl / 1000.0, fl / 1000.0]
                else:
                    cam.fovy = float(np.rad2deg(2 * np.arctan(va / (2 * fl))))
            continue
        if prim.IsA(UsdLux.DistantLight) or prim.IsA(UsdLux.SphereLight):
            lt = body.add_light(); lt.name = prim.GetPath().name; lt.pos = pos.tolist()
            R = np.zeros(9); mujoco.mju_quat2Mat(R, quat); R = R.reshape(3, 3)
            lt.dir = (R @ np.array([0, 0, -1.0])).tolist()
            lt.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL if prim.IsA(UsdLux.DistantLight) else mujoco.mjtLightType.mjLIGHT_POINT
            col = np.array(UsdLux.LightAPI(prim).GetColorAttr().Get() or (1, 1, 1), float) * float(UsdLux.LightAPI(prim).GetIntensityAttr().Get() or 1.0)
            lt.diffuse = np.clip(col, 0, 10).tolist()
            for attr, field in (("mjc:ambient", "ambient"), ("mjc:specular", "specular")):
                if prim.HasAttribute(attr):
                    setattr(lt, field, list(prim.GetAttribute(attr).Get()))
            if prim.HasAttribute("mjc:castshadow"):
                lt.castshadow = bool(prim.GetAttribute("mjc:castshadow").Get())
            continue
        # gprim (names are unique in MuJoCo; USD may reuse leaf names like "Cube" under different bodies)
        g = body.add_geom()
        gname = prim.GetPath().name
        if gname in used_geom_names:
            gname = f"{prim.GetParent().GetPath().name}_{gname}"
            k = 2
            while gname in used_geom_names:
                gname = f"{prim.GetParent().GetPath().name}_{prim.GetPath().name}_{k}"; k += 1
        used_geom_names.add(gname)
        g.name = gname; g.pos = pos.tolist(); g.quat = quat.tolist()
        collides = prim.HasAPI(UsdPhysics.CollisionAPI) and (UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not False)
        if prim.IsA(UsdGeom.Cube):
            g.type = mujoco.mjtGeom.mjGEOM_BOX; sz = float(UsdGeom.Cube(prim).GetSizeAttr().Get() or 2.0) / 2
            g.size = (scale * sz * units).tolist()
        elif prim.IsA(UsdGeom.Sphere):
            g.type = mujoco.mjtGeom.mjGEOM_SPHERE; g.size = [float(UsdGeom.Sphere(prim).GetRadiusAttr().Get()) * scale[0] * units, 0, 0]
        elif prim.IsA(UsdGeom.Cylinder) or prim.IsA(UsdGeom.Capsule):
            api = UsdGeom.Cylinder(prim) if prim.IsA(UsdGeom.Cylinder) else UsdGeom.Capsule(prim)
            g.type = mujoco.mjtGeom.mjGEOM_CYLINDER if prim.IsA(UsdGeom.Cylinder) else mujoco.mjtGeom.mjGEOM_CAPSULE
            r = float(api.GetRadiusAttr().Get()) * units; h = float(api.GetHeightAttr().Get()) * units
            g.size = [r * scale[0], h / 2 * scale[2], 0]
            axis = api.GetAxisAttr().Get() or "Z"
            if axis != "Z":   # rotate the geom so its local Z is the USD axis
                qa = np.array([np.cos(np.pi / 4), 0, np.sin(np.pi / 4), 0]) if axis == "X" else np.array([np.cos(np.pi / 4), -np.sin(np.pi / 4), 0, 0])
                qn = np.zeros(4); mujoco.mju_mulQuat(qn, quat, qa); g.quat = qn.tolist()
        elif prim.IsA(UsdGeom.Plane):
            g.type = mujoco.mjtGeom.mjGEOM_PLANE
            pl = UsdGeom.Plane(prim); w = pl.GetWidthAttr().Get(); l = pl.GetLengthAttr().Get()
            g.size = [float(w) / 2 * units if w else 0.0, float(l) / 2 * units if l else 0.0, 0.05]
        elif prim.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(prim)
            pts = np.array(mesh.GetPointsAttr().Get(), np.float64) * units * scale
            counts = np.array(mesh.GetFaceVertexCountsAttr().Get()); idx = np.array(mesh.GetFaceVertexIndicesAttr().Get())
            tris = []
            k = 0
            for c in counts:
                for t in range(1, int(c) - 1):
                    tris.append([idx[k], idx[k + t], idx[k + t + 1]])
                k += int(c)
            key = prim.GetPrototype().GetPath() if prim.IsInstance() else prim.GetPath()
            src = prim.GetReferences()  # internal references resolve through composition; use the path as key
            mname = mesh_names.get(prim.GetPath())
            if mname is None:
                mm = spec.add_mesh(); mm.name = f"mesh_{len(mesh_names)}"
                mm.uservert = pts.reshape(-1).tolist(); mm.userface = np.asarray(tris, np.int32).reshape(-1).tolist()
                pv = UsdGeom.PrimvarsAPI(prim).GetPrimvar("st")
                if pv and pv.HasValue() and pv.GetInterpolation() == UsdGeom.Tokens.faceVarying and all(counts == 3):
                    uv = np.array(pv.Get(), np.float64)
                    mm.usertexcoord = uv.reshape(-1).tolist()
                    mm.userfacetexcoord = np.arange(len(uv), dtype=np.int32).tolist()
                mesh_names[prim.GetPath()] = mm.name; mname = mm.name
            g.type = mujoco.mjtGeom.mjGEOM_MESH; g.meshname = mname
        else:
            continue
        if collides:
            for attr, field, conv in (("mjc:contype", "contype", int), ("mjc:conaffinity", "conaffinity", int), ("mjc:condim", "condim", int),
                                      ("mjc:friction", "friction", list), ("mjc:solref", "solref", list), ("mjc:priority", "priority", int),
                                      ("mjc:margin", "margin", float)):
                if prim.HasAttribute(attr) and prim.GetAttribute(attr).Get() is not None:
                    v = prim.GetAttribute(attr).Get()
                    setattr(g, field, conv(v) if conv is not list else list(v))
        else:
            g.contype = 0; g.conaffinity = 0
        if prim.HasAttribute("mjc:group"):
            g.group = int(prim.GetAttribute("mjc:group").Get())
        elif not collides:
            g.group = 2
        elif purpose == "guide":
            g.group = 3
        mb = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]   # bindings resolve on instance proxies too
        if mb and mb.GetPath() in mat_by_path:
            g.material = mat_by_path[mb.GetPath()]
        else:
            dc = UsdGeom.Gprim(prim).GetDisplayColorAttr().Get(); do = UsdGeom.Gprim(prim).GetDisplayOpacityAttr().Get()
            if dc:
                g.rgba = [*map(float, dc[0]), float(do[0]) if do else 1.0]
        if not collides and not prim.HasAttribute("mjc:group"):
            g.group = 2
    return spec


if __name__ == "__main__":
    import sys
    spec = load_usd(sys.argv[1], lossless="--generic" not in sys.argv)
    m = spec.compile()
    print(f"compiled: nbody {m.nbody} njnt {m.njnt} ngeom {m.ngeom} nu {m.nu} ncam {m.ncam} nlight {m.nlight} nmesh {m.nmesh}")

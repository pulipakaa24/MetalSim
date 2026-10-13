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


EFFORT_LIMIT = "joint"   # where a drive's maxForce goes: "joint" (actfrcrange, default since 2026-09-26) or "actuator" (archived)


def load_usd(path: str, *, lossless: bool = True, drives: bool = True, visuals: bool = True,
             effort_limit: str | None = None) -> mujoco.MjSpec:
    stage = Usd.Stage.Open(path)
    root = stage.GetDefaultPrim() or stage.GetPrimAtPath("/World")
    if lossless and root and root.HasCustomDataKey("mjc:source"):
        spec = mujoco.MjSpec.from_string(root.GetCustomDataByKey("mjc:source"))
        base = os.path.dirname(os.path.abspath(path))
        # mesh assets resolve relative to the original MJCF; the importer keeps absolute meshdir
        return spec
    return convert_stage(stage, base_dir=os.path.dirname(os.path.abspath(path)), drives=drives, visuals=visuals,
                         effort_limit=effort_limit)


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


def _record_dome_light(spec, prim, base_dir, scale_units):
    """A DomeLight has no MuJoCo light type; record it as the custom text ``usd_dome`` (JSON) so the tier-2
    renderer can apply it (``Tier2Renderer.set_environment_from_model``; a textureless dome is the ``dome``
    kwarg). Fields follow UsdLux LightAPI (radiance = texture x intensity x 2^exposure x color) and DomeLight
    (``inputs:texture:file`` resolved to a path, ``inputs:texture:format``, DomeLight_1 ``poleAxis``); ``yaw`` is the
    prim's world rotation about +z (the renderer adds Kit's azimuth, ``tier2.KIT_DOME_AZIMUTH``), ``tilt`` how far the
    authored rotation moves the pole off +z. Kit applies a tilting rotation rigidly (measured 2026-09-26, rotateX 90
    deg: docs/research/isaac_side_session_2026-09-26.md section 2); tier 2 renders only the yaw, so a tilted dome is
    reported here and rendered upright (open gap)."""
    import json
    la = UsdLux.LightAPI(prim)
    val = lambda a, d: (a.Get() if a and a.Get() is not None else d)
    rec = {"prim": prim.GetPath().pathString, "intensity": float(val(la.GetIntensityAttr(), 1.0)),
           "exposure": float(val(la.GetExposureAttr(), 0.0)), "color": [float(c) for c in val(la.GetColorAttr(), (1.0, 1.0, 1.0))]}
    fa = prim.GetAttribute("inputs:texture:file")
    ap = fa.Get() if fa else None
    if ap is not None and (ap.resolvedPath or ap.path):
        path = ap.resolvedPath or ap.path
        if not os.path.isabs(path):
            path = os.path.normpath(os.path.join(base_dir, path))
        rec["file"] = path
        rec["format"] = str(val(prim.GetAttribute("inputs:texture:format"), "automatic"))
    pa = prim.GetAttribute("poleAxis") or prim.GetAttribute("inputs:poleAxis")   # DomeLight_1: uniform token poleAxis
    if pa and pa.Get() is not None:
        rec["pole_axis"] = str(pa.Get())
    _, q, _ = _world_xform(prim, scale_units)
    R = np.zeros(9); mujoco.mju_quat2Mat(R, q); R = R.reshape(3, 3)
    rec["yaw"] = float(np.arctan2(R[1, 0], R[0, 0]))                 # rotation of the dome's x axis about +z
    rec["tilt"] = float(np.degrees(np.arccos(np.clip(R[2, 2], -1, 1))))
    t = spec.add_text(); t.name = "usd_dome"; t.data = json.dumps(rec)
    return rec


LIGHT_DIFFUSE_MAX = 10.0      # the importer's clip on a MuJoCo light's diffuse (SphereLight / DistantLight)


def _area_light(body, prim, pos, quat, scale_units, name):
    """RectLight / DiskLight / CylinderLight -> a MuJoCo light with the area light's far-field radiometry.

    UsdLux: radiance L = intensity x 2^exposure x color (divided by the emitting area when ``inputs:normalize``).
    The renderers' convention (tier 0/1/2, as for the importer's DistantLight: E = pi x diffuse) gives a MuJoCo light
    irradiance E = pi x diffuse x cos(theta_e)^exponent x cos(theta_r) / (c + l d + q d^2). Far field of a Lambertian
    emitter: E = I(theta_e) cos(theta_r) / d^2 with radiant intensity I = L x (area projected toward the receiver):
    * Rect / Disk (emit from one side, toward the prim's -Z): I = L A cos(theta_e) -> a spot light along -Z, cutoff
      90 deg (or ``inputs:shaping:cone:angle``), exponent 1, diffuse = L A / pi, inverse-square attenuation.
    * Cylinder (emits radially, axis along the prim's X, no end caps): I = L 2 r len sin(phi), phi from the axis ->
      a point light with the broadside value I = L 2 r len (isotropic: overestimates toward the axis).
    Diffuse above LIGHT_DIFFUSE_MAX is moved into the quadratic attenuation (diffuse / k, q = 1 / k), which keeps the
    irradiance exact instead of clipping it. Near field (d of a few light sizes) the point source over-estimates the
    on-axis irradiance (exact for a disk: E = pi L r^2 / (d^2 + r^2)); the size is recorded for a renderer that
    samples the true area (``usd_area_lights`` custom text). Returns the record."""
    la = UsdLux.LightAPI(prim)
    val = lambda a, d: (a.Get() if a and a.Get() is not None else d)
    color = np.array(val(la.GetColorAttr(), (1.0, 1.0, 1.0)), float)
    intensity = float(val(la.GetIntensityAttr(), 1.0)); exposure = float(val(la.GetExposureAttr(), 0.0))
    normalize = bool(val(la.GetNormalizeAttr(), False))
    radiance = color * intensity * 2.0 ** exposure
    _, _, ws = _world_xform(prim, scale_units)
    rec = {"prim": prim.GetPath().pathString, "light": name, "intensity": intensity, "exposure": exposure,
           "color": color.tolist(), "normalize": normalize}
    cone = None
    if prim.IsA(UsdLux.RectLight):
        rl = UsdLux.RectLight(prim)
        w = float(val(rl.GetWidthAttr(), 1.0)) * ws[0] * scale_units; h = float(val(rl.GetHeightAttr(), 1.0)) * ws[1] * scale_units
        area = proj = w * h
        rec.update(shape="rect", width=w, height=h)
    elif prim.IsA(UsdLux.DiskLight):
        r = float(val(UsdLux.DiskLight(prim).GetRadiusAttr(), 0.5)) * scale_units
        area = proj = np.pi * (r * ws[0]) * (r * ws[1])
        rec.update(shape="disk", radius=r * float(np.sqrt(ws[0] * ws[1])))
    else:
        cl = UsdLux.CylinderLight(prim)
        r = float(val(cl.GetRadiusAttr(), 0.5)) * float(np.sqrt(ws[1] * ws[2])) * scale_units
        ln = float(val(cl.GetLengthAttr(), 1.0)) * ws[0] * scale_units
        area, proj = 2 * np.pi * r * ln, 2 * r * ln
        rec.update(shape="cylinder", radius=r, length=ln)
    if normalize:
        radiance = radiance / max(area, 1e-12)
    if prim.HasAPI(UsdLux.ShapingAPI):
        ca = UsdLux.ShapingAPI(prim).GetShapingConeAngleAttr().Get()
        if ca is not None and float(ca) < 90.0:
            cone = float(ca)
    I = radiance * proj                                  # far-field radiant intensity (broadside / on axis)
    diffuse = I / np.pi
    k = max(float(diffuse.max()) / LIGHT_DIFFUSE_MAX, 1.0)
    lt = body.add_light(); lt.name = name; lt.pos = pos.tolist()
    R = np.zeros(9); mujoco.mju_quat2Mat(R, quat); R = R.reshape(3, 3)
    lt.dir = (R @ np.array([0, 0, -1.0])).tolist()
    if rec["shape"] == "cylinder":
        lt.type = mujoco.mjtLightType.mjLIGHT_POINT
    else:
        lt.type = mujoco.mjtLightType.mjLIGHT_SPOT
        lt.cutoff = 90.0 if cone is None else cone        # MuJoCo light cutoff is in degrees whatever compiler.degree says
        lt.exponent = 1.0                                 # cos(theta_e): Lambertian emitter
    lt.diffuse = (diffuse / k).tolist()
    lt.attenuation = [0.0, 0.0, 1.0 / k]
    if prim.HasAttribute("mjc:castshadow"):
        lt.castshadow = bool(prim.GetAttribute("mjc:castshadow").Get())
    rec.update(area=float(area), radiance=radiance.tolist(), far_field_intensity=I.tolist(), attenuation_scale=k,
               cone_angle=cone)
    return rec


def _uv_reader_varname(tex_shader):
    """The primvar a UsdUVTexture reads: its ``inputs:st`` connected to a UsdPrimvarReader_float2's ``inputs:varname``."""
    st = tex_shader.GetInput("st")
    src = st.GetConnectedSource() if st else None
    if not src:
        return None
    rd = UsdShade.Shader(src[0].GetPrim())
    vn = rd.GetInput("varname")
    if not vn:
        return None
    v = vn.Get()
    if v is None:                                       # varname may itself be connected to a material interface input
        s2 = vn.GetConnectedSource()
        if s2:
            v = s2[0].GetInput(s2[1]).Get() if hasattr(s2[0], "GetInput") else None
    return str(v) if v else None


def _corner_uvs(prim, counts, idx, npts, hint=None):
    """Per face-vertex ("corner") texture coordinates of a mesh, in MuJoCo's convention, or None.

    Reads ``hint`` (the bound material's UsdPrimvarReader varname), else ``st``, ``st0``, ``UVMap``, ``uv``; any of
    UsdGeom's interpolations: faceVarying (per corner, indexed or not), vertex / varying (per point), uniform (per
    face). USD texture coordinates have (0, 0) at the image's bottom-left (UsdUVTexture); MuJoCo's mesh texcoords
    have v measured from the top row (its OBJ loader flips v), so v -> 1 - v."""
    api = UsdGeom.PrimvarsAPI(prim)
    pv = None
    for n in [hint, "st", "st0", "UVMap", "uv"]:
        if not n:
            continue
        p = api.GetPrimvar(n)
        if p and p.HasValue():
            pv = p
            break
    if pv is None:
        return None
    vals = pv.ComputeFlattened()
    if vals is None:
        return None
    vals = np.asarray(vals, np.float64).reshape(-1, 2)
    interp = pv.GetInterpolation(); nc = int(np.sum(counts))
    if interp == UsdGeom.Tokens.faceVarying and len(vals) == nc:
        uv = vals
    elif interp in (UsdGeom.Tokens.vertex, UsdGeom.Tokens.varying) and len(vals) == npts:
        uv = vals[idx]
    elif interp == UsdGeom.Tokens.uniform and len(vals) == len(counts):
        uv = np.repeat(vals, counts, axis=0)
    else:
        return None
    uv = uv.copy(); uv[:, 1] = 1.0 - uv[:, 1]
    return uv


def _triangulate(counts, idx):
    """Fan triangulation of polygon faces: (T, 3) corner indices (into the face-vertex list) and (T,) source face."""
    corners, face_of = [], []
    k = 0
    for f, c in enumerate(counts):
        c = int(c)
        for t in range(1, c - 1):
            corners.append((k, k + t, k + t + 1)); face_of.append(f)
        k += c
    return np.asarray(corners, np.int64).reshape(-1, 3), np.asarray(face_of, np.int64)


def _material_subsets(prim, nfaces):
    """GeomSubsets of family ``materialBind`` (element type face): [(name, face indices, bound Material or None)], plus
    ("", remaining faces, None) for the faces in no subset (the mesh's own binding). Empty when the mesh has none."""
    subs = UsdGeom.Subset.GetGeomSubsets(UsdGeom.Imageable(prim), UsdGeom.Tokens.face, "materialBind")
    out, used = [], np.zeros(nfaces, bool)
    for s in subs:
        ind = s.GetIndicesAttr().Get()
        ind = np.asarray(ind if ind is not None else [], np.int64)
        ind = ind[(ind >= 0) & (ind < nfaces)]
        if len(ind) == 0:
            continue
        used[ind] = True
        mb = UsdShade.MaterialBindingAPI(s.GetPrim()).ComputeBoundMaterial()[0]
        out.append((s.GetPrim().GetName(), ind, mb))
    if out and not used.all():
        out.append(("", np.flatnonzero(~used), None))
    return out


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


def convert_stage(stage: Usd.Stage, base_dir: str = ".", drives: bool = True, visuals: bool = True,
                  effort_limit: str | None = None) -> mujoco.MjSpec:
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
    mat_uvname = {}       # material path -> the primvar its UsdUVTexture reads (UsdPrimvarReader_float2 varname)
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
                        uv_name = _uv_reader_varname(tsh)
                        if uv_name:
                            mat_uvname[prim.GetPath()] = uv_name
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
                    # UsdPhysics DriveAPI maxForce limits the drive's force on the joint (a joint-level clamp in PhysX,
                    # and Newton maps it to the joint's actuatorfrcrange). On the joint MuJoCo keeps the drive's implicit
                    # velocity derivative while clamped; as an actuator forcerange it drops it (engine_derivative.c
                    # mjd_actuator_vel), which is unstable for stiff, light joints (runs/il3/finger_unit.py)
                    if (effort_limit or EFFORT_LIMIT) == "actuator":
                        act.forcerange = [-float(mf), float(mf)]; act.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
                    else:
                        jt.actfrcrange = [-float(mf), float(mf)]; jt.actfrclimited = mujoco.mjtLimited.mjLIMITED_TRUE
                if jp.HasAttribute("mjc:ctrlrange"):
                    cr = jp.GetAttribute("mjc:ctrlrange").Get(); act.ctrlrange = [float(cr[0]), float(cr[1])]; act.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
                elif jt.limited == mujoco.mjtLimited.mjLIMITED_TRUE:
                    act.ctrlrange = list(jt.range); act.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE

    # -- geoms, cameras, lights ---------------------------------------------------------------------
    mesh_names = {}
    used_geom_names = set()
    area_lights = []
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
        if prim.IsA(UsdLux.DomeLight) or (hasattr(UsdLux, "DomeLight_1") and prim.IsA(UsdLux.DomeLight_1)):
            _record_dome_light(spec, prim, base_dir, units)
            continue
        if prim.IsA(UsdLux.RectLight) or prim.IsA(UsdLux.DiskLight) or prim.IsA(UsdLux.CylinderLight):
            area_lights.append(_area_light(body, prim, pos, quat, units, prim.GetPath().name))
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
        pieces = []           # GeomSubset (materialBind) pieces of a mesh: (subset name, mesh asset, Material or None)
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
            corners, face_of = _triangulate(counts, idx)
            mb0 = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
            subsets = _material_subsets(prim, len(counts))
            hints = [mat_uvname.get(mb0.GetPath()) if mb0 else None] + \
                    [mat_uvname.get(s[2].GetPath()) for s in subsets if s[2]]
            uvc = _corner_uvs(prim, counts, idx, len(pts), next((h for h in hints if h), None))

            def add_mesh_asset(sel, key):
                """MuJoCo mesh of the triangles ``sel`` (compacted vertices, per-corner texcoords)."""
                if key in mesh_names:
                    return mesh_names[key]
                cs = corners[sel]
                vid = idx[cs]
                used, inv = np.unique(vid.reshape(-1), return_inverse=True)
                if key == prim.GetPath():
                    verts, faces = pts, vid                      # whole mesh: the vertex list as authored
                else:
                    verts, faces = pts[used], inv.reshape(-1, 3)
                mm = spec.add_mesh(); mm.name = f"mesh_{len(mesh_names)}"
                mm.uservert = verts.reshape(-1).tolist(); mm.userface = np.asarray(faces, np.int32).reshape(-1).tolist()
                if uvc is not None:
                    tuv = np.round(uvc[cs.reshape(-1)], 7)
                    utc, tinv = np.unique(tuv, axis=0, return_inverse=True)
                    mm.usertexcoord = utc.reshape(-1).tolist()
                    mm.userfacetexcoord = tinv.reshape(-1).astype(np.int32).tolist()
                if key != prim.GetPath():
                    mm.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL   # visual pieces may be flat (one face)
                mesh_names[key] = mm.name
                return mm.name
            g.type = mujoco.mjtGeom.mjGEOM_MESH; g.meshname = add_mesh_asset(slice(None), prim.GetPath())
            for sname, faces_sel, smat in subsets:
                tri_sel = np.flatnonzero(np.isin(face_of, faces_sel))
                if len(tri_sel):
                    pieces.append((sname, add_mesh_asset(tri_sel, (prim.GetPath(), sname)), smat))
        else:
            # unsupported gprim (Cone, Points, BasisCurves, ...): drop the geom just added (left in place it was a zero-size
            # sphere that failed the compile, so a stage holding one did not load at all)
            spec.delete(g); used_geom_names.discard(gname)
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
        if pieces:
            # per-face materials: one visual geom per subset, same body and pose, its own material; the whole mesh stays
            # as one geom for collision and mass (unchanged physics) but is not drawn (alpha 0, no material)
            for sname, mname, smat in pieces:
                pg = body.add_geom()
                pn = f"{gname}_{sname or 'rest'}"
                k = 2
                while pn in used_geom_names:
                    pn = f"{gname}_{sname or 'rest'}_{k}"; k += 1
                used_geom_names.add(pn)
                pg.name = pn; pg.pos = pos.tolist(); pg.quat = quat.tolist()
                pg.type = mujoco.mjtGeom.mjGEOM_MESH; pg.meshname = mname
                pg.contype = 0; pg.conaffinity = 0; pg.density = 0.0; pg.group = g.group
                if smat and smat.GetPath() in mat_by_path:
                    pg.material = mat_by_path[smat.GetPath()]
                elif g.material:
                    pg.material = g.material
                else:
                    pg.rgba = list(g.rgba)
            g.material = ""; g.rgba = [0.5, 0.5, 0.5, 0.0]
    if area_lights:
        import json
        t = spec.add_text(); t.name = "usd_area_lights"; t.data = json.dumps(area_lights)
    return spec


if __name__ == "__main__":
    import sys
    spec = load_usd(sys.argv[1], lossless="--generic" not in sys.argv)
    m = spec.compile()
    print(f"compiled: nbody {m.nbody} njnt {m.njnt} ngeom {m.ngeom} nu {m.nu} ncam {m.ncam} nlight {m.nlight} nmesh {m.nmesh}")

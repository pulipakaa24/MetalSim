"""MJCF -> USD importer (WS1): one stage carrying visuals, UsdPhysics, materials and semantics.

Field mapping follows Isaac Sim's MJCF importer (built on NVIDIA's mujoco-usd-converter), so
assets move between the stacks:

* bodies -> Xform prims with ``PhysicsRigidBodyAPI`` + ``PhysicsMassAPI`` (mass, COM, inertia)
* joints -> ``PhysicsRevoluteJoint`` / ``PhysicsPrismaticJoint`` / ``PhysicsSphericalJoint`` /
  free joints are the absence of a joint; ``PhysicsArticulationRootAPI`` on each kinematic tree
  root; limits from ``range`` (degrees for revolute), ``PhysicsDriveAPI`` from actuators with
  fixed gain / affine bias (position: stiffness=kp, damping=kv; velocity/motor accordingly);
  ``frictionloss`` and ``armature`` kept as ``mjc:`` namespaced attributes (Isaac keeps them as
  PhysxJointAPI.jointFriction / armature; both are written)
* geoms -> Gprims (Cube/Sphere/Cylinder/Capsule/Plane/Mesh) with ``PhysicsCollisionAPI`` on
  collision geoms (contype/conaffinity != 0) and ``PhysicsMeshCollisionAPI`` (convexHull) on mesh
  colliders; visual-only geoms have no collision API
* materials -> ``UsdShade.Material`` with ``UsdPreviewSurface`` (diffuseColor, roughness from
  shininess, metallic, emissive, opacity) and a texture reader for MuJoCo 2D textures written as
  PNG next to the stage
* semantics -> ``UsdSemantics.LabelsAPI`` ("class": body name / geom name)
* cameras -> ``UsdGeom.Camera`` (focal length / apertures from sensorsize+focal or fovy)
* lights -> ``UsdLux.DistantLight`` / ``SphereLight`` (directional / point)
* the whole MJCF text is kept as ``mjc:source`` customData for lossless round trip

Units: MuJoCo is metres/kilograms/radians; the stage is authored with metersPerUnit = 1, Z up.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import mujoco
import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdSemantics, UsdShade, Vt


def _name(m, objtype, i, fallback):
    n = mujoco.mj_id2name(m, objtype, i)
    n = n if n else f"{fallback}_{i}"
    return "".join(c if c.isalnum() or c == "_" else "_" for c in n).lstrip("0123456789") or f"{fallback}_{i}"


def _quat_gf(q):  # MuJoCo (w,x,y,z) -> Gf.Quatf
    return Gf.Quatf(float(q[0]), float(q[1]), float(q[2]), float(q[3]))


def _set_xform(prim, pos, quat):
    x = UsdGeom.Xformable(prim)
    x.ClearXformOpOrder()
    x.AddTranslateOp().Set(Gf.Vec3d(*map(float, pos)))
    x.AddOrientOp().Set(_quat_gf(quat))


@dataclass
class ImportResult:
    stage: Usd.Stage
    body_paths: dict
    geom_paths: dict
    joint_paths: dict


def import_mjcf(mjcf_path: str, usd_path: str, *, write_textures: bool = True) -> ImportResult:
    spec = mujoco.MjSpec.from_file(mjcf_path)
    m = spec.compile()
    # lossless round trip: the flattened model (includes resolved) with absolute asset directories
    mjcf_dir = os.path.dirname(os.path.abspath(mjcf_path))
    spec.meshdir = os.path.join(mjcf_dir, spec.meshdir) if not os.path.isabs(spec.meshdir or "") else spec.meshdir
    spec.texturedir = os.path.join(mjcf_dir, spec.texturedir) if not os.path.isabs(spec.texturedir or "") else spec.texturedir
    flattened_xml = spec.to_xml()
    stage = Usd.Stage.CreateNew(usd_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(root.GetPrim())
    root.GetPrim().SetCustomDataByKey("mjc:source", flattened_xml)
    root.GetPrim().SetCustomDataByKey("mjc:source_path", os.path.abspath(mjcf_path))
    root.GetPrim().SetCustomDataByKey("mjc:timestep", float(m.opt.timestep))

    scene = UsdPhysics.Scene.Define(stage, "/World/physicsScene")
    g = np.asarray(m.opt.gravity)
    gn = np.linalg.norm(g)
    scene.CreateGravityDirectionAttr(Gf.Vec3f(*map(float, g / gn)) if gn > 0 else Gf.Vec3f(0, 0, -1))
    scene.CreateGravityMagnitudeAttr(float(gn))

    # -- materials and textures ----------------------------------------------------------------
    looks = UsdGeom.Scope.Define(stage, "/World/Looks")
    tex_files = {}
    out_dir = os.path.dirname(os.path.abspath(usd_path))
    if write_textures:
        for t in range(m.ntex):
            if int(m.tex_type[t]) == int(mujoco.mjtTexture.mjTEXTURE_SKYBOX):
                continue
            w, h, nc = int(m.tex_width[t]), int(m.tex_height[t]), int(m.tex_nchannel[t])
            adr = int(m.tex_adr[t])
            img = m.tex_data[adr:adr + w * h * nc].reshape(h, w, nc)[::-1]
            tname = _name(m, mujoco.mjtObj.mjOBJ_TEXTURE, t, "texture")
            fn = f"{tname}.png"
            try:
                from PIL import Image
                Image.fromarray(np.ascontiguousarray(img if nc in (3, 4) else np.repeat(img, 3, 2))).save(os.path.join(out_dir, fn))
                tex_files[t] = fn
            except ImportError:
                pass
    mat_paths = {}
    for i in range(m.nmat):
        mname = _name(m, mujoco.mjtObj.mjOBJ_MATERIAL, i, "material")
        mp = f"/World/Looks/{mname}"
        mat = UsdShade.Material.Define(stage, mp)
        sh = UsdShade.Shader.Define(stage, mp + "/Shader")
        sh.CreateIdAttr("UsdPreviewSurface")
        rgba = m.mat_rgba[i]
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*map(float, rgba[:3])))
        sh.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(rgba[3]))
        rough = float(m.mat_roughness[i]) if hasattr(m, "mat_roughness") and m.mat_roughness[i] >= 0 else 1.0 - float(m.mat_shininess[i])
        metal = float(m.mat_metallic[i]) if hasattr(m, "mat_metallic") and m.mat_metallic[i] >= 0 else 0.0
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(max(rough, 0.02))
        sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metal)
        sh.CreateInput("specular", Sdf.ValueTypeNames.Float).Set(float(m.mat_specular[i]))
        em = float(m.mat_emission[i])
        sh.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*(float(c) * em for c in rgba[:3])))
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        # MuJoCo-specific: reflectance, texrepeat, texuniform
        prim = mat.GetPrim()
        prim.CreateAttribute("mjc:reflectance", Sdf.ValueTypeNames.Float).Set(float(m.mat_reflectance[i]))
        prim.CreateAttribute("mjc:texrepeat", Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(*map(float, m.mat_texrepeat[i])))
        prim.CreateAttribute("mjc:texuniform", Sdf.ValueTypeNames.Bool).Set(bool(m.mat_texuniform[i]))
        roles = np.asarray(m.mat_texid[i]).reshape(-1)
        texid = int(roles[1]) if len(roles) > 1 else int(roles[0])
        if texid >= 0 and texid in tex_files:
            st = UsdShade.Shader.Define(stage, mp + "/stReader")
            st.CreateIdAttr("UsdPrimvarReader_float2")
            st.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
            tx = UsdShade.Shader.Define(stage, mp + "/Texture")
            tx.CreateIdAttr("UsdUVTexture")
            tx.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(f"./{tex_files[texid]}")
            tx.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st.ConnectableAPI(), "result")
            tx.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
            tx.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
            tx.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*map(float, rgba)))
            sh.GetInput("diffuseColor").ConnectToSource(tx.ConnectableAPI(), "rgb")
        # physics material (friction from the geoms that use it is per geom in MuJoCo; default here)
        UsdPhysics.MaterialAPI.Apply(prim)
        mat_paths[i] = mp

    # -- bodies (kinematic tree) ------------------------------------------------------------------
    body_paths = {0: "/World"}
    geom_paths, joint_paths = {}, {}
    order = sorted(range(1, m.nbody), key=lambda b: (m.body_parentid[b] == 0, b))
    # define in tree order: parents first
    remaining = list(range(1, m.nbody))
    while remaining:
        progressed = False
        for b in list(remaining):
            p = int(m.body_parentid[b])
            if p in body_paths:
                bname = _name(m, mujoco.mjtObj.mjOBJ_BODY, b, "body")
                path = f"{body_paths[p]}/{bname}"
                xf = UsdGeom.Xform.Define(stage, path)
                _set_xform(xf.GetPrim(), m.body_pos[b], m.body_quat[b])
                prim = xf.GetPrim()
                UsdPhysics.RigidBodyAPI.Apply(prim)
                mass = UsdPhysics.MassAPI.Apply(prim)
                mass.CreateMassAttr(float(m.body_mass[b]))
                mass.CreateCenterOfMassAttr(Gf.Vec3f(*map(float, m.body_ipos[b])))
                mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*map(float, m.body_inertia[b])))
                mass.CreatePrincipalAxesAttr(_quat_gf(m.body_iquat[b]))
                if p == 0 and int(m.body_jntnum[b]) == 0:
                    # static (world-attached, no joint): kinematic body
                    UsdPhysics.RigidBodyAPI(prim).CreateKinematicEnabledAttr(True)
                if int(m.body_rootid[b]) == b or (p == 0 and int(m.body_jntnum[b]) > 0):
                    if int(m.body_jntnum[b]) == 1 and int(m.jnt_type[m.body_jntadr[b]]) == int(mujoco.mjtJoint.mjJNT_FREE):
                        pass  # floating base: free body, no articulation root needed
                    else:
                        UsdPhysics.ArticulationRootAPI.Apply(prim)
                UsdSemantics.LabelsAPI.Apply(prim, "class").CreateLabelsAttr(Vt.TokenArray([bname]))
                prim.CreateAttribute("mjc:body_id", Sdf.ValueTypeNames.Int).Set(int(b))
                body_paths[b] = path
                remaining.remove(b)
                progressed = True
        if not progressed:
            raise RuntimeError("body tree could not be ordered")

    # -- joints ---------------------------------------------------------------------------------------
    act_by_joint = {}
    for a in range(m.nu):
        if int(m.actuator_trntype[a]) == int(mujoco.mjtTrn.mjTRN_JOINT):
            act_by_joint.setdefault(int(m.actuator_trnid[a][0]), []).append(a)
    for j in range(m.njnt):
        jt = int(m.jnt_type[j])
        b = int(m.jnt_bodyid[j])
        p = int(m.body_parentid[b])
        jname = _name(m, mujoco.mjtObj.mjOBJ_JOINT, j, "joint")
        jpath = f"{body_paths[b]}/{jname}"
        if jt == int(mujoco.mjtJoint.mjJNT_FREE):
            continue
        axis = np.asarray(m.jnt_axis[j], float)
        if jt == int(mujoco.mjtJoint.mjJNT_HINGE):
            joint = UsdPhysics.RevoluteJoint.Define(stage, jpath)
            if m.jnt_limited[j]:
                lo, hi = np.rad2deg(m.jnt_range[j])
                joint.CreateLowerLimitAttr(float(lo)); joint.CreateUpperLimitAttr(float(hi))
        elif jt == int(mujoco.mjtJoint.mjJNT_SLIDE):
            joint = UsdPhysics.PrismaticJoint.Define(stage, jpath)
            if m.jnt_limited[j]:
                joint.CreateLowerLimitAttr(float(m.jnt_range[j][0])); joint.CreateUpperLimitAttr(float(m.jnt_range[j][1]))
        else:  # ball
            joint = UsdPhysics.SphericalJoint.Define(stage, jpath)
        # joint frame: at jnt_pos in the child body, with the joint axis mapped to X (UsdPhysics convention)
        a = axis / (np.linalg.norm(axis) + 1e-12)
        x = np.array([1.0, 0, 0])
        v = np.cross(x, a); s = np.linalg.norm(v); c = float(np.dot(x, a))
        if s < 1e-9:
            q_axis = np.array([1, 0, 0, 0.0]) if c > 0 else np.array([0, 0, 0, 1.0])
        else:
            v /= s; ang = np.arctan2(s, c)
            q_axis = np.concatenate([[np.cos(ang / 2)], np.sin(ang / 2) * v])
        if jt != int(mujoco.mjtJoint.mjJNT_BALL):
            joint.CreateAxisAttr("X")
        jpos_child = np.asarray(m.jnt_pos[j], float)
        # child frame: body-local; parent frame: same point expressed in the parent body
        Rb = np.zeros(9); mujoco.mju_quat2Mat(Rb, m.body_quat[b]); Rb = Rb.reshape(3, 3)
        jpos_parent = np.asarray(m.body_pos[b], float) + Rb @ jpos_child
        q_parent = np.zeros(4); mujoco.mju_mulQuat(q_parent, m.body_quat[b], q_axis)
        joint.CreateBody0Rel().SetTargets([body_paths[p]])
        joint.CreateBody1Rel().SetTargets([body_paths[b]])
        joint.CreateLocalPos0Attr(Gf.Vec3f(*map(float, jpos_parent)))
        joint.CreateLocalRot0Attr(_quat_gf(q_parent))
        joint.CreateLocalPos1Attr(Gf.Vec3f(*map(float, jpos_child)))
        joint.CreateLocalRot1Attr(_quat_gf(q_axis))
        jprim = joint.GetPrim()
        dofadr = int(m.jnt_dofadr[j])
        jprim.CreateAttribute("mjc:armature", Sdf.ValueTypeNames.Float).Set(float(m.dof_armature[dofadr]))
        jprim.CreateAttribute("mjc:frictionloss", Sdf.ValueTypeNames.Float).Set(float(m.dof_frictionloss[dofadr]))
        jprim.CreateAttribute("mjc:damping", Sdf.ValueTypeNames.Float).Set(float(m.dof_damping[dofadr]))
        jprim.CreateAttribute("physxJoint:armature", Sdf.ValueTypeNames.Float).Set(float(m.dof_armature[dofadr]))
        jprim.CreateAttribute("physxJoint:jointFriction", Sdf.ValueTypeNames.Float).Set(float(m.dof_frictionloss[dofadr]))
        # drives from actuators (fixed gain + affine bias, as Isaac's importer supports)
        for ai in act_by_joint.get(j, []):
            gp = m.actuator_gainprm[ai]; bp = m.actuator_biasprm[ai]
            gear = float(m.actuator_gear[ai][0])
            inst = "angular" if jt == int(mujoco.mjtJoint.mjJNT_HINGE) else "linear"
            drive = UsdPhysics.DriveAPI.Apply(jprim, inst)
            drive.CreateTypeAttr("force")
            kp = float(-bp[1] * gear) if bp[1] < 0 else 0.0
            kv = float(-bp[2] * gear) if bp[2] < 0 else float(m.dof_damping[dofadr])
            if jt == int(mujoco.mjtJoint.mjJNT_HINGE):   # USD angular drives are per degree
                kp, kv = kp * np.pi / 180.0, kv * np.pi / 180.0
            drive.CreateStiffnessAttr(kp)
            drive.CreateDampingAttr(kv)
            fr = m.actuator_forcerange[ai]
            if m.actuator_forcelimited[ai]:
                drive.CreateMaxForceAttr(float(max(abs(fr[0]), abs(fr[1])) * gear))
            jprim.CreateAttribute("mjc:actuator", Sdf.ValueTypeNames.String).Set(_name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, ai, "actuator"))
            cr = m.actuator_ctrlrange[ai]
            jprim.CreateAttribute("mjc:ctrlrange", Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(float(cr[0]), float(cr[1])))
        UsdSemantics.LabelsAPI.Apply(jprim, "class").CreateLabelsAttr(Vt.TokenArray([jname]))
        joint_paths[j] = jpath

    # -- geoms ---------------------------------------------------------------------------------------
    mesh_paths = {}
    meshes_scope = UsdGeom.Scope.Define(stage, "/World/Meshes")
    for g in range(m.ngeom):
        b = int(m.geom_bodyid[g])
        gname = _name(m, mujoco.mjtObj.mjOBJ_GEOM, g, "geom")
        gpath = f"{body_paths[b]}/{gname}"
        gt = int(m.geom_type[g]); size = m.geom_size[g]
        if gt == int(mujoco.mjtGeom.mjGEOM_BOX):
            gp = UsdGeom.Cube.Define(stage, gpath); gp.CreateSizeAttr(2.0)
            UsdGeom.Xformable(gp).AddScaleOp().Set(Gf.Vec3f(*map(float, size[:3])))
            prim = gp.GetPrim()
        elif gt == int(mujoco.mjtGeom.mjGEOM_SPHERE):
            gp = UsdGeom.Sphere.Define(stage, gpath); gp.CreateRadiusAttr(float(size[0])); prim = gp.GetPrim()
        elif gt == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
            gp = UsdGeom.Cylinder.Define(stage, gpath); gp.CreateRadiusAttr(float(size[0])); gp.CreateHeightAttr(float(2 * size[1])); gp.CreateAxisAttr("Z"); prim = gp.GetPrim()
        elif gt == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
            gp = UsdGeom.Capsule.Define(stage, gpath); gp.CreateRadiusAttr(float(size[0])); gp.CreateHeightAttr(float(2 * size[1])); gp.CreateAxisAttr("Z"); prim = gp.GetPrim()
        elif gt == int(mujoco.mjtGeom.mjGEOM_PLANE):
            gp = UsdGeom.Plane.Define(stage, gpath); gp.CreateAxisAttr("Z")
            gp.CreateWidthAttr(float(2 * size[0]) if size[0] > 0 else 100.0); gp.CreateLengthAttr(float(2 * size[1]) if size[1] > 0 else 100.0)
            prim = gp.GetPrim()
        elif gt == int(mujoco.mjtGeom.mjGEOM_MESH):
            mid = int(m.geom_dataid[g])
            if mid not in mesh_paths:
                mname = _name(m, mujoco.mjtObj.mjOBJ_MESH, mid, "mesh")
                mpath = f"/World/Meshes/{mname}"
                mesh = UsdGeom.Mesh.Define(stage, mpath)
                va, vn = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid]); fa, fn = int(m.mesh_faceadr[mid]), int(m.mesh_facenum[mid])
                verts = m.mesh_vert[va:va + vn]; faces = m.mesh_face[fa:fa + fn]
                mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(verts, np.float32)))
                mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * fn))
                mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.ascontiguousarray(faces.reshape(-1), np.int32)))
                mesh.CreateSubdivisionSchemeAttr("none")
                tca = int(m.mesh_texcoordadr[mid])
                if tca >= 0:
                    ftc = m.mesh_facetexcoord[fa:fa + fn].reshape(-1)
                    uv = m.mesh_texcoord[tca + ftc]
                    pv = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
                    pv.Set(Vt.Vec2fArray.FromNumpy(np.ascontiguousarray(uv, np.float32)))
                mesh_paths[mid] = mpath
            # instance the mesh under the body via a reference (internal)
            prim = stage.DefinePrim(gpath, "Mesh")
            prim.GetReferences().AddInternalReference(mesh_paths[mid])
        else:
            continue  # hfield, sdf, ellipsoid: not yet
        _set_xform(prim, m.geom_pos[g], m.geom_quat[g])
        if gt == int(mujoco.mjtGeom.mjGEOM_BOX):
            UsdGeom.Xformable(prim).AddScaleOp().Set(Gf.Vec3f(*map(float, size[:3])))
        gprim = UsdGeom.Gprim(prim)
        mat = int(m.geom_matid[g])
        if mat >= 0:
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(UsdShade.Material(stage.GetPrimAtPath(mat_paths[mat])))
        else:
            gprim.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, m.geom_rgba[g][:3]))]))
        gprim.CreateDisplayOpacityAttr(Vt.FloatArray([float(m.geom_rgba[g][3])]))
        collides = int(m.geom_contype[g]) != 0 or int(m.geom_conaffinity[g]) != 0
        if collides:
            UsdPhysics.CollisionAPI.Apply(prim)
            if gt == int(mujoco.mjtGeom.mjGEOM_MESH):
                UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("convexHull")
            prim.CreateAttribute("mjc:contype", Sdf.ValueTypeNames.Int).Set(int(m.geom_contype[g]))
            prim.CreateAttribute("mjc:conaffinity", Sdf.ValueTypeNames.Int).Set(int(m.geom_conaffinity[g]))
            prim.CreateAttribute("mjc:condim", Sdf.ValueTypeNames.Int).Set(int(m.geom_condim[g]))
            prim.CreateAttribute("mjc:friction", Sdf.ValueTypeNames.Float3).Set(Gf.Vec3f(*map(float, m.geom_friction[g])))
            prim.CreateAttribute("mjc:solref", Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(*map(float, m.geom_solref[g][:2])))
            prim.CreateAttribute("mjc:priority", Sdf.ValueTypeNames.Int).Set(int(m.geom_priority[g]))
            prim.CreateAttribute("mjc:margin", Sdf.ValueTypeNames.Float).Set(float(m.geom_margin[g]))
        else:
            prim.CreateAttribute("physics:collisionEnabled", Sdf.ValueTypeNames.Bool).Set(False)
        if int(m.geom_group[g]) >= 3:
            UsdGeom.Imageable(prim).CreatePurposeAttr("guide")   # collision-only geoms are not rendered
        prim.CreateAttribute("mjc:group", Sdf.ValueTypeNames.Int).Set(int(m.geom_group[g]))
        prim.CreateAttribute("mjc:geom_id", Sdf.ValueTypeNames.Int).Set(int(g))
        UsdSemantics.LabelsAPI.Apply(prim, "class").CreateLabelsAttr(Vt.TokenArray([_name(m, mujoco.mjtObj.mjOBJ_BODY, b, "body")]))
        UsdSemantics.LabelsAPI.Apply(prim, "geom").CreateLabelsAttr(Vt.TokenArray([gname]))
        geom_paths[g] = gpath

    # -- sites, cameras, lights --------------------------------------------------------------------
    for c in range(m.ncam):
        b = int(m.cam_bodyid[c])
        cname = _name(m, mujoco.mjtObj.mjOBJ_CAMERA, c, "camera")
        cam = UsdGeom.Camera.Define(stage, f"{body_paths[b]}/{cname}")
        _set_xform(cam.GetPrim(), m.cam_pos[c], m.cam_quat[c])
        ss = m.cam_sensorsize[c]
        if ss[0] > 0 and ss[1] > 0:
            focal = m.cam_intrinsic[c][:2]
            cam.CreateFocalLengthAttr(float(focal[0] * 1000))              # mm
            cam.CreateHorizontalApertureAttr(float(ss[0] * 1000)); cam.CreateVerticalApertureAttr(float(ss[1] * 1000))
        else:
            fovy = float(m.cam_fovy[c]); ap = 24.0
            cam.CreateVerticalApertureAttr(ap); cam.CreateHorizontalApertureAttr(ap)
            cam.CreateFocalLengthAttr(float(ap / (2 * np.tan(np.deg2rad(fovy) / 2))))
        cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
        res = m.cam_resolution[c]
        cam.GetPrim().CreateAttribute("mjc:resolution", Sdf.ValueTypeNames.Int2).Set(Gf.Vec2i(int(res[0]), int(res[1])))
        # MuJoCo cameras look along -Z with +Y up, which is USD's camera convention: no extra rotation
    for l in range(m.nlight):
        b = int(m.light_bodyid[l])
        lname = _name(m, mujoco.mjtObj.mjOBJ_LIGHT, l, "light")
        lpath = f"{body_paths[b]}/{lname}"
        if m.light_type[l] == int(mujoco.mjtLightType.mjLIGHT_DIRECTIONAL) if hasattr(mujoco, "mjtLightType") else m.light_directional[l]:
            light = UsdLux.DistantLight.Define(stage, lpath)
        else:
            light = UsdLux.SphereLight.Define(stage, lpath)
            light.CreateRadiusAttr(0.05)
        d = np.asarray(m.light_dir[l], float); d /= (np.linalg.norm(d) + 1e-12)
        # USD lights emit along -Z of their frame: rotate -Z onto d
        z = np.array([0, 0, -1.0]); v = np.cross(z, d); s = np.linalg.norm(v); cth = float(np.dot(z, d))
        if s < 1e-9:
            q = np.array([1, 0, 0, 0.0]) if cth > 0 else np.array([0, 1, 0, 0.0])
        else:
            v /= s; ang = np.arctan2(s, cth); q = np.concatenate([[np.cos(ang / 2)], np.sin(ang / 2) * v])
        _set_xform(light.GetPrim(), m.light_pos[l], q)
        diff = np.asarray(m.light_diffuse[l], float)
        light.CreateColorAttr(Gf.Vec3f(*map(float, diff / max(diff.max(), 1e-6))))
        light.CreateIntensityAttr(float(diff.max()))
        light.GetPrim().CreateAttribute("mjc:castshadow", Sdf.ValueTypeNames.Bool).Set(bool(m.light_castshadow[l]))
        light.GetPrim().CreateAttribute("mjc:ambient", Sdf.ValueTypeNames.Float3).Set(Gf.Vec3f(*map(float, m.light_ambient[l])))
        light.GetPrim().CreateAttribute("mjc:specular", Sdf.ValueTypeNames.Float3).Set(Gf.Vec3f(*map(float, m.light_specular[l])))
    for s_ in range(m.nsite):
        b = int(m.site_bodyid[s_])
        sname = _name(m, mujoco.mjtObj.mjOBJ_SITE, s_, "site")
        xf = UsdGeom.Xform.Define(stage, f"{body_paths[b]}/{sname}")
        _set_xform(xf.GetPrim(), m.site_pos[s_], m.site_quat[s_])
        xf.GetPrim().CreateAttribute("mjc:site", Sdf.ValueTypeNames.Bool).Set(True)
    stage.GetRootLayer().Save()
    return ImportResult(stage, body_paths, geom_paths, joint_paths)


if __name__ == "__main__":
    import sys
    r = import_mjcf(sys.argv[1], sys.argv[2])
    print(f"wrote {sys.argv[2]}: {len(r.body_paths) - 1} bodies, {len(r.joint_paths)} joints, {len(r.geom_paths)} geoms")

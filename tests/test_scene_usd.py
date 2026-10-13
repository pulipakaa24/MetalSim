"""WS1: MJCF -> USD import carries physics, materials, semantics and cameras."""
import os

import mujoco
import numpy as np
import pytest

from metalsim.scene.mjcf_to_usd import import_mjcf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SO101 = os.path.join(ROOT, "assets", "so101", "scene_box_rl.xml")


def test_so101_import(tmp_path):
    from pxr import Usd, UsdGeom, UsdPhysics, UsdSemantics, UsdShade
    out = str(tmp_path / "so101.usda")
    r = import_mjcf(SO101, out)
    m = mujoco.MjModel.from_xml_path(SO101)
    st = Usd.Stage.Open(out)
    assert len(r.body_paths) - 1 == m.nbody - 1
    joints = [p for p in st.Traverse() if p.IsA(UsdPhysics.Joint)]
    assert len(joints) == 6                      # 6 hinges; the box's free joint is no joint prim
    geoms = [p for p in st.Traverse() if p.IsA(UsdGeom.Gprim)]
    assert len(geoms) >= m.ngeom                 # + shared mesh prototypes under /World/Meshes
    # articulation root on the fixed-base robot, rigid body + mass on every body
    assert st.GetPrimAtPath("/World/base").HasAPI(UsdPhysics.ArticulationRootAPI)
    sh = st.GetPrimAtPath("/World/base/shoulder")
    assert sh.HasAPI(UsdPhysics.RigidBodyAPI) and sh.HasAPI(UsdPhysics.MassAPI)
    b = int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "shoulder"))
    assert abs(UsdPhysics.MassAPI(sh).GetMassAttr().Get() - m.body_mass[b]) < 1e-6
    # joint: limits in degrees, drive gains from the position actuator (kp per degree)
    j = st.GetPrimAtPath("/World/base/shoulder/shoulder_pan")
    rj = UsdPhysics.RevoluteJoint(j)
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "shoulder_pan")
    np.testing.assert_allclose(rj.GetLowerLimitAttr().Get(), np.rad2deg(m.jnt_range[jid][0]), rtol=1e-5)
    drv = UsdPhysics.DriveAPI(j, "angular")
    aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "shoulder_pan")
    kp = -m.actuator_biasprm[aid][1]
    np.testing.assert_allclose(drv.GetStiffnessAttr().Get(), kp * np.pi / 180, rtol=1e-4)
    assert j.GetAttribute("mjc:armature").Get() > 0
    # collision only on colliding geoms; visual meshes have collision disabled
    col = [p for p in geoms if p.HasAPI(UsdPhysics.CollisionAPI)]
    n_col = int(((m.geom_contype != 0) | (m.geom_conaffinity != 0)).sum())
    assert len(col) == n_col
    # semantics and materials
    box = st.GetPrimAtPath("/World/box/box")
    assert UsdSemantics.LabelsAPI(box, "class").GetLabelsAttr().Get() == ["box"]
    floor = st.GetPrimAtPath("/World/floor")
    mat = UsdShade.MaterialBindingAPI(floor).ComputeBoundMaterial()[0]
    assert mat and mat.GetPath().name == "groundplane"
    assert os.path.exists(str(tmp_path / "groundplane.png"))
    # cameras with physical intrinsics
    cam = UsdGeom.Camera(st.GetPrimAtPath("/World/base/shoulder/upper_arm/lower_arm/wrist/gripper/camera_mount/wrist_cam"))
    np.testing.assert_allclose(cam.GetFocalLengthAttr().Get(), 3.6, rtol=1e-5)
    np.testing.assert_allclose(cam.GetHorizontalApertureAttr().Get(), 5.76, rtol=1e-5)


def test_usd_round_trip_to_mujoco(tmp_path):
    """USD -> MjSpec, both the lossless path and the generic UsdPhysics conversion."""
    from metalsim.scene.usd_to_mjcf import load_usd
    out = str(tmp_path / "so101.usda")
    import_mjcf(SO101, out)
    m0 = mujoco.MjModel.from_xml_path(SO101)
    m_l = load_usd(out, lossless=True).compile()
    assert (m_l.nbody, m_l.njnt, m_l.ngeom, m_l.nu) == (m0.nbody, m0.njnt, m0.ngeom, m0.nu)
    m1 = load_usd(out, lossless=False).compile()
    assert (m1.nbody, m1.njnt, m1.nu, m1.ncam, m1.nlight) == (m0.nbody, m0.njnt, m0.nu, m0.ncam, m0.nlight)
    assert m1.ngeom == m0.ngeom
    # forward kinematics parity on random joint angles (bodies matched by name)
    d0, d1 = mujoco.MjData(m0), mujoco.MjData(m1)
    rng = np.random.default_rng(0)
    for j in range(m0.njnt):
        if int(m0.jnt_type[j]) != int(mujoco.mjtJoint.mjJNT_HINGE):
            continue
        n = mujoco.mj_id2name(m0, mujoco.mjtObj.mjOBJ_JOINT, j)
        j1 = mujoco.mj_name2id(m1, mujoco.mjtObj.mjOBJ_JOINT, n)
        v = rng.uniform(-0.5, 0.5)
        d0.qpos[m0.jnt_qposadr[j]] = v; d1.qpos[m1.jnt_qposadr[j1]] = v
    mujoco.mj_forward(m0, d0); mujoco.mj_forward(m1, d1)
    for b in range(1, m0.nbody):
        n = mujoco.mj_id2name(m0, mujoco.mjtObj.mjOBJ_BODY, b)
        b1 = mujoco.mj_name2id(m1, mujoco.mjtObj.mjOBJ_BODY, n)
        np.testing.assert_allclose(d1.xpos[b1], d0.xpos[b], atol=1e-5, err_msg=n)
        np.testing.assert_allclose(m1.body_mass[b1], m0.body_mass[b], atol=1e-6)
    # actuator gains survive the per-degree <-> per-radian conversion
    a0 = mujoco.mj_name2id(m0, mujoco.mjtObj.mjOBJ_ACTUATOR, "shoulder_pan"); a1 = mujoco.mj_name2id(m1, mujoco.mjtObj.mjOBJ_ACTUATOR, "shoulder_pan")
    np.testing.assert_allclose(m1.actuator_biasprm[a1][:3], m0.actuator_biasprm[a0][:3], rtol=1e-4)
    # the drive's maxForce comes back as the joint's actuator force range (UsdPhysics maxForce is a joint-level clamp;
    # load_usd(effort_limit="joint"), the default since 2026-09-26)
    if m0.actuator_forcelimited[a0]:
        j1 = m1.actuator_trnid[a1][0]
        assert not m1.actuator_forcelimited[a1] and m1.jnt_actfrclimited[j1]
        np.testing.assert_allclose(m1.jnt_actfrcrange[j1], m0.actuator_forcerange[a0], rtol=1e-5)


def test_urdf_import_to_usd(tmp_path):
    """URDF -> USD through MuJoCo's URDF loader (MjSpec handles URDF), same physics mapping."""
    urdf = tmp_path / "arm.urdf"
    urdf.write_text("""<?xml version="1.0"?>
<robot name="arm2">
  <link name="base"><inertial><mass value="1.0"/><inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/></inertial>
    <visual><geometry><box size="0.1 0.1 0.1"/></geometry></visual><collision><geometry><box size="0.1 0.1 0.1"/></geometry></collision></link>
  <link name="link1"><inertial><origin xyz="0 0 0.15"/><mass value="0.5"/><inertia ixx="0.005" iyy="0.005" izz="0.001" ixy="0" ixz="0" iyz="0"/></inertial>
    <visual><origin xyz="0 0 0.15"/><geometry><cylinder radius="0.02" length="0.3"/></geometry></visual></link>
  <link name="link2"><inertial><origin xyz="0 0 0.1"/><mass value="0.3"/><inertia ixx="0.002" iyy="0.002" izz="0.0005" ixy="0" ixz="0" iyz="0"/></inertial>
    <visual><origin xyz="0 0 0.1"/><geometry><cylinder radius="0.015" length="0.2"/></geometry></visual></link>
  <joint name="j1" type="revolute"><parent link="base"/><child link="link1"/><origin xyz="0 0 0.05"/><axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="10" velocity="2"/><dynamics damping="0.1" friction="0.05"/></joint>
  <joint name="j2" type="revolute"><parent link="link1"/><child link="link2"/><origin xyz="0 0 0.3"/><axis xyz="0 1 0"/>
    <limit lower="-2.0" upper="2.0" effort="5" velocity="2"/></joint>
</robot>""")
    from pxr import Usd, UsdPhysics
    from metalsim.scene.usd_to_mjcf import load_usd
    out = str(tmp_path / "arm.usda")
    r = import_mjcf(str(urdf), out)
    st = Usd.Stage.Open(out)
    joints = [p for p in st.Traverse() if p.IsA(UsdPhysics.RevoluteJoint)]
    assert len(joints) == 2 and len(r.body_paths) - 1 == 2   # MuJoCo attaches the URDF root link to the world
    j1 = next(p for p in joints if p.GetPath().name == "j1")
    assert abs(UsdPhysics.RevoluteJoint(j1).GetUpperLimitAttr().Get() - np.rad2deg(1.57)) < 1e-3
    assert abs(j1.GetAttribute("mjc:damping").Get() - 0.1) < 1e-6 and abs(j1.GetAttribute("mjc:frictionloss").Get() - 0.05) < 1e-6
    m1 = load_usd(out, lossless=False).compile()
    m0 = mujoco.MjModel.from_xml_path(str(urdf))
    assert (m1.nbody, m1.njnt) == (m0.nbody, m0.njnt)
    np.testing.assert_allclose(sorted(m1.body_mass), sorted(m0.body_mass), atol=1e-6)


# -- USD visual import: area lights, GeomSubset per-face materials, UV interpolations (render-fix workstream C) -------
# Synthetic stages built here; scripts/diagnostics/usd_import_visuals.py reuses the builders for the before/after numbers.

def area_light_stage(path):
    """RectLight, DiskLight (normalize), CylinderLight and a DistantLight; returns the area lights' analytic specs."""
    from pxr import Gf, Usd, UsdGeom, UsdLux
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z); UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World"); stage.SetDefaultPrim(world.GetPrim())
    rect = UsdLux.RectLight.Define(stage, "/World/rect")
    rect.CreateWidthAttr(0.4); rect.CreateHeightAttr(0.2); rect.CreateIntensityAttr(1000.0); rect.CreateExposureAttr(1.0)
    rect.CreateColorAttr(Gf.Vec3f(1.0, 0.8, 0.6))
    UsdGeom.Xformable(rect.GetPrim()).AddTranslateOp().Set((0.0, 0.0, 3.0))            # emits toward its -Z: down
    disk = UsdLux.DiskLight.Define(stage, "/World/disk")
    disk.CreateRadiusAttr(0.1); disk.CreateIntensityAttr(50.0); disk.CreateNormalizeAttr(True)
    UsdGeom.Xformable(disk.GetPrim()).AddTranslateOp().Set((2.0, 0.0, 3.0))
    cyl = UsdLux.CylinderLight.Define(stage, "/World/tube")
    cyl.CreateRadiusAttr(0.05); cyl.CreateLengthAttr(0.6); cyl.CreateIntensityAttr(200.0)
    UsdGeom.Xformable(cyl.GetPrim()).AddTranslateOp().Set((-2.0, 0.0, 3.0))
    sun = UsdLux.DistantLight.Define(stage, "/World/sun"); sun.CreateIntensityAttr(1.0)
    stage.GetRootLayer().Save()
    return {"rect": dict(shape="rect", L=1000.0 * 2.0 * np.array([1.0, 0.8, 0.6]), w=0.4, h=0.2, c=np.array([0, 0, 3.0])),
            "disk": dict(shape="disk", L=50.0 / (np.pi * 0.01) * np.ones(3), r=0.1, c=np.array([2.0, 0, 3.0])),
            "tube": dict(shape="cyl", L=200.0 * np.ones(3), r=0.05, len=0.6, c=np.array([-2.0, 0, 3.0]))}


def area_light_size(s):
    return max(s.get("w", 0), s.get("h", 0), 2 * s.get("r", 0), s.get("len", 0))


def area_light_exact_irradiance(s, p, n=400):
    """Irradiance (red channel) at p on a receiver facing +z from the Lambertian area light, by midpoint quadrature."""
    L, c = float(s["L"][0]), s["c"]
    if s["shape"] in ("rect", "disk"):
        u = (np.arange(n) + 0.5) / n
        wx, wy = (s["w"], s["h"]) if s["shape"] == "rect" else (2 * s["r"], 2 * s["r"])
        X, Y = np.meshgrid((u - 0.5) * wx, (u - 0.5) * wy)
        keep = np.ones_like(X, bool) if s["shape"] == "rect" else X ** 2 + Y ** 2 <= s["r"] ** 2
        S = np.stack([X[keep] + c[0], Y[keep] + c[1], np.full(int(keep.sum()), c[2])], 1)
        v = S - p; d2 = (v ** 2).sum(1); d = np.sqrt(d2)
        cos = np.clip(v[:, 2] / d, 0, None)              # emitter faces -z, receiver +z: the same cosine
        return float((L * cos * cos / d2).sum() * wx * wy / n ** 2)
    xs = (np.arange(n) + 0.5) / n * s["len"] - s["len"] / 2; a = (np.arange(n) + 0.5) / n * 2 * np.pi
    X, A = np.meshgrid(xs, a)
    nrm = np.stack([np.zeros_like(X), np.cos(A), np.sin(A)], -1).reshape(-1, 3)        # cylinder along x, no caps
    S = c + np.stack([X, s["r"] * np.cos(A), s["r"] * np.sin(A)], -1).reshape(-1, 3)
    v = p - S; d2 = (v ** 2).sum(1); d = np.sqrt(d2)
    cos_e = np.clip((nrm * v).sum(1) / d, 0, None); cos_r = np.clip(-v[:, 2] / d, 0, None)
    return float((L * cos_e * cos_r / d2).sum() * s["len"] / n * s["r"] * 2 * np.pi / n)


def mujoco_light_irradiance(m, i, p):
    """Irradiance (red) at p on a receiver facing +z from MuJoCo light i, the renderers' convention (tier 0/1/2):
    E = pi x diffuse x spot cos^exponent x cos(theta_r) / (c + l d + q d^2)."""
    lp = m.light_pos[i]; ld = m.light_dir[i] / np.linalg.norm(m.light_dir[i])
    v = p - lp; d = np.linalg.norm(v)
    att = 1.0 / float(m.light_attenuation[i] @ [1.0, d, d * d])
    if int(m.light_type[i]) == int(mujoco.mjtLightType.mjLIGHT_SPOT):
        ca = float(np.dot(v / d, ld))
        att *= ca ** float(m.light_exponent[i]) if ca > np.cos(np.radians(m.light_cutoff[i])) else 0.0
    return float(np.pi * m.light_diffuse[i][0] * att * max(-v[2] / d, 0.0))


def test_usd_area_lights_imported(tmp_path):
    """Rect / Disk / Cylinder lights were dropped (only the DistantLight came through); now a spot (rect, disk: cutoff
    90, exponent 1, inverse-square) and a point light (cylinder) whose irradiance at 10 light sizes matches the area
    light's (quadrature) within 1 % on and 30 deg off axis; the USD shape is recorded in ``usd_area_lights``."""
    import json
    pytest.importorskip("pxr")
    from metalsim.scene.usd_to_mjcf import load_usd
    lights = area_light_stage(tmp_path / "lights.usda")
    m = load_usd(str(tmp_path / "lights.usda")).compile()
    assert m.nlight == 4
    ids = {k: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_LIGHT, k) for k in lights}
    spot, point = int(mujoco.mjtLightType.mjLIGHT_SPOT), int(mujoco.mjtLightType.mjLIGHT_POINT)
    assert [int(m.light_type[ids[k]]) for k in ("rect", "disk", "tube")] == [spot, spot, point]
    for k in ("rect", "disk"):
        assert m.light_cutoff[ids[k]] == 90.0 and m.light_exponent[ids[k]] == 1.0
        np.testing.assert_allclose(m.light_dir[ids[k]], [0, 0, -1], atol=1e-6)
    assert np.all(m.light_diffuse <= 10.0 + 1e-6)          # the importer's clip; irradiance kept through the attenuation
    tid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_TEXT, "usd_area_lights")
    rec = json.loads(bytes(m.text_data[m.text_adr[tid]:m.text_adr[tid] + m.text_size[tid]]).split(b"\0")[0].decode())
    by = {r["light"]: r for r in rec}
    assert by["rect"]["shape"] == "rect" and np.isclose(by["rect"]["width"], 0.4) and by["disk"]["normalize"]
    assert by["tube"]["shape"] == "cylinder" and np.isclose(by["tube"]["length"], 0.6)
    for k, s in lights.items():
        for off in (0.0, 30.0):
            p = s["c"] + 10 * area_light_size(s) * np.array([0.0, np.sin(np.radians(off)), -np.cos(np.radians(off))])
            ex, mj = area_light_exact_irradiance(s, p), mujoco_light_irradiance(m, ids[k], p)
            assert abs(mj / ex - 1) < 0.01, (k, off, mj, ex)


def _preview_material(stage, path, rgb=None, texture=None, varname="st"):
    from pxr import Sdf, UsdShade
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path + "/surf"); sh.CreateIdAttr("UsdPreviewSurface")
    if texture is None:
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(rgb)
    else:
        rd = UsdShade.Shader.Define(stage, path + "/reader"); rd.CreateIdAttr("UsdPrimvarReader_float2")
        rd.CreateInput("varname", Sdf.ValueTypeNames.Token).Set(varname)
        tx = UsdShade.Shader.Define(stage, path + "/tex"); tx.CreateIdAttr("UsdUVTexture")
        tx.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(texture)
        tx.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(rd.ConnectableAPI(), "result")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tx.ConnectableAPI(), "rgb")
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


def subset_stage(path, collide=False):
    """A cube mesh of 6 quads with GeomSubsets: top red, the four sides blue; the bottom face in no subset (the mesh's
    own green binding). Camera "cam" above and in front of a corner, seeing the top and two sides."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z); UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World"); stage.SetDefaultPrim(world.GetPrim())
    red = _preview_material(stage, "/World/Looks/red", Gf.Vec3f(0.9, 0.1, 0.1))
    blue = _preview_material(stage, "/World/Looks/blue", Gf.Vec3f(0.1, 0.1, 0.9))
    green = _preview_material(stage, "/World/Looks/green", Gf.Vec3f(0.1, 0.9, 0.1))
    mesh = UsdGeom.Mesh.Define(stage, "/World/box")
    h = 0.2
    pts = [(-h, -h, -h), (h, -h, -h), (h, h, -h), (-h, h, -h), (-h, -h, h), (h, -h, h), (h, h, h), (-h, h, h)]
    faces = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]   # bottom, top, sides
    mesh.CreatePointsAttr(pts); mesh.CreateFaceVertexCountsAttr([4] * 6)
    mesh.CreateFaceVertexIndicesAttr([i for f in faces for i in f]); mesh.CreateSubdivisionSchemeAttr("none")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(green)
    for name, fs, mat in (("top", [1], red), ("sides", [2, 3, 4, 5], blue)):
        s = UsdGeom.Subset.CreateGeomSubset(UsdGeom.Imageable(mesh), name, UsdGeom.Tokens.face, fs, "materialBind")
        UsdShade.MaterialBindingAPI.Apply(s.GetPrim()).Bind(mat)
    if collide:
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    cam = UsdGeom.Camera.Define(stage, "/World/cam")
    cam.CreateFocalLengthAttr(24.0); cam.CreateVerticalApertureAttr(15.0); cam.CreateHorizontalApertureAttr(15.0)
    xf = UsdGeom.Xformable(cam.GetPrim())
    xf.AddTranslateOp().Set((0.9, -0.9, 0.9)); xf.AddRotateXYZOp().Set((54.7, 0.0, 45.0))
    stage.GetRootLayer().Save()


def test_usd_geomsubset_per_face_materials(tmp_path):
    """GeomSubset (materialBind) faces keep their own material: one visual geom per subset, plus the unassigned faces
    with the mesh's binding; the whole mesh stays one undrawn geom (collision and mass unchanged)."""
    pytest.importorskip("pxr")
    from metalsim.scene.usd_to_mjcf import load_usd
    for collide in (False, True):
        subset_stage(tmp_path / f"sub{collide}.usda", collide)
        m = load_usd(str(tmp_path / f"sub{collide}.usda")).compile()
        geoms = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g): g for g in range(m.ngeom)}
        mat = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MATERIAL, int(m.geom_matid[g]))
        assert set(geoms) == {"box", "box_top", "box_sides", "box_rest"}
        assert (mat(geoms["box_top"]), mat(geoms["box_sides"]), mat(geoms["box_rest"])) == ("red", "blue", "green")
        assert m.geom_rgba[geoms["box"]][3] == 0 and m.geom_matid[geoms["box"]] < 0
        assert [int(m.mesh_facenum[m.geom_dataid[geoms[k]]]) for k in ("box_top", "box_sides", "box_rest", "box")] == [2, 8, 2, 12]
        assert all(m.geom_contype[geoms[k]] == 0 and m.geom_conaffinity[geoms[k]] == 0 for k in ("box_top", "box_sides", "box_rest"))
        assert bool(m.geom_contype[geoms["box"]]) == collide
        d = mujoco.MjData(m); mujoco.mj_forward(m, d)
        for k, axis_val in (("box_top", (2, 0.2)), ("box_rest", (2, -0.2))):   # pieces sit where the faces were
            g = geoms[k]; mid = m.geom_dataid[g]
            v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
            w = d.geom_xpos[g] + v @ d.geom_xmat[g].reshape(3, 3).T
            np.testing.assert_allclose(w[:, axis_val[0]], axis_val[1], atol=1e-6)
            assert np.allclose(np.abs(w[:, :2]), 0.2, atol=1e-6)


def write_asym_texture(path):
    """64 x 64 PNG: top half red, bottom half blue, left quarter green."""
    from PIL import Image
    img = np.zeros((64, 64, 3), np.uint8)
    img[:32] = (230, 40, 40); img[32:] = (40, 40, 230); img[:, :16] = (40, 230, 40)
    Image.fromarray(img).save(str(path))
    return img


def uv_plate_stage(path, tex, interpolation="vertex", indexed=False, triangles=False):
    """A thin plate of 6 quads, front face toward -y spanning st in [0, 1] (USD: v up), textured through a
    UsdPrimvarReader on "st": per point ("vertex"), per corner ("faceVarying"), or indexed faceVarying. Camera "cam"
    on -y looking at it."""
    from pxr import Sdf, Usd, UsdGeom, UsdShade, Vt
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z); UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World"); stage.SetDefaultPrim(world.GetPrim())
    mesh = UsdGeom.Mesh.Define(stage, "/World/plate")
    s, t = 0.15, 0.005
    pts = [(-s, -t, -s), (s, -t, -s), (s, -t, s), (-s, -t, s), (-s, t, -s), (s, t, -s), (s, t, s), (-s, t, s)]
    pt_uv = [(0, 0), (1, 0), (1, 1), (0, 1)] * 2
    faces = [(0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)]
    if triangles:                # the all-triangle layout the importer already handled (faceVarying only)
        faces = [t for f in faces for t in ((f[0], f[1], f[2]), (f[0], f[2], f[3]))]
    idx = [i for f in faces for i in f]
    mesh.CreatePointsAttr(pts); mesh.CreateFaceVertexCountsAttr([len(f) for f in faces]); mesh.CreateFaceVertexIndicesAttr(idx)
    mesh.CreateSubdivisionSchemeAttr("none")
    api = UsdGeom.PrimvarsAPI(mesh)
    tok = UsdGeom.Tokens.vertex if interpolation == "vertex" else UsdGeom.Tokens.faceVarying
    pv = api.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, tok)
    if interpolation == "vertex":
        pv.Set(Vt.Vec2fArray([tuple(map(float, u)) for u in pt_uv]))
    elif indexed:
        pv.Set(Vt.Vec2fArray([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])); pv.SetIndices(Vt.IntArray([i % 4 for i in idx]))
    else:
        pv.Set(Vt.Vec2fArray([tuple(map(float, pt_uv[i])) for i in idx]))
    mat = _preview_material(stage, "/World/Looks/tex", texture=str(tex))
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
    cam = UsdGeom.Camera.Define(stage, "/World/cam")
    cam.CreateFocalLengthAttr(24.0); cam.CreateVerticalApertureAttr(15.0); cam.CreateHorizontalApertureAttr(15.0)
    xf = UsdGeom.Xformable(cam.GetPrim()); xf.AddTranslateOp().Set((0.0, -0.6, 0.0)); xf.AddRotateXYZOp().Set((90.0, 0.0, 0.0))
    stage.GetRootLayer().Save()


def plate_expected_class(m, d, cam, W, H, img):
    """Per pixel, the colour class (0 red, 1 blue, 2 green; -1 off the front face) the plate's texture shows there."""
    fy = H / (2 * np.tan(np.radians(m.cam_fovy[cam]) / 2))
    yy, xx = np.mgrid[0:H, 0:W]
    dirs = np.stack([(xx + 0.5 - W / 2) / fy, -(yy + 0.5 - H / 2) / fy, -np.ones((H, W))], -1) @ d.cam_xmat[cam].reshape(3, 3).T
    o = d.cam_xpos[cam]
    P = o + dirs * ((-0.005 - o[1]) / dirs[..., 1])[..., None]
    u = (P[..., 0] + 0.15) / 0.3; v = (P[..., 2] + 0.15) / 0.3
    on = (u >= 0) & (u <= 1) & (v >= 0) & (v <= 1)
    col = img[np.clip(((1 - v) * 64).astype(int), 0, 63), np.clip((u * 64).astype(int), 0, 63)].astype(np.float32)
    return np.where(on, pixel_class(col), -1)


def pixel_class(rgb):
    rgb = np.asarray(rgb, np.float32)
    return np.where((rgb[..., 1] > rgb[..., 0]) & (rgb[..., 1] > rgb[..., 2]), 2, np.where(rgb[..., 0] > rgb[..., 2], 0, 1))


def plate_interior(exp, it=2):
    from scipy.ndimage import binary_erosion
    inner = np.zeros(exp.shape, bool)
    for c in range(3):           # away from the colour boundaries and the plate's edge (texel filtering, rasterisation)
        inner |= binary_erosion(exp == c, iterations=it)
    return inner


@pytest.mark.parametrize("interp,indexed,tris", [("vertex", False, False), ("faceVarying", False, False), ("faceVarying", True, False), ("faceVarying", False, True)])
def test_usd_mesh_uvs_vertex_facevarying_quads(tmp_path, interp, indexed, tris):
    """st with vertex, faceVarying and indexed faceVarying interpolation on a quad mesh reaches the MuJoCo model, USD's
    v-up mapped to MuJoCo's v-down: MuJoCo's own renderer shows the image the right way up (top red, bottom blue,
    left strip green) on >= 95 % of the plate's interior pixels (before: vertex / quads gave no texcoords at all)."""
    pytest.importorskip("pxr")
    from metalsim.scene.usd_to_mjcf import load_usd
    img = write_asym_texture(tmp_path / "asym.png")
    uv_plate_stage(tmp_path / "plate.usda", tmp_path / "asym.png", interp, indexed, tris)
    m = load_usd(str(tmp_path / "plate.usda")).compile()
    assert m.nmesh == 1 and m.mesh_texcoordadr[0] >= 0 and m.ntex == 1
    d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    W = H = 96
    r = mujoco.Renderer(m, H, W); r.update_scene(d, camera="cam"); rgb = r.render(); r.close()
    exp = plate_expected_class(m, d, 0, W, H, img)
    inner = plate_interior(exp)
    frac = float((pixel_class(rgb)[inner] == exp[inner]).mean())
    assert inner.sum() > 500 and frac > 0.95, frac

"""WS1: MJCF -> USD import carries physics, materials, semantics and cameras."""
import os

import mujoco
import numpy as np
import pytest

from orchard.scene.mjcf_to_usd import import_mjcf

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

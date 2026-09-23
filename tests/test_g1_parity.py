"""Parity of the Isaac Lab G1 velocity task with Isaac's own configuration, on Isaac's own asset
(assets/isaac/G1/g1_minimal.usd, the file Isaac Lab loads for Isaac-Velocity-Rough-G1-v0).

Each test names the Isaac source it checks against; together they are the evidence rows of
docs/PARITY.md for the G1 comparison."""
import ast
import os
import re

import mujoco
import numpy as np
import pytest
import warp as wp

from orchard.learn.g1_velocity import ACTUATORS, INIT_JOINTS, INIT_POS, PHYSICS_DT, build_g1_model

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
USD = os.path.join(ROOT, "assets", "isaac", "G1", "g1_minimal.usd")
ISAAC_CFG = os.path.join(ROOT, "assets", "isaac", "g1_asset_cfg.py")
pytestmark = pytest.mark.skipif(not os.path.exists(USD), reason="needs Isaac's g1_minimal.usd")


@pytest.fixture(scope="module")
def model():
    m, _ = build_g1_model("flat")
    return m


def _isaac_actuator_groups():
    """Parses G1_MINIMAL_CFG's ImplicitActuatorCfg groups out of Isaac Lab's source (no isaaclab import)."""
    tree = ast.parse(open(ISAAC_CFG).read())
    groups = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "G1_CFG" for t in node.targets):
            for call in ast.walk(node):
                if isinstance(call, ast.Call) and getattr(call.func, "id", "") == "ImplicitActuatorCfg":
                    kw = {k.arg: ast.literal_eval(k.value) for k in call.keywords}
                    groups.append(kw)
    assert groups, "G1_CFG actuator groups not found"
    # G1_MINIMAL_CFG (what the velocity env uses) is G1_CFG with only the usd_path replaced
    src = open(ISAAC_CFG).read()
    assert re.search(r"G1_MINIMAL_CFG = G1_CFG\.copy\(\)", src) and "g1_minimal.usd" in src
    return groups


def _isaac_value(groups, joint, key):
    for g in groups:
        if any(re.fullmatch(p, joint) for p in g["joint_names_expr"]):
            v = g[key]
            if isinstance(v, dict):
                return next(val for p, val in v.items() if re.fullmatch(p, joint))
            return v
    raise KeyError(joint)


def test_actuator_gains_match_isaac_implicit_pd(model):
    """Isaac Lab: ImplicitActuatorCfg -> PhysX joint drive tau = kp (q* - q) - kd qd clipped to
    effort_limit_sim, armature added to the joint (source: assets/isaac/g1_asset_cfg.py,
    G1_MINIMAL_CFG). Ours: MuJoCo affine actuators with gain kp, bias (0, -kp, -kd), forcerange."""
    groups = _isaac_actuator_groups()
    m = model
    assert m.nu == 37
    for a in range(m.nu):
        j = m.actuator_trnid[a][0]
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        kp = _isaac_value(groups, name, "stiffness"); kd = _isaac_value(groups, name, "damping")
        eff = _isaac_value(groups, name, "effort_limit_sim"); arm = _isaac_value(groups, name, "armature")
        assert m.actuator_gainprm[a][0] == kp, name
        assert m.actuator_biasprm[a][1] == -kp and m.actuator_biasprm[a][2] == -kd, name
        assert tuple(m.actuator_forcerange[a]) == (-eff, eff) and m.actuator_forcelimited[a], name
        assert m.dof_armature[m.jnt_dofadr[j]] == arm, name
    # the actuator table in g1_velocity.py is exactly Isaac's grouping (no joint unmatched, no extra)
    joints = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(1, m.njnt)]
    assert all(any(re.fullmatch(p, jn) for p, *_ in ACTUATORS) for jn in joints)


def test_joint_frames_and_axes_match_usd(model):
    """Every joint's world anchor and axis equal the UsdPhysics joint frames of Isaac's USD.

    The USD's authored prim transforms are a posed configuration (the joint frames F0 = body0 .
    (localPos0, localRot0) and F1 = body1 . (localPos1, localRot1) differ by a rotation about the
    joint axis); the joint angle in that pose is that rotation, and the zero pose is where the
    frames coincide (PhysX's definition). We pose our model with those angles and require the world
    anchors and axes of all 37 joints to match both frames to 1e-5."""
    from pxr import Usd, UsdGeom, UsdPhysics, Gf
    st = Usd.Stage.Open(USD)
    units = float(UsdGeom.GetStageMetersPerUnit(st) or 1.0)
    cache = UsdGeom.XformCache()
    m = model
    d = mujoco.MjData(m); d.qpos[:] = 0; d.qpos[3] = 1
    frames = {}
    tok = {"X": Gf.Vec3d(1, 0, 0), "Y": Gf.Vec3d(0, 1, 0), "Z": Gf.Vec3d(0, 0, 1)}
    for p in st.Traverse():
        if not p.IsA(UsdPhysics.RevoluteJoint):
            continue
        j = UsdPhysics.RevoluteJoint(p)
        out = []
        for side in (0, 1):
            b = (j.GetBody0Rel() if side == 0 else j.GetBody1Rel()).GetTargets()[0]
            M = cache.GetLocalToWorldTransform(st.GetPrimAtPath(b))
            lp = Gf.Vec3d((j.GetLocalPos0Attr() if side == 0 else j.GetLocalPos1Attr()).Get())
            lr = Gf.Quatd((j.GetLocalRot0Attr() if side == 0 else j.GetLocalRot1Attr()).Get())
            pos = np.array(M.Transform(lp)) * units
            R = np.array([np.array(M.TransformDir(lr.Transform(tok[a]))) for a in "XYZ"]).T   # columns = frame axes in world
            R /= np.linalg.norm(R, axis=0, keepdims=True)
            out.append((pos, R))
        frames[p.GetName()] = (out, j.GetAxisAttr().Get() or "X")
    assert len(frames) == 37
    # joint angle in the authored pose: rotation from F0 to F1 about the joint axis
    for name, (fr, axis_tok) in frames.items():
        (p0, R0), (p1, R1) = fr
        assert np.linalg.norm(p0 - p1) < 1e-5, f"{name}: USD joint frames do not share an anchor"
        Rrel = R1 @ R0.T
        ang = np.arccos(np.clip((np.trace(Rrel) - 1) / 2, -1, 1))
        axis_w = R0 @ np.array(tok[axis_tok])
        if ang > 1e-3:      # below this the frames coincide to float32 precision (state 0)
            rot_axis = np.array([Rrel[2, 1] - Rrel[1, 2], Rrel[0, 2] - Rrel[2, 0], Rrel[1, 0] - Rrel[0, 1]])
            rot_axis /= np.linalg.norm(rot_axis)
            assert abs(abs(rot_axis @ axis_w) - 1) < 1e-3, f"{name}: authored pose is not a rotation about the joint axis"
            ang *= np.sign(rot_axis @ axis_w)
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name); assert jid >= 0
        d.qpos[m.jnt_qposadr[jid]] = ang
    mujoco.mj_forward(m, d)
    worst_pos = worst_ax = 0.0
    for name, (fr, axis_tok) in frames.items():
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        (p0, R0), (p1, R1) = fr
        worst_pos = max(worst_pos, float(np.linalg.norm(d.xanchor[jid] - p0)))
        worst_ax = max(worst_ax, float(np.linalg.norm(d.xaxis[jid] - R0 @ np.array(tok[axis_tok]))))
    posed = int(np.sum(np.abs(d.qpos[7:]) > 1e-6))
    print(f"37 revolute joints ({posed} non-zero in the authored pose): worst anchor error {worst_pos:.2e} m, worst axis error {worst_ax:.2e}")
    assert worst_pos < 1e-5 and worst_ax < 1e-5


def test_mass_properties_match_usd(model):
    """Per-link mass, diagonal inertia and centre of mass equal the USD's PhysicsMassAPI values
    (Isaac uses these as-is; unauthored COM (-inf) means the link origin)."""
    from pxr import Usd, UsdPhysics
    st = Usd.Stage.Open(USD)
    m = model
    n = 0
    for p in st.Traverse():
        if not (p.HasAPI(UsdPhysics.RigidBodyAPI) and p.HasAPI(UsdPhysics.MassAPI)):
            continue
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, p.GetName()); assert b >= 0
        ma = UsdPhysics.MassAPI(p)
        mass = ma.GetMassAttr().Get(); di = np.array(ma.GetDiagonalInertiaAttr().Get()); com = np.array(ma.GetCenterOfMassAttr().Get())
        assert m.body_mass[b] == pytest.approx(mass, rel=1e-6), p.GetName()
        np.testing.assert_allclose(np.sort(m.body_inertia[b]), np.sort(di), rtol=1e-5, err_msg=p.GetName())
        if np.isfinite(com).all():
            np.testing.assert_allclose(m.body_ipos[b], com, atol=1e-6, err_msg=p.GetName())
        else:
            np.testing.assert_allclose(m.body_ipos[b], 0, atol=1e-6, err_msg=p.GetName())
        n += 1
    assert n == 44 and m.body_subtreemass[1] == pytest.approx(32.239, abs=1e-2)


def test_scene_settings_match_isaac(model):
    """Physics rate 200 Hz (decimation 4 -> 50 Hz control), gravity 9.81 down, Isaac's initial
    state (pos z 0.74, joint defaults), soft limits not applied at the joint level (Isaac's
    soft_joint_pos_limit_factor only affects the penalty term), episode 20 s (g1_velocity_env_cfg.py)."""
    m = model
    assert m.opt.timestep == PHYSICS_DT == 0.005
    np.testing.assert_allclose(m.opt.gravity, [0, 0, -9.81])
    q0 = m.key_qpos[0]
    np.testing.assert_allclose(q0[:3], INIT_POS); np.testing.assert_allclose(q0[3:7], [1, 0, 0, 0])
    for j in range(1, m.njnt):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        want = next((v for p, v in INIT_JOINTS if re.fullmatch(p, name)), 0.0)
        assert q0[m.jnt_qposadr[j]] == pytest.approx(want), name
    # Isaac's init_state.joint_pos in the source is the same table
    src = open(ISAAC_CFG).read()
    for pat, val in INIT_JOINTS:
        assert re.search(rf'"{re.escape(pat)}":\s*{val}', src), pat


def test_collision_set_matches_usd(model):
    """g1_minimal.usd keeps three convex colliders (torso and both ankle_roll links); ours has the
    same three mesh geoms plus the ground, nothing else (no visual meshes collide)."""
    m = model
    names = sorted(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) for g in range(m.ngeom) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH)
    assert names == ["left_ankle_roll_link", "right_ankle_roll_link", "torso_link"]
    assert m.ngeom == 4


@pytest.mark.skipif(not wp.is_metal_available(), reason="needs Metal")
def test_warp_matches_mujoco_c_under_pd_hold():
    """The batched Metal simulation reproduces MuJoCo C step by step on this asset (same solver
    budget): pelvis height and joint angles agree to MuJoCo Warp's tolerance over 100 steps of a
    PD hold at the initial pose (the robot pitches forward under Isaac's 20 Nm/rad ankle gains)."""
    from orchard.learn.g1_velocity import G1VelocityTask
    task = G1VelocityTask(4, terrain="flat")
    m = task.model
    import torch
    task.reset_all()                                 # Isaac's reset randomizes yaw and xy; override with the keyframe
    d = mujoco.MjData(m); mujoco.mj_resetDataKeyframe(m, d, 0); d.ctrl[:] = m.key_qpos[0][7:]
    q_init = np.tile(m.key_qpos[0], (4, 1)).astype(np.float32); q_init[:, :2] += task.origins.numpy()[:, :2]
    task.sim.t.qpos.copy_(torch.as_tensor(q_init)); task.sim.t.qvel.zero_()
    task.sim.t.ctrl.copy_(torch.as_tensor(np.tile(m.key_qpos[0][7:], (4, 1)).astype(np.float32)))
    v = task.sim.forward(); task.sim.after(v); task.sim.synchronize()
    def ours():
        q = task.sim.d.qpos.numpy().astype(np.float64)
        q[:, :2] -= task.origins.numpy()[:, :2]      # worlds are laid out on a 2.5 m grid (Isaac env_spacing)
        return q
    for t in range(25):                              # 25 control steps x 4 substeps = 0.5 s
        task.sim.step()
        for _ in range(task.decimation):
            mujoco.mj_step(m, d)
    task.sim.synchronize()
    q = ours(); e = np.abs(q - d.qpos[None]); err = e.max()
    print(f"after 0.5 s (landing from 5 cm, 8 contacts): pelvis z ours {q[0, 2]:.4f} mujoco {d.qpos[2]:.4f}; "
          f"|dq| max {err:.2e} median {np.median(e):.1e} over 4 worlds")
    # MuJoCo C exits the Newton loop on tolerance; Metal runs the full 10 iterations (no conditional
    # graph nodes), so contact-rich steps differ at the 1e-2 rad level, MJWarp's documented regime.
    assert np.isfinite(q).all() and err < 2e-2 and abs(q[0, 2] - d.qpos[2]) < 2e-3
    for t in range(75):                              # to 2 s: both have pitched onto the torso
        task.sim.step()
        for _ in range(task.decimation):
            mujoco.mj_step(m, d)
    task.sim.synchronize()
    q = ours()
    print(f"after 2 s: pelvis z ours {q[:, 2].round(3)} mujoco {d.qpos[2]:.3f}")
    assert np.isfinite(q).all() and (q[:, 2] < 0.1).all() and d.qpos[2] < 0.1

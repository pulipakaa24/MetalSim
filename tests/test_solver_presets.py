"""metalsim.physics.solver_presets: Isaac Lab 3.0's own MuJoCo Warp settings for the G1 as a preset.

Model fields against the live recording (runs/parity3/isaac/newton_mjwarp_settings.json), the once-per-tick collision
(Newton's fast path: same contacts, dist/pos from body-local witness points) against a fresh collision pass, and the
iteration cap: MuJoCo Warp exits each world at the tolerance, so a lower cap gives the same state whenever no world
reaches it."""
import json

import mujoco
import numpy as np
import pytest
import warp as wp

from metalsim.learn.g1_velocity import build_g1_model
from metalsim.physics import solver_presets
from metalsim.physics.batch import BatchSim, BatchSimOptions

pytestmark = pytest.mark.skipif(not wp.is_metal_available(), reason="needs Metal")


def test_isaaclab3_model_fields_match_the_recording():
    from metalsim.physics import contact_tuning
    m, _ = build_g1_model("flat", physics_dt=0.0025)
    contact_tuning.apply(m, "recommended")          # the task applies its contact_cfg first: the preset must override all of it
    solver_presets.apply(m, "isaaclab3")
    lim = json.load(open("runs/parity3/isaac/fidelity/newton_mjwarp/meta.json"))["newton"]["mjw_model_joint_actuator"]["jnt_solimp"]["first_world"]
    np.testing.assert_allclose(m.jnt_solimp, np.tile(np.array(lim)[0], (m.njnt, 1)), rtol=1e-6)   # one row for every joint
    rs = json.load(open("runs/parity3/isaac/fidelity/newton_mjwarp/meta.json"))["newton"]["mjw_model_joint_actuator"]["jnt_solref"]["first_world"]
    np.testing.assert_allclose(m.jnt_solref[0], np.array(rs)[0], rtol=1e-4)                     # free joint (per-name loop below)
    rec = json.load(open(solver_presets.ISAACLAB3_SETTINGS))
    assert m.opt.iterations == rec["solver"]["iterations"] == 100 and m.opt.ls_iterations == rec["solver"]["ls_iterations"] == 50
    assert m.opt.tolerance == pytest.approx(rec["solver"]["tolerance"]) and m.opt.ls_tolerance == pytest.approx(rec["solver"]["ls_tolerance"])
    assert m.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST and m.opt.cone == mujoco.mjtCone.mjCONE_PYRAMIDAL
    assert m.opt.impratio == rec["solver"]["impratio"] == 1.0
    np.testing.assert_allclose(m.geom_solref, np.tile(rec["contacts"]["geom_solref"], (m.ngeom, 1)), rtol=1e-4)
    np.testing.assert_allclose(m.geom_solimp, np.tile(rec["contacts"]["geom_solimp"], (m.ngeom, 1)), rtol=1e-6)
    np.testing.assert_allclose(m.geom_gap, rec["contacts"]["shape_defaults"]["gap"])
    # contact friction: MuJoCo and Newton both take the larger of the two geoms' coefficients (equal priorities)
    col = (m.geom_contype != 0) | (m.geom_conaffinity != 0)
    assert np.max(m.geom_friction[col, 0]) == 1.0
    for nm, v in rec["joint_limits"]["joints"].items():
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, nm)
        np.testing.assert_allclose(m.jnt_solref[j], v["solreflimit_live"], rtol=1e-6)
        np.testing.assert_allclose(m.jnt_range[j], v["range"], atol=1e-4)            # same asset limits
        assert m.dof_armature[m.jnt_dofadr[j]] == pytest.approx(v["armature"])


def _standing_sim(preset, n=4, substeps=2):
    m, _ = build_g1_model("flat", physics_dt=0.0025)
    solver_presets.apply(m, preset)
    sim = BatchSim(m, n, options=BatchSimOptions(**solver_presets.batch_options(preset, substeps=substeps, njmax=256, nconmax=64)))
    q = np.tile(m.key_qpos[0], (n, 1)).astype(np.float32)
    q[:, 2] -= 0.025                                           # feet ~5 mm into the floor: vertex contacts on both feet
    sim.set_state(q, np.zeros((n, m.nv), np.float32)); v = sim.forward(); sim.after(v); sim.synchronize()
    return m, sim


def test_once_per_tick_contacts_follow_the_bodies():
    """The second substep of a tick keeps the first substep's contacts and moves their witness points with the bodies:
    for plane-vertex contacts the refreshed distance equals a fresh collision pass on the moved bodies."""
    import mujoco_warp as mjw
    m, sim = _standing_sim("isaaclab3")
    d = sim.d; dev = sim.device
    la = wp.zeros(d.naconmax, dtype=wp.vec3, device=dev); lb = wp.zeros(d.naconmax, dtype=wp.vec3, device=dev)
    with wp.ScopedDevice(dev):
        wp.launch(solver_presets._save_witness, dim=d.naconmax, inputs=[d.nacon, d.contact.dist, d.contact.pos, d.contact.frame,
                  d.contact.geom, d.contact.worldid, sim.m.geom_bodyid, d.xpos, d.xmat, la, lb])
    sim.synchronize()
    nc = int(d.nacon.numpy()[0]); assert nc >= 8
    # move the robots: 1 mm up and a small rotation of the whole robot (larger in each world)
    q = d.qpos.numpy().copy()
    q[:, 2] += 0.001
    for e in range(sim.n):
        rq = np.zeros(4); mujoco.mju_axisAngle2Quat(rq, np.array([0.3, 0.2, 1.0]) / np.linalg.norm([0.3, 0.2, 1.0]), 0.004 * (e + 1))
        out = np.zeros(4); mujoco.mju_mulQuat(out, rq, q[e, 3:7].astype(np.float64)); q[e, 3:7] = out
    d.qpos.assign(q)
    with wp.ScopedDevice(dev):
        mjw.kinematics(sim.m, d)
        wp.launch(solver_presets._refresh_contacts, dim=d.naconmax, inputs=[d.nacon, d.contact.frame, d.contact.geom,
                  d.contact.worldid, sim.m.geom_bodyid, d.xpos, d.xmat, la, lb, d.contact.dist, d.contact.pos, d.contact.efc_address])
    sim.synchronize()
    # the foot-side witness point: refreshed contacts carry it in the body frame (lb), fresh plane-vertex contacts at
    # pos + dist/2 normal (their witness pair is aligned with the normal)
    keys = ("dist", "pos", "geom", "worldid", "frame")
    ref = {k: getattr(d.contact, k).numpy()[:nc].copy() for k in keys}
    assert np.all(d.contact.efc_address.numpy()[:nc] == -1)
    xp = d.xpos.numpy(); xm = d.xmat.numpy(); gb = m.geom_bodyid; lbn = lb.numpy()[:nc]
    pr = np.array([xp[w, gb[g[1]]] + xm[w, gb[g[1]]] @ lbn[c] for c, (w, g) in enumerate(zip(ref["worldid"], ref["geom"]))])
    with wp.ScopedDevice(dev):
        mjw.forward(sim.m, d)                                  # fresh collision on the moved bodies
    sim.synchronize()
    nf = int(d.nacon.numpy()[0])
    fresh = {k: getattr(d.contact, k).numpy()[:nf] for k in keys}
    pf = fresh["pos"] + 0.5 * fresh["dist"][:, None] * fresh["frame"][:, 0, :]
    matched = 0
    for c in range(nf):
        cand = np.nonzero((ref["worldid"] == fresh["worldid"][c]) & np.all(ref["geom"] == fresh["geom"][c], 1))[0]
        if not len(cand):
            continue
        k = cand[np.argmin(np.linalg.norm(pr[cand] - pf[c], axis=1))]
        if np.linalg.norm(pr[k] - pf[c]) < 1e-4:               # the same vertex
            np.testing.assert_allclose(ref["dist"][k], fresh["dist"][c], atol=2e-6)
            matched += 1
    assert matched >= 0.9 * nf


def test_once_per_tick_collision_step_is_stable_and_close():
    """A 0.5 s stand on the preset with collision once per 5 ms tick vs every 2.5 ms substep: finite, and the pelvis
    height within 5 mm (the two differ only in which contacts the second substep sees)."""
    out = {}
    for preset in ("isaaclab3", "isaaclab3_collide_every_substep"):
        m, sim = _standing_sim(preset, substeps=8)
        q = np.tile(m.key_qpos[0], (sim.n, 1)).astype(np.float32)
        sim.set_state(q, np.zeros((sim.n, m.nv), np.float32)); v = sim.forward(); sim.after(v); sim.synchronize()
        solver_presets.install(sim, preset)
        assert sim.collision_every == solver_presets.PRESETS[preset].collision_every
        sim.t.ctrl.copy_(sim.t.qpos[:, 7:].clone())
        for _ in range(25):
            sim.step()
        sim.synchronize()
        qp = sim.d.qpos.numpy(); assert np.all(np.isfinite(qp))
        out[preset] = qp[:, 2]
    assert np.abs(out["isaaclab3"] - out["isaaclab3_collide_every_substep"]).max() < 5e-3


def test_iteration_cap_only_changes_worlds_that_reach_it():
    """MuJoCo Warp exits every world at the tolerance (graph conditional or not): with caps 100 and 20 the states agree
    to float noise when no world needs more than 20 iterations."""
    res = {}
    for preset in ("isaaclab3", "isaaclab3_cap20"):
        m, sim = _standing_sim(preset, substeps=8)
        solver_presets.install(sim, preset)
        sim.t.ctrl.copy_(sim.t.qpos[:, 7:].clone())
        for _ in range(10):
            sim.step()
        sim.synchronize()
        res[preset] = (sim.d.qpos.numpy().copy(), int(sim.d.solver_niter.numpy().max()), sim.overflow_flags())
    assert res["isaaclab3"][1] < 20 and "ITERATIONS" not in res["isaaclab3_cap20"][2]
    # measured 2026-09-25: not bit-identical (max 1.2e-7 after 0.2 s: the extra, masked iterations still launch some
    # unmasked bookkeeping kernels), i.e. float noise
    np.testing.assert_allclose(res["isaaclab3"][0], res["isaaclab3_cap20"][0], atol=1e-5)


@pytest.mark.parametrize("contact_cfg,solver_cfg", [("default", None), ("recommended", None), ("recommended", "isaaclab3"),
                                                    ("recommended", "isaaclab3_every_substep_cap20"),
                                                    ("recommended", "isaaclab3_hardlimits"), ("default", "isaaclab3")])
def test_effective_model_fields_per_preset_combination(contact_cfg, solver_cfg):
    """The task's effective contact / limit fields equal the intended preset alone applied to the raw model: contact_cfg
    first, then solver_cfg (solver_presets docstring), with the solver preset owning every field it touches."""
    from metalsim.learn.g1_velocity import G1VelocityTask
    from metalsim.physics import contact_tuning
    task = G1VelocityTask(2, terrain="flat", seed=0, physics_dt=0.0025, reward_cfg="flat", contact_cfg=contact_cfg, solver_cfg=solver_cfg)
    ref, _ = build_g1_model("flat", physics_dt=0.0025)
    if solver_cfg is None:
        contact_tuning.apply(ref, contact_cfg)
    else:
        solver_presets.apply(ref, solver_cfg)            # alone, on MuJoCo's defaults
    m = task.model
    for f in ("geom_solref", "geom_solimp", "geom_gap", "geom_margin", "jnt_solref", "jnt_solimp", "geom_friction"):
        np.testing.assert_allclose(getattr(m, f), getattr(ref, f), rtol=1e-7, err_msg=f"{contact_cfg}+{solver_cfg}: {f}")
    for f in ("iterations", "ls_iterations", "tolerance", "ls_tolerance", "cone", "impratio", "integrator"):
        assert getattr(m.opt, f) == getattr(ref.opt, f), (contact_cfg, solver_cfg, f)


def test_g1_actuator_and_joint_fields_match_newtons_model():
    """The G1 drive as Newton builds Isaac Lab 3.0's MuJoCo Warp model (newton_generated.xml): per joint a position <general>
    (gain kp, bias -kp q) and a velocity <general> (gain kd, bias -kd qd), no actuator force limit, the effort limit on
    the joint's actuatorfrcrange. MetalSim's single affine actuator is the same sum; the effort limit is on the joint too
    (build_g1_model effort_limit="joint"; "actuator" = the archived layout)."""
    import re
    X = open("runs/parity3/isaac/fidelity/newton_mjwarp/newton_generated.xml").read()
    m, _ = build_g1_model("flat", physics_dt=0.0025)
    for a in range(m.nu):
        j = m.actuator_trnid[a][0]; nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        acts = re.findall(r'<general joint="[^"]*_%s" biastype="affine" gainprm="([^"]*)" biasprm="([^"]*)"' % nm, X)
        assert len(acts) == 2, nm
        kp = float(acts[0][0]); kd = float(acts[1][0])
        assert [float(x) for x in acts[0][1].split()] == [0.0, -kp] and [float(x) for x in acts[1][1].split()] == [0.0, 0.0, -kd]
        np.testing.assert_allclose([m.actuator_gainprm[a][0], *m.actuator_biasprm[a][:3]], [kp, 0.0, -kp, -kd])
        jx = re.search(r'<joint name="[^"]*_%s"[^>]*>' % nm, X).group(0)
        lo, hi = (float(x) for x in re.search(r'actuatorfrcrange="([^"]*)"', jx).group(1).split())
        assert not m.actuator_forcelimited[a] and m.jnt_actfrclimited[j]
        np.testing.assert_allclose(m.jnt_actfrcrange[j], [lo, hi])
        np.testing.assert_allclose(m.dof_armature[m.jnt_dofadr[j]], float(re.search(r'armature="([^"]*)"', jx).group(1)))
    old, _ = build_g1_model("flat", physics_dt=0.0025, effort_limit="actuator")
    assert old.actuator_forcelimited.all() and not old.jnt_actfrclimited[1:].any()


def test_saturated_finger_keeps_implicit_damping():
    """Unit proof (MuJoCo C): a finger driven towards its limit at 60 rad/s with its drive saturated stays bounded with the
    effort limit on the joint, and blows up within a few substeps with it on the actuator (MuJoCo drops a clamped
    actuator's velocity derivative)."""
    import copy
    base, _ = build_g1_model("flat", physics_dt=0.0025)
    for layout, ok in (("joint", True), ("actuator", False)):
        m, _ = build_g1_model("flat", physics_dt=0.0025, effort_limit=layout); m.opt.gravity[:] = 0
        solver_presets.apply(m, "isaaclab3_every_substep_cap20")
        d = mujoco.MjData(m); mujoco.mj_resetDataKeyframe(m, d, 0)
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "left_four_joint"); a = [i for i in range(m.nu) if m.actuator_trnid[i][0] == j][0]
        d.ctrl[:] = m.key_qpos[0][7:]; d.ctrl[a] = -1.84; d.qpos[m.jnt_qposadr[j]] = -1.70; d.qvel[m.jnt_dofadr[j]] = -60.0
        vmax = 0.0
        for _ in range(40):
            mujoco.mj_step(m, d); v = abs(float(d.qvel[m.jnt_dofadr[j]])); vmax = max(vmax, v if np.isfinite(v) else 1e9)
        assert (vmax < 100.0) == ok, (layout, vmax)

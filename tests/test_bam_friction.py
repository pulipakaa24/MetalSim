"""BAM's load-dependent servo friction (metalsim.physics.actuation): the MuJoCo Warp update ``BamFrictionWarp`` against
the CPU one (``BamFriction`` / ``bam_friction_budget``).

* The kernel on Warp's CPU device, on a stand-in for BatchSim's arrays: every BAM friction model (m1..m6), random and
  edge-case torques (zero, equal magnitudes, signs), constraint rows of other types, stale rows past ``nefc`` and an
  overflowed ``nefc``. Float32 against float64.
* On Metal (skipped without it): the SO-101 arm in a BatchSim, the kernel as a substep hook, against CPU MuJoCo with
  ``BamFriction.update`` before every ``mj_step``, from the same states.
"""
import os
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import warp as wp

from metalsim.physics.actuation import _BAM_FLAGS, BamFriction, BamFrictionWarp, bam_friction_budget

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SO101 = os.path.join(ROOT, "assets", "so101", "scene_box_rl.xml")
ARM = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
FRICTION_DOF = int(mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF)

# Rhoban BAM, bam/params/feetech_sts3215_7_4V/m6.json (github.com/Rhoban/bam b9e1111; so101.py BAM_M6); the m3 / m4
# non-directional load coefficients are m3.json's.
M6 = dict(model="m6", friction_base=0.0533134815909817, friction_stribeck=4.257322248151969e-05,
          load_friction_motor=0.044912012282132555, load_friction_external=0.24355483356616028,
          load_friction_motor_stribeck=0.00012448975632695385, load_friction_external_stribeck=0.1422905635963302,
          load_friction_motor_quad=0.007594943296670792, load_friction_external_quad=0.00340408727365082,
          dtheta_stribeck=0.371136775424206, alpha=9.93094944677977, kt=1.2753202369175398, R=2.753421214375944)
NON_DIR = dict(load_friction_base=0.21, load_friction_stribeck=0.05)


def _params(model):
    return dict(M6, **NON_DIR, model=model)


def _fake_sim(n, nv, njmax, device):
    """The fields of BatchSim that BamFrictionWarp reads and writes, on ``device``."""
    z = lambda *s, dt=wp.float32: wp.zeros(s, dtype=dt, device=device)
    d = SimpleNamespace(nefc=z(n, dt=wp.int32), efc=SimpleNamespace(type=z(n, njmax, dt=wp.int32), id=z(n, njmax, dt=wp.int32),
                                                                   force=z(n, njmax)),
                        qfrc_bias=z(n, nv), qfrc_constraint=z(n, nv), qfrc_actuator=z(n, nv), qvel=z(n, nv))
    hooks = []
    return SimpleNamespace(n=n, device=wp.get_device(device), d=d, m=SimpleNamespace(dof_frictionloss=z(n, nv)),
                           add_substep_hook=lambda fn, reset_fn=None: hooks.append((fn, reset_fn)), hooks=hooks,
                           synchronize=lambda: wp.synchronize_device(device))


def _cpu_reference(p, emf, dofs, nefc, etype, eid, eforce, bias, cons, act, qvel):
    """BamFriction.update's arithmetic in float64, per world."""
    out = np.zeros((len(nefc), len(dofs)))
    for w in range(len(nefc)):
        own = np.zeros(bias.shape[1])
        n = min(int(nefc[w]), etype.shape[1])
        k = etype[w, :n] == FRICTION_DOF
        np.add.at(own, eid[w, :n][k], eforce[w, :n][k])
        te = -bias[w, dofs] + cons[w, dofs] - own[dofs]
        tm = act[w, dofs] - emf * qvel[w, dofs]
        out[w] = bam_friction_budget(p, tm, te, qvel[w, dofs])
    return out


@pytest.mark.parametrize("model", sorted(_BAM_FLAGS))
def test_kernel_matches_cpu_budget(model):
    if not wp.is_cpu_available():
        pytest.skip("needs Warp's CPU device")
    rng = np.random.default_rng(7)
    n, nv, njmax = 64, 12, 24
    dofs = np.array([0, 1, 2, 3, 4, 5])
    p, emf = _params(model), M6["kt"] ** 2 / M6["R"]
    sim = _fake_sim(n, nv, njmax, "cpu")
    fh = BamFrictionWarp(sim, dofs, p, emf)
    assert len(sim.hooks) == 1
    # constraint rows: one friction row per arm dof in shuffled order among contact / limit rows; stale friction rows
    # past nefc (must be ignored); worlds 0-3 overflowed (nefc > njmax)
    etype = rng.choice([FRICTION_DOF, int(mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT),
                        int(mujoco.mjtConstraint.mjCNSTR_CONTACT_ELLIPTIC)], size=(n, njmax)).astype(np.int32)
    eid = rng.integers(0, nv, size=(n, njmax)).astype(np.int32)
    nefc = rng.integers(0, njmax, size=n).astype(np.int32)
    nefc[:4] = njmax + 5
    eforce = rng.normal(0, 0.3, size=(n, njmax))
    # torques: random, plus edge cases (zeros, equal magnitudes of either sign) and speeds around dtheta_stribeck
    bias, cons = rng.normal(0, 0.5, (n, nv)), rng.normal(0, 0.2, (n, nv))
    act = rng.normal(0, 0.8, (n, nv))
    qvel = rng.normal(0, 1, (n, nv)) * rng.choice([0.0, 0.1, 0.37, 1.0, 5.0], size=(n, nv))
    act[8:12], qvel[8:12] = 0.0, 0.0                                        # tau_m = 0
    bias[12:16], cons[12:16] = 0.0, 0.0
    nefc[12:16] = 0                                                         # tau_e = 0
    act[16:20], qvel[16:20], cons[16:20], nefc[16:20] = bias[16:20], 0.0, 0.0, 0   # tau_m = -tau_e
    act[20:24], qvel[20:24], cons[20:24], nefc[20:24] = -bias[20:24], 0.0, 0.0, 0  # tau_m = tau_e
    f32 = lambda a: a.astype(np.float32)
    bias, cons, act, qvel, eforce = map(lambda a: f32(a).astype(np.float64), (bias, cons, act, qvel, eforce))
    for a, v in ((sim.d.nefc, nefc), (sim.d.efc.type, etype), (sim.d.efc.id, eid), (sim.d.efc.force, f32(eforce)),
                 (sim.d.qfrc_bias, f32(bias)), (sim.d.qfrc_constraint, f32(cons)), (sim.d.qfrc_actuator, f32(act)),
                 (sim.d.qvel, f32(qvel))):
        a.assign(v)
    fh.update()
    got = sim.m.dof_frictionloss.numpy()[:, dofs].astype(np.float64)
    want = _cpu_reference(p, emf, dofs, nefc, etype, eid, eforce, bias, cons, act, qvel)
    np.testing.assert_allclose(got, want, rtol=2e-6, atol=2e-7)
    # the reset hook writes the budget at rest to the selected worlds only
    mask = np.zeros(n, bool); mask[::3] = True
    sim.hooks[0][1](wp.array(mask, dtype=wp.bool, device="cpu"))
    after = sim.m.dof_frictionloss.numpy()[:, dofs]
    z = np.zeros(len(dofs))
    np.testing.assert_allclose(after[mask], np.broadcast_to(bam_friction_budget(p, z, z, z), after[mask].shape), rtol=1e-6)
    np.testing.assert_array_equal(after[~mask], got[~mask].astype(np.float32))


def test_needs_per_world_frictionloss():
    if not wp.is_cpu_available():
        pytest.skip("needs Warp's CPU device")
    sim = _fake_sim(4, 12, 8, "cpu")
    sim.m.dof_frictionloss = wp.zeros((1, 12), dtype=wp.float32, device="cpu")
    with pytest.raises(ValueError, match="per_world_fields"):
        BamFrictionWarp(sim, range(6), _params("m6"))


@pytest.mark.skipif(not wp.is_metal_available(), reason="needs Metal")
def test_batchsim_matches_cpu_mujoco():
    """SO-101 arm, m6 friction on the six arm dofs, stiff friction constraint, 4 worlds holding / moving against
    gravity for 60 physics steps: MetalSim (float32) against CPU MuJoCo (float64) from the same start."""
    from metalsim.physics.batch import BatchSim, BatchSimOptions

    m = mujoco.MjModel.from_xml_path(SO101)
    dofs = np.array([m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in ARM])
    p, emf = _params("m6"), M6["kt"] ** 2 / M6["R"]
    # BAM's m6 servo at LeRobot's P 16 (so101.py "7v4_bam"): kp = kt 7.4 / R x 16 x 0.166 x 1.1617, duty clip 0.97,
    # back-EMF + viscous as joint damping, armature
    full = M6["kt"] * 7.4 / M6["R"]
    m.actuator_gainprm[:6, 0] = full * 16 * 0.166 * 1.1616979929275297
    m.actuator_biasprm[:6, 1] = -m.actuator_gainprm[:6, 0]
    m.actuator_biasprm[:6, 2] = 0.0
    m.actuator_forcerange[:6] = (-0.97 * full, 0.97 * full)
    m.dof_damping[dofs] = emf + 0.028160537042911545
    m.dof_armature[dofs] = 0.021564290129722483
    cpu = BamFriction(m, dofs, p, emf)
    cpu.stiffen()
    cpu.reset()
    n, steps = 4, 60
    rng = np.random.default_rng(3)
    q0 = np.tile(m.qpos0, (n, 1))
    q0[:, dofs] += rng.uniform(-0.3, 0.3, (n, len(dofs)))
    ctrl = np.tile(q0[:, dofs], (1, 1))
    ctrl[1::2, 1] += 0.4                                                    # worlds 1, 3: shoulder moves
    sim = BatchSim(m, n, options=BatchSimOptions(substeps=1, njmax=256, per_world_fields=("dof_frictionloss",)))
    gpu = BamFrictionWarp(sim, dofs, p, emf)
    sim.reset()
    c = np.zeros((n, m.nu), np.float32); c[:, :len(dofs)] = ctrl
    with wp.ScopedDevice(sim.device):
        sim.d.ctrl.assign(c)
    sim.set_state(q0, np.zeros((n, m.nv)))                                  # forward with the ctrl, then the budget
    gpu.update()
    qg = np.zeros((steps, n, len(dofs)))
    fg = np.zeros((steps, n, len(dofs)))
    for t in range(steps):
        sim.step()
        sim.synchronize()
        qg[t] = sim.d.qpos.numpy()[:, dofs]
        fg[t] = sim.m.dof_frictionloss.numpy()[:, dofs]
    for w in range(n):
        d = mujoco.MjData(m)
        cpu.reset()
        d.qpos[:] = q0[w]
        d.ctrl[:len(dofs)] = ctrl[w]
        mujoco.mj_forward(m, d)
        for t in range(steps):
            cpu.update(d)
            mujoco.mj_step(m, d)
            assert np.abs(d.qpos[dofs] - qg[t, w]).max() < 1e-4, (w, t)
            cpu.update(d)                                                   # the budget GPU wrote after this step
            np.testing.assert_allclose(m.dof_frictionloss[dofs], fg[t, w], rtol=1e-3, atol=1e-4)

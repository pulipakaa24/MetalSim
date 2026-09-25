"""Deformables on Metal: MuJoCo Warp flex (cloth + cable) against MuJoCo C, energy, graph replay; and the
MetalSim-native XPBD cloth (same topology) against itself on the CPU, energy, graph replay, and against
MuJoCo Warp's draped shape. GPU tests: run through scripts/gpu_run.sh."""
import os

import mujoco
import numpy as np
import pytest
import warp as wp

from metalsim.physics import deformable as dfm

pytestmark = pytest.mark.skipif(not wp.is_metal_available(), reason="needs Metal")
DEV = os.environ.get("METALSIM_DEFORMABLE_DEVICE", "metal:0")   # "cpu" for a dry run of the logic


def _box(**kw):
    return dfm.scene_model("box", **kw)


# ------------------------------------------------------------------------------------------------
# MuJoCo Warp flex path
# ------------------------------------------------------------------------------------------------

def test_mjw_flex_free_fall_matches_mujoco_c():
    """Cloth and cable under gravity with their edge-equality constraints active (kinematics, flex edge
    Jacobians, constraint rows, CG solve): every world equals MuJoCo C (float64) over 60 steps."""
    m = _box()
    sim = dfm.DeformableSim(m, 4, device=DEV)
    q = np.tile(sim.mjd0.qpos, (4, 1))
    sim.set_qpos(q)
    ref = dfm.mjc_rollout(m, q[0], 60)
    assert ref["ncon"].max() == 0
    for k in range(60):
        sim.step()
        if k in (0, 9, 59):
            v = sim.d.flexvert_xpos.numpy()
            assert np.abs(v - ref["vert"][k][None]).max() < 1e-5, k
    e = sim.d.energy.numpy()
    np.testing.assert_allclose(e, np.tile(ref["energy"][-1], (4, 1)), atol=2e-5)


def _fidelity_model(cable_pos=(0.12, 0.28, 0.40)):
    # the fidelity comparisons use a converged solver (CG 100 / 50) so solver truncation does not dominate
    return mujoco.MjModel.from_xml_string(dfm.box_scene_xml(dfm.ClothCfg(), dfm.CableCfg(pos=cable_pos),
                                                             iterations=100, ls_iterations=50))


def test_mjw_flex_contact_trajectory_matches_mujoco_c():
    """Cloth landing on the box and cable on the floor from a perturbed (tie-free) start: 100 steps (0.2 s, first
    contact at ~step 60, 50 box-cloth contacts selected from ~380 candidates) track MuJoCo C (float64) to 1 mm. Requires the metalsim-flex MuJoCo Warp branch (MuJoCo C's
    contact cap/selection, box-triangle and cable capsule contacts)."""
    m = _fidelity_model()
    sim = dfm.DeformableSim(m, 2, device=DEV)
    q0 = sim.mjd0.qpos + np.random.default_rng(0).normal(0, 2e-3, m.nq)
    sim.set_qpos(np.tile(q0, (2, 1)))
    ref = dfm.mjc_rollout(m, q0, 100)
    assert ref["ncon"][-1] >= 50
    for k in range(100):
        sim.step()
    x = sim.d.flexvert_xpos.numpy()
    assert np.abs(x - ref["vert"][-1][None]).max() < 1e-3, np.abs(x - ref["vert"][-1][None]).max()


def test_mjw_flex_one_step_matches_mujoco_c_in_contact():
    """The cable falls across the cloth on the box (geom-flex and flex-flex contacts). From MuJoCo C's own states
    (every 10th of 300 steps once in contact), one MuJoCo Warp step gives MuJoCo C's contact set (geom, flex,
    element, vertex) apart from flex-flex contacts shallower than 10 um (float32 CCD at the activation boundary),
    and its velocities: median error < 1e-4 m/s, max < 0.05 m/s."""
    m = _fidelity_model(cable_pos=(-0.15, 0.0, 0.35))
    sim = dfm.DeformableSim(m, 1, device=DEV, capture=False)
    d = mujoco.MjData(m)
    d.qpos[:] = sim.mjd0.qpos + np.random.default_rng(0).normal(0, 2e-3, m.nq)
    mujoco.mj_forward(m, d)

    def key(c, deep_only=False):
        rows = np.stack([c.geom[:, 0], c.flex[:, 1], c.elem[:, 1], c.vert[:, 1]], 1)
        keep = ~((c.geom[:, 0] < 0) & (np.abs(c.dist) < 1e-5)) if deep_only else np.ones(len(rows), bool)
        return sorted(map(tuple, rows[keep].tolist()))

    errs = []
    for k in range(300):
        if k % 10 == 0 and k >= 60:
            sim.set_qpos(d.qpos[None], d.qvel[None])
            sim.step(eager=True)
            w = sim.get_world(0)
            mujoco.mj_step(m, d)
            assert d.ncon > 0 and key(w.contact[:w.ncon], True) == key(d.contact[:d.ncon], True), k
            errs.append(np.abs(w.qvel - d.qvel).max())
        else:
            mujoco.mj_step(m, d)
    assert len(errs) >= 20
    assert np.median(errs) < 1e-4 and max(errs) < 0.05, (np.median(errs), max(errs))


def test_mjw_flex_energy_bounded_2s():
    """64 randomized worlds, 2 s: finite, no constraint/contact overflow, total energy never above its
    initial value (the edge constraints and contacts only dissipate), cloth settles on the box."""
    m = _box()
    sim = dfm.DeformableSim(m, 64, device=DEV)
    sim.randomize(seed=3)
    sim.synchronize()
    e0 = sim.d.energy.numpy().sum(1)
    emax = e0.copy()
    nsteps = int(round(2.0 / m.opt.timestep))
    for k in range(nsteps):
        sim.step()
        if k % 50 == 49:
            e = sim.d.energy.numpy().sum(1)
            assert np.isfinite(e).all(), k
            emax = np.maximum(emax, e)
    ov = sim.overflow()
    assert not {k: v for k, v in ov.items() if k in ("NEFC", "NARROWPHASE", "NCON", "CCD")}, ov
    assert (emax - e0).max() < 0.01 * np.abs(e0).max(), (emax - e0).max()
    x = sim.d.flexvert_xpos.numpy()
    ke = sim.d.energy.numpy()[:, 1]
    assert ke.max() < 0.02, ke.max()
    assert x[:, :, 2].min() > -0.01                       # nothing through the floor
    cloth = x[:, :m.flex_vertnum[0]]
    assert (cloth[:, :, 2].max(1) > 0.29).all()           # cloth lies on the 0.3 m box


def test_mjw_flex_graph_replay_equals_eager():
    """Graph replay vs eager launches of the same step. Free fall with the edge constraints (60 steps) is
    deterministic, so the two agree to float rounding. Through contact MuJoCo Warp appends contacts and their
    constraint rows with atomics, whose order varies from run to run on the GPU, and the draped cloth amplifies
    float rounding, so the reference there is a second eager run: graph-vs-eager must be of the same size as
    eager-vs-eager (measured on Metal: 3.19e-2 vs 3.18e-2 m after 150 steps)."""
    m = _box()
    a = dfm.DeformableSim(m, 16, device=DEV, capture=True)
    b = dfm.DeformableSim(m, 16, device=DEV, capture=False)
    c = dfm.DeformableSim(m, 16, device=DEV, capture=False)
    q = a.randomize(seed=5)
    b.set_qpos(q)
    c.set_qpos(q)
    for k in range(150):
        a.step()
        b.step(eager=True)
        c.step(eager=True)
        if k == 59:
            a.synchronize(); b.synchronize()
            free = np.abs(a.d.qpos.numpy() - b.d.qpos.numpy()).max()
            assert free < 1e-6, free
    a.synchronize(); b.synchronize(); c.synchronize()
    ab = np.abs(a.d.qpos.numpy() - b.d.qpos.numpy()).max()
    bc = np.abs(b.d.qpos.numpy() - c.d.qpos.numpy()).max()
    print(f"graph-vs-eager {ab:.2e}, eager-vs-eager {bc:.2e} after 150 steps")
    assert ab <= 2.0 * bc + 1e-5, (ab, bc)


# ------------------------------------------------------------------------------------------------
# MetalSim-native XPBD
# ------------------------------------------------------------------------------------------------

def test_xpbd_metal_equals_cpu():
    m = _box()
    s_gpu = dfm.XPBDSim(m, 4, device=DEV)
    s_cpu = dfm.XPBDSim(m, 4, device="cpu", capture=False)
    x = s_gpu.randomize(seed=2)
    s_cpu.set_vertices(x)
    for _ in range(100):
        s_gpu.step(); s_cpu.step()
    s_gpu.synchronize()
    assert np.abs(s_gpu.x.numpy() - s_cpu.x.numpy()).max() < 1e-4


def test_xpbd_graph_replay_equals_eager():
    m = _box()
    a = dfm.XPBDSim(m, 16, device=DEV, capture=True)
    b = dfm.XPBDSim(m, 16, device=DEV, capture=False)
    b.set_vertices(a.randomize(seed=4))
    for _ in range(150):
        a.step(); b.step(eager=True)
    a.synchronize()
    assert np.array_equal(a.x.numpy(), b.x.numpy())


def test_xpbd_energy_strain_penetration_2s():
    m = _box()
    sim = dfm.XPBDSim(m, 64, device=DEV)
    sim.randomize(seed=3)
    sim.step(); sim.synchronize()
    e0 = sim.energy_arr.numpy().sum(1)
    emax = e0.copy()
    for k in range(int(round(2.0 / sim.dt))):
        sim.step()
        if k % 50 == 49:
            e = sim.energy_arr.numpy().sum(1)
            assert np.isfinite(e).all()
            emax = np.maximum(emax, e)
    assert (emax - e0).max() < 0.01 * np.abs(e0).max()
    x = sim.x.numpy()
    r = sim.radius.numpy()
    assert x[:, :, 2].min() > r.min() - 1e-3                               # floor
    top = (np.abs(x[:, :, 0]) < 0.14) & (np.abs(x[:, :, 1]) < 0.14)
    assert x[:, :, 2][top].min() > 0.30 + r.min() - 1e-3                    # box top
    assert max(np.abs(sim.edge_strain(w)).max() for w in (0, 17, 63)) < 0.005   # < 0.5 % stretch
    assert sim.energy_arr.numpy()[:, 1].max() < 1e-3                         # at rest


def test_xpbd_drape_agrees_with_mjw_flex():
    """Same cloth, same box: XPBD's rest shape against MuJoCo Warp flex's (different constitutive models,
    so a loose bound: the cloth's centre sits on the box top and its mean height agrees to 2 cm)."""
    m = _box()
    a = dfm.DeformableSim(m, 1, device=DEV)
    b = dfm.XPBDSim(m, 1, device=DEV)
    for _ in range(750):
        a.step(); b.step()
    a.synchronize(); b.synchronize()
    nc = m.flex_vertnum[0]
    xa, xb = a.d.flexvert_xpos.numpy()[0, :nc], b.x.numpy()[0, :nc]
    centre = np.argmin(np.linalg.norm(xa[:, :2], axis=1))
    assert abs(xa[centre, 2] - xb[centre, 2]) < 0.01
    assert abs(xa[:, 2].mean() - xb[:, 2].mean()) < 0.02, (xa[:, 2].mean(), xb[:, 2].mean())


def test_g1_cloth_scenes_run():
    """Cloth + cable on Isaac's G1: MuJoCo Warp flex and XPBD (two-way through xfrc_applied) stay finite."""
    fm = dfm.g1_scene_model()
    a = dfm.DeformableSim(fm, 4, device=DEV)
    b = dfm.XPBDSim(fm, 4, rigid_model=dfm.g1_scene_model(flexes=False), device=DEV, cfg=dfm.XPBDCfg(two_way=True))
    for _ in range(200):
        a.step(); b.step()
    a.synchronize(); b.synchronize()
    assert np.isfinite(a.d.qpos.numpy()).all() and np.isfinite(b.x.numpy()).all()
    assert b.x.numpy()[:, :144, 2].max() < 1.5                               # the cloth fell onto the robot

"""Contact sensor (Isaac Lab ContactSensor equivalent, metalsim.sensors.contact) on MuJoCo Warp:
net force per body vs MuJoCo C's mj_contactForce, history, pair filtering, and Isaac's air-time
formulas (assets/isaac/sensors/contact_sensor/contact_sensor.py, v2.3.2)."""
import mujoco
import numpy as np
import pytest
import torch
import warp as wp

from metalsim.physics.batch import BatchSim, BatchSimOptions
from metalsim.sensors.contact import ContactSensor

pytestmark = pytest.mark.skipif(not wp.is_metal_available(), reason="needs Metal")

STACK = """
<mujoco><option timestep="0.002"/>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1"/>
    <body name="low" pos="0 0 0.06"><freejoint/><geom name="gl" type="box" size="0.1 0.1 0.05" mass="2"/></body>
    <body name="top" pos="0 0 0.17"><freejoint/><geom name="gt" type="box" size="0.05 0.05 0.05" mass="1"/></body>
  </worldbody>
</mujoco>"""

LIFT = """
<mujoco><option timestep="0.002"/>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1"/>
    <body name="foot" pos="0 0 0.05"><joint name="z" type="slide" axis="0 0 1" damping="20"/>
      <geom name="g" type="box" size="0.05 0.05 0.05" mass="1"/></body>
  </worldbody>
  <actuator><position name="p" joint="z" kp="2000"/></actuator>
</mujoco>"""


def _c_body_forces(m, d, bodies, total=True):
    """Reference: MuJoCo C mj_contactForce, world frame, +F on geom[1]'s body, -F on geom[0]'s."""
    out = np.zeros((len(bodies), 3)); f6 = np.zeros(6)
    for i in range(d.ncon):
        c = d.contact[i]
        if c.efc_address < 0:
            continue
        mujoco.mj_contactForce(m, d, i, f6)
        fr = c.frame.reshape(3, 3)
        f = f6[:3] @ fr if total else f6[0] * fr[0]
        for g, s in ((c.geom[1], 1.0), (c.geom[0], -1.0)):
            b = m.geom_bodyid[g]
            if b in bodies:
                out[bodies.index(b)] += s * f
    return out


def test_stack_sign_history_filter_and_graph():
    """Two stacked boxes at rest: the lower box reads +(m1+m2) g from the floor and -m2 g from the
    top box (filtered), the net force is their sum; history rolls by one per substep; the sensor
    runs inside the replayed step graph without host synchronization."""
    from metalsim.interop import warp_metal as wm
    m = mujoco.MjModel.from_xml_string(STACK)
    n = 8
    sim = BatchSim(m, n, options=BatchSimOptions(substeps=1, njmax=128))
    cs = ContactSensor(sim, ["low", "top"], history_length=3, filter=[["floor"], "top"], force_mode="total")
    assert tuple(cs.net_forces_w.shape) == (n, 2, 3) and tuple(cs.net_forces_w_history.shape) == (n, 3, 2, 3)
    assert tuple(cs.force_matrix_w.shape) == (n, 2, 2, 3) and tuple(cs.force_matrix_w_history.shape) == (n, 3, 2, 2, 3)
    for _ in range(400):
        sim.step()
    sim.synchronize()
    hist_ref = []
    c0 = wm.counters()
    for _ in range(3):
        v = sim.step(); sim.after(v)
        hist_ref.append(cs.net_forces_w.clone())
    torch.mps.synchronize()
    assert (wm.counters() - c0).syncs == 0
    o = cs.numpy()
    g = 9.81
    net, mat = o["net_forces_w"], o["force_matrix_w"]
    assert np.allclose(net[:, 0], [0, 0, 2 * g], atol=0.05), net[0]          # low: 3 g up from floor, 1 g down from top
    assert np.allclose(net[:, 1], [0, 0, 1 * g], atol=0.05), net[0]
    assert np.allclose(mat[:, 0, 0], [0, 0, 3 * g], atol=0.05) and np.allclose(mat[:, 0, 1], [0, 0, -g], atol=0.05)
    assert np.allclose(mat[:, 1, 0], 0, atol=1e-6)                           # top never touches the floor
    assert np.allclose(mat[:, 0].sum(1), net[:, 0], atol=1e-4)
    # history: index 0 = latest, index k = k substeps ago
    h = o["net_forces_w_history"]
    for k in range(3):
        assert np.array_equal(h[:, k], hist_ref[2 - k].cpu().numpy())
    # reset clears the masked worlds only
    mask = torch.zeros(n, dtype=torch.bool); mask[:3] = True
    sim.reset(mask); sim.synchronize()
    o2 = cs.numpy()
    assert np.all(o2["net_forces_w_history"][:3] == 0) and np.all(o2["net_forces_w_history"][3:] == h[3:])


def test_air_time_matches_isaac_formulas():
    """A foot lifted and lowered by a position actuator: current/last air and contact time equal
    Isaac Lab's update (replayed in numpy on the sensor's own per-substep net force), and
    compute_first_contact / compute_first_air follow."""
    m = mujoco.MjModel.from_xml_string(LIFT)
    n = 4
    sim = BatchSim(m, n, options=BatchSimOptions(substeps=1, njmax=64))
    cs = ContactSensor(sim, ["foot"], track_air_time=True, force_threshold=1.0)
    dt = m.opt.timestep
    ca = np.zeros((n, 1)); la = np.zeros((n, 1)); cc = np.zeros((n, 1)); lc = np.zeros((n, 1))
    n_first_contact = n_first_air = 0
    for k in range(600):
        up = (k // 100) % 2 == 1
        sim.t.ctrl.fill_(0.03 if up else -0.01)       # heights per world are equal; phases alternate every 0.2 s
        torch.mps.synchronize()
        sim.step(); sim.synchronize()
        o = cs.numpy()
        f = o["net_forces_w"]
        # Isaac Lab ContactSensor._update_buffers_impl, elapsed_time = dt
        is_c = np.linalg.norm(f, axis=-1) > 1.0
        first_c = (ca > 0) * is_c; first_d = (cc > 0) * ~is_c
        la = np.where(first_c, ca + dt, la); ca = np.where(~is_c, ca + dt, 0.0)
        lc = np.where(first_d, cc + dt, lc); cc = np.where(is_c, cc + dt, 0.0)
        for name, ref in (("current_air_time", ca), ("last_air_time", la), ("current_contact_time", cc), ("last_contact_time", lc)):
            assert np.allclose(o[name], ref, atol=1e-5), (k, name, o[name][0], ref[0])
        fc = cs.compute_first_contact(dt).cpu().numpy(); fa = cs.compute_first_air(dt).cpu().numpy()
        assert np.array_equal(fc, (cc > 0) & (cc < dt + 1e-8)) and np.array_equal(fa, (ca > 0) & (ca < dt + 1e-8))
        n_first_contact += int(fc.any()); n_first_air += int(fa.any())
    assert n_first_contact >= 2 and n_first_air >= 2          # the foot touched down and lifted off repeatedly
    assert (la > 0.1).all() and (lc > 0.1).all()


def test_g1_landing_vs_mujoco_c():
    """G1 (Isaac's asset) dropped 5-15 cm onto the floor, 1 s at the task's 5 ms step: per-body
    contact force on the feet and torso vs MuJoCo C.
    (a) the sensor's reduction equals MuJoCo C's mj_contactForce summed over the same contacts
        (MuJoCo Warp's contact set and efc_force decoded by C);
    (b) against an independent MuJoCo C forward from the same state, the total is within 1 % on
        every step where both engines find the same contact set. Steps where MuJoCo Warp finds fewer
        contacts are counted and reported: its plane-convex collision keeps only vertices within
        1 mm of the deepest (collision_primitive.plane_convex, threshold = max_support - 1e-3),
        so a tilted landing foot loses its shallower heel corners; MuJoCo C keeps all four."""
    import os
    from metalsim.learn.g1_velocity import build_g1_model
    if not os.path.exists(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "isaac", "G1", "g1_minimal.usd")):
        pytest.skip("needs Isaac's g1_minimal.usd")
    m, _ = build_g1_model("flat")
    n = 4
    sim = BatchSim(m, n, options=BatchSimOptions(substeps=1, njmax=256, nconmax=32, solver_iterations=10, ls_iterations=20))
    bid = lambda nm: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, nm)
    bodies = [bid("left_ankle_roll_link"), bid("right_ankle_roll_link"), bid("torso_link")]
    cs_t = ContactSensor(sim, bodies, history_length=3, force_mode="total", filter=["world"])
    cs_n = ContactSensor(sim, bodies, force_mode="normal")
    rng = np.random.default_rng(0)
    q = np.tile(m.key_qpos[0], (n, 1)); q[:, 2] += rng.uniform(0.05, 0.15, n)
    sim.set_state(q.astype(np.float32), np.zeros((n, m.nv), np.float32))
    sim.t.ctrl.copy_(torch.as_tensor(np.tile(m.key_qpos[0][7:], (n, 1)).astype(np.float32)))
    weight = mujoco.mj_getTotalmass(m) * 9.81
    agg_err = 0.0; rel_same = []; n_contact = n_diff = 0; zsum = []
    for k in range(200):
        pre = [sim.get_world(w) for w in range(n)]
        sim.step()
        ot, on = cs_t.numpy(), cs_n.numpy()
        for w in range(n):
            post = sim.get_world(w)            # MuJoCo Warp's own contacts and efc_force of this step
            agg_err = max(agg_err, np.abs(_c_body_forces(m, post, bodies) - ot["net_forces_w"][w]).max(),
                          np.abs(_c_body_forces(m, post, bodies, total=False) - on["net_forces_w"][w]).max())
            assert np.allclose(ot["force_matrix_w"][w, :, 0], ot["net_forces_w"][w], atol=1e-3)   # all contacts are with the ground
            mujoco.mj_forward(m, pre[w])       # MuJoCo C, same state, same solver budget
            ref = _c_body_forces(m, pre[w], bodies)
            tot = ref[:, 2].sum()
            if abs(tot) < 1.0:
                continue
            n_contact += 1
            zsum.append(ot["net_forces_w"][w][:, 2].sum())
            if pre[w].ncon != post.ncon:
                n_diff += 1; continue
            rel_same.append(abs(ot["net_forces_w"][w][:, 2].sum() - tot) / abs(tot))
            assert np.abs(ot["net_forces_w"][w] - ref).max() < 0.01 * max(abs(tot), weight)
    rel_same = np.array(rel_same)
    print(f"G1 landing: {n_contact} world-steps in contact; reduction vs C on the same contacts max {agg_err:.2e} N; "
          f"independent C forward, same contact set ({len(rel_same)}): total rel err median {np.median(rel_same):.1e} "
          f"max {rel_same.max():.1e}; different contact set: {n_diff}")
    assert agg_err < 1e-3
    assert rel_same.max() < 0.01 and len(rel_same) > 0.8 * n_contact
    assert min(zsum) > 0                        # feet are pushed up (+z, world frame)
    # history of the last 3 substeps: latest equals net
    assert np.array_equal(ot["net_forces_w_history"][:, 0], ot["net_forces_w"])

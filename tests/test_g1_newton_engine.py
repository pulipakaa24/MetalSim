"""Invariants of the Newton XPBD engine path of the G1 task (metalsim.physics.newton_backend.NewtonSim):
the MuJoCo-layout state it exposes means the same thing as MuJoCo's, contact "touch" signals are
physical forces, a fall is terminated and penalized like on MuJoCo Warp, resets restore Isaac's default
state, and the captured graph equals eager stepping. Shared observation/reward invariants are in
test_g1_task_terms.py (parametrized over both engines)."""
import mujoco
import numpy as np
import pytest
import warp as wp

from metalsim.learn.g1_velocity import G1VelocityTask, build_g1_model
from metalsim.learn.warp_policy import RolloutBuffers, bump, zero_int

pytestmark = pytest.mark.skipif(not wp.is_metal_available(), reason="needs Metal")
NEWTON = dict(engine="newton", newton_iterations=4, newton_dt=0.00125)


def _stepper(task):
    n = task.n
    class _Pol:
        step_idx = wp.zeros(1, dtype=int, device="metal:0")
    pol = _Pol(); bufs = RolloutBuffers(1, n, task.obs_dim, task.act_dim)
    bufs.rew = wp.zeros((1, n), dtype=float, device="metal:0"); bufs.done = wp.zeros((1, n), dtype=float, device="metal:0")
    def step(action):                         # one-row reward buffer: step index stays 1 (as g1_preflight)
        wp.launch(zero_int, dim=1, inputs=[pol.step_idx], device="metal:0")
        wp.launch(bump, dim=1, inputs=[pol.step_idx], device="metal:0")
        task.launch_apply_action(action); task.sim.launch_step(); task.launch_reward_done_reset(pol, bufs); task.launch_obs(pol.step_idx)
        task.sim.synchronize()
        return bufs.rew.numpy()[0].copy(), bufs.done.numpy()[0].copy()
    return step, pol


def test_policy_interface_identical_across_engines():
    a = G1VelocityTask(2, terrain="flat"); b = G1VelocityTask(2, terrain="flat", **NEWTON)
    assert (a.obs_dim, a.act_dim, a.max_t) == (b.obs_dim, b.act_dim, b.max_t)
    np.testing.assert_array_equal(a.default_q.numpy(), b.default_q.numpy())
    np.testing.assert_array_equal(a.group.numpy(), b.group.numpy())


def _rot_xyzw(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def test_state_conventions_match_mujoco():
    """qpos/qvel written into Newton and read back are unchanged, and the bodies move as MuJoCo says
    they move for that qpos/qvel (free-joint velocity convention, joint zero and sign)."""
    from metalsim.physics.newton_backend import NewtonSim
    m, _ = build_g1_model("flat", physics_dt=0.0025)
    n = 3; sim = NewtonSim(m, n); rng = np.random.default_rng(0)
    qpos = np.tile(m.key_qpos[0], (n, 1)); qvel = rng.normal(0, 0.5, (n, m.nv))
    for e in range(n):
        ax = rng.normal(size=3); ax /= np.linalg.norm(ax); ang = rng.uniform(-1, 1)
        qpos[e, 3:7] = [np.cos(ang / 2), *(np.sin(ang / 2) * ax)]; qpos[e, 7:] += rng.uniform(-0.3, 0.3, m.nu); qpos[e, :3] += rng.normal(0, 1, 3)
    sim.d.qpos.assign(qpos.astype(np.float32)); sim.d.qvel.assign(qvel.astype(np.float32))
    sim._reset_mask.fill_(True); sim.launch_reset()
    with wp.ScopedDevice("metal:0"):
        sim._sync_out()
    sim.synchronize()
    np.testing.assert_allclose(sim.d.qpos.numpy(), qpos, atol=1e-5)
    np.testing.assert_allclose(sim.d.qvel.numpy(), qvel, atol=1e-4)
    d = mujoco.MjData(m); bq = sim.s0.body_q.numpy().reshape(n, sim.nb, 7); bqd = sim.s0.body_qd.numpy().reshape(n, sim.nb, 6)
    labels = [l.split("/")[-1] for l in sim.model.body_label[: sim.nb]]
    for e in range(n):
        d.qpos[:] = qpos[e]; d.qvel[:] = qvel[e]; mujoco.mj_forward(m, d)
        for i, name in enumerate(labels):
            bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
            np.testing.assert_allclose(bq[e, i, :3], d.xpos[bid], atol=1e-5)
            v6 = np.zeros(6); mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, bid, v6, 0)   # at the COM, world
            # compare at the body origin: Newton merged fixed-jointed children into torso_link (moved COM)
            w = v6[:3]; v_mj = v6[3:] + np.cross(w, d.xpos[bid] - d.xipos[bid])
            x = bq[e, i]; R = _rot_xyzw(x[3:7]); com = sim.model.body_com.numpy()[i]
            v_nw = bqd[e, i, :3] + np.cross(bqd[e, i, 3:], -(R @ com))
            np.testing.assert_allclose(v_nw, v_mj, atol=1e-4)
            np.testing.assert_allclose(bqd[e, i, 3:], w, atol=1e-4)
        # feet_slide input: foot body origin, world linear velocity (MuJoCo's XBODY object velocity)
        fv = sim.d.foot_vel.numpy()[e]
        for k, f in enumerate((int(sim.foot_mj[0]), int(sim.foot_mj[1]))):
            v6 = np.zeros(6); mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_XBODY, f, v6, 0)
            np.testing.assert_allclose(fv[k], v6[3:], atol=1e-4)


def test_touch_signals_are_contact_forces_and_fall_is_terminated():
    """Standing on the default pose the feet carry the robot's weight and the torso nothing; the
    robot falls under the zero-action hold (as on MuJoCo), which ends the episode by torso contact
    with the -200 x dt = -4 termination penalty, and the reset restores Isaac's default state."""
    n = 4
    task = G1VelocityTask(n, terrain="flat", seed=3, **NEWTON); task.reset_all()
    step, _ = _stepper(task)
    zero = wp.zeros((n, task.act_dim), dtype=float, device="metal:0")
    weight = float(task.model.body_subtreemass[1]) * 9.81
    ta = task.touch_adr
    for _ in range(15):                       # 0.3 s: landed from Isaac's init height, still upright
        step(zero)
    sd = task.sim.d.sensordata.numpy()
    feet = sd[:, ta[0]] + sd[:, ta[1]]
    assert np.all(np.abs(feet - weight) < 0.25 * weight), (feet, weight)
    assert np.all(sd[:, ta[2]] == 0.0)
    ended = np.zeros(n, bool); term_r = []
    for _ in range(150):                      # 3 s: every world falls onto the torso
        r, d = step(zero)
        term_r += list(r[(d > 0) & ~ended]); ended |= d > 0
    assert ended.all()
    assert np.all(np.abs(np.array(term_r) + 4.0) < 0.2), term_r     # -200 x 0.02 plus the step's small shaping terms
    assert task.stats_i.numpy()[2] >= n and task.stats_i.numpy()[3] == 0   # fell, none blew up


def test_reset_restores_default_state():
    n = 4
    task = G1VelocityTask(n, terrain="flat", seed=4, **NEWTON); task.reset_all()
    step, _ = _stepper(task)
    a = wp.array(np.random.default_rng(0).normal(0, 1, (n, task.act_dim)).astype(np.float32), dtype=float, device="metal:0")
    for _ in range(10):
        step(a)
    task.sim._reset_mask.fill_(True)
    task.reset_all()
    with wp.ScopedDevice("metal:0"):
        task.sim._sync_out()                  # read the state back from Newton's bodies
    task.sim.synchronize()
    q = task.sim.d.qpos.numpy(); v = task.sim.d.qvel.numpy(); default = task.default_q.numpy()
    np.testing.assert_allclose(q[:, 7:], np.tile(default[7:], (n, 1)), atol=1e-5)
    np.testing.assert_allclose(q[:, 2], default[2], atol=1e-5)
    np.testing.assert_allclose(np.linalg.norm(q[:, 3:7], axis=1), 1.0, atol=1e-5)
    assert np.abs(q[:, 4:6]).max() < 1e-6                         # yaw-only initial orientation
    assert np.abs(v).max() < 1e-5


def test_graph_replay_matches_eager():
    from metalsim.learn.g1_velocity import benchmark_step
    out = []
    for capture in (True, False):
        task = G1VelocityTask(4, terrain="flat", seed=5, **NEWTON)
        benchmark_step(task, num_frames=0, warmup=0, capture=capture)   # reset only
        step, _ = _stepper(task)
        if capture:
            with wp.ScopedCapture(device="metal:0") as cap:
                task.launch_apply_action(task.action_scratch); task.sim.launch_step()
            for _ in range(20):
                wp.capture_launch(cap.graph)
        else:
            for _ in range(20):
                task.launch_apply_action(task.action_scratch); task.sim.launch_step()
        task.sim.synchronize(); out.append(task.sim.d.qpos.numpy())
    np.testing.assert_allclose(out[0], out[1], atol=1e-4)


def test_newton_heightfield_surface_matches_mujoco():
    """Isaac's rough terrain as a Newton heightfield: small spheres dropped on it rest on MuJoCo's hfield
    surface (mj_ray at their final xy). Runs on the CPU device."""
    from metalsim.learn.terrain import isaac_rough_terrain
    from metalsim.physics.newton_backend import heightfield_from_mujoco
    import newton
    hf = isaac_rough_terrain(num_rows=2, num_cols=4, seed=3)
    m, _ = build_g1_model("rough", hf); d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    only_ground = np.array([1, 0, 0, 0, 0, 0], np.uint8)
    surf = lambda x, y: 10.0 - mujoco.mj_ray(m, d, np.array([x, y, 10.0]), np.array([0, 0, -1.0]), only_ground, 1, -1, np.zeros(1, np.int32))
    rng = np.random.default_rng(0); sx, sy = hf["size"][0], hf["size"][1]
    pts = np.c_[rng.uniform(-0.9 * sx, 0.9 * sx, 40), rng.uniform(-0.9 * sy, 0.9 * sy, 40)]
    b = newton.ModelBuilder(); b.gravity = (0.0, 0.0, -9.81)
    h, X = heightfield_from_mujoco(hf); b.add_shape_heightfield(xform=X, heightfield=h)
    for x, y in pts:
        body = b.add_body(xform=wp.transform((float(x), float(y), surf(x, y) + 0.05), wp.quat_identity()))
        b.add_shape_sphere(body, radius=0.01)
    M = b.finalize("cpu"); S = newton.solvers.SolverXPBD(M, iterations=8)
    s0, s1, c = M.state(), M.state(), M.control(); pipe = newton.CollisionPipeline(M, broad_phase="explicit"); con = pipe.contacts()
    for _ in range(400):
        s0.clear_forces(); pipe.collide(s0, con); S.step(s0, s1, c, con, 0.0025); s0, s1 = s1, s0
    bq = s0.body_q.numpy()
    err = np.array([abs(bq[i, 2] - 0.01 - surf(bq[i, 0], bq[i, 1])) for i in range(len(pts))])
    assert np.median(err) < 1e-3 and np.percentile(err, 80) < 2e-3, err


def test_rough_task_runs_on_newton():
    """The rough task (heightfield, height scan, curriculum) steps on Newton; the torso pose the height
    scanner reads is MuJoCo's for the same state."""
    from metalsim.learn.g1_velocity import benchmark_step
    n = 8
    task = G1VelocityTask(n, terrain="rough", seed=0, **NEWTON)
    assert task.obs_dim == 12 + 3 * 37 + 187
    benchmark_step(task, num_frames=10, warmup=2)
    task.sim.synchronize()
    obs = task.obs.numpy(); q = task.sim.d.qpos.numpy()
    assert np.isfinite(obs).all() and np.isfinite(q).all()
    scan = obs[:, -187:]
    assert scan.std() > 0.0 and np.abs(scan).max() <= 1.0
    # consistent state (FK of the reported qpos/qvel), then the scanner's torso pose vs MuJoCo C
    sim = task.sim; sim._reset_mask.fill_(True); sim.launch_reset(); sim.synchronize()
    m = task.model; d = mujoco.MjData(m); tb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    xp = sim.d.xpos.numpy(); xm = sim.d.xmat.numpy()
    for e in range(n):
        d.qpos[:] = q[e]; mujoco.mj_kinematics(m, d)
        np.testing.assert_allclose(xp[e, tb], d.xpos[tb], atol=1e-4)
        np.testing.assert_allclose(xm[e, tb].reshape(9), d.xmat[tb], atol=1e-4)

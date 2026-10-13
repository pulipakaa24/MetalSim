"""Provable capacity bounds (metalsim.physics.capacity) and the overflow guard.

The bound must hold on the G1 flat task under random actions (contact-rich, limits driven), the auto-sized
BatchSim must run without a capacity overflow, and a capacity that is too small must raise from check_overflow
(never silent)."""
import numpy as np
import pytest
import torch
import warp as wp

wp.config.quiet = True

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs the Metal device")


def _g1(n, contact_cfg=None, **opts):
    from metalsim.learn import g1_velocity as g1v
    from metalsim.physics.batch import BatchSimOptions
    orig = g1v.BatchSimOptions
    g1v.BatchSimOptions = lambda **kw: orig(**{**kw, **opts})
    task_kw = {} if contact_cfg is None else {"contact_cfg": contact_cfg}
    try:
        task = g1v.G1VelocityTask(n, terrain="flat", physics_dt=0.0025, seed=0, **task_kw)
    finally:
        g1v.BatchSimOptions = orig
    task.reset_all()
    return task


def _random_steps(task, steps, seed=1):
    from metalsim.learn.warp_policy import RolloutBuffers, bump
    dev = task.device; n = task.n
    rng = np.random.default_rng(seed)

    class _Pol:
        step_idx = wp.zeros(1, dtype=int, device=dev)
    pol = _Pol()
    bufs = RolloutBuffers(1, n, task.obs_dim, task.act_dim)
    bufs.rew = wp.zeros((1, n), dtype=float, device=dev); bufs.done = wp.zeros((1, n), dtype=float, device=dev)
    nefc_max, nacon_max = 0, 0
    for k in range(steps):
        a = wp.array(rng.uniform(-1, 1, (n, task.act_dim)).astype(np.float32), dtype=float, device=dev)
        with wp.ScopedDevice(dev):
            wp.launch(bump, dim=1, inputs=[pol.step_idx], device=dev)
            task.launch_apply_action(a); task.sim.launch_step(); task.launch_reward_done_reset(pol, bufs); task.launch_obs(pol.step_idx)
        if k % 5 == 4:
            task.sim.synchronize()
            nefc_max = max(nefc_max, int(task.sim.d.nefc.numpy().max())); nacon_max = max(nacon_max, int(task.sim.d.nacon.numpy()[0]))
    task.sim.synchronize()
    return nefc_max, nacon_max


@pytest.mark.parametrize("contact_cfg,rows", [("recommended", 3), ("recommended_pyramidal", 4)])
def test_g1_flat_bounds_hold_under_random_actions(contact_cfg, rows):
    """The G1's colliders are condim 3: 3 rows per contact under the default elliptic cone (contact_cfg "recommended"
    since 2026-09-26 11:00), 4 under the pyramidal preset (the default when this test was written, 04:23 that day)."""
    import mujoco
    task = _g1(64, njmax="auto", nconmax="auto", contact_cfg=contact_cfg)
    b = task.sim.capacity_bounds
    assert b is not None
    elliptic = task.sim.mj_model.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC
    assert elliptic == (rows == 3)
    # 3 colliders (two feet, torso) against the plane at <= 4 each, 3 mesh-mesh pairs at <= 4 (MULTICCD), 37 limited hinges
    assert (b.ncon, b.nlimit, b.neq_rows, b.nfriction) == (24, 37, 0, 0)
    assert {r for *_, r in b.pairs} == {rows}
    njmax = {3: 112, 4: 144}[rows]                       # 37 + 24 * 3 = 109 -> 112; 37 + 24 * 4 = 133 -> 144
    assert b.nefc == 37 + 24 * rows and task.sim.d.njmax == b.njmax == njmax and task.sim.d.naconmax == 24 * 64
    nefc_max, nacon_max = _random_steps(task, 60)
    assert nefc_max <= b.nefc and nacon_max <= b.ncon * 64
    assert not task.sim.overflow_flags().keys() & {"NEFC", "NARROWPHASE", "CCD", "BROADPHASE"}
    task.sim.check_overflow()      # no raise


def test_g1_flat_task_defaults_to_the_bound():
    task = _g1(16)          # no overrides: the task's own capacities on flat terrain (elliptic cone: 37 + 24 * 3 rows)
    assert task.sim.capacity_bounds is not None and task.sim.d.njmax == task.sim.capacity_bounds.njmax == 112
    assert task.sim.d.naconmax == 24 * 16


def test_check_overflow_raises_on_dropped_rows():
    """A genuinely too small njmax: the build pose stands on 8 foot contacts, 24 elliptic rows (put_data refuses less),
    and random actions drive joints into their limits, adding rows past 24 within ~10 control steps (up to 27 seen:
    scripts/diagnostics/capacity_overflow_probe.py). The written test (njmax 32 = 8 contacts x 4 pyramidal rows)
    never overflowed once the G1 default became elliptic: 3 rows per contact, at most 27 rows."""
    task = _g1(64, njmax=24, nconmax=24)
    _random_steps(task, 20)
    assert task.sim.d.njmax == 24
    with pytest.raises(RuntimeError, match="capacity overflow"):
        task.sim.check_overflow()


def test_overflow_survives_resets():
    """mjw.reset_data zeroes d.overflow of every world it resets, and the task resets fallen worlds every control step
    (often the worlds whose dropped rows made them fall). The guard must still raise after the reset."""
    task = _g1(64, njmax=24, nconmax=24)
    _random_steps(task, 20)
    task.sim.synchronize()
    assert (task.sim.d.overflow.numpy() & 1).any()          # NEFC set in some worlds before the reset
    task.reset_all()                       # mjw.reset_data on every world
    task.sim.synchronize()
    assert not task.sim.d.overflow.numpy().any()           # MuJoCo Warp's own flags are gone
    with pytest.raises(RuntimeError, match="capacity overflow"):
        task.sim.check_overflow()


@pytest.mark.parametrize("eq,rows", [("connect", 3), ("weld", 6)])
def test_equality_block_ending_at_njmax_is_flagged(eq, rows):
    """Up to MuJoCo Warp fork cf0f8ad, a connect (weld) block was written only if it started below njmax - 3 (- 6): with
    njmax = 3 (6) its rows were dropped while nefc == njmax (no NEFC bit), and the solver ran on stale rows (the ball
    sagged 8 cm in 0.4 s; MetalSim then counted nefc >= njmax itself). The fork's metalsim branch since f52488b keeps a
    block that ends exactly at njmax, and one that does not fit leaves nefc > njmax, which MuJoCo Warp flags itself.
    njmax = rows: block written, ball held, no flag; njmax 16: the same. njmax = rows - 1 (put_data refuses an njmax
    below MuJoCo C's initial nefc, so there the equality starts inactive and is switched on in every world after
    put_data): block dropped, MuJoCo Warp's NEFC bit set, check_overflow raises, the ball falls."""
    import mujoco
    import numpy as np
    import mujoco_warp as mjw
    from metalsim.physics.batch import BatchSim, BatchSimOptions
    nefc_bit = int(mjw.OverflowType.NEFC)
    z = {}
    for njmax in (rows, 16, rows - 1):
        active = "false" if njmax < rows else "true"
        xml = f"""<mujoco><worldbody><body name='a' pos='0 0 1'><freejoint/>
        <geom type='sphere' size='0.1' contype='0' conaffinity='0'/></body></worldbody>
        <equality><{eq} body1='a' active='{active}' {"anchor='0 0 0'" if eq == "connect" else ""}/></equality></mujoco>"""
        m = mujoco.MjModel.from_xml_string(xml)
        sim = BatchSim(m, 4, options=BatchSimOptions(njmax=njmax, nconmax=1))
        if njmax < rows:
            sim.d.eq_active.assign(np.ones((4, 1), dtype=bool))
        for _ in range(200):           # 0.4 s at the default 2 ms
            sim.step()
        sim.synchronize()
        z[njmax] = float(sim.d.qpos.numpy()[:, 2].max())
        nefc = sim.d.nefc.numpy()
        assert (nefc == rows).all(), nefc
        if njmax < rows:
            assert (sim.d.overflow.numpy() & nefc_bit).all()          # MuJoCo Warp's own flag
            with pytest.raises(RuntimeError, match="capacity overflow"):
                sim.check_overflow()
        else:
            assert not (sim.d.overflow.numpy() & nefc_bit).any()
            sim.check_overflow()
    assert z[rows] > 0.995 and z[16] > 0.995 and z[rows - 1] < 0.97, z   # held at the anchor (1.0) vs block dropped


def test_bounds_refuse_unbounded_models():
    import mujoco, mujoco_warp as mjw
    from metalsim.physics import capacity
    xml = """<mujoco><asset><hfield name='h' nrow='4' ncol='4' size='1 1 0.1 0.01'/></asset>
    <worldbody><geom type='hfield' hfield='h'/><body pos='0 0 1'><freejoint/><geom type='sphere' size='0.1'/></body></worldbody></mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml)
    with wp.ScopedDevice("cpu"):
        wm = mjw.put_model(m)
    with pytest.raises(ValueError):
        capacity.bounds(m, wm)

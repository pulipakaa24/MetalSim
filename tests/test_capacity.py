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


def _g1(n, **opts):
    from metalsim.learn import g1_velocity as g1v
    from metalsim.physics.batch import BatchSimOptions
    orig = g1v.BatchSimOptions
    g1v.BatchSimOptions = lambda **kw: orig(**{**kw, **opts})
    try:
        task = g1v.G1VelocityTask(n, terrain="flat", physics_dt=0.0025, seed=0)
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


def test_g1_flat_bounds_hold_under_random_actions():
    task = _g1(64, njmax="auto", nconmax="auto")
    b = task.sim.capacity_bounds
    assert b is not None
    # 3 colliders (two feet, torso) against the plane at <= 4 each, 3 mesh-mesh pairs at <= 4 (MULTICCD), 37 limited hinges
    assert (b.ncon, b.nlimit, b.neq_rows, b.nfriction) == (24, 37, 0, 0)
    assert b.nefc == 37 + 24 * 4 and task.sim.d.njmax == b.njmax == 144 and task.sim.d.naconmax == 24 * 64
    nefc_max, nacon_max = _random_steps(task, 60)
    assert nefc_max <= b.nefc and nacon_max <= b.ncon * 64
    assert not task.sim.overflow_flags().keys() & {"NEFC", "NARROWPHASE", "CCD", "BROADPHASE"}
    task.sim.check_overflow()      # no raise


def test_g1_flat_task_defaults_to_the_bound():
    task = _g1(16)          # no overrides: the task's own capacities on flat terrain
    assert task.sim.capacity_bounds is not None and task.sim.d.njmax == 144 and task.sim.d.naconmax == 24 * 16


def test_check_overflow_raises_on_dropped_rows():
    task = _g1(64, njmax=32, nconmax=24)   # the build pose needs 32 rows (8 foot contacts); random actions drive limits and contacts past it
    _random_steps(task, 100)
    with pytest.raises(RuntimeError, match="capacity overflow"):
        task.sim.check_overflow()


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

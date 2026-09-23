"""WS2 acceptance: MuJoCo Warp on Metal against CPU MuJoCo, and no host synchronization per step."""
import os
import time

import mujoco
import numpy as np
import pytest
import torch
import warp as wp

from metalsim.physics.batch import BatchSim, BatchSimOptions

pytestmark = pytest.mark.skipif(not wp.is_metal_available(), reason="needs Metal")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MENAGERIE = os.path.join(ROOT, "upstream", "mujoco_menagerie")
SO101 = os.path.join(ROOT, "assets", "so101", "scene_box_rl.xml")

# One-step parity thresholds follow MuJoCo Warp's own forward tests (5e-4 smooth, 1e-3 step,
# 5e-3 implicit integrators), applied against float64 MuJoCo C.
ROBOTS = [
    ("so101_lift", SO101, 5e-3),
    ("panda", os.path.join(MENAGERIE, "franka_emika_panda", "scene.xml"), 5e-3),
    ("go1", os.path.join(MENAGERIE, "unitree_go1", "scene.xml"), 5e-3),
    ("g1", os.path.join(MENAGERIE, "unitree_g1", "scene.xml"), 5e-3),
]


def _cpu_states(model, n_states, seed, warm_steps=50):
    """Random reachable states from a CPU rollout with random controls."""
    rng = np.random.default_rng(seed)
    d = mujoco.MjData(model)
    states, ctrls = [], []
    for k in range(n_states):
        for _ in range(warm_steps):
            if model.nu:
                lo, hi = model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
                lim = model.actuator_ctrllimited.astype(bool)
                c = rng.uniform(-1, 1, model.nu)
                d.ctrl[:] = np.where(lim, lo + 0.5 * (c + 1) * (hi - lo), c)
            mujoco.mj_step(model, d)
        states.append((d.qpos.copy(), d.qvel.copy(), d.act.copy()))
        ctrls.append(d.ctrl.copy())
    return states, ctrls


@pytest.mark.parametrize("name,path,tol", ROBOTS, ids=[r[0] for r in ROBOTS])
def test_one_step_parity_vs_mujoco_c(name, path, tol):
    if not os.path.exists(path):
        pytest.skip(f"missing {path}")
    model = mujoco.MjModel.from_xml_path(path)
    n = 8
    states, ctrls = _cpu_states(model, n, seed=1)
    sim = BatchSim(model, n, options=BatchSimOptions(capture=False))
    # load the n states into the n worlds, step once on Metal
    sim.synchronize()
    qpos = sim.d.qpos.numpy(); qvel = sim.d.qvel.numpy(); ctrl = sim.d.ctrl.numpy()
    act = sim.d.act.numpy() if model.na else None
    for w, (q, v, a) in enumerate(states):
        qpos[w], qvel[w], ctrl[w] = q, v, ctrls[w]
        if model.na:
            act[w] = a
    sim.d.qpos.assign(qpos); sim.d.qvel.assign(qvel); sim.d.ctrl.assign(ctrl)
    if model.na:
        sim.d.act.assign(act)
    sim.forward()
    sim.step()
    sim.synchronize()
    q_gpu = sim.d.qpos.numpy(); v_gpu = sim.d.qvel.numpy()
    # reference: MuJoCo C in float64 from the same states
    worst_q = worst_v = 0.0
    for w, (q, v, a) in enumerate(states):
        d = mujoco.MjData(model)
        d.qpos[:] = q; d.qvel[:] = v; d.ctrl[:] = ctrls[w]
        if model.na:
            d.act[:] = a
        mujoco.mj_forward(model, d)
        mujoco.mj_step(model, d)
        worst_q = max(worst_q, np.abs(q_gpu[w] - d.qpos).max())
        worst_v = max(worst_v, np.abs(v_gpu[w] - d.qvel).max())
        np.testing.assert_allclose(q_gpu[w], d.qpos, atol=tol, rtol=tol, err_msg=f"{name} world {w} qpos")
        # velocities: MuJoCo Warp's solver tests allow 0.1 absolute/relative; on contact-heavy humanoid
        # states the float32 solver (Warp CPU and Metal alike) deviates by up to ~10% of the velocity
        # scale from float64 MuJoCo C, so the bound scales with the state's velocity magnitude.
        vscale = max(1.0, float(np.abs(d.qvel).max()))
        np.testing.assert_allclose(v_gpu[w], d.qvel, atol=tol * 20 * vscale, rtol=tol * 20,
                                   err_msg=f"{name} world {w} qvel")
    print(f"{name}: nq={model.nq} nv={model.nv} one-step max |dqpos|={worst_q:.2e} max |dqvel|={worst_v:.2e}")


def test_trajectory_parity_so101_arm_joints():
    """200-step rollout with identical controls: arm joints (no contact) must track CPU MuJoCo."""
    model = mujoco.MjModel.from_xml_path(SO101)
    rng = np.random.default_rng(0)
    steps = 200
    lo, hi = model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
    ctrl_seq = lo + rng.uniform(0.2, 0.8, (steps, model.nu)) * (hi - lo)
    ctrl_seq[:, 5] = lo[5]  # keep the gripper open: avoid box contact chaos in this check
    d = mujoco.MjData(model)
    sim = BatchSim(model, 4, options=BatchSimOptions(substeps=1))
    sim.synchronize()
    ref = np.zeros((steps, model.nq)); got = np.zeros((steps, model.nq))
    for k in range(steps):
        d.ctrl[:] = ctrl_seq[k]
        mujoco.mj_step(model, d)
        ref[k] = d.qpos
        sim.t.ctrl[:] = torch.as_tensor(ctrl_seq[k], dtype=torch.float32)
        torch.mps.synchronize()
        sim.step()
        sim.synchronize()
        got[k] = sim.d.qpos.numpy()[0]
    arm = slice(0, 6)
    err = np.abs(got[:, arm] - ref[:, arm])
    print(f"SO-101 arm joints over {steps} steps: max |dq| = {err.max():.2e}, final = {err[-1].max():.2e}")
    assert err.max() < 2e-2
    assert "NEFC" not in sim.overflow_flags()
    # all four worlds identical (determinism across worlds)
    q_all = sim.d.qpos.numpy()
    assert np.allclose(q_all[0], q_all[1:], atol=1e-6)


def test_step_does_not_block_host():
    """The rollout loop never waits for the GPU: runtime counters show zero host syncs and host ops,
    and host time per step does not scale with the GPU work."""
    from metalsim.interop import warp_metal as wm
    model = mujoco.MjModel.from_xml_path(SO101)
    times = {}
    for n in (64, 2048):
        sim = BatchSim(model, n, options=BatchSimOptions(substeps=4))
        sim.step(); sim.synchronize()  # warm
        # Metal queues hold at most 64 in-flight command buffers, after which creating one blocks:
        # measure the host cost inside that window (2 command buffers per step here).
        k = 12
        c0 = wm.counters()
        t0 = time.perf_counter()
        for _ in range(k):
            sim.step()
        host = (time.perf_counter() - t0) / k
        c1 = wm.counters()
        sim.synchronize()
        total = (time.perf_counter() - t0) / k
        times[n] = (host, total)
        d = c1 - c0
        print(f"N={n}: host {host * 1e3:.3f} ms/step, wall incl. GPU {total * 1e3:.3f} ms/step "
              f"-> {n * 4 / total:,.0f} physics steps/s; per step: {d.dispatches // k} dispatches, "
              f"{d.flushes / k:.1f} command buffers, {d.host_ops / k:.1f} host ops, {d.syncs / k:.2f} syncs")
        assert d.syncs == 0, "host waited for the GPU inside the loop"
        assert d.host_ops == 0, "graph replay ran host ops (each one waits for the GPU)"
    host_small, _ = times[64]
    host_big, total_big = times[2048]
    assert host_big < 0.25 * total_big, "host time is a large fraction of GPU time: the loop is blocking"
    assert host_big < 2 * host_small + 5e-4, "host time grew with GPU work"


def test_reset_and_forward_ordering():
    """reset(mask) + torch-written qpos + forward(), all GPU-ordered, then a host check."""
    model = mujoco.MjModel.from_xml_path(SO101)
    n = 16
    sim = BatchSim(model, n)
    sim.synchronize()
    v = sim.step()
    mask = torch.zeros(n, dtype=torch.bool); mask[::2] = True
    v = sim.reset(mask)
    sim.after(v)
    q = sim.t.qpos
    q[::2, 0] = 0.5                      # randomize one joint on the reset worlds (torch, ordered after reset)
    import metalsim.interop.torch_bridge as tb
    from metalsim.interop import warp_metal as wm
    ev = wm.SharedEvent("metal:0", "test-reset")
    tb.signal_event(ev, 1)
    sim.wait(ev, 1)
    sim.forward()
    sim.synchronize()
    qn = sim.d.qpos.numpy()
    xpos = sim.d.xpos.numpy()
    assert np.allclose(qn[::2, 0], 0.5) and not np.allclose(qn[1::2, 0], 0.5)
    assert np.allclose(qn[::2, 1:6], model.qpos0[1:6], atol=1e-6)     # reset to qpos0
    assert not np.allclose(xpos[0], xpos[1], atol=1e-4)                # forward ran: kinematics differ


def test_per_world_model_fields():
    """Physics DR: per-world body_mass and geom_friction as writable tensors that change dynamics."""
    from metalsim.physics.batch import BatchSim, BatchSimOptions
    model = mujoco.MjModel.from_xml_string("""
    <mujoco><option timestep="0.002"/><worldbody>
      <geom type="plane" size="2 2 0.1"/>
      <body name="b" pos="0 0 0.5"><freejoint/><geom name="g" type="box" size="0.05 0.05 0.05" mass="1"/></body>
    </worldbody><actuator/></mujoco>""")
    n = 8
    sim = BatchSim(model, n, options=BatchSimOptions(per_world_fields=("body_mass", "geom_friction"), njmax=64))
    sim.synchronize()
    assert tuple(sim.tm.body_mass.shape) == (n, model.nbody)
    b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "b")
    # push every body sideways with an external force via a torch write to xfrc_applied; heavier worlds move less
    masses = torch.linspace(0.5, 4.0, n, device="mps")
    sim.tm.body_mass[:, b] = masses
    torch.mps.synchronize()
    sim.recompute_constants()
    sim.synchronize()
    f = sim.t.xfrc_applied
    f[:, b, 0] = 2.0   # 2 N along +x
    torch.mps.synchronize()
    for _ in range(50):
        sim.step()
    sim.synchronize()
    v = sim.d.qvel.numpy()[:, 0]
    # a = F/m before friction dominates: velocity should decrease monotonically with mass
    assert np.all(np.diff(v) < 0) and v[0] > 3 * v[-1], v

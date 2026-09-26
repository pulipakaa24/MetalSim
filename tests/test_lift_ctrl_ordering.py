"""SO101LiftEnv.step orders torch's write of ctrl (on the MPS command queue) before the physics launch on Warp's
queue, and the physics of a step before the next torch write: the learner event is signalled from torch after the
ctrl write and waited on by the sim before its graph, and the sim's completion event is waited on by torch
(``sim.after``) after each step. A missing pair would let a stale or half-written ctrl reach the physics (the race
scripts/diagnostics/competitors/elliptic_check.py had before it was fixed on 2026-09-25).
"""
import numpy as np
import pytest
import torch
import warp as wp

import metalsim.learn.so101_lift as sl
from metalsim.learn.so101_lift import LiftConfig, SO101LiftEnv

pytestmark = pytest.mark.skipif(not wp.is_metal_available(), reason="needs Metal")


def _env(n=8, seed=0):
    return SO101LiftEnv(LiftConfig(num_envs=n, render=False, max_episode_steps=1000, physics_dr=False, seed=seed))


def test_step_signals_and_waits_in_order(monkeypatch):
    env = _env(); env.reset(); env.synchronize()
    log = []
    orig_signal, orig_wait, orig_step, orig_after = sl.tb.signal_event, env.sim.wait, env.sim.step, env.sim.after

    def signal(ev, v):
        log.append(("signal_learner" if ev is env.learner_event else "signal_other", v)); return orig_signal(ev, v)

    def wait(ev, v):
        log.append(("wait_learner" if ev is env.learner_event else "wait_other", v)); return orig_wait(ev, v)

    def step():
        log.append(("physics", None)); return orig_step()

    def after(v):
        log.append(("after", v)); return orig_after(v)

    monkeypatch.setattr(sl.tb, "signal_event", signal)
    env.sim.wait, env.sim.step, env.sim.after = wait, step, after
    a = torch.zeros(env.n, env.act_dim, device="mps")
    for _ in range(3):
        env.step(a)
    env.synchronize()
    # step() commits torch's writes twice: before the physics (the ctrl write) and, after resetting the envs that
    # finished (sim.reset: its own signal / wait on the sim event), before the forward pass (the randomized state);
    # only the learner-event, physics and after entries are ordered here
    kinds = [k for k, _ in log if k in ("signal_learner", "wait_learner", "physics")]
    assert kinds == ["signal_learner", "wait_learner", "physics", "signal_learner", "wait_learner"] * 3, [k for k, _ in log]
    seq = [(k, v) for k, v in log if k in ("signal_learner", "wait_learner", "physics", "after")]
    phys = [n for n, (k, _) in enumerate(seq) if k == "physics"]
    assert len(phys) == 3
    for n in phys:
        # the learner event is waited on at the value just signalled, right before the physics, and torch waits on the
        # physics' completion (after) right after it, i.e. before its next write
        assert seq[n - 2][0] == "signal_learner" and seq[n - 1][0] == "wait_learner" and seq[n - 2][1] == seq[n - 1][1]
        assert seq[n + 1][0] == "after"
    # torch's ctrl write is visible to the physics: the sim's ctrl equals what step() computed from the action
    env.synchronize()
    expect = (env.ctrl_lo + 0.5 * (a + 1.0) * (env.ctrl_hi - env.ctrl_lo)).cpu().numpy()
    np.testing.assert_allclose(env.sim.d.ctrl.numpy(), expect, atol=1e-6)


def test_step_matches_host_synchronized_stepping():
    """The event-ordered loop (no host sync) gives the same trajectory as syncing the host after every step."""
    ea, eb = _env(seed=3), _env(seed=3)
    ea.reset(); eb.reset(); ea.synchronize(); eb.synchronize()
    g = torch.Generator(device="mps").manual_seed(11)
    acts = [torch.rand(ea.n, ea.act_dim, device="mps", generator=g) * 2 - 1 for _ in range(30)]
    for a in acts:
        ea.step(a)
    ea.synchronize()
    for a in acts:
        eb.step(a); eb.synchronize()
    qa, qb = ea.sim.d.qpos.numpy(), eb.sim.d.qpos.numpy()
    np.testing.assert_allclose(qa, qb, atol=1e-4)     # a stale ctrl would move the arm to a different target (>> 1e-4 rad)

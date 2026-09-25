"""PPOWarp rollout semantics against rsl_rl's (found by the learner differential, runs/g1_flat_rslrl_ppo.log):
fresh exploration noise every rollout, no never-executed action leaking into the next rollout, stored
log-probs consistent with torch, time-outs flagged for bootstrapping, and the rsl_rl VecEnv adapter's
contract (observation after reset, time_outs vs falls)."""
import pytest
import torch
import warp as wp

pytestmark = pytest.mark.skipif(not wp.is_metal_available(), reason="needs Metal")


@pytest.fixture(scope="module")
def algo():
    from metalsim.learn.g1_velocity import G1VelocityTask, g1_ppo_config
    from metalsim.learn.ppo_warp import PPOWarp
    wp.config.quiet = True
    task = G1VelocityTask(64, terrain="flat", seed=0, physics_dt=0.0025)
    return PPOWarp(task, g1_ppo_config("flat", 1, 0))


def _rollout(algo):
    algo.rollout(); torch.mps.synchronize(); algo.task.sim.synchronize()
    b = algo.bufs
    with torch.no_grad():
        z = (b.t_act - algo.net.actor_mean(b.t_obs)) / algo.net.log_std.exp()
    return z.clone(), b.t_obs.clone(), b.t_act.clone()


def test_noise_fresh_each_rollout_and_no_phantom_action(algo):
    nj = algo.task.nj; sl = slice(12 + 2 * nj, 12 + 3 * nj)
    z0, o0, a0 = _rollout(algo)
    z1, o1, a1 = _rollout(algo)
    # before the fix the rollout row index keyed the sampler: corr(z0, z1) was 1.0
    corr = torch.corrcoef(torch.stack([z0.flatten(), z1.flatten()]))[0, 1].item()
    assert abs(corr) < 0.05, corr
    assert 0.95 < z0.std().item() < 1.05
    # the first observation of a rollout carries the action applied at the previous rollout's last step
    # (before the fix: the bootstrap-value sample, which was never executed)
    same = (o1[0, :, sl] - a0[23]).abs().max(-1).values < 1e-5
    assert bool(same.all())


def test_stored_logp_matches_torch(algo):
    algo.rollout(); torch.mps.synchronize(); algo.task.sim.synchronize()
    b = algo.bufs
    with torch.no_grad():
        lp = torch.distributions.Normal(algo.net.actor_mean(b.t_obs), algo.net.log_std.exp()).log_prob(b.t_act).sum(-1)
    assert (lp - b.t_logp).abs().max().item() < 1e-3


def test_timeouts_flagged(algo):
    task = algo.task
    old = task.max_t; task.max_t = 10
    try:
        algo.graph = None                      # max_t is a kernel argument baked into the captured graph
        algo.rollout(); torch.mps.synchronize(); task.sim.synchronize()
        d, to = algo.bufs.t_done, algo.bufs.t_timeout
        assert to.sum().item() > 0
        assert not bool(((to > 0) & (d < 0.5)).any())
    finally:
        task.max_t = old; algo.graph = None


def test_rslrl_adapter_contract():
    pytest.importorskip("rsl_rl"); pytest.importorskip("tensordict")
    from metalsim.learn.g1_velocity import G1VelocityTask
    from metalsim.learn.rslrl_adapter import G1RslRlVecEnv
    wp.config.quiet = True
    task = G1VelocityTask(32, terrain="flat", seed=0, physics_dt=0.0025)
    env = G1RslRlVecEnv(task)
    env.episode_length_buf = torch.full((32,), task.max_t - 3, dtype=torch.int32, device="mps")
    saw_timeout = False
    for _ in range(5):
        obs, rew, done, extras = env.step(torch.zeros(32, task.act_dim, device="mps"))
        to = extras["time_outs"] > 0
        assert bool((~to | (done > 0)).all())       # a time-out is a done
        if to.any():
            saw_timeout = True
            # the observation returned for a finished env is the post-reset one: zero last action
            nj = task.nj
            assert obs["policy"][to][:, 12 + 2 * nj:12 + 3 * nj].abs().max().item() == 0.0
    assert saw_timeout

"""WS7: rollout policy in Warp — parity with the torch module, in-place weight sharing with the
optimizer, and a rollout loop with no torch launches per step."""
import time

import numpy as np
import pytest
import torch
import warp as wp

from metalsim.interop import warp_metal as wm
from metalsim.learn.warp_policy import ActorCriticMLP, WarpMLPPolicy

pytestmark = pytest.mark.skipif(not wp.is_metal_available(), reason="needs Metal")


@wp.kernel
def gather_cartpole_obs(qpos: wp.array2d(dtype=float), qvel: wp.array2d(dtype=float), obs: wp.array2d(dtype=float)):
    e = wp.tid()
    obs[e, 0] = qpos[e, 0]; obs[e, 1] = qpos[e, 1]; obs[e, 2] = qvel[e, 0]; obs[e, 3] = qvel[e, 1]


def test_parity_with_torch_and_shared_weights():
    torch.manual_seed(0)
    n, obs_dim, act_dim = 256, 4, 1
    net = ActorCriticMLP(obs_dim, act_dim, hidden=(64, 32)).to("mps")
    pol = WarpMLPPolicy(net, n, obs_dim, act_dim, ctrl_lo=[-1.0], ctrl_hi=[1.0])
    obs_np = np.random.default_rng(0).standard_normal((n, obs_dim)).astype(np.float32)
    obs = wp.array(obs_np, dtype=float, device="metal:0")
    ctrl = wp.zeros((n, act_dim), dtype=float, device="metal:0")
    wp.synchronize_device("metal:0")
    pol.act(obs, ctrl)
    wp.synchronize_device("metal:0")
    with torch.no_grad():
        mean_t = net.actor(torch.as_tensor(obs_np, device="mps")).cpu().numpy()
        val_t = net.critic(torch.as_tensor(obs_np, device="mps")).cpu().numpy()
    np.testing.assert_allclose(pol.actor_act[-1].numpy(), mean_t, atol=1e-5, rtol=1e-4)
    np.testing.assert_allclose(pol.critic_act[-1].numpy(), val_t, atol=1e-5, rtol=1e-4)
    # sampled actions and log-probs are consistent with the torch distribution
    a = pol.action.numpy(); lp = pol.logp.numpy()
    d = torch.distributions.Normal(torch.as_tensor(mean_t), net.log_std.exp().detach().cpu())
    np.testing.assert_allclose(lp, d.log_prob(torch.as_tensor(a)).sum(-1).numpy(), atol=1e-4)
    assert abs(a.mean()) < 0.3 and 0.6 < a.std() < 1.5      # unit-std noise around small means
    np.testing.assert_allclose(ctrl.numpy(), np.clip(a, -1, 1), atol=1e-6)
    # optimizer updates the shared weights in place; the Warp policy sees them
    opt = torch.optim.SGD(net.parameters(), lr=0.1)
    loss = net.actor(torch.as_tensor(obs_np, device="mps")).pow(2).mean()
    opt.zero_grad(); loss.backward(); opt.step(); torch.mps.synchronize()
    pol.act(obs, ctrl); wp.synchronize_device("metal:0")
    with torch.no_grad():
        mean_t2 = net.actor(torch.as_tensor(obs_np, device="mps")).cpu().numpy()
    assert not np.allclose(mean_t2, mean_t)
    np.testing.assert_allclose(pol.actor_act[-1].numpy(), mean_t2, atol=1e-5, rtol=1e-4)


def test_rollout_without_torch_launches():
    """Cartpole (state obs): one captured graph per step (obs gather, Warp policy, buffer store,
    physics), replayed T times; torch touches nothing until the rollout boundary."""
    from metalsim.learn.cartpole_rgb import CARTPOLE_XML
    from metalsim.learn.warp_policy import RolloutBuffers
    from metalsim.physics.batch import BatchSim, BatchSimOptions
    import mujoco
    torch.manual_seed(0)
    n, T = 1024, 64
    model = mujoco.MjModel.from_xml_string(CARTPOLE_XML)
    # MuJoCo's default 100 Newton iterations would all run (no GPU-side early exit on Metal); 10 suffice here
    sim = BatchSim(model, n, options=BatchSimOptions(substeps=2, njmax=32, solver_iterations=10, ls_iterations=10))
    net = ActorCriticMLP(4, 1).to("mps")
    pol = WarpMLPPolicy(net, n, 4, 1, ctrl_lo=[-1.0], ctrl_hi=[1.0])
    bufs = RolloutBuffers(T, n, 4, 1)
    obs = wp.zeros((n, 4), dtype=float, device="metal:0")
    sim.synchronize(); torch.mps.synchronize()
    with wp.ScopedCapture(device="metal:0") as cap:
        wp.launch(gather_cartpole_obs, dim=n, inputs=[sim.d.qpos, sim.d.qvel, obs], device="metal:0")
        pol.act(obs, sim.d.ctrl)
        pol.store(obs, bufs)
        sim.launch_step()
    step_graph = cap.graph
    pol.rewind()
    c0 = wm.counters()
    t0 = time.perf_counter()
    for t in range(T):
        wp.capture_launch(step_graph)
    host = time.perf_counter() - t0
    d = wm.counters() - c0          # before the deliberate end-of-rollout synchronize
    sim.synchronize()
    total = time.perf_counter() - t0
    assert d.syncs == 0 and d.host_ops == 0
    t_obs, t_act, t_logp = bufs.t_obs, bufs.t_act, bufs.t_logp
    assert torch.isfinite(t_obs).all() and torch.isfinite(t_act).all() and torch.isfinite(t_logp).all()
    assert (t_obs[1:] != t_obs[:-1]).any() and (t_act[1] != t_act[0]).any()      # states evolve, noise differs per step
    assert pol.step_idx.numpy()[0] == T
    print(f"rollout of {T} steps x {n} envs as {T} graph replays: {n * T / total:,.0f} env-steps/s, host "
          f"{host / T * 1e3:.3f} ms/step, {d.dispatches // T} dispatches/step, 0 torch launches per step")


def test_layer_mapping_bitwise():
    """mlp_layer4 (four outputs per thread, envs fastest; the default mapping since 2026-09-26) gives bitwise the
    same activations, actions and log-probs as the original one-thread-per-(env, output) kernel: each output keeps
    the same sequential dot product, only the thread-to-work mapping differs."""
    from metalsim.learn import warp_policy as WPL
    n = 512
    for obs_dim, hidden, act_dim in ((123, (256, 128, 128), 37), (310, (512, 256, 128), 37), (4, (32, 32), 1)):
        torch.manual_seed(1)
        net = ActorCriticMLP(obs_dim, act_dim, hidden=hidden).to("mps")
        pol = WarpMLPPolicy(net, n, obs_dim, act_dim, [-1.0] * act_dim, [1.0] * act_dim)
        obs = wp.array(np.random.default_rng(0).standard_normal((n, obs_dim)).astype(np.float32), dtype=float, device="metal:0")
        ctrl = wp.zeros((n, act_dim), dtype=float, device="metal:0")
        outs = {}
        for mapping in ("A", "D"):
            WPL._MLP_MAPPING = mapping
            pol.act(obs, ctrl); wp.synchronize_device("metal:0")
            outs[mapping] = [a.numpy().copy() for a in pol.actor_act + pol.critic_act + [pol.action, ctrl]] + [pol.logp.numpy().copy()]
            # the same RNG stream: rewind the step counter so the samples match
            pol.rng_step.zero_() if hasattr(pol.rng_step, "zero_") else None
        WPL._MLP_MAPPING = "D"
        for a, b in zip(outs["A"], outs["D"]):
            assert np.array_equal(a, b)

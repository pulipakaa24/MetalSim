"""Franka-Pour PPO checkpoint round trip: a resumed learner equals the paused one (networks, Adam moments, counters,
curriculum sampler, RNG streams, logger buffers), and the STOP file pauses training after the current iteration.

    scripts/gpu_run.sh test_pour_resume check 10 -- .venv-newton152/bin/python -m pytest tests/test_franka_pour_resume.py
"""
from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")
wp = pytest.importorskip("warp")
newton = pytest.importorskip("newton")

pytestmark = [pytest.mark.skipif(not wp.is_metal_available(), reason="needs Metal"),
              # the task runs on Newton 1.5.2 (.venv-newton152); .venv's Newton 1.7 SolverMuJoCo does not build
              # against the MuJoCo Warp fork, so in .venv this test is skipped like test_franka_pour_terms
              pytest.mark.skipif(not newton.__version__.startswith("1.5.2"), reason="needs Newton 1.5.2 (.venv-newton152)")]

from metalsim.learn import franka_pour as P  # noqa: E402
from metalsim.learn import franka_pour_ppo as PPO  # noqa: E402

pytestmark = [*pytestmark, pytest.mark.skipif(not os.path.isfile(P.S.RESET_DATASET), reason="needs the Franka-Pour assets")]


def test_checkpoint_round_trip_and_stop_file(tmp_path):
    wp.config.quiet = True
    env = P.FrankaPourEnv(P.PourTaskConfig(num_envs=8, seed=0))
    cfg = PPO.PourPPOConfig(num_steps_per_env=8, save_interval=1000)
    a = PPO.PourPPO(env, cfg)
    open(tmp_path / "STOP", "w").close()
    a.train(5, out_dir=str(tmp_path), log=lambda s: None)
    assert a.it == 1 and not (tmp_path / "STOP").exists()          # paused after the first iteration
    d = torch.load(tmp_path / "model_last.pt", map_location="cpu", weights_only=False)
    b = PPO.PourPPO(env, cfg)
    b.load_state_dict(d)
    for x, y in zip(list(a.actor.parameters()) + list(a.critic.parameters()),
                    list(b.actor.parameters()) + list(b.critic.parameters())):
        assert torch.equal(x.cpu(), y.cpu())
    sa, sb = a.optimizer.state_dict()["state"], b.optimizer.state_dict()["state"]
    assert sa.keys() == sb.keys()
    for k in sa:
        for key in ("exp_avg", "exp_avg_sq", "step"):
            assert torch.equal(torch.as_tensor(sa[k][key]).cpu(), torch.as_tensor(sb[k][key]).cpu())
    assert (b.it, b.total_steps) == (a.it, a.total_steps)
    s = env.curriculum._sampler
    for k in PPO.PourPPO._SAMPLER_TENSORS:
        assert torch.equal(getattr(s, k).cpu(), d["sampler"][k])
    assert list(b.rewbuffer) == list(a.rewbuffer)
    assert torch.equal(torch.get_rng_state(), d["rng"]["torch_cpu"])

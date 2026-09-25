"""The camera-PPO fast update path (PPOConfig.fast_update) computes the same loss and gradients as the plain path.

Fast path: per-image means over a flattened view, fused uint8->centered-float pass, loss under torch.compile,
optionally the custom Metal conv-gradient kernels. Checked in fp32 on random data for the Isaac camera-cartpole
config (image-only, centered), an image+qpos uncentered config, and a 64x64 image (kernels fall back to aten).

Tolerance. Errors are max |diff| / max |reference| per tensor. The loss matches PyTorch's MPS path to 1e-4. For the
gradients the exact reference is the plain path in float64 on the CPU: measured (M4 Max, torch 2.14), PyTorch's own
fp32 MPS path is off from it by up to 4e-4 on conv1's weight/bias gradient (a sum of ~1e5-1e6 products of zero-mean
centered pixels, i.e. cancellation) and 2e-4 on conv2's weight gradient, so "within 1e-4 of PyTorch's MPS gradient"
is below PyTorch's own fp32 noise for those tensors. The test therefore asserts, per tensor, that the fast path is
within 1e-4 of the exact gradient or no less accurate than PyTorch's MPS path (x1.5), and separately that it is within
1e-4 of PyTorch's MPS gradient wherever PyTorch's MPS gradient is itself within 1e-4 of exact.
"""
import copy
import types

import pytest
import torch

from metalsim.learn.ppo import PPO, PPOConfig

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")


class _Env:
    def __init__(self, n, hw, qdim):
        self.n, self.act_dim = n, 1
        self.obs_space = {"image": (hw, hw, 3), "qpos": (qdim,)}


def _plain_loss(ppo, img, q, a, logp_old, adv, ret, v_old):
    cfg = ppo.cfg
    mean, v = ppo.net(img, q, nchw=True)
    d = ppo.net.dist(mean)
    logp = d.log_prob(a).sum(-1)
    ratio = (logp - logp_old).exp()
    pg = -torch.min(ratio * adv, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * adv).mean()
    v_c = v_old + (v - v_old).clamp(-cfg.clip, cfg.clip)
    vf = torch.max((v - ret) ** 2, (v_c - ret) ** 2).mean()
    return pg + cfg.vf_coef * vf


def _features64(self, image_u8, qpos, nchw=False):
    x = image_u8.double() / 255.0
    if self.center_images:
        x = x - x.mean(dim=(2, 3), keepdim=True)
    f = self.cnn(x)
    if self.qpos_dim > 0:
        f = torch.cat([f, self.qpos(qpos)], dim=1)
    return f


def _rel(a, b):
    a, b = a.detach().cpu().double(), b.detach().cpu().double()
    return (a - b).abs().max().item() / max(b.abs().max().item(), 1e-30)


@pytest.mark.parametrize("kernels", [False, True])
@pytest.mark.parametrize("qdim,center,hw", [(0, True, 100), (4, False, 100), (4, True, 64)])
def test_fast_update_matches_plain(qdim, center, hw, kernels):
    cfg = PPOConfig(rollout=2, epochs=1, minibatches=1, feat=512, activation="elu", qpos_dim=qdim,
                    center_images=center, clip_value=True, vf_coef=1.0, fast_update=True,
                    metal_conv_kernels=kernels, seed=3)
    ppo = PPO(_Env(256, hw, max(qdim, 1)), cfg)
    g = torch.Generator().manual_seed(0)
    n = 256
    img = torch.randint(0, 256, (n, 3, hw, hw), dtype=torch.uint8, generator=g)
    q = torch.randn(n, max(qdim, 1), generator=g)
    a = torch.randn(n, 1, generator=g)
    lo, adv, ret, vo = (torch.randn(n, generator=g) for _ in range(4))
    cpu_args = (img, q.double(), a.double(), lo.double(), adv.double(), ret.double(), vo.double())
    img, q, a, lo, adv, ret, vo = (t.to("mps") for t in (img, q, a, lo, adv, ret, vo))

    names = [k for k, _ in ppo.net.named_parameters()]
    params = list(ppo.net.parameters())
    ref64 = copy.copy(ppo); ref64.net = copy.deepcopy(ppo.net).cpu().double()
    ref64.net.features = types.MethodType(_features64, ref64.net)
    l64 = _plain_loss(ref64, *cpu_args)
    g64 = torch.autograd.grad(l64, list(ref64.net.parameters()), allow_unused=True)

    l_mps = _plain_loss(ppo, img, q, a, lo, adv, ret, vo)
    g_mps = torch.autograd.grad(l_mps, params, allow_unused=True)
    loss, *_ = ppo._mb_loss_c(img, q, ppo.net.image_mean(img), a, lo, adv, ret, vo)
    g_fast = torch.autograd.grad(loss, params, allow_unused=True)

    assert abs(loss.item() - l_mps.item()) <= 1e-4 * max(1.0, abs(l_mps.item())), (loss.item(), l_mps.item())
    report = []
    for name, ge, gm, gf in zip(names, g64, g_mps, g_fast):
        if ge is None:
            continue
        e_fast, e_torch = _rel(gf, ge), _rel(gm, ge)
        report.append(f"{name}: fast {e_fast:.0e} torch {e_torch:.0e}")
        assert e_fast <= max(1e-4, 1.5 * e_torch), f"{name}: fast {e_fast:.1e} vs exact; torch MPS {e_torch:.1e}"
        if e_torch <= 1e-4:
            assert _rel(gf, gm) <= 1e-4, f"{name}: fast vs torch MPS {_rel(gf, gm):.1e}"
    print("; ".join(report))

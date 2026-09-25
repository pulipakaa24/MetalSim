"""Profile the camera-cartpole PPO update (Isaac Lab skrl config) in isolation on MPS, and the alternatives.

Same nets (metalsim.learn.ppo.ActorCritic, feat 512 ELU, image-only 100x100, centered images), same
batch shapes (64 x 1024 = 65,536 samples, 4 epochs x 32 minibatches of 2048), same optimizer (Adam
eps 1e-5, grad clip 1.0, clipped value loss, KL-adaptive lr). Parts (select with --parts):

  update     the real PPO.update() on a stub env with random buffers (the 5.9 s number)
  split      per-minibatch split: gather+normalize, forward, forward (no_grad), fwd+bwd, clip+Adam step
  layers     per-layer conv / fc forward, input-grad and weight-grad (aten.convolution_backward with output_mask)
  fp16       fwd+bwd+step under torch.autocast(fp16)
  compile    torch.compile(inductor) of the loss fwd+bwd on MPS (if it works)
  mlx        the same NatureCNN + heads + PPO loss + clip + Adam in MLX (NHWC), eager and mx.compile
  mlxlayers  per-layer MLX conv forward / vjp

    .venv/bin/python scripts/diagnostics/camera_update_profile.py --parts update,split,layers,fp16,compile,mlx
"""
from __future__ import annotations

import argparse
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from metalsim.learn.ppo import PPO, PPOConfig

N_ENVS, T, H, W = 1024, 64, 100, 100
EPOCHS, MINIBATCHES = 4, 32
ROLLOUT_S = 1.4          # measured rollout time per iteration (docs/GAPS.md), used for implied env-steps/s


def sps(update_s):
    return N_ENVS * T / (ROLLOUT_S + update_s)


class StubEnv:
    n = N_ENVS
    act_dim = 1
    obs_space = {"image": (H, W, 3), "qpos": (4,)}


def make_ppo(seed=0):
    cfg = PPOConfig(total_steps=0, rollout=T, epochs=EPOCHS, minibatches=MINIBATCHES, lr=1e-4, gamma=0.99, lam=0.95,
                    clip=0.2, ent_coef=0.0, vf_coef=1.0, max_grad_norm=1.0, desired_kl=0.008, clip_value=True, feat=512,
                    activation="elu", qpos_dim=0, center_images=True, value_norm=True, seed=seed)
    ppo = PPO(StubEnv(), cfg)
    g = torch.Generator(device="cpu").manual_seed(seed)
    ppo.buf_img.copy_(torch.randint(0, 256, ppo.buf_img.shape, dtype=torch.uint8, generator=g))
    ppo.buf_a.copy_(torch.randn(ppo.buf_a.shape, generator=g))
    ppo.buf_logp.copy_(torch.randn(ppo.buf_logp.shape, generator=g) - 1.0)
    ppo.buf_v.copy_(torch.randn(ppo.buf_v.shape, generator=g))
    ppo.buf_r.copy_(torch.rand(ppo.buf_r.shape, generator=g))
    ppo.buf_done.copy_((torch.rand(ppo.buf_done.shape, generator=g) < 0.01).float())
    torch.mps.synchronize()
    return ppo


def timeit(fn, reps=10, warmup=3, sync=torch.mps.synchronize):
    for _ in range(warmup):
        fn()
    sync()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); sync(); ts.append(time.perf_counter() - t0)
    return float(np.median(ts)), float(np.min(ts))


def row(name, med, mn, per_update=None):
    extra = ""
    if per_update is not None:
        extra = f"  -> x128 = {per_update:6.2f} s/update  (implied {sps(per_update):7,.0f} env-steps/s)"
    print(f"{name:52s} median {med * 1e3:8.2f} ms  min {mn * 1e3:8.2f} ms{extra}", flush=True)


# ------------------------------------------------------------------------------------------------ torch
def part_update(args):
    ppo = make_ppo()
    adv, ret = ppo.gae(torch.zeros(N_ENVS, device="mps"))

    def once():
        ppo.update(adv, ret)
    med, mn = timeit(once, reps=args.update_reps, warmup=1)
    print(f"{'PPO.update (4 epochs x 32 mb of 2048), fp32':52s} median {med:8.3f} s  min {mn:8.3f} s  "
          f"(implied {sps(med):,.0f} env-steps/s)", flush=True)


def minibatch_tensors(ppo, mb=2048):
    N = T * N_ENVS
    img = ppo.buf_img.reshape(N, 3, H, W)
    idx = torch.randperm(N, device="mps")[:mb]
    return img, idx


def loss_fn(ppo, img_u8, a, logp_old, adv, ret, v_old):
    net, cfg = ppo.net, ppo.cfg
    mean, v = net(img_u8, None, nchw=True)
    d = net.dist(mean)
    logp = d.log_prob(a).sum(-1)
    ratio = (logp - logp_old).exp()
    pg = -torch.min(ratio * adv, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * adv).mean()
    v_c = v_old + (v - v_old).clamp(-cfg.clip, cfg.clip)
    vf = torch.max((v - ret) ** 2, (v_c - ret) ** 2).mean()
    return pg + cfg.vf_coef * vf


def mb_data(ppo, mb=2048, seed=1):
    g = torch.Generator(device="cpu").manual_seed(seed)
    img, idx = minibatch_tensors(ppo, mb)
    x = img[idx].contiguous()
    a = torch.randn(mb, 1, generator=g).to("mps"); logp_old = (torch.randn(mb, generator=g) - 1).to("mps")
    adv = torch.randn(mb, generator=g).to("mps"); ret = torch.randn(mb, generator=g).to("mps")
    v_old = torch.randn(mb, generator=g).to("mps")
    return img, idx, x, a, logp_old, adv, ret, v_old


def part_split(args):
    ppo = make_ppo()
    img, idx, x, a, lo, adv, ret, vo = mb_data(ppo)
    net = ppo.net
    K = EPOCHS * MINIBATCHES

    def gather():
        y = img[idx].float() * (1 / 255.0); y - y.mean(dim=(2, 3), keepdim=True)
    row("gather uint8 + float + center (2048)", *timeit(gather, 20), K * timeit(gather, 20)[0])

    def fwd_ng():
        with torch.no_grad():
            net(x, None, nchw=True)
    m = timeit(fwd_ng, 20); row("forward, no_grad", *m, K * m[0])

    def fwd():
        loss_fn(ppo, x, a, lo, adv, ret, vo)
    m = timeit(fwd, 20); row("forward + loss (grad-enabled)", *m, K * m[0])

    def fwdbwd():
        ppo.opt.zero_grad(set_to_none=True)
        loss_fn(ppo, x, a, lo, adv, ret, vo).backward()
    m_fb = timeit(fwdbwd, 20); row("forward + loss + backward", *m_fb, K * m_fb[0])

    params = list(net.parameters())

    def step():
        torch.nn.utils.clip_grad_norm_(params, 1.0); ppo.opt.step()
    fwdbwd()
    m = timeit(step, 20); row("clip_grad_norm + Adam.step", *m, K * m[0])

    def kl_item():
        (lo - lo).mean().item()
    m = timeit(kl_item, 20); row("KL .item() host sync (per minibatch)", *m, K * m[0])

    def full():
        ppo.opt.zero_grad(set_to_none=True)
        ii = idx
        loss = loss_fn(ppo, img[ii], a, lo, adv, ret, vo)
        loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); ppo.opt.step()
    m = timeit(full, 20); row("full minibatch step (gather+fwd+bwd+step)", *m, K * m[0])


def part_layers(args):
    torch.manual_seed(0)
    mb = 2048
    specs = [("conv1 3->32 k8 s4  (100->24)", (mb, 3, 100, 100), (32, 3, 8, 8), 4, False),
             ("conv2 32->64 k4 s2 (24->11)", (mb, 32, 24, 24), (64, 32, 4, 4), 2, True),
             ("conv3 64->64 k3 s1 (11->9)", (mb, 64, 11, 11), (64, 64, 3, 3), 1, True)]
    tot = 0.0
    for dtype in (torch.float32, torch.float16):
        print(f"-- per-layer, {dtype}", flush=True)
        for name, xs, ws, s, need_in in specs:
            x = torch.randn(xs, device="mps", dtype=dtype)
            w = torch.randn(ws, device="mps", dtype=dtype) * 0.05
            b = torch.zeros(ws[0], device="mps", dtype=dtype)
            y = F.conv2d(x, w, b, stride=s)
            gy = torch.randn_like(y)
            flops = 2 * y.numel() * ws[1] * ws[2] * ws[3]
            mf = timeit(lambda: F.conv2d(x, w, b, stride=s), 20)
            row(f"{name} fwd  [{flops / mf[0] / 1e12:5.2f} TF/s]", *mf)

            def bw(mask):
                return torch.ops.aten.convolution_backward(gy, x, w, [ws[0]], [s, s], [0, 0], [1, 1], False, [0, 0], 1, mask)
            mi = timeit(lambda: bw([True, False, False]), 20)
            row(f"{name} dgrad (input)  [{flops / mi[0] / 1e12:5.2f} TF/s]", *mi)
            mw = timeit(lambda: bw([False, True, True]), 20)
            row(f"{name} wgrad+bgrad    [{flops / mw[0] / 1e12:5.2f} TF/s]", *mw)
            if dtype == torch.float32:
                tot += mf[0] + mw[0] + (mi[0] if need_in else 0)
        x = torch.randn(mb, 5184, device="mps", dtype=dtype); w = torch.randn(512, 5184, device="mps", dtype=dtype) * 0.01
        gy = torch.randn(mb, 512, device="mps", dtype=dtype)
        flops = 2 * mb * 5184 * 512
        mf = timeit(lambda: x @ w.t(), 20); row(f"fc 5184->512 fwd  [{flops / mf[0] / 1e12:5.2f} TF/s]", *mf)
        mi = timeit(lambda: gy @ w, 20); row(f"fc dgrad  [{flops / mi[0] / 1e12:5.2f} TF/s]", *mi)
        mw = timeit(lambda: gy.t() @ x, 20); row(f"fc wgrad  [{flops / mw[0] / 1e12:5.2f} TF/s]", *mw)
        if dtype == torch.float32:
            tot += mf[0] + mi[0] + mw[0]
            print(f"sum of fp32 conv/fc fwd+needed grads per minibatch: {tot * 1e3:.2f} ms -> x128 = {tot * 128:.2f} s", flush=True)


def part_fp16(args):
    ppo = make_ppo()
    img, idx, x, a, lo, adv, ret, vo = mb_data(ppo)
    params = list(ppo.net.parameters())

    def full():
        ppo.opt.zero_grad(set_to_none=True)
        with torch.autocast("mps", dtype=torch.float16):
            loss = loss_fn(ppo, x, a, lo, adv, ret, vo)
        loss.float().backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); ppo.opt.step()
    m = timeit(full, 20); row("fp16 autocast fwd+bwd+step", *m, 128 * m[0])


def part_bf16(args):
    ppo = make_ppo()
    img, idx, x, a, lo, adv, ret, vo = mb_data(ppo)
    params = list(ppo.net.parameters())

    def full():
        ppo.opt.zero_grad(set_to_none=True)
        with torch.autocast("mps", dtype=torch.bfloat16):
            loss = loss_fn(ppo, x, a, lo, adv, ret, vo)
        loss.float().backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); ppo.opt.step()
    m = timeit(full, 20); row("bf16 autocast fwd+bwd+step", *m, 128 * m[0])


def part_fusedadam(args):
    ppo = make_ppo()
    img, idx, x, a, lo, adv, ret, vo = mb_data(ppo)
    params = list(ppo.net.parameters())
    for fused in (False, True):
        opt = torch.optim.Adam(params, lr=1e-4, eps=1e-5, fused=fused)
        ppo.opt.zero_grad(set_to_none=True); loss_fn(ppo, x, a, lo, adv, ret, vo).backward()

        def step():
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        m = timeit(step, 20); row(f"clip + Adam(fused={fused}) step", *m, 128 * m[0])


def s2d(x, r):
    """space-to-depth: (N,C,H,W) -> (N,C*r*r,H/r,W/r), channel order (c, ry, rx)."""
    N, C, Hh, Ww = x.shape
    return x.view(N, C, Hh // r, r, Ww // r, r).permute(0, 1, 3, 5, 2, 4).reshape(N, C * r * r, Hh // r, Ww // r)


def conv_s2d(x, w, b, s):
    """stride-s conv with kernel k (k % s == 0) as a stride-1 conv with kernel k/s on the space-to-depth input."""
    N, C, Hh, Ww = x.shape
    O, _, k, _ = w.shape
    Hc = (Hh // s) * s
    ws = w.view(O, C, k // s, s, k // s, s).permute(0, 1, 3, 5, 2, 4).reshape(O, C * s * s, k // s, k // s)
    return F.conv2d(s2d(x[:, :, :Hc, :Hc].contiguous() if Hc != Hh else x, s), ws, b)


def part_alt(args):
    """Encoder fwd+bwd variants with identical math: MPSGraph conv (baseline), space-to-depth stride-1 convs, unfold+matmul."""
    torch.manual_seed(0)
    mb = 2048
    x = torch.randn(mb, 3, 100, 100, device="mps")
    ws = [torch.randn(32, 3, 8, 8, device="mps") * 0.05, torch.randn(64, 32, 4, 4, device="mps") * 0.05,
          torch.randn(64, 64, 3, 3, device="mps") * 0.05]
    bs = [torch.zeros(o, device="mps") for o in (32, 64, 64)]
    for t in ws + bs:
        t.requires_grad_(True)
    fcw = (torch.randn(512, 5184, device="mps") * 0.01).requires_grad_(True)

    def base(x):
        h = F.relu(F.conv2d(x, ws[0], bs[0], stride=4)); h = F.relu(F.conv2d(h, ws[1], bs[1], stride=2))
        return F.relu(F.conv2d(h, ws[2], bs[2]))

    def v_s2d(x):
        h = F.relu(conv_s2d(x, ws[0], bs[0], 4)); h = F.relu(conv_s2d(h, ws[1], bs[1], 2))
        return F.relu(F.conv2d(h, ws[2], bs[2]))

    def unf(x, w, b, s):
        N = x.shape[0]; O, C, k, _ = w.shape
        Ho = (x.shape[2] - k) // s + 1
        cols = F.unfold(x, k, stride=s)                         # (N, C*k*k, L)
        y = torch.matmul(w.view(O, -1), cols) + b.view(1, O, 1)  # (N, O, L)
        return y.view(N, O, Ho, Ho)

    def v_unf(x):
        h = F.relu(unf(x, ws[0], bs[0], 4)); h = F.relu(unf(h, ws[1], bs[1], 2))
        return F.relu(unf(h, ws[2], bs[2], 1))

    ref = base(x)
    for name, f in (("MPSGraph conv (baseline)", base), ("space-to-depth stride-1 convs", v_s2d), ("unfold + matmul", v_unf)):
        err = (f(x) - ref).abs().max().item()

        def fb():
            out = F.elu(f(x).flatten(1) @ fcw.t())
            out.sum().backward()
        m = timeit(fb, 20); row(f"encoder fwd+bwd: {name} (maxerr {err:.1e})", *m, 128 * m[0])
    for name, f in (("baseline", base), ("s2d", v_s2d)):
        def fb16():
            with torch.autocast("mps", dtype=torch.float16):
                out = F.elu(f(x).flatten(1) @ fcw.t())
            out.float().sum().backward()
        m = timeit(fb16, 20); row(f"encoder fwd+bwd fp16 autocast: {name}", *m, 128 * m[0])


def part_prep(args):
    """Image preprocessing cost (uint8 NCHW gather -> float/255 -> per-image channel mean-centering)."""
    ppo = make_ppo()
    img, idx, x, *_ = mb_data(ppo)
    m = timeit(lambda: img[idx], 20); row("gather img[idx] uint8", *m, 128 * m[0])
    m = timeit(lambda: x.float() * (1 / 255.0), 20); row("float * 1/255", *m, 128 * m[0])
    xf = x.float() * (1 / 255.0)
    m = timeit(lambda: xf.mean(dim=(2, 3), keepdim=True), 20); row("mean over (2,3)", *m, 128 * m[0])
    m = timeit(lambda: xf.view(-1, H * W).mean(1), 20); row("mean over flattened HW", *m, 128 * m[0])
    mm = xf.mean(dim=(2, 3), keepdim=True)
    m = timeit(lambda: xf - mm, 20); row("subtract mean", *m, 128 * m[0])


def part_updvar(args):
    """The real PPO.update with the ActorCritic forward wrapped in torch.compile, and fused Adam."""
    import os
    print(f"TORCHINDUCTOR_LAYOUT_OPTIMIZATION={os.environ.get('TORCHINDUCTOR_LAYOUT_OPTIMIZATION', '<unset>')}", flush=True)
    for comp, fused in ((True, False), (True, True)):
        ppo = make_ppo()
        if fused:
            ppo.opt = torch.optim.Adam(ppo.net.parameters(), lr=ppo.cfg.lr, eps=1e-5, fused=True)
        if comp:
            ppo.net.forward = torch.compile(ppo.net.forward)
        adv, ret = ppo.gae(torch.zeros(N_ENVS, device="mps"))
        t0 = time.perf_counter(); ppo.update(adv, ret); torch.mps.synchronize()
        print(f"first update incl. compile: {time.perf_counter() - t0:.1f} s", flush=True)
        med, mn = timeit(lambda: ppo.update(adv, ret), reps=3, warmup=1)
        print(f"{'PPO.update compiled-net=' + str(comp) + ' fusedAdam=' + str(fused):52s} median {med:8.3f} s  "
              f"min {mn:8.3f} s  (implied {sps(med):,.0f} env-steps/s)", flush=True)


def part_compile(args):
    ppo = make_ppo()
    img, idx, x, a, lo, adv, ret, vo = mb_data(ppo)
    params = list(ppo.net.parameters())
    try:
        cl = torch.compile(lambda *z: loss_fn(ppo, *z))

        def full():
            ppo.opt.zero_grad(set_to_none=True)
            cl(x, a, lo, adv, ret, vo).backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); ppo.opt.step()
        t0 = time.perf_counter(); full(); torch.mps.synchronize()
        print(f"torch.compile first call (compile time): {time.perf_counter() - t0:.1f} s", flush=True)
        m = timeit(full, 20); row("torch.compile(inductor) fwd+bwd + eager step", *m, 128 * m[0])
    except Exception as e:  # noqa: BLE001
        print(f"torch.compile failed: {type(e).__name__}: {str(e)[:400]}", flush=True)


# ------------------------------------------------------------------------------------------------ MLX
def mlx_setup(dtype_name="float32"):
    import mlx.core as mx
    import mlx.nn as mnn
    import mlx.optimizers as mopt
    dt = getattr(mx, dtype_name)

    class MNature(mnn.Module):
        def __init__(self):
            super().__init__()
            self.c1 = mnn.Conv2d(3, 32, 8, stride=4); self.c2 = mnn.Conv2d(32, 64, 4, stride=2)
            self.c3 = mnn.Conv2d(64, 64, 3, stride=1); self.fc = mnn.Linear(64 * 9 * 9, 512)
            self.pi = mnn.Linear(512, 1); self.v = mnn.Linear(512, 1); self.log_std = mx.zeros((1,))

        def __call__(self, x):                  # x (N,H,W,3) float
            x = mnn.relu(self.c1(x)); x = mnn.relu(self.c2(x)); x = mnn.relu(self.c3(x))
            x = x.transpose(0, 3, 1, 2).reshape(x.shape[0], -1)       # NCHW flatten order, as torch
            h = mnn.elu(self.fc(x))
            return self.pi(h), self.v(h).squeeze(-1)

    net = MNature()
    if dt != mx.float32:
        net.set_dtype(dt)
    opt = mopt.Adam(learning_rate=1e-4, eps=1e-5)

    def loss(model, img_u8, a, logp_old, adv, ret, v_old):
        x = img_u8.astype(dt) * (1 / 255.0)
        x = x - x.mean(axis=(1, 2), keepdims=True)
        mean, v = model(x)
        mean = mean.astype(mx.float32); v = v.astype(mx.float32)
        std = mx.exp(mx.clip(model.log_std, -20.0, 2.0))
        logp = (-((a - mean) ** 2) / (2 * std ** 2) - mx.log(std) - 0.5 * math.log(2 * math.pi)).sum(-1)
        ratio = mx.exp(logp - logp_old)
        pg = -mx.minimum(ratio * adv, mx.clip(ratio, 0.8, 1.2) * adv).mean()
        v_c = v_old + mx.clip(v - v_old, -0.2, 0.2)
        vf = mx.maximum((v - ret) ** 2, (v_c - ret) ** 2).mean()
        return pg + vf

    lg = mnn.value_and_grad(net, loss)

    def step(img_u8, a, lo, adv, ret, vo):
        l, g = lg(net, img_u8, a, lo, adv, ret, vo)
        g, _ = mopt.clip_grad_norm(g, 1.0)
        opt.update(net, g)
        return l
    return mx, net, opt, step


def part_mlx(args):
    import mlx.core as mx
    mb = 2048
    rng = np.random.default_rng(0)
    # full rollout buffer in NHWC uint8, as MLX's conv wants
    buf = mx.array(rng.integers(0, 256, (T * N_ENVS, H, W, 3), dtype=np.uint8))
    a = mx.array(rng.standard_normal((mb, 1), dtype=np.float32)); lo = mx.array(rng.standard_normal(mb, dtype=np.float32) - 1)
    adv = mx.array(rng.standard_normal(mb, dtype=np.float32)); ret = mx.array(rng.standard_normal(mb, dtype=np.float32))
    vo = mx.array(rng.standard_normal(mb, dtype=np.float32))
    mx.eval(buf)
    sync = mx.synchronize if hasattr(mx, "synchronize") else (lambda: None)
    for dname in ("float32", "float16"):
        mx_, net, opt, step = mlx_setup(dname)
        idx = mx.array(rng.permutation(T * N_ENVS)[:mb].astype(np.int32))
        x = buf[idx]; mx.eval(x)

        def eager():
            l = step(x, a, lo, adv, ret, vo); mx.eval(l, net.parameters(), opt.state)
        m = timeit(eager, 20, sync=sync); row(f"MLX {dname} eager fwd+bwd+clip+Adam", *m, 128 * m[0])

        state = [net.state, opt.state]
        cstep = mx.compile(step, inputs=state, outputs=state)

        def comp():
            l = cstep(x, a, lo, adv, ret, vo); mx.eval(l, net.parameters(), opt.state)
        try:
            m = timeit(comp, 20, sync=sync); row(f"MLX {dname} mx.compile fwd+bwd+clip+Adam", *m, 128 * m[0])
        except Exception as e:  # noqa: BLE001
            print(f"mx.compile failed: {type(e).__name__}: {str(e)[:300]}", flush=True)

        def full_update():   # 4 epochs x 32 minibatches with gather from the NHWC uint8 buffer, lazy graph per minibatch
            for _ in range(EPOCHS):
                perm = mx.random.permutation(T * N_ENVS)
                for i in range(MINIBATCHES):
                    ii = perm[i * mb:(i + 1) * mb]
                    l = cstep(buf[ii], a, lo, adv, ret, vo)
                    mx.eval(l, net.parameters(), opt.state)
        med, mn = timeit(full_update, 3, warmup=1, sync=sync)
        print(f"{'MLX ' + dname + ' full update (128 mb, compiled step)':52s} median {med:8.3f} s  min {mn:8.3f} s  "
              f"(implied {sps(med):,.0f} env-steps/s)", flush=True)


def part_mlxlayers(args):
    import mlx.core as mx
    mb = 2048
    sync = mx.synchronize if hasattr(mx, "synchronize") else (lambda: None)
    specs = [("conv1", (mb, 100, 100, 3), (32, 8, 8, 3), 4), ("conv2", (mb, 24, 24, 32), (64, 4, 4, 32), 2),
             ("conv3", (mb, 11, 11, 64), (64, 3, 3, 64), 1)]
    for dname in ("float32", "float16"):
        dt = getattr(mx, dname)
        print(f"-- MLX per-layer, {dname}", flush=True)
        for name, xs, ws, s in specs:
            x = mx.random.normal(xs).astype(dt); w = (mx.random.normal(ws) * 0.05).astype(dt)
            y = mx.conv2d(x, w, stride=s); gy = mx.random.normal(y.shape).astype(dt); mx.eval(x, w, y, gy)
            flops = 2 * y.size * ws[1] * ws[2] * ws[3]
            m = timeit(lambda: mx.eval(mx.conv2d(x, w, stride=s)), 20, sync=sync)
            row(f"MLX {name} fwd [{flops / m[0] / 1e12:5.2f} TF/s]", *m)
            _, vjpx = mx.vjp(lambda xx: mx.conv2d(xx, w, stride=s), [x], [gy])
            m = timeit(lambda: mx.eval(mx.vjp(lambda xx: mx.conv2d(xx, w, stride=s), [x], [gy])[1]), 20, sync=sync)
            row(f"MLX {name} dgrad [{flops / m[0] / 1e12:5.2f} TF/s]", *m)
            m = timeit(lambda: mx.eval(mx.vjp(lambda ww: mx.conv2d(x, ww, stride=s), [w], [gy])[1]), 20, sync=sync)
            row(f"MLX {name} wgrad [{flops / m[0] / 1e12:5.2f} TF/s]", *m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="update,split,layers,alt,fp16,bf16,fusedadam,mlx,mlxlayers,compile")
    ap.add_argument("--update-reps", type=int, default=3)
    args = ap.parse_args()
    print(f"torch {torch.__version__}", flush=True)
    import subprocess
    for p in args.parts.split(","):
        st = subprocess.run(["python3", "scripts/gpu_lock.py", "status"], capture_output=True, text=True).stdout
        print("gpu_lock status: " + " | ".join(st.strip().splitlines()[:1]), flush=True)
        print(f"==== {p}", flush=True)
        try:
            globals()[f"part_{p}"](args)
        except Exception as e:  # noqa: BLE001  (keep measuring the other parts)
            import traceback; traceback.print_exc()
        torch.mps.empty_cache()


if __name__ == "__main__":
    main()

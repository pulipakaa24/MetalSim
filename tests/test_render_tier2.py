"""Tier-2 path tracer acceptance: analytic radiometry (Lambertian plane under a directional light,
white furnace under a uniform sky), Monte Carlo convergence, agreement of the direct-light term
with tier 0, GPU-path ordering against BatchSim, and a full Cartpole-RGB rollout at tier 2."""
import os

import mujoco
import numpy as np
import pytest
import torch
import warp as wp

from metalsim.render.tier0 import Tier0Renderer
from metalsim.render.tier2 import Tier2Renderer

pytestmark = pytest.mark.skipif(not wp.is_metal_available(), reason="needs Metal")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "measurements")

# camera straight down onto a matte grey plane; one directional light straight down; no headlight
LAMBERT = """
<mujoco>
  <visual><headlight active="0" ambient="0 0 0"/></visual>
  <asset><material name="matte" rgba="0.8 0.8 0.8 1" specular="0" shininess="0"/></asset>
  <worldbody>
    <light directional="true" pos="0 0 3" dir="0 0 -1" diffuse="0.5 0.5 0.5" specular="0 0 0" ambient="0 0 0"/>
    <camera name="cam" pos="0 0 2" xyaxes="1 0 0 0 1 0" fovy="30"/>
    <geom name="floor" type="plane" size="50 50 0.1" material="matte"/>
  </worldbody>
</mujoco>
"""

FURNACE = """
<mujoco>
  <visual><headlight active="0" ambient="0 0 0"/></visual>
  <asset>
    <texture type="skybox" builtin="flat" rgb1="0.5 0.5 0.5" rgb2="0.5 0.5 0.5" width="8" height="48"/>
    <material name="white" rgba="1 1 1 1" specular="0" shininess="0"/>
  </asset>
  <worldbody>
    <camera name="cam" pos="0 0 2" xyaxes="1 0 0 0 1 0" fovy="30"/>
    <geom name="floor" type="plane" size="50 50 0.1" material="white"/>
  </worldbody>
</mujoco>
"""

PRIMS = """
<mujoco>
  <visual><headlight ambient="0 0 0"/></visual>
  <worldbody>
    <light directional="true" pos="0 0 3" dir="0.3 0.3 -0.9" ambient="0 0 0"/>
    <camera name="cam" pos="0.6 -0.6 0.5" quat="0.85 0.4 0.15 0.3"/>
    <geom name="floor" type="plane" size="2 2 0.1" rgba="0.3 0.3 0.35 1"/>
    <body pos="0 0 0.3"><joint type="free"/><geom name="b1" type="box" size="0.06 0.04 0.05" rgba="0.9 0.2 0.2 1"/></body>
    <body pos="0.15 0.1 0.2"><joint type="free"/><geom name="s1" type="sphere" size="0.05" rgba="0.2 0.8 0.2 1"/></body>
    <body pos="-0.15 -0.1 0.25"><joint type="free"/><geom name="c1" type="cylinder" size="0.04 0.08" rgba="0.2 0.2 0.9 1"/></body>
    <body pos="0.05 -0.2 0.2"><joint type="free"/><geom name="k1" type="capsule" size="0.03 0.06" rgba="0.9 0.9 0.2 1"/></body>
  </worldbody>
</mujoco>
"""


def _fwd(model, seed=None):
    d = mujoco.MjData(model)
    if seed is not None:
        d.qpos[:] += np.random.default_rng(seed).uniform(-0.05, 0.05, model.nq)
    mujoco.mj_forward(model, d)
    return d


def _psnr(a, b):
    mse = np.mean((a.astype(float) - b.astype(float)) ** 2)
    return 10 * np.log10(255.0 ** 2 / max(mse, 1e-9))


def test_lambertian_plane_analytic():
    """Radiance of a Lambertian plane (albedo 0.8) under a directional light of MuJoCo diffuse 0.5
    at normal incidence is albedo/pi * E, with E = pi * 0.5 (the stack's radiometric convention,
    shared with tier 0): 0.4 exactly, independent of bounce count (the sky is black)."""
    model = mujoco.MjModel.from_xml_string(LAMBERT)
    for bounces in (0, 3):
        rend = Tier2Renderer(model, 1, width=64, height=64, camera="cam", spp=16, max_bounces=bounces)
        out = rend.render_host([_fwd(model)], passes=4)
        hdr = out["hdr"][0]
        assert np.isfinite(hdr).all()
        print(f"bounces {bounces}: mean {hdr.mean():.4f} (expected 0.4000), pixel std {hdr.std():.4f}")
        assert abs(hdr.mean() - 0.4) < 0.01
        assert hdr.std() < 0.01          # unshadowed, unoccluded: no noise sources
    assert out["rgb"][0].mean() == pytest.approx(0.4 * 255, abs=2)
    assert np.allclose(out["depth"][0], 2.0, atol=1e-3)
    assert (out["seg"][0] > 0).all()


def test_white_furnace_uniform_sky():
    """A white Lambertian plane under a uniform sky of radiance S reflects exactly S (energy
    conservation of the cosine-sampled diffuse lobe and its pdf)."""
    model = mujoco.MjModel.from_xml_string(FURNACE)
    rend = Tier2Renderer(model, 1, width=64, height=64, camera="cam", spp=32, max_bounces=1)
    out = rend.render_host([_fwd(model)], passes=4)
    hdr = out["hdr"][0]
    print(f"furnace: mean {hdr.mean():.4f} (expected 0.5000), pixel std {hdr.std():.4f}")
    assert abs(hdr.mean() - 0.5) < 0.01
    assert hdr.std() < 0.05


def test_monte_carlo_convergence():
    """Per-pixel noise between two independent renders falls as 1/sqrt(spp) in an indirectly-lit
    region (a box on the plane under the sky, 2 bounces)."""
    model = mujoco.MjModel.from_xml_string(PRIMS)
    d = _fwd(model)
    def noise(spp, seed):
        a = Tier2Renderer(model, 1, width=96, height=96, camera="cam", spp=spp, max_bounces=2, seed=seed).render_host([d])["hdr"][0]
        b = Tier2Renderer(model, 1, width=96, height=96, camera="cam", spp=spp, max_bounces=2, seed=seed + 100).render_host([d])["hdr"][0]
        return float(np.sqrt(np.mean((a - b) ** 2)))
    n4, n64 = noise(4, 1), noise(64, 2)
    print(f"noise rms: spp 4 -> {n4:.4f}, spp 64 -> {n64:.4f}, ratio {n4 / n64:.2f} (ideal 4.0)")
    assert n4 / n64 > 2.5


def test_direct_light_matches_tier0():
    """With bounces disabled the path tracer computes the same direct light + headlight + shadows
    that tier 0 rasterizes (same BRDF, same light convention): PSNR and identical depth/seg."""
    model = mujoco.MjModel.from_xml_string(PRIMS)
    t0 = Tier0Renderer(model, 1, width=192, height=192, camera="cam", outputs=("rgb", "depth", "seg"), include_planes=True)
    t2 = Tier2Renderer(model, 1, width=192, height=192, camera="cam", spp=16, max_bounces=0)
    psnrs = []
    for seed in range(3):
        d = _fwd(model, seed)
        a = t0.render_host([d]); b = t2.render_host([d], passes=4)
        psnrs.append(_psnr(a["rgb"][0], b["rgb"][0]))
        both = (a["seg"][0] > 0) & (b["seg"][0] > 0)
        dmed = np.median(np.abs(a["depth"][0][both] - b["depth"][0][both]))
        seg_iou = ((a["seg"][0] > 0) & (b["seg"][0] > 0)).sum() / ((a["seg"][0] > 0) | (b["seg"][0] > 0)).sum()
        assert dmed < 1e-3 and seg_iou > 0.97
    print("tier 2 (direct only) vs tier 0 PSNR:", np.round(psnrs, 2), "depth median", dmed, "seg IoU", round(seg_iou, 4))
    os.makedirs(OUT, exist_ok=True)
    try:
        import imageio.v2 as iio
        full = Tier2Renderer(model, 1, width=192, height=192, camera="cam", spp=64, max_bounces=4).render_host([d], passes=8)["rgb"][0]
        iio.imwrite(os.path.join(OUT, "tier2_vs_tier0.png"), np.concatenate([a["rgb"][0], b["rgb"][0], full], axis=1))
    except ImportError:
        pass
    assert min(psnrs) > 22.0


def test_gpu_path_from_batchsim_matches_host():
    from metalsim.physics.batch import BatchSim
    model = mujoco.MjModel.from_xml_string(PRIMS)
    n = 4
    sim = BatchSim(model, n)
    sim.synchronize()
    rend = Tier2Renderer(model, n, width=64, height=64, camera="cam", spp=8, max_bounces=1, seed=3)
    for _ in range(20):
        v = sim.step()
    vr = rend.render(sim, v); rend.wait_sim_after_render(sim, vr); rend.after(vr)
    gpu = rend.out.rgb.clone().cpu().numpy()
    sim.synchronize(); rend.ctx.synchronize()
    datas = []
    for e in range(n):
        d = mujoco.MjData(model); d.qpos[:] = sim.d.qpos.numpy()[e]; mujoco.mj_forward(model, d); datas.append(d)
    rend2 = Tier2Renderer(model, n, width=64, height=64, camera="cam", spp=8, max_bounces=1, seed=3)
    host = rend2.render_host(datas)["rgb"]
    assert rend.out.rgb.device.type == "mps" and gpu.shape == (n, 64, 64, 3)
    assert np.mean(np.abs(gpu.astype(int) - host.astype(int))) < 1.0   # same seeds, same state: same paths


def test_cartpole_rgb_rollout_tier2():
    """A full env rollout (physics + path-traced 100x100 observations + reward/reset) at tier 2."""
    from metalsim.learn.cartpole_rgb import CartpoleRGBEnv, CartpoleRGBConfig
    env = CartpoleRGBEnv(CartpoleRGBConfig(num_envs=32, tier=2, spp=2, max_bounces=2))
    obs = env.reset()
    a = torch.zeros(32, 1, device="mps")
    for _ in range(30):
        obs, r, done, info = env.step(torch.rand_like(a) * 2 - 1)
    env.synchronize()
    img = obs["image"].cpu().numpy()
    assert img.shape == (32, 100, 100, 3) and img.std() > 10 and torch.isfinite(r).all()


def test_usd_units_lambertian_sun_disk_and_dome():
    """USD / RTX mode: a DistantLight of intensity I (0.53 deg disk) gives irradiance pi*I at normal incidence,
    so a Lambertian plane (OmniPBR with specular_level 0, i.e. pure Lambert) reads albedo * I; a textureless
    DomeLight of intensity S*colour is a uniform environment of that radiance (white furnace)."""
    model = mujoco.MjModel.from_xml_string(LAMBERT)
    rend = Tier2Renderer(model, 1, width=64, height=64, camera="cam", spp=16, max_bounces=0, material_model="omnipbr",
                         usd_lights=[{"intensity": 3000.0, "color": (1, 1, 1), "angle_deg": 0.53}], headlight=False)
    hdr = rend.render_host([_fwd(model)], passes=4)["hdr"][0]
    print(f"usd sun: mean {hdr.mean():.2f} (expected {0.8 * 3000:.0f})")
    assert abs(hdr.mean() / (0.8 * 3000) - 1) < 0.01
    fm = mujoco.MjModel.from_xml_string(FURNACE)
    rend = Tier2Renderer(fm, 1, width=64, height=64, camera="cam", spp=32, max_bounces=1, material_model="omnipbr",
                         dome=(400.0, (0.75, 0.8, 0.9)), headlight=False)
    hdr = rend.render_host([_fwd(fm)], passes=4)["hdr"][0]
    exp = 400 * np.array([0.75, 0.8, 0.9])
    print("usd dome furnace:", hdr.reshape(-1, 3).mean(0), "expected", exp)
    assert np.allclose(hdr.reshape(-1, 3).mean(0) / exp, 1, atol=0.02)


def test_rtx_tonemap_and_denoisers_on_furnace():
    """RTX display transform: 8-bit output = sRGB(ACES(exposure * radiance)); both denoisers keep the mean of a
    furnace image (white plane under a uniform dome) and reduce its Monte Carlo noise."""
    from metalsim.render.tier2 import rtx_exposure
    fm = mujoco.MjModel.from_xml_string(FURNACE)
    k = rtx_exposure()
    tm = lambda L: 255 * (1.055 * np.clip(L * k * (2.51 * L * k + 0.03) / (L * k * (2.43 * L * k + 0.59) + 0.14), 0, 1) ** (1 / 2.4) - 0.055)
    ran = 0
    for dn in (None, "atrous", "oidn"):
        try:
            rend = Tier2Renderer(fm, 2, width=48, height=48, camera="cam", spp=4, max_bounces=1, dome=(400.0, (0.8, 0.8, 0.8)),
                                 headlight=False, tonemap="rtx", exposure=k, denoise=dn)
        except ImportError:
            continue
        out = rend.render_host([_fwd(fm), _fwd(fm)], passes=1)
        L = out["hdr"]
        if dn is None:
            assert abs(out["rgb"].mean() - tm(L).mean()) < 1.5
            continue
        ran += 1
        noisy, clean = out["hdr_mean"], out["hdr_denoised"]
        print(f"{dn}: mean {noisy.mean():.1f} -> {clean.mean():.1f}, std {noisy.std():.1f} -> {clean.std():.1f}")
        assert abs(clean.mean() / noisy.mean() - 1) < 0.02 and clean.std() < 0.5 * noisy.std()
        assert abs(out["rgb"].mean() - tm(clean).mean()) < 1.5
    assert ran >= 1


def test_physical_preset_batch_is_finite():
    """1024 camera-RL frames under the physical tier-2 preset (the Cartpole-RGB default): every accumulated radiance
    value is finite and the observation is not degenerate."""
    from metalsim.learn.cartpole_rgb import CartpoleRGBEnv, CartpoleRGBConfig
    env = CartpoleRGBEnv(CartpoleRGBConfig(num_envs=1024, tier=2, render_mode="physical"))
    obs = env.reset()
    for _ in range(10):
        obs, r, done, info = env.step(torch.rand(1024, 1, device="mps") * 2 - 1)
    env.synchronize()
    acc = env.rend._accum.numpy()
    assert np.isfinite(acc).all() and (acc[..., 3] > 0).all()
    img = obs["image"].float()
    assert torch.isfinite(img).all() and 40 < img.mean().item() < 200 and img.std().item() > 10

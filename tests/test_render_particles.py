"""Particle (render grain) rendering: analytic coverage and irradiance of a single grain, white-furnace energy
conservation in a grain pile, instance-count scaling, batched multi-env rendering, determinism, the GPU (Warp array)
input path, and the tier-0 raster path. See metalsim/render/PARTICLES.md."""
import time

import mujoco
import numpy as np
import pytest
import warp as wp

from metalsim.render.particles import GrainMaterial, grain_meshes
from metalsim.render.tier0 import Tier0Renderer
from metalsim.render.tier2 import Tier2Renderer

pytestmark = pytest.mark.skipif(not wp.is_metal_available(), reason="needs Metal")

# camera 1 m above the origin looking straight down; one directional light straight down; black sky; a small box
# far outside the view (a scene needs one drawable geom)
TOPDOWN = """
<mujoco>
  <visual><headlight active="0" ambient="0 0 0"/></visual>
  <worldbody>
    <light directional="true" pos="0 0 3" dir="0 0 -1" diffuse="{L} {L} {L}" specular="0 0 0" ambient="0 0 0"/>
    <camera name="cam" pos="0 0 1" xyaxes="1 0 0 0 1 0" fovy="{fovy}"/>
    <geom name="far" type="box" size="0.01 0.01 0.01" pos="30 30 -30" rgba="0 0 0 1"/>
  </worldbody>
</mujoco>
"""

FURNACE = """
<mujoco>
  <visual><headlight active="0" ambient="0 0 0"/></visual>
  <asset><texture type="skybox" builtin="flat" rgb1="0.5 0.5 0.5" rgb2="0.5 0.5 0.5" width="8" height="48"/></asset>
  <worldbody>
    <camera name="cam" pos="0.35 -0.35 0.35" xyaxes="0.707 0.707 0 -0.4 0.4 0.82" fovy="40"/>
    <geom name="far" type="box" size="0.01 0.01 0.01" pos="30 30 -30" rgba="1 1 1 1"/>
  </worldbody>
</mujoco>
"""

SPHERE_SHAPE = dict(irregularity=0.0, elongation=0.0, subdiv=3)   # an icosphere of 1280 triangles


def _datas(m, n=1):
    out = []
    for _ in range(n):
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        out.append(d)
    return out


def _pile(n=2000, seed=0, half=0.06):
    rs = np.random.default_rng(seed)
    xy = rs.uniform(-half, half, (n, 2))
    z = rs.uniform(0, 0.04, n)
    return np.concatenate([xy, z[:, None]], 1).astype(np.float32)


def test_grain_meshes_unit_volume():
    """Every grain mesh is closed, outward-oriented and has unit volume-equivalent radius."""
    for v, idx in grain_meshes(6, 1, seed=3):
        p = v[:, :3].astype(np.float64)[idx.reshape(-1, 3)]
        vol = np.einsum("ij,ij->i", p[:, 0], np.cross(p[:, 1], p[:, 2])).sum() / 6
        assert abs(vol - 4 / 3 * np.pi) < 1e-3
        assert np.allclose(np.linalg.norm(v[:, 3:6], axis=1), 1, atol=1e-5)


def test_single_grain_coverage_and_irradiance():
    """One spherical grain seen from above under a directional light along the view axis.

    Coverage: grain pixels = projected disk area pi r^2 / pixel area (within 2 %; the polyhedron is inscribed).
    Radiance: a Lambertian sphere lit head-on, seen head-on, has mean radiance over its disk
    rho * E / pi * <cos> with <cos> = 2/3 (E = pi * diffuse, the stack's convention), within 2 %."""
    L, rho, r, fovy, W = 0.7, 0.6, 0.05, 20.0, 256
    m = mujoco.MjModel.from_xml_string(TOPDOWN.format(L=L, fovy=fovy))
    rt = Tier2Renderer(m, 1, width=W, height=W, spp=16, max_bounces=1, headlight=False)
    rt.set_particles(np.zeros((1, 3), np.float32), radius=1e-7, grains_per_particle=1, grain_radius=r, radius_jitter=0.0,
                     material=GrainMaterial.uniform((rho, rho, rho), roughness=1.0, specular=0.0), grain_shape=SPHERE_SHAPE,
                     n_meshes=1)
    out = rt.render_host(_datas(m), passes=16)
    seg = out["seg"][0]
    grain = seg == rt.G + 1
    pix = 2 * 1.0 * np.tan(np.radians(fovy) / 2) / W        # pixel footprint at 1 m
    # the camera sits d = 1 m above the grain centre: the silhouette cone has tan(half angle) = r / sqrt(d^2 - r^2)
    d = 1.0
    expect = np.pi * (r / np.sqrt(d * d - r * r)) ** 2 / pix ** 2
    assert abs(grain.sum() / expect - 1) < 0.02, (grain.sum(), expect)
    hdr = out["hdr"][0][..., 0]
    yy, xx = np.mgrid[0:W, 0:W]
    inner = (np.hypot(xx + 0.5 - W / 2, yy + 0.5 - W / 2) < 0.95 * np.sqrt(expect / np.pi))   # skip the edge pixels
    # analytic mean over the inner disk of radius a*R: cos = sqrt(1 - s^2), mean over s < a: 2/3 (1 - (1 - a^2)^1.5) / a^2
    a = 0.95
    mean_cos = 2 / 3 * (1 - (1 - a * a) ** 1.5) / (a * a)
    analytic = rho * L * mean_cos
    assert abs(hdr[inner].mean() / analytic - 1) < 0.02, (hdr[inner].mean(), analytic)
    assert np.all(hdr[~grain] == 0)


def test_white_furnace_grain_pile():
    """Energy conservation with grains: white (albedo 1) Lambertian grains under a uniform sky of radiance 0.5 return
    0.5 everywhere, however many interreflections the pile makes (within 3 % on the mean, unbiased RR)."""
    m = mujoco.MjModel.from_xml_string(FURNACE)
    rt = Tier2Renderer(m, 1, width=96, height=96, spp=8, max_bounces=48, headlight=False)
    rt.set_particles(_pile(3000), radius=0.004, grains_per_particle=4,
                     material=GrainMaterial.uniform((1.0, 1.0, 1.0), roughness=1.0, specular=0.0))
    out = rt.render_host(_datas(m), passes=16)
    seg = out["seg"][0]
    grain = seg == rt.G + 1
    assert grain.mean() > 0.3
    h = out["hdr"][0][grain]
    assert abs(h.mean() / 0.5 - 1) < 0.03, h.mean()
    assert h.max() < 2.0          # no energy gain in any pixel beyond Monte Carlo noise


def test_instance_count_scaling():
    """Instances = scene instances + n_envs * particles * grains; the acceleration structure grows about linearly
    and a frame renders at every size (timings printed, not asserted)."""
    m = mujoco.MjModel.from_xml_string(FURNACE)
    rt = Tier2Renderer(m, 2, width=64, height=64, spp=1, max_bounces=2, headlight=False)
    sizes = {}
    for P, K in ((1000, 1), (10000, 8), (100000, 8)):
        rt.set_particles(_pile(P, half=0.2), radius=0.003, grains_per_particle=K)
        t0 = time.time()
        out = rt.render_host(_datas(m, 2), passes=1)
        dt = time.time() - t0
        assert rt._ias.n_total == rt.rt.n_inst + 2 * P * K
        assert (out["seg"] == rt.G + 1).any()
        sizes[P * K] = rt._ias.as_bytes
        print(f"grains {2 * P * K}: structure {rt._ias.as_bytes / 2**20:.1f} MiB, frame {dt * 1e3:.0f} ms")
    r = sizes[800000] / sizes[80000]
    assert 5 < r < 15, sizes


def test_batched_envs_and_isolation():
    """Each env draws its own particles: env 0 a pile left of the camera axis, env 1 right of it, env 2 none
    (non-finite positions: culled), env 3 the same pile as env 0 (same image as env 0)."""
    m = mujoco.MjModel.from_xml_string(FURNACE)
    rt = Tier2Renderer(m, 4, width=96, height=96, spp=4, max_bounces=3, headlight=False, seed=5)
    pile = _pile(1500)
    P = np.stack([pile + [0.0, -0.05, 0], pile + [0.05, 0.0, 0], np.full_like(pile, np.nan), pile + [0.0, -0.05, 0]])
    rt.set_particles(P, radius=0.004, grains_per_particle=4)
    out = rt.render_host(_datas(m, 4), passes=2)
    g = out["seg"] == rt.G + 1
    assert g[0].sum() > 100 and g[1].sum() > 100 and g[2].sum() == 0
    cx = [np.nonzero(g[e])[1].mean() for e in (0, 1)]
    assert cx[0] != pytest.approx(cx[1], abs=3)
    assert np.array_equal(g[0], g[3])          # identical geometry in env 0 and env 3 (separate tiles of one structure)


def test_determinism():
    """Two renderers with the same seed produce bit-identical images (grain instances are hashed from the particle
    and grain index, the structure build is deterministic)."""
    m = mujoco.MjModel.from_xml_string(FURNACE)
    imgs = []
    for _ in range(2):
        rt = Tier2Renderer(m, 1, width=64, height=64, spp=4, max_bounces=4, headlight=False, seed=11)
        rt.set_particles(_pile(4000), radius=0.004, grains_per_particle=8)
        imgs.append(rt.render_host(_datas(m), passes=3)["hdr"])
    assert np.array_equal(imgs[0], imgs[1])


def test_gpu_positions_match_host():
    """A Warp array on the Metal device (bound directly, no host copy) renders the same image as the numpy input."""
    m = mujoco.MjModel.from_xml_string(FURNACE)
    pile = _pile(3000)
    imgs = []
    for src in ("numpy", "warp"):
        rt = Tier2Renderer(m, 1, width=64, height=64, spp=2, max_bounces=3, headlight=False, seed=2)
        pos = pile if src == "numpy" else wp.array(pile, dtype=wp.vec3, device="metal:0")
        rt.set_particles(pos, radius=0.004, grains_per_particle=4)
        imgs.append(rt.render_host(_datas(m), passes=2)["hdr"])
    assert np.array_equal(imgs[0], imgs[1])


def test_tier0_grains():
    """Tier 0 draws the same grains: grain segmentation where the path tracer has grains (IoU > 0.9), the uniform
    grain colour, and per-env particles."""
    lit = FURNACE.replace("<worldbody>", '<worldbody><light directional="true" pos="0 0 3" dir="0.2 0.3 -1" diffuse="0.8 0.8 0.8"/>')
    m = mujoco.MjModel.from_xml_string(lit)
    pile = _pile(2000)
    P = np.stack([pile, np.full_like(pile, np.nan)])
    mat = GrainMaterial.uniform((0.8, 0.5, 0.2), roughness=1.0, specular=0.0)
    r0 = Tier0Renderer(m, 2, width=96, height=96, camera="cam", outputs=("rgb", "seg"))
    r0.set_particles(P, radius=0.004, grains_per_particle=4, material=mat)
    o0 = r0.render_host(_datas(m, 2))
    seg0, rgb0 = o0["seg"], o0["rgb"]
    rt = Tier2Renderer(m, 2, width=96, height=96, spp=1, max_bounces=1, headlight=False)
    rt.set_particles(P, radius=0.004, grains_per_particle=4, material=mat)
    seg2 = rt.render_host(_datas(m, 2), passes=1)["seg"]
    g0, g2 = seg0 == r0.G + 1, seg2 == rt.G + 1
    assert g0[1].sum() == 0 and g0[0].sum() > 100
    iou = (g0[0] & g2[0]).sum() / (g0[0] | g2[0]).sum()
    assert iou > 0.9, iou
    c = rgb0[0][g0[0]].astype(float).mean(0)
    assert c[0] > c[1] > c[2]                  # the orange grain colour reaches the image

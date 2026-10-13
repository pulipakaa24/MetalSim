"""MuJoCo's haze over infinite planes and its fog (metalsim.render.haze, shaders/haze.h) against mujoco.Renderer 3.14.0.

Scene: a black infinite plane under a black skybox, no lights, so MuJoCo's image is the haze alone (white, alpha
blended) and every renderer must reproduce it: the NumPy model of the faceted cone on the CPU, then tier 0, tier 1 and
tier 2. Fog: a blue plane under a red fog seen at 90 degrees (GL linear fog on eye-space depth). GPU tests:
scripts/gpu_run.sh check."""
import mujoco
import numpy as np
import pytest
import warp as wp

from metalsim.render import haze as hz

needs_metal = pytest.mark.skipif(not wp.is_metal_available(), reason="needs Metal")
W, H = 80, 60


def _scene(height=20.0, pitch=-20.0, skybox=True):
    sky = '<texture type="skybox" builtin="flat" rgb1="0 0 0" rgb2="0 0 0" width="16" height="16"/>' if skybox else ""
    xml = f"""<mujoco><visual><headlight ambient="0 0 0" diffuse="0 0 0" specular="0 0 0"/></visual>
    <statistic extent="1" center="0 0 0"/>
    <asset>{sky}<material name="black" rgba="0 0 0 1" emission="0" specular="0"/></asset>
    <worldbody><geom type="plane" size="0 0 1" material="black"/>
    <camera name="c" pos="0 0 {height}" euler="{90 + pitch} 0 0" fovy="60"/></worldbody></mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return m, d


def _mujoco(m, d):
    r = mujoco.Renderer(m, H, W)
    r.update_scene(d, camera="c")
    img = r.render().astype(float)
    r.close()
    return img


@pytest.mark.parametrize("height,pitch", [(20.0, -20.0), (1.0, -2.0), (6.0, -45.0)])
def test_cone_model_matches_mujoco(height, pitch):
    """CPU: haze_alpha of the faceted cone vs mujoco.Renderer's image: mean error < 0.002, max 0.02 (0.15 for the camera
    1 m up, whose haze is a 1-2 px band whose edges MuJoCo's multisampling blends)."""
    m, d = _scene(height, pitch)
    p = hz.haze_params(m, [0], sky_drawn=True)
    fy = 0.5 * H / np.tan(np.deg2rad(30.0))
    dirs = hz.camera_dirs(W, H, fy, fy, W / 2, H / 2, d.cam_xmat[0])
    a = hz.haze_alpha(p, d.cam_xpos[0], dirs)
    err = np.abs(_mujoco(m, d)[..., 0] / 255 - a)
    assert err.mean() < 0.002, (height, pitch, err.mean())
    assert err.max() < (0.15 if height < 2 else 0.02), (height, pitch, err.max())   # low camera: a 1-2 px band, MSAA edges


def test_params():
    m, _ = _scene()
    p = hz.haze_params(m, [0], sky_drawn=True)
    assert p[3] == pytest.approx(0.7 * m.vis.map.zfar * m.stat.extent, rel=1e-6) and hz.active(p)
    assert not hz.active(hz.haze_params(m, [0], sky_drawn=False))          # haze is drawn in the skybox pass only
    assert not hz.active(hz.haze_params(m, [0], haze=False))
    assert hz.active(hz.haze_params(m, [0], haze=False, fog=True))


@needs_metal
@pytest.mark.parametrize("which", ["tier0", "tier1", "tier2"])
def test_haze_in_every_renderer(which):
    from metalsim.render.tier0 import Tier0Renderer
    from metalsim.render.tier2 import Tier2Renderer
    m, d = _scene()
    ref = _mujoco(m, d)[..., 0] / 255
    for on in (True, False):
        if which == "tier2":
            r = Tier2Renderer(m, 2, width=W, height=H, camera="c", spp=1, max_bounces=1, center_sample=True, haze=on)
        else:
            r = Tier0Renderer(m, 2, width=W, height=H, camera="c", tier=1 if which == "tier1" else 0, far=100.0, haze=on)
        img = r.render_host([d, d])["rgb"][0].astype(float)[..., 0] / 255
        if on:
            err = np.abs(img - ref)
            assert err.max() <= 3 / 255 + 1e-6 and np.percentile(err, 99) <= 1 / 255 + 1e-6, (which, err.max())
        else:
            assert img.max() == 0                      # no haze: black plane, black sky


@needs_metal
def test_fog_matches_mujoco():
    """mjRND_FOG on (fog=True): a blue plane fogged to red by eye depth, as mujoco.Renderer."""
    from metalsim.render.tier0 import Tier0Renderer
    xml = """<mujoco><visual><map fogstart="0.5" fogend="2" znear="0.01"/><rgba fog="1 0 0 1"/>
      <headlight ambient="1 1 1" diffuse="0 0 0" specular="0 0 0"/></visual><statistic extent="1" center="0 0 0"/>
      <worldbody><geom type="box" size="5 5 0.01" pos="0 0 -1" rgba="0 0 1 1" material="m"/>
      <camera name="c" pos="0 0 0" xyaxes="1 0 0 0 1 0" fovy="90"/></worldbody>
      <asset><material name="m" emission="0" specular="0" shininess="0"/></asset></mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    r = mujoco.Renderer(m, H, W); r.update_scene(d, camera="c")
    r.scene.flags[mujoco.mjtRndFlag.mjRND_FOG] = 1
    ref = r.render().astype(float); r.close()
    ours = Tier0Renderer(m, 1, width=W, height=H, camera="c", fog=True).render_host([d])["rgb"][0].astype(float)
    assert np.abs(ours - ref)[5:-5, 5:-5].max() <= 3, np.abs(ours - ref).max()
    plain = Tier0Renderer(m, 1, width=W, height=H, camera="c").render_host([d])["rgb"][0].astype(float)
    assert plain[H // 2, W // 2, 0] < 5                 # off by default, as mujoco.Renderer

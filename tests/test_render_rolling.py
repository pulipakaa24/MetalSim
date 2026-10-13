"""Rolling shutter (metalsim.render.rolling) against the analytic skew of a moving object and of a panning camera.

Row r is drawn with the poses at time tau_r of the interval between the previous rendered frame (0) and the current one
(1), as Omniverse RTX's per-pixel shutter time. A vertical bar moving sideways at constant velocity must appear sheared:
its edge in row r sits where the pinhole projection of the bar at tau_r puts it. Checked per row in tier 0 and tier 2,
for the object moving and for the camera panning (orientation slerp), with one band per row (exact) and with 8 bands
(a staircase within half a band's motion). GPU tests: scripts/gpu_run.sh check."""
import mujoco
import numpy as np
import pytest
import warp as wp

from metalsim.render.rolling import RollingShutter

needs_metal = pytest.mark.skipif(not wp.is_metal_available(), reason="needs Metal")
W, H = 96, 64
FOVY = 50.0

XML = f"""
<mujoco>
  <visual><headlight ambient="0.5 0.5 0.5" diffuse="0.5 0.5 0.5"/></visual>
  <worldbody>
    <body name="cambody" mocap="true" pos="0 -2 0.5"><camera name="cam" xyaxes="1 0 0 0 0 1" fovy="{FOVY}"/></body>
    <body name="bar" mocap="true" pos="0 0 0.5"><geom name="bar" type="box" size="0.03 0.03 1.5" rgba="0.9 0.2 0.2 1"/></body>
  </worldbody>
</mujoco>"""


def _state(m, bar_x=0.0, yaw_deg=0.0):
    d = mujoco.MjData(m)
    d.mocap_pos[1] = (bar_x, 0, 0.5)
    q = np.zeros(4)
    mujoco.mju_axisAngle2Quat(q, np.array([0, 0, 1.0]), np.deg2rad(yaw_deg))
    d.mocap_quat[0] = q
    mujoco.mj_forward(m, d)
    return d


def _left_edge(seg_row):
    idx = np.flatnonzero(seg_row > 0)
    return idx[0] if len(idx) else -1


def _predicted_left_edge(bar_x, yaw_deg):
    """Column of the first pixel centre inside the bar's left face edge (the bar 2 m ahead of the camera)."""
    f = 0.5 * H / np.tan(np.deg2rad(FOVY) / 2)
    edge = np.array([bar_x - 0.03, 2.0 - 0.03])                 # left-near corner in (x, depth) of the camera at yaw 0
    a = np.deg2rad(yaw_deg)                                     # camera yawed by a: points rotate by -a in its frame
    xc = np.cos(a) * edge[0] + np.sin(a) * edge[1]
    zc = -np.sin(a) * edge[0] + np.cos(a) * edge[1]
    xn = np.cos(a) * (bar_x - 0.03) + np.sin(a) * 2.03          # the far corner (silhouette when rotated left)
    zn = -np.sin(a) * (bar_x - 0.03) + np.cos(a) * 2.03
    u = min(f * xc / zc, f * xn / zn) + W / 2
    return int(np.ceil(u - 0.5))


def _render(which, m, rs, prev, cur):
    from metalsim.render.tier0 import Tier0Renderer, SEG_GEOM
    from metalsim.render.tier2 import Tier2Renderer
    if which == "tier0":
        r = Tier0Renderer(m, 1, width=W, height=H, camera="cam", outputs=("rgb", "seg"), seg_mode=SEG_GEOM, rolling_shutter=rs)
        return r.render_host([cur], prev=[prev])["seg"][0]
    r = Tier2Renderer(m, 1, width=W, height=H, camera="cam", spp=1, max_bounces=1, center_sample=True, rolling_shutter=rs)
    return r.render_host([cur], prev=[prev])["seg"][0]


def test_bands():
    rs = RollingShutter.top_to_bottom(64, 0.5)
    b = rs.bands(64)
    assert len(b) == 64 and b[0][2] == pytest.approx(0.5) and b[-1][2] == pytest.approx(1.0)
    b8 = RollingShutter.top_to_bottom(64, 0.5, bands=8).bands(64)
    assert [x[:2] for x in b8] == [(8 * k, 8 * k + 8) for k in range(8)]
    assert RollingShutter(np.full(64, 1.0)).bands(64) == [(0, 64, 1.0)]


@needs_metal
@pytest.mark.parametrize("which", ["tier0", "tier2"])
@pytest.mark.parametrize("motion", ["object", "camera"])
def test_skew_matches_analytic(which, motion):
    m = mujoco.MjModel.from_xml_string(XML)
    if motion == "object":
        prev, cur = _state(m, bar_x=-0.4), _state(m, bar_x=0.4)
        x_at = lambda t: (-0.4 + 0.8 * t, 0.0)          # noqa: E731
    else:
        prev, cur = _state(m, yaw_deg=-8.0), _state(m, yaw_deg=8.0)
        x_at = lambda t: (0.0, -8.0 + 16.0 * t)         # noqa: E731
    readout = 0.75
    rs = RollingShutter.top_to_bottom(H, readout)
    seg = _render(which, m, rs, prev, cur)
    rows = range(4, H - 4)
    err = []
    for r in rows:
        t = 1.0 - readout * (H - 1 - r) / (H - 1)
        err.append(_left_edge(seg[r]) - _predicted_left_edge(*x_at(t)))
    err = np.abs(np.asarray(err))
    assert err.max() <= 1, (which, motion, err)
    # the skew is real: the top and bottom rows' edges differ by the motion over the readout (> 10 px here)
    assert abs(_left_edge(seg[4]) - _left_edge(seg[H - 5])) > 10
    # global shutter (all rows at 1): a straight edge at the current pose
    seg_g = _render(which, m, RollingShutter(np.ones(H)), prev, cur)
    assert len({_left_edge(seg_g[r]) for r in rows}) <= 2
    # 8 bands: each band at its mean time, within half a band's motion (+ 1 px)
    seg8 = _render(which, m, RollingShutter.top_to_bottom(H, readout, bands=8), prev, cur)
    band_px = abs(_predicted_left_edge(*x_at(1.0)) - _predicted_left_edge(*x_at(1.0 - readout))) / 8
    e8 = [abs(_left_edge(seg8[r]) - _predicted_left_edge(*x_at(1.0 - readout * (H - 1 - r) / (H - 1)))) for r in rows]
    assert max(e8) <= band_px / 2 + 1.5, (which, motion, max(e8), band_px)


@needs_metal
@pytest.mark.parametrize("which", ["tier0", "tier2"])
def test_gpu_path_uses_the_previous_frame(which):
    """BatchSim GPU path: the first frame has no history (no skew); the second interpolates from the first frame's
    poses, equal to the host path given that previous state."""
    import torch
    from metalsim.physics.batch import BatchSim
    from metalsim.render.tier0 import Tier0Renderer, SEG_GEOM
    from metalsim.render.tier2 import Tier2Renderer
    m = mujoco.MjModel.from_xml_string(XML)
    sim = BatchSim(m, 2)
    rs = RollingShutter.top_to_bottom(H, 0.75)

    def put(x):
        mp = sim.d.mocap_pos.numpy(); mp[:, 1] = (x, 0, 0.5)
        sim.d.mocap_pos.assign(mp); wp.synchronize_device("metal:0")
        sim.forward(); sim.synchronize()
    if which == "tier0":
        r = Tier0Renderer(m, 2, width=W, height=H, camera="cam", outputs=("seg",), seg_mode=SEG_GEOM, rolling_shutter=rs)
    else:
        r = Tier2Renderer(m, 2, width=W, height=H, camera="cam", spp=1, max_bounces=1, center_sample=True, rolling_shutter=rs)
    put(-0.4)
    v = r.render(sim); r.after(v); torch.mps.synchronize()
    s0 = r.out.seg.cpu().numpy()[0]
    assert len({_left_edge(s0[k]) for k in range(4, H - 4)}) <= 2          # no history: no skew
    put(0.4)
    v = r.render(sim); r.after(v); torch.mps.synchronize()
    s1 = r.out.seg.cpu().numpy()[0]
    ref = _render(which, m, rs, _state(m, bar_x=-0.4), _state(m, bar_x=0.4))
    assert np.mean((s1 > 0) != (ref > 0)) < 0.002

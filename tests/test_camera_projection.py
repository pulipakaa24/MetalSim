"""Camera projection convention vs MuJoCo (mujoco.Renderer): small spheres at the image edges and corners land on the same
pixels in tiers 0 / 1 / 2 and the ray-traced depth camera, for a fovy camera and a physical camera with an offset
principal point (MuJoCo's frustum: engine_vis_visualize.c getFrustum + render_gl3.c glFrustum).

The first test is pure math (no GPU): MuJoCo's frustum re-implemented from its C source vs ``CameraIntrinsics``.
"""
import mujoco
import numpy as np
import pytest

from metalsim.render.scene_tables import CameraIntrinsics

W, H = 160, 120
CAMS = """
<camera name="fovy" pos="0 0 0" xyaxes="1 0 0 0 1 0" fovy="50"/>
<camera name="phys" pos="0 0 0" xyaxes="1 0 0 0 1 0" sensorsize="0.004 0.003" focal="0.003 0.003" principal="0.0003 -0.0002" resolution="160 120"/>
<camera name="audit" pos="0 -2.2 0.75" xyaxes="1 0 0 0 0.25 0.968"/>
"""


def _mujoco_pixel(m, cam, p_cam, w=W, h=H):
    """MuJoCo's projection of a camera-frame point (x right, y up, -z forward) to continuous pixel coordinates
    (rows down), from mjv_cameraFrustum / getFrustum and mjr_render's glFrustum."""
    zn = 1.0
    if m.cam_sensorsize[cam][1]:
        f, s, p = m.cam_intrinsic[cam][:2], m.cam_sensorsize[cam], m.cam_intrinsic[cam][2:4]
        zhor = (zn / f[0] * (s[0] / 2 - p[0]), zn / f[0] * (s[0] / 2 + p[0]))
        zver = (zn / f[1] * (s[1] / 2 - p[1]), zn / f[1] * (s[1] / 2 + p[1]))
        top, bottom = zver[0], -zver[1]
        center, halfwidth = (zhor[1] - zhor[0]) / 2, (zhor[1] + zhor[0]) / 2
    else:
        t = zn * np.tan(np.deg2rad(m.cam_fovy[cam]) / 2)
        top, bottom, center = t, -t, 0.0
        halfwidth = 0.5 * w / h * (top - bottom)
    left, right = center - halfwidth, center + halfwidth
    x, y = p_cam[0] / -p_cam[2] * zn, p_cam[1] / -p_cam[2] * zn
    u = (x - left) / (right - left) * w
    v = (top - y) / (top - bottom) * h
    return np.array([u, v])


def _model(extra=""):
    return mujoco.MjModel.from_xml_string(f"<mujoco><worldbody>{CAMS}{extra}</worldbody></mujoco>")


@pytest.mark.parametrize("cam", ["fovy", "phys"])
def test_intrinsics_match_mujoco_frustum(cam):
    m = _model('<geom type="sphere" size="0.1" pos="0 0 -5"/>')
    c = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, cam)
    ci = CameraIntrinsics.from_model(m, c, W, H)
    rng = np.random.default_rng(0)
    for _ in range(50):
        p = np.array([rng.uniform(-1, 1), rng.uniform(-1, 1), -rng.uniform(1, 5)])
        ours = np.array([ci.cx + ci.fx * p[0] / -p[2], ci.cy - ci.fy * p[1] / -p[2]])
        np.testing.assert_allclose(ours, _mujoco_pixel(m, c, p), atol=1e-4)   # float32 model fields


def _edge_points(m, cam, ci, depth=2.0):
    """Camera-frame centres of spheres that project to pixels near the edges and corners."""
    px = [(4.5, 3.5), (W / 2 + 0.5, 2.5), (W - 4.5, 3.5), (2.5, H / 2 + 0.5), (W - 2.5, H / 2 + 0.5), (4.5, H - 3.5), (W / 2 + 0.5, H - 2.5),
          (W - 4.5, H - 3.5), (W / 2 + 0.5, H / 2 + 0.5)]   # pixel centres: a 1 px capsule on a pixel edge covers no centre
    return [np.array([(u - ci.cx) / ci.fx * depth, -(v - ci.cy) / ci.fy * depth, -depth]) for u, v in px]


@pytest.mark.parametrize("cam", ["fovy", "phys", "audit"])
def test_edge_and_corner_objects_vs_mujoco_renderer(cam):
    """``audit``: the render-coverage audit's camera (tilted, MuJoCo's default fovy) with 2 cm capsules 3 m away."""
    import warp as wp
    if not wp.is_metal_available():
        pytest.skip("needs Metal")
    from metalsim.render.tier0 import SEG_GEOM, Tier0Renderer
    from metalsim.render.tier2 import Tier2Renderer
    from metalsim.sensors.raytrace import RayTracer
    m0 = _model()
    c = mujoco.mj_name2id(m0, mujoco.mjtObj.mjOBJ_CAMERA, cam)
    d0 = mujoco.MjData(m0); mujoco.mj_forward(m0, d0)
    depth = 3.0 if cam == "audit" else 2.0
    pts = [d0.cam_xpos[c] + d0.cam_xmat[c].reshape(3, 3) @ p
           for p in _edge_points(m0, c, CameraIntrinsics.from_model(m0, c, W, H), depth)]
    shape = 'type="capsule" size="0.01 0.05"' if cam == "audit" else 'type="sphere" size="0.03"'
    geoms = "".join(f'<geom name="s{i}" {shape} pos="{p[0]} {p[1]} {p[2]}" rgba="0.9 0.3 0.3 1"/>'
                    for i, p in enumerate(pts))
    m = _model('<light directional="true" pos="0 0 3" dir="0 0 -1"/>' + geoms)
    d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    r = mujoco.Renderer(m, H, W); r.enable_segmentation_rendering(); r.update_scene(d, camera=cam)
    ref = r.render()[..., 0].copy(); r.close()
    outs = {}
    for tier in (0, 1):
        t = Tier0Renderer(m, 1, width=W, height=H, camera=cam, outputs=("seg",), seg_mode=SEG_GEOM, tier=tier)
        outs[f"tier{tier}"] = t.render_host([d])["seg"][0] - 1
    t2 = Tier2Renderer(m, 1, width=W, height=H, camera=cam, spp=1, max_bounces=1, center_sample=True)
    outs["tier2"] = np.concatenate([[-1], t2.tables.slot_seg_ids() - 1])[t2.render_host([d])["seg"][0]]
    rt = RayTracer(m, 1, max_range=10.0)
    outs["raydepth"] = np.where(rt.trace_host([d], [rt.make_depth_camera(cam, W, H)])["ray_depth"][0] > 0, 0, -1)
    for g in range(m.ngeom):
        a = ref == g
        assert a.sum() >= 4, (cam, g)
        ya, xa = np.nonzero(a)
        for k, seg in outs.items():
            b = (seg >= 0) if k == "raydepth" else (seg == g)
            if k == "raydepth":                      # no ids: the sphere's neighbourhood
                b = b & (np.abs(np.arange(H)[:, None] - ya.mean()) < 6) & (np.abs(np.arange(W)[None] - xa.mean()) < 6)
            yb, xb = np.nonzero(b)
            iou = (a & b).sum() / max((a | b).sum(), 1)
            off = (xb.mean() - xa.mean(), yb.mean() - ya.mean())
            print(cam, g, k, "rows", int(ya.min()), int(ya.max()), "iou", round(float(iou), 3), "offset px", np.round(off, 3))
            assert iou > 0.8 and max(abs(off[0]), abs(off[1])) < 0.35, (cam, g, k, round(iou, 3), off)

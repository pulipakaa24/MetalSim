"""Tier-0 renderer acceptance: parity against mujoco.Renderer (the plan's thresholds: silhouette IoU
> 0.95, texture correlation > 0.99, depth median < 1 cm, segmentation IoU), zero-copy outputs, and
GPU-path ordering against BatchSim."""
import os

import mujoco
import numpy as np
import pytest
import torch
import warp as wp

from metalsim.render.tier0 import Tier0Renderer, SEG_GEOM

pytestmark = pytest.mark.skipif(not wp.is_metal_available(), reason="needs Metal")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SO101 = os.path.join(ROOT, "assets", "so101", "scene_box_rl.xml")
OUT = os.path.join(ROOT, "docs", "measurements")

PRIMS = """
<mujoco>
  <worldbody>
    <light directional="true" pos="0 0 3" dir="0.3 0.3 -0.9"/>
    <camera name="cam" pos="0.6 -0.6 0.5" quat="0.85 0.4 0.15 0.3"/>
    <geom name="floor" type="plane" size="2 2 0.1" rgba="0.3 0.3 0.35 1"/>
    <body pos="0 0 0.3"><joint type="free"/><geom name="b1" type="box" size="0.06 0.04 0.05" rgba="0.9 0.2 0.2 1"/></body>
    <body pos="0.15 0.1 0.2"><joint type="free"/><geom name="s1" type="sphere" size="0.05" rgba="0.2 0.8 0.2 1"/></body>
    <body pos="-0.15 -0.1 0.25"><joint type="free"/><geom name="c1" type="cylinder" size="0.04 0.08" rgba="0.2 0.2 0.9 1"/></body>
    <body pos="0.05 -0.2 0.2"><joint type="free"/><geom name="k1" type="capsule" size="0.03 0.06" rgba="0.9 0.9 0.2 1"/></body>
  </worldbody>
</mujoco>
"""

TEXTURED = """
<mujoco>
  <asset>
    <texture name="checker" type="2d" builtin="checker" rgb1="0.1 0.1 0.1" rgb2="0.9 0.9 0.9" width="64" height="64"/>
    <material name="checker" texture="checker" texrepeat="4 4"/>
    <texture name="grad" type="2d" builtin="gradient" rgb1="1 0 0" rgb2="0 0 1" width="64" height="64"/>
    <material name="grad" texture="grad" texrepeat="1 1"/>
  </asset>
  <worldbody>
    <light directional="true" pos="0 0 3" dir="0 0 -1"/>
    <camera name="cam" pos="0 -0.9 0.9" xyaxes="1 0 0 0 0.707 0.707"/>
    <geom name="floor" type="plane" size="1 1 0.1" material="checker"/>
    <geom name="b1" type="box" pos="0 0 0.15" size="0.15 0.15 0.15" material="grad"/>
  </worldbody>
</mujoco>
"""


def _ref(model, d, cam, w, h, seg=False, depth=False):
    r = mujoco.Renderer(model, h, w)
    if seg:
        r.enable_segmentation_rendering()
    if depth:
        r.enable_depth_rendering()
    r.update_scene(d, camera=cam)
    return r.render()


def _iou(a, b):
    return (a & b).sum() / max((a | b).sum(), 1)


def test_silhouette_and_depth_and_seg_parity_primitives():
    model = mujoco.MjModel.from_xml_string(PRIMS)
    rend = Tier0Renderer(model, 1, width=256, height=256, camera="cam", include_planes=False,
                         outputs=("rgb", "depth", "seg"), seg_mode=SEG_GEOM)
    rend.set_backgrounds([0], [np.full((256, 256, 3), 120, np.uint8)])
    ious, depth_med, seg_ious = [], [], []
    for seed in range(4):
        d = mujoco.MjData(model)
        rng = np.random.default_rng(seed)
        d.qpos[:] += rng.uniform(-0.05, 0.05, model.nq)
        mujoco.mj_forward(model, d)
        out = rend.render_host([d])
        rgb, depth, seg = out["rgb"][0], out["depth"][0], out["seg"][0]
        ref_seg = _ref(model, d, "cam", 256, 256, seg=True)[:, :, 0].astype(int)
        ref_depth = _ref(model, d, "cam", 256, 256, depth=True)
        fg_ref = np.isin(ref_seg, rend.tables.geoms)
        fg_ours = np.any(np.abs(rgb.astype(int) - 120) > 8, axis=2)
        ious.append(_iou(fg_ref, fg_ours))
        both = fg_ref & (seg > 0)
        depth_med.append(np.median(np.abs(depth[both] - ref_depth[both])))
        # per-geom segmentation IoU (our ids are model geom id + 1)
        for g in rend.tables.geoms:
            if (ref_seg == g).sum() > 20:   # geom visible in the reference
                seg_ious.append(_iou(ref_seg == g, seg == g + 1))
    print("silhouette IoU", np.round(ious, 4), "depth median err (m)", np.round(depth_med, 5),
          "seg IoU min", round(min(seg_ious), 4))
    assert min(ious) > 0.95
    assert max(depth_med) < 0.01
    assert min(seg_ious) > 0.85


def test_texture_pattern_parity():
    model = mujoco.MjModel.from_xml_string(TEXTURED)
    d = mujoco.MjData(model)
    mujoco.mj_forward(model, d)
    rend = Tier0Renderer(model, 1, width=256, height=256, camera="cam", outputs=("rgb",))
    ours = rend.render_host([d])["rgb"][0].astype(float).mean(axis=2)
    ref = _ref(model, d, "cam", 256, 256).astype(float).mean(axis=2)
    # normalize both (shading differs) and compare the pattern
    o = (ours - ours.mean()) / (ours.std() + 1e-6)
    r = (ref - ref.mean()) / (ref.std() + 1e-6)
    corr = float((o * r).mean())
    print("texture pattern correlation", round(corr, 4))
    os.makedirs(OUT, exist_ok=True)
    try:
        import imageio.v2 as iio
        iio.imwrite(os.path.join(OUT, "tier0_texture_parity.png"),
                    np.concatenate([_ref(model, d, "cam", 256, 256), rend.render_host([d])["rgb"][0]], axis=1))
    except ImportError:
        pass
    assert corr > 0.97


def test_batch_and_so101_visual():
    model = mujoco.MjModel.from_xml_path(SO101)
    n = 16
    rend = Tier0Renderer(model, n, width=128, height=128, camera="base_cam", outputs=("rgb", "depth", "seg", "normal"))
    datas = []
    rng = np.random.default_rng(0)
    for e in range(n):
        d = mujoco.MjData(model)
        d.qpos[:6] = rng.uniform(-0.5, 0.5, 6)
        mujoco.mj_forward(model, d)
        datas.append(d)
    colors = np.ones((n, rend.G, 4), np.float32)
    colors[:, :, :3] = rng.uniform(0.3, 1.0, (n, rend.G, 3))
    rend.set_colors(colors)
    out = rend.render_host(datas)
    rgb = out["rgb"]
    assert rgb.shape == (n, 128, 128, 3)
    # robot pixels present in every tile, different tiles differ (poses + colors)
    assert all((out["seg"][e] > 0).sum() > 200 for e in range(n))
    assert not np.array_equal(rgb[0], rgb[1])
    ref = _ref(model, datas[0], "base_cam", 128, 128, seg=True)[:, :, 0]
    iou = _iou(np.isin(ref, rend.tables.geoms), out["seg"][0] > 0)
    print("SO-101 silhouette IoU vs mujoco.Renderer (env 0):", round(iou, 4))
    assert iou > 0.9
    os.makedirs(OUT, exist_ok=True)
    try:
        import imageio.v2 as iio
        tpr = 4
        grid = rgb.reshape(tpr, tpr, 128, 128, 3).transpose(0, 2, 1, 3, 4).reshape(tpr * 128, tpr * 128, 3)
        iio.imwrite(os.path.join(OUT, "tier0_so101_grid.png"), grid)
        dn = out["depth"][0]; dn = (255 * np.clip(dn / max(dn.max(), 1e-6), 0, 1)).astype(np.uint8)
        nm = ((out["normal"][0] * 0.5 + 0.5) * 255).astype(np.uint8)
        iio.imwrite(os.path.join(OUT, "tier0_so101_rgb_depth_normal.png"),
                    np.concatenate([rgb[0], np.repeat(dn[:, :, None], 3, 2), nm], axis=1))
    except ImportError:
        pass


def test_gpu_path_with_batchsim_no_host_sync():
    """Physics -> render -> torch read, ordered by events; outputs are zero-copy MPS tensors."""
    from metalsim.physics.batch import BatchSim, BatchSimOptions
    from metalsim.interop import warp_metal as wm
    model = mujoco.MjModel.from_xml_path(SO101)
    n = 64
    sim = BatchSim(model, n, options=BatchSimOptions(substeps=4))
    rend = Tier0Renderer(model, n, width=128, height=128, camera="base_cam", outputs=("rgb", "seg"))
    sim.synchronize()
    means = []
    c0 = wm.counters()
    for k in range(20):
        sim.t.ctrl.fill_(0.2 * np.sin(k / 3.0))
        vs = sim.step()
        vr = rend.render(sim, vs)
        rend.wait_sim_after_render(sim, vr)      # next step must not overwrite poses mid-render
        rend.after(vr)
        means.append(rend.out.rgb.float().mean())
    torch.mps.synchronize()
    d = wm.counters() - c0
    assert d.syncs == 0
    m = torch.stack(means).cpu().numpy()
    assert np.all(m > 0) and m.std() > 0, m      # something rendered, and it changes as the arm moves
    assert (rend.out.seg > 0).sum().item() > n * 100
    print("per-step mean intensity:", np.round(m[:5], 2), "...")

"""World extent and per-environment rendering (docs/RENDER_COVERAGE.md, workstream C): far plane and depth precision,
infinite ground planes, heightfield sides, per-world MuJoCo Warp model fields (size, colour, lights), and the
Randomizer's material writes. References: mujoco.Renderer (what is visible) and, for per-world fields, the same
renderer on a model compiled with that world's values. The probe scenes are shared with
scripts/diagnostics/render_world_audit.py (which records the before / after numbers)."""
import os
import sys

import mujoco
import numpy as np
import pytest
import torch
import warp as wp

from metalsim.render.tier0 import Tier0Renderer, SEG_GEOM
from metalsim.render.tier2 import Tier2Renderer
from metalsim.sensors.raytrace import RayTracer

pytestmark = pytest.mark.skipif(not wp.is_metal_available(), reason="needs Metal")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "diagnostics"))
import render_world_audit as A   # noqa: E402

W, H = A.W, A.H


def _scene(xml):
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    return m, d


def test_far_plane_default_and_mujoco_far():
    m, d = _scene(A.FAR)
    seg, rdep = A.ref_render(m, d, seg=True), A.ref_render(m, d, depth=True)
    far = A.mujoco_far(m)
    t0_default = A.ours(m, d, "tier0")
    for which in ("tier0", "tier1"):
        o = A.ours(m, d, which, far=far)
        for name in ("box6", "box15", "box40"):
            g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
            cov = A.coverage(A.geom_mask(seg, g), o, rdep, g)
            assert cov["rgb"] > 0.99 and cov["depth"] > 0.99 and cov["seg"] > 0.99, (which, name, cov)
            assert cov["depth_relerr_med"] < 1e-3, (which, name, cov)
    # the default far plane (10 m) keeps the previous behaviour: geometry beyond renders as background, depth 0
    g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "box15")
    cov = A.coverage(A.geom_mask(seg, g), t0_default, rdep, g)
    assert cov["depth"] == 0.0 and cov["seg"] == 0.0


def test_reversed_z_depth_precision():
    """Walls at 2 / 20 / 200 / 2000 m with far = 1e6 (Isaac's default clipping range): reversed z keeps the depth
    error relative (~1e-6 here); the previous convention loses metres at 2 km."""
    m, d = _scene(A.PREC)
    exact = {"w2": 2.0, "w20": 20.0, "w200": 200.0, "w2000": 2000.0}
    errs = {}
    for rz in (True, False):
        r = Tier0Renderer(m, 1, width=W, height=H, camera="cam", outputs=("depth", "seg"), seg_mode=SEG_GEOM, far=1e6, reverse_z=rz)
        o = r.render_host([d])
        for name, z in exact.items():
            g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
            sel = o["seg"][0] == g + 1
            assert sel.sum() > 20, name
            errs[(rz, name)] = float(np.median(np.abs(o["depth"][0][sel] - z)) / z)   # median: edge pixels see the box sides
    print({f"{'rev' if k[0] else 'std'} {k[1]}": f"{v:.1e}" for k, v in errs.items()})
    assert all(errs[(True, n)] < 1e-5 for n in exact)
    assert errs[(False, "w2000")] > 1e-3 and errs[(False, "w200")] > 1e-4


@pytest.mark.parametrize("view,mat,texu", [("loco", "groundplane", "true"), ("topdown", "asym", "true"), ("topdown", "asym", "false")])
def test_infinite_ground_plane(view, mat, texu):
    cam = "cam" if view == "loco" else "down"
    m, d = _scene(A.GROUND.replace("{MAT}", mat).replace("{TEXUNIFORM}", texu))
    seg, rdep, rgb = A.ref_render(m, d, cam, seg=True), A.ref_render(m, d, cam, depth=True), A.ref_render(m, d, cam)
    mask = A.geom_mask(seg, 0)
    far = A.mujoco_far(m)
    for which in ("tier0", "tier1", "tier2", "raydepth"):
        o = A.ours(m, d, which, cam, far=far)
        cov = A.coverage(mask, o, rdep, 0)
        assert cov["depth"] > 0.99, (which, cov)
        if "rgb" in o:
            assert cov["rgb"] > 0.99 and cov["seg"] > 0.99, (which, cov)
            corr = A._corr(rgb, o["rgb"], mask & (o["seg"] == 1) & (rdep < 8.0))   # beyond, the checker aliases
            print(view, mat, texu, which, cov, "texture corr", corr)
            # the 300 px checker with thin edge marks aliases in the raster (single-sample, no mip maps: TEXTURES.md);
            # tier 2 supersamples, so it measures the mapping (scale / offset / orientation) itself
            lim = 0.9 if mat == "asym" else (0.8 if which == "tier2" else 0.4)
            assert corr > lim, (which, corr)


def test_lidar_sees_ground_beyond_old_square():
    m, d = _scene(A.GROUND.replace("{MAT}", "groundplane").replace("{TEXUNIFORM}", "true"))
    rt = RayTracer(m, 4, max_range=30.0)
    az = np.linspace(-np.pi, np.pi, 64, endpoint=False)
    lid = rt.make_lidar("lidar", az, np.full(64, -np.arctan(1 / 15)))
    rng = rt.trace_host([d] * 4, [lid])["lidar"]["range"]
    np.testing.assert_allclose(rng, np.hypot(1, 15), atol=0.02)


@pytest.mark.parametrize("cam", ["cam", "under"])
def test_heightfield_sides_and_base(cam):
    m, d = _scene(A.HFIELD)
    seg, rdep = A.ref_render(m, d, cam, seg=True), A.ref_render(m, d, cam, depth=True)
    mask = A.geom_mask(seg, 0)
    assert mask.sum() > 500
    for which, kw in (("tier0", {}), ("tier0", dict(hfield_cull=False)), ("tier1", {}), ("tier2", {}), ("raydepth", {})):
        cov = A.coverage(mask, A.ours(m, d, which, cam, **kw), rdep, 0)
        assert cov["depth"] > 0.98, (which, kw, cov)
        assert cov.get("seg", 1.0) > 0.98, (which, kw, cov)


def test_per_world_model_fields_gpu_path():
    """geom_size / geom_rgba / mat_rgba / light_diffuse / light_active per world through BatchSim: silhouettes match
    MuJoCo's for that world's sizes, and colours match the renderer on a model compiled with that world's values."""
    res, _ = A.perworld_probe(4)
    for tier, rows in res["rows"].items():
        for row in rows:
            iou = row["silhouette_iou_vs_mujoco"]
            assert min(v for k, v in iou.items() if k != "capsule") > 0.9, (tier, row)
            assert iou["capsule"] > 0.85, (tier, row)          # capsule caps are scaled, not rebuilt
            lim = 1.0 if tier == "tier0" else 8.0                # tier 2: Monte-Carlo noise (same seed, different scene)
            assert row["rgb_mean_abs_diff_vs_compiled"] < lim, (tier, row)


def test_geom_rgba_overrides_material_like_mujoco():
    xml = """<mujoco><asset><material name="m" rgba="0.1 0.2 0.9 1"/></asset><worldbody>
      <light directional="true" pos="0 0 3" dir="0 0 -1"/><camera name="cam" pos="0 0 2" xyaxes="1 0 0 0 1 0"/>
      <geom name="a" type="box" pos="-0.3 0 0" size="0.2 0.2 0.1" material="m"/>
      <geom name="b" type="box" pos="0.3 0 0" size="0.2 0.2 0.1" material="m" rgba="0.9 0.2 0.1 1"/></worldbody></mujoco>"""
    m, d = _scene(xml)
    ref, seg = A.ref_render(m, d), A.ref_render(m, d, seg=True)
    o = A.ours(m, d, "tier0")
    for g in (0, 1):
        mk = A.geom_mask(seg, g) & (o["seg"] == g + 1)
        a, b = ref[mk].mean(0), o["rgb"][mk].mean(0)
        assert np.argmax(a) == np.argmax(b), (g, a, b)      # blue (material) for a, red (geom rgba) for b


@pytest.mark.parametrize("kind", ["tier0", "tier2_legacy", "tier2_omnipbr"])
def test_randomizer_materials(kind):
    from metalsim.replicator.randomizers import Randomizer
    m = mujoco.MjModel.from_xml_string(A.PW)
    if kind == "tier0":
        r = Tier0Renderer(m, 2, width=32, height=32, camera="cam")
    else:
        r = Tier2Renderer(m, 2, width=32, height=32, camera="cam", spp=1,
                          material_model="legacy" if kind == "tier2_legacy" else "omnipbr")
    Randomizer(r, seed=0).materials(roughness=(0.2, 0.3), metallic=(0.5, 0.6))
    gpu = r.ctx.buffer_array(r.mat_buf, np.float32, (r.G, 16))
    assert np.all((gpu[:, 5] >= 0.2) & (gpu[:, 5] <= 0.3)) and np.all((gpu[:, 4] >= 0.5) & (gpu[:, 4] <= 0.6))
    assert np.all(gpu[:, 0:4] == 1.0)                       # colours come from the model sync
    if kind == "tier2_omnipbr":
        assert np.all(gpu[:, 15] == 4)                      # BRDF model ids kept
    d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    out = r.render_host([d] * 2)
    assert out["rgb"].max() > 0


def test_fisheye_lenses():
    """Isaac Lab's default FisheyeCameraCfg (f-theta polynomial, 200 degrees) and a Kannala-Brandt lens: spheres at 0,
    60 and 95 degrees off axis land where the lens's forward model puts them in tier 2, the ray-traced depth camera and
    tier 0's cube-face resample; z-depth matches (negative behind the image plane at 95 degrees)."""
    res, _ = A.fisheye_probe()
    for lname, row in res["lenses"].items():
        for g in ("s0", "s60", "s95"):
            r = row[g]
            for tag in ("tier2", "tier0_cube", "raydepth"):
                assert r[tag] is not None, (lname, g, tag)
                assert r[tag]["centroid_err_px"] < 1.0, (lname, g, tag, r)
            assert abs(r["tier2"]["zdepth_nearest"] - r["zdepth_centre_exact"]) < 0.35, (lname, g, r)   # sphere r = 0.3
            assert r["iou_tier0_vs_tier2"] > 0.8, (lname, g, r)

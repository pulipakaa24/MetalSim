"""Static-geometry appearance vs MuJoCo's renderer (docs/RENDER_COVERAGE.md, 2026-10 fixes): mesh normals with hard
edges, SDF geoms, texture mapping (mesh UVs, primitives, cube textures, skybox), transparency, more than 8 lights,
spot / point light shadows. Probe scenes are synthetic MJCF (scripts/diagnostics/render_coverage_audit.py)."""
import os
import sys

import mujoco
import numpy as np
import pytest
import warp as wp

pytestmark = pytest.mark.skipif(not wp.is_metal_available(), reason="needs Metal")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts", "diagnostics"))

import render_coverage_audit as rca   # noqa: E402

from metalsim.render import builtin_geoms as bg            # noqa: E402
from metalsim.render.scene_tables import build_scene_tables  # noqa: E402

W, H = rca.W, rca.H


def _mask(seg, g):
    return (seg[..., 0] == g) & (seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM))


def _fwd(xml):
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    return m, d


def test_tables_follow_mujoco_geometry():
    """CPU: builtin tessellations are closed and outward (CCW), mesh corners keep the face normal on hard edges."""
    for P, U, N in (bg.unit_box(), bg.unit_sphere(28, 16), bg.unit_cylinder_side(28), bg.unit_disk(1, 28),
                    bg.unit_half_sphere(-1, 28, 8)):
        assert (np.einsum("fc,fc->f", bg.face_normals(P), N.mean(1)) > 0).all()
        assert U.min() >= 0 and U.max() <= 1
    m, _ = _fwd(rca.MESHBOX)
    t = build_scene_tables(m, 1)
    k = t.geoms.index(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "mesh_box"))
    mesh = t.meshes[t.geom_mesh[k]]
    idx = t.indices[mesh["i_off"]:mesh["i_off"] + mesh["i_count"]]
    nrm = np.round(t.vertices[idx, 3:6], 3)
    assert len(np.unique(nrm, axis=0)) == 6            # one normal per box face: no averaging across the edges


def test_mesh_normals_hard_edges():
    """A box written as a mesh shades like MuJoCo's (three flat faces), not rounded."""
    m, d = _fwd(rca.MESHBOX)
    seg = rca.ref_render(m, d, seg=True)
    k = _mask(seg, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "mesh_box"))
    ref = rca.ref_render(m, d).astype(np.float32)
    for r in ("tier0", "tier2"):
        img = rca.ours(m, d, r)["rgb"].astype(np.float32)
        std = img.mean(-1)[k].std()
        legacy = rca.ours(m, d, r, appearance="legacy")["rgb"].astype(np.float32).mean(-1)[k].std()
        print(r, "luminance std", round(float(std), 1), "legacy", round(float(legacy), 1), "MuJoCo", round(float(ref.mean(-1)[k].std()), 1))
        assert std < 15 and std < 0.5 * legacy


def test_sdf_geom_drawn():
    m, d = _fwd(rca.GEOMS)
    seg = rca.ref_render(m, d, seg=True); rdep = rca.ref_render(m, d, depth=True)
    g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "sdf")
    k = _mask(seg, g)
    assert k.sum() > 50
    for r in ("tier0", "tier2"):
        o = rca.ours(m, d, r)
        cov = rca.coverage(k, o, rdep, g, o.get("geoms"))
        print(r, cov)
        assert cov["rgb"] > 0.95 and cov["depth"] > 0.95 and cov["seg_correct"] > 0.9 and cov["depth_err_med_m"] < 0.01


def test_texture_mapping_matches_mujoco(tmp_path):
    m, d = _fwd(rca.texture_xml(str(tmp_path)))
    ref = rca.ref_render(m, d).astype(np.float32)
    seg = rca.ref_render(m, d, seg=True)
    for r in ("tier0", "tier2"):
        img = rca.ours(m, d, r)["rgb"].astype(np.float32)
        names = ("tex_sphere", "tex_cylinder", "tex_capsule", "tex_box_2d", "tex_box_cube") + (("tex_mesh_uv",) if r == "tier0" else ())
        for name in names:
            k = _mask(seg, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name))
            c = np.corrcoef(ref.mean(-1)[k], img.mean(-1)[k])[0, 1]
            print(r, name, "corr", round(float(c), 3))
            assert c > 0.85, (r, name, c)
        k = _mask(seg, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "tex_mesh_uv"))   # image orientation on mesh UVs
        ys = np.nonzero(k.any(1))[0]
        top, bot = ys[:len(ys) // 4], ys[-(len(ys) // 4):]
        for rows, colour in ((top, "blue"), (bot, "red")):
            c_ = img[rows][k[rows]].mean(0)
            assert (c_[2] > c_[0]) == (colour == "blue"), (r, colour, c_)
        bgm = seg[..., 0] < 0                          # skybox gradient behind everything
        c = np.corrcoef(ref.mean(-1)[bgm], img.mean(-1)[bgm])[0, 1]
        print(r, "skybox corr", round(float(c), 3), "std", round(float(img.mean(-1)[bgm].std()), 1))
        assert c > 0.9 and img.mean(-1)[bgm].std() > 10


def test_transparency_blends_like_mujoco():
    """Red box behind 25 % blue glass: the colour change the glass makes over the box matches MuJoCo's (its blend:
    sorted, front-to-back then back-to-front); segmentation and depth of the glass surface as mujoco.Renderer's."""
    m, d = _fwd(rca.TRANSPARENT)
    ref = rca.ref_render(m, d).astype(np.float32)
    m2, d2 = _fwd(rca.TRANSPARENT.replace('rgba="0.1 0.1 0.95 0.25"', 'rgba="0.1 0.1 0.95 0"'))
    back, glass = (mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n) for n in ("back", "glass"))
    k = _mask(rca.ref_render(m2, d2, seg=True), back)
    ref_delta = ref[k].mean(0) - rca.ref_render(m2, d2).astype(np.float32)[k].mean(0)
    ref_depth = np.median(rca.ref_render(m, d, depth=True)[k])
    for r in ("tier0", "tier1", "tier2"):
        o = rca.ours(m, d, r)
        delta = o["rgb"].astype(np.float32)[k].mean(0) - rca.ours(m2, d2, r)["rgb"].astype(np.float32)[k].mean(0)
        print(r, "glass delta rgb", delta.round(), "MuJoCo", ref_delta.round())
        tol = 12 if r != "tier2" else 30                  # tier 2: presence (Isaac), not MuJoCo's double blend
        assert np.abs(delta - ref_delta).max() < tol, (r, delta, ref_delta)
        seg = o["seg"] if r != "tier2" else np.concatenate([[0], np.asarray(o["geoms"]) + 1])[o["seg"]]
        assert (seg[k] == glass + 1).mean() > 0.95
        assert abs(np.median(o["depth"][k]) - ref_depth) < 0.01
        ob = rca.ours(m, d, r, transparent_ids="behind")
        sb = ob["seg"] if r != "tier2" else np.concatenate([[0], np.asarray(ob["geoms"]) + 1])[ob["seg"]]
        assert (sb[k] == back + 1).mean() > 0.95 and np.median(ob["depth"][k]) > ref_depth + 0.3


def test_more_than_eight_lights():
    m, d = _fwd(rca.LIGHTS10)
    fy = H / (2 * np.tan(np.deg2rad(m.cam_fovy[0]) / 2))

    def patches(img):
        lum = img.astype(np.float32).mean(-1)
        return np.array([lum[H // 2 - 2:H // 2 + 3, max(int(round(W / 2 + (-2.25 + 0.5 * i) / 6.0 * fy)) - 2, 0):
                             int(round(W / 2 + (-2.25 + 0.5 * i) / 6.0 * fy)) + 3].mean() for i in range(10)])
    allp = patches(rca.ours(m, d, "tier0")["rgb"])
    cap8 = patches(rca.ours(m, d, "tier0", max_lights=8)["rgb"])
    print("all", allp.round(1), "max_lights=8", cap8.round(1))
    assert allp[8:].min() > 0.8 * allp[:8].min()
    assert cap8[8:].max() < 2.0 and abs(cap8[:8] - allp[:8]).max() < 2.0


@pytest.mark.parametrize("kind", ["spot", "point"])
def test_spot_and_point_shadows(kind):
    light = {"spot": '<light pos="0 0 1.5" dir="0 0 -1" cutoff="50" exponent="1" castshadow="true" diffuse="0.9 0.9 0.9"/>',
             "point": '<light type="point" pos="0 0 1.5" castshadow="true" diffuse="0.9 0.9 0.9"/>'}[kind]
    m, d = _fwd(rca.SHADOW.replace("{LIGHT}", light))
    seg = rca.ref_render(m, d, seg=True)
    floor = _mask(seg, 0)
    dep = rca.ref_render(m, d, depth=True)
    fy = H / (2 * np.tan(np.deg2rad(m.cam_fovy[0]) / 2))
    yy, xx = np.mgrid[0:H, 0:W]
    dirs = np.stack([(xx + 0.5 - W / 2) / fy, -(yy + 0.5 - H / 2) / fy, -np.ones_like(xx, float)], -1)
    P = d.cam_xpos[0] + (dirs * dep[..., None]) @ d.cam_xmat[0].reshape(3, 3).T
    inside = floor & (np.abs(P[..., 0]) < 0.1) & (np.abs(P[..., 1]) < 0.1)
    ring = floor & (np.hypot(P[..., 0], P[..., 1]) > 0.3) & (np.hypot(P[..., 0], P[..., 1]) < 0.4)
    for r in ("tier0", "tier1"):
        lum = rca.ours(m, d, r)["rgb"].astype(np.float32).mean(-1)
        ratio = lum[inside].mean() / lum[ring].mean()
        print(kind, r, "shadow ratio", round(float(ratio), 3))
        assert ratio < 0.25, (kind, r, ratio)
        assert lum[ring].mean() > 30                       # no shadow acne on the lit floor

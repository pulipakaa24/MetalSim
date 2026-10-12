"""Non-geom visuals in every renderer and ray-traced sensor (metalsim.render.extras / dynamic_mesh), measured against
MuJoCo's own renderer as docs/RENDER_COVERAGE.md does: for each object MuJoCo draws (its segmentation mask is the
reference silhouette), the fraction of the silhouette that MetalSim covers in RGB (pixel differs from the clear
colour), depth (> 0) and segmentation (the object's own id), >= 0.99, with the median depth error.

Objects: sites, 1D / 2D / 3D flexes (cable, cloth, soft volume), skin, spatial tendons (straight, wrapping a geom,
across a pulley), deformed states (stepped MuJoCo, MuJoCo Warp on the GPU path, XPBD cloth, PhysX-port cloth), and
the default-off guarantee: scenes without these objects render byte-identically with and without the extras path.
GPU tests: run through scripts/gpu_run.sh (check).

The camera keeps the objects away from the image border: in the top few rows MetalSim and MuJoCo disagree by about a
pixel for every object, geoms included (a horizontal 2 cm capsule geom in rows 1-4 of a 160x120 image is covered 0.54 by
tier 0 and 0.77 by tier 2, the same capsule mid-image 1.0), a camera-model difference that is not part of this path."""
import os

import mujoco
import numpy as np
import pytest
import warp as wp

from metalsim.render.extras import DeformableMesh
from metalsim.render.scene_tables import build_scene_tables
from metalsim.render.tier0 import Tier0Renderer, SEG_GEOM, SEG_BODY
from metalsim.render.tier2 import Tier2Renderer
from metalsim.sensors.raytrace import RayTracer

needs_metal = pytest.mark.skipif(not wp.is_metal_available(), reason="needs Metal")
W, H = 160, 120
CAM = '<camera name="cam" pos="0 -2.2 0.95" xyaxes="1 0 0 0 0 1"/>'   # objects mid-image (see test notes)
THRESH = 0.99

EXTRAS = f"""
<mujoco>
  <worldbody>
    <light directional="true" pos="0 0 3" dir="0 0.3 -1"/>
    {CAM}
    <geom name="anchor" type="box" pos="2.5 3 2.5" size="0.01 0.01 0.01" rgba="1 1 1 1"/>
    <site name="site_sphere" type="sphere" pos="-1.0 0 1.0" size="0.1" rgba="0.9 0.3 0.3 1"/>
    <site name="site_box" type="box" pos="-0.7 0 1.0" size="0.08 0.08 0.08" rgba="0.3 0.9 0.3 1" euler="0 30 20"/>
    <site name="site_hidden" type="sphere" pos="-0.85 0 0.7" size="0.08" group="4" rgba="1 1 1 1"/>
    <body name="t1" pos="-0.45 0 1.2"><site name="ta" size="0.005" rgba="0 0 0 0"/></body>
    <body name="t2" pos="-0.25 0 0.75"><site name="tb" size="0.005" rgba="0 0 0 0"/></body>
    <body name="bone" pos="1.0 0 1.0" euler="10 20 30"/>
    <flexcomp name="cable" type="grid" count="8 1 1" spacing="0.05 0.05 0.05" pos="0.0 0 1.02" dim="1" radius="0.02"
              rgba="0.95 0.6 0.2 1"><edge equality="true"/></flexcomp>
    <flexcomp name="cloth" type="grid" count="7 7 1" spacing="0.04 0.04 0.04" pos="0.15 0 0.75" euler="90 0 0" dim="2"
              radius="0.005" rgba="0.2 0.5 0.9 1"><edge equality="true"/></flexcomp>
    <flexcomp name="soft" type="grid" count="3 3 3" spacing="0.07 0.07 0.07" pos="0.55 0 0.95" dim="3" radius="0.005"
              rgba="0.6 0.9 0.3 1"><edge equality="true"/></flexcomp>
  </worldbody>
  <tendon><spatial name="tendon" width="0.03" rgba="0.9 0.9 0.2 1"><site site="ta"/><site site="tb"/></spatial></tendon>
  <deformable>
    <skin name="skin" rgba="0.8 0.4 0.9 1" inflate="0.01"
      vertex="-0.1 -0.1 -0.1  0.1 -0.1 -0.1  0.1 0.1 -0.1  -0.1 0.1 -0.1  -0.1 -0.1 0.1  0.1 -0.1 0.1  0.1 0.1 0.1  -0.1 0.1 0.1"
      face="0 2 1 0 3 2 4 5 6 4 6 7 0 1 5 0 5 4 1 2 6 1 6 5 2 3 7 2 7 6 3 0 4 3 4 7">
      <bone body="bone" bindpos="0 0 0" bindquat="1 0 0 0" vertid="0 1 2 3 4 5 6 7" vertweight="1 1 1 1 1 1 1 1"/>
    </skin>
  </deformable>
</mujoco>"""

# a tendon wrapping a cylinder and a sphere, and a pulley (two branches); a flexed cloth on a sphere (stepped)
WRAP = f"""
<mujoco>
  <option timestep="0.005"/>
  <worldbody>
    <light directional="true" pos="0 0 3" dir="0 0.3 -1"/>
    {CAM}
    <geom name="cyl" type="cylinder" pos="-0.6 0 1.0" size="0.08 0.15" euler="90 0 0" rgba="0.5 0.5 0.5 0.0"/>
    <site name="s0" pos="-0.9 0 0.95"/><site name="s1" pos="-0.3 0 0.95"/>
    <site name="cside" pos="-0.6 0 1.2"/>
    <geom name="ball" type="sphere" pos="0.3 0 1.0" size="0.1" rgba="0.5 0.5 0.5 0.0"/>
    <site name="s2" pos="0.05 0 0.95"/><site name="s3" pos="0.55 0 0.95"/>
    <site name="p0" pos="0.8 0 1.2"/><site name="p1" pos="0.7 0 0.8"/><site name="p2" pos="0.95 0 0.8"/>
  </worldbody>
  <tendon>
    <spatial name="wrap_cyl" width="0.02" rgba="0.9 0.2 0.2 1"><site site="s0"/><geom geom="cyl" sidesite="cside"/><site site="s1"/></spatial>
    <spatial name="wrap_ball" width="0.02" rgba="0.2 0.9 0.2 1"><site site="s2"/><geom geom="ball"/><site site="s3"/></spatial>
    <spatial name="pulley" width="0.015" rgba="0.2 0.2 0.9 1"><site site="p0"/><site site="p1"/><pulley divisor="2"/><site site="p0"/><site site="p2"/></spatial>
  </tendon>
</mujoco>"""

DRAPE = f"""
<mujoco>
  <option timestep="0.004" solver="CG" iterations="30"/>
  <worldbody>
    <light directional="true" pos="0 0 3" dir="0 0.3 -1"/>
    <camera name="cam" pos="0 -1.4 1.0" xyaxes="1 0 0 0 0.5 0.866"/>
    <geom name="ball" type="sphere" pos="0 0 0.5" size="0.18" rgba="0.6 0.6 0.6 1"/>
    <flexcomp name="cloth" type="grid" count="12 12 1" spacing="0.05 0.05 0.05" pos="0 0 0.8" dim="2" radius="0.006"
              mass="0.3" rgba="0.2 0.5 0.9 1"><edge equality="true"/><contact condim="3" selfcollide="none"/></flexcomp>
    <flexcomp name="rope" type="grid" count="14 1 1" spacing="0.04 0.04 0.04" pos="-0.25 0.15 0.9" dim="1" radius="0.012"
              mass="0.05" rgba="0.9 0.6 0.1 1"><edge equality="true"/></flexcomp>
  </worldbody>
</mujoco>"""


def _ref(m, d, seg=False, depth=False):
    r = mujoco.Renderer(m, H, W)
    if seg:
        r.enable_segmentation_rendering()
    if depth:
        r.enable_depth_rendering()
    r.update_scene(d, camera="cam")
    out = r.render().copy()
    r.close()
    return out


def _render(m, datas, which, **kw):
    """Outputs per env: rgb, depth, seg (geom mode ids: geoms g + 1, extras ngeom + 1 + j)."""
    src = kw.pop("sources", None)
    if which in ("tier0", "tier1"):
        r = Tier0Renderer(m, len(datas), width=W, height=H, camera="cam", outputs=("rgb", "depth", "seg"), seg_mode=SEG_GEOM,
                          tier=0 if which == "tier0" else 1, **kw)
        o = r.render_host(datas, sources=src)
        return r, o
    if which == "tier2":
        r = Tier2Renderer(m, len(datas), width=W, height=H, camera="cam", spp=16, max_bounces=2, center_sample=True, **kw)
        o = r.render_host(datas, passes=2, sources=src)
        o["seg"] = np.concatenate([[0], r.tables.slot_seg_ids()])[o["seg"]]
        return r, o
    if "visual" not in kw:      # the sensor default leaves sites out (mj_ray skips them); compare everything MuJoCo draws
        assert not any(e["kind"] == "site" for e in RayTracer(m, 1, max_range=1.0).tables.extras)
        kw["visual"] = mujoco.MjvOption()
    rt = RayTracer(m, len(datas), max_range=100.0, **kw)
    cam = rt.make_depth_camera("cam", W, H)
    return rt, {"depth": rt.trace_host(datas, [cam], sources=src)["ray_depth"]}


def _coverage(m, d, out, env, tables, ref_seg=None, ref_depth=None, objects=None):
    """{label: {px, rgb, depth, seg, derr}} for each extra object MuJoCo draws."""
    ref_seg = _ref(m, d, seg=True) if ref_seg is None else ref_seg
    ref_depth = _ref(m, d, depth=True) if ref_depth is None else ref_depth
    res = {}
    for ex in tables.extras:
        if objects is not None and ex["name"] not in objects:
            continue
        mask = (ref_seg[..., 0] == ex["objid"]) & (ref_seg[..., 1] == int(ex["objtype"]))
        if ex["objtype"] < 0:      # user mesh: compared through a MuJoCo flex of the same name
            mask = (ref_seg[..., 0] == mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_FLEX, ex["name"])) & (ref_seg[..., 1] == int(mujoco.mjtObj.mjOBJ_FLEX))
        row = {"px": int(mask.sum())}
        if mask.any():
            if "rgb" in out:
                row["rgb"] = float((out["rgb"][env][mask].max(-1) > 3).mean())
            dv = out["depth"][env][mask]
            row["depth"] = float((dv > 0).mean())
            ok = dv > 0
            row["derr"] = float(np.median(np.abs(dv[ok] - ref_depth[mask][ok]))) if ok.any() else np.inf
            if "seg" in out:
                row["seg"] = float((out["seg"][env][mask] == ex["seg_id"]).mean())
        res[ex["name"]] = row
    return res


def _assert_covered(cov, thresh=THRESH, derr=0.01, min_px=20):
    bad = {}
    for k, r in cov.items():
        if r["px"] < min_px:
            continue
        for q in ("rgb", "depth", "seg"):
            if q in r and r[q] < thresh:
                bad[f"{k}.{q}"] = round(r[q], 3)
        if r.get("derr", 0) > derr:
            bad[f"{k}.derr"] = r["derr"]
    assert not bad, (bad, cov)


def _forward(xml, nstep=0):
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    for _ in range(nstep):
        mujoco.mj_step(m, d)
    mujoco.mj_forward(m, d)
    return m, d


# ------------------------------------------------------------------------------------------------------------------
# tables (CPU)

def test_extras_follow_mujoco_visibility():
    m, _ = _forward(EXTRAS)
    t = build_scene_tables(m, 2, visual=None)
    names = [e["name"] for e in t.extras]
    assert names == ["site_sphere", "site_box", "cable", "cloth", "soft", "skin", "tendon"]   # group-4 and alpha-0 sites hidden
    assert [e["seg_id"] for e in t.extras] == list(range(m.ngeom + 1, m.ngeom + 1 + len(names)))
    assert t.G == len(t.geoms) + len(names) and t.n_pose == m.ngeom + 2
    assert len(t.inst_table) == 2 * (len(t.geoms) + 2)          # geoms + 2 site slots, per env
    # MuJoCo's default option draws the cloth / soft as skins, the cable as edge cylinders
    kinds = {d["slot"]: (d["kind"], d.get("faces"), d.get("lines")) for d in t.dyn}
    assert sorted(kinds.values(), key=str) == sorted([("flex", False, True), ("flex", True, False), ("flex", True, False),
                                                      ("skin", None, None), ("tendon", None, None)], key=str)
    assert build_scene_tables(m, 2).extras == []                 # build_scene_tables default: geoms only
    vopt = mujoco.MjvOption()
    vopt.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = 0
    vopt.sitegroup[:] = 0
    vopt.flexgroup[:] = 0
    t2 = build_scene_tables(m, 2, visual=vopt)
    assert [e["name"] for e in t2.extras] == ["skin"]


def _host_sources(m, datas):
    n, nw = len(datas), 2 * int(m.nwrap)
    return {"flexvert_xpos": np.stack([d.flexvert_xpos for d in datas]),
            "xpos": np.stack([d.xpos for d in datas]), "xquat": np.stack([d.xquat for d in datas]),
            "ten_wrapadr": np.stack([d.ten_wrapadr for d in datas]), "ten_wrapnum": np.stack([d.ten_wrapnum for d in datas]),
            "wrap_obj": np.stack([np.asarray(d.wrap_obj).reshape(-1)[:nw] for d in datas]),
            "wrap_xpos": np.stack([np.asarray(d.wrap_xpos).reshape(-1)[:3 * nw].reshape(-1, 3) for d in datas])}


def _canon(tri_pts):
    """(F, 3, k) per-corner rows -> canonical sorted (F*3... ) array for set comparison."""
    t = np.round(np.asarray(tri_pts, np.float64), 5)
    t = np.stack([x[np.lexsort(x.T[::-1])] for x in t])           # corners sorted within each triangle
    flat = t.reshape(len(t), -1)
    return flat[np.lexsort(flat.T[::-1])]


@pytest.mark.parametrize("xml,nstep", [(EXTRAS, 0), (WRAP, 0), (DRAPE, 120)])
def test_generated_geometry_matches_mjv_update_scene(xml, nstep):
    """CPU: the generation tables, run by the NumPy reference of the kernels (dynamic_mesh.emulate), give MuJoCo's own
    flex faces / normals (mjv_updateActiveFlex), skin vertices / normals (mjv_updateActiveSkin), tendon capsules
    (addSpatialTendonGeoms + mjv_connector) and flex edge cylinders (render_gl3: quatZ2Vec, open cylinder)."""
    from metalsim.render.dynamic_mesh import _Plan, emulate
    m, d = _forward(xml, nstep)
    t = build_scene_tables(m, 1, visual=None)
    plan = _Plan(m, t)
    plan.finalize(1)
    plan.validate(1)
    rec = emulate(plan, 1, _host_sources(m, [d]))
    scn = mujoco.MjvScene(m, maxgeom=5000)
    mujoco.mjv_updateScene(m, d, mujoco.MjvOption(), mujoco.MjvPerturb(), mujoco.MjvCamera(), mujoco.mjtCatBit.mjCAT_ALL, scn)
    by_slot = {ob["slot"]: ob for ob in plan.objects}
    tm_cyl, tm_cap = plan.cyl_t[0].astype(np.float64), plan.cap_t[0].astype(np.float64)
    checked = set()
    for dd in t.dyn:
        ob = by_slot[dd["slot"]]
        R = rec[ob["obj_base"]: ob["obj_base"] + ob["nvert"]]
        if dd["kind"] == "flex" and dd["faces"]:
            f = dd["id"]
            a, u = scn.flexfaceadr[f], scn.flexfaceused[f]
            mj = np.concatenate([scn.flexface[9 * a: 9 * (a + u)].reshape(-1, 3, 3), scn.flexnormal[9 * a: 9 * (a + u)].reshape(-1, 3, 3)], -1)
            ours = R[ob["tris"]][..., :6]
            assert len(ours) == len(mj)
            np.testing.assert_allclose(_canon(ours), _canon(mj), atol=2e-5)
            checked.add("flex_faces")
        elif dd["kind"] == "flex":
            f = dd["id"]
            va, r = m.flex_vertadr[f], float(m.flex_radius[f])
            exp = []
            for (i0, i1) in m.flex_edge[m.flex_edgeadr[f]: m.flex_edgeadr[f] + m.flex_edgenum[f]]:
                p0, p1 = d.flexvert_xpos[va + i0], d.flexvert_xpos[va + i1]
                q = np.zeros(4); mujoco.mju_quatZ2Vec(q, p1 - p0)
                M = np.zeros(9); mujoco.mju_quat2Mat(M, q); M = M.reshape(3, 3)
                h = 0.5 * np.linalg.norm(p1 - p0)
                exp.append(0.5 * (p0 + p1) + (np.stack([r * tm_cyl[:, 0], r * tm_cyl[:, 1], h * tm_cyl[:, 3]], 1) @ M.T))
            np.testing.assert_allclose(R[:, :3], np.concatenate(exp), atol=2e-6)
            checked.add("flex_edges")
        elif dd["kind"] == "skin":
            i = dd["id"]
            a, nv = m.skin_vertadr[i], m.skin_vertnum[i]
            np.testing.assert_allclose(R[:, :3], scn.skinvert[3 * a: 3 * (a + nv)].reshape(-1, 3), atol=2e-6)
            np.testing.assert_allclose(R[:, 3:6], scn.skinnormal[3 * a: 3 * (a + nv)].reshape(-1, 3), atol=2e-6)
            checked.add("skin")
        elif dd["kind"] == "tendon":
            geoms = [scn.geoms[k] for k in range(scn.ngeom) if scn.geoms[k].objtype == mujoco.mjtObj.mjOBJ_TENDON and scn.geoms[k].objid == dd["id"]]
            nt = len(tm_cap)
            segs = R.reshape(-1, nt, 8)
            valid = [s for s in segs if np.ptp(s[:, :3], 0).max() > 0]
            assert len(valid) == len(geoms) > 0
            for s, g in zip(valid, geoms):
                M = np.asarray(g.mat, np.float64).reshape(3, 3)
                w, h = float(g.size[0]), float(g.size[2])
                loc = np.stack([w * tm_cap[:, 0], w * tm_cap[:, 1], w * tm_cap[:, 2] + h * tm_cap[:, 3]], 1)
                np.testing.assert_allclose(s[:, :3], np.asarray(g.pos, np.float64) + loc @ M.T, atol=2e-5)
            checked.add("tendon")
    assert checked, "nothing checked"


def test_flexface_mode_and_user_mesh_match_mujoco():
    """CPU: mjVIS_FLEXFACE without the skin flag (flat element faces offset by the radius, both windings; 3D: the
    elements of the selected layer) and a DeformableMesh with a flex's triangles and radius, against mjv_updateScene."""
    from metalsim.render.dynamic_mesh import _Plan, emulate
    m, d = _forward(DRAPE, 80)
    vopt = mujoco.MjvOption()
    vopt.flags[mujoco.mjtVisFlag.mjVIS_FLEXSKIN] = 0
    vopt.flags[mujoco.mjtVisFlag.mjVIS_FLEXFACE] = 1
    scn = mujoco.MjvScene(m, maxgeom=5000)
    mujoco.mjv_updateScene(m, d, vopt, mujoco.MjvPerturb(), mujoco.MjvCamera(), mujoco.mjtCatBit.mjCAT_ALL, scn)
    f = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_FLEX, "cloth")
    a, u = scn.flexfaceadr[f], scn.flexfaceused[f]
    mj = np.concatenate([scn.flexface[9 * a: 9 * (a + u)].reshape(-1, 3, 3), scn.flexnormal[9 * a: 9 * (a + u)].reshape(-1, 3, 3)], -1)
    t = build_scene_tables(m, 1, visual=vopt)
    plan = _Plan(m, t); plan.finalize(1)
    rec = emulate(plan, 1, _host_sources(m, [d]))
    ob = next(o for o in plan.objects if o["slot"] == next(dd["slot"] for dd in t.dyn if dd["kind"] == "flex" and dd["id"] == f))
    R = rec[ob["obj_base"]: ob["obj_base"] + ob["nvert"]]
    faces = R[ob["tris"][:len(mj)]][..., :6]           # then the edge cylinders (mjVIS_FLEXEDGE, drawn with faces)
    np.testing.assert_allclose(_canon(faces), _canon(mj), atol=2e-5)
    # the same cloth as a user mesh (radius > 0: a MuJoCo 2D skin), its vertices given as an array
    va, nv = m.flex_vertadr[f], m.flex_vertnum[f]
    tris = m.flex_elem[m.flex_elemdataadr[f]: m.flex_elemdataadr[f] + 3 * m.flex_elemnum[f]].reshape(-1, 3)
    dm = DeformableMesh("cloth_mesh", tris, nv, radius=float(m.flex_radius[f]))
    vno = mujoco.MjvOption(); vno.flexgroup[:] = 0
    t2 = build_scene_tables(m, 1, visual=vno, deformables=[dm])
    plan2 = _Plan(m, t2); plan2.finalize(1)
    rec2 = emulate(plan2, 1, {"cloth_mesh": d.flexvert_xpos[None, va:va + nv]})
    ob2 = plan2.objects[0]
    scn2 = mujoco.MjvScene(m, maxgeom=5000)
    mujoco.mjv_updateScene(m, d, mujoco.MjvOption(), mujoco.MjvPerturb(), mujoco.MjvCamera(), mujoco.mjtCatBit.mjCAT_ALL, scn2)
    a, u = scn2.flexfaceadr[f], scn2.flexfaceused[f]
    mj2 = np.concatenate([scn2.flexface[9 * a: 9 * (a + u)].reshape(-1, 3, 3), scn2.flexnormal[9 * a: 9 * (a + u)].reshape(-1, 3, 3)], -1)
    ours2 = rec2[ob2["tris"]][..., :6]
    assert len(ours2) == len(mj2)
    np.testing.assert_allclose(_canon(ours2[..., :3]), _canon(mj2[..., :3]), atol=2e-5)     # same surface
    np.testing.assert_allclose(_canon(ours2), _canon(mj2), atol=2e-5)                       # and the same normals


@needs_metal
@pytest.mark.parametrize("xml", [EXTRAS, WRAP, DRAPE])
def test_gpu_generation_matches_reference(xml):
    """The Metal kernels write the records the NumPy reference computes, per env, for a raster-only renderer and for
    a ray tracer (positions offset by the env grid)."""
    from metalsim.render.dynamic_mesh import emulate
    m, _ = _forward(xml)
    datas = []
    for n in (0, 40, 120):
        dk = mujoco.MjData(m); mujoco.mj_forward(m, dk)
        for _ in range(n):
            mujoco.mj_step(m, dk)
        mujoco.mj_forward(m, dk)
        datas.append(dk)
    r = Tier0Renderer(m, 3, width=32, height=24, camera="cam", outputs=("rgb",))
    r.render_host(datas)
    ref = emulate(r.dyn.plan, 3, _host_sources(m, datas))
    np.testing.assert_allclose(r.dyn.records(), ref, atol=5e-5)
    rt = RayTracer(m, 3, max_range=5.0)
    cam = rt.make_depth_camera("cam", 16, 12)
    rt.trace_host(datas, [cam])
    got = rt.dyn.records().astype(np.float64).copy()
    for ob in rt.dyn.objects:
        for e in range(3):
            off = np.array([(e % rt.tpr) * rt.stride, (e // rt.tpr) * rt.stride, 0.0])
            a = ob["obj_base"] + e * ob["nvert"]
            got[a:a + ob["nvert"], :3] -= off
    np.testing.assert_allclose(got, ref, atol=2e-4)


# ------------------------------------------------------------------------------------------------------------------
# coverage against mujoco.Renderer

@needs_metal
@pytest.mark.parametrize("which", ["tier0", "tier1", "tier2", "raydepth"])
def test_extras_coverage(which):
    m, d = _forward(EXTRAS)
    r, o = _render(m, [d], which)
    cov = _coverage(m, d, o, 0, r.tables)
    drawn = {k for k, v in cov.items() if v["px"] >= 20}
    assert drawn >= {"site_sphere", "site_box", "cable", "cloth", "soft", "skin", "tendon"}, cov
    _assert_covered(cov)


@needs_metal
@pytest.mark.parametrize("which", ["tier0", "tier1", "tier2", "raydepth"])
def test_tendon_wrapping_and_pulley(which):
    m, d = _forward(WRAP)
    assert d.ten_wrapnum[0] > 2 and d.ten_wrapnum[1] > 2 and (np.asarray(d.wrap_obj).reshape(-1) == -2).any()
    r, o = _render(m, [d], which)
    cov = _coverage(m, d, o, 0, r.tables)
    assert all(cov[n]["px"] > 50 for n in ("wrap_cyl", "wrap_ball", "pulley")), cov
    # These tendons are 1.5-2 px wide. A pixel-centre test of the generated triangles on the CPU covers exactly MuJoCo's
    # 120 pulley pixels, but 6 of them (4 / 117 for wrap_cyl) lie less than 1/256 px inside any triangle: near the
    # silhouette only edge-on slivers cover them, below the raster's sub-pixel precision. The ray-traced paths cover them
    # (1.0); the raster gets 0.95. The geometry is right; the raster threshold for these thin tubes is 0.95.
    _assert_covered(cov, thresh=0.95 if which in ("tier0", "tier1") else THRESH)


@needs_metal
@pytest.mark.parametrize("which", ["tier0", "tier1", "tier2", "raydepth"])
def test_flex_deformed_and_posed_skin(which):
    """A cloth draped on a sphere and a rope after 150 steps of MuJoCo C, and the skin's bone moved and turned."""
    m, d = _forward(DRAPE, nstep=150)
    v = d.flexvert_xpos[: m.flex_vertnum[0]]
    assert v[:, 2].max() - v[:, 2].min() > 0.1          # actually draped
    r, o = _render(m, [d], which)
    _assert_covered(_coverage(m, d, o, 0, r.tables))
    m2, d2 = _forward(EXTRAS)
    b = mujoco.mj_name2id(m2, mujoco.mjtObj.mjOBJ_BODY, "bone")
    m2.body_pos[b] = (0.95, 0.1, 1.05)
    m2.body_quat[b] = (0.8, 0.3, 0.4, 0.33)
    m2.body_quat[b] /= np.linalg.norm(m2.body_quat[b])
    mujoco.mj_forward(m2, d2)
    r, o = _render(m2, [d2], which)
    _assert_covered(_coverage(m2, d2, o, 0, r.tables, objects={"skin"}))


@needs_metal
def test_batched_envs_differ():
    """Three envs with different flex / tendon / skin states in one render: each env against its own reference."""
    m, d0 = _forward(DRAPE)
    datas = []
    for k, n in enumerate((0, 60, 150)):
        dk = mujoco.MjData(m)
        mujoco.mj_forward(m, dk)
        for _ in range(n):
            mujoco.mj_step(m, dk)
        mujoco.mj_forward(m, dk)
        datas.append(dk)
    for which in ("tier0", "tier1", "tier2"):
        r, o = _render(m, datas, which)
        for e, dk in enumerate(datas):
            _assert_covered(_coverage(m, dk, o, e, r.tables))


# ------------------------------------------------------------------------------------------------------------------
# GPU path: vertices straight from the simulators' buffers

def _drape_model():
    return mujoco.MjModel.from_xml_string(DRAPE)


@needs_metal
def test_gpu_path_mujoco_warp_flex():
    """N worlds of MuJoCo Warp's flex simulation (DeformableSim) rendered from its buffers on the GPU path: every env
    against MuJoCo's renderer given the same flex vertex positions."""
    from metalsim.physics.deformable import DeformableSim
    m = _drape_model()
    n = 4
    sim = DeformableSim(m, n)
    sim.randomize(seed=3, xy=0.08, z=0.05)
    for _ in range(40):
        sim.step()
    sim.synchronize()
    fx = sim.d.flexvert_xpos.numpy()
    for which in ("tier0", "tier1", "tier2"):
        if which == "tier2":
            r = Tier2Renderer(m, n, width=W, height=H, camera="cam", spp=16, max_bounces=2, center_sample=True)
            r.render(sim)
        else:
            r = Tier0Renderer(m, n, width=W, height=H, camera="cam", outputs=("rgb", "depth", "seg"), seg_mode=SEG_GEOM,
                              tier=0 if which == "tier0" else 1)
            r.render(sim)
        r.ctx.synchronize()
        o = {k: getattr(r.out, k).cpu().numpy() for k in ("rgb", "depth", "seg")}
        if which == "tier2":
            o["seg"] = np.concatenate([[0], r.tables.slot_seg_ids()])[o["seg"]]
        for e in range(n):
            dd = mujoco.MjData(m)
            mujoco.mj_forward(m, dd)
            dd.flexvert_xpos[:] = fx[e]
            _assert_covered(_coverage(m, dd, o, e, r.tables))


@needs_metal
def test_gpu_path_xpbd_and_physx_cloth():
    """XPBD cloth (the flexes of the model, vertices from XPBDSim.x) and the PhysX-port cloth (a DeformableMesh with the
    cloth's triangles and radius, vertices from PhysXClothSim.x), rendered on the GPU path; reference: MuJoCo's renderer
    drawing the same flex at those vertex positions."""
    from metalsim.physics.deformable import XPBDSim
    from metalsim.physics import physx_cloth as pc
    m = _drape_model()
    n = 3
    xp = XPBDSim(m, n)
    xp.randomize(seed=1)
    for _ in range(60):
        xp.step()
    xp.synchronize()
    xs = xp.x.numpy()
    r = Tier0Renderer(m, n, width=W, height=H, camera="cam", outputs=("rgb", "depth", "seg"), seg_mode=SEG_GEOM, tier=1)
    r.render(xp, sources={"flexvert_xpos": xp.x})
    r.ctx.synchronize()
    o = {k: getattr(r.out, k).cpu().numpy() for k in ("rgb", "depth", "seg")}
    for e in range(n):
        dd = mujoco.MjData(m); mujoco.mj_forward(m, dd); dd.flexvert_xpos[:] = xs[e]
        _assert_covered(_coverage(m, dd, o, e, r.tables))
    # PhysX-port cloth on the cloth flex's topology, drawn as a user mesh named after the flex
    f = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_FLEX, "cloth")
    mesh = pc.mesh_from_flex(m, f)
    cs = pc.PhysXClothSim(mesh, n, obstacles=[pc.plane(0.0)])
    cs.randomize(seed=2)
    for _ in range(30):
        cs.step()
    cs.synchronize()
    cx = cs.x.numpy()
    vopt = mujoco.MjvOption(); vopt.flexgroup[:] = 0           # the model's own flexes hidden: only the user mesh
    dm = DeformableMesh("cloth", mesh.tris, len(mesh.x), rgba=tuple(m.flex_rgba[f]), radius=float(m.flex_radius[f]))
    for which in ("tier0", "tier1", "tier2", "raydepth"):
        if which == "raydepth":
            rt = RayTracer(m, n, max_range=100.0, visual=vopt, deformables=[dm])
            cam = rt.make_depth_camera("cam", W, H)
            from metalsim.physics.deformable import XPBDSim as _X  # noqa: F401  (a sim with .d for the GPU path)
            rt.trace(xp, [cam], sources={"cloth": cs.x})
            rt.ctx.synchronize()
            o = {"depth": cam.numpy()}
            tables = rt.tables
        elif which == "tier2":
            r = Tier2Renderer(m, n, width=W, height=H, camera="cam", spp=16, max_bounces=2, center_sample=True, visual=vopt, deformables=[dm])
            r.render(xp, sources={"cloth": cs.x}); r.ctx.synchronize()
            o = {k: getattr(r.out, k).cpu().numpy() for k in ("rgb", "depth", "seg")}
            o["seg"] = np.concatenate([[0], r.tables.slot_seg_ids()])[o["seg"]]
            tables = r.tables
        else:
            r = Tier0Renderer(m, n, width=W, height=H, camera="cam", outputs=("rgb", "depth", "seg"), seg_mode=SEG_GEOM,
                              tier=0 if which == "tier0" else 1, visual=vopt, deformables=[dm])
            r.render(xp, sources={"cloth": cs.x}); r.ctx.synchronize()
            o = {k: getattr(r.out, k).cpu().numpy() for k in ("rgb", "depth", "seg")}
            tables = r.tables
        for e in range(n):
            dd = mujoco.MjData(m); mujoco.mj_forward(m, dd)
            va = m.flex_vertadr[f]
            dd.flexvert_xpos[va:va + len(mesh.x)] = cx[e]
            # the reference shows only the cloth flex (the rope hidden by its group)
            mr = mujoco.MjModel.from_xml_string(DRAPE.replace('mass="0.05" rgba="0.9 0.6 0.1 1"', 'mass="0.05" group="4" rgba="0.9 0.6 0.1 1"'))
            dr = mujoco.MjData(mr); mujoco.mj_forward(mr, dr); dr.flexvert_xpos[:] = dd.flexvert_xpos
            _assert_covered(_coverage(mr, dr, o, e, tables))


@needs_metal
def test_gpu_path_tendon_and_skin_batchsim():
    """Tendons (wrap points per world) and the skin (bone poses per world) from a BatchSim on the GPU path."""
    from metalsim.physics.batch import BatchSim
    xml = EXTRAS.replace('<body name="bone" pos="1.0 0 1.0" euler="10 20 30"/>',
                         '<body name="bone" pos="1.0 0 1.0" euler="10 20 30"><freejoint/><geom size="0.01" rgba="0 0 0 0" contype="0" conaffinity="0"/></body>')
    xml = xml.replace('<body name="t2" pos="-0.25 0 0.75">', '<body name="t2" pos="-0.25 0 0.75"><freejoint/><geom size="0.01" rgba="0 0 0 0" contype="0" conaffinity="0"/>')
    xml = xml.replace("<worldbody>", '<option gravity="0 0 0"/><worldbody>', 1)
    m = mujoco.MjModel.from_xml_string(xml)
    n = 3
    sim = BatchSim(m, n)
    qpos = np.tile(m.qpos0, (n, 1))
    rng = np.random.default_rng(0)
    for e in range(n):
        for jn, dp in (("t2", 0.15), ("bone", 0.1)):
            j = m.body_jntadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, jn)]
            a = m.jnt_qposadr[j]
            qpos[e, a:a + 3] += rng.uniform(-dp, dp, 3)
            q = rng.normal(size=4); q[0] += 2; qpos[e, a + 3:a + 7] = q / np.linalg.norm(q)
    sim.set_state(qpos)
    for which in ("tier0", "tier1", "tier2"):
        if which == "tier2":
            r = Tier2Renderer(m, n, width=W, height=H, camera="cam", spp=16, max_bounces=2, center_sample=True)
        else:
            r = Tier0Renderer(m, n, width=W, height=H, camera="cam", outputs=("rgb", "depth", "seg"), seg_mode=SEG_GEOM,
                              tier=0 if which == "tier0" else 1)
        r.render(sim)
        r.ctx.synchronize()
        o = {k: getattr(r.out, k).cpu().numpy() for k in ("rgb", "depth", "seg")}
        if which == "tier2":
            o["seg"] = np.concatenate([[0], r.tables.slot_seg_ids()])[o["seg"]]
        sx = sim.d.xpos.numpy()
        for e in range(n):
            dk = mujoco.MjData(m); dk.qpos[:] = qpos[e]; mujoco.mj_forward(m, dk)
            assert np.abs(dk.xpos - sx[e]).max() < 1e-4
            _assert_covered(_coverage(m, dk, o, e, r.tables))


# ------------------------------------------------------------------------------------------------------------------
# defaults: scenes without extras are unchanged; segmentation modes; annotators

@needs_metal
def test_scenes_without_extras_are_byte_identical():
    """For scenes with no visible site / flex / skin / tendon, the default (MuJoCo's option) and the extras path off
    (visual=False) give byte-identical RGB, depth and segmentation in every renderer and the depth camera."""
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    xmls = [os.path.join(ROOT, "assets", "so101", "scene_box_rl.xml")]
    prims = """<mujoco><worldbody><light directional="true" pos="0 0 3" dir="0.3 0.3 -0.9"/>
      <camera name="cam" pos="0.6 -0.6 0.5" quat="0.85 0.4 0.15 0.3"/><geom type="plane" size="2 2 .1"/>
      <site name="hidden" group="3" size="0.1"/><body pos="0 0 0.3"><joint type="free"/><geom type="box" size=".06 .04 .05"/>
      <site name="invisible" size="0.05" rgba="1 0 0 0"/></body></worldbody></mujoco>"""
    models = [mujoco.MjModel.from_xml_string(prims)] + [mujoco.MjModel.from_xml_path(x) for x in xmls if os.path.exists(x)]
    for m in models:
        d = mujoco.MjData(m); mujoco.mj_forward(m, d)
        assert build_scene_tables(m, 1, visual=None).extras == []
        cam = 0
        outs = {}
        for vis in (None, False):
            r0 = Tier0Renderer(m, 2, width=96, height=80, camera=cam, outputs=("rgb", "depth", "seg", "normal"), visual=vis)
            r1 = Tier0Renderer(m, 2, width=96, height=80, camera=cam, outputs=("rgb", "depth", "seg"), tier=1, visual=vis)
            r2 = Tier2Renderer(m, 2, width=64, height=48, camera=cam, spp=4, max_bounces=2, visual=vis)
            rt = RayTracer(m, 2, max_range=20.0, visual=vis)
            dc = rt.make_depth_camera(cam, 64, 48)
            outs[vis] = (r0.render_host([d, d]), r1.render_host([d, d]), r2.render_host([d, d]), rt.trace_host([d, d], [dc]))
        for a, b in zip(outs[None], outs[False]):
            for k in a:
                assert np.array_equal(a[k], b[k]), k


@needs_metal
def test_segmentation_modes_and_annotators():
    m, d = _forward(EXTRAS)
    r = Tier0Renderer(m, 1, width=W, height=H, camera="cam", outputs=("rgb", "depth", "seg"), seg_mode=SEG_BODY)
    o = r.render_host([d])
    ref = _ref(m, d, seg=True)
    flex = {e["name"]: e for e in r.tables.extras}
    for name in ("cloth", "skin"):
        ex = flex[name]
        mask = (ref[..., 0] == ex["objid"]) & (ref[..., 1] == int(ex["objtype"]))
        body = int(r.tables.semantic[r.tables.n_geom_slots + r.tables.extras.index(ex), 1])
        assert (o["seg"][0][mask] == body + 1).mean() > 0.99
    # Isaac-style annotators: extras are unlabelled instances with their own instance_id
    from metalsim.replicator.isaac import IsaacAnnotators
    import torch
    r2 = Tier0Renderer(m, 1, width=W, height=H, camera="cam", outputs=("rgb", "depth", "seg"), seg_mode=SEG_GEOM)
    r2.render_host([d])
    a = IsaacAnnotators(r2)
    iid, info = a.instance_id_segmentation()
    iid = iid.cpu().numpy()[0]
    cl = flex["cloth"]
    mask = (ref[..., 0] == cl["objid"]) & (ref[..., 1] == int(cl["objtype"]))
    assert (iid[mask] == cl["seg_id"]).mean() > 0.99 and info["idToLabels"][str(cl["seg_id"])].endswith("/cloth")
    inst, _ = a.instance_segmentation()
    assert (inst.cpu().numpy()[0][mask] == 1).all()          # UNLABELLED
    box = a.bounding_box_2d_tight()
    assert torch.isfinite(box["box"].float()).all()


@needs_metal
def test_dynamic_meshes_and_particle_grains_share_the_tier2_structure():
    """Particle grains (metalsim/render/particles.py) after the dynamic objects' instances in one instance acceleration
    structure: the draped cloth and rope keep their coverage, grains are drawn (segmentation n_slots + 1) and the
    cloth's ids are not taken for grains."""
    m, d = _forward(DRAPE, nstep=150)
    r = Tier2Renderer(m, 1, width=W, height=H, camera="cam", spp=16, max_bounces=2, center_sample=True)
    pts = np.stack(np.meshgrid(np.linspace(0.45, 0.65, 6), np.linspace(-0.3, -0.15, 4), [0.35], indexing="ij"), -1).reshape(-1, 3)
    r.set_particles(pts, radius=0.01)
    o = r.render_host([d], passes=2)
    assert r.rt.n_inst == r.rt.n_inst_static + len(r.rt.dyn_as)
    grains = o["seg"][0] == r.G + 1
    assert grains.sum() > 20
    o["seg"] = np.concatenate([[0], r.tables.slot_seg_ids(), [-1]])[o["seg"]]
    _assert_covered(_coverage(m, d, o, 0, r.tables))

"""Tendons as mujoco.Renderer draws them (MuJoCo 3.14.0 engine_vis_visualize.c addSpatialTendonGeoms / mjv_isCatenary /
mjv_catenary): a slack two-site tendon with a deadband spring or an unreached length limit hangs as a catenary
(numslices + 1 points, a hanging bead when nearly vertical, straight when stretched), and a default-grey tendon without
material is mixed toward vis.rgba.constraint by the impedance of its active limit.

CPU: the generation tables run by the NumPy reference (dynamic_mesh.emulate) give MuJoCo's own capsules
(mjv_updateScene) and colours. GPU (scripts/gpu_run.sh check): the Metal kernels (float32) against the reference,
coverage of MuJoCo's tendon silhouettes in tier 0 / 1 / 2 and the ray-traced depth camera, the limit colour in the
per-env colour table, and tendons="straight" reproducing the previous geometry."""
import mujoco
import numpy as np
import pytest
import warp as wp

from metalsim.render import dynamic_mesh as dm
from metalsim.render.scene_tables import build_scene_tables

needs_metal = pytest.mark.skipif(not wp.is_metal_available(), reason="needs Metal")
W, H = 160, 120

TENDONS = """
<mujoco>
  <worldbody>
    <light directional="true" pos="0 0 3" dir="0 0.3 -1"/>
    <camera name="cam" pos="0 -2.4 0.9" xyaxes="1 0 0 0 0 1"/>
    <site name="a0" pos="-0.95 0 1.2"/><site name="b0" pos="-0.2 0 1.25"/><site name="c0" pos="0.45 0 1.3"/>
    <site name="d0" pos="-0.6 0 0.75"/><site name="e0" pos="0.0 0 0.6"/>
    <!-- second ends on sliders: tendon limits need degrees of freedom (no Jacobian, no constraint row) -->
    <body pos="-0.35 0 1.2"><joint type="slide" axis="1 0 0"/><inertial pos="0 0 0" mass="0.1" diaginertia="1e-4 1e-4 1e-4"/><site name="a1"/></body>
    <body pos="0.3 0 1.05"><joint type="slide" axis="1 0 0"/><inertial pos="0 0 0" mass="0.1" diaginertia="1e-4 1e-4 1e-4"/><site name="b1"/></body>
    <body pos="0.97 0 1.3"><joint type="slide" axis="1 0 0"/><inertial pos="0 0 0" mass="0.1" diaginertia="1e-4 1e-4 1e-4"/><site name="c1"/></body>
    <body pos="-0.598 0 0.55"><joint type="slide" axis="1 0 0"/><inertial pos="0 0 0" mass="0.1" diaginertia="1e-4 1e-4 1e-4"/><site name="d1"/></body>
    <body pos="0.52 0 0.6"><joint type="slide" axis="1 0 0"/><inertial pos="0 0 0" mass="0.1" diaginertia="1e-4 1e-4 1e-4"/><site name="e1"/></body>
  </worldbody>
  <tendon>
    <spatial name="spring" width="0.012" stiffness="50" springlength="0 0.9" rgba="0.9 0.6 0.1 1">
      <site site="a0"/><site site="a1"/></spatial>
    <spatial name="rope" width="0.012" limited="true" range="0 0.75" rgba="0.2 0.6 0.9 1">
      <site site="b0"/><site site="b1"/></spatial>
    <spatial name="taut" width="0.015" limited="true" range="0 0.5"><site site="c0"/><site site="c1"/></spatial>
    <spatial name="bead" width="0.01" stiffness="10" springlength="0 0.6" rgba="0.3 0.9 0.3 1">
      <site site="d0"/><site site="d1"/></spatial>
    <spatial name="near_limit" width="0.015" limited="true" range="0.3 0.53" margin="0.05" solimplimit="0.8 0.95 0.1 0.5 2">
      <site site="e0"/><site site="e1"/></spatial>
  </tendon>
</mujoco>"""


def _forward(xml):
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return m, d


def _sources(m, datas):
    nw = 2 * int(m.nwrap)
    return {"ten_wrapadr": np.stack([d.ten_wrapadr for d in datas]), "ten_wrapnum": np.stack([d.ten_wrapnum for d in datas]),
            "wrap_obj": np.stack([np.asarray(d.wrap_obj).reshape(-1)[:nw] for d in datas]),
            "wrap_xpos": np.stack([np.asarray(d.wrap_xpos).reshape(-1)[:3 * nw].reshape(-1, 3) for d in datas]),
            "ten_length": np.stack([d.ten_length for d in datas])}


def _scene(m, d):
    scn = mujoco.MjvScene(m, maxgeom=2000)
    mujoco.mjv_updateScene(m, d, mujoco.MjvOption(), mujoco.MjvPerturb(), mujoco.MjvCamera(), mujoco.mjtCatBit.mjCAT_ALL, scn)
    return scn


def _tendon_geoms(scn, i):
    return [scn.geoms[k] for k in range(scn.ngeom)
            if scn.geoms[k].objtype == mujoco.mjtObj.mjOBJ_TENDON and scn.geoms[k].objid == i]


def test_catenary_classification_matches_mujoco():
    """Static conditions (catenary_mode) plus the slack test give exactly the tendons MuJoCo draws as catenaries
    (more than one capsule between two sites), and the expected point counts (bead: 3, full: numslices + 1)."""
    m, d = _forward(TENDONS)
    scn = _scene(m, d)
    npts = min(m.vis.quality.numslices + 1, 100)
    got = {}
    for i in range(m.ntendon):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_TENDON, i)
        mode, L = dm.catenary_mode(m, i)
        n_mj = len(_tendon_geoms(scn, i))
        if mode < 0 or (mode == 1 and d.ten_length[i] >= L):
            n_ours = 1
        else:
            adr, wx = d.ten_wrapadr[i], np.asarray(d.wrap_xpos).reshape(-1, 3)
            n_ours = len(dm.catenary_points(wx[adr], wx[adr + 1], m.opt.gravity, L, npts)) - 1
        got[name] = (n_ours, n_mj)
    assert got == {"spring": (npts - 1,) * 2, "rope": (npts - 1,) * 2, "taut": (1, 1), "bead": (2, 2), "near_limit": (1, 1)}, got


def test_generated_tendons_match_mjv_update_scene():
    """CPU reference of the kernels: every tendon capsule equals MuJoCo's (pos, mat, size) to 2e-5 m; tendons="straight"
    keeps one capsule per wrap segment."""
    m, d = _forward(TENDONS)
    t = build_scene_tables(m, 1, visual=None)
    scn = _scene(m, d)
    for tendons, catenaries in (("mujoco", True), ("straight", False)):
        plan = dm._Plan(m, t, tendons=tendons)
        plan.finalize(1)
        plan.validate(1)
        rec = dm.emulate(plan, 1, _sources(m, [d]))
        cap = plan.cap_t[0].astype(np.float64)
        by_slot = {ob["slot"]: ob for ob in plan.objects}
        for dd in t.dyn:
            ob = by_slot[dd["slot"]]
            segs = rec[ob["obj_base"]: ob["obj_base"] + ob["nvert"]].reshape(-1, len(cap), 8)
            valid = [s for s in segs if np.ptp(s[:, :3], 0).max() > 0]
            geoms = _tendon_geoms(scn, dd["id"])
            if not catenaries:
                assert len(valid) == 1
                continue
            assert len(valid) == len(geoms) > 0, (dd, len(valid), len(geoms))
            for s, g in zip(valid, geoms):
                M = np.asarray(g.mat, np.float64).reshape(3, 3)
                w, h = float(g.size[0]), float(g.size[2])
                loc = np.stack([w * cap[:, 0], w * cap[:, 1], w * cap[:, 2] + h * cap[:, 3]], 1)
                np.testing.assert_allclose(s[:, :3], np.asarray(g.pos, np.float64) + loc @ M.T, atol=2e-5)


def test_limit_colour_matches_mujoco():
    """The tint table and dm.tendon_limit_imp give the rgba MuJoCo's scene carries for the default-grey limited tendons
    (taut: past its limit, full impedance; near_limit: inside the margin, solimp's interpolated impedance)."""
    m, d = _forward(TENDONS)
    t = build_scene_tables(m, 1, visual=None)
    plan = dm._Plan(m, t)
    scn = _scene(m, d)
    tinted = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_TENDON, i): i for (i, _, _) in plan.tint}
    assert set(tinted) == {"taut", "near_limit"}, tinted     # default grey, no material, limited
    for name, i in tinted.items():
        imp = dm.tendon_limit_imp(m, i, float(d.ten_length[i]))
        assert imp > 0.1, (name, imp)
        ours = (1 - imp) * np.asarray(m.tendon_rgba[i][:3]) + imp * np.asarray(m.vis.rgba.constraint[:3])
        (g,) = _tendon_geoms(scn, i)
        np.testing.assert_allclose(ours, np.asarray(g.rgba[:3]), atol=1e-6)


def _states(m):
    """Three states: as authored, sliders moved (the rope taut, the spring further slack), and the taut one released."""
    out = []
    for dq in ((0, 0, 0, 0, 0), (0.05, 0.25, 0.0, 0.0, -0.05), (0.0, 0.0, -0.1, 0.002, 0.03)):
        d = mujoco.MjData(m)
        d.qpos[:] = dq
        mujoco.mj_forward(m, d)
        out.append(d)
    return out


@needs_metal
def test_gpu_catenary_and_tint_match_reference():
    """Metal kernels (float32 catenary) against the float64 reference for three envs in different states, and the
    per-env colour table's tendon slots against MuJoCo's scene colours."""
    from metalsim.render.tier0 import Tier0Renderer
    m, _ = _forward(TENDONS)
    datas = _states(m)
    r = Tier0Renderer(m, 3, width=32, height=24, camera="cam", outputs=("rgb",))
    r.render_host(datas)
    ref = dm.emulate(r.dyn.plan, 3, _sources(m, datas))
    np.testing.assert_allclose(r.dyn.records(), ref, atol=2e-4)
    col = r.ctx.buffer_array(r.inst_col_buf, np.float32, (3, r.G, 4))
    for e, d in enumerate(datas):
        scn = _scene(m, d)
        for (i, slot, _) in r.dyn.plan.tint:
            g = _tendon_geoms(scn, i)
            if len(g) == 1:          # straight (not hanging): MuJoCo's colour, tinted or not
                np.testing.assert_allclose(col[e, slot, :3], np.asarray(g[0].rgba[:3]), atol=2e-6)


@needs_metal
@pytest.mark.parametrize("which", ["tier0", "tier1", "tier2", "raydepth"])
def test_tendon_coverage_vs_mujoco_renderer(which):
    """Coverage of MuJoCo's tendon silhouettes (catenaries, bead, straight) >= 0.97 in rgb / depth / segmentation;
    with tendons="straight" the hanging ones lose most of it (the previous behaviour)."""
    from test_render_extras import _coverage, _render
    m, _ = _forward(TENDONS)
    datas = _states(m)
    for e in (0, 1):
        r, o = _render(m, datas, which)
        cov = _coverage(m, datas[e], o, e, r.tables)
        low = {k: v for k, v in cov.items() if v["px"] >= 15 and min(v.get(q, 1.0) for q in ("rgb", "depth", "seg")) < 0.97}
        assert not low, (which, e, cov)
    r, o = _render(m, datas[:1], which, tendons="straight")
    cov = _coverage(m, datas[0], o, 0, r.tables)
    assert cov["spring"]["depth"] < 0.6 and cov["rope"]["depth"] < 0.8, cov


@needs_metal
def test_limit_colour_in_the_image():
    """The taut default-grey tendon is drawn in MuJoCo's constraint colour in tier 0, as in mujoco.Renderer (mean rgb
    over MuJoCo's silhouette within 12 levels), grey with tendons="straight"."""
    from test_render_extras import _ref, _render
    m, d = _forward(TENDONS)
    seg = _ref(m, d, seg=True)
    ref = _ref(m, d)
    i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_TENDON, "taut")
    mask = (seg[..., 0] == i) & (seg[..., 1] == int(mujoco.mjtObj.mjOBJ_TENDON))
    assert mask.sum() > 30
    _, o = _render(m, [d], "tier0")
    ours = o["rgb"][0][mask].mean(0)
    mj = ref[mask].mean(0)
    assert np.abs(ours - mj).max() < 12, (ours, mj)
    assert mj[0] > 2 * mj[1]
    _, o = _render(m, [d], "tier0", tendons="straight")
    grey = o["rgb"][0][mask].mean(0)
    assert abs(grey[0] - grey[1]) < 8, grey

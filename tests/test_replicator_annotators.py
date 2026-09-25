"""Isaac Replicator annotators, BasicWriter layout and Isaac Lab event terms against ground truth.

Scene: a floor patch and boxes at known poses (one rotated, one partly hidden behind another, one
body made of two boxes) seen by a fixed camera. Ground truth is an analytic ray cast through every
pixel centre (u = x + 0.5, v = y + 0.5, the rasterizer's sample point). Pixels whose centre lies
within EPS px of a silhouette edge are "ambiguous" (the rasterizer's fixed-point subpixel snapping
decides them); everything else must match exactly.
"""
import json
import os

import mujoco
import numpy as np
import pytest
import torch
import warp as wp

pytestmark = pytest.mark.skipif(not wp.is_metal_available(), reason="needs Metal")

W = H = 96
EPS = 0.01   # px

XML = """
<mujoco>
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <worldbody>
    <light directional="true" dir="0 0 -1" pos="0 0 3"/>
    <camera name="cam" pos="0.15 -1.6 1.3" xyaxes="1 0 0 0 0.62 0.78" fovy="50"/>
    <geom name="floor" type="plane" size="0.8 0.8 0.05" pos="0 0 0"/>
    <body name="red_box" pos="-0.35 0.05 0.12"><freejoint/>
      <geom name="red" type="box" size="0.12 0.1 0.12" rgba="1 0 0 1"/></body>
    <body name="blue_box" pos="0.25 0.1 0.1" quat="0.9239 0 0 0.3827"><freejoint/>
      <geom name="blue" type="box" size="0.1 0.15 0.1" rgba="0 0 1 1"/></body>
    <body name="hidden_box" pos="0.3 0.45 0.15"><freejoint/>
      <geom name="hidden" type="box" size="0.12 0.08 0.15" rgba="0 1 0 1"/></body>
    <body name="table" pos="-0.2 0.5 0.0"><freejoint/>
      <geom name="top" type="box" size="0.2 0.12 0.03" pos="0 0 0.25" rgba="0.6 0.4 0.2 1"/>
      <geom name="leg" type="box" size="0.03 0.03 0.11" pos="0.1 0.05 0.11" quat="0.9659 0 0 0.2588" rgba="0.6 0.4 0.2 1"/></body>
  </worldbody>
</mujoco>"""

LABELS = {"red_box": "cube", "blue_box": "cube", "hidden_box": "crate", "table": "table"}


def _model():
    return mujoco.MjModel.from_xml_string(XML)


def _data(m, shift=None):
    d = mujoco.MjData(m)
    if shift is not None:
        for body, dp in shift.items():
            b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body)
            d.qpos[m.jnt_qposadr[m.body_jntadr[b]]:][:3] += dp
    mujoco.mj_forward(m, d)
    return d


# -- ground truth ray cast ------------------------------------------------------------------------------

def _rays(m, d, intr, du=0.0, dv=0.0):
    fx, fy, cx, cy = intr
    xs, ys = np.meshgrid(np.arange(W) + 0.5 + du, np.arange(H) + 0.5 + dv)
    dc = np.stack([(xs - cx) / fx, -(ys - cy) / fy, -np.ones_like(xs)], -1)   # z = -1: t = image-plane distance
    R = d.cam_xmat[0].reshape(3, 3)
    return d.cam_xpos[0].copy(), dc @ R.T


def _cast(m, d, intr, geoms=None, du=0.0, dv=0.0):
    """Nearest hit per pixel: (geom id or -1, t, world normal)."""
    o, dirs = _rays(m, d, intr, du, dv)
    best_t = np.full((H, W), np.inf); best_g = -np.ones((H, W), int); best_n = np.zeros((H, W, 3))
    for g in (range(m.ngeom) if geoms is None else geoms):
        R = d.geom_xmat[g].reshape(3, 3); p = d.geom_xpos[g]
        ol = R.T @ (o - p); dl = dirs @ R            # ray in the geom frame
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
            with np.errstate(divide="ignore", invalid="ignore"):
                t = -ol[2] / dl[..., 2]
            hit = ol + t[..., None] * dl
            ok = (t > 0) & (np.abs(hit[..., 0]) <= m.geom_size[g][0]) & (np.abs(hit[..., 1]) <= m.geom_size[g][1])
            n = np.broadcast_to(R[:, 2], (H, W, 3))
        else:
            s = m.geom_size[g][:3]
            with np.errstate(divide="ignore", invalid="ignore"):
                t1 = (-s - ol) / dl; t2 = (s - ol) / dl
            tn = np.minimum(t1, t2); tf = np.maximum(t1, t2)
            t = tn.max(-1); ok = (t <= tf.min(-1)) & (t > 0)
            ax = tn.argmax(-1)
            sign = -np.sign(np.take_along_axis(dl, ax[..., None], -1)[..., 0])
            n = R[:, ax].transpose(1, 2, 0) * sign[..., None]
        closer = ok & (t < best_t)
        best_t[closer] = t[closer]; best_g[closer] = g; best_n[closer] = n[closer]
    return best_g, best_t, best_n


def _cast_sure(m, d, intr, geoms=None):
    """Centre-sample truth plus an ambiguity mask (any of four EPS-offset samples disagrees)."""
    g, t, n = _cast(m, d, intr, geoms)
    amb = np.zeros_like(g, bool)
    for du, dv in ((EPS, EPS), (-EPS, EPS), (EPS, -EPS), (-EPS, -EPS)):
        amb |= _cast(m, d, intr, geoms, du, dv)[0] != g
    return g, t, n, amb


# -- fixtures ----------------------------------------------------------------------------------------------

@pytest.fixture(scope="module", params=["warp", "torch"])
def scene(request):
    """``impl``: the fused Warp kernels (default) and the first torch implementation (kept behind the
    flag) must give the same, exact results."""
    from metalsim.render.tier0 import Tier0Renderer, SEG_SLOT
    from metalsim.replicator import IsaacAnnotators, Semantics, State
    m = _model()
    n = 2
    r = Tier0Renderer(m, n, width=W, height=H, camera="cam", outputs=("rgb", "depth", "seg", "normal"), seg_mode=SEG_SLOT)
    sem = Semantics.from_rules(m, bodies={k: v for k, v in LABELS.items()})
    datas = [_data(m), _data(m, {"red_box": np.array([0.05, 0.0, 0.0])})]
    r.render_host(datas)
    ann = IsaacAnnotators(r, sem, impl=request.param)
    ann.set_state(State.from_datas(datas))
    intr = tuple(float(x) for x in (r.intrinsics.fx, r.intrinsics.fy, r.intrinsics.cx, r.intrinsics.cy))
    gts = [_cast_sure(m, d, intr) for d in datas]
    return m, r, sem, ann, datas, intr, gts


def _gid(m, name):
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)


def _bid(m, name):
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)


def _entity_of_geom(sem, g):
    return -1 if g < 0 else int(sem.geom_entity[g])


# -- tests -------------------------------------------------------------------------------------------------

def test_segmentation_exact(scene):
    m, r, sem, ann, datas, intr, gts = scene
    inst, info = ann.instance_segmentation()
    semseg, sinfo = ann.semantic_segmentation()
    iid, iinfo = ann.instance_id_segmentation()
    inst, semseg, iid = inst.cpu().numpy(), semseg.cpu().numpy(), iid.cpu().numpy()
    ent_sem, id_to_labels = sem.semantic_table()
    assert sinfo["idToLabels"] == id_to_labels
    assert id_to_labels["0"] == {"class": "BACKGROUND"} and id_to_labels["1"] == {"class": "UNLABELLED"}
    assert {v["class"] for k, v in id_to_labels.items() if int(k) > 1} == {"cube", "crate", "table"}
    total_amb = 0
    for e, (g, t, nrm, amb) in enumerate(gts):
        ent = np.vectorize(lambda x: _entity_of_geom(sem, x))(g)
        gt_inst = np.where(g < 0, 0, np.where(ent < 0, 1, ent + 2))
        gt_sem = np.where(g < 0, 0, np.where(ent < 0, 1, ent_sem[np.maximum(ent, 0)]))
        sure = ~amb
        assert np.array_equal(inst[e][sure], gt_inst[sure])
        assert np.array_equal(semseg[e][sure], gt_sem[sure])
        assert np.array_equal(iid[e][sure], (g + 1)[sure])
        total_amb += amb.sum()
        # every GT instance is present, labels resolve to the right paths / classes
        for k in np.unique(gt_inst[sure]):
            if k >= 2:
                assert info["idToLabels"][str(k)] == sem.entities[k - 2].path
                assert info["idToSemantics"][str(k)]["class"] == LABELS[sem.entities[k - 2].path.split("/")[-1]]
    assert iinfo["idToLabels"][str(_gid(m, "leg") + 1)] == "/World/table/leg"
    # the ambiguity band is a thin set of edge pixels
    assert total_amb < 0.01 * 2 * H * W, total_amb


def _bounds(mask_sure, mask_maybe):
    """Allowed [lo, hi] for each of x_min, y_min, x_max, y_max given sure-in / maybe-in masks."""
    def ext(mk):
        ys, xs = np.nonzero(mk)
        return np.array([xs.min(), ys.min(), xs.max(), ys.max()])
    a, b = ext(mask_sure), ext(mask_maybe)
    return np.minimum(a, b), np.maximum(a, b)


def test_bbox_2d_tight_and_loose_exact(scene):
    m, r, sem, ann, datas, intr, gts = scene
    tight = ann.bounding_box_2d_tight(occlusion=True)
    loose = ann.bounding_box_2d_loose(occlusion=True)
    tb, lb = tight["box"].cpu().numpy(), loose["box"].cpu().numpy()
    tc, lc = tight["count"].cpu().numpy(), loose["count"].cpu().numpy()
    occ = tight["occlusionRatio"].cpu().numpy()
    hid = int(sem.geom_entity[_gid(m, "hidden")])
    for e, (g, t, nrm, amb) in enumerate(gts):
        ent = np.vectorize(lambda x: _entity_of_geom(sem, x))(g)
        for k in range(len(sem.entities)):
            sure_in = (ent == k) & ~amb
            maybe_in = (ent == k) | amb
            if not sure_in.any():
                assert not tight["valid"][e, k]
                continue
            lo, hi = _bounds(sure_in, maybe_in)
            assert np.all(tb[e, k] >= lo) and np.all(tb[e, k] <= hi), (e, k, tb[e, k], lo, hi)
            assert sure_in.sum() <= tc[e, k] <= maybe_in.sum()
            # loose: the entity's silhouette with nothing else in the scene
            gk = sem.entities[k].geoms
            ga, _, _, amba = _cast_sure(m, datas[e], intr, gk)
            s_in, m_in = (ga >= 0) & ~amba, (ga >= 0) | amba
            lo, hi = _bounds(s_in, m_in)
            assert np.all(lb[e, k] >= lo) and np.all(lb[e, k] <= hi), (e, k, lb[e, k], lo, hi)
            assert s_in.sum() <= lc[e, k] <= m_in.sum()
            # occlusion ratio within the ambiguity of both counts
            lo_r = 1 - maybe_in.sum() / max(s_in.sum(), 1); hi_r = 1 - sure_in.sum() / max(m_in.sum(), 1)
            assert lo_r - 1e-6 <= occ[e, k] <= hi_r + 1e-6
        assert occ[e, hid] > 0.2          # the hidden box really is partly hidden
    # structured records use Isaac's dtype and info keys
    rec, info = ann.records("bounding_box_2d_tight", tight, 0)
    assert rec.dtype.names == ("semanticId", "x_min", "y_min", "x_max", "y_max", "occlusionRatio")
    assert rec.dtype["semanticId"] == np.dtype("<u4") and rec.dtype["x_min"] == np.dtype("<i4")
    assert set(info) == {"bboxIds", "idToLabels", "primPaths"} and len(info["primPaths"]) == len(rec)


def test_bbox_2d_loose_projected_vertices(scene):
    """The render-free loose box (projected render-mesh vertices) equals the analytic corner projection."""
    m, r, sem, ann, datas, intr, gts = scene
    from metalsim.replicator import IsaacAnnotators
    lb = ann.bounding_box_2d_loose(occlusion=False)["box"].cpu().numpy()
    all_v = IsaacAnnotators(r, sem, impl=ann.impl, loose_hull=False)      # every render-mesh vertex
    all_v.set_state(ann.state)
    assert np.array_equal(all_v.bounding_box_2d_loose(occlusion=False)["box"].cpu().numpy(), lb)
    fx, fy, cx, cy = intr
    signs = np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)])
    for e, d in enumerate(datas):
        Rc = d.cam_xmat[0].reshape(3, 3); pc = d.cam_xpos[0]
        for k, ent in enumerate(sem.entities):
            pts = np.concatenate([d.geom_xpos[g] + (signs * m.geom_size[g][:3]) @ d.geom_xmat[g].reshape(3, 3).T for g in ent.geoms])
            c = (pts - pc) @ Rc
            u = cx + fx * c[:, 0] / -c[:, 2]; v = cy - fy * c[:, 1] / -c[:, 2]
            want = np.array([np.ceil(u.min() - 0.5), np.ceil(v.min() - 0.5), np.floor(u.max() - 0.5), np.floor(v.max() - 0.5)])
            want = np.clip(want, 0, [W - 1, H - 1, W - 1, H - 1])
            assert np.array_equal(lb[e, k], want), (e, k, lb[e, k], want)


def test_bbox_3d_exact(scene):
    m, r, sem, ann, datas, intr, gts = scene
    out = ann.bounding_box_3d()
    ext = out["extents"].cpu().numpy(); T = out["transform"].cpu().numpy()
    for k, ent in enumerate(sem.entities):
        name = ent.path.split("/")[-1]
        if name == "table":
            # union of the top (0.2 x 0.12 x 0.03 at z 0.25) and a leg rotated 30 deg about z
            leg = _gid(m, "leg")
            c, s = np.cos(np.deg2rad(30)), np.sin(np.deg2rad(30))
            hx = 0.03 * c + 0.03 * s
            want = np.array([min(-0.2, 0.1 - hx), min(-0.12, 0.05 - hx), min(0.22, 0.0),
                             max(0.2, 0.1 + hx), max(0.12, 0.05 + hx), 0.28])
        else:
            s = m.geom_size[ent.geoms[0]][:3]
            want = np.concatenate([-s, s])
        np.testing.assert_allclose(ext[k], want, atol=2e-6)
        for e, d in enumerate(datas):
            want_T = np.eye(4)
            want_T[:3, :3] = d.xmat[ent.body].reshape(3, 3).T
            want_T[3, :3] = d.xpos[ent.body]
            np.testing.assert_allclose(T[e, k], want_T, atol=2e-6)
            # the local box transformed by T contains every world-frame corner of the entity's geoms
            corners = np.concatenate([d.geom_xpos[g] + (np.array([[a, b, cc] for a in (-1, 1) for b in (-1, 1) for cc in (-1, 1)]) * m.geom_size[g][:3]) @ d.geom_xmat[g].reshape(3, 3).T for g in ent.geoms])
            local = (np.c_[corners, np.ones(len(corners))] @ np.linalg.inv(T[e, k]))[:, :3]
            assert np.all(local >= ext[k][:3] - 1e-5) and np.all(local <= ext[k][3:] + 1e-5)
    rec, info = ann.records("bounding_box_3d", out, 0)
    assert rec.dtype["transform"].shape == (4, 4) and rec.dtype["x_min"] == np.dtype("<f4")


def test_depth_normals_pointcloud(scene):
    m, r, sem, ann, datas, intr, gts = scene
    d_ip = ann.distance_to_image_plane().cpu().numpy()
    d_c = ann.distance_to_camera().cpu().numpy()
    nr = ann.normals().cpu().numpy()
    nd = ann.normals_from_depth().cpu().numpy()
    pc = ann.pointcloud(include_unlabelled=True)
    pts, valid = pc["points"].cpu().numpy(), pc["valid"].cpu().numpy()
    fx, fy, cx, cy = intr
    xs, ys = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    ray = np.sqrt(1 + ((xs - cx) / fx) ** 2 + ((ys - cy) / fy) ** 2)
    errs_d, errs_n, errs_nd = [], [], []
    for e, (g, t, nrm, amb) in enumerate(gts):
        sure = ~amb
        hit = sure & (g >= 0)
        assert np.all(np.isinf(d_ip[e][sure & (g < 0)])) and np.all(np.isinf(d_c[e][sure & (g < 0)]))
        errs_d.append(np.abs(d_ip[e][hit] - t[hit]).max())
        np.testing.assert_allclose(d_c[e][hit], (t * ray)[hit], rtol=1e-4, atol=3e-4)
        ang_r = np.arccos(np.clip((nr[e][..., :3] * nrm).sum(-1), -1, 1))
        errs_n.append(ang_r[hit].max())
        # world points on the surfaces
        o, dirs = _rays(m, datas[e], intr)
        want = o + t[..., None] * dirs
        assert np.abs(pts[e][hit] - want[hit]).max() < 5e-4
        assert np.array_equal(valid[e][sure], (g >= 0)[sure])
        # depth-derived normals: faces interior (3x3 neighbourhood on the same geom, unambiguous)
        from scipy.ndimage import minimum_filter, maximum_filter
        # face id: geom and quantized GT normal; interior = 3x3 neighbourhood on one face, unambiguous
        face = np.where(hit, g * 1000 + np.round((nrm + 1) * 4).astype(int) @ np.array([100, 10, 1]), -9)
        interior = (minimum_filter(face, 3) == maximum_filter(face, 3)) & hit
        interior &= minimum_filter(sure.astype(int), 3) == 1
        ang = np.arccos(np.clip((nd[e][..., :3] * nrm).sum(-1), -1, 1))[interior]
        errs_nd.append(ang)
    assert max(errs_d) < 3e-4, errs_d          # depth-buffer precision (24-bit-mantissa NDC z)
    assert max(errs_n) < 1e-3, errs_n          # raster face normals, fp16 render target (2^-11 quantization)
    a = np.concatenate(errs_nd)
    assert a.max() < 5e-3, (np.median(a), np.percentile(a, 99), a.max())   # face interiors (creases are not)


def test_motion_vectors_exact(scene):
    """Env 1 has the red box 5 cm further along +x than env 0. With env 0's state as the previous frame,
    every red pixel's vector is (u_prev - u, v_prev - v) of its surface point moved back 5 cm; static
    geoms have zero motion. Isaac: +x = motion to the left, +y = motion up, so this box (moving right,
    i.e. +u) has negative x."""
    m, r, sem, ann, datas, intr, gts = scene
    from metalsim.replicator import State
    mv = ann.motion_vectors(State.from_datas([datas[0], datas[0]])).cpu().numpy()
    g, t, nrm, amb = gts[1]
    fx, fy, cx, cy = intr
    red = _gid(m, "red")
    o, dirs = _rays(m, datas[1], intr)
    p = o + t[..., None] * dirs - np.array([0.05, 0, 0])
    c = (p - o) @ datas[1].cam_xmat[0].reshape(3, 3)
    u0 = cx + fx * c[..., 0] / -c[..., 2]; v0 = cy - fy * c[..., 1] / -c[..., 2]
    xs, ys = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    sel = (g == red) & ~amb
    assert sel.sum() > 100
    np.testing.assert_allclose(mv[1][sel][:, 0], (u0 - xs)[sel], atol=2e-3)
    np.testing.assert_allclose(mv[1][sel][:, 1], (v0 - ys)[sel], atol=2e-3)
    assert np.all(mv[1][sel][:, 0] < 0)
    stat = (g >= 0) & (g != red) & ~amb
    assert np.abs(mv[1][stat][:, :2]).max() < 1e-4
    assert np.abs(mv[0][..., :2]).max() < 1e-4            # env 0: previous state == current state


def test_camera_params_project(scene):
    m, r, sem, ann, datas, intr, gts = scene
    cp = ann.camera_params()
    V = cp["cameraViewTransform"].cpu().numpy(); P = cp["cameraProjection"].cpu().numpy()
    fx, fy, cx, cy = intr
    d = datas[0]
    pts = d.geom_xpos[:4] + 0.01
    clip = np.c_[pts, np.ones(len(pts))] @ V[0] @ P[0]      # row-vector convention, as USD
    ndc = clip[:, :3] / clip[:, 3:]
    u = (ndc[:, 0] + 1) / 2 * W; v = (1 - ndc[:, 1]) / 2 * H
    c = (pts - d.cam_xpos[0]) @ d.cam_xmat[0].reshape(3, 3)
    np.testing.assert_allclose(u, cx + fx * c[:, 0] / -c[:, 2], atol=1e-3)
    np.testing.assert_allclose(v, cy - fy * c[:, 1] / -c[:, 2], atol=1e-3)
    ap = cp["cameraAperture"].cpu().numpy()[0]; f = float(cp["cameraFocalLength"][0])
    np.testing.assert_allclose(f / ap[1] * H, fy, rtol=1e-6)


def test_basic_writer_layout(scene, tmp_path):
    m, r, sem, ann, datas, intr, gts = scene
    from metalsim.replicator import BasicWriter, State
    kw = dict(rgb=True, bounding_box_2d_tight=True, bounding_box_2d_loose=True, semantic_segmentation=True,
              instance_id_segmentation=True, instance_segmentation=True, distance_to_camera=True,
              distance_to_image_plane=True, bounding_box_3d=True, normals=True, motion_vectors=True,
              camera_params=True, pointcloud=True)
    one = BasicWriter(str(tmp_path / "one"), **kw)
    one.write_batch(ann, envs=[0], prev_state=State.from_datas(datas))
    expect = {"rgb_0000.png", "normals_0000.png", "distance_to_camera_0000.npy", "distance_to_image_plane_0000.npy",
              "semantic_segmentation_0000.png", "semantic_segmentation_labels_0000.json",
              "instance_id_segmentation_0000.png", "instance_id_segmentation_mapping_0000.json",
              "instance_segmentation_0000.png", "instance_segmentation_mapping_0000.json",
              "instance_segmentation_semantics_mapping_0000.json", "motion_vectors_0000.npy", "camera_params_0000.json",
              "pointcloud_0000.npy", "pointcloud_rgb_0000.npy", "pointcloud_normals_0000.npy",
              "pointcloud_semantic_0000.npy", "pointcloud_instance_0000.npy"}
    for k in ("2d_tight", "2d_loose", "3d"):
        expect |= {f"bounding_box_{k}_0000.npy", f"bounding_box_{k}_labels_0000.json", f"bounding_box_{k}_prim_paths_0000.json"}
    assert set(os.listdir(tmp_path / "one")) == expect
    # multi render product: <rp>/<annotator>/<file>
    two = BasicWriter(str(tmp_path / "two"), rgb=True, semantic_segmentation=True, bounding_box_2d_tight=True)
    two.write_batch(ann, occlusion=False); two.write_batch(ann, occlusion=False)
    assert sorted(os.listdir(tmp_path / "two")) == ["env_0000", "env_0001"]
    assert sorted(os.listdir(tmp_path / "two" / "env_0001" / "rgb")) == ["rgb_0000.png", "rgb_0001.png"]
    # read back like a downstream tool: colours in the PNG resolve through the labels JSON
    import imageio.v2 as iio
    o = tmp_path / "one"
    img = iio.imread(o / "semantic_segmentation_0000.png")
    lab = json.load(open(o / "semantic_segmentation_labels_0000.json"))
    semseg, _ = ann.semantic_segmentation()
    sid = semseg[0].cpu().numpy()
    _, id_to_labels = sem.semantic_table()
    for col in {tuple(c) for c in img.reshape(-1, 4)}:
        key = str(tuple(int(x) for x in col))
        ids = np.unique(sid[(img == np.array(col)).all(-1)])
        assert len(ids) == 1 and lab[key] == id_to_labels[str(ids[0])]
    assert lab.get("(0, 0, 0, 0)") == {"class": "BACKGROUND"} or 0 not in sid
    bb = np.load(o / "bounding_box_2d_tight_0000.npy")
    bl = json.load(open(o / "bounding_box_2d_tight_labels_0000.json"))
    bp = json.load(open(o / "bounding_box_2d_tight_prim_paths_0000.json"))
    assert bb.dtype.names[0] == "semanticId" and len(bp) == len(bb)
    assert {bl[str(s)]["class"] for s in bb["semanticId"]} <= {"cube", "crate", "table"}
    cam = json.load(open(o / "camera_params_0000.json"))
    assert len(cam["cameraViewTransform"]) == 16 and len(cam["cameraProjection"]) == 16
    pcl = np.load(o / "pointcloud_0000.npy")
    assert pcl.shape[1] == 3 and len(np.load(o / "pointcloud_rgb_0000.npy")) == len(pcl)
    assert np.load(o / "distance_to_image_plane_0000.npy").dtype == np.float32


def test_semantics_from_usd(tmp_path):
    from metalsim.replicator import Semantics
    from metalsim.scene.mjcf_to_usd import import_mjcf
    from pxr import Usd
    xml = tmp_path / "s.xml"; xml.write_text(XML)
    import_mjcf(str(xml), str(tmp_path / "s.usda"), write_textures=False)
    m = mujoco.MjModel.from_xml_path(str(xml))
    a = Semantics.from_usd(m, Usd.Stage.Open(str(tmp_path / "s.usda")))
    b = Semantics.from_body_names(m)
    got = sorted((e.path, tuple(sorted(e.geoms)), e.labels["class"]) for e in a.entities)
    want = sorted((e.path, tuple(sorted(e.geoms)), e.labels["class"]) for e in b.entities)
    # mjcf_to_usd also labels world geoms with class "world": the floor is its own labelled instance
    assert got == sorted(want + [("/World/floor", (_gid(m, "floor"),), "world")])


# -- Isaac Lab event terms -----------------------------------------------------------------------------------

EV_XML = """
<mujoco><compiler angle="radian"/><option timestep="0.002"/><worldbody>
  <geom type="plane" size="3 3 0.1"/>
  <body name="base" pos="0 0 0.5"><freejoint/>
    <geom name="torso" type="box" size="0.2 0.1 0.05" pos="0.03 0 0" mass="4"/>
    <body name="thigh" pos="0.2 0 0"><joint name="hip" type="hinge" axis="0 1 0" range="-1 1"/>
      <geom type="capsule" fromto="0 0 0 0 0 -0.3" size="0.03" mass="1"/>
      <body name="shin" pos="0 0 -0.3"><joint name="knee" type="hinge" axis="0 1 0" range="-2 0.5"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.3" size="0.03" mass="0.5"/></body></body></body>
</worldbody>
<actuator><position joint="hip" kp="30" kv="1"/><position joint="knee" kp="20" kv="0.5"/></actuator>
<keyframe><key qpos="0 0 0.5 1 0 0 0 0.4 -0.8"/></keyframe></mujoco>"""


@pytest.fixture(scope="module")
def evsim():
    from metalsim.physics.batch import BatchSim, BatchSimOptions
    from metalsim.replicator import events as ev
    m = mujoco.MjModel.from_xml_string(EV_XML)
    fields = tuple(sorted({f for v in ev.REQUIRED_FIELDS.values() for f in v}))
    sim = BatchSim(m, 64, options=BatchSimOptions(per_world_fields=fields, njmax=64))
    sim.synchronize()
    ctx = ev.EventContext(sim, seed=1, default_qpos=torch.as_tensor(m.key_qpos[0], dtype=torch.float32, device="mps"))
    return m, sim, ctx, ev


def _com_vel(m, sim, env, body):
    """MuJoCo's COM velocity of ``body`` in world ``env``, evaluated with that world's (possibly
    randomized) COM offset: linear (COM, world), angular (world), MjData."""
    mm = mujoco.MjModel.from_xml_string(EV_XML)
    mm.body_ipos[:] = sim.tm.body_ipos.cpu().numpy()[env]
    d = sim.get_world(env)
    mujoco.mj_forward(mm, d)
    v = np.zeros(6)
    mujoco.mj_objectVelocity(mm, d, mujoco.mjtObj.mjOBJ_BODY, body, v, 0)
    return v[3:], v[:3], d


def test_event_reset_root_and_joints(evsim):
    m, sim, ctx, ev = evsim
    pr = {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)}
    vr = {k: (-0.5, 0.5) for k in ("x", "y", "z", "roll", "pitch", "yaw")}
    torch.manual_seed(0)
    ev.reset_root_state_uniform(ctx, None, pose_range=pr, velocity_range=vr)
    ev.reset_joints_by_scale(ctx, None, position_range=(0.5, 1.5), velocity_range=(0.0, 0.0))
    torch.mps.synchronize(); sim.forward(); sim.synchronize()
    q = sim.t.qpos.cpu().numpy(); v = sim.t.qvel.cpu().numpy()
    assert np.all(np.abs(q[:, 0:2]) <= 0.5) and np.allclose(q[:, 2], 0.5)
    yaw = 2 * np.arctan2(q[:, 6], q[:, 3])
    assert np.allclose(q[:, 4:6], 0, atol=1e-6) and np.all(np.abs(yaw) <= 3.14 + 1e-5)
    # joints: default * scale, inside the limits
    s_hip, s_knee = q[:, 7] / 0.4, q[:, 8] / -0.8
    assert np.all((s_hip >= 0.5 - 1e-6) & (s_hip <= 1.5 + 1e-6)) and s_hip.std() > 0.1
    assert np.all(q[:, 8] >= -2 - 1e-6) and np.all(q[:, 7] <= 1 + 1e-6)
    # the written COM velocity (world lin + ang) is what MuJoCo reports for the base body
    base = _bid(m, "base")
    lin_all, ang_all = [], []
    for e in range(4):
        lin, ang, d = _com_vel(m, sim, e, base)
        lin_all.append(lin); ang_all.append(ang)
        # recover the sampled COM velocity from qvel with the COM offset: consistent with Isaac's root_com_vel
        R = d.xmat[base].reshape(3, 3)
        w = R @ v[e, 3:6]
        assert np.all(np.abs(w) <= 0.5 + 1e-5)
        np.testing.assert_allclose(ang, w, atol=1e-5)
        assert np.all(np.abs(lin) <= 0.5 + 1e-5), lin
    # COM (not origin) velocity lies in the sampled box even though the origin is 3 cm from the COM
    assert np.std(np.array(lin_all)) > 0.05


def test_event_push_adds_com_velocity(evsim):
    m, sim, ctx, ev = evsim
    base = _bid(m, "base")
    before = [_com_vel(m, sim, e, base)[:2] for e in range(3)]
    g = torch.Generator(device="mps").manual_seed(5)
    ctx.gen = g
    ev.push_by_setting_velocity(ctx, torch.tensor([0, 1, 2]), velocity_range={"x": (0.7, 0.7), "yaw": (0.3, 0.3)})
    torch.mps.synchronize(); sim.forward(); sim.synchronize()
    for e in range(3):
        lin, ang, _ = _com_vel(m, sim, e, base)
        np.testing.assert_allclose(lin - before[e][0], [0.7, 0, 0], atol=1e-5)
        np.testing.assert_allclose(ang - before[e][1], [0, 0, 0.3], atol=1e-5)


def test_event_model_randomization(evsim):
    m, sim, ctx, ev = evsim
    base, thigh = _bid(m, "base"), _bid(m, "thigh")
    ev.randomize_rigid_body_mass(ctx, None, ev.AssetCfg(body_names="base"), mass_distribution_params=(-1.0, 3.0), operation="add")
    ev.randomize_rigid_body_com(ctx, None, com_range={"x": (-0.05, 0.05), "z": (-0.01, 0.01)}, asset_cfg=ev.AssetCfg(body_names="base"))
    ev.randomize_rigid_body_material(ctx, None, None, static_friction_range=(0.4, 1.2), dynamic_friction_range=(0.3, 0.9),
                                     restitution_range=(0, 0), num_buckets=16)
    ev.randomize_actuator_gains(ctx, None, ev.AssetCfg(joint_names="hip"), stiffness_distribution_params=(0.5, 2.0),
                                damping_distribution_params=(0.8, 1.2), operation="scale")
    ev.randomize_joint_parameters(ctx, None, None, friction_distribution_params=(0.0, 0.2), armature_distribution_params=(0.01, 0.02))
    torch.mps.synchronize(); sim.recompute_constants(); sim.synchronize()
    mass = sim.tm.body_mass.cpu().numpy(); inertia = sim.tm.body_inertia.cpu().numpy()
    assert np.all((mass[:, base] >= m.body_mass[base] - 1) & (mass[:, base] <= m.body_mass[base] + 3)) and mass[:, base].std() > 0.5
    np.testing.assert_allclose(mass[:, thigh], m.body_mass[thigh])
    np.testing.assert_allclose(inertia[:, base], m.body_inertia[base][None] * (mass[:, base] / m.body_mass[base])[:, None], rtol=1e-5)
    ipos = sim.tm.body_ipos.cpu().numpy()
    off = ipos[:, base] - m.body_ipos[base]
    assert np.all(np.abs(off[:, 0]) <= 0.05 + 1e-6) and np.allclose(off[:, 1], 0) and np.all(np.abs(off[:, 2]) <= 0.01 + 1e-6)
    fr = sim.tm.geom_friction.cpu().numpy()[:, :, 0]
    robot_geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] != 0]
    vals = np.unique(fr[:, robot_geoms])
    assert 2 <= len(vals) <= 16 and vals.min() >= 0.4 - 1e-6 and vals.max() <= 1.2 + 1e-6
    np.testing.assert_allclose(fr[:, 0], m.geom_friction[0, 0])             # the floor is not part of the asset
    np.testing.assert_allclose(ctx.material["values"].cpu().numpy()[:, robot_geoms, 0], fr[:, robot_geoms])
    g = sim.tm.actuator_gainprm.cpu().numpy(); b = sim.tm.actuator_biasprm.cpu().numpy()
    assert np.all((g[:, 0, 0] >= 15 - 1e-4) & (g[:, 0, 0] <= 60 + 1e-4)) and np.allclose(b[:, 0, 1], -g[:, 0, 0])
    assert np.all((-b[:, 0, 2] >= 0.8 - 1e-5) & (-b[:, 0, 2] <= 1.2 + 1e-5))
    np.testing.assert_allclose(g[:, 1, 0], 20)                              # knee untouched
    fl = sim.tm.dof_frictionloss.cpu().numpy()
    assert np.all(fl[:, 6:] <= 0.2) and np.allclose(fl[:, :6], 0)
    # the randomized physics runs: heavier bases sink the same free fall (no NaN), friction differs per world
    for _ in range(20):
        sim.step()
    sim.synchronize()
    assert np.isfinite(sim.t.qpos.cpu().numpy()).all()


def test_event_external_wrench_body_frame(evsim):
    m, sim, ctx, ev = evsim
    base = _bid(m, "base")
    sim.synchronize()
    ev.apply_external_force_torque(ctx, None, force_range=(2.0, 2.0), torque_range=(0.0, 0.0), asset_cfg=ev.AssetCfg(body_names="base"))
    torch.mps.synchronize()
    for _ in range(5):
        sim.step()
    sim.synchronize()
    x = sim.t.xfrc_applied.cpu().numpy(); R = sim.t.xmat.cpu().numpy().reshape(64, -1, 3, 3)
    want = np.einsum("nij,j->ni", R[:, base], [2.0, 2.0, 2.0])
    np.testing.assert_allclose(x[:, base, :3], want, atol=1e-5)
    assert np.allclose(x[:, _bid(m, "thigh")], 0)


def test_event_manager_interval_timers(evsim):
    m, sim, ctx, ev = evsim
    calls = []

    def term(c, env_ids, tag):
        calls.append((tag, None if env_ids is None else env_ids.cpu().numpy().copy()))
    mgr = ev.EventManager(ctx, {"push": ev.EventTerm(term, "interval", {"tag": "p"}, interval_range_s=(1.0, 2.0)),
                                "startup": ev.EventTerm(term, "startup", {"tag": "s"})}, step_dt=0.02)
    mgr.apply("startup")
    for _ in range(100):             # 2 s: every env fires at least once, none more than twice
        mgr.apply("interval")
    fired = np.concatenate([c[1] for c in calls if c[0] == "p"])
    counts = np.bincount(fired, minlength=64)
    assert calls[0][0] == "s" and counts.min() >= 1 and counts.max() <= 2
    assert len({len(c[1]) for c in calls if c[0] == "p"}) > 1     # per-env timers, not one global tick


def test_event_reset_root_velocity_is_com_velocity(evsim):
    """Isaac writes the root COM velocity (world); with fixed ranges MuJoCo's COM velocity equals it
    exactly although the base COM is 3 cm from the body origin."""
    m, sim, ctx, ev = evsim
    vr = {"x": (0.3, 0.3), "y": (-0.2, -0.2), "z": (0.1, 0.1), "roll": (0.4, 0.4), "pitch": (-0.6, -0.6), "yaw": (0.8, 0.8)}
    ev.reset_root_state_uniform(ctx, None, pose_range={"yaw": (-3.0, 3.0), "roll": (-0.3, 0.3)}, velocity_range=vr)
    torch.mps.synchronize(); sim.forward(); sim.synchronize()
    base = _bid(m, "base")
    for e in range(4):
        lin, ang, _ = _com_vel(m, sim, e, base)
        np.testing.assert_allclose(lin, [0.3, -0.2, 0.1], atol=1e-5)
        np.testing.assert_allclose(ang, [0.4, -0.6, 0.8], atol=1e-5)

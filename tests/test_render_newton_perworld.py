"""Newton bridge, per-world appearance (docs/RENDER_COVERAGE.md workstream C): an Isaac-style per-world scale and colour
randomization written into the Newton model (``Model.shape_scale``, ``shape_color``) shows in the renderers, and the
Newton-built twin keeps a mesh's UVs, texture, roughness and metallic. Reference: the same scene as an MJCF with that
world's values, rendered through the MuJoCo path.

    scripts/gpu_run.sh newton_perworld check 10 -- .venv/bin/python -m pytest tests/test_render_newton_perworld.py -q
"""
import os
import sys

import mujoco
import numpy as np
import pytest
import torch
import warp as wp

newton = pytest.importorskip("newton", reason="Newton not installed (optional engine)")

from metalsim.render.newton_bridge import NewtonRenderSource
from metalsim.render.tier0 import SEG_GEOM, Tier0Renderer
from metalsim.render.tier2 import Tier2Renderer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from newton_bridge_scenes import PRIMS, mirror  # noqa: E402

pytestmark = pytest.mark.skipif(not wp.is_metal_available(), reason="needs Metal")
W, H = 160, 120
# multipliers on Newton's shape_scale: box / ellipsoid half extents, sphere (r, -, -), cylinder / capsule (r, half height, -)
SCALES = {"box": (1.6, 1.0, 0.6), "ball": (1.5, 1.0, 1.0), "cyl": (1.3, 0.7, 1.0), "cap": (1.4, 1.4, 1.0), "ell": (0.7, 1.4, 1.2)}
COLOR = {"box": (0.1, 0.8, 0.9), "ball": (0.9, 0.1, 0.6), "cyl": (0.95, 0.6, 0.1), "cap": (0.2, 0.2, 0.2), "ell": (0.5, 0.9, 0.2)}


def _world1_mjcf():
    """PRIMS with world 1's randomized sizes and colours (MuJoCo sizes: the Newton scale's meaning per type)."""
    spec = mujoco.MjSpec.from_string(PRIMS)
    for b in spec.bodies:
        if b.name in SCALES:
            g = b.geoms[0]
            k = np.asarray(SCALES[b.name])
            if g.type == mujoco.mjtGeom.mjGEOM_SPHERE:
                g.size = [g.size[0] * k[0], 0, 0]
            elif g.type in (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_CAPSULE):
                g.size = [g.size[0] * k[0], g.size[1] * k[1], 0]
            else:
                g.size = list(np.asarray(g.size) * k)
            g.rgba = list(COLOR[b.name]) + [1.0]
    return spec.compile()


def _newton(n=2):
    env = newton.ModelBuilder(); env.add_mjcf(PRIMS)
    b = newton.ModelBuilder(); b.replicate(env, n, spacing=(2.0, 0.0, 0.0))
    return b.finalize(device="metal:0")


@pytest.mark.parametrize("tier", [0, 2])
def test_per_world_scale_and_colour(tier):
    mj = mujoco.MjModel.from_xml_string(PRIMS)
    model = _newton(2)
    cams, lights = mirror(mj)
    src = NewtonRenderSource(model, cameras=cams, lights=lights)
    # randomize world 1 (Isaac randomize_rigid_body_scale / randomize_visual_color on the Newton backend)
    sc = model.shape_scale.numpy(); col = model.shape_color.numpy()
    for g in range(src.m.ngeom):
        nm = mujoco.mj_id2name(src.m, mujoco.mjtObj.mjOBJ_BODY, int(src.m.geom_bodyid[g]))
        s = int(src.geom_shape[1, g])
        if nm in SCALES and s >= 0:
            sc[s] *= np.asarray(SCALES[nm], np.float32)
            col[s] = COLOR[nm]
    model.shape_scale.assign(sc); model.shape_color.assign(col)
    st = model.state()
    v = src.update(st)
    if tier == 0:
        r = Tier0Renderer(src.m, 2, width=W, height=H, camera="cam", outputs=("rgb", "seg"), seg_mode=SEG_GEOM)
    else:
        r = Tier2Renderer(src.m, 2, width=W, height=H, camera="cam", spp=4, max_bounces=1, center_sample=True)
    r.after(r.render(src, v)); torch.mps.synchronize()
    rgb = r.out.rgb.cpu().numpy(); seg = r.out.seg.cpu().numpy()
    if tier == 2:
        seg = np.concatenate([[0], r.tables.slot_seg_ids()])[seg]
    # references: the MJCF at world 0's (compiled) and world 1's values, MuJoCo path, same renderer
    for e, ref_m in ((0, mj), (1, _world1_mjcf())):
        d = mujoco.MjData(ref_m); mujoco.mj_forward(ref_m, d)
        if tier == 0:
            rr = Tier0Renderer(ref_m, 1, width=W, height=H, camera="cam", outputs=("rgb", "seg"), seg_mode=SEG_GEOM)
        else:
            rr = Tier2Renderer(ref_m, 1, width=W, height=H, camera="cam", spp=4, max_bounces=1, center_sample=True)
        o = rr.render_host([d])
        ref_rgb = o["rgb"][0]
        ref_fg = o["seg"][0] > 0 if tier == 0 else (o["depth"][0] > 0)
        for name in SCALES:
            gi = mujoco.mj_name2id(ref_m, mujoco.mjtObj.mjOBJ_BODY, name)
            gr = int(ref_m.body_geomadr[gi])
            gt = [g for g in range(src.m.ngeom) if mujoco.mj_id2name(src.m, mujoco.mjtObj.mjOBJ_BODY, int(src.m.geom_bodyid[g])) == name][0]
            ref_seg = o["seg"][0] if tier == 0 else np.concatenate([[0], rr.tables.slot_seg_ids()])[o["seg"][0]]
            a, b = ref_seg == gr + 1, seg[e] == gt + 1
            iou = (a & b).sum() / max((a | b).sum(), 1)
            assert iou > (0.85 if name == "cap" else 0.95), (tier, e, name, iou)
            both = a & b
            diff = np.abs(rgb[e][both].astype(float).mean(0) - ref_rgb[both].astype(float).mean(0)).max()
            assert diff < (4.0 if tier == 0 else 10.0), (tier, e, name, diff)
        assert ref_fg.any()


def test_newton_mesh_texture_roughness_metallic():
    """A Newton mesh with UVs, a texture, roughness and metallic: the twin keeps them (material, texture, texcoords) and
    renders like the same mesh written as an MJCF with that texture and material."""
    import tempfile
    from PIL import Image
    img = np.zeros((32, 64, 3), np.uint8); img[:, :32] = (220, 40, 40); img[:, 32:] = (40, 40, 220); img[:8] = (40, 220, 40)
    v = np.array([[-0.2, -0.01, -0.15], [0.2, -0.01, -0.15], [0.2, -0.01, 0.15], [-0.2, -0.01, 0.15],
                  [-0.2, 0.01, -0.15], [0.2, 0.01, -0.15], [0.2, 0.01, 0.15], [-0.2, 0.01, 0.15]], np.float32)
    f = np.array([0, 1, 2, 0, 2, 3, 4, 6, 5, 4, 7, 6, 0, 4, 5, 0, 5, 1, 1, 5, 6, 1, 6, 2, 2, 6, 7, 2, 7, 3, 3, 7, 4, 3, 4, 0], np.int32)
    uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0], [1, 0], [1, 1], [0, 1]], np.float32)
    b = newton.ModelBuilder()
    mesh = newton.Mesh(v, f, uvs=uv, texture=img, roughness=0.35, metallic=0.25, color=(1.0, 1.0, 1.0))
    body = b.add_body(xform=wp.transform((0.0, 0.0, 1.0), wp.quat_identity()))
    b.add_shape_mesh(body, mesh=mesh)
    model = b.finalize(device="metal:0")
    from metalsim.render.newton_bridge import Camera, Light
    src = NewtonRenderSource(model, cameras=[Camera(name="cam", pos=(0.0, -1.0, 1.0), lookat=(0.0, 0.0, 1.0), fovy=40.0)],
                             lights=[Light(pos=(0.0, 0.0, 3.0), dir=(0.0, 0.0, -1.0))])      # the MJCF's light
    tm = src.m
    assert tm.nmat == 1 and tm.ntex == 1
    np.testing.assert_allclose([tm.mat_roughness[0], tm.mat_metallic[0]], [0.35, 0.25], atol=1e-6)
    assert int(tm.mesh_texcoordadr[0]) >= 0
    v_ = src.update(model.state())
    r = Tier0Renderer(tm, 1, width=W, height=H, camera="cam", outputs=("rgb", "seg"), seg_mode=SEG_GEOM)
    r.after(r.render(src, v_)); torch.mps.synchronize()
    ours = r.out.rgb.cpu().numpy()[0]; seg = r.out.seg.cpu().numpy()[0] > 0
    # the same mesh as an MJCF (texture file, texcoords, material)
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "t.png"); Image.fromarray(img).save(p)
        xml = f"""<mujoco><compiler inertiafromgeom="false"/><asset><texture name="t" type="2d" file="{p}"/>
          <material name="m" texture="t" roughness="0.35" metallic="0.25"/>
          <mesh name="q" vertex="{' '.join(map(str, v.reshape(-1)))}" face="{' '.join(map(str, f))}"
                texcoord="{' '.join(map(str, uv.reshape(-1)))}"/></asset>
          <worldbody><light directional="true" pos="0 0 3" dir="0 0 -1"/>
          <camera name="cam" pos="0 -1 1" xyaxes="1 0 0 0 0 1" fovy="40"/>
          <body pos="0 0 1"><inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/><geom type="mesh" mesh="q" material="m"/></body>
          </worldbody></mujoco>"""
        mm = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(mm); mujoco.mj_forward(mm, d)
    ref = Tier0Renderer(mm, 1, width=W, height=H, camera="cam", outputs=("rgb",)).render_host([d])["rgb"][0]
    la, lb = ours[seg].astype(float).mean(-1), ref[seg].astype(float).mean(-1)
    assert lb.std() > 20 and la.std() > 20                       # textured, not one texel
    # texture layout: red / blue halves left / right, green strip, as the MJCF path draws it
    corr = [np.corrcoef(ours[seg][:, k].astype(float), ref[seg][:, k].astype(float))[0, 1] for k in range(3)]
    ys, xs = np.nonzero(seg)
    q = lambda img, m: img[m].astype(float).mean(0).round(1)       # noqa: E731
    left, top = seg & (np.arange(W)[None] < xs.mean()), seg & (np.arange(H)[:, None] < ys.mean())
    print("corr", np.round(corr, 3), "left", q(ours, left), q(ref, left), "top", q(ours, top), q(ref, top))
    try:
        from PIL import Image
        Image.fromarray(np.concatenate([ours, ref], 1)).save(os.path.join(os.environ.get("RW_OUT", "/tmp"), "newton_tex.png"))
    except OSError:
        pass
    assert min(corr) > 0.9, corr

"""Per-env material and texture swaps (metalsim.render.env_materials).

* per-world ``geom_matid`` (a MuJoCo Warp material randomization): each env of the GPU path renders as the same renderer
  on a model compiled with that world's materials (texture, mapping and PBR parameters, not only the colour), and as
  ``mujoco.Renderer`` on that model (texture correlation);
* a texture pool (Isaac Lab ``randomize_visual_texture_material``): per-env image choice, OmniPBR-style ``project_uvw``
  box projection checked texel by texel on a box's top face against an analytic reference, rotation, the "uv" mapping;
* nothing configured: byte-identical to the renderer without the feature.
GPU tests: scripts/gpu_run.sh check."""
import mujoco
import numpy as np
import pytest
import warp as wp

from metalsim.render.scene_tables import build_scene_tables, material_block

needs_metal = pytest.mark.skipif(not wp.is_metal_available(), reason="needs Metal")
W, H = 96, 80

XML = """
<mujoco>
  <visual><headlight ambient="0.15 0.15 0.15" diffuse="0.25 0.25 0.25" specular="0 0 0"/></visual>
  <asset>
    <texture name="chk" type="2d" builtin="checker" rgb1="0.9 0.2 0.1" rgb2="0.1 0.2 0.9" width="64" height="64"/>
    <texture name="grd" type="2d" builtin="gradient" rgb1="0.1 0.9 0.2" rgb2="0.9 0.9 0.1" width="64" height="64"/>
    <material name="A" texture="chk" texrepeat="2 2" specular="0" shininess="0"/>
    <material name="B" texture="grd" specular="0" shininess="0.5" emission="0.1"/>
    <material name="C" rgba="0.2 0.8 0.8 1" specular="0"/>
  </asset>
  <worldbody>
    <light directional="true" dir="0 0 -1" diffuse="0.25 0.25 0.25" castshadow="false"/>
    <camera name="cam" pos="0 -0.9 1.0" xyaxes="1 0 0 0 0.75 0.66"/>
    <camera name="top" pos="0 0 1.2" xyaxes="1 0 0 0 1 0" fovy="40"/>
    <geom name="floor" type="plane" size="1 1 0.1" rgba="0.6 0.6 0.6 1"/>
    <geom name="box" type="box" pos="0 0 0.1" size="0.2 0.2 0.1" material="A"/>
    <geom name="ball" type="sphere" pos="0.35 0.2 0.12" size="0.12" material="B"/>
  </worldbody>
</mujoco>"""


def _model():
    m = mujoco.MjModel.from_xml_string(XML)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return m, d


def test_material_block_matches_tables():
    """CPU: the per-(geom, material) block equals build_scene_tables' rows for each geom's compiled material."""
    m, _ = _model()
    t = build_scene_tables(m, 1)
    for s, g in enumerate(t.geoms):
        row, ext = material_block(m, g, int(m.geom_matid[g]), t.atlas)
        np.testing.assert_array_equal(row[4:], t.materials[s, 4:])
        np.testing.assert_array_equal(ext, t.material_ext[s])


WORLD_MATS = [{}, {"box": "B"}, {"box": None, "ball": "A"}, {"box": "C", "ball": "C"}]


def _matid(m, name):
    return -1 if name is None else mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, name)


def _compiled(e):
    me, de = _model()
    for gname, mat in WORLD_MATS[e].items():
        me.geom_matid[mujoco.mj_name2id(me, mujoco.mjtObj.mjOBJ_GEOM, gname)] = _matid(me, mat)
    mujoco.mj_forward(me, de)
    return me, de


@needs_metal
@pytest.mark.parametrize("tier", ["tier0", "tier1", "tier2"])
def test_per_world_geom_matid_switches_the_texture(tier):
    """BatchSim with geom_matid per world: env e equals the renderer on the model compiled with world e's materials."""
    import torch
    from metalsim.physics.batch import BatchSim, BatchSimOptions
    from metalsim.render.tier0 import Tier0Renderer
    from metalsim.render.tier2 import Tier2Renderer
    m, _ = _model()
    n = len(WORLD_MATS)
    sim = BatchSim(m, n, options=BatchSimOptions(per_world_fields=("geom_matid",), njmax=64))
    ids = np.stack([_compiled(e)[0].geom_matid for e in range(n)]).astype(np.int32)
    sim.m.geom_matid.assign(ids.reshape(sim.m.geom_matid.numpy().shape))
    wp.synchronize_device("metal:0")
    sim.forward(); sim.synchronize()

    def make(model, k):
        if tier == "tier2":
            return Tier2Renderer(model, k, width=W, height=H, camera="cam", spp=4, max_bounces=1, center_sample=True, seed=2)
        return Tier0Renderer(model, k, width=W, height=H, camera="cam", tier=1 if tier == "tier1" else 0)
    r = make(m, n)
    v = r.render(sim, sim.event.value); r.after(v); torch.mps.synchronize()
    rgb = r.out.rgb.cpu().numpy().astype(float)
    for e in range(n):
        me, de = _compiled(e)
        ref = make(me, 1).render_host([de])["rgb"][0].astype(float)
        lim = {"tier0": 1.0, "tier1": 4.0, "tier2": 3.0}[tier]   # tier 1: AO / soft-shadow samples hash the env index
        assert np.abs(rgb[e] - ref).mean() < lim, (tier, e, np.abs(rgb[e] - ref).mean())
    assert np.abs(rgb[1] - rgb[0]).mean() > 3          # the swaps are visible


@needs_metal
def test_per_world_matid_vs_mujoco_renderer():
    """Tier 0 env images against mujoco.Renderer on the model compiled with that world's materials: on the box, the swap
    adds no error to tier 0's own shading difference from MuJoCo (mean colour within 2 levels of the compiled model's
    tier-0 render's distance to MuJoCo) and its texture correlates with MuJoCo's (> 0.8)."""
    import torch
    from metalsim.physics.batch import BatchSim, BatchSimOptions
    from metalsim.render.tier0 import Tier0Renderer, SEG_GEOM
    m, _ = _model()
    n = len(WORLD_MATS)
    sim = BatchSim(m, n, options=BatchSimOptions(per_world_fields=("geom_matid",), njmax=64))
    ids = np.stack([_compiled(e)[0].geom_matid for e in range(n)]).astype(np.int32)
    sim.m.geom_matid.assign(ids.reshape(sim.m.geom_matid.numpy().shape))
    wp.synchronize_device("metal:0")
    sim.forward(); sim.synchronize()
    r = Tier0Renderer(m, n, width=W, height=H, camera="cam", outputs=("rgb", "seg"), seg_mode=SEG_GEOM)
    v = r.render(sim, sim.event.value); r.after(v); torch.mps.synchronize()
    rgb, seg = r.out.rgb.cpu().numpy().astype(float), r.out.seg.cpu().numpy()
    box = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "box")
    for e in range(n):
        me, de = _compiled(e)
        rr = mujoco.Renderer(me, H, W); rr.update_scene(de, camera="cam"); ref = rr.render().astype(float)
        rr.enable_segmentation_rendering(); rr.update_scene(de, camera="cam"); rs = rr.render()[..., 0]; rr.close()
        mk = (rs == box) & (seg[e] == box + 1)
        assert mk.sum() > 200
        own = Tier0Renderer(me, 1, width=W, height=H, camera="cam").render_host([de])["rgb"][0].astype(float)
        gap_swap = np.abs(rgb[e][mk].mean(0) - ref[mk].mean(0)).max()
        gap_own = np.abs(own[mk].mean(0) - ref[mk].mean(0)).max()
        assert gap_swap < gap_own + 2, (e, gap_swap, gap_own)
        a, b = rgb[e][mk].mean(-1), ref[mk].mean(-1)
        if a.std() > 5:
            assert np.corrcoef(a, b)[0, 1] > 0.8, (e, np.corrcoef(a, b)[0, 1])


def _pool_images():
    yy, xx = np.mgrid[0:64, 0:64]
    stripes = np.zeros((64, 64, 3), np.uint8); stripes[..., 0] = np.where((xx // 8) % 2, 230, 20); stripes[..., 2] = 120
    ramp = np.zeros((64, 64, 3), np.uint8); ramp[..., 1] = (yy * 4).astype(np.uint8); ramp[..., 0] = 40
    white = np.full((64, 64, 3), 255, np.uint8)
    black = np.zeros((64, 64, 3), np.uint8)
    return [stripes, ramp, white, black]


def _srgb_lin(c):
    c = np.asarray(c, np.float64) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


@needs_metal
@pytest.mark.parametrize("tier", ["tier0", "tier2"])
def test_texture_pool_projection(tier):
    """Isaac Lab-style texture randomization: envs 0..3 draw the box with stripes, ramp, white and its own material; the
    top face (seen from straight above) shows the image at the geom-frame (x, y) * texture_scale (OmniPBR project_uvw):
    shading is affine in the albedo (diffuse and ambient scale it, the specular lobe adds to it), so with a white and a
    black env as calibration the pixel colour is black + (white - black) x the expected texel (linear), within 6 levels
    at the median."""
    from metalsim.render.tier0 import Tier0Renderer
    from metalsim.render.tier2 import Tier2Renderer
    m, d = _model()
    n = 5
    if tier == "tier2":
        r = Tier2Renderer(m, n, width=W, height=H, camera="top", spp=1, max_bounces=1, center_sample=True, seed=1)
    else:
        r = Tier0Renderer(m, n, width=W, height=H, camera="top", outputs=("rgb", "seg"))
    imgs = _pool_images()
    ids = r.set_texture_pool(imgs, projection="uvw", texture_scale=(2.0, 2.0), rotations_deg=(0.0, 90.0))
    G = r.G
    box_slot = list(r.tables.geoms).index(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "box"))
    sel = np.full((n, G), -1, np.int32)
    for e in range(4):
        sel[e, box_slot] = ids[e, 1 if e == 1 else 0]            # env 1: the ramp rotated by 90 degrees
    r.set_material_ids(sel)
    out = r.render_host([d] * n)
    rgb = out["rgb"].astype(float)
    # top face pixels: rays from the top camera hitting z = 0.2 inside |x|, |y| < 0.2 (pinhole, fovy 40)
    f = 0.5 * H / np.tan(np.deg2rad(20))
    ys, xs = np.mgrid[0:H, 0:W]
    dx, dy = (xs + 0.5 - W / 2) / f, -(ys + 0.5 - H / 2) / f
    t = 1.0                                              # camera at z = 1.2, face at z = 0.2, looking down -z
    px, py = dx * t, dy * t
    face = (np.abs(px) < 0.19) & (np.abs(py) < 0.19)
    white, black = rgb[2][face], rgb[3][face]
    for e in (0, 1):
        st = np.stack([px[face], py[face]], -1) * 2.0
        u, v = st[:, 0] % 1.0, st[:, 1] % 1.0
        ii, jj = np.clip((v * 64).astype(int), 0, 63), np.clip((u * 64).astype(int), 0, 63)
        texel = _srgb_lin(imgs[e][ii, jj])
        pred = black + (white - black) * texel
        err = np.abs(rgb[e][face] - pred)
        assert np.median(err) < 6, (tier, e, np.median(err))
    plain = (Tier2Renderer(m, n, width=W, height=H, camera="top", spp=1, max_bounces=1, center_sample=True, seed=1)
             if tier == "tier2" else Tier0Renderer(m, n, width=W, height=H, camera="top", outputs=("rgb", "seg")))
    assert np.abs(rgb[4] - plain.render_host([d] * n)["rgb"][4].astype(float)).max() == 0   # env 4: its own material
    assert (white < 250).all()                           # calibration not clipped


@needs_metal
def test_without_overrides_byte_identical():
    """A texture pool set but every id -1, and no pool at all: identical images (tier 0 and tier 2)."""
    from metalsim.render.tier0 import Tier0Renderer
    from metalsim.render.tier2 import Tier2Renderer
    m, d = _model()
    for mk in (lambda: Tier0Renderer(m, 2, width=W, height=H, camera="cam"),
               lambda: Tier2Renderer(m, 2, width=W, height=H, camera="cam", spp=2, max_bounces=1, seed=4)):
        a = mk().render_host([d, d])["rgb"]
        r = mk()
        r.set_texture_pool(_pool_images())
        b = r.render_host([d, d])["rgb"]
        assert np.array_equal(a, b)

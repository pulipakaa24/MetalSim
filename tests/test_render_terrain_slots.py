"""Rough terrain in the renderers: the heightfield as its exact surface and the per-world box slots of
``terrain_collision="boxes_local"`` (metalsim.learn.terrain.BoxWindow) at their per-world size and position.

One env looks straight down at the riser of the inverted-stairs pit of Isaac's rough terrain (seed 0, row 3, column 4:
the cell of tests/test_terrain.py::test_step_edge_contacts_match_mujoco_c). The depth along the image row through the
riser must follow the stairs, pixel by pixel, as MuJoCo C's ``mj_ray`` sees them on the model with all 14,080 boxes;
with the slots off (``terrain_slots=False``, the previous behaviour) the same row shows the heightfield lowered 5 cm
under the boxes (flat), which is what ``mj_ray`` sees on the window model with its slots parked."""
import mujoco
import numpy as np
import pytest
import torch
import warp as wp

pytestmark = pytest.mark.skipif(not wp.is_metal_available(), reason="needs Metal")

ROW, COL = 3, 4
CAM_UP = 2.0          # camera height above the pit platform (m)
W = H = 96


def _model(hf, mode_geoms=True):
    from metalsim.learn.terrain import add_terrain_geoms, terrain_boxes
    B = terrain_boxes(hf["generator"]); cb = B[(B[:, 6] == ROW) & (B[:, 7] == COL)]
    plat = cb[-1]; x_riser = plat[0] + plat[3]; z_top = plat[2] + plat[5]
    spec = mujoco.MjSpec()
    h = spec.add_hfield(); h.name = "terrain"; h.nrow, h.ncol = hf["nrow"], hf["ncol"]; h.size = hf["size"]
    h.userdata = hf["data"].reshape(-1)
    g = spec.worldbody.add_geom(); g.name = "ground"; g.type = mujoco.mjtGeom.mjGEOM_HFIELD; g.hfieldname = "terrain"
    g.pos = [0, 0, hf["zmin"]]; g.rgba = [0.55, 0.5, 0.45, 1]
    add_terrain_geoms(spec, hf, g)
    b = spec.worldbody.add_body(); b.name = "probe"; b.add_freejoint()      # BoxWindow centres the window on qpos[0:2]
    pg = b.add_geom(); pg.type = mujoco.mjtGeom.mjGEOM_SPHERE; pg.size = [0.01, 0, 0]; pg.group = 4
    pg.contype = 0; pg.conaffinity = 0; pg.mass = 1.0
    cx, cy = x_riser + 0.3, plat[1]
    c = spec.worldbody.add_camera(); c.name = "down"; c.pos = [cx, cy, z_top + CAM_UP]; c.fovy = 60.0   # identity: looks along -z
    l = spec.worldbody.add_light(); l.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL; l.dir = [0.3, 0.2, -1.0]; l.pos = [cx, cy, 5]
    return spec.compile(), dict(x_riser=float(x_riser), z_top=float(z_top), cx=float(cx), cy=float(cy))


def _rays(m, geo, bodyexclude):
    """mj_ray z-depth along the image's middle row (tier 0's pinhole: pixel centres, camera looks along -z, +x right)."""
    d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    f = H / (2 * np.tan(np.deg2rad(60.0) / 2))
    pos = np.array([geo["cx"], geo["cy"], geo["z_top"] + CAM_UP])
    i = H // 2
    out = np.zeros(W); gid = np.zeros(1, np.int32)
    for j in range(W):
        v = np.array([(j + 0.5 - W / 2) / f, (H / 2 - i - 0.5) / f, -1.0])
        dist = mujoco.mj_ray(m, d, pos, v / np.linalg.norm(v), None, 1, bodyexclude, gid)
        out[j] = dist / np.linalg.norm(v)       # distance along the ray -> depth along the optical axis
    return out


@pytest.fixture(scope="module")
def scene():
    from metalsim.learn.terrain import isaac_rough_terrain, BoxWindow
    from metalsim.physics.batch import BatchSim, BatchSimOptions
    wp.config.quiet = True
    hf = isaac_rough_terrain(seed=0, collision="boxes_local")
    m, geo = _model(hf)
    sim = BatchSim(m, 1, options=BatchSimOptions(per_world_fields=BoxWindow.FIELDS, njmax=64))
    q = sim.d.qpos.numpy(); q[0, :3] = (geo["cx"], geo["cy"], geo["z_top"] + 0.5); q[0, 3:7] = (1, 0, 0, 0)
    sim.set_state(q[0]); win = BoxWindow(sim, hf); win.launch(); sim.synchronize()
    assert int(win.overflow.numpy()[0]) == 0
    probe = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "probe")
    ref_boxes = _rays(_model(isaac_rough_terrain(seed=0, collision="boxes"))[0], geo, probe)   # C: all 14,080 boxes
    ref_hfield = _rays(m, geo, probe)                                                           # C: slots parked
    return dict(sim=sim, m=m, geo=geo, ref_boxes=ref_boxes, ref_hfield=ref_hfield)


def _render(sc, tier, **kw):
    sim, m = sc["sim"], sc["m"]
    if tier == 0:
        from metalsim.render.tier0 import Tier0Renderer, SEG_GEOM
        r = Tier0Renderer(m, 1, width=W, height=H, camera="down", outputs=("depth", "seg"), seg_mode=SEG_GEOM, **kw)
    else:
        from metalsim.render.tier2 import Tier2Renderer
        r = Tier2Renderer(m, 1, width=W, height=H, camera="down", spp=1, max_bounces=1, center_sample=True, **kw)
    v = r.render(sim, sim.event.value); r.after(v); torch.mps.synchronize()
    depth = r.out.depth[0].cpu().numpy()[H // 2].astype(np.float64)
    seg = r.out.seg[0].cpu().numpy()[H // 2] if tier == 0 else None
    return depth, seg


def test_reference_row_crosses_risers(scene):
    """Sanity of the protocol: along the row, the stairs (C, all boxes) have several risers of > 3 cm, while the
    window model with parked slots is the flat lowered heightfield."""
    rb, rh = scene["ref_boxes"], scene["ref_hfield"]
    assert (np.abs(np.diff(rb)) > 0.03).sum() >= 3
    assert np.ptp(rh) < 0.01 and np.all(rh > rb - 1e-6)


@pytest.mark.parametrize("tier", [0, 2])
def test_slot_boxes_render_the_stairs(scene, tier):
    rb, rh = scene["ref_boxes"], scene["ref_hfield"]
    new, seg = _render(scene, tier)
    old, seg_old = _render(scene, tier, terrain_slots=False)
    err_new = np.abs(new - rb); err_old_vs_h = np.abs(old - rh)
    print(f"tier {tier}: |new - C(boxes)| median {np.median(err_new) * 1e3:.2f} mm, <1 cm {np.mean(err_new < 0.01):.3f}; "
          f"|old - C(hfield only)| median {np.median(err_old_vs_h) * 1e3:.2f} mm; |old - C(boxes)| median {np.median(np.abs(old - rb)) * 1e3:.1f} mm")
    # new: the true stairs, pixel for pixel (a pixel straddling a riser may land on either side)
    assert np.median(err_new) < 0.003 and np.mean(err_new < 0.01) > 0.95
    assert (np.abs(np.diff(new)) > 0.03).sum() == (np.abs(np.diff(rb)) > 0.03).sum()     # same step silhouette
    # old behaviour: the lowered heightfield only
    assert np.median(err_old_vs_h) < 0.003 and np.median(np.abs(old - rb)) > 0.04
    if tier == 0:       # id buffer: the row sees slot boxes (geom id + 1 >= first slot + 1), before only the heightfield
        slot0 = mujoco.mj_name2id(scene["m"], mujoco.mjtObj.mjOBJ_GEOM, "tslot0")
        ground = mujoco.mj_name2id(scene["m"], mujoco.mjtObj.mjOBJ_GEOM, "ground")
        assert np.mean(seg >= slot0 + 1) > 0.95 and np.all(seg_old == ground + 1)

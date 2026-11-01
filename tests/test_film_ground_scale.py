"""The film's ground texture lies at its real size (metalsim.render.film.render: ground_texrepeat, add_ground_material).

``LookCfg.ground_tile_m`` is the floor edge in metres one texture image shows; the render stage sets the texuniform
ground material's ``texrepeat`` to 2 / ground_tile_m, because MuJoCo (and the path tracer, which copies its texture
coordinates) lays a texuniform image once per 2 / texrepeat m on planes, boxes, heightfields and object-linear meshes.
Here, per geom type a film textures as ground, the texture coordinate the path tracer samples (the scene tables'
stored coordinates times the material's scale plus its offset, as ``shaders/appearance.h`` ``material_tex``:
fract(uv * par.xy + offset)) must advance by exactly one image per ``ground_tile_m`` metres of world x and y.
CPU only: the scene tables, no Metal."""
from __future__ import annotations

import dataclasses

import mujoco
import numpy as np
import pytest

from metalsim.render.film import cfg as C
from metalsim.render.film import render as RR
from metalsim.render.film.adapters import terrain_box_faces
from metalsim.render.scene_tables import build_scene_tables

TWIN = """
<mujoco>
  <asset><hfield name="hf" nrow="9" ncol="13" size="1.2 0.8 0.05 0.02"/></asset>
  <worldbody>
    <geom name="floor" type="plane" size="0 0 0.05"/>
    <geom name="fplane" type="plane" size="0.9 0.7 0.05" pos="-2 1 0.01" euler="0 0 30"/>
    <geom name="table_top" type="box" size="0.45 0.35 0.02" pos="2 1 -0.02"/>
    <geom name="ground" type="hfield" hfield="hf" pos="2 -2 0.003"/>
  </worldbody>
</mujoco>
"""
# Isaac's terrain boxes as the G1 rough twin merges them (adapters.G1Velocity.twin_spec): rows centre, half extents
BOXES = np.array([[0.0, -2.0, 0.05, 1.0, 0.5, 0.05], [0.6, -0.8, 0.10, 0.6, 0.4, 0.10],
                  [-1.0, -3.0, 0.15, 0.3, 0.8, 0.15]])
GROUND = ("floor", "fplane", "table_top", "ground", "terrain_boxes")


def _texture(tmp_path):
    import imageio.v3 as iio

    p = tmp_path / "tile.png"
    iio.imwrite(p, (np.random.default_rng(0).random((16, 16, 3)) * 255).astype(np.uint8))
    return str(p)


def _look(tmp_path, tile):
    return C.LookCfg(hdri=None, ground_texture=_texture(tmp_path), ground_geoms=GROUND, ground_tile_m=tile)


def _spec():
    spec = mujoco.MjSpec.from_string(TWIN)
    V, F, UV = terrain_box_faces(BOXES)
    me = spec.add_mesh(); me.name = "terrain_boxes"
    me.uservert = V.ravel().tolist(); me.userface = F.ravel().tolist()
    g = spec.worldbody.add_geom(); g.name = "terrain_boxes"; g.type = mujoco.mjtGeom.mjGEOM_MESH
    g.meshname = "terrain_boxes"
    mc = spec.compile()                       # texcoords stored divided by the geom size, as the G1 rough twin does
    size = mc.geom_size[mujoco.mj_name2id(mc, mujoco.mjtObj.mjOBJ_GEOM, "terrain_boxes")]
    me.usertexcoord = (UV / np.maximum(size[:2], 1e-9)).ravel().tolist()
    return spec


def _rates(m, T, d, name):
    """d(s, t) / d(world x, y) of the sampled texture coordinate over the geom's upward faces (least squares; the
    residual is checked): rows s, t; columns x, y."""
    g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
    slot = T.geoms.index(g)
    mesh = T.meshes[T.geom_mesh[slot]]
    idx = T.indices[mesh["i_off"]:mesh["i_off"] + mesh["i_count"]]
    v = T.vertices[idx].astype(np.float64)
    R = d.geom_xmat[g].reshape(3, 3)
    P = (v[:, 0:3] * T.slot_scale[slot]) @ R.T + d.geom_xpos[g]
    up = (v[:, 3:6] @ R.T)[:, 2] > 0.99
    st = v[up, 6:8] * T.materials[slot, 12:14] + T.material_ext[slot, 0:2]
    A = np.c_[P[up, :2], np.ones(int(up.sum()))]
    coef, *_ = np.linalg.lstsq(A, st, rcond=None)
    np.testing.assert_allclose(A @ coef, st, atol=1e-4)          # affine in world x, y: one rate per axis
    return coef[:2].T, slot


@pytest.mark.parametrize("per_world_size", [True, False])
@pytest.mark.parametrize("tile", [1.70, 2.53, 3.0])
def test_ground_period_equals_tile_per_geom_type(tmp_path, tile, per_world_size):
    spec = _spec()
    rep = RR.add_ground_material(spec, _look(tmp_path, tile))
    assert rep == pytest.approx(2.0 / tile)
    m = spec.compile()
    d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    # the path tracer's tables (Tier2Renderer -> RayTracer: plane discs, heightfield skirts, per-world sizes)
    T = build_scene_tables(m, 1, plane_ring=12.0, per_world_size=per_world_size, appearance="mujoco")
    for name in GROUND:
        J, slot = _rates(m, T, d, name)
        # one image per tile metres along both world axes (the finite plane is turned 30 deg: along its own axes)
        if name == "fplane":
            c, s = np.cos(np.pi / 6), np.sin(np.pi / 6)
            J = J @ np.array([[c, -s], [s, c]])
        np.testing.assert_allclose(np.abs(J) * tile, np.eye(2), atol=2e-4, err_msg=name)
    # the infinite plane past the env's cell (env_grid.h grid_plane): uv = local xy * scale + add, then the material
    g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    slot = T.geoms.index(g)
    us, vs = T.plane[slot, 4], T.plane[slot, 6]
    np.testing.assert_allclose(np.abs([us * T.materials[slot, 12], vs * T.materials[slot, 13]]) * tile, 1.0, rtol=1e-5)


def test_build_scene_lays_the_tile(tmp_path):
    """The render stage's scene: the look's ground geoms carry the texuniform material at texrepeat 2 / tile."""
    rec = tmp_path / "rec"; rec.mkdir()
    (rec / "twin.xml").write_text(TWIN)
    model, *_ = RR.build_scene(str(rec), None, _look(tmp_path, 1.70), C.ViewerCfg())
    mat = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, "film_ground")
    np.testing.assert_allclose(model.mat_texrepeat[mat], 2.0 / 1.70, rtol=1e-6)
    assert model.mat_texuniform[mat]
    on = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) for g in range(model.ngeom)
          if model.geom_matid[g] == mat}
    assert on == {"floor", "fplane", "table_top", "ground"}


def test_ground_needs_a_tile_and_a_flat_type(tmp_path):
    spec = mujoco.MjSpec.from_string(TWIN)
    with pytest.raises(ValueError, match="ground_tile_m"):
        RR.add_ground_material(spec, _look(tmp_path, None))
    spec = mujoco.MjSpec.from_string(TWIN.replace('type="box" size="0.45 0.35 0.02"', 'type="sphere" size="0.3"'))
    with pytest.raises(ValueError, match="table_top"):
        RR.add_ground_material(spec, _look(tmp_path, 1.5))


def test_every_film_ground_has_its_size():
    from metalsim.render.film import tasks

    for name, task in tasks.TASKS.items():
        if task.look.ground_texture:
            assert task.look.ground_tile_m and task.look.ground_tile_m > 0, name


def test_cache_key_hashes_the_resolved_texrepeat():
    """Keys hash the texrepeat a tile resolves to, under the field's earlier name: TRON1 (3 m concrete, texrepeat 2/3
    before ground_tile_m existed) keeps its key; without a ground texture the old default 1.0 stands."""
    from metalsim.render.film.tron1 import TRON1

    old = dataclasses.asdict(TRON1.look)
    old.pop("ground_tile_m")
    old["ground_repeat"] = 2.0 / 3.0
    assert RR._key_look(TRON1.look) == old
    bare = C.LookCfg(hdri=None, ground_texture=None)
    assert RR._key_look(bare)["ground_repeat"] == 1.0 and "ground_tile_m" not in RR._key_look(bare)

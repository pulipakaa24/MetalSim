"""Film look colours on geoms that carry their own rgba (metalsim.render.film.render). MuJoCo draws a geom's own rgba
over its material's unless the geom's is the default (0.5, 0.5, 0.5, 1) (``scene_tables.geom_rgba``; the path tracer
multiplies a texture by that colour), so ``build_scene`` writes the look's colour into the geom too: a
``LookCfg.materials`` entry's rgba, and the ground tint on the ground geoms. The render cache key records that write
only for takes whose pixels it changes: frames drawn before it get a new key, every other take keeps its key. CPU only."""
from __future__ import annotations

import hashlib
import json
import os

import mujoco
import numpy as np
import pytest
from PIL import Image

from metalsim.render import scene_tables as ST
from metalsim.render.film import cfg as C
from metalsim.render.film import render as RR

# floor: its own rgba (LimX's TRON1 floor); ground: a default-rgba plane; box: its own rgba (the SO-101 box, the LimX
# geoms); leg: the default rgba; plain: its own rgba and no override
TWIN = """
<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="1 1 0.1" rgba="0.5 0.6 0.7 1"/>
    <geom name="ground" type="plane" size="1 1 0.1" pos="0 0 -1"/>
    <geom name="box" type="box" size="0.02 0.02 0.03" pos="0 0 0.03" rgba="0 1 0 1"/>
    <geom name="leg" type="box" size="0.02 0.02 0.3" pos="0.5 0 -0.3"/>
    <geom name="plain" type="sphere" size="0.05" pos="0 0.5 0.05" rgba="1 0 0 1"/>
  </worldbody>
</mujoco>"""

GREEN = (0.1, 0.9, 0.15, 1.0)
GREY = (0.12, 0.12, 0.12, 1.0)
TINT = (0.9, 0.8, 0.7)


@pytest.fixture
def rec_dir(tmp_path):
    (tmp_path / "twin.xml").write_text(TWIN)
    (tmp_path / "rec.json").write_text(json.dumps(dict(fps=30.0)))
    np.savez(tmp_path / "rec.npz", xpos=np.zeros((2, 1, 5, 3), np.float32))
    return str(tmp_path)


@pytest.fixture
def texture(tmp_path):
    p = tmp_path / "tile.png"
    Image.fromarray(np.full((8, 8, 3), 128, np.uint8)).save(p)
    return str(p)


def _look(materials=None, ground=None, geoms=("floor",), tint=(1.0, 1.0, 1.0)):
    return C.LookCfg(hdri=None, ground_texture=ground, ground_geoms=geoms, ground_tile_m=2.0 if ground else None,
                     ground_tint=tint, materials=materials or {})


def _key(rec_dir, look):
    return RR.cache_key(rec_dir, C.FilmCfg(), look, C.ViewerCfg(), 1, [0])


def _key_before_the_fix(rec_dir, look):
    """``render.cache_key`` as it was before the geom rgba writes (845e72b1), for the keys that must not change."""
    film, viewer = C.FilmCfg(), C.ViewerCfg()
    h = hashlib.sha256()
    for name in ("rec.npz", "rec.json", "twin.xml"):
        h.update(name.encode())
        with open(os.path.join(rec_dir, name), "rb") as f:
            h.update(f.read())
    settings = dict(tier=film.tier, passes=film.passes, spp=film.spp, bounces=film.bounces, denoise=film.denoise,
                    look=RR._key_look(look), viewer=RR._cfg_dict(viewer), resolution=list(viewer.resolution), grid=1,
                    slots=[0])
    h.update(json.dumps(settings, sort_keys=True, default=str).encode())
    return h.hexdigest()[:20]


def test_build_scene_draws_the_look_colours(rec_dir, texture):
    look = _look({r"^box$": dict(rgba=GREEN, roughness=0.5), r"^leg$": dict(rgba=GREY, metallic=0.6),
                  r"^plain$": dict(roughness=0.3)}, ground=texture, geoms=("floor", "ground"), tint=TINT)
    model, *_ = RR.build_scene(rec_dir, {}, look, C.ViewerCfg())
    gid = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in ("floor", "ground", "box", "leg", "plain")}
    # the colour the renderers draw (MuJoCo's rule), not only the material's
    np.testing.assert_allclose(ST.geom_rgba(model, gid["box"]), GREEN, atol=1e-7)
    np.testing.assert_allclose(ST.geom_rgba(model, gid["leg"]), GREY, atol=1e-7)
    np.testing.assert_allclose(ST.geom_rgba(model, gid["plain"]), (1, 0, 0, 1))   # no rgba in the entry: its own
    for g in ("floor", "ground"):                       # the texture times the tint, not times LimX's floor colour
        np.testing.assert_allclose(ST.geom_rgba(model, gid[g]), (*TINT, 1.0), atol=1e-7)
        assert mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MATERIAL, model.geom_matid[gid[g]]) == "film_ground"
    assert float(model.mat_roughness[model.geom_matid[gid["plain"]]]) == pytest.approx(0.3)
    assert float(model.mat_metallic[model.geom_matid[gid["leg"]]]) == pytest.approx(0.6)


@pytest.mark.parametrize("look, recoloured", [
    (dict(materials={r"^box$": dict(rgba=GREEN, roughness=0.5), r"^leg$": dict(rgba=GREY)}), ["box"]),
    # the last matching entry wins, as in build_scene: a later entry without rgba leaves the box in the earlier colour
    (dict(materials={r"^box$": dict(rgba=GREEN), r"^b": dict(roughness=0.2)}), ["box"]),
    (dict(ground="tile.png"), ["floor"]),                                          # LimX's floor rgba under the tint
    (dict(ground="tile.png", geoms=("floor", "ground"), tint=TINT), ["floor"]),
], ids=["materials", "materials-last-wins", "ground", "ground-tinted"])
def test_cache_key_is_new_where_the_geom_write_changes_pixels(rec_dir, look, recoloured):
    look = _look(**look)
    assert RR.geoms_recoloured(rec_dir, look) == recoloured
    assert _key(rec_dir, look) != _key_before_the_fix(rec_dir, look)


@pytest.mark.parametrize("look", [
    dict(),                                                         # no overrides
    dict(materials={r"^box$": dict(roughness=0.5, metallic=0.2)}),  # no rgba: the geom keeps its own colour, as before
    dict(materials={r"^leg$": dict(rgba=GREY)}),     # default-rgba geom: the material's colour was drawn already
    dict(materials={r"^box$": dict(rgba=(0.0, 1.0, 0.0, 1.0))}),    # the geom's own colour
    dict(materials={r"^nothing$": dict(rgba=GREEN)}),               # matches no geom
    dict(ground="tile.png", geoms=("ground",), tint=TINT),          # default-rgba ground (the G1 films' floors)
    dict(ground="tile.png", geoms=("no_such_floor",)),
], ids=["none", "no-rgba", "default-rgba-geom", "same-colour", "no-match", "default-rgba-ground", "no-ground-geom"])
def test_cache_key_unchanged_where_pixels_do_not_change(rec_dir, look):
    look = _look(**look)
    assert RR.geoms_recoloured(rec_dir, look) == []
    assert _key(rec_dir, look) == _key_before_the_fix(rec_dir, look)

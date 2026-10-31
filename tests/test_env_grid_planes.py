"""Infinite planes under the compact env grid (metalsim/sensors/env_grid.py, shaders/env_grid.h): the plane is a disc
covering the env's cell in the acceleration structure and is continued analytically past the cell. Two ways the ground
lost a wedge (the G1 film, docs/research/readme_media_plan_2026-10-31.md section 2, defect 2):

* the cell grew (geometry neared its boundary) but the disc kept the initial cell's size, so floor between the disc's
  edge and the new cell boundary was neither in the structure nor continued;
* a camera placed outside its env's cell (Tier2Renderer.render_host(cam_pos=...)) traces up to the cell's far side, so
  floor between the camera and the disc's edge was missing.

The reference is the legacy grid (one disc to the sensor range, no cells): the same depth image."""
from __future__ import annotations

import warnings

import mujoco
import numpy as np
import pytest
import warp as wp

needs_metal = pytest.mark.skipif(not wp.is_metal_available(), reason="needs Metal")

# camera at the origin side looking along +y, 25 deg down: floor from ~2 m to the horizon; the box is behind it
XML = """<mujoco><worldbody>
  <geom name="floor" type="plane" size="0 0 0.1"/>
  <camera name="cam" pos="0 0.5 1" xyaxes="1 0 0 0 0.423 0.906" fovy="50"/>
  <body name="b" pos="0 0 0.5"><freejoint/><geom type="box" size="0.1 0.1 0.1"/></body>
</worldbody></mujoco>"""


def _renderer(m, layout, monkeypatch, n=1):
    from metalsim.render.tier2 import Tier2Renderer
    monkeypatch.setenv("METALSIM_ENV_LAYOUT", layout)
    return Tier2Renderer(m, n, width=96, height=64, camera="cam", spp=1, max_bounces=1, center_sample=True)


@needs_metal
def test_plane_disc_grows_with_the_env_grid(monkeypatch):
    m = mujoco.MjModel.from_xml_string(XML)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    ref = _renderer(m, "legacy", monkeypatch).render_host([d])["depth"][0]
    floor = ref > 0
    assert floor.mean() > 0.3
    r = _renderer(m, "compact", monkeypatch)
    d.qpos[0] = 3.0                                    # the box leaves its cell: the grid grows on a following frame
    mujoco.mj_forward(m, d)
    with pytest.warns(UserWarning, match="env grid"):
        for _ in range(6):
            out = r.render_host([d])["depth"][0]
    assert r.rt.grid_growths >= 1
    print("floor pixels missing after the growth:", int((floor & ~(out > 0)).sum()), "of", int(floor.sum()))
    assert np.array_equal(out > 0, floor)              # before the fix: a band of misses past the old disc's edge
    np.testing.assert_allclose(out[floor], ref[floor], rtol=1e-4, atol=1e-4)
    assert r.rt._plane_ring_built > r.rt._plane_ring  # the disc grew with the cell


@needs_metal
def test_camera_outside_its_cell_sees_the_whole_floor(monkeypatch):
    """A host camera 6 m from a scene whose cell is about 2.4 m wide: the cell grows to hold it (a warning), the floor
    is complete and the 4 envs of the same state render alike."""
    from metalsim.render.film.render import look_at_wxyz
    m = mujoco.MjModel.from_xml_string(XML)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    eye = np.array([4.0, -4.0, 2.5])
    q = look_at_wxyz(eye, (0.0, 0.0, 0.3))
    ref = _renderer(m, "legacy", monkeypatch).render_host([d], cam_pos=eye[None], cam_quat=q[None])["depth"][0]
    floor = ref > 0
    r = _renderer(m, "compact", monkeypatch, n=4)
    assert not ((eye >= r.rt.grid.lo) & (eye <= r.rt.grid.hi)).all()        # the camera starts outside the cell
    with pytest.warns(UserWarning, match="render camera is outside"):
        out = r.render_host([d] * 4, cam_pos=np.tile(eye, (4, 1)), cam_quat=np.tile(q, (4, 1)))["depth"]
    with warnings.catch_warnings():
        warnings.simplefilter("error")                  # contained now: no further growth
        r.render_host([d] * 4, cam_pos=np.tile(eye, (4, 1)), cam_quat=np.tile(q, (4, 1)))
    for e in range(4):
        print("env", e, "floor pixels missing:", int((floor & ~(out[e] > 0)).sum()), "of", int(floor.sum()))
        assert np.array_equal(out[e] > 0, floor), e
        np.testing.assert_allclose(out[e][floor], ref[floor], rtol=1e-4, atol=1e-4)


CHECKER = """<mujoco>
  <asset><texture name="chk" type="2d" builtin="checker" width="64" height="64" rgb1="1 1 1" rgb2="0.1 0.1 0.1"/>
         <material name="m" texture="chk" texrepeat="0.37 0.37" texuniform="true"/></asset>
  <worldbody>
  <geom name="floor" type="plane" size="0 0 0.1" material="m"/>
  <camera name="cam" pos="0 0 3" fovy="60"/>
  <body name="b" pos="0 0 0.5"><freejoint/><geom type="box" size="0.1 0.1 0.1"/></body>
</worldbody></mujoco>"""


@needs_metal
def test_neighbouring_plane_discs_do_not_overlap(monkeypatch):
    """4 envs of one state under a camera looking down on a textured infinite plane: every env's albedo equals the
    1-env render's. The disc used to reach 1 m past the cell's corners (about 1.5 m into each neighbouring cell here),
    coplanar with the neighbour's floor and its texture offset by the grid stride: the two z-fought."""
    from metalsim.render.tier2 import Tier2Renderer
    m = mujoco.MjModel.from_xml_string(CHECKER)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    monkeypatch.setenv("METALSIM_ENV_LAYOUT", "compact")
    alb = {}
    for n in (1, 4):
        r = Tier2Renderer(m, n, width=96, height=64, camera="cam", spp=1, max_bounces=1, center_sample=True, aux=True)
        alb[n] = np.asarray(r.render_host([d] * n)["albedo"])[..., :3].reshape(n, 64, 96, 3)
    assert r.rt.grid.nx * r.rt.grid.ny == 4 and alb[1].std() > 0.1           # side by side; the checker is visible
    for e in range(4):
        bad = int((np.abs(alb[4][e] - alb[1][0]).max(-1) > 1e-3).sum())
        print("env", e, "albedo pixels differing from the 1-env render:", bad)
        assert bad == 0, e

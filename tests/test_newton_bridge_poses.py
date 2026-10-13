"""Newton -> renderer bridge, geometry level (no Metal renderer needed; runs in .venv and .venv-newton152): the render
twin's drawn geometry, posed by the bridge from Newton's ``body_q``, is where MuJoCo puts the same MJCF's geometry.

Per body: the world-space vertices of every drawn mesh of the twin (``scene_tables``, the renderers' own tessellation)
against the same for the MJCF at the same qpos; bounding boxes agree to 1e-4 m. Both twins (Newton shapes, source MJCF),
batched worlds, and SolverMuJoCo stepping (MuJoCo-via-Newton)."""
import os
import sys

import mujoco
import numpy as np
import pytest
import warp as wp

newton = pytest.importorskip("newton", reason="Newton not installed (optional engine)")

from metalsim.render.newton_bridge import NewtonRenderSource, joint_q_to_mujoco_qpos, mujoco_qpos_to_joint_q
from metalsim.render.scene_tables import build_scene_tables

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from newton_bridge_scenes import PRIMS, mirror, random_qpos, so101_xml  # noqa: E402

DEVICE = "cpu"


def _model(xml, n, device=DEVICE):
    env = newton.ModelBuilder(); env.add_mjcf(xml)
    b = newton.ModelBuilder(); b.replicate(env, n)
    return b.finalize(device=device)


def _body_boxes(m, xpos, xmat):
    """{body name: (min, max)} of the world-space vertices of the drawn geoms (renderer tessellation), one env."""
    t = build_scene_tables(m, 1)
    V = t.vertices[:, :3].astype(np.float64)
    out = {}
    for slot, g in enumerate(t.geoms):
        if int(m.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_PLANE):
            continue
        mesh = t.meshes[t.geom_mesh[slot]]
        idx = np.unique(t.indices[mesh["i_off"]:mesh["i_off"] + mesh["i_count"]])
        w = V[idx] @ np.asarray(xmat[g], float).reshape(3, 3).T + xpos[g]
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) or "world"
        lo, hi = out.get(nm, (np.full(3, np.inf), np.full(3, -np.inf)))
        out[nm] = (np.minimum(lo, w.min(0)), np.maximum(hi, w.max(0)))
    return out


def _check(src, mj, qpos, tol=1e-4):
    for e in range(src.n):
        d = mujoco.MjData(mj); d.qpos[:] = qpos[e]; mujoco.mj_forward(mj, d)
        ref = _body_boxes(mj, d.geom_xpos, d.geom_xmat)
        h = src.host_datas()[e]
        got = _body_boxes(src.m, h.geom_xpos, h.geom_xmat)
        assert set(ref) == set(got), (set(ref) ^ set(got))
        for nm in ref:
            np.testing.assert_allclose(got[nm][0], ref[nm][0], atol=tol, err_msg=f"env {e} body {nm}")
            np.testing.assert_allclose(got[nm][1], ref[nm][1], atol=tol, err_msg=f"env {e} body {nm}")


@pytest.mark.parametrize("scene", ["prims", "so101"])
@pytest.mark.parametrize("twin", ["newton", "mjcf"])
def test_twin_geometry_posed_like_mujoco(scene, twin):
    xml = PRIMS if scene == "prims" else so101_xml()
    mj = mujoco.MjModel.from_xml_string(xml)
    n = 3
    model = _model(xml, n)
    state = model.state()
    if twin == "mjcf":
        src = NewtonRenderSource(model, twin=mj, device=DEVICE)
    else:
        cams, lights = mirror(mj)
        src = NewtonRenderSource(model, cameras=cams, lights=lights, device=DEVICE)
    qpos = random_qpos(mj, n, 7)
    model.joint_q.assign(mujoco_qpos_to_joint_q(mj, qpos, model, offsets=src.origins))
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    src.update(state, graph=False)
    _check(src, mj, qpos)
    # cameras: same pose as MuJoCo's (by name)
    for c in range(mj.ncam):
        nm = mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_CAMERA, c)
        tc = mujoco.mj_name2id(src.m, mujoco.mjtObj.mjOBJ_CAMERA, nm)
        d = mujoco.MjData(mj); d.qpos[:] = qpos[1]; mujoco.mj_forward(mj, d)
        h = src.host_datas()[1]
        np.testing.assert_allclose(h.cam_xpos[tc], d.cam_xpos[c], atol=1e-5)
        np.testing.assert_allclose(h.cam_xmat[tc], d.cam_xmat[c], atol=1e-5)


def test_solver_mujoco_state_posed_like_mujoco():
    """MuJoCo-via-Newton: SolverMuJoCo steps the state; the twin geometry sits where MuJoCo puts it at the qpos Newton
    reports (``joint_q``), i.e. the bridge reads ``body_q`` as the solver leaves it."""
    device = "metal:0" if wp.is_metal_available() else "cpu"
    mj = mujoco.MjModel.from_xml_string(PRIMS)
    model = _model(PRIMS, 2, device=device)
    try:
        solver = newton.solvers.SolverMuJoCo(model)
        s0, s1, ctrl = model.state(), model.state(), model.control()
        newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
        for _ in range(50):
            solver.step(s0, s1, ctrl, None, 0.002)
            s0, s1 = s1, s0
        wp.synchronize_device(device)
    except Exception as exc:          # Newton 1.7 here pins mujoco ~=3.12; the .venv has 3.14 (kernel codegen fails)
        pytest.skip(f"SolverMuJoCo unavailable in this environment: {type(exc).__name__}: {str(exc)[:120]}")
    src = NewtonRenderSource(model, twin=mj, device=device)
    src.update(s0, graph=False)
    qpos = joint_q_to_mujoco_qpos(mj, model, s0.joint_q)
    assert np.abs(qpos - mj.qpos0).max() > 0.01
    _check(src, mj, qpos, tol=2e-4)

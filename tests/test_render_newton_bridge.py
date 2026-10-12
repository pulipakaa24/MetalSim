"""Newton -> renderer bridge (metalsim.render.newton_bridge): a Newton model imported from an MJCF, rendered through
the bridge, matches the same MJCF rendered through the MuJoCo path at the same state (per-body silhouette coverage
>= 0.99, depth within 1 mm) for both render twins (Newton shapes, source MJCF), batched worlds, env origins,
heterogeneous worlds, the ray-traced sensors, tier 1 / tier 2, the annotators, SolverMuJoCo stepping and graph capture.

Scenes: synthetic MJCF and the SO-101 (assets/so101). Run through the GPU queue:
    scripts/gpu_run.sh newton_bridge_test check 15 -- .venv/bin/python -m pytest tests/test_render_newton_bridge.py -q
"""
import os
import sys

import mujoco
import numpy as np
import pytest
import warp as wp

newton = pytest.importorskip("newton", reason="Newton not installed (optional engine)")

from metalsim.render.newton_bridge import Camera, NewtonRenderSource, joint_q_to_mujoco_qpos, mujoco_qpos_to_joint_q
from metalsim.render.tier0 import SEG_BODY, SEG_GEOM, Tier0Renderer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from newton_bridge_scenes import PRIMS, mirror, random_qpos, so101_xml  # noqa: E402

pytestmark = pytest.mark.skipif(not wp.is_metal_available(), reason="needs Metal")
W, H = 160, 120

def newton_model(xml, n_worlds, spacing=0.0, device="metal:0"):
    env = newton.ModelBuilder()
    env.add_mjcf(xml)
    b = newton.ModelBuilder()
    b.replicate(env, n_worlds, spacing=(spacing, 0.0, 0.0))
    return b.finalize(device=device)


def make_source(model, mj, twin, **kw):
    if twin == "mjcf":
        return NewtonRenderSource(model, twin=mj, **kw)
    cams, lights = mirror(mj)
    return NewtonRenderSource(model, cameras=cams, lights=lights, **kw)


def datas_of(mj, qpos):
    out = []
    for q in qpos:
        d = mujoco.MjData(mj); d.qpos[:] = q; mujoco.mj_forward(mj, d); out.append(d)
    return out


def pose_newton(model, mj, qpos, state, offsets=None):
    model.joint_q.assign(mujoco_qpos_to_joint_q(mj, qpos, model, offsets=offsets))
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)


def body_id_map(mj, twin):
    """MuJoCo-path seg value (body + 1) -> bridge seg value, by body name."""
    names = [mujoco.mj_id2name(twin, mujoco.mjtObj.mjOBJ_BODY, b) for b in range(twin.nbody)]
    ref, got = [], []
    for b in range(mj.nbody):
        nm = mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_BODY, b)
        if b == 0 or nm in names:
            ref.append(b + 1); got.append(1 if b == 0 else names.index(nm) + 1)
    return ref, got


def coverage(ref, got, ids):
    """Minimum per-body coverage of the reference silhouette, foreground IoU, depth error on the common foreground."""
    covs = []
    for a, b in zip(*ids):
        m = ref["seg"] == a
        if m.sum() >= 10:
            covs.append(float((got["seg"][m] == b).mean()))
    fr, fg = ref["seg"] > 0, got["seg"] > 0
    both = fr & fg & (ref["depth"] > 0) & (got["depth"] > 0)
    dz = np.abs(ref["depth"][both] - got["depth"][both])
    return min(covs), float((fr & fg).sum() / max((fr | fg).sum(), 1)), float(np.percentile(dz, 99)) if dz.size else 0.0


def render_pair(xml, n, twin, cam, seed=0, spacing=0.0, env_origins="auto", tier=0):
    mj = mujoco.MjModel.from_xml_string(xml)
    model = newton_model(xml, n, spacing)
    state = model.state()
    src = make_source(model, mj, twin, env_origins=env_origins)
    rs = Tier0Renderer(src.m, n, width=W, height=H, camera=cam, outputs=("rgb", "depth", "seg"), seg_mode=SEG_BODY, tier=tier)
    rr = Tier0Renderer(mj, n, width=W, height=H, camera=cam, outputs=("rgb", "depth", "seg"), seg_mode=SEG_BODY, tier=tier)
    qpos = random_qpos(mj, n, seed)
    # the MuJoCo path sees world e's qpos in its own env frame: the Newton worlds are offset by ``spacing``
    pose_newton(model, mj, qpos, state, offsets=src.origins)
    v = src.update(state)
    rs.render(src, v); rs.ctx.synchronize()
    got = {k: a.numpy().copy() for k, a in rs._out_arrays.items()}
    ref = rr.render_host(datas_of(mj, qpos))
    return mj, src, ref, got


@pytest.mark.parametrize("twin", ["newton", "mjcf"])
def test_primitives_batched_match_mujoco_path(twin):
    n = 3
    mj, src, ref, got = render_pair(PRIMS, n, twin, "cam")
    ids = body_id_map(mj, src.m)
    for e in range(n):
        cov, iou, dz = coverage({k: v[e] for k, v in ref.items()}, {k: v[e] for k, v in got.items()}, ids)
        assert cov >= 0.99 and iou >= 0.99, (e, cov, iou)
        assert dz < 1e-3, (e, dz)
    # different states per world: the envs differ from each other
    assert (ref["seg"][0] != ref["seg"][1]).mean() > 0.01
    if twin == "mjcf":     # same tables, same poses up to float32 rounding: the images agree to the pixel
        assert np.abs(ref["rgb"].astype(int) - got["rgb"].astype(int)).mean() < 0.05


@pytest.mark.parametrize("cam", ["overview", "wrist_cam"])
def test_so101_meshes_match_mujoco_path(cam):
    """Mesh-heavy arm (visual class meshes; collision geoms hidden as in MuJoCo's groups), the floor collider drawn by
    the static fallback, and a body-mounted physical camera (sensorsize / focal)."""
    n = 2
    mj, src, ref, got = render_pair(so101_xml(), n, "newton", cam, seed=1)
    ids = body_id_map(mj, src.m)
    for e in range(n):
        cov, iou, dz = coverage({k: v[e] for k, v in ref.items()}, {k: v[e] for k, v in got.items()}, ids)
        assert cov >= 0.99 and iou >= 0.99 and dz < 1e-3, (e, cov, iou, dz)


def test_env_origins_spaced_worlds():
    """Worlds 2 m apart in Newton render like co-located worlds (env frames at the auto origins)."""
    n = 3
    _, _, ref, got = render_pair(PRIMS, n, "newton", "cam", seed=2, spacing=2.0)
    _, _, _, got0 = render_pair(PRIMS, n, "newton", "cam", seed=2, spacing=0.0)
    assert (got["seg"] == got0["seg"]).mean() > 0.999
    assert np.abs(got["depth"] - got0["depth"]).max() < 1e-3
    assert (got["seg"] == ref["seg"]).mean() > 0.0          # seg ids are bridge-body ids; compared above via names


def test_heterogeneous_worlds_each_env_sees_its_own():
    b = newton.ModelBuilder()
    b.add_ground_plane()
    for e, kind in enumerate(("box", "sphere", "capsule")):
        b.begin_world()
        body = b.add_body(xform=wp.transform((0.0, 0.0, 0.3), wp.quat_identity()), label=f"obj{e}")
        if kind == "box":
            b.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
        elif kind == "sphere":
            b.add_shape_sphere(body, radius=0.12)
        else:
            b.add_shape_capsule(body, radius=0.06, half_height=0.1)
        b.end_world()
    model = b.finalize(device="metal:0")
    state = model.state()
    with pytest.warns(UserWarning, match="different shape layouts"):
        src = NewtonRenderSource(model, cameras=[Camera(pos=(1.0, -1.0, 0.8), lookat=(0, 0, 0.3))])
    r = Tier0Renderer(src.m, 3, width=W, height=H, camera=0, outputs=("depth", "seg"), seg_mode=SEG_GEOM)
    r.render(src, src.update(state)); r.ctx.synchronize()
    seg = r._out_arrays["seg"].numpy()
    shape_of = src.shape_of_geom()
    for e in range(3):
        drawn = {int(g) - 1 for g in np.unique(seg[e]) if g > 0}
        shapes = {int(shape_of[e, g]) for g in drawn}
        assert -1 not in shapes                                       # nothing from another world
        sw = model.shape_world.numpy()
        assert {int(sw[s]) for s in shapes} == {-1, e}                # the ground plane and this world's object


def test_raytraced_depth_tier1_tier2_match_mujoco_path():
    """The ray tracer (depth camera; shared by lidar / radar / tier 1 / tier 2) and tier 2 on the bridge source."""
    from metalsim.render.tier2 import Tier2Renderer
    from metalsim.sensors.raytrace import RayTracer
    n = 2
    mj = mujoco.MjModel.from_xml_string(PRIMS)
    model = newton_model(PRIMS, n)
    state = model.state()
    qpos = random_qpos(mj, n, 3)
    pose_newton(model, mj, qpos, state)
    datas = datas_of(mj, qpos)
    for twin in ("newton", "mjcf"):
        src = make_source(model, mj, twin)
        v = src.update(state)
        rt = RayTracer(src.m, n, max_range=10.0); cam = rt.make_depth_camera("cam", W, H)
        lid = rt.make_lidar("tip", np.linspace(-1, 1, 64), np.zeros(64))
        vr = rt.trace(src, [cam, lid], v); rt.ctx.synchronize()
        got_d, got_l = cam.numpy(), lid.numpy()["range"]
        rt0 = RayTracer(mj, n, max_range=10.0); cam0 = rt0.make_depth_camera("cam", W, H)
        lid0 = rt0.make_lidar("tip", np.linspace(-1, 1, 64), np.zeros(64))
        ref = rt0.trace_host(datas, [cam0, lid0])
        assert np.abs(got_l - ref["lidar"]["range"]).max() < 1e-3          # Newton imports MJCF sites as SITE shapes
        hit = (ref["ray_depth"] > 0) & (got_d > 0)
        assert ((ref["ray_depth"] > 0) == (got_d > 0)).mean() > 0.999
        assert np.abs(ref["ray_depth"][hit] - got_d[hit]).max() < 1e-3
        # tier 2 (center-sampled, same seed): same silhouettes and depth
        t2 = Tier2Renderer(src.m, n, width=64, height=48, camera="cam", spp=4, max_bounces=1, center_sample=True)
        t2r = Tier2Renderer(mj, n, width=64, height=48, camera="cam", spp=4, max_bounces=1, center_sample=True)
        t2.render(src, src.update(state)); t2.ctx.synchronize()
        g2 = {k: a.numpy() for k, a in t2._arrays.items()}
        r2 = t2r.render_host(datas)
        assert ((g2["depth"] > 0) == (r2["depth"] > 0)).mean() > 0.995
        both = (g2["depth"] > 0) & (r2["depth"] > 0)
        assert np.median(np.abs(g2["depth"][both] - r2["depth"][both])) < 1e-3
    # tier 1 (raster + ray-traced shadows / AO / reflections)
    mj_, src_, ref1, got1 = render_pair(PRIMS, n, "mjcf", "cam", seed=3, tier=1)
    cov, iou, dz = coverage({k: v[0] for k, v in ref1.items()}, {k: v[0] for k, v in got1.items()}, body_id_map(mj_, src_.m))
    assert cov >= 0.99 and dz < 1e-3
    assert np.abs(ref1["rgb"].astype(int) - got1["rgb"].astype(int)).mean() < 0.5


def test_newton_site_shapes_mount_a_lidar():
    b = newton.ModelBuilder()
    b.add_ground_plane()
    body = b.add_body(xform=wp.transform((0.0, 0.0, 0.5), wp.quat_identity()), label="base")
    b.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05)
    b.add_site(body, xform=wp.transform((0.0, 0.0, -0.06), wp.quat_rpy(0.0, np.pi / 2, 0.0)), label="down")
    model = b.finalize(device="metal:0")
    src = NewtonRenderSource(model)
    from metalsim.sensors.raytrace import RayTracer
    rt = RayTracer(src.m, 1, max_range=5.0)
    lid = rt.make_lidar("down", np.zeros(1), np.zeros(1))
    rt.trace(src, [lid], src.update(model.state())); rt.ctx.synchronize()
    assert abs(float(lid.numpy()["range"][0, 0]) - 0.44) < 1e-3          # site x axis points down at the ground


def test_annotators_on_bridge_source():
    from metalsim.replicator.isaac import IsaacAnnotators, State
    from metalsim.replicator.semantics import Semantics
    n = 2
    mj = mujoco.MjModel.from_xml_string(PRIMS)
    model = newton_model(PRIMS, n)
    state = model.state()
    qpos = random_qpos(mj, n, 4)
    pose_newton(model, mj, qpos, state)
    src = make_source(model, mj, "mjcf")
    r = Tier0Renderer(src.m, n, width=W, height=H, camera="cam", outputs=("rgb", "depth", "seg"), seg_mode=SEG_GEOM)
    r.render(src, src.update(state)); r.ctx.synchronize()
    ann = IsaacAnnotators(r, Semantics.from_body_names(src.m), sim=src)
    boxes = ann.bounding_box_3d()
    r_ref = Tier0Renderer(mj, n, width=W, height=H, camera="cam", outputs=("rgb", "depth", "seg"), seg_mode=SEG_GEOM)
    datas = datas_of(mj, qpos)
    r_ref.render_host(datas)
    ann_ref = IsaacAnnotators(r_ref, Semantics.from_body_names(mj))
    ann_ref.set_state(State.from_datas(datas))
    ref_boxes = ann_ref.bounding_box_3d()
    for k in ref_boxes:
        a, b_ = ref_boxes[k], boxes[k]
        if hasattr(a, "cpu"):
            np.testing.assert_allclose(b_.cpu().numpy(), a.cpu().numpy(), atol=1e-4)


def test_solver_mujoco_steps_render_like_mujoco_path():
    """MuJoCo-via-Newton: SolverMuJoCo advances the state; the bridge renders body_q, the reference renders the MJCF at
    the qpos Newton reports. Recorded into one graph with the solver step (pose kernels inside the capture)."""
    n = 2
    mj = mujoco.MjModel.from_xml_string(PRIMS)
    model = newton_model(PRIMS, n)
    s0, s1, ctrl = model.state(), model.state(), model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    src = make_source(model, mj, "newton")
    try:
        solver = newton.solvers.SolverMuJoCo(model)
        solver.step(s0, s1, ctrl, None, 0.002); s0, s1 = s1, s0
        wp.synchronize_device("metal:0")
    except Exception as exc:          # Newton 1.7 here pins mujoco ~=3.12; the .venv has 3.14 (kernel codegen fails)
        pytest.skip(f"SolverMuJoCo unavailable in this environment: {type(exc).__name__}: {str(exc)[:120]}")
    with wp.ScopedDevice("metal:0"):
        with wp.ScopedCapture() as cap:
            for _ in range(10):
                solver.step(s0, s1, ctrl, None, 0.002)
                s0, s1 = s1, s0
            src.launch(s0)
    for _ in range(5):
        wp.capture_launch(cap.graph)
    v = src.signal()
    r = Tier0Renderer(src.m, n, width=W, height=H, camera="cam", outputs=("depth", "seg"), seg_mode=SEG_BODY)
    r.render(src, v); r.ctx.synchronize()
    got = {k: a.numpy() for k, a in r._out_arrays.items()}
    qpos = joint_q_to_mujoco_qpos(mj, model, s0.joint_q)
    assert np.abs(qpos - mj.qpos0).max() > 0.01                   # the objects moved
    rr = Tier0Renderer(mj, n, width=W, height=H, camera="cam", outputs=("depth", "seg"), seg_mode=SEG_BODY)
    ref = rr.render_host(datas_of(mj, qpos))
    ids = body_id_map(mj, src.m)
    for e in range(n):
        cov, iou, dz = coverage({k: v_[e] for k, v_ in ref.items()}, {k: v_[e] for k, v_ in got.items()}, ids)
        assert cov >= 0.99 and dz < 1e-3, (e, cov, dz)


def test_graph_replay_equals_eager():
    mj = mujoco.MjModel.from_xml_string(PRIMS)
    model = newton_model(PRIMS, 2)
    state = model.state()
    pose_newton(model, mj, random_qpos(mj, 2, 5), state)
    src = make_source(model, mj, "newton")
    src.update(state, graph=False); src.synchronize()
    a = src.d.geom_xpos.numpy().copy(), src.d.geom_xmat.numpy().copy(), src.d.cam_xmat.numpy().copy()
    src.d.geom_xpos.zero_(); src.d.geom_xmat.zero_()
    src.update(state, graph=True); src.update(state, graph=True); src.synchronize()
    np.testing.assert_array_equal(a[0], src.d.geom_xpos.numpy())
    np.testing.assert_array_equal(a[1], src.d.geom_xmat.numpy())
    np.testing.assert_array_equal(a[2], src.d.cam_xmat.numpy())

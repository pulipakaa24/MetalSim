"""Scene composition (metalsim.scene): the YAML schema (Isaac Lab names), the built MjSpec (robot + objects + camera /
lidar mounts + ground + collision proxies), camera offset conventions, reset sampling, and stability of the composed
example under MuJoCo C. CPU tests run anywhere; the ``needs_metal`` ones build BatchSim, the cameras and the lidar."""
from __future__ import annotations

import json
import os

import mujoco
import numpy as np
import pytest

from metalsim.scene import cfg as C
from metalsim.scene import compose as CP
from metalsim.scene import proxy as PX

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GO2 = os.path.join(ROOT, "configs", "scenes", "go2_boxes.yaml")
HAVE_GO2 = os.path.exists(os.path.join(ROOT, "upstream", "mujoco_menagerie", "unitree_go2", "go2_mjx.xml"))
needs_go2 = pytest.mark.skipif(not HAVE_GO2, reason="needs upstream/mujoco_menagerie (scripts/setup_*.sh)")


def _metal():
    try:
        import warp as wp
        return bool(wp.is_metal_available())
    except Exception:  # noqa: BLE001
        return False


needs_metal = pytest.mark.skipif(not _metal(), reason="needs Metal")


def test_schema_isaac_names():
    d = {"num_envs": 2, "objects": {"ball": {"spawn": {"type": "SphereCfg", "radius": 0.1, "mass_props": {"mass": 1.0}},
                                             "init_state": {"pos": [0, 0, 1]},
                                             "randomize": {"pose_range": {"x": [-1, 1]}}}},
         "cameras": {"cam": {"prim_path": "{ENV_REGEX_NS}/Camera", "offset": {"pos": [2, 0, 1], "convention": "world"},
                             "data_types": ["rgb", "distance_to_image_plane"]}},
         "background": {"ground": None, "lights": [{"type": "DistantLightCfg"}]}}
    cfg = C.from_dict(d)
    assert isinstance(cfg.objects["ball"].spawn, C.SphereCfg) and cfg.objects["ball"].spawn.mass_props.mass == 1.0
    assert cfg.cameras["cam"].offset.convention == "world" and cfg.background.ground is None
    assert isinstance(cfg.background.lights[0], C.DistantLightCfg)
    with pytest.raises(ValueError):
        C.from_dict({"objects": {"x": {"spawn": {"type": "NoSuchCfg"}}}})
    with pytest.raises(ValueError):
        C.from_dict({"num_envz": 3})


@pytest.mark.parametrize("conv,forward", [("world", (1, 0, 0)), ("ros", (0, 0, 1)), ("opengl", (0, 0, -1))])
def test_camera_conventions(conv, forward):
    """Identity offset rotation: the camera looks along the convention's forward axis (Isaac's OffsetCfg)."""
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, CP.camera_quat(C.OffsetCfg(convention=conv)))
    np.testing.assert_allclose(-R.reshape(3, 3)[:, 2], forward, atol=1e-12)    # MuJoCo cameras look along -z


def test_camera_rotation_xyzw():
    """rot is (x, y, z, w): 90 deg about +z turns a world-convention camera from +x to +y."""
    q = (0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4))
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, CP.camera_quat(C.OffsetCfg(rot=q, convention="world")))
    np.testing.assert_allclose(-R.reshape(3, 3)[:, 2], (0, 1, 0), atol=1e-12)


def test_fovy_from_pinhole():
    cam = C.CameraCfg(width=1920, height=1080, spawn=C.PinholeCameraCfg(focal_length=18.147, horizontal_aperture=20.955))
    assert abs(CP.fovy_deg(cam) - 36.0) < 0.1          # Kit's perspective camera at 16:9


def test_mount_body():
    assert CP.mount_body("{ENV_REGEX_NS}/Robot/base/head_cam") == "base"
    assert CP.mount_body("{ENV_REGEX_NS}/Camera") is None
    assert CP.mount_body("/World/envs/env_0/Robot/link7/wrist") == "link7"


def test_splat_transform_matches_splat_package():
    t = C.SplatTransformCfg(up="-y", scale=2.0, yaw_deg=30.0, translate=(1, 2, 3))
    A, b = CP.splat_transform(t)
    try:
        from metalsim.render.splat import make_transform
    except ImportError:
        pytest.skip("splat renderer not on this branch")
    A2, b2 = make_transform(up="-y", scale=2.0, yaw_deg=30.0, translate=(1, 2, 3))
    np.testing.assert_allclose(A, A2); np.testing.assert_allclose(b, b2)


def test_structure_proxy(tmp_path):
    """InteriorGS-style floor plan -> floor plane + one box per wall (shared walls merged), in group 4 (not drawn)."""
    js = tmp_path / "structure.json"
    js.write_text(json.dumps({"rooms": [{"profile": [[0, 0], [4, 0], [4, 3], [0, 3]]},
                                        {"profile": [[4, 0], [7, 0], [7, 3], [4, 3]]}]}))
    spec = mujoco.MjSpec()
    info = PX.add_proxy(spec, C.CollisionProxyCfg(type="structure", structure_json=str(js), wall_height=2.5))
    m = spec.compile()
    assert info["geoms"] == 1 + 7 and m.ngeom == 8 and set(m.geom_group) == {PX.PROXY_GROUP}
    d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    # a ray along +x at mid-height from inside room 0 hits the shared wall at x = 4 - thickness / 2
    gid = np.zeros(1, np.int32)
    dist = mujoco.mj_ray(m, d, np.array([1.0, 1.5, 1.0]), np.array([1.0, 0, 0]), None, 1, -1, gid)
    assert abs(dist - (3.0 - 0.05)) < 1e-6


def test_occupancy_proxy(tmp_path):
    import imageio.v3 as iio

    img = np.full((20, 30), 255, np.uint8); img[:, 10] = 0                   # one occupied column of pixels
    iio.imwrite(tmp_path / "occupancy.png", img)
    (tmp_path / "occupancy.json").write_text(json.dumps({"resolution": 0.1, "origin": [-1.0, -1.0]}))
    (tmp_path / "structure.json").write_text("{}")
    spec = mujoco.MjSpec()
    PX.add_proxy(spec, C.CollisionProxyCfg(type="occupancy", structure_json=str(tmp_path / "structure.json"),
                                           wall_height=2.0))
    m = spec.compile(); d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    gid = np.zeros(1, np.int32)
    dist = mujoco.mj_ray(m, d, np.array([-0.95, 0.0, 1.0]), np.array([1.0, 0, 0]), None, 1, -1, gid)
    # occupied pixel column covers x in [0.0, 0.1] (centre 0.05); grid points at pixel centres, so at half height the
    # ramp from the free centre (-0.05) is crossed at x = 0.0
    assert abs(dist - 0.95) < 1e-6


@needs_go2
def test_go2_boxes_builds_and_stands():
    """The example composes, defaults hold the standing pose, boxes rest on the ground (MuJoCo C, 2 s)."""
    cfg = C.load(GO2)
    scene = CP.ComposedScene(cfg, device=None)
    m = scene.model
    assert scene.info.robot_free == "base_freejoint"   # Menagerie's unnamed <freejoint/>, named so init_state applies
    assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "head_cam") >= 0
    assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "front_lidar") >= 0
    assert sum(1 for b in range(m.nbody) if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "").startswith("box_")) == 6
    qpos, qvel = scene.sample_states([0, 1, 2])
    assert not np.allclose(qpos[0], qpos[1])         # randomized box poses differ per env
    d = mujoco.MjData(m)
    d.qpos[:] = qpos[0]; d.qvel[:] = 0; d.ctrl[:] = scene.ctrl_default
    for _ in range(int(2.0 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base")
    assert 0.2 < d.xpos[base, 2] < 0.4               # still standing
    for b in range(m.nbody):
        if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "").startswith("box_"):
            assert abs(d.xpos[b, 2] - 0.125) < 0.02    # resting on the ground (or on another box: no box starts stacked)
    # the head camera looks forward (+x of the base)
    cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "head_cam")
    fwd = -d.cam_xmat[cid].reshape(3, 3)[:, 2]
    np.testing.assert_allclose(fwd, d.xmat[base].reshape(3, 3)[:, 0], atol=1e-9)


@needs_go2
def test_run_check_cli():
    from metalsim import run
    assert run.main([GO2, "--check"]) == 0


@needs_go2
@needs_metal
def test_go2_boxes_gpu(tmp_path):
    """BatchSim + tier-1 head camera (rgb, depth, segmentation) + ray-traced lidar on the composed example."""
    from dataclasses import replace

    cfg = replace(C.load(GO2), num_envs=2)
    scene = CP.compose(cfg)
    scene.reset()
    for _ in range(20):
        scene.step()
    scene.sim.synchronize()
    out = scene.render()["head_cam"]
    rgb, depth, seg = out["rgb"], out["distance_to_image_plane"], out["semantic_segmentation"]
    assert rgb.shape == (2, 360, 640, 3) and rgb.std() > 5
    assert depth.shape == (2, 360, 640) and np.isfinite(depth).all()
    # The first version asked for > 50 % of pixels with a hit and got exactly 1/3 (2026-10-14, 16-inch). Root cause: the
    # image was right. A forward camera 0.38 m above the floor sees sky (depth 0 = no hit) above the horizon, about half
    # the frame when level (runs/scenes/go2_boxes/run/head_cam_distance_to_image_plane_0.png: horizon at mid-height),
    # and at step 20 the dog, settling from its drop, was pitched ~6 deg nose-up in both envs: the horizon 120 rows
    # below the centre, 120 + 120 hit rows of 720 = 1/3 (the bottom-row floor depth 1.854 m matches that pitch).
    # Checked geometrically now, against the first hit of each pixel's ray on the host model (mj_ray over the geom groups
    # the renderer draws, 0-3), not an assumed floor: with the reset sampling of the scene seed, boxes sit across the
    # bottom-left pixel's ray (fresh-install check 2026-10-14: 0.866 m rendered, a box; the floor would be 1.854 m).
    m = scene.model
    cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "head_cam")
    datas = scene.host_datas()                         # the state just rendered (host_datas synchronizes; no step since)
    f = (360 / 2) / np.tan(np.radians(CP.fovy_deg(cfg.cameras["head_cam"])) / 2)
    groups = np.array([1, 1, 1, 1, 0, 0], np.uint8)
    robot = m.body_rootid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base")]
    checked = 0
    for e in range(2):
        p, R = datas[e].cam_xpos[cid], datas[e].cam_xmat[cid].reshape(3, 3)
        for row, col in ((359, 0), (359, 639), (359, 320), (300, 320), (330, 100), (330, 540)):
            v = R @ np.array([(col + 0.5 - 320) / f, -(row + 0.5 - 180) / f, -1.0])   # unit planar depth along v
            gid = np.zeros(1, np.int32)
            t = mujoco.mj_ray(m, datas[e], p, v, groups, 1, -1, gid)                   # in units of |v|: planar depth
            got = float(depth[e, row, col])
            if t < 0:
                assert got == 0.0, (e, row, col, got)                                  # nothing hit: sky
            elif m.body_rootid[m.geom_bodyid[gid[0]]] == robot:
                continue                                       # the robot's own collision vs visual meshes differ
            else:
                hit = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(gid[0]))
                assert abs(got - t) < 0.02 * t, (e, row, col, got, float(t), hit)
            checked += 1
        top = R @ np.array([0.0, 180 / f, -1.0])
        if top[2] > 0.01:                                                       # the top row looks up: sky only
            assert (depth[e, 0] == 0).all()
    assert checked >= 8
    assert len(np.unique(seg)) >= 2
    hits = scene.sense()["front_lidar"]["range"]
    assert hits.shape[0] == 2 and (np.asarray(hits) > 0).mean() > 0.05      # the boxes and the ground are hit


@needs_go2
def test_film_task_twin_exports(tmp_path):
    """--film: the scene's film entry and its render twin (mesh paths made absolute, exported MJCF reloads)."""
    from metalsim.render.film.record import export_twin, name_geoms
    from metalsim.scene.film import film_task

    cfg = C.load(GO2)
    task = film_task(cfg)
    assert task.viewer.eye == (2.4, -2.2, 1.3) and task.film.max_seconds == 5 and len(task.assets) == 3
    spec, _ = CP.build_spec(cfg)
    name_geoms(spec)
    m0 = spec.compile()
    export_twin(spec, str(tmp_path))
    m = mujoco.MjModel.from_xml_path(str(tmp_path / "twin.xml"))
    assert m.ngeom == m0.ngeom and m.nmesh == m0.nmesh


def test_lidar_keeps_config_name(monkeypatch):
    """scene.sense() is keyed by the config name: compose passes it to make_lidar (Lidar's default name is "lidar";
    batch_fix2's KeyError: 'front_lidar'). CPU: a stand-in ray tracer records the call; beams follow LidarPatternCfg."""
    import metalsim.sensors.raytrace as RT

    calls = []

    class FakeRT:
        def __init__(self, model, n, max_range):
            self.max_range = max_range

        def make_lidar(self, site, az, el, **kw):
            calls.append((site, len(az), kw))
            return type("L", (), {"name": kw.get("name", "lidar")})()

        def trace_host(self, datas, sensors):
            return {s.name: {"range": np.zeros((1, 3))} for s in sensors}

    monkeypatch.setattr(RT, "RayTracer", FakeRT)
    cfg = C.from_dict({"sensors": {"roof_lidar": {"prim_path": "{ENV_REGEX_NS}/Lidar", "offset": {"pos": [0, 0, 1]},
                                                  "pattern_cfg": {"channels": 4, "horizontal_res": 10.0}}}})
    scene = CP.ComposedScene(cfg, device=None)
    scene._build_sensors()
    assert calls == [("roof_lidar", 4 * 36, {"name": "roof_lidar"})] and list(scene.lidars) == ["roof_lidar"]
    assert list(scene.sense(datas=[object()])) == ["roof_lidar"]

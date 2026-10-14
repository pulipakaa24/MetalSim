"""The filming tool's CPU stages (metalsim.render.film): record with stand-in adapters (a Newton scene and a MuJoCo
scene), the portable render twin (MJCF + PNG textures) and its poses, the render-scene look, the doc. The GPU stages
(the task rollouts and the tier-2 render) run in the film batch; here everything is on the CPU."""
from __future__ import annotations

import json
from dataclasses import replace

import mujoco
import numpy as np
import pytest

from metalsim.render.film import cfg as C
from metalsim.render.film import record as REC

XML = """
<mujoco>
  <asset><texture name="chk" type="2d" builtin="checker" width="8" height="8" rgb1="1 0 0" rgb2="0 0 1"/>
         <material name="m" texture="chk"/></asset>
  <worldbody>
    <geom name="floor" type="plane" size="0 0 0.1"/>
    <body name="base" pos="0 0 0.5"><freejoint/>
      <geom type="box" size="0.1 0.2 0.05" material="m"/>
      <body name="arm" pos="0.2 0 0"><joint name="hinge" axis="0 1 0"/>
        <geom name="link" type="capsule" fromto="0 0 0 0.3 0 0" size="0.03"/></body>
    </body>
  </worldbody>
</mujoco>"""


class FakeMujoco:
    backend = "mujoco"

    def __init__(self, task, film, checkpoint, viewer=None):
        self.num_envs, self.frame_dt, self.k = film.num_envs, 1 / 30, 0
        self.m = mujoco.MjModel.from_xml_string(XML)

    def reset(self):
        self.k = 0

    def step(self):
        self.k += 1
        q = np.zeros((self.num_envs, self.m.nq))
        q[:, 3] = 1.0
        q[:, 0] = 0.01 * self.k + np.arange(self.num_envs)
        q[:, 2] = 0.5
        q[:, 7] = 0.05 * self.k
        done = np.array([self.k == 5 + e for e in range(self.num_envs)])
        return dict(qpos=q), done, dict(success=done.copy()), dict(sim_s=1e-3)

    def rank(self, first):
        return sorted(range(self.num_envs), key=lambda e: (not first["success"][e], e))

    def twin_spec(self):
        return mujoco.MjSpec.from_string(XML)

    def meta(self):
        return dict(policy=None, env_build_s=0.0, physics="fake")


def _task(adapter, **film):
    return C.FilmTask(name="fake", adapter=adapter, title="Fake", summary="A stand-in scene.",
                      film=C.FilmCfg(num_envs=3, max_seconds=0.5, **film),
                      viewer=C.ViewerCfg(origin_type="asset_body", body_name="base"),
                      look=C.LookCfg(hdri=None, ground_texture=None, materials={"link": dict(rgba=(0, 1, 0, 1))}),
                      assets=[C.AssetRef("cube", "here", "https://example.org/cube", "CC0")], doc="docs/renders/X.md")


def test_record_mujoco(tmp_path):
    task = _task(FakeMujoco)
    out = REC.record(task, task.film, task.viewer, str(tmp_path), None)
    assert out["envs"][:3] == [0, 1, 2] and out["frames"] == 7  # stops once every episode has ended
    R = np.load(tmp_path / "rec.npz")
    assert R["xpos"].shape == (7, 3, 3, 3) and list(R["ends"]) == [4, 5, 6]
    # poses = forward kinematics of the recorded qpos, the tracked body = base
    m = mujoco.MjModel.from_xml_path(str(tmp_path / "twin.xml"))
    d = mujoco.MjData(m)
    d.qpos[:] = 0; d.qpos[3] = 1; d.qpos[0] = 0.01 * 3 + 1; d.qpos[2] = 0.5; d.qpos[7] = 0.15
    mujoco.mj_kinematics(m, d)
    ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n) for n in out["geom_names"]]
    np.testing.assert_allclose(R["xpos"][2, 1], d.geom_xpos[ids], atol=1e-6)
    np.testing.assert_allclose(R["track"][2, 1], d.xpos[1], atol=1e-6)


def test_export_twin_buffer_texture(tmp_path):
    spec = mujoco.MjSpec()
    t = spec.add_texture(name="t", type=mujoco.mjtTexture.mjTEXTURE_2D, width=4, height=2, nchannel=3)
    img = (np.arange(24, dtype=np.uint8) * 10).reshape(2, 4, 3)
    t.data = img.tobytes()
    mt = spec.add_material(name="m"); mt.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "t"
    spec.worldbody.add_geom(size=[1, 1, 1], type=mujoco.mjtGeom.mjGEOM_BOX, material="m")
    spec.compile()
    REC.name_geoms(spec)
    REC.export_twin(spec, str(tmp_path))
    m = mujoco.MjModel.from_xml_path(str(tmp_path / "twin.xml"))
    assert m.ntex == 1 and m.ngeom == 1 and mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, 0)
    import imageio.v3 as iio
    np.testing.assert_array_equal(iio.imread(tmp_path / "twin_tex_0.png"), img)


def test_newton_twin(tmp_path):
    newton = pytest.importorskip("newton")
    import warp as wp

    class FakeNewton:
        backend = "newton"

        def __init__(self, task, film, checkpoint, viewer=None):
            self.num_envs, self.frame_dt, self.k = film.num_envs, 1 / 30, 0

        def _model(self):
            b = newton.ModelBuilder(up_axis="Z")
            b.add_ground_plane()
            body = b.add_body(xform=wp.transform(wp.vec3(0, 0, 1), wp.quat_identity()), label="/World/envs/env_0/cube")
            b.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1, label="/World/envs/env_0/cube/collider")
            v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float32) * 0.2
            f = np.array([0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3], np.int32)
            cfg = newton.ModelBuilder.ShapeConfig(has_shape_collision=False, has_particle_collision=False)
            b.add_shape_mesh(body, mesh=newton.Mesh(v, f), cfg=cfg, label="/World/envs/env_0/cube/visual")
            return b.finalize(device="cpu")

        def reset(self):
            pass

        def step(self):
            self.k += 1
            bq = np.zeros((self.num_envs, 1, 7), np.float32); bq[..., 6] = 1; bq[:, 0, 0] = 0.1 * self.k
            pq = np.zeros((self.num_envs, 5, 3), np.float32) + self.k
            return dict(body_q=bq, particles=pq), np.zeros(self.num_envs, bool), {}, dict(sim_s=1e-3)

        def rank(self, first):
            return list(range(self.num_envs))

        def twin_newton(self):
            return self._model()

        def meta(self):
            return dict(policy=None)

    task = _task(FakeNewton)
    out = REC.record(task, replace(task.film, num_envs=2, max_seconds=0.1), task.viewer, str(tmp_path), None)
    R = np.load(tmp_path / "rec.npz")
    names = out["geom_names"]
    assert "visual" in names and "collider" not in names          # the collider of a body with visuals is hidden
    assert out["twin"]["visibility"]["colliders_hidden"] == 1
    gi = names.index("visual")
    m = mujoco.MjModel.from_xml_path(str(tmp_path / "twin.xml"))
    # mesh geom frame = body pose x compiled (recentred) mesh frame
    np.testing.assert_allclose(R["xpos"][1, 0, gi], np.array([0.2, 0, 0]) + m.geom_pos[gi], atol=1e-5)
    assert R["particles"].shape == (4, 2, 5, 3) and float(R["particles"][3, 0, 0, 0]) == 4.0


def test_render_scene_and_doc(tmp_path):
    from metalsim.render.film import doc as D
    from metalsim.render.film.render import build_scene, camera, smooth_track

    task = _task(FakeMujoco)
    rec_dir = tmp_path / "rec"
    rec = REC.record(task, task.film, task.viewer, str(rec_dir), None)
    model, sky, sun, _ = build_scene(str(rec_dir), rec, task.look, task.viewer)
    assert model.ncam == 1 and model.nlight == 1 and sky is None
    g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "link")
    np.testing.assert_allclose(model.mat_rgba[model.geom_matid[g]], (0, 1, 0, 1))
    tr = smooth_track(np.load(rec_dir / "rec.npz")["track"].astype(float), 30.0, 0.1)
    eye, q = camera(task.viewer, 0, 0, tr)
    np.testing.assert_allclose(eye, tr[0, 0] + np.asarray(task.viewer.eye))
    shot = tmp_path / "hero"
    shot.mkdir()
    (shot / "render.json").write_text(json.dumps(dict(
        frames=10, fps=30.0, width=64, height=36, grid=1, panel=[64, 36], tier=2, samples_per_pixel=16, passes=4, spp=4,
        bounces=3, denoise="oidn", setup_s=1.0, probe=dict(one_pass_s=0.1, per_pass_s=0.05),
        software={"warp": {"version": "x"}})))
    (shot / "render_frames.jsonl").write_text("".join(json.dumps(dict(frame=k, seconds=0.5)) + "\n" for k in range(10)))
    text = D.write_doc(task, str(rec_dir), [("hero", str(shot))], str(tmp_path / "X.md"), ["python -m x"])
    assert "| render: hero |" in text and "https://example.org/cube" in text
    assert "/Users/" not in text and "/private/" not in text

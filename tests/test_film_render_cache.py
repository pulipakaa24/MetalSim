"""The film render stage's frame cache (metalsim.render.film.render): frames are keyed on the recording and the render
settings. A re-recording into the same shot directory re-renders every frame (the old frames, render.json and
render_frames.jsonl move to stale/, and no exposure, probe or hardware stamp carries over); an interrupted render resumes
where it stopped. CPU only: the record stage runs a stand-in MuJoCo scene, the path tracer is replaced by a stand-in
whose image depends on the recorded poses."""
from __future__ import annotations

import json
import os
from dataclasses import replace
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from metalsim.render.film import cfg as C
from metalsim.render.film import provenance
from metalsim.render.film import record as REC
from metalsim.render.film import render as RR

XML = """
<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="0 0 0.1"/>
    <body name="base" pos="0 0 0.5"><freejoint/><geom name="box" type="box" size="0.1 0.2 0.05"/></body>
  </worldbody>
</mujoco>"""


class Scene:
    """Stand-in recorder adapter: the base slides along x by ``speed`` per frame."""
    backend = "mujoco"
    speed = 0.01

    def __init__(self, task, film, checkpoint, viewer=None):
        self.num_envs, self.frame_dt, self.k = film.num_envs, 1 / 30, 0
        self.m = mujoco.MjModel.from_xml_string(XML)

    def reset(self):
        self.k = 0

    def step(self):
        self.k += 1
        q = np.zeros((self.num_envs, self.m.nq))
        q[:, 3] = 1.0
        q[:, 0] = self.speed * self.k + np.arange(self.num_envs)
        q[:, 2] = 0.5
        done = np.zeros(self.num_envs, bool)
        return dict(qpos=q), done, dict(success=done.copy()), dict(sim_s=1e-3)

    def rank(self, first):
        return list(range(self.num_envs))

    def twin_spec(self):
        return mujoco.MjSpec.from_string(XML)

    def meta(self):
        return dict(policy=None, env_build_s=0.0, physics="fake")


class FakeTier2:
    """Stand-in path tracer: radiance from the filmed base's x, so a different recording gives different frames."""
    calls = 0
    fail_after = None

    def __init__(self, model, n, *, width, height, **kw):
        self.n, self.w, self.h = n, width, height
        self.rt = SimpleNamespace(grid=SimpleNamespace(S=np.ones(3)), grid_growths=0)
        self.env_margin = kw.get("env_margin")

    def set_environment(self, *a, **k):
        pass

    def render_host(self, datas, passes=1, cam_pos=None, cam_quat=None, chunk=None):
        FakeTier2.calls += 1
        if FakeTier2.fail_after is not None and FakeTier2.calls > FakeTier2.fail_after:
            raise KeyboardInterrupt("render interrupted")
        x = float(np.max(np.abs(datas[0].geom_xpos[:, 0])))
        return {"hdr": np.full((self.n, self.h, self.w, 3), 0.1 + x, np.float32)}


@pytest.fixture
def film(tmp_path, monkeypatch):
    import metalsim.render.tier2 as T2
    monkeypatch.setattr(T2, "Tier2Renderer", FakeTier2)
    hw = {"model": "first"}
    monkeypatch.setattr(provenance, "hardware", lambda: dict(hw))
    FakeTier2.calls, FakeTier2.fail_after = 0, None
    task = C.FilmTask(name="fake", adapter=Scene, film=C.FilmCfg(num_envs=1, envs=[0], max_seconds=0.3, hold_s=0.0,
                                                                  passes=2, spp=1),
                      viewer=C.ViewerCfg(origin_type="asset_body", body_name="base", resolution=(32, 18)),
                      look=C.LookCfg(hdri=None, ground_texture=None))
    rec, out = tmp_path / "rec", tmp_path / "hero"

    def record(speed):
        Scene.speed = speed
        return REC.record(task, task.film, task.viewer, str(rec), None)

    return SimpleNamespace(task=task, rec=str(rec), out=str(out), record=record, hw=hw)


def _frames(out):
    return sorted(f for f in os.listdir(os.path.join(out, "frames")))


def _stats(out):
    return [json.loads(x)["frame"] for x in open(os.path.join(out, "render_frames.jsonl"))]


def test_rerecord_rerenders_every_frame(film):
    film.record(0.01)
    m1 = RR.render(film.task, film.rec, film.out)
    n = m1["frames"]
    assert n >= 5 and _frames(film.out) == [f"frame_{k:04d}.png" for k in range(n)]
    assert FakeTier2.calls == n + 2                     # + the timing probe's two single passes
    first = {f: open(os.path.join(film.out, "frames", f), "rb").read() for f in _frames(film.out)}
    assert m1["hardware"] == {"model": "first"} and m1["cache_key"]

    # same recording and settings: nothing to do, nothing appended
    FakeTier2.calls = 0
    RR.render(film.task, film.rec, film.out)
    assert FakeTier2.calls == 0 and _stats(film.out) == list(range(n))

    # re-record into the same rec dir (the robot moves differently), render on "another machine"
    film.record(0.05)
    film.hw["model"] = "second"
    FakeTier2.calls = 0
    m2 = RR.render(film.task, film.rec, film.out)
    assert FakeTier2.calls == n + 2                     # every frame rendered again, and a new timing probe
    assert m2["cache_key"] != m1["cache_key"] and m2["hardware"] == {"model": "second"}
    assert m2["exposure"] != m1["exposure"]             # automatic exposure measured on the new recording
    assert _stats(film.out) == list(range(n))           # no rows of the old render
    changed = [f for f in _frames(film.out) if open(os.path.join(film.out, "frames", f), "rb").read() != first[f]]
    assert len(changed) == n - 1                        # frame 0 is the same pose in both recordings
    stale = os.path.join(film.out, "stale")
    assert sorted(os.listdir(stale)) == ["frames", "render.json", "render_frames.jsonl"]
    assert json.load(open(os.path.join(stale, "render.json")))["cache_key"] == m1["cache_key"]

    # a render setting changes the key too
    FakeTier2.calls = 0
    m3 = RR.render(replace(film.task, film=replace(film.task.film, passes=3)), film.rec, film.out)
    assert FakeTier2.calls == n + 2 and m3["cache_key"] != m2["cache_key"]


def test_interrupted_render_resumes(film):
    film.record(0.01)
    FakeTier2.fail_after = 2 + 3                         # probe passes, then 3 frames, then the 4th frame is cut off
    with pytest.raises(KeyboardInterrupt):
        RR.render(film.task, film.rec, film.out)
    assert _frames(film.out) == [f"frame_{k:04d}.png" for k in range(3)]     # no partial frame or temp file left
    m1 = json.load(open(os.path.join(film.out, "render.json")))
    FakeTier2.calls, FakeTier2.fail_after = 0, None
    film.hw["model"] = "second"
    m2 = RR.render(film.task, film.rec, film.out)
    n = m2["frames"]
    assert FakeTier2.calls == n - 3                      # only the missing frames; the probe is not repeated
    assert m2["cache_key"] == m1["cache_key"] and m2["exposure"] == m1["exposure"] and m2["probe"] == m1["probe"]
    assert m2["hardware"] == {"model": "first"}          # resumed: the stamp of the run that started the shot
    assert _frames(film.out) == [f"frame_{k:04d}.png" for k in range(n)] and _stats(film.out) == list(range(n))
    assert not os.path.exists(os.path.join(film.out, "stale"))


def test_no_skip_existing_replaces_stats_rows(film):
    film.record(0.01)
    m = RR.render(film.task, film.rec, film.out)
    RR.render(film.task, film.rec, film.out, frames_arg="1:3", skip_existing=False)
    assert sorted(_stats(film.out)) == list(range(m["frames"]))                # re-rendered frames: one row each

"""CPU check of the SO101Lift controller option (no GPU: warp is blocked, BatchSim is a CPU MuJoCo stand-in).

1. Policy films unchanged: the baseline so101.py (copied before this stream's edits, scratch/so101-film/baseline/) and
   the edited one, each driving so101_lift, so101_lift_74v and so101_lift_smooth on the same CPU stand-in: snapshots,
   done flags, info, summary, rank and meta() must be equal bit for bit. The FilmTask fields that make a film (adapter,
   checkpoint, adapter_kw, film, viewer, shots) must be equal too. (look: the ground-scale stream's edit, reported.)
2. Expert wiring: the edited SO101Lift with controller "deploy_expert" on 100 CPU worlds against so101-expert's
   eval_metalsim.py --cpu-dry 100 (scratch/so101-expert/metalsim/dry100: the same loop on CPU MjData worlds):
   per-seed success, time to lift, highest box, retries and phase events, and the arm joint positions at every frame.

    .venv/bin/python runs/readme_media/scratch/so101-film/cpu_check.py [frames_policy=60]
"""
import contextlib
import copy
import json
import os
import sys
import types
from dataclasses import replace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, ROOT)

# -- no GPU: a fake warp and a CPU BatchSim ----------------------------------------------------------------------------
fake_wp = types.ModuleType("warp")
fake_wp.config = types.SimpleNamespace(quiet=False)
fake_wp.ScopedDevice = lambda dev: contextlib.nullcontext()
sys.modules["warp"] = fake_wp
import mujoco  # noqa: E402


class _State:
    """A BatchSim data field over CPU worlds: float32 readback (the GPU's precision), host writes."""

    def __init__(self, ds, name):
        self.ds, self.name = ds, name

    def numpy(self):
        return np.stack([getattr(d, self.name).copy() for d in self.ds]).astype(np.float32)

    def assign(self, a):
        for d, x in zip(self.ds, np.asarray(a)):
            getattr(d, self.name)[:] = x


class _ModelField:
    """A per-world model field (float64, as eval_metalsim.py --cpu-dry writes the CPU models)."""

    def __init__(self, ms, name):
        self.ms, self.name = ms, name

    def numpy(self):
        return np.stack([getattr(m, self.name).copy() for m in self.ms])

    def assign(self, a):
        for m, x in zip(self.ms, np.asarray(a)):
            getattr(m, self.name)[:] = x


class BatchSimOptions:
    def __init__(self, substeps=1, njmax=None, per_world_fields=(), **kw):
        self.substeps, self.njmax, self.per_world_fields = substeps, njmax, per_world_fields


class BatchSim:
    def __init__(self, m, n, options=None, device=None):
        self.opt = options or BatchSimOptions()
        self.n, self.device = n, None
        self.ms = [copy.deepcopy(m) for _ in range(n)]
        self.ds = [mujoco.MjData(mm) for mm in self.ms]
        self.d = types.SimpleNamespace(**{k: _State(self.ds, k) for k in ("qpos", "qvel", "xpos", "site_xpos", "cvel",
                                                                          "ctrl")})
        self.m = types.SimpleNamespace(body_mass=_ModelField(self.ms, "body_mass"),
                                       geom_friction=_ModelField(self.ms, "geom_friction"))
        self.steps = 0

    def reset(self):
        for mm, d in zip(self.ms, self.ds):
            mujoco.mj_resetData(mm, d)

    def set_state(self, q, qd):
        for mm, d, a, b in zip(self.ms, self.ds, q, qd):
            d.qpos[:], d.qvel[:] = a, b
            mujoco.mj_forward(mm, d)

    def step(self):
        for _ in range(self.opt.substeps):
            for mm, d in zip(self.ms, self.ds):
                mujoco.mj_step(mm, d)
        self.steps += self.opt.substeps

    def synchronize(self):
        pass

    def check_overflow(self):
        return {}

    def overflow_flags(self):
        return {}


fake_batch = types.ModuleType("metalsim.physics.batch")
fake_batch.BatchSim, fake_batch.BatchSimOptions = BatchSim, BatchSimOptions
import metalsim.physics  # noqa: E402,F401
sys.modules["metalsim.physics.batch"] = fake_batch

from metalsim.render.film import so101 as NEW  # noqa: E402


def load_baseline():
    """The pre-edit so101.py as a module (its look line adapted to the ground-scale stream's LookCfg, so it imports;
    the look is not part of the adapter)."""
    src = open(os.path.join(HERE, "baseline", "so101.py")).read()
    n = src.count("ground_repeat=1.0,")
    src = src.replace("ground_repeat=1.0,", "ground_tile_m=1.5,")
    mod = types.ModuleType("so101_baseline")
    mod.__file__ = NEW.__file__              # ROOT from the real location
    exec(compile(src, "baseline/so101.py", "exec"), mod.__dict__)
    return mod, n


def run(Adapter, task, film, frames):
    ck = task.checkpoint
    ck = os.path.join(ROOT, ck) if ck and not os.path.isabs(ck) else ck
    a = Adapter(task, film, ck)
    a.reset()
    out = []
    for _ in range(frames):
        snap, done, info, _t = a.step()
        out.append((snap["qpos"].copy(), np.asarray(done).copy(), {k: np.asarray(v).copy() for k, v in info.items()}))
    ends = np.full(a.num_envs, -1)
    return a, out, a.summary(ends)


def same(x, y):
    if isinstance(x, dict):
        return x.keys() == y.keys() and all(same(x[k], y[k]) for k in x)
    if isinstance(x, (list, tuple)):
        return len(x) == len(y) and all(same(a, b) for a, b in zip(x, y))
    if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
        return np.array_equal(np.asarray(x), np.asarray(y))
    return x == y


def check_policy(frames):
    OLD, n_patched = load_baseline()
    res = dict(baseline_look_lines_adapted=n_patched, tasks={})
    for name in ("SO101_LIFT", "SO101_LIFT_74V", "SO101_LIFT_SMOOTH"):
        to, tn = getattr(OLD, name), getattr(NEW, name)
        fields = {f: same(getattr(to, f), getattr(tn, f)) for f in ("name", "adapter", "venv", "checkpoint",
                                                                    "adapter_kw", "film", "viewer", "shots", "assets",
                                                                    "doc", "title", "summary")}
        look_diff = {k: (v, getattr(tn.look, k)) for k, v in vars(to.look).items() if v != getattr(tn.look, k)}
        film = replace(tn.film, num_envs=4)
        ao, ro, so = run(OLD.SO101Lift, to, film, frames)
        an, rn, sn = run(NEW.SO101Lift, tn, film, frames)
        steps = all(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) and same(a[2], b[2])
                    for a, b in zip(ro, rn))
        strip = lambda d: {k: v for k, v in d.items() if k != "env_build_s"}     # a wall-clock time, not a result
        mo, mn = json.dumps(strip(ao.meta()), default=str), json.dumps(strip(an.meta()), default=str)
        res["tasks"][tn.name] = dict(fields_equal=fields, look_diff_other_stream=look_diff,
                                     frames=frames, worlds=4, snapshots_done_info_equal=bool(steps),
                                     summary_equal=so == sn, rank_equal=ao.rank({}) == an.rank({}),
                                     meta_json_equal_except_env_build_s=mo == mn,
                                     max_qpos_diff=float(max(np.abs(a[0] - b[0]).max() for a, b in zip(ro, rn))),
                                     moved=float(np.abs(rn[-1][0] - rn[0][0]).max()))
        print(name, json.dumps(res["tasks"][tn.name], default=str), flush=True)
    return res


def check_expert():
    task = NEW.SO101_LIFT_DEPLOY
    film = task.film
    a, out, summ = run(NEW.SO101Lift, task, film, 300)
    print(summ.splitlines()[-1], flush=True)
    dry = json.load(open(os.path.join(ROOT, "runs/readme_media/scratch/so101-expert/metalsim/dry100/episodes.json")))
    by = {e["seed"]: e for e in dry["episodes"]}
    z = np.load(os.path.join(ROOT, "runs/readme_media/scratch/so101-expert/metalsim/dry100/traj.npz"))
    meta = a.meta()["first_episode"]
    rows, mism = [], []
    q_film = np.stack([o[0][:, a.arm_qadr] for o in out], 1)          # (E, 300, 6) after each frame
    q_dry = z["q"][:, 3::4]                                            # (E, 300, 6) after physics step 4k + 3
    for e in range(a.num_envs):
        s = a.seed0 + e
        d = by[s]
        f = dict(success=bool(a.succeeded[e]), time_to_lift_s=meta["success_t_s"][e], hmax=round(float(a.hmax[e]), 4),
                 retries=a.experts[e].retries, events=[list(x) for x in a.experts[e].events])
        g = dict(success=d["success"], time_to_lift_s=d["time_to_lift_s"], hmax=d["hmax"], retries=d["retries"],
                 events=d["events"])
        ok = all(same(f[k], g[k]) for k in f)
        live = ~np.isnan(q_dry[e]).any(1)
        qd = float(np.abs(q_film[e][live] - q_dry[e][live]).max()) if live.any() else None
        rows.append(dict(seed=s, equal=ok, q_maxdiff=qd))
        if not ok:
            mism.append(dict(seed=s, film=f, dry=g))
    eq = sum(r["equal"] for r in rows)
    qmax = max(r["q_maxdiff"] for r in rows if r["q_maxdiff"] is not None)
    res = dict(worlds=a.num_envs, frames=len(out), outcome_events_equal=eq, arm_q_max_abs_diff_rad=qmax,
               film=dict(successes=int(a.succeeded.sum()), still_up=int((a.succeeded & (a.end_height > NEW.LIFT_Z)).sum()),
                         retries=int(sum(x.retries for x in a.experts))),
               dry=dry["summary"], mismatches=mism,
               seed10051=dict(film=meta["success_t_s"][51], retries=meta["retries"][51], events=meta["events"][51]))
    print(json.dumps({k: v for k, v in res.items() if k != "mismatches"}, default=str), flush=True)
    if mism:
        print("mismatches:", json.dumps(mism, default=str)[:3000])
    return res


if __name__ == "__main__":
    frames = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    out = dict(policy=check_policy(frames), expert=check_expert())
    json.dump(out, open(os.path.join(HERE, "cpu_check.json"), "w"), indent=1, default=str)
    print("wrote", os.path.relpath(os.path.join(HERE, "cpu_check.json"), ROOT))

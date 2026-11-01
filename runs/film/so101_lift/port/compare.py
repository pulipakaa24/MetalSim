"""MetalSim against so101Sim's CPU MuJoCo reference (reference.py) for the so101_lift film: the film's adapter
(``metalsim.render.film.so101.SO101Lift``, MuJoCo Warp on Metal, one world per seed) on eval_lift_native.py's seeds.

Compares the start states (qpos, box mass and friction scales) seed by seed, the eval figures (success, mean of the
max box height, time to lift) over all seeds, the per-seed outcome, and the trajectories (box position, arm joint
positions) step by step. A GPU-queue job (render kind):

  scripts/gpu_run.sh so101-compare render 10 -- .venv/bin/python runs/film/so101_lift/port/compare.py \
      <reference_dir> <out_dir> [policy.pt]
"""
import json
import os
import sys
from types import SimpleNamespace

import numpy as np

from metalsim.render.film import provenance
from metalsim.render.film.so101 import CTRL_DT, LIFT_Z, MAX_STEPS, POLICY, SO101Lift

ref_dir, out_dir = sys.argv[1:3]
policy = sys.argv[3] if len(sys.argv) > 3 else os.path.join(provenance.ROOT, POLICY)
os.makedirs(out_dir, exist_ok=True)
R = np.load(os.path.join(ref_dir, "reference.npz"), allow_pickle=False)
RJ = json.load(open(os.path.join(ref_dir, "reference.json")))
E = int(RJ["summary"]["episodes"])
film = SimpleNamespace(num_envs=E, seed=10_000)
task = SimpleNamespace(adapter_kw={})
ad = SO101Lift(task, film, policy)
ad.reset()
T = MAX_STEPS
qpos = np.zeros((E, T + 1, ad.model.nq))
box = np.zeros((E, T + 1, 3))
qpos[:, 0], box[:, 0] = ad.start_qpos, ad.state[2]
act = np.zeros((E, T, 6), np.float32)
height = np.zeros((E, T))
for t in range(T):
    snap, done, info, _ = ad.step()
    qpos[:, t + 1], box[:, t + 1] = snap["qpos"], ad.last["box"]
    act[:, t], height[:, t] = ad.last["action"], ad.last["height"]
ov = ad.sim.check_overflow()

start = dict(qpos_max_abs_diff=float(np.abs(qpos[:, 0] - R["qpos"][:, 0]).max()),
             mass_scale_max_abs_diff=float(np.abs(ad.mass_scale - R["mass_scale"]).max()),
             friction_scale_max_abs_diff=float(np.abs(ad.fric_scale - R["friction_scale"]).max()))
ok = ad.succeeded
ref_ok = np.array([e["success"] for e in RJ["episodes"]])
st = ad.success_t
mine = dict(successes=int(ok.sum()), max_height_mean=float(ad.hmax.mean()), max_height_min=float(ad.hmax.min()),
            time_to_lift_mean_s=float(st[ok].mean() * CTRL_DT) if ok.any() else None,
            lifted_at_end=int((ad.end_height > LIFT_Z).sum()), terminated=int(0))
ref_st = np.array([e["success_t"] if e["success_t"] is not None else -1 for e in RJ["episodes"]])
both = ok & ref_ok
per_seed = dict(agree=int((ok == ref_ok).sum()), only_metalsim=[int(10_000 + e) for e in np.flatnonzero(ok & ~ref_ok)],
                only_reference=[int(10_000 + e) for e in np.flatnonzero(~ok & ref_ok)],
                time_to_lift_abs_diff_s=dict(mean=float(np.abs(st[both] - ref_st[both]).mean() * CTRL_DT),
                                             max=float(np.abs(st[both] - ref_st[both]).max() * CTRL_DT)),
                max_height_abs_diff_m=dict(mean=float(np.abs(ad.hmax - [e["hmax"] for e in RJ["episodes"]]).mean()),
                                           max=float(np.abs(ad.hmax - [e["hmax"] for e in RJ["episodes"]]).max())))


def traj(e):
    db = np.linalg.norm(box[e] - R["box"][e], axis=1)
    dq = np.abs(qpos[e][:, ad.arm_qadr] - R["qpos"][e][:, ad.arm_qadr]).max(1)
    first = lambda x, thr: (int(np.argmax(x > thr)) * CTRL_DT if (x > thr).any() else None)
    at = lambda x, s: float(x[min(int(round(s / CTRL_DT)), T)])
    return dict(seed=10_000 + e, box_m_at=dict((f"{s}s", at(db, s)) for s in (0.25, 0.5, 1.0, 2.0, 6.0)),
                box_m_max=float(db.max()), arm_rad_at=dict((f"{s}s", at(dq, s)) for s in (0.25, 0.5, 1.0, 2.0, 6.0)),
                arm_rad_max=float(dq.max()), first_box_over_1mm_s=first(db, 1e-3), first_box_over_1cm_s=first(db, 1e-2),
                action_max_abs_diff_first_10=float(np.abs(act[e, :10] - R["action"][e, :10]).max()),
                height_end=dict(metalsim=float(height[e, -1]), reference=float(R["height"][e, -1])))


trajs = [traj(e) for e in range(min(E, 5))]
dball = np.linalg.norm(box - R["box"], axis=2)                  # (E, T+1)
out = dict(script="runs/film/so101_lift/port/compare.py", policy=provenance.relpath(policy), reference=RJ["summary"],
           reference_mujoco=RJ["mujoco"], metalsim=mine, start_state=start, per_seed=per_seed, trajectories=trajs,
           box_diff_median_over_seeds_m={f"{s}s": float(np.median(dball[:, int(round(s / CTRL_DT))]))
                                         for s in (0.1, 0.25, 0.5, 1.0, 2.0, 6.0)},
           capacity_overflow=ov, adapter=ad.meta(), software=provenance.software(), hardware=provenance.hardware())
json.dump(out, open(os.path.join(out_dir, "compare.json"), "w"), indent=1,
          default=lambda o: o.item() if hasattr(o, "item") else str(o))
np.savez_compressed(os.path.join(out_dir, "compare.npz"), qpos=qpos, box=box, action=act, height=height)
print(json.dumps(dict(reference=RJ["summary"], metalsim=mine, start=start, per_seed=per_seed,
                      box_diff_median=out["box_diff_median_over_seeds_m"]), indent=1, default=str))
for tr in trajs:
    print(tr)

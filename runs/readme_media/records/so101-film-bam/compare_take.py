"""[so101-film-bam copy of so101-film/compare_take.py, for the take on 7v4_bam: the references are the 7v4_bam evals.]

The so101_lift_deploy take's own 100 worlds (rec.json) against the 7v4_bam evals, seed by seed.

Reads runs/film/so101_lift_deploy/rec/rec.json (adapter.first_episode: success, success_t_s, max / end height,
retries, phase events, per world = seed 10000 + world) and the episodes.json of
  cpu      : CPU MuJoCo 3.10.0, so101Sim's env (so101-sag eval_cpu_sag.py --patched 7v4_bam)
  ms_eval  : MetalSim, so101-bam-gpu eval_metalsim_bam.py (one GPU draw)
  cpu314   : CPU MuJoCo 3.14.0, the film's scene and reset (eval_metalsim_bam.py --cpu-dry 100)
Writes compare_take.json next to this script and prints the summary.

    .venv/bin/python runs/readme_media/scratch/so101-film-bam/compare_take.py
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
EX = os.path.join(ROOT, "runs/readme_media/scratch")
REF = dict(cpu="so101-sag/cpu/patched_7v4_bam/episodes.json", ms_eval="so101-bam-gpu/metalsim/deploy_7v4_bam/episodes.json",
           cpu314="so101-bam-gpu/dry/dry100_7v4_bam/episodes.json")
LIFT_Z = 0.10

rec = json.load(open(os.path.join(ROOT, "runs/film/so101_lift_deploy/rec/rec.json")))
assert rec["adapter"]["servo"]["servo"] == "7v4_bam", rec["adapter"]["servo"]["servo"]
fe = rec["adapter"]["first_episode"]
take = {}
for i, s in enumerate(fe["seeds"]):
    take[s] = dict(seed=s, success=fe["success"][i], time_to_lift_s=fe["success_t_s"][i], hmax=fe["max_height_m"][i],
                   final_height=fe["end_height_m"][i], up_at_end=bool(fe["success"][i] and fe["end_height_m"][i] > LIFT_Z),
                   retries=fe["retries"][i], events=fe["events"][i])


def summary(eps):
    succ = [e for e in eps if e["success"]]
    t = [e["time_to_lift_s"] for e in succ]
    return dict(successes=len(succ), up_at_6s=sum(bool(e["up_at_end"]) for e in eps),
                time_to_lift_median_s=float(np.median(t)) if t else None,
                time_to_lift_mean_s=round(float(np.mean(t)), 3) if t else None,
                p10_p90=[round(float(np.percentile(t, 10)), 3), round(float(np.percentile(t, 90)), 3)] if t else None,
                retries=sum(e["retries"] for e in eps),
                retry_seeds=[e["seed"] for e in eps if e["retries"]],
                failures=[e["seed"] for e in eps if not e["success"]],
                not_up_at_6s=[e["seed"] for e in eps if not e["up_at_end"]])


out = dict(take=dict(summary=summary(list(take.values())), filmed_envs=rec["envs"],
                     episodes_filmed={str(e): take[fe["seeds"][e]] for e in rec["envs"]},
                     capacity_overflow=rec["adapter"].get("capacity_overflow"),
                     mujoco_warp=rec["software"].get("mujoco_warp"), warp=rec["software"].get("warp"),
                     metalsim=rec["software"].get("metalsim"), power=rec["hardware"].get("power")),
           refs={}, vs={})
for k, p in REF.items():
    d = json.load(open(os.path.join(EX, p)))
    by = {e["seed"]: e for e in d["episodes"]}
    out["refs"][k] = summary(list(by.values()))
    same = [s for s in take if take[s]["success"] == by[s]["success"]]
    both = [s for s in take if take[s]["success"] and by[s]["success"]]
    dt = np.array([abs(take[s]["time_to_lift_s"] - by[s]["time_to_lift_s"]) for s in both])
    ev_same = [s for s in take if [list(x) for x in by[s]["events"]] == take[s]["events"]]
    out["vs"][k] = dict(same_success=len(same), differ=[s for s in take if s not in same], both_lift=len(both),
                        time_to_lift_within_0p1s=int((dt <= 0.1 + 1e-9).sum()), time_to_lift_equal=int((dt < 1e-9).sum()),
                        abs_dt_median_s=float(np.median(dt)) if len(dt) else None,
                        abs_dt_max_s=float(dt.max()) if len(dt) else None,
                        same_retry_count=sum(take[s]["retries"] == by[s]["retries"] for s in take),
                        phase_events_identical=len(ev_same),
                        filmed={str(s): dict(time_to_lift_s=by[s]["time_to_lift_s"], retries=by[s]["retries"],
                                             events=by[s]["events"]) for s in (fe["seeds"][e] for e in rec["envs"])})
json.dump(out, open(os.path.join(HERE, "compare_take.json"), "w"), indent=1, default=str)
print(json.dumps(dict(take=out["take"]["summary"], filmed=out["take"]["episodes_filmed"],
                      overflow=out["take"]["capacity_overflow"], mjw=out["take"]["mujoco_warp"]), default=str))
for k in REF:
    print(k, json.dumps(out["refs"][k]))
    print("  vs take:", json.dumps({x: y for x, y in out["vs"][k].items() if x != "filmed"}))

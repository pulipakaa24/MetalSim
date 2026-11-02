"""G1 (Unitree policy) falls on Isaac's random_rough tile (noise_range (0.02, 0.10)) over tile seeds, via the g1_bumps
measurement (runs/film/g1_bumps/measure/measure.py: 16 robots, 8 s, command (0.5, 0, 0); fall = pelvis < 0.4 m).
Seed 0 is the tile of the g1_bumps hand-off (CPU 11/16). Usage (repo root):
  .venv/bin/python runs/readme_media/scratch/hfield-contacts/falls/measure_seeds.py cpu|gpu <tag> <seed> [<seed> ...]
Writes falls/<mode>_<tag>.json: per seed falls, first/median fall time, forward median, mujoco_warp file."""
import json, os, sys
sys.path.insert(0, "runs/film/g1_bumps/measure")
import measure as M  # noqa: E402

mode, tag, seeds = sys.argv[1], sys.argv[2], [int(s) for s in sys.argv[3:]]
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{mode}_{tag}.json")
res = dict(mode=mode, tag=tag, ceiling=0.10, robots=M.N, seconds=M.SECONDS, seeds={})
if mode == "gpu":
  import mujoco_warp
  res["mujoco_warp"] = mujoco_warp.__file__
# CPU_PERTURB=<eps>:<rep>: add N(0, eps) rad to the start joint angles (CPU MuJoCo is deterministic; this gives it the
# run-to-run spread MuJoCo Warp has on Metal). rep seeds the noise.
if os.environ.get("CPU_PERTURB"):
  import numpy as np
  eps, rep = os.environ["CPU_PERTURB"].split(":"); eps = float(eps); rng = np.random.default_rng(int(rep))
  _start = M.U.unitree_start_qpos
  def _perturbed(*a, **k):
    q = _start(*a, **k); q = q.copy(); q[:, 7:] += rng.normal(0, eps, q[:, 7:].shape); return q
  M.U.unitree_start_qpos = _perturbed
  res["cpu_perturb"] = os.environ["CPU_PERTURB"]
for s in seeds:
  M.SEED = s
  m, terrain, fg, Q, C, NF, frame_dt, start, lanes, info = (M.run_gpu if mode == "gpu" else M.run_cpu)(0.10)
  summary, rows, g = M.summarise(m, terrain, fg, Q, C, NF, frame_dt, start, lanes, M.CMD)
  res["seeds"][str(s)] = dict(falls=summary["falls"], fall_s_first=summary["fall_s_first"], fall_s_median=summary["fall_s_median"],
                              forward_m_median=summary["forward_m_median"], nonfoot=summary["nonfoot_contacts"],
                              fell=[bool(r.get("fell", r.get("fall_s") is not None)) for r in rows], info=info)
  print(f"[{mode} {tag}] seed {s}: falls {summary['falls']}/{M.N} first {summary['fall_s_first']} median {summary['fall_s_median']} "
        f"forward {summary['forward_m_median']} {info}", flush=True)
  json.dump(res, open(out, "w"), indent=1, default=float)
tot = sum(v["falls"] for v in res["seeds"].values())
print(f"[{mode} {tag}] total falls {tot}/{M.N * len(seeds)}")

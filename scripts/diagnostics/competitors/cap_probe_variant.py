"""Newton warm-start / cap probe for a solver variant (cap_probe_ellip.py's protocol): does the variant, at cap C, change
states beyond the plain solver's cap-100 re-run noise floor? States come from a rollout with the plain solver at cap 100
(4096 envs, a policy checkpoint's mean action or uniform random actions, resets by the task). At 10 checkpoints the state
(qpos, qvel, ctrl, qacc_warmstart and the variant's warm-start buffers) is saved; from each, one control step (8 substeps
of 2.5 ms) is run with the plain solver at cap 100 (twice: the second is the floor), with the plain solver at cap 20 (the
preset's cap) and with the variant at caps 10 / 20 / 40 / 100, and compared per world (p99 |dqvel| and the count of worlds
with |dqvel| > 0.01 are the robust measures; single-world maxima are chaotic because contact order is nondeterministic).
The variant is selected by module-level knobs of the fork flipped at runtime (eager launches, no graph):
    VARIANT=extrap    solver._WARMSTART_EXTRAP (MJW_WARMSTART_EXTRAP)
    VARIANT=lsfloor   solver._LS_NOISE_FLOOR_EPS = LS_FLOOR_EPS (default 8; MJW_LS_NOISE_FLOOR)
usage: VARIANT=extrap python cap_probe_variant.py TERRAIN CONTACT_CFG random|CKPT.pt [OUT.json]"""
import sys, os, json, numpy as np, torch, warp as wp, mujoco_warp as mjw
wp.config.quiet = True
from mujoco_warp._src import solver
from metalsim.learn.g1_velocity import G1VelocityTask
from metalsim.learn.warp_policy import ActorCriticMLP, RolloutBuffers, bump
terrain, cfg, src = sys.argv[1], sys.argv[2], sys.argv[3]; OUT = sys.argv[4] if len(sys.argv) > 4 else None
VARIANT = os.environ.get("VARIANT", "extrap")
N = int(os.environ.get("CAP_N", "4096")); CAPS = [10, 20, 40, 100]


LS_FLOOR_EPS = float(os.environ.get("LS_FLOOR_EPS", "8"))


def set_variant(on: bool):
  if VARIANT == "extrap":
    solver._WARMSTART_EXTRAP = on
  elif VARIANT == "lsfloor":
    solver._LS_NOISE_FLOOR_EPS = LS_FLOOR_EPS if on else 0.0
  else:
    raise SystemExit(f"unknown VARIANT {VARIANT}")


set_variant(False)
task = G1VelocityTask(N, terrain=terrain, seed=0, physics_dt=0.0025, reward_cfg="flat" if terrain == "flat" else "rough_isaac", contact_cfg=cfg)
d, m = task.sim.d, task.sim.m
print(f"mujoco_warp {os.path.dirname(mjw.__file__)}; variant {VARIANT}; cone {m.opt.cone} impratio_invsqrt {m.opt.impratio_invsqrt.numpy().tolist()} cap {m.opt.iterations}; N {N}", flush=True)
task.reset_all()
net = None
if src != "random":
  ck = torch.load(src, map_location="mps", weights_only=False); sd = ck["net"]
  hidden = tuple(sd[k].shape[0] for k in sorted((k for k in sd if k.startswith("actor.") and k.endswith("weight")), key=lambda s: int(s.split(".")[1]))[:-1])
  net = ActorCriticMLP(task.obs_dim, task.act_dim, hidden=hidden).to("mps"); net.load_state_dict(sd); net.eval()


class _Pol:
  step_idx = wp.zeros(1, dtype=int, device="metal:0"); rng_step = wp.zeros(1, dtype=int, device="metal:0")


pol = _Pol(); bufs = RolloutBuffers(1, N, task.obs_dim, task.act_dim)
bufs.rew = wp.zeros((1, N), dtype=float, device="metal:0"); bufs.done = wp.zeros((1, N), dtype=float, device="metal:0")
rng = np.random.default_rng(0)
wp.launch(bump, dim=1, inputs=[pol.rng_step], device="metal:0"); task.launch_obs(pol.rng_step); task.sim.synchronize()
F = tuple(f for f in ("qpos", "qvel", "ctrl", "qacc_warmstart", "qacc_ws_prev", "qacc_ws_last") if getattr(d, f, None) is not None)


def control_step(cap, variant):
  set_variant(variant); m.opt.iterations = cap
  hit = np.zeros(N, bool); nmax = 0; nsum = np.zeros(N)
  with wp.ScopedDevice("metal:0"):
    for s in range(task.decimation):
      mjw.step(m, d)
      ni = d.solver_niter.numpy(); hit |= ni >= cap; nmax = max(nmax, int(ni.max())); nsum += ni
  set_variant(False)
  return hit, nmax, nsum / task.decimation


save = lambda: {f: getattr(d, f).numpy().copy() for f in F}


def load(st):
  for f in F:
    getattr(d, f).assign(st[f])


def action():
  if net is None:
    return rng.uniform(-1, 1, (N, task.act_dim)).astype(np.float32)
  task.sim.synchronize()
  with torch.no_grad():
    return net.actor(torch.as_tensor(task.obs.numpy(), device="mps")).cpu().numpy().astype(np.float32)


def diff(cur, ref, hit=None):
  dq = np.abs(cur["qpos"] - ref["qpos"]).max(1); dv = np.abs(cur["qvel"] - ref["qvel"]).max(1)
  r = {"max_dqpos": float(dq.max()), "max_dqvel": float(dv.max()), "p99_dqvel": float(np.percentile(dv, 99)), "median_dqvel": float(np.median(dv)),
       "worlds_dqvel_gt_0.01": int((dv > 0.01).sum()), "worlds_dqvel_gt_0.1": int((dv > 0.1).sum())}
  if hit is not None:
    r["max_dqvel_in_hitting"] = float(dv[hit].max()) if hit.any() else None
  return r


res = {"terrain": terrain, "contact_cfg": cfg, "actions": src, "variant": VARIANT, "caps": CAPS, "points": []}
for k in range(200):
  a = wp.array(action(), dtype=float, device="metal:0")
  task.launch_apply_action(a)
  if k % 20 == 19:
    st = save(); hit100, n100, mean100 = control_step(100, False); ref = save()
    load(st); control_step(100, False); floor = save()
    row = {"step": k, "plain_cap100": {"max_niter": n100, "mean_niter": float(mean100.mean()), "worlds_at_cap": int(hit100.sum())},
           "pelvis_z_mean": float(st["qpos"][:, 2].mean()), "pelvis_z_below_0.4": int((st["qpos"][:, 2] < 0.4).sum()), "floor": diff(floor, ref)}
    load(st); hit, nmax, meanc = control_step(20, False); cur = save()
    row["plain_cap20"] = dict(worlds_hitting_cap=int(hit.sum()), mean_niter=float(meanc.mean()), max_niter=nmax, **diff(cur, ref, hit))
    for c in CAPS:
      load(st); hit, nmax, meanc = control_step(c, True); cur = save()
      row[f"{VARIANT}_cap{c}"] = dict(worlds_hitting_cap=int(hit.sum()), mean_niter=float(meanc.mean()), max_niter=nmax, **diff(cur, ref, hit))
    res["points"].append(row); print(json.dumps(row), flush=True)
    load(ref)
  else:
    control_step(100, False)
  wp.launch(bump, dim=1, inputs=[pol.rng_step], device="metal:0")
  task.launch_reward_done_reset(pol, bufs); task.launch_obs(pol.rng_step)
P = res["points"]
keys = ["floor", "plain_cap20"] + [f"{VARIANT}_cap{c}" for c in CAPS]
for key in keys:
  s = {"p99_dqvel_max": max(p[key]["p99_dqvel"] for p in P), "p99_dqvel_mean": float(np.mean([p[key]["p99_dqvel"] for p in P])),
       "median_dqvel_max": max(p[key]["median_dqvel"] for p in P), "max_dqvel": max(p[key]["max_dqvel"] for p in P),
       "worlds_dqvel_gt_0.01_mean": float(np.mean([p[key]["worlds_dqvel_gt_0.01"] for p in P])), "worlds_dqvel_gt_0.1_mean": float(np.mean([p[key]["worlds_dqvel_gt_0.1"] for p in P]))}
  if key != "floor":
    s.update({"worlds_hitting_cap_mean_per_step": float(np.mean([p[key]["worlds_hitting_cap"] for p in P])), "mean_niter": float(np.mean([p[key]["mean_niter"] for p in P])),
              "max_niter": max(p[key]["max_niter"] for p in P)})
  res["summary_" + key] = s
res["summary_plain_cap100"] = {"mean_niter": float(np.mean([p["plain_cap100"]["mean_niter"] for p in P])), "max_niter": max(p["plain_cap100"]["max_niter"] for p in P)}
print(json.dumps({k: v for k, v in res.items() if k != "points"}, indent=1))
if OUT:
  json.dump(res, open(OUT, "w"), indent=1)

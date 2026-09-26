"""Newton iteration statistics on dumped G1-task states (g1_state_dump.py: qpos, qvel, ctrl, act, qacc_warmstart at
control-step boundaries of a policy rollout), stepping the 8 substeps of each control step from every snapshot,
elliptic (the dump's cone / impratio) vs pyramidal (impratio 1, the pyramidal preset), Newton cap 100:

  --engine c     MuJoCo C (the oracle): per world-substep solver_niter and, per Newton iteration, the solver stats
                 (neval = line-search cost evaluations, nchange = rows changing state); which start C's warmstart()
                 picked (engine_forward.c: the better of qacc_warmstart and qacc_smooth by cost); rows in the CONE state
                 at the start of the solve (forward with iterations = 0); zero-iteration exits.
  --engine warp  MuJoCo Warp on the CPU device: per world-substep solver_niter, CONE rows and QUADRATIC flips per
                 iteration (hooked _solver_iteration, first substep), under a warm-start variant:
                   ws      qacc_warmstart = the previous substep's qacc (the fork's behaviour)
                   smooth  qacc_smooth (warmstart disabled)
                   extrap  2 qacc_prev - qacc_prevprev from the second substep on (plain ws on the first)
                 Knobs of the fork (MJW_*) are read from the environment as usual.
Output: one JSON line per (engine, cone, variant) with the niter distribution (mean, p50, p90, p99, max, fraction
>= 10 and >= 20 of world-substeps) and the per-iteration statistics.
    python g1_newton_iters.py runs/competitors/g1_states/ellip10 --engine c --worlds 256 [--cone elliptic|pyramidal|both]
    python g1_newton_iters.py runs/competitors/g1_states/ellip10 --engine warp --variant ws --worlds 128 --cone elliptic"""
import argparse, json, os, sys, time
import numpy as np, mujoco

ap = argparse.ArgumentParser()
ap.add_argument("prefix"); ap.add_argument("--engine", default="c"); ap.add_argument("--worlds", type=int, default=256)
ap.add_argument("--cone", default="both"); ap.add_argument("--variant", default="ws"); ap.add_argument("--substeps", type=int, default=8)
ap.add_argument("--cap", type=int, default=100); ap.add_argument("--snapshots", default=None, help="comma list of snapshot steps (default all)")
ap.add_argument("--out", default=None)
a = ap.parse_args()
z = np.load(a.prefix + ".npz"); N = a.worlds; S = a.substeps
steps = [int(t) for t in z["steps"]] if a.snapshots is None else [int(x) for x in a.snapshots.split(",")]
cones = ["elliptic", "pyramidal"] if a.cone == "both" else [a.cone]
CONE_ST = int(mujoco.mjtConstraintState.mjCNSTRSTATE_CONE); QUAD_ST = int(mujoco.mjtConstraintState.mjCNSTRSTATE_QUADRATIC)
ELL_T = int(mujoco.mjtConstraint.mjCNSTR_CONTACT_ELLIPTIC)


def model_for(cone):
  m = mujoco.MjModel.from_binary_path(a.prefix + ".mjb")
  if cone == "pyramidal":
    m.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL; m.opt.impratio = 1.0
  else:
    m.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
  m.opt.iterations = a.cap
  return m


def dist(niter):
  n = np.asarray(niter, dtype=float).ravel()
  return dict(count=int(n.size), mean=float(n.mean()), p50=float(np.percentile(n, 50)), p90=float(np.percentile(n, 90)),
              p99=float(np.percentile(n, 99)), max=int(n.max()), frac_ge10=float((n >= 10).mean()), frac_ge20=float((n >= 20).mean()),
              frac_eq0=float((n == 0).mean()), frac_le2=float((n <= 2).mean()))


def load_world(d, k, w):
  d.qpos[:] = z[k + "_qpos"][w]; d.qvel[:] = z[k + "_qvel"][w]; d.ctrl[:] = z[k + "_ctrl"][w]
  if z[k + "_act"].size:
    d.act[:] = z[k + "_act"][w]
  d.qacc_warmstart[:] = z[k + "_qacc_warmstart"][w]; d.time = 0.0


def run_c(cone):
  m = model_for(cone); d = mujoco.MjData(m)
  niter = np.zeros((len(steps), N, S), int); chose_smooth = np.zeros((len(steps), N, S), bool)
  cone_rows0 = np.zeros((len(steps), N, S), int); ell_rows = np.zeros((len(steps), N, S), int); nefc = np.zeros((len(steps), N, S), int)
  per_it = {}   # iteration index -> list of (neval, nchange, nactive)
  ls_cap_hits = 0; total_its = 0
  t0 = time.time()
  for si, t in enumerate(steps):
    k = f"t{t}"
    for w in range(N):
      load_world(d, k, w)
      for s in range(S):
        # start of the solve: C's warmstart() choice and the constraint states there (no Newton iterations)
        m.opt.iterations = 0; mujoco.mj_forward(m, d)
        chose_smooth[si, w, s] = not np.array_equal(d.qacc, d.qacc_warmstart)
        st = d.efc_state[: d.nefc]; ty = d.efc_type[: d.nefc]
        cone_rows0[si, w, s] = int((st == CONE_ST).sum()); ell_rows[si, w, s] = int((ty == ELL_T).sum()); nefc[si, w, s] = d.nefc
        m.opt.iterations = a.cap; mujoco.mj_step(m, d)
        ni = int(d.solver_niter[0]); niter[si, w, s] = ni
        for i in range(ni):
          st_i = d.solver[i]
          per_it.setdefault(i, []).append((st_i.neval, st_i.nchange, st_i.nactive))
          ls_cap_hits += int(st_i.neval >= m.opt.ls_iterations); total_its += 1
  res = dict(engine="c", cone=cone, impratio=float(m.opt.impratio), variant="c_bestof", worlds=N, substeps=S, snapshots=steps, cap=a.cap,
             seconds=round(time.time() - t0, 1), niter=dist(niter), niter_first_substep=dist(niter[:, :, 0]), niter_later_substeps=dist(niter[:, :, 1:]),
             chose_smooth_frac=float(chose_smooth.mean()), chose_smooth_frac_first_substep=float(chose_smooth[:, :, 0].mean()),
             niter_when_chose_smooth=dist(niter[chose_smooth]) if chose_smooth.any() else None,
             niter_when_chose_ws=dist(niter[~chose_smooth]),
             nefc_mean=float(nefc.mean()), elliptic_rows_mean=float(ell_rows.mean()),
             cone_rows_at_start_mean=float(cone_rows0.mean()), cone_rows_at_start_frac_of_elliptic=float(cone_rows0.sum() / max(1, ell_rows.sum())),
             worlds_with_cone_rows_at_start_frac=float((cone_rows0 > 0).mean()),
             ls_evals_per_iteration=[[i, len(v), float(np.mean([x[0] for x in v])), float(np.percentile([x[0] for x in v], 90)), float(np.mean([x[1] for x in v]))] for i, v in sorted(per_it.items()) if i < 30],
             ls_evals_mean_all=float(np.mean([x[0] for v in per_it.values() for x in v])), ls_cap_hit_frac=ls_cap_hits / max(1, total_its),
             niter_by_cone_rows_at_start={str(c): dist(niter[cone_rows0 == c]) for c in sorted(set(np.unique(cone_rows0).tolist())) if (cone_rows0 == c).sum() > 50} if cone == "elliptic" else None)
  return res


def run_warp(cone):
  import warp as wp
  wp.config.quiet = True
  import mujoco_warp as mjw
  from mujoco_warp._src import solver, types
  m = model_for(cone)
  if a.variant == "smooth":
    m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_WARMSTART
  rec = []
  orig_iter = solver._solver_iteration

  def hooked(*args, **kw):
    dd = kw.get("d", args[1] if len(args) > 1 else None); ctx = kw.get("ctx", args[2] if len(args) > 2 else None)
    sb = dd.efc.state.numpy().copy(); orig_iter(*args, **kw)
    rec.append((sb, dd.efc.state.numpy().copy(), ctx.done.numpy().copy()))
  solver._solver_iteration = hooked
  CONE, QUAD = int(types.ConstraintState.CONE), int(types.ConstraintState.QUADRATIC); ELL = int(types.ConstraintType.CONTACT_ELLIPTIC)
  niter = np.zeros((len(steps), N, S), int); cone1 = []; flips1 = []; alive = {}; cone0 = []; ell0 = []
  t0 = time.time()
  with wp.ScopedDevice("cpu"):
    mm = mjw.put_model(m); d0 = mujoco.MjData(m)
    dd = mjw.put_data(m, d0, nworld=N, njmax=int(z["njmax"]), naconmax=int(z["nconmax"]) * N)
    for si, t in enumerate(steps):
      k = f"t{t}"
      dd.qpos.assign(z[k + "_qpos"][:N]); dd.qvel.assign(z[k + "_qvel"][:N]); dd.ctrl.assign(z[k + "_ctrl"][:N]); dd.qacc_warmstart.assign(z[k + "_qacc_warmstart"][:N])
      if z[k + "_act"].size:
        dd.act.assign(z[k + "_act"][:N])
      dd.time.zero_()
      prev = z[k + "_qacc_warmstart"][:N].copy(); prevprev = None
      for s in range(S):
        if a.variant == "extrap" and prevprev is not None:
          dd.qacc_warmstart.assign((2.0 * prev - prevprev).astype(np.float32))
        rec.clear(); mjw.step(mm, dd)
        ni = dd.solver_niter.numpy()[:N]; niter[si, :, s] = ni
        qacc = dd.qacc.numpy()[:N].copy(); prevprev, prev = prev, qacc
        if a.variant != "extrap" or True:
          dd.qacc_warmstart.assign(qacc)   # what the fork does (step already did; keeps extrap's bookkeeping explicit)
        if s == 0 and rec:
          nefc = dd.nefc.numpy()[:N]; rows = np.arange(rec[0][0].shape[1])[None, :] < nefc[:, None]
          for it, (sb, sa, done) in enumerate(rec):
            active = rows & ~done[:N, None]
            alive.setdefault(it, []).append(int((~done[:N]).sum()))
            if it == 0:
              cone1.append(((sa == CONE) & active).sum(1)); flips1.append((((sb == QUAD) != (sa == QUAD)) & active).sum(1))
              cone0.append(((sb == CONE) & rows).sum(1)); ell0.append(((dd.efc.type.numpy()[:N] == ELL) & rows).sum(1))
      print(f"  [{cone} {a.variant}] snapshot t={t}: niter mean {niter[si].mean():.2f} max {niter[si].max()} ({time.time() - t0:.0f} s)", file=sys.stderr, flush=True)
  cone1 = np.concatenate(cone1); flips1 = np.concatenate(flips1); cone0 = np.concatenate(cone0); ell0 = np.concatenate(ell0)
  res = dict(engine="warp_cpu", cone=cone, impratio=float(m.opt.impratio), variant=a.variant, worlds=N, substeps=S, snapshots=steps, cap=a.cap,
             seconds=round(time.time() - t0, 1), niter=dist(niter), niter_first_substep=dist(niter[:, :, 0]), niter_later_substeps=dist(niter[:, :, 1:]),
             cone_rows_at_start_mean=float(cone0.mean()), cone_rows_at_start_frac_of_elliptic=float(cone0.sum() / max(1, ell0.sum())), elliptic_rows_mean=float(ell0.mean()),
             cone_rows_iter1_mean=float(cone1.mean()), worlds_with_cone_rows_iter1_frac=float((cone1 > 0).mean()), quad_flips_iter1_mean=float(flips1.mean()),
             worlds_alive_by_iteration_first_substep=[float(np.mean(v)) for _, v in sorted(alive.items())][:40],
             env={k: v for k, v in os.environ.items() if k.startswith("MJW_")})
  return res


out = []
for cone in cones:
  r = run_c(cone) if a.engine == "c" else run_warp(cone)
  print(json.dumps(r), flush=True); out.append(r)
if a.out:
  with open(a.out, "w") as f:
    json.dump(out, f, indent=1)

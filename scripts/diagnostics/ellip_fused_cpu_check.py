"""Fused vs unfused per-iteration Newton launches on the CPU device (deterministic: Warp's CPU atomics run in order),
on dumped G1-task states (g1_state_dump.py), elliptic cones on the incremental mode-2 path (MJW_FUSE_H_CHOLESKY_CPU=1,
MJW_METAL_FUSE_UPDATE_CPU=1 are set here): per substep, max |difference| of qacc, qfrc_constraint, efc_force, efc_state
and solver_niter between (a) the unfused launches, (b) the fused kernel with the separate htot launch, (c) the fused kernel
with the htot fusion; and the launch count per Newton iteration (wp.launch / wp.launch_tiled calls inside
_solver_iteration) for each. Also pyramidal (impratio 1) fused vs unfused, whose kernel signature changed.
    python scripts/diagnostics/ellip_fused_cpu_check.py runs/competitors/g1_states/ellip10 [--worlds 64] [--substeps 8]"""
import argparse, os, sys
os.environ.setdefault("MJW_FUSE_H_CHOLESKY_CPU", "1"); os.environ.setdefault("MJW_METAL_FUSE_UPDATE_CPU", "1")
import numpy as np, mujoco, warp as wp
wp.config.quiet = True
import mujoco_warp as mjw
from mujoco_warp._src import solver

ap = argparse.ArgumentParser(); ap.add_argument("prefix"); ap.add_argument("--worlds", type=int, default=64); ap.add_argument("--substeps", type=int, default=8)
ap.add_argument("--snapshot", type=int, default=60); ap.add_argument("--cap", type=int, default=20)
a = ap.parse_args(); z = np.load(a.prefix + ".npz"); N = a.worlds; k = f"t{a.snapshot}"
print("mujoco_warp", os.path.dirname(mjw.__file__), "device cpu; knobs", {e: os.environ.get(e) for e in ("MJW_FUSE_H_CHOLESKY_CPU", "MJW_METAL_FUSE_UPDATE_CPU", "MJW_ELLIPTIC_INCREMENTAL")})

launches = {"n": 0}
_orig_launch, _orig_launch_tiled = wp.launch, wp.launch_tiled
def _cl(*args, **kw):
  launches["n"] += 1; return _orig_launch(*args, **kw)
def _clt(*args, **kw):
  launches["n"] += 1; return _orig_launch_tiled(*args, **kw)
per_iter = []
_orig_iter = solver._solver_iteration
def _hooked(*args, **kw):
  n0 = launches["n"]; r = _orig_iter(*args, **kw); per_iter.append(launches["n"] - n0); return r
solver._solver_iteration = _hooked


def run(cone, fuse, fuse_htot):
  solver._METAL_FUSE_UPDATE = fuse; solver._METAL_FUSE_HTOT = fuse_htot
  m = mujoco.MjModel.from_binary_path(a.prefix + ".mjb")
  if cone == "pyramidal":
    m.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL; m.opt.impratio = 1.0
  m.opt.iterations = a.cap
  out = []
  with wp.ScopedDevice("cpu"):
    mm = mjw.put_model(m); dd = mjw.put_data(m, mujoco.MjData(m), nworld=N, njmax=int(z["njmax"]), naconmax=int(z["nconmax"]) * N)
    dd.qpos.assign(z[k + "_qpos"][:N]); dd.qvel.assign(z[k + "_qvel"][:N]); dd.ctrl.assign(z[k + "_ctrl"][:N]); dd.qacc_warmstart.assign(z[k + "_qacc_warmstart"][:N])
    if z[k + "_act"].size:
      dd.act.assign(z[k + "_act"][:N])
    solver._solver_iteration, wp.launch, wp.launch_tiled = _hooked, _cl, _clt
    for s in range(a.substeps):
      per_iter.clear(); launches["n"] = 0
      mjw.step(mm, dd)
      out.append(dict(qacc=dd.qacc.numpy().copy(), qfrc=dd.qfrc_constraint.numpy().copy(), force=dd.efc.force.numpy().copy(), state=dd.efc.state.numpy().copy(),
                      niter=dd.solver_niter.numpy().copy(), qpos=dd.qpos.numpy().copy(), qvel=dd.qvel.numpy().copy(), launches_per_iter=list(per_iter), launches_step=launches["n"]))
    wp.launch, wp.launch_tiled = _orig_launch, _orig_launch_tiled
  return out


def compare(tag, A, B):
  for s, (x, y) in enumerate(zip(A, B)):
    same_niter = int((x["niter"] != y["niter"]).sum())
    print(f"  {tag} substep {s}: max|dqacc| {np.abs(x['qacc'] - y['qacc']).max():.3e}  max|dqfrc| {np.abs(x['qfrc'] - y['qfrc']).max():.3e}  "
          f"max|dforce| {np.abs(x['force'] - y['force']).max():.3e}  state rows differing {int((x['state'] != y['state']).sum())}  worlds with different niter {same_niter}  "
          f"max|dqpos| {np.abs(x['qpos'] - y['qpos']).max():.3e}  max|dqvel| {np.abs(x['qvel'] - y['qvel']).max():.3e}")


for cone in ("elliptic", "pyramidal"):
  U = run(cone, False, False); F = run(cone, True, False)
  print(f"[{cone}] launches per Newton iteration: unfused {sorted(set(U[0]['launches_per_iter']))} (per substep {U[0]['launches_step']}), fused {sorted(set(F[0]['launches_per_iter']))} (per substep {F[0]['launches_step']}); "
        f"niter mean {U[-1]['niter'].mean():.2f} max {U[-1]['niter'].max()}")
  compare(f"{cone} fused vs unfused", U, F)
  if cone == "elliptic":
    H = run(cone, True, True)
    print(f"[{cone}] launches per Newton iteration, fused + htot fusion: {sorted(set(H[0]['launches_per_iter']))} (per substep {H[0]['launches_step']})")
    compare(f"{cone} fused+htot vs unfused", U, H)

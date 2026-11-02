"""so101-bam-gpu, no GPU: where the MuJoCo Warp (Warp CPU device, float32) vs MuJoCo C (float64) gap comes from, on the
mjw_cpu_preflight.py scenario (film scene, N worlds from reset_draw, start-pose targets, odd worlds shoulder +0.4 rad,
STEPS physics steps). Variants, the same on both sides:
  real        7v4_real (no BAM friction)
  bam_frozen  7v4_bam, friction constraint stiffened, dof_frictionloss frozen at the budget at rest (no update)
  bam         7v4_bam with the update (mjw: BamFrictionWarp hook after each step; C: BamFriction.update before)
  bam_soft    bam, but MuJoCo's default friction solref / solimp (not stiffened)
  bam_iter    bam with solver iterations 100, ls_iterations 50, on both sides
  bam_tol0    bam with solver tolerance 0 on both sides (every iteration runs; no early stop in float64)
Prints max |dq| per joint, the step where shoulder_lift's |dq| first exceeds 1e-6, and the solver iterations used.

  .venv/bin/python mjw_cpu_variants.py [steps] [variant ...] [--ctrl-first]
"""
import json
import os
import sys
from types import SimpleNamespace

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import patched  # noqa: E402,F401

import warp as wp  # noqa: E402

import mujoco_warp as mjw  # noqa: E402

import metalsim.render.film.so101 as S  # noqa: E402
from metalsim.physics.actuation import BamFriction, BamFrictionWarp  # noqa: E402

wp.config.quiet = True
N = 4
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 60
CTRL_FIRST = "--ctrl-first" in sys.argv   # C side: ctrl set before the first mj_forward, as the mjw side (default: after,
#                                            the first run's order: the first budget from qfrc_actuator at ctrl 0)
VARIANTS = [a for a in sys.argv[2:] if not a.startswith("--")] or ["real", "bam_frozen", "bam", "bam_soft", "bam_iter", "bam_tol0"]


def scenario(variant):
    m = mujoco.MjModel.from_xml_path(os.path.join(S.ROOT, S.SCENE))
    ap = S.apply_servo(m, "7v4_real" if variant == "real" else "7v4_bam")
    fr = ap.get("friction")
    dofs = np.array(ap.get("friction", {}).get("dofs", range(6)))
    if variant == "bam_soft":
        m0 = mujoco.MjModel.from_xml_path(os.path.join(S.ROOT, S.SCENE))
        m.dof_solref[dofs], m.dof_solimp[dofs] = m0.dof_solref[dofs], m0.dof_solimp[dofs]
    if variant == "bam_iter":
        m.opt.iterations, m.opt.ls_iterations = 100, 50
    if variant == "bam_tol0":
        m.opt.tolerance = 0.0
    hook = fr is not None and variant != "bam_frozen"
    return m, ap, fr, dofs, hook


def run(variant):
    m, ap, fr, dofs, hook = scenario(variant)
    oid = lambda t, s: mujoco.mj_name2id(m, t, s)
    box_qadr = int(m.jnt_qposadr[oid(mujoco.mjtObj.mjOBJ_JOINT, "box")])
    arm_qadr = np.array([int(m.jnt_qposadr[oid(mujoco.mjtObj.mjOBJ_JOINT, j)]) for j in S.ARM_JOINTS])
    q0 = np.stack([S.reset_draw(10_000 + e, m.qpos0.copy(), box_qadr, arm_qadr, True)[0] for e in range(N)])
    ctrl = np.zeros((N, m.nu)); ctrl[:, ap["actuators"]] = q0[:, arm_qadr]; ctrl[1::2, ap["actuators"][1]] += 0.4
    with wp.ScopedDevice("cpu"):
        mjd = mujoco.MjData(m); mujoco.mj_forward(m, mjd)
        wm = mjw.put_model(m, batch_sizes={"dof_frictionloss": N})
        wm.opt.warn_overflow = 0
        wd = mjw.put_data(m, mjd, nworld=N, njmax=256)
        hooks = []
        sim = SimpleNamespace(n=N, device=wp.get_device("cpu"), d=wd, m=wm, synchronize=lambda: None,
                              add_substep_hook=lambda fn, reset_fn=None: hooks.append(fn))
        fh = BamFrictionWarp(sim, dofs, fr["params"], fr["emf"]) if hook else None
        wd.qpos.assign(q0.astype(np.float32)); wd.qvel.zero_(); wd.ctrl.assign(ctrl.astype(np.float32))
        mjw.forward(wm, wd)
        if fh is not None:
            fh.update()
        qg = np.zeros((STEPS, N, 6)); itg = np.zeros((STEPS, N))
        for t in range(STEPS):
            mjw.step(wm, wd)
            for h in hooks:
                h()
            qg[t] = wd.qpos.numpy()[:, arm_qadr]
            itg[t] = wd.solver_niter.numpy()
    fc = BamFriction(m, dofs, fr["params"], fr["emf"]) if hook else None
    qc = np.zeros((STEPS, N, 6)); itc = np.zeros((STEPS, N))
    rest = m.dof_frictionloss.copy()
    for w in range(N):
        d = mujoco.MjData(m)
        m.dof_frictionloss[:] = rest
        d.qpos[:] = q0[w]
        if CTRL_FIRST:
            d.ctrl[:] = ctrl[w]
        mujoco.mj_forward(m, d); d.ctrl[:] = ctrl[w]
        for t in range(STEPS):
            if fc is not None:
                fc.update(d)
            mujoco.mj_step(m, d)
            qc[t, w] = d.qpos[arm_qadr]
            itc[t, w] = d.solver_niter[0]
    err = np.abs(qg - qc)
    onset = int(np.argmax(err[:, :, 1].max(1) > 1e-6)) if (err[:, :, 1] > 1e-6).any() else None
    r = dict(max_dq=err.max((0, 1)).tolist(), onset_step_lift_1e6=onset,
             max_dq_by_world=err.max((0, 2)).tolist(),
             mjw_iters_mean=float(itg.mean()), mjw_iters_max=int(itg.max()), c_iters_mean=float(itc.mean()), c_iters_max=int(itc.max()))
    print(variant, json.dumps({k: (["%.1e" % x for x in v] if isinstance(v, list) else v) for k, v in r.items()}), flush=True)
    return r


out = {v: run(v) for v in VARIANTS}
os.makedirs(os.path.join(HERE, "check"), exist_ok=True)
json.dump(dict(steps=STEPS, worlds=N, variants=out), open(os.path.join(HERE, "check", f"mjw_cpu_variants_{STEPS}{'_ctrlfirst' if CTRL_FIRST else ''}.json"), "w"), indent=1)

"""so101-bam-gpu pre-flight, no GPU: MuJoCo Warp on Warp's CPU device (float32) with BamFrictionWarp as the hook after
every mjw.step, against CPU MuJoCo (float64) with BamFriction.update before every mj_step. Same model (film scene,
apply_servo "7v4_bam"), N worlds from reset_draw, fixed per-world targets (start pose; odd worlds shoulder +0.4 rad),
60 physics steps. Checks the real MuJoCo Warp arrays (dtypes, efc layout, batched dof_frictionloss) and the hook's
step ordering before the Metal job; it does not test Metal codegen or BatchSim's graph capture.

  .venv/bin/python mjw_cpu_preflight.py
"""
import os
import sys
import time
from types import SimpleNamespace

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import patched  # noqa: E402,F401

import warp as wp  # noqa: E402

import mujoco_warp as mjw  # noqa: E402

import metalsim.render.film.so101 as S  # noqa: E402
from metalsim.physics.actuation import BamFrictionWarp  # noqa: E402

wp.config.quiet = True
N, STEPS = 4, 60
m = mujoco.MjModel.from_xml_path(os.path.join(S.ROOT, S.SCENE))
ap = S.apply_servo(m, "7v4_bam")
fr = ap["friction"]
dofs = np.array(fr["dofs"])
oid = lambda t, s: mujoco.mj_name2id(m, t, s)
box_qadr = int(m.jnt_qposadr[oid(mujoco.mjtObj.mjOBJ_JOINT, "box")])
arm_qadr = np.array([int(m.jnt_qposadr[oid(mujoco.mjtObj.mjOBJ_JOINT, j)]) for j in S.ARM_JOINTS])
q0 = np.stack([S.reset_draw(10_000 + e, m.qpos0.copy(), box_qadr, arm_qadr, True)[0] for e in range(N)])
ctrl = np.zeros((N, m.nu)); ctrl[:, ap["actuators"]] = q0[:, arm_qadr]; ctrl[1::2, ap["actuators"][1]] += 0.4
t0 = time.time()
with wp.ScopedDevice("cpu"):
    mjd = mujoco.MjData(m); mujoco.mj_forward(m, mjd)
    wm = mjw.put_model(m, batch_sizes={"dof_frictionloss": N})
    wd = mjw.put_data(m, mjd, nworld=N, njmax=256)
    hooks = []
    sim = SimpleNamespace(n=N, device=wp.get_device("cpu"), d=wd, m=wm, synchronize=lambda: None,
                          add_substep_hook=lambda fn, reset_fn=None: hooks.append(fn))
    fh = BamFrictionWarp(sim, dofs, fr["params"], fr["emf"])
    print("dof_frictionloss", wm.dof_frictionloss.shape, wm.dof_frictionloss.dtype, "efc.type", wd.efc.type.dtype,
          "nefc", wd.nefc.dtype, flush=True)
    wd.qpos.assign(q0.astype(np.float32)); wd.qvel.zero_(); wd.ctrl.assign(ctrl.astype(np.float32))
    mjw.forward(wm, wd)
    fh.update()
    qg = np.zeros((STEPS, N, 6)); fg = np.zeros((STEPS, N, 6))
    for t in range(STEPS):
        mjw.step(wm, wd)
        for h in hooks:
            h()
        qg[t] = wd.qpos.numpy()[:, arm_qadr]
        fg[t] = wm.dof_frictionloss.numpy()[:, dofs]
print(f"mjw cpu: {time.time() - t0:.0f} s (compile included)", flush=True)
fh_cpu = S.friction_hook(ap, m)
dq, dfl = np.zeros(N), np.zeros(N)
qc_all = np.zeros((STEPS, N, 6)); fc_all = np.zeros((STEPS, N, 6))
for w in range(N):
    d = mujoco.MjData(m)
    fh_cpu.reset()
    d.qpos[:] = q0[w]; mujoco.mj_forward(m, d); d.ctrl[:] = ctrl[w]
    for t in range(STEPS):
        fh_cpu.update(d)
        mujoco.mj_step(m, d)
        qc_all[t, w] = d.qpos[arm_qadr]
        fh_cpu.update(d)
        fc_all[t, w] = m.dof_frictionloss[dofs]
print("max |q_mjwcpu - q_cpu| per joint:", ["%.1e" % x for x in np.abs(qg - qc_all).max((0, 1))])
print("max |fl_mjwcpu - fl_cpu| per joint:", ["%.1e" % x for x in np.abs(fg - fc_all).max((0, 1))])
print("budget range CPU:", fc_all.min().round(4), fc_all.max().round(4), "; motion range:",
      (qc_all.max(0) - qc_all.min(0)).max(0).round(3).tolist())

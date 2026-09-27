# CPU check: G1 C_drop under the speculative prototype; efc normal forces (signed) vs total vertical momentum change.
import os, sys, numpy as np, mujoco, warp as wp
wp.config.quiet = True
import mujoco_warp as mjw
from metalsim.learn.g1_velocity import build_g1_model
from metalsim.physics import contact_tuning as ct
p = sys.argv[1]; S0, S1 = int(sys.argv[2]), int(sys.argv[3])
m, _ = build_g1_model("flat", physics_dt=0.0025); ct.apply(m, p); m.opt.iterations = 20; m.opt.ls_iterations = 20
d = mujoco.MjData(m); d.qpos[:] = m.key_qpos[0]; d.qpos[2] = 1.0; d.ctrl[:] = m.key_qpos[0][7:]; mujoco.mj_forward(m, d)
M = m.body_subtreemass[0]; md = mujoco.MjData(m)
def Pz(q, v):
    md.qpos[:] = q; md.qvel[:] = v; mujoco.mj_forward(m, md); mujoco.mj_subtreeVel(m, md); return M * md.subtree_linvel[0][2]
with wp.ScopedDevice("cpu"):
    wm = mjw.put_model(m); wd = mjw.put_data(m, d, nworld=1, nconmax=32, njmax=256)
    P0 = Pz(wd.qpos.numpy()[0], wd.qvel.numpy()[0])
    for s in range(S1):
        mjw.step(wm, wd)
        q = wd.qpos.numpy()[0]; v = wd.qvel.numpy()[0]; P1 = Pz(q, v)
        if s >= S0:
            nc = wd.nacon.numpy()[0]; dist = wd.contact.dist.numpy()[:nc]; ea = wd.contact.efc_address.numpy()[:nc, 0]; g = wd.contact.geom.numpy()[:nc]
            f = wd.efc.force.numpy()[0]; it = wd.solver_niter.numpy()[0]
            rows = [(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g[i][1]])[:5], round(float(dist[i]) * 1e3, 2), round(float(f[ea[i]]), 0)) for i in range(nc) if ea[i] >= 0]
            Fn = sum(r[2] for r in rows)
            print(s, "niter", it, "dPz/dt + Mg = %.0f N" % ((P1 - P0) / 0.0025 + M * 9.81), "| sum efc normal %.0f N" % Fn, rows)
        P0 = P1

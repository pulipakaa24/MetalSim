# CPU-device check of the speculative-gap prototype (no GPU): G1 1 m drop, left-foot contacts around landing.
import os, sys, numpy as np, mujoco, warp as wp
wp.config.quiet = True
import mujoco_warp as mjw
from metalsim.learn.g1_velocity import build_g1_model
from metalsim.physics import contact_tuning as ct
p = sys.argv[1]
m, _ = build_g1_model("flat", physics_dt=0.0025); ct.apply(m, p); m.opt.iterations = 20; m.opt.ls_iterations = 20
d = mujoco.MjData(m); d.qpos[:] = m.key_qpos[0]; d.qpos[2] = 1.0; d.ctrl[:] = m.key_qpos[0][7:]; mujoco.mj_forward(m, d)
with wp.ScopedDevice("cpu"):
    wm = mjw.put_model(m); wd = mjw.put_data(m, d, nworld=1, nconmax=32, njmax=256)
    lf = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
    peak = 0; rows = []
    for s in range(400):
        mjw.step(wm, wd)
        nc = wd.nacon.numpy()[0]; dist = wd.contact.dist.numpy()[:nc]; g = wd.contact.geom.numpy()[:nc]; ea = wd.contact.efc_address.numpy()[:nc, 0]
        sel = [i for i in range(nc) if m.geom_bodyid[g[i][1]] == lf or m.geom_bodyid[g[i][0]] == lf]
        dd = [(round(float(dist[i]) * 1e3, 2), int(ea[i] >= 0)) for i in sel]
        if sel: peak = max(peak, -min(dist[i] for i in sel))
        if 90 <= s <= 102: print(s, "left contacts (dist mm, row)", dd, "foot z %.2f mm" % (wd.xpos.numpy()[0, lf, 2] * 1e3), "vz %.3f" % wd.cvel.numpy()[0, lf, 5])
    print(p, "SPEC", os.environ.get("MJW_SPECULATIVE_GAP"), "left-foot peak penetration %.2f mm" % (peak * 1e3), "final contacts", dd)

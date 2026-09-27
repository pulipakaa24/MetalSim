# CPU check of the speculative prototype on a box dropped on a plane: are efc forces on gap rows consistent with qacc?
import numpy as np, mujoco, warp as wp
wp.config.quiet = True
import mujoco_warp as mjw
XML = """<mujoco><option timestep="0.0025" cone="elliptic" impratio="10" iterations="20" ls_iterations="20"/>
<worldbody><geom name="ground" type="plane" size="1 1 .1" gap="0.01" solref="0.005 1" solimp="0.9 0.999 0.005 0.5 2"/>
<body pos="0 0 0.1"><freejoint/><geom type="box" size=".1 .03 .01" mass="3" solref="0.005 1" solimp="0.9 0.999 0.005 0.5 2"/></body></worldbody></mujoco>"""
m = mujoco.MjModel.from_xml_string(XML); d = mujoco.MjData(m); mujoco.mj_forward(m, d)
with wp.ScopedDevice("cpu"):
    wm = mjw.put_model(m); wd = mjw.put_data(m, d, nworld=1, nconmax=16, njmax=64)
    for s in range(200):
        mjw.step(wm, wd)
        nc = wd.nacon.numpy()[0]; ne = wd.nefc.numpy()[0]
        dist = wd.contact.dist.numpy()[:nc]; ea = wd.contact.efc_address.numpy()[:nc, 0]
        f = wd.efc.force.numpy()[0]; qacc = wd.qacc.numpy()[0]; z = wd.qpos.numpy()[0, 2]; vz = wd.qvel.numpy()[0, 2]
        if s % 5 == 0 or (nc and dist.min() < 0.004 and s < 80):
            print(s, "z %.5f vz %+.3f qacc_z %+9.2f" % (z, vz, qacc[2]), "contacts", [(round(float(dist[i]) * 1e3, 2), round(float(f[ea[i]]), 1)) for i in range(nc) if ea[i] >= 0],
                  "sum normal F %.1f N (m g = %.1f)" % (sum(f[ea[i]] for i in range(nc) if ea[i] >= 0), 3 * 9.81))

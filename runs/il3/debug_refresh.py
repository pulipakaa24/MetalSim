import numpy as np, warp as wp, mujoco, mujoco_warp as mjw, sys
sys.path.insert(0, "tests")
wp.config.quiet = True
from test_solver_presets import _standing_sim
from metalsim.physics import solver_presets
m, sim = _standing_sim("isaaclab3", n=1)
d = sim.d; dev = sim.device
nc = int(d.nacon.numpy()[0])
print("before", nc, d.contact.dist.numpy()[:nc], d.contact.pos.numpy()[:nc], d.contact.geom.numpy()[:nc], d.contact.frame.numpy()[:nc, 0])
la = wp.zeros(d.naconmax, dtype=wp.vec3, device=dev); lb = wp.zeros(d.naconmax, dtype=wp.vec3, device=dev)
with wp.ScopedDevice(dev):
    wp.launch(solver_presets._save_witness, dim=d.naconmax, inputs=[d.nacon, d.contact.dist, d.contact.pos, d.contact.frame,
              d.contact.geom, d.contact.worldid, sim.m.geom_bodyid, d.xpos, d.xmat, la, lb])
sim.synchronize(); print("la", la.numpy()[:nc], "lb", lb.numpy()[:nc])
q = d.qpos.numpy().copy(); q[:, 2] += 0.001; d.qpos.assign(q)
with wp.ScopedDevice(dev):
    mjw.kinematics(sim.m, d)
    wp.launch(solver_presets._refresh_contacts, dim=d.naconmax, inputs=[d.nacon, d.contact.frame, d.contact.geom,
              d.contact.worldid, sim.m.geom_bodyid, d.xpos, d.xmat, la, lb, d.contact.dist, d.contact.pos, d.contact.efc_address])
sim.synchronize(); print("refreshed", d.contact.dist.numpy()[:nc], d.contact.pos.numpy()[:nc])
with wp.ScopedDevice(dev): mjw.forward(sim.m, d)
sim.synchronize(); nf = int(d.nacon.numpy()[0]); print("fresh", nf, d.contact.dist.numpy()[:nf], d.contact.pos.numpy()[:nf])

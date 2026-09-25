"""Newton fork item 6: exact per-body net contact forces from SolverXPBD's contact impulses
(``SolverXPBD(body_contact_forces=True).body_contact_force``), in the layout of metalsim.sensors.contact.ContactSensor
(net_forces_w: (N, B, 3), world frame, force ON the tracked body; force_mode "normal" = normal parts only, "total" =
normal + friction). ``gather_net_forces`` below is the reference kernel for a Newton ContactSensor.

Check: G1 standing quasi-statically (feet on the ground, ankle kp 200, as rest_height.py), 1.5 s, CPU: the feet's
summed vertical force must equal the robot's weight; compared with (a) the per-contact forces of update_contacts
summed per body (vector) and (b) NewtonSim's current touch sensordata (per-shape sum of |per-contact force|, last substep).
usage: python scripts/diagnostics/newton_fork/contact_forces.py [--it 4] [--dt_ms 1.25] [--kw k=v ...]"""
import argparse, numpy as np, mujoco, warp as wp, newton
wp.config.quiet = True
from metalsim.learn.g1_velocity import build_g1_model
from metalsim.physics import newton_backend as nb
import metalsim.interop.warp_metal as wm

class _E:
    def __init__(self, *x): self.v = 0
    def next_value(self): self.v += 1; return self.v
wm.SharedEvent = _E


@wp.kernel
def gather_net_forces(body_contact_force: wp.array[wp.spatial_vector], nb: int, bodies: wp.array[int], total: int,
                      net: wp.array3d[float]):
    """(N, B, 3) net contact force on tracked body b of env e (Newton body e * nb + bodies[b])."""
    e, b = wp.tid()
    f = body_contact_force[e * nb + bodies[b]]
    v = wp.spatial_bottom(f)
    if total != 0:
        v = wp.spatial_top(f)
    net[e, b, 0] = v[0]
    net[e, b, 1] = v[1]
    net[e, b, 2] = v[2]


ap = argparse.ArgumentParser(); ap.add_argument("--it", type=int, default=4); ap.add_argument("--dt_ms", type=float, default=1.25)
ap.add_argument("--kw", nargs="*", default=[]); ap.add_argument("--n", type=int, default=2)
a = ap.parse_args()
skw = {"body_contact_forces": True, "joint_armature_inertia": "none"}
for kv in a.kw:
    k, v = kv.split("="); skw[k] = eval(v)
m, _ = build_g1_model("flat", physics_dt=0.0025); n = a.n
sim = nb.NewtonSim(m, n, iterations=a.it, dt=a.dt_ms * 1e-3, device="cpu", relaxation=0.4, solver_kw=skw)
names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, m.actuator_trnid[i, 0]) for i in range(m.nu)]
kp = sim.actuator.kp.numpy(); kd = sim.actuator.kd.numpy(); eff = sim.actuator.effort.numpy(); dof = sim.dof_of.numpy()
for e in range(n):
    for i, nm in enumerate(names):
        if "ankle" in nm:
            d = e * sim.nd + dof[i]; kp[d] = 200.0; kd[d] = 5.0; eff[d] = 300.0
sim.actuator.kp.assign(kp); sim.actuator.kd.assign(kd); sim.actuator.effort.assign(eff)
q0 = np.tile(m.key_qpos[0].astype(np.float32), (n, 1)); q0[:, 2] -= 0.0184259; q0[:, 0] = np.arange(n) * 2.5
sim.d.qpos.assign(q0); sim.d.qvel.zero_(); sim._reset_mask.fill_(True); sim.launch_reset()
for _ in range(int(1.5 / 0.02)):
    sim.launch_step()
M = sim.model; lab = [l.split("/")[-1] for l in M.body_label[: sim.nb]]
track = ["left_ankle_roll_link", "right_ankle_roll_link", "torso_link", "pelvis"]
bodies = wp.array([lab.index(t) for t in track], dtype=int, device="cpu")
net = {}
for mode, tot in (("normal", 0), ("total", 1)):
    out = wp.zeros((n, len(track), 3), dtype=float, device="cpu")
    wp.launch(gather_net_forces, dim=(n, len(track)), inputs=[sim.solver.body_contact_force, sim.nb, bodies, tot], outputs=[out], device="cpu")
    net[mode] = out.numpy()
# per-contact forces of update_contacts, summed per body as vectors (the solver stores the force on shape0's body side)
ct = sim.contacts; k = int(ct.rigid_contact_count.numpy()[0]); f = ct.force.numpy()[:k, :3]
s0, s1 = ct.rigid_contact_shape0.numpy()[:k], ct.rigid_contact_shape1.numpy()[:k]; sb = M.shape_body.numpy()
per = np.zeros((M.body_count, 3))
for c in range(k):
    if s0[c] >= 0 and sb[s0[c]] >= 0: per[sb[s0[c]]] += f[c]
    if s1[c] >= 0 and sb[s1[c]] >= 0: per[sb[s1[c]]] -= f[c]
weight = float(M.body_mass.numpy()[: sim.nb].sum()) * 9.81
touch = sim.d.sensordata.numpy()
adr = [int(m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, t + "_touch")]) for t in ("left_ankle_roll_link", "right_ankle_roll_link")]
print(f"newton at {newton.__file__}; XPBD {a.it} it / {a.dt_ms} ms, solver kwargs {skw}; robot weight {weight:.2f} N")
for e in range(n):
    feet_n = net["normal"][e, :2]; feet_t = net["total"][e, :2]
    pc = per[[e * sim.nb + lab.index(t) for t in track[:2]]]
    print(f"| env {e} | body_contact_force normal: L {np.round(feet_n[0], 2).tolist()} R {np.round(feet_n[1], 2).tolist()} sum z {feet_n[:, 2].sum():.2f} N "
          f"({feet_n[:, 2].sum() / weight:.4f} x weight) | total sum {np.round(feet_t.sum(0), 2).tolist()} | "
          f"update_contacts summed per body: sum z {pc[:, 2].sum():.2f} N ({pc[:, 2].sum() / weight:.4f}) | "
          f"NewtonSim touch (|f| sums) L {touch[e, adr[0]]:.2f} R {touch[e, adr[1]]:.2f} sum {touch[e, adr].sum():.2f} ({touch[e, adr].sum() / weight:.4f}) |")
print("torso / pelvis net force (not touching):", np.round(net["total"][0, 2:], 3).tolist())

# dynamic: Isaac's init state (18 mm drop, Isaac gains), every control step for 1.5 s: touch (sum of per-contact
# |f|) vs |net total force vector| per foot, and the normal-only vs total split
sim2 = nb.NewtonSim(m, 1, iterations=a.it, dt=a.dt_ms * 1e-3, device="cpu", relaxation=0.4, solver_kw=skw)
sim2.d.qpos.assign(m.key_qpos[0].astype(np.float32)[None]); sim2.d.qvel.zero_(); sim2._reset_mask.fill_(True); sim2.launch_reset()
feet = wp.array([lab.index(t) for t in track[:2]], dtype=int, device="cpu")
rows = []
for step in range(int(1.5 / 0.02)):
    sim2.launch_step()
    o = {}
    for mode, tot in (("normal", 0), ("total", 1)):
        out = wp.zeros((1, 2, 3), dtype=float, device="cpu")
        wp.launch(gather_net_forces, dim=(1, 2), inputs=[sim2.solver.body_contact_force, sim2.nb, feet, tot], outputs=[out], device="cpu")
        o[mode] = out.numpy()[0]
    t = sim2.d.sensordata.numpy()[0, adr]
    for i in range(2):
        nt = np.linalg.norm(o["total"][i])
        if nt > 5.0:
            rows.append((t[i] / nt, np.linalg.norm(o["total"][i][:2]) / nt, np.linalg.norm(o["normal"][i]) / nt))
r = np.array(rows)
print(f"| landing + fall, feet in contact ({len(r)} foot-steps) | touch / |net vector| p50 {np.percentile(r[:, 0], 50):.3f} "
      f"p99 {np.percentile(r[:, 0], 99):.3f} max {r[:, 0].max():.3f} | tangential share of net p50 {np.percentile(r[:, 1], 50):.3f} "
      f"max {r[:, 1].max():.3f} | |normal-only| / |total| p50 {np.percentile(r[:, 2], 50):.3f} min {r[:, 2].min():.3f} |")

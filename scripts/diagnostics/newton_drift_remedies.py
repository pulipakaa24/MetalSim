"""Remedies for Newton XPBD's joint-constraint drift on the G1, next to the 4 it / 0.625 ms option:
  project : XPBD, then after every substep rebuild body poses/twists from the joint state (eval_ik + eval_fk)
  fs      : SolverFeatherstone (reduced coordinates) with ActuatorPD, armature native in the joint-space inertia
Per setting (CPU device unless given): drift under random actions (feet vs FK of the joint angles), a 1 m drop
under the PD hold (impact penetration, finite state), and a passive energy check (drives off, zero gravity,
floating, random initial joint velocities: kinetic energy after 2 s / initial; > 1 would mean energy injection).

usage: python scripts/diagnostics/newton_drift_remedies.py [device] [setting ...]
       setting = IT:DT_MS[:project][:rc] | fs:DT_MS
Revolute limits are kept inside +-(pi - 0.15) (NewtonSim default), rc = recentred joint zeros."""
import sys, numpy as np, warp as wp, newton
wp.config.quiet = True
from metalsim.learn.g1_velocity import build_g1_model
from metalsim.physics import newton_backend as nb
import metalsim.interop.warp_metal as wm

DEV = sys.argv[1] if len(sys.argv) > 1 else "cpu"
SETTINGS = sys.argv[2:] or ["4:1.25", "4:0.625", "4:1.25:project", "fs:2.5", "fs:1.25"]
if DEV == "cpu":
    class _E:
        def __init__(self, *a): self.v = 0
        def next_value(self): self.v += 1; return self.v
    wm.SharedEvent = _E
M_MJ = build_g1_model("flat", physics_dt=0.0025)[0]


def make(setting, n):
    p = setting.split(":")
    if p[0] == "fs":
        return nb.NewtonSim(M_MJ, n, dt=float(p[1]) * 1e-3, device=DEV, solver="featherstone")
    return nb.NewtonSim(M_MJ, n, iterations=int(p[0]), dt=float(p[1]) * 1e-3, device=DEV, project="project" in p, recenter="rc" in p)


def place(sim, n, z=None, qvel=None):
    q = np.tile(M_MJ.key_qpos[0], (n, 1)).astype(np.float32)
    side = int(np.ceil(np.sqrt(n))); q[:, 0] = (np.arange(n) % side) * 2.5; q[:, 1] = (np.arange(n) // side) * 2.5
    if z is not None: q[:, 2] = z
    sim.d.qpos.assign(q); sim.d.qvel.assign(np.zeros((n, M_MJ.nv), np.float32) if qvel is None else qvel.astype(np.float32))
    sim._reset_mask.fill_(True); sim.launch_reset()


def drift(setting, n=32, steps=100, sigma=1.0):
    sim = make(setting, n); M = sim.model; place(sim, n); rng = np.random.default_rng(0)
    fk = M.state(); feet = [int(sim.foot_nb[0]), int(sim.foot_nb[1])]; F = []; alive = np.ones(n, bool)
    for _ in range(steps):
        sim.d.ctrl.assign((M_MJ.key_qpos[0][7:] + 0.5 * sigma * rng.normal(size=(n, M_MJ.nu))).astype(np.float32))
        sim.launch_step()
        newton.eval_fk(M, sim.joint_q, sim.joint_qd, fk)
        bq = sim.s0.body_q.numpy().reshape(n, sim.nb, 7); bf = fk.body_q.numpy().reshape(n, sim.nb, 7)
        if not np.isfinite(bq).all(): return None
        alive &= bq[:, 0, 2] > 0.4
        if not alive.any(): break
        F.append(np.linalg.norm(bq[alive][:, feet, :3] - bf[alive][:, feet, :3], axis=-1).ravel())
    F = np.concatenate(F) * 1e3
    return np.percentile(F, 50), np.percentile(F, 99), F.max()


@wp.kernel
def _lowest(body_q: wp.array[wp.transform], shape_body: wp.array[int], shape_X: wp.array[wp.transform], shape_scale: wp.array[wp.vec3],
            shape_type: wp.array[int], out: wp.array[float]):
    s = wp.tid()
    if shape_type[s] != 7 or shape_body[s] < 0:
        out[s] = 1.0e6
        return
    X = body_q[shape_body[s]] * shape_X[s]; h = shape_scale[s]; z = float(1.0e6)
    for i in range(8):
        c = wp.vec3(wp.where(i % 2 == 0, -h[0], h[0]), wp.where((i // 2) % 2 == 0, -h[1], h[1]), wp.where(i // 4 == 0, -h[2], h[2]))
        z = wp.min(z, wp.transform_point(X, c)[2])
    out[s] = z


def drop(setting, n=8, T=1.5):
    sim = make(setting, n); M = sim.model; place(sim, n, z=1.0)
    zc = wp.zeros(M.shape_count, dtype=float, device=DEV); pen = 0.0
    for _ in range(int(T / 0.02)):
        sim.launch_step()
        wp.launch(_lowest, dim=M.shape_count, inputs=[sim.s0.body_q, M.shape_body, M.shape_transform, M.shape_scale, M.shape_type], outputs=[zc], device=DEV)
        z = zc.numpy()
        if not np.isfinite(z).all(): return None, False
        pen = max(pen, -float(z.min()))
    return pen * 100, bool(np.isfinite(sim.s0.body_q.numpy()).all())


def kinetic(sim):
    M = sim.model; bq = sim.s0.body_q.numpy(); bqd = sim.s0.body_qd.numpy(); m = M.body_mass.numpy(); I = M.body_inertia.numpy()
    e = 0.0
    for b in range(M.body_count):
        x, y, z, w = bq[b, 3:7]
        R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                      [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
        v, om = bqd[b, :3], bqd[b, 3:]; ob = R.T @ om
        e += 0.5 * m[b] * v @ v + 0.5 * ob @ I[b] @ ob
    return e


def energy(setting, n=2, T=2.0):
    sim = make(setting, n); M = sim.model
    M.gravity.zero_(); sim.actuator.gravity = wp.vec3(0.0, 0.0, 0.0)          # per-world gravity (n_worlds, 3)
    sim.actuator.kp.zero_(); sim.actuator.kd.zero_()
    qv = np.zeros((n, M_MJ.nv)); qv[:, 6:] = np.random.default_rng(1).normal(0, 2.0, (n, M_MJ.nu))
    place(sim, n, z=3.0, qvel=qv)
    e0 = kinetic(sim)
    for _ in range(int(T / 0.02)):
        sim.launch_step()
    return kinetic(sim) / e0


print(f"device {DEV}")
print("| setting | feet vs FK(q) p50 / p99 / max [mm] | 1 m drop: impact penetration [cm], finite | passive KE after 2 s / initial |")
print("|---|---|---|---|", flush=True)
for s in SETTINGS:
    try:
        d = drift(s); dp, ok = drop(s); ek = energy(s)
        ds = "non-finite" if d is None else f"{d[0]:.2f} / {d[1]:.1f} / {d[2]:.1f}"
        print(f"| {s} | {ds} | {'-' if dp is None else f'{dp:.2f}'}, {ok} | {ek:.3f} |", flush=True)
    except Exception as ex:
        print(f"| {s} | failed: {ex!r:.120} | | |", flush=True)

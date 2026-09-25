"""Joint-constraint residual of Newton XPBD on the G1 during a random-action rollout (actions N(0, sigma), Isaac's
action scale 0.5), per solver setting: anchor separation (linear residual) and off-axis rotation (swing) of every
revolute joint, and how far the feet are from where forward kinematics of the reported joint angles puts them.

Why it matters for observations: the task's joint observations come from eval_ik, i.e. each joint's relative body
orientation projected on its axis, and the base terms from the pelvis body; so drift enters observations only as
the residual those projections discard. Compare with the observation noise Isaac adds (joint pos +-0.01 rad).

usage: python scripts/diagnostics/newton_constraint_drift.py [device] [sigma]   (default cpu, 1.0)"""
import sys, numpy as np, warp as wp, newton
wp.config.quiet = True
from metalsim.learn.g1_velocity import build_g1_model
from metalsim.physics import newton_backend as nb
import metalsim.interop.warp_metal as wm

DEV = sys.argv[1] if len(sys.argv) > 1 else "cpu"; SIGMA = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
if DEV == "cpu":
    class _E:
        def __init__(self, *a): self.v = 0
        def next_value(self): self.v += 1; return self.v
    wm.SharedEvent = _E


@wp.kernel
def residual(body_q: wp.array[wp.transform], parent: wp.array[int], child: wp.array[int], X_p: wp.array[wp.transform],
             X_c: wp.array[wp.transform], jtype: wp.array[int], qd_start: wp.array[int], axis: wp.array[wp.vec3],
             lin: wp.array[float], ang: wp.array[float]):
    j = wp.tid()
    if jtype[j] != 1:                  # revolute only
        lin[j] = 0.0; ang[j] = 0.0
        return
    Xwp = X_p[j]
    if parent[j] >= 0:
        Xwp = body_q[parent[j]] * Xwp
    Xwc = body_q[child[j]] * X_c[j]
    lin[j] = wp.length(wp.transform_get_translation(Xwp) - wp.transform_get_translation(Xwc))
    qr = wp.quat_inverse(wp.transform_get_rotation(Xwp)) * wp.transform_get_rotation(Xwc)
    a = axis[qd_start[j]]
    v = wp.vec3(qr[0], qr[1], qr[2])
    p = wp.dot(v, a)
    tw = wp.normalize(wp.quat(a[0] * p, a[1] * p, a[2] * p, qr[3]))
    sw = qr * wp.quat_inverse(tw)
    ang[j] = 2.0 * wp.acos(wp.min(wp.abs(sw[3]), 1.0))


def run(it, dt, n=64, steps=100, seed=0):
    m = build_g1_model("flat", physics_dt=0.0025)[0]
    sim = nb.NewtonSim(m, n, iterations=it, dt=dt, device=DEV); M = sim.model
    rng = np.random.default_rng(seed)
    q = np.tile(m.key_qpos[0], (n, 1)).astype(np.float32)
    side = int(np.ceil(np.sqrt(n))); q[:, 0] = (np.arange(n) % side) * 2.5; q[:, 1] = (np.arange(n) // side) * 2.5
    sim.d.qpos.assign(q); sim._reset_mask.fill_(True); sim.launch_reset()
    lin = wp.zeros(M.joint_count, dtype=float, device=DEV); ang = wp.zeros(M.joint_count, dtype=float, device=DEV)
    fk = M.state(); feet = [int(sim.foot_nb[0]), int(sim.foot_nb[1])]
    L, A, F = [], [], []
    alive = np.ones(n, bool)
    for t in range(steps):
        sim.d.ctrl.assign((m.key_qpos[0][7:] + 0.5 * SIGMA * rng.normal(size=(n, m.nu))).astype(np.float32))
        sim.launch_step()
        wp.launch(residual, dim=M.joint_count, inputs=[sim.s0.body_q, M.joint_parent, M.joint_child, M.joint_X_p, M.joint_X_c,
                  M.joint_type, M.joint_qd_start, M.joint_axis], outputs=[lin, ang], device=DEV)
        newton.eval_fk(M, sim.joint_q, sim.joint_qd, fk)          # sim.joint_q: eval_ik of the current state
        bq = sim.s0.body_q.numpy().reshape(n, sim.nb, 7); bf = fk.body_q.numpy().reshape(n, sim.nb, 7)
        alive &= bq[:, 0, 2] > 0.4                                 # upright envs only (not lying on the ground)
        if not alive.any():
            break
        L.append(lin.numpy().reshape(n, -1)[alive]); A.append(ang.numpy().reshape(n, -1)[alive])
        F.append(np.linalg.norm(bq[alive][:, feet, :3] - bf[alive][:, feet, :3], axis=-1))
    L, A, F = np.concatenate(L), np.concatenate(A), np.concatenate(F)
    return L, A, F


print(f"G1, random actions N(0, {SIGMA}) x 0.5 rad, 64 envs, 2 s, upright envs only, device {DEV}")
print("| setting | anchor separation p50 / p99 / max [mm] | off-axis rotation p50 / p99 / max [mrad] | feet vs FK(q) p50 / p99 / max [mm] |")
print("|---|---|---|---|")
for it, dt in ((2, 0.00125), (4, 0.00125), (6, 0.00125), (8, 0.00125), (16, 0.00125), (8, 0.0025), (4, 0.000625)):
    L, A, F = run(it, dt)
    L = L[L > 0] * 1e3; A = A[A > 0] * 1e3; F = F * 1e3
    q = lambda x: f"{np.percentile(x, 50):.2f} / {np.percentile(x, 99):.2f} / {x.max():.1f}"
    print(f"| {it} it, {dt*1e3:.3g} ms | {q(L)} | {q(A)} | {q(F)} |", flush=True)

"""Isaac's actuator model under Newton XPBD (metalsim.physics.newton_backend.ActuatorPD): do the effort
limits (300 Nm legs/arms, 20 Nm ankles) and armature (0.01 / 0.001 kg m^2) matter on the G1, and what do
they change? Floating PD hold from Isaac's init state (2 s) and a 1 m drop (1 s), 4 envs, XPBD 4 it at
1.25 ms and 8 it at 2.5 ms, joint relaxation 0.4/0.4. Reports peak |tau| per group, % of DOF-steps at the
limit, and the pelvis trajectory for: full model / no effort limit / no armature."""
import numpy as np, warp as wp
wp.config.quiet = True
from metalsim.physics.newton_backend import G1XPBD

GROUPS = {"legs": ("hip", "knee", "torso"), "ankles": ("ankle",), "arms": ("shoulder", "elbow"),
          "hands": ("zero", "one", "two", "three", "four", "five", "six")}


def run(z0, T, it, dt, effort=True, armature=True):
    sim = G1XPBD(4, iterations=it, dt=dt, drive="ipd", z0=z0, armature_inertia="iso" if armature else False)
    if not effort:
        sim.actuator.effort.fill_(1e6)
    m = sim.model; nj = m.joint_count // 4; qds = m.joint_qd_start.numpy()
    names = {int(qds[j]): m.joint_label[j].split("/")[-1] for j in range(nj) if qds[j + 1] - qds[j] == 1}
    grp = {g: [d for d, n in names.items() if any(k in n for k in keys)] for g, keys in GROUPS.items()}
    lim = np.array([300.0 if g != "ankles" else 20.0 for g in GROUPS])
    nd = m.joint_dof_count // 4
    peak = {g: 0.0 for g in GROUPS}; sat = {g: 0 for g in GROUPS}; zs = []; n = 0
    for k in range(int(T / dt)):
        sim.step(1); n += 1
        tau = sim.control.joint_f.numpy().reshape(4, nd)
        for gi, (g, ds) in enumerate(grp.items()):
            a = np.abs(tau[:, ds]); peak[g] = max(peak[g], float(a.max())); sat[g] += int((a >= lim[gi] * 0.999).sum())
        if (k + 1) % int(0.25 / dt) == 0:
            zs.append(float(sim.pelvis_z().mean()))
    tot = {g: n * 4 * len(ds) for g, ds in grp.items()}
    return peak, {g: 100.0 * sat[g] / tot[g] for g in GROUPS}, zs


for label, z0, T in (("PD hold from Isaac's init state", 0.74, 2.0), ("1 m drop (root z 1.74)", 1.74, 1.0)):
    print(label)
    for it, dt in ((4, 0.00125), (8, 0.0025)):
        for eff, arm in ((True, True), (False, True), (True, False)):
            try:
                peak, sat, zs = run(z0, T, it, dt, eff, arm)
                tag = f"it {it} dt {dt*1e3:.2f} ms, {'effort limit' if eff else 'NO effort limit'}, {'armature' if arm else 'NO armature'}"
                print(f"  {tag:52s}: peak |tau| " + ", ".join(f"{g} {v:.0f}" for g, v in peak.items())
                      + " Nm | at limit " + ", ".join(f"{g} {v:.1f}%" for g, v in sat.items()) + f" | pelvis z {np.round(zs, 3).tolist()}", flush=True)
            except Exception as e:
                print(f"  it {it} eff {eff} arm {arm}: failed {e!r}")

"""G1 drop test under Newton XPBD on Metal, comparable to scripts/diagnostics/g1_contact_stiffness.py
(MuJoCo C): root released at z = 1.0 m from Isaac's default pose under the PD hold (ActuatorPD, Isaac
gains), 3 s at the given step. Penetration = depth of the lowest collider-box corner below the ground
plane (the G1 has 3 box colliders: 2 feet, torso), max over the run (impact) and mean over t > 2 s
(rest), max over 16 envs. MuJoCo C reference (same protocol, PARITY 2.1): default solref 0.02 ->
2.97 cm impact / 0.06 cm rest; tau 5 ms + impedance 0.99 -> 0.85 / 0.00 cm."""
import numpy as np, warp as wp
wp.config.quiet = True
from metalsim.physics.newton_backend import G1XPBD

N = 16


@wp.kernel
def lowest_corner(body_q: wp.array[wp.transform], shape_body: wp.array[int], shape_X: wp.array[wp.transform],
                  shape_scale: wp.array[wp.vec3], shape_type: wp.array[int], out: wp.array[float]):
    s = wp.tid()
    if shape_type[s] != 7 or shape_body[s] < 0:        # boxes on bodies only (ground plane excluded)
        out[s] = 1.0e6
        return
    X = body_q[shape_body[s]] * shape_X[s]
    h = shape_scale[s]
    zmin = float(1.0e6)
    for i in range(8):
        c = wp.vec3(wp.where(i % 2 == 0, -h[0], h[0]), wp.where((i // 2) % 2 == 0, -h[1], h[1]), wp.where(i // 4 == 0, -h[2], h[2]))
        zmin = wp.min(zmin, wp.transform_point(X, c)[2])
    out[s] = zmin


def drop(it, dt, T=3.0, **kw):
    sim = G1XPBD(N, iterations=it, dt=dt, drive="ipd", z0=1.0, **kw); m = sim.model
    zc = wp.zeros(m.shape_count, dtype=float, device=m.device)
    imp, rest, zs = 0.0, [], []
    for k in range(int(T / dt)):
        sim.step(1)
        wp.launch(lowest_corner, dim=m.shape_count, inputs=[sim.s0.body_q, m.shape_body, m.shape_transform, m.shape_scale, m.shape_type], outputs=[zc], device=m.device)
        pen = max(0.0, -float(zc.numpy().min()))
        imp = max(imp, pen)
        if k * dt > 2.0: rest.append(pen)
        if (k + 1) % int(0.5 / dt) == 0: zs.append(round(float(sim.pelvis_z().mean()), 3))
    bad = int((~np.isfinite(sim.s0.body_q.numpy())).any())
    return imp, float(np.mean(rest)), zs, bad


if __name__ == "__main__":
    print(f"G1 drop from root z 1.0 m, {N} envs, Newton XPBD (joint relaxation 0.4/0.4, ActuatorPD), Metal")
    for it, dt in ((2, 0.0025), (4, 0.0025), (8, 0.0025), (4, 0.00125), (8, 0.00125)):
        imp, rest, zs, bad = drop(it, dt)
        print(f"  it {it} dt {dt*1e3:.2f} ms: impact penetration {imp*100:.2f} cm, resting penetration {rest*100:.2f} cm, "
              f"pelvis z every 0.5 s {zs}{'  NON-FINITE' if bad else ''}", flush=True)

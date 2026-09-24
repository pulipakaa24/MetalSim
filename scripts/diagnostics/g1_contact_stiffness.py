"""How hard can MuJoCo's contacts be made? G1 drop test (1.0 m) and standing rest in MuJoCo C at the
2.5 ms step: penetration and peak foot force vs contact solref/solimp, margin (speculative contact),
and noslip. Isaac/PhysX reference: penetration near zero (rigid unilateral constraints)."""
import sys, numpy as np, mujoco
from metalsim.learn.g1_velocity import build_g1_model
dt = 0.0025
def run(solref, solimp, margin=0.0, noslip=0, label=""):
    m, _ = build_g1_model("flat", physics_dt=dt)
    m.geom_solref[:] = solref; m.geom_solimp[:] = solimp; m.geom_margin[:] = margin; m.opt.noslip_iterations = noslip
    d = mujoco.MjData(m); mujoco.mj_resetDataKeyframe(m, d, 0); d.qpos[2] = 1.0; d.ctrl[:] = m.key_qpos[0][7:]
    pen_max = 0.0; f_peak = 0.0; pen_rest = []
    for t in range(int(3.0 / dt)):
        mujoco.mj_step(m, d)
        if d.ncon:
            pen = -min(c.dist for c in d.contact[:d.ncon]); pen_max = max(pen_max, pen)
            f_peak = max(f_peak, d.sensordata[:2].max())
            if t > int(2.0 / dt): pen_rest.append(pen)
    print(f"{label:40s} solref {solref} solimp {solimp[:3]} margin {margin} noslip {noslip}: impact penetration {pen_max*100:.2f} cm, "
          f"resting penetration {np.mean(pen_rest)*100 if pen_rest else 0:.2f} cm, peak foot force {f_peak:.0f} N ({f_peak/(32.24*9.81):.1f}x weight), pelvis z {d.qpos[2]:.3f}", flush=True)
run([0.02, 1.0], [0.9, 0.95, 0.001, 0.5, 2.0], label="MuJoCo default (ours today)")
run([0.01, 1.0], [0.9, 0.95, 0.001, 0.5, 2.0], label="tau 10 ms")
run([0.005, 1.0], [0.9, 0.95, 0.001, 0.5, 2.0], label="tau 5 ms (= 2 dt floor)")
run([0.005, 1.0], [0.99, 0.999, 0.001, 0.5, 2.0], label="tau 5 ms, impedance 0.99-0.999")
run([0.005, 1.0], [0.99, 0.999, 0.001, 0.5, 2.0], margin=0.01, label="+ margin 1 cm (speculative contact)")
run([0.005, 1.0], [0.99, 0.999, 0.001, 0.5, 2.0], margin=0.01, noslip=5, label="+ noslip 5 (hard friction)")
run([-2e5, -2e3], [0.99, 0.999, 0.001, 0.5, 2.0], label="explicit stiffness 2e5 N/m, damping 2e3")

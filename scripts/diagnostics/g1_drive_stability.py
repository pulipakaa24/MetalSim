"""Does the actuator-driven blow-up depend on the physics timestep / integrator? MuJoCo C, Isaac's
gains (kp 200, effort 300), random position targets of amplitude `amp` (3 sigma of the policy's scale)."""
import sys, numpy as np, mujoco
from metalsim.learn.g1_velocity import build_g1_model
amp = float(sys.argv[1]); dt = float(sys.argv[2]); integ = sys.argv[3]
m, _ = build_g1_model("flat"); m.opt.timestep = dt
m.opt.integrator = {"implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST, "implicit": mujoco.mjtIntegrator.mjINT_IMPLICIT, "euler": mujoco.mjtIntegrator.mjINT_EULER}[integ]
sub = int(round(0.02 / dt))
rng = np.random.default_rng(0); worst = 0.0; blow = 0
for e in range(8):
    d = mujoco.MjData(m); mujoco.mj_resetDataKeyframe(m, d, 0)
    for t in range(400):
        d.ctrl[:] = m.key_qpos[0][7:] + 0.5 * rng.normal(size=m.nu) * amp
        for _ in range(sub): mujoco.mj_step(m, d)
        v = np.abs(d.qvel).max(); worst = max(worst, v)
        if v > 1e3: blow += 1; break
print(f"C flat amp {amp} dt {dt} {integ}: worlds blown {blow}/8, |qvel| max {worst:.1f}", flush=True)

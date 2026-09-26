"""Shared benchmark settings: same MJCF, controller, timestep, env count and solver budget in both engines."""
import os
# a sparse checkout of google-deepmind/mujoco_menagerie with unitree_go2 and unitree_g1 (see README.md here)
MEN = os.environ.get("MENAGERIE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "menagerie"))
ROBOTS = {
    "go2": dict(robot=f"{MEN}/unitree_go2/go2.xml", scene=f"{MEN}/unitree_go2/scene.xml", kp=20.0, kd=0.5),
    "g1":  dict(robot=f"{MEN}/unitree_g1/g1.xml",   scene=f"{MEN}/unitree_g1/scene.xml",  kp=100.0, kd=2.0),
}
DT = 0.004
WARMUP, STEPS, RESAMPLE, AMP = 100, 1000, 20, 0.3

def force_limits(m):
    """Per-actuator force limits: the actuator's ctrlrange (Go2 motors) or its joint's actuatorfrcrange (G1)."""
    import numpy as np
    lim = m.actuator_ctrlrange.copy()
    for i in range(m.nu):
        if not m.actuator_ctrllimited[i] or lim[i, 0] == lim[i, 1]:
            lim[i] = m.jnt_actfrcrange[m.actuator_trnid[i, 0]]
    assert (lim[:, 1] > lim[:, 0]).all(), lim
    return lim

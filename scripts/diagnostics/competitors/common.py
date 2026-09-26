"""Shared benchmark settings: same MJCF, controller, timestep, env count and solver budget in both engines."""
import os
# a sparse checkout of google-deepmind/mujoco_menagerie with unitree_go2 and unitree_g1 (see README.md here)
MEN = os.environ.get("MENAGERIE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "menagerie"))
ROBOTS = {
    "go2": dict(robot=f"{MEN}/unitree_go2/go2.xml", scene=f"{MEN}/unitree_go2/scene.xml", kp=20.0, kd=0.5),
    "g1":  dict(robot=f"{MEN}/unitree_g1/g1.xml",   scene=f"{MEN}/unitree_g1/scene.xml",  kp=100.0, kd=2.0),
    # audit models (2026-09-25): SO-101 arm + box (Menagerie's own position servos, elliptic cones, impratio 10);
    # DeepMind humanoid from the MuJoCo Warp test data (motors -> PD, gear 20-120 so kp is per unit gear)
    "so101": dict(robot=f"{MEN}/robotstudio_so101/so101.xml", scene=f"{MEN}/robotstudio_so101/scene_box.xml", kp=None, kd=None),
    "humanoid": dict(robot=os.environ.get("HUMANOID_XML", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "upstream/mujoco_warp-ellip2/mujoco_warp/test_data/humanoid/humanoid.xml")),
                     scene=None, kp=5.0, kd=0.1),
}
for _r in ROBOTS.values():
    if _r["scene"] is None:
        _r["scene"] = _r["robot"]


def _g1task_model():
    """MetalSim's G1 velocity-task model (Isaac's asset, its PD actuators, 2.5 ms) with a contact_tuning preset
    (G1_PRESET, default the task's "recommended"); stepped with 8 substeps per call like the task."""
    from metalsim.learn.g1_velocity import build_g1_model
    from metalsim.physics import contact_tuning
    m, _ = build_g1_model("flat", physics_dt=0.0025)
    contact_tuning.apply(m, os.environ.get("G1_PRESET", "recommended"))
    return m


ROBOTS["g1task"] = dict(robot=None, scene=None, kp=None, kd=None, builder=_g1task_model, substeps=8)
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

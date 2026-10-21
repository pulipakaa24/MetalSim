"""Rigid-body bitwise neutrality check of a Warp tree: MuJoCo Warp (metalsim BatchSim) on Metal, 200 steps with
seeded random controls, for So-101, Panda, Go1 and G1; prints a hash of every world's qpos / qvel per robot.
Run once per Warp tree (PYTHONPATH) and compare the hashes.

    python rigid_neutral.py
"""
import hashlib
import os

import mujoco
import numpy as np
import warp as wp

from metalsim.physics.batch import BatchSim, BatchSimOptions

wp.config.quiet = True
ROOT = "."
ROBOTS = [
    ("so101_lift", os.path.join(ROOT, "assets", "so101", "scene_box_rl.xml")),
    ("panda", os.path.join(ROOT, "upstream", "mujoco_menagerie", "franka_emika_panda", "scene.xml")),
    ("go1", os.path.join(ROOT, "upstream", "mujoco_menagerie", "unitree_go1", "scene.xml")),
    ("g1", os.path.join(ROOT, "upstream", "mujoco_menagerie", "unitree_g1", "scene.xml")),
]
for name, path in ROBOTS:
    if not os.path.exists(path):
        print(name, "missing")
        continue
    model = mujoco.MjModel.from_xml_path(path)
    n = 16
    sim = BatchSim(model, n, options=BatchSimOptions(capture=False))
    rng = np.random.default_rng(0)
    h = hashlib.sha256()
    for step in range(200):
        if model.nu:
            c = rng.uniform(-1, 1, (n, model.nu)).astype(np.float32)
            lo, hi = model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
            lim = model.actuator_ctrllimited.astype(bool)
            c = np.where(lim, lo + 0.5 * (c + 1) * (hi - lo), c).astype(np.float32)
            sim.d.ctrl.assign(c)
        sim.step()
        if step % 20 == 19:
            h.update(sim.d.qpos.numpy().tobytes())
            h.update(sim.d.qvel.numpy().tobytes())
    print(name, h.hexdigest()[:16], "finite", bool(np.isfinite(sim.d.qpos.numpy()).all()))

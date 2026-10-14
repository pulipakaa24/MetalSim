"""CPU reproduction of test_go2_boxes_gpu's env 0 / 1 state (reset sampling with the scene seed, 20 control steps of
MuJoCo C with the hold controls) and the first hit of the bottom-left pixel's ray (mj_ray), vs the floor distance."""
import sys
from dataclasses import replace

import mujoco
import numpy as np

sys.path.insert(0, ".")
from metalsim.scene import cfg as C, compose as CP

cfg = replace(C.load("configs/scenes/go2_boxes.yaml"), num_envs=2)
sc = CP.ComposedScene(cfg, device=None)
m = sc.model
qpos, qvel = sc.sample_states([0, 1])
cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "head_cam")
f = (360 / 2) / np.tan(np.radians(CP.fovy_deg(cfg.cameras["head_cam"])) / 2)
for e in range(2):
    d = mujoco.MjData(m)
    d.qpos[:] = qpos[e]; d.qvel[:] = qvel[e]; d.ctrl[:] = sc.ctrl_default
    for _ in range(20 * cfg.physics.substeps):
        mujoco.mj_step(m, d)
    mujoco.mj_forward(m, d)
    p, R = d.cam_xpos[cid].copy(), d.cam_xmat[cid].reshape(3, 3)
    v = R @ np.array([(0.5 - 320) / f, -(359.5 - 180) / f, -1.0])      # unit planar depth
    t_floor = -p[2] / v[2]
    gid = np.zeros(1, np.int32)
    n = v / np.linalg.norm(v)
    dist = mujoco.mj_ray(m, d, p, n, None, 1, -1, gid)
    planar = dist / np.linalg.norm(v)
    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(gid[0])) if gid[0] >= 0 else None
    print(f"env {e}: cam {p.round(3)}, floor planar {t_floor:.4f}, first hit planar {planar:.4f} on geom {name}")
    for b in range(m.nbody):
        bn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        if bn.startswith("box_"):
            print("   ", bn, d.xpos[b].round(3))

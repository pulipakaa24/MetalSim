"""so101-tilt: geometry of so101Sim's 'pickup' keyframe (the TCP calibration source of the scripted expert) and of
the TCP in the gripper body frame. Read-only on so101Sim. Main venv (MuJoCo 3.14).
  .venv/bin/python runs/readme_media/scratch/so101-tilt/keyframe.py <so101Sim>/mujoco_menagerie/robotstudio_so101/scene_box_rl.xml"""
import sys
import numpy as np
import mujoco
sys.path.insert(0, ".")
import metalsim.render.film.so101_expert as X

m = mujoco.MjModel.from_xml_path(sys.argv[1])
d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "pickup"))
mujoco.mj_forward(m, d)
site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
gb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "gripper")
bb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "box")
Rs, Rg, Rb = (d.site_xmat[site].reshape(3, 3), d.xmat[gb].reshape(3, 3), d.xmat[bb].reshape(3, 3))
tcp = d.site_xpos[site] + Rs @ X.TCP_LOCAL
print("box centre (world)", d.xpos[bb].round(4), " box z axis (world)", Rb[:, 2].round(3),
      " -> box long (6 cm) axis tilt from vertical", round(float(np.degrees(np.arccos(abs(Rb[2, 2])))), 1), "deg")
print("box long axis in the gripper frame", (Rg.T @ Rb[:, 2]).round(3), "(gripper x = fixed-pad normal, y = jaw hinge axis)")
print("gripper joint at the keyframe", round(float(d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, 'gripper')]]), 4), "rad")
print("TCP in the gripper body frame", (Rg.T @ (tcp - d.xpos[gb])).round(4))
print("jaw axis (JAW_LOCAL) in the gripper body frame", (Rg.T @ Rs @ X.JAW_LOCAL).round(3))
print("box contacts at the keyframe:")
f6 = np.zeros(6)
for i in range(d.ncon):
    c = d.contact[i]
    if bb not in (m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]):
        continue
    o = c.geom2 if m.geom_bodyid[c.geom1] == bb else c.geom1
    mujoco.mj_contactForce(m, d, i, f6)
    print("  ", mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, o) or f"#{o}", "pos in box frame mm",
          (Rb.T @ (c.pos - d.xpos[bb]) * 1e3).round(1), "fn", round(f6[0], 2))
for g in ("fixed_jaw_box4", "fixed_jaw_box5", "fixed_jaw_box6", "fixed_jaw_box7", "fixed_jaw_sph_tip1"):
    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, g)
    print(f"  {g} centre in gripper frame mm", (m.geom_pos[gid] * 1e3).round(1), "half-size mm", (m.geom_size[gid] * 1e3).round(2))

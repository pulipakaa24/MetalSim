"""In-hand creep of the SO-101 lift-task box under pyramidal vs elliptic friction cones (MuJoCo's own recommendation
for grasping is elliptic cones + large impratio + Newton). Protocol (MetalSim's, not from a paper):
  1. home pose, gripper opened (1.0 rad) for 0.5 s; the box (4 x 4 x 6 cm, 4 cm face between the jaws) is placed
     between the jaw tips, aligned with the gripper frame, and the jaws close (ctrl -0.17 rad, force-limited servo);
  2. 1.0 s settle; then 3 s of (a) static hold, (b) wrist_roll sinusoid +-0.6 rad at 1 Hz, (c) shoulder_lift sinusoid
     +-0.25 rad at 1 Hz (inertial load), each from the settled state;
  3. creep = displacement of the box centre in the gripper-site frame between the settled state and the end (mm),
     and whether the box was dropped (box more than 6 cm from the site).
Settings: pyramidal, elliptic impratio 1 / 10 (the scene's own) / 100. Engines: MuJoCo C (--engine c, 1 world) and
MuJoCo Warp on Metal (--engine warp, N identical worlds; run through the GPU queue).
usage: python so101_creep.py --engine c|warp [--n 64] [--settings pyr,ell1,ell10,ell100]"""
import argparse, os, sys, json, numpy as np, mujoco
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SCENE = os.path.join(ROOT, "assets", "so101", "scene_box_rl.xml")
# name: (cone, impratio, newton iterations, contact solimp d0 on every geom or None)
SETTINGS = {"pyr": (0, 1.0, 10, None), "ell1": (1, 1.0, 10, None), "ell10": (1, 10.0, 10, None), "ell100": (1, 100.0, 10, None),
            "pyr_it50": (0, 1.0, 50, None), "ell100_it50": (1, 100.0, 50, None),
            "pyr_imp99": (0, 1.0, 10, 0.99), "ell100_imp99": (1, 100.0, 10, 0.99), "ell10_imp99": (1, 10.0, 10, 0.99)}
ap = argparse.ArgumentParser(); ap.add_argument("--engine", default="c"); ap.add_argument("--n", type=int, default=64)
ap.add_argument("--settings", default="pyr,ell1,ell10,ell100"); ap.add_argument("--json", default=None); a = ap.parse_args()


def build(cone, impratio, iterations=10, solimp0=None):
    m = mujoco.MjModel.from_xml_path(SCENE); m.opt.cone = cone; m.opt.impratio = impratio; m.opt.iterations = iterations
    if solimp0 is not None:  # stiffer contact impedance (MuJoCo's other slip remedy): d0 -> solimp0, dwidth 0.999, width 1 mm
        m.geom_solimp[:, 0] = solimp0; m.geom_solimp[:, 1] = 0.999; m.geom_solimp[:, 2] = 0.001
    return m


class Engine:
    def __init__(self, m, n):
        self.m, self.n = m, n
        self.sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
        self.box = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "box")
        self.bq = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "box")]
        self.tips = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, g) for g in ("fixed_jaw_sph_tip1", "moving_jaw_sph_tip1")]
        self.ctrl0 = np.zeros(m.nu); self.ctrl0[:] = m.qpos0[[m.jnt_qposadr[m.actuator_trnid[i, 0]] for i in range(m.nu)]]
        self.names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]
        if a.engine == "c":
            self.d = mujoco.MjData(m)
        else:
            from metalsim.physics.batch import BatchSim, BatchSimOptions
            import torch
            self.torch = torch
            self.sim = BatchSim(m, n, options=BatchSimOptions(njmax=512)); self.sim.set_state(m.qpos0); self.sim.forward(); self.sim.synchronize()

    def set_qpos(self, q):
        if a.engine == "c":
            self.d.qpos[:] = q; self.d.qvel[:] = 0; mujoco.mj_forward(self.m, self.d)
        else:
            self.sim.set_state(q); self.sim.t.qvel.zero_(); self.sim.forward(); self.sim.synchronize()

    def step(self, ctrl):
        if a.engine == "c":
            self.d.ctrl[:] = ctrl; mujoco.mj_step(self.m, self.d)
        else:
            # ctrl written through Warp after the previous replay finished (a torch/MPS copy into the shared buffer runs on
            # another command queue and can land while the previous step's graph is still executing)
            self.sim.synchronize(); self.sim.d.ctrl.assign(np.tile(ctrl, (self.n, 1)).astype(np.float32)); self.sim.step()

    def state(self):
        """box centre in the gripper-site frame (n, 3), box world pos (n, 3), tip positions (n, 2, 3)"""
        if a.engine == "c":
            xp, sp, sm, gp = self.d.xpos[None], self.d.site_xpos[None], self.d.site_xmat[None], self.d.geom_xpos[None]
        else:
            self.sim.synchronize(); xp = self.sim.d.xpos.numpy(); sp = self.sim.d.site_xpos.numpy(); sm = self.sim.d.site_xmat.numpy(); gp = self.sim.d.geom_xpos.numpy()
        R = sm[:, self.sid].reshape(-1, 3, 3); rel = np.einsum("nji,nj->ni", R, xp[:, self.box] - sp[:, self.sid])
        return rel, xp[:, self.box], gp[:, self.tips]


def run(name):
    cone, imp, its, si = SETTINGS[name]; m = build(cone, imp, its, si); e = Engine(m, a.n); dt = m.opt.timestep; gi = e.names.index("gripper")
    wr, sl = e.names.index("wrist_roll"), e.names.index("shoulder_lift")
    # 1. open the gripper at the home pose
    q = m.qpos0.copy(); q[e.bq:e.bq + 3] = [0.6, 0.0, 0.03]; e.set_qpos(q)
    c = e.ctrl0.copy(); c[gi] = 1.0
    for _ in range(int(0.5 / dt)): e.step(c)
    rel, _, tips = e.state(); mid = tips[0].mean(0)
    # box centre at the midpoint of the jaw tips, aligned with the site frame (site z along the jaws' gap in this pose)
    if a.engine == "c":
        R = e.d.site_xmat[e.sid].reshape(3, 3); qb = e.d.qpos.copy()
    else:
        e.sim.synchronize(); R = e.sim.d.site_xmat.numpy()[0, e.sid].reshape(3, 3); qb = e.sim.d.qpos.numpy()[0].copy()
    quat = np.zeros(4); mujoco.mju_mat2Quat(quat, R.reshape(-1))
    qb[e.bq:e.bq + 3] = mid + R @ np.array([0.0, 0.0, 0.0]); qb[e.bq + 3:e.bq + 7] = quat
    e.set_qpos(qb)
    for _ in range(int(0.1 / dt)): e.step(c)
    # 2. close and settle
    c[gi] = -0.17
    for _ in range(int(1.5 / dt)): e.step(c)
    rel0, box0, _ = e.state()
    if a.engine == "c": q_settled = e.d.qpos.copy()
    else: e.sim.synchronize(); q_settled = e.sim.d.qpos.numpy()[0].copy()
    out = {"settle_rel_mm": (1e3 * rel0[0]).round(2).tolist(), "settle_box_z": float(box0[0, 2])}
    for phase in ("hold", "wrist_roll", "shoulder_lift"):
        e.set_qpos(q_settled)
        creep_t = []
        for k in range(int(3.0 / dt)):
            cc = c.copy(); t = k * dt
            if phase == "wrist_roll": cc[wr] = e.ctrl0[wr] + 0.6 * np.sin(2 * np.pi * t)
            if phase == "shoulder_lift": cc[sl] = e.ctrl0[sl] + 0.25 * np.sin(2 * np.pi * t)
            e.step(cc)
            if (k + 1) % int(0.5 / dt) == 0:
                rel, box, _ = e.state(); creep_t.append(float(np.linalg.norm(rel - rel0, axis=1).mean() * 1e3))
        rel, box, _ = e.state(); d = np.linalg.norm(rel - rel0, axis=1) * 1e3
        out[phase] = {"creep_mm_mean": float(d.mean()), "creep_mm_max": float(d.max()), "dropped": int((np.linalg.norm(rel, axis=1) > 0.06).sum()),
                      "creep_mm_vs_t": [round(x, 2) for x in creep_t]}
    print(f"{name:12s} cone={cone} impratio={imp:5.1f} it={its} solimp0={si}: settle rel {out['settle_rel_mm']} mm, box z {out['settle_box_z']:.3f}; " +
          "; ".join(f"{p}: creep {out[p]['creep_mm_mean']:.2f} mm (max {out[p]['creep_mm_max']:.2f}, dropped {out[p]['dropped']}/{a.n if a.engine=='warp' else 1}) t={out[p]['creep_mm_vs_t']}" for p in ("hold", "wrist_roll", "shoulder_lift")), flush=True)
    return out


res = {s: run(s) for s in a.settings.split(",")}
if a.json: json.dump(res, open(a.json, "w"), indent=1)

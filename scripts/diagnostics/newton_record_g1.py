"""Newton side of the Isaac fidelity protocol: metalsim/parity/record_g1.py's physics recording (A_hold 150 steps, B_random
with Isaac's recorded action sequence by joint name 250 steps, C_drop from root z 1.0 m 150 steps; joint state, root pose and
body-frame velocities, joint torques, contact forces, per control step, Isaac's joint order), run on the Newton engine
(NewtonSim, Isaac's actuator via ActuatorPD) instead of MuJoCo Warp. No rendering. Contact "force" per body is the XPBD
contact force magnitude on the foot / torso colliders (MuJoCo Warp: the touch sensor's normal force), written as z.
Compare with: python -m metalsim.parity.compare --isaac runs/parity/isaac/parity_out2/rt --metalsim OUT --out REPORT

usage: python scripts/diagnostics/newton_record_g1.py --isaac DIR --out DIR --iterations 4 --dt_ms 1.25 [--device cpu]"""
import argparse, json, os
import numpy as np, mujoco, warp as wp
from metalsim.learn.g1_velocity import build_g1_model, ACTION_SCALE
from metalsim.physics import newton_backend as nb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--isaac", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--iterations", type=int, default=4); ap.add_argument("--dt_ms", type=float, default=1.25)
    ap.add_argument("--num_envs", type=int, default=4); ap.add_argument("--device", default="cpu")
    a = ap.parse_args(); wp.config.quiet = True
    if a.device == "cpu":
        import metalsim.interop.warp_metal as wm
        class _E:
            def __init__(self, *x): self.v = 0
            def next_value(self): self.v += 1; return self.v
        wm.SharedEvent = _E
    os.makedirs(a.out, exist_ok=True)
    meta = json.load(open(os.path.join(a.isaac, "meta.json")))
    seqB = np.load(os.path.join(a.isaac, "action_sequence_B.npz"))
    isaac_joints = [str(s) for s in seqB["joint_names"]]
    m, _ = build_g1_model("flat", physics_dt=0.0025)
    our_joints = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(1, m.njnt)]
    to_isaac = np.array([our_joints.index(n) for n in isaac_joints])
    act_of_joint = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, m.actuator_trnid[i][0]): i for i in range(m.nu)}
    act_isaac = np.array([act_of_joint[n] for n in isaac_joints])
    n = a.num_envs; dt = a.dt_ms * 1e-3
    sim = nb.NewtonSim(m, n, iterations=a.iterations, dt=dt, control_dt=meta["control_dt"], device=a.device)
    default = m.key_qpos[0].copy()
    contact_bodies = meta["contact_bodies"]; touch = {"left_ankle_roll_link": 0, "right_ankle_roll_link": 1, "torso_link": 2}
    tadr = [int(sim.touch_adr[i]) for i in range(3)]
    protocols = {"A_hold": (150, lambda t: np.zeros((n, m.nu), np.float32), None),
                 "B_random": (250, lambda t: np.repeat(seqB["actions"][t][None], n, 0), None),
                 "C_drop": (150, lambda t: np.zeros((n, m.nu), np.float32), 1.0)}
    import subprocess, sys
    stamp = json.loads(subprocess.run([sys.executable, "scripts/diagnostics/newton_stamp.py", "--json"], capture_output=True, text=True).stdout)
    json.dump({"our_joints": our_joints, "isaac_joints": isaac_joints, "engine": "newton", "iterations": a.iterations,
               "physics_dt": dt, "decimation": sim.substeps, "device": a.device, "install": stamp}, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    for tag, (T, act_fn, drop_z) in protocols.items():
        q = np.tile(default, (n, 1)).astype(np.float32)
        if drop_z is not None: q[:, 2] = drop_z
        sim.d.qpos.assign(q); sim.d.qvel.zero_(); sim.actuator.qd_prev.zero_(); sim._reset_mask.fill_(True); sim.launch_reset()
        sim.synchronize()
        rec = {k: [] for k in ("joint_pos", "joint_vel", "root_pos", "root_quat", "root_lin_vel_b", "root_ang_vel_b", "torque", "contact")}
        for t in range(T):
            ctrl = np.tile(default[7:], (n, 1)).astype(np.float32)
            ctrl[:, act_isaac] = default[7:][act_isaac] + ACTION_SCALE * act_fn(t)
            sim.d.ctrl.assign(ctrl)
            with wp.ScopedDevice(a.device):
                sim.launch_step()
            sim.synchronize()
            qp = sim.d.qpos.numpy(); qv = sim.d.qvel.numpy(); tau = sim.d.qfrc_actuator.numpy(); sd = sim.d.sensordata.numpy()
            Rm = np.zeros((n, 9)); [mujoco.mju_quat2Mat(Rm[e], qp[e, 3:7].astype(np.float64)) for e in range(n)]
            R = Rm.reshape(n, 3, 3)
            rec["joint_pos"].append(qp[:, 7:][:, to_isaac]); rec["joint_vel"].append(qv[:, 6:][:, to_isaac])
            rec["root_pos"].append(qp[:, :3].copy()); rec["root_quat"].append(qp[:, 3:7].copy())
            rec["root_lin_vel_b"].append(np.einsum("eji,ej->ei", R, qv[:, :3])); rec["root_ang_vel_b"].append(qv[:, 3:6].copy())
            rec["torque"].append(tau[:, 6:][:, to_isaac])
            cf = np.zeros((n, len(contact_bodies), 3), np.float32)
            for k, nm in enumerate(contact_bodies):
                if nm in touch: cf[:, k, 2] = sd[:, tadr[touch[nm]]]
            rec["contact"].append(cf)
        np.savez_compressed(os.path.join(a.out, f"{tag}.npz"), **{k: np.stack(v) for k, v in rec.items()})
        print(f"[newton {a.iterations} it {a.dt_ms} ms] {tag}: {T} steps, final root z {rec['root_pos'][-1][:, 2]}", flush=True)


if __name__ == "__main__":
    main()

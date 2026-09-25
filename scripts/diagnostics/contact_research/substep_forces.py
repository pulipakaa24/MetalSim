"""Per-substep contact forces for the fidelity protocols (A_hold, B_random, C_drop), to compare MuJoCo's
contact-force peaks with Isaac's on the same reporting window.

Isaac Lab 2.3.2 records `ContactSensor.data.net_forces_w` after `env.step()`: `get_net_contact_forces(dt=physics_dt)`
from the LAST PhysX step (5 ms) of the 4-step decimation (the sensor refreshes every physics step because
history_length = 3 > 0; ManagerBasedRLEnv.step calls scene.update(physics_dt) after each sim.step), i.e. the
contact impulse of that 5 ms step / 5 ms. `metalsim.parity.record_g1` records the force of the LAST 2.5 ms
substep of the 8. This script records every substep so that any window can be formed afterwards.

    python scripts/diagnostics/contact_research/substep_forces.py --engine c --tuning default
    scripts/gpu_run.sh substep_forces timing 3 -- .venv/bin/python scripts/diagnostics/contact_research/substep_forces.py --engine warp --tuning default

Writes runs/contact_research/substep_<engine>_<tuning>.npz: per protocol `<tag>_force` (T*dec, B, 3) net normal force
per Isaac contact body (world), `<tag>_rootvel` (T*dec, 6), `<tag>_rootpos` (T*dec, 3), plus `bodies`, `dec`, `dt`.
"""
import argparse, json, os, sys, time
import numpy as np, mujoco

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)
from metalsim.learn.g1_velocity import build_g1_model, ACTION_SCALE
from metalsim.physics import contact_tuning

ap = argparse.ArgumentParser()
ap.add_argument("--engine", choices=["c", "warp"], default="c")
ap.add_argument("--tuning", default="default", choices=sorted(contact_tuning.PRESETS))
ap.add_argument("--solref", default=None, help="custom contact solref 'tc,dr' (or '-k,-b'); overrides the preset's")
ap.add_argument("--solimp", default=None, help="custom contact solimp 'd0,dw,width[,mid,power]'")
ap.add_argument("--label", default=None, help="output name instead of the preset name")
ap.add_argument("--isaac", default=os.path.join(ROOT, "runs/parity/isaac/parity_out2/rt"))
ap.add_argument("--physics_dt", type=float, default=0.0025)
ap.add_argument("--protocols", default="A_hold,B_random,C_drop")
ap.add_argument("--out", default=os.path.join(ROOT, "runs/contact_research"))
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
meta = json.load(open(os.path.join(a.isaac, "meta.json")))
seqB = np.load(os.path.join(a.isaac, "action_sequence_B.npz"))
isaac_joints = [str(s) for s in seqB["joint_names"]]

# same model as metalsim.parity.record_g1 (visual camera/lights do not touch the physics; skipped)
m, info = build_g1_model("flat", visuals=False, physics_dt=a.physics_dt)
contact_tuning.apply(m, a.tuning)
if a.solref or a.solimp:     # custom contact parameters on top of the preset (joint limits etc. kept)
    contact_tuning.set_contacts(m, solref=[float(x) for x in a.solref.split(",")] if a.solref else None,
                                solimp=[float(x) for x in a.solimp.split(",")] if a.solimp else None)
act_of_joint = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, m.actuator_trnid[i][0]): i for i in range(m.nu)}
act_isaac = np.array([act_of_joint[n] for n in isaac_joints])
dec = int(round(meta["control_dt"] / a.physics_dt))
bodies = list(meta["contact_bodies"])
body_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in range(m.nbody)]
default = m.key_qpos[0].copy()
protocols = {"A_hold": (150, lambda t: np.zeros(m.nu, np.float32), None),
             "B_random": (250, lambda t: seqB["actions"][t], None),
             "C_drop": (150, lambda t: np.zeros(m.nu, np.float32), 1.0)}

def ctrl_of(a_isaac):
    c = default[7:].astype(np.float32).copy()
    c[act_isaac] = default[7:][act_isaac] + ACTION_SCALE * a_isaac
    return c

out = {"bodies": np.array(bodies), "dec": dec, "dt": a.physics_dt, "tuning": a.tuning, "engine": a.engine}
t0 = time.time()
if a.engine == "c":
    body_row = {b: bodies.index(body_names[b]) for b in range(m.nbody) if body_names[b] in bodies}
    f6 = np.zeros(6)
    for tag in a.protocols.split(","):
        T, act_fn, drop_z = protocols[tag]
        d = mujoco.MjData(m); d.qpos[:] = default; d.qvel[:] = 0
        if drop_z is not None: d.qpos[2] = drop_z
        mujoco.mj_forward(m, d)
        F = np.zeros((T * dec, len(bodies), 3), np.float32); V = np.zeros((T * dec, 6), np.float32); P = np.zeros((T * dec, 3), np.float32)
        for t in range(T):
            d.ctrl[:] = ctrl_of(act_fn(t))
            for s in range(dec):
                mujoco.mj_step(m, d)
                k = t * dec + s
                for i in range(d.ncon):
                    c = d.contact[i]
                    if c.efc_address < 0: continue
                    mujoco.mj_contactForce(m, d, i, f6)
                    fw = c.frame[:3] * f6[0]                       # normal component, world frame
                    b1, b0 = m.geom_bodyid[c.geom[1]], m.geom_bodyid[c.geom[0]]
                    if b1 in body_row: F[k, body_row[b1]] += fw
                    if b0 in body_row: F[k, body_row[b0]] -= fw
                V[k] = d.qvel[:6]; P[k] = d.qpos[:3]
        out[f"{tag}_force"] = F; out[f"{tag}_rootvel"] = V; out[f"{tag}_rootpos"] = P
        print(f"[c] {tag}: {T*dec} substeps, final z {d.qpos[2]:.3f}, {time.time()-t0:.0f} s", flush=True)
else:
    import torch, warp as wp
    wp.config.quiet = True
    from metalsim.physics.batch import BatchSim, BatchSimOptions
    from metalsim.sensors.contact import ContactSensor
    n = 1
    sim = BatchSim(m, n, options=BatchSimOptions(substeps=1, njmax=256, nconmax=32, solver_iterations=10, ls_iterations=20)); sim.synchronize()
    cs_bodies = [nm for nm in bodies if nm in body_names]
    csens = ContactSensor(sim, cs_bodies, force_mode="normal")
    rows = [bodies.index(nm) for nm in cs_bodies]
    for tag in a.protocols.split(","):
        T, act_fn, drop_z = protocols[tag]
        q = np.tile(default, (n, 1)).astype(np.float32)
        if drop_z is not None: q[:, 2] = drop_z
        sim.set_state(q, np.zeros((n, m.nv), np.float32)); v = sim.forward(); sim.after(v); sim.synchronize()
        F = np.zeros((T * dec, len(bodies), 3), np.float32); V = np.zeros((T * dec, 6), np.float32); P = np.zeros((T * dec, 3), np.float32)
        for t in range(T):
            sim.t.ctrl.copy_(torch.as_tensor(np.tile(ctrl_of(act_fn(t)), (n, 1)))); sim.synchronize()
            for s in range(dec):
                sim.step(); sim.synchronize()
                k = t * dec + s
                F[k, rows] = csens._net.numpy()[0]
                V[k] = sim.d.qvel.numpy()[0, :6]; P[k] = sim.d.qpos.numpy()[0, :3]
        out[f"{tag}_force"] = F; out[f"{tag}_rootvel"] = V; out[f"{tag}_rootpos"] = P
        print(f"[warp] {tag}: {T*dec} substeps, final z {P[-1, 2]:.3f}, {time.time()-t0:.0f} s", flush=True)
out["solref"] = str(a.solref); out["solimp"] = str(a.solimp)
fn = os.path.join(a.out, f"substep_{a.engine}_{a.label or a.tuning}.npz")
np.savez_compressed(fn, **out); print("wrote", fn)

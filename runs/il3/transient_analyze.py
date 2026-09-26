"""Offline (MuJoCo C, CPU) replay of the blown worlds' last 24 substeps dumped by transient_trace.py: from the first
buffered state, step 24 substeps with the recorded ctrl under (a) the run's own settings and (b) MuJoCo's default
contacts, and report per substep: contacts per foot, the largest contact normal force and its geom pair, the fastest
joint and whether it exceeds 3x the 37 rad/s limit (and when first), the joint-limit constraint force on that joint, the
actuator force on it, solver iterations and the total mechanical energy (mj_energyPos + mj_energyVel).
usage: python runs/il3/transient_analyze.py DUMP.npz TERRAIN SOLVER_CFG"""
import sys, json, copy, numpy as np, mujoco
from metalsim.learn.g1_velocity import build_g1_model
from metalsim.physics import contact_tuning, solver_presets
path, terrain, preset = sys.argv[1], sys.argv[2], sys.argv[3]
Z = np.load(path, allow_pickle=True); meta = json.loads(str(Z["meta"])); res = json.loads(str(Z["res"]))
hf = None
if terrain != "flat":
    from metalsim.learn.terrain import isaac_rough_terrain
    hf = isaac_rough_terrain(seed=0, collision="hfield")
base, _ = build_g1_model(terrain, hf, physics_dt=0.0025)
def model(kind):
    m = copy.deepcopy(base)
    if kind == "run":
        if preset.startswith("contact:"): contact_tuning.apply(m, preset.split(":", 1)[1])
        else: contact_tuning.apply(m, "recommended"); solver_presets.apply(m, preset)
    else:
        contact_tuning.apply(m, "default"); m.opt.iterations = 20
    m.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_ENERGY
    return m
names = lambda m, t, i: mujoco.mj_id2name(m, t, i)
def replay(m, dd, torso_scale):
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    m.body_mass[b] *= torso_scale; m.body_inertia[b] *= torso_scale
    d = mujoco.MjData(m); mujoco.mj_setConst(m, d)
    d.qpos[:] = dd["qpos"][0]; d.qvel[:] = dd["qvel"][0]
    feet = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in ("left_ankle_roll_link", "right_ankle_roll_link")]
    rows = []; first3x = None
    for s in range(dd["qpos"].shape[0]):
        d.ctrl[:] = dd["ctrl"][s]
        mujoco.mj_step(m, d)
        if not (np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()):
            rows.append({"s": s, "nonfinite": True}); break
        f6 = np.zeros(6); per = [0, 0]; fmax = 0.0; pair = None
        for i in range(d.ncon):
            c = d.contact[i]
            if c.efc_address < 0: continue
            mujoco.mj_contactForce(m, d, i, f6)
            bb = (m.geom_bodyid[c.geom[0]], m.geom_bodyid[c.geom[1]])
            for fi, fb in enumerate(feet):
                if fb in bb: per[fi] += 1
            if f6[0] > fmax: fmax = float(f6[0]); pair = (names(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom[0]), names(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom[1]))
        sp = np.abs(d.qvel[6:]); j = int(sp.argmax()); jn = names(m, mujoco.mjtObj.mjOBJ_JOINT, j + 1)
        lim = 0.0
        for r in range(d.nefc):
            if d.efc_type[r] == mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT and d.efc_id[r] == j + 1: lim += float(d.efc_force[r])
        if first3x is None and sp.max() > 3 * 37.0: first3x = (s, jn)
        rows.append({"s": s, "contacts_per_foot": per, "max_contact_force": round(fmax, 1), "pair": pair, "fastest_joint": jn,
                     "speed": round(float(sp.max()), 1), "limit_force_on_it": round(lim, 1), "actuator_force_on_it": round(float(d.qfrc_actuator[6 + j]), 1),
                     "niter": int(d.solver_niter[0]), "energy": round(float(d.energy[0] + d.energy[1]), 2),
                     "ncon": int(d.ncon), "min_dist": round(float(min([d.contact[i].dist for i in range(d.ncon)], default=0.0)), 4)})
    return rows, first3x
out = {"dump": path, "trace_summary": res, "worlds": []}
for i, mm in enumerate(meta):
    dd = {k: Z[f"d{i}_{k}"] for k in ("qpos", "qvel", "ctrl")}
    w = {"env": mm["env"], "step": mm["step"], "first_episode": mm["first_episode"], "mass_scale": mm["mass_scale"]}
    for kind in ("run", "default_contacts"):
        rows, f3 = replay(model(kind), dd, mm["mass_scale"])
        w[kind] = {"first_joint_over_3x": f3, "energy_first_last": [rows[0].get("energy"), rows[-1].get("energy")],
                   "nonfinite": any(r.get("nonfinite") for r in rows), "rows": rows}
    out["worlds"].append(w)
    r = w["run"]; dflt = w["default_contacts"]
    print(f"env {w['env']} step {w['step']} first-episode {w['first_episode']}: run energy {r['energy_first_last']} first>3x {r['first_joint_over_3x']} nonfinite {r['nonfinite']} | "
          f"default contacts energy {dflt['energy_first_last']} first>3x {dflt['first_joint_over_3x']} nonfinite {dflt['nonfinite']}")
json.dump(out, open(path.replace(".npz", "_analysis.json"), "w"), indent=1, default=str)

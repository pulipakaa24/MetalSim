"""so101-bam-gpu: MetalSim's BAM friction update (actuation.BamFrictionWarp, wired in SO101Lift by so101_patched.py)
against CPU MuJoCo with BamFriction.update before every mj_step, the way so101-servo's gpu_feeder_check.py checks the
feeder: SO101Lift (MuJoCo Warp, servo "7v4_bam", CtrlPlanFeeder, 4 physics substeps per control step) and CPU MuJoCo
(this venv's, the same scene assets/so101/scene_box_rl.xml, the same reset_draw, the same SetpointLayer fed per physics
step) from the same states (ctrl at the start pose for the first forward pass on both sides, as SO101Lift.reset), for
fixed action schedules; 40 control steps (160 physics steps), E worlds = seeds
10000.. . Prints, per schedule:

* max |q_gpu - q_cpu| per arm joint (open loop, 40 control steps);
* max |dof_frictionloss_gpu - dof_frictionloss_cpu| at the end of each control step (the budget each side computed from
  its own state after the last substep);
* the kernel alone: GPU dof_frictionloss against bam_friction_budget (float64, numpy) on the GPU's own arrays after
  the step (d.efc, d.qfrc_*, d.qvel): no trajectory divergence in it;
* negative controls: CPU with the friction frozen at the budget at rest (the update never runs), and CPU with the
  update one physics step late (the budget written before step k from the data before step k - 1).

  scripts/gpu_run.sh so101-bam-gpu render 25 -- bash runs/readme_media/scratch/so101-bam-gpu/gpu_job.sh
  (--cpu-only: the CPU side alone, no GPU, to test the script)
"""
import json
import os
import sys
from types import SimpleNamespace

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import patched  # noqa: E402,F401  (so101_patched.py as metalsim.render.film.so101)

import metalsim.render.film.so101 as S  # noqa: E402
from metalsim.physics.actuation import bam_friction_budget  # noqa: E402
from metalsim.render.film.so101 import apply_servo, friction_hook, reset_draw, servo_layer  # noqa: E402

CPU_ONLY = "--cpu-only" in sys.argv
E, T = 8, 40
SERVO = "7v4_bam"
FRICTION_DOF = int(mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF)


def sched_steps(t):
    """gpu_feeder_check.py's schedule: shoulder_lift and wrist_flex steps, odd worlds mirrored."""
    a = np.zeros((E, 6), np.float32)
    a[:, 1] = -0.2 if t < 10 else 0.3
    a[:, 3] = 0.4 if t < 20 else -0.4
    a[:, 5] = 0.5
    a[1::2] *= -1
    return a


def sched_slow(t):
    """Slow moves and holds: shoulder_lift, elbow_flex and wrist_flex ramp at 0.4-0.8 rad/s of command (below the
    Stribeck speed 0.37 rad/s at the joint after the P-loop lag for part of it), then hold (stick), then reverse."""
    a = np.zeros((E, 6), np.float32)
    s = np.clip(t / 12.0, 0, 1) - np.clip((t - 24) / 12.0, 0, 1)
    a[:, 1] = -0.1 + 0.25 * s
    a[:, 2] = 0.1 - 0.2 * s
    a[:, 3] = 0.2 * s
    a[:, 5] = 0.3
    a[1::2, :4] *= -1
    return a


SCHEDULES = dict(steps=sched_steps, slow=sched_slow)
m = mujoco.MjModel.from_xml_path(os.path.join(S.ROOT, S.SCENE))
ap = apply_servo(m, SERVO)
fr = ap["friction"]
dofs = np.array(fr["dofs"])
oid = lambda t, s: mujoco.mj_name2id(m, t, s)
box_body, box_geom = oid(mujoco.mjtObj.mjOBJ_BODY, "box"), oid(mujoco.mjtObj.mjOBJ_GEOM, "box")
box_qadr = int(m.jnt_qposadr[oid(mujoco.mjtObj.mjOBJ_JOINT, "box")])
arm_qadr = np.array([int(m.jnt_qposadr[oid(mujoco.mjtObj.mjOBJ_JOINT, j)]) for j in S.ARM_JOINTS])
mass0, fric0 = float(m.body_mass[box_body]), m.geom_friction[box_geom].copy()
rng_ = m.actuator_ctrlrange.copy()
lim = m.actuator_ctrllimited.astype(bool).reshape(-1)
lo, hi = np.where(lim, rng_[:, 0], -np.pi), np.where(lim, rng_[:, 1], np.pi)
n_sub = max(1, int(round(S.CTRL_DT / m.opt.timestep)))


def cpu_run(sched, mode):
    """mode: "update" (BamFriction.update before every mj_step) or "frozen" (the budget at rest throughout)."""
    q = np.zeros((E, T, 6)); fl = np.zeros((E, T, 6))
    fh = friction_hook(ap, m)
    for e in range(E):
        d = mujoco.MjData(m)
        q0, ms, fs = reset_draw(10_000 + e, m.qpos0.copy(), box_qadr, arm_qadr, True)
        m.body_mass[box_body] = mass0 * ms
        m.geom_friction[box_geom] = fric0 * fs
        fh.reset()
        d.qpos[:] = q0
        c0 = np.zeros(m.nu); c0[ap["actuators"]] = d.qpos[arm_qadr]
        d.ctrl[:] = c0                             # servos powered on holding (SO101Lift.reset: the first budget's torque)
        mujoco.mj_forward(m, d)
        lay = servo_layer(ap, 1)
        lay.reset(c0)
        for t in range(T):
            a = sched(t)[e].astype(np.float64)
            cmd = lo + 0.5 * (a + 1.0) * (hi - lo)
            for u in lay.plan(cmd[None], n_sub)[0]:
                d.ctrl[:] = u
                if mode == "update":
                    fh.update(d)
                mujoco.mj_step(m, d)
            q[e, t] = d.qpos[arm_qadr]
            if mode == "update":
                fh.update(d)                       # what the GPU kernel computed after this control step's last substep
            fl[e, t] = m.dof_frictionloss[dofs]
    return q, fl


def cpu_late(sched):
    """The update one physics step late: before step k, the budget from the data before step k - 1."""
    q = np.zeros((E, T, 6))
    fh = friction_hook(ap, m)
    for e in range(E):
        d = mujoco.MjData(m)
        q0, ms, fs = reset_draw(10_000 + e, m.qpos0.copy(), box_qadr, arm_qadr, True)
        m.body_mass[box_body] = mass0 * ms
        m.geom_friction[box_geom] = fric0 * fs
        fh.reset()
        d.qpos[:] = q0
        c0 = np.zeros(m.nu); c0[ap["actuators"]] = d.qpos[arm_qadr]
        d.ctrl[:] = c0                             # servos powered on holding (SO101Lift.reset: the first budget's torque)
        mujoco.mj_forward(m, d)
        lay = servo_layer(ap, 1)
        lay.reset(c0)
        nxt = m.dof_frictionloss[dofs].copy()
        for t in range(T):
            a = sched(t)[e].astype(np.float64)
            cmd = lo + 0.5 * (a + 1.0) * (hi - lo)
            for u in lay.plan(cmd[None], n_sub)[0]:
                d.ctrl[:] = u
                use = nxt
                fh.update(d)                       # computed now, used one step later
                nxt = m.dof_frictionloss[dofs].copy()
                m.dof_frictionloss[dofs] = use
                mujoco.mj_step(m, d)
            q[e, t] = d.qpos[arm_qadr]
    return q


def gpu_run(sched):
    _t = [0]

    def act(P, obs):
        a = sched(_t[0]); _t[0] += 1
        return a

    S.policy_act = act
    ad = S.SO101Lift(SimpleNamespace(adapter_kw={"servo": SERVO}), SimpleNamespace(num_envs=E, seed=10_000), None)
    assert ad.friction is not None and ad.sim.m.dof_frictionloss.shape[0] == E
    ad.reset()
    q = np.zeros((E, T, 6)); fl = np.zeros((E, T, 6)); kern = np.zeros((E, T, 6))
    sd, sm = ad.sim.d, ad.sim.m
    for t in range(T):
        ad.step()
        q[:, t] = ad.state[0][:, ad.arm_qadr]
        fl[:, t] = sm.dof_frictionloss.numpy()[:, dofs]
        # the kernel alone: numpy on the GPU's own arrays after the step
        nefc = sd.nefc.numpy(); et, ei, ef = sd.efc.type.numpy(), sd.efc.id.numpy(), sd.efc.force.numpy().astype(np.float64)
        bias, cons = sd.qfrc_bias.numpy().astype(np.float64), sd.qfrc_constraint.numpy().astype(np.float64)
        act_, qv = sd.qfrc_actuator.numpy().astype(np.float64), sd.qvel.numpy().astype(np.float64)
        for w in range(E):
            n = min(int(nefc[w]), et.shape[1])
            own = np.zeros(m.nv)
            k = et[w, :n] == FRICTION_DOF
            np.add.at(own, ei[w, :n][k], ef[w, :n][k])
            te = -bias[w, dofs] + cons[w, dofs] - own[dofs]
            tm = act_[w, dofs] - fr["emf"] * qv[w, dofs]
            kern[w, t] = bam_friction_budget(fr["params"], tm, te, qv[w, dofs])
    ov = ad.sim.overflow_flags()
    return q, fl, kern, ov, ad.state[2][:, 2].copy()


out = dict(servo=SERVO, worlds=E, control_steps=T, substeps=n_sub, seeds=[10_000, 10_000 + E - 1], schedules={})
for name, sched in SCHEDULES.items():
    qc, flc = cpu_run(sched, "update")
    qf, _ = cpu_run(sched, "frozen")
    ql = cpu_late(sched)
    r = dict(motion_range=(qc.max(1) - qc.min(1)).max(0).round(3).tolist(),
             budget_range_cpu=[float(flc.min()), float(flc.max())])
    if not CPU_ONLY:
        qg, flg, kern, ov, boxz = gpu_run(sched)
        r.update(max_dq=np.abs(qg - qc).max((0, 1)).tolist(),
                 max_dq_per_world=np.abs(qg - qc).max((1, 2)).tolist(),
                 max_dq_first10=np.abs(qg - qc)[:, :10].max((0, 2)).tolist(),
                 max_dfl=np.abs(flg - flc).max((0, 1)).tolist(),
                 kernel_vs_numpy_max_abs=float(np.abs(flg - kern).max()),
                 kernel_vs_numpy_max_rel=float((np.abs(flg - kern) / np.abs(kern)).max()),
                 neg_frozen_max_dq=np.abs(qg - qf).max((0, 1)).tolist(),
                 neg_late_max_dq=np.abs(qg - ql).max((0, 1)).tolist(),
                 overflow=ov, box_z=boxz.round(4).tolist())
    else:
        r.update(cpu_frozen_vs_update_max_dq=np.abs(qf - qc).max((0, 1)).tolist(),
                 cpu_late_vs_update_max_dq=np.abs(ql - qc).max((0, 1)).tolist())
    out["schedules"][name] = r
    print(name, json.dumps({k: (["%.2e" % x for x in v] if isinstance(v, list) and v and isinstance(v[0], float) else v)
                            for k, v in r.items()}), flush=True)
os.makedirs(os.path.join(HERE, "check"), exist_ok=True)
json.dump(out, open(os.path.join(HERE, "check", "cpu_only.json" if CPU_ONLY else "gpu_bam_check.json"), "w"), indent=1)

"""MetalSim (MuJoCo Warp) evaluation of the deployment-style expert, the CPU harness's loop on the GPU.

The film adapter's model (metalsim/render/film/so101.py: assets/so101/scene_box_rl.xml, ``apply_servo``) in a BatchSim
of one world per seed 10000 + e, reset as SO101Lift.reset does (so101Sim's seeded draw ``reset_draw``: start qpos, box
mass and friction per world), but stepped one 5 ms physics step per call (substeps 1) so that the 30 Hz control ticks
(physics step round(k * 200 / 30)) and the servo layer's per-step targets are exact, as in eval_cpu.py. At each tick,
every live world's qpos / qvel are copied to a CPU MjData of the same model, ``mj_forward`` runs (fresh kinematics) and
that world's DeployExpert computes its absolute targets; the servo layer (command delay + firmware profile, all worlds
at once) turns the held command into each physics step's ctrl. Success, up-at-6 s, out-of-bounds and the 20 ms sampling
are eval_cpu.py's. Writes <out>/episodes.json, <out>/smooth.json, <out>/traj.npz, as eval_cpu.py.

``--cpu-dry N``: the same loop on N CPU MjData worlds (a stand-in with BatchSim's interface) instead of the GPU, to
test the harness without the queue; compare with eval_cpu.py on the same seeds.

GPU (queue, from the MetalSim root):
  scripts/gpu_run.sh so101-expert-metalsim render 15 -- .venv/bin/python \
      runs/readme_media/scratch/so101-expert/eval_metalsim.py 7v4_real runs/readme_media/scratch/so101-expert/metalsim/deploy_7v4_real
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT_MS = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.join(HERE, "..", "so101-servo"))
sys.path.append(ROOT_MS)
import mujoco  # noqa: E402

import metalsim.render.film.so101_expert as X  # noqa: E402
from metalsim.render.film.so101 import (ARM_JOINTS, LIFT_Z, OOB_R, ROOT, SCENE, SUCCESS_HOLD, apply_servo,  # noqa: E402
                                        reset_draw, servo_layer)
from sag import steady_error  # noqa: E402
from smoothness import joint_motion  # noqa: E402

args = sys.argv[1:]
DRY = None
if "--cpu-dry" in args:
    i = args.index("--cpu-dry"); DRY = int(args[i + 1]); del args[i:i + 2]
servo, out_dir = args[0], args[1]
E = DRY or (int(args[2]) if len(args) > 2 else 100)
SEED0 = 10_000
PHASES = ("approach", "descend", "grasp", "lift")
HZ = X.CTRL_HZ
os.makedirs(out_dir, exist_ok=True)

m = mujoco.MjModel.from_xml_path(os.path.join(ROOT, SCENE))
applied = apply_servo(m, servo)
layer = servo_layer(applied, E)
act = np.array(applied["actuators"])
oid = lambda t, s: mujoco.mj_name2id(m, t, s)
grip_site, box_body, box_geom = oid(mujoco.mjtObj.mjOBJ_SITE, "gripperframe"), oid(mujoco.mjtObj.mjOBJ_BODY, "box"), \
    oid(mujoco.mjtObj.mjOBJ_GEOM, "box")
box_qadr = int(m.jnt_qposadr[oid(mujoco.mjtObj.mjOBJ_JOINT, "box")])
arm_qadr = np.array([int(m.jnt_qposadr[oid(mujoco.mjtObj.mjOBJ_JOINT, j)]) for j in ARM_JOINTS])
arm_vadr = np.array([int(m.jnt_dofadr[oid(mujoco.mjtObj.mjOBJ_JOINT, j)]) for j in ARM_JOINTS])
ids = dict(grip_site=grip_site, box_body=box_body, box_qadr=box_qadr, arm_qadr=arm_qadr, arm_vadr=arm_vadr)
mass0, fric0, home = float(m.body_mass[box_body]), m.geom_friction[box_geom].copy(), m.qpos0.copy()
DT = float(m.opt.timestep)
NPH, NS = int(round(6.0 / DT)), int(round(0.02 / DT))


class _Arr:                                    # BatchSim-like array view over CPU worlds (dry run)
    def __init__(self, ds, name):
        self.ds, self.name = ds, name

    def numpy(self):
        return np.stack([getattr(d, self.name).copy() for d in self.ds]).astype(np.float32)

    def assign(self, a):
        for d, x in zip(self.ds, np.asarray(a)):
            getattr(d, self.name)[:] = x


class CpuWorlds:
    """N CPU MuJoCo worlds with the BatchSim calls this script uses (dry run only)."""

    def __init__(self, m, n):
        self.ms = [mujoco.MjModel.from_xml_path(os.path.join(ROOT, SCENE)) for _ in range(n)]
        for mm in self.ms:
            apply_servo(mm, servo)
        self.ds = [mujoco.MjData(mm) for mm in self.ms]
        self.d = type("D", (), {k: _Arr(self.ds, k) for k in ("qpos", "qvel", "xpos", "ctrl")})()
        self.device = None

    def reset_worlds(self, q, mass, fric):
        for mm, d, qq, ms, fs in zip(self.ms, self.ds, q, mass, fric):
            mujoco.mj_resetData(mm, d)
            mm.body_mass[box_body] = mass0 * ms
            mm.geom_friction[box_geom] = fric0 * fs
            d.qpos[:] = qq
            mujoco.mj_forward(mm, d)

    def step(self):
        for mm, d in zip(self.ms, self.ds):
            mujoco.mj_step(mm, d)

    def synchronize(self):
        pass

    def check_overflow(self):
        return {}

    def overflow_flags(self):
        return {}


draws = [reset_draw(SEED0 + e, home, box_qadr, arm_qadr, True) for e in range(E)]
q0 = np.stack([x[0] for x in draws]); mass_s = np.array([x[1] for x in draws]); fric_s = np.array([x[2] for x in draws])
if DRY:
    sim = CpuWorlds(m, E)
    sim.reset_worlds(q0, mass_s, fric_s)
    scoped = None
else:
    import warp as wp

    from metalsim.physics.batch import BatchSim, BatchSimOptions

    wp.config.quiet = True
    sim = BatchSim(m, E, options=BatchSimOptions(substeps=1, njmax=512, per_world_fields=("body_mass", "geom_friction")))
    sim.reset()                                     # SO101Lift.reset's sequence
    sim.synchronize()
    bm = sim.m.body_mass.numpy(); bm[:, box_body] = mass0 * mass_s; sim.m.body_mass.assign(bm)
    gf = sim.m.geom_friction.numpy(); gf[:, box_geom] = fric0[None] * fric_s[:, None]; sim.m.geom_friction.assign(gf)
    sim.set_state(q0, np.zeros((E, m.nv)))
    sim.synchronize()
    scoped = lambda: wp.ScopedDevice(sim.device)

kins = [mujoco.MjData(m) for _ in range(E)]
ctls = [X.DeployExpert(m, ids, hz=HZ) for _ in range(E)]
qpos, qvel = sim.d.qpos.numpy().astype(np.float64), sim.d.qvel.numpy().astype(np.float64)
start_err = float(np.abs(qpos - q0).max())
layer.reset(qpos[:, arm_qadr])
cmd = qpos[:, arm_qadr].copy()
for e in range(E):
    ctls[e].reset(qpos[e, arm_qadr])
ticks = {int(round(k / HZ / DT)) for k in range(int(6.0 * HZ) + 1)}
R = dict(q=np.full((E, NPH, 6), np.nan), qd=np.full((E, NPH, 6), np.nan), tgt=np.full((E, NPH, 6), np.nan),
         cmd=np.full((E, NPH, 6), np.nan), phase=np.full((E, NPH), -1, np.int8), box=np.full((E, NPH // NS + 1, 3), np.nan))
alive = np.ones(E, bool)
held = np.zeros(E, int); success_t = np.full(E, -1); hmax = np.zeros(E); term = np.zeros(E, bool)
R["box"][:, 0] = sim.d.xpos.numpy()[:, box_body]
ctrl_full = np.zeros((E, m.nu), np.float32)
t0 = time.time()
for n in range(NPH):
    if n in ticks:
        for e in np.flatnonzero(alive):
            k = kins[e]
            k.qpos[:], k.qvel[:] = qpos[e], qvel[e]
            mujoco.mj_forward(m, k)
            cmd[e] = ctls[e].act(k, n * DT)
    tg = layer.tick(cmd)
    ctrl_full[:, act] = tg
    if scoped is None:
        sim.d.ctrl.assign(ctrl_full)
        sim.step()
    else:
        with scoped():
            sim.d.ctrl.assign(ctrl_full)
            sim.step()
    sim.synchronize()
    qpos, qvel = sim.d.qpos.numpy().astype(np.float64), sim.d.qvel.numpy().astype(np.float64)
    a = alive
    R["cmd"][a, n], R["tgt"][a, n] = cmd[a], tg[a]
    R["phase"][a, n] = [PHASES.index(ctls[e].phase) for e in np.flatnonzero(a)]
    R["q"][a, n], R["qd"][a, n] = qpos[a][:, arm_qadr], qvel[a][:, arm_vadr]
    if (n + 1) % NS == 0:
        k = (n + 1) // NS
        box = sim.d.xpos.numpy().astype(np.float64)[:, box_body]
        R["box"][a, k] = box[a]
        h = box[:, 2]
        hmax[a] = np.maximum(hmax[a], h[a])
        held = np.where(h > LIFT_Z, held + 1, 0)
        new = a & (held >= SUCCESS_HOLD) & (success_t < 0)
        success_t[new] = k - 1
        oob = a & (np.linalg.norm(box[:, :2], axis=1) > OOB_R)
        term |= oob
        alive &= ~oob
wall = time.time() - t0
box = sim.d.xpos.numpy().astype(np.float64)[:, box_body]
end_h = np.array([R["box"][e][~np.isnan(R["box"][e][:, 2])][-1, 2] for e in range(E)])
eps = [dict(seed=SEED0 + e, success=bool(success_t[e] >= 0), success_t=int(success_t[e]) if success_t[e] >= 0 else None,
            time_to_lift_s=round(float(success_t[e]) * 0.02, 3) if success_t[e] >= 0 else None, hmax=round(float(hmax[e]), 4),
            final_height=round(float(end_h[e]), 4), up_at_end=bool(end_h[e] > LIFT_Z and not term[e]), terminated=bool(term[e]),
            phase_end=ctls[e].phase, retries=ctls[e].retries, events=ctls[e].events) for e in range(E)]
succ = [x for x in eps if x["success"]]
summary = dict(episodes=E, successes=len(succ), up_at_6s=int(sum(x["up_at_end"] for x in eps)),
               time_to_lift_mean_s=float(np.mean([x["time_to_lift_s"] for x in succ])) if succ else None,
               time_to_lift_median_s=float(np.median([x["time_to_lift_s"] for x in succ])) if succ else None,
               max_height_mean=float(np.mean(hmax)), terminated=int(term.sum()), retries=int(sum(x["retries"] for x in eps)),
               phase_end={p: int(sum(x["phase_end"] == p for x in eps)) for p in PHASES})
SM = joint_motion(R["q"], R["qd"], DT)
SAG = steady_error(R["q"], R["qd"], R["tgt"], R["cmd"], R["phase"], DT)
try:
    from metalsim.render.film import provenance
    prov = dict(software=provenance.software(), hardware=provenance.hardware())
except Exception as ex:                                      # provenance is best effort
    prov = dict(provenance_error=repr(ex))
try:
    overflow = sim.check_overflow()
except RuntimeError as ex:                                   # recorded, outputs still written
    overflow = repr(ex)
meta = dict(script="runs/readme_media/scratch/so101-expert/eval_metalsim.py", backend="cpu-dry" if DRY else "metalsim",
            controller="metalsim/render/film/so101_expert.py DeployExpert", ctrl_hz=HZ, servo=applied,
            scene=SCENE, seeds=[SEED0, SEED0 + E - 1], start_qpos_max_err=start_err, wall_s=round(wall, 1),
            capacity_overflow=overflow, solver_flags=sim.overflow_flags(), **prov)
json.dump(dict(**meta, summary=summary, episodes=eps), open(os.path.join(out_dir, "episodes.json"), "w"), indent=1,
          default=lambda o: o.item() if hasattr(o, "item") else str(o))
json.dump(dict(**meta, summary=summary, smoothness=SM, sag=SAG), open(os.path.join(out_dir, "smooth.json"), "w"), indent=1,
          default=lambda o: o.item() if hasattr(o, "item") else str(o))
np.savez_compressed(os.path.join(out_dir, "traj.npz"), **R, seeds=np.arange(SEED0, SEED0 + E), dt=DT)
print(json.dumps(summary))

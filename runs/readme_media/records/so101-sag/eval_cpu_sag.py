"""[so101-sag copy of so101-expert/eval_cpu.py: `servo` is a sag_servo.py variant (7v4_real, bam_m1, bam_m6, ...),
or with --patched a servo option of the patched so101.py (patch/new/: e.g. 7v4_bam); the BAM friction hook runs before
every mj_step; nothing else changed.]

CPU evaluation of the deployment-style expert (metalsim/render/film/so101_expert.py DeployExpert) on a servo option.

so101Sim's own SO101LiftEnv (unchanged; imported through sys.path) supplies the scene, the seeded reset (seeds
10000 + ep, eval_lift_native.py's) and the success test; the servo is applied with metalsim.render.film.so101
``apply_servo`` and ``servo_layer`` (the "_real" options' command delay + firmware profile at every 5 ms physics step),
as runs/readme_media/scratch/so101-servo/servo_cpu.py does. The control loop is this script's, because LeRobot's 30 Hz
does not fit env.step's 20 ms: the physics runs 5 ms steps; control tick k falls on physics step round(k * 200 / 30)
(ticks 30 or 35 ms apart, 30 Hz on average); at a tick the controller reads a kinematic copy of the state (qpos, qvel,
mj_forward: "fresh kinematics", as eval_metalsim.py and so101-servo's eval_cpu.py --fresh-kin), and its absolute
targets are held (zero-order hold) until the next tick, through the servo layer. Success as so101Sim's env: box
centre above LIFT_Z = 0.10 m for SUCCESS_HOLD = 10 consecutive 20 ms samples; out of bounds (> 0.45 m) ends the
episode; 6 s (1200 physics steps).

Writes <out>/episodes.json (per-episode outcome, phase timeline), <out>/traj.npz (per physics step: arm q, qdot, the
servo target after the layer, the streamed command; per 20 ms: box position) and <out>/smooth.json (smoothness.
joint_motion + sag.py's steady-state tracking error).

  <so101Sim>/urlab_bridge/.venv/bin/python eval_cpu.py <so101Sim root> <servo> <out_dir> [episodes] [--hz 30]
      [--seeds a,b,...] [--over '{"accel_limit": Infinity}'] (apply_servo overrides)
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
METALSIM = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
args = sys.argv[1:]
HZ = 30.0
if "--hz" in args:
    i = args.index("--hz"); HZ = float(args[i + 1]); del args[i:i + 2]
SEEDS = None
OVER = {}
PATCHED = "--patched" in args          # run the patch (patch/new/) through so101.apply_servo + friction_hook
if PATCHED:
    args.remove("--patched")
if "--over" in args:
    i = args.index("--over"); OVER = json.loads(args[i + 1]); del args[i:i + 2]
if "--seeds" in args:
    i = args.index("--seeds"); SEEDS = [int(s) for s in args[i + 1].split(",")]; del args[i:i + 2]
root, servo, out_dir = args[0:3]
EPISODES = int(args[3]) if len(args) > 3 else 100
sys.path.insert(0, os.path.join(root, "rl", "native"))
sys.path.append(METALSIM)
sys.path.insert(1, HERE)
sys.path.insert(2, os.path.join(HERE, "..", "so101-servo"))
sys.path.insert(3, os.path.join(HERE, "..", "so101-expert"))
if PATCHED:
    import patched  # noqa: E402,F401
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from so101_lift import LIFT_Z, SUCCESS_HOLD, SO101LiftEnv  # noqa: E402

import metalsim.render.film.so101_expert as X  # noqa: E402
from metalsim.render.film.so101 import servo_layer  # noqa: E402
import sag_servo as SS  # noqa: E402
from sag import steady_error  # noqa: E402
from smoothness import joint_motion  # noqa: E402

PHASES = ("approach", "descend", "grasp", "lift")
seeds = SEEDS or [10_000 + e for e in range(EPISODES)]
E = len(seeds)
os.makedirs(out_dir, exist_ok=True)
env = SO101LiftEnv()
m, d = env.model, env.data
if PATCHED:
    from metalsim.render.film.so101 import apply_servo, friction_hook  # noqa: E402
    applied = apply_servo(m, servo, **OVER)
    _fh = friction_hook(applied, m)
    hook = None if _fh is None else type("H", (), dict(reset=lambda self: _fh.reset(), update=lambda self: _fh.update(d)))()
else:
    applied, hook = SS.apply(m, d, servo, **OVER)
layer = servo_layer(applied, 1)
act = np.array(applied.get("actuators", range(6)))
ids = dict(grip_site=env._grip_site, box_body=env._box_body, box_qadr=env._box_qadr, arm_qadr=env._arm_qadr,
           arm_vadr=env._arm_vadr)
ctl = X.DeployExpert(m, ids, hz=HZ)
kin = mujoco.MjData(m)
DT = m.opt.timestep
NPH = int(round(6.0 / DT))                      # 1200 physics steps
NS = env._n_sub                                 # 4: the 20 ms success sampling
ticks = sorted({int(round(k / HZ / DT)) for k in range(int(6.0 * HZ) + 1)})
tickset = set(ticks)
R = dict(q=np.full((E, NPH, 6), np.nan), qd=np.full((E, NPH, 6), np.nan), tgt=np.full((E, NPH, 6), np.nan),
         cmd=np.full((E, NPH, 6), np.nan), phase=np.full((E, NPH), -1, np.int8),
         box=np.full((E, NPH // NS + 1, 3), np.nan), boxq=np.full((E, NPH // NS + 1, 4), np.nan))
eps = []
t0 = time.time()
for ei, seed in enumerate(seeds):
    env.reset(seed=seed)
    qa = d.qpos[env._arm_qadr].copy()
    if layer is not None:
        layer.reset(qa[None])
    ctl.reset(qa)
    if hook is not None:
        hook.reset()
    cmd = qa.copy()
    held, success_t, hmax, term = 0, None, 0.0, False
    R["box"][ei, 0] = d.xpos[env._box_body]
    for n in range(NPH):
        if n in tickset:
            kin.qpos[:], kin.qvel[:] = d.qpos, d.qvel
            mujoco.mj_forward(m, kin)
            cmd = ctl.act(kin, d.time)
        if layer is not None:
            tg = layer.tick(cmd[None])[0]
            d.ctrl[act] = tg
        else:
            tg = cmd
            d.ctrl[act] = cmd
        R["cmd"][ei, n], R["tgt"][ei, n] = cmd, tg
        R["phase"][ei, n] = PHASES.index(ctl.phase)
        if hook is not None:
            hook.update()
        mujoco.mj_step(m, d)
        R["q"][ei, n], R["qd"][ei, n] = d.qpos[env._arm_qadr], d.qvel[env._arm_vadr]
        if (n + 1) % NS == 0:                      # a 20 ms control step of so101Sim's env ends here
            k = (n + 1) // NS
            box = d.xpos[env._box_body].copy()
            R["box"][ei, k] = box
            R["boxq"][ei, k] = d.xquat[env._box_body]
            h = float(box[2])
            hmax = max(hmax, h)
            held = held + 1 if h > LIFT_Z else 0
            if held >= SUCCESS_HOLD and success_t is None:
                success_t = k - 1                  # eval_lift_native.py's t (0-based control step)
            if np.linalg.norm(box[:2]) > 0.45:
                term = True
                break
    end_h = float(d.xpos[env._box_body][2])
    tilt = float(np.degrees(np.arccos(np.clip(d.xmat[env._box_body].reshape(3, 3)[2, 2], -1, 1))))
    eps.append(dict(seed=seed, success=success_t is not None, success_t=success_t,
                    time_to_lift_s=None if success_t is None else round(success_t * 0.02, 3), hmax=round(hmax, 4),
                    final_height=round(end_h, 4), up_at_end=bool(end_h > LIFT_Z and not term), terminated=term,
                    phase_end=ctl.phase, retries=ctl.retries, box_tilt_end_deg=round(tilt, 1),
                    events=ctl.events))
wall = time.time() - t0
succ = [e for e in eps if e["success"]]
summary = dict(episodes=E, successes=len(succ), up_at_6s=int(sum(e["up_at_end"] for e in eps)),
               time_to_lift_mean_s=float(np.mean([e["time_to_lift_s"] for e in succ])) if succ else None,
               time_to_lift_median_s=float(np.median([e["time_to_lift_s"] for e in succ])) if succ else None,
               max_height_mean=float(np.mean([e["hmax"] for e in eps])), terminated=int(sum(e["terminated"] for e in eps)),
               retries=int(sum(e["retries"] for e in eps)),
               phase_end={p: int(sum(e["phase_end"] == p for e in eps)) for p in PHASES})
SM = joint_motion(R["q"], R["qd"], DT)
SAG = steady_error(R["q"], R["qd"], R["tgt"], R["cmd"], R["phase"], DT)
meta = dict(script="runs/readme_media/scratch/so101-sag/eval_cpu_sag.py", controller="metalsim/render/film/so101_expert.py "
            "DeployExpert", ctrl_hz=HZ, vmax=X.VMAX.tolist(), amax=X.AMAX.tolist(), t_settle=X.T_SETTLE, servo=applied,
            env="so101Sim rl/native/so101_lift.py SO101LiftEnv (reset, scene, success test; unchanged)",
            over=OVER, mujoco=mujoco.__version__, seeds=[seeds[0], seeds[-1]] if SEEDS is None else seeds, wall_s=round(wall, 1))
json.dump(dict(**meta, summary=summary, episodes=eps), open(os.path.join(out_dir, "episodes.json"), "w"), indent=1)
json.dump(dict(**meta, summary=summary, smoothness=SM, sag=SAG), open(os.path.join(out_dir, "smooth.json"), "w"), indent=1)
np.savez_compressed(os.path.join(out_dir, "traj.npz"), **R, seeds=np.array(seeds), dt=DT)
print(json.dumps(summary))

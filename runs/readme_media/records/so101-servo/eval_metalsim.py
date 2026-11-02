"""MetalSim evaluation for the so101-servo stream (so101-smooth's eval_metalsim.py with servo overrides and the arm's
q / qdot recorded at every 5 ms physics step by a substep hook, for smoothness.joint_motion -> smooth.json): the film's adapter (metalsim.render.film.so101.SO101Lift, MuJoCo
Warp, one world per seed 10000 + e, so101Sim's control loop) with a servo model and a policy file, 300 control steps.
Records every control step's action, ctrl, arm joint velocities and arm servo forces (``actuator_force``, the last of
the 4 physics steps, as runs/film/so101_lift_74v/port/compare_servo.py) while each world's episode runs, and reports
metrics.control_style; with a CPU reference.json of the same policy and servo (eval_cpu.py), also the per-seed
agreement of success.

``expert`` in place of the policy file: so101Sim's scripted expert (rl/native/scripted_expert.py ``Expert``, its
defaults) drives the MuJoCo Warp worlds. One Expert per world, each on a CPU MjData of so101Sim's own scene (its
``pickup`` keyframe calibrates the expert's TCP; assets/so101/scene_box_rl.xml has no keyframe): at every control step
the world's qpos / qvel are copied from the GPU into that MjData, ``mj_forward`` runs, and the expert's ctrl
(``norm_ctrl``) replaces the policy's action (``so101.policy_act`` is replaced in this process only). The expert
reads kinematics of the state after the control step's last physics step; on CPU (eval_cpu.py) it reads env.data
after ``mj_step``, whose positions lag that state by one 5 ms physics step. Needs the so101Sim root: argument 5 or
``SO101SIM``.

A GPU-queue job (render kind), from the repository root:
  scripts/gpu_run.sh so101-smooth-eval render 10 -- .venv/bin/python \
      runs/readme_media/scratch/so101-servo/eval_metalsim.py <policy.pt|expert> <servo> <out_dir> [cpu reference.json]
      [so101Sim root] [--over '{...}']
"""
import json
import os
import sys
from types import SimpleNamespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.join(HERE, "..", "so101-smooth"))
from metrics import control_style, row  # noqa: E402 (so101-smooth's)
from smoothness import joint_motion  # noqa: E402

from metalsim.render.film import provenance  # noqa: E402
from metalsim.render.film.so101 import ARM_JOINTS, LIFT_Z, MAX_STEPS, SO101Lift  # noqa: E402

ARGS = sys.argv[1:]
OVER = {}
if "--over" in ARGS:
    i = ARGS.index("--over"); OVER = json.loads(ARGS[i + 1]); del ARGS[i:i + 2]
sys.argv = [sys.argv[0]] + ARGS
policy, servo, out_dir = sys.argv[1:4]
ref = json.load(open(sys.argv[4])) if len(sys.argv) > 4 and sys.argv[4] != "-" else None
E = 100
os.makedirs(out_dir, exist_ok=True)
EXPERT = policy == "expert"
ROOT_MS = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
ad = SO101Lift(SimpleNamespace(adapter_kw={"servo": servo, "servo_over": OVER}), SimpleNamespace(num_envs=E, seed=10_000),
               os.path.join(ROOT_MS, "runs/film/so101_lift/policy/so101_lift_ppo.pt") if EXPERT else policy)
if EXPERT:
    for k in ("EXPERT_DQ", "EXPERT_GRIP_OPEN"):
        if k in os.environ:
            raise SystemExit(f"{k} is set: the expert must run with its defaults")
    so101sim = sys.argv[5] if len(sys.argv) > 5 else os.environ["SO101SIM"]
    sys.path.insert(0, os.path.join(so101sim, "rl", "native"))
    import mujoco
    import scripted_expert as SE
    from so101_lift import SO101LiftEnv

    import metalsim.render.film.so101 as S101

    kin = SO101LiftEnv()
    assert kin.model.nq == ad.model.nq and kin.model.nv == ad.model.nv
    assert (kin._arm_qadr == ad.arm_qadr).all() and kin._box_qadr == ad.box_qadr
    shims = [SimpleNamespace(model=kin.model, data=mujoco.MjData(kin.model), _grip_site=kin._grip_site,
                             _box_body=kin._box_body, _box_qadr=kin._box_qadr, _arm_qadr=kin._arm_qadr,
                             _arm_vadr=kin._arm_vadr) for _ in range(E)]
    experts = [SE.Expert(sh) for sh in shims]

    def expert_act(P, obs):
        qpos, qvel = ad.state[0], ad.state[1]
        out = np.zeros((E, 6), np.float32)
        for e in range(E):
            dd = shims[e].data
            dd.qpos[:], dd.qvel[:] = qpos[e], qvel[e]
            mujoco.mj_forward(kin.model, dd)
            out[e] = SE.norm_ctrl(kin.model, experts[e].act())
        return out

    S101.policy_act = expert_act
    for ex in experts:
        ex.reset()
oid = lambda j: ad.mujoco.mj_name2id(ad.model, ad.mujoco.mjtObj.mjOBJ_ACTUATOR, j)
act_ids = np.array([oid(j) for j in ARM_JOINTS])
lim = ad.model.actuator_forcerange[act_ids, 1]
import warp as wp  # noqa: E402


@wp.kernel
def _rec(qpos: wp.array2d(dtype=wp.float32), qvel: wp.array2d(dtype=wp.float32), qadr: wp.array(dtype=wp.int32),
         vadr: wp.array(dtype=wp.int32), k: wp.array(dtype=wp.int32), out: wp.array3d(dtype=wp.float32)):
    w, j = wp.tid()
    out[w, k[0], j] = qpos[w, qadr[j]]
    out[w, k[0], 6 + j] = qvel[w, vadr[j]]


@wp.kernel
def _inc(k: wp.array(dtype=wp.int32)):
    k[0] = k[0] + 1


NS = ad.n_sub
with wp.ScopedDevice(ad.sim.device):
    rec_buf = wp.zeros((E, NS, 12), dtype=wp.float32)
    rec_k = wp.zeros(1, dtype=wp.int32)
    qadr_w = wp.array(ad.arm_qadr.astype(np.int32), dtype=wp.int32)
    vadr_w = wp.array(ad.arm_vadr.astype(np.int32), dtype=wp.int32)


def _hook():
    wp.launch(_rec, dim=(E, 6), inputs=[ad.sim.d.qpos, ad.sim.d.qvel, qadr_w, vadr_w, rec_k], outputs=[rec_buf])
    wp.launch(_inc, dim=1, inputs=[rec_k])


ad.sim.add_substep_hook(_hook)
ad.reset()
T = MAX_STEPS
SUB = np.full((E, T * NS, 12), np.nan)
action, ctrl = np.zeros((E, T, 6)), np.zeros((E, T, 6))
qd, force, alive = np.zeros((E, T, 6)), np.zeros((E, T, 6)), np.zeros((E, T), bool)
for t in range(T):
    alive[:, t] = ad.alive
    with wp.ScopedDevice(ad.sim.device):
        rec_k.assign(np.zeros(1, np.int32))
    ad.step()
    SUB[:, t * NS:(t + 1) * NS] = rec_buf.numpy().astype(np.float64)
    action[:, t], ctrl[:, t] = ad.last["action"], ad.last["ctrl"]
    qd[:, t] = ad.state[1][:, ad.arm_vadr]
    force[:, t] = ad.sim.d.actuator_force.numpy()[:, act_ids].astype(np.float64)
ov = ad.sim.check_overflow()
ends = alive.sum(1) - 1
for e in range(E):                                    # NaN after each world's first episode ends (as eval_cpu)
    SUB[e, (ends[e] + 1) * NS:] = np.nan
SM = joint_motion(SUB[..., 0:6], SUB[..., 6:12], float(ad.model.opt.timestep))
sat = np.abs(force) >= lim * (1 - 1e-4)
M = control_style(action, ctrl, ends, sat, alive, qd, alive, ad.success_t.copy(), ad.end_height > LIFT_Z, ad.hmax)
out = dict(script="runs/readme_media/scratch/so101-servo/eval_metalsim.py", over=OVER, smoothness=SM, policy=("so101Sim rl/native/scripted_expert.py Expert (defaults: DQ_MAX %s, grip %s / %s)"
                   % (SE.DQ_MAX, SE.GRIP_OPEN, SE.GRIP_CLOSE)) if EXPERT else provenance.relpath(policy),
           servo=ad.servo, metrics=M, seeds=[10_000, 10_000 + E - 1], capacity_overflow=ov,
           per_seed=dict(success=ad.succeeded.tolist(), success_t=ad.success_t.tolist(),
                         end_height=ad.end_height.round(4).tolist()),
           software=provenance.software(), hardware=provenance.hardware())
if ref is not None:
    r_ok = np.array([e["success"] for e in ref["episodes"]])
    out["cpu_agreement"] = dict(same_outcome=int((r_ok == ad.succeeded).sum()),
                                only_metalsim=[10_000 + int(e) for e in np.flatnonzero(ad.succeeded & ~r_ok)],
                                only_cpu=[10_000 + int(e) for e in np.flatnonzero(~ad.succeeded & r_ok)])
json.dump(out, open(os.path.join(out_dir, "metrics.json"), "w"), indent=1,
          default=lambda o: o.item() if hasattr(o, "item") else str(o))
np.savez_compressed(os.path.join(out_dir, "metalsim.npz"), action=action, ctrl=ctrl, qd=qd, force=force, alive=alive,
                    sub=SUB.astype(np.float32))
json.dump(dict(script=out["script"], servo=ad.servo, over=OVER, smoothness=SM, successes=int(ad.succeeded.sum()),
               cpu_agreement=out.get("cpu_agreement")), open(os.path.join(out_dir, "smooth.json"), "w"), indent=1,
          default=lambda o: o.item() if hasattr(o, "item") else str(o))
print(row(f"metalsim {os.path.basename(policy)} {servo}", M))
print("cpu agreement:", out.get("cpu_agreement"), "overflow:", ov)

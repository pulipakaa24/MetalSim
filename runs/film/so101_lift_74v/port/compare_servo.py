"""MetalSim against so101Sim's CPU reference with a servo model of metalsim/render/film/so101.py (``SERVOS``): runs
runs/film/so101_lift/port/compare.py unchanged, with the film's adapter (``SO101Lift``) given ``adapter_kw["servo"]``,
and also records each arm servo's force (``actuator_force``, MuJoCo Warp, the last of the 4 physics steps of every
20 ms control step) to report the fraction of control steps each joint spends at its force limit.

A GPU-queue job (render kind), from the repository root:
  scripts/gpu_run.sh so101-74v-compare render 10 -- .venv/bin/python runs/film/so101_lift_74v/port/compare_servo.py \
      <servo> <reference_dir of the same servo> <out_dir>
Writes <out_dir>/compare.json (compare.py's, plus ``servo`` and ``torque_saturation``) and compare.npz.
"""
import json
import os
import runpy
import sys
from types import SimpleNamespace

import numpy as np

import metalsim.render.film.so101 as S

HERE = os.path.dirname(os.path.abspath(__file__))
COMPARE = os.path.join(S.ROOT, "runs", "film", "so101_lift", "port", "compare.py")
servo, ref_dir, out_dir = sys.argv[1:4]
rec = dict(force=[], alive=[])


class SO101LiftServo(S.SO101Lift):
    def __init__(self, task, film, checkpoint, viewer=None):
        task = SimpleNamespace(adapter_kw={**(getattr(task, "adapter_kw", None) or {}), "servo": servo})
        super().__init__(task, film, checkpoint, viewer)
        oid = lambda j: self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_ACTUATOR, j)
        self.act_ids = np.array([oid(j) for j in S.ARM_JOINTS])

    def step(self):
        alive = self.alive.copy()
        out = super().step()
        rec["alive"].append(alive)
        rec["force"].append(self.sim.d.actuator_force.numpy()[:, self.act_ids].astype(np.float64))
        return out


S.SO101Lift = SO101LiftServo
sys.argv = [COMPARE, ref_dir, out_dir]
g = runpy.run_path(COMPARE, run_name="__main__")

ad = g["ad"]
F, A = np.stack(rec["force"], 1), np.stack(rec["alive"], 1)              # (E, T, 6), (E, T)
lim = ad.model.actuator_forcerange[ad.act_ids, 1]
sat = np.abs(F) >= lim * (1 - 1e-4)
frac = {j: float(sat[..., k][A].mean()) for k, j in enumerate(S.ARM_JOINTS)}
p = os.path.join(out_dir, "compare.json")
J = json.load(open(p))
J["wrapper"] = "runs/film/so101_lift_74v/port/compare_servo.py"
J["servo"] = ad.servo
J["torque_saturation"] = dict(
    frac_per_joint=frac, forcerange_Nm=lim.round(4).tolist(),
    note="fraction of control steps (while the episode runs) at which |actuator_force| is at the force limit, sampled "
         "at the last of each control step's 4 physics steps (MuJoCo Warp actuator_force)")
json.dump(J, open(p, "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print("servo:", servo, "torque-saturated fraction:", {k: round(v, 3) for k, v in frac.items()})

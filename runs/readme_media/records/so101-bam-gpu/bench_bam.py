"""so101-bam-gpu: cost per physics step of "7v4_bam" (BAM friction kernel as a substep hook) against "7v4_real" on
MetalSim, as RATIOS only (this Mac is not a timing machine; no absolute time is printed or written).

Three BatchSims of the film scene, built as SO101Lift builds them (substeps 4, njmax 512), N worlds from reset_draw's
seeded states, every world commanded to its start pose + 0.3 rad at shoulder_lift and elbow_flex (a move, then a hold
against gravity; contacts as they come):
  real   : 7v4_real, per_world_fields (body_mass, geom_friction)
  bam    : 7v4_bam, + dof_frictionloss per world, BamFrictionWarp hook after every substep
  frozen : 7v4_bam, + dof_frictionloss per world, no hook (the budget stays at rest): the physics change alone
Graph replays of 50 step() calls (200 physics steps) then one synchronize, interleaved real / bam / frozen, 7 rounds;
ratio of medians. Writes bench/ratios.json.
"""
import json
import os
import sys
import time

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import patched  # noqa: E402,F401

import warp as wp  # noqa: E402

import metalsim.render.film.so101 as S  # noqa: E402
from metalsim.physics.batch import BatchSim, BatchSimOptions  # noqa: E402

wp.config.quiet = True
SUB, CALLS, ROUNDS = 4, 50, 7


def build(servo, n, hook):
    m = mujoco.MjModel.from_xml_path(os.path.join(S.ROOT, S.SCENE))
    ap = S.apply_servo(m, servo)
    pwf = ("body_mass", "geom_friction") + (("dof_frictionloss",) if "friction" in ap else ())
    sim = BatchSim(m, n, options=BatchSimOptions(substeps=SUB, njmax=512, per_world_fields=pwf))
    fh = S.friction_hook(ap, m, sim) if hook else None
    oid = lambda t, s: mujoco.mj_name2id(m, t, s)
    box_qadr = int(m.jnt_qposadr[oid(mujoco.mjtObj.mjOBJ_JOINT, "box")])
    arm_qadr = np.array([int(m.jnt_qposadr[oid(mujoco.mjtObj.mjOBJ_JOINT, j)]) for j in S.ARM_JOINTS])
    q = np.stack([S.reset_draw(10_000 + e, m.qpos0.copy(), box_qadr, arm_qadr, True)[0] for e in range(n)])
    sim.reset(); sim.synchronize()
    sim.set_state(q, np.zeros((n, m.nv)))
    if fh is not None:
        fh.update()
    c = np.zeros((n, m.nu), np.float32)
    c[:, ap["actuators"]] = q[:, arm_qadr]
    c[:, ap["actuators"][1]] += 0.3
    c[:, ap["actuators"][2]] += 0.3
    with wp.ScopedDevice(sim.device):
        sim.d.ctrl.assign(c)
    for _ in range(10):                                   # warm-up (graph already captured and compiled)
        sim.step()
    sim.synchronize()
    return sim


def run(sim):
    sim.synchronize()
    t0 = time.perf_counter()
    for _ in range(CALLS):
        sim.step()
    sim.synchronize()
    return (time.perf_counter() - t0) / (CALLS * SUB)


out = dict(substeps=SUB, calls=CALLS, rounds=ROUNDS, note="ratios only; this Mac is not a timing machine")
for n in (100, 1024):
    sims = dict(real=build("7v4_real", n, False), bam=build("7v4_bam", n, True), frozen=build("7v4_bam", n, False))
    t = {k: [] for k in sims}
    for _ in range(ROUNDS):
        for k, s in sims.items():
            t[k].append(run(s))
    med = {k: float(np.median(v)) for k, v in t.items()}
    spread = {k: float((max(v) - min(v)) / np.median(v)) for k, v in t.items()}
    r = dict(bam_over_real=med["bam"] / med["real"], frozen_over_real=med["frozen"] / med["real"],
             bam_over_frozen=med["bam"] / med["frozen"], relative_spread=spread,
             overflow={k: s.overflow_flags() for k, s in sims.items()})
    out[f"n{n}"] = r
    print(n, json.dumps(r), flush=True)
    del sims
os.makedirs(os.path.join(HERE, "bench"), exist_ok=True)
json.dump(out, open(os.path.join(HERE, "bench", "ratios.json"), "w"), indent=1)

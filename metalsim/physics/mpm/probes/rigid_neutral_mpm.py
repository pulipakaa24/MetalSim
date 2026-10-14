"""Rigid-body neutrality of the MPM layer: does a scene without MPM run the same kernels and produce the same
trajectory when an MPM scene was built and stepped earlier in the same process?

    python rigid_neutral_mpm.py --mode plain --json A.json      # MuJoCo Warp only
    python rigid_neutral_mpm.py --mode after-mpm --json B.json  # an MPMCoupling scene first, then the same rigid run
    python rigid_neutral_mpm.py --compare A.json A2.json B.json  # A vs A2: run-to-run baseline; A vs B: neutrality

Rigid scenes: a free box tumbling on a floor (contacts) and, when MuJoCo Menagerie is present ($MENAGERIE, default
<repo>/upstream/mujoco_menagerie), the Franka panda with
seeded random controls; N = 4 worlds, 200 steps, captured graphs (BatchSim defaults). Records a hash of qpos / qvel
every 20 steps and the hash of every loaded MuJoCo Warp kernel module (code + compile options).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4]))

import numpy as np  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[4]
BOX = """<mujoco><worldbody><geom type="plane" size="0 0 1"/>
<body pos="0 0 0.3" euler="20 30 10"><freejoint/><geom type="box" size=".05 .07 .04"/></body></worldbody></mujoco>"""


def scenes():
    import mujoco

    out = [("box", mujoco.MjModel.from_xml_string(BOX))]
    menagerie = pathlib.Path(os.environ.get("MENAGERIE", ROOT / "upstream" / "mujoco_menagerie"))
    cand = menagerie / "franka_emika_panda" / "scene.xml"
    if cand.exists():
        out.append(("panda", mujoco.MjModel.from_xml_path(str(cand))))
    return out


def rigid_run(model, n=4, steps=200):
    from metalsim.physics.batch import BatchSim

    sim = BatchSim(model, n)
    rng = np.random.default_rng(0)
    h = hashlib.sha256()
    for s in range(steps):
        if model.nu:
            c = rng.uniform(-1, 1, (n, model.nu)).astype(np.float32)
            lo, hi = model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
            c = np.where(model.actuator_ctrllimited.astype(bool), lo + 0.5 * (c + 1) * (hi - lo), c).astype(np.float32)
            sim.d.ctrl.assign(c)
        sim.step()
        if s % 20 == 19:
            h.update(sim.d.qpos.numpy().tobytes())
            h.update(sim.d.qvel.numpy().tobytes())
    return h.hexdigest()[:16]


def module_hashes():
    from warp._src import context

    out = {}
    for name, m in sorted(context.user_modules.items()):
        if not name.startswith("mujoco_warp") or not m.execs:
            continue
        for (_ctx, bd) in sorted(m.execs, key=lambda k: str(k)):
            out[f"{name}@{bd}"] = m.get_module_hash(bd).hex()[:16]
    return out


def mpm_first():
    import mujoco

    from metalsim.physics.batch import BatchSim
    from metalsim.physics.mpm.coupled import SAND, GridEmitter, MPMConfig, MPMCoupling

    m = mujoco.MjModel.from_xml_string(BOX)
    sim = BatchSim(m, 2)
    mpm = MPMCoupling(sim, MPMConfig(voxel_size=0.03, max_iterations=20))
    mpm.add_emitter(GridEmitter(lower=(-0.1, -0.1, 0.0), upper=(0.1, 0.1, 0.05), material=SAND))
    mpm.finalize()
    for _ in range(5):
        mpm.step()
    sim.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["plain", "after-mpm"])
    ap.add_argument("--json")
    ap.add_argument("--compare", nargs="+")
    a = ap.parse_args()
    if a.compare:
        rs = [json.load(open(p)) for p in a.compare]
        base = rs[0]
        ok = True
        for p, r in zip(a.compare[1:], rs[1:]):
            traj = {k: base["traj"][k] == r["traj"].get(k) for k in base["traj"]}
            common = set(base["modules"]) & set(r["modules"])
            mods = [k for k in common if base["modules"][k] != r["modules"][k]]
            print(f"{a.compare[0]} vs {p} ({r['mode']}): trajectories equal {traj}; module hashes compared "
                  f"{len(common)}, differing {mods}")
            ok &= not mods
        sys.exit(0 if ok else 4)
    import warp as wp

    wp.config.quiet = True
    if a.mode == "after-mpm":
        mpm_first()
    res = {"mode": a.mode, "traj": {name: rigid_run(m) for name, m in scenes()}, "modules": module_hashes(),
           "warp_config": {k: repr(getattr(wp.config, k)) for k in dir(wp.config) if not k.startswith("_")},
           "pid": os.getpid()}
    print(json.dumps({k: v for k, v in res.items() if k != "warp_config"}))
    json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()

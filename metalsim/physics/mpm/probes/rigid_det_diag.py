"""Where does a deterministic BatchSim stop repeating? Per-step, per-world hashes of a rigid scene, two runs in one
process (and a JSON for cross-process comparison).

    python rigid_det_diag.py --scene stack --worlds 256 --steps 150 [--deterministic 1] [--json out.json]
    python rigid_det_diag.py --compare A.json B.json

Scenes: ``box`` (one box tumbling onto the floor), ``stack`` (two boxes landing on each other and on the floor: the
case tests/test_batch_deterministic.py failed at 256 worlds in batch_02), ``panda`` (Menagerie Franka, random
controls).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4]))

import numpy as np  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[4]
BOX = """<mujoco><worldbody><geom type="plane" size="0 0 1"/>
<body pos="0 0 0.3" euler="20 30 10"><freejoint/><geom type="box" size=".05 .07 .04"/></body></worldbody></mujoco>"""
STACK = """<mujoco><worldbody><geom type="plane" size="0 0 1"/>
<body pos="0 0 0.3" euler="20 30 10"><freejoint/><geom type="box" size=".05 .07 .04"/></body>
<body pos="0.02 0 0.5" euler="10 0 40"><freejoint/><geom type="box" size=".04 .04 .04"/></body></worldbody></mujoco>"""


def model_of(scene):
    import mujoco

    if scene == "box":
        return mujoco.MjModel.from_xml_string(BOX)
    if scene == "stack":
        return mujoco.MjModel.from_xml_string(STACK)
    for cand in (ROOT / "upstream/mujoco_menagerie/franka_emika_panda/scene.xml",
                 ROOT.parent.parent.parent / "upstream/mujoco_menagerie/franka_emika_panda/scene.xml"):
        if cand.exists():
            return mujoco.MjModel.from_xml_path(str(cand))
    raise FileNotFoundError("MuJoCo Menagerie panda")


def run(model, n, steps, deterministic, capture):
    from metalsim.physics.batch import BatchSim, BatchSimOptions

    sim = BatchSim(model, n, options=BatchSimOptions(deterministic=deterministic, capture=capture))
    rng = np.random.default_rng(0)
    per_step, per_world, ncon = [], [], []
    for _ in range(steps):
        if model.nu:
            c = rng.uniform(-1, 1, (n, model.nu)).astype(np.float32)
            lo, hi = model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
            sim.d.ctrl.assign(np.where(model.actuator_ctrllimited.astype(bool), lo + 0.5 * (c + 1) * (hi - lo), c)
                              .astype(np.float32))
        sim.step()
        q = np.concatenate([sim.d.qpos.numpy(), sim.d.qvel.numpy()], 1)
        per_step.append(hashlib.sha256(q.tobytes()).hexdigest()[:16])
        per_world.append([hashlib.sha256(q[w].tobytes()).hexdigest()[:8] for w in range(n)])
        ncon.append(int(sim.d.nacon.numpy()[0]) if hasattr(sim.d, "nacon") else -1)
    sim.check_overflow()
    return {"per_step": per_step, "per_world": per_world, "ncon": ncon}


def first_diff(a, b):
    i = next((k for k in range(len(a["per_step"])) if a["per_step"][k] != b["per_step"][k]), None)
    if i is None:
        return None
    worlds = [w for w in range(len(a["per_world"][i])) if a["per_world"][i][w] != b["per_world"][i][w]]
    return {"step": i, "worlds_differing": len(worlds), "first_worlds": worlds[:8], "ncon_a": a["ncon"][i],
            "ncon_b": b["ncon"][i]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="stack")
    ap.add_argument("--worlds", type=int, default=256)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--deterministic", type=int, default=1)
    ap.add_argument("--capture", type=int, default=1)
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--json")
    ap.add_argument("--compare", nargs=2)
    a = ap.parse_args()
    if a.compare:
        A, B = (json.load(open(p)) for p in a.compare)
        d = first_diff(A["runs"][0], B["runs"][0])
        print(json.dumps({"cross_process_first_diff": d}))
        sys.exit(0 if d is None else 3)
    import warp as wp

    wp.config.quiet = True
    m = model_of(a.scene)
    runs = [run(m, a.worlds, a.steps, bool(a.deterministic), bool(a.capture)) for _ in range(a.runs)]
    res = {"scene": a.scene, "worlds": a.worlds, "deterministic": a.deterministic, "capture": a.capture,
           "in_process_first_diff": [first_diff(runs[0], r) for r in runs[1:]], "runs": runs[:1],
           "warp_cache": wp.config.kernel_cache_dir}
    print(json.dumps({k: v for k, v in res.items() if k != "runs"}))
    if a.json:
        json.dump(res, open(a.json, "w"))
    sys.exit(0 if all(d is None for d in res["in_process_first_diff"]) else 3)


if __name__ == "__main__":
    main()

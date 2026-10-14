"""Contact-slot audit of a deterministic BatchSim around the step where two stacked boxes first touch.

Before each audited step every contact slot's worldid is set to -1 (sentinel); after the step the probe counts, per
world, the box-box and box-floor contacts written into [0, nacon), and the slots below nacon still holding the sentinel
(reserved but never written: the signature of a phase-0 / phase-1 counter mismatch in Warp's two-phase deterministic
counters). Two runs; the per-world counts and the holes are compared.

    python contact_audit.py [--worlds 256] [--first 130] [--last 145] [--nativeccd-off] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4]))

import numpy as np  # noqa: E402

STACK = """<mujoco><worldbody><geom type="plane" size="0 0 1"/>
<body pos="0 0 0.3" euler="20 30 10"><freejoint/><geom type="box" size=".05 .07 .04"/></body>
<body pos="0.02 0 0.5" euler="10 0 40"><freejoint/><geom type="box" size=".04 .04 .04"/></body></worldbody></mujoco>"""


def audit(n, first, last, nativeccd_off, deterministic):
    import mujoco

    from metalsim.physics.batch import BatchSim, BatchSimOptions

    m = mujoco.MjModel.from_xml_string(STACK)
    if nativeccd_off:
        m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD)
    sim = BatchSim(m, n, options=BatchSimOptions(deterministic=deterministic, capture=False))
    out = []
    for s in range(last + 1):
        if s >= first:
            sim.d.contact.worldid.fill_(-1)
        sim.step()
        if s < first:
            continue
        nacon = int(sim.d.nacon.numpy()[0])
        wid = sim.d.contact.worldid.numpy()[:nacon]
        geom = sim.d.contact.geom.numpy()[:nacon]
        holes = int((wid < 0).sum())
        ok = wid >= 0
        bb = ok & (geom[:, 0] > 0) & (geom[:, 1] > 0)
        bf = ok & ~bb
        bb_w = np.bincount(wid[bb], minlength=n)
        bf_w = np.bincount(wid[bf], minlength=n)
        q = np.concatenate([sim.d.qpos.numpy(), sim.d.qvel.numpy()], 1)
        out.append({"step": s, "nacon": nacon, "holes": holes,
                    "bb_counts": np.bincount(bb_w).tolist(), "bf_counts": np.bincount(bf_w).tolist(),
                    "bb_per_world": bb_w.tolist(), "bf_per_world": bf_w.tolist(),
                    "world_hash": [hash(q[w].tobytes()) & 0xFFFFFFFF for w in range(n)]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", type=int, default=256)
    ap.add_argument("--first", type=int, default=130)
    ap.add_argument("--last", type=int, default=145)
    ap.add_argument("--nativeccd-off", action="store_true")
    ap.add_argument("--deterministic", type=int, default=1)
    ap.add_argument("--json")
    a = ap.parse_args()
    import warp as wp

    wp.config.quiet = True
    runs = [audit(a.worlds, a.first, a.last, a.nativeccd_off, bool(a.deterministic)) for _ in range(2)]
    summary = []
    for x, y in zip(*runs):
        diff_w = [w for w in range(a.worlds) if x["world_hash"][w] != y["world_hash"][w]]
        bbdiff = [w for w in range(a.worlds) if x["bb_per_world"][w] != y["bb_per_world"][w]]
        summary.append({"step": x["step"], "nacon": (x["nacon"], y["nacon"]), "holes": (x["holes"], y["holes"]),
                        "bb_count_hist": (x["bb_counts"], y["bb_counts"]), "worlds_state_differ": diff_w[:8],
                        "n_worlds_state_differ": len(diff_w), "worlds_bb_count_differ": bbdiff[:8]})
    for r in summary:
        print(json.dumps(r))
    if a.json:
        json.dump({"summary": summary, "runs": runs}, open(a.json, "w"))


if __name__ == "__main__":
    main()

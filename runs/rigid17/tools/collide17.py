"""Franka-Pour rigid collide on the Newton 1.7 line (fork branches metalsim / metalsim-hashtable-metal).

scale_pour.py --part rigid needs SolverMuJoCo, which does not build against the MuJoCo Warp fork 3.14 on the 1.7 line
(contact_force_fn signature), so this probe reuses scale_pour's scene and times CollisionPipeline.collide alone.

    collide17.py --worlds N [--det] [--state-out F.npz | --state-in F.npz] [--settle 60] [--steps 20] [--json F]

--state-out: settle the scene with SolverXPBD for --settle ticks (collide + 3 substeps) and save body_q / body_qd, so
every tree is timed and hashed at the same, in-contact state (--state-in). The contact hash covers every contact
array up to rigid_contact_count after one collide (meaningful for bitwise comparisons with --det only).
"""
import argparse
import hashlib
import json
import os
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--worlds", type=int, default=64)
ap.add_argument("--det", action="store_true")
ap.add_argument("--state-in")
ap.add_argument("--state-out")
ap.add_argument("--settle", type=int, default=60)
ap.add_argument("--steps", type=int, default=20)
ap.add_argument("--warmup", type=int, default=3)
ap.add_argument("--max-triangle-pairs", type=int, default=None)
ap.add_argument("--json")
ap.add_argument("--kprof", action="store_true", help="per-kernel GPU time of 4 collides (set WP_METAL_PROFILE=1)")
a = ap.parse_args()

SP = os.path.join(os.path.dirname(__file__), "../../../metalsim/physics/mpm/probes")
sys.argv = ["scale_pour.py", "--part", "rigid", "--worlds", str(a.worlds)]
sys.path.insert(0, os.path.abspath(SP))
import numpy as np  # noqa: E402

import scale_pour as sp  # noqa: E402  (parses the argv above; runs nothing at import)

wp, newton = sp.wp, sp.newton
DEV = sp.DEV
res = {"worlds": a.worlds, "det": a.det, "newton": newton.__file__, "warp": wp.__version__}
with wp.ScopedDevice(DEV):
    model, src, tgt, bpw, build_s = sp.build(a.worlds, robot=True, media=False, source_mass=0.05)
    kw = {} if a.max_triangle_pairs is None else {"max_triangle_pairs": a.max_triangle_pairs}
    pipeline = newton.CollisionPipeline(model, deterministic=a.det, **kw)
    contacts = pipeline.contacts()
    s0, s1 = model.state(), model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    res["rigid_contact_max"] = int(contacts.rigid_contact_max)
    if a.state_out:
        solver = newton.solvers.SolverXPBD(model, iterations=4)
        control = model.control()
        for _ in range(a.settle):
            pipeline.collide(s0, contacts)
            for _ in range(3):
                s0.clear_forces()
                solver.step(s0, s1, control, contacts, sp.RIGID_DT)
                s0, s1 = s1, s0
        np.savez(a.state_out, body_q=s0.body_q.numpy(), body_qd=s0.body_qd.numpy())
        res["finite"] = bool(np.isfinite(s0.body_q.numpy()).all())
    elif a.state_in:
        z = np.load(a.state_in)
        s0.body_q.assign(z["body_q"])
        s0.body_qd.assign(z["body_qd"])

    def collide():
        pipeline.collide(s0, contacts)

    for _ in range(a.warmup):
        collide()
    sp.sync()
    n = int(contacts.rigid_contact_count.numpy()[0])
    h = hashlib.sha256()
    for name in ("rigid_contact_shape0", "rigid_contact_shape1", "rigid_contact_point0", "rigid_contact_point1",
                 "rigid_contact_offset0", "rigid_contact_offset1", "rigid_contact_normal", "rigid_contact_margin0",
                 "rigid_contact_margin1"):
        arr = getattr(contacts, name, None)
        if arr is not None:
            h.update(arr.numpy()[: min(n, arr.shape[0])].tobytes())
    res["rigid_contacts"] = n
    res["contact_sha"] = h.hexdigest()[:16]
    ts = sp.timed_loop(collide, a.steps)
    res["collide_ms"] = sp.stats(ts)
    # a second hash after the timed collides: the same state, so it must not change (repeatability within a run)
    h2 = hashlib.sha256()
    for name in ("rigid_contact_shape0", "rigid_contact_shape1", "rigid_contact_point0", "rigid_contact_point1",
                 "rigid_contact_offset0", "rigid_contact_offset1", "rigid_contact_normal", "rigid_contact_margin0",
                 "rigid_contact_margin1"):
        arr = getattr(contacts, name, None)
        if arr is not None:
            h2.update(arr.numpy()[: min(n, arr.shape[0])].tobytes())
    res["contact_sha_after"] = h2.hexdigest()[:16]
    if a.kprof:
        sp.args.kprof_ticks = 4
        sp.kprof("collide", {"collide": collide})
        res["kprof"] = sp.RESULT["kprof"]
print(json.dumps(res, indent=1))
if a.json:
    with open(a.json, "w") as fh:
        json.dump(res, fh, indent=1)

"""Two-way coupling's ordered body-force sum: compacted (fork) vs full-scan lanes, on synthetic data.

Loads the kernels from the given ``example_mpm_twoway_coupling.py`` (Newton fork, with the compaction) and checks on
``--device``: the compaction keeps exactly the body nodes in increasing order; the per-body forces of the compacted
path equal the full-scan path to float32 rounding (relative 1e-5) and the float64 reference; two calls are bitwise
equal; and times both paths (median of ``--reps`` synchronized calls, after warm-up).

    PYTHONPATH=<warp>:<newton> python body_force_compact_check.py EXAMPLE_PY [--device metal:0] [--nodes 655360]
"""
import argparse
import importlib.util
import sys
import time

import numpy as np
import warp as wp

ap = argparse.ArgumentParser()
ap.add_argument("example")
ap.add_argument("--device", default="metal:0")
ap.add_argument("--nodes", type=int, default=655360)
ap.add_argument("--bodies", type=int, default=12)
ap.add_argument("--contact", type=int, default=6000, help="nodes in contact with some body")
ap.add_argument("--reps", type=int, default=50)
args = ap.parse_args()
wp.config.quiet = True
wp.init()
spec = importlib.util.spec_from_file_location("ex2w", args.example)
ex = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ex)
from warp.utils import array_scan  # noqa: E402

rng = np.random.default_rng(0)
N, B, L = args.nodes, args.bodies, 256
ncoll = B + 3                                     # colliders: one per body, plus 3 static ones (body -1)
body_ids_np = np.concatenate([np.arange(B), -np.ones(3, int)]).astype(np.int32)
cid = -np.ones(N, np.int32)
contact = rng.choice(N, size=args.contact, replace=False)
cid[contact] = rng.integers(0, ncoll, size=args.contact)
imp = rng.standard_normal((N, 3)).astype(np.float32)
pos = rng.standard_normal((N, 3)).astype(np.float32)
bq = np.zeros((B, 7), np.float32); bq[:, 6] = 1; bq[:, :3] = rng.standard_normal((B, 3))
com = rng.standard_normal((B, 3)).astype(np.float32) * 0.1
dt = 0.01
ok = True
with wp.ScopedDevice(args.device):
    d = dict(cid=wp.array(cid, dtype=int), bids=wp.array(body_ids_np, dtype=int), imp=wp.array(imp, dtype=wp.vec3),
             pos=wp.array(pos, dtype=wp.vec3), bq=wp.array(bq, dtype=wp.transform), com=wp.array(com, dtype=wp.vec3))
    lanes = wp.zeros((B, L), dtype=wp.spatial_vector)
    flags, positions, nodes = (wp.zeros(N, dtype=int) for _ in range(3))
    count = wp.zeros(1, dtype=int)

    def full():
        f = wp.zeros(B, dtype=wp.spatial_vector)
        wp.launch(ex.compute_body_forces_lanes, dim=(B, L), inputs=[dt, N, d["cid"], d["imp"], d["pos"], d["bids"], d["bq"], d["com"], lanes])
        wp.launch(ex.sum_body_force_lanes, dim=B, inputs=[lanes, f])
        return f

    def compact():
        f = wp.zeros(B, dtype=wp.spatial_vector)
        wp.launch(ex.flag_body_nodes, dim=N, inputs=[N, d["cid"], d["bids"], flags])
        array_scan(flags, positions, inclusive=False)
        wp.launch(ex.compact_body_nodes, dim=N, inputs=[N, flags, positions, nodes, count])
        wp.launch(ex.compute_body_forces_compact_lanes, dim=(B, L),
                  inputs=[dt, count, nodes, d["cid"], d["imp"], d["pos"], d["bids"], d["bq"], d["com"], lanes])
        wp.launch(ex.sum_body_force_lanes, dim=B, inputs=[lanes, f])
        return f

    a, c1, c2 = full().numpy(), compact().numpy(), compact().numpy()
    keep = np.nonzero((cid >= 0) & (body_ids_np[np.clip(cid, 0, None)] >= 0))[0]
    n = int(count.numpy()[0])
    comp_ok = n == keep.size and np.array_equal(nodes.numpy()[:n], keep)
    # float64 reference
    ref = np.zeros((B, 6))
    for i in keep:
        b = body_ids_np[cid[i]]
        fw = imp[i].astype(np.float64) / dt
        r = pos[i].astype(np.float64) - (bq[b, :3].astype(np.float64) + com[b])
        ref[b, :3] += fw; ref[b, 3:] += np.cross(r, fw)
    scale = np.abs(ref).max()
    e_full, e_comp = np.abs(a - ref).max() / scale, np.abs(c1 - ref).max() / scale
    e_pair = np.abs(a - c1).max() / scale
    rep = np.array_equal(c1, c2)
    ok = comp_ok and rep and e_comp < 1e-5 and e_pair < 1e-5
    print(f"compaction exact {comp_ok} ({n} nodes); vs float64: full {e_full:.2e}, compact {e_comp:.2e}; "
          f"full vs compact {e_pair:.2e}; compact repeat bitwise {rep}; bitwise equal to full {np.array_equal(a, c1)}")
    for name, fn in (("full", full), ("compact", compact)):
        for _ in range(5):
            fn()
        wp.synchronize_device()
        ts = []
        for _ in range(args.reps):
            t0 = time.perf_counter(); fn(); wp.synchronize_device(); ts.append((time.perf_counter() - t0) * 1e3)
        print(f"{name}: median {np.median(ts):.3f} ms per call ({args.device}, {N} nodes, {B} bodies)")
print("OK" if ok else "FAIL")
sys.exit(0 if ok else 1)

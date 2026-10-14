"""SolverImplicitMPM's ordered proxy-wrench harvest on Metal: compacted (fork) vs full world-range scan, synthetic data.

Uses the kernels of the imported Newton (fork with ``_harvest_mpm_proxy_wrench_compact_lanes_kernel``). Builds
``--worlds`` environment-first collider-node ranges with ``--bodies`` proxy bodies per world (plus non-proxy and
static colliders) and checks: the per-body lane partials folded in lane order equal the full scan to float32 rounding
(relative 1e-5) and a float64 reference; two calls repeat bitwise; the single-world case (empty offsets) too. Times
both paths (median of ``--reps`` synchronized calls).

    PYTHONPATH=<warp>:<newton> python harvest_compact_check.py [--device metal:0] [--worlds 1] [--bodies 16]
"""
import argparse
import sys
import time

import numpy as np
import warp as wp

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="metal:0")
ap.add_argument("--worlds", type=int, default=1)
ap.add_argument("--bodies", type=int, default=16, help="proxy bodies per world")
ap.add_argument("--nodes-per-world", type=int, default=150000)
ap.add_argument("--contact-per-world", type=int, default=4000)
ap.add_argument("--reps", type=int, default=30)
args = ap.parse_args()
wp.config.quiet = True
wp.init()
import newton  # noqa: E402
from newton._src.solvers.implicit_mpm import solver_implicit_mpm as S  # noqa: E402
from warp.utils import array_scan  # noqa: E402

rng = np.random.default_rng(1)
W, Bw, Nw = args.worlds, args.bodies, args.nodes_per_world
B = W * (Bw + 1)                              # per world: Bw proxy bodies + 1 non-proxy dynamic body
L = S._HARVEST_LANES
proxy_flag = int(newton.BodyFlags.PROXY)
body_world = np.repeat(np.arange(W), Bw + 1).astype(np.int32)
body_flags = np.zeros(B, np.int32)
b2p = -np.ones(B, np.int32)
for w in range(W):
    for j in range(Bw):
        b = w * (Bw + 1) + j
        body_flags[b] = proxy_flag
        b2p[b] = w * Bw + j
ncol = B + 2                                   # colliders: one per body, two static (body -1)
collider_body = np.concatenate([np.arange(B), [-1, -1]]).astype(np.int32)
N = W * Nw
cid = -np.ones(N, np.int32)
for w in range(W):
    idx = w * Nw + rng.choice(Nw, size=args.contact_per_world, replace=False)
    choices = np.concatenate([w * (Bw + 1) + np.arange(Bw + 1), [B, B + 1]])
    cid[idx] = rng.choice(choices, size=idx.size)
offsets_np = (np.arange(W + 1) * Nw).astype(np.int32)
imp = rng.standard_normal((N, 3)).astype(np.float32)
pos = rng.standard_normal((N, 3)).astype(np.float32)
bq = np.zeros((B, 7), np.float32); bq[:, 6] = 1; bq[:, :3] = rng.standard_normal((B, 3))
com = (rng.standard_normal((B, 3)) * 0.1).astype(np.float32)
dt = 0.01
ok = True
with wp.ScopedDevice(args.device):
    a = lambda x, t: wp.array(x, dtype=t)  # noqa: E731
    d = dict(cid=a(cid, int), cb=a(collider_body, int), b2p=a(b2p, int), bf=a(body_flags, wp.int32), bw=a(body_world, wp.int32),
             imp=a(imp, wp.vec3), pos=a(pos, wp.vec3), bq=a(bq, wp.transform), com=a(com, wp.vec3))
    partial = wp.zeros((B, L), dtype=wp.spatial_vector)
    flags, npos = wp.zeros(N + 1, dtype=int), wp.zeros(N + 1, dtype=int)
    nodes = wp.zeros(N, dtype=int)

    def fold():
        out = wp.zeros(W * Bw, dtype=wp.spatial_vector)
        wp.launch(S._sum_mpm_proxy_wrench_lanes_kernel, dim=B, inputs=[partial, d["b2p"], proxy_flag, d["bf"]], outputs=[out])
        return out.numpy()

    def full(offs):
        wp.launch(S._harvest_mpm_proxy_wrench_lanes_kernel, dim=(B, L), inputs=[dt, L, offs, d["cid"], d["imp"], d["pos"], d["cb"],
                  d["b2p"], proxy_flag, d["bf"], d["bw"], d["com"], d["bq"], partial])
        return fold()

    def compact(offs):
        wp.launch(S._flag_proxy_harvest_nodes_kernel, dim=N + 1, inputs=[offs, d["cid"], d["cb"], d["b2p"], proxy_flag, d["bf"], flags])
        array_scan(flags, npos, inclusive=False)
        wp.launch(S._compact_proxy_harvest_nodes_kernel, dim=N, inputs=[flags, npos, nodes])
        wp.launch(S._harvest_mpm_proxy_wrench_compact_lanes_kernel, dim=(B, L), inputs=[dt, L, offs, npos, nodes, d["cid"], d["imp"],
                  d["pos"], d["cb"], d["b2p"], proxy_flag, d["bf"], d["bw"], d["com"], d["bq"], partial])
        return fold()

    # float64 reference (world ranges are exact here: every node of world w carries only world-w colliders)
    ref = np.zeros((W * Bw, 6))
    for i in np.nonzero(cid >= 0)[0]:
        b = collider_body[cid[i]]
        if b < 0 or b2p[b] < 0:
            continue
        fw = imp[i].astype(np.float64) / dt
        r = pos[i].astype(np.float64) - (bq[b, :3].astype(np.float64) + com[b])
        ref[b2p[b], :3] += fw; ref[b2p[b], 3:] += np.cross(r, fw)
    scale = np.abs(ref).max()
    for label, offs in (("world ranges", a(offsets_np, int)), ("no ranges (every body scans every node)", wp.empty(0, dtype=int))):
        f, c1, c2 = full(offs), compact(offs), compact(offs)
        e_f, e_c, e_fc = (np.abs(x - y).max() / scale for x, y in ((f, ref), (c1, ref), (f, c1)))
        rep = np.array_equal(c1, c2)
        good = rep and e_c < 1e-5 and e_fc < 1e-5
        ok &= good
        print(f"{label}: vs float64 full {e_f:.2e} compact {e_c:.2e}; full vs compact {e_fc:.2e}; repeat bitwise {rep} -> {'ok' if good else 'FAIL'}")
        for name, fn in (("full", full), ("compact", compact)):
            for _ in range(3):
                fn(offs)
            wp.synchronize_device(); ts = []
            for _ in range(args.reps):
                t0 = time.perf_counter(); fn(offs); wp.synchronize_device(); ts.append((time.perf_counter() - t0) * 1e3)
            print(f"   {name}: {np.median(ts):.3f} ms per call ({W} worlds x {Bw} proxy bodies, {N} nodes)")
print("OK" if ok else "FAIL")
sys.exit(0 if ok else 1)

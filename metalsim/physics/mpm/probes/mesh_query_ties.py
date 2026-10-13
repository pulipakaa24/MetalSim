"""Closest-face queries on box meshes (Newton's MPM collider meshes) at grid points that include exact ties (points
equidistant from two or three faces). The face chosen at a tie depends on the BVH (tree layout and traversal), so the
output fingerprints a device's BVH: CUDA builds an LBVH by default, a CPU tree uses SAH.

    python mesh_query_ties.py --device cuda:0|metal:0|cpu [--constructor lbvh|sah|median] --out FILE.npz
"""
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--device", required=True)
ap.add_argument("--constructor", default=None)
ap.add_argument("--leaf-size", type=int, default=None, help="BVH leaf size (64: one leaf, no tree)")
ap.add_argument("--out", required=True)
args = ap.parse_args()

import numpy as np
import warp as wp

wp.config.quiet = True
wp.init()


def box_mesh(center, half):
    c, h = np.asarray(center, np.float32), np.asarray(half, np.float32)
    corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], np.float32) * h + c
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    tris = []
    for a, b, cc, d in quads:
        tris += [(a, b, cc), (a, cc, d)]
    return corners, np.array(tris, np.int32).reshape(-1)


@wp.kernel
def query(mesh: wp.uint64, pts: wp.array[wp.vec3], face: wp.array[int], uv: wp.array[wp.vec2], sign: wp.array[float]):
    i = wp.tid()
    q = wp.mesh_query_point(mesh, pts[i], 10.0)
    face[i] = wp.where(q.result, q.face, -1)
    uv[i] = wp.vec2(q.u, q.v)
    sign[i] = q.sign


# boxes: Newton's granular cube collider, the column scene's walls, a unit cube; a lattice of points around each, on
# voxel-aligned positions so edge and corner ties occur
cases = [((0.75, 0.0, 0.8), (0.5, 2.0, 0.8), 0.05), ((-0.25, 0.05, 0.7), (0.25, 0.6, 0.7), 0.05 / 3.0),
         ((0.0, 0.0, 0.0), (0.5, 0.5, 0.5), 0.0625)]
out = {}
with wp.ScopedDevice(args.device):
    for k, (c, h, step) in enumerate(cases):
        pts_np, idx = box_mesh(c, h)
        kw = {} if args.constructor is None else {"bvh_constructor": args.constructor}
        if args.leaf_size:
            kw["bvh_leaf_size"] = args.leaf_size
        mesh = wp.Mesh(wp.array(pts_np, dtype=wp.vec3), wp.array(idx, dtype=int), **kw)
        lo, hi = np.asarray(c) - np.asarray(h) - 4 * step, np.asarray(c) + np.asarray(h) + 4 * step
        axes = [np.arange(lo[d], hi[d] + 1e-9, step, dtype=np.float64).astype(np.float32) for d in range(3)]
        g = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
        n = g.shape[0]
        face, uv, sign = wp.zeros(n, dtype=int), wp.zeros(n, dtype=wp.vec2), wp.zeros(n, dtype=float)
        wp.launch(query, n, inputs=[mesh.id, wp.array(g, dtype=wp.vec3), face, uv, sign])
        out[f"pts{k}"], out[f"face{k}"], out[f"uv{k}"], out[f"sign{k}"] = g, face.numpy(), uv.numpy(), sign.numpy()
np.savez_compressed(args.out, **out)
print("saved", args.out, {k: v.shape for k, v in out.items() if k.startswith("face")})

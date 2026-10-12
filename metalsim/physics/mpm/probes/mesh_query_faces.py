"""Per-face closest-point arithmetic for given points: for each triangle of a box mesh, the barycentrics returned by
``mesh_query_point`` on a one-triangle mesh, and the squared distance computed exactly as ``mesh_query_point``
compares faces (``c = u*p + v*q + w*r``, ``length_sq(c - point)``). Run on two devices to find which step rounds
differently at ties.

    python mesh_query_faces.py --device D --points FILE.npz --case K --out OUT.npz
"""
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--device", required=True)
ap.add_argument("--points", required=True, help="npz with pts{case} (mesh_query_ties.py output) and idx")
ap.add_argument("--case", type=int, default=0)
ap.add_argument("--out", required=True)
args = ap.parse_args()

import numpy as np
import warp as wp

wp.config.quiet = True
wp.init()

CASES = [((0.75, 0.0, 0.8), (0.5, 2.0, 0.8)), ((-0.25, 0.05, 0.7), (0.25, 0.6, 0.7)), ((0.0, 0.0, 0.0), (0.5, 0.5, 0.5))]


def box_mesh(center, half):
    c, h = np.asarray(center, np.float32), np.asarray(half, np.float32)
    corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], np.float32) * h + c
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    tris = []
    for a, b, cc, d in quads:
        tris += [(a, b, cc), (a, cc, d)]
    return corners, np.array(tris, np.int32)


@wp.kernel
def per_face(mesh: wp.uint64, p: wp.vec3, q: wp.vec3, r: wp.vec3, pts: wp.array[wp.vec3], uv: wp.array[wp.vec2],
             dist: wp.array[float]):
    i = wp.tid()
    res = wp.mesh_query_point_no_sign(mesh, pts[i], 10.0)
    u = res.u
    v = res.v
    w = 1.0 - u - v
    c = u * p + v * q + w * r
    uv[i] = wp.vec2(u, v)
    dist[i] = wp.length_sq(c - pts[i])


src = np.load(args.points)
pts_np = src["pts"] if "pts" in src.files else src[f"pts{args.case}"]
corners, tris = box_mesh(*CASES[args.case])
out = {"pts": pts_np}
with wp.ScopedDevice(args.device):
    P = wp.array(pts_np, dtype=wp.vec3)
    n = pts_np.shape[0]
    for f, (a, b, c) in enumerate(tris):
        mesh = wp.Mesh(wp.array(corners[[a, b, c]], dtype=wp.vec3), wp.array([0, 1, 2], dtype=int))
        uv, dist = wp.zeros(n, dtype=wp.vec2), wp.zeros(n, dtype=float)
        wp.launch(per_face, n, inputs=[mesh.id, wp.vec3(*corners[a]), wp.vec3(*corners[b]), wp.vec3(*corners[c]), P, uv,
                                       dist])
        out[f"uv{f}"], out[f"dist{f}"] = uv.numpy(), dist.numpy()
np.savez_compressed(args.out, **out)
print("saved", args.out, n, "points x", len(tris), "faces")

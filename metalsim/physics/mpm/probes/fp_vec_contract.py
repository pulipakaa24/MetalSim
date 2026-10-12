"""Intermediate values of mesh_query_point's face distance (c = u*p + v*q + w*r; length_sq(c - x)) on a device, plus
scalar spellings of the same arithmetic, to locate where Metal and CUDA fuse differently.

    python fp_vec_contract.py --device D --faces FACES.npz --points DIFFPTS.npz --out OUT.npz
"""
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--device", required=True)
ap.add_argument("--faces", required=True)
ap.add_argument("--points", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()

import numpy as np
import warp as wp

wp.config.quiet = True
wp.init()


@wp.kernel
def steps(pts: wp.array[wp.vec3], uvs: wp.array[wp.vec2], p: wp.vec3, q: wp.vec3, r: wp.vec3,
          c_out: wp.array[wp.vec3], d_out: wp.array[wp.vec3], l_out: wp.array[float], s_out: wp.array[wp.vec3]):
    i = wp.tid()
    u = uvs[i][0]
    v = uvs[i][1]
    w = 1.0 - u - v
    c = u * p + v * q + w * r
    d = c - pts[i]
    c_out[i] = c
    d_out[i] = d
    l_out[i] = wp.length_sq(d)
    # scalar spelling of the x component of c
    s_out[i] = wp.vec3(u * p[0] + v * q[0] + w * r[0], u * p[1] + v * q[1] + w * r[1], u * p[2] + v * q[2] + w * r[2])


F = np.load(args.faces)
P = np.load(args.points)["pts"]
corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], np.float32) * np.float32(
    [0.5, 2.0, 0.8]) + np.float32([0.75, 0.0, 0.8])
quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
tris = [t for a, b, c, d in quads for t in ((a, b, c), (a, c, d))]
out = {}
with wp.ScopedDevice(args.device):
    n = P.shape[0]
    X = wp.array(P, dtype=wp.vec3)
    for f in (0, 11):
        a, b, c = tris[f]
        co, do, so = (wp.zeros(n, dtype=wp.vec3) for _ in range(3))
        lo = wp.zeros(n, dtype=float)
        wp.launch(steps, n, inputs=[X, wp.array(F[f"uv{f}"], dtype=wp.vec2), wp.vec3(*corners[a]), wp.vec3(*corners[b]),
                                    wp.vec3(*corners[c]), co, do, lo, so])
        out[f"c{f}"], out[f"d{f}"], out[f"l{f}"], out[f"s{f}"] = co.numpy(), do.numpy(), lo.numpy(), so.numpy()
np.savez(args.out, **out)
print("saved", args.out)

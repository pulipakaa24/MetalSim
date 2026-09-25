"""Structured tetrahedral meshes (hexahedral grid, each cell split into 6 Freudenthal tetrahedra, positive volume) as USDA
files with one UsdGeom.TetMesh, for the Isaac Lab 3.0 deformable mesh-resolution sweep (Isaac Lab's automatic
tetrahedralization needs pytetwild, which aborts the Kit process on the parity VM; a pre-tetrahedralized TetMesh under
the prim is its documented alternative). Rod 0.5 x 0.02 x 0.02 m with n cells across and 2.5 n cells along (cell
aspect 2.5, as PhysX 5.1's 11x2x2 hexahedral rod at n = 1); cube 0.2 m with n cells per edge. Centred at the origin.
usage: make_tetmeshes.py OUTDIR"""
import itertools, os, sys
import numpy as np

PERMS = list(itertools.permutations(range(3)))


def grid_tets(size, cells):
    nx, ny, nz = cells
    xs = [np.linspace(-s / 2, s / 2, c + 1) for s, c in zip(size, cells)]
    idx = lambda i, j, k: (i * (ny + 1) + j) * (nz + 1) + k
    pts = np.array([[xs[0][i], xs[1][j], xs[2][k]] for i in range(nx + 1) for j in range(ny + 1) for k in range(nz + 1)])
    tets = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                for p in PERMS:   # path 000 -> 111 along axes in order p: one tetrahedron per permutation
                    c = np.zeros(3, int); vs = [idx(i, j, k)]
                    for ax in p:
                        c[ax] += 1; vs.append(idx(i + c[0], j + c[1], k + c[2]))
                    a, b, cc, d = (pts[v] for v in vs)
                    if np.dot(np.cross(b - a, cc - a), d - a) < 0:
                        vs[2], vs[3] = vs[3], vs[2]
                    tets.append(vs)
    return pts, np.array(tets, np.int32)


def write_usda(path, name, pts, tets):
    p = ", ".join(f"({x:.7g}, {y:.7g}, {z:.7g})" for x, y, z in pts)
    t = ", ".join(f"({a}, {b}, {c}, {d})" for a, b, c, d in tets)
    with open(path, "w") as f:
        f.write(f'''#usda 1.0
(
    defaultPrim = "{name}"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "{name}"
{{
    def TetMesh "sim_mesh"
    {{
        point3f[] points = [{p}]
        int4[] tetVertexIndices = [{t}]
    }}
}}
''')


if __name__ == "__main__":
    out = sys.argv[1]; os.makedirs(out, exist_ok=True)
    for n in (1, 2, 4, 8):
        pts, tets = grid_tets((0.5, 0.02, 0.02), (int(round(10 * n)), n, n))
        write_usda(os.path.join(out, f"rod_n{n}.usda"), "Rod", pts, tets); print(f"rod_n{n}", len(pts), "points", len(tets), "tets")
    for n in (2, 4, 8):
        pts, tets = grid_tets((0.2, 0.2, 0.2), (n, n, n))
        write_usda(os.path.join(out, f"cube_n{n}.usda"), "Cube", pts, tets); print(f"cube_n{n}", len(pts), "points", len(tets), "tets")

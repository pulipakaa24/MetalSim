"""so101-film-bam: how far the rendered jaws go into the box, per frame of a take (rec/rec.npz poses, rec/twin.xml):
every vertex of each mesh geom on the gripper and moving-jaw bodies (the visual STLs the renderer draws, and the
collision mesh) is put in the box geom's frame; depth = how far inside the 4 x 4 x 6 cm box it is (0 if outside).
Vertex-in-box is exact for the vertices; edges and faces crossing a box edge are not counted.

    .venv/bin/python runs/readme_media/scratch/so101-film-bam/jaw_depth.py <rec dir> [...]
"""
import os
import sys

import mujoco
import numpy as np

for rec in sys.argv[1:]:
    m = mujoco.MjModel.from_xml_path(os.path.join(rec, "twin.xml"))
    z = np.load(os.path.join(rec, "rec.npz"))
    X, P = z["xmat"][:, 0].astype(float), z["xpos"][:, 0].astype(float)
    box = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "box")
    half = m.geom_size[box]
    bodies = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in ("gripper", "moving_jaw_so101_v1")}
    geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] in bodies and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH]
    print(rec, "box half-size", half, "jaw mesh geoms", [(g, mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, m.geom_dataid[g]),
                                                       "visual" if m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0 else "collision") for g in geoms])
    worst = {}
    for g in geoms:
        mid = m.geom_dataid[g]
        v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].astype(float)
        depth = np.zeros(len(X))
        for f in range(len(X)):
            w = v @ X[f, g].reshape(3, 3).T + P[f, g]                      # world
            b = (w - P[f, box]) @ X[f, box].reshape(3, 3)                  # box frame
            inside = np.all(np.abs(b) < half, axis=1)
            if inside.any():
                depth[f] = float(np.min(half - np.abs(b[inside]), axis=1).max())
        worst[g] = depth
        kind = "visual" if m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0 else "collision"
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, mid)
        hold = depth[150:]
        print(f"  geom {g} {name} ({kind}): max depth {depth.max() * 1e3:.2f} mm at frame {int(depth.argmax())}; "
              f"hold (frames 150-299) median {np.median(hold) * 1e3:.2f} mm, max {hold.max() * 1e3:.2f} mm; "
              f"frames with any vertex inside {int((depth > 0).sum())}")

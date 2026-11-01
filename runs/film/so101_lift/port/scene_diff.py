"""Compare two compiled MJCF scenes field by field (every numeric MjModel array and the physics options).

Usage: <python> runs/film/so101_lift/port/scene_diff.py <scene_a.xml> <scene_b.xml>
Run with so101Sim's interpreter (MuJoCo 3.10.0, the version its env runs) for the so101_lift film's check: so101Sim's
mujoco_menagerie/robotstudio_so101/scene_box_rl.xml against MetalSim's assets/so101/scene_box_rl.xml.
"""
import sys

import mujoco
import numpy as np

a, b = (mujoco.MjModel.from_xml_path(p) for p in sys.argv[1:3])
print("mujoco", mujoco.__version__)
for k in ("nq", "nv", "nu", "nbody", "njnt", "ngeom", "nsite", "nmesh", "ncam", "ntex", "nmat", "nkey", "nlight"):
    va, vb = getattr(a, k), getattr(b, k)
    print(f"{k:8s} {va:6d} {vb:6d}" + ("" if va == vb else "   DIFFERS"))
for k in dir(a.opt):
    if k.startswith("_"):
        continue
    va, vb = getattr(a.opt, k), getattr(b.opt, k)
    if callable(va):
        continue
    if not np.array_equal(np.asarray(va), np.asarray(vb)):
        print(f"opt.{k}: {va} vs {vb}   DIFFERS")
diff = []
for k in dir(a):
    if k.startswith("_") or k.startswith(("name_", "cam_", "tex_", "mat_", "light_", "key_", "vis", "stat")):
        continue
    va = getattr(a, k)
    if not isinstance(va, np.ndarray) or va.dtype.kind not in "fiub":
        continue
    vb = getattr(b, k)
    if va.shape != vb.shape:
        diff.append(f"{k}: shape {va.shape} vs {vb.shape}")
    elif not np.array_equal(va, vb):
        idx = np.argwhere(va != vb)[:4].tolist()
        diff.append(f"{k}: {int((va != vb).sum())} entries differ, first {idx}")
print("\n".join(diff) if diff else "all numeric model arrays equal (names, cameras, textures, materials, lights, keys, visual and stat skipped)")

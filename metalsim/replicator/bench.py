"""Per-annotator GPU cost at a batch of envs (default 1024 x 128x128, the SO-101 lift scene).

    python -m metalsim.replicator.bench [--envs 1024] [--size 128] [--reps 10]

Each row: median wall time of the annotator over ``reps`` calls with ``torch.mps.synchronize()`` on
both sides (so it includes launch overhead), after one warm-up call, on top of an already-rendered
frame. ``render`` is the tier-0 frame itself (rgb + depth + id buffer + normals) for reference.
"""
from __future__ import annotations

import argparse
import os
import statistics
import time

import mujoco
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SO101 = os.path.join(ROOT, "assets", "so101", "scene_box_rl.xml")


def _time(fn, reps):
    fn(); torch.mps.synchronize()
    ts = []
    for _ in range(reps):
        torch.mps.synchronize(); t0 = time.perf_counter()
        fn()
        torch.mps.synchronize(); ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=1024)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--xml", default=SO101)
    a = ap.parse_args(argv)
    from metalsim.physics.batch import BatchSim, BatchSimOptions
    from metalsim.render.tier0 import Tier0Renderer, SEG_SLOT
    from metalsim.replicator import IsaacAnnotators, Semantics, State
    from metalsim.replicator import events as ev

    m = mujoco.MjModel.from_xml_path(a.xml)
    n = a.envs
    sim = BatchSim(m, n, options=BatchSimOptions(substeps=1))
    cam = "base_cam" if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "base_cam") >= 0 else 0
    r = Tier0Renderer(m, n, width=a.size, height=a.size, camera=cam, outputs=("rgb", "depth", "seg", "normal"),
                      seg_mode=SEG_SLOT, decimate_faces=2000)
    sem = Semantics.from_body_names(m)
    ann = IsaacAnnotators(r, sem, sim=sim, decimate_faces=2000)
    alt = IsaacAnnotators(r, sem, sim=sim, decimate_faces=2000, impl="torch")
    allv = IsaacAnnotators(r, sem, sim=sim, decimate_faces=2000, loose_hull=False)
    sim.synchronize()
    sim.forward(); sim.synchronize()

    def render():
        vr = r.render(sim, sim.event.value); r.after(vr)
    render(); torch.mps.synchronize(); r.ctx.synchronize()
    prev = State.from_sim(sim, clone=True)

    def rows_for(x):
        return [
            ("rgb (RGBA)", lambda: x.rgb()),
            ("distance_to_image_plane", lambda: x.distance_to_image_plane()),
            ("distance_to_camera", lambda: x.distance_to_camera()),
            ("normals (raster)", lambda: x.normals()),
            ("normals_from_depth", lambda: x.normals_from_depth()),
            ("semantic_segmentation", lambda: x.semantic_segmentation()),
            ("instance_segmentation", lambda: x.instance_segmentation()),
            ("instance_id_segmentation", lambda: x.instance_id_segmentation()),
            ("bounding_box_2d_tight", lambda: x.bounding_box_2d_tight(occlusion=False)),
            ("bounding_box_2d_loose (projected hull vertices)", lambda: x.bounding_box_2d_loose(occlusion=False)),
            ("unoccluded pass (loose + occlusionRatio, exact)", lambda: x.unoccluded()),
            ("bounding_box_3d", lambda: x.bounding_box_3d()),
            ("pointcloud (dense)", lambda: x.pointcloud()),
            ("motion_vectors", lambda: x.motion_vectors(prev)),
            ("camera_params", lambda: x.camera_params()),
        ]
    ctx = ev.EventContext(sim, seed=0)
    extra = [("render (tier 0: rgb+depth+ids+normals)", render)]
    if ctx.joints:
        extra.append(("event reset_joints_by_scale", lambda: ev.reset_joints_by_scale(ctx, None, (0.5, 1.5), (0.0, 0.0))))
    extra.append(("event randomize_visual_color", lambda: ev.randomize_visual_color(r, None, {"r": (0, 1), "g": (0, 1), "b": (0, 1)})))
    inst, _ = alt.instance_segmentation()
    rejected = [
        ("[rejected] extents with torch.bincount counts", lambda: alt._torch_extents_from_ids(inst, counts="bincount")),
        ("[rejected] loose box, every render-mesh vertex", lambda: allv.bounding_box_2d_loose(occlusion=False)),
    ]
    print(f"envs {n}, {a.size}x{a.size}, instances {ann.E}, render slots {r.G}, scene {os.path.basename(a.xml)}")
    print(f"{'annotator':52s} {'warp ms':>9s} {'torch ms':>9s}   warp us/env")
    for (name, fw), (_, ft) in zip(rows_for(ann), rows_for(alt)):
        tw, tt = _time(fw, a.reps), _time(ft, a.reps)
        print(f"{name:52s} {tw * 1e3:9.2f} {tt * 1e3:9.2f}   {tw / n * 1e6:9.2f}")
    for name, fn in extra + rejected:
        t = _time(fn, a.reps)
        print(f"{name:52s} {t * 1e3:9.2f} {'':9s}   {t / n * 1e6:9.2f}")


if __name__ == "__main__":
    main()

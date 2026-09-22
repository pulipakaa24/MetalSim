"""Render-only and physics+render throughput of the native tier-0 renderer.

Same methodology as mjbatch-metal's bench.py / end_to_end.py (env-frames/s and steps/s), so the
numbers compare directly with its published M4 Max table.
Usage: python -m orchard.bench.render_bench [scene.xml]
"""
import sys
import time

import mujoco
import numpy as np
import torch
import warp as wp

from orchard.physics.batch import BatchSim, BatchSimOptions
from orchard.render.tier0 import Tier0Renderer

PRIMS = """
<mujoco>
  <worldbody>
    <light directional="true" pos="0 0 3" dir="0.3 0.3 -0.9"/>
    <camera name="cam" pos="0.6 -0.6 0.5" quat="0.85 0.4 0.15 0.3"/>
    <geom name="floor" type="plane" size="2 2 0.1" rgba="0.3 0.3 0.35 1"/>
    <body pos="0 0 0.3"><joint type="free"/><geom name="b1" type="box" size="0.06 0.04 0.05" rgba="0.9 0.2 0.2 1"/></body>
    <body pos="0.15 0.1 0.2"><joint type="free"/><geom name="s1" type="sphere" size="0.05" rgba="0.2 0.8 0.2 1"/></body>
  </worldbody>
</mujoco>
"""


def render_only(model, cam, grid):
    print("| N envs | res | env-frames/s (render only, GPU path) |")
    print("|---|---|---|")
    for n, res in grid:
        sim = BatchSim(model, n)
        rend = Tier0Renderer(model, n, width=res, height=res, camera=cam, outputs=("rgb",))
        sim.forward(); sim.synchronize()
        v = sim.event.value
        for _ in range(3):
            rend.render(sim, v)
        rend.ctx.synchronize()
        reps = max(20, min(300, 20000 // n))
        t0 = time.perf_counter()
        for _ in range(reps):
            rend.render(sim, v)
        rend.ctx.synchronize()
        dt = time.perf_counter() - t0
        print(f"| {n} | {res} | {n * reps / dt:,.0f} |", flush=True)
        del rend, sim


def end_to_end(model, cam, n, res, substeps=1, steps=200):
    sim = BatchSim(model, n, options=BatchSimOptions(substeps=substeps))
    rend = Tier0Renderer(model, n, width=res, height=res, camera=cam, outputs=("rgb",))
    sim.synchronize()
    rng = np.random.default_rng(0)
    if model.nu:
        sim.t.ctrl.copy_(torch.as_tensor(rng.uniform(-1, 1, (n, model.nu)), dtype=torch.float32))
        torch.mps.synchronize()

    def one():
        vs = sim.step()
        vr = rend.render(sim, vs)
        rend.wait_sim_after_render(sim, vr)
    for _ in range(5):
        one()
    rend.ctx.synchronize(); sim.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        one()
    rend.ctx.synchronize(); sim.synchronize()
    dt = time.perf_counter() - t0
    return n * steps / dt


if __name__ == "__main__":
    wp.config.quiet = True
    if len(sys.argv) > 1:
        model = mujoco.MjModel.from_xml_path(sys.argv[1])
        cam = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, 0)
        render_only(model, cam, [(64, 128), (256, 128), (1024, 64)])
        print(f"| physics+render | N=64 @128px | {end_to_end(model, cam, 64, 128):,.0f} steps/s |")
        print(f"| physics+render | N=1024 @64px | {end_to_end(model, cam, 1024, 64):,.0f} steps/s |")
    else:
        model = mujoco.MjModel.from_xml_string(PRIMS)
        render_only(model, "cam", [(16, 64), (64, 64), (256, 64), (1024, 64), (16, 128), (64, 128), (256, 128),
                                   (1024, 128), (64, 256), (256, 256)])

"""Render cost of the tier-2 environment map (measured; run through the GPU queue as a `timing` job).

Two configurations, each with and without a Poly Haven map (2048x1024, sampling grid 512x256):
  camera-RL: Cartpole-RGB physical preset, 1024 envs x 100x100, 4 spp, 2 bounces (the RL default);
  gallery:   G1 hero scene, 1 env x 1024x768, 32 spp per pass, 4 bounces (and 16 passes = 512 spp).
Prints ms per frame (median of timed frames, host path with synchronisation) and ms per spp.
"""
import time

import numpy as np
import mujoco
import warp as wp
wp.config.quiet = True

from metalsim.render.tier2 import Tier2Renderer, mujoco_scene_kwargs
from metalsim.render.hdr import load_hdr
from metalsim.learn.cartpole_rgb import CARTPOLE_XML
from metalsim.learn.g1_velocity import build_g1_model

HDR = "assets/polyhaven/hdri/train/lebombo.hdr"


def timed(fn, warm=3, reps=15):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    return 1e3 * float(np.median(ts)), 1e3 * float(np.min(ts))


def main():
    hdr = load_hdr(HDR)
    rows = []
    # camera-RL
    cm = mujoco.MjModel.from_xml_string(CARTPOLE_XML)
    datas = []
    for k in range(1024):
        d = mujoco.MjData(cm); d.qpos[:] = np.random.default_rng(k).uniform(-0.2, 0.2, cm.nq); mujoco.mj_forward(cm, d); datas.append(d)
    r = Tier2Renderer(cm, 1024, width=100, height=100, camera="cam", spp=4, max_bounces=2, seed=0, **mujoco_scene_kwargs(cm))
    for label, on in (("no map", False), ("map", True)):
        r.set_environment(hdr, key="lebombo") if on else r.set_environment(None)
        med, mn = timed(lambda: r.render_host(datas, passes=1))
        rows.append(("camera-RL 1024 x 100x100, 4 spp, 2 bounces", label, med, mn, 4, 1024))
    # gallery
    m, info = build_g1_model("flat", visuals=True)
    spec = info["spec"]
    cam = spec.worldbody.add_camera(); cam.name = "hero"; cam.pos = [2.2, -1.6, 1.1]; cam.fovy = 38
    f = np.array([0.0, 0.0, 0.62]) - np.array(cam.pos); f /= np.linalg.norm(f); rr = np.cross(f, [0, 0, 1]); rr /= np.linalg.norm(rr); u = np.cross(rr, f)
    q = np.zeros(4); mujoco.mju_mat2Quat(q, np.stack([rr, u, -f], 1).reshape(-1)); cam.quat = q.tolist()
    m = spec.compile()
    d = mujoco.MjData(m); mujoco.mj_resetDataKeyframe(m, d, 0); d.qpos[2] = 0.78; mujoco.mj_forward(m, d)
    r = Tier2Renderer(m, 1, width=1024, height=768, camera="hero", spp=32, max_bounces=4, material_model="omnipbr", headlight=False, tonemap="rtx", exposure=1.0)
    for label, on in (("no map", False), ("map", True)):
        r.set_environment(hdr, key="lebombo") if on else r.set_environment(None)
        med, mn = timed(lambda: r.render_host([d], passes=1), reps=10)
        rows.append(("gallery 1 x 1024x768, 32 spp, 4 bounces", label, med, mn, 32, 1))
        med16, mn16 = timed(lambda: r.render_host([d], passes=16), warm=1, reps=5)
        rows.append(("gallery 1 x 1024x768, 512 spp (16 passes), 4 bounces", label, med16, mn16, 512, 1))
    print("\n| configuration | environment | ms / frame (median) | min | ms / spp (per frame) |")
    print("|---|---|---|---|---|")
    for cfg, label, med, mn, spp, n in rows:
        print(f"| {cfg} | {label} | {med:.1f} | {mn:.1f} | {med / spp:.3f} |")


if __name__ == "__main__":
    main()

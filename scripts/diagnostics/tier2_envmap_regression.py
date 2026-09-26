"""Bit-for-bit regression of the tier-2 parity presets across the environment-map port.

Renders the two presets with no map set and writes the raw accumulators (hdr) and 8-bit frames to an .npz with
their SHA-256: run once from a checkout of the previous renderer (PYTHONPATH=<worktree>) and once from the
current one, then `--compare a.npz b.npz`.

  (a) G1 parity scene (metalsim.parity.record_g1.hero_spec on Isaac's meta.json) under usd_scene_kwargs,
      2 envs x 1024x576, 16 spp, 3 bounces;
  (b) Cartpole-RGB physical preset (mujoco_scene_kwargs), 64 envs x 100x100, 4 spp, 2 bounces.
"""
import hashlib
import json
import os
import sys

import numpy as np


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def render_all():
    import mujoco
    import warp as wp
    wp.config.quiet = True
    import metalsim
    from metalsim.render.tier2 import Tier2Renderer, usd_scene_kwargs, mujoco_scene_kwargs
    from metalsim.learn.g1_velocity import build_g1_model
    from metalsim.parity.record_g1 import hero_spec
    from metalsim.learn.cartpole_rgb import CARTPOLE_XML
    print("metalsim from", os.path.dirname(metalsim.__file__))
    out = {}
    meta = json.load(open("runs/parity3/isaac/fidelity/isaacsim_physx/meta.json"))
    m, info = build_g1_model("flat", visuals=True)
    m = hero_spec(info["spec"], meta).compile()
    datas = []
    for k in range(2):
        d = mujoco.MjData(m); mujoco.mj_resetDataKeyframe(m, d, 0); d.qpos[2] = 0.78 + 0.05 * k; d.qpos[7:] += 0.02 * k; mujoco.mj_forward(m, d); datas.append(d)
    r = Tier2Renderer(m, 2, width=1024, height=576, camera="hero", spp=16, max_bounces=3, seed=0, **usd_scene_kwargs(meta))
    o = r.render_host(datas, passes=1)
    out["g1_hdr"], out["g1_rgb"], out["g1_depth"] = o["hdr"], o["rgb"], o["depth"]
    cm = mujoco.MjModel.from_xml_string(CARTPOLE_XML)
    datas = []
    for k in range(64):
        d = mujoco.MjData(cm); d.qpos[:] = np.random.default_rng(k).uniform(-0.2, 0.2, cm.nq); mujoco.mj_forward(cm, d); datas.append(d)
    r = Tier2Renderer(cm, 64, width=100, height=100, camera="cam", spp=4, max_bounces=2, seed=0, **mujoco_scene_kwargs(cm))
    o = r.render_host(datas, passes=1)
    out["cp_hdr"], out["cp_rgb"] = o["hdr"], o["rgb"]
    for k, v in out.items():
        print(f"{k}: shape {v.shape} sha256 {sha(v)} mean {float(v.mean()):.6f}")
    return out


def main():
    if sys.argv[1] == "--compare":
        a, b = np.load(sys.argv[2]), np.load(sys.argv[3])
        ok = True
        for k in a.files:
            x, y = a[k], b[k]
            diff = float(np.abs(x.astype(np.float64) - y.astype(np.float64)).max())
            same = sha(x) == sha(y)
            ok &= same
            print(f"{k}: sha {sha(x)} vs {sha(y)} {'IDENTICAL' if same else 'DIFFERENT'}, max |diff| {diff:g}")
        print("bit-for-bit:", "YES" if ok else "NO")
        return
    out = render_all()
    os.makedirs(os.path.dirname(sys.argv[1]), exist_ok=True)
    np.savez(sys.argv[1], **out)
    print("wrote", sys.argv[1])


if __name__ == "__main__":
    main()

"""Render cost of the rough task's terrain (G1 rough, ``terrain_collision="boxes_local"``): N envs at their Isaac terrain
origins (seed 0), a 128 x 128 head camera on ``torso_link`` (0.1 m ahead, 0.35 m up, 40 degrees down), no physics steps
between frames (the pose buffers are fixed; each frame waits on the same sim event). Renderer configurations:

  before   terrain_slots=False, draw_hfields=False   the renderer as it was (no heightfield; slots baked at 2 mm)
  slots    terrain_slots=True,  draw_hfields=False   the per-world box slots only
  hfield   terrain_slots=False, draw_hfields=True    the exact heightfield only
  after    terrain_slots=True,  draw_hfields=True    both (the new default)

ms per frame = best of 3 x ``--frames`` back-to-back renders, one synchronize each; tier 2 = Cartpole-RGB's physical
settings (4 spp, 2 bounces, exposure 0.25). A 4 x 4 tile of the first 16 envs is saved per configuration.

    scripts/gpu_run.sh terrain_render_cost timing 10 -- python scripts/diagnostics/terrain_render_cost.py --n 1024
"""
import argparse, json, os, time
import numpy as np, mujoco, torch, warp as wp
wp.config.quiet = True

CONFIGS = {"before": dict(terrain_slots=False, draw_hfields=False), "slots": dict(terrain_slots=True, draw_hfields=False),
           "hfield": dict(terrain_slots=False, draw_hfields=True), "after": dict(terrain_slots=True, draw_hfields=True)}


def rough_scene(n, seed=0, visuals=False):
    from metalsim.learn.g1_velocity import build_g1_model
    from metalsim.learn.terrain import isaac_rough_terrain, BoxWindow
    from metalsim.physics.batch import BatchSim, BatchSimOptions
    hf = isaac_rough_terrain(seed=seed, collision="boxes_local")
    m, info = build_g1_model("rough", hf, visuals=visuals)
    spec = info["spec"]
    tb = next(b for b in spec.bodies if b.name == "torso_link")
    a = np.deg2rad(40.0)
    cam = tb.add_camera(); cam.name = "head"; cam.pos = [0.1, 0.0, 0.35]; cam.fovy = 70.0
    x = np.array([0.0, -1.0, 0.0]); y = np.array([np.sin(a), 0.0, np.cos(a)]); R = np.stack([x, y, np.cross(x, y)], 1)
    qc = np.zeros(4); mujoco.mju_mat2Quat(qc, R.reshape(-1)); cam.quat = qc.tolist()     # looks forward, 40 deg down
    l = spec.worldbody.add_light(); l.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL; l.dir = [0.3, 0.2, -1.0]; l.pos = [0, 0, 10]
    l.castshadow = True
    m = spec.compile()
    sim = BatchSim(m, n, options=BatchSimOptions(per_world_fields=BoxWindow.FIELDS, njmax=256, nconmax=128))
    q = np.tile(m.key_qpos[0], (n, 1)).astype(np.float32)
    o = hf["origins"](n, seed); q[:, :2] = o[:, :2]; q[:, 2] += o[:, 2]
    sim.set_state(q, np.zeros((n, m.nv), np.float32))
    win = BoxWindow(sim, hf); win.launch(); sim.synchronize()
    return m, sim, hf, win


def time_renderer(r, sim, frames, tier):
    ev = sim.event.value
    for _ in range(3):
        r.render(sim, ev)
    r.ctx.synchronize()
    best = []
    for _ in range(3):
        t0 = time.perf_counter()
        for _ in range(frames):
            r.render(sim, ev)
        r.ctx.synchronize()
        best.append((time.perf_counter() - t0) / frames * 1e3)
    return min(best), float(np.median(best))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=1024); ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--frames", type=int, default=10); ap.add_argument("--tiers", default="0,2"); ap.add_argument("--configs", default="before,slots,hfield,after")
    ap.add_argument("--out", default="runs/terrain_render")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    import imageio.v2 as iio
    from metalsim.render.tier0 import Tier0Renderer
    from metalsim.render.tier2 import Tier2Renderer, mujoco_scene_kwargs
    m, sim, hf, win = rough_scene(a.n)
    ov = win.overflow.numpy()
    print(f"G1 rough (boxes_local), {a.n} envs, {a.size}x{a.size} head camera; box window overflow {int(ov[0])}, most boxes in a window {int(ov[1])}", flush=True)
    rows = []
    for tier in [int(t) for t in a.tiers.split(",")]:
        for name in a.configs.split(","):
            kw = CONFIGS[name]
            t0 = time.perf_counter()
            if tier == 0:
                r = Tier0Renderer(m, a.n, width=a.size, height=a.size, camera="head", outputs=("rgb", "depth"), **kw)
            else:
                r = Tier2Renderer(m, a.n, width=a.size, height=a.size, camera="head", spp=4, max_bounces=2,
                                  **mujoco_scene_kwargs(m, exposure=0.25), **kw)
            build_s = time.perf_counter() - t0
            best, med = time_renderer(r, sim, a.frames, tier)
            rgb = r.out.rgb[:16].cpu().numpy(); dep = r.out.depth[:16].cpu().numpy()
            tile = np.concatenate([np.concatenate(list(rgb[i * 4:(i + 1) * 4]), 1) for i in range(4)], 0)
            iio.imwrite(os.path.join(a.out, f"tier{tier}_{name}.png"), tile)
            row = dict(tier=tier, config=name, n=a.n, size=a.size, ms_best=best, ms_median=med, fps_envs=a.n / best * 1e3,
                       drawn_slots=int(r.tables.G), tris_per_env=int(sum(r.tables.meshes[k]["i_count"] for k in r.tables.geom_mesh) // 3), build_s=build_s,
                       depth_valid=float((dep > 0).mean()))
            rows.append(row)
            print(json.dumps(row), flush=True)
            del r; torch.mps.synchronize()
    with open(os.path.join(a.out, "cost.jsonl"), "a") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()

"""Render cost of the tier-2 options at the two sizes that matter: 1024 envs x 100x100 (camera RL, Cartpole-RGB
scene, GPU path from BatchSim) and 1024x576 hero frames of the G1 parity scene (batch 1 and 4, host path).

    python -m metalsim.render.bench_cost [--hero_isaac runs/parity/isaac/parity_out2/rt]
"""
import argparse, json, time

import mujoco
import numpy as np


def _time(fn, reps):
    fn(); fn()
    t = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); t.append(time.perf_counter() - t0)
    return float(np.median(t))


RL_CONFIGS = {
    "legacy": {},
    "sun_disk+tonemap": dict(usd_lights=[{"intensity": 0.8, "color": (1, 1, 1), "angle_deg": 0.53}], tonemap="rtx", exposure=1.0),
    "sun_disk+tonemap+atrous": dict(usd_lights=[{"intensity": 0.8, "color": (1, 1, 1), "angle_deg": 0.53}], tonemap="rtx", exposure=1.0, denoise="atrous"),
    "sun_disk+tonemap+atrous3": dict(usd_lights=[{"intensity": 0.8, "color": (1, 1, 1), "angle_deg": 0.53}], tonemap="rtx", exposure=1.0, denoise="atrous", atrous_iters=3),
    "sun_disk+tonemap+oidn": dict(usd_lights=[{"intensity": 0.8, "color": (1, 1, 1), "angle_deg": 0.53}], tonemap="rtx", exposure=1.0, denoise="oidn", denoise_quality="fast"),
}


def bench_rl(n=1024, spp=1, bounces=2, configs=None, reps=10):
    import torch
    from metalsim.learn.cartpole_rgb import CARTPOLE_XML
    from metalsim.physics.batch import BatchSim
    from metalsim.render.tier2 import Tier2Renderer
    m = mujoco.MjModel.from_xml_string(CARTPOLE_XML)
    sim = BatchSim(m, n); sim.synchronize()
    v = sim.step(); sim.synchronize()
    out = {}
    for name in (configs or RL_CONFIGS):
        r = Tier2Renderer(m, n, width=100, height=100, camera="cam", spp=spp, max_bounces=bounces, **RL_CONFIGS[name])
        def go():
            vr = r.render(sim, v); r.after(vr); torch.mps.synchronize()
        out[name] = _time(go, reps) * 1e3
        print(f"[bench] RL {n}x100x100 spp {spp} bounces {bounces} {name}: {out[name]:.2f} ms/batch", flush=True)
        del r
    return out


def fidelity_rl(n=256, spp=1, bounces=2, gt_passes=256, configs=("sun_disk+tonemap", "sun_disk+tonemap+atrous", "sun_disk+tonemap+atrous3", "sun_disk+tonemap+oidn")):
    """PSNR of the batched 100x100 observation at the RL setting (1 spp) against the converged image (256 passes of the
    same renderer, no denoiser), per denoiser option, on the Cartpole-RGB scene after 20 random-action steps."""
    import torch
    from metalsim.learn.cartpole_rgb import CARTPOLE_XML
    from metalsim.physics.batch import BatchSim
    from metalsim.render.tier2 import Tier2Renderer
    m = mujoco.MjModel.from_xml_string(CARTPOLE_XML)
    sim = BatchSim(m, n); sim.synchronize()
    sim.t.ctrl.copy_(torch.rand(n, m.nu, device="mps") * 2 - 1)
    for _ in range(20): v = sim.step()
    sim.synchronize()
    def img(cfg, passes, seed=0):
        r = Tier2Renderer(m, n, width=100, height=100, camera="cam", spp=spp, max_bounces=bounces, seed=seed, **RL_CONFIGS[cfg])
        vr = r.render(sim, v, passes=passes); r.after(vr); torch.mps.synchronize()
        return r.out.rgb.cpu().numpy().astype(np.float64) / 255
    gt = img("sun_disk+tonemap", gt_passes, seed=99)
    out = {}
    for cfg in configs:
        x = img(cfg, 1)
        out[cfg] = float(np.mean([10 * np.log10(1 / max(((x[e] - gt[e]) ** 2).mean(), 1e-12)) for e in range(n)]))
        print(f"[fidelity] RL {n}x100x100 {spp} spp {cfg}: {out[cfg]:.2f} dB vs {gt_passes}-pass converged", flush=True)
    return out


def bench_hero(isaac, presets, batch=(1, 4), spp=16, passes=4, bounces=3, reps=3):
    from metalsim.render.rtx_parity import build_scene, preset_kwargs
    from metalsim.render.tier2 import Tier2Renderer
    meta = json.load(open(f"{isaac}/meta.json"))
    m = build_scene(meta); cam = meta["camera"]
    out = {}
    for b in batch:
        datas = [mujoco.MjData(m) for _ in range(b)]
        for d in datas: mujoco.mj_forward(m, d)
        for p in presets:
            r = Tier2Renderer(m, b, width=cam["width"], height=cam["height"], camera="hero", spp=spp, max_bounces=bounces, **preset_kwargs(p, meta))
            ms = _time(lambda: r.render_host(datas, passes=passes), reps) * 1e3 / b
            out[f"{p}_b{b}"] = ms
            print(f"[bench] hero 1024x576 batch {b} {spp} spp x {passes} {p}: {ms:.1f} ms/frame", flush=True)
            del r
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--hero_isaac", default="runs/parity/isaac/parity_out2/rt")
    ap.add_argument("--presets", default="legacy,tonemap,atrous,oidn"); ap.add_argument("--passes", type=int, default=4)
    ap.add_argument("--rl_only", action="store_true"); ap.add_argument("--fidelity_rl", action="store_true"); ap.add_argument("--out", default="")
    a = ap.parse_args()
    import warp as wp; wp.config.quiet = True
    if a.fidelity_rl:
        print(json.dumps({"rl_fidelity_1spp": fidelity_rl(), "rl_fidelity_4spp": fidelity_rl(spp=4, gt_passes=64)}, indent=1)); raise SystemExit
    res = {"rl_1spp_2b": bench_rl(), "rl_4spp_2b": bench_rl(spp=4)}
    if not a.rl_only:
        res["hero"] = bench_hero(a.hero_isaac, a.presets.split(","), passes=a.passes)
    print(json.dumps(res, indent=1))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)

"""Cost of the lidar extras (metalsim.sensors.raytrace.Lidar) per 64-beam scan at 1024 envs, on the
lidar-navigation scene (12 boxes + robot + floor per env). The lidar kernel alone is timed by
encoding it REPS times in one command buffer (acceleration structures already built); the full
trace (instance refit + TLAS rebuild + kernel) is timed with the default lidar for reference.

    scripts/gpu_run.sh lidar_extras_bench timing 5 -- .venv/bin/python scripts/diagnostics/bench_lidar_extras.py
"""
import time

import numpy as np

from metalsim.learn.lidar_nav import LidarNavConfig, LidarNavEnv

REPS = 200


def main():
    import subprocess, datetime
    print('gpu_lock status at start', datetime.datetime.now().isoformat(timespec='seconds'), flush=True)
    print(subprocess.run(['python3', 'scripts/gpu_lock.py', 'status'], capture_output=True, text=True).stdout, flush=True)
    env = LidarNavEnv(LidarNavConfig(num_envs=1024))
    env.reset(); env.synchronize()
    rt, base = env.rt, env.lidar
    az, el = base.beams[:, 0], base.beams[:, 1]
    site = base.site
    opts = {
        "default (1 ray, first hit)": {},
        "reflectance table (ext kernel, 1 ray)": dict(reflectance={".*": 0.5}),
        "multi-return 2 (re-cast)": dict(max_returns=2),
        "multi-return 4 (re-cast)": dict(max_returns=4),
        "divergence 4 rays": dict(divergence_deg=(0.5, 0.5), spot_rays=4),
        "divergence 8 rays": dict(divergence_deg=(0.5, 0.5), spot_rays=8),
        "divergence 16 rays": dict(divergence_deg=(0.5, 0.5), spot_rays=16),
        "divergence 32 rays": dict(divergence_deg=(0.5, 0.5), spot_rays=32),
        "divergence 16 rays + 2 returns (re-cast)": dict(divergence_deg=(0.5, 0.5), spot_rays=16, max_returns=2),
    }
    lidars = {k: rt.make_lidar(site, az, el, **v) for k, v in opts.items()}
    # one full trace to build the structures from the sim state
    v = rt.trace(env.sim, list(lidars.values())); rt.ctx.synchronize()
    c = rt.ctx
    def time_kernel(l):
        cb = c.command_buffer()
        for _ in range(REPS):
            l.encode(cb, sim=env.sim)
        t0 = time.perf_counter(); c.commit(cb); c.synchronize()
        return (time.perf_counter() - t0) / REPS * 1e6
    def time_trace(l, reps=100):
        t0 = time.perf_counter()
        for _ in range(reps):
            rt.trace(env.sim, [l])
        c.synchronize()
        return (time.perf_counter() - t0) / reps * 1e6
    for l in lidars.values():
        time_kernel(l)   # warm-up / pipeline compile
    res = {k: [time_kernel(l) for _ in range(5)] for k, l in lidars.items()}
    full = [time_trace(lidars["default (1 ray, first hit)"]) for _ in range(3)]
    base_us = np.median(res["default (1 ray, first hit)"])
    print(f"lidar kernel per 64-beam scan x 1024 envs (65,536 beams), median of 5 x {REPS} encodes in one command buffer")
    for k, v in res.items():
        med = np.median(v)
        print(f"  {k:44s} {med:9.1f} us   ({med / base_us:5.2f}x default)   runs {np.round(v, 1).tolist()}")
    print(f"  full trace (refit + TLAS rebuild + default kernel), synchronized per 100: {np.median(full):.1f} us")
    # radar-lite: 64 x 8 rays over 120 x 20 deg + Doppler/RCS kernel, full trace
    from metalsim.sensors.radar import Radar
    radar = Radar(rt, env.sim, site, fov_deg=(120.0, 20.0), n_az=64, n_el=8)
    radar.trace(); env.sim.synchronize()
    def time_radar(reps=100):
        t0 = time.perf_counter()
        for _ in range(reps):
            radar.trace()
        env.sim.synchronize()
        return (time.perf_counter() - t0) / reps * 1e6
    rr = [time_radar() for _ in range(3)]
    print(f"  radar-lite 512 rays x 1024 envs, full trace + Doppler/RCS kernel, synchronized per 100: {np.median(rr):.1f} us")


if __name__ == "__main__":
    main()

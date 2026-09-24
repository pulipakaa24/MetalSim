"""Physics-only throughput on Isaac's G1, idle GPU, graph replay: MuJoCo Warp (our BatchSim, 10 Newton it.,
20 LS, 2.5 ms, 8 substeps per graph) vs Newton XPBD with working drives (ActuatorPD: Isaac gains, effort
limits, armature; joint relaxation 0.4/0.4) at the settings that track MuJoCo C (newton_xpbd_drives.py C),
at 256 / 1024 / 4096 envs. Rates are normalized to simulated time: "2.5 ms-steps/s" = N * simulated
seconds per wall second / 2.5 ms (a 1.25 ms setting runs 2 substeps per 2.5 ms step); env-steps/s at
Isaac's 50 Hz control = that / 8. Also checks that graph replay reproduces eager stepping.

usage: python scripts/diagnostics/newton_xpbd_throughput.py [N ...]"""
import sys, time, numpy as np, warp as wp
wp.config.quiet = True
from metalsim.physics.newton_backend import G1XPBD

H = 0.0025
SIM_T = 0.5                         # simulated seconds per timing
SETTINGS = [("ipd", 4, 0.00125), ("ipd", 8, 0.0025), ("ipd", 4, 0.0025), ("xpbd-drive", 4, 0.0025)]
NS = [int(a) for a in sys.argv[1:]] or [256, 1024, 4096]


def mjwarp_rate(N):
    from metalsim.learn.g1_velocity import build_g1_model
    from metalsim.physics.batch import BatchSim, BatchSimOptions
    m, _ = build_g1_model("flat", physics_dt=H)
    sim = BatchSim(m, N, options=BatchSimOptions(substeps=8, njmax=256, nconmax=32, solver_iterations=10, ls_iterations=20))
    sim.synchronize()
    for _ in range(3): sim.step()
    sim.synchronize(); calls = int(SIM_T / (8 * H)); t0 = time.perf_counter()
    for _ in range(calls): sim.step()
    sim.synchronize(); r = N * calls * 8 / (time.perf_counter() - t0)
    del sim
    return r


def newton_rate(N, drive, it, dt):
    sim = G1XPBD(N, iterations=it, dt=dt, drive="xpbd" if drive == "xpbd-drive" else drive,
                 armature_inertia="iso" if drive == "ipd" else False,
                 **({} if drive == "ipd" else {"solver_kw": {"joint_linear_relaxation": 0.7, "joint_angular_relaxation": 0.4}}))
    block = int(round(0.02 / dt))                     # one 50 Hz control step per graph launch
    sim.step(block, graph=True); wp.synchronize()
    calls = int(SIM_T / 0.02); t0 = time.perf_counter()
    for _ in range(calls): sim.step(block, graph=True)
    wp.synchronize(); el = time.perf_counter() - t0
    ok = bool(np.isfinite(sim.s0.body_q.numpy()).all())
    return N * calls * 0.02 / el / H, ok


def graph_matches_eager():
    a = G1XPBD(4, iterations=4, dt=0.00125, drive="ipd"); b = G1XPBD(4, iterations=4, dt=0.00125, drive="ipd")
    for _ in range(25): a.step(16, graph=True)
    b.step(400); wp.synchronize()
    return float(np.abs(a.pelvis_z() - b.pelvis_z()).max()), float(a.pelvis_z().mean())


if __name__ == "__main__":
    d, z = graph_matches_eager()
    print(f"graph replay vs eager after 0.5 s (ipd 4 it 1.25 ms): max pelvis z diff {d:.2e} m (z {z:.3f})", flush=True)
    for N in NS:
        mj = mjwarp_rate(N)
        row = [f"N={N}: MuJoCo Warp {mj:,.0f} 2.5ms-steps/s ({mj/8:,.0f} env-steps/s)"]
        for drive, it, dt in SETTINGS:
            r, ok = newton_rate(N, drive, it, dt)
            row.append(f"XPBD {drive} {it} it {dt*1e3:.2f} ms: {r:,.0f} ({r/8:,.0f} env-steps/s, {r/mj:.1f}x){'' if ok else ' NON-FINITE'}")
        print("\n    ".join(row), flush=True)

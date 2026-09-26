"""MuJoCo (C, float64) and MuJoCo Warp (float32, CPU or Metal) on the closed-loop test mechanisms (mechanisms.py):
soft loop closure through `connect` equality constraints.

    .venv/bin/python scripts/diagnostics/closed_loops/mjw_loops.py traj --mech fourbar --dt 0.0025 --engine c
    scripts/gpu_run.sh mjw_loops low 20 -- .venv/bin/python .../mjw_loops.py traj --engine mjw --device metal:0 --worlds 1024
    scripts/gpu_run.sh mjw_loops low 10 -- .venv/bin/python .../mjw_loops.py bench --device metal:0 --worlds 1024

traj: records world-0 body direction angles every step, loop-closure error (max and p99 over worlds) and
non-finite / peak joint speed, with --torque none | sigma3 (sigma3 runs use mechanisms.DAMPING) (kamino_probe.torque_table: 3 x SIGMA_TAU, Gaussian,
held 20 ms, per-world seeds). bench: graph-replayed env-steps/s (constant random torques).
"""
import argparse, json, math, os, sys, time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mechanisms as M  # noqa: E402
from mechanisms import torque_table  # noqa: E402


class _Sim:
    """MuJoCo Warp batch on any Warp device: on Metal the same setup as metalsim.physics.batch.BatchSim (graph
    replay, graph_conditional off); on the CPU device plain launches."""

    def __init__(self, m, nworld, device):
        import mujoco, mujoco_warp as mjw, warp as wp
        self.mjw, self.wp = mjw, wp
        self.device = wp.get_device(device)
        d = mujoco.MjData(m); mujoco.mj_forward(m, d)
        with wp.ScopedDevice(self.device):
            self.m = mjw.put_model(m)
            if getattr(self.device, "is_metal", False):
                self.m.opt.graph_conditional = False
            self.m.opt.warn_overflow = 0
            self.d = mjw.put_data(m, d, nworld=nworld, njmax=64)
            self.graph = None
            if getattr(self.device, "is_metal", False):
                with wp.ScopedCapture(device=self.device) as cap:
                    mjw.step(self.m, self.d)
                self.graph = cap.graph

    def step(self):
        with self.wp.ScopedDevice(self.device):
            if self.graph is not None:
                self.wp.capture_launch(self.graph)
            else:
                self.mjw.step(self.m, self.d)

    def forward(self):
        with self.wp.ScopedDevice(self.device):
            self.mjw.forward(self.m, self.d)

    def synchronize(self):
        self.wp.synchronize_device(self.device)

    def overflow_flags(self):
        self.synchronize()
        ov = self.d.overflow.numpy()
        return int((ov != 0).sum())


def mj_angles(xmat):
    """(..., nbody, 9) -> direction angles of bodies 1.. (local +x in the x-z plane)."""
    return np.arctan2(xmat[..., 1:, 6], xmat[..., 1:, 0])


def parse_solimp(s):
    return tuple(float(x) for x in s.split(","))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("traj", "bench"))
    ap.add_argument("--mech", default="fourbar", choices=list(M.GEOMS))
    ap.add_argument("--engine", default="c", choices=("c", "mjw"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dt", type=float, default=0.0025)
    ap.add_argument("--T", type=float, default=5.0)
    ap.add_argument("--worlds", type=int, default=1)
    ap.add_argument("--torque", default="none", choices=("none", "sigma3"))
    ap.add_argument("--amp", type=float, default=3.0, help="torque amplitude in sigmas (sigma3 mode)")
    ap.add_argument("--solref", default="0.02,1")
    ap.add_argument("--solimp", default="0.9,0.95,0.001,0.5,2")
    ap.add_argument("--integrator", default="implicitfast")
    ap.add_argument("--no_eq", action="store_true")
    ap.add_argument("--bench_steps", type=int, default=2000)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    import mujoco
    xml = M.mjcf(a.mech, dt=a.dt, solref=tuple(float(x) for x in a.solref.split(",")), solimp=parse_solimp(a.solimp),
                 integrator=a.integrator, eq_on=not a.no_eq, damped=a.torque == "sigma3")
    m = mujoco.MjModel.from_xml_string(xml)
    nb = m.nbody - 1
    nsteps = int(round(a.T / a.dt)); hold = int(round(0.02 / a.dt))
    tau = torque_table(a.mech, a.worlds, a.T, amp=a.amp) if a.torque == "sigma3" else None
    meta = dict(mode=a.mode, mech=a.mech, engine=a.engine, device=a.device if a.engine == "mjw" else "cpu(C)", dt=a.dt, T=a.T,
                worlds=a.worlds, torque=a.torque, amp=a.amp, solref=a.solref, solimp=a.solimp, integrator=a.integrator, eq=not a.no_eq,
                mujoco=mujoco.__version__)

    if a.engine == "c":
        ds = [mujoco.MjData(m) for _ in range(a.worlds)]
        ang = np.zeros((nsteps + 1, nb)); clos = np.zeros(nsteps + 1); p99 = np.zeros(nsteps + 1)
        for d in ds:
            mujoco.mj_forward(m, d)
        ang[0] = mj_angles(ds[0].xmat); peak = 0.0; nonfin = np.zeros(a.worlds, bool)
        t0 = time.time()
        for k in range(nsteps):
            ce = np.zeros(a.worlds)
            for w, d in enumerate(ds):
                if tau is not None and k % hold == 0:
                    d.ctrl[:] = tau[k // hold, w]
                mujoco.mj_step(m, d)
                if d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number > 0:   # MuJoCo C resets the world itself
                    nonfin[w] = True
                ce[w] = M.mj_closure_error(a.mech, d.xpos, d.xmat)
                if not np.isfinite(d.qvel).all():
                    nonfin[w] = True
                else:
                    peak = max(peak, float(np.abs(d.qvel).max()))
            ang[k + 1] = mj_angles(ds[0].xmat)
            fin = np.isfinite(ce)
            clos[k + 1] = ce[fin].max() if fin.any() else np.nan
            p99[k + 1] = np.percentile(ce[fin], 99) if fin.any() else np.nan
        wall = time.time() - t0
    else:
        import warp as wp
        wp.config.quiet = True
        sim = _Sim(m, a.worlds, a.device)
        meta["device"] = str(sim.device)
        if a.mode == "bench":
            rng = np.random.default_rng(0)
            sig = np.array([M.SIGMA_TAU[a.mech][j] for j in M.GEOMS[a.mech]()["actuated"]])
            sim.d.ctrl.assign((rng.standard_normal((a.worlds, m.nu)) * sig).astype(np.float32))
            for _ in range(20):
                sim.step()
            sim.synchronize()
            t0 = time.time()
            for _ in range(a.bench_steps):
                sim.step()
            sim.synchronize()
            el = time.time() - t0
            qv = sim.d.qvel.numpy()
            ce = M.mj_closure_error(a.mech, sim.d.xpos.numpy(), sim.d.xmat.numpy().reshape(a.worlds, m.nbody, 9))
            meta.update(steps=a.bench_steps, s=el, env_steps_per_s=a.worlds * a.bench_steps / el,
                        nonfinite=int((~np.isfinite(qv).all(1)).sum()), closure_max_end=float(np.nanmax(ce)),
                        overflow=sim.overflow_flags())
            print(json.dumps(meta), flush=True)
            return
        ang = np.zeros((nsteps + 1, nb)); clos = np.zeros(nsteps + 1); p99 = np.zeros(nsteps + 1)
        sim.forward(); sim.synchronize()
        ang[0] = mj_angles(sim.d.xmat.numpy().reshape(a.worlds, m.nbody, 9)[0]); peak = 0.0
        t0 = time.time()
        for k in range(nsteps):
            if tau is not None and k % hold == 0:
                sim.synchronize()
                sim.d.ctrl.assign(tau[k // hold].astype(np.float32))
            sim.step(); sim.synchronize()
            xm = sim.d.xmat.numpy().reshape(a.worlds, m.nbody, 9)
            ce = M.mj_closure_error(a.mech, sim.d.xpos.numpy(), xm)
            ang[k + 1] = mj_angles(xm[0])
            fin = np.isfinite(ce)
            clos[k + 1] = ce[fin].max() if fin.any() else np.nan
            p99[k + 1] = np.percentile(ce[fin], 99) if fin.any() else np.nan
            if k % 20 == 19:
                qv = sim.d.qvel.numpy(); ok = np.isfinite(qv).all(1)
                if ok.any():
                    peak = max(peak, float(np.abs(qv[ok]).max()))
        wall = time.time() - t0
        qv = sim.d.qvel.numpy(); nonfin = ~np.isfinite(qv).all(1)
        meta["overflow"] = sim.overflow_flags()
    E = M.energy(a.mech, ang, a.dt)
    meta.update(closure_max=float(np.nanmax(clos)), closure_mean=float(np.nanmean(clos)), closure_p99_max=float(np.nanmax(p99)),
                nonfinite_worlds=int(np.sum(nonfin)), peak_joint_speed=peak, wall_s=wall,
                energy_start=float(E[0]), energy_end=float(E[-1]))
    print(json.dumps(meta), flush=True)
    if a.out:
        np.savez(a.out, t=np.arange(nsteps + 1) * a.dt, ang=ang, closure=clos, closure_p99=p99, meta=json.dumps(meta))


if __name__ == "__main__":
    main()

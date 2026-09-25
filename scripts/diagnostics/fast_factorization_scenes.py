"""Fast-factorization settings (``metal_register_cholesky_max`` 48, ``m_dense_max`` 0; the G1's throughput
settings, docs/research/mjwarp_throughput_2026-09-25.md §5) on every other scene, against the previous
defaults (40 / MuJoCo Warp's M_BLOCK_DENSE_MAX 64). One process per configuration (the register bound is
process-wide warp.config, read at module build).

Per scene, with the scene's own BatchSimOptions (substeps, njmax, nconmax, iterations, per-world fields):
  floor    two graph-replay instances of this configuration, max |dqpos| / |dqvel| (MuJoCo Warp's atomics
           order contacts and rows nondeterministically, so identical configurations already differ)
  eager    graph replay vs eager launches of the same configuration
  C        worlds 0-1 vs mj_step (float64) under the same ctrl sequence
  tp       env steps/s at N_TP envs (fixed ctrl, 50 timed steps after 5 warm-up, one synchronize)
The state after 1 / 10 / 50 / 100 / 200 steps (random per-world ctrl in ctrlrange, seed 0) is saved; ``--ref`` compares
with the other configuration's file.

usage: python scripts/diagnostics/fast_factorization_scenes.py {old,new} --out F.npz [--ref G.npz] [--n 64] [--ntp 4096] [--scenes a,b]
"""
import argparse, json, os, sys, time
import numpy as np, mujoco, warp as wp
wp.config.quiet = True

CFG = {"old": dict(metal_register_cholesky_max=40, m_dense_max=64), "new": dict(metal_register_cholesky_max=48, m_dense_max=0)}
CHECK = (1, 10, 50, 100, 200)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def scenes():
    from metalsim.learn.cartpole_rgb import CARTPOLE_XML
    from metalsim.learn.lidar_nav import build_xml
    from metalsim.learn.so101_lift import SCENE
    from metalsim.learn.tron1_wf import build_train_model
    from metalsim.tron1.sim import build_model
    from metalsim.tron1.realism import SimParams
    men = os.path.join(ROOT, "upstream", "mujoco_menagerie")
    tron_f = ("body_mass", "body_ipos", "body_inertia", "geom_friction", "actuator_gainprm", "actuator_biasprm", "dof_frictionloss")
    return {   # name: (model factory, the scene's BatchSimOptions kwargs)
        "cartpole_rgb": (lambda: mujoco.MjModel.from_xml_string(CARTPOLE_XML), dict(substeps=2, njmax=32)),
        "cartpole_state": (lambda: mujoco.MjModel.from_xml_string(CARTPOLE_XML),
                           dict(substeps=2, njmax=32, solver_iterations=10, ls_iterations=10)),
        "so101_lift": (lambda: mujoco.MjModel.from_xml_path(SCENE), dict(substeps=4, njmax=512, per_world_fields=("body_mass", "geom_friction"))),
        "lidar_nav": (lambda: mujoco.MjModel.from_xml_string(build_xml()),
                      dict(substeps=5, njmax=512, nconmax=96, solver_iterations=10, ls_iterations=10)),
        "tron1_wf": (build_train_model, dict(substeps=1, njmax=64, nconmax=16, solver_iterations=10, ls_iterations=20, per_world_fields=tron_f)),
        # tests/test_tron1.py's model (MuJoCo C there; no GPU task of its own): torque motors at 10 % of the range,
        # capacity for its wheel/foot contacts
        "tron1_sim": (lambda: build_model(SimParams.nominal()), dict(substeps=10, njmax=256, nconmax=64), 0.1),
        "go1": (lambda: mujoco.MjModel.from_xml_path(os.path.join(men, "unitree_go1", "scene.xml")), dict(substeps=10)),
        "panda": (lambda: mujoco.MjModel.from_xml_path(os.path.join(men, "franka_emika_panda", "scene.xml")), dict(substeps=10)),
    }


def ctrl_seq(m, n, T, seed=0, scale=1.0):
    """Uniform in ctrlrange (or [-1, 1]); ``scale`` shrinks it about its centre."""
    rng = np.random.default_rng(seed)
    lo = np.where(m.actuator_ctrllimited.astype(bool), m.actuator_ctrlrange[:, 0], -1.0)
    hi = np.where(m.actuator_ctrllimited.astype(bool), m.actuator_ctrlrange[:, 1], 1.0)
    if scale != 1.0:
        c, h = 0.5 * (lo + hi), 0.5 * (hi - lo) * scale
        lo, hi = c - h, c + h
    return (lo + (hi - lo) * rng.uniform(size=(T, n, m.nu))).astype(np.float32)


def init_state(m):
    d = mujoco.MjData(m)
    if m.nkey:
        mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    return d


def run_sim(m, kw, n, ctrls, capture=True):
    from metalsim.physics.batch import BatchSim, BatchSimOptions
    sim = BatchSim(m, n, options=BatchSimOptions(capture=capture, **kw))
    d0 = init_state(m)
    sim.set_state(np.tile(d0.qpos, (n, 1)), np.tile(d0.qvel, (n, 1)))
    sim.synchronize()
    out = []
    for t in range(ctrls.shape[0]):
        sim.d.ctrl.assign(ctrls[t]); sim.step()
        if t + 1 in CHECK:
            sim.synchronize()
            out.append((sim.d.qpos.numpy().copy(), sim.d.qvel.numpy().copy()))
    ov = sim.overflow_flags()
    info = dict(nv=m.nv, sparse_M=bool((sim.m.qLD_block_adr.numpy() == -1).any()) if hasattr(sim.m, "qLD_block_adr") else None,
                qLD_block_total=int(getattr(sim.m, "qLD_block_total", -1)), overflow=ov)
    finite = all(np.isfinite(q).all() and np.isfinite(v).all() for q, v in out)
    return out, info, finite, sim


def run_c(m, substeps, ctrls, worlds=2):
    res = []
    for w in range(worlds):
        d = init_state(m); o = []
        for t in range(ctrls.shape[0]):
            d.ctrl[:] = ctrls[t, w]
            for _ in range(substeps):
                mujoco.mj_step(m, d)
            if t + 1 in CHECK:
                o.append(d.qpos.copy())
        res.append(o)
    return res       # [world][check] -> qpos


def throughput(m, kw, n):
    from metalsim.physics.batch import BatchSim, BatchSimOptions
    sim = BatchSim(m, n, options=BatchSimOptions(**kw))
    d0 = init_state(m)
    sim.set_state(np.tile(d0.qpos, (n, 1)), np.tile(d0.qvel, (n, 1)))
    sim.d.ctrl.assign(ctrl_seq(m, n, 1, seed=1)[0])
    for _ in range(5):
        sim.step()
    sim.synchronize()
    best = []
    for rep in range(3):
        t0 = time.perf_counter()
        for _ in range(50):
            sim.step()
        sim.synchronize()
        best.append(n * 50 / (time.perf_counter() - t0))
    return max(best), float(np.median(best))


def mx(a, b):
    return float(np.abs(a - b).max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", choices=list(CFG)); ap.add_argument("--out", required=True); ap.add_argument("--ref")
    ap.add_argument("--n", type=int, default=64); ap.add_argument("--ntp", type=int, default=4096)
    ap.add_argument("--scenes", default=""); ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--tp_only", action="store_true", help="throughput only (no correctness runs); prints one line per scene")
    a = ap.parse_args()
    if a.tp_only:
        cfg = CFG[a.config]; allsc = scenes(); res = {}
        for name in [s for s in a.scenes.split(",") if s] or list(allsc):
            fac, kw0, *_ = allsc[name]; m = fac(); kw = dict(kw0, **cfg)
            tp, tpm = throughput(m, kw, a.ntp); res[name] = (tp, tpm)
            print(f"tp {a.config} {name}: {tp:,.0f} env-steps/s best of 3 (median {tpm:,.0f}) at {a.ntp} envs x {kw['substeps']} substeps", flush=True)
        with open(a.out, "a") as f:
            f.write(json.dumps({"config": a.config, "ntp": a.ntp, "tp": res}) + "\n")
        return
    cfg = CFG[a.config]
    allsc = scenes(); names = [s for s in a.scenes.split(",") if s] or list(allsc)
    save = {}; summary = {}
    print(f"=== config {a.config} {cfg}; N={a.n} worlds, {a.steps} steps, checkpoints {CHECK}; throughput at N={a.ntp}", flush=True)
    for name in names:
        fac, kw0, *rest = allsc[name]; m = fac(); kw = dict(kw0, **cfg)
        ctrls = ctrl_seq(m, a.n, a.steps, scale=rest[0] if rest else 1.0)
        A, info, fa, _ = run_sim(m, kw, a.n, ctrls); B, _, fb, _ = run_sim(m, kw, a.n, ctrls)
        E, _, fe, _ = run_sim(m, kw, a.n, ctrls, capture=False)
        C = run_c(m, kw["substeps"], ctrls)
        tp, tpm = throughput(m, kw, a.ntp)
        floor = [(mx(qa, qb), mx(va, vb)) for (qa, va), (qb, vb) in zip(A, B)]
        eager = [(mx(qa, qe), mx(va, ve)) for (qa, va), (qe, ve) in zip(A, E)]
        vsc = [max(mx(A[k][0][w].astype(np.float64), C[w][k]) for w in range(len(C))) for k in range(len(CHECK))]
        # free-joint quaternions: compare as-is (both sides normalise the same way)
        s = dict(info, finite=fa and fb and fe, floor=floor, eager=eager, vs_c=vsc, tp_best=tp, tp_median=tpm,
                 substeps=kw["substeps"])
        summary[name] = s
        save[f"{name}_qpos"] = np.stack([q for q, _ in A]); save[f"{name}_qvel"] = np.stack([v for _, v in A])
        save[f"{name}_vsc"] = np.array(vsc)
        print(f"--- {name}: nv {info['nv']} qLD_block_total {info['qLD_block_total']} sparse-M dofs {info['sparse_M']} "
              f"overflow {info['overflow']} finite {s['finite']}", flush=True)
        print("   floor (qpos/qvel)   " + "  ".join(f"{k}:{q:.1e}/{v:.1e}" for k, (q, v) in zip(CHECK, floor)))
        print("   replay vs eager     " + "  ".join(f"{k}:{q:.1e}/{v:.1e}" for k, (q, v) in zip(CHECK, eager)))
        print("   vs MuJoCo C (qpos)  " + "  ".join(f"{k}:{e:.1e}" for k, e in zip(CHECK, vsc)))
        print(f"   throughput {a.ntp} envs x {kw['substeps']} substeps: {tp:,.0f} env-steps/s best of 3 (median {tpm:,.0f}); "
              f"{tp * kw['substeps']:,.0f} physics steps/s", flush=True)
    np.savez_compressed(a.out, summary=json.dumps(summary), config=a.config, **save)
    if a.ref:
        z = np.load(a.ref); rs = json.loads(str(z["summary"]))
        print(f"=== {a.config} vs reference {z['config']} (graph replay, same ctrl sequence)")
        for name in names:
            if f"{name}_qpos" not in z:
                continue
            dq = [mx(x, y) for x, y in zip(save[f"{name}_qpos"], z[f"{name}_qpos"])]
            dv = [mx(x, y) for x, y in zip(save[f"{name}_qvel"], z[f"{name}_qvel"])]
            fl = [max(p[0], q[0]) for p, q in zip(summary[name]["floor"], rs[name]["floor"])]
            print(f"--- {name}: config diff qpos " + "  ".join(f"{k}:{q:.1e}" for k, q in zip(CHECK, dq))
                  + " | qvel " + "  ".join(f"{k}:{v:.1e}" for k, v in zip(CHECK, dv)))
            print(f"   floor (max of both) " + "  ".join(f"{k}:{q:.1e}" for k, q in zip(CHECK, fl)))
            print(f"   vs C: {a.config} " + " ".join(f"{e:.1e}" for e in summary[name]["vs_c"]) + f" | {z['config']} "
                  + " ".join(f"{e:.1e}" for e in rs[name]["vs_c"]))
            print(f"   throughput {a.config} {summary[name]['tp_best']:,.0f} vs {z['config']} {rs[name]['tp_best']:,.0f} env-steps/s "
                  f"({summary[name]['tp_best'] / rs[name]['tp_best'] - 1:+.1%}; medians {summary[name]['tp_median']:,.0f} / {rs[name]['tp_median']:,.0f})")
    print("=== done")


if __name__ == "__main__":
    main()

"""Which of the G1 throughput fast paths a scene actually takes, and what it measures at N envs.

The three paths (docs/research/mjwarp_throughput_2026-09-25.md §5, all Metal-only) and their conditions:
  register Cholesky   Warp fork: tile_cholesky in one SIMD group without workspaces, nv <= wp.config.metal_register_cholesky_max
                      (48); MuJoCo Warp launches the single-tile Newton Cholesky off CUDA for nv <= MJW_METAL_DENSE_CHOL_MAX (64)
  M layout            BatchSimOptions.m_dense_max (32): trees above it factor M with the one-thread-per-world sparse L'DL,
                      smaller ones with the dense tile
  fused Hessian       incremental Hessian update fused into the Cholesky launch: Newton solver, non-elliptic cone
                      (MuJoCo Warp's incremental path is pyramidal-only), dense constraint Jacobian, nv <= 64

Throughput uses fast_factorization_scenes.throughput (fixed random ctrl, 5 warm-up, 3 x 50 steps, median) with the
scene's own BatchSimOptions; --kernels adds the eager per-kernel GPU time of one env step (WP_METAL_PROFILE=1).

usage: python scripts/diagnostics/g1_tp_generality.py SCENE [N] [--cone pyramidal|elliptic] [--kernels] [--reps 3]
  SCENE: a key of fast_factorization_scenes.scenes() (so101_lift, panda, go1, tron1_wf, ...) or go2 (menagerie scene, 10 substeps)"""
import os, sys, time, json
import numpy as np, mujoco, warp as wp
wp.config.quiet = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fast_factorization_scenes as ffs
from metalsim.physics.batch import BatchSim, BatchSimOptions

args = [a for i, a in enumerate(sys.argv[1:], 1) if not a.startswith("--") and sys.argv[i - 1] not in ("--cone", "--reps")]
SCENE = args[0]
N = int(args[1]) if len(args) > 1 else 4096
CONE = sys.argv[sys.argv.index("--cone") + 1] if "--cone" in sys.argv else None
REPS = int(sys.argv[sys.argv.index("--reps") + 1]) if "--reps" in sys.argv else 3
KERNELS = "--kernels" in sys.argv
if KERNELS and os.environ.get("WP_METAL_PROFILE") != "1":
    sys.exit("--kernels needs WP_METAL_PROFILE=1 in the environment (read at load time)")

sc = ffs.scenes()
if SCENE == "go2":
    men = os.path.join(ffs.ROOT, "upstream", "mujoco_menagerie")
    sc["go2"] = (lambda: mujoco.MjModel.from_xml_path(os.path.join(men, "unitree_go2", "scene.xml")), dict(substeps=10))
factory, kw = sc[SCENE][0], dict(sc[SCENE][1])
m = factory()
if CONE:
    m.opt.cone = {"pyramidal": mujoco.mjtCone.mjCONE_PYRAMIDAL, "elliptic": mujoco.mjtCone.mjCONE_ELLIPTIC}[CONE]
cone = {0: "pyramidal", 1: "elliptic"}[int(m.opt.cone)]
tag = f"{SCENE} {cone}"

sim = BatchSim(m, N, options=BatchSimOptions(capture=not KERNELS, **kw))
wm = sim.m
reg_max = int(getattr(wp.config, "metal_register_cholesky_max", 40))
dense_chol_max = int(os.environ.get("MJW_METAL_DENSE_CHOL_MAX", "64"))
newton = int(wm.opt.solver) == int(mujoco.mjtSolver.mjSOL_NEWTON)
sparse_M = bool((wm.qLD_block_adr.numpy() == -1).any()) if hasattr(wm, "qLD_block_adr") else None
paths = dict(
    nv=int(m.nv), nu=int(m.nu), ngeom=int(m.ngeom), cone=cone, solver="newton" if newton else "cg", jacobian="sparse" if wm.is_sparse else "dense",
    iterations=int(wm.opt.iterations), ls_iterations=int(wm.opt.ls_iterations), substeps=kw.get("substeps", 1),
    register_cholesky=bool(newton and m.nv <= min(reg_max, dense_chol_max)),
    single_tile_cholesky=bool(newton and m.nv <= dense_chol_max),
    M_layout="sparse L'DL" if sparse_M else "dense tile",
    incremental_hessian=bool(newton and cone != "elliptic"),
    fused_hessian_cholesky=bool(newton and cone != "elliptic" and not wm.is_sparse and m.nv <= dense_chol_max
                                and os.environ.get("MJW_METAL_FUSE_H_CHOLESKY", "1") != "0"),
)
print(f"[{tag}] N={N} paths {json.dumps(paths)}", flush=True)
del sim

if not KERNELS:
    best = []
    ffs_tp = ffs.throughput
    for _ in range(REPS):
        best.append(ffs_tp(m, kw, N))
    mx, med = max(b[0] for b in best), float(np.median([b[1] for b in best]))
    sub = kw.get("substeps", 1)
    print(f"[{tag}] throughput {med:,.0f} env-steps/s median ({mx:,.0f} max; {REPS} builds x 3 x 50 steps) = {med*sub:,.0f} physics steps/s "
          f"({sub} substeps of {m.opt.timestep*1e3:g} ms)", flush=True)
    print(f"[{tag}] SUMMARY nv {m.nv} {cone} register {paths['register_cholesky']} M {paths['M_layout']} fused {paths['fused_hessian_cholesky']} "
          f"| {med:,.0f} env-steps/s | {med*sub:,.0f} physics steps/s", flush=True)
    sys.exit(0)

# per-kernel attribution: eager launches, one command buffer per dispatch
sim = BatchSim(m, N, options=BatchSimOptions(capture=False, **kw))
d0 = ffs.init_state(m)
sim.set_state(np.tile(d0.qpos, (N, 1)), np.tile(d0.qvel, (N, 1)))
sim.d.ctrl.assign(ffs.ctrl_seq(m, N, 1, seed=1)[0])
for _ in range(5):
    sim.step()
sim.synchronize()
core = wp._src.context.runtime.core
core.wp_metal_profile_report()          # clear
K = 3
for _ in range(K):
    sim.step()
sim.synchronize(); time.sleep(0.2)
rep = core.wp_metal_profile_report().decode()
rows, tot = [], 0.0
for ln in rep.strip().splitlines():
    f = ln.split(None, 3)
    ms = float(f[0]) / K; tot += ms; rows.append((ms, int(f[2]) // K, f[3]))
rows.sort(reverse=True)
chol = sum(r[0] for r in rows if "cholesky" in r[2].lower())
hess = sum(r[0] for r in rows if "jtdaj" in r[2].lower() or "h_incremental" in r[2].lower())
ldl = sum(r[0] for r in rows if "factor_i" in r[2] or "solve_LD" in r[2] or "_factor_" in r[2])
print(f"[{tag}] KERNELS {tot:.2f} ms GPU per env step (sum of per-dispatch times, {len(rows)} kernels, {sum(r[1] for r in rows)} dispatches); "
      f"Newton Cholesky {chol:.2f} ms ({chol/tot:.0%}), Hessian assembly {hess:.2f} ms ({hess/tot:.0%}), M factor/solve {ldl:.2f} ms ({ldl/tot:.0%})", flush=True)
for ms, cnt, kname in rows[:14]:
    print(f"   {ms:8.3f} ms  x{cnt:4d}  {kname}")

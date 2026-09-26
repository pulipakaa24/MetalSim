"""Physics-difference protocol for the elliptic-cone solver variants (Go2 / G1 Menagerie scenes, same model,
controller and solver budget as metalsim_step.py, CONE=elliptic|pyramidal):

  1. floor: two graph-replay instances of the same configuration in one process (MuJoCo Warp's atomics order
     contacts and rows nondeterministically, so identical configurations already differ by float noise)
  2. this configuration vs a reference file (--ref, written by --out from another process / worktree / mode)
  3. MuJoCo C oracle: 4 worlds of PD hold + a fixed random target sequence vs mj_step (the same MjModel),
     |dq| max / median at several times (a variant must stay within the reference's envelope)
  4. capacity: nefc / nacon maxima and overflow flags

usage: [CONE=elliptic] [MJW_JTCJ_MODE=world] python elliptic_check.py go2|g1 [N] [STEPS] [--out F.npz] [--ref BASE.npz]"""
import json, os, sys, time
import numpy as np, torch, mujoco, warp as wp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from metalsim.physics.batch import BatchSim, BatchSimOptions
import mujoco_warp

args = [a for a in sys.argv[1:] if not a.startswith("--")]
name = args[0]; N = int(args[1]) if len(args) > 1 else 512; S = int(args[2]) if len(args) > 2 else 200
OUT = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None
REF = sys.argv[sys.argv.index("--ref") + 1] if "--ref" in sys.argv else None
R = ROBOTS[name]
SNAP = sorted(set([1, 2, 4, 10, 25, 50, 100] + [S]))


def build():
    spec = mujoco.MjSpec.from_file(R["scene"])
    LIM = force_limits(mujoco.MjModel.from_xml_path(R["scene"])) if R["kp"] is not None else []
    for a, (lo, hi) in zip(spec.actuators, LIM):
        a.set_to_position(kp=R["kp"], kv=R["kd"])
        a.ctrllimited = mujoco.mjtLimited.mjLIMITED_FALSE
        a.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE; a.forcerange = [lo, hi]
    spec.option.timestep = DT
    if os.environ.get("CONE"):
        spec.option.cone = {"pyramidal": mujoco.mjtCone.mjCONE_PYRAMIDAL, "elliptic": mujoco.mjtCone.mjCONE_ELLIPTIC}[os.environ["CONE"]]
    spec.option.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_MULTICCD)
    spec.option.iterations, spec.option.ls_iterations = 10, 20      # the "matched" budget, also in the MuJoCo C oracle
    for gm in spec.geoms:
        gm.margin = 0.0
    return spec.compile()


m = build()
home = m.key_qpos[0].copy() if m.nkey else m.qpos0.copy()
qadr = [m.jnt_qposadr[m.actuator_trnid[i, 0]] for i in range(m.nu)]
rng = np.random.default_rng(0)
targets = [(home[qadr] + AMP * rng.uniform(-1, 1, (N, m.nu))).astype(np.float32) for _ in range(S // RESAMPLE + 1)]


class Runner:
    def __init__(self, n):
        self.sim = BatchSim(m, n, options=BatchSimOptions(solver_iterations=10, ls_iterations=20, jacobian=os.environ.get("JAC"), **({"njmax": int(os.environ["NJMAX"])} if os.environ.get("NJMAX") else {})))
        self.sim.set_state(home); self.sim.d.ctrl.assign(targets[0][:n]); self.sim.forward(); self.sim.synchronize()
        self.nefc_max = 0; self.nacon_max = 0

    def run(self, steps):
        out = []
        for k in range(steps):
            if k % RESAMPLE == 0:
                # write ctrl through Warp after the previous replay has finished: a torch (MPS) copy into the shared
                # buffer is on another command queue and can land while the previous step's graph still runs
                self.sim.synchronize(); self.sim.d.ctrl.assign(targets[k // RESAMPLE][: self.sim.n])
            self.sim.step()
            if k + 1 in SNAP:
                self.sim.synchronize(); d = self.sim.d
                self.nefc_max = max(self.nefc_max, int(d.nefc.numpy().max())); self.nacon_max = max(self.nacon_max, int(d.nacon.numpy()[0]))
                out.append((k + 1, d.qpos.numpy().copy(), d.qvel.numpy().copy()))
        return out


def diff(a, b):
    return [(k, float(np.abs(qa - qb).max()), float(np.median(np.abs(qa - qb))), float(np.abs(va - vb).max()), float(np.mean(np.abs(qa - qb).max(axis=1) > 1e-3)))
            for (k, qa, va), (_, qb, vb) in zip(a, b)]


def show(title, rows):
    print(f"--- {title}")
    for k, mx, md, vx, frac in rows:
        print(f"   step {k:4d}: qpos max {mx:.2e} median {md:.1e} | qvel max {vx:.2e} | worlds with |dq| > 1e-3: {100*frac:.1f} %")


def mujoco_c_protocol(sim, T=100):
    rng = np.random.default_rng(7)
    tg = [home[qadr] + 0.5 * rng.uniform(-0.5, 0.5, m.nu) for _ in range(T)]
    d = mujoco.MjData(m); (mujoco.mj_resetDataKeyframe(m, d, 0) if m.nkey else None)
    # state written through Warp (torch/MPS writes into the shared buffers are on another command queue)
    sim.synchronize(); sim.set_state(home); sim.d.qvel.zero_(); sim.d.qacc_warmstart.zero_(); sim.forward(); sim.synchronize()
    out = []
    for t in range(T):
        d.ctrl[:] = tg[t]
        sim.synchronize(); sim.d.ctrl.assign(np.tile(tg[t], (sim.n, 1)).astype(np.float32))
        sim.step(); mujoco.mj_step(m, d)
        if t + 1 in (1, 2, 4, 12, 25, 50, 100):
            sim.synchronize(); q = sim.d.qpos.numpy().astype(np.float64)
            e = np.abs(q[:4] - d.qpos[None]); out.append((t + 1, float(e.max()), float(np.median(e))))
    return out


mode = os.environ.get("MJW_JTCJ_MODE", "(default)")
print(f"=== {name} cone={'elliptic' if m.opt.cone else 'pyramidal'} N={N} S={S} MJW_JTCJ_MODE={mode} mujoco_warp={os.path.dirname(mujoco_warp.__file__)}", flush=True)
ra, rb = Runner(N), Runner(N)
print(f"njmax {ra.sim.d.njmax} naconmax {ra.sim.d.naconmax} sparse {ra.sim.m.is_sparse} JAC={os.environ.get('JAC')} nv {m.nv} ndof_tri {ra.sim.m.dof_tri_row.size}", flush=True)
qa, qb = ra.run(S), rb.run(S)
show("floor: two graph-replay instances of this configuration", diff(qa, qb))
for nm, r in (("replay", ra), ("replay 2", rb)):
    print(f"capacity {nm}: nefc max {r.nefc_max} (njmax {r.sim.d.njmax}), nacon max {r.nacon_max} (naconmax {r.sim.d.naconmax}); overflow {r.sim.overflow_flags()}")
proto = mujoco_c_protocol(ra.sim)
print("--- MuJoCo C oracle (4 worlds: PD hold + fixed random targets; |dq| max / median)")
print("   " + "  ".join(f"t={k*DT:.3f}s {mx:.2e}/{md:.1e}" for k, mx, md in proto), flush=True)
if OUT:
    np.savez_compressed(OUT, steps=np.array([k for k, *_ in qa]), qpos=np.stack([q for _, q, _ in qa]), qvel=np.stack([v for _, _, v in qa]),
                        proto=np.array(proto), label=f"{name} {mode} JAC={os.environ.get('JAC')} {os.path.dirname(mujoco_warp.__file__)}")
if REF:
    z = np.load(REF)
    ref = [(int(k), q, v) for k, q, v in zip(z["steps"], z["qpos"], z["qvel"])]
    show(f"this configuration vs reference [{z['label']}] (graph replay)", diff(qa, ref))
    print("   reference MuJoCo C oracle: " + "  ".join(f"t={k*DT:.3f}s {mx:.2e}/{md:.1e}" for k, mx, md in z["proto"]))
print("=== done")

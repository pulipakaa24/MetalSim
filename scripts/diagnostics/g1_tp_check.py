"""Correctness checks for a throughput variant of the G1 flat task (MuJoCo Warp, 2.5 ms), against the
committed configuration and against MuJoCo C.

  1. replay == eager: the full env step as one replayed graph vs the same kernels launched eagerly
     (same config, same deterministic actions), max |dq| after S control steps
  2. variant vs baseline: the same env-step loop under MJW_TP_VARIANT vs the committed config, compared
     with the run-to-run floor (baseline vs a second baseline instance: MuJoCo Warp's atomics order
     contacts and constraint rows nondeterministically, so identical configs already differ by float noise)
  3. physics protocol vs MuJoCo C: 4 worlds of PD hold at the init keyframe plus a fixed random
     target sequence (8 substeps per control step), max |dq| vs mj_step at 0.25 / 0.5 / 1 s, for baseline
     and variant (the variant must stay within the baseline's error envelope)
  4. capacity: nefc / nacon maxima and overflow flags over the checks

usage: MJW_TP_VARIANT='{"njmax":128}' python scripts/diagnostics/g1_tp_check.py [N] [STEPS] [--out F.npz] [--ref BASE.npz]"""
import json, os, sys
import numpy as np, torch, mujoco, warp as wp
wp.config.quiet = True
import mujoco_warp as mjw
import metalsim.learn.g1_velocity as g1v
from metalsim.physics.batch import BatchSimOptions
from metalsim.learn.warp_policy import RolloutBuffers, bump

VAR = json.loads(os.environ.get("MJW_TP_VARIANT", "{}") or "{}")
args = [a for a in sys.argv[1:] if not a.startswith("--")]
N = int(args[0]) if args else 512
S = int(args[1]) if len(args) > 1 else 50
_cur = {}


def _opts(**kw):
    for k in ("njmax", "nconmax", "jacobian", "block_dim", "m_dense_max", "metal_register_cholesky_max"):
        if k in _cur:
            kw[k] = _cur[k]
    return BatchSimOptions(**kw)


g1v.BatchSimOptions = _opts
_kin = mjw.kinematics


def make(var):
    _cur.clear(); _cur.update(var)
    t = g1v.G1VelocityTask(N, terrain=os.environ.get("MJW_TP_TERRAIN", "flat"), physics_dt=0.0025, seed=0)
    t.reset_all()
    return t


class Runner:
    """Env-step loop of benchmark_step with deterministic actions (a fixed smooth pattern per env)."""

    def __init__(self, task, var, capture=True):
        self.task, self.var, dev = task, var, task.device

        class _Pol:
            step_idx = wp.zeros(1, dtype=int, device=dev)
        self.pol = _Pol()
        self.bufs = RolloutBuffers(1, N, task.obs_dim, task.act_dim)
        self.bufs.rew = wp.zeros((1, N), dtype=float, device=dev); self.bufs.done = wp.zeros((1, N), dtype=float, device=dev)
        rng = np.random.default_rng(123)
        self.actions = [wp.array(rng.uniform(-1, 1, (N, task.act_dim)).astype(np.float32), dtype=float, device=dev) for _ in range(4)]
        self.graphs = None
        if capture:
            self.graphs = []
            for a in self.actions:
                with wp.ScopedDevice(dev), wp.ScopedCapture(device=dev) as cap:
                    self.body(a)
                self.graphs.append(cap.graph)
        self.nefc_max = 0; self.nacon_max = 0

    def body(self, a):
        task = self.task
        mjw.kinematics = (lambda m, d: None) if self.var.get("no_kin") else _kin
        try:
            wp.launch(bump, dim=1, inputs=[self.pol.step_idx], device=task.device)
            task.launch_apply_action(a)
            task.sim.launch_step()
            task.launch_reward_done_reset(self.pol, self.bufs)
            task.launch_obs(self.pol.step_idx)
        finally:
            mjw.kinematics = _kin

    def run(self, steps):
        qs = []
        for k in range(steps):
            if self.graphs is not None:
                wp.capture_launch(self.graphs[k % 4])
            else:
                with wp.ScopedDevice(self.task.device):
                    self.body(self.actions[k % 4])
            if k in (0, 1, 4) or k % 10 == 9 or k == steps - 1:
                self.task.sim.synchronize()
                d = self.task.sim.d
                self.nefc_max = max(self.nefc_max, int(d.nefc.numpy().max())); self.nacon_max = max(self.nacon_max, int(d.nacon.numpy()[0]))
                qs.append((k + 1, d.qpos.numpy().copy(), self.task.obs.numpy().copy(), self.bufs.rew.numpy().copy()))
        return qs


def diff(a, b):
    return [(k, float(np.abs(qa - qb).max()), float(np.median(np.abs(qa - qb))), float(np.abs(oa - ob).max()), float(np.abs(ra - rb).max()))
            for (k, qa, oa, ra), (_, qb, ob, rb) in zip(a, b)]


def show(title, rows):
    print(f"--- {title}")
    for k, mx, md, ox, rx in rows:
        print(f"   step {k:4d}: qpos max {mx:.2e} median {md:.1e} | obs max {ox:.2e} | rew max {rx:.2e}")


def mujoco_c_protocol(task, T=50):
    """PD hold at the init keyframe + a fixed random target sequence; 4 worlds vs mj_step."""
    m = task.model; sim = task.sim; n = task.n
    rng = np.random.default_rng(7)
    targets = [m.key_qpos[0][7:] + 0.5 * rng.uniform(-0.5, 0.5, m.nu) for _ in range(T)]
    d = mujoco.MjData(m); mujoco.mj_resetDataKeyframe(m, d, 0)
    q0 = np.tile(m.key_qpos[0], (n, 1)).astype(np.float32); q0[:, :2] += task.origins.numpy()[:, :2]
    sim.t.qpos.copy_(torch.as_tensor(q0)); sim.t.qvel.zero_(); sim.t.qacc_warmstart.zero_()
    v = sim.forward(); sim.after(v); sim.synchronize()
    out = []
    for t in range(T):
        d.ctrl[:] = targets[t]
        sim.t.ctrl.copy_(torch.as_tensor(np.tile(targets[t], (n, 1)).astype(np.float32)))
        torch.mps.synchronize()
        sim.step()
        for _ in range(task.decimation):
            mujoco.mj_step(m, d)
        if t + 1 in (1, 2, 4, 12, 25, 50):
            sim.synchronize()
            q = sim.d.qpos.numpy().astype(np.float64); q[:, :2] -= task.origins.numpy()[:, :2]
            e = np.abs(q[:4] - d.qpos[None])
            out.append((t + 1, float(e.max()), float(np.median(e))))
    return out


OUT = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None
REF = sys.argv[sys.argv.index("--ref") + 1] if "--ref" in sys.argv else None
print(f"=== variant {VAR or 'baseline'}; N={N}, {S} control steps, deterministic actions", flush=True)
# Some knobs are process-wide (warp.config, module builds), so each configuration runs in its own process:
# two graph-replay instances (the run-to-run floor of this configuration), one eager instance, the MuJoCo C
# protocol; the snapshots are saved (--out) and compared with a reference configuration's file (--ref).
a = make(VAR); b = make(VAR)
print(f"njmax {a.sim.d.njmax} naconmax {a.sim.d.naconmax} sparse {a.sim.m.is_sparse}", flush=True)
ra, rb = Runner(a, VAR), Runner(b, VAR)
qa, qb = ra.run(S), rb.run(S)
show("floor: two graph-replay instances of this configuration", diff(qa, qb))
e = make(VAR); re_ = Runner(e, VAR, capture=False); qe = re_.run(S)
show("graph replay vs eager launches (same configuration)", diff(qa, qe))
for name, r in (("replay", ra), ("replay 2", rb), ("eager", re_)):
    print(f"capacity {name}: nefc max {r.nefc_max} (njmax {r.task.sim.d.njmax}), nacon max {r.nacon_max} "
          f"(naconmax {r.task.sim.d.naconmax}); overflow {r.task.sim.overflow_flags()}")
print("--- physics protocol vs MuJoCo C (4 worlds: PD hold + fixed random targets; |dq| max / median)")
proto = mujoco_c_protocol(a)
print("   " + "  ".join(f"t={k*0.02:.2f}s {mx:.2e}/{md:.1e}" for k, mx, md in proto), flush=True)
if OUT:
    np.savez_compressed(OUT, steps=np.array([k for k, *_ in qa]), qpos=np.stack([q for _, q, _, _ in qa]),
                        obs=np.stack([o for _, _, o, _ in qa]), rew=np.stack([r for _, _, _, r in qa]),
                        proto=np.array(proto), variant=json.dumps(VAR))
if REF:
    z = np.load(REF)
    ref = [(int(k), q, o, r) for k, q, o, r in zip(z["steps"], z["qpos"], z["obs"], z["rew"])]
    show(f"this configuration vs reference {z['variant']} (graph replay)", diff(qa, ref))
    print("   reference MuJoCo C protocol: " + "  ".join(f"t={k*0.02:.2f}s {mx:.2e}/{md:.1e}" for k, mx, md in z["proto"]))
print("=== done")

"""G1 checks on Newton XPBD for the solver-level fixes in the Newton fork (upstream/newton, branch metalsim).
Run with either environment: .venv (pinned upstream Newton 45458023 = "before") or .venv-newtonfork (the fork =
"after"); the Newton path/commit is printed first.

  energy  : passive energy (drives off, zero gravity, floating, random initial joint velocities N(0, 2) rad/s,
            2 envs, 2 s at 50 Hz): kinetic energy after 2 s / initial (> 1 = energy injected). As
            scripts/diagnostics/newton_drift_remedies.py.
  drift   : feet vs FK of the reported joint angles under random actions N(0, 1) x 0.5 rad, 32 envs, 100 control
            steps, upright envs only: p50 / p99 / max [mm] (as newton_drift_remedies.py).
  drop    : 8 G1s dropped from 1 m under the PD hold, 1.5 s: max penetration of any box collider [cm], finite.
  all     : energy, drift, drop.
  stand   : 4 G1s PD-holding Isaac's init pose on flat ground, 2 s: pelvis z every 0.25 s and max |q - q_MuJoCoC| per
            joint group at 0.5 s (MuJoCo C, same model and gains, 2.5 ms).
  sigma3  : CPU version of g1_preflight.stability: N envs, S control steps of targets default + 0.5 * N(0, 3);
            an env whose torso touches (> 1 N) is reset (default pose, random yaw), one whose qpos/qvel is non-finite
            or > 1000 is counted as blown up and reset. Reports blown-up episodes, falls and peak |joint speed|.

Options: --noclamp keeps the USD's revolute limits (elbow pitch up to 3.421 rad) instead of NewtonSim's clamp to
+-(pi - 0.15); --it/--dt_ms solver settings; --relax joint relaxation; --kw k=v extra SolverXPBD kwargs.

usage: python scripts/diagnostics/newton_fork/g1_checks.py energy|sigma3 [--noclamp] [--n 256] [--steps 400]"""
import argparse, sys, time, numpy as np, warp as wp
wp.config.quiet = True
import newton
from metalsim.learn.g1_velocity import build_g1_model
from metalsim.physics import newton_backend as nb
import metalsim.interop.warp_metal as wm

ap = argparse.ArgumentParser()
ap.add_argument("check", choices=("energy", "sigma3", "drift", "drop", "all", "stand"))
ap.add_argument("--drive", default="actuator", choices=("actuator", "solver"),
                help="actuator: NewtonSim's ActuatorPD (joint_f); solver: the model's ke/kd solved by SolverXPBD (fork)")
ap.add_argument("--noclamp", action="store_true"); ap.add_argument("--n", type=int, default=256)
ap.add_argument("--steps", type=int, default=400); ap.add_argument("--it", type=int, default=4)
ap.add_argument("--dt_ms", type=float, default=1.25); ap.add_argument("--relax", type=float, default=0.4)
ap.add_argument("--lin", type=float); ap.add_argument("--ang", type=float)
ap.add_argument("--kw", nargs="*", default=[]); ap.add_argument("--device", default="cpu"); ap.add_argument("--seed", type=int, default=1)
a = ap.parse_args()
DEV = a.device
if DEV == "cpu":
    class _E:
        def __init__(self, *x): self.v = 0
        def next_value(self): self.v += 1; return self.v
    wm.SharedEvent = _E
if a.noclamp:                       # NewtonSim has no switch for the limit clamp: patch the builder default
    _gb = nb.g1_builder
    nb.g1_builder = lambda *x, **k: _gb(*x, **{**k, "limit_margin": None})
if a.lin is not None or a.ang is not None:   # NewtonSim sets both factors to --relax: override per factor
    _X = newton.solvers.SolverXPBD
    class _XR(_X):
        def __init__(self, m, **k):
            if a.lin is not None: k["joint_linear_relaxation"] = a.lin
            if a.ang is not None: k["joint_angular_relaxation"] = a.ang
            super().__init__(m, **k)
    newton.solvers.SolverXPBD = _XR
skw = {}
import inspect as _insp
if "joint_armature_inertia" in _insp.signature(newton.solvers.SolverXPBD.__init__).parameters:
    skw["joint_armature_inertia"] = "none"     # NewtonSim's builder already adds the armature to the link inertia
for kv in a.kw:
    k, v = kv.split("="); skw[k] = eval(v)
import importlib.metadata as md, json, os
try:
    du = json.loads(md.distribution("newton").read_text("direct_url.json") or "{}")
except Exception:
    du = {}
print(f"newton {md.version('newton')} at {os.path.dirname(newton.__file__)} {du.get('vcs_info', du.get('dir_info', ''))}")
M_MJ = build_g1_model("flat", physics_dt=0.0025)[0]


@wp.kernel
def _target_to_coord(target: wp.array[float], dof: wp.array[int], coord: wp.array[int], out: wp.array[float]):
    i = wp.tid()
    out[coord[i]] = target[dof[i]]


def make(n):
    kw = dict(skw)
    if a.drive == "solver":
        kw.setdefault("joint_drive_mode", "pd")
    sim = nb.NewtonSim(M_MJ, n, iterations=a.it, dt=a.dt_ms * 1e-3, device=DEV, relaxation=a.relax, solver_kw=kw or None)
    if a.drive == "solver":
        # the solver's own drive with Isaac's gains; ActuatorPD's gains zeroed (it then writes no joint_f), and its
        # per-DOF targets (written from ctrl) copied into Control.joint_target_q (coordinate layout) every substep
        m, act = sim.model, sim.actuator
        m.joint_target_ke.assign(act.kp.numpy()); m.joint_target_kd.assign(act.kd.numpy())
        act.kp.zero_(); act.kd.zero_()
        qs, qds, jt = m.joint_q_start.numpy(), m.joint_qd_start.numpy(), m.joint_type.numpy()
        rev = [j for j in range(m.joint_count) if qds[j + 1] - qds[j] == 1]
        dof = wp.array([int(qds[j]) for j in rev], dtype=int, device=DEV); coord = wp.array([int(qs[j]) for j in rev], dtype=int, device=DEV)
        _apply = act.apply
        def apply(state, control):
            _apply(state, control)                                  # eval_ik (joint_q/qd for qacc); no torque
            wp.launch(_target_to_coord, dim=len(rev), inputs=[act.target, dof, coord], outputs=[control.joint_target_q], device=DEV)
        act.apply = apply
    return sim


def place(sim, n, z=None, qvel=None, mask=None, rng=None):
    q = np.tile(M_MJ.key_qpos[0], (n, 1)).astype(np.float32)
    side = int(np.ceil(np.sqrt(n))); q[:, 0] = (np.arange(n) % side) * 2.5; q[:, 1] = (np.arange(n) // side) * 2.5
    if rng is not None:
        yaw = rng.uniform(-np.pi, np.pi, n); q[:, 3] = np.cos(yaw / 2); q[:, 4:6] = 0; q[:, 6] = np.sin(yaw / 2)
    if z is not None: q[:, 2] = z
    qv = np.zeros((n, M_MJ.nv), np.float32) if qvel is None else qvel.astype(np.float32)
    if mask is not None:                      # keep the other envs' state
        q0 = sim.d.qpos.numpy(); v0 = sim.d.qvel.numpy(); q0[mask] = q[mask]; v0[mask] = qv[mask]; q, qv = q0, v0
        sim._reset_mask.assign(mask)
    else:
        sim._reset_mask.fill_(True)
    sim.d.qpos.assign(q); sim.d.qvel.assign(qv); sim.launch_reset()


def kinetic(sim):
    M = sim.model; bq = sim.s0.body_q.numpy(); bqd = sim.s0.body_qd.numpy(); m = M.body_mass.numpy(); I = M.body_inertia.numpy()
    e = 0.0
    for b in range(M.body_count):
        x, y, z, w = bq[b, 3:7]
        R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                      [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
        v, om = bqd[b, :3], bqd[b, 3:]; ob = R.T @ om
        e += 0.5 * m[b] * v @ v + 0.5 * ob @ I[b] @ ob
    return e


def energy(n=2, T=2.0):
    sim = make(n); M = sim.model
    M.gravity.zero_(); sim.actuator.gravity = wp.vec3(0.0, 0.0, 0.0)
    sim.actuator.kp.zero_(); sim.actuator.kd.zero_(); M.joint_target_ke.zero_(); M.joint_target_kd.zero_()
    qv = np.zeros((n, M_MJ.nv)); qv[:, 6:] = np.random.default_rng(a.seed).normal(0, 2.0, (n, M_MJ.nu))
    place(sim, n, z=3.0, qvel=qv)
    e0 = kinetic(sim); peak = 0.0; ratios = []
    for _ in range(int(T / 0.02)):
        sim.launch_step()
        ratios.append(kinetic(sim) / e0); peak = max(peak, float(np.abs(sim.d.qvel.numpy()[:, 6:]).max()))
    return ratios[-1], max(ratios), peak


def sigma3(n, steps, amp=3.0):
    sim = make(n); rng = np.random.default_rng(a.seed); place(sim, n, rng=rng)
    adr = int(M_MJ.sensor_adr[[i for i in range(M_MJ.nsensor) if M_MJ.sensor(i).name == "torso_link_touch"][0]])
    blown = falls = 0; peak = 0.0; t0 = time.time()
    for t in range(steps):
        sim.d.ctrl.assign((M_MJ.key_qpos[0][7:] + 0.5 * amp * rng.normal(size=(n, M_MJ.nu))).astype(np.float32))
        sim.launch_step()
        qp, qv, sd = sim.d.qpos.numpy(), sim.d.qvel.numpy(), sim.d.sensordata.numpy()
        bad = ~np.isfinite(qp).all(1) | ~np.isfinite(qv).all(1) | (np.abs(qp) > 1000).any(1) | (np.abs(qv) > 1000).any(1)
        fell = (sd[:, adr] > 1.0) & ~bad
        ok = ~bad
        if ok.any(): peak = max(peak, float(np.abs(qv[ok, 6:]).max()))
        blown += int(bad.sum()); falls += int(fell.sum())
        done = bad | fell
        if done.any():
            place(sim, n, mask=done, rng=rng)
    return blown, falls, peak, time.time() - t0


rl = f"{a.lin if a.lin is not None else a.relax}/{a.ang if a.ang is not None else a.relax}"
def drift(n=32, steps=100, sigma=1.0):
    sim = make(n); M = sim.model; place(sim, n); rng = np.random.default_rng(0)
    fk = M.state(); feet = [int(sim.foot_nb[0]), int(sim.foot_nb[1])]; F = []; alive = np.ones(n, bool)
    for _ in range(steps):
        sim.d.ctrl.assign((M_MJ.key_qpos[0][7:] + 0.5 * sigma * rng.normal(size=(n, M_MJ.nu))).astype(np.float32))
        sim.launch_step()
        newton.eval_fk(M, sim.joint_q, sim.joint_qd, fk)
        bq = sim.s0.body_q.numpy().reshape(n, sim.nb, 7); bf = fk.body_q.numpy().reshape(n, sim.nb, 7)
        if not np.isfinite(bq).all(): return None
        alive &= bq[:, 0, 2] > 0.4
        if not alive.any(): break
        F.append(np.linalg.norm(bq[alive][:, feet, :3] - bf[alive][:, feet, :3], axis=-1).ravel())
    F = np.concatenate(F) * 1e3
    return np.percentile(F, 50), np.percentile(F, 99), F.max()


@wp.kernel
def _lowest(body_q: wp.array[wp.transform], shape_body: wp.array[int], shape_X: wp.array[wp.transform], shape_scale: wp.array[wp.vec3],
            shape_type: wp.array[int], out: wp.array[float]):
    s = wp.tid()
    if shape_type[s] != 7 or shape_body[s] < 0:
        out[s] = 1.0e6
        return
    X = body_q[shape_body[s]] * shape_X[s]; h = shape_scale[s]; z = float(1.0e6)
    for i in range(8):
        c = wp.vec3(wp.where(i % 2 == 0, -h[0], h[0]), wp.where((i // 2) % 2 == 0, -h[1], h[1]), wp.where(i // 4 == 0, -h[2], h[2]))
        z = wp.min(z, wp.transform_point(X, c)[2])
    out[s] = z


def drop(n=8, T=1.5):
    sim = make(n); M = sim.model; place(sim, n, z=1.0)
    zc = wp.zeros(M.shape_count, dtype=float, device=DEV); pen = 0.0
    for _ in range(int(T / 0.02)):
        sim.launch_step()
        wp.launch(_lowest, dim=M.shape_count, inputs=[sim.s0.body_q, M.shape_body, M.shape_transform, M.shape_scale, M.shape_type], outputs=[zc], device=DEV)
        z = zc.numpy()
        if not np.isfinite(z).all(): return None, False
        pen = max(pen, -float(z.min()))
    return pen * 100, bool(np.isfinite(sim.s0.body_q.numpy()).all())


tag = f"{a.it} it / {a.dt_ms} ms, drive {a.drive}, relax {rl}, limits {'USD (elbow 3.421)' if a.noclamp else 'clamped pi-0.15'}{', ' + ' '.join(a.kw) if a.kw else ''}"
def stand(n=4, T=2.0):
    import mujoco
    GROUPS = {"legs": ("hip", "knee", "torso"), "ankles": ("ankle",), "arms": ("shoulder", "elbow"),
              "hands": ("zero", "one", "two", "three", "four", "five", "six")}
    grp = lambda nm: next(g for g, keys in GROUPS.items() if any(k in nm for k in keys))
    m = M_MJ; d = mujoco.MjData(m); mujoco.mj_resetDataKeyframe(m, d, 0); d.ctrl[:] = m.key_qpos[0][7:]
    zmj, qmj = [], None
    for k in range(int(T / 0.25)):
        for _ in range(int(round(0.25 / m.opt.timestep))): mujoco.mj_step(m, d)
        zmj.append(d.qpos[2])
        if k == 1: qmj = d.qpos[7:].copy()
    sim = make(n); place(sim, n); zs, err = [], None
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, m.actuator_trnid[i, 0]) for i in range(m.nu)]
    for k in range(int(T / 0.25)):
        for _ in range(int(round(0.25 / 0.02))): sim.launch_step()
        qp = sim.d.qpos.numpy(); zs.append(float(qp[0, 2]))
        if k == 1:
            dq = np.abs(qp[:, 7:] - qmj).max(0)
            err = {g: max(dq[i] for i in range(m.nu) if grp(names[i]) == g) for g in GROUPS}
    return zmj, zs, err


if a.check == "stand":
    zmj, zs, err = stand()
    print(f"| stand | MuJoCo C | z {np.round(zmj, 3).tolist()} |")
    print(f"| stand | {tag}, drive {a.drive} | z {np.round(zs, 3).tolist()} | dq@0.5s " + ", ".join(f"{g} {v:.3f}" for g, v in err.items()) + " |")
elif a.check == "all":
    fin, mx, pk = energy(); d = drift(); dp, ok = drop()
    ds = "non-finite" if d is None else f"{d[0]:.2f} / {d[1]:.1f} / {d[2]:.1f}"
    print(f"| all | {tag} | drift p50/p99/max {ds} mm | drop pen {'-' if dp is None else f'{dp:.2f}'} cm, finite {ok} | KE(2 s)/KE0 {fin:.3f}, max {mx:.3f} |")
elif a.check == "energy":
    fin, mx, pk = energy()
    print(f"| energy | {tag} | KE(2 s)/KE0 {fin:.3f} | max KE/KE0 {mx:.3f} | peak |qd| {pk:.1f} rad/s |")
elif a.check == "drift":
    d = drift()
    print(f"| drift | {tag} | " + ("non-finite" if d is None else f"{d[0]:.2f} / {d[1]:.1f} / {d[2]:.1f} mm") + " |")
elif a.check == "drop":
    dp, ok = drop()
    print(f"| drop | {tag} | {'-' if dp is None else f'{dp:.2f}'} cm, finite {ok} |")
else:
    b, f, pk, el = sigma3(a.n, a.steps)
    print(f"| sigma3 | {tag} | {a.n} envs x {a.steps} steps | blown {b} | falls {f} | peak |qd| {pk:.0f} rad/s | {el:.0f} s |")

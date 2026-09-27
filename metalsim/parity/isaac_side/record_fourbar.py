"""PhysX four-bar: the closed-loop crank-rocker of scripts/diagnostics/closed_loops/mechanisms.py (``fourbar``), built as
a PhysX articulation with the loop closed by an extra revolute joint marked physics:excludeFromArticulation
(UsdPhysics; "a joint in the loop may use its excludeFromArticulation attribute flag ... and at this point the loop is
then broken"; PhysX 5 Articulations > "Closing loops": "adding rigid-body Joints between articulation links"), then
5 s of free, undamped swing from rest at crank -20 deg (the MuJoCo / exact-reference initial state).

Topology (all hinge axes world +y, gravity -z, no colliders, no contacts):
  world -FixedJoint- ground (articulation root link, fixed base)
  ground -rev "crank" @A(0,0,0)- crank -rev "coupler" @B- coupler        (articulation tree)
  ground -rev "rocker" @D(0.3,0,0)- rocker                                 (articulation tree)
  coupler -rev "loop0" @C- rocker        physics:excludeFromArticulation = true (maximal-coordinate PhysX joint)
Link frames at their pivots rotated to the rod direction (rod along local +x), mass/COM/inertia exactly as
mechanisms._rod (uniform rod, r = 1 cm); joint frames world-aligned at the start pose (coordinate 0 there), as
mechanisms.newton_builder does. Links: linear/angular damping 0, sleep/stabilization thresholds 0.

Records per physics step: body_pos_w, body_quat_w (wxyz), joint_pos/vel (articulation DOFs), body direction angles
in mechanisms.body_order ('crank', 'coupler', 'rocker'), and the loop-closure gap |C_coupler - C_rocker| [m].
Stepping and reading through Isaac Lab 2.3.2's SimulationContext + Articulation (omni.physics.tensors underneath).

    python record_fourbar.py --headless --dt 0.0025 --out ~/parity3/fourbar_il2/dt0.0025 [--device cpu] [--pos_iters 8 --vel_iters 4]
"""
import argparse, os, json, math, sys
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--out", required=True)
parser.add_argument("--dt", type=float, default=0.0025)
parser.add_argument("--T", type=float, default=5.0)
parser.add_argument("--pos_iters", type=int, default=8)   # Isaac Lab G1 / most robot cfgs: 8 position, 4 velocity
parser.add_argument("--vel_iters", type=int, default=4)
parser.add_argument("--solver", default="TGS", choices=["TGS", "PGS"])
parser.add_argument("--no_loop", action="store_true", help="diagnostic: omit the loop joint (open tree: crank-coupler double pendulum + rocker pendulum)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np, torch
from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, Sdf, Gf
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg

dev = args.device if args.device else "cuda:0"
os.makedirs(args.out, exist_ok=True)
G = 9.81; RAD = 0.01
def rod(m, L): return dict(m=m, L=L, Ia=0.5 * m * RAD * RAD, It=m * L * L / 12.0 + 0.25 * m * RAD * RAD)
def d2(th): return np.array([math.cos(th), 0.0, math.sin(th)])
# geometry: identical to mechanisms.fourbar_geometry(theta_crank = -20 deg)
th_c = math.radians(-20.0); A = np.zeros(3); D = np.array([0.30, 0, 0]); a_, b_, c_ = 0.10, 0.30, 0.25
Bp = A + a_ * d2(th_c); dd = np.linalg.norm(D - Bp); ex = (D - Bp) / dd
x = (dd * dd + b_ * b_ - c_ * c_) / (2 * dd); h = math.sqrt(b_ * b_ - x * x); ez = np.array([-ex[2], 0, ex[0]])
C1 = Bp + x * ex + h * ez; C2 = Bp + x * ex - h * ez; C = C1 if C1[2] > C2[2] else C2
ang = lambda P, Q: math.atan2(Q[2] - P[2], Q[0] - P[0])
BODIES = {"crank": (A, th_c, rod(0.2, a_)), "coupler": (Bp, ang(Bp, C), rod(0.6, b_)), "rocker": (D, ang(D, C), rod(0.5, c_))}
def qy(th):   # (w, x, y, z) turning local +x to direction angle th (MuJoCo _quat_y)
    return Gf.Quatf(math.cos(-th / 2), 0.0, math.sin(-th / 2), 0.0)
def local(bn, pw):
    piv, th, _ = BODIES[bn]; d = pw - piv
    return Gf.Vec3f(math.cos(th) * d[0] + math.sin(th) * d[2], 0.0, -math.sin(th) * d[0] + math.cos(th) * d[2])

usd_path = os.path.join(args.out, "fourbar.usda")
st = Usd.Stage.CreateNew(usd_path)
UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.z); UsdGeom.SetStageMetersPerUnit(st, 1.0)
root = UsdGeom.Xform.Define(st, "/fourbar"); st.SetDefaultPrim(root.GetPrim())
UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
pa = PhysxSchema.PhysxArticulationAPI.Apply(root.GetPrim())
pa.CreateSolverPositionIterationCountAttr(args.pos_iters); pa.CreateSolverVelocityIterationCountAttr(args.vel_iters)
pa.CreateEnabledSelfCollisionsAttr(False); pa.CreateSleepThresholdAttr(0.0); pa.CreateStabilizationThresholdAttr(0.0)
def link(name, pos, rot, m, com, diag):
    x = UsdGeom.Xform.Define(st, f"/fourbar/{name}")
    x.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in pos])); x.AddOrientOp().Set(rot)
    p = x.GetPrim(); UsdPhysics.RigidBodyAPI.Apply(p)
    mp = UsdPhysics.MassAPI.Apply(p); mp.CreateMassAttr(float(m)); mp.CreateCenterOfMassAttr(Gf.Vec3f(*com))
    mp.CreateDiagonalInertiaAttr(Gf.Vec3f(*diag)); mp.CreatePrincipalAxesAttr(Gf.Quatf(1, 0, 0, 0))
    rb = PhysxSchema.PhysxRigidBodyAPI.Apply(p)
    rb.CreateLinearDampingAttr(0.0); rb.CreateAngularDampingAttr(0.0); rb.CreateSleepThresholdAttr(0.0); rb.CreateStabilizationThresholdAttr(0.0)
    rb.CreateMaxAngularVelocityAttr(1e7); rb.CreateMaxLinearVelocityAttr(1e7)
    return p
link("ground", (0, 0, 0), Gf.Quatf(1, 0, 0, 0), 1.0, (0, 0, 0), (1e-3, 1e-3, 1e-3))
for bn, (piv, th, r) in BODIES.items():
    link(bn, piv, qy(th), r["m"], (r["L"] / 2, 0, 0), (r["Ia"], r["It"], r["It"]))
fj = UsdPhysics.FixedJoint.Define(st, "/fourbar/joints/world_fixed"); fj.CreateBody1Rel().SetTargets(["/fourbar/ground"])
def rev(name, b0, p0, r0, b1, p1, r1, exclude=False):
    j = UsdPhysics.RevoluteJoint.Define(st, f"/fourbar/joints/{name}")
    j.CreateBody0Rel().SetTargets([b0]); j.CreateBody1Rel().SetTargets([b1]); j.CreateAxisAttr("Y")
    j.CreateLocalPos0Attr(p0); j.CreateLocalRot0Attr(r0); j.CreateLocalPos1Attr(p1); j.CreateLocalRot1Attr(r1)
    if exclude: j.CreateExcludeFromArticulationAttr(True)
    pj = PhysxSchema.PhysxJointAPI.Apply(j.GetPrim()); pj.CreateJointFrictionAttr(0.0); pj.CreateMaxJointVelocityAttr(1e6)
    return j
inv = lambda bn: qy(-BODIES[bn][1])
I = Gf.Quatf(1, 0, 0, 0)
rev("crank", "/fourbar/ground", Gf.Vec3f(*A), I, "/fourbar/crank", Gf.Vec3f(0, 0, 0), inv("crank"))
rev("coupler", "/fourbar/crank", local("crank", Bp), inv("crank"), "/fourbar/coupler", Gf.Vec3f(0, 0, 0), inv("coupler"))
rev("rocker", "/fourbar/ground", Gf.Vec3f(*D), I, "/fourbar/rocker", Gf.Vec3f(0, 0, 0), inv("rocker"))
if not args.no_loop: rev("loop0", "/fourbar/coupler", local("coupler", C), inv("coupler"), "/fourbar/rocker", local("rocker", C), inv("rocker"), exclude=True)
st.GetRootLayer().Save()
print("[fourbar] wrote", usd_path, "C =", C.tolist(), flush=True)

sim_cfg = sim_utils.SimulationCfg(dt=args.dt, device=dev, gravity=(0.0, 0.0, -G))
sim_cfg.physx.solver_type = 1 if args.solver == "TGS" else 0
sim = sim_utils.SimulationContext(sim_cfg)
cfg = ArticulationCfg(prim_path="/World/fourbar", spawn=sim_utils.UsdFileCfg(usd_path=usd_path),
                      init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), joint_pos={".*": 0.0}, joint_vel={".*": 0.0}),
                      actuators={"passive": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=0.0, damping=0.0)})
art = Articulation(cfg)
sim.reset()
print("[fourbar] bodies", art.body_names, "joints", art.joint_names, "fixed base", art.is_fixed_base, flush=True)
order = ["crank", "coupler", "rocker"]; bid = [art.body_names.index(n) for n in order]
ic, ir = art.body_names.index("coupler"), art.body_names.index("rocker")

def qrot(q, v):   # wxyz, batched
    w, u = q[..., :1], q[..., 1:]; t = 2 * np.cross(u, v); return v + w * t + np.cross(u, t)
N = int(round(args.T / args.dt))
rec = {k: [] for k in ("body_pos_w", "body_quat_w", "joint_pos", "joint_vel")}
def snap():
    rec["body_pos_w"].append(art.data.body_pos_w[0].cpu().numpy().astype(np.float64)); rec["body_quat_w"].append(art.data.body_quat_w[0].cpu().numpy().astype(np.float64))
    rec["joint_pos"].append(art.data.joint_pos[0].cpu().numpy().astype(np.float64)); rec["joint_vel"].append(art.data.joint_vel[0].cpu().numpy().astype(np.float64))
art.update(args.dt); snap()
import time; t0 = time.time()
for k in range(N):
    art.write_data_to_sim(); sim.step(render=False); art.update(args.dt); snap()
wall = time.time() - t0
R = {k: np.stack(v) for k, v in rec.items()}
q = R["body_quat_w"]; p = R["body_pos_w"]
dirs = qrot(q, np.array([1.0, 0, 0]))
angs = np.arctan2(dirs[..., 2], dirs[..., 0])[:, bid]
tipc = p[:, ic] + qrot(q[:, ic], np.array([b_, 0, 0])); tipr = p[:, ir] + qrot(q[:, ir], np.array([c_, 0, 0]))
gap = np.linalg.norm(tipc - tipr, axis=-1)
yoff = np.abs(p[..., 1]).max()
t = np.arange(N + 1) * args.dt
np.savez_compressed(os.path.join(args.out, "fourbar_none.npz"), t=t, ang=angs, closure_gap=gap, body_names=np.array(art.body_names), joint_names=np.array(art.joint_names), **R)
meta = {"dt": args.dt, "T": args.T, "device": dev, "solver": args.solver, "pos_iters": args.pos_iters, "vel_iters": args.vel_iters,
        "body_order_ang": order, "closure_gap_max_m": float(gap.max()), "closure_gap_mean_m": float(gap.mean()), "closure_gap_final_m": float(gap[-1]),
        "max_abs_y_m": float(yoff), "wall_s": wall, "C": C.tolist(), "crank_final_rad": float(angs[-1, 0]),
        "isaaclab": "2.3.x", "no_loop": args.no_loop, "usd": "fourbar.usda"}
json.dump(meta, open(os.path.join(args.out, "meta.json"), "w"), indent=1)
print("[fourbar] summary", json.dumps(meta), flush=True)
open(os.path.join(args.out, "DONE"), "w").write("ok")
sys.stdout.flush()
os._exit(0)   # SimulationApp.close() hangs after a bare SimulationContext on this build; all data is written

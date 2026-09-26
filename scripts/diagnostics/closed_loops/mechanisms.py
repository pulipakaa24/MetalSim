"""Closed-loop test mechanisms, defined once and emitted three ways (2026-09-25, closed-loop study):

* ``mjcf(name, ...)``: MuJoCo MJCF, spanning tree of hinges + one ``connect`` equality per loop (soft closure);
* ``newton_builder(name, ...)``: Newton ModelBuilder with every joint a revolute, loop-closing joint outside the
  articulation (hard closure in SolverKamino), same frames, masses and inertias;
* ``planar_reference(name, ...)``: an independent planar maximal-coordinate DAE (each rod x, z, phi; pin joints as
  point equalities; index-1 form with light Baumgarte) integrated by scipy at rtol 1e-10: the "exact" hard-loop
  trajectory both engines are compared with.

Mechanisms (all in the world x-z plane, hinge axes world +y, gravity -z, no contacts):

* ``fourbar``: Grashof crank-rocker. Ground pivots A = (0,0,0), D = (0.30,0,0); crank a = 0.10 m (0.2 kg), coupler
  b = 0.30 m (0.6 kg), rocker c = 0.25 m (0.5 kg); uniform rods of radius 0.01 m. Crank starts at -20 deg: the
  passive swing (crank -20 .. about -188 deg) stays well inside the potential well (the upper equilibrium is near
  +80 deg; a start at 60 deg grazes it and makes trajectories hypersensitive). 1 DoF.
  Crank torque actuator.
* ``leg``: hanging leg with a parallelogram knee drive (Minitaur/Digit-style: the knee motor sits at the hip and
  drives the shank through a crank and a rod parallel to the thigh). Thigh 0.30 m / 2.0 kg (hip hinge, actuated),
  shank 0.30 m / 1.0 kg (knee hinge, passive, limit -30..95 deg), knee crank r = 0.06 m / 0.1 kg coaxial with the
  hip (actuated), rod 0.30 m / 0.1 kg from the crank tip to a bell-crank point on the shank 0.06 m from the knee,
  125 deg off the shank axis (parallelogram singular at knee -55 / 125 deg, 25 deg outside the limits). Starts at
  thigh -70 deg (20 deg off vertical), knee 20 deg: the passive swing stays inside the limits. 2 DoF.

Angle convention: a rod's direction angle theta is measured in the x-z plane from +x toward +z; a hinge about +y
with coordinate q rotates by -q in theta (right-hand rule), so child theta = parent theta - q.
"""
from __future__ import annotations

import math

import numpy as np

G = 9.81
RAD = 0.01


def _rod(m, L, r=RAD):
    return dict(m=m, L=L, Ia=0.5 * m * r * r, It=m * L * L / 12.0 + 0.25 * m * r * r)


def _dir(th):
    return np.array([math.cos(th), 0.0, math.sin(th)])


def fourbar_geometry(theta_crank=math.radians(-20.0)):
    A = np.zeros(3); D = np.array([0.30, 0, 0])
    a, b, c = 0.10, 0.30, 0.25
    B = A + a * _dir(theta_crank)
    # C: intersection of |C-B| = b and |C-D| = c, upper branch
    d = np.linalg.norm(D - B); ex = (D - B) / d
    x = (d * d + b * b - c * c) / (2 * d); h = math.sqrt(b * b - x * x)
    ez = np.array([-ex[2], 0, ex[0]])            # ex rotated +90 deg in x-z
    C1 = B + x * ex + h * ez; C2 = B + x * ex - h * ez
    C = C1 if C1[2] > C2[2] else C2
    th = lambda P, Q: math.atan2(Q[2] - P[2], Q[0] - P[0])
    bodies = {  # name: (pivot, theta, rod, parent)
        "crank": (A, theta_crank, _rod(0.2, a), None),
        "coupler": (B, th(B, C), _rod(0.6, b), "crank"),
        "rocker": (D, th(D, C), _rod(0.5, c), None),
    }
    # loops: (body1, point in body1 local, body2, point in body2 local)
    loops = [("coupler", np.array([b, 0, 0]), "rocker", np.array([c, 0, 0]))]
    joints = [("crank", None, "crank"), ("coupler", "crank", "coupler"), ("rocker", None, "rocker")]  # (name, parent, child)
    actuated = ["crank"]
    limits = {}
    return dict(bodies=bodies, loops=loops, joints=joints, actuated=actuated, limits=limits, points=dict(A=A, B=B, C=C, D=D))


def leg_geometry(theta_thigh=math.radians(-70.0), q_knee=math.radians(20.0), gamma=math.radians(125.0)):
    H = np.zeros(3)
    Lt, Ls, r, Lr = 0.30, 0.30, 0.06, 0.30
    K = H + Lt * _dir(theta_thigh)
    th_s = theta_thigh - q_knee
    th_c = th_s + gamma
    P = H + r * _dir(th_c)
    Lp = K + r * _dir(th_c)                     # bell-crank point on the shank
    bodies = {
        "thigh": (H, theta_thigh, _rod(2.0, Lt, 0.015), None),
        "shank": (K, th_s, _rod(1.0, Ls, 0.015), "thigh"),
        "crank": (H, th_c, _rod(0.1, r), None),
        "rod": (P, theta_thigh, _rod(0.1, Lr), "crank"),
    }
    lever_local = r * np.array([math.cos(gamma), 0, math.sin(gamma)])
    loops = [("rod", np.array([Lr, 0, 0]), "shank", lever_local)]
    joints = [("hip", None, "thigh"), ("knee", "thigh", "shank"), ("crank", None, "crank"), ("rodpin", "crank", "rod")]
    actuated = ["hip", "crank"]
    limits = {"knee": (math.radians(-30.0), math.radians(95.0))}
    # joint coordinate of the knee at the start is q_knee (hinge coordinate relative to the start pose is 0 in
    # MuJoCo; the limit range is therefore shifted by -q_knee there, see mjcf())
    return dict(bodies=bodies, loops=loops, joints=joints, actuated=actuated, limits=limits, q0={"knee": q_knee},
                points=dict(H=H, K=K, P=P, L=Lp))


GEOMS = {"fourbar": fourbar_geometry, "leg": leg_geometry}
# Torque scale sigma [Nm] per actuated joint: about half the largest static gravity torque on that joint (four-bar
# crank 0.65 Nm; leg hip ~8 Nm with the leg horizontal, knee crank ~1.5 Nm); the stress test applies Gaussian
# torques of 3 sigma (the repo's 3-sigma convention), resampled at 50 Hz.
SIGMA_TAU = {"fourbar": {"crank": 0.3}, "leg": {"hip": 4.0, "crank": 0.75}}
# Viscous joint damping [N m s/rad] used with torque inputs (motor back-EMF / gearbox losses; without it random
# torques pump energy without bound, e.g. 95-240 rad/s after 8 s): actuated joints about sigma / 20 rad/s,
# passive tree joints small. Passive (torque-free) runs are undamped. The loop-closing joint is undamped.
DAMPING = {"fourbar": {"crank": 0.015, "coupler": 0.002, "rocker": 0.002},
           "leg": {"hip": 0.2, "knee": 0.02, "crank": 0.04, "rodpin": 0.002}}


def torque_table(mech, nworld, T, hold=0.02, amp=3.0, seed=0):
    """(n_ctrl, nworld, n_act) Gaussian torques [Nm], amp x SIGMA_TAU, in the actuated-joint order; one generator
    per world (seed, w), so world w's sequence does not depend on nworld or T (prefix-consistent)."""
    sig = np.array([SIGMA_TAU[mech][j] for j in GEOMS[mech]()["actuated"]])
    n = int(math.ceil(T / hold)) + 1
    out = np.empty((n, nworld, len(sig)))
    for w in range(nworld):
        out[:, w] = np.random.default_rng([seed, w]).standard_normal((n, len(sig))) * amp * sig
    return out


def _quat_y(th):
    """MuJoCo quaternion (w, x, y, z) turning local +x to direction angle th."""
    return np.array([math.cos(-th / 2), 0.0, math.sin(-th / 2), 0.0])


def mjcf(name, dt=0.0025, solref=(0.02, 1.0), solimp=(0.9, 0.95, 0.001, 0.5, 2.0), integrator="implicitfast",
         eq_on=True, extra_option="", damped=False):
    g = GEOMS[name]()
    B = g["bodies"]
    jname_of = {child: j for j, _, child in g["joints"]}
    q0 = g.get("q0", {})

    def body_xml(bn, indent):
        piv, th, rod, parent = B[bn]
        if parent is None:
            pos, rel = piv, th
        else:
            ppiv, pth, prod, _ = B[parent]
            dp = piv - ppiv
            # parent-local coordinates of the child pivot (rotate world offset by -pth about the plane)
            pos = np.array([math.cos(pth) * dp[0] + math.sin(pth) * dp[2], 0.0, -math.sin(pth) * dp[0] + math.cos(pth) * dp[2]])
            rel = th - pth
        j = jname_of[bn]
        rng = ""
        if j in g["limits"]:
            lo, hi = g["limits"][j]
            rng = f' limited="true" range="{lo - q0.get(j, 0.0):.9f} {hi - q0.get(j, 0.0):.9f}"'
        if damped:
            rng += f' damping="{DAMPING[name][j]:g}"'
        q = _quat_y(rel)
        s = (f'{indent}<body name="{bn}" pos="{pos[0]:.12g} {pos[1]:.12g} {pos[2]:.12g}" quat="{q[0]:.15g} {q[1]:.15g} {q[2]:.15g} {q[3]:.15g}">\n'
             f'{indent}  <joint name="{j}" type="hinge" axis="0 1 0"{rng}/>\n'
             f'{indent}  <inertial pos="{rod["L"] / 2:.12g} 0 0" mass="{rod["m"]}" diaginertia="{rod["Ia"]:.12g} {rod["It"]:.12g} {rod["It"]:.12g}"/>\n'
             f'{indent}  <geom type="capsule" fromto="0 0 0 {rod["L"]:.12g} 0 0" size="{RAD}" contype="0" conaffinity="0" mass="0"/>\n')
        for cn, (_, _, _, par) in B.items():
            if par == bn:
                s += body_xml(cn, indent + "  ")
        return s + f"{indent}</body>\n"

    roots = "".join(body_xml(bn, "    ") for bn, v in B.items() if v[3] is None)
    sr = f'solref="{solref[0]:g} {solref[1]:g}" solimp="{" ".join(f"{x:g}" for x in solimp)}"'
    eqs = "".join(f'    <connect body1="{b1}" body2="{b2}" anchor="{p1[0]:.12g} {p1[1]:.12g} {p1[2]:.12g}" {sr}/>\n'
                  for b1, p1, b2, p2 in g["loops"]) if eq_on else ""
    acts = "".join(f'    <motor name="{j}" joint="{j}" gear="1" ctrllimited="false"/>\n' for j in g["actuated"])
    return f"""<mujoco model="{name}">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{dt}" gravity="0 0 -{G}" integrator="{integrator}" {extra_option}/>
  <worldbody>
{roots}  </worldbody>
  <equality>
{eqs}  </equality>
  <actuator>
{acts}  </actuator>
</mujoco>
"""


def mj_closure_error(name, xpos, xmat):
    """Loop-closure error [m] per world from MuJoCo body poses. xpos (..., nbody, 3), xmat (..., nbody, 9).
    Body ids follow MJCF order: world 0 then depth-first as written by mjcf()."""
    g = GEOMS[name]()
    order = _mj_body_order(g)
    errs = []
    for b1, p1, b2, p2 in g["loops"]:
        i1, i2 = order.index(b1) + 1, order.index(b2) + 1
        w1 = xpos[..., i1, :] + np.einsum("...ij,j->...i", xmat[..., i1, :].reshape(*xmat.shape[:-2], 3, 3), p1)
        w2 = xpos[..., i2, :] + np.einsum("...ij,j->...i", xmat[..., i2, :].reshape(*xmat.shape[:-2], 3, 3), p2)
        errs.append(np.linalg.norm(w1 - w2, axis=-1))
    return np.max(np.stack(errs, 0), 0)


def _mj_body_order(g):
    B = g["bodies"]; out = []
    def rec(bn):
        out.append(bn)
        for cn, v in B.items():
            if v[3] == bn:
                rec(cn)
    for bn, v in B.items():
        if v[3] is None:
            rec(bn)
    return out


def body_order(name):
    return _mj_body_order(GEOMS[name]())


# ---------------------------------------------------------------------------------------------------------------
# Newton builder (hard loop closure in SolverKamino)

def newton_builder(name, builder=None, actuator_mode="effort", damped=False):
    import warp as wp
    import newton
    from newton import JointTargetMode
    g = GEOMS[name]()
    B = g["bodies"]
    q0 = g.get("q0", {})
    own = builder is None
    b = builder if builder is not None else newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-G)
    ids = {}
    for bn in _mj_body_order(g):
        piv, th, rod, _ = B[bn]
        q = _quat_y(th)                       # (w, x, y, z) -> warp (x, y, z, w)
        rot = wp.quat(float(q[1]), float(q[2]), float(q[3]), float(q[0]))
        I = wp.mat33(rod["Ia"], 0.0, 0.0, 0.0, rod["It"], 0.0, 0.0, 0.0, rod["It"])
        ids[bn] = b.add_link(label=bn, mass=rod["m"], inertia=I, com=wp.vec3(rod["L"] / 2, 0.0, 0.0),
                             xform=wp.transform(wp.vec3(*[float(x) for x in piv]), rot), lock_inertia=True)

    def local(bn, pw):
        piv, th, _, _ = B[bn]
        d = pw - piv
        return wp.vec3(math.cos(th) * d[0] + math.sin(th) * d[2], 0.0, -math.sin(th) * d[0] + math.cos(th) * d[2])

    def qinv(bn):
        q = _quat_y(-B[bn][1])
        return wp.quat(float(q[1]), float(q[2]), float(q[3]), float(q[0]))

    def dof(jn):
        mode = JointTargetMode.EFFORT if jn in g["actuated"] else JointTargetMode.NONE
        kw = dict(axis=newton.Axis.Y, actuator_mode=mode)
        if damped and jn in DAMPING[name]:
            kw.update(damping=DAMPING[name][jn])
        if jn in g["limits"]:
            lo, hi = g["limits"][jn]
            kw.update(limit_lower=lo - q0.get(jn, 0.0), limit_upper=hi - q0.get(jn, 0.0))
        if mode == JointTargetMode.EFFORT:
            kw.update(effort_limit=math.inf)
        return newton.ModelBuilder.JointDofConfig(**kw)

    # joint frames: world-aligned at the start pose (joint coordinate 0 there)
    art = []
    for jn, par, child in g["joints"]:
        piv = B[child][0]
        if par is None:
            pxf = wp.transform(wp.vec3(*[float(x) for x in piv]), wp.quat_identity())
            pid = -1
        else:
            pxf = wp.transform(local(par, piv), qinv(par)); pid = ids[par]
        cxf = wp.transform(local(child, piv), qinv(child))
        art.append(b.add_joint_revolute(label=jn, parent=pid, child=ids[child], axis=dof(jn), parent_xform=pxf, child_xform=cxf))
    for k, (b1, p1, b2, p2) in enumerate(g["loops"]):
        piv1 = B[b1][0]; th1 = B[b1][1]
        pw = piv1 + np.array([math.cos(th1) * p1[0] - math.sin(th1) * p1[2], 0.0, math.sin(th1) * p1[0] + math.cos(th1) * p1[2]])
        b.add_joint_revolute(label=f"loop{k}", parent=ids[b1], child=ids[b2], axis=dof(f"loop{k}"),
                             parent_xform=wp.transform(local(b1, pw), qinv(b1)), child_xform=wp.transform(local(b2, pw), qinv(b2)))
    b.add_articulation(art)
    return b, ids, g


def newton_closure_error(name, body_q):
    """body_q (..., nbody, 7) Newton transforms (px, py, pz, qx, qy, qz, qw), bodies in body_order()."""
    g = GEOMS[name]()
    order = _mj_body_order(g)

    def rot(q, v):
        x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        u = np.stack([x, y, z], -1)
        t = 2 * np.cross(u, np.broadcast_to(v, u.shape))
        return v + w[..., None] * t + np.cross(u, t)
    errs = []
    for b1, p1, b2, p2 in g["loops"]:
        i1, i2 = order.index(b1), order.index(b2)
        w1 = body_q[..., i1, :3] + rot(body_q[..., i1, 3:], p1)
        w2 = body_q[..., i2, :3] + rot(body_q[..., i2, 3:], p2)
        errs.append(np.linalg.norm(w1 - w2, axis=-1))
    return np.max(np.stack(errs, 0), 0)


# ---------------------------------------------------------------------------------------------------------------
# Planar exact reference (maximal coordinates, hard pins, index-1 DAE)

class PlanarRef:
    """Bodies in body_order(); coordinates per body (x, z, phi) of the pivot... stored as COM (xc, zc, th)."""

    def __init__(self, name, damped=False):
        g = GEOMS[name]()
        self.damp = DAMPING[name] if damped else {}
        self.g = g
        self.order = _mj_body_order(g)
        B = g["bodies"]
        self.nb = len(self.order)
        self.m = np.array([B[b][2]["m"] for b in self.order])
        self.I = np.array([B[b][2]["It"] for b in self.order])
        self.L = np.array([B[b][2]["L"] for b in self.order])
        # pins: list of (bodyA or -1, local point A (x along rod, z), bodyB, local point B), world points for -1
        pins = []
        jn_child = {}
        for jn, par, child in g["joints"]:
            piv = B[child][0]
            ci = self.order.index(child)
            if par is None:
                pins.append((-1, np.array([piv[0], piv[2]]), ci, np.array([0.0, 0.0])))
            else:
                pi_ = self.order.index(par)
                pins.append((pi_, self._loc(par, piv), ci, np.array([0.0, 0.0])))
            jn_child[jn] = (par, child)
        for b1, p1, b2, p2 in g["loops"]:
            pins.append((self.order.index(b1), np.array([p1[0], p1[2]]), self.order.index(b2), np.array([p2[0], p2[2]])))
        self.pins = pins
        self.jn_child = jn_child
        # limits (knee) as hard one-sided constraints are not modelled: the reference is used only in
        # windows where limits are inactive (checked by the caller)
        q = np.zeros(3 * self.nb)
        for i, bn in enumerate(self.order):
            piv, th, rod, _ = B[bn]
            q[3 * i:3 * i + 3] = [piv[0] + 0.5 * rod["L"] * math.cos(th), piv[2] + 0.5 * rod["L"] * math.sin(th), th]
        self.q0 = q

    def _loc(self, bn, pw):
        piv, th, _, _ = self.g["bodies"][bn]
        d = pw - piv
        return np.array([math.cos(th) * d[0] + math.sin(th) * d[2], -math.sin(th) * d[0] + math.cos(th) * d[2]])

    def _pt(self, q, i, p):
        """world point of body i's pivot-frame local point p (x along rod from pivot, z normal)."""
        xc, zc, th = q[3 * i:3 * i + 3]
        c, s = math.cos(th), math.sin(th)
        lx = p[0] - 0.5 * self.L[i]; lz = p[1]
        return np.array([xc + c * lx - s * lz, zc + s * lx + c * lz]), np.array([-s * lx - c * lz, c * lx - s * lz])

    def constraints(self, q, qd):
        n = 3 * self.nb
        rows = []; Gm = []; gdot = []
        for a, pa, b, pb in self.pins:
            J = np.zeros((2, n)); dd = np.zeros(2)
            if a >= 0:
                wa, dwa = self._pt(q, a, pa)
                J[:, 3 * a:3 * a + 2] += np.eye(2); J[:, 3 * a + 2] += dwa
                th = q[3 * a + 2]; w = qd[3 * a + 2]
                dd += -(wa - q[3 * a:3 * a + 2]) * w * w
            else:
                wa = pa
            wb, dwb = self._pt(q, b, pb)
            J[:, 3 * b:3 * b + 2] -= np.eye(2); J[:, 3 * b + 2] -= dwb
            w = qd[3 * b + 2]
            dd -= -(wb - q[3 * b:3 * b + 2]) * w * w
            rows.append(wa - wb); Gm.append(J); gdot.append(dd)
        return np.concatenate(rows), np.vstack(Gm), np.concatenate(gdot)

    def joint_angles(self, q):
        """hinge coordinates (MuJoCo convention: q = parent theta - child theta, relative to the start pose)."""
        out = {}
        th0 = self.q0[2::3]; th = q[2::3]
        for jn, (par, child) in self.jn_child.items():
            ci = self.order.index(child)
            if par is None:
                out[jn] = -(th[ci] - th0[ci])
            else:
                pi_ = self.order.index(par)
                out[jn] = -((th[ci] - th[pi_]) - (th0[ci] - th0[pi_]))
        return out

    def rhs(self, t, y, tau_fn):
        n = 3 * self.nb
        q, qd = y[:n], y[n:]
        Mi = np.zeros(n)
        Mi[0::3] = self.m; Mi[1::3] = self.m; Mi[2::3] = self.I
        F = np.zeros(n)
        F[1::3] = -self.m * G
        # joint torques: +tau on the child about +y means theta decreases (tau_theta = -tau), reaction on parent
        tau = dict(tau_fn(t))
        for jn, c in self.damp.items():          # joint rate qd = -(omega_child - omega_parent)
            par, child = self.jn_child[jn]
            w = qd[3 * self.order.index(child) + 2] - (qd[3 * self.order.index(par) + 2] if par is not None else 0.0)
            tau[jn] = tau.get(jn, 0.0) + c * w
        for jn, val in tau.items():
            par, child = self.jn_child[jn]
            F[3 * self.order.index(child) + 2] -= val
            if par is not None:
                F[3 * self.order.index(par) + 2] += val
        gpos, J, gd = self.constraints(q, qd)
        gvel = J @ qd
        alpha = 50.0
        rhs_c = -gd - 2 * alpha * gvel - alpha * alpha * gpos     # d2g/dt2 = J qdd + gd
        # [M J^T; J 0] [qdd; -lam] = [F; rhs_c]   (J qdd = rhs_c); redundant rows handled by lstsq
        K = np.zeros((n + J.shape[0], n + J.shape[0]))
        K[:n, :n] = np.diag(Mi); K[:n, n:] = J.T; K[n:, :n] = J
        sol = np.linalg.lstsq(K, np.concatenate([F, rhs_c]), rcond=None)[0]
        return np.concatenate([qd, sol[:n]])

    def simulate(self, T, tau_fn=lambda t: {}, t_eval=None):
        from scipy.integrate import solve_ivp
        y0 = np.concatenate([self.q0, np.zeros_like(self.q0)])
        return solve_ivp(self.rhs, (0, T), y0, args=(tau_fn,), rtol=1e-10, atol=1e-12, t_eval=t_eval, method="DOP853")


# ---------------------------------------------------------------------------------------------------------------
# Engine-independent diagnostics from body direction angles (N, nb) sampled every dt

def pivots_and_coms(name, ang):
    """world pivots and COMs (N, nb, 2) in (x, z) from body direction angles, following the tree."""
    g = GEOMS[name]()
    B = g["bodies"]; order = _mj_body_order(g)
    N = ang.shape[0]
    piv = np.zeros((N, len(order), 2)); com = np.zeros_like(piv)
    for i, bn in enumerate(order):
        p, th0, rod, par = B[bn]
        if par is None:
            piv[:, i] = [p[0], p[2]]
        else:
            j = order.index(par)
            pp, pth0 = B[par][0], B[par][1]
            d = p - pp
            lx = math.cos(pth0) * d[0] + math.sin(pth0) * d[2]; lz = -math.sin(pth0) * d[0] + math.cos(pth0) * d[2]
            c, s = np.cos(ang[:, j]), np.sin(ang[:, j])
            piv[:, i, 0] = piv[:, j, 0] + c * lx - s * lz
            piv[:, i, 1] = piv[:, j, 1] + s * lx + c * lz
        com[:, i, 0] = piv[:, i, 0] + 0.5 * rod["L"] * np.cos(ang[:, i])
        com[:, i, 1] = piv[:, i, 1] + 0.5 * rod["L"] * np.sin(ang[:, i])
    return piv, com


def energy(name, ang, dt):
    """total mechanical energy [J] (N-2,) at interior samples, central-difference velocities."""
    g = GEOMS[name]()
    B = g["bodies"]; order = _mj_body_order(g)
    a = np.unwrap(ang, axis=0)
    _, com = pivots_and_coms(name, a)
    v = (com[2:] - com[:-2]) / (2 * dt); w = (a[2:] - a[:-2]) / (2 * dt)
    m = np.array([B[b][2]["m"] for b in order]); I = np.array([B[b][2]["It"] for b in order])
    ke = 0.5 * (m * (v ** 2).sum(-1)).sum(-1) + 0.5 * (I * w ** 2).sum(-1)
    pe = (m * G * com[1:-1, :, 1]).sum(-1)
    return ke + pe

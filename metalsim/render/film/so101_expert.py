"""A deployment-style version of so101Sim's scripted IK expert for the SO-101 cube lift.

so101Sim's expert (``rl/native/scripted_expert.py`` ``Expert``) sets, every 20 ms tick, joint target = *measured* q +
clip(damped-least-squares step, +-DQ_MAX 0.06 rad). On a P-only servo (LeRobot writes P 16, I 0) the target must lead
the joint by load / kp to hold it; re-anchored to the measured q every tick, its lead never exceeds 0.06 rad
(0.32 N.m at kp 5.26 N.m/rad on ``7v4_real``) and the arm sinks where gravity needs more (0/100 on ``7v4_real``).

A real SO-101 under LeRobot is commanded differently: every control tick the host writes *absolute* goal positions
(``SOFollower.send_action``: ``sync_write("Goal_Position", ...)``; ``max_relative_target`` None by default), at the
loop's fps (30: ``DatasetRecordConfig.fps``, ``RolloutConfig.fps``), and the servo's own P loop tracks them, lagging and
sagging under load. LeRobot's own scripted move (``rollout/strategies/core.py`` ``return_to_initial_position``) is the
same pattern: absolute targets interpolated from the present position, streamed open loop. LeRobot's end-effector
pipeline (``robots/so_follower/robot_kinematic_processor.py`` ``InverseKinematicsEEToJoints``) turns a Cartesian target
into absolute joint goals with a kinematic IK; nothing in LeRobot compensates gravity on the SO arms (the only gravity
compensation in its source is the Unitree G1's, off by default).

``DeployExpert`` keeps so101Sim's plan: the same waypoints (cube + 10 cm, cube + 8 mm, lift to 0.20 m over the grasp
point), the same IK (damped least squares on the jaw-centre point "TCP", lambda 1e-4, five arm joints, wrist-roll yaw
alignment as a secondary task, each iteration clipped at DQ_MAX) and the same phases (approach, descend, grasp, lift,
retry if the cube stays low). What changes is the command:

* each waypoint's joint goal is the IK solved to convergence on the *kinematic model* (a scratch MjData at the
  commanded configuration), not one step from the measured, sagging state;
* the command moves to that goal along a straight line in joint space with a synchronised trapezoidal speed profile,
  per-joint speed and acceleration limits ``VMAX`` / ``AMAX`` (derived below from the ``7v4_real`` servo), re-targeted
  every tick to the latest goal (the cube is perceived at the control rate, as the original re-targets every tick);
* it is streamed as absolute targets at ``CTRL_HZ`` = 30 Hz (zero-order hold between ticks);
* the gripper gets absolute targets (open 1.4 rad, close -0.15 rad, so101Sim's), as steps; the servo's firmware profile
  and LeRobot's 50 % torque cap (``Max_Torque_Limit`` 500) shape the motion;
* phase transitions use the measured state with the original's tolerances, plus timeouts a script on a sagging
  P-only arm needs: once the command has arrived, a phase moves on when the measured TCP is within tolerance *or*
  ``T_SETTLE`` has passed (the 7v4_real step response's 2 % settle time at the shoulder, 310 ms). The grasp waits for
  the jaws to stop (measured gripper speed below ``V_MOVING``) after at least the original's 0.3 s, with a timeout.
* no gravity compensation (LeRobot has none for the SO arms).

Inputs every tick: an MjData holding the measured state with fresh kinematics (``mj_forward`` of qpos / qvel). The
cube pose is read from it, as the original reads it (a real script would get it from a camera).

``python -m``-free: pure numpy + mujoco; used by runs/readme_media/scratch/so101-expert/ (CPU and MetalSim evals).
"""

import numpy as np

# -- so101Sim's plan (rl/native/scripted_expert.py at d0546025, module constants and Expert.act) ---------------------
GRIP_OPEN = 1.4             # rad
GRIP_CLOSE = -0.15          # rad
DQ_MAX = 0.06               # rad: the IK's per-iteration clip (here: per solver iteration, not per tick)
APPROACH_DZ = 0.10          # m above the cube centre
GRASP_DZ = 0.008            # m: TCP target above the cube centre
LIFT_Z = 0.20               # m: TCP lift target
IK_LAMBDA = 1e-4            # damped least squares
APPROACH_TOL, YAW_TOL = 0.02, 0.12      # m, rad: approach -> descend (measured)
DESCEND_TOL = 0.012         # m: descend -> grasp (measured)
EXPERT_TICK = 0.02          # s: the original's control period, for its tick counts below
DESCEND_MAX = 70 * EXPERT_TICK          # s: "far spawns sometimes never hit the tight tolerance: close anyway"
GRASP_MIN = 15 * EXPERT_TICK            # s: the original's whole grasp phase
LIFT_CHECK, CUBE_LOW = 25 * EXPERT_TICK, 0.07   # s, m: "empty-handed? -> retry"
# The TCP (point between the jaws) and the jaw-opening axis in the gripperframe site's frame, as Expert.__init__
# calibrates them from so101Sim's 'pickup' keyframe (scene_box_rl.xml in so101Sim's menagerie clone; MuJoCo 3.10.0).
# Stored because MetalSim's copy of the scene has no keyframe; check: scratch/so101-expert/check_port.py.
TCP_LOCAL = np.array([-0.007388469181794655, -0.01061171842330552, 0.012775475294526169])
JAW_LOCAL = np.array([-5.851083600530068e-01, -4.196739854641918e-16, 8.109551202107791e-01])

# -- deployment ---------------------------------------------------------------------------------------------------
CTRL_HZ = 30.0              # LeRobot: lerobot-record --dataset.fps default 30, lerobot-rollout --fps default 30
T_SETTLE = 0.310            # s: 7v4_real step response, 2 % settle time, shoulder_lift (so101-servo step_tables.md)
V_MOVING = 0.05             # rad/s: "moving" threshold (scratch/so101-servo/smoothness.py V_HYST)
ARRIVED = 2 * np.pi / 4096  # rad: command within one encoder step of the goal counts as arrived
# Per-joint planning limits (shoulder_pan .. wrist_roll), from the 7v4_real servo (forcerange 1.912 N.m, Coulomb
# 0.0515 N.m, back-EMF + viscous damping 0.3417 N.m.s/rad) and the arm's gravity load and inertia at the task's
# waypoints (scratch/so101-expert/limits.py, 100 seeds): R = 1.912 - 0.0515 - g_max is the torque left over gravity;
# half of it goes to speed (VMAX = R / 2b), half to acceleration (AMAX = R / 2M_max), so at the trapezoid's corner the
# servo just reaches its duty limit and elsewhere it has headroom. That split is the time-optimal trapezoid for a move
# of D = M R / b^2 (0.4-0.6 rad here, the size of the task's moves). Capped by the firmware profile (5.29 rad/s,
# 39.0 rad/s^2), which the servo applies anyway (no joint reaches the cap). limits.py's output (limits.json):
# g_max [0.0, 0.599, 0.812, 0.316, 0.034] N.m (elbow: the lift with the heaviest box), M_max [0.044, 0.044, 0.035, 0.027, 0.026] kg.m^2.
VMAX = np.array([2.723, 1.846, 1.534, 2.261, 2.673])           # rad/s
AMAX = np.array([21.36, 14.25, 14.83, 28.40, 34.92])             # rad/s^2


def braking_speed(dist, acc, dt):
    """Largest speed from which steps of ``dt`` slowing by ``acc * dt`` each stop within ``dist`` (discrete)."""
    return acc * dt * (-0.5 + np.sqrt(0.25 + 2.0 * np.maximum(dist, 0.0) / (acc * dt * dt)))


class LinePlanner:
    """Online straight-line joint trajectory with a synchronised trapezoidal speed profile.

    Each tick the command moves toward the (possibly moved) goal along the line to it: path speed s' (fraction of the
    remaining displacement d per second) is at most min_j vmax_j / |d_j| and the discrete braking speed for the
    remaining path under min_j amax_j / |d_j|; the joint-velocity change per tick is at most amax_j * dt (scaled
    uniformly, so a static goal gives exactly the straight line). It lands on the goal when the next step would reach
    it. From rest to a static goal this is the classic synchronised trapezoid: all joints start and stop together."""

    def __init__(self, vmax, amax, dt):
        self.vmax, self.amax, self.dt = np.asarray(vmax, float), np.asarray(amax, float), float(dt)

    def reset(self, q):
        self.q = np.asarray(q, float).copy()
        self.v = np.zeros_like(self.q)

    def advance(self, goal):
        dt, d = self.dt, np.asarray(goal, float) - self.q
        big = np.abs(d) > 1e-12
        if not big.any():
            self.v[:] = 0.0
            return self.q.copy(), True
        ad = np.where(big, np.abs(d), 1.0)
        sv = np.min(np.where(big, self.vmax / ad, np.inf))        # path speed limit (1/s)
        sa = np.min(np.where(big, self.amax / ad, np.inf))        # path acceleration limit (1/s^2)
        s_des = min(sv, braking_speed(1.0, sa, dt))
        dv = s_des * d - self.v
        k = np.min(np.where(np.abs(dv) > 1e-12, self.amax * dt / np.maximum(np.abs(dv), 1e-12), np.inf))
        self.v = self.v + min(1.0, k) * dv
        step = self.v * dt
        if float(step @ d) >= float(d @ d):                       # reaches (or passes) the goal this tick
            self.q, self.v = np.asarray(goal, float).copy(), np.zeros_like(self.q)
            return self.q.copy(), True
        self.q = self.q + step
        return self.q.copy(), bool(np.max(np.abs(goal - self.q)) < ARRIVED and not self.v.any())


class DeployExpert:
    """One episode's controller. ``act(meas, t)`` at the control ticks returns the six absolute goal positions (rad).

    ``model``: the MjModel (any compile of the SO-101 lift scene with ``gripperframe`` site, ``box`` body / freejoint
    and the six arm joints / actuators in ``ARM_JOINTS`` order). ``ids``: dict(grip_site, box_body, box_qadr, arm_qadr,
    arm_vadr). ``vmax`` / ``amax``: the five arm joints' planning limits (defaults ``VMAX`` / ``AMAX``)."""

    def __init__(self, model, ids, vmax=None, amax=None, hz=CTRL_HZ):
        import mujoco

        self.mj, self.m = mujoco, model
        self.site, self.box_body = int(ids["grip_site"]), int(ids["box_body"])
        self.box_qadr, self.qadr, self.vadr = int(ids["box_qadr"]), np.asarray(ids["arm_qadr"]), np.asarray(ids["arm_vadr"])
        self.site_body = int(model.site_bodyid[self.site])
        self.lo, self.hi = model.actuator_ctrlrange[:5, 0].copy(), model.actuator_ctrlrange[:5, 1].copy()
        vmax = VMAX if vmax is None else vmax
        amax = AMAX if amax is None else amax
        if vmax is None or amax is None:
            raise ValueError("planning limits not set: pass vmax / amax (scratch/so101-expert/limits.py)")
        self.dt = 1.0 / float(hz)
        self.plan = LinePlanner(vmax, amax, self.dt)
        self.pd = mujoco.MjData(model)          # planning model: commanded configuration + perceived cube
        self.sc = mujoco.MjData(model)          # scratch for the roll gradient
        self.log = []

    # -- kinematics (ports of Expert._tcp, _jaw_yaw_of, _yaw_alignment_error, _roll_correction, _ik_step) ----------
    def _kin(self, d):
        self.mj.mj_kinematics(self.m, d)
        self.mj.mj_comPos(self.m, d)

    def tcp(self, d):
        R = d.site_xmat[self.site].reshape(3, 3)
        return d.site_xpos[self.site] + R @ TCP_LOCAL

    def _jaw_yaw(self, d):
        jaw = d.site_xmat[self.site].reshape(3, 3) @ JAW_LOCAL
        return np.arctan2(jaw[1], jaw[0])

    def yaw_error(self, d):
        q = d.qpos[self.box_qadr + 3:self.box_qadr + 7]
        cube_yaw = np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
        err = cube_yaw - self._jaw_yaw(d)
        return (err + np.pi / 4) % (np.pi / 2) - np.pi / 4

    def _roll_correction(self, d):
        err = self.yaw_error(d)
        sc = self.sc
        sc.qpos[:] = d.qpos
        y0 = self._jaw_yaw(d)
        sc.qpos[self.qadr[4]] += 0.02
        self.mj.mj_kinematics(self.m, sc)
        y1 = self._jaw_yaw(sc)
        g = (((y1 - y0 + np.pi) % (2 * np.pi)) - np.pi) / 0.02
        if abs(g) < 0.3:
            return 0.0
        return float(np.clip(0.5 * err / g, -DQ_MAX, DQ_MAX))

    def _ik_iter(self, target, align):
        m, d = self.m, self.pd
        tcp = self.tcp(d)
        err = target - tcp
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        self.mj.mj_jac(m, d, jacp, jacr, tcp, self.site_body)
        J = jacp[:, self.vadr[:5]]
        dq = J.T @ np.linalg.solve(J @ J.T + IK_LAMBDA * np.eye(3), err)
        dq = np.clip(dq, -DQ_MAX, DQ_MAX)
        if align:
            dq[4] += self._roll_correction(d)
        q = np.clip(d.qpos[self.qadr[:5]] + dq, self.lo, self.hi)
        return q, float(np.linalg.norm(err))

    def solve(self, meas, target, align, q0, iters=200, tol=1e-5):
        """The waypoint's joint goal: the original's IK iterated to convergence on the kinematic model from q0, the
        cube pose (for the yaw alignment) taken from the measured state."""
        d = self.pd
        d.qpos[:] = meas.qpos
        d.qpos[self.qadr[:5]] = q0
        self._kin(d)
        for _ in range(iters):
            q, e = self._ik_iter(target, align)
            step = float(np.max(np.abs(q - d.qpos[self.qadr[:5]])))
            d.qpos[self.qadr[:5]] = q
            self._kin(d)
            if step < tol:
                break
        return d.qpos[self.qadr[:5]].copy(), float(np.linalg.norm(target - self.tcp(d)))

    # -- episode --------------------------------------------------------------------------------------------------
    def reset(self, q_meas6):
        """The servos power on holding where they are: the command starts at the measured arm position."""
        q = np.asarray(q_meas6, float)
        self.plan.reset(q[:5])
        self.goal = q[:5].copy()
        self.grip = float(q[5])
        self.phase, self.t_phase0, self.t_arrive = "approach", 0.0, None
        self.lift_xy = None
        self.retries = 0
        self.ik_err = 0.0
        self.log = []
        self.events = []          # (t, from, to, reason)

    def _enter(self, phase, t, why):
        self.events.append((round(float(t), 4), self.phase, phase, why))
        self.phase, self.t_phase0, self.t_arrive = phase, t, None

    def act(self, meas, t):
        """``meas``: MjData with the measured qpos / qvel and fresh kinematics. ``t``: the tick's time (s)."""
        cube = meas.xpos[self.box_body].copy()
        tcp = self.tcp(meas)
        ph = self.phase
        if ph == "approach":
            tgt, align, grip = cube + np.array([0, 0, APPROACH_DZ]), True, GRIP_OPEN
        elif ph == "descend":
            tgt, align, grip = cube + np.array([0, 0, GRASP_DZ]), True, GRIP_OPEN
        elif ph == "grasp":
            tgt, align, grip = cube + np.array([0, 0, GRASP_DZ]), False, GRIP_CLOSE
        else:
            tgt, align, grip = np.array([self.lift_xy[0], self.lift_xy[1], LIFT_Z]), False, GRIP_CLOSE
        self.goal, self.ik_err = self.solve(meas, tgt, align, self.goal)
        q_cmd, arrived = self.plan.advance(self.goal)
        if arrived and self.t_arrive is None:
            self.t_arrive = t
        elif not arrived:
            self.t_arrive = None
        settled = self.t_arrive is not None and t - self.t_arrive >= T_SETTLE - 1e-9
        e = float(np.linalg.norm(tgt - tcp))                       # measured TCP error (the original's e)
        tp = t - self.t_phase0
        if ph == "approach":
            if self.t_arrive is not None and e < APPROACH_TOL and abs(self.yaw_error(meas)) < YAW_TOL:
                self._enter("descend", t, "measured")
            elif settled:
                self._enter("descend", t, "settle timeout")
        elif ph == "descend":
            if self.t_arrive is not None and e < DESCEND_TOL:
                self._enter("grasp", t, "measured")
            elif settled:
                self._enter("grasp", t, "settle timeout")
            elif tp >= DESCEND_MAX - 1e-9:
                self._enter("grasp", t, "descend cap")
        elif ph == "grasp":
            moving = abs(float(meas.qvel[self.vadr[5]])) > V_MOVING
            if tp >= GRASP_MIN - 1e-9 and not moving:
                self._enter("lift", t, "jaws stopped")
                self.lift_xy = cube[:2].copy()
            elif tp >= GRASP_TIMEOUT - 1e-9:
                self._enter("lift", t, "grasp timeout")
                self.lift_xy = cube[:2].copy()
        else:
            if tp >= LIFT_CHECK - 1e-9 and settled and cube[2] < CUBE_LOW:
                self.retries += 1
                self._enter("approach", t, "cube low: retry")
        self.grip = grip
        self.log.append((t, ph, e, self.ik_err))
        return np.concatenate([q_cmd, [grip]])


# The grasp's timeout: a free close (1.4 -> -0.15 rad) at the gripper's no-load speed under LeRobot's 50 % torque cap
# (half the duty: half the 7v4_real no-load speed, 5.445 / 2 rad/s) plus the settle time.
GRASP_TIMEOUT = (GRIP_OPEN - GRIP_CLOSE) / (52 * 2 * np.pi / 60 / 2) + T_SETTLE

"""Position hold on top of LimX's stock (Remote Control Mode) controller.

The stock controller is closed and owns the drives; its only input is a velocity command
(`request_twist {x, y, z}` on ws://10.192.1.2:5000, which LimX's own navigation stack sends at
30 Hz). It has no position feedback, so standing still it creeps (LimX's open policy, our stand-in:
~1 m and ~90 deg in 55 s in the sim). This outer loop closes that gap:

    odometry  wheel encoders + IMU from the low-level SDK stream, which the robot publishes in
              controller mode too (probe 2026-09-23: state ~2 kHz, IMU 500 Hz while the stock
              controller was in its damping state). Heading from the IMU's yaw (the wheel-angle
              difference fails when the policy spins the tyres: 140 deg off in one sim robot);
              forward speed r * (mean wheel rate + pitch rate) (the wheel
              encoders measure relative to the body, so body pitch has to be added back).
    hold      once the operator has let go and the robot has settled, anchor the pose. Default
              mode "pulse": the stock controller ignores |vx| up to ~0.15-0.2 m/s and tracks
              above it (probe 2026-09-24, runs/probe_vx.npz), so there is no small command to make
              continuous corrections with. Instead, once the error passes `pulse_engage`, drive
              back at `v_move` (just above the deadzone), cut to 0 `stop_lead` early (the stop
              overshoots), wait `cooldown` for the lean to settle, re-check. Heading the same way.
              Mode "pd" (continuous vx = -kp e - kd v) limit-cycles through that deadzone: on the
              robot the command built up unanswered, crossed it, lurched, and rocked at 1 Hz.
    operator  while the remote's sticks are off-centre we send nothing (the remote drives the
              stock controller directly), and the anchor follows the robot.

Run on the Jetson (dry run by default, prints what it would send):

    ROBOT_TYPE=WF_TRON1A .venv/bin/python -m metalsim.tron1.stock_hold [--send] [--log runs/hold.npz]
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .sdk import WHEEL, ImuData, RobotState
from .sim import WHEEL_RADIUS, quat_to_mat


@dataclass
class HoldGains:
    # Gentle on purpose: a balancing robot must lean before it can move, so a step in the command
    # makes the stock controller tip, lurch and overshoot; a fast outer loop then chases that and
    # rocks it (seen on the robot 2026-09-23 with kp 1.5 / kd 0.4 and no ramp).
    kp: float = 0.5                # (m/s) per m of position error
    kd: float = 0.1                # (m/s) per m/s of (low-passed) measured speed
    kp_psi: float = 0.8            # (rad/s) per rad of heading error
    kd_psi: float = 0.1            # (rad/s) per rad/s of (low-passed) yaw rate
    engage: float = 0.05           # m; start correcting beyond this error ...
    release: float = 0.01          # ... and stop (command ramps to 0) once back inside this
    engage_psi: float = np.radians(3.0)
    release_psi: float = np.radians(0.5)
    accel_max: float = 0.15        # m/s^2; the sent speed command ramps, never steps
    yaw_accel_max: float = 0.4     # rad/s^2
    v_filter: float = 0.3          # s; time constant of the speed / yaw-rate low-pass for the D terms
    v_max: float = 0.3             # m/s; the hold never asks for more
    w_max: float = 0.3             # rad/s
    settle_speed: float = 0.05     # m/s; anchor once slower than this ...
    settle_yaw_rate: float = 0.05  # rad/s
    settle_time: float = 0.5       # ... for this long, or after settle_timeout regardless
    settle_timeout: float = 2.0
    err_limit: float = 0.5         # m; if pushed further, the anchor is dragged along (no runaway)
    err_limit_psi: float = np.radians(30)
    stick_deadzone: float = 0.05   # remote axes (-1..1) below this count as centred
    # --- pulse mode
    pulse_engage: float = 0.08     # m; start a correction move beyond this error
    v_move: float = 0.25           # m/s; stock ignored +0.15 and -0.2, tracked +-0.25 (probe)
    stop_lead: float = 0.04        # m; cut the command this far before the anchor (stop overshoot)
    pulse_engage_psi: float = np.radians(5.0)
    w_move: float = 0.3            # rad/s; must clear the yaw deadzone (0.22 was ignored; probe wz)
    stop_lead_psi: float = np.radians(2.0)
    stop_ramp: float = 1.0         # s; ramp the command to 0 instead of stepping (a step from 0.25
                                   # lurched 7.5 deg). Only helps if the deadzone gates starting only.
    cooldown: float = 1.5          # s; after each move, let the lean settle before re-checking
    move_timeout: float = 3.0      # s; give up a move that does not get there


class WheelOdometry:
    """Planar pose from the SDK stream. `track` = distance between the tyre contact points.

    heading="imu": the IMU's own yaw (its AHRS output); immune to tyre slip, drifts slowly with gyro
    bias. heading="wheels": r (theta_R - theta_L) / track; no drift, but wrong once a tyre slips.
    """

    def __init__(self, track: float, radius: float = WHEEL_RADIUS, heading: str = "imu"):
        self.track, self.r, self.heading = track, radius, heading
        self.x = self.y = 0.0
        self.psi0 = None
        self.t_last = None
        self.psi = self.v = self.yaw_rate = 0.0

    def update(self, st: RobotState, imu: ImuData, t: float):
        R = quat_to_mat(imu.quat)
        yaw = np.arctan2(R[1, 0], R[0, 0])
        w_world = R @ np.asarray(imu.gyro)
        pitch_rate = -np.sin(yaw) * w_world[0] + np.cos(yaw) * w_world[1]
        self.yaw_rate = float(w_world[2])
        qw, dqw = np.asarray(st.q)[WHEEL], np.asarray(st.dq)[WHEEL]
        psi = yaw if self.heading == "imu" else self.r * (qw[1] - qw[0]) / self.track
        if self.psi0 is None:
            self.psi0 = psi
        self.psi = float(np.angle(np.exp(1j * (psi - self.psi0))))
        self.v = float(self.r * (np.mean(dqw) + pitch_rate))
        dt = 0.0 if self.t_last is None else float(np.clip(t - self.t_last, 0.0, 0.05))
        self.t_last = t
        self.x += self.v * np.cos(self.psi) * dt
        self.y += self.v * np.sin(self.psi) * dt


class StockHold:
    """Outer position/heading loop. `step(...) -> (vx, vy, wz)` to send, or None to stay silent."""

    def __init__(self, track: float, gains: HoldGains | None = None, heading: str = "imu", mode: str = "pulse"):
        self.g = gains or HoldGains()
        self.mode = mode
        self.move = None               # pulse: (axis 0 = position / 1 = heading, direction, t_start)
        self.stop = None               # pulse: ramping down after a move (axis, direction, t_start)
        self.cool_until = 0.0
        self.odom = WheelOdometry(track, heading=heading)
        self.anchor = None             # (x, y, psi) once holding
        self.still = self.idle = 0.0
        self.t_last = None
        self.v_f = self.w_f = 0.0      # low-passed speed and yaw rate
        self.cmd = np.zeros(2)         # last sent (vx, wz), for the ramp
        self.active = [False, False]   # correcting position / heading (hysteresis)

    def update_odometry(self, st: RobotState, imu: ImuData, t: float):
        self.odom.update(st, imu, t)

    def step(self, t: float, sticks=(0.0, 0.0, 0.0)):
        g, o = self.g, self.odom
        dt = 0.0 if self.t_last is None else float(np.clip(t - self.t_last, 0.0, 0.2))
        self.t_last = t
        a = dt / (g.v_filter + dt) if dt > 0 else 0.0
        self.v_f += a * (o.v - self.v_f)
        self.w_f += a * (o.yaw_rate - self.w_f)
        if np.max(np.abs(sticks)) > g.stick_deadzone:          # operator driving: stay out of it
            self.anchor, self.still, self.idle = None, 0.0, 0.0
            self.cmd[:], self.active = 0.0, [False, False]
            self.move = self.stop = None
            return None
        if self.anchor is None:
            slow = abs(o.v) < g.settle_speed and abs(o.yaw_rate) < g.settle_yaw_rate
            self.still = self.still + dt if slow else 0.0
            self.idle += dt
            if self.still < g.settle_time and self.idle < g.settle_timeout:
                return self._ramp(np.zeros(2), dt)
            self.anchor = [o.x, o.y, o.psi]
        ax, ay, apsi = self.anchor
        c, s = np.cos(o.psi), np.sin(o.psi)
        e = c * (o.x - ax) + s * (o.y - ay)                    # along the heading (diff drive)
        if abs(e) > g.err_limit:                               # pushed far: drag the anchor
            self.anchor[0] += c * (e - np.sign(e) * g.err_limit)
            self.anchor[1] += s * (e - np.sign(e) * g.err_limit)
            e = np.sign(e) * g.err_limit
        e_psi = float(np.angle(np.exp(1j * (o.psi - apsi))))
        if abs(e_psi) > g.err_limit_psi:
            self.anchor[2] = o.psi - np.sign(e_psi) * g.err_limit_psi
            e_psi = np.sign(e_psi) * g.err_limit_psi
        if self.mode == "pulse":
            return self._pulse(t, e, e_psi)
        # Hysteresis instead of a deadband: sway inside `engage` is left alone; once engaged, drive
        # all the way back to `release` so it does not dither at the edge.
        if abs(e) > g.engage:
            self.active[0] = True
        elif abs(e) < g.release:
            self.active[0] = False
        if abs(e_psi) > g.engage_psi:
            self.active[1] = True
        elif abs(e_psi) < g.release_psi:
            self.active[1] = False
        vx = np.clip(-g.kp * e - g.kd * self.v_f, -g.v_max, g.v_max) if self.active[0] else 0.0
        wz = np.clip(-g.kp_psi * e_psi - g.kd_psi * self.w_f, -g.w_max, g.w_max) if self.active[1] else 0.0
        return self._ramp(np.array([vx, wz]), dt)

    def _pulse(self, t, e, e_psi):
        g = self.g
        if self.move is not None:
            axis, d, t0 = self.move
            rem = -d * (e if axis == 0 else e_psi)             # distance still to go (> 0)
            lead = g.stop_lead if axis == 0 else g.stop_lead_psi
            if rem > lead and t - t0 < g.move_timeout:
                return (d * g.v_move, 0.0, 0.0) if axis == 0 else (0.0, 0.0, d * g.w_move)
            self.move, self.stop = None, (axis, d, t)
            self.cool_until = t + g.stop_ramp + g.cooldown
        if self.stop is not None:
            axis, d, t0 = self.stop
            f = 1.0 - (t - t0) / g.stop_ramp
            if f > 0:
                return (d * g.v_move * f, 0.0, 0.0) if axis == 0 else (0.0, 0.0, d * g.w_move * f)
            self.stop = None
        if t < self.cool_until:
            return (0.0, 0.0, 0.0)
        if abs(e_psi) > g.pulse_engage_psi:                    # heading first: position moves along it
            self.move = (1, -np.sign(e_psi), t)
        elif abs(e) > g.pulse_engage:
            self.move = (0, -np.sign(e), t)
        else:
            return (0.0, 0.0, 0.0)
        return self._pulse(t, e, e_psi)

    def _ramp(self, target, dt):
        g = self.g
        lim = np.array([g.accel_max, g.yaw_accel_max]) * dt
        self.cmd = self.cmd + np.clip(target - self.cmd, -lim, lim)
        return (float(self.cmd[0]), 0.0, float(self.cmd[1]))


def track_from_q(q8) -> float:
    """Tyre-to-tyre distance for the current leg angles (LimX's MJCF kinematics)."""
    from .classical import Tron1Model
    return float(abs(np.diff(Tron1Model().wheel_centers(np.asarray(q8))[:, 1])[0]))


# ------------------------------------------------------------------------------------------------
# Sim check: LimX's open RL policy stands in for the stock controller.

def simulate(hold: bool, params, seed=0, seconds=55.0, hold_hz=30.0, link_delay=0.03,
             gains: HoldGains | None = None, pushes=(), trace=None, heading="imu",
             mode="pulse", deadzone=(0.17, 0.25)):
    """Standing still for `seconds`; the hold loop runs at `hold_hz` and its twist reaches the
    controller `link_delay` later (websocket + stock controller's command handling, [est])."""
    from .limx_policy import LimxPolicyController
    from .sim import Tron1Sim
    c = LimxPolicyController()
    sim = Tron1Sim(params, seed=seed).reset(c.initial_q())
    h = StockHold(track_from_q(c.initial_q()), gains, heading, mode) if hold else None
    period, hold_every = 2, int(round(1000 / hold_hz))          # controller at 500 Hz, 1 ms ticks
    pending, cmd = [], (0.0, 0.0, 0.0)
    pushes = sorted(pushes)
    p0, yaw0 = None, None
    peak = 0.0
    for k in range(int(seconds * 1000)):
        t = sim.t_us * 1e-6
        while pushes and pushes[0][0] <= t:
            sim.push(pushes.pop(0)[1])
        if h is not None and sim.latest_state is not None:
            h.update_odometry(sim.latest_state, sim.latest_imu, t)
            if k % hold_every == 0:
                out = h.step(t)
                if out is not None:
                    pending.append((t + link_delay, out))
        while pending and pending[0][0] <= t:
            cmd = pending.pop(0)[1]
        if k % period == 0:
            # The stock controller ignores small commands (probe: |vx| <= ~0.15-0.2 m/s).
            dz = (cmd[0] if abs(cmd[0]) > deadzone[0] else 0.0, 0.0, cmd[2] if abs(cmd[2]) > deadzone[1] else 0.0)
            sim.send(c.step(sim.latest_state, sim.latest_imu, dz, t))
        sim.tick()
        if k == 1000:                                            # measure from t = 1 s
            tr = sim.truth()
            p0, yaw0 = tr["pos"][:2].copy(), tr["yaw"]
        if p0 is not None:
            peak = max(peak, float(np.linalg.norm(sim.truth()["pos"][:2] - p0)))
        if trace is not None and k % 100 == 0 and h is not None:
            tr = sim.truth()
            trace.append((t, h.odom.x, h.odom.psi, h.odom.v, tr["pos"][0], tr["pos"][1], tr["yaw"], tr["v_head"][0], *cmd))
        if sim.fallen:
            break
    tr = sim.truth()
    return dict(fallen=bool(sim.fallen), drift=float(np.linalg.norm(tr["pos"][:2] - p0)), peak=peak,
                heading=float(np.degrees(abs(np.angle(np.exp(1j * (tr["yaw"] - yaw0)))))))


def _sim_job(a):
    hold, cond, seed, gains, *rest = a
    from .realism import SimParams
    p = SimParams.nominal() if cond == "nominal" else SimParams.nominal().sample(np.random.default_rng(20_000 + seed))
    return hold, cond, simulate(hold, p, seed=seed, gains=gains, **(rest[0] if rest else {}))


def sim_report(n=12, gains=None):
    from concurrent.futures import ProcessPoolExecutor
    jobs = [(hold, cond, s, gains) for hold in (False, True) for cond in ("nominal", "random") for s in range(n)]
    with ProcessPoolExecutor() as ex:
        rows = list(ex.map(_sim_job, jobs))
    for hold in (False, True):
        for cond in ("nominal", "random"):
            r = [x for h, cn, x in rows if h == hold and cn == cond]
            ok = [x for x in r if not x["fallen"]]
            f = lambda k: f"{np.median([x[k] for x in ok]):.2f} ({max(x[k] for x in ok):.2f})" if ok else "-"
            print(f"{'hold' if hold else 'stock'} / {cond:7s}  falls {len(r) - len(ok)}/{len(r)}  "
                  f"drift {f('drift')} m  peak {f('peak')} m  heading {f('heading')} deg")


# ------------------------------------------------------------------------------------------------
# On the robot (Jetson): SDK stream in, request_twist out.

def main():
    import argparse, json, os, threading, time, uuid
    ap = argparse.ArgumentParser()
    ap.add_argument("--send", action="store_true", help="actually send request_twist (default: dry run)")
    ap.add_argument("--hz", type=float, default=30.0)
    ap.add_argument("--seconds", type=float, default=0.0, help="stop after this long (0 = until Ctrl+C)")
    ap.add_argument("--log", default=None)
    ap.add_argument("--sim-report", action="store_true", help="run the sim comparison instead")
    ap.add_argument("--mode", choices=("pulse", "pd"), default="pulse")
    ap.add_argument("--probe", choices=("vx", "wz", "ramp"), default=None,
                    help="instead of holding, send a fixed staircase of commands (+a, 0, -a, 0 for rising a) "
                         "and log the response; moving a stick aborts it")
    ap.add_argument("--set", nargs="*", default=[], metavar="NAME=VALUE",
                    help="override HoldGains, e.g. --set kp=0.3 accel_max=0.1 (angles in radians)")
    a = ap.parse_args()
    gains = HoldGains()
    for kv in a.set:
        k, v = kv.split("=")
        if not hasattr(gains, k):
            raise SystemExit(f"unknown gain {k}; have {list(vars(gains))}")
        setattr(gains, k, float(v))
    if a.sim_report:
        return sim_report()

    ws = None
    if a.send:
        import websocket
        ws = websocket.create_connection("ws://10.192.1.2:5000", timeout=2)
    import limxsdk.robot.Robot as Robot, limxsdk.robot.RobotType as RobotType
    from .sdk import RobotState as S, ImuData as I
    os.environ.setdefault("ROBOT_TYPE", "WF_TRON1A")
    lock = threading.Lock()
    last = {}
    robot = Robot(RobotType.PointFoot)
    if not robot.init("10.192.1.2"):
        raise SystemExit("limxsdk init failed")
    def keep(k):
        def f(m):
            with lock:
                last[k] = (m, time.monotonic())
        return f
    robot.subscribeRobotState(keep("state")); robot.subscribeImuData(keep("imu")); robot.subscribeSensorJoy(keep("joy"))

    def send(v):
        msg = dict(accid="WF_TRON1A_445", title="request_twist", timestamp=int(time.time() * 1000),
                   guid=uuid.uuid4().hex, data=dict(x=v[0], y=v[1], z=v[2]))
        if ws is not None:
            ws.send(json.dumps(msg))

    import signal
    def _stop(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGHUP, _stop)
    hold, rows, t0 = None, [], time.monotonic()
    period = 1.0 / a.hz
    probe = None
    if a.probe:
        # (start, end, command): 3 s quiet, then per level 2 s at +a, 3 s at 0, 2 s at -a, 3 s at 0.
        probe, tt = [], 3.0
        if a.probe == "ramp":
            # Does the deadzone gate only starting? Kick at +-0.25 for 1 s, then ramp to 0 over 2 s
            # in 0.1 s steps: if the speed follows the ramp below 0.2, it does.
            for sgn in (1, -1, 1, -1):
                probe.append((tt, tt + 1.0, sgn * 0.25)); tt += 1.0
                for k in range(20):
                    probe.append((tt, tt + 0.1, sgn * 0.25 * (1 - (k + 1) / 20))); tt += 0.1
                probe.append((tt, tt + 3.0, 0.0)); tt += 3.0
            levels = "kick 0.25 then 2 s ramp, x4"
        else:
            levels = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3) if a.probe == "vx" else (0.1, 0.2, 0.3, 0.4)
            for lv in levels:
                for val, dur in ((lv, 2.0), (0.0, 3.0), (-lv, 2.0), (0.0, 3.0)):
                    probe.append((tt, tt + dur, val)); tt += dur
        a.seconds = tt + 1.0
        print(f"probe {a.probe}: levels {levels}, {tt:.0f} s; move a stick to abort", flush=True)
    aborted = False
    t_sent = 0.0
    try:
        while True:
            time.sleep(0.002)
            with lock:
                ms, mi, mj = last.get("state"), last.get("imu"), last.get("joy")
            if ms is None or mi is None:
                continue
            now = time.monotonic()
            if now - ms[1] > 0.05 or now - mi[1] > 0.05:
                print("stale SDK data; hold paused", flush=True)
                continue
            st = S(stamp=ms[0].stamp, q=np.array(ms[0].q), dq=np.array(ms[0].dq), tau=np.array(ms[0].tau))
            imu = I(stamp=mi[0].stamp, acc=np.array(mi[0].acc), gyro=np.array(mi[0].gyro), quat=np.array(mi[0].quat))
            if hold is None:
                hold = StockHold(track_from_q(st.q), gains, mode=a.mode)
                print(a.mode, gains, flush=True)
                print(f"track {hold.odom.track:.3f} m; {'SENDING' if ws else 'dry run'}", flush=True)
            hold.update_odometry(st, imu, now)
            R = quat_to_mat(imu.quat)
            pitch = float(np.arcsin(np.clip(-R[2, 0], -1, 1)))
            o = hold.odom
            sticks = tuple(mj[0].axes[:4]) if mj is not None else (0.0,)
            if probe is not None:
                # Log every sample (~500 Hz) so the lean / wheel response is visible.
                tp = now - t0
                cmd = next((c for t_a, t_b, c in probe if t_a <= tp < t_b), 0.0)
                if np.max(np.abs(sticks)) > gains.stick_deadzone and not aborted:
                    aborted = True
                    print("stick moved: probe aborted, sending zero", flush=True)
                out = (0.0, 0.0, 0.0) if aborted else ((0.0, 0.0, cmd) if a.probe == "wz" else (cmd, 0.0, 0.0))
                rows.append([tp, o.x, o.y, o.psi, o.v, o.yaw_rate, *out, pitch])
                if now - t_sent >= period:
                    t_sent = now
                    send(out)
                    if int(tp * a.hz) % int(a.hz) == 0:
                        print(f"t {tp:5.1f}  cmd {out[2] if a.probe == 'wz' else out[0]:+.2f}  x {o.x:+.3f}  "
                              f"psi {np.degrees(o.psi):+6.1f}  v {o.v:+.2f}  pitch {np.degrees(pitch):+5.1f}", flush=True)
                if a.seconds and tp > a.seconds:
                    break
                continue
            if now - (hold.t_last or 0.0) < period:
                continue
            out = hold.step(now, sticks)
            if out is not None:
                send(out)
            rows.append([now - t0, o.x, o.y, o.psi, o.v, o.yaw_rate, *(out or (np.nan,) * 3), pitch])
            if len(rows) % int(a.hz) == 0:
                print(f"t {now - t0:6.1f}  x {o.x:+.3f} y {o.y:+.3f} psi {np.degrees(o.psi):+6.1f} deg  "
                      f"v {o.v:+.2f}  -> {'(operator)' if out is None else f'vx {out[0]:+.2f} wz {out[2]:+.2f}'}", flush=True)
            if a.seconds and now - t0 > a.seconds:
                break
    except KeyboardInterrupt:
        pass
    finally:
        # Log first: sending can fail (e.g. the robot closed the socket) and must not lose it.
        if a.log and rows:
            np.savez(a.log, rows=np.array(rows), cols="t x y psi v yaw_rate vx vy wz pitch")
            print(f"saved {len(rows)} rows to {a.log}", flush=True)
        if ws is not None:
            try:
                send((0.0, 0.0, 0.0))
                ws.close()
            except Exception as ex:
                print(f"could not send the final zero: {ex}", flush=True)


if __name__ == "__main__":
    main()

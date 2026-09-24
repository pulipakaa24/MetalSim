"""Run our controller on the real TRON1 (Developer Mode), driven by the LimX remote.

    python -m metalsim.tron1.limx_bridge --ip 10.192.1.2 --gains runs/tron1_gains.json   # classical, on the Jetson
    python -m metalsim.tron1.limx_bridge --controller limx                              # LimX's open policy + hold
    python -m metalsim.tron1.limx_bridge --controller trained [--policy x.onnx]         # our policy + hold
    python -m metalsim.tron1.limx_bridge --backend sim [--controller limx]              # same code vs Tron1Sim

The robot must be in Developer Mode (R1 + Left; the stock controller is then off) and
zero-calibrated (hang it, L1 + R1, diagnostic "calibration" code 0). In Developer Mode the remote's
sticks and buttons reach this program through the SDK (`subscribeSensorJoy`).

Remote (buttons by the symbols printed on the LimX remote; verified on the robot 2026-09-24:
top = triangle, right = circle, bottom = cross, left = square):

    L1 + △ (triangle, top)    STAND: legs blend to the stance over 3 s, wheels limp. Safe hung.
    L1 + ✕ (cross, bottom)    BALANCE: full controller (wheels drive). Only after STAND, on the ground.
    L1 + □ (square, left)     DAMP: every motor Kp 0 / Kd 1 (LimX's stop), from any mode. The
                              robot sags: catch it or keep it on the rope.
    left stick up/down = forward speed, right stick left/right = turn rate.
    (LimX's own config calls these Y / A / X, Xbox-style; that "X" is the square, not the cross.)

LimX's firmware keeps its own reserved stops (both sticks pressed = e-stop; right stick click =
damping), plus the physical e-stop button.

Watchdogs (-> DAMP, and it stays there until L1 + Y): state or IMU older than 50 ms; tilt over
35 deg in BALANCE (or 60 deg in STAND); EtherCAT or IMU diagnostic ERROR; motor names not in the
order this controller assumes. BALANCE starts from a 0.5 s blend so the wheel torque does not
jump. The first sessions should cap speed and wheel torque (`--max-vx`, `--wheel-torque`).

Every tick is logged; the log is written to `runs/tron1_bridge_<time>.npz` on exit.
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .classical import ClassicalController, ClassicalGains
from .sdk import LEG, N_MOTORS, WF_MOTOR_NAMES, WHEEL, ImuData, RobotCmd, RobotState
from .sim import quat_to_mat
from .tuner import BELIEF, apply_flat

BTN = dict(A=0, B=1, X=2, Y=3, L1=4, R2=5, L2=6, R1=7)
# Internal names follow LimX's Xbox-style config; on the remote: A = ✕ cross (bottom), B = ○ circle
# (right), X = □ square (left), Y = △ triangle (top). Verified on the robot 2026-09-24.
BUTTON_LABEL = {0: "✕ cross (bottom)", 1: "○ circle (right)", 2: "□ square (left)", 3: "△ triangle (top)",
                4: "L1", 5: "R2", 6: "L2", 7: "R1", 8: "SELECT", 9: "START", 12: "UP", 13: "DOWN", 14: "LEFT",
                15: "RIGHT", 16: "MENU", 17: "BACK"}
AX_LEFT_V, AX_RIGHT_H = 1, 2
DAMP, STAND, BALANCE = "DAMP", "STAND", "BALANCE"


@dataclass
class BridgeConfig:
    max_vx: float = 0.5            # m/s at full stick
    max_wz: float = 0.8            # rad/s at full stick
    turn_sign: float = 1.0         # flip if right stick right turns left on your remote
    deadzone: float = 0.08
    wheel_torque: float = 20.0     # N m per wheel cap for the first sessions (controller max 40)
    stand_time: float = 3.0        # s, leg blend to the stance
    balance_blend: float = 0.5     # s, wheel torque ramp when BALANCE starts
    stale_s: float = 0.05
    max_tick_gap: float = 0.02     # s between control ticks; a longer stall (GC, CPU) -> DAMP
    tilt_balance_deg: float = 35.0
    tilt_stand_deg: float = 60.0
    require_calibration: bool = True


@dataclass
class Joy:
    axes: list = field(default_factory=lambda: [0.0] * 8)
    buttons: list = field(default_factory=lambda: [0] * 20)


class Bridge:
    """Mode logic + safety around ClassicalController. Pure: update() in, RobotCmd out."""

    def __init__(self, ctrl: ClassicalController, cfg: BridgeConfig):
        self.ctrl, self.cfg = ctrl, cfg
        self.mode = DAMP
        self.reason = "start"
        self.mode_t0 = 0.0
        self.q_start = None
        self.prev_buttons = set()
        self.calibration = -1
        self.diag_error = None
        self.robot_mode = {}          # the robot's own "working_mode" / "controller" diagnostics
        self.prev_refuse = None

    # -- inputs from the SDK callbacks
    def on_diagnostic(self, name, level, code, message):
        if name == "calibration":
            self.calibration = code
        if name in ("working_mode", "controller"):
            if self.robot_mode.get(name) != message:
                print(f"[bridge] robot reports {name} = {message!r}", flush=True)
            self.robot_mode[name] = message
        if name in ("ethercat", "imu") and level >= 2:            # 2 = ERROR in LimX's enum
            self.diag_error = f"{name}: {message}"

    def _pressed(self, joy: Joy, *names):
        return all(joy.buttons[BTN[n]] == 1 for n in names)

    def _to(self, mode, now, reason):
        if mode != self.mode:
            banner = "!!!!!!!! " if mode == DAMP else ""
            print(f"[bridge] {banner}{self.mode} -> {mode}: {reason}", flush=True)
            self.prev_refuse = None
        self.mode, self.mode_t0, self.reason = mode, now, reason

    def stick_command(self, joy: Joy):
        dz = lambda v: 0.0 if abs(v) < self.cfg.deadzone else float(np.clip(v, -1, 1))
        return (dz(joy.axes[AX_LEFT_V]) * self.cfg.max_vx, 0.0,
                self.cfg.turn_sign * dz(joy.axes[AX_RIGHT_H]) * self.cfg.max_wz)

    def update(self, st: RobotState, imu: ImuData, joy: Joy, now: float, st_age: float, imu_age: float,
               tick_gap: float = 0.0) -> RobotCmd:
        cfg = self.cfg
        if self.mode != DAMP and tick_gap > cfg.max_tick_gap:
            self._to(DAMP, now, f"control loop stalled {tick_gap * 1e3:.0f} ms")
        # ---- faults
        if self.mode != DAMP:
            R = quat_to_mat(imu.quat)
            tilt = np.degrees(np.arccos(np.clip(R[2, 2], -1, 1)))
            limit = cfg.tilt_balance_deg if self.mode == BALANCE else cfg.tilt_stand_deg
            if st_age > cfg.stale_s or imu_age > cfg.stale_s:
                self._to(DAMP, now, f"stale data (state {st_age * 1e3:.0f} ms, imu {imu_age * 1e3:.0f} ms)")
            elif tilt > limit:
                self._to(DAMP, now, f"tilt {tilt:.0f} deg > {limit:.0f}")
            elif self.diag_error:
                self._to(DAMP, now, self.diag_error)
        # ---- echo every button change, so the physical layout can be checked while hung
        held = {i for i, b in enumerate(joy.buttons) if b == 1}
        if held != self.prev_buttons:
            if held:
                print("[bridge] buttons held: " + " + ".join(BUTTON_LABEL.get(i, f"#{i}") for i in sorted(held)), flush=True)
            self.prev_buttons = held
        # ---- buttons (combos)
        if self._pressed(joy, "L1", "X"):
            self._to(DAMP, now, "L1 + □ square")
        elif self._pressed(joy, "L1", "Y") and self.mode == DAMP:
            if cfg.require_calibration and self.calibration != 0:
                print(f"[bridge] refusing STAND: calibration code {self.calibration} (hang the robot, L1 + R1)", flush=True)
            else:
                self.q_start = np.asarray(st.q, dtype=float).copy()
                self.diag_error = None
                self._to(STAND, now, "L1 + △ triangle")
        elif self._pressed(joy, "L1", "A"):
            if self.mode == STAND and now - self.mode_t0 >= cfg.stand_time:
                self.ctrl.reset()
                self._to(BALANCE, now, "L1 + ✕ cross")
            elif self.mode != STAND and self.mode != BALANCE and self.prev_refuse != "balance":
                print(f"[bridge] L1 + ✕ ignored: BALANCE only from STAND (now {self.mode}); L1 + △ first", flush=True)
                self.prev_refuse = "balance"
            elif self.mode == STAND and self.prev_refuse != "early":
                print(f"[bridge] L1 + ✕ ignored: stand-up still running ({now - self.mode_t0:.1f} / {cfg.stand_time:.0f} s)", flush=True)
                self.prev_refuse = "early"

        # ---- commands
        if self.mode == DAMP:
            return RobotCmd.damping(1.0)
        if self.mode == STAND:
            a = min((now - self.mode_t0) / cfg.stand_time, 1.0)
            cmd = RobotCmd()
            cmd.q[LEG] = (1 - a) * self.q_start[LEG] + a * self.ctrl.q_stance[LEG]
            cmd.Kp[LEG] = self.ctrl.g.leg_kp
            cmd.Kd[LEG] = self.ctrl.g.leg_kd
            cmd.Kp[[0, 4]] = self.ctrl.g.abad_kp
            cmd.Kd[[0, 4]] = self.ctrl.g.abad_kd
            cmd.Kd[WHEEL] = 1.0                                     # wheels limp
            return cmd
        vel = self.stick_command(joy)
        cmd = self.ctrl.step(st, imu, vel, now)
        # Wheel torque cap, ramped in over balance_blend. The drive applies
        # tau + Kd (dq_cmd - dq): the classical controller commands tau (Kd 0), LimX's policy a
        # velocity through Kd, so clip the total and put it back where it came from.
        ramp = min((now - self.mode_t0) / cfg.balance_blend, 1.0)
        dq = np.asarray(st.dq)[WHEEL]
        total = cmd.tau[WHEEL] + cmd.Kd[WHEEL] * (cmd.dq[WHEEL] - dq)
        capped = np.clip(total, -cfg.wheel_torque * ramp, cfg.wheel_torque * ramp)
        vel_mode = cmd.Kd[WHEEL] > 0
        cmd.dq[WHEEL] = np.where(vel_mode, dq + (capped - cmd.tau[WHEEL]) / np.where(vel_mode, cmd.Kd[WHEEL], 1.0), cmd.dq[WHEEL])
        cmd.tau[WHEEL] = np.where(vel_mode, cmd.tau[WHEEL], capped)
        return cmd


# ------------------------------------------------------------------------------------------------
class LimxBackend:
    """limxsdk-lowlevel (Linux amd64/aarch64). Callbacks copy the latest messages under a lock."""

    def __init__(self, ip):
        import os
        os.environ.setdefault("ROBOT_TYPE", "WF_TRON1A")        # limxsdk aborts without it
        import limxsdk.datatypes as datatypes
        import limxsdk.robot.Robot as Robot
        import limxsdk.robot.RobotType as RobotType
        self.dt = datatypes
        self.robot = Robot(RobotType.PointFoot)
        if not self.robot.init(ip):
            raise SystemExit(f"limxsdk: init({ip}) failed")
        # limxsdk 4.1.1 returns 7 names for 8 motors on this robot, two glued together
        # ('wheel_L_Jointabad_R_Joint'); compare the concatenation, which still catches reordering.
        names = list(self.robot.getMotorNames())
        if self.robot.getMotorNumber() != N_MOTORS or "".join(names) != "".join(WF_MOTOR_NAMES):
            raise SystemExit(f"motors {names} != expected {list(WF_MOTOR_NAMES)}; refusing to run")
        self.raw_state = self.raw_imu = self.raw_joy = None
        self.t_state = self.t_imu = 0.0
        self.bridge = None
        self.robot.subscribeRobotState(self._on_state)
        self.robot.subscribeImuData(self._on_imu)
        self.robot.subscribeSensorJoy(self._on_joy)
        self.robot.subscribeDiagnosticValue(self._on_diag)

    # Callbacks run on SDK threads ~3000 times a second in total and hold the GIL: keep them to a
    # reference swap; the conversion happens once per control tick in read().
    def _on_state(self, m):
        self.raw_state, self.t_state = m, time.monotonic()

    def _on_imu(self, m):
        self.raw_imu, self.t_imu = m, time.monotonic()

    def _on_joy(self, m):
        self.raw_joy = m

    def _on_diag(self, m):
        if self.bridge is not None:
            self.bridge.on_diagnostic(m.name, int(m.level), int(m.code), m.message)

    def read(self):
        ms, mi, mj = self.raw_state, self.raw_imu, self.raw_joy
        ts, ti = self.t_state, self.t_imu
        now = time.monotonic()
        st = None if ms is None else RobotState(stamp=ms.stamp, q=np.array(ms.q, float), dq=np.array(ms.dq, float),
                                               tau=np.array(ms.tau, float))
        imu = None if mi is None else ImuData(stamp=mi.stamp, acc=np.array(mi.acc, float), gyro=np.array(mi.gyro, float),
                                             quat=np.array(mi.quat, float))
        joy = Joy() if mj is None else Joy(list(mj.axes) + [0.0] * 8, list(mj.buttons) + [0] * 20)
        return st, imu, joy, now - ts, now - ti

    def publish(self, c: RobotCmd):
        m = self.dt.RobotCmd()
        m.stamp = time.time_ns()
        m.mode = [0] * N_MOTORS
        m.q, m.dq, m.tau = [float(v) for v in c.q], [float(v) for v in c.dq], [float(v) for v in c.tau]
        m.Kp, m.Kd = [float(v) for v in c.Kp], [float(v) for v in c.Kd]
        m.parallel_solve_required = [False] * N_MOTORS
        m.motor_names = [""] * N_MOTORS
        self.robot.publishRobotCmd(m)


class SimBackend:
    """Tron1Sim in lockstep, with a scripted remote: joy_script(t) -> Joy."""

    def __init__(self, sim, joy_script):
        self.sim, self.joy_script = sim, joy_script
        self.bridge = None

    def read(self):
        return self.sim.latest_state, self.sim.latest_imu, self.joy_script(self.sim.t_us * 1e-6), 0.0, 0.0

    def publish(self, c: RobotCmd):
        self.sim.send(c)


def run(backend, bridge: Bridge, hz=500, duration=None, clock=None, step_sim=None, log_path=None):
    """Control loop. Real time for the robot; lockstep when `step_sim` advances a sim."""
    import gc
    backend.bridge = bridge
    period = 1.0 / hz
    # Preallocated numeric log (a Python list of tuples grew ~1 MB/s and its garbage collection
    # stalled the loop 40 ms on the robot). Doubles when full, only in DAMP.
    log = np.zeros((int(hz * 900), len(LOG_COLUMNS)), dtype=np.float32)     # 15 min, ~95 MB at 500 Hz
    n_log = 0
    gc.collect()
    gc.freeze()
    import signal
    def _stop(*_):
        raise KeyboardInterrupt
    for sig in (signal.SIGTERM, signal.SIGHUP):              # closing the terminal = Ctrl-C
        try:
            signal.signal(sig, _stop)
        except ValueError:
            pass
    t0 = time.monotonic()
    nxt = t0
    last_tick = None
    gaps = []
    last_status = 0.0
    try:
        while duration is None or (clock() if clock else time.monotonic() - t0) < duration:
            st, imu, joy, st_age, imu_age = backend.read()
            now = clock() if clock else time.monotonic() - t0
            if st is None or imu is None:
                time.sleep(0.01)
                continue
            # No garbage collection while the legs are held or the robot balances; collect in DAMP.
            if bridge.mode == DAMP:
                if not gc.isenabled():
                    gc.enable()
            elif gc.isenabled():
                gc.disable()
            gap = 0.0 if last_tick is None else now - last_tick
            last_tick = now
            gaps.append(gap)
            cmd = bridge.update(st, imu, joy, now, st_age, imu_age, gap)
            backend.publish(cmd)
            if step_sim is None and now - last_status >= 1.0:
                last_status = now
                g = np.array(gaps[-int(hz):]) * 1e3
                vx, _, wz = bridge.stick_command(joy)
                print(f"[bridge] {bridge.mode:7s} loop mean {g.mean():.2f} ms p99 {np.percentile(g, 99):.2f} ms max {g.max():.1f} ms | "
                      f"state age {st_age * 1e3:.1f} ms | calib {bridge.calibration} | robot mode {bridge.robot_mode.get('working_mode', '?')} "
                      f"{bridge.robot_mode.get('controller', '?')} | stick vx {vx:+.2f} wz {wz:+.2f}", flush=True)
            if n_log == len(log):
                if bridge.mode != DAMP:              # never reallocate under control: wrap instead
                    n_log = 0
                else:
                    log = np.concatenate([log, np.zeros_like(log)])
            row = log[n_log]
            row[0], row[1] = now, {DAMP: 0, STAND: 1, BALANCE: 2}[bridge.mode]    # t: float32, ~0.1 ms at 15 min
            row[2:10], row[10:18], row[18:26] = st.q, st.dq, st.tau
            row[26:30], row[30:33] = imu.quat, imu.gyro
            row[33:41], row[41:49], row[49:53] = cmd.q, cmd.tau, joy.axes[:4]
            hc = getattr(bridge.ctrl, "last_hold_cmd", (0.0, 0.0, 0.0))
            row[53], row[54] = hc[0], hc[2]
            row[55:63] = cmd.dq
            n_log += 1
            if step_sim is not None:
                step_sim()
            else:
                nxt += period
                time.sleep(max(0.0, nxt - time.monotonic()))
    except KeyboardInterrupt:
        print("[bridge] Ctrl-C: damping", flush=True)
    finally:
        # A second Ctrl-C (or the terminal closing) must not cut the damping or lose the log.
        import signal
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            try:
                signal.signal(sig, signal.SIG_IGN)
            except ValueError:                       # not the main thread (tests)
                pass
        for _ in range(50):
            backend.publish(RobotCmd.damping(1.0))
            if step_sim is not None:
                step_sim()
            else:
                time.sleep(0.02)
        gc.enable()
        log = log[:n_log]
        if log_path:
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            np.savez(log_path, log=log, columns=np.array(LOG_COLUMNS))      # uncompressed: fast
            print(f"[bridge] log: {log_path} ({len(log)} ticks)", flush=True)
    return log


LOG_COLUMNS = (["t", "mode"] + [f"q{i}" for i in range(8)] + [f"dq{i}" for i in range(8)] + [f"tau{i}" for i in range(8)]
               + ["qw", "qx", "qy", "qz", "gx", "gy", "gz"] + [f"q_cmd{i}" for i in range(8)]
               + [f"tau_cmd{i}" for i in range(8)] + [f"ax{i}" for i in range(4)] + ["cmd_vx", "cmd_wz"]
               + [f"dq_cmd{i}" for i in range(8)])


def make_controller(gains_file=None, kind="classical", hold=True, policy=None):
    if kind in ("limx", "trained"):
        from .policy_hold import PolicyHoldController
        return PolicyHoldController(hold=hold, policy="limx" if kind == "limx" else (policy or "trained"))
    ctrl = ClassicalController()
    if gains_file:
        g = json.loads(Path(gains_file).read_text())
        apply_flat(ctrl, g)
        if any(k in g for k in BELIEF):
            b = {"belief_mass": ctrl.model.payload_mass, "belief_com_x": ctrl.model.payload_pos[0],
                 "belief_com_y": ctrl.model.payload_pos[1], "belief_com_z": ctrl.model.payload_pos[2], **g}
            ctrl.set_pack_belief(b["belief_mass"], (b["belief_com_x"], b["belief_com_y"], b["belief_com_z"]))
    return ctrl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="limx", choices=["limx", "sim"])
    ap.add_argument("--ip", default="10.192.1.2")
    ap.add_argument("--gains", default=None)
    ap.add_argument("--controller", default="classical", choices=["classical", "limx", "trained"],
                    help="limx = LimX's open RL policy; trained = our policy (metalsim.learn.tron1_wf); both with a "
                         "position/heading hold (metalsim.tron1.policy_hold)")
    ap.add_argument("--policy", default=None, help="trained: ONNX path (default assets/tron1/metalsim_policy/policy.onnx)")
    ap.add_argument("--no-hold", action="store_true", help="limx: policy alone, no position hold")
    ap.add_argument("--max-vx", type=float, default=0.5)
    ap.add_argument("--max-wz", type=float, default=0.8)
    ap.add_argument("--turn-sign", type=float, default=1.0)
    ap.add_argument("--wheel-torque", type=float, default=None,
                    help="N m per wheel; default 20 for classical (first sessions), 40 for limx (its trained "
                         "limit: at 20 the policy loses heading, 0.6 m / up to 115 deg drift in the sim)")
    ap.add_argument("--hz", type=float, default=500.0)
    a = ap.parse_args()
    wheel_torque = a.wheel_torque if a.wheel_torque is not None else (40.0 if a.controller in ("limx", "trained") else 20.0)
    cfg = BridgeConfig(max_vx=a.max_vx, max_wz=a.max_wz, turn_sign=a.turn_sign, wheel_torque=wheel_torque)
    ctrl = make_controller(a.gains, a.controller, hold=not a.no_hold, policy=a.policy)
    log_path = f"runs/tron1_bridge_{time.strftime('%Y%m%d_%H%M%S')}.npz"
    if a.backend == "limx":
        be = LimxBackend(a.ip)
        print(f"[bridge] connected to {a.ip} ({a.controller}). L1 + △ triangle = stand (hung); "
              f"L1 + ✕ cross = balance (on the ground, after the stand); L1 + □ square = damp (robot sags). "
              f"Button presses are echoed below.", flush=True)
        run(be, Bridge(ctrl, cfg), hz=a.hz, log_path=log_path)
    else:
        from .realism import SimParams
        from .sim import Tron1Sim
        log = simulate_procedure(ctrl, cfg, hz=a.hz, log_path=log_path)
        print(summarize(log))


# The first-session procedure, scripted: robot hanging from a rope with the wheels ~8 cm up,
# limp (damping) at start; stand; lower until the rope is slack; balance; drive; turn; stop.
PROCEDURE = dict(stand=0.2, lower=(3.6, 4.6), balance=4.2, drive=(6.0, 9.0), turn=(9.0, 11.0), damp=13.5, end=14.5)


def demo_joy(t):
    j = Joy()
    press = lambda *names: [j.buttons.__setitem__(BTN[n], 1) for n in names]
    P = PROCEDURE
    if P["stand"] < t < P["stand"] + 0.2:
        press("L1", "Y")
    if P["balance"] < t < P["balance"] + 0.2:
        press("L1", "A")
    if P["drive"][0] < t < P["drive"][1]:
        j.axes[AX_LEFT_V] = 0.8
    if P["turn"][0] < t < P["turn"][1]:
        j.axes[AX_RIGHT_H] = 0.6
    if P["damp"] < t < P["damp"] + 0.2:
        press("L1", "X")
    return j


def simulate_procedure(ctrl, cfg, params=None, hz=500.0, log_path=None, seed=0):
    """The whole hung-start procedure against Tron1Sim through the same Bridge code."""
    from .realism import SimParams
    from .sim import Tron1Sim
    sim = Tron1Sim(params or SimParams.nominal(), seed=seed)
    limp = ctrl.q_stance.copy()
    limp[[1, 5]] *= 0.3                                      # legs hanging, not at the stance
    limp[[2, 6]] *= 0.3
    sim.reset(limp)
    sim.d.qpos[2] += 0.08
    import mujoco
    mujoco.mj_forward(sim.m, sim.d)
    z_hung = float(sim.d.xpos[sim.base_id][2] + 0.03)
    sim.set_hoist(z_hung)
    lo0, lo1 = PROCEDURE["lower"]

    def step():
        t = sim.t_us * 1e-6
        if lo0 <= t <= lo1:
            sim.set_hoist(z_hung - 0.4 * (t - lo0) / (lo1 - lo0))
        for _ in range(int(round(1000 / hz))):
            sim.tick()

    bcfg = BridgeConfig(**{**vars(cfg), "require_calibration": False})
    bridge = Bridge(ctrl, bcfg)
    log = run(SimBackend(sim, demo_joy), bridge, hz=hz, duration=PROCEDURE["end"], clock=lambda: sim.t_us * 1e-6,
              step_sim=step, log_path=log_path)
    log = {"log": log, "fallen": sim.fallen, "base_z": float(sim.d.qpos[2]), "pos": sim.d.qpos[:2].copy()}
    return log


def summarize(res):
    L = res["log"]
    t, mode = L[:, 0], L[:, 1]
    changes = [(round(float(t[i]), 2), {0: DAMP, 1: STAND, 2: BALANCE}[int(mode[i])]) for i in range(1, len(t)) if mode[i] != mode[i - 1]]
    return f"mode changes {changes}; fallen {res['fallen']}; final base height {res['base_z']:.3f} m; moved to {np.round(res['pos'], 2)}"


if __name__ == "__main__":
    main()

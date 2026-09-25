"""TRON1 wheel-foot sim: model, hardware-realism layer, classical controller, LimX baseline, terrain."""
from dataclasses import replace

import mujoco
import numpy as np
import pytest

from metalsim.tron1.classical import ClassicalController, Tron1Model, wip_lqr
from metalsim.tron1.realism import SimParams
from metalsim.tron1.sdk import WF_MOTOR_NAMES, RobotCmd
from metalsim.tron1.sim import Tron1Sim, build_model


def test_model_matches_limx_description():
    m = build_model(SimParams(payload_mass=0.0))
    pack = m.body_mass[m.body("sensor_pack").id]                   # 1 g floor when payload_mass = 0
    assert abs(m.body_subtreemass[1] - pack - 22.273) < 1e-6       # tron1-robot-description WF_TRON1A
    # SDK motor order is the model's joint order after the free joint (LimX's simulator: qpos[7 + i]).
    assert [m.joint(i).name for i in range(1, m.njnt)] == list(WF_MOTOR_NAMES)
    assert m.opt.timestep == 0.0005


def test_ideal_sensing_is_exact_and_mode0_law():
    sim = Tron1Sim(SimParams.ideal()).reset(np.zeros(8))
    st = sim.latest_state
    assert np.allclose(st.q, sim.d.qpos[sim.jq])
    c = RobotCmd()
    c.Kp[:] = 10.0
    c.q[:] = 0.1
    c.tau[:] = 1.0
    sim.send(c)
    sim.tick()
    # After delivery the drive applies Kp (q - q_meas) + Kd (dq - dq_meas) + tau, no lag.
    expected = 10.0 * (0.1 - sim.q_enc) + 1.0
    sim._drive()
    assert np.allclose(sim.tau_motor, expected, atol=1e-9)


def test_sensor_pack_live_change_and_belief():
    c = ClassicalController()
    sim = Tron1Sim(SimParams.nominal()).reset(c.q_stance)
    m0 = sim.total_mass
    sim.set_pack(3.0, (0.02, 0.0, 0.12))
    assert abs(sim.total_mass - (m0 + 1.5)) < 1e-9 and np.allclose(sim.m.body_ipos[sim.pack_id], (0.02, 0.0, 0.12))
    out = sim.run(c, 4.0)
    assert not out["fallen"]
    c.set_pack_belief(3.0, (0.02, 0.0, 0.12))
    assert abs(c.model.total_mass - sim.total_mass) < 1e-6
    assert not sim.run(c, 3.0)["fallen"]


def test_command_latency():
    p = replace(SimParams.ideal(), cmd_delay=0.005)
    sim = Tron1Sim(p).reset(np.zeros(8))
    c = RobotCmd()
    c.tau[:] = 5.0
    sim.send(c)
    for _ in range(4):
        sim.tick()
        assert np.allclose(sim._active_cmd.tau, 0.0)          # still the initial damping command
    sim.tick()
    assert np.allclose(sim._active_cmd.tau, 5.0)              # arrived at 5 ms


def test_encoder_quantization_and_velocity_from_differences():
    p = replace(SimParams.ideal(), encoder_bits=12)
    sim = Tron1Sim(p).reset(np.array([0.0, 0.3, 0.5, 0.0, 0.0, -0.3, -0.5, 0.0]))
    sim.tick()
    step = 2 * np.pi / 4096
    q = sim.latest_state.q
    assert np.allclose(np.round(q / step) * step, q)
    assert np.allclose(np.round(sim.latest_state.dq * 1e-3 / step), sim.latest_state.dq * 1e-3 / step)


def test_torque_speed_envelope_limits_motoring_only():
    p = replace(SimParams.nominal(), current_lag=0.0)
    sim = Tron1Sim(p).reset(np.zeros(8))
    sim.d.qvel[sim.jv[3]] = p.speed_no_load[3]                # wheel spinning at no-load speed
    c = RobotCmd()
    c.tau[3] = 30.0                                           # motoring: no torque left
    c.tau[7] = 0.0
    sim._active_cmd = c
    sim._drive()
    assert abs(sim.tau_motor[3]) < 1e-9
    c.tau[3] = -30.0                                          # braking: full torque
    sim._drive()
    assert abs(sim.tau_motor[3] + 30.0) < 1e-9


def test_stance_ik_and_lqr():
    model = Tron1Model()
    for h in (0.6, 0.7):
        x = model.stance(h)
        assert np.linalg.norm(model._stance_residual(x, h)) < 1e-8
        K, A, B = wip_lqr(model.pendulum(model.mirror(*x)))
        assert np.all(np.linalg.eigvals(A - B @ K).real < 0)


def test_classical_holds_position_and_survives_shoves():
    c = ClassicalController()
    sim = Tron1Sim(SimParams.nominal(), seed=3).reset(c.q_stance)
    out = sim.run(c, 12.0, pushes=[(3.0, (0.0, 0.3)), (7.0, (0.4, 0.0))])
    assert not out["fallen"]
    assert np.linalg.norm(out["pos"][-1, :2] - out["pos"][0, :2]) < 0.15
    assert abs(np.degrees(out["yaw"][-1] - out["yaw"][0])) < 3.0


def test_classical_tracks_velocity_while_turning():
    c = ClassicalController()
    sim = Tron1Sim(SimParams.nominal(), seed=4).reset(c.q_stance)
    out = sim.run(c, 8.0, commands=lambda t: (0.0, 0.0, 0.0) if t < 1 else (0.8, 0.0, 0.5))
    assert not out["fallen"]
    k = out["t"] > 4
    assert abs(out["v_head"][k, 0].mean() - 0.8) < 0.1
    assert abs(out["yaw_rate"][k].mean() - 0.5) < 0.1


def test_limx_policy_balances_in_nominal_sim():
    pytest.importorskip("onnxruntime")
    from metalsim.tron1.limx_policy import LimxPolicyController
    c = LimxPolicyController()
    sim = Tron1Sim(SimParams.nominal(), seed=5).reset(c.initial_q())
    out = sim.run(c, 6.0, commands=lambda t: (0.0, 0.0, 0.0) if t < 2 else (0.5, 0.0, 0.0))
    assert not out["fallen"]


def test_scan_terrain_attaches_and_supports_robot():
    from metalsim.tron1.scene import ScanTerrain
    h = np.zeros((80, 80), np.float32)
    h[:, 60:] = 0.5                                           # a wall 1 m ahead
    t = ScanTerrain(h, 0.05, (-2.0, -2.0), (0.0, 0.0))
    c = ClassicalController()
    sim = Tron1Sim(SimParams.nominal(), terrain=t).reset(c.q_stance)
    assert sim.m.geom("scan_terrain").type == mujoco.mjtGeom.mjGEOM_HFIELD
    out = sim.run(c, 3.0)
    assert not out["fallen"] and abs(out["pos"][-1, 2] - out["pos"][0, 2]) < 0.05


def test_bridge_procedure_and_safety():
    from metalsim.tron1.limx_bridge import (BALANCE, DAMP, STAND, Bridge, BridgeConfig, Joy, BTN, make_controller,
                                            simulate_procedure)
    res = simulate_procedure(make_controller(), BridgeConfig())
    L = res["log"]
    modes = L[:, 1]
    seq = [int(m) for i, m in enumerate(modes) if i == 0 or m != modes[i - 1]]
    assert seq == [0, 1, 2, 0]                                       # DAMP -> STAND -> BALANCE -> DAMP
    t_damp = L[np.argmax((modes[1:] == 0) & (modes[:-1] == 2)) + 1, 0]
    assert t_damp > 13.0                                             # no fault ended BALANCE early

    c = make_controller()
    sim = Tron1Sim(SimParams.nominal()).reset(c.q_stance)
    b = Bridge(c, BridgeConfig())
    st, imu = sim.latest_state, sim.latest_imu
    j = Joy()
    j.buttons[BTN["L1"]] = j.buttons[BTN["Y"]] = 1
    b.update(st, imu, j, 0.0, 0.0, 0.0)
    assert b.mode == DAMP                                            # not calibrated -> refused
    b.on_diagnostic("calibration", 0, 0, "")
    b.update(st, imu, j, 0.0, 0.0, 0.0)
    assert b.mode == STAND
    j = Joy()
    j.buttons[BTN["L1"]] = j.buttons[BTN["A"]] = 1
    b.update(st, imu, j, 1.0, 0.0, 0.0)
    assert b.mode == STAND                                           # stand-up not finished yet
    b.update(st, imu, j, 3.5, 0.0, 0.0)
    assert b.mode == BALANCE
    b.update(st, imu, Joy(), 3.6, 0.2, 0.0)
    assert b.mode == DAMP and "stale" in b.reason                    # old state -> damp
    cmd = b.update(st, imu, Joy(), 3.7, 0.0, 0.0)
    assert np.all(cmd.Kp == 0) and np.all(cmd.Kd == 1.0)


def test_stock_hold_stops_policy_drift_and_yields_to_operator():
    pytest.importorskip("onnxruntime")                               # LimX policy (optional: pip install -e ".[tron1]")
    from metalsim.tron1.stock_hold import StockHold, simulate
    free = simulate(False, SimParams.nominal(), seed=1, seconds=20)
    held = simulate(True, SimParams.nominal(), seed=1, seconds=20, pushes=[(8.0, (0.4, 0.0))])
    assert not held["fallen"]
    assert held["drift"] < 0.1 and held["heading"] < 10.0          # pulse engages at 5 deg
    assert free["heading"] > 10.0                                    # the stand-in really drifts
    h = StockHold(track=0.5)
    assert h.step(0.0, sticks=(0.3, 0.0, 0.0, 0.0)) is None          # operator driving -> silent


def test_bridge_runs_limx_policy_with_hold():
    pytest.importorskip("onnxruntime")                               # LimX policy (optional: pip install -e ".[tron1]")
    from metalsim.tron1.limx_bridge import BridgeConfig, make_controller, simulate_procedure
    ctrl = make_controller(kind="limx")
    res = simulate_procedure(ctrl, BridgeConfig(wheel_torque=40.0))
    assert not res["fallen"] and res["base_z"] > 0.5


def test_controller_reset_is_fast_enough_for_the_control_loop():
    pytest.importorskip("onnxruntime")                               # LimX policy (optional: pip install -e ".[tron1]")
    # The bridge resets the controller inside the 500 Hz loop when BALANCE starts; a slow reset
    # (building the MuJoCo model, 125 ms on the Jetson) tripped the 20 ms stall watchdog.
    import time
    from metalsim.tron1.limx_bridge import make_controller
    for kind in ("limx", "classical"):
        c = make_controller(kind=kind)
        t0 = time.perf_counter()
        c.reset()
        assert time.perf_counter() - t0 < 0.005, kind

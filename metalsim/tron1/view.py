"""Live MuJoCo viewer for the TRON1 wheel-foot sim, in real time.

    .venv/bin/mjpython -m metalsim.tron1.view [--controller classical|limx] [--params ideal|nominal|random]
                                             [--seed N] [--scene PATH.npz]

Keys: arrows drive (up/down = forward speed, left/right = turn), space = stop, P = shove the
robot sideways, O = shove it forward, R = reset (new random robot with --params random) and reload the controller code from disk.
The robot restarts automatically when it falls. The terminal prints speed, pitch and roll.
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np

from .realism import SimParams
from .sim import Tron1Sim

# The macOS viewer (GLFW) changes the working directory, which breaks re-imports when the package
# was found through the cwd; pin the repo root on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT, KEY_SPACE = 265, 264, 263, 262, 32


def make_controller(name):
    from . import classical, limx_policy
    importlib.reload(classical)
    if name == "limx":
        importlib.reload(limx_policy)
        return limx_policy.LimxPolicyController()
    return classical.ClassicalController()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--controller", default="classical", choices=["classical", "limx"])
    ap.add_argument("--params", default="nominal", choices=["ideal", "nominal", "random"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scene", default=None)
    a = ap.parse_args()

    terrain = None
    if a.scene:
        from .scene import ScanTerrain
        terrain = ScanTerrain.load(a.scene)

    state = {"vx": 0.0, "wz": 0.0, "push": None, "reset": False, "seed": a.seed}

    def key(k):
        if k == KEY_UP:
            state["vx"] = min(state["vx"] + 0.2, 1.5)
        elif k == KEY_DOWN:
            state["vx"] = max(state["vx"] - 0.2, -1.5)
        elif k == KEY_LEFT:
            state["wz"] = min(state["wz"] + 0.25, 1.5)
        elif k == KEY_RIGHT:
            state["wz"] = max(state["wz"] - 0.25, -1.5)
        elif k == KEY_SPACE:
            state["vx"] = state["wz"] = 0.0
        elif k == ord("P"):
            state["push"] = (0.0, 0.4)
        elif k == ord("O"):
            state["push"] = (0.5, 0.0)
        elif k == ord("R"):
            state["reset"] = True

    state["key"] = key

    def build():
        if a.params == "random":
            p = SimParams.nominal().sample(np.random.default_rng(state["seed"]))
            state["seed"] += 1
        else:
            p = getattr(SimParams, a.params)()
        ctrl = make_controller(a.controller)
        sim = Tron1Sim(p, seed=state["seed"], terrain=terrain)
        start = ctrl.initial_q() if hasattr(ctrl, "initial_q") else ctrl.q_stance
        sim.reset(start, xy=terrain.spawn_xy if terrain is not None else (0.0, 0.0))
        return sim, ctrl

    while True:
        sim, ctrl = build()
        if not run_window(sim, ctrl, state):
            break


def run_window(sim, ctrl, state):
    """Run one robot until the window closes (False) or R / a fall asks for a new robot (True)."""
    with mujoco.viewer.launch_passive(sim.m, sim.d, key_callback=state["key"]) as v:
        v.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        v.cam.trackbodyid = sim.base_id
        v.cam.distance, v.cam.elevation, v.cam.azimuth = 3.0, -15, 120
        k, last_print = 0, 0.0
        t_wall = time.perf_counter()
        while v.is_running():
            if state["reset"] or sim.fallen:
                if sim.fallen:
                    print("fell - restarting", flush=True)
                    time.sleep(1.0)
                state["reset"] = False
                return True
            if state["push"] is not None:
                sim.push(state["push"])
                state["push"] = None
            vel = (state["vx"], 0.0, state["wz"])
            with v.lock():
                for _ in range(16):             # 16 ms of sim per frame
                    if k % 2 == 0:              # controller at 500 Hz
                        sim.send(ctrl.step(sim.latest_state, sim.latest_imu, vel, sim.t_us * 1e-6))
                    sim.tick()
                    k += 1
                    if sim.fallen:
                        break
            v.sync()
            t_wall += 0.016
            dt = t_wall - time.perf_counter()
            if dt > 0:
                time.sleep(dt)
            else:
                t_wall = time.perf_counter()
            tr = sim.truth()
            if tr["t"] - last_print > 0.5:
                last_print = tr["t"]
                print(f"t={tr['t']:6.1f}s  cmd vx={vel[0]:+.1f} wz={vel[2]:+.2f}  "
                      f"v={tr['v_head'][0]:+.2f} m/s  yaw rate={tr['yaw_rate']:+.2f}  "
                      f"pitch={np.degrees(tr['pitch']):+5.1f} roll={np.degrees(tr['roll']):+5.1f} deg", flush=True)
    return False


if __name__ == "__main__":
    main()

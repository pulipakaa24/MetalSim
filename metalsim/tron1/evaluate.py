"""Batch evaluation of TRON1 controllers across sim realism levels and randomized robots.

    .venv/bin/python -m metalsim.tron1.evaluate [--episodes 100] [--seconds 20] [--out docs/measurements/tron1_eval.json]

Each episode: the robot starts standing; every 2.5 s a new velocity command is drawn from LimX's
training ranges (vx in [-1, 1] m/s, yaw rate in [-0.6, 0.6] rad/s; tron1-rl-isaacgym
wheelfoot_flat_config.py), 25 % of segments are "stand still"; two horizontal shoves of
0.2-0.5 m/s in random directions at random times. Conditions:

* ideal     - LimX's tron1-mujoco-sim (exact sensing, no delay, ideal torque, 1 cm tyre disc)
* nominal   - best single guess of the real robot (`SimParams.nominal()`), noise seeds vary
* random    - one randomized robot per episode (`SimParams.nominal().sample(rng)`)

Metrics: fall rate; RMS forward-speed and yaw-rate tracking error over non-fallen time (skipping
0.5 s after each command change); peak |pitch| and |roll|.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

CONDITIONS = ("ideal", "nominal", "random")
CONTROLLERS = ("classical", "limx")


def episode(args):
    controller, condition, seed, seconds = args
    from .realism import SimParams
    from .sim import Tron1Sim
    rng = np.random.default_rng(10_000 + seed)
    if condition == "ideal":
        p = SimParams.ideal()
    elif condition == "nominal":
        p = SimParams.nominal()
    else:
        p = SimParams.nominal().sample(rng)
    if controller == "classical":
        from .classical import ClassicalController
        c = ClassicalController()
        q0 = c.q_stance
    else:
        from .limx_policy import LimxPolicyController
        c = LimxPolicyController()
        q0 = c.initial_q()

    seg = 2.5
    n = int(np.ceil(seconds / seg))
    cmds = np.stack([rng.uniform(-1, 1, n), np.zeros(n), rng.uniform(-0.6, 0.6, n)], 1)
    cmds[rng.random(n) < 0.25] = 0.0
    cmds[0] = 0.0
    commands = lambda t: tuple(cmds[min(int(t / seg), n - 1)])
    pushes = [(float(rng.uniform(1.0, seconds - 1.0)), tuple(rng.uniform(0.2, 0.5) * np.array([np.cos(a), np.sin(a)])))
              for a in rng.uniform(0, 2 * np.pi, 2)]

    sim = Tron1Sim(p, seed=seed).reset(q0)
    o = sim.run(c, seconds, commands=commands, pushes=pushes)
    t = o["t"]
    settled = (t % seg) > 0.5
    vx_err = o["v_head"][settled, 0] - o["cmd"][settled, 0]
    wz_err = o["yaw_rate"][settled] - o["cmd"][settled, 2]
    return dict(controller=controller, condition=condition, seed=seed, fallen=bool(o["fallen"]), t_end=float(o["t_end"]),
                vx_rmse=float(np.sqrt(np.mean(vx_err ** 2))) if len(vx_err) else float("nan"),
                wz_rmse=float(np.sqrt(np.mean(wz_err ** 2))) if len(wz_err) else float("nan"),
                pitch_max=float(np.degrees(np.abs(o["pitch"]).max())), roll_max=float(np.degrees(np.abs(o["roll"]).max())))


def summarize(rows):
    out = {}
    for c in CONTROLLERS:
        for k in CONDITIONS:
            r = [x for x in rows if x["controller"] == c and x["condition"] == k]
            if not r:
                continue
            ok = [x for x in r if not x["fallen"]]
            out[f"{c}/{k}"] = dict(
                episodes=len(r), falls=sum(x["fallen"] for x in r),
                vx_rmse_median=float(np.median([x["vx_rmse"] for x in ok])) if ok else None,
                wz_rmse_median=float(np.median([x["wz_rmse"] for x in ok])) if ok else None,
                pitch_max_p95=float(np.percentile([x["pitch_max"] for x in ok], 95)) if ok else None,
                roll_max_p95=float(np.percentile([x["roll_max"] for x in ok], 95)) if ok else None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--controllers", default=",".join(CONTROLLERS))
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--out", default="docs/measurements/tron1_eval.json")
    a = ap.parse_args()
    jobs = [(c, k, s, a.seconds) for c in a.controllers.split(",") for k in a.conditions.split(",")
            for s in range(a.episodes)]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 2)) as ex:
        rows = list(ex.map(episode, jobs, chunksize=4))
    summary = summarize(rows)
    for k, v in summary.items():
        print(f"{k:22s} falls {v['falls']:3d}/{v['episodes']:<3d}  vx rmse {v['vx_rmse_median'] or float('nan'):.3f} m/s  "
              f"wz rmse {v['wz_rmse_median'] or float('nan'):.3f} rad/s  pitch p95 {v['pitch_max_p95'] or float('nan'):5.1f}  "
              f"roll p95 {v['roll_max_p95'] or float('nan'):5.1f} deg")
    print(f"{len(rows)} episodes x {a.seconds:.0f} s in {time.time() - t0:.0f} s wall")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(dict(summary=summary, episodes=rows, seconds=a.seconds), f, indent=1)


if __name__ == "__main__":
    main()

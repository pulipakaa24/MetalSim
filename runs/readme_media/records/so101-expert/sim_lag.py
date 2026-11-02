"""Fontaine's lag metric on the simulated episodes: mean |command[t] - q[t+1]| per arm joint over the control ticks
(30 Hz), degrees, from <run>/traj.npz (eval_cpu.py / eval_metalsim.py); plus joint speed p99 / max (rad/s).
  python sim_lag.py <run dir> [hz]"""
import sys
import numpy as np
R = np.load(sys.argv[1] + "/traj.npz"); dt = float(R["dt"]); hz = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
ticks = sorted({int(round(k / hz / dt)) for k in range(int(6 * hz) + 1)})
ticks = [t for t in ticks if t < R["q"].shape[1]]
c, q = R["cmd"][:, ticks[:-1], :5], R["q"][:, [t - 1 for t in ticks[1:]], :5]   # command at tick k, q just before tick k+1
lag = np.degrees(np.nanmean(np.abs(c - q), axis=(0, 1)))
v = np.abs(R["qd"][..., :5]).reshape(-1, 5)
print("lag mean per arm joint (deg)", np.round(lag, 2), "all", round(float(lag.mean()), 2))
print("speed p99 (rad/s)", np.round(np.nanpercentile(v, 99, axis=0), 2), "max", np.round(np.nanmax(v, axis=0), 2))

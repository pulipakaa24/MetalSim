"""Runs `python -m metalsim.learn.g1_velocity <args>` unchanged, plus host-side timing of the Warp -> torch hand-over.

Per iteration it prints:
  encode_s  host time from rollout() start to BatchSim.after(v) (the wait PyTorch encodes on its command buffer)
  tail_s    time from after(v) until Warp's event reaches v on the host: how long the PyTorch command buffer that holds
            the wait (and the first part of update(): GAE, the clones of obs/act/logp) sits on the GPU waiting
  update_s  host time of update()
No code path of the training is changed: only wrappers that read the clock and a thread that waits on the event.

usage: g1_probe.py <g1_velocity args...>
"""
import runpy
import sys
import threading
import time

from metalsim.learn import ppo_warp
from metalsim.physics import batch

T = {}
_orig_rollout, _orig_update, _orig_after = ppo_warp.PPOWarp.rollout, ppo_warp.PPOWarp.update, batch.BatchSim.after


def _old_rollout(self):
    """PPOWarp.rollout as of main b29ffb2a (GPU-side hand-over only), for PROBE_OLD_HANDOVER=1."""
    import warp as wp
    if self.graph is None:
        self._capture()
    self.pol.rewind()
    for _ in range(self.cfg.rollout):
        wp.capture_launch(self.graph)
    self._obs()
    task = self.task
    self.pol.act(task.obs, task.action_scratch if hasattr(task, "launch_apply_action") else task.sim.d.ctrl)
    v = self.task.sim._signal()
    self.task.sim.after(v)


import os  # noqa: E402
if os.environ.get("PROBE_OLD_HANDOVER") == "1":
    _orig_rollout = _old_rollout
print(f"[probe] ppo_warp from {ppo_warp.__file__}, old hand-over {os.environ.get('PROBE_OLD_HANDOVER') == '1'}", flush=True)
it = [0]


def rollout(self):
    it[0] += 1
    T["r0"] = time.perf_counter()
    return _orig_rollout(self)


def after(self, value):
    t = time.perf_counter()
    T["after"] = t
    ev = self.event
    done = {}

    def waiter():
        ev.obj.waitUntilSignaledValue_timeoutMS_(value, 120_000)
        done["t"] = time.perf_counter()

    th = threading.Thread(target=waiter, daemon=True)
    th.start()
    T["waiter"] = (th, done)
    return _orig_after(self, value)


def update(self):
    u0 = time.perf_counter()
    out = _orig_update(self)
    u1 = time.perf_counter()
    th, done = T.pop("waiter", (None, {}))
    if th is not None:
        th.join(timeout=60)
    tail = done.get("t", float("nan")) - T.get("after", float("nan"))
    print(f"[probe] it {it[0]} encode_s {T.get('after', 0) - T.get('r0', 0):.3f} tail_s {tail:.3f} "
          f"update_s {u1 - u0:.3f} wall {time.strftime('%H:%M:%S')}", flush=True)
    return out


ppo_warp.PPOWarp.rollout, ppo_warp.PPOWarp.update, batch.BatchSim.after = rollout, update, after
sys.argv = ["metalsim.learn.g1_velocity"] + sys.argv[1:]
runpy.run_module("metalsim.learn.g1_velocity", run_name="__main__", alter_sys=True)

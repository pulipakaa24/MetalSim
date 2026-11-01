"""`python -m metalsim.learn.g1_velocity <args>` (the training CLI, unchanged) with one change to MetalSim's capacity
guard for these re-runs: MuJoCo Warp's EPA_HORIZON flag is reported but does not abort training. Every other capacity
flag (NEFC, NARROWPHASE, CCD, HFIELD, ...) still raises.

    .venv/bin/python runs/rough_rerun_2026-10-31/train_wrap.py 4096 rough train 1500 <log> <ckpt> 0.0025 <flags>

Why (smoke test 2026-10-31, runs/rough_rerun_2026-10-31/smoke/): on the fixed fork (93e5ce1) Isaac's G1 foot meshes
collide with heightfield prisms through GJK/EPA; EPA's horizon buffer is a compile-time 24 edges (MJ_MAX_EPAHORIZON,
mujoco_warp/_src/types.py:31). When it fills, _epa returns no face (collision_gjk.py ~1441-1500) and ccd() returns
FLOAT_MAX (no contact) for that prism in that substep, and sets OverflowType.EPA_HORIZON. BatchSim.check_overflow()
(batch.py:448) raises on it at every PPO log point since the latched guard (870d0d7f, 2026-10-13), so the rough_il3 run
died at iteration 1-2 (1-2 of 4096 worlds flagged). No task setting can enlarge the buffer; it needs a fork change. The
original runs (2026-09-25/26) had no guard at all. Here the flag stays visible: the AnomalyMonitor logs
"[anomaly] it N: MuJoCo Warp EPA_HORIZON capacity overflow in K worlds" every iteration (K = worlds flagged since the
start, latched) and in <log>.anomalies.jsonl, and this wrapper prints a line whenever K changes."""
import runpy, sys

from metalsim.physics import capacity

_orig = capacity.capacity_overflows
_last = [0]


def _capacity_overflows(flags: dict) -> dict:
    cap = _orig(flags)
    n = cap.pop("EPA_HORIZON", 0)
    if n != _last[0]:
        print(f"[rr-guard] EPA_HORIZON flagged in {n} worlds since the start (was {_last[0]}): a heightfield prism's EPA "
              f"horizon overflowed and that contact was dropped for the substep; not fatal in this re-run", flush=True)
        _last[0] = n
    return cap


capacity.capacity_overflows = _capacity_overflows
print("[rr-guard] capacity guard: EPA_HORIZON reported, not fatal; all other capacity flags fatal", flush=True)
sys.argv = ["metalsim.learn.g1_velocity"] + sys.argv[1:]
runpy.run_module("metalsim.learn.g1_velocity", run_name="__main__", alter_sys=True)

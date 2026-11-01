"""Run a script with G1VelocityTask keyword defaults overridden, so a protocol recorded under older task defaults is
reproduced without editing the shared script. Explicit keyword arguments of the script still win (setdefault).

    RR_TASK_DEFAULTS='{"contact_cfg": "default", "effort_limit": "actuator"}' \
        .venv/bin/python runs/rough_rerun_2026-10-31/task_defaults.py scripts/diagnostics/g1_rough_transfer.py <args>

Why: scripts/diagnostics/g1_rough_transfer.py builds G1VelocityTask with today's defaults (contact_cfg "recommended" =
elliptic since 1a1cf0db 2026-09-26 11:00; effort_limit "joint" since 18180101 2026-09-26 05:07). The transfer records it
is compared against (runs/g1_rough_transfer.jsonl, e88db43f 2026-09-25 07:36; runs/terrain_walls/transfer_reps.jsonl,
2026-09-25 morning) predate both: MuJoCo-default contacts (contact_cfg did not exist before b1473429, 14:52; "default" =
no tuning, contact_tuning.py:61) and the actuator forcerange (effort_limit "actuator")."""
import atexit, functools, json, os, runpy, sys

import metalsim.learn.g1_velocity as g1v

over = json.loads(os.environ.get("RR_TASK_DEFAULTS", "{}") or "{}")
_init = g1v.G1VelocityTask.__init__


@functools.wraps(_init)
def _patched(self, *a, **kw):
    for k, v in over.items():
        kw.setdefault(k, v)
    _init(self, *a, **kw)
    print(f"[task_defaults] G1VelocityTask n={self.n} terrain_collision={self.terrain_collision} scan_surface={self.scan_surface} "
          f"contact_cfg={self.contact_cfg} effort_limit={self.effort_limit} base_velocity={self.base_velocity} "
          f"solver_cfg={self.solver_cfg} cone={int(self.model.opt.cone)} (0 pyramidal, 1 elliptic) impratio={self.model.opt.impratio}",
          flush=True)
    atexit.register(_report_overflow, self)


def _report_overflow(task):
    """The transfer script never checks MuJoCo Warp's overflow flags; print them at exit (latched since construction:
    worlds with dropped rows / contacts, e.g. EPA_HORIZON on the fixed fork, and the solver-cap counts)."""
    try:
        print(f"[task_defaults] overflow flags at exit (worlds): {task.sim.overflow_flags()}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[task_defaults] overflow flags unavailable: {e!r}", flush=True)


g1v.G1VelocityTask.__init__ = _patched
print(f"[task_defaults] overrides {over}", flush=True)
script = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(script, run_name="__main__")

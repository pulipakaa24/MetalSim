"""Per-solve iteration and residual logging for the sand-fidelity probes (research_notes/sand_fidelity/PLAN_2026-10-14.md
section 8.1: does the rheology solve ever reach its tolerance, or does it always stop at ``max_iterations``?).

``MPMCoupling.finalize`` installs ``newton_compat`` with ``log=False`` and captures the coupled step, so the iteration
count is never read. :func:`enable` re-installs the same patches with ``log=True`` (``install`` is re-callable: it keeps
the original functions); the caller must build ``MPMConfig(capture=False)``, so that each solve runs through the
chunk-graph path, which reads the iteration counter back once per solve (same kernels, same order, same stopping
iteration as the captured step: ``newton_compat`` module docstring, ``gpu_exit``). :func:`drain` returns the log since
the last call; :func:`summary` reduces a list of drained logs.
"""
from __future__ import annotations

import numpy as np


def enable(cfg) -> None:
    from metalsim.physics.mpm import newton_compat

    if cfg.capture:
        raise ValueError("solve_log.enable needs MPMConfig(capture=False): a captured step never reads the count back")
    newton_compat.install(accumulate="float32", loop="cuda", log=True, chunk_graph=cfg.chunk_graph,
                          gpu_exit=cfg.gpu_exit, rigid_deterministic=None, device_args=None,
                          metal_fast_math=cfg.metal_fast_math)
    newton_compat.drain_log()


def drain() -> dict:
    from metalsim.physics.mpm import newton_compat

    return newton_compat.drain_log()


def summary(logs: list, cap: int) -> dict:
    """Fraction of solves that stopped at the cap, iteration and residual percentiles."""
    if not logs:
        return {"solves": 0}
    it = np.concatenate([np.asarray(x["iterations"]) for x in logs])
    l2 = np.concatenate([np.asarray(x["res_l2"], float) for x in logs])
    li = np.concatenate([np.asarray(x["res_linf"], float) for x in logs])
    if it.size == 0:
        return {"solves": 0}
    pct = lambda a: [float(v) for v in np.nanpercentile(a, [10, 50, 90, 100])] if np.isfinite(a).any() else None  # noqa: E731
    return {"solves": int(it.size), "cap": int(cap), "frac_at_cap": float(np.mean(it >= cap)),
            "iterations_pct_10_50_90_100": pct(it.astype(float)), "res_l2_pct_10_50_90_100": pct(l2),
            "res_linf_pct_10_50_90_100": pct(li)}

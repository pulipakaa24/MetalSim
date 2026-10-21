"""Which ``SolverImplicitMPM`` MetalSim runs.

Isaac Lab 3.0-EA's MPM is Newton 1.5.2's implicit solver; MetalSim's Metal changes to it live on the Newton fork's
``metalsim-1.5.2`` branch. In ``.venv-newton152`` that branch is the installed ``newton``. The main environment
(``.venv``) has the Newton 1.7 line, whose own implicit MPM is a later algorithm, so MetalSim carries the fork's
solver package as a generated copy (``metalsim/physics/mpm/implicit_mpm``, ``scripts/vendor_mpm152.py``): same
kernels, run against the environment's Newton for its ``Model`` / ``State`` types.

``METALSIM_MPM_SOLVER=newton`` or ``vendored`` forces one (the parity tests run both in ``.venv-newton152``).
"""
from __future__ import annotations

import os


def solver_kind() -> str:
    kind = os.environ.get("METALSIM_MPM_SOLVER", "auto")
    if kind not in ("auto", "newton", "vendored"):
        raise ValueError(f"METALSIM_MPM_SOLVER must be auto, newton or vendored, got {kind!r}")
    if kind != "auto":
        return kind
    from metalsim.physics.mpm.newton_compat import newton_is_152

    return "newton" if newton_is_152() else "vendored"


VARIANTS = {"parity": None, "pr4660": "metalsim.physics.mpm.implicit_mpm_pr4660",
            "pr4660c": "metalsim.physics.mpm.implicit_mpm_pr4660c"}   # pr4660c: its collider hunks only (research)


def solver_class(variant: str = "parity"):
    """``SolverImplicitMPM`` of the Newton 1.5.2 fork (Newton's own on the 1.5.2 line, else the vendored copy).

    ``variant`` other than ``"parity"``: a flag-gated variant generated from the vendored copy plus an upstream change
    (``scripts/vendor_mpm_variant.py``; ``"pr4660"``: Newton draft PR #4660's strain and collider robustness). Variants
    exist only as vendored packages, so they refuse ``METALSIM_MPM_SOLVER=newton``."""
    if variant not in VARIANTS:
        raise ValueError(f"solver variant must be one of {sorted(VARIANTS)}, got {variant!r}")
    if variant != "parity":
        if solver_kind() == "newton":
            raise ValueError(f"solver variant {variant!r} runs the vendored package; unset METALSIM_MPM_SOLVER=newton "
                             "(or use .venv rather than .venv-newton152)")
        import importlib  # noqa: PLC0415

        return importlib.import_module(VARIANTS[variant]).SolverImplicitMPM
    if solver_kind() == "newton":
        from newton.solvers import SolverImplicitMPM
    else:
        from metalsim.physics.mpm.implicit_mpm import SolverImplicitMPM
    return SolverImplicitMPM

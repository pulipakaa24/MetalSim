# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
# Modified by MetalSim (Apache-2.0 section 4): vendored from pulipakaa24/newton @ 3fd1f052a5ad031578960d9642aaaedc03834e4d
# (newton/_src/solvers/implicit_mpm/__init__.py) by scripts/vendor_mpm152.py; the only change is that relative imports leaving the package
# (``from ...x import`` / ``from ..x import``) are rewritten to absolute ``newton._src`` imports. Do not edit:
# change the fork, move the pin, and regenerate (metalsim/physics/mpm/MPM.md, section 11).

from .solver_implicit_mpm import SolverImplicitMPM

__all__ = ["SolverImplicitMPM"]

# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
# Modified by MetalSim (Apache-2.0 section 4): MetalSim's vendored 1.5.2 copy of this file (metalsim/physics/mpm/implicit_mpm/__init__.py)
# with the collider part (rasterized_collisions.py) of Newton draft PR #4660 at 1de47831 applied, by scripts/vendor_mpm_variant.py (variant 'pr4660c', selected with
# MPMConfig(solver_variant=...)). Do not edit: change the patch and regenerate.

from .solver_implicit_mpm import SolverImplicitMPM

__all__ = ["SolverImplicitMPM"]

"""Metal acceleration-structure size limits.

Metal supports, per acceleration structure (``MTLAccelerationStructureUsage.extendedLimits`` documentation):

| | standard | extended limits |
|---|---|---|
| primitives in a primitive structure | 2^28 | 2^30 |
| geometries in a primitive structure | 2^24 | 2^30 |
| instances in an instance structure | 2^24 | 2^30 |
| visibility mask bits | 8 | 32 |

A structure built past the standard limits without ``MTLAccelerationStructureUsageExtendedLimits`` is outside the
API's supported range (2026-10-12: a 25.1 M-instance build on macOS 26 kept the GPU busy after its process was killed
and preceded machine crashes), and one built with the flag must be intersected with the ``extended_limits`` tag in the
shader. ``check`` therefore fails loudly unless the caller has an extended-limits path.
"""
from __future__ import annotations

import Metal

STANDARD = {"primitives": 1 << 28, "geometries": 1 << 24, "instances": 1 << 24}
EXTENDED = 1 << 30


def check(kind: str, count: int, label: str, extended_ok: bool = False) -> bool:
    """Whether a structure with ``count`` ``kind`` ("primitives", "geometries", "instances") needs extended limits.
    Raises ValueError past the extended limit, or past the standard one when the caller has no extended-limits path
    (``extended_ok=False``: its shaders intersect without the ``extended_limits`` tag)."""
    if kind not in STANDARD:
        raise ValueError(f"unknown acceleration-structure limit {kind!r}")
    count = int(count)
    if count > EXTENDED:
        raise ValueError(f"{label}: {count:,} {kind} exceed Metal's extended limit of {EXTENDED:,} per acceleration "
                         f"structure; split the geometry over several structures")
    if count > STANDARD[kind]:
        if not extended_ok:
            raise ValueError(f"{label}: {count:,} {kind} exceed Metal's standard limit of {STANDARD[kind]:,} per "
                             f"acceleration structure (this path has no extended-limits support); split the geometry "
                             f"over several structures")
        return True
    return False


def apply(descriptor, extended: bool) -> None:
    """Set ``MTLAccelerationStructureUsageExtendedLimits`` on ``descriptor`` when ``extended``."""
    if extended:
        descriptor.setUsage_(descriptor.usage() | Metal.MTLAccelerationStructureUsageExtendedLimits)

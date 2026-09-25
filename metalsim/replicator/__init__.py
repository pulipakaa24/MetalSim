"""Replicator equivalent (WS6): randomizers over the batched scene, annotators from the renderer
and sensors, and dataset writers.

* ``Randomizer``: per-env GPU randomizers (colour, camera, intrinsics, light, ambient, backgrounds,
  materials).
* ``events``: Isaac Lab's ``mdp.events`` terms (physics and state randomization) and an
  ``EventManager`` with startup / reset / interval modes.
* ``Semantics``: labels on prims (Isaac's Semantics schema / ``UsdSemantics.LabelsAPI``).
* ``IsaacAnnotators``: Isaac Replicator's annotators (names, ids, dtypes, info dicts) computed on the
  GPU from the renderer's id/depth/normal buffers and the physics state; ``Annotators`` is the older
  per-slot interface.
* ``BasicWriter``: Isaac Replicator's BasicWriter file layout; ``CocoWriter``, ``KittiWriter``,
  ``NpzWriter``.
"""
from metalsim.replicator.randomizers import Randomizer  # noqa: F401
from metalsim.replicator.annotators import Annotators  # noqa: F401
from metalsim.replicator.semantics import Semantics  # noqa: F401
from metalsim.replicator.isaac import IsaacAnnotators, State  # noqa: F401
from metalsim.replicator.basic_writer import BasicWriter  # noqa: F401
from metalsim.replicator.writers import CocoWriter, KittiWriter, NpzWriter  # noqa: F401

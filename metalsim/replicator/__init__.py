"""Replicator equivalent (WS6): randomizers over the batched scene, annotators from the renderer
and sensors, and dataset writers (COCO, KITTI, raw tensors).

Randomization is per-env instance data (colors, camera pose deltas, lights, backgrounds, material
parameters) written on the GPU, so it costs nothing at render time; annotators are the renderer's
outputs (rgb, depth, normals, instance/semantic segmentation, 2-D/3-D bounding boxes derived from
segmentation and physics state); writers run on the CPU after one event wait per batch.
"""
from metalsim.replicator.randomizers import Randomizer  # noqa: F401
from metalsim.replicator.annotators import Annotators  # noqa: F401
from metalsim.replicator.writers import BasicWriter, CocoWriter, KittiWriter  # noqa: F401

"""3D Gaussian splat rendering for MetalSim (design, measurements and how to use: metalsim/render/SPLAT.md).

Public API (what a scene / filming layer needs):

* ``load_scene(path, A, t, crop="auto")`` -> ``PackedSplats``: load any supported file (``.ply`` 3DGS or SuperSplat
  compressed, ``.spz``, ``.splat``, ``.nurec``, or a prepared ``.npz``), apply the file -> sim affine map
  ``x_sim = A x_file + t`` exactly, crop floaters, pack for the GPU; cached on disk.
* ``make_transform(up=, scale=, yaw_deg=, translate=)`` / ``floor_plane_transform(normal, point)`` -> (A, t).
* ``SplatRenderer(splats, n_envs, width=, height=, ..., mesh=Tier0Renderer(...))``: N cameras per call into
  tier0-layout buffers (``rgb`` (N, H, W, 3) uint8, ``depth`` (N, H, W) float32 planar metres); ``render(sim)`` GPU path,
  ``render_host(cam_pos, cam_quat=)`` host path; ``mesh=`` composites the robot / objects with correct occlusion.

``SplatRenderer`` needs Metal and is imported lazily (``from metalsim.render.splat import SplatRenderer`` works; importing
this package alone does not touch the GPU, so loaders and the float64 reference run anywhere).
"""
from metalsim.render.splat.formats import GaussianCloud, load, load_nurec, load_ply, load_splat, load_spz  # noqa: F401
from metalsim.render.splat.scene import (PackedSplats, UP_AXES, auto_crop_box, floor_plane_transform, load_scene,  # noqa: F401
                                         make_transform, prepare)


def __getattr__(name):
    if name == "SplatRenderer":
        from metalsim.render.splat.renderer import SplatRenderer
        return SplatRenderer
    if name == "SplatSceneRenderer":
        from metalsim.render.splat.backend import SplatSceneRenderer
        return SplatSceneRenderer
    raise AttributeError(name)

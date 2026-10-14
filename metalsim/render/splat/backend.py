"""``SceneRenderer``-compatible splat backend: ``render(pos, quat_wxyz) -> {"rgb": (H, W, 3) uint8, "depth": (H, W) float32}``
with a MuJoCo camera pose (looks along -z, +y up), the same single-camera contract as ``Tier0Renderer.render_host``'s
camera override, so a harness or data engine can swap a mesh scene for a splat scene behind one interface.

Depth for the safety layer (``depth_source``): ``"splat"`` = median splat depth (z where transmittance crosses 0.5, planar,
0 = nothing), ``"mesh"`` = the planar depth of a mesh renderer at the same pose (``mesh_renderer``: any object with the
SceneRenderer ``render`` signature returning "depth"), ``"both"`` = splat depth in "depth" plus "depth_mesh".
"""
from __future__ import annotations

import math
import time

import numpy as np

from metalsim.render.splat.renderer import SplatRenderer
from metalsim.render.splat.scene import PackedSplats, load_scene, make_transform

# named file -> sim transforms; pass an (A, t) tuple for anything else
TRANSFORMS = {"none": lambda: (np.eye(3), np.zeros(3)), "y_up": lambda: make_transform(up="y"),
              "neg_y_up": lambda: make_transform(up="-y")}


def intrinsics(width, height, hfov_deg=None, vfov_deg=None):
    if hfov_deg is not None:
        fx = (width / 2) / math.tan(math.radians(hfov_deg) / 2)
    else:
        fx = (height / 2) / math.tan(math.radians(vfov_deg) / 2)
    K = np.array([[fx, 0, width / 2], [0, fx, height / 2], [0, 0, 1]])
    return K, float(np.degrees(2 * np.arctan(width / 2 / fx))), float(np.degrees(2 * np.arctan(height / 2 / fx)))


class SplatSceneRenderer:
    def __init__(self, splat, width=640, height=360, hfov_deg=None, vfov_deg=None, outputs=("rgb", "depth"),
                 transform="none", crop="auto", depth_source="splat", mesh_renderer=None, clear_color=(0.0, 0.0, 0.0),
                 near=0.05, log=print, **renderer_kw):
        t0 = time.perf_counter()
        self.width, self.height = width, height
        self.K, self.hfov, self.vfov = intrinsics(width, height, hfov_deg, vfov_deg)
        if isinstance(splat, PackedSplats):
            self.splats = splat
        else:
            A, t = TRANSFORMS[transform]() if isinstance(transform, str) else transform
            self.splats = load_scene(splat, A, t, crop=crop, log=log)
        self.load_s = time.perf_counter() - t0
        if depth_source not in ("splat", "mesh", "both"):
            raise ValueError(depth_source)
        if depth_source != "splat" and mesh_renderer is None:
            raise ValueError("depth_source mesh/both needs mesh_renderer")
        self.depth_source, self.mesh_renderer = depth_source, mesh_renderer
        want = ("rgb",) + (("depth",) if ("depth" in outputs or depth_source != "mesh") else ())
        t0 = time.perf_counter()
        self.r = SplatRenderer(self.splats, 1, width=width, height=height, fx=self.K[0, 0], fy=self.K[1, 1], cx=self.K[0, 2],
                               cy=self.K[1, 2], outputs=want, clear_color=clear_color, near=near, **renderer_kw)
        self.build_s = time.perf_counter() - t0
        self.outputs = tuple(outputs)

    def render(self, pos, quat_wxyz):
        out = self.r.render_host(np.asarray(pos, np.float32)[None], cam_quat=np.asarray(quat_wxyz, np.float64)[None])
        res = {"rgb": out["rgb"][0]}
        if "depth" in self.outputs or self.depth_source != "splat":
            if self.depth_source == "splat":
                res["depth"] = out["depth"][0]
            else:
                md = self.mesh_renderer.render(pos, quat_wxyz)["depth"]
                if self.depth_source == "mesh":
                    res["depth"] = md
                else:
                    res["depth"], res["depth_mesh"] = out["depth"][0], md
        return res

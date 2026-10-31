"""Configuration of the filming tool, shaped after Isaac Lab 3.0's own classes so a task's film entry reads like its
env config:

* ``ViewerCfg`` mirrors ``isaaclab.envs.ViewerCfg`` / ``KitVisualizerCfg`` (``source/isaaclab/isaaclab/envs/common.py``:
  ``eye``, ``lookat``, ``origin_type`` in world / env / asset_root / asset_body, ``env_index``, ``asset_name``,
  ``body_name``, ``resolution``), plus the vertical field of view and a tracking smoothing time.
* ``FilmCfg`` mirrors ``VideoRecorderCfg`` (``source/isaaclab/isaaclab/envs/utils/video_recorder_cfg.py``: ``fps``
  defaults to the env's ``render_fps`` = 1 / step_dt, ``video_length``, ``output_filename_prefix``) and adds the
  offline-render settings (tier, samples, lighting) that Isaac leaves to the Kit renderer.
* ``FilmTask`` is the per-task entry: which recorder adapter runs the rollout, its default viewer and film settings,
  the look (HDRI, ground texture, particle material, material overrides) and the asset list with sources and licences
  that the generated doc prints.

Nothing here imports a simulator: the CLI reads these in either virtual environment.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace


@dataclass
class ViewerCfg:
    """Camera of the film (Isaac Lab ``ViewerCfg`` fields). ``eye`` / ``lookat`` are in the frame ``origin_type``
    names: "world", "env" (the filmed env's origin; recordings store every env in its own frame, so "env" and "world"
    agree for them), "asset_root" / "asset_body" (follow the recorded track of ``body_name``; the offset ``eye - lookat``
    is kept, ``lookat`` is relative to the tracked point, smoothed over ``smooth_s`` seconds)."""

    eye: tuple = (1.4, 1.4, 0.9)
    lookat: tuple = (0.5, 0.0, 0.1)
    origin_type: str = "env"
    env_index: int = 0
    asset_name: str | None = None
    body_name: str | None = None
    resolution: tuple = (1920, 1080)
    fovy: float = 40.0
    smooth_s: float = 1.0
    orbit_deg_s: float = 0.0     # slow azimuth orbit about the look-at point, symmetric about mid-clip (0: none)


@dataclass
class LookCfg:
    """Offline-render look. ``hdri``: path relative to the Poly Haven root (``POLYHAVEN`` or assets/polyhaven);
    ``sun``: split the HDRI's sun into an analytic sun disk (outdoor maps; ``metalsim.render.hdr.extract_sun``);
    ``ground_texture``: Poly Haven texture for the geoms named in ``ground_geoms`` (texrepeat ``ground_repeat`` per m);
    ``particle_palette``: (r, g, b, weight) linear colours of the render grains; ``materials``: {geom-name regex:
    dict(rgba, roughness, metallic, specular)} overrides; ``exposure``: linear exposure (0 = automatic from the first
    frame: median luminance to 0.2 before ACES)."""

    hdri: str | None = "hdri/train/poly_haven_studio.hdr"
    hdri_yaw_deg: float = 0.0
    sun: bool = False
    ground_texture: str | None = None
    ground_geoms: tuple = ("ground_plane", "ground", "floor")
    ground_repeat: float = 1.0
    ground_tint: tuple = (1.0, 1.0, 1.0)
    particle_palette: tuple = ((0.60, 0.49, 0.35, 0.74), (0.76, 0.68, 0.55, 0.14), (0.40, 0.30, 0.21, 0.08),
                               (0.14, 0.13, 0.12, 0.04))
    grains_per_particle: int = 4
    grain_scale: float = 1.2
    materials: dict = field(default_factory=dict)
    exposure: float = 0.0


@dataclass
class FilmCfg:
    """What to film and how to render it (Isaac Lab ``VideoRecorderCfg`` names where they exist). ``num_envs``: envs
    simulated (more than filmed, so the selection rule can pick e.g. a successful episode); ``envs``: filmed envs,
    "auto" (the adapter's ranking) or a list; ``grid``: 1 = one env full frame, 2 = a 2 x 2 grid of the first four;
    ``video_length``: frames (None: until the filmed episodes end, at most ``max_seconds``); ``hold_s``: seconds the
    last frame of a finished episode is held; ``tier``: 2 = path tracer (Tier2Renderer), 0 = raster (Tier0Renderer);
    ``passes`` x ``spp`` samples per pixel (tier 2), ``bounces``, ``denoise`` ("oidn" or None). ``allow_early_end``:
    film an env whose first episode ends by a termination (an adapter's ``info["terminated"]``: a fall, not a time-out
    or a success); by default the record stage refuses such a film."""

    num_envs: int = 4
    envs: object = "auto"
    grid: int = 1
    seed: int = 7
    fps: float | None = None
    video_length: int | None = None
    max_seconds: float = 12.0
    hold_s: float = 1.0
    tier: int = 2
    passes: int = 64
    spp: int = 4
    bounces: int = 4
    denoise: str | None = "oidn"
    crf: int = 22
    stills: tuple = (0.25, 0.5, 0.8)
    output_filename_prefix: str = "film"
    allow_early_end: bool = False


@dataclass
class ShotCfg:
    """One rendered shot of a recording: ``grid`` 1 (one env) or 2 (2 x 2), its own camera (None: the task's viewer)
    and sample count (None: the film's)."""

    name: str = "hero"
    grid: int = 1
    viewer: ViewerCfg | None = None
    passes: int | None = None


@dataclass
class AssetRef:
    """One asset in the film: what it is, where it comes from, its licence, and how it is fetched."""

    name: str
    source: str
    url: str
    licence: str
    fetch: str = ""


@dataclass
class FilmTask:
    """A task's film entry. ``adapter``: "module:Class" of the recorder adapter (``metalsim.render.film.adapters``);
    ``venv``: interpreter the rollout needs ("main" = .venv, "newton152" = .venv-newton152: the Newton 1.5.2 tasks);
    ``checkpoint``: default policy (None: the adapter's zero / scripted actions); ``adapter_kw``: the adapter's own
    settings (e.g. task settings the checkpoint does not save, from its training log); ``title`` / ``summary`` /
    ``notes``: prose for the generated doc (``notes``: markdown, "How it was made")."""

    name: str
    adapter: str
    venv: str = "main"
    checkpoint: str | None = None
    adapter_kw: dict = field(default_factory=dict)
    title: str = ""
    summary: str = ""
    viewer: ViewerCfg = field(default_factory=ViewerCfg)
    film: FilmCfg = field(default_factory=FilmCfg)
    look: LookCfg = field(default_factory=LookCfg)
    assets: list = field(default_factory=list)
    notes: str = ""
    doc: str | None = None
    shots: list = field(default_factory=lambda: [ShotCfg()])

    def with_overrides(self, **kw) -> "FilmTask":
        return replace(self, **kw)


def to_dict(obj):
    return asdict(obj)

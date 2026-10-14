"""Command-line overrides shared by the film stages (each stage can run alone, in its own venv)."""
from __future__ import annotations

from dataclasses import replace


def add_override_args(ap) -> None:
    ap.add_argument("--checkpoint", default=None, help="policy checkpoint (default: the task entry's)")
    ap.add_argument("--num-envs", type=int, default=None, help="envs simulated (filmed ones are selected among them)")
    ap.add_argument("--envs", default=None, help="filmed envs: 'auto' or comma-separated ids")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--seconds", type=float, default=None, help="maximum film length [s]")
    ap.add_argument("--grid", type=int, default=None, choices=(1, 2), help="1: one env; 2: a 2 x 2 grid")
    ap.add_argument("--eye", type=float, nargs=3, default=None)
    ap.add_argument("--lookat", type=float, nargs=3, default=None)
    ap.add_argument("--origin-type", default=None, choices=("world", "env", "asset_root", "asset_body"))
    ap.add_argument("--body", default=None, help="tracked body (origin_type asset_body)")
    ap.add_argument("--fovy", type=float, default=None)
    ap.add_argument("--resolution", type=int, nargs=2, default=None)
    ap.add_argument("--tier", type=int, default=None, choices=(2,))
    ap.add_argument("--passes", type=int, default=None)
    ap.add_argument("--spp", type=int, default=None)
    ap.add_argument("--bounces", type=int, default=None)
    ap.add_argument("--hdri", default=None, help="Poly Haven HDRI path relative to the Poly Haven root")
    ap.add_argument("--exposure", type=float, default=None)
    ap.add_argument("--no-denoise", action="store_true",
                    help="render without the OIDN denoiser (recorded in render.json and the doc; more Monte Carlo noise)")
    ap.add_argument("--skip-preflight", action="store_true", help="do not check the later stages' dependencies first")


def apply_overrides(task, a):
    f, v, lk = task.film, task.viewer, task.look
    fk = {}
    if a.num_envs is not None:
        fk["num_envs"] = a.num_envs
    if a.envs is not None:
        fk["envs"] = "auto" if a.envs == "auto" else [int(x) for x in a.envs.split(",")]
    for name in ("seed", "grid", "tier", "passes", "spp", "bounces"):
        if getattr(a, name) is not None:
            fk[name] = getattr(a, name)
    if a.seconds is not None:
        fk["max_seconds"] = a.seconds
    if getattr(a, "no_denoise", False):
        fk["denoise"] = None
    vk = {}
    for name, key in (("eye", "eye"), ("lookat", "lookat"), ("origin_type", "origin_type"), ("body", "body_name"),
                      ("fovy", "fovy")):
        if getattr(a, name) is not None:
            vk[key] = tuple(getattr(a, name)) if isinstance(getattr(a, name), list) else getattr(a, name)
    if a.resolution is not None:
        vk["resolution"] = tuple(a.resolution)
    lkk = {}
    if a.hdri is not None:
        lkk["hdri"] = a.hdri
    if a.exposure is not None:
        lkk["exposure"] = a.exposure
    out = replace(task, film=replace(f, **fk), viewer=replace(v, **vk), look=replace(lk, **lkk))
    if a.checkpoint:
        out = replace(out, checkpoint=a.checkpoint)
    return out

"""Camera poses for splat scenes, in MuJoCo's camera convention (frame columns: x right, y up, z back; the camera looks
along -z), i.e. what ``SplatRenderer.render_host(cam_pos, cam_mat=)`` and ``Tier0Renderer`` take.

Used by the real-data tests and ``scripts/splat_bench.py``; handy for any tool that films a splat scene.
"""
from __future__ import annotations

import json
import os

import numpy as np

# data directory of the fetched public scenes (scripts/fetch_splat_scenes.sh)
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA_DIR = os.environ.get("METALSIM_SPLAT_DATA", os.path.join(REPO, "assets", "splats"))
INTERIORGS_REF = os.path.join(DATA_DIR, "interiorgs", "0204_840158")        # the reference scene (SPLAT.md §7b)
DXGL_REF = os.path.join(DATA_DIR, "dxgl", "modern_arm_chair.ply")
NUREC_LOUNGE = os.path.join(DATA_DIR, "nurec", "zh_lounge", "export_last.nurec")    # optional, CC-BY-4.0


def yaw_pitch_mat(yaw: float, pitch_down_deg: float = 0.0) -> np.ndarray:
    """Camera frame for a heading ``yaw`` (rad, about +z, 0 = looking along +x) pitched down by ``pitch_down_deg``."""
    th = np.radians(pitch_down_deg)
    f = np.array([np.cos(th) * np.cos(yaw), np.cos(th) * np.sin(yaw), -np.sin(th)])
    right = np.array([np.sin(yaw), -np.cos(yaw), 0.0])
    up = np.cross(right, f)
    return np.stack([right, up, -f], 1)


def look_at(pos, target, world_up=(0.0, 0.0, 1.0)) -> np.ndarray:
    p = np.asarray(pos, float); f = np.asarray(target, float) - p; f /= np.linalg.norm(f)
    up = np.asarray(world_up, float)
    if abs(f @ up) > 0.99:
        up = np.array([0.0, 1.0, 0.0])
    x = np.cross(f, up); x /= np.linalg.norm(x); y = np.cross(x, f)
    return np.stack([x, y, -f], 1)


def _inside(poly: np.ndarray, p: np.ndarray) -> bool:
    x, y = p; inside = False
    for (x0, y0), (x1, y1) in zip(poly, np.roll(poly, -1, 0)):
        if (y0 > y) != (y1 > y) and x < x0 + (y - y0) * (x1 - x0) / (y1 - y0):
            inside = not inside
    return inside


def _edge_dist(poly: np.ndarray, p: np.ndarray) -> float:
    d = np.inf
    for a, b in zip(poly, np.roll(poly, -1, 0)):
        ab = b - a; t = np.clip((p - a) @ ab / max(ab @ ab, 1e-12), 0, 1)
        d = min(d, float(np.linalg.norm(a + t * ab - p)))
    return d


def room_views(structure_json: str, height=0.83, pitch_down_deg=22.0, spacing=1.5, wall_margin=0.8, n_yaw=8):
    """InteriorGS ``structure.json`` (room floor polygons, z-up metres) -> (positions (M, 3), cam mats (M, 3, 3)):
    a ``spacing`` grid of floor points inside every room and at least ``wall_margin`` from its walls, at ``height``,
    ``n_yaw`` headings each, pitched down. Deterministic (sorted grid, fixed headings)."""
    rooms = json.load(open(structure_json)).get("rooms", [])
    pos, mats = [], []
    for r in rooms:
        poly = np.asarray(r.get("profile", []), float)[:, :2]
        if len(poly) < 3:
            continue
        lo, hi = poly.min(0), poly.max(0)
        for x in np.arange(lo[0] + spacing / 2, hi[0], spacing):
            for y in np.arange(lo[1] + spacing / 2, hi[1], spacing):
                p = np.array([x, y])
                if _inside(poly, p) and _edge_dist(poly, p) >= wall_margin:
                    for k in range(n_yaw):
                        pos.append([x, y, height]); mats.append(yaw_pitch_mat(2 * np.pi * k / n_yaw, pitch_down_deg))
    return np.asarray(pos, float).reshape(-1, 3), np.asarray(mats, float).reshape(-1, 3, 3)


def orbit_views(center, radius, n=8, elevation_deg=20.0):
    """``n`` cameras on a circle around ``center`` (z-up), looking at it from ``elevation_deg`` above the horizon."""
    c = np.asarray(center, float); e = np.radians(elevation_deg)
    pos = np.array([c + radius * np.array([np.cos(a) * np.cos(e), np.sin(a) * np.cos(e), np.sin(e)])
                    for a in np.linspace(0, 2 * np.pi, n, endpoint=False)])
    return pos, np.array([look_at(p, c) for p in pos])

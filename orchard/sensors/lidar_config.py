"""Lidar specifications in Isaac Sim's vocabulary.

Isaac's RTX lidar is described by `OmniSensorGenericLidarCoreAPI` attributes (scanType, rotation,
near/far range, range resolution/accuracy, emitter azimuth/elevation tables, fire times, returns)
and legacy JSON configs. ``LidarSpec`` holds the subset this stack implements today and produces
the beam table for ``RayTracer.make_lidar``; ``from_isaac_json`` reads the legacy JSON profile
format (``profile.emitterStates[0].azimuthDeg/elevationDeg``, ``nearRangeM``, ``farRangeM``,
``rangeAccuracyM``, ``scanRateBaseHz``, ``numberOfEmitters``), and ``rotary`` builds the common
spinning multi-channel pattern (channels = elevation rows, azimuth samples per revolution).

Range noise (``range_accuracy_m``, one-sigma) is applied on the torch side by ``apply_noise`` so the
ray kernel stays deterministic.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class LidarSpec:
    azimuth_deg: np.ndarray
    elevation_deg: np.ndarray
    near_range_m: float = 0.3
    far_range_m: float = 200.0
    range_accuracy_m: float = 0.02
    range_resolution_m: float = 0.004
    scan_rate_hz: float = 10.0
    name: str = "lidar"
    extra: dict = field(default_factory=dict)

    @property
    def n_beams(self) -> int:
        return int(len(self.azimuth_deg))

    def beams_rad(self):
        return np.deg2rad(self.azimuth_deg).astype(np.float32), np.deg2rad(self.elevation_deg).astype(np.float32)

    @classmethod
    def rotary(cls, channels_elevation_deg, azimuth_samples: int, **kw):
        el = np.asarray(channels_elevation_deg, np.float64)
        az = np.linspace(-180.0, 180.0, azimuth_samples, endpoint=False)
        A, E = np.meshgrid(az, el)
        return cls(A.ravel(), E.ravel(), **kw)

    @classmethod
    def from_isaac_json(cls, path: str):
        with open(path) as f:
            cfg = json.load(f)
        prof = cfg.get("profile", cfg)
        st = prof["emitterStates"][0]
        az = np.asarray(st["azimuthDeg"], np.float64)
        el = np.asarray(st["elevationDeg"], np.float64)
        return cls(az, el,
                   near_range_m=float(prof.get("nearRangeM", 0.3)), far_range_m=float(prof.get("farRangeM", 200.0)),
                   range_accuracy_m=float(prof.get("rangeAccuracyM", 0.02)),
                   range_resolution_m=float(prof.get("rangeResolutionM", 0.004)),
                   scan_rate_hz=float(prof.get("scanRateBaseHz", 10.0)),
                   name=cfg.get("name", "lidar"), extra={k: v for k, v in prof.items() if k != "emitterStates"})

    def make(self, rt, site: str | int):
        az, el = self.beams_rad()
        lidar = rt.make_lidar(site, az, el)
        lidar.spec = self
        return lidar

    def apply_noise(self, ranges: torch.Tensor, gen: torch.Generator | None = None) -> torch.Tensor:
        """Gaussian range noise (one sigma = range accuracy), quantized to the range resolution, on
        returns within [near, far]; non-returns stay 0."""
        noise = torch.randn(ranges.shape, generator=gen, device=ranges.device) * self.range_accuracy_m
        r = ranges + noise
        if self.range_resolution_m > 0:
            r = torch.round(r / self.range_resolution_m) * self.range_resolution_m
        valid = (ranges > 0) & (ranges >= self.near_range_m) & (ranges <= self.far_range_m)
        return torch.where(valid, r, torch.zeros_like(r))

"""Radar-lite: ray-cast radar detections with Doppler (radial velocity) and a Lambertian RCS proxy.

Isaac Sim's RTX radar (``OmniRadar`` prim, ``OmniSensorGenericRadarWpmDmatAPI``) ray traces the scene
with a motion BVH, forms a range-Doppler-angle data cube by digital beamforming and runs CFAR
detection; its point cloud (``IsaacComputeRTXRadarPointCloud``) reports per detection radial
distance, azimuth, elevation, radial velocity (m/s) and RCS (dBsm). This module reproduces that
output format with the smallest model that gives the same fields from physics state:

* detections = hits of a regular (azimuth x elevation) ray grid over the field of view, cast by
  the Metal ray tracer (the default lidar kernel: exact first hits);
* radial velocity = d(range)/dt = (v_hit_point - v_sensor) . ray direction (positive receding),
  from MuJoCo's body velocities (``cvel`` about ``subtree_com``) at the hit point and at the site;
* RCS proxy: each ray covers a solid angle Omega; the patch it hits (projected area r^2 Omega) is a
  Lambertian scatterer, sigma = 4 rho r^2 Omega cos(theta), with rho the per-geom reflectance of the
  lidar tables (so a plate at normal incidence sums to 4 rho A); reported in dBsm, detections under
  ``rcs_min_dbsm`` dropped (Isaac's RCS threshold coefficient);
* optional Gaussian noise on range / radial velocity / angles (``apply_noise``, torch side).

Not modelled: multipath, beam pattern / sidelobes, range-Doppler binning and ambiguity, CFAR,
clustering of rays into per-object detections.

GPU ordering: ``trace(after_value)`` runs the ray-cast after the sim event, then a Warp kernel on
the sim's queue; it returns a value of ``sim.event`` after which ``out`` tensors are valid.
"""
from __future__ import annotations

import numpy as np
import torch
import warp as wp

from metalsim.interop import torch_bridge as tb
from metalsim.interop import warp_metal as wm


@wp.kernel
def _radar_post(
    rng: wp.array2d(dtype=float), pts: wp.array3d(dtype=float), nrm: wp.array3d(dtype=float),
    slot: wp.array2d(dtype=int), beams: wp.array(dtype=wp.vec2), solid_angle: wp.array(dtype=float),
    slot_body: wp.array(dtype=int), slot_refl: wp.array(dtype=float), body_rootid: wp.array(dtype=int),
    site: int, site_body: int, site_xpos: wp.array2d(dtype=wp.vec3),
    cvel: wp.array2d(dtype=wp.spatial_vector), subtree_com: wp.array2d(dtype=wp.vec3),
    rcs_min_dbsm: float,
    out_range: wp.array2d(dtype=float), out_az: wp.array2d(dtype=float), out_el: wp.array2d(dtype=float),
    out_vr: wp.array2d(dtype=float), out_rcs: wp.array2d(dtype=float), out_valid: wp.array2d(dtype=wp.bool),
):
    w, k = wp.tid()
    r = rng[w, k]
    s = slot[w, k]
    ab = beams[k]
    out_az[w, k] = ab[0]
    out_el[w, k] = ab[1]
    if r <= 0.0 or s < 0:
        out_range[w, k] = 0.0
        out_vr[w, k] = 0.0
        out_rcs[w, k] = -1000.0
        out_valid[w, k] = False
        return
    p = wp.vec3(pts[w, k, 0], pts[w, k, 1], pts[w, k, 2])
    n = wp.vec3(nrm[w, k, 0], nrm[w, k, 1], nrm[w, k, 2])
    o = site_xpos[w, site]
    d = wp.normalize(p - o)
    b = slot_body[s]
    vp = wp.vec3(0.0)
    if b > 0:
        cv = cvel[w, b]
        om = wp.spatial_top(cv)
        vp = wp.spatial_bottom(cv) + wp.cross(om, p - subtree_com[w, body_rootid[b]])
    vs = wp.vec3(0.0)
    if site_body > 0:
        cs = cvel[w, site_body]
        oms = wp.spatial_top(cs)
        vs = wp.spatial_bottom(cs) + wp.cross(oms, o - subtree_com[w, body_rootid[site_body]])
    cos_t = wp.max(-wp.dot(n, d), 0.0)
    sigma = 4.0 * slot_refl[s] * r * r * solid_angle[k] * cos_t
    db = 10.0 * wp.log10(wp.max(sigma, 1.0e-30))
    out_range[w, k] = r
    out_vr[w, k] = wp.dot(vp - vs, d)
    out_rcs[w, k] = db
    out_valid[w, k] = db >= rcs_min_dbsm


class Radar:
    """Ray-cast radar on a ``RayTracer`` and a ``BatchSim``.

    Args:
        rt, sim: the ray tracer and the MuJoCo Warp batch sim it traces.
        site: sensor site (x forward, as the lidar).
        fov_deg: (azimuth, elevation) full field of view; n_az x n_el rays over it.
        reflectance: per-geom reflectance overrides as for ``Lidar`` (dict or (ngeom,) array).
        rcs_min_dbsm: detections with a lower RCS are reported invalid.
    """

    def __init__(self, rt, sim, site, fov_deg=(120.0, 20.0), n_az: int = 64, n_el: int = 8, *, reflectance=None,
                 rcs_min_dbsm: float = -40.0, range_std_m: float = 0.0, vr_std_mps: float = 0.0, angle_std_deg: float = 0.0):
        self.rt, self.sim = rt, sim
        m = rt.m
        daz, del_ = np.deg2rad(fov_deg[0]) / n_az, np.deg2rad(fov_deg[1]) / n_el
        az = (np.arange(n_az) + 0.5) * daz - np.deg2rad(fov_deg[0]) / 2
        el = (np.arange(n_el) + 0.5) * del_ - np.deg2rad(fov_deg[1]) / 2
        A, E = np.meshgrid(az, el)
        self.lidar = rt.make_lidar(site, A, E)
        self.lidar.name = "radar_rays"
        self.n_rays = self.lidar.n_beams
        omega = (daz * del_ * np.cos(E.ravel())).astype(np.float32)   # solid angle per ray
        refl = rt.slot_reflectance(reflectance)
        dev = sim.device
        geoms = np.asarray(rt.tables.geoms)
        self.site = self.lidar.site
        self.site_body = int(m.site_bodyid[self.site])
        self._beams = wp.array(self.lidar.beams, dtype=wp.vec2, device=dev)
        self._omega = wp.array(omega, dtype=float, device=dev)
        self._slot_body = wp.array(m.geom_bodyid[geoms].astype(np.int32), dtype=int, device=dev)
        self._slot_refl = wp.array(refl.astype(np.float32), dtype=float, device=dev)
        self._rootid = wp.array(m.body_rootid.astype(np.int32), dtype=int, device=dev)
        self.rcs_min_dbsm = float(rcs_min_dbsm)
        self.range_std_m, self.vr_std_mps, self.angle_std = range_std_m, vr_std_mps, np.deg2rad(angle_std_deg)
        n = sim.n
        z = lambda dt=float: wp.zeros((n, self.n_rays), dtype=dt, device=dev)
        self._arrays = {"range": z(), "azimuth": z(), "elevation": z(), "radial_velocity": z(), "rcs_dbsm": z(), "valid": z(wp.bool)}
        wp.synchronize_device(dev)
        self.out = {k: tb.mps_tensor(a) for k, a in self._arrays.items()}
        self.out["points"] = self.lidar.out["points"]; self.out["slot"] = self.lidar.out["slot"]

    def trace(self, after_value: int | None = None) -> int:
        """Ray-cast after the sim reaches ``after_value``, then compute Doppler/RCS on the sim queue.
        Returns the ``sim.event`` value after which ``out`` is valid."""
        sim, rt, L = self.sim, self.rt, self.lidar
        v = rt.trace(sim, [L], after_value)
        wm.wait(rt.event, v, sim.device)
        a = self._arrays
        wp.launch(_radar_post, dim=(sim.n, self.n_rays), device=sim.device, inputs=[
            L._arrays["range"], L._arrays["points"], L._arrays["normals"], L._arrays["slot"], self._beams, self._omega,
            self._slot_body, self._slot_refl, self._rootid, self.site, self.site_body, sim.d.site_xpos,
            sim.d.cvel, sim.d.subtree_com, self.rcs_min_dbsm,
            a["range"], a["azimuth"], a["elevation"], a["radial_velocity"], a["rcs_dbsm"], a["valid"]])
        return sim._signal()

    def apply_noise(self, gen: torch.Generator | None = None) -> dict:
        """Noisy copies (torch) of range / radial velocity / azimuth / elevation on valid detections."""
        o = self.out; valid = o["valid"]
        def add(x, std):
            if std <= 0:
                return x.clone()
            return torch.where(valid, x + torch.randn(x.shape, generator=gen, device=x.device) * std, x)
        return {"range": add(o["range"], self.range_std_m), "radial_velocity": add(o["radial_velocity"], self.vr_std_mps),
                "azimuth": add(o["azimuth"], self.angle_std), "elevation": add(o["elevation"], self.angle_std),
                "rcs_dbsm": o["rcs_dbsm"].clone(), "valid": valid.clone()}

    def numpy(self) -> dict:
        wp.synchronize_device(self.sim.device)
        out = {k: a.numpy().copy() for k, a in self._arrays.items()}
        out["points"] = self.lidar._arrays["points"].numpy().copy(); out["slot"] = self.lidar._arrays["slot"].numpy().copy()
        return out

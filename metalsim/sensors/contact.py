"""Contact sensor on the MuJoCo Warp path, equivalent to Isaac Lab's ``ContactSensor`` (v2.3.2,
reference copy under ``assets/isaac/sensors/contact_sensor``).

Per tracked body and per env, from MuJoCo Warp's contact buffer and constraint forces
(``d.contact.*``, ``d.efc.force``), as Warp kernels launched after every physics substep inside the
captured step graph (``BatchSim.add_substep_hook``):

* ``net_forces_w`` (N, B, 3): net contact force on the body, world frame. ``force_mode="normal"``
  (default) sums only the normal components, which is what Isaac's ``net_forces_w`` documents
  ("sum of the normal contact forces ... not the total contact forces"); ``"total"`` adds the
  friction components, i.e. the sum of MuJoCo's ``mj_contactForce`` over the body's contacts.
* ``net_forces_w_history`` (N, T, B, 3): the last T substeps, index 0 the most recent (Isaac's
  ``history_length``; T = 1 when history_length is 0, as Isaac's ``unsqueeze(1)``).
* ``force_matrix_w`` (N, B, M, 3) and its history (N, T, B, M, 3): the same force restricted to
  contacts with each of M filter sets (Isaac's ``filter_prim_paths_expr``; a filter entry is a body
  (all its geoms) or an explicit set of geoms).
* air / contact time (``track_air_time``): ``current_air_time``, ``last_air_time``,
  ``current_contact_time``, ``last_contact_time`` (N, B), updated with Isaac's formulas, contact
  iff ``|net_forces_w| > force_threshold``; ``compute_first_contact(dt)`` / ``compute_first_air(dt)``
  as Isaac's.

Sign convention: MuJoCo's contact frame normal points from geom[0] to geom[1] and the contact
force acts on geom[1]'s body with + sign and on geom[0]'s body with - sign (``mj_rnePostConstraint``),
so a body resting on the floor reads +m g along +z, as Isaac's.

The values after a substep are the forces MuJoCo computed for that substep (the constraint solve
at the start of ``mj_step``), same as reading ``mj_contactForce`` after ``mj_step`` in MuJoCo C.
"""
from __future__ import annotations

import re
from typing import Sequence

import mujoco
import numpy as np
import torch
import warp as wp

from mujoco_warp._src.support import contact_force_fn
from mujoco_warp._src.types import vec5

from metalsim.interop import torch_bridge as tb


@wp.kernel
def _accumulate(
    opt_cone: int,
    contact_frame: wp.array(dtype=wp.mat33),
    contact_friction: wp.array(dtype=vec5),
    contact_dim: wp.array(dtype=int),
    contact_efc_address: wp.array2d(dtype=int),
    contact_adhesion: wp.array(dtype=float),
    contact_geom: wp.array(dtype=wp.vec2i),
    contact_worldid: wp.array(dtype=int),
    efc_force: wp.array2d(dtype=float),
    njmax: int,
    nacon: wp.array(dtype=int),
    geom_track: wp.array(dtype=int),     # geom -> tracked body index, -1 not tracked
    geom_filter: wp.array(dtype=int),    # geom -> filter index, -1 none
    n_filter: int,
    total: int,                          # 1: normal + friction, 0: normal only
    acc: wp.array3d(dtype=float),        # (N, B, 3)
    acc_mat: wp.array4d(dtype=float),    # (N, B, max(M,1), 3)
):
    cid = wp.tid()
    if cid >= nacon[0]:
        return
    g0 = contact_geom[cid][0]
    g1 = contact_geom[cid][1]
    if g0 < 0 or g1 < 0:
        return
    t0 = geom_track[g0]
    t1 = geom_track[g1]
    if t0 < 0 and t1 < 0:
        return
    w = contact_worldid[cid]
    f6 = contact_force_fn(opt_cone, contact_frame, contact_friction, contact_dim, contact_efc_address,
                          contact_adhesion, efc_force, njmax, nacon, w, cid, False)
    fr = contact_frame[cid]
    # world = frame^T f (rows of frame are normal, tangent1, tangent2)
    fw = wp.vec3(fr[0, 0], fr[0, 1], fr[0, 2]) * f6[0]
    if total != 0:
        fw = fw + wp.vec3(fr[1, 0], fr[1, 1], fr[1, 2]) * f6[1] + wp.vec3(fr[2, 0], fr[2, 1], fr[2, 2]) * f6[2]
    if t1 >= 0:     # force on geom[1]'s body: +F
        for k in range(3):
            wp.atomic_add(acc, w, t1, k, fw[k])
        if n_filter > 0:
            fi = geom_filter[g0]
            if fi >= 0:
                for k in range(3):
                    wp.atomic_add(acc_mat, w, t1, fi, k, fw[k])
    if t0 >= 0:     # force on geom[0]'s body: -F
        for k in range(3):
            wp.atomic_add(acc, w, t0, k, -fw[k])
        if n_filter > 0:
            fi = geom_filter[g1]
            if fi >= 0:
                for k in range(3):
                    wp.atomic_add(acc_mat, w, t0, fi, k, -fw[k])


@wp.kernel
def _commit(
    acc: wp.array3d(dtype=float), acc_mat: wp.array4d(dtype=float), n_filter: int, M1: int, T: int,
    track_air: int, threshold: float, dt: float,
    net: wp.array3d(dtype=float), hist: wp.array4d(dtype=float),
    mat: wp.array4d(dtype=float), mat_hist: wp.array4d(dtype=float),   # mat_hist (N, T, B*M1, 3)
    cur_air: wp.array2d(dtype=float), last_air: wp.array2d(dtype=float),
    cur_con: wp.array2d(dtype=float), last_con: wp.array2d(dtype=float),
):
    w, b = wp.tid()
    fx = acc[w, b, 0]
    fy = acc[w, b, 1]
    fz = acc[w, b, 2]
    acc[w, b, 0] = 0.0
    acc[w, b, 1] = 0.0
    acc[w, b, 2] = 0.0
    net[w, b, 0] = fx
    net[w, b, 1] = fy
    net[w, b, 2] = fz
    # history: roll by one (index 0 most recent), as Isaac's .roll(1, dims=1)
    for tt in range(T - 1):
        t = T - 1 - tt
        for k in range(3):
            hist[w, t, b, k] = hist[w, t - 1, b, k]
    hist[w, 0, b, 0] = fx
    hist[w, 0, b, 1] = fy
    hist[w, 0, b, 2] = fz
    for fi in range(n_filter):
        for k in range(3):
            v = acc_mat[w, b, fi, k]
            acc_mat[w, b, fi, k] = 0.0
            mat[w, b, fi, k] = v
            for tt in range(T - 1):
                t = T - 1 - tt
                mat_hist[w, t, b * M1 + fi, k] = mat_hist[w, t - 1, b * M1 + fi, k]
            mat_hist[w, 0, b * M1 + fi, k] = v
    if track_air != 0:
        # Isaac Lab ContactSensor._update_buffers_impl, elapsed_time = physics dt
        is_contact = wp.sqrt(fx * fx + fy * fy + fz * fz) > threshold
        ca = cur_air[w, b]
        cc = cur_con[w, b]
        if ca > 0.0 and is_contact:
            last_air[w, b] = ca + dt
        if cc > 0.0 and not is_contact:
            last_con[w, b] = cc + dt
        if is_contact:
            cur_air[w, b] = 0.0
            cur_con[w, b] = cc + dt
        else:
            cur_air[w, b] = ca + dt
            cur_con[w, b] = 0.0


@wp.kernel
def _reset(
    mask: wp.array(dtype=wp.bool), n_filter: int, M1: int, T: int,
    net: wp.array3d(dtype=float), hist: wp.array4d(dtype=float),
    mat: wp.array4d(dtype=float), mat_hist: wp.array4d(dtype=float),   # mat_hist (N, T, B*M1, 3)
    cur_air: wp.array2d(dtype=float), last_air: wp.array2d(dtype=float),
    cur_con: wp.array2d(dtype=float), last_con: wp.array2d(dtype=float),
):
    w, b = wp.tid()
    if not mask[w]:
        return
    for k in range(3):
        net[w, b, k] = 0.0
        for t in range(T):
            hist[w, t, b, k] = 0.0
        for fi in range(n_filter):
            mat[w, b, fi, k] = 0.0
            for t in range(T):
                mat_hist[w, t, b * M1 + fi, k] = 0.0
    cur_air[w, b] = 0.0
    last_air[w, b] = 0.0
    cur_con[w, b] = 0.0
    last_con[w, b] = 0.0


def _resolve_bodies(m: mujoco.MjModel, keys) -> list[int]:
    """Body ids from ids, exact names or regular expressions (full match), in the order given."""
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) or "" for i in range(m.nbody)]
    out = []
    for k in ([keys] if isinstance(keys, (str, int)) else keys):
        if isinstance(k, (int, np.integer)):
            out.append(int(k)); continue
        hit = [i for i, nm in enumerate(names) if nm == k] or [i for i, nm in enumerate(names) if re.fullmatch(k, nm)]
        if not hit:
            raise ValueError(f"no body matches {k!r}")
        out += [i for i in hit if i not in out]
    return out


class ContactSensor:
    """Isaac Lab ``ContactSensor`` equivalent on a ``BatchSim`` (MuJoCo Warp).

    Args:
        sim: the ``BatchSim``. With ``attach=True`` the sensor registers itself as a substep hook,
            so every ``sim.step()`` / ``sim.launch_step()`` updates it and ``sim.reset(mask)`` clears it.
        bodies: tracked bodies (ids, names or regexes; Isaac's ``prim_path`` leaf pattern).
        history_length: substeps of force history (Isaac's ``history_length``).
        filter: list of filter entries (Isaac's ``filter_prim_paths_expr``); each a body name/id or a
            list/tuple of geom names/ids. ``force_matrix_w[..., j, :]`` is the force from contacts
            with entry j.
        track_air_time, force_threshold: as Isaac's.
        force_mode: ``"normal"`` (Isaac's documented net normal force) or ``"total"`` (normal +
            friction, = sum of ``mj_contactForce``).
    """

    def __init__(self, sim, bodies, *, history_length: int = 0, filter: Sequence | None = None,
                 track_air_time: bool = False, force_threshold: float = 1.0, force_mode: str = "normal",
                 attach: bool = True):
        if force_mode not in ("normal", "total"):
            raise ValueError("force_mode must be 'normal' or 'total'")
        self.sim = sim
        m = sim.mj_model
        self.body_ids = _resolve_bodies(m, bodies)
        self.body_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in self.body_ids]
        self.B = len(self.body_ids)
        self.T = max(1, int(history_length))
        self.history_length = int(history_length)
        self.track_air_time = bool(track_air_time)
        self.force_threshold = float(force_threshold)
        self.total = 1 if force_mode == "total" else 0
        self.dt = float(m.opt.timestep)
        geom_track = np.full(m.ngeom, -1, np.int32)
        for i, b in enumerate(self.body_ids):
            geom_track[m.geom_bodyid == b] = i
        geom_filter = np.full(m.ngeom, -1, np.int32)
        filter = list(filter or [])
        for j, f in enumerate(filter):
            if isinstance(f, (list, tuple, set, np.ndarray)):
                gids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, g) if isinstance(g, str) else int(g) for g in f]
                if min(gids, default=0) < 0:
                    raise ValueError(f"unknown geom in filter entry {f!r}")
            else:
                (bid,) = _resolve_bodies(m, [f])[:1]
                gids = list(np.nonzero(m.geom_bodyid == bid)[0])
            for g in gids:
                if geom_filter[g] >= 0:
                    raise ValueError(f"geom {g} appears in more than one filter entry")
                geom_filter[g] = j
        self.M = len(filter)
        dev = sim.device
        n, B, T, M1 = sim.n, self.B, self.T, max(self.M, 1)
        z = lambda *s: wp.zeros(s, dtype=float, device=dev)
        self.geom_track = wp.array(geom_track, dtype=int, device=dev)
        self.geom_filter = wp.array(geom_filter, dtype=int, device=dev)
        self._acc = z(n, B, 3); self._acc_mat = z(n, B, M1, 3)
        self._net = z(n, B, 3); self._hist = z(n, T, B, 3)
        self._mat = z(n, B, M1, 3); self._mat_hist = z(n, T, B * M1, 3)
        self._cur_air = z(n, B); self._last_air = z(n, B); self._cur_con = z(n, B); self._last_con = z(n, B)
        wp.synchronize_device(dev)
        t = tb.mps_tensor
        self.net_forces_w = t(self._net)
        self.net_forces_w_history = t(self._hist)
        self.force_matrix_w = t(self._mat)[:, :, : self.M] if self.M else None
        self.force_matrix_w_history = t(self._mat_hist).view(n, T, B, M1, 3)[:, :, :, : self.M] if self.M else None
        if self.track_air_time:
            self.current_air_time = t(self._cur_air); self.last_air_time = t(self._last_air)
            self.current_contact_time = t(self._cur_con); self.last_contact_time = t(self._last_con)
        if attach:
            sim.add_substep_hook(self.launch, self.launch_reset)

    # -- graph pieces -----------------------------------------------------------------------------

    def launch(self) -> None:
        """Reduce this substep's contacts and update history / air time (one substep's worth)."""
        sim = self.sim
        d, mm = sim.d, sim.m
        wp.launch(_accumulate, dim=d.naconmax, device=sim.device, inputs=[
            int(mm.opt.cone), d.contact.frame, d.contact.friction, d.contact.dim, d.contact.efc_address,
            d.contact.adhesion, d.contact.geom, d.contact.worldid, d.efc.force, d.njmax, d.nacon,
            self.geom_track, self.geom_filter, self.M, self.total, self._acc, self._acc_mat])
        wp.launch(_commit, dim=(sim.n, self.B), device=sim.device, inputs=[
            self._acc, self._acc_mat, self.M, max(self.M, 1), self.T, int(self.track_air_time), self.force_threshold, self.dt,
            self._net, self._hist, self._mat, self._mat_hist,
            self._cur_air, self._last_air, self._cur_con, self._last_con])

    def launch_reset(self, mask: wp.array) -> None:
        wp.launch(_reset, dim=(self.sim.n, self.B), device=self.sim.device, inputs=[
            mask, self.M, max(self.M, 1), self.T, self._net, self._hist, self._mat, self._mat_hist,
            self._cur_air, self._last_air, self._cur_con, self._last_con])

    # -- Isaac helpers (torch, on the sensor's tensors; order after the sim event) -------------------

    def compute_first_contact(self, dt: float, abs_tol: float = 1.0e-8) -> torch.Tensor:
        if not self.track_air_time:
            raise RuntimeError("track_air_time is off")
        c = self.current_contact_time
        return (c > 0.0) * (c < (dt + abs_tol))

    def compute_first_air(self, dt: float, abs_tol: float = 1.0e-8) -> torch.Tensor:
        if not self.track_air_time:
            raise RuntimeError("track_air_time is off")
        a = self.current_air_time
        return (a > 0.0) * (a < (dt + abs_tol))

    def numpy(self) -> dict:
        """Host copies of all outputs (synchronizes; tests/tools)."""
        wp.synchronize_device(self.sim.device)
        out = {"net_forces_w": self._net.numpy().copy(), "net_forces_w_history": self._hist.numpy().copy()}
        if self.M:
            out["force_matrix_w"] = self._mat.numpy()[:, :, : self.M].copy()
            n, T, M1 = self.sim.n, self.T, max(self.M, 1)
            out["force_matrix_w_history"] = self._mat_hist.numpy().reshape(n, T, self.B, M1, 3)[:, :, :, : self.M].copy()
        if self.track_air_time:
            out.update(current_air_time=self._cur_air.numpy().copy(), last_air_time=self._last_air.numpy().copy(),
                       current_contact_time=self._cur_con.numpy().copy(), last_contact_time=self._last_con.numpy().copy())
        return out

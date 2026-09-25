"""Fused per-pixel annotator kernels (Warp on Metal), one pass over the batch each.

The torch formulations of these annotators materialize (N,H,W,3,3) gathers and several full-size
temporaries; at 1024 envs x 128^2 that costs 100-450 ms per annotator. These kernels read the
renderer's own output arrays (depth, id buffer, normals: Warp arrays aliased as the MPS tensors) and
write persistent Warp arrays that are returned as zero-copy MPS tensors (overwritten by the next
call, like Isaac Lab's camera buffers).

Ordering: ``Bridge.run`` makes the Warp queue wait for torch's pending work (which already waits on
the render), launches, and makes torch's later work wait for the kernel, with one shared event and
no host synchronization.
"""
from __future__ import annotations

import warp as wp

from metalsim.interop import torch_bridge as tb
from metalsim.interop import warp_metal as wm


class Bridge:
    def __init__(self, device="metal:0"):
        self.device = device
        self.ev = wm.SharedEvent(device, "replicator.kernels")
        self._bufs = {}

    def buffer(self, name, shape, dtype=float):
        """Persistent Warp array + its MPS alias, (re)allocated on shape change."""
        a = self._bufs.get(name)
        if a is None or tuple(a[0].shape) != tuple(shape):
            arr = wp.zeros(shape, dtype=dtype, device=self.device)
            wp.synchronize_device(self.device)      # the zero-fill must finish before any torch access
            a = (arr, tb.mps_tensor(arr))
            self._bufs[name] = a
        return a

    def run(self, kernel, dim, inputs):
        v = self.ev.next_value()
        tb.signal_event(self.ev, v)
        wm.wait(self.ev, v, self.device)
        wp.launch(kernel, dim=dim, inputs=inputs, device=self.device)
        v = self.ev.next_value()
        wm.signal(self.ev, v, self.device)
        tb.wait_event(self.ev, v)


@wp.func
def _cam_point(depth: float, x: int, y: int, cs: wp.array2d(dtype=float), e: int):
    fx = cs[e, 0]; fy = cs[e, 1]; cx = cs[e, 2]; cy = cs[e, 3]
    u = float(x) + 0.5
    v = float(y) + 0.5
    return wp.vec3(depth * (u - cx) / fx, -depth * (v - cy) / fy, -depth)


@wp.kernel
def pixel_kernel(depth: wp.array3d(dtype=float), normal_in: wp.array4d(dtype=float), cs: wp.array2d(dtype=float),
                 cam_pos: wp.array(dtype=wp.vec3), cam_rot: wp.array(dtype=wp.mat33), has_normal: int,
                 dist_cam: wp.array3d(dtype=float), world: wp.array4d(dtype=float), normals: wp.array4d(dtype=float)):
    """distance_to_camera (inf background), world points (0 background), unit normals with w flag."""
    e, y, x = wp.tid()
    d = depth[e, y, x]
    inf = float(1.0e30) * float(1.0e30)
    if d <= 0.0:
        dist_cam[e, y, x] = inf
        for i in range(3):
            world[e, y, x, i] = 0.0
        for i in range(4):
            normals[e, y, x, i] = 0.0
        return
    pc = _cam_point(d, x, y, cs, e)
    dist_cam[e, y, x] = wp.length(pc)
    pw = cam_rot[e] * pc + cam_pos[e]
    for i in range(3):
        world[e, y, x, i] = pw[i]
    if has_normal != 0:
        n = wp.vec3(normal_in[e, y, x, 0], normal_in[e, y, x, 1], normal_in[e, y, x, 2])
        ln = wp.length(n)
        if ln > 0.0:
            n = n / ln
        for i in range(3):
            normals[e, y, x, i] = n[i]
        normals[e, y, x, 3] = 1.0


@wp.func
def _valid_same(depth: wp.array3d(dtype=float), seg: wp.array3d(dtype=wp.int32), e: int, y: int, x: int,
                y2: int, x2: int, H: int, W: int):
    if y2 < 0 or y2 >= H or x2 < 0 or x2 >= W:
        return False
    return depth[e, y2, x2] > 0.0 and seg[e, y2, x2] == seg[e, y, x]


@wp.kernel
def normals_from_depth_kernel(depth: wp.array3d(dtype=float), seg: wp.array3d(dtype=wp.int32), cs: wp.array2d(dtype=float),
                              cam_rot: wp.array(dtype=wp.mat33), world_frame: int, out: wp.array4d(dtype=float)):
    """Depth-derived normals: central differences where both neighbours are on the same render prim,
    one-sided otherwise; oriented towards the camera."""
    e, y, x = wp.tid()
    H = depth.shape[1]
    W = depth.shape[2]
    for i in range(4):
        out[e, y, x, i] = 0.0
    d = depth[e, y, x]
    if d <= 0.0:
        return
    p = _cam_point(d, x, y, cs, e)
    okr = _valid_same(depth, seg, e, y, x, y, x + 1, H, W)
    okl = _valid_same(depth, seg, e, y, x, y, x - 1, H, W)
    okd = _valid_same(depth, seg, e, y, x, y + 1, x, H, W)
    oku = _valid_same(depth, seg, e, y, x, y - 1, x, H, W)
    if not (okr or okl) or not (okd or oku):
        return
    dx = wp.vec3(0.0)
    if okr and okl:
        dx = 0.5 * (_cam_point(depth[e, y, x + 1], x + 1, y, cs, e) - _cam_point(depth[e, y, x - 1], x - 1, y, cs, e))
    elif okr:
        dx = _cam_point(depth[e, y, x + 1], x + 1, y, cs, e) - p
    else:
        dx = p - _cam_point(depth[e, y, x - 1], x - 1, y, cs, e)
    dy = wp.vec3(0.0)
    if okd and oku:
        dy = 0.5 * (_cam_point(depth[e, y + 1, x], x, y + 1, cs, e) - _cam_point(depth[e, y - 1, x], x, y - 1, cs, e))
    elif okd:
        dy = _cam_point(depth[e, y + 1, x], x, y + 1, cs, e) - p
    else:
        dy = p - _cam_point(depth[e, y - 1, x], x, y - 1, cs, e)
    n = wp.cross(dx, dy)
    ln = wp.length(n)
    if ln <= 0.0:
        return
    n = n / ln
    if wp.dot(n, p) > 0.0:
        n = -n
    if world_frame != 0:
        n = cam_rot[e] * n
    for i in range(3):
        out[e, y, x, i] = n[i]
    out[e, y, x, 3] = 1.0


@wp.kernel
def motion_kernel(depth: wp.array3d(dtype=float), seg: wp.array3d(dtype=wp.int32), raw_geom: wp.array(dtype=wp.int32),
                  cs: wp.array2d(dtype=float), cam_pos: wp.array(dtype=wp.vec3), cam_rot: wp.array(dtype=wp.mat33),
                  pcam_pos: wp.array(dtype=wp.vec3), pcam_rot: wp.array(dtype=wp.mat33),
                  gpos: wp.array2d(dtype=wp.vec3), grot: wp.array2d(dtype=wp.mat33),
                  pgpos: wp.array2d(dtype=wp.vec3), pgrot: wp.array2d(dtype=wp.mat33), out: wp.array4d(dtype=float)):
    """(u_prev - u, v_prev - v, 0, 1): the pixel's surface point carried back rigidly with its geom
    and projected with the previous camera (Isaac: +x = motion to the left, +y = motion up)."""
    e, y, x = wp.tid()
    for i in range(4):
        out[e, y, x, i] = 0.0
    d = depth[e, y, x]
    g = raw_geom[seg[e, y, x]]
    if d <= 0.0 or g < 0:
        return
    pw = cam_rot[e] * _cam_point(d, x, y, cs, e) + cam_pos[e]
    local = wp.transpose(grot[e, g]) * (pw - gpos[e, g])
    p0 = pgrot[e, g] * local + pgpos[e, g]
    c = wp.transpose(pcam_rot[e]) * (p0 - pcam_pos[e])
    z = wp.max(-c[2], 1.0e-6)
    u0 = cs[e, 2] + cs[e, 0] * c[0] / z
    v0 = cs[e, 3] - cs[e, 1] * c[1] / z
    out[e, y, x, 0] = u0 - (float(x) + 0.5)
    out[e, y, x, 1] = v0 - (float(y) + 0.5)
    out[e, y, x, 3] = 1.0


@wp.kernel
def extents_kernel(seg: wp.array3d(dtype=wp.int32), raw_inst: wp.array(dtype=wp.int32), only: int,
                   box: wp.array3d(dtype=wp.int32), count: wp.array2d(dtype=wp.int32)):
    """Per (env, row): reduce the row's run of each instance, then one atomic per (row, instance)
    change point. box (N,E,4) must be preset to (big, big, -1, -1); ``only`` >= 0 restricts to one
    instance index (entity k -> id k + 2)."""
    e, y = wp.tid()
    W = seg.shape[2]
    x = int(0)
    while x < W:
        k = raw_inst[seg[e, y, x]] - 2
        x0 = x
        while x + 1 < W and raw_inst[seg[e, y, x + 1]] - 2 == k:
            x += 1
        if k >= 0 and (only < 0 or k == only):
            wp.atomic_min(box, e, k, 0, x0)
            wp.atomic_max(box, e, k, 2, x)
            wp.atomic_min(box, e, k, 1, y)
            wp.atomic_max(box, e, k, 3, y)
            wp.atomic_add(count, e, k, x - x0 + 1)
        x += 1


@wp.kernel
def reset_extents(box: wp.array3d(dtype=wp.int32), count: wp.array2d(dtype=wp.int32)):
    e, k = wp.tid()
    box[e, k, 0] = 1 << 30
    box[e, k, 1] = 1 << 30
    box[e, k, 2] = -1
    box[e, k, 3] = -1
    count[e, k] = 0

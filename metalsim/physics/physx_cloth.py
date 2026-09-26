# SPDX-License-Identifier: MIT AND BSD-3-Clause
# MetalSim code, MIT (see LICENSE). Algorithm ported from NVIDIA PhysX 5.6.1's GPU particle-cloth path:
# Copyright (c) 2008-2025, NVIDIA Corporation, BSD-3-Clause (licenses/BSD-3-Clause-PhysX.txt). See THIRD_PARTY_NOTICES.md.
"""PhysX 5 PBD particle cloth (PhysX 5.6.1, Isaac Sim 5.1) as Warp kernels, batched over worlds, graph-capturable.

A port of the algorithm of PhysX's GPU particle-cloth path (BSD-3; read in upstream/PhysX-5.6.1, see
docs/research/physx_deformables_port_2026-09-25.md §2 for every line reference). One ``step(dt)`` is one
``PxScene::simulate(dt)`` of a PBD particle system with the TGS solver:

* pre-integration: v <- (v + g dt)(1 - min(1, damping dt)) (``ps_preIntegrateLaunch``);
* particle-shape contacts generated once per step at the step-start positions: contact when the distance to the
  shape is <= particle contact offset + shape contact offset, error = distance - (particle rest offset + shape
  rest offset) (``cudaParticleSystem.cu`` ``particlePrimitiveCollision``, ``particleCollision.cuh``
  ``contactPointBox``); plane and box shapes;
* ``iterations`` TGS position iterations of h = dt / iterations, each: x <- x + v h (``ps_stepParticlesLaunch``);
  springs gathered into per-spring copies, one launch per combined partition (8) of the implicit spring-damper
  solve with PhysX's clamp (``ps_solveSpringsLaunch``), copies chained between partitions and blended back with
  1/(max copies + 1) (``NpParticleClothPreProcessor``, ``ps_averageVertsLaunch``), then v += d/h, x += d
  (``ps_updateParticleLaunch``); contacts with accumulated normal impulse, Coulomb friction bounded by the
  accumulated normal impulse, relaxation min(0.7, bias coefficient), averaged over the particle's active contacts
  (``solvePCOutputDeltaVTGS``, ``ps_accumulateDeltaVParticleLaunch``), then v += d/h, x += d;
* velocity iterations (5.6.1 semantics: the same position passes without the x += v h step);
* finalisation: if |v| > |dx_step / dt| then v <- s v + (1 - s) dx_step / dt with s the TGS bias coefficient
  min(0.9, 2 sqrt(1 / iterations)) (``ps_finalizeParticlesLaunch``).

Springs: PhysX's ``ExtParticleClothCooker`` rules (stretch = triangle edges, shear = both quad diagonals,
bending = distance springs between neighbours on a straight line through a vertex), or an explicit spring list
(``springs=``; e.g. the USD ``physxParticle:springIndices/...`` that Isaac Sim writes after cooking — the omni.physx
cooker itself is closed source, so the cooker rules are an assumption until those are available).

Not ported yet (raise or are absent): particle self-collision (hash grid), triangle-mesh / height-field one-way
contacts, two-way coupling (obstacles are read from arrays, e.g. MuJoCo Warp ``geom_xpos``/``geom_xmat``, i.e.
one-way), aerodynamics, inflatables, max-velocity clamp (PhysX default: infinite).

    cloth = grid_cloth(21, 1.0, z0=0.5)                           # vertices, triangles (PhysX recorder's cloth)
    sim = PhysXClothSim(cloth, num_envs=1024, obstacles=[plane(), box((0, 0, 0.2), (0.2, 0.2, 0.2))])
    sim.step()                                                    # graph replay
    sim.vertices()                                                # (N, n, 3) zero-copy MPS tensor
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import warp as wp

PARTITIONS_TEMP = 32      # NpParticleBuffer.cpp PARTICLE_MAX_NUM_PARTITIONS_TEMP
PARTITIONS_FINAL = 8      # PARTICLE_MAX_NUM_PARTITIONS_FINAL
SHAPE_PLANE, SHAPE_BOX = 0, 1

# spring classes (PxParticleClothConstraint::eTYPE_*)
HORIZONTAL, VERTICAL, DIAGONAL, BENDING = 1, 2, 4, 8


# ----------------------------------------------------------------------------------------------------------
# Host side: cloth, spring cooking, partitioning
# ----------------------------------------------------------------------------------------------------------

@dataclass
class ClothMesh:
    x: np.ndarray                 # (n, 3) rest = initial positions
    tris: np.ndarray              # (m, 3)
    inv_mass: np.ndarray          # (n,) 0 = fixed


def grid_cloth(n: int = 21, size: float = 1.0, z0: float = 0.5, mass: float = 0.02, center=(0.0, 0.0),
               vertical: bool = False) -> ClothMesh:
    """n x n vertices, `size` m, flat at height z0 (``vertical``: hanging in the x-z plane with its top row at z0),
    triangles split like the PhysX recorder's (a, b, d), (a, d, c)."""
    xs = np.linspace(-size / 2, size / 2, n)
    if vertical:
        pts = [(center[0] + x, center[1], z0 - (size / 2 - y)) for y in xs[::-1] for x in xs]
    else:
        pts = [(center[0] + x, center[1] + y, z0) for y in xs for x in xs]
    tris = []
    for j in range(n - 1):
        for i in range(n - 1):
            a, b, c, d = j * n + i, j * n + i + 1, (j + 1) * n + i, (j + 1) * n + i + 1
            tris += [(a, b, d), (a, d, c)]
    return ClothMesh(np.asarray(pts, np.float32), np.asarray(tris, np.int32), np.full(n * n, 1.0 / mass, np.float32))


def cook_springs(mesh: ClothMesh, vertical_dir=(0.0, 0.0, 1.0), bend_max_angle: float = math.radians(20.0),
                 types: int = HORIZONTAL | VERTICAL | DIAGONAL | BENDING):
    """``ExtParticleClothCooker::cookConstraints`` (all constraint types) → (i, j, rest, type) arrays.

    Deviation: PhysX iterates its edge hash set in bucket order; here unique edges are in sorted (a, b) order.
    The set of springs is the same; their order (and so the greedy partition assignment) can differ."""
    x = mesh.x.astype(np.float64)
    tris = mesh.tris
    edges = {}
    for t, (v0, v1, v2) in enumerate(tris):
        p0, p1, p2 = x[v0], x[v1], x[v2]
        l0, l1, l2 = np.linalg.norm(p0 - p1), np.linalg.norm(p1 - p2), np.linalg.norm(p2 - p0)
        longest = 0 if l0 > max(l1, l2) else (1 if l1 > l2 else 2)       # MaxArg semantics (ties → later)
        for k, (a, b) in enumerate(((v0, v1), (v1, v2), (v2, v0))):
            key = (min(a, b), max(a, b))
            if key not in edges:
                edges[key] = [t, -1, longest == k, False]
            else:
                edges[key][1] = t
                edges[key][3] = longest == k
    uniq = sorted(edges)
    ci, cj, rest, typ = [], [], [], []
    cos45 = math.cos(math.radians(45.0))
    vdir = np.asarray(vertical_dir, np.float64)
    extra = []
    for (a, b) in uniq:
        ta, tb, la, lb = edges[(a, b)]
        L = float(np.linalg.norm(x[a] - x[b]))
        if la and lb and (types & DIAGONAL):
            ci.append(a); cj.append(b); rest.append(L); typ.append(DIAGONAL)
            oa = [v for v in tris[ta] if v != a and v != b][0]
            ob = [v for v in tris[tb] if v != a and v != b][0]
            ci.append(min(oa, ob)); cj.append(max(oa, ob)); rest.append(float(np.linalg.norm(x[oa] - x[ob]))); typ.append(DIAGONAL)
            extra.append((min(oa, ob), max(oa, ob), True, True))
        else:
            d = (x[a] - x[b]) / L
            t = VERTICAL if float(vdir @ d) > cos45 else HORIZONTAL
            if types & t:
                ci.append(a); cj.append(b); rest.append(L); typ.append(t)
    if types & BENDING:
        # adjacency without the diagonals (eTYPE_DIAGONAL_BENDING_CONSTRAINT not set)
        nbr = [[] for _ in range(len(x))]
        for (a, b) in uniq:
            _, _, la, lb = edges[(a, b)]
            if la and lb:
                continue
            nbr[a].append(b); nbr[b].append(a)
        max_cos = math.cos(bend_max_angle)
        for i in range(len(x)):
            adj = nbr[i]
            for p in range(len(adj) - 1):
                d1 = x[adj[p]] - x[i]; d1 /= np.linalg.norm(d1)
                best, bq = -1.0, -1
                for q in range(p + 1, len(adj)):
                    d2 = x[adj[q]] - x[i]; d2 /= np.linalg.norm(d2)
                    c = abs(float(d1 @ d2))
                    if c > best:
                        best, bq = c, q
                if best > max_cos:
                    a, b = adj[p], adj[bq]
                    ci.append(a); cj.append(b); rest.append(float(np.linalg.norm(x[a] - x[b]))); typ.append(BENDING)
    return (np.asarray(ci, np.int32), np.asarray(cj, np.int32), np.asarray(rest, np.float32), np.asarray(typ, np.int32))


def partition_springs(si: np.ndarray, sj: np.ndarray, n_particles: int):
    """``NpParticleClothPreProcessor::partitionSprings``: greedy 32-bit partitions, combined into 8, copy chains.

    Returns (order, part_end[8], remap_out[2*ns], copy_end[n], blend_scale): springs reordered by combined
    partition; remap_out[s] for copy slot s (s < ns: end 0 of spring s, s >= ns: end 1 of spring s - ns) is the copy
    slot of the same particle in a later partition, or 2*ns + an accumulation slot; copies of particle p's
    accumulation slots are [copy_end[p-1], copy_end[p])."""
    ns = len(si)
    # classifySprings / writeSprings (same assignment, done once)
    part_of = np.full(ns, -1, np.int64)
    pending = list(range(ns))
    start = 0
    while pending:
        progress = [0] * n_particles
        nxt = []
        for s in pending:
            a, b = int(si[s]), int(sj[s])
            mask = ~(progress[a] | progress[b]) & 0xFFFFFFFF
            if mask == 0:
                nxt.append(s)
                continue
            bit = (mask & -mask).bit_length() - 1
            progress[a] |= 1 << bit
            progress[b] |= 1 << bit
            part_of[s] = start + bit
        pending = nxt
        start += PARTITIONS_TEMP
    n_parts = 0
    counts = np.bincount(part_of, minlength=start)
    while n_parts < len(counts) and counts[n_parts] > 0:
        n_parts += 1
    order_by_part = [[s for s in range(ns) if part_of[s] == p] for p in range(n_parts)]   # stable = PhysX write order
    # combinePartitions
    max_acc = (n_parts + PARTITIONS_FINAL - 1) // PARTITIONS_FINAL
    arr = max_acc * PARTITIONS_FINAL
    table = np.full((n_particles, arr), -1, np.int64)
    order, part_end = [], []
    for i in range(PARTITIONS_FINAL):
        for j in range(max_acc):
            pid = i + PARTITIONS_FINAL * j
            if pid >= n_parts:
                continue
            index = i * max_acc + j
            for s in order_by_part[pid]:
                cnt = len(order)
                order.append(s)
                table[si[s], index] = cnt
                table[sj[s], index] = cnt + ns
        part_end.append(len(order))
    remap_tab = np.full((n_particles, arr), -1, np.int64)
    ncopies = np.zeros(n_particles, np.int64)
    for p in range(n_particles):
        occupied = np.zeros(arr, bool)
        for j in range(PARTITIONS_FINAL):
            start_ind = j * max_acc
            next_start = (j + 1) * max_acc
            for k in range(max_acc):
                index = start_ind + k
                if table[p, index] != -1:
                    found = False
                    for h in range(next_start, arr):
                        if table[p, h] != -1 and not occupied[h]:
                            remap_tab[p, index] = table[p, h]
                            occupied[h] = True
                            found = True
                            next_start += 1
                            break
                    if not found:
                        ncopies[p] += 1
    copy_end = np.cumsum(ncopies)
    remap_out = np.full(2 * ns, -1, np.int64)
    for p in range(n_particles):
        acc = 0
        for j in range(arr):
            slot = table[p, j]
            if slot != -1:
                r = remap_tab[p, j]
                if r == -1:
                    r = 2 * ns + (copy_end[p - 1] if p > 0 else 0) + acc
                    acc += 1
                remap_out[slot] = r
    blend = 1.0 / (int(ncopies.max()) + 1) if ns else 1.0
    return (np.asarray(order, np.int32), np.asarray(part_end, np.int32), remap_out.astype(np.int32),
            copy_end.astype(np.int32), float(blend))


# ----------------------------------------------------------------------------------------------------------
# Obstacles
# ----------------------------------------------------------------------------------------------------------

@dataclass
class Shape:
    kind: int
    pos: tuple = (0.0, 0.0, 0.0)
    rot: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float32))
    half: tuple = (0.0, 0.0, 0.0)
    contact_offset: float = 0.02      # PxShape default
    rest_offset: float = 0.0


def plane(z: float = 0.0, **kw) -> Shape:
    """Plane z = const, normal +z (PhysX's plane normal is local +x; MuJoCo's and this one's is +z)."""
    return Shape(SHAPE_PLANE, (0.0, 0.0, z), **kw)


def box(pos, half, rot=None, **kw) -> Shape:
    return Shape(SHAPE_BOX, tuple(pos), np.eye(3, dtype=np.float32) if rot is None else np.asarray(rot, np.float32), tuple(half), **kw)


# ----------------------------------------------------------------------------------------------------------
# Kernels
# ----------------------------------------------------------------------------------------------------------

@wp.func
def _basis(n: wp.vec3):
    """PxComputeBasisVectors."""
    if wp.abs(n[1]) <= 0.9999:
        r = wp.normalize(wp.vec3(n[2], 0.0, -n[0]))
        u = wp.vec3(n[1] * r[2], n[2] * r[0] - n[0] * r[2], -n[1] * r[0])
        return r, u
    r = wp.vec3(1.0, 0.0, 0.0)
    u = wp.normalize(wp.vec3(0.0, n[2], -n[1]))
    return r, u


@wp.kernel
def _pre_integrate(x: wp.array2d(dtype=wp.vec3), v: wp.array2d(dtype=wp.vec3), x0: wp.array2d(dtype=wp.vec3),
                   inv_mass: wp.array(dtype=float), gravity: wp.vec3, dt: float, damping: float):
    w, i = wp.tid()
    x0[w, i] = x[w, i]
    if inv_mass[i] == 0.0:
        v[w, i] = wp.vec3(0.0)
        return
    v[w, i] = (v[w, i] + gravity * dt) * (1.0 - wp.min(1.0, damping * dt))


@wp.kernel
def _contacts(x0: wp.array2d(dtype=wp.vec3), shape_kind: wp.array(dtype=int), shape_pos: wp.array2d(dtype=wp.vec3),
              shape_rot: wp.array2d(dtype=wp.mat33), shape_half: wp.array(dtype=wp.vec3), shape_cdist: wp.array(dtype=float),
              shape_rdist: wp.array(dtype=float), c_normal: wp.array3d(dtype=wp.vec3), c_err: wp.array3d(dtype=float),
              c_active: wp.array3d(dtype=int), c_force: wp.array3d(dtype=wp.vec2)):
    w, i, k = wp.tid()
    p = x0[w, i]
    kind = shape_kind[k]
    c = shape_pos[w % shape_pos.shape[0], k]
    R = shape_rot[w % shape_rot.shape[0], k]
    n = wp.vec3(0.0, 0.0, 1.0)
    dist = float(0.0)
    if kind == 0:
        n = R * wp.vec3(0.0, 0.0, 1.0)
        dist = wp.dot(p - c, n)
    else:
        he = shape_half[k]
        q = wp.transpose(R) * (p - c)
        cp = wp.vec3(wp.clamp(q[0], -he[0], he[0]), wp.clamp(q[1], -he[1], he[1]), wp.clamp(q[2], -he[2], he[2]))
        inside = he[0] >= wp.abs(q[0]) and he[1] >= wp.abs(q[1]) and he[2] >= wp.abs(q[2])
        if inside:
            ds = he - wp.vec3(wp.abs(cp[0]), wp.abs(cp[1]), wp.abs(cp[2]))
            nl = wp.vec3(0.0, 0.0, 1.0)
            if cp[2] < 0.0:
                nl = wp.vec3(0.0, 0.0, -1.0)
            dist = -ds[2]
            if ds[0] <= ds[1] and ds[0] <= ds[2]:
                nl = wp.vec3(1.0, 0.0, 0.0)
                if cp[0] < 0.0:
                    nl = wp.vec3(-1.0, 0.0, 0.0)
                dist = -ds[0]
            elif ds[1] <= ds[2]:
                nl = wp.vec3(0.0, 1.0, 0.0)
                if cp[1] < 0.0:
                    nl = wp.vec3(0.0, -1.0, 0.0)
                dist = -ds[1]
            n = R * nl
        else:
            d = q - cp
            dist = wp.length(d)
            n = R * (d / wp.max(dist, 1.0e-20))
    c_force[w, i, k] = wp.vec2(0.0, 0.0)
    if dist <= shape_cdist[k]:
        c_active[w, i, k] = 1
        c_normal[w, i, k] = n
        c_err[w, i, k] = dist - shape_rdist[k]
    else:
        c_active[w, i, k] = 0


@wp.kernel
def _step_particles(x: wp.array2d(dtype=wp.vec3), x0: wp.array2d(dtype=wp.vec3), v: wp.array2d(dtype=wp.vec3),
                    dp: wp.array2d(dtype=wp.vec3), inv_mass: wp.array(dtype=float), h: float, first: int):
    w, i = wp.tid()
    if inv_mass[i] == 0.0:
        return
    src = x[w, i]
    d0 = dp[w, i]
    if first != 0:
        src = x0[w, i]
        d0 = wp.vec3(0.0)
    step = v[w, i] * h
    x[w, i] = src + step
    dp[w, i] = d0 + step


@wp.kernel
def _gather(x: wp.array2d(dtype=wp.vec3), v: wp.array2d(dtype=wp.vec3), si: wp.array(dtype=int), sj: wp.array(dtype=int),
            cp: wp.array2d(dtype=wp.vec3), cv: wp.array2d(dtype=wp.vec3)):
    w, s = wp.tid()
    ns = si.shape[0]
    k = s % ns
    idx = si[k]
    if s >= ns:
        idx = sj[k]
    cp[w, s] = x[w, idx]
    cv[w, s] = v[w, idx]


@wp.kernel
def _springs(cp: wp.array2d(dtype=wp.vec3), cv: wp.array2d(dtype=wp.vec3), si: wp.array(dtype=int), sj: wp.array(dtype=int),
             rest: wp.array(dtype=float), stiff: wp.array(dtype=float), damp: wp.array(dtype=float),
             remap: wp.array(dtype=int), inv_mass: wp.array(dtype=float), start: int, h: float):
    w, k0 = wp.tid()
    k = start + k0
    ns = si.shape[0]
    xi = cp[w, k]
    xj = cp[w, k + ns]
    vi = cv[w, k]
    vj = cv[w, k + ns]
    wi = inv_mass[si[k]]
    wj = inv_mass[sj[k]]
    wsum = wi + wj
    if wsum > 0.0:
        xij = xi - xj
        dsq = wp.dot(xij, xij)
        if dsq > 1.0e-8:
            l = wp.sqrt(dsq)
            e = rest[k] - l
            kk = stiff[k]
            if kk < 0.0:                   # tether: allows compression
                e = wp.min(e, 0.0)
                kk = -kk
            g = xij / l
            dcdt = wp.dot(g, vi - vj)
            b = h * kk
            d = h * damp[k]
            a = h * b + d
            xx = 1.0 / (1.0 + a * wsum)
            dl = (xx * b * e - xx * a * dcdt) * h           # TGS: no lambda accumulation
            eb = e / wsum
            if wp.abs(dl) > wp.abs(eb):
                dl = eb
            dv = g * dl
            xi = xi + dv * wi
            xj = xj - dv * wj
            vi = vi + dv * (wi / h)
            vj = vj - dv * (wj / h)
    cp[w, remap[k]] = xi
    cp[w, remap[k + ns]] = xj
    cv[w, remap[k]] = vi
    cv[w, remap[k + ns]] = vj


@wp.kernel
def _average(x: wp.array2d(dtype=wp.vec3), cp: wp.array2d(dtype=wp.vec3), acc: wp.array2d(dtype=wp.vec3),
             copy_end: wp.array(dtype=int), inv_mass: wp.array(dtype=float), n_slots: int, blend: float):
    w, i = wp.tid()
    if inv_mass[i] == 0.0:
        return
    a = 0
    if i > 0:
        a = copy_end[i - 1]
    b = copy_end[i]
    if b == a:
        return
    xi = x[w, i]
    diff = wp.vec3(0.0)
    for c in range(a, b):
        diff = diff + (cp[w, n_slots + c] - xi)
    acc[w, i] = acc[w, i] + diff * blend


@wp.kernel
def _update(x: wp.array2d(dtype=wp.vec3), v: wp.array2d(dtype=wp.vec3), dp: wp.array2d(dtype=wp.vec3),
            acc: wp.array2d(dtype=wp.vec3), inv_h: float):
    w, i = wp.tid()
    a = acc[w, i]
    v[w, i] = v[w, i] + a * inv_h
    x[w, i] = x[w, i] + a
    dp[w, i] = dp[w, i] + a
    acc[w, i] = wp.vec3(0.0)


@wp.kernel
def _solve_contacts(dp: wp.array2d(dtype=wp.vec3), acc: wp.array2d(dtype=wp.vec3), inv_mass: wp.array(dtype=float),
                    c_normal: wp.array3d(dtype=wp.vec3), c_err: wp.array3d(dtype=float), c_active: wp.array3d(dtype=int),
                    c_force: wp.array3d(dtype=wp.vec2), friction: float, relaxation: float):
    w, i = wp.tid()
    im = inv_mass[i]
    if im == 0.0:
        return
    delta = dp[w, i]
    vm = 1.0 / im                      # static shape: unit response = particle inverse mass only
    total = wp.vec3(0.0)
    cnt = float(0.0)
    for k in range(c_active.shape[2]):
        if c_active[w, i, k] == 0:
            continue
        nrm = -c_normal[w, i, k]        # PhysX flips the normal to point into the shape
        t0, t1 = _basis(nrm)
        f = c_force[w, i, k]
        sep = c_err[w, i, k] - wp.dot(nrm, delta)
        df = wp.max(-f[0], -sep * vm)
        fx = wp.max(0.0, f[0] + df)
        maxf = wp.abs(fx) * friction
        r0 = -wp.dot(t0, delta) * vm
        r1 = -wp.dot(t1, delta) * vm
        req = wp.sqrt(r0 * r0 + r1 * r1)
        rr = float(0.0)
        if req >= 1.0e-16:
            rr = 1.0 / req
        dfr = wp.min(req + f[1], maxf) - f[1]
        c_force[w, i, k] = wp.vec2(fx, f[1] + dfr)
        dl = (-(nrm * df) + t0 * (dfr * r0 * rr) + t1 * (dfr * r1 * rr)) * im * relaxation
        total = total + dl
        if df != 0.0:
            cnt = cnt + 1.0
    if cnt != 0.0 or wp.length_sq(total) > 0.0:
        acc[w, i] = acc[w, i] + total * (1.0 / wp.max(1.0, cnt))


@wp.kernel
def _finalize(v: wp.array2d(dtype=wp.vec3), dp: wp.array2d(dtype=wp.vec3), inv_mass: wp.array(dtype=float),
              inv_dt: float, scale: float):
    w, i = wp.tid()
    if inv_mass[i] == 0.0:
        return
    dv = dp[w, i] * inv_dt
    vi = v[w, i]
    if wp.dot(vi, vi) > wp.dot(dv, dv):
        v[w, i] = vi * scale + dv * (1.0 - scale)


@wp.kernel
def _energy(x: wp.array2d(dtype=wp.vec3), v: wp.array2d(dtype=wp.vec3), inv_mass: wp.array(dtype=float), gravity: wp.vec3,
            out: wp.array2d(dtype=float)):
    w, i = wp.tid()
    im = inv_mass[i]
    if im == 0.0:
        return
    m = 1.0 / im
    wp.atomic_add(out, w, 0, -m * wp.dot(gravity, x[w, i]))
    wp.atomic_add(out, w, 1, 0.5 * m * wp.dot(v[w, i], v[w, i]))


# ----------------------------------------------------------------------------------------------------------
# Simulation
# ----------------------------------------------------------------------------------------------------------

@dataclass
class PhysXClothCfg:
    """Particle-system, material and spring parameters (names as in PhysX / omni.physx; defaults = the values of
    the Isaac recorder's cloth scene, metalsim/parity/isaac_side/record_deformables.py)."""
    dt: float = 1.0 / 200.0
    iterations: int = 16                  # solverPositionIterationCount (= TGS substeps of the whole island)
    velocity_iterations: int = 1          # PhysX default minVelocityIters
    rest_offset: float = 0.025            # particle-system rest offset (added to the shape's rest offset)
    contact_offset: float = 0.0375        # particle-system contact offset (added to the shape's contact offset)
    friction: float = 0.6                 # PBD material friction
    damping: float = 0.0                  # PBD material (velocity) damping
    stretch_stiffness: float = 1.0e4      # springStretchStiffness
    shear_stiffness: float = 100.0        # springShearStiffness
    bend_stiffness: float = 200.0         # springBendStiffness
    spring_damping: float = 0.2           # springDamping
    gravity: tuple = (0.0, 0.0, -9.81)
    self_collision: bool = False          # not ported yet


class PhysXClothSim:
    """N worlds of one particle cloth stepped with PhysX's PBD particle-cloth algorithm."""

    def __init__(self, mesh: ClothMesh, num_envs: int, obstacles: list[Shape] | None = None, cfg: PhysXClothCfg | None = None,
                 springs: tuple | None = None, device: str = "metal:0", capture: bool = True):
        self.cfg = cfg = cfg or PhysXClothCfg()
        if cfg.self_collision:
            raise NotImplementedError("particle self-collision is not ported yet")
        self.n = num_envs
        self.mesh = mesh
        self.device = wp.get_device(device)
        npart = len(mesh.x)
        self.npart = npart
        if springs is None:
            si, sj, rest, typ = cook_springs(mesh)
            k = np.where(typ == DIAGONAL, cfg.shear_stiffness, np.where(typ == BENDING, cfg.bend_stiffness, cfg.stretch_stiffness))
            d = np.full(len(si), cfg.spring_damping, np.float32)
        else:
            si, sj, rest, k, d = springs
            typ = np.zeros(len(si), np.int32)
        self.spring_types = typ
        order, part_end, remap, copy_end, blend = partition_springs(si, sj, npart)
        self.part_end = [0] + [int(e) for e in part_end]
        self.blend = blend
        ns = len(si)
        self.ns = ns
        self.n_slots = 2 * ns
        n_copy = int(copy_end[-1]) if npart else 0
        obstacles = obstacles if obstacles is not None else [plane()]
        self.shapes = obstacles
        nsh = len(obstacles)
        with wp.ScopedDevice(self.device):
            self.inv_mass = wp.array(mesh.inv_mass, dtype=float)
            self.si = wp.array(si[order], dtype=int)
            self.sj = wp.array(sj[order], dtype=int)
            self.rest = wp.array(np.asarray(rest, np.float32)[order], dtype=float)
            self.stiff = wp.array(np.asarray(k, np.float32)[order], dtype=float)
            self.sdamp = wp.array(np.asarray(d, np.float32)[order], dtype=float)
            self.remap = wp.array(remap, dtype=int)
            self.copy_end = wp.array(copy_end, dtype=int)
            self.x = wp.array(np.tile(mesh.x, (num_envs, 1, 1)), dtype=wp.vec3)
            self.v = wp.zeros_like(self.x)
            self.x0 = wp.zeros_like(self.x)
            self.dp = wp.zeros_like(self.x)
            self.acc = wp.zeros_like(self.x)
            self.cp = wp.zeros((num_envs, 2 * ns + n_copy), dtype=wp.vec3)
            self.cv = wp.zeros((num_envs, 2 * ns + n_copy), dtype=wp.vec3)
            self.shape_kind = wp.array(np.array([s.kind for s in obstacles], np.int32), dtype=int)
            self.shape_pos = wp.array(np.array([s.pos for s in obstacles], np.float32)[None], dtype=wp.vec3)
            self.shape_rot = wp.array(np.array([s.rot for s in obstacles], np.float32)[None], dtype=wp.mat33)
            self.shape_half = wp.array(np.array([s.half for s in obstacles], np.float32), dtype=wp.vec3)
            self.shape_cdist = wp.array(np.array([cfg.contact_offset + s.contact_offset for s in obstacles], np.float32), dtype=float)
            self.shape_rdist = wp.array(np.array([cfg.rest_offset + s.rest_offset for s in obstacles], np.float32), dtype=float)
            self.c_normal = wp.zeros((num_envs, npart, nsh), dtype=wp.vec3)
            self.c_err = wp.zeros((num_envs, npart, nsh), dtype=float)
            self.c_active = wp.zeros((num_envs, npart, nsh), dtype=int)
            self.c_force = wp.zeros((num_envs, npart, nsh), dtype=wp.vec2)
            self.energy_arr = wp.zeros((num_envs, 2), dtype=float)
            self.graph = None
            if capture and (getattr(self.device, "is_metal", False) or self.device.is_cuda):
                with wp.ScopedCapture(device=self.device) as cap:
                    self.launch()
                self.graph = cap.graph
        self._t = {}

    def bind_obstacle_poses(self, geom_xpos: wp.array, geom_xmat: wp.array) -> None:
        """Read the obstacles' poses from per-world arrays (e.g. MuJoCo Warp ``Data.geom_xpos/geom_xmat`` sliced to the
        obstacle geoms, shape (N, nshape)); one-way coupling. Re-captures the graph."""
        self.shape_pos, self.shape_rot = geom_xpos, geom_xmat
        if self.graph is not None:
            with wp.ScopedDevice(self.device), wp.ScopedCapture(device=self.device) as cap:
                self.launch()
            self.graph = cap.graph

    def launch(self) -> None:
        c = self.cfg
        n, npart, dt = self.n, self.npart, c.dt
        h = dt / c.iterations
        bias = min(0.9, 2.0 * math.sqrt(1.0 / c.iterations))        # PxgContext.cpp:2004
        relax = min(0.7, bias)
        g = wp.vec3(*c.gravity)
        wp.launch(_pre_integrate, dim=(n, npart), inputs=[self.x, self.v, self.x0, self.inv_mass, g, dt, c.damping])
        wp.launch(_contacts, dim=(n, npart, len(self.shapes)),
                  inputs=[self.x0, self.shape_kind, self.shape_pos, self.shape_rot, self.shape_half, self.shape_cdist, self.shape_rdist,
                          self.c_normal, self.c_err, self.c_active, self.c_force])
        for it in range(c.iterations + c.velocity_iterations):
            if it < c.iterations:
                wp.launch(_step_particles, dim=(n, npart), inputs=[self.x, self.x0, self.v, self.dp, self.inv_mass, h, int(it == 0)])
            if self.ns:
                wp.launch(_gather, dim=(n, 2 * self.ns), inputs=[self.x, self.v, self.si, self.sj, self.cp, self.cv])
                for p in range(PARTITIONS_FINAL):
                    a, b = self.part_end[p], self.part_end[p + 1]
                    if b > a:
                        wp.launch(_springs, dim=(n, b - a), inputs=[self.cp, self.cv, self.si, self.sj, self.rest, self.stiff, self.sdamp,
                                                                     self.remap, self.inv_mass, a, h])
                wp.launch(_average, dim=(n, npart), inputs=[self.x, self.cp, self.acc, self.copy_end, self.inv_mass, self.n_slots, self.blend])
                wp.launch(_update, dim=(n, npart), inputs=[self.x, self.v, self.dp, self.acc, 1.0 / h])
            wp.launch(_solve_contacts, dim=(n, npart), inputs=[self.dp, self.acc, self.inv_mass, self.c_normal, self.c_err, self.c_active,
                                                                self.c_force, c.friction, relax])
            wp.launch(_update, dim=(n, npart), inputs=[self.x, self.v, self.dp, self.acc, 1.0 / h])
        wp.launch(_finalize, dim=(n, npart), inputs=[self.v, self.dp, self.inv_mass, 1.0 / dt, bias])
        self.energy_arr.zero_()
        wp.launch(_energy, dim=(n, npart), inputs=[self.x, self.v, self.inv_mass, g, self.energy_arr])

    def step(self, eager: bool = False) -> None:
        with wp.ScopedDevice(self.device):
            if self.graph is not None and not eager:
                wp.capture_launch(self.graph)
            else:
                self.launch()

    def synchronize(self) -> None:
        wp.synchronize_device(self.device)

    def set_state(self, x: np.ndarray, v: np.ndarray | None = None) -> None:
        self.synchronize()
        self.x.assign(np.asarray(x, np.float32).reshape(self.n, self.npart, 3))
        self.v.assign(np.zeros((self.n, self.npart, 3), np.float32) if v is None else np.asarray(v, np.float32).reshape(self.n, self.npart, 3))

    def randomize(self, seed: int = 0, xy: float = 0.05, z: float = 0.02, worlds_equal_first: bool = True) -> np.ndarray:
        rng = np.random.default_rng(seed)
        off = np.stack([rng.uniform(-xy, xy, self.n), rng.uniform(-xy, xy, self.n), rng.uniform(0, z, self.n)], 1).astype(np.float32)
        if worlds_equal_first:
            off[0] = 0
        x = self.mesh.x[None] + off[:, None, :]
        self.set_state(x)
        return x

    def tensor(self, name: str):
        t = self._t.get(name)
        if t is None:
            from metalsim.interop import torch_bridge as tb
            t = self._t[name] = tb.mps_tensor(getattr(self, name))
        return t

    def vertices(self):
        return self.tensor("x")

    def energy(self):
        return self.tensor("energy_arr")

    def spring_strain(self, world: int = 0, kind: int | None = HORIZONTAL | VERTICAL) -> np.ndarray:
        """Relative length error of the springs of the given classes in one world."""
        self.synchronize()
        x = self.x.numpy()[world]
        i, j, r = self.si.numpy(), self.sj.numpy(), self.rest.numpy()
        l = np.linalg.norm(x[i] - x[j], axis=1)
        s = (l - r) / r
        if kind is None:
            return s
        typ = self._ordered_types()
        return s[(typ & kind) != 0]

    def _ordered_types(self) -> np.ndarray:
        if not hasattr(self, "_otypes"):
            si, sj = self.si.numpy(), self.sj.numpy()
            ci, cj, _, typ = cook_springs(self.mesh)
            lut = {(int(a), int(b)): int(t) for a, b, t in zip(ci, cj, typ)}
            self._otypes = np.array([lut.get((int(a), int(b)), 0) for a, b in zip(si, sj)], np.int32)
        return self._otypes


def benchmark(num_envs: int, seconds: float = 2.0, n: int = 21, cfg: PhysXClothCfg | None = None, warmup: int = 10,
              device: str = "metal:0") -> dict:
    """The recorder's cloth scene (21x21, 1 m, 0.4 m static box on a plane) at `num_envs` worlds, graph replay."""
    mesh = grid_cloth(n, 1.0, z0=0.5)
    sim = PhysXClothSim(mesh, num_envs, obstacles=[plane(), box((0, 0, 0.2), (0.2, 0.2, 0.2))], cfg=cfg, device=device)
    sim.randomize(seed=1)
    for _ in range(warmup):
        sim.step()
    sim.synchronize()
    steps = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        for _ in range(10):
            sim.step()
        sim.synchronize()
        steps += 10
    wall = time.perf_counter() - t0
    x = sim.x.numpy()
    return {"num_envs": num_envs, "steps": steps, "wall_s": wall, "env_steps_per_s": num_envs * steps / wall,
            "sim_s_per_wall_s": steps * sim.cfg.dt / wall, "finite": bool(np.isfinite(x).all()), "particles": sim.npart,
            "springs": sim.ns, "iterations": sim.cfg.iterations, "z_mean_world0": float(x[0, :, 2].mean())}


def _main():
    import argparse, json
    ap = argparse.ArgumentParser(description="PhysX-style PBD cloth throughput on Metal (run through scripts/gpu_run.sh)")
    ap.add_argument("--envs", type=int, nargs="+", default=[1024, 4096])
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--device", default="metal:0")
    a = ap.parse_args()
    wp.config.quiet = True
    for n in a.envs:
        print(json.dumps(benchmark(n, a.seconds, device=a.device)), flush=True)


if __name__ == "__main__":
    _main()


def mesh_from_flex(model, flex: int = 0) -> ClothMesh:
    """A MuJoCo 2D flex (e.g. a ``flexcomp`` grid, with ``<pin>`` vertices fixed) as a ClothMesh: same vertices,
    triangles and vertex masses as MuJoCo Warp flex and MetalSim ``XPBDSim`` see."""
    import mujoco
    d = mujoco.MjData(model)
    mujoco.mj_forward(model, d)
    va, vn = model.flex_vertadr[flex], model.flex_vertnum[flex]
    ea, en = model.flex_elemdataadr[flex], model.flex_elemnum[flex]
    assert model.flex_dim[flex] == 2, "cloth = 2D flex"
    tris = model.flex_elem[ea:ea + 3 * en].reshape(-1, 3).astype(np.int32)
    body = model.flex_vertbodyid[va:va + vn]
    mass = model.body_mass[body]
    inv = np.where([model.body_dofnum[b] > 0 for b in body], 1.0 / np.maximum(mass, 1e-12), 0.0).astype(np.float32)
    return ClothMesh(d.flexvert_xpos[va:va + vn].astype(np.float32).copy(), tris, inv)

"""Sphere-on-heightfield contacts on the g1_bumps tile: MuJoCo C, MuJoCo Warp (MetalSim fork's per-triangle plane
contact, and upstream's GJK/EPA against each prism) and the exact answer, for Unitree's 5 mm sole spheres.

Why: on this tile Unitree's policy falls far more often in CPU MuJoCo than in MetalSim's MuJoCo Warp (measure.py). The
fork's heightfield path (``collision_convex.HFIELD_PLANE_CONTACTS``, docs/PARITY.md section 1.6) treats a primitive
geom by its single support point along each triangle's normal (docs/research/elliptic_cones_2026-09-25.md section 7.4).
This checks each engine's contacts against the geometry itself.

Placements: N spheres (radius 5 mm, Unitree's) over the tile's rough area, each pressed ``pen`` = 0.5 ... 3 mm into the
triangulated surface (MuJoCo's triangulation of the heightfield): a random surface point p, its triangle normal n,
centre c = p + (r - pen) n. Exact: the closest point q on the triangulated surface (all triangles of the 4 x 4 cells
around c), depth r - |c - q|, normal (c - q) / |c - q|. Engines: their contacts between the sphere and the
heightfield at that pose, one world per placement; compared: the deepest contact's depth and normal, and the sum of
all contact normals weighted by depth (the direction the contact set pushes).

Usage: .venv/bin/python runs/film/g1_bumps/measure/contact_check.py c|plane|prism <out_dir>  (CPU only: MuJoCo Warp on
Warp's CPU device), then ``... compare <out_dir>``. The MuJoCo Warp copy is whichever imports (``PYTHONPATH``); the
plane / prism runs record it in ``contacts_<mode>.json``. "prism" is upstream's GJK/EPA per prism, which the MetalSim
fork has used for primitives since ``b630530`` and, with its heightfield fixes, for meshes since ``93e5ce1``
(docs/research/hfield_contacts_2026-10-31.md); "plane" is the fork's per-triangle plane path.
First take (2026-10-31 afternoon): out_dir runs/readme_media/scratch/g1-bumps/measure_npz/contact_check, copied to
runs/film/g1_bumps/measure/contact_check.json (its "plane" column measured a stale checkout, ``07a51a6``; its "prism"
column is the fork before the heightfield fixes). MetalSim-physics take: out_dir
runs/readme_media/scratch/g1-bumps-metalsim/cc (fork ``787a5b3``), copied to
runs/film/g1_bumps/measure/mjw_787a5b3/contact_check.json.
"""
import json
import os
import sys

import numpy as np

N, R, SEED = 3000, 0.005, 7


def build_model(hf):
    """The tile as the film builds it (an MjSpec heightfield from ``userdata``, geom at z = zmin) and one free 5 mm
    sphere. (MJCF's ``elevation`` attribute is laid out differently from ``userdata``: not used.)"""
    import mujoco
    spec = mujoco.MjSpec()
    t = spec.add_hfield(); t.name = "t"; t.nrow, t.ncol = int(hf["nrow"]), int(hf["ncol"])
    t.size = [float(v) for v in hf["size"]]; t.userdata = np.asarray(hf["data"], np.float32).reshape(-1).tolist()
    g = spec.worldbody.add_geom(); g.name = "floor"; g.type = mujoco.mjtGeom.mjGEOM_HFIELD; g.hfieldname = "t"
    g.pos = [0, 0, float(hf["zmin"])]
    b = spec.worldbody.add_body(); b.name = "ball"; b.pos = [0, 0, 1]; b.add_freejoint()
    s_ = b.add_geom(); s_.name = "ball"; s_.type = mujoco.mjtGeom.mjGEOM_SPHERE; s_.size = [R, 0, 0]; s_.mass = 0.1
    return spec.compile()


def surface(hf):
    """MuJoCo's triangulated heightfield as world arrays: Z [row (y), col (x)], spacing, origin."""
    D = np.asarray(hf["data"], np.float64).reshape(hf["nrow"], hf["ncol"]) * hf["size"][2] + hf["zmin"]
    return D, hf["res"], -hf["size"][0], -hf["size"][1]


def tri_of(D, res, x0, y0, x, y):
    """Vertices (3, 3) of the triangle under (x, y), MuJoCo's split along (c, r)-(c+1, r+1)."""
    u, v = (x - x0) / res, (y - y0) / res
    c, r = int(np.floor(u)), int(np.floor(v))
    fx, fy = u - c, v - r
    P = lambda rr, cc: np.array([x0 + cc * res, y0 + rr * res, D[rr, cc]])
    if fx >= fy:
        return np.stack([P(r, c), P(r, c + 1), P(r + 1, c + 1)])
    return np.stack([P(r, c), P(r + 1, c + 1), P(r + 1, c)])


def closest_on_triangle(p, a, b, c):
    """Closest point to p on triangle abc (Ericson, Real-Time Collision Detection 5.1.5)."""
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = ab @ ap, ac @ ap
    if d1 <= 0 and d2 <= 0:
        return a
    bp = p - b; d3, d4 = ab @ bp, ac @ bp
    if d3 >= 0 and d4 <= d3:
        return b
    vc = d1 * d4 - d3 * d2
    if vc <= 0 and d1 >= 0 and d3 <= 0:
        return a + d1 / (d1 - d3) * ab
    cp = p - c; d5, d6 = ab @ cp, ac @ cp
    if d6 >= 0 and d5 <= d6:
        return c
    vb = d5 * d2 - d1 * d6
    if vb <= 0 and d2 >= 0 and d6 <= 0:
        return a + d2 / (d2 - d6) * ac
    va = d3 * d6 - d5 * d4
    if va <= 0 and (d4 - d3) >= 0 and (d5 - d6) >= 0:
        return b + (d4 - d3) / ((d4 - d3) + (d5 - d6)) * (c - b)
    den = 1.0 / (va + vb + vc)
    return a + ab * vb * den + ac * vc * den


def exact(D, res, x0, y0, c):
    """Closest point on the triangulated surface around c (4 x 4 cells), its depth and normal."""
    best = (np.inf, None)
    ci, ri = int(np.floor((c[0] - x0) / res)), int(np.floor((c[1] - y0) / res))
    for r in range(ri - 1, ri + 3):
        for cc in range(ci - 1, ci + 3):
            P = lambda rr, k: np.array([x0 + k * res, y0 + rr * res, D[rr, k]])
            for t in ((P(r, cc), P(r, cc + 1), P(r + 1, cc + 1)), (P(r, cc), P(r + 1, cc + 1), P(r + 1, cc))):
                q = closest_on_triangle(c, *t)
                d = np.linalg.norm(c - q)
                if d < best[0]:
                    best = (d, q)
    d, q = best
    return R - d, (c - q) / d


def exact_set(D, res, x0, y0, c, tol=1e-6):
    """Every distinct closest feature of the triangulated surface within the sphere (4 x 4 cells around c): the true
    contact set (a sphere in a concave crease touches two faces). Returns [(depth, normal)], deepest first."""
    ci, ri = int(np.floor((c[0] - x0) / res)), int(np.floor((c[1] - y0) / res))
    pts = []
    for r in range(ri - 1, ri + 3):
        for cc in range(ci - 1, ci + 3):
            P = lambda rr, k: np.array([x0 + k * res, y0 + rr * res, D[rr, k]])
            for t in ((P(r, cc), P(r, cc + 1), P(r + 1, cc + 1)), (P(r, cc), P(r + 1, cc + 1), P(r + 1, cc))):
                q = closest_on_triangle(c, *t)
                d = np.linalg.norm(c - q)
                if d < R and not any(np.linalg.norm(q - q2) < tol for q2, _ in pts):
                    pts.append((q, d))
    return sorted(((R - d, (c - q) / d) for q, d in pts), key=lambda x: -x[0])


def placements(hf):
    D, res, x0, y0 = surface(hf)
    rng = np.random.default_rng(SEED)
    out = []
    while len(out) < N:
        x, y = rng.uniform(-3.6, 3.6, 2)
        tri = tri_of(D, res, x0, y0, x, y)
        n = np.cross(tri[1] - tri[0], tri[2] - tri[0]); n /= np.linalg.norm(n); n *= np.sign(n[2])
        z = tri[0, 2] + (-(n[0] * (x - tri[0, 0]) + n[1] * (y - tri[0, 1])) / n[2])
        pen = rng.uniform(0.0005, 0.003)
        c = np.array([x, y, z]) + (R - pen) * n
        dep, nrm = exact(D, res, x0, y0, c)
        out.append((c, dep, nrm))
    C = np.array([o[0] for o in out]); dep = np.array([o[1] for o in out]); nrm = np.array([o[2] for o in out])
    return C, dep, nrm


def summarise_contacts(nw, world, dist, normal):
    """Per world: deepest depth, its normal, the depth-weighted normal sum (normalised), contact count."""
    deep = np.full(nw, np.nan); nd = np.full((nw, 3), np.nan); ns = np.zeros((nw, 3)); cnt = np.zeros(nw, int)
    for w, d, n in zip(world, dist, normal):
        n = n * np.sign(n[2]) if abs(n[2]) > 1e-12 else n
        cnt[w] += 1
        ns[w] += max(-d, 0.0) * n
        if np.isnan(deep[w]) or -d > deep[w]:
            deep[w] = -d; nd[w] = n
    nn = np.linalg.norm(ns, axis=1, keepdims=True)
    return deep, nd, np.where(nn > 0, ns / np.where(nn > 0, nn, 1), np.nan), cnt


def run_c(hf, C):
    import mujoco
    m = build_model(hf)
    d = mujoco.MjData(m)
    W, Dd, Nn = [], [], []
    for i, c in enumerate(C):
        d.qpos[:3] = c; d.qpos[3:7] = (1, 0, 0, 0)
        mujoco.mj_forward(m, d)
        for k in range(d.ncon):
            W.append(i); Dd.append(d.contact.dist[k]); Nn.append(np.array(d.contact.frame[k, :3]))
    return np.array(W, int), np.array(Dd), np.array(Nn).reshape(-1, 3)


def run_mjw(hf, C, plane: bool):
    import mujoco
    import warp as wp
    wp.config.quiet = True
    from mujoco_warp._src import collision_convex
    collision_convex.HFIELD_PLANE_CONTACTS = plane
    import mujoco_warp as mjw
    m = build_model(hf)
    d0 = mujoco.MjData(m)
    with wp.ScopedDevice("cpu"):
        mm = mjw.put_model(m)
        dd = mjw.put_data(m, d0, nworld=len(C), nconmax=64, njmax=256)
        q = np.tile(np.array([0, 0, 1, 1, 0, 0, 0], np.float32), (len(C), 1)); q[:, :3] = C
        dd.qpos.assign(q)
        mjw.forward(mm, dd)
        n = int(dd.nacon.numpy()[0])
        W = dd.contact.worldid.numpy()[:n]; Dd = dd.contact.dist.numpy()[:n]
        Nn = dd.contact.frame.numpy()[:n, 0, :]
    return W.astype(int), Dd.astype(np.float64), Nn.astype(np.float64)


def compare(out):
    ref = np.load(os.path.join(out, "contacts_exact.npz"))
    dep, nrm = ref["depth"], ref["normal"]
    tilt = np.degrees(np.arccos(np.clip(nrm[:, 2], -1, 1)))
    res = dict(placements=int(len(dep)), sphere_radius=R, depth_mm=[0.5, 3.0], exact_tilt_deg_p50_p95_max=
               [round(float(np.percentile(tilt, q)), 1) for q in (50, 95, 100)], engines={})
    sets = dict(all=np.ones(len(dep), bool), steep=tilt > 20.0, flat=tilt <= 20.0)
    for eng in ("c", "plane", "prism"):
        p = os.path.join(out, f"contacts_{eng}.npz")
        if not os.path.exists(p):
            continue
        z = np.load(p)
        deep, nd, ns, cnt = summarise_contacts(len(dep), z["world"], z["dist"], z["normal"])
        r = {}
        for sname, sel in sets.items():
            miss = np.isnan(deep[sel])
            ang_d = np.degrees(np.arccos(np.clip((nd[sel] * nrm[sel]).sum(1), -1, 1)))
            ang_s = np.degrees(np.arccos(np.clip((ns[sel] * nrm[sel]).sum(1), -1, 1)))
            derr = (deep[sel] - dep[sel]) * 1e3
            f = lambda a: [round(float(np.nanpercentile(a, q)), 2) for q in (50, 95)] if np.isfinite(a).any() else None
            r[sname] = dict(n=int(sel.sum()), no_contact=int(miss.sum()), contacts_mean=round(float(cnt[sel].mean()), 2),
                            deepest_normal_err_deg_p50_p95=f(ang_d), weighted_normal_err_deg_p50_p95=f(ang_s),
                            depth_err_mm_p50_p95=f(np.abs(derr)),
                            horizontal_normal_share=round(float(np.nanmean(nd[sel][:, 2] < 0.5)), 3))
        res["engines"][{"c": "MuJoCo C 3.14 (mj_forward)", "plane": "MuJoCo Warp, per-triangle plane contacts (HFIELD_PLANE_CONTACTS True)",
                        "prism": "MuJoCo Warp, GJK/EPA per prism (HFIELD_PLANE_CONTACTS False)"}[eng]] = r
        src = os.path.join(out, f"contacts_{eng}.json")
        if os.path.exists(src):
            r["mujoco_warp"] = json.load(open(src))
    # every engine contact against the true contact set (exact_set): normal error to the nearest true contact, and
    # contact counts against the true count
    from metalsim.render.film.unitree import random_rough_tile
    D, res_, x0, y0 = surface(random_rough_tile(None, 0))
    C = ref["centre"]
    E = [exact_set(D, res_, x0, y0, c) for c in C]
    ne = np.array([len(e) for e in E])
    res["exact_contacts_per_sphere"] = {str(k): int((ne == k).sum()) for k in sorted(set(ne.tolist()))}
    for eng in ("c", "plane", "prism"):
        p = os.path.join(out, f"contacts_{eng}.npz")
        if not os.path.exists(p):
            continue
        z = np.load(p)
        err, cnt = [], np.zeros(len(C), int)
        for w, dd, n in zip(z["world"], z["dist"], z["normal"]):
            n = n * np.sign(n[2]) if abs(n[2]) > 1e-12 else n
            cnt[w] += 1
            err.append(min(np.degrees(np.arccos(np.clip(n @ en, -1, 1))) for _, en in E[w]))
        err = np.array(err)
        name = [k for k in res["engines"] if k.startswith({"c": "MuJoCo C", "plane": "MuJoCo Warp, per-triangle", "prism": "MuJoCo Warp, GJK/EPA"}[eng])][0]
        res["engines"][name]["all_contacts_vs_true_set"] = dict(
            contacts=int(len(err)), normal_err_deg_p50_p95_p99=[round(float(np.percentile(err, q)), 2) for q in (50, 95, 99)],
            share_over_10deg=round(float((err > 10).mean()), 4),
            count_equal_true=round(float((cnt == ne).mean()), 3), fewer_than_true=int((cnt < ne).sum()),
            more_than_true=int((cnt > ne).sum()))
    json.dump(res, open(os.path.join(out, "contact_check.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


def main():
    mode, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    if mode == "compare":
        return compare(out)
    from metalsim.render.film.unitree import random_rough_tile
    hf = random_rough_tile(None, 0)
    p = os.path.join(out, "contacts_exact.npz")
    if not os.path.exists(p):
        C, dep, nrm = placements(hf)
        np.savez(p, centre=C, depth=dep, normal=nrm)
    C = np.load(p)["centre"]
    W, Dd, Nn = run_c(hf, C) if mode == "c" else run_mjw(hf, C, plane=(mode == "plane"))
    np.savez(os.path.join(out, f"contacts_{mode}.npz"), world=W, dist=Dd, normal=Nn)
    if mode != "c":
        import mujoco_warp

        from metalsim.render.film.provenance import git_sha
        import subprocess
        f = mujoco_warp.__file__
        root = os.path.dirname(os.path.dirname(os.path.realpath(f)))      # the copy's top directory
        top = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=root, capture_output=True, text=True).stdout.strip()
        arch = os.path.join(root, "ARCHIVE_OF")                            # a `git archive` copy names its commit here
        json.dump(dict(file=os.path.relpath(f), git=(git_sha(root) or None) if top and os.path.realpath(top) == root else None,
                       archive_of=open(arch).read().strip() if os.path.exists(arch) else None),
                  open(os.path.join(out, f"contacts_{mode}.json"), "w"), indent=1)
    print(mode, "contacts", len(W), "worlds with contact", len(set(W.tolist())), "of", len(C))


if __name__ == "__main__":
    main()

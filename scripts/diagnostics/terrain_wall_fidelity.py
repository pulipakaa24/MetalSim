"""Off-grid collision-surface fidelity of the rough-terrain options against Isaac's own terrain mesh (CPU, no GPU).

Oracle: Isaac Lab v2.3.2's TerrainGenerator run here (tests/isaac_terrain_ref.py, CPU torch box heights) and Isaac's
RayCaster operation (wp.mesh_query_ray straight down on its float32 terrain mesh). Ours: the top of the MuJoCo collision
geometry each option builds (metalsim.learn.terrain.isaac_rough_terrain(collision=...)): the heightfield with MuJoCo's
own triangulation (the prism layout of mjc_ConvexHField / MuJoCo Warp's hfield kernel: diagonal (x0, y1)-(x1, y0)) and,
for the box options, the highest box top above the point. The analytic surface is checked against MuJoCo's mj_ray on
the compiled model at a subsample (metalsim.learn.terrain.collision_top). Points are uniform in each sub-terrain type (all 10 rows), off the grid.

    python scripts/diagnostics/terrain_wall_fidelity.py --points 40000 --out runs/terrain_walls/fidelity.json
"""
import argparse, json, os, sys, time
import numpy as np
import mujoco
import warp as wp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tests"))
from metalsim.learn.terrain import isaac_rough_terrain, add_terrain_geoms, collision_top


@wp.kernel
def _raycast_down(mesh: wp.uint64, xy: wp.array(dtype=wp.vec2), out: wp.array(dtype=float)):
    i = wp.tid()
    q = wp.mesh_query_ray(mesh, wp.vec3(xy[i][0], xy[i][1], 20.0), wp.vec3(0.0, 0.0, -1.0), 1.0e6)
    out[i] = 20.0 - q.t if q.result else -1.0e9


def isaac_raycast(mesh, xy):
    m = wp.Mesh(points=wp.array(np.asarray(mesh.vertices, np.float32), dtype=wp.vec3, device="cpu"),
                indices=wp.array(np.asarray(mesh.faces, np.int32).ravel(), dtype=wp.int32, device="cpu"))
    out = wp.zeros(len(xy), dtype=float, device="cpu")
    wp.launch(_raycast_down, dim=len(xy), inputs=[m.id, wp.array(xy.astype(np.float32), dtype=wp.vec2, device="cpu"), out], device="cpu")
    return out.numpy().astype(np.float64)


def terrain_only_model(hf):
    spec = mujoco.MjSpec()
    h = spec.add_hfield(); h.name = "terrain"; h.nrow, h.ncol = hf["nrow"], hf["ncol"]; h.size = hf["size"]
    h.userdata = hf["data"].reshape(-1)
    g = spec.worldbody.add_geom(); g.name = "ground"; g.type = mujoco.mjtGeom.mjGEOM_HFIELD; g.hfieldname = "terrain"
    g.pos = [0, 0, hf["zmin"]]; g.friction = [0.8, 0.005, 0.0001]
    add_terrain_geoms(spec, hf, g)
    return spec.compile()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", type=int, default=40000, help="per sub-terrain type")
    ap.add_argument("--mj_ray", type=int, default=1500, help="mj_ray validation points per mode")
    ap.add_argument("--modes", nargs="+", default=["hfield", "hfield_fine:0.05", "hfield_fine:0.025", "boxes", "meshes"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args(); wp.config.quiet = True
    import isaac_terrain_ref as R
    t0 = time.time(); isaac = R.isaac_rough_generator(seed=0); print(f"Isaac generator {time.time() - t0:.1f} s", flush=True)
    base = isaac_rough_terrain(seed=0, torch_device="cpu")
    gen = base["generator"]
    assert np.array_equal(gen.terrain_origins, isaac.terrain_origins)
    rng = np.random.default_rng(0)
    types = {}
    for s in gen.sub_terrains:
        types.setdefault(s.name, []).append((s.row, s.col))
    pts = {}
    for name, cells in types.items():
        k = rng.integers(0, len(cells), a.points); rc = np.array(cells)[k]
        lx, ly = rng.uniform(0, 8, a.points), rng.uniform(0, 8, a.points)
        pts[name] = np.stack([rc[:, 0] * 8.0 - 40.0 + lx, rc[:, 1] * 8.0 - 80.0 + ly], 1)
    ref = {n: isaac_raycast(isaac.terrain_mesh, p) for n, p in pts.items()}
    results = {"points_per_type": a.points, "oracle": "Isaac Lab v2.3.2 TerrainGenerator (CPU torch), wp.mesh_query_ray", "modes": {}}
    for spec_ in a.modes:
        mode, _, fr = spec_.partition(":")
        hf = isaac_rough_terrain(seed=0, torch_device="cpu", collision=mode, fine_res=float(fr or 0.025))
        row = {}
        allerr = []
        surf = lambda p: collision_top(hf, p[:, 0], p[:, 1])
        for n, p in pts.items():
            # a point within 0.1 mm of a wall takes the better side: Isaac's float32 vertices put walls up to ~1e-7 m
            # from where the float64 box geoms put them (one such point at 40 k samples reads 0.10 m off otherwise)
            e = np.min([np.abs(surf(p + np.array(o)) - ref[n]) for o in ((0, 0), (1e-4, 0), (-1e-4, 0), (0, 1e-4), (0, -1e-4))], 0)
            allerr.append(e)
            row[n] = {"mean": float(e.mean()), "p99": float(np.percentile(e, 99)), "max": float(e.max()), "frac_over_1cm": float((e > 0.01).mean())}
        e = np.concatenate(allerr)
        row["all"] = {"mean": float(e.mean()), "p99": float(np.percentile(e, 99)), "max": float(e.max()), "frac_over_1cm": float((e > 0.01).mean())}
        # validation: the analytic surface is the compiled MuJoCo model's (mj_ray, straight down)
        m = terrain_only_model(hf); d = mujoco.MjData(m); mujoco.mj_forward(m, d)
        vv = []
        gid = np.zeros(1, np.int32)
        for n, p in pts.items():
            for i in range(a.mj_ray // len(pts)):
                x, y = p[i]
                dist = mujoco.mj_ray(m, d, np.array([x, y, 20.0]), np.array([0, 0, -1.0]), None, 1, -1, gid)
                za = collision_top(hf, [x], [y])[0]
                vv.append(abs((20.0 - dist) - za))
        row["mj_ray_vs_analytic_max"] = float(np.max(vv))
        row["ngeom"] = int(m.ngeom); row["hfield_samples"] = int(m.nhfielddata)
        results["modes"][spec_] = row
        print(spec_, json.dumps({k: (v if not isinstance(v, dict) else {kk: round(vv_, 4) for kk, vv_ in v.items()}) for k, v in row.items()}), flush=True)
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True); json.dump(results, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()

"""PhysX deformable protocol on MetalSim: the three Isaac-side scenes (record_deformables*.py) rebuilt for both MetalSim
backends (MuJoCo Warp flex, MetalSim XPBD), bulk metrics, residual tables and a parameter fit.

A per-vertex comparison is meaningless across formulations (PhysX particle cloth / FEM hexahedral-tet meshes vs MuJoCo
flex grids vs XPBD chains have different nodes), so every metric is a bulk quantity computed the same way from each
side's node positions/velocities sampled at 200 Hz.

    python physx_protocol.py metrics  REC_DIR                       # PhysX metrics of a recording
    python physx_protocol.py run      REC_DIR --backend flex|xpbd [--device cpu|metal:0] [--params JSON]
    python physx_protocol.py fit      REC_DIR --backend flex|xpbd --object cloth|rope|cube [--device cpu]
"""
from __future__ import annotations

import argparse, itertools, json, os, sys, time
import numpy as np

HZ = 200.0
T_END = 5.0


# --------------------------------------------------------------------------------------------------
# Bulk metrics (identical code for every source)
# --------------------------------------------------------------------------------------------------

def _settle_time(ke, dt, frac=0.01):
    """First time after which kinetic energy stays below frac * its maximum."""
    thr = frac * ke.max()
    above = np.where(ke > thr)[0]
    return float((above[-1] + 1) * dt) if len(above) else 0.0


def cloth_metrics(pos, vel, mass_per_node, box=0.4, cell=0.1):
    dt = 1.0 / HZ
    ke = 0.5 * mass_per_node * (vel ** 2).sum((1, 2))
    x = pos[-1]
    on = (np.abs(x[:, 0]) < box / 2) & (np.abs(x[:, 1]) < box / 2)
    # height map over [-0.6, 0.6]^2 (max z per cell) and xy silhouette at t = 5 s
    edges = np.arange(-0.6, 0.6 + 1e-9, cell)
    ix = np.clip(np.digitize(x[:, 0], edges) - 1, 0, len(edges) - 2)
    iy = np.clip(np.digitize(x[:, 1], edges) - 1, 0, len(edges) - 2)
    hmap = np.full((len(edges) - 1, len(edges) - 1), np.nan)
    for a, b, z in zip(ix, iy, x[:, 2]):
        hmap[a, b] = z if np.isnan(hmap[a, b]) else max(hmap[a, b], z)
    return {"rest_top_z": float(x[on, 2].mean()) if on.any() else float("nan"), "rest_mean_z": float(x[:, 2].mean()),
            "rest_min_z": float(x[:, 2].min()), "xy_extent": float(0.5 * (np.ptp(x[:, 0]) + np.ptp(x[:, 1]))),
            "settle_time": _settle_time(ke, dt), "ke_peak": float(ke.max()), "ke_peak_t": float(ke.argmax() * dt),
            "min_z_any_t": float(pos[:, :, 2].min()), "_hmap": hmap}


def rope_metrics(pos, pinned_mask):
    dt = 1.0 / HZ
    pin = pos[:, pinned_mask].mean(1)
    x0 = pos[0]
    far = x0[:, 0] > x0[:, 0].max() - 0.011
    tip = pos[:, far].mean(1)
    rel = tip - pin
    L = np.linalg.norm(rel[0])
    ang = np.degrees(np.arctan2(rel[:, 2], rel[:, 0]))          # 0 = horizontal toward +x, -90 = hanging
    t_bottom = float(np.argmax(ang < -89.0) * dt) if (ang < -89.0).any() else float("nan")
    xr = rel[:, 0]
    # swing: zero crossings of the horizontal offset after the first pass
    zc = np.where(np.diff(np.sign(xr)) != 0)[0]
    period = float(2 * np.mean(np.diff(zc)) * dt) if len(zc) > 2 else float("nan")
    peaks = [np.abs(xr[a:b]).max() for a, b in zip(zc[:-1], zc[1:])] if len(zc) > 2 else []
    decr = float(np.mean(np.log(np.array(peaks[:-1]) / np.array(peaks[1:])))) if len(peaks) > 2 else float("nan")
    return {"length0": float(L), "t_first_vertical": t_bottom, "swing_period": period, "log_decrement": decr,
            "swing_peaks": [round(float(v), 3) for v in peaks[:8]],
            "first_backswing_x": float(xr[zc[0]:zc[1]].min()) if len(zc) > 1 else float("nan"),
            "rest_tip_drop": float(-rel[-1, 2]), "rest_length_ratio": float(np.linalg.norm(rel[-1]) / L),
            "_angle": ang}


def cube_metrics(pos, vel, mass_per_node, size=0.2):
    dt = 1.0 / HZ
    cz = pos[:, :, 2].mean(1)
    ke = 0.5 * mass_per_node * (vel ** 2).sum((1, 2))
    kmin = int(np.argmin(cz[: int(1.0 / dt)]))
    after = cz[kmin:]
    peak = kmin + int(np.argmax(after[: int(0.5 / dt)]))
    return {"cz_every_0.1s": [round(float(v), 3) for v in cz[::20][:21]],
            "t_impact_min": float(kmin * dt), "min_centroid_z": float(cz[kmin]), "bounce_peak_z": float(cz[peak]),
            "bounce_peak_t": float(peak * dt), "rest_centroid_z": float(cz[-1]), "rest_height": float(np.ptp(pos[-1, :, 2])),
            "max_compression": float(1 - np.ptp(pos[kmin, :, 2]) / size), "settle_time": _settle_time(ke, dt),
            "min_node_z": float(pos[:, :, 2].min()), "rest_min_node_z": float(pos[-1, :, 2].min()), "_cz": cz}


def physx_metrics(rec_dir):
    r = np.load(os.path.join(rec_dir, "record.npz"))
    meta = json.load(open(os.path.join(rec_dir, "meta.json")))
    out = {}
    c = meta["scenes"]["cloth"]
    out["cloth"] = cloth_metrics(r["cloth_pos"], r["cloth_vel"], c["particle_mass"])
    rope_pos = r["rope_pos"].copy(); rope_pos[:, :, 0] -= meta["scenes"]["rope"]["x0"]
    out["rope"] = rope_metrics(rope_pos, r["pinned"].astype(bool))
    cube_pos = r["cube_pos"].copy(); cube_pos[:, :, 0] -= meta["scenes"]["cube"]["x0"]
    out["cube"] = cube_metrics(cube_pos, r["cube_vel"], meta["scenes"]["cube"]["mass"] / cube_pos.shape[1])
    return out, meta


# --------------------------------------------------------------------------------------------------
# MetalSim scenes. Physical mappings (PhysX -> MetalSim) are marked PHYS; everything else is a free (fitted) parameter.
# --------------------------------------------------------------------------------------------------

def cloth_xml(meta, p, dt):
    c = meta["scenes"]["cloth"]; g = meta["ground"]
    n, w = c["verts_per_side"], c["size_m"]
    sp = w / (n - 1)
    fr = max(c["pbd_friction"], g["dynamic_friction"])                      # PHYS (MuJoCo combines by max)
    if p.get("young"):     # FEM membrane (PhysX surface deformable / flex StVK), needs the Euler integrator
        integ = ""
        stretch = (f'<edge equality="false"/><elasticity young="{p["young"]}" poisson="{p["poisson"]}" thickness="{p["thickness"]}" '
                   f'damping="{p.get("elastic_damping", 0)}" elastic2d="{p.get("elastic2d", "stretch")}"/>')
    else:
        integ = 'integrator="implicitfast"'
        stretch = f'<edge equality="true" solref="{p["edge_solref"]}"/>'
    return f"""<mujoco><option timestep="{dt}" solver="CG" iterations="{p.get('iterations', 50)}" ls_iterations="20" jacobian="sparse"
      {integ}><flag energy="disable"/></option>
    <worldbody><geom type="plane" size="3 3 0.1" friction="{g['dynamic_friction']} 0.005 0.0001"/>
    <geom type="box" size="{c['box_size']/2} {c['box_size']/2} {c['box_size']/2}" pos="0 0 {c['box_size']/2}" friction="{g['dynamic_friction']} 0.005 0.0001"/>
    <flexcomp name="cloth" type="grid" count="{n} {n} 1" spacing="{sp} {sp} {sp}" pos="0 0 {c['z0']}" dim="2"
      radius="{c['particle_rest_offset']}" mass="{c['mass']}">
      {stretch}
      <contact condim="3" friction="{fr}" solref="{p['contact_solref']}" selfcollide="none"/>
    </flexcomp></worldbody></mujoco>"""


def _rod_pin_grid(nx):
    return " ".join(f"0 {j} {k}" for j in (0, 1) for k in (0, 1))


def rope_xml(meta, p, dt):
    r = meta["scenes"]["rope"]; m = r["material"]
    nx = p.get("rope_nx", 11)
    sp = r["length"] / (nx - 1)
    return f"""<mujoco><option timestep="{dt}" solver="CG" iterations="{p.get('iterations', 50)}" ls_iterations="20" jacobian="sparse"/>
    <worldbody><geom type="plane" size="3 3 0.1"/>
    <flexcomp name="rope" type="grid" count="{nx} 2 2" spacing="{sp} {r['thickness']} {r['thickness']}" pos="{r['length']/2} 0 {r['z0']}"
      dim="3" radius="0.001" mass="{r['mass']}">
      <pin grid="{_rod_pin_grid(nx)}"/>
      <elasticity young="{m['youngs_modulus']}" poisson="{m['poissons_ratio']}" damping="{p['elastic_damping']}"/>
      <contact selfcollide="none"/>
    </flexcomp></worldbody></mujoco>"""


def cube_xml(meta, p, dt):
    c = meta["scenes"]["cube"]; m = c["material"]; g = meta["ground"]
    n = p.get("cube_n", 5)
    sp = c["size"] / (n - 1)
    fr = 0.5 * (m["dynamic_friction"] + g["dynamic_friction"])              # PHYS (PhysX default combine: average)
    return f"""<mujoco><option timestep="{dt}" solver="CG" iterations="{p.get('iterations', 50)}" ls_iterations="20" jacobian="sparse"/>
    <worldbody><geom type="plane" size="3 3 0.1" friction="{fr} 0.005 0.0001"/>
    <flexcomp name="cube" type="grid" count="{n} {n} {n}" spacing="{sp} {sp} {sp}" pos="0 0 {c['z0']}" dim="3"
      radius="{p.get('cube_radius', 0.001)}" mass="{c['mass']}">
      <elasticity young="{m['youngs_modulus']}" poisson="{m['poissons_ratio']}" damping="{p['elastic_damping']}"/>
      <contact condim="3" friction="{fr}" solref="{p['contact_solref']}" solimp="{p.get('contact_solimp', '0.9 0.95 0.001 0.5 2')}"
        margin="{p.get('contact_margin', 0)}" selfcollide="none"/>
    </flexcomp></worldbody></mujoco>"""


FLEX_DEFAULTS = {
    # PHYS: the spring k [N/m] on a particle pair of mass m acts on the relative coordinate with effective mass m/2,
    # i.e. an acceleration-level stiffness 2k/m and damping 2d/m, which is what a negative MuJoCo solref encodes.
    "cloth": {"edge_solref": None, "contact_solref": "0.01 1", "dt": 0.005},
    # PhysX elasticity_damping 0.005 has no usable flex counterpart: MuJoCo's flex elasticity damping is explicit and
    # blows up above ~1e-4 (rod) / 1e-3 (cube) at 0.5 ms (measured in MuJoCo C and Warp): fitted instead. Young's modulus
    # and Poisson's ratio map directly (PHYS; different constitutive laws: PhysX co-rotational vs flex StVK).
    "rope": {"elastic_damping": 1e-4, "dt": 0.0005},
    "cube": {"elastic_damping": 1e-3, "contact_solref": "0.01 1", "dt": 0.0005},
}
XPBD_DEFAULTS = {
    # PHYS: identical formulation to PhysX particle cloth: compliance 1/k per spring, damping d per spring, 16 substeps
    "cloth": {"stretch_compliance": 1e-4, "bend_compliance": 5e-3, "stretch_damping": 0.2, "bend_damping": 0.2, "damping": 0.0,
              "substeps": 16},
    "rope": {"damping": 0.0, "substeps": 16},             # stiffness/damping: physical mapping in run_xpbd
}


def _flex_params(obj, meta, params):
    p = dict(FLEX_DEFAULTS[obj]); p.update(params or {})
    if obj == "cloth" and p.get("edge_solref") is None and not p.get("young"):
        c = meta["scenes"]["cloth"]
        k, d, m = c["spring_stretch_stiffness"], c["spring_damping"], c["particle_mass"]
        p["edge_solref"] = f"{-2 * k / m} {-2 * d / m}"
    return p


def run_flex(obj, meta, params=None, device="cpu", nworld=1):
    import mujoco, warp as wp
    wp.config.quiet = True
    from metalsim.physics import deformable as dfm
    p = _flex_params(obj, meta, params)
    dt = float(p["dt"])
    from mujoco_warp._src import flex_damping
    flex_damping.ENABLE = bool(p.get("implicit_damping", False))
    from mujoco_warp._src import collision_flex as _cfx
    _cfx.FLEX_MAXCONPAIR = int(p.get("maxconpair", 50))
    xml = {"cloth": cloth_xml, "rope": rope_xml, "cube": cube_xml}[obj](meta, p, dt)
    m = mujoco.MjModel.from_xml_string(xml)
    sim = dfm.DeformableSim(m, nworld, device=device, capture=device != "cpu")
    sub = int(round(1.0 / HZ / dt))
    nframes = int(round(T_END * HZ))
    P, V = [], []
    vert_body = m.flex_vertbodyid

    def grab():
        x = sim.d.flexvert_xpos.numpy()[0].copy()
        # vertex velocities from body linear velocity (cvel is (ang, lin) at the subtree com; vertex bodies are points)
        v = np.zeros_like(x)
        dof = m.body_dofadr[vert_body]
        qv = sim.d.qvel.numpy()[0]
        ok = vert_body > 0
        v[ok] = np.stack([qv[dof[ok]], qv[dof[ok] + 1], qv[dof[ok] + 2]], 1)
        P.append(x); V.append(v)

    grab()
    t0 = time.time()
    for _ in range(nframes):
        for _ in range(sub):
            sim.step()
        grab()
    wall = time.time() - t0
    pos, vel = np.array(P), np.array(V)
    pinned = np.array([m.body_dofnum[b] == 0 for b in vert_body])
    return pos, vel, {"model_nv": int(m.nv), "nvert": int(m.nflexvert), "wall_s": wall, "params": p, "pinned": pinned}


def make_xpbd(obj, meta, params=None, device="cpu", nworld=1, _info=None):
    import mujoco, warp as wp
    wp.config.quiet = True
    from metalsim.physics import deformable as dfm
    p = dict(XPBD_DEFAULTS[obj]); p.update(params or {})
    # the same flex topology/masses as the flex scene; XPBD solves it with its own kernels at 200 Hz x substeps
    fp = _flex_params(obj, meta, {})
    xml = {"cloth": cloth_xml, "rope": rope_xml}[obj](meta, fp, 1.0 / HZ)
    rope_phys = None
    if obj == "rope":
        # XPBD rod on the rod's centre line, physical mapping from PhysX's rod (PHYS):
        #  - clamp: PhysX holds the whole end face (4 nodes), which fixes position and direction; here node 0 at
        #    x = -l and node 1 at x = 0 are held (node 1 is the attachment point used by the metrics)
        #  - segment l = 0.05 m (PhysX's hexahedral resolution: 10 cells over 0.5 m)
        #  - lumped masses: rho A l per interior node, rho A l / 2 at the tip (trapezoid rule)
        #  - stretch k = E A / l, rod bending k = 4 E I / l^3 (I = t^4 / 12), per-constraint damping d = beta k with
        #    beta = PhysX elasticity damping (stiffness-proportional damping, PhysX's XPBD form)
        r = meta["scenes"]["rope"]; mat = r["material"]
        nseg = int(p.get("rope_segments", 10))
        l = r["length"] / nseg
        E, t, beta = mat["youngs_modulus"], r["thickness"], mat["elasticity_damping"]
        A, I = t * t, t ** 4 / 12.0
        # bending stiffness: "continuum" = E t^4/12, or the effective EI of the 11x2x2 hexahedral FEM rod that PhysX
        # simulates (measured on the same mesh in flex, static cantilever: 1.65e-2 N m^2, 12.4x the continuum value:
        # one-cell cross-sections lock in bending), p["bend_EI"]
        EI = p.get("bend_EI", E * I)
        ks, kb = E * A / l, 4.0 * EI / l ** 3
        p.setdefault("stretch_compliance", 1.0 / ks); p.setdefault("bend_compliance", 1.0 / kb)
        beta = p.get("beta", beta)
        p.setdefault("stretch_damping", beta * ks); p.setdefault("bend_damping", beta * kb)
        p.setdefault("rope_bending", "midpoint")
        rope_phys = {"l": l, "rhoAl": mat["density"] * A * l, "k_stretch": ks, "k_bend": kb, "beta": beta}
        xml = f"""<mujoco><option timestep="{1.0/HZ}"/><worldbody><geom type="plane" size="3 3 0.1"/>
          <flexcomp name="rope" type="grid" count="{nseg + 2} 1 1" spacing="{l} 0.02 0.02" pos="{(r['length'] - l)/2} 0 {r['z0']}"
            dim="1" radius="{min(t / 2, 0.45 * l)}" mass="{r['mass']}"><pin id="0 1"/><edge equality="true"/></flexcomp></worldbody></mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml)
    cfg = dfm.XPBDCfg(substeps=int(p["substeps"]), stretch_compliance=p["stretch_compliance"], bend_compliance=p["bend_compliance"],
                      stretch_damping=p["stretch_damping"], bend_damping=p["bend_damping"], damping=p["damping"],
                      friction=meta["scenes"]["cloth"]["pbd_friction"] if obj == "cloth" else 0.5,
                      rope_bending=p.get("rope_bending", "distance"))
    sim = dfm.XPBDSim(m, nworld, device=device, cfg=cfg, capture=device != "cpu")
    if rope_phys is not None and p.get("lumped_mass", True):
        nvx = sim.nvert
        mass = np.full(nvx, rope_phys["rhoAl"], np.float32); mass[-1] *= 0.5
        inv = 1.0 / mass; inv[:2] = 0.0
        sim.mass.assign(mass); sim.inv_mass.assign(inv.astype(np.float32))
    if _info is not None:
        _info.update(p=p, rope_phys=rope_phys)
    return sim


def run_xpbd(obj, meta, params=None, device="cpu", nworld=1):
    info = {}
    sim = make_xpbd(obj, meta, params, device, nworld, info)
    p, rope_phys = info["p"], info["rope_phys"]
    nframes = int(round(T_END * HZ))
    P, V = [sim.x.numpy()[0].copy()], [sim.v.numpy()[0].copy()]
    t0 = time.time()
    for _ in range(nframes):
        sim.step()
        P.append(sim.x.numpy()[0].copy()); V.append(sim.v.numpy()[0].copy())
    wall = time.time() - t0
    pinned = sim.inv_mass.numpy() == 0
    if rope_phys is not None:           # attachment point = the pinned node at x = 0 (PhysX's pinned face)
        pinned = np.zeros(sim.nvert, bool); pinned[1] = True
    return np.array(P), np.array(V), {"nvert": int(sim.nvert), "wall_s": wall, "params": p, "pinned": pinned}


def ours_metrics(obj, pos, vel, info, meta):
    if obj == "cloth":
        return cloth_metrics(pos, vel, meta["scenes"]["cloth"]["mass"] / pos.shape[1])
    if obj == "rope":
        return rope_metrics(pos, info["pinned"])
    return cube_metrics(pos, vel, meta["scenes"]["cube"]["mass"] / pos.shape[1])


# --------------------------------------------------------------------------------------------------
# Residuals and fit
# --------------------------------------------------------------------------------------------------

KEYS = {
    "cloth": [("rest_top_z", 0.01), ("rest_mean_z", 0.01), ("xy_extent", 0.02), ("settle_time", 0.25), ("ke_peak_t", 0.02)],
    "rope": [("t_first_vertical", 0.02), ("swing_period", 0.05), ("log_decrement", 0.1), ("first_backswing_x", 0.05),
             ("rest_tip_drop", 0.01)],
    "cube": [("min_centroid_z", 0.005), ("bounce_peak_z", 0.005), ("rest_centroid_z", 0.002), ("t_impact_min", 0.01),
             ("settle_time", 0.25)],
}


def residuals(obj, ref, ours):
    out = {}
    for k, scale in KEYS[obj]:
        a, b = ref[obj][k] if isinstance(ref.get(obj), dict) else ref[k], ours[k]
        out[k] = {"physx": a, "metalsim": b, "diff": (b - a) if np.isfinite(a) and np.isfinite(b) else float("nan"), "scale": scale}
    if obj == "cloth":
        h1, h2 = ref["cloth"]["_hmap"] if "cloth" in ref else ref["_hmap"], ours["_hmap"]
        both = ~np.isnan(h1) & ~np.isnan(h2)
        occ1, occ2 = ~np.isnan(h1), ~np.isnan(h2)
        out["heightmap_rmse"] = {"metalsim": float(np.sqrt(np.mean((h1[both] - h2[both]) ** 2))), "scale": 0.01}
        out["silhouette_iou"] = {"metalsim": float((occ1 & occ2).sum() / max(1, (occ1 | occ2).sum())), "scale": None}
    return out


def loss(res):
    tot = 0.0
    for k, v in res.items():
        if v.get("scale") is None:
            continue
        d = v.get("diff", v.get("metalsim"))
        tot += (d / v["scale"]) ** 2 if np.isfinite(d) else 1e3
    return float(tot)


GRIDS = {
    # the physical mapping (solref -2k/m, -2d/m = -1e6, -20) is unstable at 5 ms (measured: diverges); at 1 ms it runs.
    # Positive solrefs are MuJoCo's usual (timeconst, dampratio) and are fitted.
    ("flex", "cloth"): {"edge_solref": ["phys", "0.01 1", "0.02 1"], "contact_solref": ["0.01 1", "0.02 1"], "dt": [0.001, 0.005]},
    ("flex", "rope"): {"elastic_damping": [0.0, 3e-5, 1e-4]},
    ("flex", "cube"): {"elastic_damping": [3e-4, 1e-3, 2e-3], "contact_solref": ["0.005 1", "0.01 1", "0.02 1"]},
    ("xpbd", "cloth"): {"bend_compliance": [5e-3, 5e-2], "damping": [0.0, 0.5]},
    ("xpbd", "rope"): {"bend_compliance": [1e-3, 1e-2, 1e-1], "damping": [0.0, 0.2]},
}


def expand(backend, obj, meta, combo):
    p = dict(combo)
    if backend == "flex" and obj == "cloth" and p.get("edge_solref") == "phys":
        p["edge_solref"] = None
    if backend == "flex" and obj == "cloth" and "edge_scale" in p:
        c = meta["scenes"]["cloth"]; s = p.pop("edge_scale")
        p["edge_solref"] = f"{-2 * c['spring_stretch_stiffness'] * s / c['particle_mass']} {-2 * c['spring_damping'] / c['particle_mass']}"
    return p


def evaluate(backend, obj, meta, ref, params, device):
    run = run_flex if backend == "flex" else run_xpbd
    pos, vel, info = run(obj, meta, params, device=device)
    om = ours_metrics(obj, pos, vel, info, meta)
    res = residuals(obj, ref, om)
    return {"params": info["params"], "loss": loss(res), "residuals": res, "wall_s": info["wall_s"], "nvert": info["nvert"],
            "finite": bool(np.isfinite(pos).all())}


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items() if not str(k).startswith("_")}
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["metrics", "run", "fit"])
    ap.add_argument("rec")
    ap.add_argument("--backend", default="flex", choices=["flex", "xpbd"])
    ap.add_argument("--object", default="all")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--params", default="{}")
    a = ap.parse_args()
    ref, meta = physx_metrics(a.rec)
    if a.cmd == "metrics":
        print(json.dumps(_clean(ref), indent=1)); return
    objs = ["cloth", "rope", "cube"] if a.object == "all" else [a.object]
    if a.backend == "xpbd":
        objs = [o for o in objs if o != "cube"]
    for obj in objs:
        if a.cmd == "run":
            r = evaluate(a.backend, obj, meta, ref, json.loads(a.params).get(obj, {}), a.device)
            print(json.dumps({"backend": a.backend, "object": obj, **_clean(r)}), flush=True)
        else:
            grid = GRIDS[(a.backend, obj)]
            best = None
            for combo in itertools.product(*grid.values()):
                params = expand(a.backend, obj, meta, dict(zip(grid.keys(), combo)))
                try:
                    r = evaluate(a.backend, obj, meta, ref, params, a.device)
                except Exception as e:  # noqa: BLE001
                    print(json.dumps({"backend": a.backend, "object": obj, "params": params, "error": repr(e)[:200]}), flush=True)
                    continue
                print(json.dumps({"backend": a.backend, "object": obj, "fit_candidate": True, **_clean(r)}), flush=True)
                if r["finite"] and (best is None or r["loss"] < best["loss"]):
                    best = r
            print(json.dumps({"backend": a.backend, "object": obj, "best": True, **_clean(best)}) if best else "{}", flush=True)


if __name__ == "__main__":
    main()


def run_physxcloth(obj, meta, params=None, device="cpu", nworld=1):
    """MetalSim's port of PhysX 5.6.1's PBD particle cloth (metalsim.physics.physx_cloth) on the 5.1 cloth protocol:
    every parameter physical (the recorder's ParticleClothDemo values)."""
    assert obj == "cloth"
    from metalsim.physics import physx_cloth as pc
    c = meta["scenes"]["cloth"]; g = meta["ground"]
    p = dict(params or {})
    cfg = pc.PhysXClothCfg(iterations=c["solver_position_iterations"], rest_offset=c["particle_rest_offset"], contact_offset=c["contact_offset"],
                           friction=c["pbd_friction"], stretch_stiffness=c["spring_stretch_stiffness"], shear_stiffness=c["spring_shear_stiffness"],
                           bend_stiffness=c["spring_bend_stiffness"], spring_damping=c["spring_damping"], **p)
    mesh = pc.grid_cloth(c["verts_per_side"], c["size_m"], z0=c["z0"], mass=c["particle_mass"])
    sim = pc.PhysXClothSim(mesh, nworld, obstacles=[pc.plane(), pc.box((0, 0, c["box_size"] / 2), (c["box_size"] / 2,) * 3)], cfg=cfg,
                           device=device, capture=device != "cpu")
    P, V = [sim.x.numpy()[0].copy()], [sim.v.numpy()[0].copy()]
    t0 = time.time()
    for _ in range(int(round(T_END * HZ))):
        sim.step(); P.append(sim.x.numpy()[0].copy()); V.append(sim.v.numpy()[0].copy())
    return np.array(P), np.array(V), {"nvert": len(mesh.x), "wall_s": time.time() - t0, "params": p, "pinned": None}

"""Mesh-resolution sweep on MetalSim's backends (shear-locking test, companion of record_deformables_il3.py --resolutions).
flex rod: static cantilever (reduced gravity, heavily damped via implicit damping) -> effective bending stiffness EI_eff
          vs cells across the 2 cm cross-section, at 5 cm and at cubic cells along the length;
flex cube: rest height and drop bounce vs cells per edge;
XPBD rod: swing period vs segment count with the `physical` preset (a 1D rod has no cross-section to lock).
usage: mesh_sweep.py OUT.json"""
import warp as wp
wp.set_device("cpu")   # CPU device only: no GPU work, no queue needed
import sys, os, json, numpy as np, mujoco, warp as wp
wp.config.quiet = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import physx_protocol as pp
import mujoco_warp as mjw
from mujoco_warp._src import flex_damping as fd
from metalsim.physics import deformable as dfm

out = {"rod_flex": [], "cube_flex": [], "rod_xpbd": []}
E, nu, t, L, rho = 1e5, 0.4, 0.02, 0.5, 1000.0
EI_c = E * t ** 4 / 12


def settle(xml, gscale, nsteps, damping_on=True):
    fd.ENABLE = damping_on
    m = mujoco.MjModel.from_xml_string(xml); m.opt.gravity[:] = [0, 0, -9.81 * gscale]
    d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    with wp.ScopedDevice("cpu"):
        M = mjw.put_model(m); M.opt.warn_overflow = 0; D = mjw.put_data(m, d, nworld=1, nconmax=4000, njmax=20000)
        for _ in range(nsteps):
            mjw.step(M, D)
        return D.flexvert_xpos.numpy()[0], float(np.abs(D.qvel.numpy()).max()), m


for across, along in [(1, 0.05), (2, 0.05), (4, 0.05), (1, 0.02), (2, 0.01)]:
    nx = int(round(L / along)) + 1; n = across + 1; h = t / across
    pins = " ".join(f"0 {j} {k}" for j in range(n) for k in range(n))
    dt = 2.5e-4 if along >= 0.02 else 1e-4
    xml = f"""<mujoco><option timestep="{dt}" solver="CG" iterations="50" ls_iterations="20" jacobian="sparse"/><worldbody>
      <flexcomp name="rod" type="grid" count="{nx} {n} {n}" spacing="{along} {h} {h}" pos="{L/2} 0 1" dim="3" radius="0.0005" mass="{rho*L*t*t}">
      <pin grid="{pins}"/><elasticity young="{E}" poisson="{nu}" damping="0.05"/><contact selfcollide="none" contype="0" conaffinity="0"/></flexcomp></worldbody></mujoco>"""
    x, vmax, m = settle(xml, 0.01, int(round(1.5 / dt)))
    tip = x[x[:, 0] > L - 1e-3][:, 2].mean() - 1.0
    w = rho * t * t * 9.81 * 0.01
    EI = w * L ** 4 / (8 * abs(tip))
    r = {"cells_across": across, "cell_along_m": along, "nverts": int(m.nflexvert), "tip_deflection_m": float(tip),
         "EI_eff": float(EI), "EI_over_continuum": float(EI / EI_c), "max_residual_speed": vmax}
    out["rod_flex"].append(r); print(json.dumps(r), flush=True)

meta = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "runs", "deformable", "isaac51", "meta.json")))
for cells in (2, 4, 8):
    P = {"elastic_damping": 0.1, "implicit_damping": True, "contact_solref": "-1e6 -300", "cube_n": cells + 1,
         "dt": 5e-4 if cells <= 4 else 2.5e-4}
    pos, vel, info = pp.run_flex("cube", meta, P, device="cpu")
    mm = pp.cube_metrics(pos, vel, 8.0 / pos.shape[1])
    r = {"cells_per_edge": cells, "nverts": int(pos.shape[1]), **{k: float(mm[k]) for k in ("bounce_peak_z", "settle_time", "min_centroid_z",
         "rest_centroid_z", "rest_height", "min_node_z")}, "finite": bool(np.isfinite(pos).all())}
    out["cube_flex"].append(r); print(json.dumps(r), flush=True)

for nseg in (5, 10, 20, 40):
    P = {"rope_segments": nseg, "beta": 0.005}
    pos, vel, info = pp.run_xpbd("rope", meta, P, device="cpu")
    mm = pp.rope_metrics(pos, info["pinned"])
    r = {"segments": nseg, **{k: float(mm[k]) for k in ("swing_period", "log_decrement", "first_backswing_x", "rest_tip_drop")}}
    out["rod_xpbd"].append(r); print(json.dumps(r), flush=True)

json.dump(out, open(sys.argv[1], "w"), indent=1)

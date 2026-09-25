"""Thin-rod references for the Isaac Lab 3.0 rope sweep: MetalSim XPBD rod (`physical` preset, 20 and 40 segments) with
each backend's material, analysed with il3_analysis.rope (same metric code as the recordings)."""
import warp as wp
wp.set_device("cpu")   # CPU device only: no GPU work, no queue needed
import sys, os, json, copy, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import physx_protocol as pp, il3_analysis as ia

base = json.load(open("runs/deformable/isaac51/meta.json"))
mats = {"isaacsim_physx": dict(E=1e6, nu=0.45, rho=1000.0, beta=0.005),
        # Newton VBD: k_mu = k_lambda = 1e5 -> E = mu(3 lam + 2 mu)/(lam + mu) = 2.5e5, nu = 0.25; density 1 kg/m^3; k_damp 0
        "newton_vbd": dict(E=2.5e5, nu=0.25, rho=1.0, beta=0.0)}
out = {}
for B, mt in mats.items():
    meta = copy.deepcopy(base)
    r = meta["scenes"]["rope"]; r["material"].update(youngs_modulus=mt["E"], poissons_ratio=mt["nu"], density=mt["rho"], elasticity_damping=mt["beta"])
    r["mass"] = mt["rho"] * 0.5 * 0.02 * 0.02
    for nseg in (20, 40):
        pos, vel, info = pp.run_xpbd("rope", meta, {"rope_segments": nseg}, device="cpu")
        pin = np.zeros(pos.shape[1], bool); pin[1] = True
        m = ia.rope(pos, pin); out[f"{B}_xpbd{nseg}"] = m
        print(B, nseg, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items()}, flush=True)
json.dump(out, open("runs/parity3/isaac/deformable/il3_thinrod_xpbd.json", "w"), indent=1)

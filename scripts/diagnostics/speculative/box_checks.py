"""Analytic box checks of the speculative-contact rule (CPU device, 2026-09-27). One MuJoCo Warp configuration per call
(PYTHONPATH / MJW_* from the environment); --c runs MuJoCo C (float64) on the same models, --oracle MODE the
speculative rule in MuJoCo C (c_oracle.SpecC, friction rows at d0).

Box 0.2 x 0.06 x 0.02 m, 3 kg, on a plane with gap 10 mm, elliptic cone impratio 10, contact tau 5 ms (the prototype's
setting), friction 1, 2.5 ms step, Newton 20 iterations (ls_iterations 50 in C).
  A drop   : released 0.1 m above contact, flat. Rigid, restitution 0 (PhysX's model; restitution 0 in Isaac's G1 cfg):
             post-impact v_z = 0, normal impulse over the impact = m v_z-, rest force = m g.
  B slide  : released 20 mm above contact with v_x = 1 m/s. Rigid Coulomb impact: the impact friction impulse is
             bounded by mu * normal impulse, so v_x+ = v_x- - mu v_z- (while sliding), then dv_x/dt = -mu g.
  C skim   : gravity off, flat box 2 mm above the plane, v_x = 1 m/s, v_z = 0: no contact force at all (x keeps 1 m/s).
  D edge   : released 0.1 m above contact, tilted 10 deg about x (lands on one long edge, slaps onto the other).
Prints one JSON line per case.
"""
import json, os, sys, numpy as np, mujoco
H = 0.0025; M = 3.0; G = 9.81; MU = 1.0
XML = """<mujoco><option timestep="0.0025" cone="elliptic" impratio="10" iterations="20" ls_iterations="{ls}" gravity="0 0 {g}"/>
<worldbody><geom name="ground" type="plane" size="5 5 .1" gap="0.01" friction="1 0.005 0.0001" solref="0.005 1" solimp="0.9 0.999 0.005 0.5 2"/>
<body pos="0 0 {z}" euler="{ex} 0 0"><freejoint/><geom type="box" size=".1 .03 .01" mass="3" friction="1 0.005 0.0001" solref="0.005 1" solimp="0.9 0.999 0.005 0.5 2"/></body></worldbody></mujoco>"""
CASES = {"A_drop": dict(z=0.11, ex=0, vx=0.0, g=-G, T=0.6), "B_slide": dict(z=0.03, ex=0, vx=1.0, g=-G, T=0.4),
         "C_skim": dict(z=0.012, ex=0, vx=1.0, g=0.0, T=0.1), "D_edge": dict(z=0.11 + 0.1 * np.sin(np.radians(10)), ex=10, vx=0.0, g=-G, T=0.6)}
use_c = "--c" in sys.argv or "--oracle" in sys.argv
ORACLE = sys.argv[sys.argv.index("--oracle") + 1] if "--oracle" in sys.argv else None     # c_oracle.SpecC mode (friction rows at d0)
if not use_c:
    import warp as wp
    wp.config.quiet = True
    import mujoco_warp as mjw


def run(case):
    c = CASES[case]
    m = mujoco.MjModel.from_xml_string(XML.format(ls=50 if use_c else 20, g=c["g"], z=c["z"], ex=c["ex"]))
    d = mujoco.MjData(m); d.qvel[0] = c["vx"]; mujoco.mj_forward(m, d)
    S = int(round(c["T"] / H)); rec = []
    if use_c:
        if ORACLE:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from c_oracle import SpecC
            orc = SpecC(m, ORACLE, True, "d0"); m = orc.mc
        for s in range(S):
            if ORACLE: orc.step(d)
            else: mujoco.mj_step(m, d)
            fn = 0.0; dmin = 1.0
            for i in range(d.ncon):
                f = np.zeros(6); mujoco.mj_contactForce(m, d, i, f)
                if d.contact[i].efc_address >= 0: fn += f[0]
                dmin = min(dmin, d.contact[i].dist)
            rec.append((d.qpos[:3].copy(), d.qvel[:6].copy(), fn, dmin))
    else:
        with wp.ScopedDevice("cpu"):
            wm = mjw.put_model(m); wd = mjw.put_data(m, d, nworld=1, nconmax=32, njmax=128)
            for s in range(S):
                mjw.step(wm, wd)
                nc = wd.nacon.numpy()[0]; ea = wd.contact.efc_address.numpy()[:nc, 0]; f = wd.efc.force.numpy()[0]
                dist = wd.contact.dist.numpy()[:nc]
                fn = float(sum(f[ea[i]] for i in range(nc) if ea[i] >= 0)); dmin = float(dist.min()) if nc else 1.0
                rec.append((wd.qpos.numpy()[0, :3].copy(), wd.qvel.numpy()[0, :6].copy(), fn, dmin))
    q = np.array([r[0] for r in rec]); v = np.array([r[1] for r in rec]); fn = np.array([r[2] for r in rec]); dmin = np.array([r[3] for r in rec])
    out = {"case": case}
    k0 = int(np.argmax(fn > 1e-6)) if (fn > 1e-6).any() else -1
    out["first_force_step"] = k0; out["first_force_dist_mm"] = round(float(dmin[k0]) * 1e3, 3) if k0 >= 0 else None
    out["peak_pen_mm"] = round(float(-dmin[dmin < 1].min()) * 1e3, 3) if (dmin < 1).any() else None
    if case in ("A_drop", "D_edge", "B_slide") and k0 >= 0:
        vzm = float(v[k0 - 1, 2]) if k0 > 0 else float(v[0, 2]); out["vz_minus"] = round(vzm, 4)
        w = slice(k0, k0 + 8)                                             # 20 ms impact window
        Jn = float(fn[w].sum() * H); out["impulse_Ns"] = round(Jn, 4)
        out["impulse_expected_Ns (m|vz-| + m g 20ms)"] = round(M * -vzm + M * -c["g"] * 8 * H, 4)
        out["vz_after_20ms"] = round(float(v[k0 + 7, 2]), 4)
        out["wx_after_20ms"] = round(float(v[k0 + 7, 3]), 4)
        out["vz_wx_after_60ms"] = [round(float(v[k0 + 23, 2]), 4), round(float(v[k0 + 23, 3]), 4)]
        out["peak_wx"] = round(float(np.abs(v[k0:k0 + 40, 3]).max()), 3)
        out["rest_z_mm"] = round(float(q[-1, 2] - 0.01) * 1e3, 4); out["rest_force_N"] = round(float(fn[-20:].mean()), 3)
        if case == "B_slide":
            out["vx_minus"] = round(float(v[k0 - 1, 0]), 4); out["vx_5ms_after_first_force"] = round(float(v[k0 + 1, 0]), 4)
            out["vx_expected_after_impact"] = round(1.0 - MU * -vzm, 4)
            # deceleration while sliding, 40-120 ms after impact
            out["stop_ms_expected (rigid Coulomb)"] = round(max(1.0 - MU * -vzm, 0.0) / (MU * G) * 1e3, 1)
            ks = int(np.argmax(np.abs(v[k0:, 0]) < 1e-3)); out["stop_ms_after_impact"] = round(ks * H * 1e3, 1)
    if case == "C_skim":
        out["max_Fn_N"] = round(float(fn.max()), 4); out["vx_end"] = round(float(v[-1, 0]), 6); out["z_change_mm"] = round(float(q[-1, 2] - q[0, 2]) * 1e3, 4)
    return out


for case in CASES:
    print(json.dumps(run(case)), flush=True)

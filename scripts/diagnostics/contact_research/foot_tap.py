"""Foot-tap experiment (MuJoCo C, 2.5 ms, the G1 foot collider: box 0.2031 x 0.0655 x 0.0185 m as in g1_minimal.usd,
3 kg effective mass): touch-down at 0.5 m/s, loaded stance (250 N extra down force), heel-off roll onto the toe edge
(pitch torque ramp with the down force ramping out), lift-off (up force). Per contact setting, the timeline of
Isaac's contact flag (|net normal force| > 1 N) at 2.5 ms (ours: ContactSensor every substep), on 5 ms averages
(PhysX's report: impulse over a 5 ms step / 5 ms), and the feet_slide window (max over the history: ours 6 x 2.5 ms,
Isaac 3 x 5 ms), against the geometric contact (min distance < 0).

    python scripts/diagnostics/contact_research/foot_tap.py
"""
import os, sys
import numpy as np, mujoco
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)
from metalsim.physics import contact_tuning

XML = """<mujoco><option timestep="0.0025" integrator="implicitfast" cone="pyramidal" iterations="10" ls_iterations="20"/>
<worldbody><geom name="ground" type="plane" size="0 0 0.05" friction="0.8 0.005 0.0001"/>
<body name="foot" pos="0 0 0.02"><freejoint/><geom name="sole" type="box" size="0.10155 0.03275 0.00925" mass="3"/></body></worldbody></mujoco>"""
SETTINGS = {"default": {}, "tau10_impact": dict(solref=(0.01, 1.0), solimp=(0.9, 0.999, 0.005, 0.5, 2.0)),
            "tau5_imp99": dict(solref=(0.005, 1.0), solimp=(0.99, 0.999, 0.001, 0.5, 2.0)),
            "il3mapped (0.005, 1.375)": dict(solref=(0.005, 1.375)), "tau10_impact_dr2": dict(solref=(0.01, 2.0), solimp=(0.9, 0.999, 0.005, 0.5, 2.0))}
T_TOUCH, T_ROLL, T_LIFT, T_END = 0.0, 0.30, 0.45, 0.60


def run(kw):
    m = mujoco.MjModel.from_xml_string(XML)
    if kw: contact_tuning.set_contacts(m, **kw)
    d = mujoco.MjData(m); d.qvel[2] = -0.5; mujoco.mj_forward(m, d)
    n = int(T_END / m.opt.timestep); F = np.zeros(n); geo = np.zeros(n, bool); f6 = np.zeros(6)
    for k in range(n):
        t = k * m.opt.timestep; d.xfrc_applied[:] = 0
        if t < T_ROLL: d.xfrc_applied[1, 2] = -250.0
        elif t < T_LIFT:                                           # heel-off: roll onto the toe edge
            s = (t - T_ROLL) / (T_LIFT - T_ROLL); d.xfrc_applied[1, 2] = -250.0 * (1 - s); d.xfrc_applied[1, 4] = -40.0 * s
        else:
            d.xfrc_applied[1, 2] = 60.0                            # lift
        mujoco.mj_step(m, d)
        for i in range(d.ncon):
            if d.contact[i].efc_address < 0: continue
            mujoco.mj_contactForce(m, d, i, f6); F[k] += f6[0]
        geo[k] = d.ncon > 0 and min(c.dist for c in d.contact[:d.ncon]) < 0
    return F, geo


def trans(c):
    return np.flatnonzero(np.diff(c.astype(int)) != 0) + 1


print("| setting | first geometric touch / first flag (2.5 ms) / first flag (5 ms avg) [ms] | flag transitions 2.5 ms / 5 ms avg (touch-down, stance+roll, lift-off) | last flag / last geometric contact [ms] | slide window on at lift: ours 6x2.5 / Isaac-style 3x5 [ms after last flag] |")
print("|---|---|---|---|---|")
for name, kw in SETTINGS.items():
    F, geo = run(kw); dt = 0.0025
    c = F > 1.0; c5 = F.reshape(-1, 2).mean(1) > 1.0
    tr = trans(c); tr5 = trans(c5)
    first_geo = np.argmax(geo) * dt * 1e3; first = np.argmax(c) * dt * 1e3; first5 = np.argmax(c5) * 5.0
    last = (len(c) - np.argmax(c[::-1]) - 1) * dt * 1e3; last_geo = (len(geo) - np.argmax(geo[::-1]) - 1) * dt * 1e3
    ph = lambda tt, step: f"{(tt * step < 0.05).sum()}, {((tt * step >= 0.05) & (tt * step < T_LIFT)).sum()}, {(tt * step >= T_LIFT).sum()}"
    # slide window: ours = any flag over the last 6 substeps; Isaac-style = any 5 ms-average flag over the last 3 x 5 ms
    h6 = np.array([c[max(0, k - 5):k + 1].any() for k in range(len(c))]); h3 = np.repeat(np.array([c5[max(0, j - 2):j + 1].any() for j in range(len(c5))]), 2)
    lk = int(last / 1e3 / dt)
    on6 = (np.argmax(~h6[lk:]) if (~h6[lk:]).any() else len(h6) - lk) * dt * 1e3; on3 = (np.argmax(~h3[lk:]) if (~h3[lk:]).any() else len(h3) - lk) * dt * 1e3
    print(f"| {name} | {first_geo:.1f} / {first:.1f} / {first5:.1f} | {len(tr)} ({ph(tr, dt)}) / {len(tr5)} ({ph(tr5, 0.005)}) | {last:.1f} / {last_geo:.1f} | {on6:.1f} / {on3:.1f} |")

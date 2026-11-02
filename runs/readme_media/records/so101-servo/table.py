"""Smoothness table (markdown) from cpu/*/smooth.json (and metalsim/*/smooth.json when present)."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
J = ["pan", "lift", "elbow", "wflex", "wroll", "grip"]
rows = [("expert", "scripted expert"), ("orig", "original PPO"), ("isaac_scaled", "isaac_scaled"),
        ("isaac_unscaled_raw", "isaac_unscaled_raw")]
f = lambda x: " / ".join(f"{v:.0f}" if v >= 10 else f"{v:.1f}" for v in x)
out = ["| controller | servo | success | RMS acc rad/s² (pan / lift / elbow / wflex / wroll / grip) | RMS jerk ×10³ rad/s³ | reversals /s | vel RMS >5 Hz rad/s | share >5 Hz | peak speed rad/s |",
       "|---|---|---|---|---|---|---|---|---|"]
for key, name in rows:
    for sv in ("7v4", "7v4_real"):
        p = os.path.join(HERE, "cpu", f"{key}_{sv}", "smooth.json")
        if not os.path.exists(p):
            continue
        s = json.load(open(p)); S = s["smoothness"]
        out.append(f"| {name} | {'old 7.4 V' if sv == '7v4' else '**7v4_real**'} | {s['summary']['successes']} | {f(S['rms_acc'])} | "
                   f"{f([x / 1e3 for x in S['rms_jerk']])} | {' / '.join(f'{x:.2f}' for x in S['reversals_per_s'])} | "
                   f"{' / '.join(f'{x:.3f}' for x in S['osc_rms'])} | {' / '.join(f'{x:.2f}' for x in S['osc_frac'])} | "
                   f"{' / '.join(f'{x:.1f}' for x in S['peak_speed'])} |")
print("\n".join(out))

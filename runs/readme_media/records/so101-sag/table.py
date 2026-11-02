"""Results table (markdown) for the so101-sag hand-off: the deploy expert on CPU, 100 seeds, per servo variant
(cpu/<variant>/{episodes,smooth}.json from eval_cpu_sag.py), plus the sim replays' sag against the real followers
(compare.json). Columns as so101-expert's table.py."""
import json
import os

H = os.path.dirname(os.path.abspath(__file__))
f = lambda x, d=1: " / ".join(f"{v:.{d}f}" for v in x)
V = ("7v4_real", "bam_m1", "bam_m6", "7v4_real+m6f")
out = ["| servo | success | up at 6 s | time to lift median (mean) s | retries | RMS acc rad/s² (pan / lift / elbow / "
       "wflex / wroll / grip) | RMS jerk ×10³ rad/s³ | reversals /s | vel RMS >5 Hz rad/s | peak speed rad/s |",
       "|---" * 10 + "|"]
for v in V:
    p = os.path.join(H, "cpu", v, "smooth.json")
    if not os.path.exists(p):
        continue
    s = json.load(open(p))
    S, U = s["smoothness"], s["summary"]
    ttl = "-" if U["time_to_lift_median_s"] is None else f"{U['time_to_lift_median_s']:.2f} ({U['time_to_lift_mean_s']:.2f})"
    out.append(f"| {v} | {U['successes']} | {U['up_at_6s']} | {ttl} | {U['retries']} | {f(S['rms_acc'])} | "
               f"{f([x / 1e3 for x in S['rms_jerk']], 2)} | {f(S['reversals_per_s'], 2)} | {f(S['osc_rms'], 3)} | "
               f"{f(S['peak_speed'])} |")
J = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
out += ["", "Expert episodes, steady-state |target - q| (sag.py: median, p90 mrad; n):", "",
        "| servo | phase | pan | lift | elbow | wflex | wroll |", "|---|---|---|---|---|---|---|"]
for v in V:
    p = os.path.join(H, "cpu", v, "smooth.json")
    if not os.path.exists(p):
        continue
    G = json.load(open(p))["sag"]
    for ph in ("free", "lift"):
        out.append(f"| {v} | {ph} | " + " | ".join(
            f"{G[j][ph]['median_mrad']:.0f}, {G[j][ph]['p90_mrad']:.0f} (n {G[j][ph]['n']})" if G[j][ph].get("n") else "-"
            for j in J) + " |")
C = json.load(open(os.path.join(H, "compare.json")))
out += ["", "Replay of the real teleop commands: median sag (mrad) per |g| quintile of the real samples, all steady "
        "samples / after moving with gravity / against gravity:", ""]
for ds, joints in C["bins"].items():
    for j, srcs in joints.items():
        out.append(f"**{ds} {j}** (share of samples after moving with gravity: " + ", ".join(
            f"{k} {C['with_frac'][ds][j][k] * 100:.0f} %" for k in ("real", "7v4_real", "bam_m6")) + ")")
        out.append("")
        out.append("| source | " + " | ".join(f"|g| ≈ {c['g']}" for c in srcs["real"]) + " |")
        out.append("|---" * (1 + len(srcs["real"])) + "|")
        for k in ("real",) + V:
            if k in srcs:
                out.append(f"| {k} | " + " | ".join(
                    f"{c['all'][0]} / {c['with_'][0]} / {c['against'][0]}" for c in srcs[k]) + " |")
        out.append("")
out += ["Median |sim - real| (mrad; signed in brackets) over the |g| bins with ≥ 30 samples, both datasets:", "",
        "| servo | lift all | lift with | lift against | elbow all | elbow with | elbow against | wflex all | wflex with | "
        "wflex against |", "|---" * 10 + "|"]
for v in V:
    S = C["summary"].get(v)
    if S:
        out.append(f"| {v} | " + " | ".join(f"{x['median_abs']} ({x['median_signed']:+.1f})" for x in S.values()) + " |")
print("\n".join(out))

"""Results table (markdown) for the hand-off: the deployment expert's runs (cpu/*, metalsim/*) and, for reference, the
old incremental expert on 7v4_real (so101-servo cpu/expert_7v4_real_freshkin, metalsim/expert_7v4_real)."""
import json, os
H = os.path.dirname(os.path.abspath(__file__)); SV = os.path.join(H, "..", "so101-servo")
f = lambda x, d=1: " / ".join(f"{v:.{d}f}" for v in x)
rows = [("deploy, 7v4_real, CPU", f"{H}/cpu/deploy_7v4_real"), ("deploy, 7v4_real, MetalSim", f"{H}/metalsim/deploy_7v4_real"),
        ("deploy, 12v_real, CPU", f"{H}/cpu/deploy_12v_real"), ("deploy, old 7v4 (kp 998), CPU", f"{H}/cpu/deploy_7v4_old"),
        ("abl.: no firmware accel limit", f"{H}/cpu/abl_no_accel_limit"), ("abl.: host latency 10 ms", f"{H}/cpu/abl_host_10ms"),
        ("abl.: BAM stall 3.41 N.m", f"{H}/cpu/abl_bam_stall"), ("abl.: 50 Hz stream", f"{H}/cpu/abl_hz50")]
out = ["| run | success | up at 6 s | time to lift median (mean) s | retries | RMS acc rad/s² (pan / lift / elbow / wflex / wroll / grip) "
       "| RMS jerk ×10³ rad/s³ | reversals /s | vel RMS >5 Hz rad/s | peak speed rad/s |", "|---" * 10 + "|"]
for name, d in rows:
    p = os.path.join(d, "smooth.json")
    if not os.path.exists(p):
        out.append(f"| {name} | (not run) |||||||||"); continue
    s = json.load(open(p)); S, U = s["smoothness"], s["summary"]
    ttl = "-" if U["time_to_lift_median_s"] is None else f"{U['time_to_lift_median_s']:.2f} ({U['time_to_lift_mean_s']:.2f})"
    out.append(f"| {name} | {U['successes']} | {U['up_at_6s']} | {ttl} | {U['retries']} | {f(S['rms_acc'])} | "
               f"{f([x / 1e3 for x in S['rms_jerk']], 2)} | {f(S['reversals_per_s'], 2)} | {f(S['osc_rms'], 3)} | {f(S['peak_speed'])} |")
for name, p in (("old expert, 7v4_real, CPU fresh kin (so101-servo)", f"{SV}/cpu/expert_7v4_real_freshkin/smooth.json"),
                ("old expert, 7v4_real, MetalSim (so101-servo)", f"{SV}/metalsim/expert_7v4_real/smooth.json")):
    if os.path.exists(p):
        s = json.load(open(p)); S = s["smoothness"]
        n = s["summary"]["successes"] if "summary" in s else s.get("successes")
        out.append(f"| {name} | {n} | 0 | - | - | {f(S['rms_acc'])} | {f([x / 1e3 for x in S['rms_jerk']], 2)} | "
                   f"{f(S['reversals_per_s'], 2)} | {f(S['osc_rms'], 3)} | {f(S['peak_speed'])} |")
out += ["", "Steady-state tracking error |target - q| (median, p90 in mrad; n steady samples), arm joints:", "",
        "| run | phase | pan | lift | elbow | wflex | wroll |", "|---|---|---|---|---|---|---|"]
J = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
for name, d in rows[:2]:
    p = os.path.join(d, "smooth.json")
    if not os.path.exists(p):
        continue
    G = json.load(open(p))["sag"]
    for ph in ("free", "lift"):
        out.append(f"| {name} | {ph} | " + " | ".join(
            f"{G[j][ph]['median_mrad']:.0f}, {G[j][ph]['p90_mrad']:.0f} (n {G[j][ph]['n']})" if G[j][ph].get("n") else "-" for j in J) + " |")
print("\n".join(out))

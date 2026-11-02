"""so101-bam-gpu: the deploy expert on 7v4_bam ("bam") / 7v4_real ("real") across backends, 100 seeds (10000-10099). Rows: summary, joint
smoothness (so101-servo smoothness.joint_motion) and steady-state sag in the lift phase (so101-expert sag.py), as
so101-sag's table.py; then per-seed agreement between pairs of runs (outcome, time to lift, retries).

  .venv/bin/python runs/readme_media/scratch/so101-bam-gpu/compare_runs.py > runs/readme_media/scratch/so101-bam-gpu/table.md
"""
import json
import os

H = os.path.dirname(os.path.abspath(__file__))
SC = os.path.dirname(H)
RUNS = {
    "bam CPU 3.10 so101Sim env, ctrl-0 first budget (so101-sag reference)": os.path.join(SC, "so101-sag", "cpu", "patched_7v4_bam"),
    "bam MetalSim": os.path.join(H, "metalsim", "deploy_7v4_bam"),
    "bam MetalSim, ctrl-0 first budget": os.path.join(H, "metalsim", "deploy_7v4_bam_legacy0"),
    "bam CPU 3.14 film scene (--cpu-dry)": os.path.join(H, "dry", "dry100_7v4_bam"),
    "bam CPU 3.14 film scene, ctrl-0 first budget (--cpu-dry)": os.path.join(H, "dry", "dry100_7v4_bam_legacy0"),
    "bam, datasheet stall clip 1.91 N.m, kp / damping kept (CPU 3.14 --cpu-dry, ablation)":
        os.path.join(H, "dry", "dry100_7v4_bam_ds_stall"),
    "real CPU 3.10 so101Sim env (so101-expert)": os.path.join(SC, "so101-expert", "cpu", "deploy_7v4_real"),
    "real MetalSim (so101-expert)": os.path.join(SC, "so101-expert", "metalsim", "deploy_7v4_real"),
    "real MetalSim (this harness, rerun)": os.path.join(H, "metalsim", "deploy_7v4_real_rerun"),
    "real CPU 3.14 film scene (--cpu-dry)": os.path.join(H, "dry", "dry100_7v4_real"),
}
PAIRS = [("bam MetalSim, ctrl-0 first budget", "bam CPU 3.10 so101Sim env, ctrl-0 first budget (so101-sag reference)"),
         ("bam MetalSim", "bam CPU 3.10 so101Sim env, ctrl-0 first budget (so101-sag reference)"),
         ("bam MetalSim", "bam CPU 3.14 film scene (--cpu-dry)"),
         ("bam MetalSim, ctrl-0 first budget", "bam CPU 3.14 film scene, ctrl-0 first budget (--cpu-dry)"),
         ("bam MetalSim", "bam MetalSim, ctrl-0 first budget"),
         ("bam CPU 3.14 film scene (--cpu-dry)", "bam CPU 3.14 film scene, ctrl-0 first budget (--cpu-dry)"),
         ("bam CPU 3.14 film scene, ctrl-0 first budget (--cpu-dry)",
          "bam CPU 3.10 so101Sim env, ctrl-0 first budget (so101-sag reference)"),
         ("bam, datasheet stall clip 1.91 N.m, kp / damping kept (CPU 3.14 --cpu-dry, ablation)",
          "bam CPU 3.14 film scene (--cpu-dry)"),
         ("real MetalSim (so101-expert)", "real CPU 3.10 so101Sim env (so101-expert)"),
         ("real MetalSim (this harness, rerun)", "real MetalSim (so101-expert)")]
f = lambda x, d=1: " / ".join(f"{v:.{d}f}" for v in x)
J = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
have = {k: p for k, p in RUNS.items() if os.path.exists(os.path.join(p, "smooth.json"))}
out = ["| run | success | up at 6 s | time to lift median (mean) s | retries | RMS acc rad/s² (pan / lift / elbow / wflex / "
       "wroll / grip) | RMS jerk ×10³ rad/s³ | reversals /s | vel RMS >5 Hz rad/s | lift-phase sag median mrad (pan / lift "
       "/ elbow / wflex / wroll) |", "|---" * 10 + "|"]
for k, p in have.items():
    s = json.load(open(os.path.join(p, "smooth.json")))
    S, U, G = s["smoothness"], s["summary"], s["sag"]
    ttl = "-" if U["time_to_lift_median_s"] is None else f"{U['time_to_lift_median_s']:.2f} ({U['time_to_lift_mean_s']:.2f})"
    sag = " / ".join(f"{G[j]['lift']['median_mrad']:.0f}" if G[j]["lift"].get("n") else "-" for j in J)
    out.append(f"| {k} | {U['successes']} | {U['up_at_6s']} | {ttl} | {U['retries']} | {f(S['rms_acc'])} | "
               f"{f([x / 1e3 for x in S['rms_jerk']], 2)} | {f(S['reversals_per_s'], 2)} | {f(S['osc_rms'], 3)} | {sag} |")
out += ["", "Per-seed agreement (A against B):", "",
        "| A | B | same outcome | both lift | time to lift within ≤ 0.1 s (of both lift) | median |Δ time to lift| s | "
        "seeds that differ (outcome; or time to lift > 0.1 s: A / B s) | retry seeds A ; B |", "|---" * 8 + "|"]
eps = {k: {e["seed"]: e for e in json.load(open(os.path.join(p, "episodes.json")))["episodes"]} for k, p in have.items()}
for a, b in PAIRS:
    if a not in eps or b not in eps:
        continue
    A, B = eps[a], eps[b]
    seeds = sorted(set(A) & set(B))
    same = [s for s in seeds if A[s]["success"] == B[s]["success"]]
    both = [s for s in seeds if A[s]["success"] and B[s]["success"]]
    dt = {s: abs(A[s]["time_to_lift_s"] - B[s]["time_to_lift_s"]) for s in both}
    close = [s for s in both if dt[s] <= 0.1 + 1e-9]
    diff = [f"{s} ({A[s]['success']}/{B[s]['success']})" for s in seeds if s not in same] + \
           [f"{s} ({A[s]['time_to_lift_s']:.2f}/{B[s]['time_to_lift_s']:.2f})" for s in both if s not in close]
    med = sorted(dt.values())[len(dt) // 2] if dt else float("nan")
    ra = [s for s in seeds if A[s]["retries"]]
    rb = [s for s in seeds if B[s]["retries"]]
    out.append(f"| {a} | {b} | {len(same)}/{len(seeds)} | {len(both)} | {len(close)}/{len(both)} | {med:.2f} | "
               f"{', '.join(diff[:20]) + (f' (+{len(diff) - 20} more)' if len(diff) > 20 else '') or '-'} | "
               f"{', '.join(map(str, ra)) or '-'} ; {', '.join(map(str, rb)) or '-'} |")
print("\n".join(out))

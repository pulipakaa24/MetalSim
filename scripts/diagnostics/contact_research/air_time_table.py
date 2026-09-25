"""Table of the air_time_rollout.py results (runs/contact_research/air_time/*.json) next to Isaac's training-log values."""
import glob, json, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
ISAAC_LOG = {"isaac1000": (0.0446, -0.0127), "isaac500": (0.0364, -0.0148), "ours1000": (0.0446, -0.0127), "ours400": (0.0349, -0.0175)}
print("| run | contact | action | feet_air_time | feet_slide (origin vel / COM vel) | Isaac log at that iteration (air / slide) | falls / episodes | air phase median / mean s | contact phase median s | phases < 20 ms air / contact | single / double stance / flight |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for f in sorted(glob.glob(os.path.join(ROOT, "runs/contact_research/air_time/*.json"))):
    n = os.path.basename(f)[:-5]
    if n == "smoke": continue
    r = json.load(open(f)); g = r["gait"]; e = r["Episode_Reward"]; key = n.split("_")[0]
    il = ISAAC_LOG.get(key, ("", ""))
    print(f"| {n} | {r['contact_tuning']} | {'stochastic' if r['stochastic'] else 'mean'}{'' if r['seed'] == 1 else ', seed ' + str(r['seed'])} | {e['feet_air_time']:.4f} | "
          f"{e['feet_slide']:.4f} / {r['feet_slide_com_velocity']:.4f} | {il[0]} / {il[1]} | {r['falls']} / {r['episodes']} | {g['air_phase_median_s']:.3f} / {g['air_phase_mean_s']:.3f} | "
          f"{g['contact_phase_median_s']:.3f} | {g['air_phases_lt_20ms_frac']:.2f} / {g['contact_phases_lt_20ms_frac']:.2f} | {g['single_stance_frac']:.2f} / {g['double_stance_frac']:.2f} / {g['flight_frac']:.3f} |")

"""Learning curves of the G1 flat task at matched iterations: Isaac (rsl_rl on PhysX, runs/parity/isaac_train_g1_flat_terms.txt),
rsl_rl 3.1.2 on MetalSim (runs/g1_flat_rslrl_ppo.log, metalsim.learn.train_g1_rslrl) and PPOWarp (runs/g1_flat_ppo_1500_dt25_fixed.log).
Isaac / rsl_rl report the mean of the last 100 finished episodes (0-based iterations); PPOWarp and the rsl_rl "iter" columns
the mean over episodes finished in that iteration (PPOWarp is 1-based).

    python scripts/diagnostics/g1_learning_curves.py

Rough task (Isaac-Velocity-Rough-G1-v0, rsl_rl on PhysX, runs/parity/isaac_train_g1_rough_terms.txt, vs PPOWarp on
MetalSim, runs/g1_rough_ppowarp_fixed.log): episode length, return, linear tracking and the terrain-level curriculum
metric (Isaac's Curriculum/terrain_levels = ours "terrain it N mean_level", both the mean level over all envs).

    python scripts/diagnostics/g1_learning_curves.py rough [isaac_terms.txt] [ours.log]
"""
import re, json, sys
ITS = [50, 100, 150, 200, 250, 300, 400, 500, 750, 1000, 1500]


def isaac_log(path):
    """Isaac rsl_rl console log (or its grep): per iteration length, return, action std, tracking, terrain level."""
    out = {}; it = None
    pats = {"ret": r"Mean reward: ([-\d.]+)", "len": r"Mean episode length: ([-\d.]+)", "std": r"noise std: ([-\d.]+)",
            "trk": r"Episode_Reward/track_lin_vel_xy_exp: ([-\d.]+)", "trk_yaw": r"Episode_Reward/track_ang_vel_z_exp: ([-\d.]+)",
            "level": r"Curriculum/terrain_levels: ([-\d.]+)", "fall": r"Episode_Termination/base_contact: ([-\d.]+)",
            "itime": r"Iteration time: ([\d.]+)s"}
    for line in open(path, errors="replace"):
        line = re.sub(r"\x1b\[[0-9;]*m", "", line)
        m = re.search(r"(?:ITER|Learning iteration) (\d+)/", line)
        if m: it = int(m.group(1)); out[it] = {}; continue
        if it is None: continue
        for k, pat in pats.items():
            m = re.search(pat, line)
            if m: out[it][k] = float(m.group(1))
    return out


def ppowarp_rough(path):
    out = ppowarp(path)
    for line in open(path):
        m = re.match(r"terrain it +(\d+) mean_level ([-\d.]+)", line)
        if m: out.setdefault(int(m.group(1)), {})["level"] = float(m.group(2))
    return out


def window(d, it, k, half=5):
    """mean of d[i][k] over iterations it-half+1 .. it+half (ours: per-iteration means are noisy)."""
    v = [d[i][k] for i in range(it - half + 1, it + half + 1) if i in d and k in d[i]]
    return sum(v) / len(v) if v else None


def rough_main(isaac_path="runs/parity/isaac_train_g1_rough_terms.txt", ours_path="runs/g1_rough_ppowarp_fixed.log"):
    I = isaac_log(isaac_path); P = ppowarp_rough(ours_path)
    f = lambda v, fmt="{:.1f}": fmt.format(v) if v is not None else "-"
    print(f"Isaac (PhysX + rsl_rl): {isaac_path}, {len(I)} iterations; MetalSim (MuJoCo Warp + PPOWarp): {ours_path}, {len(P)} iterations")
    print("Isaac: mean of the last 100 finished episodes (0-based it); MetalSim: episodes finished in that iteration (1-based it),"
          " point value and mean over it-4..it+5; terrain level: mean over all envs at that iteration on both sides")
    print("| iteration | Isaac length | Isaac return | Isaac terrain level | MetalSim length (point / ±5) | MetalSim return (point / ±5) | MetalSim terrain level |")
    print("|---|---|---|---|---|---|---|")
    for it in ITS:
        ii = min(it, max(I) if I else it); pi = it
        print(f"| {it} | {f(I.get(ii, {}).get('len'))} | {f(I.get(ii, {}).get('ret'), '{:+.2f}')} | {f(I.get(ii, {}).get('level'), '{:.3f}')} | "
              f"{f(P.get(pi, {}).get('len'))} / {f(window(P, pi, 'len'))} | {f(P.get(pi, {}).get('ret'), '{:+.2f}')} / {f(window(P, pi, 'ret'), '{:+.2f}')} | "
              f"{f(P.get(pi, {}).get('level'), '{:.3f}')} |")
    extra = [(it, I[it]) for it in ITS if it in I]
    if extra:
        print("\nIsaac per-iteration detail: it | linear tracking | yaw tracking | falls (base_contact fraction) | action std | iteration time")
        for it, d in extra:
            print(f"{it} | {f(d.get('trk'), '{:.3f}')} | {f(d.get('trk_yaw'), '{:.3f}')} | {f(d.get('fall'), '{:.3f}')} | {f(d.get('std'), '{:.2f}')} | {f(d.get('itime'), '{:.2f}')} s")

def isaac():
    out = {}; it = None
    for line in open("runs/parity/isaac_train_g1_flat_terms.txt"):
        line = re.sub(r"\x1b\[[0-9;]*m", "", line)
        m = re.search(r"ITER (\d+)/", line)
        if m: it = int(m.group(1)); out[it] = {}
        m = re.search(r"Mean reward: ([-\d.]+)", line)
        if m: out[it]["ret"] = float(m.group(1))
        m = re.search(r"Mean episode length: ([-\d.]+)", line)
        if m: out[it]["len"] = float(m.group(1))
        m = re.search(r"noise std: ([-\d.]+)", line)
        if m: out[it]["std"] = float(m.group(1))
        m = re.search(r"track_lin_vel_xy_exp: ([-\d.]+)", line)
        if m: out[it]["trk"] = float(m.group(1))
    return out
def ppowarp(path):
    out = {}
    for line in open(path):
        m = re.match(r"it +(\d+) .*ep_ret +([-\d.]+) ep_len +([-\d.]+).*lr ([\d.e+-]+)", line)
        if m: out[int(m.group(1))] = {"ret": float(m.group(2)), "len": float(m.group(3)), "lr": float(m.group(4))}
    return out
def rslrl(path):
    out = {}
    for line in open(path):
        m = re.match(r"RSLRL it +(\d+) mean_reward +([-\d.na]+) mean_ep_len +([-\d.na]+) \| iter_ep_ret +([-\d.]+) iter_ep_len +([-\d.]+).*lr ([\d.e+-]+) std ([\d.]+)", line)
        if m: out[int(m.group(1))] = {"ret100": float(m.group(2)), "len100": float(m.group(3)), "ret": float(m.group(4)), "len": float(m.group(5)), "lr": float(m.group(6)), "std": float(m.group(7))}
    return out
def terms(path):
    out = {}
    try:
        for line in open(path):
            d = json.loads(line); out[d["it"]] = d.get("Episode_Reward", {})
    except FileNotFoundError: pass
    return out
if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "rough":
    rough_main(*sys.argv[2:4]); sys.exit(0)
I = isaac(); P = ppowarp("runs/g1_flat_ppo_1500_dt25_fixed.log"); R = rslrl("runs/g1_flat_rslrl_ppo.log"); RT = terms("runs/g1_flat_rslrl_ppo.terms.jsonl")
extra = {}
for name, path in [("PPOWarp-fixed", "runs/g1_flat_ppowarp_fixed.log"), ("Newton-PPOWarp-fixed", "runs/g1_flat_newton_ppo_fixed.log")]:
    try: extra[name] = ppowarp(path)
    except FileNotFoundError: pass
try: RF = rslrl("runs/g1_flat_rslrl_ppo_isaacflatcfg.log"); RFT = terms("runs/g1_flat_rslrl_ppo_isaacflatcfg.terms.jsonl")
except FileNotFoundError: RF, RFT = {}, {}
def g(d, it, k, fmt="{:.1f}"):
    v = d.get(it, {}).get(k)
    return fmt.format(v) if v is not None else "-"
print("it | Isaac len/ret/trk/std | rsl_rl@MetalSim len100/ret100 (iter len/ret) trk std lr | PPOWarp-orig len/ret lr | " + " | ".join(f"{k} len/ret lr" for k in extra) + " | rsl_rl flatcfg len100/ret100 trk")
for it in ITS:
    ri = it  # rsl_rl and Isaac are 0-based, PPOWarp 1-based: compare rsl_rl/Isaac it with PPOWarp it (off by one, negligible)
    row = [str(it), f"{g(I,it,'len')}/{g(I,it,'ret','{:.2f}')}/{g(I,it,'trk','{:.3f}')}/{g(I,it,'std','{:.2f}')}",
           f"{g(R,ri,'len100')}/{g(R,ri,'ret100','{:.2f}')} ({g(R,ri,'len')}/{g(R,ri,'ret','{:.2f}')}) {RT.get(ri,{}).get('track_lin_vel_xy_exp','-')} {g(R,ri,'std','{:.3f}')} {g(R,ri,'lr','{:.1e}')}",
           f"{g(P,it,'len')}/{g(P,it,'ret','{:.2f}')} {g(P,it,'lr','{:.1e}')}"]
    for k, d in extra.items(): row.append(f"{g(d,it,'len')}/{g(d,it,'ret','{:.2f}')} {g(d,it,'lr','{:.1e}')}")
    row.append(f"{g(RF,ri,'len100')}/{g(RF,ri,'ret100','{:.2f}')} {RFT.get(ri,{}).get('track_lin_vel_xy_exp','-')}")
    print(" | ".join(row))

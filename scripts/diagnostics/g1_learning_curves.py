"""Learning curves of the G1 flat task at matched iterations: Isaac (rsl_rl on PhysX, runs/parity/isaac_train_g1_flat_terms.txt),
rsl_rl 3.1.2 on MetalSim (runs/g1_flat_rslrl_ppo.log, metalsim.learn.train_g1_rslrl) and PPOWarp (runs/g1_flat_ppo_1500_dt25_fixed.log).
Isaac / rsl_rl report the mean of the last 100 finished episodes (0-based iterations); PPOWarp and the rsl_rl "iter" columns
the mean over episodes finished in that iteration (PPOWarp is 1-based).

    python scripts/diagnostics/g1_learning_curves.py
"""
import re, json, sys
ITS = [50, 100, 150, 200, 300, 500, 750, 1000, 1500]
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
I = isaac(); P = ppowarp("runs/g1_flat_ppo_1500_dt25_fixed.log"); R = rslrl("runs/g1_flat_rslrl_ppo.log"); RT = terms("runs/g1_flat_rslrl_ppo.terms.jsonl")
extra = {}
for name, path in [("PPOWarp-fixed", "runs/g1_flat_ppowarp_fixed_legacytask.log")]:
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

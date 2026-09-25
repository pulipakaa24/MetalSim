"""Learning curves of the G1 flat task at matched iterations: Isaac (rsl_rl on PhysX, runs/parity/isaac_train_g1_flat_terms.txt),
rsl_rl 3.1.2 on MetalSim (runs/g1_flat_rslrl_ppo.log, metalsim.learn.train_g1_rslrl) and PPOWarp (runs/g1_flat_ppo_1500_dt25_fixed.log).
Isaac / rsl_rl report the mean of the last 100 finished episodes (0-based iterations); PPOWarp and the rsl_rl "iter" columns
the mean over episodes finished in that iteration (PPOWarp is 1-based).

    python scripts/diagnostics/g1_learning_curves.py

Rough task (Isaac-Velocity-Rough-G1-v0, rsl_rl on PhysX, runs/parity/isaac_train_g1_rough_terms.txt, vs PPOWarp on
MetalSim, runs/g1_rough_ppowarp_fixed.log): episode length, return, linear tracking and the terrain-level curriculum
metric (Isaac's Curriculum/terrain_levels = ours "terrain it N mean_level", both the mean level over all envs).

    python scripts/diagnostics/g1_learning_curves.py rough [isaac_terms.txt] [ours.log]

Isaac Lab 3.0-EA (rsl_rl 5.4.1 on Isaac's Newton/MuJoCo-Warp and PhysX backends, runs/parity3/isaac/train/
train_<terrain>_<backend>_terms.txt) vs PPOWarp on MetalSim's il3 task (reward_cfg flat_il3 / rough_il3, whose log carries
"terms it N {...}" lines with the per-term episode means): length, return, tracking, terrain level at matched iterations,
every Episode_Reward term at chosen iterations, and the training-loop throughput. Isaac: mean of the last 100 finished
episodes (rsl_rl's rewbuffer / lenbuffer, maxlen 100), Episode_Reward/<term> = per-step mean over the envs reset in that
step of (episode sum of value x dt) / 20 s, averaged over the steps of the iteration that had resets; MetalSim: mean over
the episodes that finished in that iteration (about 100 per iteration at full length with 4096 envs x 24 steps), each
term's episode sum / 20 s averaged over those episodes. Isaac's iterations are 0-based, PPOWarp's 1-based.

    python scripts/diagnostics/g1_learning_curves.py il3 flat runs/il3/train_flat_il3_isaaclab3.log [--at 1000,1499]
"""
import re, json, sys
import numpy as np
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
IL3_TERMS = {  # Isaac 3.0 Episode_Reward/<name> -> MetalSim term (joint deviation is one term here, four in Isaac)
    "track_lin_vel_xy_exp": "track_lin_vel_xy_exp", "track_ang_vel_z_exp": "track_ang_vel_z_exp", "feet_air_time": "feet_air_time",
    "feet_slide": "feet_slide", "joint_deviation (hip+arms+fingers+torso)": "joint_deviation",
    "flat_orientation_l2": "flat_orientation_l2", "action_rate_l2": "action_rate_l2", "termination_penalty": "termination_penalty",
    "lin_vel_z_l2": "lin_vel_z_l2", "ang_vel_xy_l2": "ang_vel_xy_l2", "dof_torques_l2": "dof_torques_l2", "dof_acc_l2": "dof_acc_l2",
    "dof_pos_limits": "dof_pos_limits"}


def isaac3_log(path):
    """Isaac Lab 3.0 rsl_rl 5.4.1 blocks (stage3_train.sh's *_terms.txt): every numeric 'key: value' line per iteration."""
    out = {}; it = None
    for line in open(path, errors="replace"):
        line = re.sub(r"\x1b\[[0-9;]*m", "", line)
        m = re.search(r"Learning iteration (\d+)/", line)
        if m: it = int(m.group(1)); out[it] = {}; continue
        if it is None or ":" not in line: continue
        k, v = line.split(":", 1); v = v.strip().split()
        try: out[it][k.strip()] = float(v[0].rstrip("s"))
        except (ValueError, IndexError): pass
    for d in out.values():
        dev = [d.get(f"Episode_Reward/joint_deviation_{g}") for g in ("hip", "arms", "fingers", "torso")]
        if all(x is not None for x in dev): d["Episode_Reward/joint_deviation (hip+arms+fingers+torso)"] = sum(dev)
    return out


def ppowarp_il3(path):
    out = ppowarp(path)
    for line in open(path):
        m = re.match(r"it +(\d+) steps +(\d+) sps +([\d,]+)", line)
        if m: out.setdefault(int(m.group(1)), {})["sps"] = float(m.group(3).replace(",", ""))
        m = re.match(r"terms it +(\d+) (\{.*\})", line)
        if m: out.setdefault(int(m.group(1)), {})["terms"] = json.loads(m.group(2))
        m = re.match(r"terrain it +(\d+) mean_level ([-\d.]+)", line)
        if m: out.setdefault(int(m.group(1)), {})["level"] = float(m.group(2))
    return out


def _wmean(P, it, fn, half=5):
    v = [fn(P[i]) for i in range(it - half + 1, it + half + 1) if i in P and fn(P[i]) is not None]
    return sum(v) / len(v) if v else None


def il3_main(terrain, ours_path, at=(1000, 1499)):
    backends = ["newton_mjwarp", "isaacsim_physx"] if terrain == "flat" else ["newton_mjwarp"]
    I = {b: isaac3_log(f"runs/parity3/isaac/train/train_{terrain}_{b}_terms.txt") for b in backends}
    P = ppowarp_il3(ours_path)
    f = lambda v, fmt="{:.2f}": fmt.format(v) if v is not None else "-"
    print(f"Isaac Lab 3.0 ({', '.join(backends)}; rsl_rl 5.4.1, last-100-episode means, 0-based it) vs MetalSim {ours_path} "
          f"(PPOWarp, episodes finished in the iteration, 1-based it; ±5-iteration mean in brackets)")
    hdr = " | ".join(f"Isaac {b}: length / return / lin track / yaw track" + (" / level" if terrain != "flat" else "") for b in backends)
    print(f"| iteration | {hdr} | MetalSim: length / return / lin track / yaw track" + (" / level" if terrain != "flat" else "") + " |")
    print("|---|" + "---|" * (len(backends) + 1))
    for it in [50, 100, 150, 200, 250, 300, 400, 500, 750, 1000, 1250, 1499]:
        cells = []
        for b in backends:
            d = I[b].get(it, {})
            c = f"{f(d.get('Mean episode length'), '{:.0f}')} / {f(d.get('Mean reward'), '{:+.1f}')} / {f(d.get('Episode_Reward/track_lin_vel_xy_exp'), '{:.3f}')} / {f(d.get('Episode_Reward/track_ang_vel_z_exp'), '{:.3f}')}"
            if terrain != "flat": c += f" / {f(d.get('Curriculum/terrain_levels'))}"
            cells.append(c)
        pi = it + 1 if it == 1499 else it
        t = lambda k: (lambda r: (r.get("terms") or {}).get(k))
        c = (f"{f(P.get(pi, {}).get('len'), '{:.0f}')} ({f(window(P, pi, 'len'), '{:.0f}')}) / {f(P.get(pi, {}).get('ret'), '{:+.1f}')} "
             f"({f(window(P, pi, 'ret'), '{:+.1f}')}) / {f(_wmean(P, pi, t('track_lin_vel_xy_exp')), '{:.3f}')} / {f(_wmean(P, pi, t('track_ang_vel_z_exp')), '{:.3f}')}")
        if terrain != "flat": c += f" / {f(P.get(pi, {}).get('level'))}"
        print(f"| {it} | " + " | ".join(cells) + f" | {c} |")
    for it in at:
        pi = it + 1 if it == 1499 else it
        print(f"\nPer-term at iteration {it} (Isaac: that iteration's log; MetalSim: ±5-iteration mean of the per-iteration episode means)")
        print("| term | " + " | ".join(f"Isaac {b}" for b in backends) + " | MetalSim |"); print("|---|" + "---|" * (len(backends) + 1))
        for ik, ok in IL3_TERMS.items():
            iv = [I[b].get(it, {}).get(f"Episode_Reward/{ik}") for b in backends]
            ov = _wmean(P, pi, lambda r, ok=ok: (r.get("terms") or {}).get(ok))
            print(f"| {ik} | " + " | ".join(f(v, "{:+.4f}") for v in iv) + f" | {f(ov, '{:+.4f}')} |")
        fi = [I[b].get(it, {}).get("Episode_Termination/base_contact") for b in backends]
        fo = _wmean(P, pi, lambda r: (r.get("terms") or {}).get("base_contact"))
        print(f"| falls (base_contact fraction of episode ends) | " + " | ".join(f(v, "{:.4f}") for v in fi) + f" | {f(fo, '{:.4f}')} |")
    print("\nThroughput (training loop, env-steps/s):")
    for b in backends:
        sps = [d.get("Steps per second") for d in I[b].values() if d.get("Steps per second")]
        tt = sum(d.get("Iteration time", 0.0) for d in I[b].values())
        print(f"  Isaac {b}: median {np.median(sps):,.0f} (L4), total iteration time {tt / 60:.1f} min over {len(I[b])} iterations")
    sps = [d["sps"] for d in P.values() if "sps" in d]
    if sps: print(f"  MetalSim: median {np.median(sps):,.0f} (M4 Max, incl. the monitor), {len(sps)} iterations")


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "il3":
    at = (1000, 1499)
    if "--at" in sys.argv: at = tuple(int(x) for x in sys.argv[sys.argv.index("--at") + 1].split(","))
    il3_main(sys.argv[2], sys.argv[3], at); sys.exit(0)
def isaac_terms_at(it, path="runs/parity/isaac_train_g1_flat_terms.txt"):
    """Isaac's Episode_Reward/<term> values of one (0-based) iteration."""
    out = {}; cur = None
    for line in open(path):
        line = re.sub(r"\x1b\[[0-9;]*m", "", line)
        m = re.search(r"ITER (\d+)/", line)
        if m: cur = int(m.group(1))
        m = re.search(r"Episode_Reward/(\w+): ([-\d.]+)", line)
        if m and cur == it: out[m.group(1)] = float(m.group(2))
    return out


def flatcmp_main(a_path, b_path, at=(100, 150, 200, 300, 500, 1000), terms_a=None, terms_b=None):
    """Isaac vs two PPOWarp flat-config logs (e.g. default contacts vs a contact preset) at matched iterations; ours as
    the mean over +-5 iterations (per-iteration episode means are noisy), Isaac's 100-episode mean. Optional per-term
    comparison at iteration 1000 from g1_reward_terms.py JSONs of the two final checkpoints."""
    I = isaac(); A = ppowarp(a_path); B = ppowarp(b_path)
    print(f"it | Isaac len/ret | A {a_path} len/ret (+-5 it mean) | B {b_path} len/ret | B - A ret")
    for it in at:
        la, ra = window(A, it, "len"), window(A, it, "ret"); lb, rb = window(B, it, "len"), window(B, it, "ret")
        f = lambda x, fmt="{:.2f}": fmt.format(x) if x is not None else "-"
        ii = I.get(it - 1, {})                                          # Isaac 0-based, PPOWarp 1-based
        print(f"{it} | {f(ii.get('len'), '{:.0f}')}/{f(ii.get('ret'))} | {f(la, '{:.0f}')}/{f(ra)} | {f(lb, '{:.0f}')}/{f(rb)} | "
              f"{f(rb - ra) if ra is not None and rb is not None else '-'}")
    if terms_a and terms_b:
        IT = isaac_terms_at(999); TA = json.load(open(terms_a))["Episode_Reward"]; TB = json.load(open(terms_b))["Episode_Reward"]
        IT["joint_deviation_all"] = sum(v for k, v in IT.items() if k.startswith("joint_deviation"))
        print("term | Isaac it 999 (training log) | A | B")
        for k in TA:
            v = IT.get(k)
            print(f"{k} | {v if v is not None else '-'} | {TA[k]:.4f} | {TB[k]:.4f}")


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "flatcmp":
    # python scripts/diagnostics/g1_learning_curves.py flatcmp A.log B.log [--terms A.json B.json]
    ta = tb = None
    if "--terms" in sys.argv: i = sys.argv.index("--terms"); ta, tb = sys.argv[i + 1], sys.argv[i + 2]
    flatcmp_main(sys.argv[2], sys.argv[3], terms_a=ta, terms_b=tb); sys.exit(0)
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

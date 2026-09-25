"""Summarize an Isaac Lab 3.0 rsl_rl training log (the per-iteration blocks extracted by stage3_train.sh into
train_<terrain>_<backend>_terms.txt) at chosen iterations: episode length, return, every Episode_Reward/ term,
terminations, action std and the loop throughput. Values are averaged over +-`--window` iterations.

    python -m metalsim.parity.isaac_side.isaaclab3.summarize_training runs/parity3/isaac/train/train_flat_newton_mjwarp_terms.txt
"""
import argparse, json, re
import numpy as np


def parse(path):
    its, cur = [], None
    for line in open(path, errors="ignore"):
        m = re.search(r"Learning iteration (\d+)/(\d+)", line)
        if m:
            cur = {"iteration": int(m.group(1))}; its.append(cur); continue
        if cur is None or ":" not in line:
            continue
        k, v = line.split(":", 1)
        v = v.strip().split()[0].rstrip("s") if v.strip() else ""
        try:
            cur[k.strip()] = float(v)
        except ValueError:
            pass
    return its


def summarize(its, at=(100, 200, 300, 500, 1000, 1499), window=5):
    out = {}
    for a in at:
        sel = [r for r in its if abs(r["iteration"] - a) <= window]
        if not sel:
            continue
        keys = sorted({k for r in sel for k in r if k != "iteration"})
        out[a] = {k: float(np.mean([r[k] for r in sel if k in r])) for k in keys}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("terms"); ap.add_argument("--window", type=int, default=5); ap.add_argument("--json")
    a = ap.parse_args()
    its = parse(a.terms)
    s = summarize(its, window=a.window)
    wall = sum(r.get("Iteration time", 0.0) for r in its)
    print(f"{len(its)} iterations, total iteration time {wall / 60:.1f} min, mean {wall / max(1, len(its)):.2f} s/it")
    for it, row in s.items():
        short = {k.replace("Episode_Reward/", "R/").replace("Episode_Termination/", "T/"): round(v, 4) for k, v in row.items()}
        print(it, json.dumps(short))
    if a.json:
        json.dump({"file": a.terms, "iterations": len(its), "total_iteration_time_s": wall, "at": s}, open(a.json, "w"), indent=1)

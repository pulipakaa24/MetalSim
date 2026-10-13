"""Franka-Pour learning curves side by side: metalsim's ``log.jsonl`` (``franka_pour_ppo train``) and Isaac Lab's
rsl_rl console log (``isaaclab train``, e.g. ``runs/parity/isaaclab3_pour/train_1024_curve.log``).

    python -m metalsim.learn.franka_pour_curve --mac runs/franka_pour/m414_2048/log.jsonl \
        --isaac runs/parity/isaaclab3_pour/train_1024_curve.log [--every 10] [--json out.json]

Columns per iteration: rsl_rl's mean reward and episode length (last 100 episodes), Isaac's
``Episode_Termination/success`` (the logged success metric: fraction of envs whose last finished episode ended in
transfer success), and for metalsim the success rate of the episodes finished in that iteration.
"""
from __future__ import annotations

import argparse
import json
import re


def read_mac(path):
    out = {}
    for line in open(path):
        r = json.loads(line)
        out[r["it"]] = {"mean_reward": r["mean_reward"], "mean_episode_length": r["mean_episode_length"],
                        "success_term": r["extras"].get("Episode_Termination/success"),
                        "episode_success_rate": r["success_rate"], "env_steps": r["steps"], "fps": r["fps"]}
    return out


def read_isaac(path):
    text = re.sub(r"\x1b\[[0-9;]*m", "", open(path, errors="replace").read())
    out = {}
    for block in re.split(r"Learning iteration ", text)[1:]:
        it = int(block.split("/")[0])
        def num(key):
            m = re.search(rf"{re.escape(key)}:\s*([-0-9.eE+]+)", block)
            return float(m.group(1)) if m else None
        out[it] = {"mean_reward": num("Mean reward"), "mean_episode_length": num("Mean episode length"),
                   "success_term": num("Episode_Termination/success"), "collection_s": num("Collection time"),
                   "learning_s": num("Learning time")}
    return out


def window_stats(runs: list[dict], key: str, window: int, n_its: int):
    """Per ``window``-iteration window: each run's mean of ``key`` over the window, then mean, sample std and count
    over runs (seeds are independent samples: neither side's physics is run-to-run deterministic)."""
    import statistics

    out = []
    for a in range(0, n_its, window):
        per_run = []
        for r in runs:
            vals = [r[i][key] for i in range(a, a + window) if i in r and r[i].get(key) is not None]
            if vals:
                per_run.append(sum(vals) / len(vals))
        if per_run:
            sd = statistics.stdev(per_run) if len(per_run) > 1 else float("nan")
            out.append((a, a + window - 1, sum(per_run) / len(per_run), sd, len(per_run), per_run))
    return out


def seeds_main(argv):
    ap = argparse.ArgumentParser(description="mean +- std over seeds per window")
    ap.add_argument("--mac", nargs="*", default=[])
    ap.add_argument("--isaac", nargs="*", default=[])
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--iterations", type=int, default=60)
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    res = {}
    for side, runs in (("metalsim", [read_mac(p) for p in a.mac]), ("isaac", [read_isaac(p) for p in a.isaac])):
        if runs:
            res[side] = window_stats(runs, "success_term", a.window, a.iterations)
            print(f"{side} ({len(runs)} runs): Episode_Termination/success per {a.window}-iteration window")
            for lo, hi, m, sd, n, per in res[side]:
                print(f"  {lo:3d}-{hi:3d}: {m:.3f} +- {sd:.3f} (n={n})  runs: {', '.join(f'{v:.3f}' for v in per)}")
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


def main():
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "seeds":
        return seeds_main(sys.argv[2:])
    ap = argparse.ArgumentParser()
    ap.add_argument("--mac")
    ap.add_argument("--isaac")
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--json")
    a = ap.parse_args()
    mac = read_mac(a.mac) if a.mac else {}
    isaac = read_isaac(a.isaac) if a.isaac else {}
    its = sorted(set(mac) | set(isaac))
    print(f"{'it':>5} | {'metalsim rew':>12} {'len':>6} {'succ_term':>9} {'ep_succ':>7} | {'Isaac rew':>9} {'len':>6} "
          f"{'succ_term':>9}")
    for it in its:
        if it % a.every and it != its[-1]:
            continue
        m, i = mac.get(it, {}), isaac.get(it, {})
        f = lambda v, w, p: (f"{v:{w}.{p}f}" if isinstance(v, (int, float)) else " " * (w - 1) + "-")  # noqa: E731
        print(f"{it:5d} | {f(m.get('mean_reward'), 12, 3)} {f(m.get('mean_episode_length'), 6, 1)} "
              f"{f(m.get('success_term'), 9, 3)} {f(m.get('episode_success_rate'), 7, 3)} | "
              f"{f(i.get('mean_reward'), 9, 3)} {f(i.get('mean_episode_length'), 6, 1)} {f(i.get('success_term'), 9, 3)}")
    if a.json:
        json.dump({"metalsim": mac, "isaac": isaac}, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()

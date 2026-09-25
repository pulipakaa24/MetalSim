"""Compare two PPO training logs (metalsim.learn.ppo_warp format) at the same iteration counts:
episode return and length (mean over a +-W iteration window, episodes weighted by count), throughput,
and wall time to reach given episode lengths.

usage: python scripts/diagnostics/compare_training_logs.py A.log B.log [--its 1,50,100,...] [--window 5]"""
import re, sys
import numpy as np

PAT = re.compile(r"^it\s+(\d+) steps\s+(\d+) sps\s+([\d,]+) \| ep_ret\s+(-?[\d.]+) ep_len\s+([\d.]+) \(n=(\d+)\)")


def load(path):
    rows = []
    for line in open(path):
        mt = PAT.match(line)
        if mt:
            it, steps, sps, ret, ln, cnt = mt.groups()
            rows.append((int(it), int(steps), float(sps.replace(",", "")), float(ret), float(ln), int(cnt)))
    return np.array(rows, float)


def at(r, it, w):
    sel = r[(r[:, 0] >= it - w) & (r[:, 0] <= it + w) & (r[:, 5] > 0)]
    if not len(sel):
        return float("nan"), float("nan")
    wts = sel[:, 5]
    return float((sel[:, 3] * wts).sum() / wts.sum()), float((sel[:, 4] * wts).sum() / wts.sum())


def main():
    a, b = sys.argv[1], sys.argv[2]
    its = [1, 25, 50, 100, 200, 300, 400, 500, 750, 1000, 1250, 1500]; w = 5
    if "--its" in sys.argv: its = [int(x) for x in sys.argv[sys.argv.index("--its") + 1].split(",")]
    if "--window" in sys.argv: w = int(sys.argv[sys.argv.index("--window") + 1])
    ra, rb = load(a), load(b)
    print(f"A = {a} ({len(ra)} iterations, final sps {ra[-1, 2]:,.0f})\nB = {b} ({len(rb)} iterations, final sps {rb[-1, 2]:,.0f})")
    print(f"| iteration | A ep_ret | B ep_ret | A ep_len | B ep_len |   (mean over it +-{w}, weighted by episode count)")
    print("|---|---|---|---|---|")
    for it in its:
        if it > max(ra[-1, 0], rb[-1, 0]):
            break
        ra_, la = at(ra, it, w); rb_, lb = at(rb, it, w)
        print(f"| {it} | {ra_:.2f} | {rb_:.2f} | {la:.1f} | {lb:.1f} |")
    for target in (100, 200, 400, 600, 800):
        out = []
        for r in (ra, rb):
            hit = r[r[:, 4] >= target]
            out.append(f"it {int(hit[0, 0])} ({hit[0, 1] / hit[0, 2] / 60:.1f} min)" if len(hit) else "not reached")
        print(f"first iteration with ep_len >= {target}: A {out[0]}, B {out[1]}")


if __name__ == "__main__":
    main()

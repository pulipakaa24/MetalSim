"""Old-vs-new rough training curves for the rough re-runs (2026-10-31).

Reads PPOWarp training logs (old: plane-path kernel, 2026-09-25/26; new: runs/rough_rerun_2026-10-31/logs) and prints,
per run: length / return / terrain level at the PARITY milestones (point value, as PARITY §1.5's table, and the
it-4..it+5 window mean, as scripts/diagnostics/g1_learning_curves.py), blow-ups (sum of the per-iteration "blown N"),
anomaly counts and ranges (contact penetration, contact force, joint speed), and capacity flags. A log that holds a
restarted run (old seed 2) is cut to its last start. Throughput is printed as per-iteration seconds for the record
only (14-inch: not a performance figure).

    .venv/bin/python runs/readme_media/scratch/rough-parity/analyze_training.py > runs/readme_media/scratch/rough-parity/training.md
"""
import json, re, statistics, sys

R = "runs/rough_rerun_2026-10-31/logs"
RUNS = [
    ("2.3.2 seed 0", "runs/g1_rough_ppowarp_fixed.log", f"{R}/g1_rough_ppowarp_fixed_s0.log"),
    ("2.3.2 seed 1", "runs/g1_rough_ppowarp_fixed_seed1.log", f"{R}/g1_rough_ppowarp_fixed_s1.log"),
    ("2.3.2 seed 2", "runs/g1_rough_ppowarp_fixed_seed2.log", f"{R}/g1_rough_ppowarp_fixed_s2.log"),
    ("rough_il3 seed 0", "runs/il3/il3fix_rough_s0.log", f"{R}/il3fix_rough_s0.log"),
]
ITS = [100, 150, 200, 300, 500, 750, 1000, 1250, 1499, 1500]
IT_RE = re.compile(r"^it +(\d+) steps +(\d+) sps +([\d,]+) \| ep_ret +([-\d.]+) ep_len +([-\d.]+) \(n=(\d+)\)(.*)$")


def parse(path):
    lines = open(path, errors="replace").read().splitlines()
    starts = [i for i, l in enumerate(lines) if l.startswith("G1 rough PPO:")]
    restarts = len(starts) - 1
    lines = lines[starts[-1]:] if starts else lines
    header = lines[0] if lines else ""
    d, an = {}, {"pen": [], "force": [], "speed": [], "dropped_rows": [], "epa": [], "overflow": []}
    for l in lines:
        m = IT_RE.match(l)
        if m:
            it = int(m.group(1)); rest = m.group(7)
            b = re.search(r"blown (\d+)", rest)
            d.setdefault(it, {}).update(steps=int(m.group(2)), sps=float(m.group(3).replace(",", "")),
                                        ret=float(m.group(4)), len=float(m.group(5)), n=int(m.group(6)),
                                        blown=int(b.group(1)) if b else 0)
            continue
        m = re.match(r"terrain it +(\d+) mean_level ([-\d.]+)", l)
        if m:
            d.setdefault(int(m.group(1)), {})["level"] = float(m.group(2)); continue
        m = re.match(r"terms it +(\d+) (\{.*\})", l)
        if m:
            try:
                d.setdefault(int(m.group(1)), {})["terms"] = json.loads(m.group(2).replace("'", '"'))
            except Exception:
                pass
            continue
        m = re.match(r"\[anomaly\] it (\d+): (.*)", l)
        if m:
            it, msg = int(m.group(1)), m.group(2)
            if (x := re.match(r"contact penetration ([\d.]+) cm", msg)): an["pen"].append((it, float(x.group(1))))
            elif (x := re.match(r"contact force ([\d.]+) N", msg)): an["force"].append((it, float(x.group(1))))
            elif (x := re.match(r"joint speed ([\d.]+) rad/s", msg)): an["speed"].append((it, float(x.group(1))))
            elif "non-finite rows dropped" in msg: an["dropped_rows"].append((it, msg))
            elif "EPA_HORIZON" in msg: an["epa"].append((it, msg))
            elif "overflow" in msg.lower(): an["overflow"].append((it, msg))
    return header, restarts, d, an


def window(d, it, k, half=5):
    v = [d[i][k] for i in range(it - half + 1, it + half + 1) if i in d and k in d[i]]
    return sum(v) / len(v) if v else None


def per_it_seconds(d, a, b):
    """median per-iteration seconds over iterations a..b from the cumulative steps / sps."""
    t = {i: d[i]["steps"] / d[i]["sps"] for i in d if "steps" in d[i] and d[i]["sps"] > 0}
    xs = [t[i] - t[i - 1] for i in range(a, b + 1) if i in t and i - 1 in t]
    return statistics.median(xs) if xs else None


def summary(label, path):
    header, restarts, d, an = parse(path)
    out = {"path": path, "header": header, "restarts_cut": restarts, "iterations": max(d) if d else 0}
    out["milestones"] = {it: {"len": d.get(it, {}).get("len"), "ret": d.get(it, {}).get("ret"),
                              "level": d.get(it, {}).get("level"), "ret_w": window(d, it, "ret"),
                              "len_w": window(d, it, "len")} for it in ITS}
    blown = {i: d[i]["blown"] for i in d if d[i].get("blown")}
    out["blown_total"] = sum(blown.values())
    out["blown_its"] = len(blown)
    out["blown_early_le_30"] = sum(v for i, v in blown.items() if i <= 30)
    out["blown_late_ge_100"] = sum(v for i, v in blown.items() if i >= 100)
    out["blown_first_last"] = (min(blown), max(blown)) if blown else None

    def rng(xs, a, b):
        v = [x for i, x in xs if a <= i <= b]
        return (len(v), min(v), max(v)) if v else (0, None, None)
    out["pen"] = {"all": rng(an["pen"], 1, 10 ** 9), "it1-50": rng(an["pen"], 1, 50), "it1400-1500": rng(an["pen"], 1400, 1500)}
    out["force"] = {"all": rng(an["force"], 1, 10 ** 9), "it1-50": rng(an["force"], 1, 50), "it1400-1500": rng(an["force"], 1400, 1500)}
    out["speed"] = {"all": rng(an["speed"], 1, 10 ** 9)}
    out["dropped_rows"] = an["dropped_rows"]
    out["epa_lines"] = len(an["epa"]); out["overflow_lines"] = an["overflow"][:5]
    out["sec_per_it_median_100_1500"] = per_it_seconds(d, 100, 1500)
    # per-term at 1000 and 1499 (window mean), il3 logs only
    for at in (1000, 1499):
        keys = set()
        for i in range(at - 4, at + 6):
            keys |= set(d.get(i, {}).get("terms", {}).keys())
        if keys:
            out[f"terms_{at}"] = {k: statistics.mean([d[i]["terms"][k] for i in range(at - 4, at + 6)
                                                      if i in d and k in d[i].get("terms", {})]) for k in sorted(keys)}
    return out


def fmt(v, f="{:.2f}"):
    return "-" if v is None else f.format(v)


def main():
    res = {}
    for label, old, new in RUNS:
        res[label] = {"old": summary(label, old), "new": summary(label, new)}
    json.dump(res, open("runs/readme_media/scratch/rough-parity/training.json", "w"), indent=1, default=str)
    for label in res:
        print(f"\n## {label}")
        for side in ("old", "new"):
            s = res[label][side]
            print(f"- {side}: `{s['path']}` ({s['iterations']} iterations; restarts cut: {s['restarts_cut']})")
            print(f"  header: {s['header'][:400]}")
        print("\n| it | old len / ret / level | new len / ret / level | old ret ±5 | new ret ±5 |")
        print("|---|---|---|---|---|")
        for it in ITS:
            o = res[label]["old"]["milestones"][it]; n = res[label]["new"]["milestones"][it]
            print(f"| {it} | {fmt(o['len'], '{:.0f}')} / {fmt(o['ret'], '{:+.2f}')} / {fmt(o['level'], '{:.3f}')} | "
                  f"{fmt(n['len'], '{:.0f}')} / {fmt(n['ret'], '{:+.2f}')} / {fmt(n['level'], '{:.3f}')} | "
                  f"{fmt(o['ret_w'], '{:+.2f}')} | {fmt(n['ret_w'], '{:+.2f}')} |")
        for side in ("old", "new"):
            s = res[label][side]
            print(f"- {side}: blow-ups {s['blown_total']} in {s['blown_its']} iterations (it <= 30: {s['blown_early_le_30']},"
                  f" it >= 100: {s['blown_late_ge_100']}, first/last {s['blown_first_last']}); dropped-row lines {len(s['dropped_rows'])};"
                  f" penetration anomalies {s['pen']}; contact-force anomalies {s['force']}; joint-speed anomalies {s['speed']};"
                  f" EPA_HORIZON lines {s['epa_lines']}; other overflow lines {s['overflow_lines']};"
                  f" median s/it 100-1500 {fmt(s['sec_per_it_median_100_1500'])} (14-inch for new: not a performance figure)")
        for at in (1000, 1499):
            o = res[label]["old"].get(f"terms_{at}"); n = res[label]["new"].get(f"terms_{at}")
            if o or n:
                print(f"\nper-term ±5 mean at {at}:")
                for k in sorted(set(o or {}) | set(n or {})):
                    print(f"  {k}: old {fmt((o or {}).get(k), '{:+.4f}')} new {fmt((n or {}).get(k), '{:+.4f}')}")
    # seed statistics (2.3.2)
    print("\n## 2.3.2 three-seed statistics (point values; sample sd)")
    for it in (300, 500, 1000, 1500):
        for side in ("old", "new"):
            r = [res[f"2.3.2 seed {s}"][side]["milestones"][it]["ret"] for s in range(3)]
            lv = [res[f"2.3.2 seed {s}"][side]["milestones"][it]["level"] for s in range(3)]
            ln = [res[f"2.3.2 seed {s}"][side]["milestones"][it]["len"] for s in range(3)]
            rw = [res[f"2.3.2 seed {s}"][side]["milestones"][it]["ret_w"] for s in range(3)]
            print(f"it {it} {side}: return {', '.join(f'{x:+.2f}' for x in r)} mean {statistics.mean(r):+.2f} sd {statistics.stdev(r):.2f}"
                  f" (range {max(r) - min(r):.2f}); ±5 return mean {statistics.mean(rw):+.2f} sd {statistics.stdev(rw):.2f};"
                  f" level {', '.join(f'{x:.3f}' for x in lv)} mean {statistics.mean(lv):.3f} sd {statistics.stdev(lv):.3f};"
                  f" length {', '.join(f'{x:.0f}' for x in ln)} mean {statistics.mean(ln):.0f}")


if __name__ == "__main__":
    main()

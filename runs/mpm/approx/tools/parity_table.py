"""Key parity_report.py numbers side by side per fast-math mode."""
import re

B = "/Users/aditya/robosim/.claude/worktrees/agent-ae8661f0d09092d4b/runs/mpm/approx/parity_"
rows = {}
for m in ("off", "cuda", "relaxed"):
    for line in open(B + m + ".md"):
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        if len(c) < 5 or not re.match(r"R\d", c[0]):
            continue
        k = c[0]
        if k.startswith("R0"):
            v = f"T1 {c[5]} / {c[9]} x floor, {c[10]}"
        elif k.startswith("R7"):
            v = f"force rms {float(c[6]) * 100:.2f} %, final {c[5]} N (L4 {c[4]}), {c[11]}"
        else:
            body = f", body {float(c[7]) * 100:.2f} %" if c[7] else ""
            v = f"com {float(c[4]) * 100:.3f} %{body}, it {c[9]}, {c[12]}"
        rows.setdefault(k, {})[m] = v
print("| run | off | cuda | relaxed |\n|---|---|---|---|")
for k, v in rows.items():
    print(f"| {k} | {v.get('off', '')} | {v.get('cuda', '')} | {v.get('relaxed', '')} |")

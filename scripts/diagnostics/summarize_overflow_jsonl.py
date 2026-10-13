"""Summarize the overflow flags recorded at every log point by metalsim.learn.monitor.AnomalyMonitor (*.anomalies.jsonl):
the largest number of worlds per flag over the run, capacity flags and solver caps separately.

    python scripts/diagnostics/summarize_overflow_jsonl.py runs/fixfail/train/*.anomalies.jsonl
"""
import json
import sys

CAP = {"NEFC", "NJMAX_NNZ", "BROADPHASE", "NARROWPHASE", "CCD", "HFIELD", "EPA_HORIZON", "CONTACT_MATCH", "NVMAX", "TACTILE"}
for path in sys.argv[1:]:
    rows = [json.loads(line) for line in open(path)]
    cap, solver = {}, {}
    for r in rows:
        for k, v in (r.get("overflow") or {}).items():
            d = cap if k in CAP else solver
            d[k] = max(d.get(k, 0), v)
    print(f"{path}: {len(rows)} log points; capacity flags (max worlds) {cap or 'none'}; solver caps (max worlds) {solver}")

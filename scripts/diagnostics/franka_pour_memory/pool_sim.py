"""Replay a TemporaryStore borrow/redeem trace under different pool policies; report bytes held at the end.
    pool_sim.py EVENTS.json"""
import bisect, collections, json, sys

ev = json.load(open(sys.argv[1]))


def simulate(policy):
    pools = collections.defaultdict(lambda: {"caps": [], "held": 0})  # available capacities (sorted)
    out = {}  # ptr -> (key, cap)
    for op, key, req, ptr, phase in ev:
        p = pools[key]
        if op == "b":
            caps = p["caps"]
            i = bisect.bisect_left(caps, req)
            take = i < len(caps)
            if take and policy.get("max_ratio") and req >= policy.get("min_bytes", 0) and caps[i] > policy["max_ratio"] * req:
                take = False
            if take:
                cap = caps.pop(i)
            else:
                cap = req
                if policy.get("grow") and caps:
                    cap = max(int(caps[-1] * 1.5), req)
                p["held"] += cap
            out[ptr] = (key, cap)
        else:
            rec = out.pop(ptr, None)
            if rec is None:
                continue
            k, cap = rec
            bisect.insort(pools[k]["caps"], cap)
    return {k: v["held"] for k, v in pools.items()}


for name, pol in (("current (best fit, 1.5x growth)", {"grow": True}), ("exact new buffers", {}),
                  ("exact, refuse > 2x for >= 1 MB", {"max_ratio": 2.0, "min_bytes": 1 << 20}),
                  ("exact, refuse > 1.25x for >= 1 MB", {"max_ratio": 1.25, "min_bytes": 1 << 20}),
                  ("grow, refuse > 2x for >= 1 MB", {"grow": True, "max_ratio": 2.0, "min_bytes": 1 << 20})):
    h = simulate(pol)
    print(f"{name:40s} held {sum(h.values())/1e9:7.3f} GB  ", {k: round(v / 1e9, 3) for k, v in h.items()})

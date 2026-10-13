#!/usr/bin/env python3
"""Hold timing jobs only (e.g. while on battery); every other kind runs normally.

    python3 scripts/gpu_hold_timing.py start   # background loop: waiting timing tickets are parked as kind "low"
    python3 scripts/gpu_hold_timing.py stop    # restore them as "timing" and end the hold
    python3 scripts/gpu_hold_timing.py fair    # background loop: a timing job waiting > FAIR_S (20 min) goes next
                                               # (check tickets are parked until it is granted, then restored)

Parked tickets keep their queue time; a placeholder "low" ticket at the front of the low class (t=0, owned by this
loop) keeps them from being granted even when nothing else waits. Waiters decide from the ticket files, so this works
for waiters that are already blocked in an older gpu_lock.py as well.
"""
import json, os, signal, subprocess, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Q = os.path.join(ROOT, "runs", "gpu_queue"); PIDF = os.path.join(ROOT, "runs", ".gpu_hold_timing.pid")


def _rekind(src, dst):
    for f in os.listdir(Q):
        p = os.path.join(Q, f)
        try: t = json.load(open(p))
        except Exception: continue
        if src == "timing" and t.get("kind") == "timing":
            t["kind"], t["held_from"] = "low", "timing"
        elif src == "held" and t.get("held_from") == "timing":
            t["kind"] = "timing"; t.pop("held_from")
        else: continue
        tmp = p + ".tmp"; json.dump(t, open(tmp, "w")); os.replace(tmp, p)


FAIR_S = float(os.environ.get("GPU_FAIR_S", "1200"))
FPIDF = os.path.join(ROOT, "runs", ".gpu_fair.pid")


def _tickets():
    out = []
    for f in os.listdir(Q):
        p = os.path.join(Q, f)
        try: out.append((p, json.load(open(p))))
        except Exception: pass
    return out


def _set(p, t):
    tmp = p + ".tmp"; json.dump(t, open(tmp, "w")); os.replace(tmp, p)


def fair():
    def end(*_):
        for p, t in _tickets():
            if t.get("held_from") == "check": t["kind"] = "check"; t.pop("held_from"); _set(p, t)
        try: os.remove(FPIDF)
        except OSError: pass
        sys.exit(0)
    signal.signal(signal.SIGTERM, end); signal.signal(signal.SIGINT, end)
    while True:
        ts = _tickets()
        starved = [t for _, t in ts if t.get("kind") == "timing" and t.get("t", 0) > 0 and time.time() - t["t"] > FAIR_S]
        for p, t in ts:
            if starved and t.get("kind") == "check":
                t["kind"], t["held_from"] = "low", "check"; _set(p, t)
            elif not starved and t.get("held_from") == "check":
                t["kind"] = "check"; t.pop("held_from"); _set(p, t)
        time.sleep(3)


def loop():
    me = os.getpid(); ph = os.path.join(Q, f"0.000_3_HOLD_TIMING_{me}.json")
    def end(*_):
        _rekind("held", None)
        for p in (ph, PIDF):
            try: os.remove(p)
            except OSError: pass
        sys.exit(0)
    signal.signal(signal.SIGTERM, end); signal.signal(signal.SIGINT, end)
    while True:
        if not os.path.exists(ph):
            json.dump({"name": "HOLD_TIMING", "kind": "low", "minutes": 99999, "pid": me, "t": 0.0,
                       "cmd": "timing jobs held (battery)", "cwd": ROOT}, open(ph, "w"))
        _rekind("timing", None)
        time.sleep(3)


if __name__ == "__main__":
    os.makedirs(Q, exist_ok=True)
    if sys.argv[1:] == ["start"]:
        if os.path.exists(PIDF): print("already holding"); sys.exit(0)
        p = subprocess.Popen([sys.executable, __file__, "_loop"], start_new_session=True,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        open(PIDF, "w").write(str(p.pid)); print(f"timing jobs held (pid {p.pid})")
    elif sys.argv[1:] == ["stop"]:
        try: os.kill(int(open(PIDF).read()), signal.SIGTERM); print("timing hold released")
        except (OSError, ValueError): _rekind("held", None); print("no live hold; restored parked tickets")
    elif sys.argv[1:] == ["_loop"]:
        loop()
    elif sys.argv[1:] == ["fair"]:
        if os.path.exists(FPIDF): print("fair scheduler already running"); sys.exit(0)
        p = subprocess.Popen([sys.executable, __file__, "_fair"], start_new_session=True,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        open(FPIDF, "w").write(str(p.pid)); print(f"fair scheduler running (pid {p.pid})")
    elif sys.argv[1:] == ["unfair"]:
        try: os.kill(int(open(FPIDF).read()), signal.SIGTERM); print("fair scheduler stopped")
        except (OSError, ValueError): print("not running")
    elif sys.argv[1:] == ["_fair"]:
        fair()

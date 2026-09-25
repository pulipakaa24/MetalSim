#!/usr/bin/env python3
"""GPU arbitration for one shared Apple GPU: a priority queue with atomic acquisition.

Classes (granted in this order when the GPU frees up; FIFO within a class):
  timing  (0)  short measurements that need an idle GPU (benchmarks, profiles); <= 15 min
  render  (1)  renders / short rollouts
  train   (2)  long training runs
A holder is recorded in runs/.gpu_lock (text: name, kind, pid, start) and runs/.gpu_lock.d/ (the atomic
token); a holder whose pid is gone is released automatically. Scripts that only check the file still
interoperate. Waiting jobs register tickets in runs/gpu_queue/.

    python scripts/gpu_lock.py acquire NAME --kind timing --minutes 10   # blocks until granted
    python scripts/gpu_lock.py release NAME
    python scripts/gpu_lock.py status
    scripts/gpu_run.sh NAME KIND MINUTES -- command args...               # acquire, run, always release
"""
import argparse, json, os, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCK = os.path.join(ROOT, "runs", ".gpu_lock"); LOCKD = LOCK + ".d"; Q = os.path.join(ROOT, "runs", "gpu_queue")
PRIO = {"timing": 0, "render": 1, "train": 2}


def alive(pid):
    try: os.kill(pid, 0); return True
    except OSError: return False


def holder():
    if not os.path.exists(LOCK) and not os.path.isdir(LOCKD): return None
    try: h = json.load(open(LOCK))
    except Exception: h = {"name": open(LOCK).read().strip()[:80] if os.path.exists(LOCK) else "?", "pid": None}
    if h.get("pid") and not alive(h["pid"]):
        _release(force=True); return None
    return h


def _release(force=False):
    for p in (LOCK,):
        if os.path.exists(p): os.remove(p)
    if os.path.isdir(LOCKD): os.rmdir(LOCKD)


def tickets():
    os.makedirs(Q, exist_ok=True); out = []
    for f in sorted(os.listdir(Q)):
        try: t = json.load(open(os.path.join(Q, f)))
        except Exception: continue
        if t.get("pid") and not alive(t["pid"]): os.remove(os.path.join(Q, f)); continue
        out.append((PRIO.get(t["kind"], 2), t["t"], f, t))
    out.sort(); return out


def acquire(name, kind, minutes, pid):
    os.makedirs(Q, exist_ok=True)
    tf = os.path.join(Q, f"{time.time():.3f}_{PRIO[kind]}_{name}.json")
    json.dump({"name": name, "kind": kind, "minutes": minutes, "pid": pid, "t": time.time()}, open(tf, "w"))
    while True:
        h = holder()
        if h is None:
            q = tickets()
            if q and q[0][2] == os.path.basename(tf):
                try:
                    os.mkdir(LOCKD)                                   # atomic
                except FileExistsError:
                    time.sleep(2); continue
                json.dump({"name": name, "kind": kind, "minutes": minutes, "pid": pid, "start": time.time()}, open(LOCK, "w"))
                os.remove(tf); return
        time.sleep(5)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["acquire", "release", "status"]); ap.add_argument("name", nargs="?")
    ap.add_argument("--kind", default="train", choices=list(PRIO)); ap.add_argument("--minutes", type=float, default=30); ap.add_argument("--pid", type=int, default=os.getppid())
    a = ap.parse_args()
    if a.cmd == "acquire": acquire(a.name, a.kind, a.minutes, a.pid)
    elif a.cmd == "release":
        h = holder()
        if h and a.name and h.get("name") not in (a.name, None): print(f"lock held by {h.get('name')}, not {a.name}; not released"); sys.exit(1)
        _release()
    else:
        h = holder(); print("holder:", json.dumps(h) if h else "none")
        for pr, t, f, tk in tickets(): print(f"  waiting [{tk['kind']}] {tk['name']} est {tk['minutes']} min, queued {time.time() - t:.0f} s ago")


if __name__ == "__main__":
    main()

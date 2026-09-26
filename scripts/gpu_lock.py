#!/usr/bin/env python3
"""GPU arbitration for one shared Apple GPU: a priority queue with atomic acquisition.

Classes (granted in this order when the GPU frees up; FIFO within a class):
  timing  (0)  short measurements that need an idle GPU (benchmarks, profiles); <= 15 min
  render  (1)  renders / short rollouts
  train   (2)  long training runs
  low     (3)  background optimisation work; granted only when nothing else waits
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
HIST = os.path.join(ROOT, "runs", "gpu_history.jsonl")      # one line per grant and per release (the dashboard reads it)


def _hist(ev, **kw):
    try:
        with open(HIST, "a") as f: f.write(json.dumps(dict(ev=ev, t=time.time(), **kw)) + "\n")
    except OSError: pass
PRIO = {"timing": 0, "render": 1, "train": 2, "low": 3}


def alive(pid):
    try: os.kill(pid, 0); return True
    except ProcessLookupError: return False
    except PermissionError: return True      # the process exists; a sandbox denied the signal (a sandboxed status call once deleted every ticket)
    except OSError: return True


def holder():
    if not os.path.exists(LOCK) and not os.path.isdir(LOCKD): return None
    try: h = json.load(open(LOCK))
    except Exception: h = {"name": open(LOCK).read().strip()[:80] if os.path.exists(LOCK) else "?", "pid": None}
    if h.get("pid") and not alive(h["pid"]):
        _release(force=True); return None
    return h


def _release(force=False, rc=None):
    if os.path.exists(LOCK):
        try: h = json.load(open(LOCK)); _hist("release", name=h.get("name"), kind=h.get("kind"), start=h.get("start"), rc=rc, forced=force)
        except Exception: pass
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


def acquire(name, kind, minutes, pid, cmd=None, front=False):
    os.makedirs(Q, exist_ok=True)
    t0 = 0.0 if front else time.time()                 # front: ahead of every waiter of its class (tickets sort by class, then t)
    tf = os.path.join(Q, f"{t0:.3f}_{PRIO[kind]}_{name}_{pid}.json")   # pid in the name: two tickets of one name can never be confused
    json.dump({"name": name, "kind": kind, "minutes": minutes, "pid": pid, "t": t0, "cmd": cmd, "cwd": os.getcwd()}, open(tf, "w"))
    ticket = {"name": name, "kind": kind, "minutes": minutes, "pid": pid, "t": t0, "cmd": cmd, "cwd": os.getcwd()}
    while True:
        if not os.path.exists(tf): json.dump(ticket, open(tf, "w"))      # self-heal: a ticket removed by another process is re-created
        h = holder()
        if h is None:
            q = tickets()
            if q and q[0][2] == os.path.basename(tf):
                try:
                    os.mkdir(LOCKD)                                   # atomic
                except FileExistsError:
                    time.sleep(2); continue
                json.dump({"name": name, "kind": kind, "minutes": minutes, "pid": pid, "start": time.time(), "queued": os.path.getmtime(tf), "cmd": cmd, "cwd": os.getcwd()}, open(LOCK, "w"))
                _hist("grant", name=name, kind=kind, minutes=minutes, cmd=cmd)
                os.remove(tf); return
        time.sleep(5)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["acquire", "release", "status", "setpid"]); ap.add_argument("name", nargs="?")
    ap.add_argument("--kind", default="train", choices=list(PRIO)); ap.add_argument("--minutes", type=float, default=30); ap.add_argument("--pid", type=int, default=os.getppid())
    ap.add_argument("--cmd", dest="job_cmd", default=None, help="the job's command line, recorded for the dashboard"); ap.add_argument("--rc", type=int, default=None, help="exit code, recorded on release")
    ap.add_argument("--json", action="store_true"); ap.add_argument("--front", action="store_true", help="queue ahead of every waiter of the same class (the pause placeholder)")
    a = ap.parse_args()
    if a.cmd == "acquire": acquire(a.name, a.kind, a.minutes, a.pid, a.job_cmd, a.front)
    elif a.cmd == "setpid":                       # the wrapper registers the real job process once spawned
        h = holder()
        if h and h.get("name") == a.name: h["pid"] = a.pid; json.dump(h, open(LOCK, "w"))
    elif a.cmd == "release":
        h = holder()
        if h and a.name and h.get("name") not in (a.name, None): print(f"lock held by {h.get('name')}, not {a.name}; not released"); sys.exit(1)
        if h and a.pid != os.getppid() and h.get("pid") and h["pid"] != a.pid:   # a late release from an older job of the same name must not free its successor
            print(f"lock held by pid {h['pid']}, release asked for pid {a.pid}; not released"); sys.exit(1)
        if h is None and a.rc is not None:            # a waiter already freed the dead holder; still record the exit code
            _hist("exit", name=a.name, rc=a.rc); return
        _release(rc=a.rc)
    elif a.json:
        print(json.dumps({"holder": holder(), "waiting": [tk for _, _, _, tk in tickets()], "now": time.time()}))
    else:
        h = holder(); print("holder:", json.dumps(h) if h else "none")
        for pr, t, f, tk in tickets(): print(f"  waiting [{tk['kind']}] {tk['name']} est {tk['minutes']} min, queued {time.time() - t:.0f} s ago")


if __name__ == "__main__":
    main()

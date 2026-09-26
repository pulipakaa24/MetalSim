#!/usr/bin/env python3
"""Live view of the GPU job queue with per-job progress. Nothing here is per job: every job that goes
through scripts/gpu_run.sh (or gpu_lock.py) appears on its own, and progress is read from whatever the
job writes, so new kinds of jobs need no changes.

How a job's progress is found (in this order, all automatic):
  1. its log: every regular file the job's process tree holds open for writing (lsof), newest first;
     the ticket's recorded command line is searched for log paths as a fallback
  2. a counter in the log tail: "it 123", "iter 123", "iteration 123", "epoch 12", "step 123",
     "123/1500" (also tqdm's "123/1500 ["), "progress 45 %" or a tqdm "45%|"; the total comes from the same line ("it 12/500"),
     from the command line (--iters/--iterations/--max_iterations/--epochs/--steps/--frames/--n, or a
     bare integer after a run script's name), or from the largest counter the job has reached so far
  3. otherwise elapsed time against the minutes the job asked for when it queued
The rate of the counter is measured between refreshes (state kept in runs/.gpu_dashboard_state.json),
which gives the ETA. The last log line is always shown, so a job with a format nobody anticipated is
still readable. Finished jobs come from runs/gpu_history.jsonl (written by gpu_lock.py).

    python3 scripts/gpu_dashboard.py               # print once
    python3 scripts/gpu_dashboard.py --watch       # refresh in the terminal every 10 s
    python3 scripts/gpu_dashboard.py --serve 8765  # http://localhost:8765 (auto-refresh page + /json)
    python3 scripts/gpu_dashboard.py --html runs/gpu_dashboard.html
"""
from __future__ import annotations
import argparse, html, json, os, re, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import gpu_lock  # noqa: E402

STATE = os.path.join(ROOT, "runs", ".gpu_dashboard_state.json")
HIST = os.path.join(ROOT, "runs", "gpu_history.jsonl")
LOG_EXT = (".log", ".txt", ".out", ".stdout", ".jsonl", ".csv")


# ---------------------------------------------------------------- processes and logs
def _ps():
    out = subprocess.run(["ps", "-axo", "pid=,ppid=,etime=,command="], capture_output=True, text=True).stdout
    procs = {}
    for line in out.splitlines():
        parts = line.split(None, 3)
        if len(parts) < 3: continue
        procs[int(parts[0])] = (int(parts[1]), parts[2], parts[3] if len(parts) > 3 else "")
    return procs


def tree(pid, procs):
    if pid not in procs: return []
    out, todo = [], [pid]
    while todo:
        p = todo.pop(); out.append(p)
        todo += [c for c, (pp, _, _) in procs.items() if pp == p]
    return out


def open_logs(pids):
    """Regular files under this checkout that the processes hold open for writing, newest mtime first."""
    if not pids: return []
    r = subprocess.run(["lsof", "-a", "-p", ",".join(map(str, pids)), "-Fna", "-d", "^cwd,^txt,^rtd,^mem"],
                       capture_output=True, text=True).stdout
    files, mode = set(), None
    for line in r.splitlines():
        if line[:1] == "a": mode = line[1:]
        elif line[:1] == "n" and mode in ("w", "u"):
            f = line[1:]
            if f.startswith(ROOT) and os.path.isfile(f) and not f.endswith((".pt", ".npz", ".mp4", ".png", ".json")):
                files.add(f)
    return sorted(files, key=lambda f: -os.path.getmtime(f))


def cmd_logs(cmd):
    return [t.strip("'\"") for t in re.findall(r"\S+", cmd or "") if t.strip("'\"").endswith(LOG_EXT) and os.path.isfile(t.strip("'\""))]


def tail(path, n=40, nbytes=16000):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2); size = f.tell(); f.seek(max(0, size - nbytes))
            txt = f.read().decode("utf-8", "replace")
        lines = [l.rstrip() for l in txt.replace("\r", "\n").splitlines() if l.strip()]
        return lines[-n:]
    except OSError:
        return []


# ---------------------------------------------------------------- progress parsing
CUR = re.compile(r"(?:^|[\s|\[])(?:it|iter|iteration|epoch|step|steps|frame|episode|update)\s*[:=]?\s*(\d+)(?:\s*/\s*(\d+))?", re.I)
FRAC = re.compile(r"(?<![\d.])(\d+)\s*/\s*(\d+)(?:\s*\[|\s|$)")
PCT = re.compile(r"(?:(?:progress|done|complete|completed|finished|rendered|processed)\D{0,12}|\|\s*)(\d{1,3}(?:\.\d+)?)\s*%|(\d{1,3}(?:\.\d+)?)%\|", re.I)   # tqdm bars or "progress 37 %", not "cpu > 40 %"
TOTAL_FLAGS = re.compile(r"--(?:iters?|iterations|max[_-]iter(?:ations)?|num[_-]iter(?:ations)?|epochs|steps|max[_-]steps|frames|n|num[_-]envs?_steps|updates)[= ](\d+)")


def parse_progress(lines, cmd):
    """Return (current, total, percent, counter_name, source) from the newest informative line."""
    total = None
    m = TOTAL_FLAGS.search(cmd or "")
    if m: total = int(m.group(1))
    elif cmd:                                      # run scripts often take the iteration count as a bare argument
        ints = [int(t) for t in re.findall(r"(?<![\w./-])(\d{2,7})(?![\w./-])", cmd)]
        if ints: total = max(ints)
    for line in reversed(lines):
        if line.startswith("[anomaly]") or line.startswith("[gpu_run]"): continue
        m = CUR.search(line)
        if m:
            cur = int(m.group(1)); tot = int(m.group(2)) if m.group(2) else total
            pct = 100.0 * cur / tot if tot else None
            return cur, tot, pct, m.group(0).strip().split()[0].strip("[|").lower(), "counter"
        m = FRAC.search(line)
        if m and int(m.group(2)) >= int(m.group(1)) > 0:
            cur, tot = int(m.group(1)), int(m.group(2)); return cur, tot, 100.0 * cur / tot, "fraction", "fraction"
        m = PCT.search(line)
        if m:
            v = float(m.group(1) or m.group(2))
            if v <= 100: return None, None, v, "percent", "percent"
    return None, None, None, None, None


def load_state():
    try: return json.load(open(STATE))
    except Exception: return {}


def save_state(st):
    try: json.dump(st, open(STATE, "w"))
    except OSError: pass


def rate_and_eta(st, key, cur, total, now):
    """Counter rate over the observed window of this job (first sample vs now), ETA to total."""
    if cur is None: return None, None
    s = st.setdefault(key, {})
    if "t0" not in s or cur < s.get("last", 0):       # new job under an old name, or a restart
        s.update(t0=now, c0=cur, last=cur, tl=now); return None, None
    s["last"] = cur; s["tl"] = now
    dt = now - s["t0"]; dc = cur - s["c0"]
    if dt < 20 or dc <= 0: return None, None
    rate = dc / dt
    eta = (total - cur) / rate if total and rate > 0 else None
    return rate, eta


# ---------------------------------------------------------------- assembling the view
def gpu_info():
    util = None
    try: util = int(subprocess.run([os.path.join(ROOT, "scripts", "gpu_util.sh")], capture_output=True, text=True, timeout=5).stdout.strip() or -1)
    except Exception: pass
    top = []
    try:
        r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "gpu_top.py"), "--interval", "1", "--min", "1"], capture_output=True, text=True, timeout=10).stdout
        for line in r.splitlines():
            if re.match(r"^\s*pid\s+\d+", line): top.append(re.sub(r"^\s*pid\s+", "", line.strip()))
    except Exception: pass
    return util, top[:6]


def history(n=12):
    grants, done = {}, []
    try:
        for line in open(HIST):
            try: e = json.loads(line)
            except Exception: continue
            if e["ev"] == "grant": grants[e["name"]] = e
            elif e["ev"] == "release":
                g = grants.pop(e.get("name"), {})
                done.append(dict(name=e.get("name"), kind=e.get("kind") or g.get("kind"), start=e.get("start") or g.get("t"), end=e["t"],
                                 rc=e.get("rc"), forced=e.get("forced"), cmd=g.get("cmd")))
            elif e["ev"] == "exit":                   # the wrapper's exit code, logged after a waiter freed the dead holder
                for d in reversed(done):
                    if d["name"] == e.get("name") and d.get("rc") is None: d["rc"] = e.get("rc"); break
    except OSError: pass
    return done[-n:][::-1]


def job_view(rec, procs, st, now, running):
    pid = rec.get("pid"); pids = tree(pid, procs) if pid else []
    cmd = rec.get("cmd") or (procs.get(pid, (0, "", ""))[2] if pid in procs else "")
    cmd = re.sub(r"^.*gpu_run\.sh\s+\S+\s+\S+\s+\S+\s+(--\s+)?", "", cmd) if cmd else ""
    logs = open_logs(pids) if running else []
    for f in cmd_logs(cmd):
        if f not in logs: logs.append(f)
    view = dict(name=rec["name"], kind=rec["kind"], minutes=rec.get("minutes"), pid=pid, cmd=cmd, logs=[os.path.relpath(f, ROOT) for f in logs],
                alive=bool(pids), state="running" if running else "waiting")
    if running:
        start = rec.get("start", now); view["elapsed"] = now - start
        lines = tail(logs[0]) if logs else []
        cur, tot, pct, name, src = parse_progress(lines, cmd)
        rate, eta = rate_and_eta(st, f"{rec['name']}:{int(start)}", cur, tot, now)
        if pct is None and rec.get("minutes"):
            pct = min(99.0, 100.0 * view["elapsed"] / (60.0 * rec["minutes"])); src = "elapsed / estimate"
            eta = max(0.0, 60.0 * rec["minutes"] - view["elapsed"])
        view.update(current=cur, total=tot, percent=pct, counter=name, progress_source=src, rate=rate, eta=eta,
                    last_line=(lines[-1][:200] if lines else ""), last_lines=lines[-4:])
        # busy check: the newest log should be moving
        view["log_age"] = (now - os.path.getmtime(logs[0])) if logs else None
    else:
        view["queued_for"] = now - rec.get("t", now)
    return view


def snapshot():
    now = time.time(); procs = _ps(); st = load_state()
    h = gpu_lock.holder(); waiting = [tk for _, _, _, tk in gpu_lock.tickets()]
    jobs = []
    if h: jobs.append(job_view(h, procs, st, now, True))
    jobs += [job_view(tk, procs, st, now, False) for tk in waiting]
    # drop state of jobs no longer present
    live = {f"{j['name']}:{int(h['start'])}" for j in jobs if j["state"] == "running"} if h else set()
    for k in list(st):
        if k not in live: st.pop(k)
    save_state(st)
    util, top = gpu_info()
    return dict(now=now, gpu_util=util, gpu_top=top, jobs=jobs, finished=history(), watchdog=os.path.exists(os.path.join(ROOT, "runs", ".gpu_watchdog")))


# ---------------------------------------------------------------- renderers
def fmt_dur(s):
    if s is None: return "–"
    s = int(s); return f"{s // 3600}h{(s % 3600) // 60:02d}m" if s >= 3600 else (f"{s // 60}m{s % 60:02d}s" if s >= 60 else f"{s}s")


def bar(pct, width=24):
    if pct is None: return "[" + "?" * width + "]"
    n = int(round(width * min(100.0, max(0.0, pct)) / 100)); return "[" + "#" * n + "." * (width - n) + "]"


def render_text(snap):
    out = [f"GPU queue  {time.strftime('%H:%M:%S', time.localtime(snap['now']))}   device utilisation {snap['gpu_util']}%"]
    if any(j["name"] == "PAUSE" and j["state"] == "running" for j in snap["jobs"]): out.append("  QUEUE PAUSED (scripts/gpu_resume.sh to continue); waiting jobs stay queued")
    for j in snap["jobs"]:
        if j["state"] == "running":
            prog = f"{j['current']}/{j['total']}" if j.get("total") else (f"{j['current']}" if j.get("current") is not None else "")
            out.append(f"  RUNNING [{j['kind']}] {j['name']}  {bar(j.get('percent'))} {j.get('percent') or 0:5.1f}%  {prog}  "
                       f"elapsed {fmt_dur(j['elapsed'])}  eta {fmt_dur(j.get('eta'))}  ({j.get('progress_source') or 'no progress signal'})")
            if j.get("logs"): out.append(f"      log {j['logs'][0]}  (last write {fmt_dur(j.get('log_age'))} ago)")
            if j.get("last_line"): out.append(f"      > {j['last_line'][:150]}")
        else:
            out.append(f"  waiting [{j['kind']}] {j['name']}  est {j['minutes']} min  queued {fmt_dur(j['queued_for'])} ago" + ("" if j["alive"] else "  (process gone)"))
    if not snap["jobs"]: out.append("  idle: nothing running or waiting")
    if snap["gpu_top"]: out.append("  GPU by process: " + " | ".join(snap["gpu_top"][:4]))
    if snap["finished"]:
        out.append("  finished:")
        for f in snap["finished"][:8]:
            out.append(f"    {time.strftime('%H:%M', time.localtime(f['end']))} [{f.get('kind')}] {f['name']}  {fmt_dur((f['end'] - f['start']) if f.get('start') else None)}  rc {f.get('rc')}")
    return "\n".join(out)


def render_html(snap, refresh=10):
    def e(x): return html.escape(str(x))
    rows = []
    for j in snap["jobs"]:
        if j["state"] == "running":
            pct = j.get("percent"); prog = f"{j['current']} / {j['total']}" if j.get("total") else (f"{j['current']}" if j.get("current") is not None else "")
            rows.append(f"""<div class="job run"><div class="head"><span class="kind">{e(j['kind'])}</span><b>{e(j['name'])}</b>
              <span class="pct">{'' if pct is None else f'{pct:.1f}%'}</span></div>
              <div class="bar"><div style="width:{0 if pct is None else min(100, pct):.1f}%"></div></div>
              <div class="meta">{e(prog)} · elapsed {fmt_dur(j['elapsed'])} · eta {fmt_dur(j.get('eta'))} · {e(j.get('progress_source') or 'no progress signal')}
              {('· rate %.2f/s' % j['rate']) if j.get('rate') else ''}</div>
              <div class="meta">log: {e(j['logs'][0]) if j.get('logs') else '(none found)'}{f" · last write {fmt_dur(j.get('log_age'))} ago" if j.get('log_age') is not None else ''}</div>
              <pre>{e(chr(10).join(j.get('last_lines') or []))}</pre>
              <div class="cmd">{e(j['cmd'][:300])}</div></div>""")
        else:
            rows.append(f"""<div class="job wait"><div class="head"><span class="kind">{e(j['kind'])}</span><b>{e(j['name'])}</b>
              <span class="pct">est {e(j['minutes'])} min · queued {fmt_dur(j['queued_for'])} ago{'' if j['alive'] else ' · process gone'}</span></div>
              <div class="cmd">{e(j['cmd'][:300])}</div></div>""")
    fin = "".join(f"<tr><td>{time.strftime('%H:%M', time.localtime(f['end']))}</td><td>{e(f.get('kind'))}</td><td>{e(f['name'])}</td>"
                  f"<td>{fmt_dur((f['end'] - f['start']) if f.get('start') else None)}</td><td>{'' if f.get('rc') is None else e(f['rc'])}{' forced' if f.get('forced') else ''}</td></tr>"
                  for f in snap["finished"])
    top = "<br>".join(e(t) for t in snap["gpu_top"][:5])
    return f"""<!doctype html><html><head><meta charset="utf-8"><meta http-equiv="refresh" content="{refresh}"><title>GPU queue</title>
<style>:root{{--bg:#101318;--fg:#e6e6e6;--mut:#9aa4b2;--run:#2e7d32;--wait:#3b4252;--card:#181c24}}
@media (prefers-color-scheme: light){{:root{{--bg:#f6f7f9;--fg:#111;--mut:#555;--card:#fff;--wait:#dfe3ea}}}}
body{{background:var(--bg);color:var(--fg);font:14px/1.4 -apple-system,Helvetica,sans-serif;margin:0;padding:16px;max-width:980px}}
h1{{font-size:18px;margin:0 0 4px}} .sub{{color:var(--mut);margin-bottom:12px}}
.job{{background:var(--card);border-radius:8px;padding:10px 12px;margin:8px 0;border-left:4px solid var(--wait)}} .job.run{{border-left-color:var(--run)}}
.head{{display:flex;gap:10px;align-items:baseline}} .kind{{font-size:11px;text-transform:uppercase;color:var(--mut)}} .pct{{margin-left:auto;color:var(--mut)}}
.bar{{height:8px;background:var(--wait);border-radius:4px;margin:6px 0;overflow:hidden}} .bar div{{height:100%;background:var(--run)}}
.meta,.cmd{{color:var(--mut);font-size:12px;word-break:break-all}} pre{{font-size:11px;white-space:pre-wrap;margin:6px 0;color:var(--fg);opacity:.85}}
table{{border-collapse:collapse;font-size:12px}} td{{padding:2px 8px;border-bottom:1px solid var(--wait)}}</style></head><body>
<h1>GPU queue</h1>{'<div class="job run" style="border-left-color:#c62828"><b>QUEUE PAUSED</b> · scripts/gpu_resume.sh to continue · waiting jobs stay queued</div>' if any(j["name"] == "PAUSE" and j["state"] == "running" for j in snap["jobs"]) else ''}<div class="sub">{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(snap['now']))} · device utilisation {snap['gpu_util']}% · refreshes every {refresh} s</div>
{''.join(rows) or '<div class="job">idle: nothing running or waiting</div>'}
<h1>GPU by process</h1><div class="meta">{top or '–'}</div>
<h1>Finished</h1><table>{fin or '<tr><td>none recorded yet</td></tr>'}</table>
</body></html>"""


def serve(port, refresh):
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            snap = snapshot()
            body, ctype = (json.dumps(snap, indent=1), "application/json") if self.path.startswith("/json") else (render_html(snap, refresh), "text/html; charset=utf-8")
            self.send_response(200); self.send_header("Content-Type", ctype); self.send_header("Cache-Control", "no-store"); self.end_headers()
            self.wfile.write(body.encode())

        def log_message(self, *a): pass
    print(f"GPU dashboard at http://localhost:{port}/  (json at /json)", flush=True)
    HTTPServer(("127.0.0.1", port), H).serve_forever()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--watch", action="store_true"); ap.add_argument("--serve", type=int, default=None)
    ap.add_argument("--html", default=None); ap.add_argument("--json", action="store_true"); ap.add_argument("--refresh", type=int, default=10)
    a = ap.parse_args()
    if a.serve: serve(a.serve, a.refresh); return
    while True:
        snap = snapshot()
        if a.html: open(a.html, "w").write(render_html(snap, a.refresh))
        if a.json: print(json.dumps(snap, indent=1))
        elif a.watch: os.system("clear"); print(render_text(snap))
        elif not a.html: print(render_text(snap))
        if not a.watch: break
        time.sleep(a.refresh)

if __name__ == "__main__":
    main()

"""The GPU queue's command-line interface: `acquire NAME --cmd "..."` must take the lock (a positional
`cmd` and an option `--cmd` once shared a destination, so every job skipped the lock), record the job's
command line, and `release --rc` must write the history the dashboard reads."""
import importlib.util, json, os, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("gpu_lock", os.path.join(ROOT, "scripts", "gpu_lock.py"))
g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)


def _sandbox():
    d = tempfile.mkdtemp()
    g.LOCK = os.path.join(d, "lock"); g.LOCKD = g.LOCK + ".d"; g.Q = os.path.join(d, "q"); g.HIST = os.path.join(d, "h.jsonl"); g.SH = os.path.join(d, "shared")


def _main(*argv):
    sys.argv = ["gpu_lock", *argv]; g.main()


def test_acquire_with_cmd_takes_the_lock_and_records_the_command():
    _sandbox()
    _main("acquire", "t1", "--kind", "low", "--minutes", "1", "--pid", str(os.getpid()), "--cmd", "python x --iters 5")
    h = json.load(open(g.LOCK))
    assert h["name"] == "t1" and h["kind"] == "low" and h["cmd"] == "python x --iters 5" and os.path.isdir(g.LOCKD)
    assert g.holder()["name"] == "t1"
    _main("release", "t1", "--pid", str(os.getpid()), "--rc", "0")
    assert not os.path.exists(g.LOCK) and not os.path.isdir(g.LOCKD)
    ev = [json.loads(l) for l in open(g.HIST)]
    assert [e["ev"] for e in ev] == ["grant", "release"] and ev[1]["rc"] == 0 and ev[0]["cmd"] == "python x --iters 5"


def test_acquire_without_cmd_still_works():
    _sandbox()
    _main("acquire", "t2", "--kind", "train", "--minutes", "2", "--pid", str(os.getpid()))
    assert g.holder()["name"] == "t2" and g.holder()["cmd"] is None
    _main("release", "t2", "--pid", str(os.getpid()))
    assert g.holder() is None


def test_exit_code_recorded_when_a_waiter_freed_the_holder():
    _sandbox()
    _main("acquire", "t3", "--kind", "low", "--minutes", "1", "--pid", str(os.getpid()))
    g._release(force=True)                      # a waiter saw the dead pid first
    _main("release", "t3", "--pid", str(os.getpid()), "--rc", "3")
    ev = [json.loads(l) for l in open(g.HIST)]
    assert [e["ev"] for e in ev] == ["grant", "release", "exit"] and ev[2]["rc"] == 3


def test_front_ticket_sorts_ahead_of_older_tickets_of_its_class():
    _sandbox()
    os.makedirs(g.Q, exist_ok=True)
    json.dump({"name": "older", "kind": "timing", "minutes": 1, "pid": os.getpid(), "t": 1.0}, open(os.path.join(g.Q, "1.000_0_older.json"), "w"))
    tf = os.path.join(g.Q, f"0.000_0_PAUSE_{os.getpid()}.json")
    json.dump({"name": "PAUSE", "kind": "timing", "minutes": 1, "pid": os.getpid(), "t": 0.0}, open(tf, "w"))
    assert g.tickets()[0][3]["name"] == "PAUSE"


def test_ticket_filename_carries_the_pid():
    _sandbox()
    _main("acquire", "t4", "--kind", "low", "--minutes", "1", "--pid", str(os.getpid()))
    assert g.holder()["name"] == "t4"          # granted at once (empty queue), ticket consumed
    assert os.listdir(g.Q) == []
    _main("release", "t4", "--pid", str(os.getpid()))


def test_alive_treats_permission_error_as_alive(monkeypatch):
    def deny(pid, sig): raise PermissionError
    monkeypatch.setattr(g.os, "kill", deny)
    assert g.alive(12345) is True
    def gone(pid, sig): raise ProcessLookupError
    monkeypatch.setattr(g.os, "kill", gone)
    assert g.alive(12345) is False


def test_check_jobs_share_and_exclusive_waits():
    """Check jobs run side by side (up to SHARED_MAX); an exclusive acquirer waits until they release."""
    import threading, time
    _sandbox()
    g.acquire("c1", "check", 1, os.getpid()); g.acquire("c2", "check", 1, os.getpid())
    assert sorted(h["name"] for h in g.shared_holders()) == ["c1", "c2"] and g.holder() is None
    done = threading.Event()
    t = threading.Thread(target=lambda: (g.acquire("ex", "timing", 1, os.getpid()), done.set())); t.start()
    time.sleep(1.0); assert not done.is_set()
    for h in g.shared_holders(): os.remove(h["_file"])
    t.join(30); assert done.is_set() and g.holder()["name"] == "ex"


def test_gpu_run_job_runs_in_the_callers_directory(tmp_path):
    """gpu_run.sh runs the job in the caller's cwd (not the repo root), with its queue files under its own repo root.

    Before, every job ran in the repo root: ``python -m pytest`` from an agent worktree then imported the main
    checkout's packages, because ``-m`` puts the cwd ahead of PYTHONPATH."""
    import shutil, subprocess
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "runs").mkdir()
    for f in ("gpu_run.sh", "gpu_lock.py", "gpu_top.py", "gpu_util.sh", "gpu_pause.sh"):
        if os.path.exists(os.path.join(ROOT, "scripts", f)):
            shutil.copy(os.path.join(ROOT, "scripts", f), repo / "scripts" / f)
    caller = tmp_path / "caller dir"          # a space too: the cd must be quoted
    caller.mkdir()
    out = subprocess.run(["bash", str(repo / "scripts" / "gpu_run.sh"), "cwdtest", "low", "1", "--", "pwd", "-P"],
                         cwd=caller, capture_output=True, text=True, timeout=900)
    assert out.returncode == 0, out.stderr
    assert os.path.realpath(caller) in out.stdout.splitlines()
    hist = [json.loads(l) for l in open(repo / "runs" / "gpu_history.jsonl")]
    assert [e["ev"] for e in hist if e.get("name") == "cwdtest"][:2] == ["grant", "release"]

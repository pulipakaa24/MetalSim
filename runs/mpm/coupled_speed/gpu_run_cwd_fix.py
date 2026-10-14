"""Apply the gpu_run.sh caller-cwd fix to a checkout of origin/main (used by batch_07 on branch fix/gpu-run-caller-cwd).
    python3 gpu_run_cwd_fix.py CHECKOUT_DIR
Edits scripts/gpu_run.sh, tests/test_gpu_queue.py and docs/GUIDE.md (section 4); fails loudly if an anchor moved."""
import sys

D = sys.argv[1]


def edit(path, old, new, count=1):
    p = f"{D}/{path}"
    s = open(p).read()
    assert s.count(old) == count, (path, old[:60], s.count(old))
    open(p, "w").write(s.replace(old, new))


# --- scripts/gpu_run.sh
edit("scripts/gpu_run.sh",
     '  NAME=$1; KIND=$2; MIN=$3; shift 3; [ "$1" = "--" ] && shift\n  cd "$(dirname "$0")/.."\n',
     '  NAME=$1; KIND=$2; MIN=$3; shift 3; [ "$1" = "--" ] && shift\n'
     '  # The job runs in the caller\'s directory; only the queue\'s own files (scripts/, runs/) are taken from the repo\n'
     '  # root. Before 2026-10-13 the job ran in the repo root, so `python -m ...` (which puts the cwd ahead of\n'
     '  # PYTHONPATH on sys.path) launched from an agent worktree imported the MAIN checkout\'s packages: tests passed or\n'
     '  # failed against code the caller never meant to run (research_notes/coupled_speed/NOTE.md, section B).\n'
     '  CALLER_CWD=$PWD\n'
     '  cd "$(dirname "$0")/.."\n')
edit("scripts/gpu_run.sh",
     '  "$@" &\n  CHILD=$!\n',
     '  (cd "$CALLER_CWD" && exec "$@") &   # exec: the subshell becomes the job, so $! is the job\'s pid\n  CHILD=$!\n')

# --- tests/test_gpu_queue.py
p = f"{D}/tests/test_gpu_queue.py"
s = open(p).read()
s += '''

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
'''
open(p, "w").write(s)

# --- docs/GUIDE.md, section 4
edit("docs/GUIDE.md",
     "reason (it also kills the command if the wrapper is killed). `MINUTES` is an estimate shown to others.\n",
     "reason (it also kills the command if the wrapper is killed). `MINUTES` is an estimate shown to others.\n"
     "The command runs in the caller's directory (since 2026-10-13; before, it ran in the repo root, so `python -m ...`\n"
     "from a worktree imported the main checkout's packages); the queue's own files stay under the repo root.\n")
print("applied to", D)

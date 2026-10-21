#!/bin/bash
# gpu_run.sh runs the job in the caller's directory (branch fix/gpu-run-caller-cwd off origin/main, its own PR).
# No GPU work: the new test drives a copy of the queue scripts in a temporary repo (its own runs/), so it can run
# while other GPU jobs are queued. Steps: worktree off origin/main, apply gpu_run_cwd_fix.py, run
# tests/test_gpu_queue.py (with and without the fix: the new test must fail on the old script), commit, push, PR.
# Log: runs/mpm/coupled_speed/batch_07.log
set -u
W=.
R=$W/runs/mpm/coupled_speed
M=.
D=$W/upstream/gpufix
exec > >(tee -a "$R/batch_07.log") 2>&1
echo "== $(date) batch_07 start"
stop() { echo "STOP: $*"; echo "== $(date) batch_07 stopped"; exit 1; }
git -C $W fetch -q origin || stop "fetch"
if [ ! -d $D ]; then
  git -C $W worktree add -q -b fix/gpu-run-caller-cwd $D origin/main || stop "worktree add"
fi
cd $D || stop "cd"
echo "base $(git rev-parse --short HEAD) on branch $(git branch --show-current)"
if ! grep -q CALLER_CWD scripts/gpu_run.sh; then
  python3 $R/gpu_run_cwd_fix.py $D || stop "apply"
  # the new test must fail on the old gpu_run.sh (swap the file, no stash: the stash stack is shared)
  cp scripts/gpu_run.sh $R/_gpu_run_fixed.sh
  git show HEAD:scripts/gpu_run.sh > scripts/gpu_run.sh
  $M/.venv/bin/python -P -m pytest -q -p no:cacheprovider tests/test_gpu_queue.py -k callers 2>&1 | tail -1 | sed 's/^/old script: /'
  cp $R/_gpu_run_fixed.sh scripts/gpu_run.sh && rm -f $R/_gpu_run_fixed.sh
fi
bash -n scripts/gpu_run.sh || stop "syntax"
grep -q CALLER_CWD scripts/gpu_run.sh || stop "fix missing"
$M/.venv/bin/python -P -m pytest -q -p no:cacheprovider tests/test_gpu_queue.py 2>&1 | tail -2 | tee $R/_tq_new.txt
grep -q ' passed' $R/_tq_new.txt && ! grep -q failed $R/_tq_new.txt || stop "tests fail with the fix"
rm -f $R/_tq_new.txt
git add scripts/gpu_run.sh tests/test_gpu_queue.py docs/GUIDE.md
git commit -s -q -F - <<'MSG' || echo "(nothing to commit)"
gpu_run.sh: run the job in the caller's directory

Jobs ran in the repo root (gpu_run.sh cds there for the queue's own
files). `python -m ...` puts the cwd ahead of PYTHONPATH on sys.path, so
a job launched from an agent worktree imported the main checkout's
packages: tests/test_mpm.py run that way failed
test_coupled_contact_filter_keeps_source_order against an older
newton_compat in the main checkout's working tree. The job now runs in
the caller's directory (exec, so the recorded pid is still the job's);
the queue files stay under the repo root. Test:
test_gpu_run_job_runs_in_the_callers_directory (fails on the old script).

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
MSG
git log --oneline -1
git push -q -u origin fix/gpu-run-caller-cwd 2>&1 | grep -v '^remote:'
gh pr create --base main --head fix/gpu-run-caller-cwd --title "gpu_run.sh: run the job in the caller's directory" --body "$(cat <<'BODY'
Jobs launched through `scripts/gpu_run.sh` ran in the repo root, because the script `cd`s there for the queue's own files. `python -m ...` puts the current directory ahead of `PYTHONPATH` on `sys.path`. So a job launched from an agent worktree imported the **main checkout's** packages instead of the worktree's.

Found in the coupled-speed work: `tests/test_mpm.py` run this way failed `test_coupled_contact_filter_keeps_source_order` 19 times out of 19. It was testing an older `newton_compat.py` from the main checkout's working tree. The same code run as a script passed. Details: `research_notes/coupled_speed/NOTE.md` section B, on branch `mpm-coupled-speed`.

Change:
- `gpu_run.sh` saves the caller's directory and starts the job there with `(cd "$CALLER_CWD" && exec "$@") &`. Because of `exec`, `$!` is still the job's pid, so setpid, the heartbeat and the kill-on-exit are unchanged.
- The queue's own files (`runs/`, `scripts/`) stay relative to the repo root.

Tests:
- New: `tests/test_gpu_queue.py::test_gpu_run_job_runs_in_the_callers_directory`. It drives a copy of the queue scripts in a temporary repo, so it uses its own queue files, and checks that the job runs in the caller's directory (a path with a space). It fails on the old script.
- The other queue tests pass. Log: `runs/mpm/coupled_speed/batch_07.log` on `mpm-coupled-speed`.

Docs: one line in `docs/GUIDE.md` section 4.

Until this is merged, launch pytest through the queue as `python -P -m pytest`, or `cd` into the worktree first.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
BODY
)" || echo "FAIL: gh pr create"
echo "== $(date) batch_07 done"

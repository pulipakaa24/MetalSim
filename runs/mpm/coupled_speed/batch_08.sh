#!/bin/bash
# Coupled-scene speed: fork commits, validation on the pinned trees, fork pushes, pin bump and re-vendor (2026-10-13).
# Run only after mpm-main's PR (Newton fork head 99843d09, vendored solver, scripts/vendor_mpm152.py) is on origin/main.
# Docs and the PR come in batch_09 (written after reading the rebased docs).
#   0  preconditions: origin/main pins Newton 99843d09; clean tree; merge origin/main into mpm-coupled-speed (no rebase:
#      the branch is already pushed and is never force-pushed)
#   1  fork clones: Warp `metalsim` (must be at 95f6f121), Newton `metalsim-1.5.2` (must be at origin/main's pin);
#      apply the two patches, add METALSIM_CHANGES rows, commit locally (no push yet)
#   2  validation on the new commits (exclusive timing jobs; stop on a failed test or GPU-health problem):
#      tests/test_mpm.py (-P); floors bitwise vs batch_02's patched trees (expected equal: single-world scenes);
#      Franka-Pour policy step at 2,048 envs, tiled choice on vs off, two interleaved reps
#   3  push both forks (fast-forward only), pin bump, patch files, re-vendor + --check, tests/test_mpm_main.py, commit,
#      push branch mpm-coupled-speed
# Log: runs/mpm/coupled_speed/batch_08.log. Resumable (done markers in $R/b08/).
set -u
# commit identity for the fresh fork clones (the user's identity is set only in the metalsim repo's config)
export GIT_AUTHOR_NAME="Aditya Pulipaka" GIT_AUTHOR_EMAIL="adipu@utexas.edu"
export GIT_COMMITTER_NAME="Aditya Pulipaka" GIT_COMMITTER_EMAIL="adipu@utexas.edu"
W=/Users/aditya/robosim/.claude/worktrees/agent-a8f809a11de334dd0
R=$W/runs/mpm/coupled_speed
B=$R/b08
M=/Users/aditya/robosim
PY=$M/.venv-newton152/bin/python
PYM=$M/.venv/bin/python
PYP=$M/.claude/worktrees/agent-a278824da4c05bf29/.venv-pour/bin/python
GR=$M/scripts/gpu_run.sh
T=$W/upstream/trees
WI=$M/upstream/warp-innate
OLD_W=95f6f121b7e67025958335c7c15e338ac04a798a
mkdir -p $B
exec > >(tee -a "$R/batch_08.log") 2>&1
echo "== $(date) batch_08 start"
export METALSIM_ROOT=$M
stop() { echo "STOP: $*"; echo "== $(date) batch_08 stopped"; exit 1; }
health() {
  [ -e $M/runs/GPU_RUNAWAY_ALERT ] && stop "runs/GPU_RUNAWAY_ALERT exists"
  local u
  for i in $(seq 1 60); do u=$($M/scripts/gpu_util.sh); [ "${u:-0}" -lt 10 ] && return 0; sleep 2; done
  python3 $M/scripts/gpu_top.py 2>/dev/null | grep -q "none >= 1 %" && stop "GPU at ${u}% with no owning process after '$1'"
  echo "note: GPU at ${u}% after '$1', owned by another process"
}
job() { local name=$1 min=$2 log=$3; shift 4; $GR "$name" timing "$min" -- "$@" > "$log" 2>&1; local rc=$?; health "$name"; return $rc; }

# ---- 0. preconditions and rebase
cd $W || stop cd
git fetch -q origin || stop fetch
OLD_N=$(git show origin/main:scripts/setup_newton152.sh | sed -n 's/^PIN=\([0-9a-f]\{40\}\).*/\1/p')
echo "origin/main $(git rev-parse --short origin/main); Newton pin there: $OLD_N"
[ "${OLD_N:0:8}" = 99843d09 ] || stop "origin/main does not pin Newton 99843d09 yet (mpm-main's PR not merged)"
git show origin/main:scripts/vendor_mpm152.py > /dev/null 2>&1 || stop "origin/main has no scripts/vendor_mpm152.py"
if [ ! -f $B/merged ]; then
  git add runs/mpm/coupled_retime/batch_01.log runs/mpm/coupled_speed research_notes/coupled_speed 2>/dev/null
  git diff --cached --quiet || git commit -s -q -m "Coupled speed: logs before merging origin/main

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
  [ -z "$(git status --porcelain --untracked-files=no)" ] || { git status --short; stop "tracked changes left before the merge"; }
  git merge -q --no-edit origin/main || { git merge --abort; stop "merging origin/main conflicts (merge aborted)"; }
  touch $B/merged
fi
git log --oneline -3

# ---- 1. fork commits (local)
WF=$T/warp_fork; NF=$T/newton_fork
if [ ! -f $B/forks_committed ]; then
  rm -rf $WF $NF
  GIT_LFS_SKIP_SMUDGE=1 git clone -q --no-checkout $WI $WF || stop "warp clone"
  git -C $WF remote set-url origin https://github.com/pulipakaa24/warp.git
  git -C $WF fetch -q origin metalsim || stop "warp fetch"
  [ "$(git -C $WF rev-parse origin/metalsim)" = $OLD_W ] || stop "warp fork metalsim moved: $(git -C $WF rev-parse origin/metalsim)"
  GIT_LFS_SKIP_SMUDGE=1 git -C $WF checkout -q -B metalsim origin/metalsim || stop "warp checkout"
  git -C $WF apply $R/patches/warp_bsr_mv_tiles.diff || stop "warp patch"
  python3 $R/changes_row.py $WF $R/row_warp.txt || stop "warp changes row"
  git -C $WF add warp/config.py warp/_src/sparse.py METALSIM_CHANGES.md
  git -C $WF commit -s -q -m "Metal: bsr_mv takes CUDA's tiled-kernel choice for long rows

warp.sparse.bsr_mv used its tiled kernel only on CUDA; Metal summed every
row serially in one thread. Rows with many blocks (Newton implicit MPM's
rigidity operator body rows, ~1,170 blocks in example_mpm_twoway_coupling)
were latency-bound. warp.config.metal_bsr_mv_tiles (WP_METAL_BSR_MV_TILES,
default on) makes CUDA's choice on Metal; Metal's tile_sum is CUDA's
shuffle tree, so the summation order is CUDA's. METALSIM_CHANGES.md row.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" || stop "warp commit"
  mkdir -p $WF/warp/bin && cp $WI/warp/bin/libwarp*.dylib $WF/warp/bin/   # native code unchanged by this commit
  git clone -q -b metalsim-1.5.2 https://github.com/pulipakaa24/newton.git $NF || stop "newton clone"
  [ "$(git -C $NF rev-parse HEAD)" = "$OLD_N" ] || stop "newton fork metalsim-1.5.2 is at $(git -C $NF rev-parse HEAD), not $OLD_N"
  git -C $NF apply $R/patches/newton_ijtm_tiled.diff || stop "newton patch"
  python3 $R/changes_row.py $NF $R/row_newton.txt || stop "newton changes row"
  git -C $NF add newton/_src/solvers/implicit_mpm/solve_rheology.py METALSIM_CHANGES.md
  git -C $NF commit -s -q -m "Metal: long-row rigidity product through bsr_mv's tiled kernel

In the rheology loop, IJtm (one block per collider node of each body) went
through the listed-rows kernel, one thread per row. When warp.sparse.bsr_mv
would tile the matrix (CUDA's choice, taken on Metal by the Warp fork's
metal_bsr_mv_tiles), the product now goes through bsr_mv itself: CUDA's
kernel and summation order. Unchanged without that Warp option.
METALSIM_CHANGES.md row.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" || stop "newton commit"
  git -C $NF apply $R/patches/newton_harvest_compact.diff || stop "newton harvest patch"
  python3 $R/changes_row.py $NF $R/row_newton_harvest.txt || stop "newton harvest row"
  git -C $NF add newton/_src/solvers/implicit_mpm/solver_implicit_mpm.py METALSIM_CHANGES.md
  git -C $NF commit -s -q -m "Metal: compact the proxy-body nodes before the ordered wrench harvest

coupling_harvest_proxy_wrenches on Metal had every proxy body scan every
collider node of its world (with one world: every active node) in 32
fixed lanes. A stable integer-scan compaction (capture-safe) lists the
proxy-body nodes first; the lanes walk only those. Same per-node terms in
another fixed order. MetalSim multibody_sand (1 world): harvest
15.6 -> 1.5 ms, coupled step 49.3 -> 34.2 ms (16 bodies); Franka-Pour
unchanged. NEWTON_METAL_HARVEST_COMPACT=0 restores the full scan.
METALSIM_CHANGES.md row.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" || stop "newton harvest commit"
  git -C $NF apply $R/patches/newton_body_force_compact.diff || stop "newton body-force patch"
  python3 $R/changes_row.py $NF $R/row_newton_bodyforce.txt || stop "newton body-force row"
  git -C $NF add newton/examples/mpm/example_mpm_twoway_coupling.py METALSIM_CHANGES.md
  git -C $NF commit -s -q -m "example_mpm_twoway_coupling: compact the body nodes before the ordered force sum

On Metal every body scanned all grid nodes (655,360) in every substep. A
stable integer-scan compaction lists the nodes in contact with a body
first; the 256 fixed lanes sum over that list. 1.05 -> 0.38 ms per call;
captured two-way 40.0 -> 37.6 ms/frame (M4 Max). Repeats bitwise.
NEWTON_METAL_BODY_FORCE_COMPACT=0 restores the full scan.
METALSIM_CHANGES.md row.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" || stop "newton body-force commit"
  touch $B/forks_committed
fi
NEW_W=$(git -C $WF rev-parse HEAD); NEW_N=$(git -C $NF rev-parse HEAD)
echo "warp $OLD_W -> $NEW_W; newton $OLD_N -> $NEW_N"
export PYTHONPATH=$WF:$NF:$W WARP_CACHE_PATH=$R/kcache_pinned
$PY -c "import warp, newton; print(warp.__file__, warp.config.metal_bsr_mv_tiles, newton.__file__)" || stop "pinned trees import"

# ---- 2. validation
if [ ! -f $B/tests_ok ]; then
  job cs8_test_mpm 30 $B/test_mpm.log -- $PY -P -m pytest -q -rf -p no:cacheprovider $W/tests/test_mpm.py
  echo "test_mpm: $(grep -E '[0-9]+ (passed|failed)' $B/test_mpm.log | tail -1)"
  grep -qE ' failed' $B/test_mpm.log && stop "tests/test_mpm.py fails on the pinned trees"
  touch $B/tests_ok
fi
STAGES=floors KIND=timing WARP_CACHE_PATH=$R/kcache_pinned bash $W/metalsim/physics/mpm/record_variant_suite.sh \
    $B/floors_pinned $WF $NF $W | grep -v '^skip'
health floors
echo "-- floors vs batch_02's patched trees (expected: equal except twoway, which now has the body-force compaction)"
$PY -P $W/research_notes/mpm_parity_closeout/bitwise_dirs.py $R/floors_new $B/floors_pinned | tee $B/bitwise_floors_vs_batch02.txt
echo "-- twoway vs batch_bf1's (IJtm + body-force compaction; expected equal)"
$PY -P $W/research_notes/mpm_parity_closeout/bitwise_dirs.py $R/bf1/floors_bf1 $B/floors_pinned | tee $B/bitwise_twoway_vs_bf1.txt
FB=$W/scripts/diagnostics/franka_pour_memory/franka_bitwise.py
for t in 1 2; do
  [ -f $B/pour_hash_$t.json ] && continue
  job cs8_pour_hash_$t 30 $B/pour_hash_$t.log -- $PYP -P $FB 64 30 $B/pour_hash_$t.json || echo "FAIL: pour hash $t"
done
python3 - $B/pour_hash_1.json $B/pour_hash_2.json $R/hv1/pour_hash_on1.json <<'PYEOF' || echo "FAIL: pour hash comparison"
import json, sys
h = [json.load(open(p))["hashes"] for p in sys.argv[1:4]]
def cmp(a, b):
    i = next((k for k, (x, y) in enumerate(zip(a, b)) if x != y), None)
    return f"bitwise equal over {len(a)} steps" if i is None else f"first differs at step {i}"
print("Franka-Pour pinned run 1 vs 2:", cmp(h[0], h[1]))
print("Franka-Pour pinned vs batch_hv1 (IJtm + harvest on the old Newton base; differs only if the GS fix touches it):", cmp(h[0], h[2]))
PYEOF
for rep in 1 2; do
  for tiles in 1 0; do
    id=pour2048_tiles${tiles}_r$rep
    [ -f $B/$id.json ] && continue
    WP_METAL_BSR_MV_TILES=$tiles job cs8_$id 20 $B/$id.log -- env WP_METAL_BSR_MV_TILES=$tiles $PYP -P -m metalsim.learn.franka_pour_ppo \
        bench --num-envs 2048 --mode step --steps 15 --warmup 3 --json $B/$id.json || echo "FAIL: $id"
    python3 -c "import json; d=json.load(open('$B/$id.json')); print('$id', round(d['step_s_median'], 3), 's/step', round(d['env_steps_per_s']), 'env-steps/s')" 2>/dev/null
  done
done

# ---- 3. push forks, pins, vendor, commit
if [ ! -f $B/forks_pushed ]; then
  git -C $WF push origin metalsim:metalsim || stop "warp push (not a fast-forward?)"
  git -C $NF push origin metalsim-1.5.2:metalsim-1.5.2 || stop "newton push (not a fast-forward?)"
  touch $B/forks_pushed
fi
cd $W
python3 $R/pin_bump.py $W $OLD_W $NEW_W $OLD_N $NEW_N || stop "pin bump"
git -C $WF diff ce15f6bb2545e2d6c4a4c1fac8e40bb800db7265 $NEW_W > patches/warp/metalsim-${NEW_W:0:8}.diff || stop "warp patch file"
git -C $NF fetch -q --tags origin
git -C $NF diff v1.5.2 $NEW_N > patches/newton/metalsim-1.5.2-${NEW_N:0:8}.diff || stop "newton patch file"
git rm -q --ignore-unmatch patches/warp/metalsim-${OLD_W:0:8}.diff patches/newton/metalsim-1.5.2-${OLD_N:0:8}.diff
$PYM scripts/vendor_mpm152.py $NF || stop "vendor"
$PYM scripts/vendor_mpm152.py --check $NF || stop "vendor --check"
export PYTHONPATH=$WF:$W
job cs8_test_mpm_main 30 $B/test_mpm_main.log -- $PYM -P -m pytest -q -rf -p no:cacheprovider $W/tests/test_mpm_main.py
echo "test_mpm_main: $(grep -E '[0-9]+ (passed|failed|skipped)' $B/test_mpm_main.log | tail -1)"
git add -A patches scripts metalsim/physics/mpm/implicit_mpm README.md docs/GUIDE.md metalsim/physics/mpm/MPM.md \
    THIRD_PARTY_NOTICES.md runs/mpm/coupled_speed
git commit -s -q -m "Pin Warp ${NEW_W:0:8} and Newton ${NEW_N:0:8}: tiled long-row bsr_mv on Metal; re-vendor

Warp metalsim ${OLD_W:0:8} -> ${NEW_W:0:8} (bsr_mv takes CUDA's tiled-kernel choice
on Metal), Newton metalsim-1.5.2 ${OLD_N:0:8} -> ${NEW_N:0:8} (long-row rigidity
product through bsr_mv); patches/ regenerated; vendored implicit_mpm
regenerated (scripts/vendor_mpm152.py). Validation logs in
runs/mpm/coupled_speed/b08/.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" || echo "(nothing to commit)"
git push -q -u origin mpm-coupled-speed 2>&1 | grep -v '^remote:'
git log --oneline -3
echo "== $(date) batch_08 done (docs + PR follow in batch_09)"

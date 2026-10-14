#!/bin/bash
# ANYmal on sand, captured (1.21x the L4 on the IJtm trees): where the Metal frame goes, and two bitwise-safe levers
# for its Gauss-Seidel solve. Measurement only (no code change kept, nothing pushed to the forks).
# Prerequisite (checked): batch_02's trees (upstream/trees/warp_new, newton_new); ANYmal does not use bsr_mv, so they
# equal the pinned trees for this scene. Independent of batch_08 / batch_bf1 (run them one after another, not together:
# all commit to the same branch).
#   1  first run of Newton's reordered Gauss-Seidel ("gs-soa", new to Metal here) under a 300 s / 60 s-stall watchdog
#   2  parts (robot graph vs eager sand step), per-frame hashes, 60 frames after 5, 3 interleaved reps of:
#        base (gs, 2 threadgroups per core) | gs, 1 / 4 / 8 per core | gs-soa
#      the threadgroup count must not change results (colour blocks are independent): hashes equal to base;
#      gs-soa may (different code path): hashes reported
#   3  per-kernel profile of the robot graph alone and of the sand step alone (isolated dispatches)
#   4  commit and push the logs
# Only GPU-health problems stop it; failures listed in any1_failures.txt. Resumable. Log: batch_any1.log
set -u
W=/Users/aditya/robosim/.claude/worktrees/agent-a8f809a11de334dd0
R=$W/runs/mpm/coupled_speed
O=$R/any1
M=/Users/aditya/robosim
PY=$M/.venv-newton152/bin/python
GR=$M/scripts/gpu_run.sh
T=$W/upstream/trees
F=$O/any1_failures.txt
P=$W/metalsim/physics/mpm/probes/anymal_probe.py
mkdir -p $O
exec > >(tee -a "$R/batch_any1.log") 2>&1
echo "== $(date) batch_any1 start"
export METALSIM_ROOT=$M
stop() { echo "STOP: $*"; echo "== $(date) batch_any1 stopped"; exit 1; }
health() {
  [ -e $M/runs/GPU_RUNAWAY_ALERT ] && stop "runs/GPU_RUNAWAY_ALERT exists"
  local u
  for i in $(seq 1 60); do u=$($M/scripts/gpu_util.sh); [ "${u:-0}" -lt 10 ] && return 0; sleep 2; done
  python3 $M/scripts/gpu_top.py 2>/dev/null | grep -q "none >= 1 %" && stop "GPU at ${u}% with no owning process after '$1'"
  echo "note: GPU at ${u}% after '$1', owned by another process"
}
sjob() { local name=$1 min=$2 log=$3; shift 4; $GR "$name" timing "$min" -- "$@" > "$log" 2>&1; local rc=$?; health "$name"
         [ $rc -eq 0 ] || { echo "FAIL: $name exit $rc ($log)"; echo "$name exit $rc" >> $F; }; return 0; }
[ -f $T/.trees_ok ] || stop "batch_02 trees missing"
export PYTHONPATH=$T/warp_new:$T/newton_new:$W WARP_CACHE_PATH=$R/kcache_new

# ---- 1. gs-soa first run under a watchdog
if [ ! -f $O/soa_first.ok ]; then
  sjob cany_soa_first 10 $O/soa_first.log -- python3 $W/runs/rigid17/tools/watch.py --log $O/soa_first_watch.log --timeout 300 \
      --stall 60 -- $PY -P $P --solver gs-soa --frames 5 --warmup 1 --json $O/soa_first.json
  grep -v '^Module' $O/soa_first_watch.log | tail -3
  [ -f $O/soa_first.json ] && touch $O/soa_first.ok || echo "gs-soa first run failed: the gs-soa variant is skipped below"
fi

# ---- 2. variants, 3 interleaved reps
for rep in 1 2 3; do
  for v in base k1 k4 k8 soa; do
    id=${v}_r$rep
    [ -f $O/$id.json ] && continue
    case $v in
      base) a=() ;;
      k1) a=(--color-blocks-per-core 1) ;;
      k4) a=(--color-blocks-per-core 4) ;;
      k8) a=(--color-blocks-per-core 8) ;;
      soa) [ -f $O/soa_first.ok ] || continue; a=(--solver gs-soa) ;;
    esac
    sjob cany_$id 15 $O/$id.log -- $PY -P $P --frames 60 --warmup 5 ${a[@]+"${a[@]}"} --json $O/$id.json
    tail -1 $O/$id.log
  done
done
python3 - "$O" <<'PYEOF' | tee $O/summary.md
import glob, json, statistics, sys
O = sys.argv[1]
base = json.load(open(f"{O}/base_r1.json"))["hashes"] if glob.glob(f"{O}/base_r1.json") else None
print("| variant | robot ms (runs) | sand ms (runs) | total ms | hashes vs base r1 |")
print("|---|---|---|---|---|")
for v in ("base", "k1", "k4", "k8", "soa"):
    runs = [json.load(open(f)) for f in sorted(glob.glob(f"{O}/{v}_r*.json"))]
    if not runs:
        continue
    rb = [r["robot_ms_median"] for r in runs]; sd = [r["sand_ms_median"] for r in runs]; tt = [r["total_ms_median"] for r in runs]
    eq = []
    for r in runs:
        if base is None:
            eq.append("?"); continue
        first = next((i for i, (a, b) in enumerate(zip(base, r["hashes"])) if a != b), None)
        eq.append("equal" if first is None else f"differs from frame {first}")
    print(f"| {v} | {statistics.median(rb):.2f} ({' / '.join(f'{x:.1f}' for x in rb)}) | {statistics.median(sd):.2f} "
          f"({' / '.join(f'{x:.1f}' for x in sd)}) | {statistics.median(tt):.2f} | {'; '.join(eq)} |")
PYEOF

# ---- 3. kernel profile of the parts (isolated dispatches; the robot graph eager, then the sand step)
if [ ! -f $O/kprof_anymal.json ]; then
  sjob cany_kprof 20 $O/kprof_anymal.log -- env WP_METAL_PROFILE=1 $PY -P $W/metalsim/physics/mpm/probes/kernel_profile.py \
      example:mpm_anymal --mode profile --warmup 5 --frames 6 --json $O/kprof_anymal.json
fi
python3 $R/l4_anymal_split.py $W/runs/mpm/profile/kernels_l4/anymal_capture_cuda_gpu_kern_sum.csv 4 | tee $O/l4_split.txt

# ---- 4. commit and push
cd $W
git add runs/mpm/coupled_speed metalsim/physics/mpm/probes/anymal_probe.py
git commit -s -q -m "Coupled speed: ANYmal parts, Gauss-Seidel launch and solver variants (logs)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" || echo "(nothing to commit)"
git push -q origin mpm-coupled-speed 2>&1 | grep -v '^remote:'; git log --oneline -1
echo "failures: $(cat $F 2>/dev/null | tr '\n' ';')"
echo "== $(date) batch_any1 done"

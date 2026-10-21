#!/bin/bash
# Two-way coupling's ordered body-force sum: compact the body nodes first (stable scan), then the fixed-lane sum
# (patches/newton_body_force_compact.diff, Newton example_mpm_twoway_coupling.py; research_notes/coupled_speed/NOTE.md C).
# Stacks on the tiled-IJtm trees of batch_02 (upstream/trees/{warp_new,newton_new}); independent of batch_08.
# Prerequisite (checked): those trees exist. Franka-Pour is not run: the change is in the two-way example only.
#   1  Metal kernel check under a watchdog: compaction exact, forces = full-scan to float32 rounding and = float64
#      reference, repeat bitwise, per-call time full vs compact (655,360 nodes, 12 bodies)
#   2  two-way floor recording with the change, twice (repeatability) + the floor ensemble vs the L4
#   3  timing (L4 protocol), two-way captured and eager, IJtm-only trees vs IJtm + compaction, 3 interleaved reps
#   4  per-kernel profile of two-way late, joined with the L4's
#   5  commit and push the logs (branch mpm-coupled-speed)
# Only GPU-health problems stop it; failures are listed in bf1_failures.txt. Resumable. Log: batch_bf1.log
set -u
W=.
R=$W/runs/mpm/coupled_speed
O=$R/bf1
M=.
PY=$M/.venv-newton152/bin/python
GR=$M/scripts/gpu_run.sh
T=$W/upstream/trees
F=$O/bf1_failures.txt
mkdir -p $O/timing $O/profile
exec > >(tee -a "$R/batch_bf1.log") 2>&1
echo "== $(date) batch_bf1 start"
export METALSIM_ROOT=$M
stop() { echo "STOP: $*"; echo "== $(date) batch_bf1 stopped"; exit 1; }
health() {
  [ -e $M/runs/GPU_RUNAWAY_ALERT ] && stop "runs/GPU_RUNAWAY_ALERT exists"
  local u
  for i in $(seq 1 60); do u=$($M/scripts/gpu_util.sh); [ "${u:-0}" -lt 10 ] && return 0; sleep 2; done
  python3 $M/scripts/gpu_top.py 2>/dev/null | grep -q "none >= 1 %" && stop "GPU at ${u}% with no owning process after '$1'"
  echo "note: GPU at ${u}% after '$1', owned by another process"
}
sjob() { local name=$1 min=$2 log=$3; shift 4; $GR "$name" timing "$min" -- "$@" > "$log" 2>&1; local rc=$?; health "$name"
         [ $rc -eq 0 ] || { echo "FAIL: $name exit $rc ($log)"; echo "$name exit $rc" >> $F; }; return 0; }
[ -f $T/.trees_ok ] && [ -d $T/warp_new/warp ] && [ -d $T/newton_new/newton ] || stop "batch_02 trees missing"
# tree: IJtm trees + the body-force compaction
if [ ! -f $T/.bf_ok ]; then
  rm -rf $T/newton_bf && mkdir -p $T/newton_bf && cp -R $T/newton_new/newton $T/newton_bf/ || stop "copy"
  patch -p1 -d $T/newton_bf < $R/patches/newton_body_force_compact.diff || stop "patch"
  touch $T/.bf_ok
fi
ij_env() { export PYTHONPATH=$T/warp_new:$T/newton_new:$W WARP_CACHE_PATH=$R/kcache_new; }
bf_env() { export PYTHONPATH=$T/warp_new:$T/newton_bf:$W WARP_CACHE_PATH=$R/kcache_bf; }
EX=$T/newton_bf/newton/examples/mpm/example_mpm_twoway_coupling.py

# ---- 1. kernel check
if ! grep -q '^OK' $O/kernel_check.log 2>/dev/null; then
  bf_env
  sjob cbf_check 10 $O/kernel_check.log -- python3 $W/runs/rigid17/tools/watch.py --log $O/kernel_check_watch.log --timeout 300 \
      --stall 60 -- $PY -P $W/metalsim/physics/mpm/probes/body_force_compact_check.py $EX --device metal:0 --nodes 655360 --contact 6000
  grep -v '^Module' $O/kernel_check_watch.log | tail -4
  grep -q '^OK' $O/kernel_check_watch.log && cp $O/kernel_check_watch.log $O/kernel_check.log
  grep -q '^OK' $O/kernel_check.log 2>/dev/null || stop "kernel check failed (see kernel_check_watch.log)"
fi

# ---- 2. floors: two-way with the change, twice; ensemble
for rep in 1 2; do
  d=$O/floors_bf$rep/F_twoway
  [ -f $d/meta.json ] && continue
  bf_env
  sjob cbf_twoway_$rep 20 $O/floors_bf${rep}_twoway.log -- $PY $W/metalsim/parity/isaac_side/record_mpm.py --device metal:0 \
      --chunk-graph --gpu-exit --scene example:mpm_twoway_coupling --seconds 1 --stride 2 --out $d
done
$PY -P $W/research_notes/mpm_parity_closeout/bitwise_dirs.py $O/floors_bf1 $O/floors_bf2 | sed 's/^/repeat: /'
$PY -P $W/research_notes/mpm_parity_closeout/bitwise_dirs.py $O/floors_bf1 $R/floors_new | sed 's/^/vs IJtm-only (expected to differ): /'
L4=$M/runs/parity/isaaclab3_mpm_floors
(cd $W && $PY -P -m metalsim.physics.mpm.floors_dist $L4/mpm_floors $L4/mpm_floors2 --metal $R/floors_new $O/floors_bf1 \
    --scenes twoway --md $O/floors_dist.md --json $O/floors_dist.json > $O/floors_dist.log 2>&1) || echo "FAIL: floors_dist"
grep -i 'verdict\|worst frame' $O/floors_dist.md

# ---- 3. timing
for rep in 1 2 3; do
  for t in ij bf; do
    for s in twoway_capture twoway; do
      case $s in
        twoway) a=(--scene example:mpm_twoway_coupling) ;;
        twoway_capture) a=(--scene example:mpm_twoway_coupling --set capture=true) ;;
      esac
      id=T_${s}_${t}_r$rep
      [ -f $O/timing/$id/meta.json ] && continue
      ${t}_env
      sjob cbf_$id 15 $O/timing/$id.log -- $PY $W/metalsim/parity/isaac_side/record_mpm.py --device metal:0 --chunk-graph \
          --gpu-exit --no-log --seconds 2 --stride 1000 "${a[@]}" --out $O/timing/$id
      grep '^done' $O/timing/$id.log
    done
  done
done
python3 - "$O" <<'PYEOF' | tee $O/timing/summary.md
import glob, json, statistics, sys
O = sys.argv[1]
L4 = {"twoway": 73.0, "twoway_capture": 31.0}
print("| scene | L4 | IJtm only (runs) | IJtm + compaction (runs) | IJtm / L4 | + compaction / L4 |")
print("|---|---|---|---|---|---|")
for s in ("twoway_capture", "twoway"):
    v = {t: [statistics.median(json.load(open(f))["wall_s"]) * 1e3 for f in sorted(glob.glob(f"{O}/timing/T_{s}_{t}_r*/meta.json"))]
         for t in ("ij", "bf")}
    if not v["ij"] or not v["bf"]:
        print(f"| {s} | missing | {v} |"); continue
    a, b = statistics.median(v["ij"]), statistics.median(v["bf"])
    print(f"| {s} | {L4[s]} | {a:.1f} ({' / '.join(f'{x:.1f}' for x in v['ij'])}) | {b:.1f} ({' / '.join(f'{x:.1f}' for x in v['bf'])}) | {a / L4[s]:.2f}x | {b / L4[s]:.2f}x |")
PYEOF

# ---- 4. profile
if [ ! -f $O/profile/kprof_twoway_late_bf.json ]; then
  bf_env
  sjob cbf_kprof 20 $O/profile/kprof_twoway_late_bf.log -- env WP_METAL_PROFILE=1 $PY $W/metalsim/physics/mpm/probes/kernel_profile.py \
      example:mpm_twoway_coupling --mode profile --warmup 140 --frames 6 --json $O/profile/kprof_twoway_late_bf.json
fi
if [ -f $O/profile/kprof_twoway_late_bf.json ]; then
  mkdir -p $O/profile/join && cp $O/profile/kprof_twoway_late_bf.json $O/profile/join/profile_twoway_coupling.json
  rm -f $O/profile/joined_late_bf.md
  (cd $O/profile && python3 $W/runs/mpm/profile/kernels_l4/kjoin.py $W/runs/mpm/profile/kernels_l4 join twoway_coupling 4 \
      joined_late_bf.md twoway_coupling_capture > /dev/null)
  head -14 $O/profile/joined_late_bf.md
fi

# ---- 5. commit and push the logs
cd $W
git add runs/mpm/coupled_speed metalsim/physics/mpm/probes/body_force_compact_check.py research_notes/coupled_speed
git commit -s -q -m "Coupled speed: two-way body-force compaction validated and timed (logs)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" || echo "(nothing to commit)"
git push -q origin mpm-coupled-speed 2>&1 | grep -v '^remote:'; git log --oneline -1
echo "failures: $(cat $F 2>/dev/null | tr '\n' ';')"
echo "== $(date) batch_bf1 done"

#!/bin/bash
# Coupled-scene speed, step 1 (2026-10-13): validate and measure the tiled IJtm product on Metal, in trees built
# from the pinned forks plus two small patches (patches/warp_bsr_mv_tiles.diff, patches/newton_ijtm_tiled.diff).
# Nothing is pushed to the forks or pinned here; this batch only measures. At the end it commits the logs to
# branch mpm-coupled-speed and pushes that branch.
#
# Phases (each GPU job is one exclusive gpu_run.sh `timing` job; health is checked after each; the batch STOPS on a
# failed job or a GPU that stays busy with no owner):
#   0  trees: copies of the installed Warp (95f6f121) and Newton (3fd1f052) packages with the patches applied
#   1  first launches of the tiled kernel on Metal (small to large, under a 300 s / 60 s-stall watchdog)
#   2  the two in-loop rigidity products timed separately, base vs new (eager, 150 frames of two-way)
#   3  tests/test_mpm.py on the new trees
#   4  the six floor recordings, base and new, plus a second new two-way run; bitwise and floor-ensemble checks
#   5  Franka-Pour trajectory hashes at 64 envs: base, new, new again (repeatability)
#   6  timing, three interleaved repetitions: two-way captured / eager and ANYmal captured (L4 protocol, 2 s)
#   7  per-kernel profile of two-way late in the run on the new trees, joined with the L4's
#   8  commit the logs and push the branch
# Log: runs/mpm/coupled_speed/batch_02.log. Re-running skips finished steps.
set -u
W=.
R=$W/runs/mpm/coupled_speed
M=.
PY=$M/.venv-newton152/bin/python
GR=$M/scripts/gpu_run.sh
T=$W/upstream/trees
mkdir -p "$R" "$T"
exec > >(tee -a "$R/batch_02.log") 2>&1
echo "== $(date) batch_02 start"
export METALSIM_ROOT=$M

stop() { echo "STOP: $*"; echo "== $(date) batch_02 stopped"; exit 1; }
health() {  # after a job: GPU back under 10 % within 120 s, or at least not busy with no owner; no runaway alert
  [ -e $M/runs/GPU_RUNAWAY_ALERT ] && stop "runs/GPU_RUNAWAY_ALERT exists"
  local u
  for i in $(seq 1 60); do u=$($M/scripts/gpu_util.sh); [ "${u:-0}" -lt 10 ] && return 0; sleep 2; done
  if python3 $M/scripts/gpu_top.py 2>/dev/null | grep -q "none >= 1 %"; then
    stop "GPU at ${u}% with no owning process after '$1' (possible runaway GPU work; do not start more jobs)"
  fi
  echo "note: GPU at ${u}% after '$1', owned by another process"
}
job() {  # job NAME MINUTES -- cmd...   (exclusive timing job; stops the batch on failure)
  local name=$1 min=$2; shift 3
  $GR "$name" timing "$min" -- "$@"; local rc=$?
  health "$name"
  [ $rc -eq 0 ] || stop "$name exit $rc"
}
base_env() { export PYTHONPATH=$W; export WARP_CACHE_PATH=$R/kcache_base; }
new_env() { export PYTHONPATH=$T/warp_new:$T/newton_new:$W; export WARP_CACHE_PATH=$R/kcache_new; }

# ---- 0. trees
WI=$M/upstream/warp-innate; NI=$M/upstream/newton-metalsim-1.5.2
echo "installed warp $(git -C $WI rev-parse HEAD) newton $(git -C $NI rev-parse HEAD)"
git -C $WI status --porcelain --untracked-files=no -- warp | head -5
git -C $NI status --porcelain --untracked-files=no -- newton | head -5
[ "$(git -C $WI rev-parse HEAD)" = 95f6f121b7e67025958335c7c15e338ac04a798a ] || stop "warp-innate not at the pin"
[ "$(git -C $NI rev-parse HEAD)" = 3fd1f052a5ad031578960d9642aaaedc03834e4d ] || stop "newton-metalsim-1.5.2 not at the pin"
[ -z "$(git -C $WI status --porcelain --untracked-files=no -- warp)" ] || stop "warp-innate has local changes"
[ -z "$(git -C $NI status --porcelain --untracked-files=no -- newton)" ] || stop "newton-metalsim-1.5.2 has local changes"
if [ ! -f $T/.trees_ok ]; then
  rm -rf "$T/warp_new" "$T/newton_new"; mkdir -p $T/warp_new $T/newton_new
  cp -R $WI/warp $T/warp_new/ && cp -R $NI/newton $T/newton_new/ || stop "copy failed"
  patch -p1 -d $T/warp_new < $R/patches/warp_bsr_mv_tiles.diff || stop "warp patch"
  patch -p1 -d $T/newton_new < $R/patches/newton_ijtm_tiled.diff || stop "newton patch"
  touch $T/.trees_ok
fi
new_env; $PY -c "import warp as wp, newton; from newton._src.solvers.implicit_mpm import solve_rheology as s; print(wp.__file__, wp.config.metal_bsr_mv_tiles, newton.__file__, hasattr(s, '_bsr_mv_uses_tiles'))" || stop "new trees do not import"
base_env; $PY -c "import warp as wp, newton; print(wp.__file__, getattr(wp.config, 'metal_bsr_mv_tiles', None), newton.__file__)"

# ---- 1. first launches of the tiled kernel
if ! grep -q "^bmt ok" $R/phase1.done 2>/dev/null; then
  new_env
  job cs_bmt_check 10 -- python3 $W/runs/rigid17/tools/watch.py --log $R/bsr_mv_tiles_check.log --timeout 300 --stall 60 \
      -- $PY $W/metalsim/physics/mpm/probes/bsr_mv_tiles_check.py
  cat $R/bsr_mv_tiles_check.log
  echo "bmt ok" > $R/phase1.done
fi

# ---- 2. the two products separately
for t in base new; do
  [ -f $R/rigidity_mv_$t.json ] && continue
  ${t}_env
  job cs_rmv_$t 20 -- $PY $W/metalsim/physics/mpm/probes/rigidity_mv_probe.py example:mpm_twoway_coupling --frames 150 \
      --json $R/rigidity_mv_$t.json > $R/rigidity_mv_$t.log 2>&1
  grep '^frame' $R/rigidity_mv_$t.log | tail -4
done

# ---- 3. tests
if [ ! -f $R/test_mpm_new.done ]; then
  new_env
  (cd $W && job cs_test_mpm 30 -- $PY -m pytest -q -p no:cacheprovider $W/tests/test_mpm.py > $R/test_mpm_new.log 2>&1)
  [ $? -eq 0 ] || stop "test_mpm (see test_mpm_new.log)"
  tail -3 $R/test_mpm_new.log; touch $R/test_mpm_new.done
fi

# ---- 4. floors
REC=$W/metalsim/physics/mpm/record_variant_suite.sh
STAGES=floors KIND=timing WARP_CACHE_PATH=$R/kcache_base bash $REC $R/floors_base - - $W || stop "floors base"
STAGES=floors KIND=timing WARP_CACHE_PATH=$R/kcache_new bash $REC $R/floors_new $T/warp_new $T/newton_new $W || stop "floors new"
if [ ! -f $R/floors_new2/F_twoway/meta.json ]; then
  new_env
  job cs_twoway_rep 20 -- $PY $W/metalsim/parity/isaac_side/record_mpm.py --device metal:0 --chunk-graph --gpu-exit \
      --scene example:mpm_twoway_coupling --seconds 1 --stride 2 --out $R/floors_new2/F_twoway > $R/floors_new2_twoway.log 2>&1
fi
health floors
base_env
echo "-- bitwise base vs new (expected: equal except twoway)"; $PY $W/research_notes/mpm_parity_closeout/bitwise_dirs.py $R/floors_base $R/floors_new | tee $R/bitwise_base_new.txt
echo "-- bitwise new vs new again (repeatability)"; $PY $W/research_notes/mpm_parity_closeout/bitwise_dirs.py $R/floors_new2 $R/floors_new | tee $R/bitwise_new_new2.txt
L4=$M/runs/parity/isaaclab3_mpm_floors
(cd $W && $PY -m metalsim.physics.mpm.floors_dist $L4/mpm_floors $L4/mpm_floors2 --metal $R/floors_base $R/floors_new \
    --scenes twoway granular multi_material column_a2 plate_drag --md $R/floors_dist.md --json $R/floors_dist.json > $R/floors_dist.log 2>&1) \
    || echo "floors_dist failed (see floors_dist.log)"
grep -i 'verdict\|worst frame' $R/floors_dist.md | head -30

# ---- 5. Franka-Pour hashes
FB=$W/scripts/diagnostics/franka_pour_memory/franka_bitwise.py
for t in base new new2; do
  [ -f $R/pour_hash_$t.json ] && continue
  case $t in base) base_env ;; *) new_env ;; esac
  job cs_pour_$t 30 -- $PY $FB 64 30 $R/pour_hash_$t.json > $R/pour_hash_$t.log 2>&1
done
base_env
python3 - "$R" <<'PYEOF'
import json, sys
R = sys.argv[1]
h = {t: json.load(open(f"{R}/pour_hash_{t}.json")) for t in ("base", "new", "new2")}
def seq(d): return d["hashes"] if isinstance(d, dict) else d
for a, b in (("new", "new2"), ("base", "new")):
    x, y = seq(h[a]), seq(h[b])
    first = next((i for i, (p, q) in enumerate(zip(x, y)) if p != q), None)
    print(f"pour {a} vs {b}: {'bitwise equal over %d steps' % len(x) if first is None else 'first differs at step %d' % first}")
PYEOF

# ---- 6. timing (L4 protocol), three interleaved repetitions
for rep in 1 2 3; do
  for t in base new; do
    for s in twoway_capture twoway anymal_capture; do
      case $s in
        twoway) a=(--scene example:mpm_twoway_coupling) ;;
        twoway_capture) a=(--scene example:mpm_twoway_coupling --set capture=true) ;;
        anymal_capture) a=(--scene example:mpm_anymal --set capture=true) ;;
      esac
      id=T_${s}_${t}_r$rep
      [ -f $R/timing/$id/meta.json ] && continue
      ${t}_env
      job cs_$id 15 -- $PY $W/metalsim/parity/isaac_side/record_mpm.py --device metal:0 --chunk-graph --gpu-exit --no-log \
          --seconds 2 --stride 1000 "${a[@]}" --out $R/timing/$id > $R/timing/$id.log 2>&1
      grep '^done' $R/timing/$id.log
    done
  done
done
python3 - "$R" <<'PYEOF'
import glob, json, statistics, sys
R = sys.argv[1]
L4 = {"twoway": 73.0, "twoway_capture": 31.0, "anymal_capture": 24.2}
print("| scene | L4 | base (runs) | new (runs) | base / L4 | new / L4 |")
print("|---|---|---|---|---|---|")
for s in ("twoway_capture", "twoway", "anymal_capture"):
    v = {}
    for t in ("base", "new"):
        v[t] = [statistics.median(json.load(open(f))["wall_s"]) * 1e3 for f in sorted(glob.glob(f"{R}/timing/T_{s}_{t}_r*/meta.json"))]
    mb, mn = statistics.median(v["base"]), statistics.median(v["new"])
    print(f"| {s} | {L4[s]} | {mb:.1f} ({' / '.join(f'{x:.1f}' for x in v['base'])}) | {mn:.1f} ({' / '.join(f'{x:.1f}' for x in v['new'])}) | {mb / L4[s]:.2f}x | {mn / L4[s]:.2f}x |")
PYEOF

# ---- 7. per-kernel profile, new trees, two-way late
if [ ! -f $R/profile/kprof_twoway_late_new.json ]; then
  mkdir -p $R/profile; new_env
  WP_METAL_PROFILE=1 job cs_kprof_late 20 -- env WP_METAL_PROFILE=1 $PY $W/metalsim/physics/mpm/probes/kernel_profile.py \
      example:mpm_twoway_coupling --mode profile --warmup 140 --frames 6 --json $R/profile/kprof_twoway_late_new.json \
      > $R/profile/kprof_twoway_late_new.log 2>&1
fi
mkdir -p $R/profile/join_late_new && cp $R/profile/kprof_twoway_late_new.json $R/profile/join_late_new/profile_twoway_coupling.json
rm -f $R/profile/joined_late_new.md
(cd $R/profile && python3 $W/runs/mpm/profile/kernels_l4/kjoin.py $W/runs/mpm/profile/kernels_l4 join_late_new twoway_coupling 4 \
    joined_late_new.md twoway_coupling_capture > /dev/null)
head -12 $R/profile/joined_late_new.md

# ---- 8. commit and push the logs
cd $W
git add runs/mpm/coupled_speed metalsim/physics/mpm/probes/rigidity_mv_probe.py metalsim/physics/mpm/probes/bsr_mv_tiles_check.py \
    runs/mpm/coupled_retime/batch_01.log
git commit -s -q -m "Coupled speed, step 1: tiled IJtm product validated and timed on Metal (logs)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" || echo "(nothing to commit)"
git push -q origin mpm-coupled-speed 2>&1 | grep -v '^remote:' ; git log --oneline -1
echo "== $(date) batch_02 done"

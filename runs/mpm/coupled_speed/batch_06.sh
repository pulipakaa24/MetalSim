#!/bin/bash
# Coupled-scene speed, step 1, finishing the validation (2026-10-13). No fork pushes, no pin changes.
#   A  tests/test_mpm.py on base and new trees with `python -P` (batch_02-05 ran `python -m pytest` from the main checkout,
#      where gpu_run.sh starts every job; -m puts the cwd ahead of PYTHONPATH, so those runs imported the MAIN
#      checkout's metalsim, an older newton_compat without the coupled-solver determinism: the "failing"
#      test_coupled_contact_filter_keeps_source_order was that, not a bug). Plus the single test x3 per tree.
#   B  Franka-Pour: new trees with WP_METAL_BSR_MV_TILES=0 must equal the base trees bitwise (then the base-vs-new
#      difference from step 0 is the tiled kernel choice and nothing else)
#   C  timing, three interleaved repetitions (batch_05 never ran it: its output directory was missing)
#   D  dry run: the Newton patch applied to the Newton fork's GS-colouring head 99843d09 (clone, `git apply --check`)
#   E  commit and push the logs (branch mpm-coupled-speed)
# Only GPU-health problems stop the batch; failed steps go to batch_06_failures.txt. Resumable.
set -u
W=.
R=$W/runs/mpm/coupled_speed
M=.
PY=$M/.venv-newton152/bin/python
PYP=$M/.claude/worktrees/agent-a278824da4c05bf29/.venv-pour/bin/python
GR=$M/scripts/gpu_run.sh
T=$W/upstream/trees
WI=$M/upstream/warp-innate; NI=$M/upstream/newton-metalsim-1.5.2
F=$R/batch_06_failures.txt
mkdir -p $R/timing $R/tests_P
exec > >(tee -a "$R/batch_06.log") 2>&1
echo "== $(date) batch_06 start"
export METALSIM_ROOT=$M
stop() { echo "STOP: $*"; echo "== $(date) batch_06 stopped"; exit 1; }
health() {
  [ -e $M/runs/GPU_RUNAWAY_ALERT ] && stop "runs/GPU_RUNAWAY_ALERT exists"
  local u
  for i in $(seq 1 60); do u=$($M/scripts/gpu_util.sh); [ "${u:-0}" -lt 10 ] && return 0; sleep 2; done
  python3 $M/scripts/gpu_top.py 2>/dev/null | grep -q "none >= 1 %" && stop "GPU at ${u}% with no owning process after '$1'"
  echo "note: GPU at ${u}% after '$1', owned by another process"
}
sjob() {  # sjob NAME MINUTES LOG -- cmd...  (exclusive timing job; output to LOG; failure recorded, batch continues)
  local name=$1 min=$2 log=$3; shift 4
  $GR "$name" timing "$min" -- "$@" > "$log" 2>&1; local rc=$?
  health "$name"
  [ $rc -eq 0 ] || { echo "FAIL: $name exit $rc ($log)"; echo "$name exit $rc" >> $F; }
  return 0
}
base_env() { export PYTHONPATH=$WI:$NI:$W WARP_CACHE_PATH=$R/kcache_base; unset WP_METAL_BSR_MV_TILES; }
new_env() { export PYTHONPATH=$T/warp_new:$T/newton_new:$W WARP_CACHE_PATH=$R/kcache_new; unset WP_METAL_BSR_MV_TILES; }
result() { grep -E '[0-9]+ (passed|failed)' "$1" | tail -1; }
[ -f $T/.trees_ok ] || stop "trees missing (run batch_02 phase 0)"

# ---- A. tests with -P
for t in base new; do
  L=$R/tests_P/suite_$t.log
  if ! grep -qE '[0-9]+ (passed|failed)' $L 2>/dev/null; then
    ${t}_env; sjob cs6_suite_$t 30 $L -- $PY -P -m pytest -q -rf -p no:cacheprovider $W/tests/test_mpm.py
  fi
  echo "suite $t: $(result $L)"
  $PY -P -c "import metalsim; print('metalsim from', metalsim.__file__)" | tee -a $R/tests_P/imports_$t.txt
  for i in 1 2 3; do
    L=$R/tests_P/cf_${t}_$i.log
    grep -qE '[0-9]+ (passed|failed)' $L 2>/dev/null && continue
    sjob cs6_cf_${t}_$i 10 $L -- $PY -P -m pytest -q -p no:cacheprovider $W/tests/test_mpm.py -k test_coupled_contact_filter_keeps_source_order
    echo "contact filter $t $i: $(result $L)"
  done
done

# ---- B. Franka-Pour: new trees with the tiled choice off == base?
FB=$W/scripts/diagnostics/franka_pour_memory/franka_bitwise.py
if [ ! -f $R/pour_hash_new_notiles.json ]; then
  new_env; export WP_METAL_BSR_MV_TILES=0
  sjob cs6_pour_notiles 30 $R/pour_hash_new_notiles.log -- $PYP $FB 64 30 $R/pour_hash_new_notiles.json
  unset WP_METAL_BSR_MV_TILES
fi
python3 - "$R" <<'PYEOF' || echo "FAIL: pour comparison"
import json, sys
R = sys.argv[1]
def h(t): return json.load(open(f"{R}/pour_hash_{t}.json"))["hashes"]
for a, b in (("base", "new_notiles"), ("base", "new"), ("new", "new2")):
    x, y = h(a), h(b)
    first = next((i for i, (p, q) in enumerate(zip(x, y)) if p != q), None)
    print(f"pour {a} vs {b}: " + (f"bitwise equal over {len(x)} steps" if first is None else f"first differs at step {first}"))
PYEOF

# ---- C. timing (L4 protocol), three interleaved repetitions
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
      sjob cs6_$id 15 $R/timing/$id.log -- $PY $W/metalsim/parity/isaac_side/record_mpm.py --device metal:0 --chunk-graph \
          --gpu-exit --no-log --seconds 2 --stride 1000 "${a[@]}" --out $R/timing/$id
      grep '^done' $R/timing/$id.log
    done
  done
done
python3 - "$R" <<'PYEOF' | tee $R/timing/summary.md
import glob, json, statistics, sys
R = sys.argv[1]
L4 = {"twoway": 73.0, "twoway_capture": 31.0, "anymal_capture": 24.2}
print("| scene | L4 | base (runs) | new (runs) | base / L4 | new / L4 |")
print("|---|---|---|---|---|---|")
for s in ("twoway_capture", "twoway", "anymal_capture"):
    v = {t: [statistics.median(json.load(open(f))["wall_s"]) * 1e3
             for f in sorted(glob.glob(f"{R}/timing/T_{s}_{t}_r*/meta.json"))] for t in ("base", "new")}
    if not v["base"] or not v["new"]:
        print(f"| {s} | missing runs | {v} |"); continue
    mb, mn = statistics.median(v["base"]), statistics.median(v["new"])
    print(f"| {s} | {L4[s]} | {mb:.1f} ({' / '.join(f'{x:.1f}' for x in v['base'])}) | {mn:.1f} "
          f"({' / '.join(f'{x:.1f}' for x in v['new'])}) | {mb / L4[s]:.2f}x | {mn / L4[s]:.2f}x |")
PYEOF

# ---- D. dry run of the Newton patch on the GS-colouring head
D=$T/newton_gs_head
if [ ! -d $D/.git ]; then git clone -q https://github.com/pulipakaa24/newton.git $D || echo "FAIL: clone" | tee -a $F; fi
if [ -d $D/.git ]; then
  git -C $D fetch -q origin && git -C $D checkout -q 99843d09 2>&1 && echo "newton head $(git -C $D rev-parse HEAD)"
  git -C $D apply --check $R/patches/newton_ijtm_tiled.diff && echo "newton patch applies on 99843d09" \
      || { echo "FAIL: newton patch does not apply on 99843d09"; echo "patch on 99843d09" >> $F; }
fi

# ---- E. commit and push the logs
cd $W
git add runs/mpm/coupled_speed research_notes/coupled_speed
git commit -s -q -m "Coupled speed: test harness fix (-P), Franka-Pour tiling check, timing (logs)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" || echo "(nothing to commit)"
git push -q origin mpm-coupled-speed 2>&1 | grep -v '^remote:'; git log --oneline -1
echo "failures: $(cat $F 2>/dev/null | tr '\n' ';')"
echo "== $(date) batch_06 done"

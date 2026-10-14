#!/bin/bash
# SolverImplicitMPM's ordered proxy-wrench harvest (Metal): compact the proxy-body collider nodes first
# (patches/newton_harvest_compact.diff, Newton fork solver_implicit_mpm.py; the vendored copy takes the same patch).
# Paths it serves: Franka-Pour (SolverCoupledProxy, separate worlds) and the main simulator's MPMCoupling (one BatchSim
# world = every collider node scanned per body before). Toggle: NEWTON_METAL_HARVEST_COMPACT=0 (full scan, as before).
# Prerequisites (checked): batch_02's trees (upstream/trees/warp_new, newton_new); the MPM-main worktree with
# metalsim/physics/mpm/probes/multibody_sand.py. Independent of batch_08 / bf1 / any1 (run one after another).
#   0  trees: newton_hv = newton_new + body-force + harvest patches; mainsim_hv = copy of the MPM-main worktree's
#      metalsim package with the harvest patch applied to its vendored solver
#   1  kernel check on Metal under a watchdog (probes/harvest_compact_check.py): 1 world x 16 / 64 bodies, 64 worlds
#   2  MPMCoupling scene (multibody_sand.py): bodies 4 / 16 / 64, compaction off vs on, 2 interleaved reps
#      (median coupled-step ms) + harvest-only profile; state hashes: on twice (repeatability), off once
#   3  Franka-Pour (64 envs, 30 steps): hashes on twice (repeatability), off once (= batch_02 trees: expected equal)
#      and the 2,048-env policy step on vs off, 2 interleaved reps
#   4  commit and push the logs
# Only GPU-health problems stop it; failures listed in hv1_failures.txt. Resumable. Log: batch_hv1.log
set -u
W=/Users/aditya/robosim/.claude/worktrees/agent-a8f809a11de334dd0
R=$W/runs/mpm/coupled_speed
O=$R/hv1
M=/Users/aditya/robosim
PY=$M/.venv-newton152/bin/python
PYM=$M/.venv/bin/python
PYP=$M/.claude/worktrees/agent-a278824da4c05bf29/.venv-pour/bin/python
MW=$M/.claude/worktrees/agent-ab17e9d0efefed741
GR=$M/scripts/gpu_run.sh
T=$W/upstream/trees
F=$O/hv1_failures.txt
mkdir -p $O
exec > >(tee -a "$R/batch_hv1.log") 2>&1
echo "== $(date) batch_hv1 start"
export METALSIM_ROOT=$M
stop() { echo "STOP: $*"; echo "== $(date) batch_hv1 stopped"; exit 1; }
health() {
  [ -e $M/runs/GPU_RUNAWAY_ALERT ] && stop "runs/GPU_RUNAWAY_ALERT exists"
  local u
  for i in $(seq 1 60); do u=$($M/scripts/gpu_util.sh); [ "${u:-0}" -lt 10 ] && return 0; sleep 2; done
  python3 $M/scripts/gpu_top.py 2>/dev/null | grep -q "none >= 1 %" && stop "GPU at ${u}% with no owning process after '$1'"
  echo "note: GPU at ${u}% after '$1', owned by another process"
}
sjob() { local name=$1 min=$2 log=$3; shift 4; $GR "$name" timing "$min" -- "$@" > "$log" 2>&1; local rc=$?; health "$name"
         [ $rc -eq 0 ] || { echo "FAIL: $name exit $rc ($log)"; echo "$name exit $rc" >> $F; }; return 0; }
first_diff() { python3 - "$1" "$2" <<'PYEOF'
import json, sys
a, b = (json.load(open(p))["hashes"] for p in sys.argv[1:3])
i = next((k for k, (x, y) in enumerate(zip(a, b)) if x != y), None)
print(f"bitwise equal over {len(a)} steps" if i is None else f"first differs at step {i}")
PYEOF
}

# ---- 0. trees
[ -f $T/.trees_ok ] || stop "batch_02 trees missing"
[ -f $MW/metalsim/physics/mpm/probes/multibody_sand.py ] || stop "MPM-main probe multibody_sand.py missing"
if [ ! -f $T/.hv_ok ]; then
  rm -rf $T/newton_hv $T/mainsim_hv
  mkdir -p $T/newton_hv $T/mainsim_hv
  cp -R $T/newton_new/newton $T/newton_hv/ || stop copy
  patch -p1 -d $T/newton_hv -i $R/patches/newton_body_force_compact.diff || stop "patch bf"
  patch -p1 -d $T/newton_hv -i $R/patches/newton_harvest_compact.diff || stop "patch harvest"
  cp -R $MW/metalsim $T/mainsim_hv/ || stop "copy mainsim"
  patch $T/mainsim_hv/metalsim/physics/mpm/implicit_mpm/solver_implicit_mpm.py -i $R/patches/newton_harvest_compact.diff \
      || stop "patch vendored solver"
  echo "MPM-main worktree $(git -C $MW rev-parse --short HEAD) $(git -C $MW status --porcelain | wc -l | tr -d ' ') local changes" | tee $O/mainsim_source.txt
  touch $T/.hv_ok
fi

# ---- 1. kernel check
if ! grep -q '^OK' $O/kernel_check.log 2>/dev/null; then
  export PYTHONPATH=$T/warp_new:$T/newton_hv WARP_CACHE_PATH=$R/kcache_hv
  sjob chv_check 10 $O/kernel_check_job.log -- python3 $W/runs/rigid17/tools/watch.py --log $O/kernel_check_watch.log --timeout 400 \
      --stall 60 -- bash -c "$PY -P $W/metalsim/physics/mpm/probes/harvest_compact_check.py --worlds 1 --bodies 16 &&
          $PY -P $W/metalsim/physics/mpm/probes/harvest_compact_check.py --worlds 1 --bodies 64 &&
          $PY -P $W/metalsim/physics/mpm/probes/harvest_compact_check.py --worlds 64 --bodies 2 --nodes-per-world 8000 --contact-per-world 700"
  grep -v '^Module' $O/kernel_check_watch.log | grep -v 'Kernel cache\|initialized\|Devices\|"cpu"\|"metal\|CUDA not\|^ *$\|/1.18'
  [ "$(grep -c '^OK' $O/kernel_check_watch.log)" = 3 ] && cp $O/kernel_check_watch.log $O/kernel_check.log
  grep -q '^OK' $O/kernel_check.log 2>/dev/null || stop "harvest kernel check failed (kernel_check_watch.log)"
fi

# ---- 2. MPMCoupling scene
export PYTHONPATH=$T/warp_new:$T/mainsim_hv WARP_CACHE_PATH=$R/kcache_mainsim
MS=$T/mainsim_hv/metalsim/physics/mpm/probes/multibody_sand.py
for rep in 1 2; do
  for nb in 4 16 64; do
    for c in 0 1; do
      id=ms_b${nb}_c${c}_r$rep
      [ -f $O/$id.json ] && continue
      sjob chv_$id 15 $O/$id.log -- env NEWTON_METAL_HARVEST_COMPACT=$c $PYM -P $MS --bodies $nb --steps 60 --warmup 10 --json $O/$id.json
      tail -1 $O/$id.log
    done
  done
done
for nb in 16 64; do for c in 0 1; do
  id=ms_profile_b${nb}_c$c
  [ -f $O/$id.log ] && grep -q 'harvest only' $O/$id.log && continue
  sjob chv_$id 10 $O/$id.log -- env NEWTON_METAL_HARVEST_COMPACT=$c $PYM -P $MS --bodies $nb --profile
  grep -i 'harvest only\|coupled step' $O/$id.log | sed "s/^/$id: /"
done; done
for t in on1 on2 off; do
  [ -f $O/ms_hash_$t.json ] && continue
  c=1; [ $t = off ] && c=0
  sjob chv_mshash_$t 15 $O/ms_hash_$t.log -- env NEWTON_METAL_HARVEST_COMPACT=$c $PYM -P $W/metalsim/physics/mpm/probes/mainsim_hash.py \
      --bodies 16 --steps 30 --json $O/ms_hash_$t.json
done
echo "MPMCoupling on vs on: $(first_diff $O/ms_hash_on1.json $O/ms_hash_on2.json)"
echo "MPMCoupling on vs off: $(first_diff $O/ms_hash_on1.json $O/ms_hash_off.json)"
python3 - "$O" <<'PYEOF' | tee $O/mainsim_summary.md
import glob, json, statistics, sys
O = sys.argv[1]
print("| bodies | off: ms/step (runs) | on: ms/step (runs) | on / off |")
print("|---|---|---|---|")
for nb in (4, 16, 64):
    v = {c: [json.load(open(f))["ms_per_step_median"] for f in sorted(glob.glob(f"{O}/ms_b{nb}_c{c}_r*.json"))] for c in (0, 1)}
    if v[0] and v[1]:
        a, b = statistics.median(v[0]), statistics.median(v[1])
        print(f"| {nb} | {a:.2f} ({' / '.join(f'{x:.2f}' for x in v[0])}) | {b:.2f} ({' / '.join(f'{x:.2f}' for x in v[1])}) | {b / a:.2f} |")
PYEOF

# ---- 3. Franka-Pour
FB=$W/scripts/diagnostics/franka_pour_memory/franka_bitwise.py
export PYTHONPATH=$T/warp_new:$T/newton_hv:$W WARP_CACHE_PATH=$R/kcache_hv
for t in on1 on2 off; do
  [ -f $O/pour_hash_$t.json ] && continue
  c=1; [ $t = off ] && c=0
  sjob chv_pour_$t 30 $O/pour_hash_$t.log -- env NEWTON_METAL_HARVEST_COMPACT=$c $PYP -P $FB 64 30 $O/pour_hash_$t.json
done
echo "Franka-Pour on vs on: $(first_diff $O/pour_hash_on1.json $O/pour_hash_on2.json)"
echo "Franka-Pour off vs batch_02 trees (expected equal): $(first_diff $O/pour_hash_off.json $R/pour_hash_new.json)"
echo "Franka-Pour on vs off: $(first_diff $O/pour_hash_on1.json $O/pour_hash_off.json)"
for rep in 1 2; do for c in 1 0; do
  id=pour2048_c${c}_r$rep
  [ -f $O/$id.json ] && continue
  sjob chv_$id 20 $O/$id.log -- env NEWTON_METAL_HARVEST_COMPACT=$c $PYP -P -m metalsim.learn.franka_pour_ppo bench --num-envs 2048 \
      --mode step --steps 15 --warmup 3 --json $O/$id.json
  python3 -c "import json; d=json.load(open('$O/$id.json')); print('$id', round(d['step_s_median'], 3), 's/step')" 2>/dev/null
done; done

# ---- 4. commit and push
cd $W
git add runs/mpm/coupled_speed metalsim/physics/mpm/probes/harvest_compact_check.py metalsim/physics/mpm/probes/mainsim_hash.py
git commit -s -q -m "Coupled speed: compacted proxy-wrench harvest validated and timed (Franka-Pour, MPMCoupling) (logs)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" || echo "(nothing to commit)"
git push -q origin mpm-coupled-speed 2>&1 | grep -v '^remote:'; git log --oneline -1
echo "failures: $(cat $F 2>/dev/null | tr '\n' ';')"
echo "== $(date) batch_hv1 done"

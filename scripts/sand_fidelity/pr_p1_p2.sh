#!/bin/bash
# Open P1 and P2 as their own draft PRs (research_notes/sand_fidelity/PLAN_2026-10-14.md section 13). Run outside the
# agent sandbox, alone, BEFORE batch 02c (02c's rest runs use the amended P2):   bash scripts/sand_fidelity/pr_p1_p2.sh
#
#   P2  In the fork worktree upstream/warp-innate-p2 (branch metalsim-metal-scratch-bytes, f8a1d679): amend the growth
#       policy back to doubling (scripts/sand_fidelity/p2/amend_growth.py; 8-byte elements kept), run the fork's Metal
#       primitive / sparse / volume unit tests and the opt-in > 2 GiB sort test (exclusive: ~7 GB), commit, push the
#       branch to the fork remote (pulipakaa24/warp), open a draft PR into its branch metalsim.
#   P1  A worktree upstream/metalsim-p1-pr on a new branch mpm-solver-variant-pr4660 from origin/main; cherry-pick the
#       P1 commit 73530b84 (flag-gated Newton #4660 variant, default off); CPU checks (variant regeneration, the
#       variant tests on CPU, no machine paths); GPU: the variant tests on Metal and both parity suites, on main's
#       code; push; draft PR into main.
# Each part stops at its first failure and pushes nothing. Resumable (STATUS.txt). Expected ~12 min of GPU: P2 unit
# 26 s + large 192 s (logged in the P2 batch); P1 variant tests 114 s, parity_main 240 s, parity_152 113 s (logged in
# 02b), plus ~20 s per job.
set -u
W=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
R=$(dirname "$(git -C "$W" rev-parse --path-format=absolute --git-common-dir)")
PY=$R/.venv/bin/python
PY152=$R/.venv-newton152/bin/python
WT_P2=$R/upstream/warp-innate-p2
P2_BR=metalsim-metal-scratch-bytes
P1WT=$R/upstream/metalsim-p1-pr
P1_BR=mpm-solver-variant-pr4660
P1_COMMIT=73530b84
OUT=$W/runs/sand_fidelity/pr_p1_p2
L=$OUT/logs
STATUS=$OUT/STATUS.txt
export GIT_AUTHOR_NAME="Aditya Pulipaka" GIT_AUTHOR_EMAIL="adipu@utexas.edu"
export GIT_COMMITTER_NAME="Aditya Pulipaka" GIT_COMMITTER_EMAIL="adipu@utexas.edu"
mkdir -p "$L"; touch "$STATUS"
exec > >(tee -a "$L/batch.log") 2>&1
say() { echo "[$(date +%H:%M:%S)] $*"; }
done_() { grep -qE "^$1 (PASS|DONE)( |$)" "$STATUS"; }
mark() { echo "$1 $2 $(date +%H:%M:%S) ${3:-}" >> "$STATUS"; say "$1 -> $2 ${3:-}"; }
TRAILER="Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
echo "=== P1/P2 PRs start $(date)"

job() {   # job NAME KIND EST CWD -- cmd...   (returns the command's rc; exits on an anomaly)
  local name=$1 kind=$2 est=$3 cwd=$4; shift 5
  local tmo=$(( est * 3 > 600 ? est * 3 : 600 )) rc
  say "gpu job $name ($kind, estimate $est s) -> logs/$name.log"
  "$R/scripts/gpu_run.sh" "prp_$name" "$kind" $(( tmo / 60 + 5 )) -- python3 "$W/scripts/watchdog.py" --stall 180 \
      --rss-gb 40 --timeout "$tmo" --cwd "$cwd" -- "$@" > "$L/$name.log" 2>&1 < /dev/null
  rc=$?
  tail -3 "$L/$name.log" | sed 's/^/    /'
  if [ $rc -ge 124 ] && [ $rc -le 126 ] || [ $rc -eq 75 ]; then mark "$name" ANOMALY "rc=$rc"; exit 91; fi
  return $rc
}

# ------------------------------------------------------------------------------------------------ P2
p2() {
  if ! done_ P2_AMEND; then
    [ -d "$WT_P2" ] && git -C "$WT_P2" merge-base --is-ancestor f8a1d679 HEAD || { mark P2_AMEND FAIL "no P2 worktree at f8a1d679"; return 1; }
    "$PY" "$W/scripts/sand_fidelity/p2/amend_growth.py" "$WT_P2" && "$PY" -m py_compile "$WT_P2/warp/_src/metal_primitives.py" \
      && mark P2_AMEND PASS || { mark P2_AMEND FAIL; return 1; }
  fi
  if ! done_ P2_UNIT; then
    job p2_unit check 60 "$WT_P2" -- env PYTHONPATH="$WT_P2" "$PY" -P -m unittest warp.tests.test_metal_primitives \
        warp.tests.test_metal_sparse warp.tests.geometry.test_volume_metal && mark P2_UNIT PASS || { mark P2_UNIT FAIL; return 1; }
  fi
  if ! done_ P2_LARGE; then
    job p2_large train 200 "$WT_P2" -- env PYTHONPATH="$WT_P2" WARP_METAL_LARGE_TESTS=1 "$PY" -P -m unittest -v \
        warp.tests.test_metal_primitives.TestMetalPrimitives.test_radix_sort_pairs_scratch_over_2gib \
      && grep -q "^OK" "$L/p2_large.log" && mark P2_LARGE PASS || { mark P2_LARGE FAIL; return 1; }
  fi
  if ! done_ P2_COMMIT; then
    git -C "$WT_P2" add warp/_src/metal_primitives.py warp/tests/test_metal_primitives.py
    git -C "$WT_P2" diff --cached --quiet || git -C "$WT_P2" commit -q -s -m "Metal scratch pool: grow by doubling again" -m \
"The pool keeps outgrown buffers alive (captured graphs may reference
them), so a 1.25x growth step retained a chain of ~5x the largest buffer:
an S2 rasterisation at 1.57M particles reached 57.7 GB against a 55.7 GB
working set. Doubling keeps the chain at ~2x. Buffers stay in 8-byte
elements, so they may still exceed 2 GiB." -m "$TRAILER" || { mark P2_COMMIT FAIL; return 1; }
    mark P2_COMMIT PASS "$(git -C "$WT_P2" rev-parse --short HEAD)"
  fi
  if ! done_ P2_PR; then
    git -C "$WT_P2" push -q fork "$P2_BR" 2>&1 | grep -v "^remote:" | tail -3
    git -C "$WT_P2" ls-remote fork "refs/heads/$P2_BR" | grep -q "$(git -C "$WT_P2" rev-parse HEAD)" || { mark P2_PUSH FAIL; return 1; }
    cat > "$OUT/p2_pr_body.md" <<'EOF'
Metal scratch pool (`warp/_src/metal_primitives.py`, `_DeviceState.scratch_ptr`): allocate in 8-byte elements so a
scratch buffer may exceed 2 GiB (Warp array shapes are int32; a byte-element pool failed above 2 GiB), growing by
doubling (outgrown buffers stay alive for captured graphs, so the retained chain stays ~2x the largest buffer). The
Metal volume builder's state subclasses `_DeviceState` and shares the pool. Scratch contents are undefined, so sort
results are unchanged.

Found by MetalSim's sand-fidelity work: implicit-MPM S2 collider rasterisation at 1.57M particles needed a radix-sort
scratch above 2 GiB.

Tests: `test_metal_primitives` / `test_metal_sparse` / `geometry.test_volume_metal` pass on Metal; new
`test_scratch_pool_over_2gib`; opt-in `test_radix_sort_pairs_scratch_over_2gib` (WARP_METAL_LARGE_TESTS=1, ~7 GB):
a 280M-pair sort with 2.24 GB of scratch matches the host order.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
    ( cd "$WT_P2" && gh pr create --repo pulipakaa24/warp --draft --base metalsim --head "$P2_BR" \
        --title "Metal scratch pool: 8-byte elements (buffers above 2 GiB), doubling growth" \
        --body-file "$OUT/p2_pr_body.md" ) > "$L/p2_pr.log" 2>&1
    cat "$L/p2_pr.log"; grep -q "github.com" "$L/p2_pr.log" && mark P2_PR PASS "$(tail -1 "$L/p2_pr.log")" || mark P2_PR FAIL
  fi
}

# ------------------------------------------------------------------------------------------------ P1
p1() {
  if ! done_ P1_BRANCH; then
    git -C "$W" fetch -q origin main || { mark P1_BRANCH FAIL "fetch"; return 1; }
    [ -d "$P1WT" ] || git -C "$W" worktree add -q -B "$P1_BR" "$P1WT" origin/main || { mark P1_BRANCH FAIL "worktree"; return 1; }
    if ! git -C "$P1WT" merge-base --is-ancestor "$P1_COMMIT" HEAD; then
      git -C "$P1WT" cherry-pick "$P1_COMMIT" > "$L/p1_cherry.log" 2>&1 \
        || { git -C "$P1WT" cherry-pick --abort; mark P1_BRANCH FAIL "cherry-pick conflicts onto main (logs/p1_cherry.log)"; return 1; }
    fi
    mark P1_BRANCH PASS "$(git -C "$P1WT" rev-parse --short HEAD) on $P1_BR from origin/main"
  fi
  if ! done_ P1_CPU; then
    ( cd "$P1WT" && "$PY" -P scripts/vendor_mpm_variant.py --check && env METALSIM_VARIANT_TESTS_METAL=0 "$PY" -P -m pytest -q \
        -p no:cacheprovider tests/test_mpm_variant.py tests/test_mpm_variant_pr4660.py tests/test_no_machine_paths.py ) \
        > "$L/p1_cpu.log" 2>&1 && mark P1_CPU PASS || { mark P1_CPU FAIL "logs/p1_cpu.log"; return 1; }
  fi
  if ! done_ P1_GPU; then
    job p1_metal check 114 "$P1WT" -- env PYTHONPATH="$P1WT" METALSIM_VARIANT_TESTS_METAL=1 METALSIM_VARIANT_TESTS_CPU=0 \
        "$PY" -P -m pytest -q -p no:cacheprovider tests/test_mpm_variant.py tests/test_mpm_variant_pr4660.py || { mark P1_GPU FAIL "p1_metal"; return 1; }
    job p1_parity_main check 240 "$P1WT" -- env PYTHONPATH="$P1WT" "$PY" -P -m pytest -q -p no:cacheprovider tests/test_mpm_main.py \
      || { mark P1_GPU FAIL "parity_main"; return 1; }
    job p1_parity_152 check 113 "$P1WT" -- env PYTHONPATH="$P1WT" "$PY152" -P -m pytest -q -p no:cacheprovider tests/test_mpm.py \
      || { mark P1_GPU FAIL "parity_152"; return 1; }
    mark P1_GPU PASS
  fi
  if ! done_ P1_PR; then
    git -C "$P1WT" push -q origin "$P1_BR" 2>&1 | grep -v "^remote:" | tail -3
    git -C "$P1WT" ls-remote origin "refs/heads/$P1_BR" | grep -q "$(git -C "$P1WT" rev-parse HEAD)" || { mark P1_PUSH FAIL; return 1; }
    cat > "$OUT/p1_pr_body.md" <<'EOF'
Flag-gated implicit-MPM solver variant with Newton draft PR #4660 ("Improve implicit MPM strain and collider
robustness", head 1de47831): `MPMConfig(solver_variant="pr4660")`. Default `"parity"` is the vendored Newton 1.5.2
fork solver (Isaac Lab's), untouched and not imported alongside.

* `scripts/vendor_mpm_variant.py` generates `metalsim/physics/mpm/implicit_mpm_pr4660/` from the vendored copy plus
  `patches/newton/pr4660-1de47831-implicit-mpm.diff` (the PR's four solver files; every hunk applies without fuzz);
  `tests/test_mpm_variant.py` regenerates and compares, checks headers, selection, and that the parity path never loads
  the variant.
* `tests/test_mpm_variant_pr4660.py`: the PR's own tests against the variant (CPU and Metal).
* Parity suites (`tests/test_mpm_main.py`, `tests/test_mpm.py`) pass unchanged on this branch.

Why: in MetalSim's sand-fidelity batches (PRs #64/#67 and the 02b batch), the variant with P1d strain lifts the
lifted-cylinder heap from 28.2 to 30.7 deg at R/dx 20 (atan(mu) = 34.2), puts a dropped sphere at Uehara's
penetration (16.3 mm), and removes a particle leak at a plane / frictionless-wall junction that blew up a tilted-box
test (the PR's collider-sign fix). Notices: THIRD_PARTY_NOTICES.md.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
    ( cd "$P1WT" && gh pr create --draft --base main --head "$P1_BR" \
        --title "MPM: flag-gated solver variant with Newton PR #4660 (default off)" --body-file "$OUT/p1_pr_body.md" ) \
        > "$L/p1_pr.log" 2>&1
    cat "$L/p1_pr.log"; grep -q "github.com" "$L/p1_pr.log" && mark P1_PR PASS "$(tail -1 "$L/p1_pr.log")" || mark P1_PR FAIL
  fi
}

p2 || say "P2 stopped (see STATUS)"
p1 || say "P1 stopped (see STATUS)"
say "=== end $(date)"; cat "$STATUS"

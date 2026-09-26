#!/bin/zsh
# Queued (kind low): Kamino on Metal with Newton's own switch NEWTON_KAMINO_DISABLE_TILE_TRANSPOSE_UPDATE=1.
# Why: the LLTB backward solve's native snippet (_tile_builtins.make_tile_matmul_left_transpose_update_func) stages
# the tiles in `__shared__` arrays only under __CUDA_ARCH__; on Metal the #else branch declares per-thread arrays
# that each thread fills only with its own tile registers, so the transpose-matmul reads uninitialised values
# (LLTB solve tests: x error up to 2.3 on Metal, pass on CPU). The switch selects the generic tile_transpose +
# tile_matmul path (same arithmetic).
cd "$(dirname "$0")/../../.."
python3 scripts/gpu_lock.py status
export NEWTON_KAMINO_DISABLE_TILE_TRANSPOSE_UPDATE=1
O=runs/closed_loops/kamino; mkdir -p $O
K=scripts/diagnostics/closed_loops/kamino_probe.py; TO=(python3 scripts/diagnostics/closed_loops/run_to.py)
F=.venv-newtonfork/bin/python
${TO[@]} 900 $F scripts/diagnostics/closed_loops/run_kamino_tests.py --device metal:0 --modules test_kamino_linalg_factorize_llt_blocked test_kamino_linalg_solver_llt_blocked test_kamino_linalg_solver_llt_blocked_rcm test_kamino_solver_kamino test_kamino_solvers_padmm
${TO[@]} 600 $F $K --mech fourbar --device metal:0 --dt 0.0025 --T 5 --out $O/fork_fourbar_none_dt0.0025_metal_nottu.npz
${TO[@]} 600 $F $K --mech fourbar --device metal:0 --dt 0.0025 --T 5 --alpha 0.5 --out $O/fork_fourbar_none_dt0.0025_metal_alpha0.5_nottu.npz
${TO[@]} 600 $F $K --mech leg --device metal:0 --dt 0.0025 --T 2 --out $O/fork_leg_none_dt0.0025_metal_nottu.npz
${TO[@]} 900 $F $K --mech leg --device metal:0 --dt 0.0025 --T 2 --worlds 1024 --torque sigma3 --alpha 0.5 --out $O/fork_leg_sigma3_w1024_dt0.0025_metal_alpha0.5_nottu.npz
${TO[@]} 600 $F $K --mech leg --device metal:0 --dt 0.0025 --T 0.05 --worlds 1024 --bench_steps 200 --capture 1
echo "kamino_jobs2 done"

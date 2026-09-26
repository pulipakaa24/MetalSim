# Cross-engine benchmarks (MetalSim vs Genesis vs MuJoCo-MLX-Cpp)

Findings: `docs/HANDOFF_elliptic_cone_perf.md` (the elliptic-cone issue) and `docs/research/competitors_2026-09-25.md`
(the full survey and every measured result). Logs: `runs/competitors/`. `*_contacts.py`: contacts per world.

```
git clone --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git scripts/diagnostics/competitors/menagerie
git -C scripts/diagnostics/competitors/menagerie sparse-checkout set unitree_go2 unitree_g1
cd scripts/diagnostics/competitors
python metalsim_step.py go2 4096 matched                  # .venv; mode matched = Newton 10 / 20 iterations, default = model's
CONE=pyramidal python metalsim_step.py go2 4096 matched   # override the model's friction cone
python genesis_step.py go2 4096 matched                   # separate venv with genesis-world 1.4.2 (Python <= 3.13)
WP_METAL_PROFILE=1 python metalsim_profile.py go2 4096 kernels
python metalsim_profile.py go2 4096 variants              # floor-only contacts / capsules instead of cylinders
python metalsim_mlxprotocol.py <model.xml> <name> 4096    # MuJoCo-MLX-Cpp's bench_batched_step_file protocol
```

Run timing jobs through `scripts/gpu_run.sh NAME timing MINUTES -- ...` with absolute paths (the wrapper
changes to the repository root). Genesis ignores an MJCF's elliptic cone unless `CONE=elliptic` is set.

## Elliptic-cone work (2026-09-25 evening, `docs/research/elliptic_cones_2026-09-25.md`)

```
scripts/gpu_run.sh NAME timing 10 -- scripts/diagnostics/competitors/elliptic_bench.sh MJW_WORKTREE LABEL [go2 g1 so101 humanoid]
   # pyramidal + elliptic throughput and the elliptic per-kernel profile; env: MJW_JTCJ_MODE, JAC, NJMAX, CONES, PROFILE, BENCH
scripts/gpu_run.sh NAME render 8 -- scripts/diagnostics/competitors/elliptic_check.sh MJW_WORKTREE LABEL [REF_LABEL]
   # Metal physics-difference protocol (floor, vs a reference snapshot, MuJoCo C oracle); elliptic_check_compare.py A.npz B.npz offline
python scripts/diagnostics/competitors/elliptic_cpu_check.py [steps] [models]      # Warp CPU device vs mj_step per MJW_JTCJ_MODE (no GPU)
python scripts/diagnostics/competitors/elliptic_cpu_time.py [nworld] [steps]       # CPU-device step time per mode
scripts/diagnostics/competitors/elliptic_fidelity_g1.sh MJW_WORKTREE rec|transfer|air|cost   # G1 fidelity protocol for the elliptic presets
python scripts/diagnostics/competitors/so101_creep.py --engine c|warp             # SO-101 in-hand creep per cone / impratio
```
`elliptic_presets.py` registers the elliptic variants of the adopted G1 preset and runs another script. `MENAGERIE`
defaults to `scripts/diagnostics/competitors/menagerie`; this machine's checkout is `upstream/mujoco_menagerie`.
The scripts write `ctrl` and reset state through Warp after `sim.synchronize()` (a torch/MPS write into the shared
buffer races with a still-running graph; `metalsim_step.py` keeps the torch write, which is harmless for timing).

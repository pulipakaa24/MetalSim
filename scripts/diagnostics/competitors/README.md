# Cross-engine benchmarks (MetalSim vs Genesis vs MuJoCo-MLX-Cpp)

Findings: `docs/HANDOFF_elliptic_cone_perf.md`. Logs: `runs/competitors/`.

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

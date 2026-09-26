# Handoff: elliptic friction cones are 3–4× slower on Metal (generalizing beyond the G1)

Branch `perf-elliptic-cone`, based on `main` at `fbaab25`. It contains the findings, the benchmark
scripts (`scripts/diagnostics/competitors/`) and their logs (`runs/competitors/`). It makes no code
changes. Everything below was measured on 2026-09-25 on an M4 Max (40-core GPU, 64 GB, on AC power),
with every timing job run through `scripts/gpu_run.sh ... timing` on an idle GPU.

## Summary

MetalSim's throughput work was done on the Isaac G1, which uses **pyramidal** friction cones. Any model
that uses **elliptic** cones takes a MuJoCo Warp solver path that was never optimized for Metal, and it
costs 2.9–3.9× throughput. Elliptic cones are common:

- MuJoCo Menagerie's Go2 declares `<option cone="elliptic" impratio="100"/>`.
- MuJoCo's documentation recommends elliptic cones with a high `impratio` for grasping. The
  so101Sim/YAM grasp study found the same thing: 1–2 mm in-hand creep with elliptic cones against
  1–2.8 cm with pyramidal.

So manipulation tasks and many quadruped models will hit this.

## Measurements

The setup is the same Menagerie MJCF in both engines:
- joint-space PD tracking random targets around the `home` keyframe, resampled every 20 steps;
- 4 ms step, one substep, 4096 envs;
- Newton solver at 10 iterations / 20 line-search iterations;
- synchronized timing over 1000 steps after 100 warm-up steps.

Scripts: `scripts/diagnostics/competitors/{metalsim_step,genesis_step}.py`; `CONE=pyramidal|elliptic`
overrides the model's cone. Genesis is v1.4.2 (`a4b45ff`) on `gs.metal`.

| model, cone | MetalSim steps/s | Genesis steps/s | MetalSim cost of elliptic | Genesis cost of elliptic |
|---|---|---|---|---|
| Go2, pyramidal | 1,023,772 | 1,133,984 | | |
| Go2, elliptic (the model's own) | 262,371 | 507,474 | **3.9×** | 2.2× |
| G1, pyramidal (the model's own) | 229,447 | 186,542 | | |
| G1, elliptic | 80,024 | 108,205 | **2.9×** | 1.7× |

With pyramidal cones MetalSim is on par with Genesis on the Go2 (0.90×) and faster on the G1 (1.23×).
With elliptic cones it is 0.52× and 0.74×.

**Caveat for any comparison:** Genesis ignores an MJCF's `cone="elliptic"` unless
`RigidOptions(friction_cone=gs.friction_cone.elliptic)` is set (`genesis/utils/mjcf.py:267`, which only
warns). MuJoCo-MLX-Cpp defines an elliptic enum, but nothing in its source uses it. Out-of-the-box
comparisons on the Go2 therefore pit MetalSim's elliptic solve against their pyramidal one. That was
the source of an apparent 4.3× Genesis lead on the Go2.

## Where the time goes (Go2, elliptic, 4096 envs)

This is the per-kernel GPU time with `WP_METAL_PROFILE=1`
(`scripts/diagnostics/competitors/metalsim_profile.py go2 4096 kernels`). In total: 13.45 ms per step,
187 dispatches.

| kernel | ms/step | share | launches/step |
|---|---|---|---|
| `_update_gradient_JTCJ_dense` | 9.71 | **72.2 %** | 11 |
| `_linesearch_iterative_kernel` | 0.62 | 4.6 % | 10 |
| `_update_gradient_JTDAJ_dense_tiled` | 0.56 | 4.1 % | 11 |
| `_update_gradient_cholesky` | 0.50 | 3.7 % | 11 |
| all collision kernels (ccd, primitive narrowphase) | < 0.3 | < 2 % | |

**Collision is not the cause.** The variants in the same script (floor-only contacts, cylinders replaced
by capsules, both) all run at 262–271K steps/s.

## Why `_update_gradient_JTCJ_dense` is slow on Metal

The file is `upstream/mujoco_warp` (fork branch `metalsim`), `mujoco_warp/_src/solver.py`, in
`_update_gradient` and `_update_gradient_JTCJ_dense`.

1. **Launch size.** The elliptic-cone Hessian term is launched as `dim=(dim_block, dof_tri_row.size)`.
   On CUDA, `dim_block` is sized from the SM count (`sm_count * 6 * 256 / ntri`). Off CUDA, the
   fallback, commented "fall back for CPU", is `dim_block = d.naconmax`. That is one thread per
   contact *slot* per lower-triangle Hessian entry. At 4096 envs, `naconmax` = 196,608 (48 per world)
   and the Go2 has 171 triangle entries, so the launch is **33.6 M threads**, 11 times per step. About
   20K contacts exist (~5 per world), so over 99 % of the threads only read `nacon` and return. Per
   `docs/research/mjwarp_throughput_2026-09-25.md` §3, a capacity-sized launch still costs a full drain
   on Metal.
2. **Atomics.** Each active contact thread adds into `ctx_h_out[worldid, dof1id, dof2id]` with `+=`,
   which Warp lowers to an atomic add. All contacts of a world contend on the same entries.
3. **No fast path.** The fork's fused path (the incremental Hessian update fused into the register
   Cholesky, `ccfaffb`) and the incremental update only apply to pyramidal cones. With elliptic cones
   the solver rebuilds H from scratch every iteration: `JTDAJ_dense_tiled`, then `JTCJ_dense`, then
   the Cholesky, ×11 per step.

## Suggested fixes, cheapest first

Measure each with `metalsim_step.py go2 4096 matched` and `CONE=elliptic metalsim_step.py g1 4096 matched`.
Check physics against the unpatched fork with the same state-difference protocol as
`scripts/diagnostics/g1_tp_check.py`.

1. **A Metal branch for `dim_block`.** Size the grid from the GPU instead of `naconmax`, for example
   `dim_block = ceil(cores * k * 256 / ntri)`, with k swept over 2–16 (the M4 Max has 40 cores; Warp
   exposes the device, and `sysctl`/IOKit give the core count if Warp doesn't). Each thread then loops
   `nblocks_perblock` times over contacts, as on CUDA. This is a small change and should remove most of
   the 9.7 ms.
2. **Loop over `nacon`, not `naconmax`.** Early-exit on `conid >= nacon` already exists, but the launch
   is sized to capacity. With graph capture, the count has to stay static. The option is a
   world-major variant (one thread per world × triangle entry, looping over that world's contacts),
   which also removes the atomics. The ~5 contacts per world make the loop short.
3. **Fold the cone term into the tiled JᵀDJ build.** Compute the elliptic cone Hessian blocks per
   world inside `_update_gradient_JTDAJ_dense_tiled` (same tile, no second pass over H), or into the
   fused register-Cholesky kernel when `nv ≤ metal_register_cholesky_max`.
4. **Extend the incremental update to cone rows.** This is harder: cone rows change state
   (`CONE`/`QUADRATIC`/`SATISFIED`) more often, and their Hessian block is not rank-1.

Upstream: fix 1 is a natural pull request to google-deepmind/mujoco_warp, since the "fall back for
CPU" branch affects every non-CUDA backend. List it in `METALSIM_CHANGES.md` and
`scripts/diagnostics/newton_upstream/`-style `FILED.md` if filed.

## Done when

- Elliptic-cone throughput on the Go2 and G1 improves, with physics unchanged to float noise against
  the current fork.
- The 316 MuJoCo Warp fork tests still pass on Metal.
- A PARITY/GAPS row states the elliptic-cone throughput, and DECISIONS records any rejected variant.

## Also found in this session (separate issue, not investigated)

On this machine, `scripts/diagnostics/g1_tp_variants.py 4096` at `main` `fbaab25` with the forks at
their `metalsim` heads (Warp `4127c48`, MuJoCo Warp `07a51a6`) measures:
- physics only **84.1K** (README 81.0K);
- full env step **55.2K** (README 67.6K);
- full PPO loop **50.2K** (README 56.6K).

`g1_step_profile.py` puts the difference in `solver.solve`: 5.50 ms per substep here against 4.23 ms in
`runs/mjw_tp/final_checks.log`. Kernels and dispatch counts are the same. Candidates:
- the flex merge `8fbf965` in the MuJoCo Warp fork, which the headline numbers predate (they were taken
  at `ccfaffb`/`1791414`);
- a difference between the machines (the reference ran under `/Users/aditya/robosim`).

A back-to-back run at `1791414` against `07a51a6` on one machine settles it. The logs are in
`runs/competitors/metalsim_g1_*.log`.

## Other competitor results from the same session (for context; full survey in `docs/research/competitors_2026-09-25.md`)

- **Genesis `go2_train.py`** (its own PPO example, 4096 envs, Metal/MPS): 170–171K env-steps/s for the
  full PPO loop, reward rising over 30 iterations. That is Go2, 12 actuators, 2 substeps of 10 ms, no
  self-collision and pyramidal cones, so it is not comparable to the G1 task's 50–57K.
  Log: `runs/competitors/genesis_go2_4096.log`.
- **MuJoCo-MLX-Cpp** (`20niship/MuJoCo-MLX-Cpp` `3ae0e4a`, its own `bench_baseline`, Euler, zero control,
  solver budget from the XML):
  - DeepMind humanoid: 23–40K / hung (3 of 3) / 103K at 256 / 2048 / 4096 envs, against MetalSim's
    46K / 254K / 396K under the same protocol (`metalsim_mlxprotocol.py`).
  - Go2: 34–41K / 126–139K / {hung, 5.7K, 116K}, against MetalSim's 22K / 40K / 43K. The Go2 gap is
    this elliptic-cone issue again: MuJoCo-MLX-Cpp does not implement elliptic cones.
  - Logs: `runs/competitors/mlx_only.log`, `metalsim_mlxprotocol.log`.

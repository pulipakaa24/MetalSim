# MPM parity close-out: multi_material and ANYmal against the L4 (2026-10-12)

Question: why was Newton's `multi_material` example on Metal outside the L4's own spread (1.15x the largest of 16
perturbed L4 runs, in every fast_math mode), and was ANYmal's 1.22x at frames 4-6 real? Method: the column work's,
one level deeper. Find the first step where Metal leaves the L4 by more than the L4's own arithmetic perturbations
do, then the first array, then the first operation. Everything below was measured on the M4 Max and on the L4
(`aditya-isaac`, Isaac Lab 3.0-EA's Warp 1.16.0 / Newton 1.5.2, driver 580.178.04). The scripts are in this
directory. Data: `runs/parity/isaaclab3_mpm_closeout/` (L4) and `runs/mpm/closeout/` (Metal). Particle recordings
are not committed.

## 1. Where multi_material leaves the L4

Per material (`mm_per_material.py`, `early_frames.py`): the kinematic block never moves. Sand and mud match the L4
to the floor until mud lands (frame 24). **Snow** differs at frame 6, while it is still in free fall:

* L4 vs L4 (no-FMA, exact math, cubin, SAH): 35-43 nm. Metal vs L4: **150 nm in every math mode**. Metal vs Metal
  across modes: 36-42 nm.
* The pattern is a coherent vertical compression (+20 to +38 nm at the bottom, -60 to -70 nm at the top;
  `snow_sand_probe.py`), not random rounding.

So the cause was systematic, and it sat outside the fast_math kernels. The ensemble could not cover it: none of the
L4 perturbations touches the cause, as section 2 shows.

Per step, per field (`dump_steps.py` on both GPUs, `steps_cmp.py`, `jp_step1.py`):

* After step 1, snow's plastic volume ratio `Jp` differed from the L4's on 5,406 particles (by 1-2 ulp). Under the
  L4's no-FMA perturbation it differed on 402, and under exact math on 0.
* Snow is the only material with hardening (Newton: yield pressure x `sinh(-10 log Jp)`, so snow starts with zero
  yield strength). Its `Jp = exp(p_rate * rate)` and its yield surface go through `exp`, `log` and `sinh`.

First-solve dumps (`dfs2.py`, `dfs_cmp.py`, `dump_p2g.py`, `npz_cmp.py`) showed that before the strain solve, the
node mass and momentum from particle-to-grid already differed:

* `inv_mass` differed on 7,381 of 10,252 nodes, by up to +-10 ulp, symmetric.
* No-FMA and exact math on the L4 leave these arrays **bitwise equal**.
* The inputs were bitwise equal on both GPUs: particle lists per cell, node-element lists, densities and weights.
* Metal with contraction off gave the same 7,381 nodes, so contraction does not cause the difference.

## 2. Root causes (all of class (b): Metal took a different algorithm path than CUDA)

1. **warp.fem's dispatched assembly is tiled on CUDA only.** In `warp/_src/fem/integrate.py`
   (`_generate_auxiliary_kernels`; the same code in the L4's Warp 1.16.0, lines 1308-1325), a linear form whose
   quadrature is "long" (`qp_count * node_element_count >= 3 * 32 * node_count * element_count`) is summed by 32
   lanes per node: strided per-lane sums, then `tile_sum`, which is the warp shuffle-down tree in
   `native/tile_reduce.h`.
   * The condition starts `not device.is_cuda or ...`, so Metal (and the CPU) always summed sequentially. That is a
     different summation order for every node's mass and momentum.
   * The L4 takes the tiled path in 5 of the 6 floor scenes (`tile_decisions.py`: multi_material, granular,
     column, plate and ANYmal; not two-way).
   * The Metal fork's SIMD-group reduction already reproduces CUDA's shuffle tree (`tile_reduce.h`,
     `metal_reduce::simd_reduce`). With the heuristic taking CUDA's choice on Metal, the first step's node mass and
     velocity **equal the L4's bit for bit**. The tiling decisions of all six scenes equal the L4's call for call.
     In granular, the cell counts are equal through step 1 and the decisions equal throughout.
2. **`log` / `exp` / `sinh` are libdevice's on CUDA and the Metal library's on Metal.** Fingerprints over all 2^32
   float32 inputs (`libm_hash.py` on both GPUs, `libm_cmp.py`):

   | function | Metal equals CUDA | correctly rounded: Metal / CUDA |
   |---|---|---|
   | `log` | 76 % of a 2^22 random sample (bit-pattern uniform); 0 % of 4,096-input chunks in [0.5, 2) | 55 % / 97 % (Metal is biased low) |
   | `exp` | 97 % | 95 % / 95 % |
   | `sinh` | 97 % | 95 % / 98 % |

   The CUDA side is the PTX that NVRTC inlines from libdevice ([libdevice user guide](https://docs.nvidia.com/cuda/libdevice-users-guide/)),
   dumped from the L4's kernel cache (`runs/parity/isaaclab3_mpm_closeout/ptx/libm_main.txt`):
   * `logf` is an FMA polynomial.
   * `expf` is a range reduction with `fma.rm` and `ex2.approx.ftz.f32`.
   * `sinhf` is a polynomial below 1, and `ex2.approx` with `div.approx` above 1.

   The PTX ISA bounds `ex2.approx` at 2 ulp and leaves its bits to the hardware
   ([PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)). No L4 perturbation changes these
   functions: they are not in fast_math kernels, and NVRTC's `--fmad` does not touch libdevice's explicit `fma`.
3. **svd3 / eig3 use `::rsqrt` on CUDA and `1 / sqrt` elsewhere** (`native/svd.h`, `recipSqrt`). CUDA's float
   `rsqrt` is always `rsqrt.approx.f32`, which is MUFU.RSQ (compiled evidence in
   `research_notes/Metal MPM speed and scalable determinism/cuda_approx_math.md`). MUFU.RSQ is correctly rounded on
   78 % of [1, 4); Metal's `1/sqrt` equals it on 67 % (`rsqrt_check.py`). Newton calls `svd3` per particle per step
   and `eig3` in its contact solve and in the Newton-to-MuJoCo inertia conversion. In multi_material this changed
   nothing before frame 12, because the deformation gradients are still near the identity there. It is a faithful
   fix, not the cause.

## 3. Fixes (Warp fork, branch `metalsim`)

* `integrate.py`: Metal takes CUDA's tiling choice (`tiles_ok = device.is_cuda or device.is_metal`).
* `native/metal_cuda_libm.h`: `wp_cuda_logf` / `wp_cuda_expf` / `wp_cuda_sinhf`, built as follows:
  * They follow the PTX operation by operation: `fma` where the PTX has `fma.rn`, and `fma(a, b, signed 0)` for
    single-rounded products, so the Metal compiler cannot contract them.
  * `fma.rm(s, 252, 12582913)` is computed exactly as an integer floor.
  * Subnormal inputs and results follow CUDA's non-ftz forms on the bit patterns, because Apple GPUs flush float32
    subnormals.
  * `ex2.approx.ftz` is MUFU.EX2 through Benjamin Carleton's sass-math model
    ([pyxis-roc/sass-math](https://github.com/pyxis-roc/sass-math), MIT, `ex2.c` and `reduction/rro_ex2.c`). This is
    the same multifunction-interpolator family as Oberman & Siu, ARITH-17 2005.
  * **Equal to the L4 on all 2^32 inputs of each function**: every 4,096-input chunk fingerprint matches.
* `native/metal_cuda_fast_math.h`: `wp_cuda_rsqrt` is MUFU.RSQ through sass-math's rsqrt table, plus the subnormal
  pre-scale that ptxas emits for the non-ftz form. It is bitwise equal to the numpy model (`mufu_model.py`, already
  validated bitwise on the L4 over all of [1, 4)) on all 2^24 inputs in [1, 4) and over exponent shifts 4^-63..4^62.
  `svd.h` uses it on Metal.
* `warp.config.metal_cuda_libm` (`WP_METAL_CUDA_LIBM`, default on) selects both. When on, every module hash is
  salted once, so stale Metal binaries are not reused; when off, the hashes and binaries are those of before.
  Test: `TestMetal.test_cuda_libm_matches_l4`, which checks L4 result bits including subnormal cases.
  * Run on the closeout tree with these changes applied (not on the pushed commits): `test_metal`,
    `fem.test_fem_integrate` and `test_math` pass (75, 9 skipped).
  * `test_arithmetic`, `test_special_values`, `test_bf16`, `matrix.test_mat` and `test_codegen` pass except
    `test_inverse_float16_metal_0`, which fails identically on the unmodified tree.
  * MetalSim's `tests/test_mpm.py`: 14 passed, 1 xfailed.

Effect on the first steps of multi_material (`steps_cmp.py`):

* Snow's `Jp` after step 1 equals the L4's on all 30,256 particles.
* The first solve's iteration counts become the L4's (solve 6: 100, not 95).
* Through step 8, Metal vs L4 is at the size of the L4's exact-math perturbation: snow positions 1.55e-7 m vs
  1.32e-7 (exact math) and 1.46e-7 (no-FMA).
* Before the solve, only the Delassus eigen-decomposition differs, on 468 of 7,959 nodes (max relative 1.7e-3). That
  is FMA-placement noise; no-FMA on the L4 changes 7,948 nodes by up to 27 %.

## 4. Six scenes against the floor ensemble (default `metal_fast_math="off"`; `floors_dist.py`)

Ensemble: the 16 perturbed L4 runs, plus 6 new rigid-jitter runs for ANYmal (section 5). The ratio is Metal / the
largest perturbed-L4 distance, at the worst recorded frame.

| scene | before | after | at the last frame (after) | verdict |
|---|---|---|---|---|
| multi_material | 1.15 (outside) | **0.80** | 0.75; p95 pairwise 0.71 | inside |
| ANYmal | 1.22 against 16 runs (1.00 against 22) | **0.94** | 0.84 | inside |
| two-way | 0.83 | 0.83 | 0.61 | inside (L4 `base2` itself not bitwise) |
| granular | 0.66 | 0.66 | 0.40 | inside (collider-tie event at frame 6, unchanged) |
| plate drag (force curve) | 0.99, final force 240 N | **0.97**, final force 250 N (L4 244-282 N) | | inside |
| column a = 2 | 0.165 um at 3 s | 0.172 um at 3 s (L4 exact-math 0.155, float32 floor 0.19) | 76.58 mean iterations = L4 | inside |

With `"cuda"` fast math:

* multi_material 0.81, ANYmal 0.94, two-way 0.95, plate 0.38 (but final force 283 N, 1 N above the L4's range).
* Column a = 2: one convergence check (solve 76) lands on the other side of the tolerance (30 iterations against
  the L4's 25). The column then grows from 0.16 um to 10 um at 3 s. Before that frame, Metal-cuda is 1.61e-7 vs the
  L4, the same as `"off"` (1.62e-7) and the L4's exact-math run (1.52e-7). This is a near-threshold stopping event,
  the kind the L4's own no-FMA run takes at solve 11. It does not show a lowering defect: the fixes are identical in
  both modes, and only fast_math kernels differ between them. It does fail the strict column T1 for `"cuda"`, and it
  is one more reason the default stays `"off"`.

## 5. ANYmal frames 4-8: a floor-estimation gap (class (a))

Per run (`per_run.py`), in the first 10 frames the robot is moved by exactly one L4 perturbation class: no-FMA. Its
three runs (`nofma`, `nofma_cubin`, `nofma_j7`) are almost identical. Every jitter and exact-math run leaves the
robot bitwise at the base pose. So the early-frame floor was effectively one sample of the rigid-side divergence.
Metal moves the robot by 0.9-1.2e-7 m, which is less than no-FMA's 1.3e-7.

New L4 runs (`record_mpm.py --jitter-body-ulp 1`, stage `isaaclab3/stage_mpm_floors3.sh`, seeds 1-6; `runs/parity/isaaclab3_mpm_floors/mpm_floors3/`)
perturb the robot's initial joint coordinates by 1 ulp, the rigid counterpart of the particle jitter already in the
ensemble:

* They move the robot by 0.6-1.0e-7 m at frame 2.
* They move the sand by up to 7.85 um at frame 6, against Metal's 7.0 um.

Against the 22-run ensemble, the worst ANYmal frame is 0.94 after the fixes and 1.00 before them. The 1.22 was the
gap, not a Metal defect.

## 6. Cost

There is no measurable cost (timing class, interleaved, 60 frames; `runs/mpm/closeout/speed/`).

* `speed_libm.sh`, multi_material, three repetitions: `metal_cuda_libm` on 52.7 / 55.7 / 54.2 ms per frame, off
  54.5 / 54.4 / 55.1.
* `speed_ab.sh` (before the closeout vs after it), two repetitions:
  * granular 47.8 / 47.5 vs 47.3 / 47.8
  * column a = 2: 31.0 / 31.3 vs 30.8 / 31.0
  * ANYmal 48.4 / 47.8 vs 47.8 / 48.1
  * multi_material 56.8 / 55.5 vs 59.3 / 59.2

The multi_material A/B is inside the run-to-run spread of the libm A/B above: the device was not idle when these jobs
started, and identical configurations varied by about 3 ms.

## Verdicts

* **multi_material**: (b), a systematic difference, now fixed. The main cause was the CUDA-only fem tiling
  heuristic (summation order of every particle-to-grid sum). The libdevice-vs-Metal `exp` / `log` / `sinh` also
  added to the snow plastic-flow difference. Inside the floor (0.80).
* **ANYmal**: (a), a floor-estimation gap: the ensemble had a single rigid-side perturbation. Inside with rigid
  jitter (0.94).
* No Metal bug was found in the sense of (c). The Metal fork's tile reduction, MUFU models and FMA contraction all
  reproduced CUDA where tested.

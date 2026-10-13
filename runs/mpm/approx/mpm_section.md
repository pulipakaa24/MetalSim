### 1b. CUDA-faithful fast_math (Warp branch `metalsim-cuda-fast-math`, 2026-10-11)

Research and measurements: `research_notes/Metal MPM speed and scalable determinism/cuda_approx_math.md`.

**What the L4 computes** (Warp 1.16 fast_math kernels; PTX from the kernel cache, SASS via ptxas 12.8 + cuobjdump):
every `/` is `div.approx.ftz.f32` = `MUFU.RCP` + `FMUL.FTZ` (100 % of 2^24 pairs and a special-value grid; not
`div.full`), `wp.sqrt` is `sqrt.approx.ftz.f32` = `MUFU.SQRT`, `1/sqrt` stays `MUFU.SQRT` + `MUFU.RCP`. MUFU.RCP is
correctly rounded on 86.79 % of inputs (biased up), MUFU.SQRT on 83.13 % (biased down, -1 ulp on 15.5 %); both are
exponent-invariant (checked over the whole normal range). Benjamin Carleton's ptx-math / sass-math interpolator model
(MIT) reproduces both on the L4 for every input (2^23 + 2^24 exhaustive), and Warp's compositions (`length`,
`normalize`, `cw_div`, `x/z +- y/z`, constant divisors, ...) are exactly per-operation compositions of it.

**Metal emulation** (`warp/native/metal_cuda_fast_math.h`, `warp.config.metal_fast_math = "cuda"`): the interpolator in
32-bit integer arithmetic (packed 128-row tables, the truncated squarer as a carry-free nibble-spread product), division
as `fma(a, rcp, signed zero)` so the Metal compiler cannot contract the quotient into a following subtraction (it does
for a plain product, even under `fp contract(off)`). Bit-identical to the L4 on all inputs of the sweeps, on 2^22
random divisions and on the 16 Warp compositions (2^20 inputs each); `"relaxed"` agrees on 66-88 %, `"off"` on 58-87 %.
Per-op cost (dependent chains, 2^20 threads; FMA 1.28 ps): division 10.8 ps (precise 3.7, fast 1.3), sqrt 13.6 ps
(precise 2.7, fast 1.3). With `"off"` / `"relaxed"` the recordings are bitwise those of the integrated tree's
no-fast-math / fast-math runs (all six floor scenes), so non-fast_math kernels are unchanged.

**Parity** (floor ensemble: 16 perturbed L4 runs, `floors_dist.py`; runs `runs/mpm/fm_{off,cuda,relaxed}` of the
worktree; M / max floor at the worst picked frame; verdict at the picked frames):

| scene | off | cuda | relaxed |
|---|---|---|---|
| column a = 2 (M at frame 180; L4 floor p50 0.178 um) | 0.165 um, inside | **0.144 um**, inside (L4 iterations 76.58) | 0.163 um, inside |
| granular | 0.66, inside | 0.66, inside | 0.66, inside |
| multi_material | 1.15, outside | 1.20, outside | 1.13, outside |
| twoway | 0.83, inside | 0.94, inside | 1.01, outside |
| anymal (frame 50) | 0.99, inside | 1.01, outside | 1.01, outside |
| plate drag (force curve) | 0.99, inside, final 240 N (L4 244-282) | **0.95**, inside, 250 N | 1.08, outside, 244 N |

parity_report.py (R suite): every T3 row PASS in all three modes; T1 column 0.83 / 0.87 (off), 0.93 / 0.76 (cuda),
0.89 / 0.87 (relaxed) x floor; T1 granular FAIL in all (collider ties, section 5).

**Speed** (timing class, 60 frames after 5, modes interleaved per scene, one repetition; ms/frame):

| scene | off | cuda | relaxed |
|---|---|---|---|
| granular | 48.2 | 83.0 (1.72x) | 37.0 |
| multi_material | 53.5 | 118.6 (2.22x) | 42.2 |
| twoway (captured) | 45.5 | 53.2 (1.17x) | 42.9 |
| anymal (captured) | 36.5 | 47.1 (1.29x) | 34.3 |
| plate drag | 47.6 | 80.4 (1.69x) | 36.0 |
| column a = 2 | 31.0 | 43.1 (1.39x) | 21.7 |
| column a = 8 | 39.8 | 64.3 (1.61x) | 25.6 |

**Default: `"off"`.** The faithful mode is CUDA's arithmetic exactly (closest to the L4 on the column and the plate,
the L4's iteration counts), but it does not pass the ensemble at least as well as `"off"` (anymal at 1.01 x vs 0.99 x,
multi_material 1.20 vs 1.15; differences at the size of one chaotic sample) and it is 1.2-2.2x slower: exact MUFU
results cost about 3x a correctly rounded division on Apple GPUs, so it cannot recover fast_math's speed. The column
"25 x" of the earlier fast-divide / precise-sqrt variant is consistent with a stopping-rule change (mean iterations
76.83 vs the L4's 76.58: one convergence check landing on the other side of the tolerance moves the trajectory by far
more than rounding), not with a lowering defect: all three modes here keep 76.58 and stay inside the column floor; that
mixed variant itself was not re-recorded.
`install(metal_fast_math="cuda")` remains for bitwise studies; `"relaxed"` is the fastest (0.64-0.94x off) at a small
parity cost (twoway, plate, anymal just outside the floor).


# CUDA approximate float32 division / sqrt / rcp / rsqrt, and bit-exact emulation on Apple GPUs

Research note, 2026-10-11. Goal: reproduce, bit for bit on Apple GPUs (Metal), what NVRTC `--use_fast_math` code (as emitted by Warp) computes on an NVIDIA L4 (sm_89, Ada).

**Evidence used.** NVIDIA, Apple, LLVM, Mesa and Warp documentation and source, quoted directly; forum posts and papers; and fresh compiles on Compiler Explorer (godbolt.org API, compilers `nvcc133` = NVCC 13.3.0 and `nvcc118` = NVCC 11.8). The compiles produced PTX and `nvdisasm` SASS; no GPU was used. Anything marked **[inferred]** or **[unverified]** has not been checked on hardware.

---

## Summary (key conclusions)

| # | Conclusion | Confidence |
|---|---|---|
| 1 | Under `--use_fast_math`, plain float `a / b` becomes **PTX `div.approx.ftz.f32`**. `1.0f/x` becomes `rcp.approx.ftz.f32`, `sqrtf(x)` becomes `sqrt.approx.ftz.f32`, and both `rsqrtf(x)` and `1.0f/sqrtf(x)` become `rsqrt.approx.ftz.f32`. | High (compiled with NVCC 13.3 and 11.8; NVIDIA's guide says the same) |
| 2 | `--use_fast_math` is **not the same** as spelling out `-ftz=true -prec-div=false -prec-sqrt=false -fmad=true`. The spelled-out flags give `div.full.ftz.f32` for `a/b`. `--use_fast_math` also rewrites `x/y` to `__fdividef(x,y)`, which is `div.approx`. | High (compiled) |
| 3 | On **sm_89 SASS**: `div.approx.ftz.f32` is `MUFU.RCP r, b; FMUL.FTZ q, a, r` with no range checks. `rcp.approx.ftz` is a single `MUFU.RCP`. `sqrt.approx.ftz` is a single **`MUFU.SQRT`**, which exists on sm_52 through sm_120. `rsqrt.approx.ftz` is a single `MUFU.RSQ`. `div.full.ftz` adds a step: if `|b| > 2^126`, both operands are multiplied by 0.25 before `MUFU.RCP` and `FMUL.FTZ`. The same sequences appear on sm_75, sm_80, sm_86, sm_90 and sm_120. | High (compiled) |
| 4 | **ptxas optimizes around MUFU.** It shares one `MUFU.RCP` between divisions by the same divisor. It contracts `a/c + b/c` into `FMUL` then `FFMA` (`fma(a, rcp(c), b*rcp(c))`). It folds division by a constant into a multiply by an FP32 reciprocal constant. So a per-operator emulation is not enough; the emulator must follow what ptxas actually emitted. | High for these observations (compiled); how general they are is **[inferred]** |
| 5 | **MUFU.RCP, RSQ and SQRT internals.** They are piecewise-quadratic table interpolators (Oberman & Siu, ARITH 2005). The reciprocal uses a 7-bit table index (128 entries), coefficient widths C0/C1/C2 = 26/16/10 bits, and a truncated squarer. The paper reports **0.98 ulp maximum error and 87% correctly rounded**, monotonic. The reciprocal result depends **only on the 23-bit significand and the sign**; the exponent is just negated. RSQ and SQRT add the exponent's parity to the table index. | High for the design (paper and an independent bit-accurate model agree exactly on the parameters); **[unverified on sm_89]** |
| 6 | **A bit-accurate MUFU emulation exists:** `bcarlet/ptx-math` and its fork `pyxis-roc/sass-math` (MIT licence, C). They were validated exhaustively over all 2^32 inputs on sm_5x, sm_6x, sm_70 and sm_75. They model `MUFU.RCP`, `RSQ`, `SQRT`, `EX2`, `LG2`, `SIN` and `COS`. My port of the RCP model reproduces the paper's figures: max error 0.975 ulp, 86.79% correctly rounded over all 2^23 significands. A 2026 preprint probed the MUFU ops exhaustively on Ampere, Ada (L40/L40S) and Hopper and stored the outputs as tables, but did not publish per-architecture findings. **No public source confirms the ptx-math tables on sm_80, sm_86 or sm_89.** | High that the models exist; medium that they still hold on sm_89 (needs one exhaustive check on the L4) |
| 7 | **Apple (Metal) fast math.** MSL 4.1 Table 8.2 (fast math): `1.0/x` ≤ 1 ulp, `x/y` ≤ 2.5 ulp, `rsqrt` ≤ 2 ulp, and `sqrt` "Implemented as `x * rsqrt(x)` with special cases handled correctly." On the hardware (AGX, G13 and later) there is `rcp`, `rsqrt` and `rsqrt_special`, and **no sqrt instruction**. Fast divide is `RECIP32 + FMUL32`. Mesa notes the hardware reciprocal "is sometimes off by 1 ULP". Apple's rounding pattern is therefore *different* from MUFU, so Apple's `fast::divide` / `fast::sqrt` **cannot** reproduce CUDA fast-math bits. | High (spec, Mesa, applegpu docs, microbenchmarks) |
| 8 | **Path to bit-exactness on Metal.** Do not use Apple's approximate instructions. In safe math mode with `-ffp-contract=off`, implement `MUFU.RCP`, `SQRT` and `RSQ` as integer table interpolation (128- or 64-entry tables, about 15 integer ops), using the ptx-math algorithm. Use correctly rounded `*` to form `RN(a * rcp(b))`, add FTZ handling, and write explicit `fma()` wherever ptxas emitted `FFMA`. | Medium (design inference; needs measurement) |
| 9 | **Warp.** `div` returns `a/b` and `wp.sqrt` returns `::sqrtf(x)`. NVRTC gets `--use_fast_math` only when the module option `fast_math=True` is set (the default is `False`). Newton's `implicit_mpm` rheology kernels set `fast_math: True`, so those kernels are the ones that hit `div.approx.ftz` and `MUFU.*`. Warp's `div_approx` / `inverse_approx` builtins emit inline PTX `div.approx.f32` / `rcp.approx.f32` (non-ftz), which on sm_89 become a denormal pre-scale plus MUFU.RCP. | High (source) |

---

## 1. NVRTC/nvcc `--use_fast_math`: implied flags and what `/`, `sqrtf`, `rsqrtf` compile to

### Documentation
- nvcc: "`--use_fast_math` implies `--ftz=true --prec-div=false --prec-sqrt=false --fmad=true`." On `--prec-div`: "This option controls single-precision floating-point division and reciprocals. `--prec-div=true` enables the IEEE round-to-nearest mode and `--prec-div=false` enables the fast approximation mode." — [NVCC docs §4.2.7.7–4.2.7.11](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html)
- NVRTC has the same wording: "`--use_fast_math` … implies `--ftz=true --prec-div=false --prec-sqrt=false --fmad=true`." — [NVRTC docs, supported compile options](https://docs.nvidia.com/cuda/nvrtc/index.html)
- CUDA Programming Guide (v13.4), Built-In Arithmetic Operators: "The nvcc compilation flag `-prec-div=false`, also included in `--use_fast_math`, has the following effect on the maximum ULP error for the division operator `/` for the single-precision type `float`: `x / y → __fdividef(x, y)`: 2 ULP; `1 / x`: 1 ULP." — [CUDA Programming Guide §5.5.6](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html)
- Same guide, Table 60 "Functions Directly Affected by `--use_fast_math`": `x/y, fdividef(x, y)` → `__fdividef(x, y)`. The sqrt entry: "`sqrtf(x)` ▪ 0 ULP ▪ 1 ULP with `--use_fast_math`", and "`rsqrtf(x)` 2 ULP". — [CUDA Programming Guide §5.5.7/§5.5.9.3](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html)
- `__fdividef` table entry: "2 for |y| ∈ [2^-126, 2^126]". The Math API adds: "`__fdividef(∞, y)` returns NaN for 2^126 < |y| < 2^128" and "`__fdividef(x, y)` returns 0 for 2^126 < |y| < 2^128 and finite x." — [CUDA Programming Guide Table 59](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html), [CUDA Math API, single-precision intrinsics](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html)
- NVIDIA's advice on portability: "users are recommended to use the default settings … `-ftz=false`, `-prec-div=true`, and `-prec-sqrt=true`, and not use the `--use_fast_math` option." — [CUDA Programming Guide §5.5](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html)

### Compiled evidence (Compiler Explorer, NVCC 13.3.0, `-arch=sm_89`)
Kernels: `o=a/b`, `o=1.0f/a`, `o=sqrtf(a)`, `o=rsqrtf(a)`, `o=1.0f/sqrtf(a)`, `o=__fdividef(a,b)`. — [Compiler Explorer](https://godbolt.org) (REST API `/api/compiler/nvcc133/compile` with `produceDevice`)

| Flags | `a/b` | `1/a` | `sqrtf` | `rsqrtf`, `1/sqrtf` | `__fdividef` |
|---|---|---|---|---|---|
| `--use_fast_math` | `div.approx.ftz.f32` | `rcp.approx.ftz.f32` | `sqrt.approx.ftz.f32` | `rsqrt.approx.ftz.f32` | `div.approx.ftz.f32` |
| `-ftz=true -prec-div=false -prec-sqrt=false -fmad=true` (the "implied" set) | **`div.full.ftz.f32`** | `rcp.approx.ftz.f32` | `sqrt.approx.ftz.f32` | `rsqrt.approx.ftz.f32` | `div.approx.ftz.f32` |
| `-prec-div=false -prec-sqrt=false` | `div.full.f32` | `rcp.approx.f32` | `sqrt.approx.f32` | `rsqrt.approx.f32` | `div.approx.f32` |
| `--use_fast_math -ftz=false` | `div.approx.f32` | `rcp.approx.f32` | `sqrt.approx.f32` | `rsqrt.approx.f32` | `div.approx.f32` |
| default | `div.rn.f32` | `rcp.rn.f32` | `sqrt.rn.f32` | `rsqrt.approx.f32` / `sqrt.rn`+`rcp.rn` | `div.approx.f32` |

- So the `__fdividef` mapping that the guide describes for `-prec-div=false` only happens with `--use_fast_math`. `-prec-div=false` alone gives `div.full`. — [Compiler Explorer](https://godbolt.org) (compiled; NVCC 11.8 gives the same PTX under `--use_fast_math`)
- `rsqrtf` is always `rsqrt.approx`, even without fast math. `1.0f/sqrtf(x)` is only fused into `rsqrt.approx` under fast math. — [Compiler Explorer](https://godbolt.org)
- **[inferred]** NVRTC uses the same NVVM front end as nvcc and documents the same option semantics, so Warp's NVRTC compiles should produce the same PTX. Confirm by dumping Warp's PTX/CUBIN on the VM (see Measurements).

### LLVM NVPTX (clang/Triton/Numba paths, not NVVM)
- `nvptx-prec-divf32` values: `0` = "Use div.approx", `1` = "Use div.full", `2` = "Use IEEE Compliant F32 div.rnd if available (default)", `3` = "… no FTZ". Without the option, `getDivF32Level` returns `Approx` when the node has the `afn` (approximate-functions) flag and `IEEE754` otherwise. `nvptx-prec-sqrtf32`: "0 use sqrt.approx, 1 use sqrt.rn". — [LLVM NVPTXISelLowering.cpp (main)](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/NVPTX/NVPTXISelLowering.cpp)
- LLVM regression test: fast-math division gives `div.approx.f32`; non-fast gives `div.rn.f32`. — [LLVM test fast-math.ll](https://llvm.googlesource.com/llvm/+/refs/heads/release_40/test/CodeGen/NVPTX/fast-math.ll), [D28508 getSqrtEstimate review](https://reviews.llvm.org/D28508)

---

## 2. PTX ISA semantics and error bounds (quoted, PTX ISA 9.4)

All quotes are from [PTX ISA, §9.7.3 Floating Point Instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-div).

- **div**: "`div.approx.f32` implements a fast approximation to divide, computed as `d = a * (1/b)`. For |b| in [2^-126, 2^126], the maximum ulp error is 2. For 2^126 < |b| < 2^128, if a is infinity, `div.approx.f32` returns NaN, otherwise it returns a sign-preserving zero."
- "`div.full.f32` implements a relatively fast, full-range approximation that scales operands to achieve better accuracy, but is not fully IEEE 754 compliant and does not support rounding modifiers. The maximum ulp error is 2 across the full range of inputs."
- Subnormals: "`div.ftz.f32` flushes subnormal inputs and results to sign-preserving zero." "For PTX ISA version 1.4 and later, one of `.approx`, `.full`, or `.rnd` is required."
- **rcp**: "`rcp.approx.f32` implements a fast approximation to reciprocal. The maximum ulp error is 1 across the full range of inputs." Special cases: −Inf→−0.0, −0.0→−Inf, +0.0→+Inf, +Inf→+0.0, NaN→NaN.
- **sqrt**: "`sqrt.approx.f32` implements a fast approximation to square root. The maximum relative error over the entire positive finite floating-point range is 2^-23." Special cases: −Inf→NaN, −normal→NaN, −0.0→−0.0, +0.0→+0.0, +Inf→+Inf.
- **rsqrt**: "The maximum relative error for `rsqrt.f32` over the entire positive finite floating-point range is 2^-22.9." Special cases: −0.0→−Inf, +0.0→+Inf, +Inf→+0.0, negative→NaN. "`rsqrt.approx.f64` is emulated in software and are relatively slow."
- **ex2 / lg2** (for completeness): `ex2.approx.f32` has a "maximum ulp error is 2 ulp from correctly rounded result"; `lg2.approx.f32` has a "maximum absolute error is 2^-22" in the stated interval, otherwise relative error 2^-22.
- The PTX ISA does **not** specify the exact bits. These are bounds only, so the bits are set by the hardware (MUFU) and by ptxas's lowering. — [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)

---

## 3. Lowering to SASS (sm_75 / sm_80 / sm_86 / sm_89 / sm_90 / sm_120)

Compiled with NVCC 13.3.0. Load/store/addressing instructions removed. — [Compiler Explorer](https://godbolt.org)

**`--use_fast_math` (ftz), sm_89:**
```
a/b, __fdividef      : MUFU.RCP R9, Rb ; FMUL.FTZ R9, Ra, R9
1.0f/a               : MUFU.RCP R7, Ra
sqrtf(a)             : MUFU.SQRT R7, Ra
rsqrtf(a), 1/sqrtf(a): MUFU.RSQ R7, Ra
```
sm_75, sm_80, sm_86, sm_90 and sm_120 give the same arithmetic sequences. NVCC 11.8 gives the same for sm_52, sm_61, sm_70 and sm_89. So **`MUFU.SQRT` is used on sm_52 and later, including sm_89**.

**`div.full.ftz.f32` (the spelled-out flags), sm_89:**
```
FSETP.GT.FTZ.AND P0, PT, |Rb|, 8.507059e+37 (=2^126), PT
@P0 FMUL.FTZ Rb, Rb, 0.25
@P0 FMUL.FTZ Ra, Ra, 0.25
MUFU.RCP R9, Rb
FMUL.FTZ R9, R9, Ra
```
This is the "scales operands" range fix that the ISA describes. For |b| ≤ 2^126 it is bit-identical to `div.approx.ftz`.

**Non-ftz approx variants (`--use_fast_math -ftz=false`, or Warp's inline `div.approx.f32` / `rcp.approx.f32`):** the compiler adds a subnormal pre-scale. `div.approx.f32` becomes `if |b| < 2^-126: b*=2^24, a*=2^24; MUFU.RCP; FMUL`. `sqrt.approx.f32` becomes `if |x|<2^-126: x*=2^24; MUFU.SQRT; r*=2^-12`. `rsqrt.approx.f32` becomes `x*=2^24; MUFU.RSQ; r*=2^12`. `rcp.approx.f32` scales by 0.25 when |x| > 2^126 and by 2^24 for subnormals. This shows **MUFU itself flushes subnormal inputs (and presumably outputs) to zero**, so the non-ftz PTX forms are built around it. — [Compiler Explorer](https://godbolt.org) **[inferred: MUFU FTZ behaviour]**

**IEEE paths (default flags), for contrast:** `sqrt.rn.f32` uses **MUFU.RSQ** plus `FMUL`/`FFMA` Newton correction and a slow-path call (`__cuda_sm20_sqrt_rn_f32_slowpath`), not MUFU.SQRT. `div.rn.f32` uses `MUFU.RCP`, `FCHK`, an FFMA refinement and a slow path. `rcp.rn.f32` uses MUFU.RCP plus an FFMA correction. — [Compiler Explorer](https://godbolt.org); njuffa describes the same RSQ-based IEEE sqrt in [Performance tweak for single-precision square root (NVIDIA forum)](https://forums.developer.nvidia.com/t/performance-tweak-for-single-precision-square-root/173274)

**ptxas transformations that matter for bit-exactness** (sm_89, fast math):
- `a/c + b/c` becomes `MUFU.RCP r,c; FMUL.FTZ t,b,r; FFMA.FTZ o,a,r,t`. One reciprocal is shared and the multiply-add is **contracted across two divisions**, so the result is `fma(a,r,RN(b*r))`, not `RN(RN(a*r)+RN(b*r))`.
- `a/3.0f` becomes `FMUL.FTZ a, 0.3333333433` (0x3EAAAAAB); `a/10.0f` becomes `FMUL a, 0.1f`; `a/2.0f` becomes `FMUL a, 0.5`. The constant reciprocal is folded at compile time. For 3 and 10, RN(1/c) and the modelled MUFU.RCP(c) are the same bits, so these cases do not show which one ptxas uses. **[unverified in general]**
- `3.0f/a` becomes `MUFU.RCP r,a; FMUL r,r,3`.
- `a/sqrtf(b)` becomes `MUFU.SQRT s,b; MUFU.RCP r,s; FMUL a,r`. This is **not** fused into `a*rsqrt(b)`.
- `-a/b` becomes `MUFU.RCP; FMUL.FTZ a, -r`.
- `sqrtf(a*a+b*b)` becomes `FMUL; FFMA; MUFU.SQRT` (FMA contraction under `--fmad=true`).

— all from [Compiler Explorer](https://godbolt.org)

**Architecture history:**
- Fermi/Kepler-era SASS had `MUFU.{LG2|EX2|SIN|COS|RCP|RSQ}`. There, `sqrt.approx.f32` was "implemented via MUFU.RSQ and MUFU.RCP". — [Fermi and Kepler GPU Special Function Units (NVIDIA forum)](https://devtalk.nvidia.com/default/topic/531855/fermi-and-kepler-gpu-special-function-units)
- ptx-math models `sqrt.approx` on sm_5x as `ptxm_rcp_sm5x(ptxm_rsqrt_sm5x(x))` and on sm_6x, sm_70 and sm_75 as a separate SQRT table. That points to a dedicated MUFU.SQRT from Pascal (sm_6x) onward. But NVCC 11.8 emits `MUFU.SQRT` for sm_52 as well, so it likely exists on Maxwell 2 and older toolchains simply did not use it. **[conflict noted; irrelevant for sm_89, where MUFU.SQRT is what runs]** — [ptx-math README](https://github.com/bcarlet/ptx-math), [Compiler Explorer](https://godbolt.org)
- Throughput: "MUFU operations have one-eighth the throughput of simple FP32 operations on recent architectures" (njuffa, paraphrased by the fetch tool). — [Weekend project: replacing MUFU.RCP and MUFU.RSQ (NVIDIA forum)](https://forums.developer.nvidia.com/t/weekend-project-exploring-the-feasibility-of-replacing-mufu-rcp-and-mufu-rsq/346566)

---

## 4. How the SFU/MUFU computes RCP/RSQ/SQRT; bit-exact reverse engineering

### Oberman & Siu 2005 (NVIDIA)
- Design targets: "Reciprocal approximation accurate to within 1 ulp and monotonic", and reciprocal square root "within 2 ulps and monotonic". — [Oberman & Siu, "A High-Performance Area-Efficient Multifunction Interpolator", ARITH-17 (2005)](http://www.acsel-lab.com/arithmetic/arith17/papers/ARITH17_Oberman.pdf)
- Method: the upper m bits of the significand index three coefficient tables (C0, C1, C2), and the result is `C0 + C1·Xl + C2·Xl²` computed with a truncated special squarer and "function-specific biases". "The 1/X function with m = 7 requires 128 entries, while the other m = 6 functions require 64 entries each. 1/√X requires two sets of m = 6 tables, one for inputs in the range [1,2), and another for those in [2,4). The total ROM size is therefore 448 x 52b." — same paper
- Table 1 (quoted): `1/X [1,2) m=7 26,16,10 → 24.02 good bits, 0.98 ulp, 87% exactly rounded, monotonic, 6.50Kb`; `1/√X [1,4) m=6 26,16,10 → 23.40 bits, 1.52 ulp, 78%, monotonic`; `2^X [0,1) m=6 → 22.51 bits, 1.41 ulp, 74%`. The rounding is *not* round-to-nearest. The result is a truncated sum plus bias, which is faithful-ish and correctly rounded 87% of the time for RCP. — same paper
- Related theory (minimax quadratic interpolation; "m = 7 is necessary for computing the reciprocal, logarithm, and reciprocal square root" at 1 ulp): [Piñeiro, Oberman, Muller, Bruguera, IEEE TC 2005](https://perso.ens-lyon.fr/jean-michel.muller/QuadraticIEEETC0305.pdf)

### Bit-accurate reverse-engineered models (the key prior art)
- **ptx-math** (B. Carleton): "PTXM provides bit-accurate implementations of the elementary function approximation instructions available in Nvidia's PTX ISA", covering `rcp.approx.f32`, `sqrt.approx.f32`, `rsqrt.approx.f32`, `sin/cos/lg2/ex2.approx.f32`. Validation: "`ptxmvalidate` utility performs exhaustive simulation of the selected function across all single-precision inputs. Function output is compared to that of the corresponding PTX instruction executed on the selected GPU device." Its results table shows `ptxm_rcp_sm5x` and `ptxm_rsqrt_sm5x` matching on **sm_5x, sm_6x, sm_70 and sm_75**, and `ptxm_sqrt_sm6x` matching on sm_6x through sm_75. — [bcarlet/ptx-math](https://github.com/bcarlet/ptx-math)
- **sass-math** (Carleton & Valpey, University of Rochester PyXIS group, 2021, MIT): "SASSM provides bit-accurate implementations of the elementary function approximation instructions available in Nvidia's SASS ISA". It models `MUFU.RCP`, `MUFU.RSQ`, `MUFU.SQRT`, `MUFU.EX2`, `MUFU.LG2`, `MUFU.SIN/COS`, `RRO.*`, `MUFU.RCP64H` and `MUFU.RSQ64H`. Caveat in its README: "SASS is architecture-dependent, and thus semantics are not necessarily preserved across architectures." — [pyxis-roc/sass-math](https://github.com/pyxis-roc/sass-math)
- Model details, from [sass-math `src/sassm/rcp.c`](https://github.com/pyxis-roc/sass-math/blob/master/src/sassm/rcp.c) and [`tables/rcp_table.h`](https://github.com/pyxis-roc/sass-math/blob/master/src/sassm/tables/rcp_table.h):
  - `RCP_M 7, RCP_T 26, RCP_P 16, RCP_Q 10`, a 128×3 table. This is **exactly the Oberman & Siu 1/X configuration**.
  - `xh` = top 7 significand bits, `xl` = low 16 bits. `sum = (c0<<30) − (c1·xl<<17) + c2·(sq_approx(xl<<1)>>2) + bias(0x67e7…>>31)`, where `sq_approx` is a truncated partial-product squarer (antidiagonal plus 9 rows, truncated to 19 columns). The result significand is bits [56−23, 56) of the sum, renormalised by one bit if needed.
  - Then `ldexpf(r, −ilogb(x))`: **the result depends only on the significand and sign; the exponent is just negated.**
  - SASS-level special cases: NaN→NaN, ±Inf→±0, ±0→±Inf, **subnormal input→±Inf** (FTZ).
  - `MUFU.SQRT`: `SQRT_M 6` with widths 25/17/12, and an index extended by the exponent's parity (128 rows). `MUFU.RSQ`: `RSQRT_M 6` with widths 26/17/11, parity-extended, and powers of four with even exponent returned exactly. — [sass-math sqrt.c](https://github.com/pyxis-roc/sass-math/blob/master/src/sassm/sqrt.c), [rsqrt.c](https://github.com/pyxis-roc/sass-math/blob/master/src/sassm/rsqrt.c)
  - **[unverified]** The model's subnormal-input case for `MUFU.RSQ` returns `+0.0` and its result path does not flush subnormal outputs (it uses `ldexpf`). Check both against the L4 at the range edges.
- **My check of the RCP model.** I ported `rcp.c` to numpy and ran it over all 2^23 significands in [1,2), measuring error exactly with an FP64 residual. Max error is **0.9749 ulp** and **86.79%** of results are correctly rounded (45.7% fall below the exact value). This matches Oberman & Siu's "0.98 ulp, 87%". Examples: `rcp(1.1)` model gives 0x3F68BA2E where RN gives 0x3F68BA2F; `rcp(1.5)` and `rcp(1.25)` equal RN. — scratch computation from the sass-math table, not a GPU measurement.
- **Exhaustive probing on Ada/Hopper (no tables published):** "Instructions like MUFU.RSQ …, MUFU.EX2 …, and MUFU.RCP … rely on architecture-specific silicon lookup tables." The author "exhaustively probed these instructions via inline PTX" and "cached their exact hardware outputs (e.g. a 4GB table mapping every 32-bit input for MUFU.EX2), ensuring bit-exact SFU emulation without needing GPU access at verification time". Outputs "are deterministic but deviate from IEEE-correct rounding by up to ±2 ULP". Bit-exact results are reported on A100, L40, L40S and H100. The paper does **not** say whether MUFU tables differ between Ampere, Ada and Hopper. — [Cankaya, "Bit-Exact AI Inference Verification Without Performance Tradeoffs", arXiv:2606.00279](https://arxiv.org/html/2606.00279v1)
- **CuLifter** says that "NVIDIA's hardware reciprocal (MUFU.RCP) … returns an approximation biased by approximately +3 ULPs relative to IEEE 754", and that its runtime "must bias the IEEE reciprocal result by +3 ULPs". This conflicts with the ≤1 ulp bound in the PTX spec and both models. It most likely describes the integer-division idiom (`I2F → MUFU.RCP → IADD3 +0x0ffffffe → F2I`), which deliberately biases the reciprocal. **Do not use it as a MUFU model.** — [Zhao et al., "CuLifter: Lifting GPU Binaries to Typed IR", arXiv:2604.27486](https://arxiv.org/html/2604.27486)
- **GPGPU-Sim** is *not* bit-exact. Its PTX functional model computes `rcp` as `data.f32 = 1.0f / src1_data.f32` and `sqrt` with libm `sqrt`. — [gpgpu-sim cuda-sim/instructions.cc](https://github.com/gpgpu-sim/gpgpu-sim_distribution/blob/dev/src/cuda-sim/instructions.cc)
- njuffa's statements: "the built-in reciprocal square root instruction of the GPU, `MUFU.RSQ` is very accurate". His correctly rounded sqrt (RSQ plus four FFMA-class ops) "passes an exhaustive test with both ftz=true and ftz=false". — [NVIDIA forum, sqrt performance tweak](https://forums.developer.nvidia.com/t/performance-tweak-for-single-precision-square-root/173274)
- Answers to the specific questions:
  - *Fraction of inputs MUFU.RCP rounds correctly:* about 87% (paper), 86.79% (sass-math model).
  - *Exponent-independent?* Yes in the model, apart from special and edge cases: subnormal inputs flush, and inputs with |x| > 2^126 give subnormal results that are presumably flushed.
  - *Validated on sm_89?* **Not publicly.**

---

## 5. Warp (NVIDIA/warp)

- Float `div` (macro `DECLARE_FLOAT_OPS`): `inline CUDA_CALLABLE T div(T a, T b) { DO_IF_FPCHECK(...) return a/b; }`. Float `sqrt`: `return ::sqrtf(x);`. — [warp/native/builtin.h (main)](https://github.com/NVIDIA/warp/blob/main/warp/native/builtin.h)
- Approximate intrinsics (device only): `asm("rcp.approx.f32 %0, %1;" ...)` and `asm("div.approx.f32 %0, %1, %2;" ...)`. f64 `approx_rcp` uses `rcp.approx.ftz.f64`, and f64 `approx_div` is `a * approx_rcp(b)`. On the CPU they fall back to exact `1.0f/a` and `a/b`. They are exposed in Python as `div_approx` ("Divide two values using approximate GPU intrinsics. Falls back to exact division on CPU.") and `inverse_approx`. — [builtin.h](https://github.com/NVIDIA/warp/blob/main/warp/native/builtin.h), [warp/_src/builtins.py](https://github.com/NVIDIA/warp/blob/main/warp/_src/builtins.py)
  - Because these are **non-ftz** PTX, they become *pre-scale + MUFU + FMUL* on sm_89 even inside a fast-math module (see §3).
- NVRTC options (`warp/native/warp.cu`): `if (fast_math) opts.push_back("--use_fast_math");` then `opts.push_back(fuse_fp ? "--fmad=true" : "--fmad=false")`. — [warp/native/warp.cu](https://github.com/NVIDIA/warp/blob/main/warp/native/warp.cu)
- Module option defaults: `"fast_math": False`, `"fuse_fp": True`. Docs: "**fast_math**: Enable fast math for CUDA compilation, defaults to `False`." CHANGELOG (old): "fast math is now *disabled* by default". — [warp/_src/context.py](https://github.com/NVIDIA/warp/blob/main/warp/_src/context.py), [Warp CHANGELOG](https://github.com/NVIDIA/warp/blob/main/CHANGELOG.md)
- Checked in the local Warp fork (1.18.0.dev) and against upstream `main`. Versions 1.10–1.16 are assumed to have the same `div`/`sqrt`/NVRTC behaviour, since the `--use_fast_math` push and the `a/b` / `::sqrtf` bodies are long-standing. **[inferred for each exact version]**
- Project relevance: Newton's implicit MPM rheology kernels use `kernel_options={"fast_math": True}` (`newton/_src/solvers/implicit_mpm/rheology_solver_kernels.py`, four kernels). The raycast and warp_raytrace kernels default `fast_math=True`. So every `/`, `wp.sqrt`, `1/x` and `1/wp.sqrt` in those kernels runs `MUFU.RCP`/`SQRT`/`RSQ` on the L4. — local fork of [newton-physics/newton](https://github.com/newton-physics/newton)

---

## 6. Apple GPU / Metal

### MSL specification (v4.1, 2026-06-04)
All quotes in this subsection are from the [Metal Shading Language Specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf).
- `-fmetal-math-mode=<fast, relaxed, safe>`, default **fast**:
  - "fast … no NaNs, no INFs, no signed zeros, allow reciprocal, allow reassociation, and FP contract to be fast."
  - "relaxed … honors INFs and NaNs. These include no signed zeros, allow reciprocal, allow reassociation, and FP contract to be fast."
  - "safe … disables unsafe floating-point optimizations by preventing the compiler from making any transformations that might affect the results. This sets the FP contract to on."
  - `-fmetal-math-fp32-functions=<fast|precise>` defaults to fast. `-ffast-math` is equivalent to fp32-functions=fast plus math-mode=fast; `-fno-fast-math` is equivalent to precise plus safe. Pragmas: `#pragma METAL fp math_mode([relaxed | safe | fast])` and `#pragma METAL fp contract([off | on | fast])`; flag `-ffp-contract=off`. (§1.6)
  - In LLVM fast-math-flag terms: fast ≈ `nnan ninf nsz arcp reassoc contract(fast)` plus fast function variants (`afn`-like). relaxed drops `nnan`/`ninf`. **[inferred mapping]**
- `metal::fast` and `metal::precise` namespaces select variants of most Table 6.4 functions, including `T divide(T x, T y)` "Compute x / y" and `sqrt`. (§6.6)
- Table 8.1 (precise / no fast math): `1.0 / x`, `x / y`, `sqrt` and `rsqrt` are all **"Correctly rounded"**. (§8.4)
- **Table 8.2 (fast math, the default):**
  - `1.0 / x` "<= 1 ulp for x in the domain of 2^-126 to 2^126"
  - `x / y` "<= 2.5 ulp for y in the domain of 2^-126 to 2^126"
  - `rsqrt` "<= 2 ulp"
  - `sqrt(x)` "Implemented as x * rsqrt(x) with special cases handled correctly."
  - (§8.4)
- Denormals: "Denormalized single-precision … numbers passed as input to or produced as the output of … arithmetic operations may be flushed to zero." §8.5 adds that when flushing, "the sign of the zero is undefined". Rounding: "Either round ties to even or round toward zero rounding mode may be supported". (§8.1–8.2, §8.5)

### Hardware (AGX / G13 and later) and compilers
- ISA (Dougall Johnson's reverse engineering) has unary `rcp`, `rsqrt` and `rsqrt_special`. "`rsqrt_special` can be used to implement fast `sqrt` as `rsqrt_special(x) * x`, by handling special-cases differently." There is **no sqrt or divide instruction**. The FP32 source and destination wrappers are modelled with `flush_to_zero=True` (FP16 with `False`). — [applegpu ISA docs](https://dougallj.github.io/applegpu/docs.html), [dougallj/applegpu](https://github.com/dougallj/applegpu)
- Mesa AGX compiler (Asahi):
  - `.lower_fdiv = true`, so `a/b` becomes `a * frcp(b)`.
  - `UNOP(frcp, rcp)`, `UNOP(frsq, rsqrt)`, and `case nir_op_fsqrt: return agx_fmul_to(b, dst, s0, agx_srsqrt(b, s0));`.
  - Comment: "The hardware frcp instruction is sometimes off by 1 ULP. For correctly rounded frcp, a refinement step is required. This routine has been exhaustively tested with a modified math_bruteforce." The refinement is `u_2 = fma(fma(-x, u, 1), u, u)`, keeping `u` when the result is NaN.
  - — [Mesa src/asahi/compiler/agx_compile.c](https://gitlab.freedesktop.org/mesa/mesa/-/blob/main/src/asahi/compiler/agx_compile.c), [agx_compile.h](https://gitlab.freedesktop.org/mesa/mesa/-/blob/main/src/asahi/compiler/agx_compile.h)
- Apple's own Metal compiler (microbenchmarks, Philip Turner):
  - "`DIV32` = `RECIP32 + FMUL32`".
  - Throughput: RECIP32 6, RSQRT32 8, SQRT32 8, DIV32 6.01. Precise RECIP32 10.46, Precise DIV32 ≤30.65, Precise SQRT32 15.03. "Precise RSQRT32" costs the same as RSQRT32 (8).
  - So fast sqrt costs the same as rsqrt, consistent with `x*rsqrt_special(x)`. The equal cost of "Precise RSQRT32" suggests **[inferred]** either that the hardware rsqrt already meets the precise (correctly rounded) spec or that precise rsqrt is not separately implemented. Unverified.
  - — [philipturner/metal-benchmarks](https://github.com/philipturner/metal-benchmarks)
- FP32 denormals on Apple GPUs: Modular reports that the Apple M5 "flushes float32 denormals to zero on arithmetic inputs" (verified on device). An Apple developer-forum post finds no control or capability query for preserving denormals. — [Modular docs, decode_e2m1_to_f32_inject](https://docs.modular.com/api/mojo/linalg/fp4_utils/decode_e2m1_to_f32_inject/), [Apple Developer Forums thread 844598](https://developer.apple.com/forums/thread/844598) **[M1–M4 behaviour unverified here]**
- **Implication:** Apple's `rcp` is a different table design from MUFU.RCP (Mesa: "sometimes off by 1 ULP"; MUFU: about 13% not correctly rounded). Apple fast sqrt is `x*rsqrt(x)`, while CUDA fast sqrt is a dedicated `MUFU.SQRT` table. So **neither `fast::divide` nor `fast::sqrt` can reproduce CUDA `--use_fast_math` bits**, and `precise::` (correctly rounded) cannot either, because MUFU is only 87% correctly rounded. **[high confidence that they differ; the size of the mismatch is to be measured]**

---

## 7. Prior art on cross-vendor bit-identical approximate math

- **Hardware-model emulation** is the established technique: sass-math/ptx-math (bit-accurate MUFU in C, see §4), and Cankaya 2026 (exhaustive MUFU probe tables, emulator bit-exact on A100/L40/L40S/H100). — [sass-math](https://github.com/pyxis-roc/sass-math), [arXiv:2606.00279](https://arxiv.org/html/2606.00279v1)
- The same approach applied to tensor cores (bit-accurate models of V100/A100/H100/B200 MMA): "Accurate Models of NVIDIA Tensor Cores". — [Khattak & Mikaitis, arXiv:2512.07004](https://arxiv.org/html/2512.07004v3)
- **Correctly rounded approach** (portable, but does *not* match CUDA fast math): RLIBM-32 "produce the correctly rounded results for all float inputs" — [RLIBM-32, arXiv:2104.04043](https://arxiv.org/pdf/2104.04043); LLVM libc's correctly rounded float math (also built for GPUs) — [LLVM libc math](https://libc.llvm.org/math/); accuracy survey of CUDA/HIP math — [Precision and Performance Analysis of C Standard Math Library Functions on GPUs (SC-W 2023)](https://dl.acm.org/doi/fullHtml/10.1145/3624062.3624166)
- **Correction steps** (one Newton/FMA step on top of MUFU.RCP or Apple rcp gives the correctly rounded reciprocal): Mesa `libagx_frcp` (exhaustively tested) — [agx_compile.c](https://gitlab.freedesktop.org/mesa/mesa/-/blob/main/src/asahi/compiler/agx_compile.c); a Helion PR describing rcp.rn's fast path as "MUFU.RCP + one Newton step, bit-identical to the IEEE reciprocal on the reachable domain" — [pytorch/helion PR #3868](https://github.com/pytorch/helion/pull/3868)
- **Reproducibility background** (non-associativity, atomics, cross-generation MMA differences): [arXiv:2408.05148](https://arxiv.org/html/2408.05148v3), [Verde/RepOps, arXiv:2502.19405](https://arxiv.org/pdf/2502.19405)
- I found no public work that emulates NVIDIA MUFU bit-exactly *on another vendor's GPU*. The sass-math algorithm is pure integer arithmetic (64-bit adds and products, a 128×3 table, and a bit-deposit squarer), so it ports directly to MSL. **[inferred feasibility]**

### Recommended emulation recipe (for implementing in Metal; inferred, to be measured)
1. Compile the emulation kernels with `-fmetal-math-mode=safe`, `-fmetal-math-fp32-functions=precise` and `-ffp-contract=off`, or the matching pragmas. This stops Metal from inserting `arcp`, reassociating or contracting on its own.
2. `mufu_rcp(x)`, `mufu_sqrt(x)`, `mufu_rsq(x)`: port the sass-math integer interpolation. The tables are 128×3, 128×3 and 128×3 `uint` in `constant` memory. Implement `square_approx` with shifts and masks (no `pdep` on Apple, so unroll the 16-bit deposit). Add FTZ special cases.
3. `div.approx.ftz(a,b)` = `ftz(RN(ftz(a) * mufu_rcp(ftz(b))))` using Metal's correctly rounded `*`. `div.full.ftz` adds the 0.25 pre-scale when `|b| > 2^126`.
4. Follow the actual SASS: emit `fma()` where ptxas emitted FFMA (for example `a/c + b/c`), share one rcp between divisions by the same divisor, and use multiply-by-folded-constant for literal divisors. Generate the Metal side from the SASS the Warp kernels actually produced, not from the source.
5. Validate exhaustively against the L4. Cost is 2^23 significands for RCP, 2^24 for SQRT and RSQ, plus the exponent edge cases (|x| near 2^±126 and subnormals).

---

## Measurements (2026-10-11; L4 = GCP g2 NVIDIA L4, sm_89, driver 580.178, Warp 1.16.0 in Isaac Lab 3.0-EA's env; Mac = M4 Max)

Scripts (metalsim, `metalsim/physics/mpm/probes/`): `cuda_approx_dump.py` and `cuda_approx_compose.py` (L4),
`mufu_model.py` (numpy model), `metal_approx_probe.py`, `metal_compose_check.py`, `metal_approx_timing.py` (Mac).

### What Warp's fast_math kernels execute on the L4 (measured, not inferred)
- Warp 1.16's NVRTC PTX for a `fast_math` kernel: `a / b` is **`div.approx.ftz.f32`**, `wp.sqrt` is `sqrt.approx.ftz.f32`,
  `1.0 / x` is `rcp.approx.ftz.f32` (PTX in the kernel cache; ptxas 12.8 `-arch=sm_89` + cuobjdump: `MUFU.RCP` +
  `FMUL.FTZ`, `MUFU.SQRT`). This settles the `div.full` question for Warp: NVRTC's `--use_fast_math` behaves as nvcc's
  (div.approx), not as the four spelled-out flags.
- Bitwise over 2^24 random pairs (|a|, |b| in 2^-40..2^40) plus a 15x15 grid of special values (0, -0, inf, NaN,
  denormals, 2^-126, 2^126 .. 3e38): fast_math `a / b` == `div.approx.ftz.f32` == `mul.rn.ftz(a, rcp.approx.ftz(b))`
  on **100 %**; `div.full.ftz.f32` differs only for |b| > 2^126 (99.9946 %).
- fast_math `wp.sqrt(x)` == `sqrt.approx.ftz.f32` on 100 % of [1, 4); fast_math `1.0 / x` == `rcp.approx.ftz.f32`
  100 %; `1.0 / wp.sqrt(x)` is **not** `rsqrt.approx` (61.2 % equal): Warp's `1/sqrt` stays `MUFU.SQRT` + `MUFU.RCP`.
- Exponent invariance, exhaustive: `rcp.approx.ftz(m 2^e) = rcp(m) 2^-e` for every m in [1, 2) and every e in
  [-126, 125] (2.1e9 inputs, 0 mismatches); `sqrt.approx.ftz` / `rsqrt.approx.ftz` of `m 4^e` for m in [1, 4),
  e in [-63, 62]: 0 mismatches. `rcp.approx.f32` == `rcp.approx.ftz.f32` on all normal inputs.
- Correctly rounded fractions (exhaustive): MUFU.RCP **86.79 %** (+1 ulp 8.75 %, -1 ulp 4.47 %); MUFU.SQRT **83.13 %**
  (-1 ulp 15.51 %, +1 ulp 1.35 %: biased low); MUFU.RSQ 78.34 % (errors up to 2 ulp).
- **The sass-math model is bit-exact on sm_89**: `mufu_model.py` (reimplementation of sass-math rcp/sqrt/rsqrt with its
  tables) equals the L4 on all 2^23 RCP inputs in [1, 2) and all 2^24 SQRT and RSQ inputs in [1, 4); with exponent
  invariance this covers every normal float32. Special cases (division grid above) match the model plus FTZ.
- Compositions in a Warp fast_math kernel (2^20 random triples, `cuda_approx_compose.py`): `x/3.0`, `x/0.1`,
  `x/z + y/z`, `x/z - y/z`, `x/sqrt|y|`, `wp.length`, `wp.normalize`, `wp.cw_div`, `vec / z`, `1.0/x`, `(x*y)/z`,
  `x/(y+z)`, `sqrt(x*x+y*y)`, `2.0*x/z` all equal the per-operation model (MUFU.RCP + FMUL.FTZ per `/`, MUFU.SQRT per
  sqrt, Warp's usual FMA contraction of the dot products) on 100 % of inputs. Unlike the nvcc examples in section 3,
  Warp's `x/z + y/z` is **not** contracted into an FFMA, and `x/3.0`, `x/0.1` are equal to `x * MUFU.RCP(c)` (for these
  constants MUFU.RCP(c) is also the correctly rounded reciprocal, so folding is not distinguishable).

### Metal (M4 Max) against CUDA, bit for bit (`metal_approx_probe.py`, `metal_compose_check.py`)
Reference: the MUFU model, which equals the L4 on every input of these sweeps. rcp: all 2^23 floats in [1, 2);
sqrt: all 2^24 in [1, 4); division: 2^22 random pairs, |a|, |b| in 2^-30..2^30, against FMUL.FTZ(a, MUFU.RCP(b)).

| op | Metal form | = CUDA | correctly rounded | ulp vs CUDA (-1 / +1, others) |
|---|---|---|---|---|
| rcp | `fast::divide(1, x)` (relaxed mode's reciprocal) | 86.14 % | 89.65 % | 4.00 / 9.86 % |
| rcp | `precise::divide(1, x)`, or fast rcp + one FMA Newton step | 86.79 % | 100 % | 8.75 / 4.47 % |
| rcp | **`wp_cuda_rcp` (MUFU.RCP emulation)** | **100 %** | 86.79 % | - |
| sqrt | `fast::sqrt` = `x * fast::rsqrt(x)` (bit-identical) | 67.11 % | 71.57 % | 7.71 / 24.96 %, 0.22 % at 2 ulp |
| sqrt | `1 / fast::rsqrt(x)` | 64.94 % | 70.83 % | 7.36 / 27.53 % |
| sqrt | `precise::sqrt` | 83.13 % | 100 % | 1.35 / 15.51 % |
| sqrt | **`wp_cuda_sqrt` (MUFU.SQRT emulation)** | **100 %** | 83.13 % | - |
| a / b | `fast::divide` = `a * fast rcp(b)` | 87.50 % | 71.56 % | 3.04 / 7.45 %, 2.0 % at 2 ulp |
| a / b | `a * precise rcp(b)` | 88.03 % | 72.90 % | 6.57 / 3.37 %, 2.0 % at 2 ulp |
| a / b | `precise::divide` (correctly rounded) | 70.24 % | 100 % | 17.30 / 12.37 % |
| a / b | **`wp_cuda_div`** | **100 %** | 70.24 % | - |

(CUDA's own division is correctly rounded on 70.24 % of these pairs.) Apple's fast rcp is closer to correctly rounded
than MUFU.RCP but a different function (86 % agreement); Apple's fast sqrt is `x * rsqrt(x)` and much further from
MUFU.SQRT (67 %) than the correctly rounded sqrt (83 %), and biased the other way (+1 ulp 25 % vs CUDA's -1 ulp 15.5 %).

Warp fast_math kernels (16 expressions x 2^20 inputs, `warp.config.metal_fast_math` per mode):
`"cuda"` 100 % on all 16 (division, `1/x`, `x/sqrt(y)`, `length`, `normalize`, `cw_div`, vec / scalar, `x/z +- y/z`,
constant divisors); `"relaxed"` 66-88 % (100 % only for `x/3`, `x/0.1`); `"off"` 58-87 %.

Two Metal compiler details the emulation has to handle: (1) the Metal compiler contracts a visible float product into a
following subtraction (`x/z - y/z` became `fma(x, r, -(y r))`: 82 %), even with `#pragma clang fp contract(off)` inside
the division helper and through `fma(a, r, -0.0f)` (folded back to a multiply); `fma(a, r, z)` with `z` the zero
carrying the product's sign is `RN(a r)` bit for bit and stays uncontracted (100 %). (2) The module's fast-math form
must be fixed when the module is hashed (Warp hashes `module="unique"` kernels at definition), otherwise a later build
can store one form under another's hash.

### Cost and end-to-end result
- Per op (dependent chains, 2^20 threads, M4 Max; `metal_approx_timing.py`): FMA 1.28 ps; division: fast 1.32,
  correctly rounded 3.72, **CUDA emulation 10.8** ps; sqrt: fast 1.25, correctly rounded 2.72, **CUDA emulation
  13.6** ps. A first version (bit-serial squarer, three table loads, 64-bit sums) cost 11.9 / 16.3 ps. Exact MUFU
  results cost about 3x a correctly rounded operation on Apple GPUs (Apple's hardware has a near-correctly-rounded
  reciprocal, so IEEE results are cheap; NVIDIA's non-IEEE rounding is not).
- Newton MPM (Warp branch `metalsim-cuda-fast-math`, MetalSim MPM.md section 8, 1b): `"cuda"` is 1.17-2.22x the
  ms/frame of `"off"` (correctly rounded), `"relaxed"` 0.64-0.94x. Parity against the L4's 16-run floor ensemble:
  `"cuda"` is closest to the L4 on the column (0.144 um vs 0.165 um at 3 s, the L4's iteration counts) and the plate
  (0.95x the largest floor, final force inside the L4's range), but not better overall than `"off"` (anymal 1.01 vs
  0.99x, multi_material 1.20 vs 1.15x). MetalSim's default for Newton MPM: `"off"`.
- Updates to the summary table above, now measured: rows 1-3 hold for Warp 1.16 / NVRTC on sm_89 (div.approx, not
  div.full; `1/sqrt` is **not** fused into `MUFU.RSQ` in Warp kernels, unlike nvcc's `1.0f/sqrtf`); row 4's ptxas
  transformations do not occur in Warp's fast_math kernels (`x/z + y/z` is not contracted; measured); row 6: the
  sass-math tables hold on sm_89 for RCP and SQRT (RSQ also matches over [1, 4)); row 8: implemented and exact.

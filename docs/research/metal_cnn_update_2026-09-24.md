# Camera-RL update on Metal: why conv backward is slow on PyTorch MPS, and what to do (2026-09-24)

Scope: the Isaac Lab / skrl camera-cartpole PPO update (metalsim/learn/ppo.py, train_cartpole_rgb.py):
NatureCNN (conv 3→32 k8 s4, 32→64 k4 s2, 64→64 k3 s1, ReLU; flatten 64·9·9 = 5184 → Linear 512 ELU;
linear actor/critic heads), 4 epochs × 32 minibatches of 2048 over 65,536 images of 3×100×100, Adam,
grad clip 1.0. Machine: M4 Max (40-core GPU), macOS 26, torch 2.14.0, mlx 0.32.2.

Labels: **[source]** = read in the cited source/issue, **[measured]** = measured here (section 4, exact
commands given), **[estimated]** = arithmetic or judgement, not measured.

## 0. The arithmetic that bounds the problem [estimated]

Output sizes 24×24, 11×11, 9×9. FLOPs per minibatch of 2048 (2·MACs):

| layer | forward | input grad | weight grad |
|---|---|---|---|
| conv1 3→32 k8 s4 | 14.5 G | not needed (input is data) | 14.5 G |
| conv2 32→64 k4 s2 | 16.2 G | 16.2 G | 16.2 G |
| conv3 64→64 k3 s1 | 12.2 G | 12.2 G | 12.2 G |
| fc 5184→512 | 10.9 G | 10.9 G | 10.9 G |

Total ≈ 147 GFLOP per minibatch, ≈ 18.8 TFLOP per update (128 minibatches). The 5.9 s update
is therefore ≈ 3.2 TFLOP/s effective. The M4 Max's fp32 peak is ≈ 16 TFLOP/s (40 cores × 128 ALUs ×
2 × ~1.6 GHz; fp16 has the same FMA rate on Apple GPUs, its gain is bandwidth/registers). So the
update runs at ~20 % of peak and is compute-sized: at 2048 per minibatch the per-op launch overhead
(tens of µs × ~100 ops) is ~1 % of a ~46 ms minibatch. The floor at a realistic 60–70 % of peak is
≈ 1.7–2.0 s per update.

## 1. Which path PyTorch MPS conv2d takes today (torch 2.14.0) [source]

All three stages are MPSGraph ops, one cached graph per (shape, dtype, stride, padding, layout) key;
there is no native Metal conv2d training kernel. File:
`aten/src/ATen/native/mps/operations/Convolution.mm` at tag v2.14.0
(https://raw.githubusercontent.com/pytorch/pytorch/v2.14.0/aten/src/ATen/native/mps/operations/Convolution.mm).

- Forward (`_mps_convolution_impl`, L554): `convolution2DWithSourceTensor:weightsTensor:descriptor:`
  (L699), bias added in-graph. Descriptor (`fill_conv_desc`, L464–492): `dataLayout` NCHW for
  contiguous input, NHWC only for exactly-channels-last input on macOS ≥ 15; `weightsLayout = OIHW`
  always ("PyTorch always uses OIHW memory layout for weights", L490).
- Input gradient (`mps_convolution_backward_input`, L761):
  `convolution2DDataGradientWithIncomingGradientTensor:weightsTensor:outputShape:forwardConvolutionDescriptor:`
  (L895), descriptor always built with `at::MemoryFormat::Contiguous` (L892) — i.e. NCHW regardless of
  the tensors' layout. Computed only when `output_mask[0]` (L1109): skipped for the first layer here.
- Weight gradient (`mps_convolution_backward_weights`, L926):
  `convolution2DWeightsGradientWithIncomingGradientTensor:sourceTensor:outputShape:forwardConvolutionDescriptor:`
  (L1062), also NCHW (L1059).
- Native Metal kernels in `aten/src/ATen/native/mps/kernels/Convolution.metal` (v2.14.0): conv3d
  implicit-GEMM (`conv3d_simd`, `conv3d_mpp`) forward only (PR #188802,
  https://github.com/pytorch/pytorch/pull/188802, "4–12× faster than MPSGraph" on an M5 Pro), and a
  forward-only `conv2d` kernel used only for filters with a spatial dimension ≥ 256 (PR #188359,
  https://github.com/pytorch/pytorch/pull/188359, a correctness fix). `Im2Col.metal`/`Col2Im.metal`
  back `F.unfold`/`F.fold`.
- Why channels-last was 4× slower here: the backward descriptors are NCHW-only in 2.14, so an NHWC
  `grad_output` is gathered through MPSGraph's strided path. Issue #193349
  (https://github.com/pytorch/pytorch/issues/193349: bf16 conv backward 0.77 ms contiguous vs 22.87 ms
  channels-last) was fixed by PR #193623 (https://github.com/pytorch/pytorch/pull/193623, landed as commit
  b6ddff2e038e on 2026-08-14, NHWC data-grad and HWIO weight-grad paths), which is on `main` only, not in 2.14.0.
  Its only published gain over contiguous is 0.84 → 0.63 ms on one bf16 shape (≈ 1.3×).
- torch.compile: issue #192551 (https://github.com/pytorch/pytorch/issues/192551) — compiled CNN
  training on MPS 4.5× slower than eager because Inductor's layout optimisation forces channels-last
  into this NCHW-only backward; with `TORCHINDUCTOR_LAYOUT_OPTIMIZATION=0` compiled was 0.72–0.87× of
  eager step time. PR #193191 (skip layout opt for MPS training) was closed unmerged. Inductor's MPS
  backend does not generate conv/matmul kernels (it calls the same aten ops), so it can only fuse the
  pointwise ops around them (see docs/research/pytorch_mps_interop_2026-09-22.md §4).
- Release notes: 2.12 (https://github.com/pytorch/pytorch/releases/tag/v2.12.0) and 2.13
  (https://github.com/pytorch/pytorch/releases/tag/v2.13.0) have no conv2d-performance items for MPS
  (2.13 adds the Conv3d NDHWC fast path #184612); 2.14
  (https://github.com/pytorch/pytorch/releases/tag/v2.14.0) has #188359 and #192303 (correctness:
  large kernels, non-contiguous weights) and #188802 (3-D only). Nothing for conv2d backward.
- Open, not in 2.14: PR #189019 `torch.mps.MetalGraph` capture/replay (dispatch overhead only).
- Optimizer: fused Adam exists for MPS (`"mps"` in `_get_fused_kernels_supported_devices`,
  https://raw.githubusercontent.com/pytorch/pytorch/v2.14.0/torch/utils/_foreach_utils.py L13).

Conclusion (a): the time is inside Apple's MPSGraph convolution-gradient kernels. PyTorch does not
choose the algorithm; the only levers from PyTorch are the layout/dtype fed to those graphs and
replacing the op with a different formulation (unfold+matmul, space-to-depth, a custom kernel).

## 2. Alternatives

- **MLX 0.32.2** (https://pypi.org/project/mlx/0.32.2/, NHWC throughout) [source]:
  forward dispatch `dispatch_conv_2D_gpu` (mlx/backend/metal/conv.cpp L1400–1481,
  https://raw.githubusercontent.com/ml-explore/mlx/v0.32.2/mlx/backend/metal/conv.cpp): Winograd only
  for 3×3 stride-1 with C,O multiples of 32 and C+O ≥ 256 (none of our layers); otherwise implicit GEMM
  (specialised for C ≤ 4 or C%16==0 — all three of our layers), else explicit im2col+GEMM, which is
  also forced whenever input dilation ≠ 1. VJP (`Convolution::vjp`, mlx/primitives.cpp ~L1438,
  https://raw.githubusercontent.com/ml-explore/mlx/v0.32.2/mlx/primitives.cpp): the input gradient is a
  conv with `input_dilation = stride`, so for our stride-4/stride-2 layers it takes the explicit-GEMM
  path over a zero-dilated gradient map (wasteful); the weight gradient uses
  `conv_weight_backward_patches` (as_strided patches, materialised, then `matmul(cotanᵀ, patches)`),
  i.e. im2col + one big GEMM on MLX's steel GEMM kernels. Published numbers: mlx-benchmark
  (https://github.com/TristanBilot/mlx-benchmark, forward-only conv) M4 Max: MLX 0.88 ms vs torch MPS
  0.71 ms; M5, MLX 0.32.2 vs torch 2.13: 0.80 vs 0.55 ms (MLX slower). Forward+backward conv,
  ml-explore/mlx#1313 (https://github.com/ml-explore/mlx/issues/1313, 2024, "PyTorch (MPS) is faster
  than MLX in backward of convolution layer"): PyTorch 3–6× faster on M1 Pro/Max, MLX faster on M3 Max
  (numbers are in an image in the issue; closed 2024-10-31 after mlx PR #1541). No published MLX-vs-MPS number for these shapes or versions.
  Interop: zero-copy DLPack on Metal both directions (MLX PR #3531, see the 2026-09-22 interop file),
  no synchronisation carried — `torch.mps.synchronize()` / `mx.eval` + `mx.synchronize()` at the seam.
- **MPSCNNConvolutionGradient / MPSGraph direct**: exists, not deprecated
  (https://developer.apple.com/documentation/metalperformanceshaders/mpscnnconvolutiongradient); no
  Python binding; no published performance data found. MPSGraph is what torch already calls, so calling
  it ourselves buys only fewer graph executions, not faster kernels [estimated].
- **Custom Metal kernel** (implicit-GEMM or Winograd via `torch.mps.compile_shader` or a `.mm`
  extension): feasible with the documented interop; PyTorch's own conv3d implicit-GEMM (#188802) shows
  4–12× over MPSGraph for 3-D. No public conv2d-backward Metal kernel found (ggml/llama.cpp
  `conv.metal` is inference-only, https://github.com/ggml-org/llama.cpp/tree/master/ggml/src/ggml-metal).
  Cost: three layers × (dgrad, wgrad) kernels plus tuning; high effort.
- **tinygrad 0.14.0** Metal: its CI skips full CIFAR training on Metal as "slow on metal"
  (https://raw.githubusercontent.com/tinygrad/tinygrad/master/.github/workflows/benchmark.yml
  L158–169); no published M-series training numbers. Not pursued.
- **JAX-Metal**: last release 0.1.1, 2024-10-08 (https://pypi.org/pypi/jax-metal/json), effectively
  abandoned; the community PJRT plugins jax-mps (https://github.com/tillahoffmann/jax-mps) and
  metaljax (https://github.com/eterevsky/metaljax) lower convolution to MLX, so they inherit MLX's
  kernels. Not pursued separately.
- **Apple's ML frameworks**: ML Compute deprecated since macOS 14.3
  (https://developer.apple.com/documentation/mlcompute/mlcdevice); Core ML `MLUpdateTask` is
  Swift/ObjC and for updatable NeuralNetwork models only — not usable for a Python PPO loop.

## 3. Where the time goes, per others [source]

No published split of MPS conv backward into input- vs weight-gradient was found. #192551 reports
conv backward at 84 % of backward time (in the channels-last-penalty regime). #112956
(https://github.com/pytorch/pytorch/issues/112956) shows non-fused `optim.step()` dominating for small
MLPs on MPS; irrelevant at our size unless measured otherwise. Our own measurement follows.

## 4. Measurements [measured]

M4 Max, macOS 26, torch 2.14.0, mlx 0.32.2, each run holding the GPU through the queue
(`scripts/gpu_run.sh`, lock holder printed before every part; no other GPU job running). Logs:
runs/camera_update/profile_1.log, profile_2.log. Commands:

    scripts/gpu_run.sh camera_update_profile timing 12 -- .venv/bin/python scripts/diagnostics/camera_update_profile.py
    .venv/bin/python scripts/diagnostics/camera_update_profile.py --parts prep,updvar
    TORCHINDUCTOR_LAYOUT_OPTIMIZATION=0 .venv/bin/python scripts/diagnostics/camera_update_profile.py --parts updvar

Implied env-steps/s = 1024 × 64 / (1.4 s rollout + update).

### 4.1 Whole update (4 epochs × 32 minibatches of 2048)

| variant | update (s) | vs baseline | implied env-steps/s |
|---|---|---|---|
| PPO.update as shipped, fp32 eager | 6.67 | 1.00× | 8,120 |
| per-minibatch fp16 autocast, × 128 | 5.56 | 1.20× | 9,414 |
| per-minibatch bf16 autocast, × 128 | 5.59 | 1.19× | 9,371 |
| PPO.update, net.forward under torch.compile (inductor) | 5.38 | 1.24× | 9,665 |
| same, + fused Adam, TORCHINDUCTOR_LAYOUT_OPTIMIZATION=0 | 5.29 | 1.26× | 9,793 |
| MLX fp32, whole update in MLX (gather, loss, clip, Adam; mx.compile step) | 4.44 | 1.50× | 11,218 |
| MLX fp16, whole update in MLX | 3.79 | 1.76× | 12,625 |

(The isolated update measures 6.67 s, not the 5.9 s quoted in docs/GAPS.md; the training log
runs/camera_cartpole_tier0.log implies 6.7 s/iteration of update at 76 % of 8.76 s, consistent with 6.67.)

### 4.2 Split of one fp32 minibatch (51.7 ms total)

| piece | ms | × 128 (s) |
|---|---|---|
| gather `img[idx]` uint8 (random rows, 61 MB) | 3.2 | 0.40 |
| `float()*1/255` | 1.9 | 0.24 |
| per-image mean over dims (2,3) | 3.8 | 0.48 (0.6 ms if taken over a flattened HW view) |
| subtract mean | 2.3 | 0.29 |
| conv/fc forward (sum of layers) | 8.1 | 1.04 |
| backward (conv1 wgrad 8.9, conv2 dgrad 6.4 + wgrad 4.3, conv3 dgrad 1.6 + wgrad 3.8, fc 2.1) | 27.0 | 3.46 |
| clip_grad_norm + Adam (fused: 4.7) | 5.1 | 0.65 |

Forward+loss+backward alone: 44.7 ms (5.72 s/update). Conv backward = 25 ms of it, i.e. about 48 %
of the update; preprocessing ≈ 11 ms (21 %); optimizer 10 %; the KL `.item()` sync is 0.13 ms (0.3 %).

### 4.3 Per layer, fp32 (fwd / input grad / weight grad, ms, and achieved TFLOP/s)

| layer | torch MPS (MPSGraph) | MLX 0.32.2 |
|---|---|---|
| conv1 k8 s4 | 3.6 (4.1) / [23.3, not needed] / **8.9 (1.6)** | 2.7 (5.3) / [31.1] / 7.5 (1.9) |
| conv2 k4 s2 | 2.0 (8.0) / **6.4 (2.5)** / 4.3 (3.8) | 1.9 (8.7) / 3.6 (4.5) / 4.7 (3.5) |
| conv3 k3 s1 | 1.3 (9.6) / 1.6 (7.9) / 3.8 (3.3) | 1.5 (8.3) / 2.0 (6.0) / 3.3 (3.7) |
| fc 5184→512 | 1.2 / 1.0 / 1.1 (~10) | – |

fp16 changes these by 0–20 % (same FMA rate). Formulation changes inside torch, fp32 encoder fwd+bwd:
MPSGraph 35.9 ms, space-to-depth (stride-1 convs) 35.6 ms, unfold+matmul 68.0 ms — no gain.

### 4.4 What the numbers say

- The forward and the conv3 / fc gradients already run at 8–11 TFLOP/s. The losses are concentrated:
  conv1 weight gradient (1.6 TFLOP/s, 8.9 ms), conv2 input gradient (2.5 TFLOP/s, 6.4 ms), and ~11 ms
  of unfused preprocessing per minibatch (a pure PyTorch-eager cost, not a conv cost).
- MLX's kernels are not uniformly faster (conv2 dgrad 1.8× faster, conv1 wgrad 1.2×, conv3 slower);
  its whole-update 1.50× (fp32) comes as much from fusing preprocessing/optimizer under `mx.compile` as
  from the convs.
- torch.compile works on MPS for this net in 2.14 (0.8–6 s compile, no graph breaks, layout
  optimisation on or off makes < 2 % difference here) and gives 1.24–1.26×.

## 5. Recommendation

Criterion from the brief: implement only if ≥ 1.5× on the update while PPO stays in PyTorch.

1. **MLX encoder behind a torch.autograd.Function** — the only measured alternative at ≥ 1.5×, but that
   1.50× is for the *whole* update in MLX. Keeping the heads, loss, and Adam in torch (fused Adam 4.7
   ms, heads/loss ~1 ms) and adding two sync seams per minibatch gives an **estimated 5.0–5.3 s
   (1.25–1.35×) in fp32** and ~4.1 s (1.6×) in fp16 — the fp16 variant cannot meet the 1e-4 fp32
   gradient-equality test. Estimated, so it should be prototyped and measured before being adopted;
   expect it to miss the bar in fp32.
2. **torch.compile of the ActorCritic forward + fused Adam** — measured 1.26× (5.29 s), zero new
   dependencies, bit-for-bit the same model. Adding per-image means precomputed once per rollout
   (instead of 4 × per sample) and folding centering into conv1's bias is estimated to take another
   0.2–0.4 s.
3. **Targeted custom Metal kernels** (via `torch.mps.compile_shader`) for the two slow gradients only:
   conv1 weight gradient (a 32 × 192 × 1.18 M reduction GEMM) and conv2 input gradient (a transposed
   stride-2 conv). Bringing both to ~7 TFLOP/s saves ~10 ms per minibatch ≈ 1.3 s per update.
   Combined with (2): **estimated ≈ 3.9–4.2 s (1.6–1.7×, ~12K env-steps/s)**. Highest effort, and the
   only path estimated to clear 1.5× in fp32 with PPO in torch.

Estimated best achievable within this design: update ≈ 4 s, i.e. ≈ 12K env-steps/s including training
(vs 8.1K measured now, 32K Isaac/4090). Beyond that the floor is ≈ 1.7–2.0 s from the arithmetic in §0.

## 6. Step 3: what was implemented, and the result [measured]

Decision taken: option 2, then option 3 (no MLX wrapper). Code: `metalsim/learn/ppo.py`
(`PPOConfig.fast_update`, `PPOConfig.metal_conv_kernels`, both default on; the plain path is unchanged
behind `fast_update=False`), `metalsim/learn/metal_conv.py` (kernels). Tests: `tests/test_metal_conv.py`,
`tests/test_camera_update_fast.py` (11 pass). Logs: runs/camera_update/profile_3*.log, profile_final.log.

**3a — fast update path** (same math): per-image channel means computed once per rollout step over a
flattened view (0.6 ms vs 3.8 ms for `mean(dim=(2,3))`) and stored with the rollout; uint8 → centered
float in one expression (one fused kernel under compile); the minibatch loss (forward, distribution,
clipped policy and value loss) under `torch.compile(dynamic=False)`; grad-norm clipping as one compiled
graph; fused Adam. Two measured surprises: `nn.utils.clip_grad_norm_` costs 4.1–4.3 ms per call on MPS
for this 2.7 M-parameter net (foreach is not supported on MPS) vs 0.26 ms compiled — worth more than the
preprocessing fix; and folding the centering into conv1's bias (as first proposed) is exact algebra but
loses fp32 precision through cancellation (conv1 weight-gradient error vs fp64 3e-5 instead of 4e-6,
CPU), so the centering is an explicit fused pass instead.

**3b — two Metal kernels** via `torch.mps.compile_shader`, each a `torch.autograd.Function` (forward
= `F.conv2d`, backward swaps in the kernel for that one gradient) whose kernel entry is a
`torch.library.custom_op`, so it sits inside the compiled region:

| op (minibatch 2048, fp32) | MPSGraph | Metal kernel | speed-up |
|---|---|---|---|
| conv1 weight gradient (3→32, k8 s4) | 6.1–8.9 ms, 1.6–2.4 TFLOP/s | 1.55–1.69 ms, 8.6–9.3 TFLOP/s | 3.6–5.3× |
| conv2 input gradient (32→64, k4 s2) | 6.5–6.9 ms, 2.4–2.5 TFLOP/s | 2.8–2.9 ms, 5.5–5.7 TFLOP/s | 2.3× |

- conv1 wgrad: split-K GEMM (32 × 192 output, reduction over N·576 positions). For fixed
  (n, c, kh, oh) the im2col block over 8 consecutive ow × 8 kw is an 8×8 matrix with row stride 4, read
  directly with `simdgroup_load` (no im2col buffer, no threadgroup memory); 12 simdgroups per
  threadgroup each own 2 column tiles × 4 row tiles, 4 samples per threadgroup, then a partial-sum
  reduction. The sweep that picked this (runs/camera_update/profile_3e.log): 3 column tiles per
  simdgroup runs at 2.4 TFLOP/s (register spill), 2 at 9.3; staging through threadgroup memory
  (1.4–1.7) and software pipelining (3.0–4.8) were both slower than the direct loads.
- conv2 dgrad: per input parity (py, px), a stride-1 2×2 correlation with a sub-kernel; with the output
  gradient zero-padded to 13×13 the A tile (8 positions × 8 channels) is a transposed `simdgroup_load`
  straight from memory; four (N·156 × 256)·(256 × 32) GEMMs, output scattered to the stride-2 pixels.
- Accuracy: kernels vs PyTorch's MPS gradients ≤ 1e-4 (max-abs relative) on random data, including
  a ragged batch (37). End-to-end, the exact reference is fp64 on the CPU: PyTorch's own fp32 MPS path
  is 4e-4 off on conv1's weight/bias gradient (cancellation over zero-mean centered pixels) and 2e-4 on
  conv2's weight gradient; the fast path is 4e-4 / 8e-7 — never less accurate than PyTorch's, and within
  1e-4 of PyTorch's gradient everywhere PyTorch is within 1e-4 of exact (that is what the test asserts).

**Update time** (same process, same random buffers, 5 reps each, median;
`--parts update,fast,fastk,update --update-reps 5`):

| variant | update (s) | vs plain | implied env-steps/s (1.4 s rollout) |
|---|---|---|---|
| plain (fast_update=False), bracketing runs | 7.18 / 7.13 | 1.00× | 7,636 / 7,686 |
| 3a: fast_update, no kernels | 5.23 | 1.37× | 9,880 |
| 3b: fast_update + Metal kernels (shipped default) | 3.86 | 1.86× | 12,460 |
| hardware floor, §0 (estimated) | 1.7–2.0 | 3.6–4.2× | 19,000–20,000 |

The plain update measured 6.67–6.71 s earlier in the day and 7.13–7.20 s in the later runs (the
machine had been under continuous GPU load; the lock was held in every run). The ratios are therefore
taken within one process. In the final run another agent's process
(scripts/diagnostics/newton_transfer.py) was running outside the queue; the numbers match the
clean run just before it (runs/camera_update/profile_3f.log: 7.20 / 5.20 / 3.86 s).

Where the 3.9 s goes now (per minibatch, runs/camera_update/profile_3e.log with the earlier conv1
kernel, adjusted): gather + compiled forward ≈ 12 ms, backward ≈ 17 ms (conv2 and conv3 weight
gradients through MPSGraph at 3.3–3.8 TFLOP/s ≈ 8 ms are now the largest single items), clip + Adam
≈ 1 ms. Remaining gap to the floor: those two MPSGraph weight gradients, conv1's forward at 4 TFLOP/s,
and the uint8 gather (3 ms).

### 6.1 Camera-cartpole training with the shipped defaults [measured]

    scripts/gpu_run.sh camera_cartpole_fast train 20 -- .venv/bin/python -m metalsim.learn.train_cartpole_rgb \
        --envs 1024 --tier 0 --steps 8000000 --log runs/camera_cartpole_tier0_fast.log

Same config and seed (42) as runs/camera_cartpole_tier0.log. Return / episode length, mean over ±2
iterations at matched iterations (cumulative env-steps/s in the last columns):

| it | env-steps | return ref | return fast | length ref | length fast | steps/s ref | steps/s fast |
|---|---|---|---|---|---|---|---|
| 1 | 65,536 | 13.4 | 13.0 | 33.2 | 33.0 | 6,733 | 10,131 |
| 10 | 655,360 | 45.4 | 44.6 | 72.1 | 71.2 | 7,324 | 11,467 |
| 20 | 1.31 M | 68.5 | 67.2 | 102.1 | 98.7 | 7,421 | 11,387 |
| 40 | 2.62 M | 82.0 | 78.1 | 124.3 | 117.7 | 7,463 | 10,983 |
| 60 | 3.93 M | 82.7 | 87.7 | 128.1 | 133.0 | 7,484 | 11,183 |
| 100 | 6.55 M | 86.2 | 90.0 | 134.3 | 137.0 | 7,474 | 11,363 |
| 123 | 8.06 M | 84.9 | 91.4 | 132.4 | 140.2 | 7,484 | 11,416 |

Last 20 iterations: return 86.3 (ref) vs 89.7 (fast), length 135.1 vs 137.5. The curves agree within
what one seed per side can resolve (the fast path is not bitwise identical, so the runs diverge in
their random streams from the first update). Whole run: **8.06 M steps in 706 s = 11,416 env-steps/s
including training** (ref 1077 s, 7,483/s: 1.53×); time split env 0.5 % / policy (rollout incl.
rendering) 32.9 % / update 66.6 %. Isaac Lab publishes 32K on an RTX 4090. With the update at the
estimated hardware floor (1.7–2.0 s) and the rollout unchanged, the same pipeline would run at
≈ 16–17K env-steps/s (estimated); past that the rollout (~1.9 s per iteration here) is the next limit.

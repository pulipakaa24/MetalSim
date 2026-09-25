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

## 4. Measurements
(filled in below)

# Tier 2 vs Isaac Sim RTX on the G1 parity scene: what RTX computes, and the plan to match it

2026-09-25. Scope: close the measured rendering gap between MetalSim tier 2 and Isaac Sim 5.1's RTX renderer on
the fidelity-protocol scene (PARITY §1.7: G1, sun DistantLight 3000 (1, 0.98, 0.95) rotated 50° about y, dome 400
(0.75, 0.8, 0.9), grey PreviewSurface ground diffuse 0.5 roughness 0.7, camera (3.2, −2.4, 1.4) → (0.3, 0, 0.6),
42° vertical FOV, 1024×576). Numbers are **measured** unless labelled **estimated**; "unverified" means no primary
source was found.

## 1. Research

### 1.1 Materials: OmniPBR and UsdPreviewSurface

The G1 asset (`assets/isaac/G1/g1_minimal.usd`) binds 43 OmniPBR materials that author **only**
`diffuse_color_constant` (0.7 grey plates, 0.2 dark parts); everything else is at OmniPBR's defaults
(`reflection_roughness_constant` 0.5, `metallic_constant` 0, `specular_level` 0.5, no emission, no texture).

**OmniPBR** ([OmniPBR.mdl, OmniPBR_ClearCoat.mdl, OmniPBRBase.mdl in NVIDIA/simready-foundation](https://github.com/NVIDIA/simready-foundation/blob/main/sample_content/common_assets/robots_general/ur10/simready_usd/)), verbatim structure of `OmniPBRBase`:

```
diffuse_bsdf           = df::diffuse_reflection_bsdf(tint: base_color, roughness: 0.0)        // Lambert
ggx_smith_bsdf         = df::microfacet_ggx_smith_bsdf(roughness_u = r², roughness_v = r², tint 1, scatter_reflect)
custom_curve_layer     = df::custom_curve_layer(normal_reflectivity 0.08, grazing_reflectivity 1.0, exponent 5,
                                                weight: specular_level, layer: ggx_smith_bsdf, base: diffuse_bsdf)
metal                  = df::tint(base_color, ggx_smith_bsdf)                                  // no Fresnel
omni_PBR_bsdf          = df::weighted_layer(weight: metallic, layer: metal, base: custom_curve_layer)
emission               = enable_emission ? emissive_color * emissive_intensity : 0             // df::diffuse_edf
```

So: GGX alpha = roughness², Schlick-shaped curve 0.08 → 1 with exponent 5 scaled by `specular_level` (F0 = 0.04 at
the default 0.5, F90 = 0.5), Lambert base attenuated by the layer, metals = GGX tinted by the base colour.
`albedo_desaturation/add/brightness` only act on a diffuse **texture**; with a constant colour they do nothing;
`diffuse_tint` multiplies. Emission: MDL's default EDF mode is radiant exitance (unverified for RTX).

**UsdPreviewSurface** ([spec](https://openusd.org/release/spec_usdpreviewsurface.html); Hydra Storm's reference
[previewSurface.glslfx](https://github.com/PixarAnimationStudios/OpenUSD/blob/release/pxr/usd/plugin/usdShaders/shaders/previewSurface.glslfx)):
metallic workflow by default, `ior` 1.5 → F0 = ((1−ior)/(1+ior))² = 0.04 for dielectrics, F0 = F90 = albedo for
metals; roughness "usually squared before use with a GGX lobe"; Storm: GGX D with alpha = r², Schlick-GGX geometry
k = alpha/2, F = mix(F0, F90, (1−E·H)⁵), diffuse = albedo/π · (1−metallic) · (1−F). RTX claims compliance with the
spec ([Omniverse UsdPreviewSurface](https://docs.omniverse.nvidia.com/materials-and-rendering/latest/templates/UsdPreviewSurface.html));
its exact MDL mapping is unverified.

**MetalSim before this work** (`pathtrace.metal`): one model for all materials: Lambert·(1−F)(1−metallic) + GGX
(alpha = r², Smith height-correlated-style G), F0 = 0.08·specular, F90 = 1, i.e. already GGX/Schlick with the same F0
as both references. Differences: OmniPBR's F90 is 0.5 (not 1) at the default `specular_level`, the base is attenuated
by the layer's directional reflectance rather than by F(V·H), Storm uses k = alpha/2 geometry. The importer
(`usd_to_mjcf.py`) already carries `diffuse_color_constant`·`diffuse_tint`, roughness, metallic, opacity and emission.
**Expected effect on the robot: small** (dielectric, F0 0.04, roughness 0.5).

### 1.2 Lights: units RTX applies to USD lights

* UsdLux ([LightAPI](https://openusd.org/release/api/class_usd_lux_light_a_p_i.html)): a light is normalised so that
  1 nit at EV0 gives pixel value 1; DistantLight `sizeFactor = π·sin²(θmax)`; with `normalize` on, `intensity` becomes
  the illuminance in lux, independent of `angle` ([DistantLight](https://openusd.org/release/api/class_usd_lux_distant_light.html):
  `angle` = angular diameter, 0.53° for the Sun). Dome lights: radiance = intensity × colour (nits), `sizeFactor` 1.
* RTX ([Omniverse lighting](https://docs.omniverse.nvidia.com/materials-and-rendering/latest/lighting.html)): "Previously, RTX did
  not respect the value of `inputs:normalize`, assuming that the light was always normalized"; the new behaviour needs
  `omni:rtx:usdluxVersion` ≥ 2505, default 2411 → Isaac Sim 5.1 (Kit 107.3) uses the old, always-normalised semantics.
* **Measured from Isaac's own frames** (this work, `runs/parity/isaac/parity_out2/{rt,pt}`): fitting a single
  exposure and tone curve to the sky pixel (dome seen directly, radiance 400·colour) and the sunlit ground
  (0.5/π · (E_sun + π·400·colour)), the only combination that fits both to < 1 level of 8 bits is
  **E_sun = π · 3000 · cos θ** with an ACES curve (rms 0.9 levels; with E_sun = 3000·cos θ every curve misses by
  12–21 levels, table in §1.3). So in Kit 107.3 a DistantLight of intensity I behaves like a surface of radiance I
  seen over π sr: irradiance normal to it = π·I. This is exactly the stack's existing convention (irradiance =
  π · MuJoCo `diffuse`), so a light `diffuse` of 3000·colour and a dome of 400·colour are RTX's units.
  (The documented UsdLux reading, irradiance = I lux, does not fit the frames; recorded as a discrepancy.)
* The sun disk: 0.53° angular diameter → penumbra width ≈ d·tan 0.53° = 9 mm per metre of occluder distance
  (2–3 px on the robot's shadow at this camera). Same normalisation as Cycles' sun (`area = π sin²(angle/2)`,
  [Cycles light.cpp](https://projects.blender.org/blender/blender/src/branch/main/intern/cycles/scene/light.cpp)).
* MetalSim before: sun `diffuse` 0.9·colour and sky = dome colour (≈0.82), i.e. sun : dome irradiance ≈ 1.1 : 1 where
  RTX's is π·3000·0.643 : π·400·0.82 ≈ 5.9 : 1 (dome-dominated → blue cast), plus MuJoCo's default headlight
  (0.4, view-aligned, unshadowed) that Isaac does not have, and a delta sun (hard shadow).

### 1.3 Exposure and tone mapping

* RTX post-processing ([rtx_post-processing](https://docs.omniverse.nvidia.com/materials-and-rendering/latest/rtx_post-processing.html)):
  operators Clamp, Linear, Reinhard, Reinhard (modified), HejlHableAlu, HableUc2, Aces, Iray; physical camera
  `filmIso` 100, `cameraShutter` 50, `fNumber` 5 (defaults as used by Falcor's Kit-USD importer,
  [USDImporter.cpp](https://github.com/NVIDIAGameWorks/Falcor/blob/master/Source/plugins/importers/USDImporter/USDImporter.cpp));
  exposure = 0.01·ISO / (shutter·N²) = 8.0·10⁻⁴ (Falcor's ToneMapper formula; RTX's own formula unverified).
  Default operator, white point, `cm2Factor` and auto-exposure: **unverified** (no defaults on the docs page).
  Isaac Sim 5.1's `isaacsim.exp.base.kit` and Isaac Lab's `apps/*.kit` set no tonemap keys, so Kit's defaults apply.
* **Measured**: the RTX sky reads (168.0, 173.0, 179.9) for a dome radiance of (300, 320, 360) nits. A linear
  exposure + sRGB can match the sky alone (k = 1.27·10⁻³) but then puts the lit ground at 200 vs 230; the Narkowicz
  ACES fit with k = 8.7·10⁻⁴ and π·I sun matches sky and ground to 0.9 levels, and also reproduces the sky's
  *desaturation* (the dome colour 0.75/0.8/0.9 reads 168/173/180, less blue than a linear sRGB encode would give).
  k = 8.7·10⁻⁴ is 9 % above the documented 8.0·10⁻⁴; the residual is attributed to the unverified ACES variant
  (RRT+ODT vs Narkowicz's fit) and is reported, not hidden: both constants are measured below.
* Fit of a single exposure k per (curve, sun convention) to RTX PT's sky (168/173/180) and near sunlit ground
  (229.8/229.5/229.2), measured from the frames (rms error in 8-bit levels):

  | curve | E_sun = I·cos θ | E_sun = π·I·cos θ |
  |---|---|---|
  | linear + sRGB | k 1.52e-3, rms 12.3 | k 1.29e-3, rms 18.1 (ground clips) |
  | ACES (Narkowicz) + sRGB | k 1.15e-3, rms 18.8 | **k 8.73e-4, rms 0.9** |
  | Hable (Uncharted 2) + sRGB | k 6.89e-3, rms 19.4 | k 4.74e-3, rms 1.3 |
  | Reinhard + sRGB | k 3.55e-3, rms 21.4 | k 2.47e-3, rms 6.2 |

* MetalSim before: linear radiance clamped to 8 bits, no sRGB encode → plates and ground darker, sky bluer
  (191, 204, 229).

### 1.4 Denoising in RTX, and on Apple silicon

* Isaac Sim 5.1 `SimulationApp` defaults ([simulation_app.py v5.1.0](https://github.com/isaac-sim/IsaacSim/blob/v5.1.0/source/extensions/isaacsim.simulation_app/isaacsim/simulation_app/simulation_app.py)):
  renderer `RaytracedLighting`, anti-aliasing 3 (DLSS), `samples_per_pixel_per_frame` 64 written to the path
  tracer's `spp`/`totalSpp`, `denoiser` True → `/rtx/pathtracing/optixDenoiser/enabled`. PT mode: OptiX AI denoiser
  ([rtx-renderer_pt](https://docs.omniverse.nvidia.com/materials-and-rendering/latest/rtx-renderer_pt.html));
  our recording ran PT at 32 spp. RT mode: Real-Time 2.0 uses DLSS Ray Reconstruction "if supported by the GPU and
  not optional" ([rtx-renderer_rt](https://docs.omniverse.nvidia.com/materials-and-rendering/latest/rtx-renderer_rt.html));
  Isaac Lab's `rendering_modes/*.kit` toggle `rtx-transient.dldenoiser`. Isaac Lab's rendering kit disables
  reflections, translucency and indirect diffuse in RT mode and enables sampled direct lighting at 1 spp
  ([isaaclab.python.rendering.kit](https://github.com/isaac-sim/IsaacLab/blob/main/apps/isaaclab.python.rendering.kit)) —
  which is why RT's shadows are brighter than PT's (measured: shadow 166 vs 126 levels).
* **Isaac's RT vs Isaac's PT on identical states (measured, 22 frames)**: whole frame 28.1 dB / 0.983 SSIM, robot-only
  25.9 dB / 0.91 SSIM. This is RTX's own mode-to-mode spread and the realistic ceiling for any single physical
  model compared against both.
* Reusable denoisers on this machine:
  * **Intel Open Image Denoise 2.x**: Metal device since 2.2 (Apple silicon, macOS 13+), shared MTLBuffers since
    2.3.1 (`oidnNewMetalDevice`, `oidnNewSharedBufferFromMetal`) ([CHANGELOG](https://github.com/RenderKit/oidn/blob/master/CHANGELOG.md),
    [README](https://github.com/RenderKit/oidn)); used by Blender Cycles on Apple GPUs since 4.1
    ([release notes](https://developer.blender.org/docs/release_notes/4.1/cycles/)). **Installable here**: PyPI
    `pyoidn` 2.5.0.1 bundles `libOpenImageDenoise` 2.5.0 with `libOpenImageDenoise_device_metal`; Homebrew
    `open-image-denoise` 2.5.1 is bottled. We call its C API through ctypes on the renderer's own command queue with
    zero-copy buffers (`metalsim/render/denoise.py`). Inputs: HDR colour, albedo, normal (`cleanAux` when the aux
    images are sample-averaged).
  * **MPSSVGF / MPSSVGFDenoiser / MPSTemporalAA** (MetalPerformanceShaders, macOS 10.15+, not deprecated,
    [MPSSVGF](https://developer.apple.com/documentation/metalperformanceshaders/mpssvgf)): texture-based, needs
    packed depth+normal textures and (optional) motion vectors, encodes into an external command buffer. Usable in
    principle, but it works on 2-D textures one image at a time (our batch is N env tiles in one buffer) and its
    value is temporal reprojection, which progressive tier-2 accumulation already provides for static cameras.
  * **MetalFX `MTLFXTemporalDenoisedScaler`** (macOS 26): denoise + upscale, needs colour, depth, motion, normal,
    diffuse/specular albedo, roughness and camera matrices; single image per call; not tried here (tier-1 candidate).
  * Core Image `CINoiseReduction`: an LDR sharpening/noise filter, not a Monte-Carlo denoiser; not suitable.
  * Not on Metal: NVIDIA NRD (D3D/Vulkan only), OptiX, DLSS-RR; Mitsuba 3 (LLVM/CUDA), pbrt-v4 GPU (CUDA), Falcor.
  * Costs from the literature: SVGF 4.4 ms at 720p / 10.2 ms at 1080p on a Titan X
    ([Schied et al. 2017](https://research.nvidia.com/sites/default/files/pubs/2017-07_Spatiotemporal-Variance-Guided-Filtering%3A//svgf_preprint.pdf));
    edge-avoiding à-trous 5 iterations 42.6 ms at 1080p on a GTX 285 ([Dammertz et al. 2010](https://jo.dreggn.org/home/2010_atrous.pdf));
    OIDN on an M2 Ultra 0.29 s for Blender's Junkshop frame (Blender 4.1 notes). A small learned denoiser of our
    own would repeat OIDN's U-Net without its training set; not pursued.
  * Batched RL frames (1024 envs × 100×100) are the throughput-critical case: OIDN runs one filter per image; a
    single-dispatch edge-avoiding à-trous over all env tiles inside the render command buffer is the cheap option
    (implemented as `denoise="atrous"`, the spatial half of SVGF, guided by averaged albedo/normal and depth,
    radiance demodulated by albedo).

## 2. Plan (effects on robot-only PSNR are **estimated**)

| step | change | expected effect on robot-only PSNR (estimated) | cost (estimated) |
|---|---|---|---|
| 0 | kinematic replay of Isaac's recorded states (`metalsim/render/rtx_parity.py`) so every frame compares the same pose | removes physics from the metric (all 110 RT / 22 PT frames usable) | – |
| 1 | per-material BRDF: OmniPBR (custom_curve_layer over Lambert) for the robot, UsdPreviewSurface (Storm) for the ground, ground roughness 0.7 | +0.0 – 0.3 dB | ≈0 |
| 2 | USD light units (sun E = π·I·cos, dome radiance = I·colour), sun as a 0.53° disk, no headlight | +2 – 4 dB (removes the dome-dominated blue cast; correct sun/shadow contrast) | +2 random numbers per shadow ray |
| 3 | RTX display transform: Kit exposure 8·10⁻⁴, ACES, sRGB; sky must read 168/173/180 with no per-image gain | +2 – 4 dB raw (plates and ground brighten to RTX's levels) | ≈0 |
| 4 | denoiser: OIDN (Metal) for hero frames, in-buffer à-trous for batches; ground truth 256 spp | vs converged: +3 – 6 dB at 16 spp; vs RTX: +0.2 – 1 dB | OIDN: tens of ms per 1024×576 frame; à-trous: a few ms per 10 M pixels |
| 5 | re-run the three protocols + compare, refresh gallery composites | – | – |

What is expected **not** to be matchable: RT mode's approximations (1-spp sampled direct light, disabled indirect
diffuse, DLSS / DL denoiser; its shadows are 40 levels brighter than PT's); NVIDIA's denoisers (OptiX, DLSS-RR);
the exact ACES variant and white point (unverified); undocumented MDL layer-weighting details of
`custom_curve_layer` (directional-albedo estimate).

## 3. Measurements (all **measured**, 2026-09-25, M4 Max)

Harness: `python -m metalsim.render.rtx_parity --isaac runs/parity/isaac/parity_out2/{rt,pt} --preset P` re-renders
every Isaac frame from Isaac's own recorded root pose and joint angles (kinematic replay: identical states, silhouette
IoU 0.999, robot depth RMSE < 1 mm), so all 110 RT / 22 PT frames compare shading only. Metrics are
`metalsim.parity.compare.image_metrics` (whole frame; robot = union silhouette, grey outside, crop): PSNR / SSIM /
LPIPS (AlexNet) / FLIP. Tier 2 at 16 spp × 8 passes, 3 bounces unless noted. Per-run outputs and `summary.json` in
`runs/render_parity/{rt,pt}_<preset>`; logs `runs/render_parity/*.log`; table script `runs/render_parity/analyze.py`.
"ms/frame" = host path (upload, render, readback) at batch 4, 1024×576.

**Reference points.** Isaac RT vs Isaac PT on identical states: whole 28.1 dB / 0.983, robot 25.9 dB / 0.910.

### 3.1 Steps against RTX real-time (110 frames) and RTX path traced (22 frames)

| step (preset) | vs RTX RT: whole PSNR / SSIM / LPIPS / FLIP | robot PSNR / SSIM / LPIPS / FLIP | vs RTX PT: whole | robot | sky (top rows) | ms/frame |
|---|---|---|---|---|---|---|
| 0 before (`legacy`) | 18.10 / 0.833 / 0.450 / 0.428 | 13.43 / 0.578 / 0.188 / 0.130 | 18.28 / 0.824 | 13.86 / 0.611 | 190 / 203 / 227 | 68 |
| 1a BRDF v1: Storm UsdPreviewSurface, curve-weighted OmniPBR (`brdf_v1`) | 17.77 / 0.830 / 0.453 / 0.434 | 12.85 / 0.530 / 0.202 / 0.137 | 18.04 / 0.821 | 13.24 / 0.564 | 190 / 203 / 227 | 70 |
| 1b BRDF: MDL layering, base × (1 − E_ggx(N·V)) (`brdf`) | 18.02 / 0.833 / 0.450 / 0.430 | 13.31 / 0.566 / 0.191 / 0.131 | 18.20 / 0.825 | 13.73 / 0.600 | 190 / 203 / 227 | 68 |
| 2 + USD light units, 0.53° sun disk, no headlight, shown at the old level (`lights`) | 16.40 / 0.960 / 0.132 / 0.384 | 16.36 / 0.702 / 0.123 / 0.119 | 14.85 / 0.942 | 16.83 / 0.731 | 196 / 208 / 232 | 65 |
| 3 + Kit exposure 8.0e-4, ACES, sRGB (`tonemap`) | 23.49 / 0.973 / 0.092 / 0.148 | 23.98 / 0.856 / 0.060 / 0.040 | 20.07 / 0.958 | 28.24 / 0.931 | 162 / 166 / 174 | 68 |
| 3′ exposure 8.725e-4 (fitted once, global) (`tonemap_cal`) | 23.10 / 0.973 / 0.090 / 0.130 | 24.04 / 0.860 / 0.058 / 0.039 | 19.70 / 0.957 | 27.54 / 0.931 | 168 / 172 / 180 | 68 |
| 3c ground base × Fresnel(N·V) (`tonemap_cal_fvg`) | 24.96 / 0.964 / 0.088 / 0.117 | 23.89 / 0.855 / 0.060 / 0.039 | 28.14 / 0.968 | 28.49 / 0.932 | 168 / 172 / 180 | 67 |
| 3c′ robot and ground × Fresnel(N·V) (`tonemap_cal_fv`) | 24.96 / 0.964 / 0.088 / 0.117 | 23.57 / 0.849 / 0.064 / 0.040 | 28.14 / 0.968 | 28.06 / 0.928 | 168 / 172 / 180 | 68 |
| 3d + camera far clip 100 m (Isaac's `clipping_range`) (`tonemap_cal_fvg` + clip) | 27.84 / 0.980 / 0.064 / 0.106 | 23.89 / 0.855 / 0.060 / 0.039 | 44.87 / 0.995 | 28.50 / 0.932 | 168 / 172 / 180 | 68 |
| 3d with Kit exposure 8.0e-4 (`tonemap_fvg` + clip) | 25.55 / 0.978 / 0.068 / 0.154 | 23.46 / 0.849 / 0.064 / 0.042 | 34.31 / 0.995 | 28.15 / 0.929 | 162 / 166 / 174 | 67 |
| 4a + à-trous denoiser (`atrous_cal_fvg` + clip) | 28.07 / 0.982 / 0.060 / 0.107 | 24.09 / 0.863 / 0.057 / 0.038 | 42.22 / 0.997 | 28.63 / 0.938 | 168 / 172 / 180 | 69 |
| **4b + OIDN (`oidn_cal_fvg` + clip) = new default for the parity scene** | 27.85 / 0.982 / 0.062 / 0.108 | 24.03 / 0.860 / 0.057 / 0.039 | **45.57 / 0.997** (LPIPS 0.005, FLIP 0.027) | **28.83 / 0.938** (LPIPS 0.023, FLIP 0.022) | 168 / 172 / 180 | 90 |

(Sky: mean of rows 5–60; RTX 168 / 173 / 180 in both modes. The step-2 row is shown at the old absolute level, so
its whole-frame PSNR falls; its robot SSIM/LPIPS show the gain.)

Readings:
* Step 1 alone moves little (≤ 0.6 dB either way), as expected for F0 0.04 dielectrics; the Storm variant is
  worse than the MDL-layered one.
* Steps 2 + 3 are the gap: the sun/dome ratio (RTX's sun is π·I·cos, 5.9× the dome's irradiance; ours was 1.1×), the
  headlight, and the missing tone curve/sRGB. Robot PSNR 13.4 → 24.0 dB vs RT, 13.9 → 28.2 dB vs PT.
* 3c is a finding about RTX's UsdPreviewSurface: its diffuse base is weighted by the Fresnel curve at the viewing
  angle, not by a roughness-averaged albedo, so a roughness-0.7 ground darkens to ~110 levels at the horizon (PT).
  With that weighting our ground reproduces RTX PT's row profile to ≤ 1.5 levels at every row (r140 108.8 vs 110.3,
  r150 133.9 vs 134.5, r200 190.1 vs 190.7, r300 218.0 vs 218.3, r560 229.2 vs 229.2). On the robot (OmniPBR) the
  same weighting is 0.4 dB worse, so OmniPBR keeps the albedo-weighted layering.
* 3d: Isaac's camera clips at 100 m (the dome shows beyond); with the same far plane the whole frame vs PT is 44.9 dB.
* Exposure: Kit's documented default (8.0·10⁻⁴) leaves the sky 6 levels dark (162/166/174) and costs 0.3–2.3 dB;
  a single global constant 8.725·10⁻⁴ (fitted to RTX's sky and sunlit ground, no per-image gain) puts the sky at
  167.7/171.7/179.7 vs 168/173/180. The 9 % difference is attributed to RTX's ACES implementation (RRT+ODT) vs the
  Narkowicz fit we use; recorded as not matched exactly.

### 3.2 Denoisers: 16 spp × 4 passes vs 256 spp converged, same model (55 RT-state / 22 PT-state frames)

| option | vs 256 spp: whole / robot PSNR | vs RTX RT: whole / robot | vs RTX PT: whole / robot | hero ms/frame (GPU bench, 16 spp × 4, batch 4) |
|---|---|---|---|---|
| none (64 spp) | 48.4 / 36.2 dB | 27.82 / 23.68 | 44.16 / 28.16 | 30.1 |
| à-trous, 5 iterations, in the render command buffer | 43.8 / 36.7 | 28.07 / 23.92 | 42.11 / 28.39 | 31.6 (+1.5) |
| OIDN 2.5 (Metal device, quality high, shared buffers) | **51.4 / 37.7** | 27.84 / 23.96 | **45.48 / 28.78** | 52.9 (+22.8) |
| (for scale) 256 spp, no denoiser | – | – | – | ≈ 4 × 30 |

OIDN at 64 spp reaches or exceeds the 128-spp undenoised render vs RTX PT and is the most faithful option (+3 dB whole,
+1.5 dB robot vs the converged image); the à-trous filter helps the robot (+0.5 dB) but over-smooths the ground's
horizon gradient (−4.6 dB whole vs converged). **OIDN is the default for hero frames** (fidelity first).

### 3.3 Batched RL frames (Cartpole-RGB, 100×100): cost at 1024 envs, fidelity at 256 envs vs converged

| option | 1 spp: ms / batch (1024 envs) | 1 spp: PSNR vs 256-pass converged | 4 spp: ms / batch | 4 spp: PSNR vs 64-pass converged |
|---|---|---|---|---|
| legacy (no tone map) | 11.05 | – | 37.82 | – |
| sun disk + RTX tone map, no denoiser | 11.28 (+0.2) | 29.71 | 38.55 | 36.69 |
| + à-trous 5 it. | 24.54 | **31.13** | 52.14 | 35.84 |
| + à-trous 3 it. | 19.99 | 30.60 | 47.35 | 36.52 |
| + OIDN (quality fast, envs stacked 163 per filter) | 116.80 | 31.00 | 144.22 | **37.69** |

At the RL budget samples beat denoising: 4 spp without a denoiser (38.6 ms) is 5.6 dB closer to converged than 1 spp
with the best denoiser (24.5 ms). The in-command-buffer à-trous filter is the cheap batched option at 1 spp (+1.4 dB
for +13 ms at 1024 envs), OIDN's is 10× the render cost at this size. Sun disk and tone map cost 0.2 ms per batch.

**Applied to camera RL (2026-09-25, follow-up).** `CartpoleRGBConfig.render_mode="physical"` is now the tier-2
default: MDL-layered OmniPBR BRDF, 0.53° sun disk, ACES + sRGB with `tier2_exposure` 0.25, 4 spp, no denoiser
(`render_mode="legacy"` restores the old path, 1 spp). Measured (`runs/render_parity/job8b.log`, `job9.log`, `job10.log`):

| | env step, 1024 envs (physics + render + reward) | PPO incl. training, 10 it. | return at it. 5 / 10 |
|---|---|---|---|
| legacy, 1 spp | 30.5 ms (33,615 env-steps/s) | 11,301 env-steps/s | 24.8 / 45.7 |
| legacy, 4 spp | 56.9 ms | – | – |
| physical, 4 spp, exposure 1.0 | 56.9 ms | 7,561 | 9.3 / 6.7 (**no learning**: image mean 186 vs 95, contrast halved) |
| **physical, 4 spp, exposure 0.25 (default)** | 56.9 ms | 8,809 | 29.5 / 45.7 |

Exposure 1.0 washes out a scene authored for a linear-clamp display; 0.25 maps its mid-grey to the old display value
and the learning signal matches the legacy run over the first 10 iterations (same seed). Cost: +26 ms per 1024-env
step, all from 4 spp (the tone map and sun disk are free); −22 % end-to-end PPO throughput.

**Full 8 M-step run** (`runs/camera_cartpole_physical_full.log`, train queue, 2026-09-25): physical tier 2, 4 spp, 1024
envs: return 7.8 / 45.7 / 70.0 / 82.8 / 89.4 / 86.0 at iterations 1 / 10 / 20 / 40 / 80 / 123 (mean of the last 20
iterations 85.3), vs tier 0 (`runs/camera_cartpole_tier0.log`) 7.0 / 47.1 / 67.8 / 83.9 / 86.9 / 85.4 (86.3);
8,788 env-steps/s including training. An earlier attempt failed at iteration 1 with NaN losses: its PPO-update MPS
command buffers ran out of GPU memory (`kIOGPUCommandBufferCallbackErrorOutOfMemory`) because the job ran without the
GPU lock; the renderer's output was identical to the successful runs (`runs/camera_cartpole_physical_full_failed_oom.log`).
The path tracer now also zeroes any non-finite path estimate before accumulation, and
`test_physical_preset_batch_is_finite` checks 1024 frames.

**Gallery videos** (`docs/gallery/g1_stage_*.mp4`, 12 files, via `scripts/gallery/g1_stage_videos.sh` + round 2):
re-rendered with `metalsim.parity.side_by_side --tier2_mode rtx` (preset `oidn_cal_fvg`, 16 spp × 4 passes).

### 3.4 Protocol replay (physics re-simulated, `runs/parity/metalsim2`, reports `runs/parity/report_{rt4,pt4}`)

| | whole frame PSNR / SSIM / LPIPS / FLIP | robot, states agreeing (IoU > 0.7): PSNR / SSIM / LPIPS / FLIP (brightness-matched PSNR) | n | robot depth RMSE |
|---|---|---|---|---|
| tier 2 vs RTX RT, before (report_rt2) | 18.0 / 0.876 / 0.401 / 0.446 | 12.5 / 0.43 / 0.144 / 0.099 (13.4) | 52 | 2.7 cm |
| **tier 2 vs RTX RT, now** | **26.8 / 0.971 / 0.082 / 0.112** | **21.5 / 0.714 / 0.071 / 0.055 (21.5)** | 53 | 2.2 cm |
| tier 2 vs RTX PT, before (report_pt2) | 17.6 / 0.860 / 0.435 / 0.448 | 12.8 / 0.45 / 0.146 / 0.104 (13.7) | 12 | 2.9 cm |
| **tier 2 vs RTX PT, now** | **34.6 / 0.984 / 0.029 / 0.034** | **23.0 / 0.762 / 0.050 / 0.046 (23.2)** | 12 | 1.9 cm |

The protocol numbers are below the kinematic ones because "agreeing" states still differ by a few cm (IoU 0.90), which
PSNR/SSIM on a white robot punish at every edge. Tier 0 is unchanged (8.1 dB whole vs RT; it keeps MuJoCo's units).

### 3.5 Render cost per frame, before and after (1024×576, G1 parity scene)

| configuration | GPU bench, 16 spp × 4, batch 4 | protocol setting 16 spp × 8, host path, batch 4 |
|---|---|---|
| before (legacy) | 30.8 ms | 68 ms |
| after, no denoiser (tone map, sun disk, layered BRDFs, clip) | 30.1 ms | 68 ms |
| after, OIDN (default for the parity scene) | 52.9 ms | 90 ms |

## 4. What cannot be matched, and why

* **RTX real-time's approximations.** Isaac Lab's rendering kit disables indirect diffuse, reflections and translucency
  in RT mode and samples direct light at 1 spp with a DL denoiser / DLSS; its shadows are ~40 levels brighter than PT's
  and its horizon darkening weaker. A physically based renderer matches PT, not RT: our robot is 24.0 dB from RT vs
  28.8 dB from PT, and RTX's own RT-vs-PT spread is 25.9 dB on the robot. Emulating RT's shortcuts would mean
  un-physical options (ambient-lit shadows); not done.
* **NVIDIA's denoisers** (OptiX AI denoiser in PT, DLSS / DL denoiser in RT): CUDA/driver-only, no Metal port. OIDN
  (a different U-Net) is the closest available; residual differences are sub-1-dB on the robot.
* **The exact ACES variant / exposure formula** of Kit 107.3 is not documented; our curve (Narkowicz fit) with a
  global exposure 9 % above the documented camera formula matches the sky to ≤ 1.3 levels. A different curve shape
  in the highlights (robot's sunlit plates at 240–246) is the likely remaining brightness error.
* **Undocumented MDL behaviours**: how `custom_curve_layer` and RTX's UsdPreviewSurface weight the base (we inferred
  Fresnel-at-N·V for UsdPreviewSurface from the horizon profile, albedo-weighted for OmniPBR from the robot metrics);
  MDL's multiple-scattering compensation in `microfacet_ggx_smith_bsdf` is not modelled.
* **The light-unit discrepancy**: UsdLux documents DistantLight intensity as illuminance (lux); Kit 107.3's frames fit
  irradiance = π·I. We follow the frames; with `omni:rtx:usdluxVersion` ≥ 2505 RTX may switch behaviour.
* Physics: in the protocol replay the states still differ (the robot-only rows depend on it); see PARITY §1.7.

## 5. Decisions (options with numbers, chosen, how to re-enable the rest)

| decision | options (robot / whole PSNR vs RTX PT, measured) | chosen, why | re-enable the others |
|---|---|---|---|
| BRDF for OmniPBR | legacy 13.86 (old lights); Storm-like curve (`omnipbr_curve`) 13.24; MDL layering, base × (1 − E_ggx(N·V)) (`omnipbr`) 13.73; base × curve(N·V) (`omnipbr_fv`) 28.06 vs 28.49 (with final lights) | `omnipbr`: best on the robot at every stage | `material_model={None: "omnipbr_curve"}` / `"omnipbr_fv"` / `"legacy"` |
| BRDF for UsdPreviewSurface | Storm (`usd_preview_storm`), MDL albedo layering (`usd_preview`): whole vs PT 19.7; Fresnel(N·V) base (`usd_preview_fv`): 28.1 → 44.9 with clip | `usd_preview_fv`: reproduces RTX PT's ground row profile to ≤ 1.5 levels | `material_model={"ground": "usd_preview"}` etc. |
| DistantLight units | irradiance = I (UsdLux doc): every tone curve misses sky + ground by 12–21 levels; irradiance = π·I: ACES fit 0.9 levels | π·I (the frames) | pass `intensity/π` in `usd_lights` |
| sun shape | delta direction vs 0.53° disk (same mean irradiance) | disk (physical penumbra; +0.2 ms per RL batch) | `angle_deg=0` |
| tone curve | linear clamp (legacy); linear + sRGB (fits sky only: ground 200 vs 230); Hable (0.3 levels worse than ACES); ACES + sRGB | ACES + sRGB | `tonemap="linear"` |
| exposure | Kit formula 8.0e-4 (sky 162/166/174; robot vs PT 28.15); global fit 8.725e-4 (sky 168/172/180; 28.50) | 8.725e-4 as the parity default, stated as fitted; 8.0e-4 kept as `rtx_exposure()` | preset `tonemap_fvg` or `exposure=rtx_exposure()` |
| far clip | none vs 100 m (Isaac's camera): whole vs PT 28.1 → 44.9 | 100 m for the parity scene (it is the recorded camera) | `clip_far=0` |
| denoiser, hero frames | none 44.16 / 28.16; à-trous 42.11 / 28.39; OIDN 45.48 / 28.78 (64 spp); vs 256 spp: 48.4 / 43.8 / 51.4 whole | OIDN (most faithful; +23 ms/frame) | `denoise=None` / `"atrous"` |
| denoiser, RL batches | none / à-trous 5 / à-trous 3 / OIDN at 1 spp: 29.7 / 31.1 / 30.6 / 31.0 dB vs converged, 11.3 / 24.5 / 20.0 / 116.8 ms per 1024-env batch | no change to the RL default here (owned by the task code); recommendation: spend on spp (4 spp: 36.7 dB at 38.6 ms); à-trous is the batched option when a denoiser is wanted | `denoise="atrous"`, `atrous_iters` |
| MPSSVGF / MetalFX denoised scaler | not tried: single-image texture APIs, temporal (our frames accumulate progressively); OIDN covered the need | – | – |

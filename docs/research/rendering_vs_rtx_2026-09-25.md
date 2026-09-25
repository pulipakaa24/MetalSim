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
  12–21 levels, table in §3.1). So in Kit 107.3 a DistantLight of intensity I behaves like a surface of radiance I
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

## 3. Measurements

(filled in below as each step is measured)

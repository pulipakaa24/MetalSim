# MaterialX materials: what Isaac Sim reads, what our renderer can carry (2026-09-25)

Scope: USD scenes whose materials are authored in MaterialX, either as `.mtlx` documents referenced
from USD (UsdMtlx) or as MaterialX node networks written directly as UsdShade prims
(`outputs:mtlx:surface`, shader `info:id = ND_*`). Goal: flatten them into the per-geom material
table of `metalsim/render/scene_tables.py` (rgba, metallic, roughness, specular, emission, one RGB
texture with repeat, reflectance).

## (a) Isaac Sim 5.x and MaterialX

* Isaac Sim 5.0 runs on Kit 107.3.x (5.0.0 release notes: "updated to Kit 107.3.1"); 6.0 moved to
  Kit 110.1. In Omniverse, "Materials ... are supported using MDL, and MaterialX using an MDL
  backend" [1]: RTX never shades MaterialX natively; the document (from a `.mtlx` reference or from
  UsdShade prims) is translated to MDL with the MaterialX SDK's MDL code generator
  (`MaterialXGenMdl`), then MDL distilling reduces it for the real-time mode [2, 3].
* Both loading routes are supported: "directly in OpenUSD as a graph of UsdShade nodes" and
  "referencing .mtlx documents into an OpenUSD layer" [3]. Kit 106.1 fixed "MaterialX issues with
  range annotations, nested node graphs, geomprop lookups, saving, and loading" [4]; later RTX
  releases added document version upgrade in `rtx.materialx` [5].
* Node graphs: anything MaterialXGenMdl can generate is accepted in principle (stdlib + pbrlib +
  bxdf: `standard_surface`, `open_pbr_surface`, `UsdPreviewSurface`, `gltf_pbr`,
  `disney_principled`). In practice decomposed BSDF graphs (`layer_bsdf`, `conductor_bsdf`,
  `subsurface_bsdf`, e.g. Blender 5.1's Principled export) fail in the MDL translation and render
  **black** in Isaac Sim 5.1; when a material carries both `outputs:surface` (UsdPreviewSurface) and
  `outputs:mtlx:surface`, RTX resolves the MaterialX terminal, not the preview one [6]. NVIDIA's
  recommendation in that thread is to rebuild as OmniPBR (MDL) or UsdPreviewSurface.
* OpenPBR: implemented as the MaterialX reference graph translated to MDL; the path-traced mode
  supports the full model, RTX Real-Time 2.0 does not support sheen (fuzz) and thin film; SSS must be
  enabled, refraction may need more bounces [2]. Kit 110 (Isaac Sim 6) adds an OpenPBR material
  template [7]. NVIDIA's PhysicalAI-SimReady-Materials are UsdShade MaterialX graphs on
  `open_pbr_surface` (OpenPBR 1.1, OpenUSD 24.08, MaterialX 1.38.10-OpenPBR) [8]. So for Isaac-side
  content the practically relevant graphs are: open_pbr_surface (SimReady), standard_surface (DCC
  exports), UsdPreviewSurface-as-MaterialX; Isaac's own robot assets still use OmniPBR MDL.
* NVIDIA does not publish a per-parameter support table for MaterialX; coverage is "whatever the
  MaterialX→MDL generator produces", limited by the real-time mode as above.

## (b) Packages here

* PyPI `MaterialX` latest = **1.39.5** (queried 2026-09-25; requires Python ≥ 3.9); installed in
  `.venv` = 1.39.5, with the data libraries (`stdlib`, `pbrlib`, `bxdf` incl. `open_pbr_surface.mtlx`,
  `usd_preview_surface.mtlx`, `gltf_pbr.mtlx` and the translation graphs
  `standard_surface_to_usd`, `open_pbr_to_standard_surface`, `standard_surface_to_open_pbr`),
  807 node definitions, and the MDL/GLSL/MSL/OSL/Slang generators.
* `usd-core` 26.8 (PyPI wheel): **`from pxr import UsdMtlx` fails** (ImportError) and no `.mtlx`
  Sdf file format is registered (`Sdf.FileFormat.FindAllFileFormatExtensions()` = usd, usda, usdc,
  usdz). The PyPI wheel is built without MaterialX support. Consequence: a `references = @x.mtlx@`
  arc fails to compose ("Cannot determine file format for @x.mtlx:SDF_FORMAT_ARGS:target=usd@")
  and every prim under it is missing, so bindings to `/…/MaterialX/Materials/<name>` resolve to
  nothing. Our loader therefore does UsdMtlx's job itself: it reads the document with the MaterialX
  Python package (XInclude, version upgrade, `fileprefix`, node definitions/defaults from the
  standard libraries), writes an equivalent UsdShade layer in memory, and swaps the arc in the
  stage's session layer (the user's files are not touched).

## (c) Parameter mapping to our metallic-roughness model

Renderer (tier 0/1/2 shaders, read-only here): `base = rgba.rgb * texture`, GGX with
`a = roughness²` (perceptual roughness, same convention as both MaterialX models),
`F0 = mix(0.08 * specular, base, metallic)`, emitted radiance `= emission * base`, alpha from rgba.a.

| our field | standard_surface | open_pbr_surface | UsdPreviewSurface (ND_) | notes |
|---|---|---|---|---|
| rgba.rgb | `base * base_color` | `base_weight * base_color` | `diffuseColor` | exact |
| texture (RGB) + texrepeat | image/tiledimage on `base_color` (× constant factors → rgba multiplier; `uvtiling`, `place2d.scale`, texcoord multiply → repeat) | same | `UsdUVTexture` (`scale` → multiplier, `UsdTransform2d.scale` → repeat) | one texture slot; bias, rotation, offset not carried |
| metallic | `metalness` | `base_metalness` | `metallic` | exact (constant); textured → texture mean |
| roughness | `specular_roughness` | `specular_roughness` | `roughness` | exact (constant); textured → texture mean |
| specular (F0 = 0.08·s) | `specular · mean(specular_color) · ((IOR−1)/(IOR+1))²` | `specular_weight · mean(specular_color) · ((ior−1)/(ior+1))²` | `((ior−1)/(ior+1))²`, or `specularColor` with `useSpecularWorkflow` | IOR 1.5 → 0.04 → s = 0.5 (MuJoCo default); s clipped to [0, 1] (IOR ≤ 1.8) |
| emission (× base) | `emission · emission_color` | `emission_luminance [cd/m²] / 1000 · emission_color` | `emissiveColor` | scalar: luminance ratio to the base colour; the emission *hue* is not carried (renderer emits in the base colour); open_pbr nit scale is a convention (1000 cd/m² = 1) |
| rgba.a | `mean(opacity)` | `geometry_opacity` | `opacity` | renderer alpha semantics unchanged |

Cannot map (recorded per material as `unmapped` when authored away from the default):
coat / clearcoat (second specular lobe and its absorption), sheen / fuzz, transmission (refraction,
dispersion, scatter), subsurface, thin film, specular anisotropy and tangent, diffuse roughness
(Oren–Nayar), metal edge tint (`specular_color` on metals; open_pbr's F82), normal / bump / coat
normal maps, occlusion and displacement, textures on inputs other than base colour (reduced to the
image mean, recorded as an approximation), procedural nodes (noise, ramps, …; the input falls back
to its default and is recorded), decomposed BSDF graphs (`surface`, `layer_bsdf`, … — the same graphs
that fail in Isaac; the material keeps whatever the regular importer found).

## Sources

1. Omniverse Materials overview — https://docs.omniverse.nvidia.com/materials-and-rendering/latest/materials.html
2. OpenPBR template (Omniverse) — https://docs.omniverse.nvidia.com/materials-and-rendering/latest/templates/OpenPBR.html
3. NVIDIA blog, "Unlock seamless material interchange … OpenUSD, MaterialX and OpenPBR" — https://developer.nvidia.com/blog/unlock-seamless-material-interchange-for-virtual-worlds-with-openusd-materialx-and-openpbr
4. Materials release notes Kit 106.1 — https://docs.omniverse.nvidia.com/materials-and-rendering/latest/materials_release-notes/106_1_0.html
5. RTX renderer release notes — https://docs.omniverse.nvidia.com/materials-and-rendering/latest/rtx-renderer_release_notes.html
6. isaac-sim/IsaacSim discussion #661, "MaterialX from Blender 5.1 renders black in Isaac Sim 5.1" — https://github.com/isaac-sim/IsaacSim/discussions/661
7. Materials release notes Kit 110.0 — https://docs.omniverse.nvidia.com/materials-and-rendering/latest/materials_release-notes/110_0_0.html
8. NVIDIA-Omniverse/PhysicalAI-SimReady-Materials — https://github.com/NVIDIA-Omniverse/PhysicalAI-SimReady-Materials
9. Isaac Sim release notes (Kit versions) — https://docs.isaacsim.omniverse.nvidia.com/6.0.1/overview/release_notes.html ; Isaac Sim 5.1 GA — https://github.com/isaac-sim/IsaacSim/discussions/286
10. MaterialX on PyPI — https://pypi.org/project/MaterialX/ (JSON API queried for the version)

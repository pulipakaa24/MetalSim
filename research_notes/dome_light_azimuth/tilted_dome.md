# Tilted DomeLight: full dome orientation in tier 2 (2026-10-13)

Question: tier 2 applied only a DomeLight's yaw (rotation about +z). A rotation that tilts the pole (a `rotateX` on the
dome or a parent, a DomeLight_1 `poleAxis`) was drawn upright. What should the full orientation be, and how does it
compose with the Kit azimuth convention (`tier2.KIT_DOME_AZIMUTH`, `usd_vs_kit_latlong.md`)?

## Sources

1. **OpenUSD `UsdLuxDomeLight_1`** (openusd.org/release/api/class_usd_lux_dome_light__1.html; user guide
   openusd.org/dev/user_guides/schemas/usdLux/DomeLight_1.html). `uniform token poleAxis = "scene"`, values
   `scene` | `Y` | `Z`: "the starting alignment of the dome light's top pole"; `scene` = the stage's up axis. With a
   Y pole the map follows OpenEXR's latlong (pole +y, longitude 0 toward +z); with a Z pole, latitude ±π/2 is ±z and
   longitude 0 faces −y. The alignment "is intended to be applied by a renderer to only the dome itself" and is
   not inherited by the dome's namespace children. So it is a rotation in the dome's **local** frame, before the
   prim's world transform: `world = R_prim · P_pole · local`. The order is not spelled out, but "not inherited by
   children" only makes sense for a local-frame rotation.
2. **OpenUSD `UsdLuxDomeLight`** (legacy). No `poleAxis`: by the spec the pole is always +Y (OpenEXR).
   `OrientToStageUpAxis()` adds a +90° `rotateX` op on z-up stages so a +Y-pole dome stands upright. NVIDIA's USD
   Exchange SDK (`usdex.core` Light Prims docs) still authors the legacy schema because DomeLight_1 "is not widely
   supported yet".
3. **Kit/RTX, measured** (Isaac Sim 5.1 / Kit 107.3.3 and 6.1 / Kit 110.3.0, legacy `UsdLux.DomeLight` on a z-up stage,
   `docs/research/isaac_side_session_2026-09-26.md` §2, recipe `metalsim/parity/isaac_side/record_dome_azimuth.py`):
   - the un-rotated dome has pole **+z** (Kit ignores the spec's +Y pole for the legacy schema), with the image centre
     toward −x, i.e. the DomeLight_1 Z-pole frame turned by −90° about +z (`KIT_DOME_AZIMUTH`);
   - `rotateZ 90°` turns the map by +90° about +z;
   - **`rotateX 90°` is applied rigidly**: the BLUE marker moved from (−0.252, 0.182, 0.950) to
     (−0.252, −0.950, 0.182) = R_x(90°)·d exactly. So Kit renders `d_world = R_prim · R_z(KIT_DOME_AZIMUTH) · d_spec`,
     where `d_spec` is the DomeLight_1 Z-pole direction of (u, v) and `R_prim` the prim's full world rotation.
     The tilt is not reduced to a yaw, and adding `OrientToStageUpAxis`'s +90° x-rotation on a z-up stage tips the
     map over in Kit.
4. **Omniverse "Lighting" docs** (docs.omniverse.nvidia.com/materials-and-rendering/latest/lighting.html, UsdLux 25.05
   section): "Previously, RTX oriented the dome light such that a default camera would be looking at the 'bottom' of
   the image. Now the default orientation is such that a default camera will be looking at the horizon". The
   section also says Kit previously applied an extra correcting transform to dome lights, and that transform has
   been removed. This is gated per light by `int omni:rtx:usdluxVersion` (absent = 2411, the old behaviour; Kit
   ≥ 109.0 *creates* lights with 2505). An identity camera looks down −z. "Looking at the bottom of the image" is
   the old pole-+z behaviour we measured. "Looking at the horizon" means the new behaviour is the spec's +Y pole.
   Our measurement scripts author the dome through `UsdLux.DomeLight.Define`, without the attribute, so they
   measured the 2411 behaviour on both Kit 107 and Kit 110.
5. Stage `upAxis`: the importer (`usd_to_mjcf`) does not re-orient y-up stages, so MuJoCo world = stage coordinates.
   Kit's legacy-dome rendering does not depend on upAxis: the pole is +z in the dome's local frame (source 3, and
   source 4's "looking at the bottom"). Kit's own Create menu compensates for a y-up stage with an authored rotation.
   This is not measured here; only its effect is, and the rigid composition handles it.

## Adopted convention (importer → tier 2)

`d_world = R_prim · P · R_z(KIT_DOME_AZIMUTH) · d_spec(u, v)`, where `d_spec` is the shader's dome frame (DomeLight_1
Z pole, `hdr.env_direction(u, v)`), and:

| dome | P |
|---|---|
| legacy `DomeLight` (any stage upAxis; measured on z-up) | I (Kit: pole +z locally) |
| `DomeLight_1`, `poleAxis = "Z"`, or `"scene"` on a z-up stage | I |
| `DomeLight_1`, `poleAxis = "Y"`, or `"scene"` on a y-up stage | R_x(−90°) (Z-pole frame → OpenEXR +Y-pole frame: pole +y, the spec's longitude 0 toward +z) |

- Un-tilted (`R_prim · P` a pure rotation about +z): tier 2 keeps the old yaw-only code path (`env_yaw = prim yaw +
  KIT_DOME_AZIMUTH`), so those renders are byte-identical to before.
- Otherwise the importer records the full 3×3 rotation `R_prim · P`, and `set_environment_from_model` passes
  `rotation = R_prim · P, yaw = KIT_DOME_AZIMUTH`. `set_environment(rotation=R, yaw=y)` uploads `R · R_z(y)` to the shader
  (`PTConsts.env_rot`, FLAG_ENVROT). The shader maps a world direction to the map with `Rᵀ d` and a sampled map
  direction back with `R d`. The lookup (`env_radiance`), the pdf (`env_pdf`) and the sampler (`env_sample`) all use
  that one matrix. A rotation preserves solid angle, so the pdf needs no Jacobian beyond the latlong one, and MIS
  stays consistent.

## Not measured (assumptions, flagged)

- How Kit renders a **DomeLight_1** (any `poleAxis`). Applying the Kit azimuth about the dome's own pole is the
  natural extension, but no recording exists. A record with `pole_axis` says so.
- A dome with `omni:rtx:usdluxVersion ≥ 2505` (Kit ≥ 109 UI-created lights). Source 4 implies a +Y pole (the spec's
  frame), but neither its azimuth nor the effect of the removed Kit transform is measured. The importer records the
  value (`rtx_usdlux_version`) and warns, and tier 2 renders the measured 2411 convention. Re-run
  `record_dome_azimuth.py` with the attribute set to settle this.
- A non-uniform scale or shear on the dome is dropped: only the rotation is used, as Kit does for a direction-only
  light.

## Implementation and results (M4 Max, 2026-10-13; logs `runs/dome_tilt/`)

- Shader: `env_uv_c` / `env_dir_c` wrap the lookup, `env_pdf` and `env_sample`. The rotated branch is compiled only
  when a rotation is set (`ENV_ROT` define, as `ENV_MATS`; `PTConsts.env_rot` = words 76..87, before the EnvGrid).
  A first version used only a runtime flag branch. Its upright renders differed from main by 1–2 ulp in 17–21 % of
  HDR values (same paths: rgb, depth, normal and seg were identical). Fast-math codegen had changed around the
  branch (`runs/dome_tilt/suite_v1_runtime_branch.log`). With the compile-time variant,
  `scripts/diagnostics/dome_untilted_bytes.py` finds all 54 output arrays byte-identical to origin/main c3a25d9d.
  The arrays cover the sky colour, `set_environment` at three yaws, `set_environment_pose`, a sky-only camera, and a USD
  `rotateZ 30` dome through the importer.
- Lookup (`test_env_map_tilted_dome_lookup`): a map that stores its own dome-frame direction is seen by a sky-only
  camera. The tilted render equals Rᵀ·(the untilted render) to < 1e-4 for rotateX 30°, rotateX 30° plus a yaw, a general
  quaternion, and the y-up pole with the Kit azimuth. The yaw-only lookup would be off by 0.5–1.4.
- Importance sampling (`test_env_map_tilted_importance_sampling`), map turned 50° about x with a yaw of 0.9:
  - A constant map gives 0.4993 (furnace, expected 0.5).
  - A bright window gives 0.534 by MIS against 0.5331 integrated on the host over the rotated map. The upright
    map would give 0.472.
  - The uniform table gives 0.5367 (|Δ| 0.0027 against 4 SE = 0.024), with 6.3× the per-pixel noise.
- `tests/test_render_tier2.py`: 19 passed. The rest of the render suite (render, replicator, scene_usd, area-light
  and camera tests): 185 passed, 1 skipped.

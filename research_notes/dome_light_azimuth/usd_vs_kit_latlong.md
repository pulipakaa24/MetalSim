# DomeLight latlong orientation: OpenUSD spec vs Kit/RTX vs our tier-2 shader (2026-10-13)

Question: `tests/test_render_tier2.py::test_env_map_usd_dome_import` authored a DomeLight with `rotateZ 30°` and expected
the tier-2 constant `env_yaw` (consts[20]) to be +30°. The renderer stores −60°. Which of importer, renderer and test
has the wrong convention?

## Sources

1. **OpenEXR `ImfEnvmap.cpp`, `LatLongMap`** (github.com/AcademySoftwareFoundation/openexr, `src/lib/OpenEXR/ImfEnvmap.cpp`):
   `longitude = -2π · ((x − min.x)/(max.x − min.x) − 0.5)`, `latitude = −π · ((y − min.y)/(max.y − min.y) − 0.5)`,
   `direction = (sin(lon)·cos(lat), sin(lat), cos(lon)·cos(lat))`. So the pole is +y, the top row is +y, the image
   centre (u = 0.5, lon 0) faces **+z**, u = 0.25 (lon +π/2) faces **+x**, u = 0 faces −z.
2. **OpenUSD `UsdLuxDomeLight_1`** (openusd.org/release/api/class_usd_lux_dome_light__1.html): orientation is set by
   `uniform token poleAxis = "scene"` (`scene` | `Y` | `Z`; note: not an `inputs:` attribute). With a Y pole it
   "adheres to the OpenEXR specification for latlong environment maps"; on a Z-up stage (`scene` or `Z`) "latitudes
   −π/2 and +π/2 correspond to the negative and positive Z direction" and "latitude 0, longitude 0 will instead point
   into the negative Y direction". That is OpenEXR's map rotated +90° about x: pole +z, **u = 0.5 faces −y**,
   u = 0.25 faces +x. The pole-alignment rotation applies to the dome only, not to its namespace children. The legacy
   `UsdLuxDomeLight` (no poleAxis) is always the OpenEXR +Y-pole frame.
3. **Kit/RTX, measured** (Isaac Sim 5.1 / Kit 107.3.3 and Isaac Sim 6.1 / Kit 110.3.0, path-traced frames of a
   three-marker map, 2026-09-26; `docs/research/isaac_side_session_2026-09-26.md` §2, recipe
   `metalsim/parity/isaac_side/record_dome_azimuth.py`): pole +z on a z-up stage for a legacy DomeLight (NVIDIA's
   usd-exchange-samples: "Kit/RTX treats the DomeLight environment as Z-up"), not mirrored,
   **u = −(atan2(y, x) − yaw)/2π**: u = 0 faces +x, u = 0.25 faces −y, **u = 0.5 faces −x**, u = 0.75 faces +y.
   A `rotateZ` on the prim turns the map by +yaw (measured +90.0 ± 0.3° for rotateZ 90°); a `rotateX 90°` is applied
   as a rigid rotation.

## Conventions side by side (z-up stage, un-rotated dome)

| | u = 0 | u = 0.25 | u = 0.5 (centre) | u = 0.75 |
|---|---|---|---|---|
| OpenUSD DomeLight_1, z-up (spec) | +y | +x | −y | −x |
| tier-2 shader `env_uv` / `hdr.env_direction` | +y | +x | −y | −x |
| Kit/RTX (measured) | +x | −y | −x | +y |

Kit = spec turned −90° about +z (clockwise seen from above). The shader frame is exactly the spec's.

## What MetalSim does with a USD dome

- Importer (`usd_to_mjcf._record_dome_light`): records `yaw` = the prim's world rotation about +z (30° here; the test's
  `rec["yaw"]` check passes), `tilt`, and now DomeLight_1's `poleAxis` (it read `inputs:poleAxis`, which the schema
  does not define, so it never recorded it).
- Tier 2 (`set_environment_from_model`): `env_yaw = rec.yaw + KIT_DOME_AZIMUTH` (−90°), added 2026-09-26 19:46
  (commit 62ebf0d9) after the Kit measurement, so a USD stage renders as Isaac renders it: −60° here.
- Direct `set_environment(yaw=)` / `Randomizer.environment`: the spec frame (no offset).
- Tier 0 / tier 1 have no environment map (documented: `docs/RENDER_COVERAGE.md` "HDR dome map per episode ❌ T0",
  "DomeLight (texture) ◐ Tier 2 only"); they draw MuJoCo's skybox texture, as `mujoco.Renderer` does, and the USD
  importer writes no skybox. MuJoCo itself has no dome light. So there is no second convention to reconcile.

## Verdict

The **test** was stale: it was committed 2026-09-26 01:34 (in 302bab5c) with the spec-frame expectation, before the
Kit measurement and the importer-path offset of 19:46 the same day; the offset commit did not update it. Importer and
renderer agree with each other and with Kit. The test now asserts `env_yaw = 30° + KIT_DOME_AZIMUTH` and, to make the
convention explicit, that the resulting world directions are Kit's measured ones (centre → (−cos 30°, −sin 30°, 0),
u = 0 → (cos 30°, sin 30°, 0), u = 0.25 → (sin 30°, −cos 30°, 0)).

Comment fixes: the shader, `set_environment` and `hdr.env_direction` described the shader frame as "as Kit/RTX
renders a DomeLight"; it is the OpenUSD spec frame, and Kit is −90° from it.

## Open (not changed here)

- A dome whose authored rotation tilts the pole (Kit applies it rigidly) is rendered upright by tier 2 (only yaw is
  passed to the shader); the importer records `tilt`. Needs a full rotation in `PTConsts`.
- Whether Kit honours an explicit DomeLight_1 `poleAxis = "Y"` was not measured (all recordings used the legacy
  DomeLight).

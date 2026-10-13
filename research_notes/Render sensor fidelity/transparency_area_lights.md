# Transparent surfaces in depth cameras and lidar; true area lights

Context: metalsim's ray-traced depth camera (`RayDepthCamera`) and lidar (`Lidar`, kernels `lidar` / `lidar_ext` in
`metalsim/render/shaders/rt.metal`) treated every geom as opaque, including MuJoCo geoms with `rgba` alpha < 1, which
workstream B made transparent in tier 0/1/2. USD Rect / Disk / Cylinder / Sphere lights were imported as spot / point
lights with far-field radiometry, 20-25 % too bright one light size away. Researched 2026-10-13. Labels: **[doc]**
documented by the vendor, **[src]** read from source, **[meas]** measured here, **[inf]** my inference.

## 1. References: what each one does with transparency

### MuJoCo (the parity reference for depth, segmentation and `mj_ray`)
- **[src]** `ray_eliminate` (src/engine/engine_ray.c, main @147dc67, lines ~68-95) skips a geom when: its body is
  `bodyexclude`; `geom_matid < 0 && geom_rgba[3] == 0`; `geom_matid >= 0 && mat_rgba[3] == 0` (with a material, the
  material's alpha decides); `!flg_static && body_weldid == 0`; `geomgroup != NULL && geomgroup[clamp(group,0,5)] == 0`.
  **Alpha exactly 0, no threshold**: alpha 0.3 is hit. https://github.com/google-deepmind/mujoco/blob/main/src/engine/engine_ray.c
  MuJoCo Warp mirrors it (`mujoco_warp/_src/ray.py::_ray_eliminate`, read locally).
- **[src]** Rangefinder: `mj_ray(m, d, origin, site_z, NULL, 1, site_bodyid, ...)` (engine_sensor.c:752): no group
  filter, static geoms included, the site's body excluded.
- **[src]** Renderer (render_gl3.c ~972): a geom is transparent when `rgba[3] < 0.995` (or a plane seen from behind);
  transparent geoms are drawn after the opaque ones.
- **[meas]** `mujoco.Renderer` (MuJoCo 3.3.x python) on a wall at 3 m behind 1 cm panes at 2 m:

  | pane alpha | depth | segmentation | `mj_ray` |
  |---|---|---|---|
  | 0.25 | 1.995 (pane) | pane | pane |
  | 0.99 | 1.995 (pane) | pane | pane |
  | 0    | 2.950 (wall) | wall | wall |

  MuJoCo's three outputs agree: **every surface with alpha > 0 is hit; alpha 0 is invisible**. (The source reading
  found `glDepthMask(GL_FALSE)` in the transparent pass; the measured depth shows the second transparent pass writes
  depth, as workstream B also measured: 1.904 m on its probe.)
- MuJoCo Warp's batch renderer (`render.py`) draws every geom opaque (alpha is only used for splats). **[src]**

### Isaac Sim (the parity reference for RTX sensors)
- **Depth on glass [forum, NVIDIA staff]**: "materials (e.g. glass) are not used as optics to calculate depth";
  a glass cylinder's depth "is calculated on the surface of the cylinder".
  https://forums.developer.nvidia.com/t/how-to-make-an-object-invisible-to-a-camera-in-isaac-sim/273765
- **Cutout opacity [doc]**: OmniPBR / OmniGlass `opacity_threshold`: "If threshold is equal to 0, then use fractional
  opacity. If threshold is greater than 0, then object is opaque when opacity is greater than threshold."
  https://docs-prod.omniverse.nvidia.com/materials-and-rendering/latest/templates/OmniGlass.html ;
  the path tracer's `/rtx/pathtracing/fractionalCutoutOpacity` (default true) treats fractional opacity as surface
  "presence", sampled stochastically. https://docs.omniverse.nvidia.com/materials-and-rendering/latest/rtx-renderer_rt.html
  Workstream B: RTX Real-Time (Isaac Sim <= 5) cuts partially opaque objects out of RGB, depth and segmentation
  (`transparent_ids="behind"`). No documented "translucent objects not written to depth" rule was found.
- **RTX lidar [doc]**: "an RTX-based Lidar can model returns from light interaction with transparent or reflective
  surfaces" (https://docs.isaacsim.omniverse.nvidia.com/6.1.0/sensors/isaacsim_sensors_rtx.html). Materials are
  "non-visual" USD tokens `omni:simready:nonvisual:base / :coating / :attributes`; bases include `clear_glass` (19),
  `frosted_glass` (20), `one_way_mirror` (21), `mirror` (22), `ceramic_glass`, `plexiglass`, `water`, `ice`,
  `calibration_lambertian`; attribute `visually_transparent`
  (https://nvidia.github.io/simready-foundation/2026.06.0/capabilities/nonvisual_sensors/nonvisual_materials/nonvisual_attributes_table.html,
  https://docs.isaacsim.omniverse.nvidia.com/6.1.0/sensors/isaacsim_sensors_rtx_materials.html). Each material is a
  BSDF with "reflection and transmission sub components" (omni.sensors.nv.materials 2.0); no numbers are published.
- **Echo settings [doc]** (https://docs.omniverse.nvidia.com/kit/docs/omni.sensors.nv.lidar/latest/lidar_extension.html):
  `maxReturns`, `reflectionPowerFraction` / `transmissionPowerFraction` ("fraction of the reflection [transmission]
  power for multiple returns"), `minDistBetweenEchosM`; transcoder `returnType` 0 First, 1 Last, 2 Strongest,
  3 MultipleReturns (default). Intensity is "between 0 and 1", scaled by `intensityScalePercent`, mapped
  LINEAR / NONLINEAR; `minReflectance` (%) at `minReflectionRangeM` is the detection threshold. **No intensity
  formula is published.** Known issue: `echoId` is always 0.
- **[forum, 2026]** Glass let the lidar through in Isaac Sim 4.5 but not in 6.0 (points always on the surface);
  NVIDIA: "This seems to be a bug" (ticket 6426827), suggested maxReturns 3 and a glass / `visually_transparent`
  material. https://forums.developer.nvidia.com/t/isaac-sim-6-0-what-are-the-functions-of-the-rtx-sensor-non-visual-material/374657
- Radar: "return behavior depends on material properties" only. [doc]

### Real lidars on glass
- Fresnel at n = 1.5: R = ((n-1)/(n+1))^2 = 0.04 per surface; a pane reflects 2R/(1+R) = 0.077 and transmits ~92 % at
  normal incidence [physics].
- Foster et al. ICRA 2013: glass "behaves like a diffuse object in a minimal range of angles (near-normal
  incidence)". https://web.eecs.umich.edu/~kuipers/papers/Foster-icra-13.pdf
- Cartographer_glass (arXiv 2212.08633): near normal "most of the incident beam is reflected directly back ... a sharp
  increase in intensity", otherwise "most of it passes through".
- HDL-32E reflection study (arXiv 1909.12483): intensity peaks at perpendicular incidence and "drops quickly"; mirror
  ghosts: "The sensor, unaware of the reflection, will report the reflected point to be located behind the glass."
- 3DRef (arXiv 2403.06538; Ouster OS0-128, Livox Avia, Hesai QT64): glass points are 0.9-7.8 % of a cloud, 99.6 % of
  them first returns; "as the incident angle increases, fewer specular returns persist".
- Return modes: Velodyne Strongest / Last / Dual (strongest + last; when the strongest is the last, the second
  strongest; VLP-16 manual https://usermanual.wiki/Pdf/VLP16Manual.1719942037/html, arXiv 1909.12483); Ouster dual =
  strongest + second strongest (https://www.mathworks.com/help/lidar/ousterlidar/ref/ousterlidar.html); Hesai Last /
  Strongest / First, dual Last+Strongest (default), Last+First, First+Strongest (reseller listings); Livox
  first-or-strongest, dual, triple.

## 2. Decisions (implemented)

Transparency is the material-resolved `rgba` alpha the renderers use (`SceneTables.transparent`: alpha < 0.995, MuJoCo's
test; height fields and sized terrain slots stay opaque; sites / flexes / skins / tendons are opaque).

**Depth camera** (`RayTracer.make_depth_camera(..., transparent_ids=..., alpha_threshold=...)`, names as workstream B's
annotators):
- `"surface"` (default; alias `"mujoco"`): the first surface whose alpha > 0. Equal to MuJoCo's renderer depth and
  `mj_ray` (alpha 0 invisible) and to Isaac's depth on glass. Same default as tier 0/1/2.
- `"behind"`: transparent surfaces are skipped: RTX Real-Time opacity cutout (B's mode).
- `alpha_threshold=t`: Isaac's `opacity_threshold` rule: a transparent surface counts only when alpha > t
  (`"surface"` is t = 0, `"behind"` t = 1).
- `"opaque"`: the previous behaviour (every drawn surface stops the ray; geoms of alpha 0 are left out of the scene
  tables, as MuJoCo's renderer does, so they are invisible in every mode).

**Lidar** (`transparency=` on `Lidar` / `make_lidar`):
- `"transmit"` (default): a transparent surface of alpha a and reflectance rho returns `P a rho cos(theta) / r^2`
  (alpha as surface presence / coverage, as tier 2 and Isaac's fractional cutout read it) and the beam continues with
  `P (1 - a)`. An object counts once (its exit faces are skipped, as the existing multi-echo model does), at most 8
  transparent objects per sub-ray. Echoes are grouped and chosen as before (`minDistBetweenEchosM`) with a
  `return_mode`. Physical for real lidars (most power passes glass) and the intent of Isaac's
  `transmissionPowerFraction`; Isaac 6.0's opaque glass is a reported bug.
- `lidar_materials={geom name / regex: preset or dict}`: a per-geom lidar BSDF overriding alpha, named after the
  SimReady non-visual bases: `clear_glass`, `frosted_glass`, `mirror`, `one_way_mirror`, `plexiglass`, `water`,
  `calibration_lambertian`. Fields: `reflectance` (Lambertian rho), `coverage` (presence of the diffuse part),
  `specular` (reflectance at normal incidence, Schlick in angle), `roughness` (GGX, alpha = roughness^2),
  `transmission` (of the uncovered part), `ghost` (follow the mirror reflection; the echo is reported along the beam at
  the reflected path length, as real lidars do). Specular backscatter: `pi F D G / (4 cos^2(theta)) cos(theta) / r^2`,
  the GGX BRDF at the retro direction in the units of the Lambertian `rho cos / r^2` (= pi f cos / r^2).
- `"surface"`: MuJoCo's rangefinder / `mj_ray` (alpha > 0 stops the beam). `"behind"`, `"opaque"` as for depth.
- `return_mode`: `"strongest"` (default, the existing behaviour; Isaac `returnType` 2, or 3 with `maxReturns` > 1),
  `"first"` (0), `"last"` (1), and for `max_returns=2` the dual modes `"strongest_last"` (Velodyne dual, Hesai default
  dual), `"first_last"`, `"first_strongest"`. `min_reflectance` / `min_reflectance_range` (Isaac `minReflectance` %,
  `minReflectionRangeM`) drop echoes weaker than a Lambertian target of that reflectance at that range.
- Intensity keeps the existing model (Lambertian `rho cos / r^2`; the radar's RCS proxy uses the same reflectance
  table). Isaac's 0-1 normalisation is undocumented and not reproduced.

Scenes without transparent geoms or lidar materials take the previous code paths: outputs are byte-identical (tested).

## 3. Area lights

### Units [doc]
- UsdLux LightAPI (https://openusd.org/release/api/class_usd_lux_light_a_p_i.html): `intensity` is luminance L in nits;
  `exposure`: L x 2^exposure; `normalize` (default false): L / (world-space surface area of the light's shape);
  `color` multiplies L. RectLight emits from one side along -Z, `width` / `height` default 1; DiskLight `radius` 0.5,
  along -Z; CylinderLight axis along X, `radius` 0.5, `length` 1, emits from its side (no caps); SphereLight
  `radius` 0.5, `treatAsPoint` a hint that renderers supporting area lights may ignore.
- RTX: lighting in nits (`omni:rtx:usdluxVersion` 2411 default; 2505 follows UsdLux 25.05, Kit 109+).
  https://docs-prod.omniverse.nvidia.com/materials-and-rendering/latest/lighting.html
- metalsim: a USD radiance L is a radiance in the renderer's units (a MuJoCo light of diffuse D gives E = pi D at
  normal incidence; the importer's DistantLight convention); the importer records shape, size and radiance per light
  (`usd_area_lights` custom text).

### Analytic irradiance used for validation
- Coaxial disk, radius R, Lambertian radiance L, distance d (receiver parallel): E = pi L R^2 / (R^2 + d^2)
  (view factor, e.g. https://uweb.engr.arizona.edu/~dial/ece425/notes7.pdf).
- Polygon (Lambert's formula; Arvo 1994 https://www.graphics.cornell.edu/pubs/1994/Arv94.pdf):
  E = (L/2) sum_i Theta_i (Gamma_i . n), Theta_i the angle edge i subtends at the receiver, Gamma_i the unit normal of
  the plane through the receiver and edge i. Used directly (any receiver position / orientation) in the tests.
- Sphere (radius R, distance d to the centre, receiver facing it): E = pi L (R/d)^2.

## 4. Results (2026-10-13, M4 Max, Warp 95f6f121, on main f3b5c62e)

Full tables in the repo: `docs/RENDER_COVERAGE.md` §3e, `docs/PARITY.md`.
- Depth "surface" equals `mj_ray` through every pixel to < 0.2 mm; "behind" / `alpha_threshold` as specified.
- Lidar transmit: pane / wall ranges to 0.2 mm, intensities a rho cos / r^2 and (1 - a) rho cos / r^2 to 1e-4; "surface"
  equals `mj_ray`; return modes, clear glass (no echo 4-14 deg off normal, 48x the wall's echo at normal), mirror
  ghost at the path length.
- Scenes without transparency: 36 / 36 sensor arrays byte-identical to main.
- Cost, 16 x 360 lidar at 4096 envs: plain 6.9 ms; glass scene (33 % of beams through panes) opaque 9.4, surface 9.8,
  behind 16.3, transmit 23.3 ms; 64 x 64 depth behind 9.2 vs 3.2 ms.
- Area lights, tier 2: floor irradiance mean within 0.1 %, worst 4 x 4 block 1.3 % at 0.5 / 1 / 3 light sizes (rect,
  disk, sphere, cylinder); far-field import was +19 % on axis at one size. Tier 1: median 0.3-0.5 % (8-bit), soft
  shadow within 0.9 % mean of tier 2.

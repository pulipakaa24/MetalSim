# ground-scale: film ground textures at their real size

## Status
DONE. The suspicion was right, and the error applies to every geom type the films texture as ground. With
`texuniform`, MuJoCo, and the path tracer that copies its texture coordinates, lay one texture image per
**2 / texrepeat** metres, not per 1 / texrepeat m as MuJoCo's XML reference says.

Fixed in one place:
- `LookCfg.ground_tile_m` (metres of floor per image) replaces `ground_repeat`;
- `render.ground_texrepeat` sets texrepeat = 2 / tile.

Every film now takes its tile from Poly Haven's `dimensions`. TRON1's pixels and cache keys are unchanged. Nothing
was re-rendered.

## Changed
- `metalsim/render/film/cfg.py`: `LookCfg.ground_repeat` is replaced by `ground_tile_m` (float or None, default None;
  required when a ground texture is set). Docstring updated.
- `metalsim/render/film/render.py`, ground part only:
  - a comment block citing MuJoCo's source;
  - `GROUND_TEXUNIFORM_PERIOD_M = 2.0`;
  - `ground_texrepeat()`;
  - `add_ground_material()`: the code `build_scene` had inline. It now refuses a ground geom that is not a plane, box,
    hfield or mesh.
  - `_key_look()`: `cache_key` hashes the resolved texrepeat under the old name `ground_repeat` (1.0 when there is no
    texture), so a take whose floor did not change keeps its key. This follows the `after_end` precedent.
  - `render.json` `look` gains `ground_tile_m` and `ground_texrepeat`.
- `metalsim/render/film/tasks.py`:
  - new constants `LAMINATE_TILE_M = 1.70` and `SANDY_GRAVEL_TILE_M = 2.53` (lines 31-34);
  - the looks of Franka-Pour, G1 flat, G1 rough, G1 Unitree and G1 bumps use them;
  - doc text: Franka "Rendering" (l. 89), `G1_RENDER_NOTES` (l. 182-183), G1 Unitree "Rendering" (l. 499-500), G1
    bumps "Look" (l. 687-698, rewrapped).
- `metalsim/render/film/tron1.py`: `ground_tile_m=3.0` (l. 454). The "Rendering" paragraph (l. 399-412) is rewritten
  and rewrapped.
- `metalsim/render/film/so101.py`: only the look line, `ground_repeat=1.0` -> `ground_tile_m=1.5` (l. 1313).
- `tests/test_film_ground_scale.py`: new, CPU only, 10 tests.
- Scratch, all in `runs/readme_media/scratch/ground-scale/`:
  - scripts: `still.py`, `measure.py`, `key_check.py`, `twin_check.py`;
  - `polyhaven_info/*.json`; `mujoco_src/` (MuJoCo 3.14.0 `render_context.c`, `render_gl3.c`, `XMLreference.rst`);
  - `still/`; `look/`; `measure.json`, `key_check.json`, `twin_check*.json`.

## Per film

| film (README cell) | texture: Poly Haven `dimensions` | ground geoms | old tile | new tile | factor | visible change, re-render? |
|---|---|---|---|---|---|---|
| g1_unitree (in README) | laminate_floor_02, 1700 mm | `floor` infinite plane | 4.00 m | 1.70 m | 2.35x | **yes**: planks 50 -> 21 cm wide |
| g1_rough (in README) | laminate_floor_02, 1700 mm | `ground` hfield + `terrain_boxes` mesh | 4.00 m | 1.70 m | 2.35x | **yes** |
| g1_flat | laminate_floor_02, 1700 mm | `ground` infinite plane | 4.00 m | 1.70 m | 2.35x | **yes**, if it is shown |
| franka_pour (in README) | laminate_floor_02, 1700 mm | `ground` infinite plane | 2.00 m | 1.70 m | 1.18x | slight (planks 25 -> 21 cm); re-render for true scale, low priority |
| g1_bumps | sandy_gravel_02, 2530 mm | `floor` hfield + `backdrop` infinite plane | 5.06 m | 2.53 m | 2.00x | **yes**: pebbles half size, horizon bands twice as frequent |
| so101_lift (and its _74v, _smooth, _deploy variants) | wood_table_001, 1500 mm | `table_top` box (0.9 x 0.7 m) | 2.00 m | 1.50 m | 1.33x | slight (grain 25 % finer); re-render for true scale |
| tron1 | concrete_floor_worn_001, 3000 mm | `floor` infinite plane | 3.00 m | 3.00 m | 1 | **no**: same pixels, same cache keys |

Notes on the table:
- Franka's geom type comes from the 2026-10-14 take's twin, `~/metalsim-fresh/film-20261014-0004/out/rec/twin.xml`:
  `ground` is `size="0 0 0.05"`, an infinite plane. The repository has no Franka twin (`runs/film/franka_pour/rec`
  holds only `rec.json`).
- Looked at: `scratch/ground-scale/look/current_readme_cells.jpg` (current g1_unitree, g1_bumps and so101 README
  frames). The G1 Unitree laminate planks are wider than the robot's foot is long. The G1 bumps sand is visibly
  coarse. The so101 dark wenge grain change will be subtle at 320x240.

## Doc lines for the README stream (do not hand-edit generated docs; `--stage doc` rewrites them from the notes)
Lines that state tile sizes or `ground_repeat`:
- `docs/renders/TRON1.md:118-121`: "`ground_repeat` 2/3 lays one tile per 3 m".
  - The pixels are unchanged, so `--stage doc` alone updates it.
- `docs/renders/G1_BUMPS.md:160-161`: "`ground_repeat` = 1 / 2.53 per m".
  - The take actually shows 5.06 m tiles, so the current page is wrong about its own frames.
  - Re-render, then run `--stage doc`.
  - Re-check by eye the notes' visual claims at :164-168 ("texture adds its own pebbles", "faint bands towards the
    horizon"): the bands now repeat every 2.53 m.
- New "real size" text appears after a doc stage at:
  - `docs/renders/G1_FLAT.md:123`;
  - `G1_ROUGH.md:142`;
  - `G1_UNITREE.md:135`;
  - `FRANKA_POUR.md:85`.
- `docs/renders/SO101_LIFT.md:183` names `wood_table_001` without a size.
  - It is generated from `so101.py:764`, which the so101-film stream owns. It could say "at its real size, 1.5 m per
    image".

## Verified
**MuJoCo's semantics.** Sources: MuJoCo 3.14.0, the installed version, fetched from raw.githubusercontent at tag
`3.14.0`, copies in `scratch/ground-scale/mujoco_src/`.
- `render_context.c` `makePlane`:
  - lines 211-212 and 219-220: a finite axis gives u = (x + s) / 2s;
  - lines 214 and 222: an infinite axis gives u = 0.5 x;
  - line 185: the infinite plane's grid step is `sX = 2/m->mat_texrepeat`, i.e. one image.
- `render_context.c` box list, 1059-1078: each face spans 0..1 over the unit cube, then is scaled by the size.
- `render_gl3.c` `settexture`:
  - 129-148: scl = texrepeat, and with texuniform x size when size > 0;
  - 172-199: the object-linear mapping, `0.5*scl` with scl = texrepeat / size x size.
- So every case is one image per 2 / texrepeat m.
- The XML reference (`doc/XMLreference.rst:1965-1972`) says "repeated N times over one spatial unit". That is off by
  2x for these geom types.
- MuJoCo Warp agrees with the code: `upstream/mujoco_warp/mujoco_warp/_src/render.py:175-201` (`_texture_plane`) and
  `:294-328` (`_texture_box`).
- Our mirror is `metalsim/render/builtin_geoms.py:189-197` and `:254-280`. The shader samples
  `fract(uv * par.xy + offset)` (`shaders/appearance.h:55`). The analytic plane is `env_grid.h:55-75` with
  `pl_uv` from `scene_tables.plane_uv_affine`.
- Web search found no upstream issue on the doc wording; none was filed.

**Texture sizes.** From `https://api.polyhaven.com/info/<id>`, field `dimensions` in mm, read 2026-10-31 and saved
in `scratch/ground-scale/polyhaven_info/`:

| texture | dimensions (mm) |
|---|---|
| laminate_floor_02 | 1700 x 1700 |
| sandy_gravel_02 | 2530 x 2530 |
| concrete_floor_worn_001 | 3000 x 3000 |
| wood_table_001 | 1500 x 1500 |
| coast_sand_01 (see Open) | 15000 x 15000 |

The laminate image's seam rows, measured in the 1k PNG, fall every 128 px: 8 planks of 0.2125 m per image.

**Unit test (CPU).** `.venv/bin/python -m pytest -q tests/test_film_ground_scale.py`: **10 passed**.
- For every geom type it takes the texture coordinate the path tracer samples, from `build_scene_tables` (stored uv x
  material scale + offset), as an affine function of world x and y.
- Its period equals `ground_tile_m` to 2e-4, for tiles 1.70, 2.53 and 3.0, with `per_world_size` True and False.
- Geom types covered:
  - infinite plane, both the in-cell disc and the analytic continuation (`tables.plane`);
  - finite plane rotated 30 deg;
  - box top;
  - heightfield with skirts;
  - `terrain_boxes`-style mesh, built with `adapters.terrain_box_faces`.
- Also tested: `build_scene` puts texrepeat 2 / tile on the named geoms; a missing tile or a sphere raises; every
  registered film with a texture has a tile; TRON1's key look equals its pre-fix form.
- Mutation check: with `GROUND_TEXUNIFORM_PERIOD_M = 1.0` (the doc's reading), the geometric check measures 3.4 m for
  a 1.70 m tile on all five geom types.

**Real twins (CPU).** `twin_check.py` builds the render-stage scene and the tracer tables from each recorded
`runs/film/<task>/rec/twin.xml`. Output: `twin_check.json` and `twin_check_g1_rough.json`.

| take | geoms | old period | new period |
|---|---|---|---|
| g1_unitree | `floor` | 4.0 m | 1.7 m |
| g1_flat | `ground` | 4.0 m | 1.7 m |
| g1_rough | `ground` hfield and `terrain_boxes` | 4.0 m | 1.7 m |
| g1_bumps | `floor` hfield and `backdrop` | 5.06 m | 2.53 m |
| so101_lift | `table_top` | 2.0 m | 1.5 m |
| tron1 | `floor` | 3.0 m | 3.0 m |

**TRON1 unchanged.** `key_check.py` (output `key_check.json`):
- the new code reproduces the stored `cache_key` of both shots: hero `e380feb638b0e45fd9f2`, readme
  `f43293eac6050257a3fe`;
- texrepeat is bit-identical: `2.0 / 3.0 == ground_texrepeat(3.0)` gives 0.6666666666666666.
- So `--stage render` would reuse its frames.

**Other takes' keys.** For every other take with a recording, the pre-fix key reproduces the stored key and the new key
differs, so they re-render: g1_bumps, g1_flat, g1_rough, g1_unitree, so101_lift, so101_lift_74v and so101_lift_smooth,
both shots each.

**GPU still.**
- Command: `scripts/gpu_run.sh ground_scale render 10 -- .venv/bin/python runs/readme_media/scratch/ground-scale/still.py`.
- It waited ~58 min behind the rough-terrain training job, ran 20:36:44-49 and exited 0. I did not touch the training
  job.
- Power, read live at 20:38: AC, 81 %, charging, `powermode 2`. Mac16,6; no timing is quoted.
- The scene, built by `build_scene` -> `add_ground_material`:
  - a top-down pinhole camera, 2048x2048, fovy 40, z = 21.98 m: 128 px/m nominal;
  - an infinite plane, a finite plane (upper left), a box (upper right) and a flat heightfield (lower right);
  - 4 black 5 cm markers at +-4.5 m;
  - tier 2, 8 x 4 spp, no denoiser;
  - the same model also drawn by `mujoco.Renderer` (OpenGL).
- Two cases: new (1.70 m) and old (texrepeat 0.5). The env cell is +-7.11 m, so |x| or |y| > 7.11 m is drawn by
  `grid_plane`'s analytic continuation.
- `measure.py` (output `measure.json`):
  - pixel scale from the markers: 127.99 px/m (tier 2), 128.03 (GL);
  - plank pitch: the high-passed row profile's autocorrelation;
  - image period: the exact-repeat autocorrelation peak.
  - The method was self-tested on the texture resampled at 1.70 m and 4.0 m per image: it recovers 1.703 and 4.000 m
    (the 1.703 is the resample rounding to 218 px). Plank pitch reads 0.5 % low: 0.2117 against 0.2129.

  | case | plank pitch | image period | regions |
  |---|---|---|---|
  | **new, tier 2** | 0.2116-0.2117 m (27.1 px) | **1.7006-1.7013 m** (217.7 px) | finite plane, box top, heightfield, infinite plane in the cell, infinite plane past the cell |
  | new, MuJoCo GL | — | 1.7000-1.7006 m | same |
  | **old, tier 2** | 0.500 m (64 px) | **4.000 m** (512 px) | |
  | old, MuJoCo GL | 0.4997-0.5000 m | 3.999-4.000 m | |

  - The quadrants hold fewer than two old images, so the old image period comes from the 15.8 m bands.
  - One measured cell is spurious: the GL plank pitch of 0.08 m in the 0.75 m-tall "past the cell, bottom" strip.
    That strip is too short across the planks.
- Frames I looked at, all in `scratch/ground-scale/look/`:
  - `still_new_old_t2.jpg` and `still_new_old_gl.jpg`: the whole stills. Markers are visible; there is a key-light
    specular gradient towards the lower right.
  - `still_center_crop_native.png`: native crop at the quadrant gaps. New planks ~27 px, old ~64 px. The quadrant
    seams show as phase jumps.
  - `still_cell_edge_corner.png`: the corner past the env cell. No seam where the disc ends and the analytic plane
    begins.
  - `current_readme_cells.jpg` and `wood_sand_textures.jpg`.

**Tests run.**
- CPU: `.venv/bin/python -m pytest -q tests/test_film_ground_scale.py tests/test_film.py tests/test_film_render_cache.py tests/test_no_machine_paths.py`
  gave **18 passed, 1 skipped**. The skip is `test_newton_twin`: no newton in the main venv.
- Earlier, `tests/test_compose.py tests/test_env_grid_planes.py` (with the above): 32 passed, 5 skipped (no newton; no
  Menagerie for 4 compose tests).
- `tests/test_tron1.py`: 16 passed.
- **Protocol slip:** `tests/test_env_grid_planes.py` has `needs_metal` tests, and they ran on the GPU outside the queue
  for a few seconds, during the training job.
- I did not run the GPU texture-parity tests (`tests/test_render_appearance.py::test_texture_mapping_matches_mujoco`,
  `tests/test_render_tier0.py::test_texture_pattern_parity`). `builtin_geoms.py` and `scene_tables.py` are unchanged;
  only the material's texrepeat value changes.

## Open
- **ANYmal sand (README cell `anymal_sand.webp`), not in my files.**
  - The brief's "ANYmal sand film, 2.53 m" is in fact the G1 bumps film.
  - The ANYmal demo, `scripts/particle_demo/render_anymal.py:123`, puts `coast_sand_01` on finite planes with
    texrepeat 0.8 and texuniform. That is one image per 2.5 m against Poly Haven's 15 x 15 m, so its ground texture is
    6x too small.
  - Decoration grains cover much of it. Whether it shows is not checked.
  - Not fixed: the file is not mine, and the clip comes from frames outside the repo (`~/metalsim-fresh/runs_render/sand25m`).
- `parity_g1_metalsim.webp` comes from `metalsim.parity.side_by_side`, not the film tool. Its floor was not checked.
- Box side faces keep MuJoCo's quirk: the ±X faces scale u by size[0] while u spans y. Only the SO-101 table edge
  (4 cm) shows a side face; MuJoCo draws it the same way.
- Concurrency:
  - The queued `so101-film-deploy` job (`runs/readme_media/scratch/so101-film/film_job.sh`) runs after my change, so
    `so101_lift_deploy` renders with the 1.5 m tile, at true scale.
  - Any so101-film code still passing `ground_repeat=` now raises a TypeError.
- G1 bumps' "faint bands towards the horizon" comment is a visual claim. Re-check it after the re-render.

## Next
For the README stream:
1. Re-render, kind render via the queue, each from its existing recording: `--stage render`, then
   `--stage encode --shots hero`, the README WebP command in each doc, then `--stage doc`.
   - The recordings stay valid: only the look changed. No `--stage record` is needed.
   - Films: g1_unitree, g1_rough, g1_bumps (largest visible change); then franka_pour and so101_lift (subtle).
   - franka_pour has no `rec.npz` in the repo: find its recording in `~/metalsim-fresh` (`film-20261014-0004/out/rec`)
     or re-record in `.venv-newton152`.
   - g1_flat only if it is still shown.
2. TRON1: run only `python -m metalsim.render.film tron1 --stage doc`. Its pixels and keys are unchanged.
3. Look at a native-resolution hero crop of the G1 Unitree floor. Expected: planks about 21 cm wide, about the length
   of the G1's foot. Today they are about 50 cm, wider than the foot is long.
4. Optionally, for whoever owns `scripts/particle_demo`: decide whether the ANYmal sand (6x too small) matters.

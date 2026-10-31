# Hand-off: film-render (bugs 2 and 3 of plan section 2), 2026-10-31

## Status
Done for both bugs, with an extra defect found and fixed on the way (neighbour discs z-fighting in multi-env renders),
plus the real cause of plan defect 5 (the grey box) fixed in render.py. Nothing committed. The patch is
`runs/readme_media/handoff/film-render.patch`: `git diff` of worktree `.claude/worktrees/agent-a992d31c678aa6957`
(base 9caff7aa) plus the two new test files. Apply it from the repo root with `git apply`. It reverse-applies cleanly to the worktree.

## Cause (bug 2, plane wedge): two defects in the compact env grid's plane disc, plus a camera outside the design's assumption
Sources: commit 6502f562 (compact grid: "doubles the cell and rewrites every buffer holding the grid (consts, lidar
copies, tier 1 RTOffset, sbvh arguments, per-env instance offsets)"); merge e485e2a6 ("C's infinite-plane disc covers
the env's cell, and past the cell infinite planes are continued analytically"); docs/RENDER_COVERAGE.md:665-676 (env
grid; cell = content box incl. cameras *at the initial state*) and :353-356 (legacy grid "grows the env stride ~4 % to
keep neighbouring discs apart"); metalsim/sensors/env_grid.py docstring; shaders/env_grid.h `grid_tmax` / `grid_plane`;
raytrace.py `__init__` (disc apothem computed once) and `_check_grid`.
1. **Disc not regrown on grid growth (the wedge).** The disc apothem is fixed at construction for the initial cell. When
   the cell doubles, `grid_plane` still continues the plane only past the *new* cell's exit, so floor between the old
   disc's edge and the new boundary is drawn by nothing. The growth landed in 6502f562 and the cell-covering disc in the
   later merge, which never added the disc to the growth path's buffer list. Hypothesis test: I predicted the gap
   analytically (initial octagon apothem 2.738 m, grid grown once) and drew its outline on before frame 100. The outline
   traces the wedge's edges exactly: `scratch/film-render/predict_gap_f100.png`. The G1 film grows the grid at frame 0
   because the twin's touch sites are drawn at the qpos0 pose (feet sites at z = -0.74, inside the 0.5 m slack of the
   cell floor at -1). It grows again when the robot walks away.
2. **Camera outside its cell.** The design assumes cameras are in the content box. The film's tracking camera
   (`render_host(cam_pos=...)`) sits at 3.5 m, outside the 2.4 m cell. Such a ray is traced up to the cell's far side,
   through neighbours' cells. Unfixed measurement (4 envs, eye 6 m out): 434 / 1131 / 0 / 486 floor pixels missing per
   env, and env 2 has 75 pixels of other content (`scratchpad/cam_outside_old.py`, run against the main checkout).
3. **Neighbouring discs overlap (found after fixing 1).** The disc reaches about 41 % of the cell's half-width plus 1 m
   into each neighbour's cell. It is coplanar with the neighbour's floor and its texture is offset by the stride, so the
   two z-fight. G1 film frame 300, 4 envs: 4,467 / 22,349 / 0 / 42,806 albedo pixels differ from the 1-env render
   (`scratch/film-render/albedo_envs_f300.png`). Before, this was masked because the grid grew past the disc. Plumbing
   `env_margin` alone (the brief's minimum) would have exposed it.

## Changed
- `metalsim/sensors/raytrace.py`: `_fit_plane_discs()` fits the disc to the cell at build time and on every growth and
  rebuilds its primitive structure in place. The disc covers the cell. For planes whose axes lie along the cell's
  (every floor), it is clipped to the cell's rectangle plus 0.25 mm. Its apothem is at least (1+√2)·half-extent, so all
  outer vertices clamp onto corners and no corner is cut; the first try without this left 4-13 crack pixels.
  `contain(points)` grows the cell until the points are inside it. `_grow_grid()` is factored out of `_check_grid`.
- `metalsim/render/tier2.py`: new `env_margin` argument passed to RayTracer. `render_host` calls `rt.contain(render camera)`
  before tracing.
- `metalsim/render/film/render.py`:
  - `film_env_margin()` sizes the cell up front from the recorded geoms (± rbound) and every camera position, so the
    film has no growth and no warning.
  - Sites are hidden (`visual.sitegroup = 0`). They are never recorded and sat at qpos0 all film: the grey box at the
    origin is the `torso_link_touch` site, not the `Cube` geoms. This is defect 5's real cause; the record/adapters
    stream should not hide `Cube` geoms, which are robot meshes.
  - Bug 3: `cache_key()` is a sha256 of rec.npz + rec.json + twin.xml + tier/passes/spp/bounces/denoise, the full look,
    the viewer (resolution included), grid and slots. It is written to render.json with an `env_grid` record. With
    another key, `_invalidate()` moves frames/, render.json and render_frames.jsonl to `<out>/stale/` (replacing an
    older stale/), and exposure, probe and hardware start fresh. With the same key, the render resumes.
  - Frames and render.json are written atomically (temp file + rename). Stats rows of re-rendered frames are replaced,
    not appended.
- `tests/test_env_grid_planes.py` (new, GPU): disc grows with the grid; camera outside its cell (4 envs) sees the whole
  floor; neighbouring discs don't overlap (checker plane, albedo per env == 1-env).
- `tests/test_film_render_cache.py` (new, CPU, fake renderer): re-record re-renders every frame; interrupted render
  resumes; --no-skip-existing replaces stats rows.

## Verified
All GPU work went through `scripts/gpu_run.sh` from the worktree, which imported the worktree's metalsim (checked). Scratch is `runs/readme_media/scratch/film-render/`.
- Before (unfixed code), `before.log`: two warnings per run ("geometry neared its env's cell boundary", at frame 0 and
  about frame 200). Frames `before_1env/frames/frame_{0000..0300 step 50}.png`, `before_4env/...`, sheets
  `before_1env_sheet.jpg`, `before_4env_sheet.jpg`: wedge and grey box in every frame.
- After, `final.log`: no env-grid warning at 1 env or with 4 envs. The cell is 17.9 x 18.0 x 18.7 m (margin 8.74 m) and
  render.json has `env_grid.growths = 0`. Frames `after_1env/`, `after_4env/` (frames 0..300 step 50), sheets
  `after_1env_sheet.jpg`, `after_4env_sheet.jpg`. I looked at the sheets and at full-size after_4env frame 300: the floor
  is complete to the horizon, there is no grey box, and no seams.
- Growth path alone (up-front margin disabled, `scratchpad/render_nomargin.py`): `after_1env_growth_only/`. Two
  "render camera is outside its env's cell" warnings, then the floor is complete.
- G1 albedo, 4 envs vs 1 env at frame 300 (`albedo_envs_f300_after.png`): 0 differing pixels in every env (max 0.0016).
- Tests (`tests.log`): `tests/test_env_grid_planes.py`, test_render_extras `test_env_grid_layouts` /
  `test_envs_render_alike` / `test_env_grid_grows_when_geometry_leaves_its_cell`, `tests/test_render_world.py` (infinite
  ground plane, lidar beyond the old square), `tests/test_render_haze.py`, `tests/test_film.py`,
  `tests/test_film_render_cache.py`: **36 passed, 1 skipped**.
- Against the unfixed code (main checkout via PYTHONPATH): the disc-growth test loses 2,074 of 6,144 floor pixels, the
  camera-outside test fails, and all 3 cache tests fail.
- Cache on GPU (`final.log`, `cache_demo/`): 0:3 renders 3 frames; 0:6 prints "resuming: 3 of 6" and renders 3. After an
  in-place re-record of rec_demo/rec.npz, 0:6 prints "cache key 4376a99… differs (1354559…): 6 old frames … moved to
  stale" and renders 6, with a new automatic exposure (0.7778 vs 0.7782). The jsonl has 6 rows; stale/ holds the old 6.

## Open
- The camera containment check is host-side only (`render_host`). `Tier2Renderer.render(sim)` and sensors whose camera
  leaves the content box mid-run still rely on geometry-triggered growth.
- Non-axis-aligned infinite planes (tilted or yawed) keep the unclipped covering octagon, so neighbour overlap can remain.
  Walls: the disc apothem uses the cell's x/y extent only (pre-existing).
- On resume the hardware stamp stays that of the run that started the shot (as before). The cache key doesn't cover
  renderer code changes or asset file contents (HDRI/texture are keyed by path).
- No timing numbers were taken (Low Power Mode). The full film was not re-rendered.
- `runs/film/g1_flat/hero` was not touched. Its next render will move its 301 old frames to `stale/`, since it has no key.

## Next
Apply the patch in the main checkout and run the same test list:
`scripts/gpu_run.sh t render 10 -- .venv/bin/python -m pytest -q tests/test_env_grid_planes.py tests/test_render_extras.py::test_env_grid_layouts tests/test_render_extras.py::test_envs_render_alike tests/test_render_extras.py::test_env_grid_grows_when_geometry_leaves_its_cell tests/test_render_world.py tests/test_render_haze.py tests/test_film.py tests/test_film_render_cache.py`
Then re-render the G1 hero with the fixed checkpoint (plan defect 1) via
`python -m metalsim.render.film g1_flat --stage render`. Check that render.json shows `env_grid.growths: 0` and a
`cache_key`, and look at a contact strip. Optionally add a GPU-side camera-in-cell flag for `render(sim)`.

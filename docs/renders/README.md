# Renders

Tasks filmed with metalsim's own renderer, each with a doc that says what is shown, how long every stage took on
which machine, every asset with its source and licence, and the commands to reproduce it.

| film | what | doc |
|---|---|---|
| Franka-Pour | trained policy pours granular MPM media between two cups (Isaac Lab 3.0-EA task); one env and a 2 x 2 grid | [FRANKA_POUR.md](FRANKA_POUR.md) |
| G1 flat walk | Isaac Lab's G1 flat velocity task with NVIDIA's published Isaac Lab 2.3 policy (rsl_rl, iteration 1499, trained in PhysX; fetched by `scripts/fetch_isaac_assets.sh`); 6 s, 1280x720 hero and a 640x480 orbiting shot for the README, 64 spp | [G1_FLAT.md](G1_FLAT.md) |
| G1 rough walk | Isaac Lab's G1 rough velocity task (stairs, slopes, boxes) with NVIDIA's published policy (iteration 2999); the G1 climbs out of a pit of inverted stairs; same two shots | [G1_ROUGH.md](G1_ROUGH.md) |

## The README's media

The animations at the top of the repository's `README.md` are in this directory, linked by relative path, as MuJoCo
Warp does. Every cell is a lossy animated WebP, 320x240, looping, written by Pillow at quality 70 in one call
(MuJoCo Warp's `mujoco_warp/record.py` recipe), through `python -m metalsim.render.film.encode webp`: a 4:3 crop of
the source, Lanczos downscale, one `.json` beside the output with the crop, window, frame count, PSNR against the
source and the exact command. Those `.json` files, the frame directories and the scratch work are in
`runs/readme_media/`, which stays local (gitignored); the commands are copied below. On 2026-10-31 every encode
command below, and the chart's, was re-run from its source and gave a byte-identical file; the commands that record
or render were not re-run (`runs/readme_media/handoff/provenance-docs.md`). Paths are from the repository root.

No timing figure comes from these media. The G1 films were recorded and rendered on the 14-inch MacBook Pro
(Mac16,6), not in timing runs (MetalSim's timing runs are made on the 16-inch through `scripts/gpu_run.sh <name> timing
...`); their docs give the power state each stage stamped. The lidar clip was made on the 14-inch on battery (the
`pmset` log shows battery power at 02:14 on 2026-10-31, during its render); its power-mode setting was not recorded. The chart's numbers are measurements from the logs it cites.

| README row | cells (left to right) |
|---|---|
| 1 | [G1 flat](#g1-flat-g1_flatwebp), [G1 rough](#g1-rough-g1_roughwebp), [ANYmal on sand](#anymal-c-on-sand-anymal_sandwebp) |
| 2 | [Franka-Pour](#franka-pour-franka_pourwebp), [Isaac Sim and MetalSim, same policy](#same-policy-in-isaac-sim-and-metalsim-parity_g1_isaacwebp-parity_g1_metalsimwebp) (two cells) |
| 3 | [lidar through glass](#lidar-through-glass-lidar_glass_rgbwebp-lidar_glass_depthwebp-lidar_glass_lidarwebp) (three cells) |
| below | [throughput chart](#throughput-chart-throughput_darksvg-throughput_lightsvg) |

### G1 flat (`g1_flat.webp`)

- **Shows:** Unitree G1 walking on flat ground: Isaac Lab's G1 flat velocity task on MuJoCo Warp, driven by NVIDIA's
  published Isaac Lab 2.3 policy for the task (rsl_rl, iteration 1499, trained in PhysX, played without retraining),
  path traced at tier 2; the camera tracks the pelvis and orbits at 10 deg/s, MuJoCo Warp's orbit rate. 0.5-5.5 s of a
  6 s take, 150 frames at 33 ms, 0.74 MB. The quick, short steps (3.8 gait cycles per second in the filmed env, 7 cm
  swing apex) are Isaac's own gait; MuJoCo Warp adds 3.4-5.9 % to the cadence of Isaac-trained G1 checkpoints
  ([G1_FLAT.md](G1_FLAT.md), "Gait").
- **Source:** the `readme` shot of the take in `runs/film/g1_flat/` (`readme/frames`, local; `rec/rec.json` and
  `readme/render.json` tracked).
- **Reproduce:** the checkpoint first, `scripts/fetch_isaac_assets.sh` (NVIDIA's file, SHA-256 checked, not in the
  repository). The take, as made on 2026-10-31 (GPU-queue jobs of kind render, one at a time, then CPU stages):
  ```
  scripts/gpu_run.sh g1-switch-rec-flat render 15 -- .venv/bin/python -m metalsim.render.film g1_flat --stage record
  scripts/gpu_run.sh g1-switch-render-flat render 20 -- .venv/bin/python -m metalsim.render.film g1_flat --stage render
  .venv/bin/python -m metalsim.render.film g1_flat --stage encode --shots hero
  python -m metalsim.render.film.encode webp runs/film/g1_flat/readme/frames docs/renders/g1_flat.webp --start 0.5 --seconds 5 --src-fps 50 --fps 30
  .venv/bin/python -m metalsim.render.film g1_flat --stage doc
  ```
  The README caption's `python -m metalsim.render.film g1_flat` records, renders and encodes the mp4s but does not
  write this WebP. A new recording is not bitwise the same take. Our own checkpoint instead of NVIDIA's: add
  `--checkpoint runs/g1regress/ckpt14/flat_main_s0.pt` to the record stage (tracked, training log beside it).
- **Provenance:** [G1_FLAT.md](G1_FLAT.md): checkpoint `assets/isaac/pretrained/rsl_rl/Isaac-Velocity-Flat-G1-v0/checkpoint.pt`
  (NVIDIA Isaac Lab, iteration 1499, sha256 `fdf0f242...`; source URL and published hash also in `rec/rec.json`
  `checkpoint`), task settings, how the env was chosen, gait, camera, timings and the power state each stage
  stamped. Hero video and stills: [g1_flat.mp4](g1_flat.mp4), `g1_flat_still{1,2,3}.jpg`.

### G1 rough (`g1_rough.webp`)

- **Shows:** the G1 on Isaac Lab's rough terrain (difficulty row 5 of rows 0-9) climbing out of a pit of inverted
  stairs, risers 0.14-0.16 m, its pelvis up to 0.97 m above its start, about six risers; driven by NVIDIA's published
  Isaac Lab 2.3 policy for the rough task (rsl_rl, iteration 2999, with the height scan; trained in PhysX, played
  without retraining); same camera, orbit and window as the flat cell; 1.44 MB.
- **Source:** the `readme` shot of `runs/film/g1_rough/`.
- **Reproduce:** `scripts/fetch_isaac_assets.sh` first (NVIDIA's checkpoint). The take, as made on 2026-10-31:
  ```
  scripts/gpu_run.sh g1-switch-rec-rough render 15 -- .venv/bin/python -m metalsim.render.film g1_rough --stage record
  scripts/gpu_run.sh g1-switch-render-rough render 25 -- .venv/bin/python -m metalsim.render.film g1_rough --stage render
  .venv/bin/python -m metalsim.render.film g1_rough --stage encode --shots hero
  python -m metalsim.render.film.encode webp runs/film/g1_rough/readme/frames docs/renders/g1_rough.webp --start 0.5 --seconds 5 --src-fps 50 --fps 30
  .venv/bin/python -m metalsim.render.film g1_rough --stage doc
  ```
  Our own checkpoint instead: add `--checkpoint runs/g1regress/ckpt14/rough_main_s0.pt` to the record stage.
- **Provenance:** [G1_ROUGH.md](G1_ROUGH.md): checkpoint `assets/isaac/pretrained/rsl_rl/Isaac-Velocity-Rough-G1-v0/checkpoint.pt`
  (NVIDIA Isaac Lab, iteration 2999, sha256 `d85a9ade...`; source URL and published hash also in `rec/rec.json`
  `checkpoint`), terrain, how the env was chosen, gait, timings and the power state each stage stamped. Hero video
  and stills: [g1_rough.mp4](g1_rough.mp4), `g1_rough_still{1,2,3}.jpg`.

### ANYmal C on sand (`anymal_sand.webp`)

- **Shows:** ANYmal C walking on Newton 1.5.2's `example_mpm_anymal` MPM sand with that example's walking policy;
  669,372 sand particles plus 261,061 decoration particles drawn as 27 grains each, 25.1 M grain instances, path traced
  at 1920x1080, 128 spp (32 passes x 4), OIDN. 3.0-8.0 s of the 8 s clip, 150 frames at 33 ms, 3.34 MB (above MuJoCo
  Warp's largest clip, 2.0 MB).
- **Source:** the clip's PNG frames 0090-0239, `~/metalsim-fresh/runs_render/sand25m/demo/frames` on the 14-inch,
  outside the repository (with `anymal_sand_25m_twolevel.mp4` and `log.txt` beside them). They were rendered on
  2026-10-13 by `runs_render/sand25m/demo.sh` (origin/main `94b75be`, two-level grain structures, `ramp.py` driving
  `scripts/particle_demo/render_anymal.py`) from the simulation in `runs_render/sim1`.
- **Reproduce:** the clip: [`docs/DEMO.md`, "The 25 M-grain render"](../DEMO.md#the-25-m-grain-render-anymal-on-sand-at-demo-scale)
  (`scripts/particle_demo/sim_anymal.py` in the Newton venv, then `render_anymal.py --frames 0:240 --width 1920
  --height 1080 --passes 32 --grains 27`). This file, centred 1440x1080 crop
  (command from `runs/readme_media/assets/anymal_sand.webp.json`):
  `python -m metalsim.render.film.encode webp ~/metalsim-fresh/runs_render/sand25m/demo/frames docs/renders/anymal_sand.webp --start 3 --seconds 5`.
- **Provenance:** `metalsim/render/PARTICLES.md`, "Demo" (the 2026-10-12 result at these settings, timings, and the
  credits: Menagerie `anybotics_anymal_c` BSD-3, Poly Haven `goegap` HDRI and `coast_sand_01` texture CC0, Newton's
  example and policy Apache-2.0); `metalsim/physics/mpm/MPM.md` §8k for the coupled step against Isaac.

### Franka-Pour (`franka_pour.webp`)

- **Shows:** Isaac Lab 3.0-EA's Franka-Pour on metalsim (MuJoCo Warp arm coupled to implicit MPM, 735 MPM particles
  per cup drawn as 4 render grains each), a policy trained here; the whole 3.5 s hero shot, 0.29 MB.
- **Source:** [franka_pour.mp4](franka_pour.mp4), the hero shot's 1280x720 docs copy (H.264, CRF 22). This WebP is a
  second-generation cut of that mp4: the 1920x1080 frames and master stay on the 16-inch that rendered them.
  Left-aligned 960x720 crop, which keeps both cups and the pour; the wrist is clipped by about 10 output pixels in the
  first few frames. libwebp merges near-identical frames, so the 105 source frames are stored as 83 (3.465 s).
- **Reproduce:**
  `python -m metalsim.render.film.encode webp docs/renders/franka_pour.mp4 docs/renders/franka_pour.webp --start 0 --seconds 0 --crop-x 0`;
  the film: `python -m metalsim.render.film franka_pour`.
- **Provenance:** [FRANKA_POUR.md](FRANKA_POUR.md) (16-inch; checkpoint `runs/franka_pour/logs/m414_2048_model_299.pt`).

### Same policy in Isaac Sim and MetalSim (`parity_g1_isaac.webp`, `parity_g1_metalsim.webp`)

- **Shows:** Isaac Lab's own rsl_rl G1 flat policy at iteration 1499 (trained in Isaac Lab with PhysX), rolled out from
  the same start (Isaac's default state, command 0.5 m/s forward) under the same camera and lights: left Isaac Sim 5.1,
  PhysX + RTX real-time, recorded on the L4; right MetalSim, MuJoCo Warp at 2.5 ms + tier 2. Physics and renderer both
  differ, so the two cells drift apart; they are not the same frames rendered twice. 0.5-4.5 s of the 8 s video, 100
  frames at 40 ms (25 fps, the source's rate), 0.21 / 0.28 MB.
- **Source:** [`docs/gallery/g1_stage_isaac_it1499.mp4`](../gallery/g1_stage_isaac_it1499.mp4) (2048x576, two
  1024x576 panels), made by `metalsim.parity.side_by_side` (`scripts/gallery/g1_stage_videos.sh`; re-rendered with
  the RTX-parity preset on 2026-09-25, `a14b6a87`). The Isaac recording and checkpoint
  (`runs/parity/isaac/parity_out/play/isaac_it1499`, `runs/parity/isaac/ckpt_out/model_1499.pt`) are not in the
  repository, so the video is the source.
- **Reproduce:** a 540x405 crop at (340, 166) of the left panel and at (1364, 166), the same place in the right
  panel (ffmpeg rounds the height to 404 for yuv420p), then the encoder:
  ```
  ffmpeg -ss 0.5 -t 4 -i docs/gallery/g1_stage_isaac_it1499.mp4 -vf crop=540:405:340:166 -start_number 0 isaac/frame_%04d.png
  ffmpeg -ss 0.5 -t 4 -i docs/gallery/g1_stage_isaac_it1499.mp4 -vf crop=540:405:1364:166 -start_number 0 metalsim/frame_%04d.png
  python -m metalsim.render.film.encode webp isaac docs/renders/parity_g1_isaac.webp --start 0 --seconds 0 --src-fps 25 --fps 25
  python -m metalsim.render.film.encode webp metalsim docs/renders/parity_g1_metalsim.webp --start 0 --seconds 0 --src-fps 25 --fps 25
  ```
- **Provenance:** [`docs/PARITY.md` §1.5](../PARITY.md#15-learning-on-the-g1-task): over the 8 s video this policy
  travels 3.11 m in Isaac and 3.21 m in MetalSim; Isaac's checkpoints at iterations 500 / 1000 / 1499 end at pelvis
  heights 0.62-0.64 m in Isaac and 0.61-0.63 m in MetalSim. Still to come (plan
  `docs/research/readme_media_plan_2026-10-31.md` §4b, next L4 session): one rollout recorded on the Isaac side and
  replayed in MetalSim tier 2 from Isaac's recorded poses (`metalsim/render/rtx_parity.py`, the §1.7 protocol), so
  that the two cells differ only in rendering.

### Lidar through glass (`lidar_glass_rgb.webp`, `lidar_glass_depth.webp`, `lidar_glass_lidar.webp`)

- **Shows:** one scene through three sensors, a ball rolling from open floor to behind a glass pane (alpha 0.3) and
  back: the tier-2 path-traced camera (through the glass); the ray-traced depth camera, which stops at the pane as
  MuJoCo's `mj_ray` and Isaac's depth do (inferno, near bright, 0.4-2.8 m); a 32 x 1024 lidar with two returns (cyan:
  echoes on glass; the rest coloured by range on the depth scale; the ring is the sensor; the empty bottom quarter is
  the lidar's blind zone near the sensor). A 5 s seamless loop, 0.30 / 0.06 / 0.63 MB (the depth view has 74 stored
  frames: libwebp merges identical ones).
- **Source:** `scripts/readme_media/lidar_glass.py` (its docstring has the scene and every sensor setting); frames in
  `runs/readme_media/assets/lidar_glass_{rgb,depth,lidar}_frames/` (640x480, 150 at 30 fps, local) and its json,
  `runs/readme_media/scratch/lidar-glass/lidar_glass.json` (settings, exposure, per-frame timings, software and
  hardware; local).
- **Reproduce:**
  `scripts/gpu_run.sh lidar-glass render 10 -- .venv/bin/python scripts/readme_media/lidar_glass.py --compare 75`, then
  per view `python -m metalsim.render.film.encode webp runs/readme_media/assets/lidar_glass_rgb_frames docs/renders/lidar_glass_rgb.webp --start 0 --seconds 5 --src-fps 30`
  (`_depth`, `_lidar` likewise).
- **Provenance:** sensor settings from `tests/test_sensor_transparency.py` and `docs/RENDER_COVERAGE.md` §3e;
  `hits_per_ray=1` and `min_echo_sep=0.1` differ from the defaults, for the reasons in the script's docstring. At frame
  75, the lidar gets 30,933 echoes with the panes (12,072 on glass, 8,438 beams with two returns) and 18,886 without
  them. Rendered on the 14-inch on battery (power-mode setting not recorded); its per-frame times are not timing figures.

### Throughput chart (`throughput_dark.svg`, `throughput_light.svg`)

- **Shows:** env-steps per second, like-for-like pairs only: G1 flat and rough env step at 4,096 envs and Franka-Pour's
  full PPO loop at 1,024 envs, MetalSim on the M4 Max against Isaac Lab 3.0 (Newton / MuJoCo Warp and PhysX) on the
  L4. Dark and light variants for a `<picture>` element, full width under the grid.
- **Source and reproduce:** `.venv/bin/python scripts/readme_media/throughput_chart.py --out docs/renders` (standard
  library only; default output `runs/readme_media/assets/`).
- **Provenance:** every number sits in the script beside its doc section (`docs/PARITY.md` §1.4, §1.8, §1b.4) and raw
  log; its docstring lists what is not drawn and why (NVIDIA's RTX 4090 figure and Isaac's rough PhysX number, which
  have no matching MetalSim measurement).

## Making one

Needs the documented install (`pip install -e ".[full]"`, docs/GUIDE.md): Open Image Denoise comes with the core
dependency `pyoidn` (a macOS arm64 wheel that bundles OIDN; MIT binding, OIDN Apache-2.0), ffmpeg with the `film`
extra (`imageio-ffmpeg`). The tool checks both before it records anything and prints the fix; `--no-denoise` renders
without OIDN (said so in the doc).

```bash
python -m metalsim.render.film --list                 # filmable tasks
python -m metalsim.render.film franka_pour            # record, render, encode, doc
python -m metalsim.render.film franka_pour --stage render --frames 0:30:10 --passes 8   # quick preview frames
```

The tool (`metalsim/render/film/`) runs in four file-based stages, so the path-traced render happens offline, after
the rollout, in the venv that has the renderer:

1. **record** (the task's venv): the task's recorder adapter rolls out the policy; per frame the bodies (and MPM
   particles) are captured and turned into the world poses of a render twin: an MJCF of the scene's visual geometry
   (Newton scenes through `newton_bridge.NewtonRenderSource`, MuJoCo scenes by forward kinematics).
2. **render** (main venv): tier-2 path tracing per shot (one env, or a 2 x 2 grid), particles as render grains, OIDN.
3. **encode**: H.264 master next to the frames, a 1280-wide copy and JPEG stills here; with `--webp`, also a 320x240
   animated WebP of a window of the clip (`--webp-window START:SECONDS`).
4. **doc**: `docs/renders/<TASK>.md` generated from the stage JSONs (`runs/film/<task>/`).

Camera and recording settings use Isaac Lab's names (`ViewerCfg`: `eye`, `lookat`, `origin_type` world / env /
asset_root / asset_body, `env_index`, `body_name`, `resolution`; `VideoRecorderCfg`: `fps` = 1 / step_dt,
`video_length`). A new task needs a `FilmTask` entry in `metalsim/render/film/tasks.py` (camera, envs, look, assets)
and a recorder adapter in `adapters.py` (build the env, load the policy, step); see the Franka-Pour (Newton) and G1
(MuJoCo Warp) adapters.

Fits but not yet ported: the ANYmal-on-sand demo (`scripts/particle_demo/`) is the same record-then-render shape; its
extras (a sand bed larger than the sim's, decoration grains, a surface-depth cull of the 25 M grains, a tracking orbit)
would become look options and an adapter around Newton's `example_mpm_anymal`.

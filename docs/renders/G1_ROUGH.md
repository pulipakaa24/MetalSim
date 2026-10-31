# G1 rough-terrain walk: a metalsim render

Unitree G1 climbing Isaac Lab's rough terrain: the G1 rough velocity task (stairs, slopes, boxes, height scan) on metalsim (MuJoCo Warp on Metal), driven by a policy trained here with the task's PPO settings. This take was filmed on the 14-inch M4 Max on battery in Low Power Mode, so its Performance table is not a timing figure.

**hero** (301 frames, 6.0 s at 50 fps): [video](g1_rough.mp4) (2.3 MB, H.264)

![hero, frame 75](g1_rough_still1.jpg)
![hero, frame 150](g1_rough_still2.jpg)
![hero, frame 240](g1_rough_still3.jpg)

## What is shown

- 40 envs simulated, filmed: 8, 13, 15, 14 (selection: automatic, the adapter ranking).
  - env 8: terminated False, time_out False, base_z 0.000, level 0, col 0, end_frame -1
  - env 13: terminated False, time_out False, base_z 0.000, level 0, col 0, end_frame -1
  - env 15: terminated False, time_out False, base_z 0.000, level 0, col 0, end_frame -1
  - env 14: terminated False, time_out False, base_z 0.000, level 0, col 0, end_frame -1
- Policy: `runs/g1regress/ckpt14/rough_main_s0.pt` (iteration 1000); physics: MuJoCo Warp BatchSim (metalsim.learn.g1_velocity).

## Performance

Measured on the machine below; one frame = one policy step (Isaac Lab's `render_fps` = 1 / step_dt).

| stage | measurement |
|---|---|
| simulate (record) | 301 frames x 40 envs: 77 ms/frame mean (76 median, first 114 incl. graph capture), 23.2 s total; env build 4.2 s; stage wall 33.5 s |
| render twin (CPU) | 48 geoms, 47 meshes, 0 textures; build 0.8 s, poses 0.0 s |
| render: hero | 1280x720 (1x1 panels of 1280x720), tier 2 path tracing, 64 spp (16 passes x 4), 3 bounces, denoiser oidn; 0.44 s/frame mean (0.44 median, 0.51 max), 2.2 min for 301 frames; first frame: one pass 65 ms, 22 ms per further pass; set-up 1.9 s |
| encode: hero | 3.1 s (master CRF 16 + docs copy CRF 22, libx264 slow) |
| render: readme | 640x480 (1x1 panels of 640x480), tier 2 path tracing, 64 spp (16 passes x 4), 3 bounces, denoiser oidn; 0.17 s/frame mean (0.17 median, 0.19 max), 51.7 s for 301 frames; first frame: one pass 23 ms, 8 ms per further pass; set-up 1.7 s |
| end to end | 3.7 min (record + render + encode, sum of stage walls) |

## Hardware and pins

- Mac16,6 (Apple M4 Max, 40 GPU cores, 64 GB), macOS 27.0.1 (26A434).
- metalsim `687f3d7b1d+dirty`; record stage (main venv): warp 1.18.0.dev20260917 (`4127c48183`), mujoco 3.14.0, mujoco_warp 3.14.0 (`07a51a63bc`), torch 2.14.0, numpy 2.5.3.
- render stage (main venv): warp 1.18.0.dev20260917 (`4127c48183`), mujoco 3.14.0, numpy 2.5.3.

## Assets

| asset | source | licence | where from |
|---|---|---|---|
| Unitree G1 (g1_minimal.usd, Isaac 5.1 asset root) | NVIDIA Isaac Lab assets | NVIDIA Isaac Sim asset terms (fetched, not redistributed) | https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/IsaacLab/Robots/Unitree/G1/g1_minimal.usd (`scripts/fetch_isaac_assets.sh`) |
| Studio HDRI (poly_haven_studio, 2k) | Poly Haven | CC0 1.0 (Poly Haven; fetched, not committed) | https://polyhaven.com/a/poly_haven_studio (`scripts/fetch_polyhaven_assets.sh train`) |
| Floor texture (laminate_floor_02, 1k diffuse) | Poly Haven | CC0 1.0 (Poly Haven; fetched, not committed) | https://polyhaven.com/a/laminate_floor_02 (`scripts/fetch_polyhaven_assets.sh train`) |
| Rough terrain: Isaac Lab 2.3.2 ROUGH_TERRAINS_CFG (10 x 20 sub-terrains of 8 m), generated from the task seed | Isaac Lab task config (port: metalsim.learn.isaac_terrain) | BSD-3-Clause | https://github.com/isaac-sim/IsaacLab (`procedural (metalsim.learn.terrain)`) |
| Policy checkpoint rough_main_s0.pt (trained on metalsim, 1,000 iterations; sha256 b5b25ad3...) | this repository | this repository's licence | `runs/g1regress/ckpt14/rough_main_s0.pt` (`committed`) |

## How it was made

**The policy.** `runs/g1regress/ckpt14/rough_main_s0.pt`, sha256
`b5b25ad377ba40dbacc18fd90462a64c20203ee7fb6b97b89c3110755cf9ca6f` (2,614,138 bytes; the record stage writes the same
hash into `rec.json`). Trained on main `ad97888e` on 2026-10-13 on the 14-inch M4 Max with the documented G1 rough
run's settings (`runs/ellip_default/run.sh`: `python -m metalsim.learn.g1_velocity 4096 rough train 1000 <log>
<checkpoint> 0.0025 --reward_cfg rough_isaac --seed 0`): 4,096 envs, physics dt 2.5 ms with decimation 8 (50 Hz
control), rollouts of 24 steps, 1,000 iterations (98.3 M env steps), Isaac's terrain curriculum, actor MLP 512-256-128
on a 310-dim observation that includes the 187-ray height scan, the elliptic contact default. Training log:
`runs/g1regress/ckpt14/rough_main_s0.log` (header with every setting; iteration 1000: return +7.55, episode length
985, mean terrain level 5.86). `docs/research/ppowarp_checkpoint_load_2026-10-13.md` sets its curve beside the
documented run of 2026-09-26 (`runs/ellip_default/g1_rough_ellip_default_s0.log`, `docs/PARITY.md` §1.5:
return +8.23, episode length 973, level 5.65 at iteration 1000) and replays it on the 16-inch: 1,016 of 1,024
episodes run to the 20 s time-out, 9 falls over 1,200 control steps (`runs/g1regress/replay_main_rough_it1000.log`),
and over levels 0 to 9 at 4,096 envs 3,983 of 4,096 first episodes run to the time-out
(`runs/g1regress/main_rough_it1000_all_levels.log`).
The file is the training run's output, copied unchanged (the log's last line names it) and tracked so that a fresh
clone can film. Not used: `g1_rough_ppowarp_fixed_it1500.pt` walks only under its own pyramidal-era settings
(`docs/research/readme_media_plan_2026-10-31.md` section 0).

**Task settings.** `metalsim.learn.g1_velocity.G1VelocityTask(40, terrain="rough", seed=0)` with the settings saved
in the checkpoint (`reward_cfg="rough_isaac"`, height-scan ray order `xy`, exact height scan, terrain collision
`boxes_local`, `contact_cfg="recommended"`, velocities at the COM; `rec.json` `adapter.saved_task_settings`) plus
physics dt 2.5 ms from the training log's header (`G1_TRAIN_TASK_KW`). Seed 0 is the training run's task seed, which
generates the terrain, so the film's terrain is the one the policy trained on: Isaac Lab 2.3.2's
`ROUGH_TERRAINS_CFG` (10 difficulty rows by 20 columns of 8 m sub-terrains; by column, pyramid stairs, inverted
pyramid stairs, random boxes, random rough ground, slopes and inverted slopes). `adapter_kw["terrain_level"] = 5`
starts every env on difficulty row 5 instead of the task's random rows 0 to 5: training ended at mean level 5.86, so
this is the difficulty the policy last trained on, and on this row the inverted stairs have risers of 0.14 to
0.16 m. The 40 envs sit two per column (`rec.json` `adapter.terrain_types`). Actions, commands, terminations and the
6 s, 50 fps film are as in the flat film.

**How the env was chosen.** `envs="auto"` ranks the 40 envs (`G1Rough.rank`): first the envs whose first episode was
not ended by a termination, then the highest climb of the pelvis above its start, then the longest path. The record
stage refuses to film an env whose first episode terminated unless `--allow-early-end` is given. In this take none of
the 40 terminated; env 8, filmed, is on column 4 (inverted pyramid stairs) and climbed out of the pit: its pelvis rose
0.75 m, about five risers, over a 4.12 m path (next: env 13, 0.67 m; env 15, 0.63 m; `rec.json`
`adapter.first_episode`). Under "What is shown", `base_z`, `level` and `col` are written only when an episode ends;
with `end_frame` -1 (still running when the film ends) their zeros are placeholders, not the env's level (5) or
column (4). That list is the top four of the ranking (the record keeps spares); both shots film one env, the first
(`render.json` `envs`).

**Terrain in the render.** The twin carries the task's own collision surface: its heightfield (2001 x 1201 samples
at 0.1 m) and Isaac's 14,080 terrain boxes merged into one mesh, `terrain_boxes`, with planar UVs on each face so
that the floor texture lies flat on treads and risers; both take the laminate texture (`ground_geoms`).

**Camera.** Both shots track the pelvis as Isaac's `ViewerCfg(origin_type="asset_body", body_name="pelvis")` does,
the tracked point smoothed by a centred 1 s moving average (`smooth_s`), with Kit's default perspective lens (36 deg
vertical field of view). `hero` (1280x720; the video and stills on this page): eye (2.6, -2.0, 1.3) m from the pelvis,
looking at (0, 0, -0.05) m, 3.55 m away. `readme` (640x480, 4:3, for the README grid): the same direction and
look-at at 0.75 times the distance (eye (1.95, -1.5, 0.98) m, 2.66 m), so the robot fills about three quarters of the
frame height, plus a slow orbit of 10 deg/s about the vertical through the look-at point, symmetric about the middle
of the film: -30 deg at 0 s, 0 at 3 s (the hero's direction), +30 deg at 6 s, counter-clockwise seen from above, as
MuJoCo's azimuth grows. The orbit copies MuJoCo Warp's README clips: `mujoco_warp/record.py` (google-deepmind/
mujoco_warp at `d41a47c0`, lines 43 and 142) sets `cam.azimuth = 90 + (step - nstep / 2) * cam_azimuth_speed` with
0.05 deg per physics step, and its G1 scenes step at 5 ms (`benchmarks/unitree_g1/unitree_g1_mjlab.xml`, line 14),
so 10 deg/s.

**Rendering.** Tier 2 (the path tracer), 64 samples per pixel (16 passes x 4), 3 bounces, Open Image Denoise, ACES
tone map, sRGB; lit by a Poly Haven studio HDRI as an importance-sampled dome, with a laminate floor texture on the
task's ground instead of Isaac's grey ground. The render twin is the task's own G1 model with Isaac's visual meshes
(`build_g1_model(visuals=True)`), posed per frame from the recorded joint state. On bodies that have visual meshes the
colliders are hidden (the three `*Cube` boxes, `rec.json` `twin.visibility`), and the touch-sensor sites are hidden:
one of them, `torso_link_touch`, was the grey box of the earlier G1 film. The ray tracer's env cell is sized up front
from the recorded geometry and the whole camera path, so the floor is drawn complete to the horizon (`render.json`
`env_grid.growths` 0). Each `render.json` carries a `cache_key`, a hash of the recording and of every setting that
changes a pixel, so a re-recording into the same directory re-renders rather than reusing old frames
(`docs/research/readme_media_plan_2026-10-31.md` section 2, defects 2, 3 and 5).

**README clip.** [g1_rough.webp](g1_rough.webp) is the `readme` shot from 0.5 s to 5.5 s (frames 25 to 274), resampled
from 50 to 30 fps and written as MuJoCo Warp writes its README clips: 320x240 (Lanczos), lossy animated WebP from
Pillow at quality 70, 33 ms per frame, looping; 150 frames. Command (re-run on 2026-10-31 from `runs/film/g1_rough`, byte
for byte the same file):
`python -m metalsim.render.film.encode webp runs/film/g1_rough/readme/frames docs/renders/g1_rough.webp --start 0.5
--seconds 5 --src-fps 50 --fps 30`. The film tool's route, `python -m metalsim.render.film g1_rough --stage encode
--shots readme --webp --webp-window 0.5:5`, uses the same encoder but names its output `g1_rough_readme.webp` and also
writes a `g1_rough_readme.mp4` and a still.

**This take, and where its timings come from.** Recorded, rendered and encoded on 2026-10-31 on the 14-inch M4 Max
(Mac16,6) **on battery in Low Power Mode**, at metalsim `687f3d7b` plus the uncommitted film-tool fixes (hence
`+dirty` in the pins). The Performance table above records how long each stage took there. It is not a timing figure:
quote nothing on this page as MetalSim's speed. MetalSim's throughput figures are in `docs/PARITY.md` §1.4 and §1.8
(16-inch, on power). Record and render ran as GPU-queue jobs of kind render, one at a time: `--stage record` into
`runs/readme_media/film/g1_rough`, then `--stage render` for both shots there. The take was then moved to
`runs/film/g1_rough` (the tool's default `--out`), where `--stage encode --shots hero` and `--stage doc` ran (CPU).
Nothing was re-recorded. A new recording is another take of the same policy, not this one: the GPU rollout is not
bitwise repeatable, so which env is filmed, and its path, can change.

## Reproduce

```bash
scripts/fetch_isaac_assets.sh                # NVIDIA assets (not redistributed)
scripts/fetch_polyhaven_assets.sh train      # CC0 HDRI and textures
python -m metalsim.render.film g1_rough   # record, render, encode, doc (main venv)
# or stage by stage (resumable):
python -m metalsim.render.film g1_rough --stage record   # runs in the .venv venv
python -m metalsim.render.film g1_rough --stage render
python -m metalsim.render.film g1_rough --stage encode
python -m metalsim.render.film g1_rough --stage doc
```

Generated by `python -m metalsim.render.film g1_rough --stage doc` from the stage logs (`runs/film/g1_rough/rec/rec.json`, `render.json`, `render_frames.jsonl`, `encode.json` per shot).

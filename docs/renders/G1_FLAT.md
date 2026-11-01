# G1 flat-terrain walk: a metalsim render

Unitree G1 walking on flat ground: Isaac Lab's G1 flat velocity task on metalsim (MuJoCo Warp on Metal), driven by NVIDIA's published Isaac Lab 2.3 policy for the task (rsl_rl, iteration 1499, trained in PhysX), played without retraining. Rendered on the 14-inch MacBook Pro (Mac16,6), not in a timing run, so its Performance table is not a timing figure.

**hero** (301 frames, 6.0 s at 50 fps): [video](g1_flat.mp4) (1.8 MB, H.264)

![hero, frame 75](g1_flat_still1.jpg)
![hero, frame 150](g1_flat_still2.jpg)
![hero, frame 240](g1_flat_still3.jpg)

## What is shown

- 16 envs simulated, filmed: 3, 1, 6, 2 (selection: automatic, the adapter ranking).
  - env 3: terminated False, time_out False, base_z 0.000, level 0, col 0, end_frame -1
  - env 1: terminated False, time_out False, base_z 0.000, level 0, col 0, end_frame -1
  - env 6: terminated False, time_out False, base_z 0.000, level 0, col 0, end_frame -1
  - env 2: terminated False, time_out False, base_z 0.000, level 0, col 0, end_frame -1
- Policy: `assets/isaac/pretrained/rsl_rl/Isaac-Velocity-Flat-G1-v0/checkpoint.pt` (iteration 1499); physics: MuJoCo Warp BatchSim (metalsim.learn.g1_velocity).

## Performance

Measured on the machine below; one frame = one policy step (Isaac Lab's `render_fps` = 1 / step_dt).

| stage | measurement |
|---|---|
| simulate (record) | 301 frames x 16 envs: 41 ms/frame mean (41 median, first 66 incl. graph capture), 12.4 s total; env build 1.3 s; stage wall 16.5 s |
| render twin (CPU) | 47 geoms, 46 meshes, 0 textures; build 0.4 s, poses 0.0 s |
| render: hero | 1280x720 (1x1 panels of 1280x720), tier 2 path tracing, 64 spp (16 passes x 4), 3 bounces, denoiser oidn; 0.25 s/frame mean (0.25 median, 0.28 max), 74.1 s for 301 frames; first frame: one pass 52 ms, 7 ms per further pass; set-up 1.0 s |
| encode: hero | 4.7 s (master CRF 16 + docs copy CRF 22, libx264 slow) |
| render: readme | 640x480 (1x1 panels of 640x480), tier 2 path tracing, 64 spp (16 passes x 4), 3 bounces, denoiser oidn; 0.11 s/frame mean (0.10 median, 0.13 max), 31.7 s for 301 frames; first frame: one pass 19 ms, 4 ms per further pass; set-up 0.9 s |
| end to end | 2.1 min (record + render + encode, sum of stage walls) |

## Hardware and pins

- Mac16,6 (Apple M4 Max, 40 GPU cores, 64 GB), macOS 27.0.1 (26A434).
- Power as each stage stamped it (`pmset -g batt` and `pmset -g`; `hardware.power`: the record stage's when it wrote rec.json, a render shot's at its first frame): record: on battery (66 %, discharging), power-mode setting High Power (pmset `powermode 2`), 2026-10-31 04:51 (-0700); render hero: on battery (66 %, discharging), power-mode setting High Power (pmset `powermode 2`), 2026-10-31 04:51 (-0700); render readme: on battery (62 %, discharging), power-mode setting High Power (pmset `powermode 2`), 2026-10-31 04:53 (-0700).
- metalsim `ef346e7efc+dirty`; record stage (main venv): warp 1.18.0.dev20260917 (`4127c48183`), mujoco 3.14.0, mujoco_warp 3.14.0 (`07a51a63bc`), torch 2.14.0, numpy 2.5.3.
- render stage (main venv): warp 1.18.0.dev20260917 (`4127c48183`), mujoco 3.14.0, numpy 2.5.3.

## Assets

| asset | source | licence | where from |
|---|---|---|---|
| Unitree G1 (g1_minimal.usd, Isaac 5.1 asset root) | NVIDIA Isaac Lab assets | NVIDIA Isaac Sim asset terms (fetched, not redistributed) | https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/IsaacLab/Robots/Unitree/G1/g1_minimal.usd (`scripts/fetch_isaac_assets.sh`) |
| Studio HDRI (poly_haven_studio, 2k) | Poly Haven | CC0 1.0 (Poly Haven; fetched, not committed) | https://polyhaven.com/a/poly_haven_studio (`scripts/fetch_polyhaven_assets.sh train`) |
| Floor texture (laminate_floor_02, 1k diffuse) | Poly Haven | CC0 1.0 (Poly Haven; fetched, not committed) | https://polyhaven.com/a/laminate_floor_02 (`scripts/fetch_polyhaven_assets.sh train`) |
| Policy: Isaac Lab 2.3 pretrained rsl_rl checkpoint (Isaac-Velocity-Flat-G1-v0, iteration 1499; sha256 fdf0f242...) | NVIDIA Isaac Lab | NVIDIA Isaac Sim asset terms (fetched, not redistributed) | https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/IsaacLab/PretrainedCheckpoints/rsl_rl/Isaac-Velocity-Flat-G1-v0/checkpoint.pt (`scripts/fetch_isaac_assets.sh`) |

## How it was made

**The policy.** NVIDIA's published Isaac Lab 2.3 checkpoint for `Isaac-Velocity-Flat-G1-v0`: rsl_rl, iteration
1499, the last of the task's 1,500 (`assets/isaac/g1_rsl_rl_ppo_cfg.py`, `G1FlatPPORunnerCfg`), trained by NVIDIA in
Isaac Sim with PhysX. MetalSim plays it as published, without retraining or fine-tuning. Source:
<https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/IsaacLab/PretrainedCheckpoints/rsl_rl/Isaac-Velocity-Flat-G1-v0/checkpoint.pt>
(the path Isaac Lab's `--use_pretrained_checkpoint` resolves, `isaaclab_rl/utils/pretrained_checkpoint.py`, under
the Isaac 5.1 asset root), sha256 `fdf0f242bb8f3bcfc1aa9b7ff2761af4c014cf1cdc7e49242c9b92f3581720e7`, 2,027,170
bytes. It is NVIDIA's file under NVIDIA's asset terms: `scripts/fetch_isaac_assets.sh` downloads it to
`assets/isaac/pretrained/rsl_rl/Isaac-Velocity-Flat-G1-v0/checkpoint.pt` and checks the hash; `.gitignore` (`*.pt`)
keeps it out of the repository. The record stage refuses a file whose sha256 is not the published one and writes the
source URL, the published hash, the format and the iteration into `rec.json` `checkpoint` (`in_repo` false). The
actor is an ELU MLP, 123 -> 256-128-128 -> 37 with no observation normaliser, read from the checkpoint's `actor.*`
weights with `torch.load(weights_only=True)` (`adapters.load_g1_checkpoint`); the film plays the actor's mean.

Our own checkpoint stays selectable and tracked: `--checkpoint runs/g1regress/ckpt14/flat_main_s0.pt` (500 iterations trained
in MetalSim, its training log beside it) films it with the task settings that checkpoint saved, and this page's text
then does not describe the policy that ran. The previous take of this page, in git history at `ef346e7e`, filmed it.

**Task settings.** `metalsim.learn.g1_velocity.G1VelocityTask(16, terrain="flat", seed=3)` with the settings Isaac's
checkpoint implies (`adapters.ISAAC_TASK_SETTINGS["flat"]`: `reward_cfg="flat"`, which carries `G1FlatEnvCfg`'s
command ranges, sideways velocity within +-0.5 m/s) and the task's defaults (`contact_cfg="recommended"`, base and
feet-slide velocities at the COM), plus physics dt 2.5 ms with decimation 8 (`G1_TRAIN_TASK_KW`). `rec.json`
`adapter.task_resolved` lists what ran.
The policy's observation is Isaac's policy group in Isaac's order (base linear and angular velocity, projected
gravity, command, joint positions, joint velocities, last action; on rough terrain then the 187 height-scan rays in
Isaac's "xy" order), computed by the task's own kernel. The joint blocks and the actions are in Isaac's joint order
(PhysX's breadth-first articulation order, `adapters.ISAAC_G1_JOINTS`), mapped to MetalSim's by joint name as
`metalsim/parity/side_by_side.py` does; the actions are Isaac's joint-position targets (scale 0.5 about the default
pose). The task always applies Isaac's training observation noise, which Isaac's PLAY configuration turns off; on flat
ground that changed no measured gait figure (`runs/readme_media/handoff/isaac-pretrained.md`, Verified B). Commands are the task's own, as in
training (Isaac's uniform velocity command with a heading target, resampled every 10 s). An episode ends on the task's
terminations (torso contact above 1 N, a non-finite state) or its 20 s time-out. The film is 6 s: 301 frames, one per
20 ms policy step (50 fps).

**How the env was chosen.** 16 envs are simulated, and `envs="auto"` lets the adapter rank them (`G1Flat.rank`).
First come the envs whose first episode was not ended by a termination, then those whose pelvis never dropped more
than 0.25 m below its start (a robot down without torso contact, which the task does not terminate). Next comes a
natural-gait preference: envs whose lower swing apex of the two feet is at least the median of those candidates, and
whose larger stance slip is at most their median (`adapters.natural_gait`; relative to the take's own envs, because no
absolute reference exists). The longest path breaks ties. The record stage refuses to film an env whose first episode
terminated unless `--allow-early-end` is given. In this take none of the 16 terminated and the largest pelvis drop was
0.13 m. Envs 1 and 3 passed the gait preference (`rec.json` `gait.natural_gait_envs`), and env 3, filmed, walked the
longer path: 4.84 m in 6 s (env 1: 3.18 m; `rec.json` `adapter.first_episode`). Under "What is shown", `base_z`,
`level` and `col` are written only when an episode ends; with `end_frame` -1 (still running when the film ends) their
zeros are placeholders. That list is the top four of the ranking (the record keeps spares); both shots film one env,
the first (`render.json` `envs`).

**Gait.** Measured by the record stage on every simulated env from 1 s to the end of the film (`rec.json` `gait`, with
its definition; `record.GAIT_DEFINITION`). Env 3, filmed: 0.85 m/s at a commanded 0.98 m/s; 3.78 gait cycles per
second on both feet (touch-downs 3.59 / 3.79 per second, left / right); swing apex 7.1 / 7.0 cm above the stance
height; stance slip 0.019 / 0.015 m/s; airborne 56 / 47 % of the time; 0.225 m covered per gait cycle. Medians over
the 16 envs: 0.53 m/s (commanded 0.62), 3.91 Hz, apex 7.3 / 5.7 cm, slip 0.019 / 0.013 m/s, 0.134 m per cycle.

**The quick, short-stepping gait is Isaac's own.** Isaac Lab's G1 policies walk this way in PhysX too. Checkpoints of
the flat task trained with Isaac Lab in Isaac Sim (this project's L4 training run, iterations 500 / 1000 / 1499,
played in Isaac Sim at 0.5 m/s) step at 3.85 / 3.60 / 3.23 Hz with 0.108 / 0.110 / 0.123 m per gait cycle
(`docs/research/speculative_contacts_2026-09-27.md`, Verdict 1 and the cadence table of §2). MuJoCo Warp adds 3.4 to
5.9 % to the cadence of the same checkpoints under every contact setting measured (same note). NVIDIA's published
checkpoint itself was not played in PhysX for this page: Isaac Sim does not run on this machine.

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

**README clip.** [g1_flat.webp](g1_flat.webp) is the `readme` shot from 0.5 s to 5.5 s (frames 25 to 274), resampled
from 50 to 30 fps and written as MuJoCo Warp writes its README clips: 320x240 (Lanczos), lossy animated WebP from
Pillow at quality 70, 33 ms per frame, looping; 150 frames. Command, run from the repository root after the render
stage: `python -m metalsim.render.film.encode webp runs/film/g1_flat/readme/frames docs/renders/g1_flat.webp --start 0.5
--seconds 5 --src-fps 50 --fps 30`. The film tool's route, `python -m metalsim.render.film g1_flat --stage encode
--shots readme --webp --webp-window 0.5:5`, uses the same encoder but names its output `g1_flat_readme.webp` and also
writes a `g1_flat_readme.mp4` and a still.

**This take, and where its timings come from.** Recorded, rendered and encoded on 2026-10-31 on the 14-inch MacBook
Pro (Mac16,6), at metalsim `ef346e7e` plus the uncommitted switch to Isaac's checkpoints (hence `+dirty` in the pins).
This was not a timing run: MetalSim's timing runs are made on the 16-inch through the GPU queue's timing kind
(`scripts/gpu_run.sh <name> timing <minutes> -- <command>`, which waits for an idle device), and its throughput
figures are in `docs/PARITY.md` §1.4 and §1.8. The Performance table above records how long each stage took here;
quote nothing on this page as MetalSim's speed. Power, as the stages stamped it (`pmset -g batt` and `pmset -g`;
"Hardware and pins" below): on battery and discharging at every stamp (66 % when the record stage wrote `rec.json` at 04:51, 66 % at the hero shot's first frame at 04:51 and 62 % at the readme shot's at 04:53, local time), with pmset's power-mode setting at High Power (`powermode 2`: the configured setting, which does not show that the machine ran in High Power Mode); the encode and doc stages stamp nothing. Record and render ran as GPU-queue jobs of kind render, one at a time,
into `runs/film/g1_flat` (the tool's default `--out`): `--stage record`, then `--stage render` (both shots); then
`--stage encode --shots hero`, the README WebP command above and `--stage doc` (CPU). A new recording is another take
of the same policy, not this one: the GPU rollout is not bitwise repeatable, so which env is filmed, and its path, can
change.

## Reproduce

```bash
scripts/fetch_isaac_assets.sh                # NVIDIA assets (not redistributed)
scripts/fetch_polyhaven_assets.sh train      # CC0 HDRI and textures
python -m metalsim.render.film g1_flat   # record, render, encode, doc (main venv)
# or stage by stage (resumable):
python -m metalsim.render.film g1_flat --stage record   # runs in the .venv venv
python -m metalsim.render.film g1_flat --stage render
python -m metalsim.render.film g1_flat --stage encode
python -m metalsim.render.film g1_flat --stage doc
```

Generated by `python -m metalsim.render.film g1_flat --stage doc` from the stage logs (`runs/film/g1_flat/rec/rec.json`, `render.json`, `render_frames.jsonl`, `encode.json` per shot).

# Renders

Tasks filmed with metalsim's own renderer, each with a doc that says what is shown, how long every stage took on
which machine, every asset with its source and licence, and the commands to reproduce it.

| film | what | doc |
|---|---|---|
| Franka-Pour | trained policy pours granular MPM media between two cups ([Isaac Lab](https://github.com/isaac-sim/IsaacLab) 3.0-EA task); one env and a 2 x 2 grid | [FRANKA_POUR.md](FRANKA_POUR.md) |
| G1 flat walk | Isaac Lab's G1 flat velocity task with NVIDIA's published Isaac Lab 2.3 policy ([rsl_rl](https://github.com/leggedrobotics/rsl_rl), iteration 1499, trained in [PhysX](https://github.com/NVIDIA-Omniverse/PhysX); fetched by `scripts/fetch_isaac_assets.sh`); 6 s, 1280x720 hero and a 640x480 orbiting shot for the README, 64 spp | [G1_FLAT.md](G1_FLAT.md) |
| G1 rough walk | Isaac Lab's G1 rough velocity task (stairs, slopes, boxes) with NVIDIA's published policy (iteration 2999); the G1 climbs out of a pit of inverted stairs; same two shots | [G1_ROUGH.md](G1_ROUGH.md) |
| G1 walk, Unitree's policy | Unitree's own pretrained G1 walking policy ([unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym) `deploy/pre_train/g1/motion.pt`, BSD-3-Clause; the file Unitree's MuJoCo and real-robot deploy configs both load), played by Unitree's [MuJoCo](https://github.com/google-deepmind/mujoco) sim2sim loop ported to [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp) on Unitree's G1 model; 0.8 s gait clock, 1.25 gait cycles per second; fetched by `scripts/fetch_unitree_rl_gym.sh`; same two shots | [G1_UNITREE.md](G1_UNITREE.md) |
| G1 on rough ground, Unitree's policy | the same blind, flat-trained policy on Isaac Lab's random rough terrain (`ROUGH_TERRAINS_CFG["random_rough"]`, one 8 x 8 m tile: bumps of 2 to 10 cm every 0.1 m; Isaac's only setting, difficulty does not change it), played by Unitree's loop on [my fork of MuJoCo Warp](https://github.com/pulipakaa24/mujoco_warp/tree/metalsim) (with the heightfield contact fixes); 10 to 12 of 16 robots fall within 8 s on the film's tile (CPU MuJoCo also 10 to 12), 75 % over five tiles (CPU MuJoCo 68 %), and the film shows a typical one, down at 5.3 s; outdoor HDRI (goegap) and CC0 ground texture (`scripts/fetch_polyhaven_assets.sh demo`); 8 s, same two shots, lower camera | [G1_BUMPS.md](G1_BUMPS.md) |
| SO-101 cube lift | so101Sim's SO-101 cube lift (the owner's project) on MuJoCo Warp, four films: its state-based PPO lift teacher (97-98/100 on MuJoCo Warp against 100/100 on so101Sim's CPU MuJoCo; the realism check finds the motion beyond the standard 7.4 V follower's servos), the same policy on a 7.4 V servo model (`so101_lift_74v`), a smoothness retrain (`so101_lift_smooth`), and so101Sim's scripted IK plan streamed at [LeRobot](https://github.com/huggingface/lerobot)'s 30 Hz to Rhoban [BAM](https://github.com/Rhoban/bam)'s identified 7.4 V servo model with load-dependent gearbox friction (`so101_lift_deploy`, 100/100 on MuJoCo Warp and on CPU MuJoCo); wooden table top at its real size; 6 s, 1280x720 hero and a 640x480 shot for the README, 64 spp | [SO101_LIFT.md](SO101_LIFT.md) |
| TRON1 wheel-legged drive, LimX's policy | LimX's TRON1 (WF_TRON1A) driven by LimX's own RL policy (the isaacgym ONNX export in LimX's [`tron1-rl-deploy-python`](https://github.com/limxdynamics/tron1-rl-deploy-python) at [`035e4c9`](https://github.com/limxdynamics/tron1-rl-deploy-python/commit/035e4c9e2f24e916fa5e8363ac6f0f788a2e1e8c), Apache-2.0, in `assets/tron1/`) in MetalSim's TRON1 simulation (`Tron1Sim`, CPU MuJoCo) at its nominal realism settings (latency, drive model, sensor noise, joint friction, sensor pack), unchanged; stood up for 3 s, then drive, turn, slow down, stop, each command change ramped over 0.5 s; 0 of 16 robots fell; industrial HDRI and concrete floor ([Poly Haven](https://polyhaven.com), `fetch_polyhaven_assets.sh heldout`); same two shots | [TRON1.md](TRON1.md) |

## The README's media

The animations at the top of the repository's `README.md` are in this directory, linked by relative path, as MuJoCo
Warp does. Every cell is a lossy animated WebP, 320x240, looping, written by
[Pillow](https://github.com/python-pillow/Pillow) at quality 70 in one call (MuJoCo Warp's `mujoco_warp/record.py`
recipe), through `python -m metalsim.render.film.encode webp`: a 4:3 crop of the source, Lanczos downscale, one `.json`
beside the output with the crop, window, frame count, PSNR against the source and the exact command. Those `.json`
files, the frame directories and the scratch work are in `runs/readme_media/`, which stays local (gitignored); the
commands are copied below. On 2026-10-31 every encode command below, and the chart's, was re-run from its source and
gave a byte-identical file (for the cells made before that day, `runs/readme_media/handoff/provenance-docs.md`); the
commands that record or render were not re-run. On 2026-11-01 the TRON1 and SO-101 cells were rendered again from their
recordings, after a film-tool fix: a look's colours (the ground texture's tint, `LookCfg.materials`) now replace a
geom's own `rgba`, which MuJoCo otherwise draws instead. Their encode commands were then run twice, with byte-identical
output. Paths are from the repository root.

No timing figure comes from these media. The G1, TRON1 and SO-101 films were recorded and rendered on the 14-inch
MacBook Pro (Mac16,6), not in timing runs (MetalSim's timing runs are made on the 16-inch MacBook Pro through
`scripts/gpu_run.sh <name> timing ...`); their docs give the power state each stage stamped. The lidar clip was made on
the 14-inch MacBook Pro on battery (the `pmset` log shows battery power at 02:14 on 2026-10-31, during its render); its
power-mode setting was not recorded. The chart's numbers are measurements from the logs it cites.

| README row | cells (left to right) |
|---|---|
| 1 | [G1 walk, Unitree's policy](#g1-walk-unitrees-policy-g1_unitreewebp), [G1 on rough ground](#g1-on-rough-ground-g1_bumpswebp), [ANYmal on sand](#anymal-c-on-sand-anymal_sandwebp) |
| 2 | [Franka-Pour](#franka-pour-franka_pourwebp), [TRON1](#tron1-tron1webp), [SO-101 cube lift](#so-101-cube-lift-deployment-style-expert-so101_lift_deploywebp) |
| 3 | [lidar through glass](#lidar-through-glass-lidar_glass_rgbwebp-lidar_glass_depthwebp-lidar_glass_lidarwebp) (three cells) |
| below | [throughput chart](#throughput-chart-throughput_darksvg-throughput_lightsvg) |

Earlier cells, kept with their media in this directory but no longer on the front page:
[G1 flat with NVIDIA's policy](#g1-flat-g1_flatwebp), [G1 rough with NVIDIA's policy](#g1-rough-g1_roughwebp) and
[the same Isaac Lab policy in Isaac Sim and MetalSim](#same-policy-in-isaac-sim-and-metalsim-parity_g1_isaacwebp-parity_g1_metalsimwebp).

### G1 walk, Unitree's policy (`g1_unitree.webp`)

- **Shows:** Unitree G1 walking on flat ground with Unitree's own pretrained walking policy (`unitree_rl_gym`'s
  `deploy/pre_train/g1/motion.pt`, BSD-3-Clause, the file Unitree's MuJoCo and real-robot deploy configs both load),
  played by Unitree's MuJoCo sim2sim loop ported to MuJoCo Warp, unchanged; 1.25 gait cycles per second (Unitree's
  0.8 s gait clock), feet landing flat. Path traced at tier 2 on a laminate floor at its real size (1.70 m per image,
  21 cm planks, about one G1 foot long); the camera tracks the pelvis and orbits at 10 deg/s. 0.5-5.5 s of a 6 s take,
  150 frames at 33 ms, 0.63 MB.
- **Source:** the `readme` shot of `runs/film/g1_unitree/` (`readme/frames`, local; `rec/rec.json` and
  `readme/render.json` tracked). The recording is the 15:52 re-take on [my fork of MuJoCo
  Warp](https://github.com/pulipakaa24/mujoco_warp/tree/metalsim) at
  [`8eb51fe`](https://github.com/pulipakaa24/mujoco_warp/commit/8eb51fe637437108185f5a4750b7f5f64ef26624); the frames
  are its 21:48 re-render at the floor's real size ([G1_UNITREE.md](G1_UNITREE.md), "This take";
  `runs/readme_media/handoff/revalidate.md`).
- **Reproduce:** Unitree's files first, `scripts/fetch_unitree_rl_gym.sh` (pinned commit, not in the repository). The
  take, as made on 2026-10-31 (GPU-queue jobs of kind render, then CPU stages):
  ```
  scripts/gpu_run.sh <name> render 15 -- .venv/bin/python -m metalsim.render.film g1_unitree --stage record
  scripts/gpu_run.sh <name> render 10 -- .venv/bin/python -m metalsim.render.film g1_unitree --stage render
  .venv/bin/python -m metalsim.render.film g1_unitree --stage encode --shots hero
  python -m metalsim.render.film.encode webp runs/film/g1_unitree/readme/frames docs/renders/g1_unitree.webp --start 0.5 --seconds 5 --src-fps 50 --fps 30
  .venv/bin/python -m metalsim.render.film g1_unitree --stage doc
  ```
  A new recording is not bitwise the same take (the GPU rollout is not repeatable run to run by default).
- **Provenance:** [G1_UNITREE.md](G1_UNITREE.md): the port against Unitree's own loop on CPU MuJoCo (pelvis within
  1.4-1.8 um at 5 s, `runs/film/g1_unitree/sim2sim_check/`), gait, camera, timings and the power state each stage
  stamped. Hero video and stills: [g1_unitree.mp4](g1_unitree.mp4), `g1_unitree_still{1,2,3}.jpg`.

### G1 on rough ground (`g1_bumps.webp`)

- **Shows:** Unitree G1 with Unitree's own pretrained walking policy (the G1 Unitree cell's: blind, trained on flat
  ground only) on Isaac Lab's random rough terrain (`ROUGH_TERRAINS_CFG["random_rough"]`, one 8 x 8 m tile, env seed
  0: bumps of 2 to 10 cm every 0.1 m), played by Unitree's MuJoCo sim2sim loop ported to MuJoCo Warp, unchanged.
  Nothing is tuned and the robot falls: over five tiles, three runs each, 180 of 240 robots fall within 8 s (75.0 %;
  CPU MuJoCo 68.3 to 68.8 %). In the take 12 of 16 fall, and the clip shows the typical outcome, the fallen robot
  nearest the median fall time: high, stumbling steps, then down at 5.34 s, falling backwards, the policy still acting
  on the ground. Path traced at tier 2 under Poly Haven's `goegap` (its sun as an analytic light), on
  `sandy_gravel_02` at its scanned 2.53 m tile; the camera, lower than the G1 films', tracks the pelvis and orbits at
  10 deg/s. 1.0-6.0 s of an 8 s take, 150 frames at 33 ms, 1.46 MB.
- **Source:** the `readme` shot of `runs/film/g1_bumps/` (`readme/frames`, local; `rec/rec.json` and
  `readme/render.json` tracked). The take ran on [my fork of MuJoCo
  Warp](https://github.com/pulipakaa24/mujoco_warp/tree/metalsim), which has the heightfield contact fixes; on its older
  commit [`8eb51fe`](https://github.com/pulipakaa24/mujoco_warp/commit/8eb51fe637437108185f5a4750b7f5f64ef26624) the
  sphere feet lose contacts in the creases between bumps and most robots keep walking (54 of 288 fell).
- **Reproduce:** `scripts/fetch_unitree_rl_gym.sh` and `scripts/fetch_polyhaven_assets.sh demo`, and [my fork of MuJoCo
  Warp](https://github.com/pulipakaa24/mujoco_warp/tree/metalsim) as MetalSim's install sets it up. The take, as made on 2026-10-31 (the record
  and render ran as one GPU-queue job, with a check of the filmed robot against the measurement runs between them;
  G1_BUMPS.md, "The take"):
  ```
  scripts/gpu_run.sh <name> render 15 -- .venv/bin/python -m metalsim.render.film g1_bumps --stage record
  scripts/gpu_run.sh <name> render 15 -- .venv/bin/python -m metalsim.render.film g1_bumps --stage render
  .venv/bin/python -m metalsim.render.film g1_bumps --stage encode --shots hero
  python -m metalsim.render.film.encode webp runs/film/g1_bumps/readme/frames docs/renders/g1_bumps.webp --start 1.0 --seconds 5 --src-fps 50 --fps 30
  .venv/bin/python -m metalsim.render.film g1_bumps --stage doc
  ```
  A new recording is another take (the GPU rollout is not repeatable run to run): how many robots fall and which one
  is filmed can change.
- **Provenance:** [G1_BUMPS.md](G1_BUMPS.md): the terrain, the contact check against the exact geometry, falls per
  bump height and over five tiles against CPU MuJoCo, how the robot was chosen, its gait, the look and the power state
  each stage stamped; the measurement scripts and their JSON in `runs/film/g1_bumps/measure/`. Hero video and stills:
  [g1_bumps.mp4](g1_bumps.mp4), `g1_bumps_still{1,2,3}.jpg`.

### ANYmal C on sand (`anymal_sand.webp`)

- **Shows:** ANYmal C walking on [Newton](https://github.com/newton-physics/newton) 1.5.2's `example_mpm_anymal` MPM
  sand with that example's walking policy; 669,372 sand particles plus 261,061 decoration particles drawn as 27 grains
  each, 25.1 M grain instances, path traced at 1920x1080, 128 spp (32 passes x 4),
  [OIDN](https://www.openimagedenoise.org). 3.0-8.0 s of the 8 s clip, 150 frames at 33 ms, 3.34 MB (above MuJoCo Warp's
  largest clip, 2.0 MB).
- **Source:** the clip's PNG frames 0090-0239, `~/metalsim-fresh/runs_render/sand25m/demo/frames` on the 14-inch,
  outside the repository (with `anymal_sand_25m_twolevel.mp4` and `log.txt` beside them). They were rendered on
  2026-10-13 by `runs_render/sand25m/demo.sh` (two-level grain structures, `ramp.py` driving
  `scripts/particle_demo/render_anymal.py`) from the simulation in `runs_render/sim1`.
- **Reproduce:** the clip: [`docs/DEMO.md`, "The 25 M-grain render"](../DEMO.md#the-25-m-grain-render-anymal-on-sand-at-demo-scale)
  (`scripts/particle_demo/sim_anymal.py` in the Newton venv, then `render_anymal.py --frames 0:240 --width 1920
  --height 1080 --passes 32 --grains 27`). This file, centred 1440x1080 crop
  (command from `runs/readme_media/assets/anymal_sand.webp.json`):
  `python -m metalsim.render.film.encode webp ~/metalsim-fresh/runs_render/sand25m/demo/frames docs/renders/anymal_sand.webp --start 3 --seconds 5`.
- **Provenance:** `metalsim/render/PARTICLES.md`, "Demo" (the 2026-10-12 result at these settings, timings, and the
  credits: [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) `anybotics_anymal_c` BSD-3, Poly
  Haven `goegap` HDRI and `coast_sand_01` texture CC0, Newton's example and policy Apache-2.0);
  `metalsim/physics/mpm/MPM.md` §8k for the coupled step against Isaac.

### Franka-Pour (`franka_pour.webp`)

- **Shows:** Isaac Lab 3.0-EA's Franka-Pour on metalsim (MuJoCo Warp arm coupled to implicit MPM, 735 MPM particles
  per cup drawn as 4 render grains each), a policy trained here; the whole 3.5 s hero shot, 0.29 MB.
- **Source:** [franka_pour.mp4](franka_pour.mp4), the hero shot's 1280x720 docs copy (H.264, CRF 22). This WebP is a
  second-generation cut of that mp4: the 1920x1080 frames and master stay on the 16-inch MacBook Pro that rendered them.
  Left-aligned 960x720 crop, which keeps both cups and the pour; the wrist is clipped by about 10 output pixels in the
  first few frames. libwebp merges near-identical frames, so the 105 source frames are stored as 83 (3.465 s).
- **Reproduce:**
  `python -m metalsim.render.film.encode webp docs/renders/franka_pour.mp4 docs/renders/franka_pour.webp --start 0 --seconds 0 --crop-x 0`;
  the film: `python -m metalsim.render.film franka_pour`.
- **Provenance:** [FRANKA_POUR.md](FRANKA_POUR.md) (16-inch MacBook Pro; checkpoint
  `runs/franka_pour/logs/m414_2048_model_299.pt`).

### TRON1 (`tron1.webp`)

- **Shows:** LimX's TRON1 wheel-legged robot (WF_TRON1A) driven by LimX's own open RL policy (the isaacgym ONNX export
  in LimX's `tron1-rl-deploy-python` at
  [`035e4c9`](https://github.com/limxdynamics/tron1-rl-deploy-python/commit/035e4c9e2f24e916fa5e8363ac6f0f788a2e1e8c),
  Apache-2.0, in `assets/tron1/limx_policy/`; not the closed controller of the robot's Remote Control Mode) in
  MetalSim's TRON1 simulation, `Tron1Sim` on CPU MuJoCo, at its nominal realism settings (command and sensor latency,
  quantized encoders, joint friction, IMU noise, a 1.5 kg sensor pack), unchanged: after a 3 s standing pre-roll, drive
  at 1.0 m/s, turn left, slow down, stop, each command change ramped over 0.5 s. 0 of 16 robots fell over 8 s; RMS
  tracking error 0.27 m/s and 0.14 rad/s; the filmed robot is the most typical tracker. The robot in its real colours
  (black; LimX's description files colour it white). Path traced at tier 2 under Poly Haven's `machine_shop_02`, on
  `concrete_floor_worn_001` at its real 3 m tile and in its own colours, at a grey card's exposure. 0.5-5.5 s of a 7 s
  take, 150 frames at 33 ms, 0.91 MB.
- **Source:** the `readme` shot of `runs/film/tron1/` (`readme/frames`, local; `rec/rec.json` and `readme/render.json`
  tracked).
- **Reproduce:** `pip install -e ".[tron1]"` (onnxruntime) and `scripts/fetch_polyhaven_assets.sh heldout`. The take,
  as made (recorded on 2026-10-31, these frames rendered on 2026-11-01):
  ```
  scripts/gpu_run.sh <name> render 15 -- .venv/bin/python -m metalsim.render.film tron1 --stage record
  scripts/gpu_run.sh <name> render 15 -- .venv/bin/python -m metalsim.render.film tron1 --stage render
  .venv/bin/python -m metalsim.render.film tron1 --stage encode --shots hero
  python -m metalsim.render.film.encode webp runs/film/tron1/readme/frames docs/renders/tron1.webp --start 0.5 --seconds 5 --src-fps 50 --fps 30
  .venv/bin/python -m metalsim.render.film tron1 --stage doc
  ```
  The physics is CPU MuJoCo, so a new recording gives the same take.
- **Provenance:** [TRON1.md](TRON1.md): LimX's files with their sha256, the command schedule, tracking per robot, the
  look and the power state each stage stamped. Hero video and stills: [tron1.mp4](tron1.mp4),
  `tron1_still{1,2,3}.jpg`.

### SO-101 cube lift, deployment-style expert (`so101_lift_deploy.webp`)

- **Shows:** the SO-101 arm grasping a 4 x 4 x 6 cm box and lifting it off a wooden table: so101Sim's scripted IK lift
  plan, streamed as absolute joint goals at LeRobot's 30 Hz (`metalsim/render/film/so101_expert.py`) to Rhoban BAM's
  identified model of the standard follower's 7.4 V servos with load-dependent gearbox friction (`7v4_bam`), on MuJoCo
  Warp. Scripted, not learned; the box pose comes from the simulator. 100/100 seeds lift on MuJoCo Warp and on CPU
  MuJoCo; the filmed seed, 10051, is typical (lifts at 2.26 s, no retry). The box tilts with the hand, which the lift
  pitches by about 36 deg, as real teleoperated lifts do; the grasp comes down 26-28 deg off vertical (real operators:
  4-13 deg) and the closing jaw tips the box on the table. Path traced at tier 2, the wooden table top in its
  texture's own colours at its real size. 0-5 s of a 6 s take, 150 frames at 33 ms, 0.69 MB.
- **Source:** the `readme` shot of `runs/film/so101_lift_deploy/` (`readme/frames`, local; `rec/rec.json` and
  `readme/render.json` tracked).
- **Reproduce:** the take, as made (recorded on 2026-10-31, these frames rendered on 2026-11-01; record and render as
  GPU-queue jobs of kind render):
  ```
  scripts/gpu_run.sh <name> render 20 -- .venv/bin/python -m metalsim.render.film so101_lift_deploy --stage record
  scripts/gpu_run.sh <name> render 20 -- .venv/bin/python -m metalsim.render.film so101_lift_deploy --stage render
  .venv/bin/python -m metalsim.render.film so101_lift_deploy --stage encode --shots hero
  python -m metalsim.render.film.encode webp runs/film/so101_lift_deploy/readme/frames docs/renders/so101_lift_deploy.webp --start 0 --seconds 5 --src-fps 50 --fps 30
  .venv/bin/python -m metalsim.render.film so101_lift --stage doc
  ```
  The four SO-101 films share one doc, written by the `so101_lift` doc stage.
- **Provenance:** [SO101_LIFT.md](SO101_LIFT.md#deployment-style-scripted-expert-on-7v4_bam) (the expert, the take
  against the 100-seed evaluation, the grasp and tilt measurements, the caveats) and its servo-model section (BAM's
  parameters and sources, the GPU friction update and its check against CPU MuJoCo). Hero video and stills:
  [so101_lift_deploy.mp4](so101_lift_deploy.mp4), `so101_lift_deploy_still{1,2,3}.jpg`.

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
  them. Rendered on the 14-inch MacBook Pro on battery (power-mode setting not recorded); its per-frame times are not
  timing figures.

### Throughput chart (`throughput_dark.svg`, `throughput_light.svg`)

- **Shows:** env-steps per second, like-for-like pairs only: G1 flat and rough env step at 4,096 envs and Franka-Pour's
  full PPO loop at 1,024 envs, MetalSim on the M4 Max against Isaac Lab 3.0 (Newton / MuJoCo Warp and PhysX) on an
  NVIDIA L4 GPU. Dark and light variants for a `<picture>` element, full width under the grid.
- **Source and reproduce:** `.venv/bin/python scripts/readme_media/throughput_chart.py --out docs/renders` (standard
  library only; default output `runs/readme_media/assets/`).
- **Provenance:** every number sits in the script beside its doc section (`docs/PARITY.md` §1.4, §1.8, §1b.4) and raw
  log; its docstring lists what is not drawn and why (NVIDIA's RTX 4090 figure and Isaac's rough PhysX number, which
  have no matching MetalSim measurement).

## Earlier README media

No longer on the front page; the files stay in this directory.

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

### Same policy in Isaac Sim and MetalSim (`parity_g1_isaac.webp`, `parity_g1_metalsim.webp`)

- **Shows:** Isaac Lab's own rsl_rl G1 flat policy at iteration 1499 (trained in Isaac Lab with PhysX), rolled out from
  the same start (Isaac's default state, command 0.5 m/s forward) under the same camera and lights: left Isaac Sim 5.1,
  PhysX + RTX real-time, recorded on an NVIDIA L4 GPU; right MetalSim, MuJoCo Warp at 2.5 ms + tier 2. Physics and
  renderer both differ, so the two cells drift apart; they are not the same frames rendered twice. 0.5-4.5 s of the 8 s
  video, 100 frames at 40 ms (25 fps, the source's rate), 0.21 / 0.28 MB.
- **Source:** [`docs/gallery/g1_stage_isaac_it1499.mp4`](../gallery/g1_stage_isaac_it1499.mp4) (2048x576, two 1024x576
  panels), made by `metalsim.parity.side_by_side` (`scripts/gallery/g1_stage_videos.sh`; re-rendered with the RTX-parity
  preset on 2026-09-25,
  [`a14b6a87`](https://github.com/pulipakaa24/MetalSim/commit/a14b6a8736baf4030960f195810063bb4ed2eb03)). The Isaac
  recording and checkpoint (`runs/parity/isaac/parity_out/play/isaac_it1499`,
  `runs/parity/isaac/ckpt_out/model_1499.pt`) are not in the repository, so the video is the source.
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
  `docs/research/readme_media_plan_2026-10-31.md` §4b, the next session on the L4 GPU): one rollout recorded on the
  Isaac side and replayed in MetalSim tier 2 from Isaac's recorded poses (`metalsim/render/rtx_parity.py`, the §1.7
  protocol), so that the two cells differ only in rendering.

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

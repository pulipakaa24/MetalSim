"""Film entries: one declarative ``FilmTask`` per task (camera, envs, look, assets). The only code a task adds is its
recorder adapter (``adapters.py``: how to build the env and load the policy).

    python -m metalsim.render.film --list
"""
from __future__ import annotations

from metalsim.render.film.cfg import AssetRef, FilmCfg, FilmTask, LookCfg, ShotCfg, ViewerCfg

ISAAC61 = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.1/Isaac"
NVIDIA_TERMS = "NVIDIA Isaac Sim asset terms (fetched, not redistributed)"
PH_CC0 = "CC0 1.0 (Poly Haven; fetched, not committed)"
FETCH_ISAAC = "scripts/fetch_isaac_assets.sh"
FETCH_PH = "scripts/fetch_polyhaven_assets.sh train"

# Isaac Lab's Kit perspective camera: focal length 18.147 mm, horizontal aperture 20.955 mm -> 60 deg horizontal,
# 36 deg vertical at 16:9 (isaaclab.sim.PinholeCameraCfg defaults / Kit's /OmniverseKit_Persp)
KIT_FOVY_16_9 = 36.0

FRANKA_POUR = FilmTask(
    name="franka_pour",
    adapter="metalsim.render.film.adapters:FrankaPour",
    venv="newton152",
    checkpoint="runs/franka_pour/logs/m414_2048_model_299.pt",
    title="Franka-Pour",
    summary=("A Franka Panda grasps a cup holding 735 MPM particles of granular media and pours them into a second "
             "cup: Isaac Lab 3.0-EA's `IsaacContrib-Franka-Pour` task on metalsim (Newton 1.5.2 coupled MuJoCo Warp + "
             "implicit MPM on Metal), driven by a policy trained here with Isaac's PPO configuration."),
    notes="""
**The policy.** Trained on metalsim on the 14-inch M4 Max: 2,048 envs, 300 iterations of the task's rsl_rl PPO
configuration (96 steps per env per iteration, MLP 512-256-128-64, learning rate 1.5e-4; 59.0 M env steps, about
16.4 h at ~945 env-steps/s; `runs/franka_pour/logs/m414_2048_config.json`, `m414_2048_log.jsonl`). Logged training
success over the last iteration's 1,704 episodes: 63 %. Evaluated over 2,048 first episodes (seed 7, uniform reset
rows as in Isaac's play mode, the actor's mean action): **47.7 % success**, 82.2 % for episodes that start with the
cup already grasped and 15.2 % for those that start before the grasp
(`runs/franka_pour/eval/eval_2048_isaac_budget.json`). Learning against Isaac Lab on the L4 at matched settings:
`docs/PARITY.md` §1b.5. The film simulates 24 envs with the evaluation's seed and films the best-ranked: successful
episodes that start before the grasp first (approach, grasp, lift and pour in one take), then successful ones that
start with the cup in hand.

**How Isaac Lab presents the task** (research, `upstream/IsaacLab3` at `ae37b028`). `pour_env_cfg.py` sets a default
viewer at eye (1.4, 1.4, 0.9) looking at (0.5, 0, 0.1) in env 0's frame, and its play mode (one env) a closer camera
at eye (0.9, 0.65, 0.5) with the Newton RTX visualizer drawing the media as particles of the media's `visual_color`
(0.85, 0.72, 0.45). Videos come from `VideoRecorderCfg` at `render_fps` = 1 / step_dt = 30 fps. The scene is lit by a
grey dome light (colour 0.75, intensity 3000). The hero shot here uses the play-mode camera, the grid shot the default
viewer, both with Kit's default perspective lens (36 deg vertical at 16:9) and one frame per policy step (30 fps).

**Rendering.** As the ANYmal-on-sand demo (`scripts/particle_demo/`): simulate first, render offline. Tier 2 (the path
tracer) is the best tier that draws MPM particles with full lighting (tier 1 draws them without ray-traced shadows,
`docs/RENDER_COVERAGE.md` §1c). The physics model loads colliders only, as Isaac's Newton path does, so the render
twin is built by the same scene builder with Isaac's USD visual meshes loaded (Newton 1.5.2's USD importer) and its
bodies matched one to one to the simulated world's; a body that has visual meshes hides its colliders. Each MPM
particle (radius 2.5 mm) is drawn as 4 render grains (`metalsim/render/PARTICLES.md`). Lighting is a Poly Haven studio
HDRI as an importance-sampled dome instead of Isaac's plain grey dome (it gives the metal and plastic reflections);
the floor texture sits on Isaac's ground plane 1.05 m below the tabletop. OIDN denoiser, ACES tone map, sRGB.
The tool is generic: `python -m metalsim.render.film <task>`; a task adds a `FilmTask` entry and a small recorder
adapter (`metalsim/render/film/`).
""",
    # Isaac Lab's play-mode camera for this task (pour_env_cfg.py:821-833: NewtonRTXVisualizerCfg eye (0.9, 0.65, 0.5),
    # lookat (0.5, 0, 0.1), env frame)
    viewer=ViewerCfg(eye=(0.9, 0.65, 0.5), lookat=(0.5, 0.0, 0.1), origin_type="env", fovy=KIT_FOVY_16_9,
                     resolution=(1920, 1080)),
    film=FilmCfg(num_envs=24, envs="auto", grid=1, seed=7, max_seconds=12.0, hold_s=1.0, passes=64, spp=4, bounces=4,
                 output_filename_prefix="franka_pour"),
    look=LookCfg(hdri="hdri/train/poly_haven_studio.hdr", hdri_yaw_deg=0.0, sun=False,
                 ground_texture="tex/train/laminate_floor_02.png", ground_repeat=1.0,
                 # Isaac's media colour (media.py: MPMGridCfg visual_color (0.85, 0.72, 0.45)) with a little spread
                 particle_palette=((0.85, 0.72, 0.45, 0.6), (0.74, 0.60, 0.36, 0.25), (0.55, 0.43, 0.27, 0.15)),
                 grains_per_particle=4, grain_scale=1.2,
                 materials={r"bowl": dict(rgba=(0.86, 0.88, 0.90, 1.0), roughness=0.35, specular=0.5)}),
    assets=[
        AssetRef("Franka Emika Panda (USD, Isaac 6.1 asset root; converted by NVIDIA from MuJoCo Menagerie's mjx_panda)",
                 "NVIDIA Isaac Sim assets", f"{ISAAC61}/IsaacLab/Robots/FrankaEmika/franka_panda.usda", NVIDIA_TERMS,
                 FETCH_ISAAC),
        AssetRef("Panda description the USD derives from (MuJoCo Menagerie franka_emika_panda)", "Google DeepMind",
                 "https://github.com/google-deepmind/mujoco_menagerie/tree/main/franka_emika_panda", "Apache-2.0", ""),
        AssetRef("Source and receiver cups (franka_pour_cups.usda, variants source / target)", "NVIDIA Isaac Lab 3.0-EA",
                 f"{ISAAC61}/IsaacLab/Contrib/MPM/Pour/franka_pour_cups.usda", NVIDIA_TERMS, FETCH_ISAAC),
        AssetRef("SeattleLab table (table_instanceable.usd)", "NVIDIA Isaac Sim assets",
                 f"{ISAAC61}/Props/Mounts/SeattleLabTable/table_instanceable.usd", NVIDIA_TERMS, FETCH_ISAAC),
        AssetRef("Reset dataset (20,000 start states, reset_dataset.pt)", "NVIDIA Isaac Lab 3.0-EA",
                 f"{ISAAC61}/IsaacLab/Contrib/MPM/Pour/reset_dataset.pt", NVIDIA_TERMS, FETCH_ISAAC),
        AssetRef("Studio HDRI (poly_haven_studio, 2k)", "Poly Haven", "https://polyhaven.com/a/poly_haven_studio", PH_CC0,
                 FETCH_PH),
        AssetRef("Floor texture (laminate_floor_02, 1k diffuse)", "Poly Haven", "https://polyhaven.com/a/laminate_floor_02",
                 PH_CC0, FETCH_PH),
        AssetRef("Granular media: 735 MPM particles per cup, density 1500 kg/m^3, friction 0.7, 5 mm lattice; render "
                 "grains coloured with Isaac's visual_color (0.85, 0.72, 0.45)", "Isaac Lab 3.0-EA task config "
                 "(pour_env_cfg.py:97-100, 152-156; media.py)", "https://github.com/isaac-sim/IsaacLab", "BSD-3-Clause",
                 "procedural (metalsim.learn.franka_pour_scene.media_lattice)"),
        AssetRef("Policy checkpoint m414_2048_model_299.pt (trained on metalsim)", "this repository",
                 "runs/franka_pour/logs/m414_2048_model_299.pt", "this repository's licence", "committed"),
    ],
    # hero: one env from Isaac's play-mode camera; grid: four envs from Isaac's default viewer for the task
    # (pour_env_cfg.py:603-608: KitVisualizerCfg eye (1.4, 1.4, 0.9), lookat (0.5, 0, 0.1), origin_type "env")
    shots=[ShotCfg("hero", 1), ShotCfg("grid", 2, ViewerCfg(eye=(1.4, 1.4, 0.9), lookat=(0.5, 0.0, 0.1),
                                                            origin_type="env", fovy=KIT_FOVY_16_9,
                                                            resolution=(1920, 1080)))],
    doc="docs/renders/FRANKA_POUR.md",
)

G1_ASSETS = [
    AssetRef("Unitree G1 (g1_minimal.usd, Isaac 5.1 asset root)", "NVIDIA Isaac Lab assets",
             "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/IsaacLab/Robots/"
             "Unitree/G1/g1_minimal.usd", NVIDIA_TERMS, FETCH_ISAAC),
    AssetRef("Studio HDRI (poly_haven_studio, 2k)", "Poly Haven", "https://polyhaven.com/a/poly_haven_studio", PH_CC0,
             FETCH_PH),
    AssetRef("Floor texture (laminate_floor_02, 1k diffuse)", "Poly Haven", "https://polyhaven.com/a/laminate_floor_02",
             PH_CC0, FETCH_PH),
]
# the G1 film camera: follows the pelvis (Isaac's origin_type "asset_body"), Kit's lens, 1280 x 720
G1_VIEWER = ViewerCfg(eye=(2.6, -2.0, 1.3), lookat=(0.0, 0.0, -0.05), origin_type="asset_body", body_name="pelvis",
                      fovy=KIT_FOVY_16_9, resolution=(1280, 720), smooth_s=1.0)
# README grid cell (4:3, shown at 320 x 240): the hero camera's direction, fov and pelvis tracking, 0.75 x closer so the
# robot fills ~3/4 of the frame height (hero: ~0.56 at 3.55 m with a 36 deg vertical fov)
G1_README_VIEWER = ViewerCfg(eye=(1.95, -1.5, 0.98), lookat=(0.0, 0.0, -0.05), origin_type="asset_body", body_name="pelvis",
                             fovy=KIT_FOVY_16_9, resolution=(640, 480), smooth_s=1.0,
                             orbit_deg_s=10.0)   # MuJoCo Warp's G1 clip: 0.05 deg per 5 ms step
G1_SHOTS = [ShotCfg("hero", 1), ShotCfg("readme", 1, G1_README_VIEWER)]
# Task settings the g1regress checkpoints do not save, from their training log headers
# (runs/g1regress/ckpt14/{flat,rough}_main_s0.log: "physics dt 0.0025 (decimation 8)"); the rest (reward_cfg,
# contact_cfg, terrain collision, ...) are read from the checkpoint by the adapter.
G1_TRAIN_TASK_KW = {"physics_dt": 0.0025}

# Doc prose shared by the two G1 films (FilmTask.notes; the doc stage prints it under "How it was made")
G1_CAMERA_NOTES = """
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
"""

G1_RENDER_NOTES = """
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
"""


def _g1_take_notes(name: str) -> str:
    """The README WebP and the timings paragraph of one G1 take (``name``: the task and its media prefix)."""
    return f"""
**README clip.** [{name}.webp]({name}.webp) is the `readme` shot from 0.5 s to 5.5 s (frames 25 to 274), resampled
from 50 to 30 fps and written as MuJoCo Warp writes its README clips: 320x240 (Lanczos), lossy animated WebP from
Pillow at quality 70, 33 ms per frame, looping; 150 frames. Command (re-run on 2026-10-31 from `runs/film/{name}`, byte
for byte the same file):
`python -m metalsim.render.film.encode webp runs/film/{name}/readme/frames docs/renders/{name}.webp --start 0.5
--seconds 5 --src-fps 50 --fps 30`. The film tool's route, `python -m metalsim.render.film {name} --stage encode
--shots readme --webp --webp-window 0.5:5`, uses the same encoder but names its output `{name}_readme.webp` and also
writes a `{name}_readme.mp4` and a still.

**This take, and where its timings come from.** Recorded, rendered and encoded on 2026-10-31 on the 14-inch M4 Max
(Mac16,6) **on battery in Low Power Mode**, at metalsim `687f3d7b` plus the uncommitted film-tool fixes (hence
`+dirty` in the pins). The Performance table above records how long each stage took there. It is not a timing figure:
quote nothing on this page as MetalSim's speed. MetalSim's throughput figures are in `docs/PARITY.md` §1.4 and §1.8
(16-inch, on power). Record and render ran as GPU-queue jobs of kind render, one at a time: `--stage record` into
`runs/readme_media/film/{name}`, then `--stage render` for both shots there. The take was then moved to
`runs/film/{name}` (the tool's default `--out`), where `--stage encode --shots hero` and `--stage doc` ran (CPU).
Nothing was re-recorded. A new recording is another take of the same policy, not this one: the GPU rollout is not
bitwise repeatable, so which env is filmed, and its path, can change.
"""


G1_FLAT = FilmTask(
    name="g1_flat",
    adapter="metalsim.render.film.adapters:G1Flat",
    venv="main",
    # trained on main 2026-10-13 (500 iterations, docs/research/ppowarp_checkpoint_load_2026-10-13.md; log beside it)
    checkpoint="runs/g1regress/ckpt14/flat_main_s0.pt",
    adapter_kw={"task": G1_TRAIN_TASK_KW},
    title="G1 flat-terrain walk",
    summary=("Unitree G1 walking on flat ground: Isaac Lab's G1 flat velocity task on metalsim (MuJoCo Warp on Metal), "
             "driven by a policy trained here with the task's PPO settings. This take was filmed on the 14-inch M4 Max "
             "on battery in Low Power Mode, so its Performance table is not a timing figure."),
    notes="""
**The policy.** `runs/g1regress/ckpt14/flat_main_s0.pt`, sha256
`6fbfbfe8b950ab27e82afc912fbc3d5484c5b5050dc288cdecd32350b9917057` (675,619 bytes; the record stage writes the same
hash into `rec.json`). Trained on main on 2026-10-13 on the 14-inch M4 Max with the documented G1 flat run's
settings (`runs/ellip_default/run.sh`: `python -m metalsim.learn.g1_velocity 4096 flat train 500 <log> <checkpoint>
0.0025 --reward_cfg flat --seed 0`): 4,096 envs, physics dt 2.5 ms with decimation 8 (50 Hz control), rollouts of 24
steps, 500 iterations (49.2 M env steps), actor MLP 256-128-128 on a 123-dim observation, the elliptic contact
default (`contact_cfg="recommended"`). Training log: `runs/g1regress/ckpt14/flat_main_s0.log` (its header has every
setting; iteration 500: return +18.64, episode length 996 of 1,000 control steps).
`docs/research/ppowarp_checkpoint_load_2026-10-13.md` compares the curve with the three documented elliptic-default
seeds of 2026-09-26 (`runs/ellip_default/g1_flat_ellip_default_s{0,1,2}.log`, `docs/PARITY.md` §1.5: +17.0 to +17.4
at iteration 500) and replays the policy on the 16-inch: 1,012 of 1,024 episodes run to the 20 s time-out, 17 falls
over 1,200 control steps (`runs/g1regress/replay_main_flat_it500.log`; the replay checks that the loaded weights are
the ones that run, after the PPOWarp load bug that note fixes; the film's adapter evaluates its own torch module and
was never affected). The file is the training run's output, copied unchanged (the log's last line names it) and
tracked so that a fresh clone can film. It replaces `runs/policies/g1_flat_1500.pt`, the earlier default, which has
no training log and falls in 16 of 16 envs within 8 s (`docs/research/readme_media_plan_2026-10-31.md` section 2,
defect 1).

**Task settings.** `metalsim.learn.g1_velocity.G1VelocityTask(16, terrain="flat", seed=3)` with the settings saved
in the checkpoint (`reward_cfg="flat"`, `contact_cfg="recommended"`, base and feet-slide velocities at the COM, no
Isaac Lab 3 events; `rec.json` `adapter.saved_task_settings`) plus the one it does not
save, physics dt 2.5 ms, from the training log's header (`G1_TRAIN_TASK_KW`). `rec.json` `adapter.task_resolved`
lists what ran. The policy acts with its actor's mean. Commands are the task's own, as in training (Isaac's uniform
velocity command with a heading target, resampled every 10 s). An episode ends on the task's terminations (torso
contact above 1 N, a non-finite state) or its 20 s time-out. The film is 6 s: 301 frames, one per 20 ms policy step
(50 fps).

**How the env was chosen.** 16 envs are simulated, and `envs="auto"` lets the adapter rank them (`G1Flat.rank`):
first the envs whose first episode was not ended by a termination, then those whose pelvis never dropped more than
0.25 m below its start (a robot down without torso contact, which the task does not terminate), then the longest
path. The record stage refuses to film an env whose first episode terminated unless `--allow-early-end` is given.
In this take none of the 16 terminated, the largest pelvis drop was 0.11 m, and env 6, filmed, walked the longest
path: 5.61 m in 6 s (`rec.json` `adapter.first_episode`). Under "What is shown", `base_z`, `level` and `col` are
written only when an episode ends; with `end_frame` -1 (still running when the film ends) their zeros are
placeholders. That list is the top four of the ranking (the record keeps spares); both shots film one env, the first
(`render.json` `envs`).
""" + G1_CAMERA_NOTES + G1_RENDER_NOTES + _g1_take_notes("g1_flat"),
    viewer=G1_VIEWER,
    # 16 envs, "auto": the longest walk among envs that neither terminated nor went down (G1Flat.rank)
    film=FilmCfg(num_envs=16, envs="auto", grid=1, seed=3, max_seconds=6.0, hold_s=0.0, passes=16, spp=4, bounces=3,
                 output_filename_prefix="g1_flat"),
    look=LookCfg(hdri="hdri/train/poly_haven_studio.hdr", sun=False, ground_texture="tex/train/laminate_floor_02.png",
                 ground_geoms=("ground", "floor", "ground_plane"), ground_repeat=0.5),
    assets=G1_ASSETS + [
        AssetRef("Policy checkpoint flat_main_s0.pt (trained on metalsim, 500 iterations; sha256 6fbfbfe8...)",
                 "this repository", "runs/g1regress/ckpt14/flat_main_s0.pt", "this repository's licence", "committed"),
    ],
    shots=G1_SHOTS,
    doc="docs/renders/G1_FLAT.md",
)

G1_ROUGH = FilmTask(
    name="g1_rough",
    adapter="metalsim.render.film.adapters:G1Rough",
    venv="main",
    # trained on main 2026-10-13 (1000 iterations, terrain level 5.86; log beside it). Seed 0 = the training run's
    # task seed, so the film's terrain is the one the policy was trained on.
    checkpoint="runs/g1regress/ckpt14/rough_main_s0.pt",
    # level 5 of 10: inverted-stairs risers 0.14-0.16 m, inverted-slope grade 0.21-0.22 on this terrain; the training
    # curriculum ended at mean level 5.86, so the policy trained on this difficulty
    adapter_kw={"task": G1_TRAIN_TASK_KW, "terrain_level": 5},
    title="G1 rough-terrain walk",
    summary=("Unitree G1 climbing Isaac Lab's rough terrain: the G1 rough velocity task (stairs, slopes, boxes, height "
             "scan) on metalsim (MuJoCo Warp on Metal), driven by a policy trained here with the task's PPO settings. "
             "This take was filmed on the 14-inch M4 Max on battery in Low Power Mode, so its Performance table is not "
             "a timing figure."),
    notes="""
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
""" + G1_CAMERA_NOTES + G1_RENDER_NOTES + _g1_take_notes("g1_rough"),
    viewer=G1_VIEWER,
    # 40 envs: two per terrain type (Isaac's column assignment); "auto" films the one that climbed most without
    # terminating (G1Rough.rank)
    film=FilmCfg(num_envs=40, envs="auto", grid=1, seed=0, max_seconds=6.0, hold_s=0.0, passes=16, spp=4, bounces=3,
                 output_filename_prefix="g1_rough"),
    look=LookCfg(hdri="hdri/train/poly_haven_studio.hdr", sun=False, ground_texture="tex/train/laminate_floor_02.png",
                 ground_geoms=("ground", "terrain_boxes"), ground_repeat=0.5),
    assets=G1_ASSETS + [
        AssetRef("Rough terrain: Isaac Lab 2.3.2 ROUGH_TERRAINS_CFG (10 x 20 sub-terrains of 8 m), generated from the "
                 "task seed", "Isaac Lab task config (port: metalsim.learn.isaac_terrain)",
                 "https://github.com/isaac-sim/IsaacLab", "BSD-3-Clause", "procedural (metalsim.learn.terrain)"),
        AssetRef("Policy checkpoint rough_main_s0.pt (trained on metalsim, 1,000 iterations; sha256 b5b25ad3...)",
                 "this repository", "runs/g1regress/ckpt14/rough_main_s0.pt", "this repository's licence", "committed"),
    ],
    shots=G1_SHOTS,
    doc="docs/renders/G1_ROUGH.md",
)


TASKS = {t.name: t for t in (FRANKA_POUR, G1_FLAT, G1_ROUGH)}


def get(name: str) -> FilmTask:
    if name not in TASKS:
        raise KeyError(f"no film entry {name!r}; known: {sorted(TASKS)}")
    return TASKS[name]

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
ISAAC51 = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac"

# Isaac Lab 2.3's published rsl_rl checkpoints (isaaclab_rl/utils/pretrained_checkpoint.py,
# get_published_pretrained_checkpoint_path: ISAACLAB_NUCLEUS_DIR/PretrainedCheckpoints/rsl_rl/<task id>/checkpoint.pt,
# under the Isaac 5.1 asset root). NVIDIA's files under NVIDIA's terms: fetched by scripts/fetch_isaac_assets.sh to
# ``path`` (SHA-256 checked, as fetched on 2026-10-31), never committed. Both: rsl_rl model_state_dict (actor.*,
# critic.*, std), optimizer_state_dict, iter, infos (None); no observation normaliser.
ISAAC_PRETRAINED = {
    tid: dict(path=f"assets/isaac/pretrained/rsl_rl/{tid}/checkpoint.pt",
              url=f"{ISAAC51}/IsaacLab/PretrainedCheckpoints/rsl_rl/{tid}/checkpoint.pt", sha256=sha, iteration=it,
              actor=actor)
    for tid, sha, it, actor in (
        ("Isaac-Velocity-Flat-G1-v0", "fdf0f242bb8f3bcfc1aa9b7ff2761af4c014cf1cdc7e49242c9b92f3581720e7", 1499,
         "123 -> 256-128-128 -> 37 (ELU)"),
        ("Isaac-Velocity-Rough-G1-v0", "d85a9adea70e5476675f130cbc70f41bdef7a28d52f443fae902afc0e2358851", 2999,
         "310 -> 512-256-128 -> 37 (ELU)"))}
FETCH_PH = "scripts/fetch_polyhaven_assets.sh train"
# Ground textures at their real size (LookCfg.ground_tile_m): the edge in metres of the floor each image shows, Poly
# Haven's "dimensions" in mm (https://api.polyhaven.com/info/<id>, read 2026-10-31)
LAMINATE_TILE_M = 1.70           # laminate_floor_02: 1700 x 1700 mm, eight planks of 21 cm across the image
SANDY_GRAVEL_TILE_M = 2.53       # sandy_gravel_02: 2530 x 2530 mm
# The G1 films' floor before ground_tile_m (ground_repeat 0.5: one image per 2 / 0.5 = 4.0 m;
# runs/readme_media/handoff/ground-scale.md, "Per film"), which the G1 Unitree page's re-render replaced
LAMINATE_OLD_G1_TILE_M = 4.0


def _tile(take, real_m):
    """(the floor tile of the take's frames in metres, whether it is the texture's real size ``real_m``), the tile read
    from the take's render logs (``doc.Take.ground_tile_m``); (None, False) when the logs do not show it, as for a
    take rendered before ``ground_tile_m`` existed, at another tile. Notes state a floor size only from this."""
    t = take.ground_tile_m
    return t, t is not None and abs(t - real_m) < 1e-9


# Unitree's unitree_rl_gym (BSD-3-Clause), fetched by scripts/fetch_unitree_rl_gym.sh at ``commit`` (main on
# 2026-10-31) into ``root`` (git-ignored, never committed): the pretrained G1 walking policy Unitree plays in MuJoCo
# (deploy/pre_train/g1/motion.pt, TorchScript; sha256 as fetched), its sim2sim playback and config, the G1 model.
UNITREE_REPO = "https://github.com/unitreerobotics/unitree_rl_gym"
UNITREE_COMMIT = "276801e46c5d433564f24658bac64f254b7d2d4b"
UNITREE_RL_GYM = dict(
    root="assets/unitree_rl_gym", repo=UNITREE_REPO, commit=UNITREE_COMMIT, licence="BSD-3-Clause",
    fetch="scripts/fetch_unitree_rl_gym.sh",
    policy="deploy/pre_train/g1/motion.pt",
    policy_url=f"{UNITREE_REPO}/blob/{UNITREE_COMMIT}/deploy/pre_train/g1/motion.pt",
    policy_sha256="cf668f75b90d1abf73d2b87612a6e76bccc61ff7e083b63582d3f6aaa3c1759d",
    config="deploy/deploy_mujoco/configs/g1.yaml", script="deploy/deploy_mujoco/deploy_mujoco.py",
    scene="resources/robots/g1_description/scene.xml")

# Isaac Lab's Kit perspective camera: focal length 18.147 mm, horizontal aperture 20.955 mm -> 60 deg horizontal,
# 36 deg vertical at 16:9 (isaaclab.sim.PinholeCameraCfg defaults / Kit's /OmniverseKit_Persp)
KIT_FOVY_16_9 = 36.0


def _franka_pour_notes(take) -> str:
    t, real = _tile(take, LAMINATE_TILE_M)
    floor = (f" (Poly Haven's `laminate_floor_02` at {'its real size, ' if real else ''}{t:.2f} m per image; "
             "`ground_tile_m`) sits on\n" if t else " sits on ")
    return f"""
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
the floor texture{floor}Isaac's ground plane 1.05 m below the tabletop. OIDN denoiser, ACES tone map, sRGB.
The tool is generic: `python -m metalsim.render.film <task>`; a task adds a `FilmTask` entry and a small recorder
adapter (`metalsim/render/film/`).
"""


FRANKA_POUR = FilmTask(
    name="franka_pour",
    adapter="metalsim.render.film.adapters:FrankaPour",
    venv="newton152",
    checkpoint="runs/franka_pour/logs/m414_2048_model_299.pt",
    title="Franka-Pour",
    summary=("A Franka Panda grasps a cup holding 735 MPM particles of granular media and pours them into a second "
             "cup: Isaac Lab 3.0-EA's `IsaacContrib-Franka-Pour` task on metalsim (Newton 1.5.2 coupled MuJoCo Warp + "
             "implicit MPM on Metal), driven by a policy trained here with Isaac's PPO configuration."),
    notes=_franka_pour_notes,
    # Isaac Lab's play-mode camera for this task (pour_env_cfg.py:821-833: NewtonRTXVisualizerCfg eye (0.9, 0.65, 0.5),
    # lookat (0.5, 0, 0.1), env frame)
    viewer=ViewerCfg(eye=(0.9, 0.65, 0.5), lookat=(0.5, 0.0, 0.1), origin_type="env", fovy=KIT_FOVY_16_9,
                     resolution=(1920, 1080)),
    film=FilmCfg(num_envs=24, envs="auto", grid=1, seed=7, max_seconds=12.0, hold_s=1.0, passes=64, spp=4, bounces=4,
                 output_filename_prefix="franka_pour"),
    look=LookCfg(hdri="hdri/train/poly_haven_studio.hdr", hdri_yaw_deg=0.0, sun=False,
                 ground_texture="tex/train/laminate_floor_02.png", ground_tile_m=LAMINATE_TILE_M,
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
# Task settings a checkpoint does not save, applied on top of the ones it implies (adapters.G1Velocity):
# physics dt 2.5 ms (decimation 8, 50 Hz control). Isaac's published checkpoints (trained at PhysX's 5 ms x 4): the
# setting MetalSim plays Isaac's G1 checkpoints at, the default physics_dt of metalsim/parity/side_by_side.py and
# scripts/diagnostics/g1_rough_transfer.py (docs/PARITY.md §1.5: the smaller step keeps the kp-200 drives stable).
# Our g1regress checkpoints (--checkpoint runs/g1regress/ckpt14/*.pt): their training setting, from their log
# headers (runs/g1regress/ckpt14/{flat,rough}_main_s0.log: "physics dt 0.0025 (decimation 8)"); the rest of their
# settings (reward_cfg, contact_cfg, terrain collision, ...) are read from the checkpoint by the adapter.
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


def _g1_render_notes(take) -> str:
    """The G1 films' "Rendering" paragraph; the floor's size as the take's render logs show it (``_tile``)."""
    t, real = _tile(take, LAMINATE_TILE_M)
    if t is None:
        size = ""
    elif real:            # the laminate image is eight planks across
        size = (f", at its real size (Poly Haven's `laminate_floor_02`: {t:.2f} m per image,\neight {t / 8 * 100:.0f} "
                "cm planks; `ground_tile_m`)")
    else:
        size = f", at {t:.2f} m per image (`ground_tile_m`; `laminate_floor_02` is {LAMINATE_TILE_M:.2f} m across)"
    return f"""
**Rendering.** Tier 2 (the path tracer), 64 samples per pixel (16 passes x 4), 3 bounces, Open Image Denoise, ACES
tone map, sRGB; lit by a Poly Haven studio HDRI as an importance-sampled dome, with a laminate floor texture on the
task's ground instead of Isaac's grey ground{size}. The render twin is the task's own G1 model with Isaac's visual meshes
(`build_g1_model(visuals=True)`), posed per frame from the recorded joint state. On bodies that have visual meshes the
colliders are hidden (the three `*Cube` boxes, `rec.json` `twin.visibility`), and the touch-sensor sites are hidden:
one of them, `torso_link_touch`, was the grey box of the earlier G1 film. The ray tracer's env cell is sized up front
from the recorded geometry and the whole camera path, so the floor is drawn complete to the horizon (`render.json`
`env_grid.growths` 0). Each `render.json` carries a `cache_key`, a hash of the recording and of every setting that
changes a pixel, so a re-recording into the same directory re-renders rather than reusing old frames
(`docs/research/readme_media_plan_2026-10-31.md` section 2, defects 2, 3 and 5).
"""


def _g1_take_notes(take, name: str, uncommitted: str) -> str:
    """The README WebP and the timings paragraph of one G1 take (``name``: the task and its media prefix;
    ``uncommitted``: what the take's uncommitted changes were). Day, machine, commit and power come from the take's
    stage logs."""
    this_take = take.fill(f"""
**This take, and where its timings come from.** Recorded, rendered and encoded on {take.day} on the {take.machine},
at metalsim {take.metalsim_at(uncommitted)}. This was not a timing run: MetalSim's timing runs are made on the 16-inch
through the GPU queue's timing kind (`scripts/gpu_run.sh <name> timing <minutes> -- <command>`, which waits for an idle
device), and its throughput figures are in `docs/PARITY.md` §1.4 and §1.8. The Performance table above records how
long each stage took here; quote nothing on this page as MetalSim's speed. Power, as the stages stamped it (`pmset -g
batt` and `pmset -g`; "Hardware and pins" below): {take.power_prose()}. Record and render ran as GPU-queue jobs of kind
render, one at a time, into `runs/film/{name}` (the tool's default `--out`): `--stage record`, then `--stage render`
(both shots); then `--stage encode --shots hero`, the README WebP command above and `--stage doc` (CPU). A new
recording is another take of the same policy, not this one: the GPU rollout is not bitwise repeatable, so which env is
filmed, and its path, can change.""")
    return f"""
**README clip.** [{name}.webp]({name}.webp) is the `readme` shot from 0.5 s to 5.5 s (frames 25 to 274), resampled
from 50 to 30 fps and written as MuJoCo Warp writes its README clips: 320x240 (Lanczos), lossy animated WebP from
Pillow at quality 70, 33 ms per frame, looping; 150 frames. Command, run from the repository root after the render
stage: `python -m metalsim.render.film.encode webp runs/film/{name}/readme/frames docs/renders/{name}.webp --start 0.5
--seconds 5 --src-fps 50 --fps 30`. The film tool's route, `python -m metalsim.render.film {name} --stage encode
--shots readme --webp --webp-window 0.5:5`, uses the same encoder but names its output `{name}_readme.webp` and also
writes a `{name}_readme.mp4` and a still.

{this_take}
"""


def _g1_notes(story: str, name: str, uncommitted: str = "the uncommitted switch to Isaac's checkpoints"):
    """FilmTask.notes of a G1 Isaac film: its fixed ``story`` (policy, task, env, gait, camera), then the parts the
    take's stage logs decide (``_g1_render_notes``, ``_g1_take_notes``)."""
    return lambda take: story + _g1_render_notes(take) + _g1_take_notes(take, name, uncommitted)


# Doc prose for the Isaac checkpoints, shared by the two G1 films
G1_OURS_NOTE = """
Our own checkpoint stays selectable and tracked: `--checkpoint runs/g1regress/ckpt14/{ckpt}` ({it} iterations trained
in MetalSim, its training log beside it) films it with the task settings that checkpoint saved, and this page's text
then does not describe the policy that ran. The previous take of this page, in git history at `ef346e7e`, filmed it.
"""
G1_ISAAC_IO_NOTE = """The policy's observation is Isaac's policy group in Isaac's order (base linear and angular velocity, projected
gravity, command, joint positions, joint velocities, last action; on rough terrain then the 187 height-scan rays in
Isaac's "xy" order), computed by the task's own kernel. The joint blocks and the actions are in Isaac's joint order
(PhysX's breadth-first articulation order, `adapters.ISAAC_G1_JOINTS`), mapped to MetalSim's by joint name as
`metalsim/parity/side_by_side.py` does; the actions are Isaac's joint-position targets (scale 0.5 about the default
pose). The task always applies Isaac's training observation noise, which Isaac's PLAY configuration turns off; on flat
ground that changed no measured gait figure (`runs/readme_media/handoff/isaac-pretrained.md`, Verified B)."""
G1_ISAAC_GAIT_NOTE = """
**The quick, short-stepping gait is Isaac's own.** Isaac Lab's G1 policies walk this way in PhysX too. Checkpoints of
the flat task trained with Isaac Lab in Isaac Sim (this project's L4 training run, iterations 500 / 1000 / 1499,
played in Isaac Sim at 0.5 m/s) step at 3.85 / 3.60 / 3.23 Hz with 0.108 / 0.110 / 0.123 m per gait cycle
(`docs/research/speculative_contacts_2026-09-27.md`, Verdict 1 and the cadence table of §2). MuJoCo Warp adds 3.4 to
5.9 % to the cadence of the same checkpoints under every contact setting measured (same note). NVIDIA's published
checkpoint itself was not played in PhysX for this page: Isaac Sim does not run on this machine.
"""

G1_FLAT = FilmTask(
    name="g1_flat",
    adapter="metalsim.render.film.adapters:G1Flat",
    venv="main",
    # NVIDIA's published Isaac Lab 2.3 checkpoint (rsl_rl, iteration 1499), fetched by scripts/fetch_isaac_assets.sh;
    # ours stays selectable: --checkpoint runs/g1regress/ckpt14/flat_main_s0.pt (tracked, log beside it)
    checkpoint=ISAAC_PRETRAINED["Isaac-Velocity-Flat-G1-v0"]["path"],
    adapter_kw={"task": G1_TRAIN_TASK_KW},
    title="G1 flat-terrain walk",
    summary=("Unitree G1 walking on flat ground: Isaac Lab's G1 flat velocity task on metalsim (MuJoCo Warp on Metal), "
             "driven by NVIDIA's published Isaac Lab 2.3 policy for the task (rsl_rl, iteration 1499, trained in "
             "PhysX), played without retraining. Rendered on the 14-inch MacBook Pro (Mac16,6), not in a timing run, so "
             "its Performance table is not a timing figure."),
    notes=_g1_notes("""
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
""" + G1_OURS_NOTE.format(ckpt="flat_main_s0.pt", it="500") + """
**Task settings.** `metalsim.learn.g1_velocity.G1VelocityTask(16, terrain="flat", seed=3)` with the settings Isaac's
checkpoint implies (`adapters.ISAAC_TASK_SETTINGS["flat"]`: `reward_cfg="flat"`, which carries `G1FlatEnvCfg`'s
command ranges, sideways velocity within +-0.5 m/s) and the task's defaults (`contact_cfg="recommended"`, base and
feet-slide velocities at the COM), plus physics dt 2.5 ms with decimation 8 (`G1_TRAIN_TASK_KW`). `rec.json`
`adapter.task_resolved` lists what ran.
""" + G1_ISAAC_IO_NOTE + """ Commands are the task's own, as in
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
""" + G1_ISAAC_GAIT_NOTE + G1_CAMERA_NOTES, "g1_flat"),
    viewer=G1_VIEWER,
    # 16 envs, "auto": the longest walk among envs that neither terminated nor went down (G1Flat.rank)
    film=FilmCfg(num_envs=16, envs="auto", grid=1, seed=3, max_seconds=6.0, hold_s=0.0, passes=16, spp=4, bounces=3,
                 output_filename_prefix="g1_flat"),
    look=LookCfg(hdri="hdri/train/poly_haven_studio.hdr", sun=False, ground_texture="tex/train/laminate_floor_02.png",
                 ground_geoms=("ground", "floor", "ground_plane"), ground_tile_m=LAMINATE_TILE_M),
    assets=G1_ASSETS + [
        AssetRef("Policy: Isaac Lab 2.3 pretrained rsl_rl checkpoint (Isaac-Velocity-Flat-G1-v0, iteration 1499; "
                 "sha256 fdf0f242...)", "NVIDIA Isaac Lab", ISAAC_PRETRAINED["Isaac-Velocity-Flat-G1-v0"]["url"],
                 NVIDIA_TERMS, FETCH_ISAAC),
    ],
    shots=G1_SHOTS,
    doc="docs/renders/G1_FLAT.md",
)

G1_ROUGH = FilmTask(
    name="g1_rough",
    adapter="metalsim.render.film.adapters:G1Rough",
    venv="main",
    # NVIDIA's published Isaac Lab 2.3 checkpoint (rsl_rl, iteration 2999), fetched by scripts/fetch_isaac_assets.sh;
    # ours stays selectable: --checkpoint runs/g1regress/ckpt14/rough_main_s0.pt (tracked, log beside it)
    checkpoint=ISAAC_PRETRAINED["Isaac-Velocity-Rough-G1-v0"]["path"],
    # level 5 of 10: inverted-stairs risers 0.14-0.16 m, inverted-slope grade 0.21-0.22 on this terrain; Isaac Lab's
    # own training of this task holds a mean terrain level of 5.9-6.0 from iteration ~750 (docs/PARITY.md §1.5)
    adapter_kw={"task": G1_TRAIN_TASK_KW, "terrain_level": 5},
    title="G1 rough-terrain walk",
    summary=("Unitree G1 climbing Isaac Lab's rough terrain: the G1 rough velocity task (stairs, slopes, boxes, height "
             "scan) on metalsim (MuJoCo Warp on Metal), driven by NVIDIA's published Isaac Lab 2.3 policy for the task "
             "(rsl_rl, iteration 2999, trained in PhysX), played without retraining. Rendered on the 14-inch MacBook "
             "Pro (Mac16,6), not in a timing run, so its Performance table is not a timing figure."),
    notes=_g1_notes("""
**The policy.** NVIDIA's published Isaac Lab 2.3 checkpoint for `Isaac-Velocity-Rough-G1-v0`: rsl_rl, iteration
2999, the last of the task's 3,000 (`assets/isaac/g1_rsl_rl_ppo_cfg.py`, `G1RoughPPORunnerCfg`), trained by NVIDIA in
Isaac Sim with PhysX on Isaac's terrain curriculum. MetalSim plays it as published, without retraining or fine-tuning.
Source:
<https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/IsaacLab/PretrainedCheckpoints/rsl_rl/Isaac-Velocity-Rough-G1-v0/checkpoint.pt>
(the path Isaac Lab's `--use_pretrained_checkpoint` resolves, `isaaclab_rl/utils/pretrained_checkpoint.py`, under
the Isaac 5.1 asset root), sha256 `d85a9adea70e5476675f130cbc70f41bdef7a28d52f443fae902afc0e2358851`, 7,842,530
bytes. It is NVIDIA's file under NVIDIA's asset terms: `scripts/fetch_isaac_assets.sh` downloads it to
`assets/isaac/pretrained/rsl_rl/Isaac-Velocity-Rough-G1-v0/checkpoint.pt` and checks the hash; `.gitignore` (`*.pt`)
keeps it out of the repository. The record stage refuses a file whose sha256 is not the published one and writes the
source URL, the published hash, the format and the iteration into `rec.json` `checkpoint` (`in_repo` false). The
actor is an ELU MLP, 310 -> 512-256-128 -> 37 with no observation normaliser (123 inputs as on flat ground plus the
187-ray height scan), read from the checkpoint's `actor.*` weights with `torch.load(weights_only=True)`
(`adapters.load_g1_checkpoint`); the film plays the actor's mean. NVIDIA publishes no training log for it (the
checkpoint's `infos` is None).
""" + G1_OURS_NOTE.format(ckpt="rough_main_s0.pt", it="1,000") + """
**Task settings.** `metalsim.learn.g1_velocity.G1VelocityTask(40, terrain="rough", seed=0)` with the settings Isaac's
checkpoint implies (`adapters.ISAAC_TASK_SETTINGS["rough"]`: `reward_cfg="rough_isaac"`, the exact port of
`G1RoughEnvCfg`, forward and turning commands only; height-scan ray order `xy`, Isaac's, which the adapter requires
for an Isaac policy) and the task's defaults (exact height scan, terrain collision `boxes_local`,
`contact_cfg="recommended"`, velocities at the COM), plus physics dt 2.5 ms with decimation 8 (`G1_TRAIN_TASK_KW`).
The exact scan and `boxes_local` collision are what `scripts/diagnostics/g1_rough_transfer.py` played Isaac's rough
checkpoints with (that tool read the base velocity at the pelvis origin, the film at the COM as Isaac Lab 2.3.2 does):
the final one of the project's L4 run fell 0 times in 32 wall-cell episodes (`docs/PARITY.md` §1.5, "Transfer of
Isaac's checkpoints on the wall cells"). `rec.json` `adapter.task_resolved` lists what ran.
""" + G1_ISAAC_IO_NOTE + """ The terrain is Isaac Lab 2.3.2's
`ROUGH_TERRAINS_CFG` (10 difficulty rows by 20 columns of 8 m sub-terrains; by column, pyramid stairs, inverted
pyramid stairs, random boxes, random rough ground, slopes and inverted slopes), generated from task seed 0, whose
origin table equals Isaac's at env seed 0 (`docs/PARITY.md` §1.5). Which random draw NVIDIA trained on is not
published. `adapter_kw["terrain_level"] = 5` starts every env on difficulty row 5 instead of the task's random rows 0
to 5: Isaac Lab's own training of this task holds a mean terrain level of 5.9 to 6.0 from about iteration 750
(`docs/PARITY.md` §1.5, the like-for-like rough table), and on this row the inverted stairs have risers of 0.14 to
0.16 m. The 40 envs sit two per column (`rec.json` `adapter.terrain_types`). Actions, commands, terminations and the
6 s, 50 fps film are as in the flat film.

**How the env was chosen.** `envs="auto"` ranks the 40 envs (`G1Rough.rank`). First come the envs whose first episode
was not ended by a termination, then the natural-gait preference of the flat film (`adapters.natural_gait`: lower
swing apex at least the candidates' median and larger stance slip at most their median), then the highest climb of
the pelvis above its start, then the longest path. The record stage refuses to film an env whose first episode
terminated unless `--allow-early-end` is given. In this take none of the 40 terminated, and 10 envs passed the gait
preference (`rec.json` `gait.natural_gait_envs`). Env 8, filmed, is on column 4 (inverted pyramid stairs) and climbed
out of the pit: its pelvis rose 0.97 m, about six risers, over a 5.11 m path, the highest climb of all 40 envs (next
among the preferred: env 15, 0.79 m; env 14, 0.65 m; `rec.json` `adapter.first_episode`). Under "What is shown",
`base_z`, `level` and `col` are written only when an episode ends; with `end_frame` -1 (still running when the film
ends) their zeros are placeholders, not the env's level (5) or column (4). That list is the top four of the ranking
(the record keeps spares); both shots film one env, the first (`render.json` `envs`).

**Gait.** Measured as on flat ground (`rec.json` `gait`, with its definition; `record.GAIT_DEFINITION`), with each
foot's height taken above the highest terrain under its footprint. Env 8, filmed: 0.88 m/s at a commanded 0.89 m/s;
3.19 / 3.39 touch-downs per second (left / right), about 0.27 m per gait cycle; stance slip 0.043 / 0.044 m/s;
airborne 57 / 65 % of the time. While it climbs, the foot rises from the lower tread over the next riser, so its swing
apex, 28 / 27 cm, includes the riser, and the foot-height spectrum mixes the climb with the steps (its peak, 2.6 Hz,
is not the cadence: read the touch-downs). Medians over the 40 envs, most of them not on stairs: 0.48 m/s (commanded
0.40), 3.39 touch-downs per second on each foot (spectral peak 3.31 / 3.30 Hz), apex 11.9 / 13.7 cm, slip 0.049 /
0.029 m/s.
""" + G1_ISAAC_GAIT_NOTE + """
**Terrain in the render.** The twin carries the task's own collision surface: its heightfield (2001 x 1201 samples
at 0.1 m) and Isaac's 14,080 terrain boxes merged into one mesh, `terrain_boxes`, with planar UVs on each face so
that the floor texture lies flat on treads and risers; both take the laminate texture (`ground_geoms`).
""" + G1_CAMERA_NOTES, "g1_rough"),
    viewer=G1_VIEWER,
    # 40 envs: two per terrain type (Isaac's column assignment); "auto" films the one that climbed most without
    # terminating (G1Rough.rank)
    film=FilmCfg(num_envs=40, envs="auto", grid=1, seed=0, max_seconds=6.0, hold_s=0.0, passes=16, spp=4, bounces=3,
                 output_filename_prefix="g1_rough"),
    look=LookCfg(hdri="hdri/train/poly_haven_studio.hdr", sun=False, ground_texture="tex/train/laminate_floor_02.png",
                 ground_geoms=("ground", "terrain_boxes"), ground_tile_m=LAMINATE_TILE_M),
    assets=G1_ASSETS + [
        AssetRef("Rough terrain: Isaac Lab 2.3.2 ROUGH_TERRAINS_CFG (10 x 20 sub-terrains of 8 m), generated from the "
                 "task seed", "Isaac Lab task config (port: metalsim.learn.isaac_terrain)",
                 "https://github.com/isaac-sim/IsaacLab", "BSD-3-Clause", "procedural (metalsim.learn.terrain)"),
        AssetRef("Policy: Isaac Lab 2.3 pretrained rsl_rl checkpoint (Isaac-Velocity-Rough-G1-v0, iteration 2999; "
                 "sha256 d85a9ade...)", "NVIDIA Isaac Lab", ISAAC_PRETRAINED["Isaac-Velocity-Rough-G1-v0"]["url"],
                 NVIDIA_TERMS, FETCH_ISAAC),
    ],
    shots=G1_SHOTS,
    doc="docs/renders/G1_ROUGH.md",
)


FETCH_UNITREE = UNITREE_RL_GYM["fetch"]
UNITREE_BLOB = f"{UNITREE_REPO}/blob/{UNITREE_COMMIT}"
# The fixed part of G1_UNITREE's notes (policy, loop, sim2sim check, gait, world, camera). The sim2sim table's distances
# are the full-precision pelvis distances of the re-run on MuJoCo Warp 8eb51fe (runs/readme_media/handoff/revalidate.md,
# "Old against new": 1.44-1.77 um at 5 s, 3.84-4.43 um at 8 s); compare.json rounds positions to 0.1 mm, so they are
# not in the take's files for the doc stage to read.
G1_UNITREE_STORY = f"""
**The policy.** Unitree's own pretrained G1 walking policy from Unitree's `unitree_rl_gym`
(<{UNITREE_REPO}>, BSD-3-Clause, commit `{UNITREE_COMMIT}`): `deploy/pre_train/g1/motion.pt`, the file both of
Unitree's playback scripts load, the MuJoCo sim2sim one (`deploy/deploy_mujoco/configs/g1.yaml`, `policy_path`) and the
one for the real robot (`deploy/deploy_real/configs/g1.yaml`, line 10). sha256
`{UNITREE_RL_GYM['policy_sha256']}`, 145,745 bytes, as fetched. It is a TorchScript module: an LSTM (47 inputs, 64
hidden) and an ELU MLP 64 -> 32 -> 12, loaded with `torch.jit.load` and played as published. Unitree trained it with
`legged_gym` on Isaac Gym (`legged_gym/envs/g1/g1_config.py`, `G1RoughCfg` / `G1RoughCfgPPO`: `ActorCriticRecurrent`,
plane terrain, friction and added-mass randomization, pushes every 5 s; physics 5 ms x decimation 4, so 50 Hz). Its
gait is set by the training task, not left to the policy: `g1_env.py` feeds a fixed gait clock (period 0.8 s, so
1.25 gait cycles per second; the right leg half a cycle behind the left) to the policy as sin / cos of the phase, and
rewards foot contact in the stance part of each leg's cycle (phase < 0.55, `_reward_contact`) and an 8 cm swing
height of the ankle-roll link (`_reward_feet_swing_height`). `scripts/fetch_unitree_rl_gym.sh` fetches the repository
at that commit into `assets/unitree_rl_gym/` (git-ignored, not committed); the record stage checks the policy's
sha256 and writes the repository, commit, file, licence and hash into `rec.json` `checkpoint` and `adapter`.

**Unitree's loop, on MuJoCo Warp.** Unitree plays the policy in MuJoCo with `deploy/deploy_mujoco/deploy_mujoco.py`
and `configs/g1.yaml`: physics step 2 ms, a policy step every 10 physics steps (50 Hz), PD gains kp (100, 100, 100,
150, 40, 40) and kd (2, 2, 2, 4, 2, 2) per leg (hip pitch, roll, yaw, knee, ankle pitch, roll), default angles
(-0.1, 0, 0, 0.3, -0.2, 0) per leg, scales: angular velocity 0.25, joint position 1, joint velocity 0.05, action 0.25,
command (2, 2, 0.25); 12 actions, 47 observations, command (0.5, 0, 0) m/s, m/s, rad/s. The model is Unitree's
`resources/robots/g1_description/scene.xml` (`g1_12dof.xml`: the 12 leg joints; Euler integrator, pyramidal cones,
Newton solver, MuJoCo's defaults; four 5 mm collision spheres under each foot; `<motor>` actuators whose torque the
joints' `actuatorfrcrange` clamps). The film runs the same loop on MetalSim's MuJoCo Warp (`metalsim.physics.batch.BatchSim`,
4 worlds of that model; `metalsim/render/film/unitree.py`, `UnitreeG1`): Unitree computes the PD torque from
`d.qpos` / `d.qvel` before every `mj_step`, and so does the film, every 2 ms physics step: a Warp kernel before the
first physics step of each policy step and, through `BatchSim.add_substep_hook`, after every physics step inside the
captured step graph (`metalsim/physics/batch.py`, `_launch_substeps`), so each step is driven by the torque of the
state it starts from. The observation is built from the state after the tenth step, as Unitree's, in float64 and
stored as float32, with the gait clock read from the physics step counter (`counter * 0.002`). The policy keeps its
LSTM state inside the module (batch 1), so each world has its own loaded copy. No episode ends: Unitree's playback
has no termination or reset (a pelvis below 0.4 m would be reported as down; none was).

**Against Unitree's own run.** Unitree's loop itself (`deploy_mujoco.py`'s code unchanged, without its viewer and
wall-clock sleep) on CPU MuJoCo 3.14.0, 8 s, against the film's adapter on MuJoCo Warp with 4 worlds, both measured
with the same code (`record.gait_metrics`, window 1 to 8 s). Scripts and results: `runs/film/g1_unitree/sim2sim_check/`
(`reference.py`, CPU only, writes `reference.json`; `compare.py`, a GPU-queue job, writes `compare.json`).

| | Unitree's loop, CPU MuJoCo | MetalSim, MuJoCo Warp (4 worlds) |
|---|---|---|
| pelvis at 5 s (x, y, z) | (2.2244, -0.0988, 0.7703) m | the same to 0.1 mm; distance 1.4 to 1.8 um |
| pelvis at 8 s | (3.6469, -0.2573, 0.7723) m | distance 3.8 to 4.4 um |
| gait cycles per second (FFT of foot height), left / right | 1.251 / 1.251 | 1.251 / 1.251 |
| touch-downs per second, left / right | 1.14 / 1.29 | 1.14 / 1.29 |
| swing apex above stance (sole), left / right | 4.2 / 4.4 cm | 4.2 / 4.4 cm |
| ankle-roll link height in swing (95th percentile), left / right | 7.7 / 8.1 cm | 7.7 / 8.1 cm |
| stance slip, left / right | 0.004 / 0.004 m/s | 0.004 / 0.004 m/s |
| pelvis speed (command 0.5 m/s) | 0.48 m/s | 0.48 m/s |
| foot contact, frame by frame | | identical in all 400 frames |

The two runs are not bitwise equal (MuJoCo Warp computes in float32, MuJoCo in float64; the four GPU worlds also
differ from one another by a few micrometres), but no gait figure differs at the printed precision, and the largest
difference in any policy output over the 400 policy steps is 3e-5.

**Gait.** Measured by the record stage on every simulated world from 1 s to the end of the 6 s film (`rec.json`
`gait`, with its definition, `record.GAIT_DEFINITION`). A foot's point is the mean of its four collision spheres'
centres, 5 mm above the floor in stance, and it is in contact when one of them touches the floor
(`unitree.unitree_feet`, `unitree_contacts`). The filmed world: 0.48 m/s at a commanded 0.50 m/s; 1.245 gait cycles
per second on both feet (the 0.8 s clock's 1.25 Hz; touch-downs 1.20 / 1.39 per second, left / right, 6 and 7 in the
5 s window); swing apex 4.2 / 4.4 cm of sole clearance, which puts the ankle-roll link at about 8 cm, Unitree's swing
target; stance slip 0.004 m/s; airborne 43 / 47 % of the time; 0.386 m covered per gait cycle. The four worlds agree
to the printed precision. The arms do not move: in Unitree's 12-dof model they are fixed to the pelvis.

**How the world was chosen.** Unitree's playback has one start (the model's `qpos0`: pelvis at 0.793 m, joints at
zero, PD targets at the default angles until the first policy step at 20 ms), so the four worlds are copies of it and
differ only by float noise. `envs="auto"` ranks them as the G1 films do (`UnitreeG1.rank`: not down, then the
natural-gait preference, then the longest path): the four tie on gait, and world 2's path was longest by float noise.
Under "What is shown", `base_z` is written only when an episode ends; with `end_frame` -1 (still walking when the
film ends) its 0 is a placeholder (the pelvis stays near 0.77 m). Both shots film one world, the first of that list
(`render.json` `envs`).
""" + G1_CAMERA_NOTES


def _g1_unitree_notes(take) -> str:
    """G1_UNITREE's notes: the story, then rendering, README clip and this take, whose floor size, day, machine,
    commits, fork pins and power are read from the take's stage logs (``doc.Take``)."""
    t, real = _tile(take, LAMINATE_TILE_M)
    if t is None:
        size = ""
    elif real:            # the laminate image is eight planks across
        size = (f", at its real size (Poly Haven's `laminate_floor_02`: {t:.2f} m\nper image, eight {t / 8 * 100:.0f} "
                "cm planks; `ground_tile_m`)")
    else:
        size = f", at {t:.2f} m per image (`ground_tile_m`; `laminate_floor_02` is {LAMINATE_TILE_M:.2f} m across)"
    mjw, warp, hero_warp = take.commit("mujoco_warp", n=7), take.commit("warp"), take.commit("warp", "hero")
    mjw = (f"on the pinned MuJoCo Warp fork commit `{mjw}`" if take.at_pin("mujoco_warp") else
           f"on the MuJoCo Warp fork commit `{mjw}`, not its pin `{take.pin('mujoco_warp', 7)}`")
    warp = (f"Warp itself at its pin `{warp}`" if take.at_pin("warp") else
            f"Warp itself at `{warp}`, not its pin `{take.pin('warp')}`")
    hero_warp = (f"`{hero_warp}`, its pin" if take.at_pin("warp", "hero") else
                 f"`{hero_warp}`, not its pin `{take.pin('warp')}`")
    hero = take.at("hero")
    # the frames this re-render replaced: the G1 films' floor before ground_tile_m
    old_floor = (f", after the film tool started laying the floor texture at its real size (`ground_tile_m`; the "
                 f"earlier frames drew it {LAMINATE_OLD_G1_TILE_M / t:.2f} times too large, as "
                 f"{LAMINATE_OLD_G1_TILE_M / 8 * 100:.0f} cm planks)" if real else "")
    this_take = take.fill(f"""
**This take, and where its timings come from.** Recorded, rendered and encoded on {take.day} on the {take.machine},
at metalsim {take.metalsim_at("uncommitted changes")}, {mjw} ({warp}). It replaces a take recorded at 13:55 the same
day on a stale fork checkout (`07a51a6`); the two recordings differ by at most 2 um in any body position. The frames on
this page were rendered again on {hero[:10]} at {hero[11:16]} from that same recording (`rec.npz`, `rec.json` and
`twin.xml` unchanged; that render ran on Warp {hero_warp}, as the render stage's pins above say){old_floor}. This was
not a timing run: MetalSim's timing runs are made on the 16-inch through the GPU queue's timing kind, and its
throughput figures are in `docs/PARITY.md` §1.4 and §1.8; quote nothing on this page as MetalSim's speed. Power, as
the stages stamped it ("Hardware and pins" below): {take.power_prose("the two render stamps are the re-render's")}.
Record and render ran as GPU-queue jobs of kind render, one at a time, into `runs/film/g1_unitree`: `--stage record`,
then `--stage render` (both shots); later a second `--stage render` job (both shots, the real-size floor); then
`--stage encode --shots hero`, the README WebP command above and `--stage doc` (CPU).""")
    return G1_UNITREE_STORY + f"""
**Rendering.** Tier 2 (the path tracer), 64 samples per pixel (16 passes x 4), 3 bounces, Open Image Denoise, ACES
tone map, sRGB; lit by the Poly Haven studio HDRI as an importance-sampled dome, with the laminate floor texture on
the scene's `floor` plane in place of Unitree's checker{size}, the same look as the G1 flat film. The render twin is
Unitree's own `scene.xml` with its STL meshes, posed per frame from the recorded joint state. Unitree's model gives
each link a visual mesh and a colliding copy; the colliding copies and the foot spheres are hidden (`rec.json`
`twin.visibility`), and the meshes keep Unitree's two greys. The scene's own lights, sky and checker floor are
replaced by the film's look.

**README clip.** [g1_unitree.webp](g1_unitree.webp) is the `readme` shot from 0.5 s to 5.5 s (frames 25 to 274),
resampled from 50 to 30 fps and written as MuJoCo Warp writes its README clips: 320x240 (Lanczos), lossy animated
WebP from Pillow at quality 70, 33 ms per frame, looping; 150 frames. Command, run from the repository root after the
render stage: `python -m metalsim.render.film.encode webp runs/film/g1_unitree/readme/frames
docs/renders/g1_unitree.webp --start 0.5 --seconds 5 --src-fps 50 --fps 30`.

{this_take}
"""


G1_UNITREE = FilmTask(
    name="g1_unitree",
    adapter="metalsim.render.film.unitree:UnitreeG1",
    venv="main",
    # Unitree's own pretrained G1 walking policy (TorchScript), fetched by scripts/fetch_unitree_rl_gym.sh
    checkpoint=f"{UNITREE_RL_GYM['root']}/{UNITREE_RL_GYM['policy']}",
    adapter_kw={"cmd": (0.5, 0.0, 0.0)},          # configs/g1.yaml cmd_init
    title="G1 walk, Unitree's policy",
    summary=("Unitree G1 walking on flat ground with Unitree's own pretrained walking policy (unitree_rl_gym, "
             "`deploy/pre_train/g1/motion.pt`, BSD-3-Clause), played by Unitree's MuJoCo sim2sim loop "
             "(`deploy_mujoco.py`, `configs/g1.yaml`) ported to metalsim's MuJoCo Warp on Metal, on Unitree's own G1 "
             "model and meshes. Rendered on the 14-inch MacBook Pro (Mac16,6), not in a timing run, so its "
             "Performance table is not a timing figure."),
    notes=_g1_unitree_notes,
    viewer=G1_VIEWER,
    # 4 copies of Unitree's single start (they differ only by GPU float noise); "auto": not down, natural gait, path
    film=FilmCfg(num_envs=4, envs="auto", grid=1, seed=0, max_seconds=6.0, hold_s=0.0, passes=16, spp=4, bounces=3,
                 output_filename_prefix="g1_unitree"),
    look=LookCfg(hdri="hdri/train/poly_haven_studio.hdr", sun=False, ground_texture="tex/train/laminate_floor_02.png",
                 ground_geoms=("floor",), ground_tile_m=LAMINATE_TILE_M),
    assets=[
        AssetRef("Unitree G1, 12-dof legs model (g1_12dof.xml in scene.xml) and its STL meshes", "Unitree Robotics "
                 "unitree_rl_gym", f"{UNITREE_BLOB}/resources/robots/g1_description", "BSD-3-Clause", FETCH_UNITREE),
        AssetRef(f"Policy: Unitree's pretrained G1 walking policy deploy/pre_train/g1/motion.pt (TorchScript; sha256 "
                 f"{UNITREE_RL_GYM['policy_sha256'][:8]}...)", "Unitree Robotics unitree_rl_gym",
                 UNITREE_RL_GYM["policy_url"], "BSD-3-Clause", FETCH_UNITREE),
        AssetRef("Sim2sim playback loop and config (deploy_mujoco.py, configs/g1.yaml), ported to MuJoCo Warp",
                 "Unitree Robotics unitree_rl_gym", f"{UNITREE_BLOB}/deploy/deploy_mujoco", "BSD-3-Clause",
                 FETCH_UNITREE),
        G1_ASSETS[1], G1_ASSETS[2],
    ],
    shots=G1_SHOTS,
    doc="docs/renders/G1_UNITREE.md",
)


FETCH_PH_DEMO = "scripts/fetch_polyhaven_assets.sh demo"
# Isaac Lab's random_rough sub-terrain as ROUGH_TERRAINS_CFG defines it (noise_range (0.02, 0.10), its only setting:
# random_uniform_terrain ignores difficulty), one 8 x 8 m tile, env seed 0 (unitree.random_rough_tile)
G1_BUMPS_TERRAIN = {"noise_range": (0.02, 0.10), "seed": 0}
# 16 robots in one row of lanes across the tile (x = -2 m, y = -3 ... 3 m, 0.4 m apart), as the measurement
# (runs/film/g1_bumps/measure/measure.py); filmed: a lane with the tile on the camera's side (y -1.4 ... 1.0 m)
G1_BUMPS_STARTS = {"x0": -2.0, "y": [-3.0, 3.0]}
G1_BUMPS_FILM_LANES = (-1.4, 1.0)
# Lower than the G1 films' camera (eye 0.75 m instead of 1.3 m above the pelvis, looking at 0.35 m below it, 18.5 deg
# down instead of 22): the bumps read in profile around the feet. Same direction, lens, tracking and smoothing.
G1_BUMPS_VIEWER = ViewerCfg(eye=(2.6, -2.0, 0.75), lookat=(0.0, 0.0, -0.35), origin_type="asset_body",
                            body_name="pelvis", fovy=KIT_FOVY_16_9, resolution=(1280, 720), smooth_s=1.0)
# README cell (4:3): the hero's direction and elevation at 0.75 x the distance, looking 0.15 m below the pelvis (at
# 0.35 m below, the closer frame cut the head off: look test, runs/readme_media/scratch/g1-bumps/look)
G1_BUMPS_README_VIEWER = ViewerCfg(eye=(1.95, -1.5, 0.675), lookat=(0.0, 0.0, -0.15), origin_type="asset_body",
                                   body_name="pelvis", fovy=KIT_FOVY_16_9, resolution=(640, 480), smooth_s=1.0,
                                   orbit_deg_s=10.0)
G1_BUMPS_MJW = "787a5b3"      # the MuJoCo Warp fork commit the measurements below ran on (runs/film/g1_bumps/measure/mjw_787a5b3)
# The record the take check accepted (rec.json hardware.power.at): the selection paragraph describes that record only
G1_BUMPS_CHECKED_RECORD = "2026-10-31T23:37:22-0700"
G1_BUMPS_STORY = f"""
**What this film is.** The policy of the G1 flat film ([G1_UNITREE.md](G1_UNITREE.md)): Unitree's own pretrained G1
walking policy, `deploy/pre_train/g1/motion.pt` from `unitree_rl_gym` (<{UNITREE_REPO}>, BSD-3-Clause, commit
`{UNITREE_COMMIT}`, sha256 `{UNITREE_RL_GYM['policy_sha256']}`), played by Unitree's MuJoCo sim2sim loop
(`deploy/deploy_mujoco/deploy_mujoco.py`, `configs/g1.yaml`, command (0.5, 0, 0)) ported to MetalSim's MuJoCo Warp, as
in the G1 flat film, with one change: Unitree's floor plane is Isaac Lab's random rough ground. The policy is blind (47
observations: body rates, gravity, command, joint state, last action, gait clock; no height scan) and Unitree trained
it on flat ground only (`legged_gym`, `legged_robot_config.py` line 15 `mesh_type = 'plane'`, not overridden by
`g1_config.py`), with pushes, friction and mass randomization. Nothing is tuned, and the robot falls: this page shows
what happens.

**Terrain.** Isaac Lab 2.3.2's `ROUGH_TERRAINS_CFG["random_rough"]` (`isaaclab/terrains/config/rough.py`):
`HfRandomUniformTerrainCfg(proportion=0.2, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25)`, on one 8 x 8
m tile at the generator's horizontal scale 0.1 m and vertical scale 5 mm (slope threshold 0.75), with the generator's 20
m flat border around it, built by MetalSim's port of Isaac's terrain generator (`metalsim.learn.isaac_terrain`, the
generator of the G1 rough film) for env seed 0 (`unitree.random_rough_tile`). Isaac's `random_uniform_terrain`
ignores its difficulty (`height_field/hf_terrains.py`, its docstring: "The difficulty parameter is ignored for this
terrain"): at every curriculum level every 0.1 m grid point is an independent uniform draw from 2, 4, 6, 8 or 10 cm
(`downsampled_scale` defaults to the horizontal scale, so the spline interpolation changes nothing). So Isaac Lab has
one random rough terrain, not a range of levels: bumps 2 to 10 cm high every 0.1 m, up to 8 cm between neighbours, on
a 0.25 m flat rim (z = 0) around the 7.5 m rough square. This film uses exactly that terrain; the levels in the table
below with a lower ceiling are variants made here to see how the outcome depends on bump height, not Isaac's. The
tile enters Unitree's `scene.xml` as a MuJoCo heightfield (481 x 481 samples, 48 m with the border) that replaces the
`floor` plane: the same geom, name and contact parameters (MuJoCo's defaults, as Unitree's plane), type heightfield
(`unitree.unitree_terrain_spec`). Isaac builds a triangle mesh from the height field, in which the 1014 of 12,960 grid
edges whose height jump exceeds the slope threshold (a 2 cm sample next to a 10 cm one: 8 cm over 0.1 m) become
vertical walls; MuJoCo's heightfield interpolates them as 0.1 m ramps (the grid heights are the top of Isaac's mesh,
`isaac_terrain.height_field_surface`). Everything else in Unitree's loop is unchanged. For the picture only, an
infinite plane 2 mm under the flat border continues the ground to the horizon (`backdrop`; it collides with nothing).

**Where the robots start.** 16 robots, each on its own lane across the tile: pelvis at x = -2 m, y from -3 to 3 m in
0.4 m steps, all walking towards +x. Each starts in Unitree's start pose (`qpos0`: joints at zero, sole spheres 1.1 mm
above the floor), raised by the highest terrain under either sole at its spot, so its soles start 1.1 mm above every
bump under them (`unitree.unitree_start_qpos`), as Isaac spawns a robot above the highest point near its terrain
origin. In 8 s at the commanded 0.5 m/s a robot covers about 4 m in x, which keeps it on the rough square, but a
robot that veers can leave it sideways: on Isaac's terrain the robot on the outermost lane y = -3 m crossed the
square's edge (|y| > 3.75 m) at 2.38 s and fell 0.3 s later in the MetalSim measurement run, and in CPU MuJoCo both
outermost robots did (at 2.34 and 4.64 s; down 0.7 and 0.2 s later), near the tile's 0.25 m flat rim. (The outer
lanes were a poor choice; `measure.py` keeps them so the takes and the measurements stay comparable.)

**Physics: MetalSim's MuJoCo Warp.** The film plays Unitree's loop on MetalSim's MuJoCo Warp
(`metalsim.physics.batch.BatchSim`, 16 worlds of Unitree's model with the heightfield; `unitree.UnitreeG1`: the PD
torque before every 2 ms physics step, the policy every tenth), the port the G1 flat film plays. On flat ground it
matches Unitree's own CPU MuJoCo run: re-checked on this fork, the pelvis agrees to the printed 0.1 mm at every
checkpoint from 0.02 to 8 s and every gait figure is equal (`runs/film/g1_bumps/measure/mjw_787a5b3/flat_sim2sim_compare.json`,
G1_UNITREE.md). The MuJoCo Warp fork is MetalSim's `metalsim` branch at `{G1_BUMPS_MJW}`: heightfield contacts checked
against the exact geometry (fork `93e5ce1`, docs/research/hfield_contacts_2026-10-31.md) and EPA's horizon built as
MuJoCo C builds it (`{G1_BUMPS_MJW}`). Those fixes are why this film can run on MetalSim's physics. The G1's feet collide
through four 5 mm spheres each, and on this tile a sphere in a crease between two bumps touches two faces at once.
Before the fixes the fork lost such contacts and kept most robots walking (54 of 288 robots fell, 18.8 %, fork
`8eb51fe` over tile seeds 0 to 4; research note section 8.3), so the first take of this film (2026-10-31, afternoon)
ran Unitree's loop on CPU MuJoCo instead (`adapter_kw["engine"] = "mujoco_c"`). A static check on this tile
(`runs/film/g1_bumps/measure/contact_check.py`; `mjw_787a5b3/contact_check.json`; the "before" row from the first take's
`contact_check.json`): 3,000 spheres pressed 0.5 to 3 mm into the triangulated surface at random spots, each engine's
contacts compared with the exact contact set (every distinct closest point of the surface's triangles within the
sphere; 661 of the 3,000 touch two or more faces).

| | contacts per sphere equal to the exact set | spheres with fewer contacts | spheres with no contact | contacts more than 10 deg off the nearest true normal |
|---|---|---|---|---|
| CPU MuJoCo 3.14 | 99.6 % (11 spheres get a duplicate) | 0 | 0 | 0 of 3,736 |
| MetalSim's MuJoCo Warp, fork `{G1_BUMPS_MJW}` (this film) | 99.5 % | 14 (13 of them touch 5 or 6 faces, more than the 4 contacts MuJoCo keeps per pair) | 0 | 1 of 3,707 |
| MetalSim's MuJoCo Warp before the fixes (fork `8eb51fe`) | 88.8 % (211 spheres get extra contacts) | 126 | 0 | 6.1 % of 3,781 |

(The first take's page listed 81.9 % for "MetalSim's MuJoCo Warp"; that run had imported a stale fork checkout,
`07a51a6`: research note section 2.)

**Measured before filming** (`runs/film/g1_bumps/measure/measure.py`: `cpu_levels.json`, the first take's CPU MuJoCo
run; `mjw_787a5b3/gpu_levels.json`, MetalSim on fork `{G1_BUMPS_MJW}`). Per level 16 robots for 8 s, the lanes above,
command (0.5, 0, 0), tile seed 0. A fall: the pelvis below 0.4 m (`unitree.FALLEN_Z`; the robot is on the ground;
Unitree's training ends an episode on pelvis contact). Gait per robot from 1 s to its fall or to 8 s
(`record.gait_metrics`), the foot's height taken above the highest terrain under its sole (a 1 cm grid over the
quadrilateral of its four sole spheres, `unitree.sole_grid`); medians over the robots with at least 2 s of walking
(all 16, except 14 for MetalSim on Isaac's terrain). Swing apex: 95th percentile of the sole's height above its stance
height. Forward: pelvis x at the fall, or at 8 s, minus the start. On flat ground (G1_UNITREE.md): 1.25 Hz, 1.14 /
1.29 touch-downs per second, apex 4.2 / 4.4 cm, 3.65 m forward in 8 s, in both engines.

| bumps (noise_range) | engine | falls of 16 | first / median fall | gait cycles per s, L / R | touch-downs per s, L / R | swing apex, L / R | forward, median (range) |
|---|---|---|---|---|---|---|---|
| 2 to 4 cm (0.02, 0.04), variant | CPU MuJoCo | 2 | 6.58 / 7.08 s | 1.25 / 1.25 | 1.29 / 1.29 | 4.7 / 5.1 cm | 3.42 m (1.62 to 3.74) |
| | MetalSim | 1 | 5.82 / 5.82 s | 1.25 / 1.25 | 1.14 / 1.29 | 4.5 / 4.9 cm | 3.41 m (2.64 to 3.60) |
| 2 to 6 cm (0.02, 0.06), variant | CPU MuJoCo | 6 | 3.34 / 6.78 s | 1.26 / 1.25 | 1.36 / 1.29 | 5.0 / 5.6 cm | 2.96 m (0.18 to 4.25) |
| | MetalSim | 4 | 3.80 / 4.58 s | 1.25 / 1.25 | 1.29 / 1.29 | 5.6 / 5.8 cm | 3.08 m (0.35 to 4.01) |
| 2 to 8 cm (0.02, 0.08), variant | CPU MuJoCo | 7 | 3.46 / 5.06 s | 1.25 / 1.26 | 1.43 / 1.29 | 6.4 / 6.2 cm | 2.63 m (0.58 to 3.43) |
| | MetalSim | 9 | 3.40 / 6.22 s | 1.24 / 1.26 | 1.39 / 1.35 | 5.6 / 6.1 cm | 2.56 m (0.65 to 3.41) |
| **2 to 10 cm (0.02, 0.10), Isaac's** | CPU MuJoCo | **11** | 3.02 / 4.62 s | 1.17 / 1.25 | 1.71 / 1.44 | 7.0 / 6.5 cm | 1.95 m (-0.55 to 3.87) |
| | MetalSim | **10** | 2.66 / 5.11 s | 1.25 / 1.26 | 1.43 / 1.29 | 6.8 / 6.2 cm | 2.07 m (-0.30 to 3.55) |

Each row is one run. The two engines agree in the aggregate, not robot by robot: on Isaac's terrain 7 of the 16 lanes
have a different outcome in the two runs. The touch-down rate and the swing apex grow with the bump height; in the
frames the extra touch-downs are short catching steps after a stumble (a reading of the frames, not a separate
measurement). On Isaac's terrain 15 of 16 robots also touch the terrain with some geom other than a sole sphere (a
mesh of the leg, arm or body; for some within the first 0.2 s, so this is not by itself a stumble) within the 8 s, in
both engines (`nonfoot_contacts`).

**How often it falls, over five tiles** (the research note's protocol, section 8.3: Isaac's terrain, tile seeds 0 to
4, 16 robots each, 8 s; `mjw_787a5b3/falls_summary.json`, `mjw_787a5b3/falls/`). MuJoCo Warp on Metal is not
run-to-run repeatable here (the order in which the GPU accumulates contacts varies, and a stumbling gait amplifies
the difference), so each tile is run three times; CPU MuJoCo is deterministic, so its spread comes from nudging the
start joints by 1e-6 rad.

| engine | runs | falls of 16, tile seed 0 / 1 / 2 / 3 / 4 (each run) | falls per 80-robot run | total |
|---|---|---|---|---|
| CPU MuJoCo 3.14 | 1 | 11 / 10 / 14 / 12 / 8 | 55 | 55 / 80 = 68.8 % |
| CPU MuJoCo, start joints + N(0, 1e-6 rad) | 3 | 12, 11, 10 / 13, 12, 11 / 13, 11, 11 / 11, 11, 10 / 10, 13, 5 | 59, 58, 47 | 164 / 240 = 68.3 % |
| MetalSim's MuJoCo Warp, fork `{G1_BUMPS_MJW}` (this film) | 3 | 11, 11, 11 / 13, 15, 15 / 12, 12, 12 / 15, 11, 11 / 9, 11, 11 | 60, 60, 60 | 180 / 240 = 75.0 % |
| MetalSim's MuJoCo Warp before the fixes (fork `8eb51fe`) | 3 to 6 | 2, 5, 4, 3, 1, 3 / 3, 3, 6 / 1, 6, 5 / 3, 1, 4 / 1, 1, 2 | 10, 12, 20 | 54 / 288 = 18.8 % |

MetalSim's robots fall a little more often than CPU MuJoCo's over the five tiles (60 against 55 per 80 robots, less
than one standard deviation of CPU MuJoCo's own nudged runs, 6.7) and as often on the film's tile (seed 0: 10 to 12 of
16 in the six MetalSim runs listed on this page, 10 to 12 in CPU MuJoCo). The research note names float32 against
float64 and MuJoCo C's duplicate contacts at shared edges as candidates for the remaining gap; it is not resolved.
"""


def _g1_bumps_take(take) -> dict | None:
    """The filmed robot's facts from the take's own logs (rec.json, rec.npz): env, lane, fall, path, gait, the
    record's falls and its typical-pick numbers. None without a record."""
    rec = take.rec
    if not rec.get("envs") or "adapter" not in rec:
        return None
    a, e, dt = rec["adapter"], int(rec["envs"][0]), float(rec["frame_dt"])
    d = a["down_frame"][e]
    out = dict(env=e, lane=a["starts"]["xy"][e][1], fell=d >= 0, fall_s=(d + 1) * dt if d >= 0 else None,
               frame=d, falls=sum(1 for f in a["down_frame"] if f >= 0), n=len(a["down_frame"]),
               pick=a.get("pick_info") or {}, gait=(rec.get("gait") or {}).get("per_env", {}).get(str(e)),
               engine=a.get("engine"), fwd=None, lat=None, path=a["first_episode"]["path_m"][e])
    try:
        import os

        import numpy as np
        T = np.load(os.path.join(take.rec_dir, "rec.npz"))["track"][:, 0]      # the filmed env's pelvis per frame
        k = d if d >= 0 else len(T) - 1
        out["fwd"], out["lat"] = float(T[k, 0] - T[0, 0]), float(T[k, 1] - T[0, 1])
    except (OSError, KeyError, IndexError):
        pass
    return out


def _g1_bumps_notes(take) -> str:
    """G1_BUMPS's notes: the story and measurements, then the take (filmed robot, selection, gait), look, camera,
    rendering, README clip and this take's facts, read from the take's stage logs (``doc.Take``)."""
    f = _g1_bumps_take(take)
    if f is None or f["engine"] != "mujoco_warp":
        the_take = ("**The take.** This page's stage logs do not hold a MetalSim-physics record; run the record stage "
                    "(`--stage record`) before the doc stage.")
        gait = ""
    else:
        side = "left" if (f["lat"] or 0) > 0 else "right"
        path = (f"walks {f['fwd']:.2f} m forward over the bumps while veering {abs(f['lat']):.2f} m to its {side} "
                f"({f['path']:.2f} m of path)" if f["fwd"] is not None else f"walks {f['path']:.2f} m over the bumps")
        if f["fell"]:
            what = (f"{path} and is down at {f['fall_s']:.2f} s (frame {f['frame']}), falling backwards. The film does not "
                    "stop at the fall: Unitree's playback has no termination or reset, so the policy keeps acting on the "
                    "ground (in the clip one leg swings up about 0.3 s after it lands), and the render plays the "
                    "recorded simulation to 8 s (`FilmCfg.after_end = \"play\"`)")
        else:
            what = f"{path} and is still walking at 8 s"
        p = f["pick"]
        checked = take.at() == G1_BUMPS_CHECKED_RECORD
        selection = (
            " Whether the filmed robot itself is typical was checked before rendering, against a rule set before this "
            "record ran: re-record only if its own outcome contradicts its lane's outcome in every one of the four "
            "MetalSim measurement runs of this tile (`measure.py` and the three seed-0 runs above). Its lane, y = -0.6 m, "
            "fell in 1 of those 4 runs (and in CPU MuJoCo, at 5.82 s) and walked in the other 3: a lane where falling and "
            "walking both happen, not one that always walks. An earlier record, also 12 falls of 16, was not rendered: "
            "a first, stricter rule (the record's fall count within the four measurement runs' 10 to 11) rejected it, "
            "and was then replaced by the rule above because a count range from four runs is narrower than the spread "
            "between runs; under the rule above that record would have passed too (its filmed robot, lane y = 0.2 m, "
            "fell in all four measurement runs). The rejected record is not kept in the repository."
            if checked else
            " This record was not checked against the measurement runs (the check described in this page's history "
            "applied to the record of 2026-10-31 23:37).")
        the_take = take.fill(f"""
**The take.** The level is Isaac's own terrain because it is the only one Isaac Lab defines, not because of its outcome;
its bumps (2 to 10 cm) are clearly visible. The record stage is one more run of the 16 lanes on MetalSim's MuJoCo Warp
(fork `{take.commit("mujoco_warp", n=7)}`), so its outcome is one draw from the spread above: {f['falls']} of {f['n']}
robots fall in it (the four measurement runs of this tile: 10, 11, 11 and 11). The film shows the record's typical
outcome (`adapter_kw["pick"] = "typical"`, `UnitreeG1._rank_typical`): as at least half of the 16 fell, the fallen robot
whose fall time is nearest the median fall time of all {p.get('envs_down', f['falls'])} ({p.get('median', 0):.2f} s),
among the lanes from y = -1.4 to 1.0 m (`film_lanes`: there the camera, which stands 2 m to the robot's right, sees the
rough tile around the robot rather than its flat border; the lane decides which bumps a robot meets, not how hard they
are). That is the robot on lane y = {f['lane']:.1f} m, which {what}.{selection}""")
        g = f["gait"] or {}
        gait = "" if not g else take.fill(f"""
**Gait of the filmed robot** (`rec.json` `gait`, {g['window_s'][0]:.0f} s to {'its fall at ' if f['fell'] else ''}{g['window_s'][1]:.2f}
s): {g['speed_mps']:.2f} m/s along its path at a commanded {g['cmd_speed_mps']:.2f}; {g['cadence_hz'][0]:.2f} /
{g['cadence_hz'][1]:.2f} gait cycles per second (touch-downs {g['touchdowns_hz'][0]:.2f} / {g['touchdowns_hz'][1]:.2f} per
s, left / right); swing apex {g['apex_m'][0] * 100:.1f} / {g['apex_m'][1] * 100:.1f} cm above the highest terrain under the
sole (4.2 / 4.4 cm on flat ground): high, stumbling steps; stance slip {g['slip_mps'][0]:.3f} / {g['slip_mps'][1]:.3f}
m/s.""")
    clip_fall = (f" the fall at {f['fall_s']:.2f} s and the robot on the ground after it" if f and f["fall_s"] else
                 " the robot walking throughout")
    t, real = _tile(take, SANDY_GRAVEL_TILE_M)
    if t is None:
        tile = ""
    elif real:
        tile = (f" at its scanned size, {t:.2f} x {t:.2f} m per image (Poly Haven's `dimensions`; `ground_tile_m` {t:.2f}, "
                f"which the film tool turns into MuJoCo's `texrepeat` 2 / {t:.2f}: MuJoCo lays a `texuniform` texture once "
                "per 2 / `texrepeat` m on planes and heightfields alike)")
    else:
        tile = f" at {t:.2f} m per image (`ground_tile_m`; the scan is {SANDY_GRAVEL_TILE_M:.2f} m across)"
    bands = (f" Beyond the tile the flat ground repeats the {t:.2f} m texture image, which shows as regular stripes over the "
             "far ground, plainly in the hero frames and still visible in the README clip at 320x240." if t else "")
    look = take.fill(f"""
**Look.** Outdoors: Poly Haven's `goegap` HDRI (Goegap Nature Reserve, Namaqualand; 8k) as an importance-sampled dome,
its sun split out as an analytic directional light (`look.sun`), and the CC0 Poly Haven ground texture `sandy_gravel_02`
(Namaqualand too) on the heightfield and the backdrop{tile}. The HDRI's sun stands 46.5 deg high, which the photo fixes;
the dome's yaw (`hdri_yaw_deg` 0) puts it 89 deg to the left of the hero camera's view, so it rakes across the bumps
and every bump has a lit and a shadowed side. The camera is lower than the G1 films' (eye 0.75 m above the pelvis
instead of 1.3 m, looking 0.35 m below it, 18.5 deg down instead of 22): the bumps read in profile around the feet. On
screen they read as uneven, shadow-dappled ground and in the robot's ragged shadow, softer than their 8 cm of relief:
the path tracer shades a heightfield with smooth vertex normals (`metalsim/render/scene_tables.py`, `hfield_mesh`), and
the texture adds its own pebbles.{bands} Thin black slits show on some bumps in the middle distance (seen in this take's
frames and in the first take's, which had the same terrain and renderer); their cause is not investigated.""")
    clip = take.fill(f"""
**README clip.** [g1_bumps.webp](g1_bumps.webp) is the `readme` shot from 1.0 s to 6.0 s (frames 50 to 299), so that
the clip holds{clip_fall} (the first take's clip ran from 0.5 s, its fall being earlier), resampled from 50 to 30 fps and written as MuJoCo Warp writes
its README clips: 320x240 (Lanczos), lossy animated WebP from Pillow at quality 70, 33 ms per frame, looping; 150
frames. Command, run from the repository root after the render stage: `python -m metalsim.render.film.encode webp
runs/film/g1_bumps/readme/frames docs/renders/g1_bumps.webp --start 1.0 --seconds 5 --src-fps 50 --fps 30`.""")
    mjw = take.commit("mujoco_warp", n=7)
    mjw = (f"on the pinned MuJoCo Warp fork commit `{mjw}`" if take.at_pin("mujoco_warp") else
           f"on the MuJoCo Warp fork commit `{mjw}`, not this tree's pin `{take.pin('mujoco_warp', 7)}`")
    warp = (f"Warp at its pin `{take.commit('warp')}`" if take.at_pin("warp") else
            f"Warp at `{take.commit('warp')}`, not its pin `{take.pin('warp')}`")
    this_take = take.fill(f"""
**This take, and where its timings come from.** Recorded, rendered and encoded on {take.day} on the {take.machine},
at metalsim {take.metalsim_at("the uncommitted G1 bumps work")}, {mjw} ({warp}). This was not a timing run: MetalSim's
timing runs are made on the 16-inch through the GPU queue's timing kind, and its throughput figures are in
`docs/PARITY.md` §1.4 and §1.8; quote nothing on this page as MetalSim's speed. Power, as the stages stamped it
("Hardware and pins" below): {take.power_prose()}. The measurements, the record stage and the render stage (both shots)
ran as GPU-queue jobs of kind render into `runs/film/g1_bumps`, one at a time, the record and the render in one job
after the take check; then `--stage encode --shots hero`, the README WebP command above and `--stage doc` (CPU). A new
recording is another take, not this one: the GPU rollout is not bitwise repeatable, so how many robots fall, which one
is filmed, and its path can change.""")
    return G1_BUMPS_STORY + f"""
{the_take}

{gait}

**What the model cannot show.** Only the four sole spheres of each foot collide; the visual sole between them does
not. On bumps 10 cm apart a bump can rise between the spheres: in stance on Isaac's terrain the terrain stands above
the quadrilateral of the sphere bottoms by 15 mm at the median, 44 mm at the 95th percentile and 81 mm at most in the
MetalSim measurement run (`mjw_787a5b3/gpu_levels.json` `sole_penetration_m`; CPU MuJoCo 15, 49 and 77 mm), so a bump
can show through a foot. In the hero frames checked at full resolution (100 to 111 and 180, both feet) the planted
foot's toe or heel looks a little sunk into the ground at times, no foot floats in stance, and no bump shows through a
sole. That is Unitree's model, unchanged.

{look}

**Camera.** Both shots track the pelvis as Isaac's `ViewerCfg(origin_type="asset_body", body_name="pelvis")` does, the
tracked point smoothed by a centred 1 s moving average, with Kit's default perspective lens (36 deg vertical field of
view), as in the G1 films (G1_UNITREE.md), but lower. `hero` (1280x720; the video and stills on this page): eye (2.6,
-2.0, 0.75) m from the pelvis, looking at (0, 0, -0.35) m, 3.46 m away, 18.5 deg down. `readme` (640x480, 4:3, for
the README grid): the same direction and elevation at 0.75 times the distance (2.60 m), looking at (0, 0, -0.15) m so
the standing robot fits the 4:3 frame, plus MuJoCo Warp's README orbit of 10 deg/s about the vertical through the
look-at point, symmetric about the middle of the 8 s film: -40 deg at 0 s, 0 at 4 s (the hero's direction), +40 deg at
8 s, counter-clockwise seen from above (`mujoco_warp/record.py` at `d41a47c0`, lines 43 and 142: 0.05 deg per 5 ms
physics step). When the robot falls the tracked pelvis drops with it, and so does the camera.

**Rendering.** Tier 2 (the path tracer), 64 samples per pixel (16 passes x 4), 3 bounces, Open Image Denoise, ACES
tone map, sRGB. The render twin is Unitree's `scene.xml` with the heightfield and the backdrop plane, its STL meshes
posed per frame from the recorded joint state; the colliding copies and the foot spheres are hidden (`rec.json`
`twin.visibility`), and the meshes keep Unitree's two greys.

{clip}

{this_take}
"""

G1_BUMPS = FilmTask(
    name="g1_bumps",
    adapter="metalsim.render.film.unitree:UnitreeG1",
    venv="main",
    checkpoint=f"{UNITREE_RL_GYM['root']}/{UNITREE_RL_GYM['policy']}",
    # MetalSim's own physics: the adapter's default engine, MuJoCo Warp (BatchSim). Since the fork's heightfield fixes
    # (93e5ce1; EPA horizon 787a5b3) its contacts for the G1's 5 mm sphere feet match the exact contact set as MuJoCo C
    # does (runs/film/g1_bumps/measure/mjw_787a5b3/contact_check.json) and the policy falls about as often as on CPU
    # MuJoCo (mjw_787a5b3/falls_summary.json); the first take (2026-10-31 afternoon) ran on "mujoco_c" because the fork
    # then kept most robots walking on dropped contacts
    adapter_kw={"cmd": (0.5, 0.0, 0.0), "terrain": G1_BUMPS_TERRAIN, "starts": G1_BUMPS_STARTS, "pick": "typical",
                "film_lanes": G1_BUMPS_FILM_LANES},
    title="G1 on rough ground, Unitree's policy",
    summary=("Unitree G1 walking onto Isaac Lab's random rough ground (the `random_rough` sub-terrain of "
             "`ROUGH_TERRAINS_CFG`: bumps of 2 to 10 cm every 0.1 m) with Unitree's own pretrained walking policy, which "
             "is blind and was trained on flat ground (unitree_rl_gym, `deploy/pre_train/g1/motion.pt`, BSD-3-Clause), "
             "played by Unitree's MuJoCo sim2sim loop ported to metalsim's MuJoCo Warp on Metal (fork `787a5b3`, whose "
             "heightfield contacts match the exact geometry), on Unitree's own G1 model, and rendered by metalsim's path "
             "tracer. Rendered on the 14-inch MacBook Pro (Mac16,6), not in a timing run, so its Performance table is "
             "not a timing figure."),
    notes=_g1_bumps_notes,
    viewer=G1_BUMPS_VIEWER,
    # 16 robots on 16 lanes of the tile, 8 s; "auto" with pick "typical": the take's typical outcome (a fall when at
    # least half fall); the render keeps playing the simulation after a fall (after_end "play")
    film=FilmCfg(num_envs=16, envs="auto", grid=1, seed=0, max_seconds=8.0, video_length=401, hold_s=0.0, passes=16,
                 spp=4, bounces=3, output_filename_prefix="g1_bumps", allow_early_end=True, after_end="play"),
    # goegap (Namaqualand, outdoor midday) with its sun as an analytic light; sandy_gravel_02 (Namaqualand too) at its
    # scanned size, 2.53 m per tile (Poly Haven "dimensions"), on the heightfield and the backdrop plane
    look=LookCfg(hdri="hdri/demo/goegap_8k.hdr", hdri_yaw_deg=0.0, sun=True,
                 ground_texture="tex/demo/sandy_gravel_02_diff_2k.png", ground_geoms=("floor", "backdrop"),
                 ground_tile_m=SANDY_GRAVEL_TILE_M),
    assets=[
        AssetRef("Unitree G1, 12-dof legs model (g1_12dof.xml in scene.xml) and its STL meshes", "Unitree Robotics "
                 "unitree_rl_gym", f"{UNITREE_BLOB}/resources/robots/g1_description", "BSD-3-Clause", FETCH_UNITREE),
        AssetRef(f"Policy: Unitree's pretrained G1 walking policy deploy/pre_train/g1/motion.pt (TorchScript; sha256 "
                 f"{UNITREE_RL_GYM['policy_sha256'][:8]}...)", "Unitree Robotics unitree_rl_gym",
                 UNITREE_RL_GYM["policy_url"], "BSD-3-Clause", FETCH_UNITREE),
        AssetRef("Sim2sim playback loop and config (deploy_mujoco.py, configs/g1.yaml), ported to MuJoCo Warp",
                 "Unitree Robotics unitree_rl_gym", f"{UNITREE_BLOB}/deploy/deploy_mujoco", "BSD-3-Clause",
                 FETCH_UNITREE),
        AssetRef("Terrain: Isaac Lab 2.3.2 ROUGH_TERRAINS_CFG['random_rough'] (HfRandomUniformTerrainCfg, noise_range "
                 "(0.02, 0.10), noise_step 0.02, border_width 0.25), one 8 x 8 m tile with the generator's 20 m border, "
                 "0.1 m / 5 mm scales, env seed 0", "Isaac Lab task config (port: metalsim.learn.isaac_terrain)",
                 "https://github.com/isaac-sim/IsaacLab", "BSD-3-Clause",
                 "procedural (metalsim.render.film.unitree.random_rough_tile)"),
        AssetRef("Outdoor HDRI (goegap, 8k; Namaqualand, South Africa; its sun drawn as an analytic light)",
                 "Poly Haven (Greg Zaal)", "https://polyhaven.com/a/goegap", PH_CC0, FETCH_PH_DEMO),
        AssetRef("Ground texture (sandy_gravel_02, 2k diffuse; scanned 2.53 x 2.53 m, Namaqualand collection)",
                 "Poly Haven (Dario Barresi)", "https://polyhaven.com/a/sandy_gravel_02", PH_CC0, FETCH_PH_DEMO),
    ],
    shots=[ShotCfg("hero", 1), ShotCfg("readme", 1, G1_BUMPS_README_VIEWER)],
    doc="docs/renders/G1_BUMPS.md",
)


TASKS = {t.name: t for t in (FRANKA_POUR, G1_FLAT, G1_ROUGH, G1_UNITREE, G1_BUMPS)}

# LimX TRON1 with LimX's own RL policy: its entry lives with its adapter (no simulator import at module level)
from metalsim.render.film.tron1 import TRON1  # noqa: E402

TASKS[TRON1.name] = TRON1

# so101Sim's SO-101 cube lift (its PPO teacher on three servo models; the deployment-style scripted expert on Rhoban
# BAM's identified 7.4 V servo, "7v4_bam"): the entries live with their adapter (no simulator import at module level)
from metalsim.render.film.so101 import SO101_TASKS  # noqa: E402

TASKS.update({t.name: t for t in SO101_TASKS})


def get(name: str) -> FilmTask:
    if name not in TASKS:
        raise KeyError(f"no film entry {name!r}; known: {sorted(TASKS)}")
    return TASKS[name]

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

G1_FLAT = FilmTask(
    name="g1_flat",
    adapter="metalsim.render.film.adapters:G1Flat",
    venv="main",
    checkpoint="runs/policies/g1_flat_1500.pt",
    title="G1 flat-terrain walk",
    summary="Unitree G1 walking on flat ground with a policy trained on Isaac Lab's G1 flat velocity task (MuJoCo Warp).",
    viewer=ViewerCfg(eye=(2.6, -2.0, 1.3), lookat=(0.0, 0.0, -0.05), origin_type="asset_body", body_name="pelvis",
                     fovy=KIT_FOVY_16_9, resolution=(1280, 720), smooth_s=1.0),
    film=FilmCfg(num_envs=4, envs=[0], grid=1, seed=3, max_seconds=5.0, hold_s=0.0, passes=16, spp=4, bounces=3,
                 output_filename_prefix="g1_flat"),
    look=LookCfg(hdri="hdri/train/poly_haven_studio.hdr", sun=False, ground_texture="tex/train/laminate_floor_02.png",
                 ground_geoms=("ground", "floor", "ground_plane"), ground_repeat=0.5),
    assets=[
        AssetRef("Unitree G1 (g1_minimal.usd, Isaac 5.1 asset root)", "NVIDIA Isaac Lab assets",
                 "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/IsaacLab/Robots/"
                 "Unitree/G1/g1_minimal.usd", NVIDIA_TERMS, FETCH_ISAAC),
        AssetRef("Studio HDRI (poly_haven_studio, 2k)", "Poly Haven", "https://polyhaven.com/a/poly_haven_studio", PH_CC0,
                 FETCH_PH),
        AssetRef("Floor texture (laminate_floor_02, 1k diffuse)", "Poly Haven", "https://polyhaven.com/a/laminate_floor_02",
                 PH_CC0, FETCH_PH),
    ],
    doc=None,
)

TASKS = {t.name: t for t in (FRANKA_POUR, G1_FLAT)}


def get(name: str) -> FilmTask:
    if name not in TASKS:
        raise KeyError(f"no film entry {name!r}; known: {sorted(TASKS)}")
    return TASKS[name]

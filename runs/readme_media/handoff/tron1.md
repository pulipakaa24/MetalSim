# Hand-off: tron1 (2026-10-31)

## Status
**Done.** I filmed LimX's TRON1 (WF_TRON1A) driven by LimX's own RL policy in MetalSim's TRON1 sim, at the
**nominal** realism settings, unchanged. The policy is unchanged too.
- It does not fall: 0 of 16 robots fell in the 8 s measurement, and 0 of 16 in the 7 s take.
- Both shots are rendered, the hero is encoded, and the README WebP was cut with the brief's exact command.
- Doc and README row are written; the three test files pass.

Nothing is committed. README.md is untouched.
- Worktree: `.claude/worktrees/agent-a47b2178f152b22ab`.
- Patch of the full diff, binaries included: `runs/readme_media/handoff/tron1.patch`.
  - It is against `ed66d8ef` (`g1-unitree-sim2sim`). The worktree had been created at `main` (`ef346e7e`), so I
    fast-forwarded it to that branch first. That made no commit.
  - `git apply --check -R` passes in the worktree.

## Measurements (nominal settings, `SimParams.nominal()`)
Command schedule `tron1.SCHEDULE`, within `evaluate.py`'s ranges:

| from | vx | yaw rate | phase |
|---|---|---|---|
| 0 s | 0 | 0 | stand |
| 0.5 s | 1.0 m/s | 0 | drive |
| 2.0 s | 0.8 m/s | 0.6 rad/s | turn left |
| 3.5 s | 0.4 m/s | 0 | slow down |
| 4.5 s | 0 | 0 | stop |

Error definitions are `evaluate.episode`'s, sampled every 20 ms.

| run | falls | vx RMSE, median (range) | yaw-rate RMSE, median (range) | pitch p95 | roll p95 |
|---|---|---|---|---|---|
| measurement, 16 robots x 8 s | **0/16** | 0.240 m/s (0.235-0.247) | 0.130 rad/s (0.127-0.133) | 14.3 deg | 3.4 deg |
| the take, 16 x 7 s | **0/16** | 0.264 m/s | 0.143 rad/s | 14.3 deg | 3.4 deg |
| TRON1.md eval, nominal (100 x 20 s, random commands and shoves) | 0/100 | 0.27 | 0.17 | 15 | 5 |

- The take reproduces the measurement frame for frame: the filmed robot's per-frame vx differs by 0.0 over 351 frames.
- Behaviour, medians over the 16 robots:
  - slow acceleration: 0.63 m/s from 1.0 to 1.5 s after the 1.0 m/s step, and 0.8 m/s first reached at about 2.0 s;
  - in the turn, 0.87 rad/s of yaw against the 0.6 commanded (74 deg of heading against 52), at 0.64 m/s;
  - 0.38 m/s against the commanded 0.4;
  - stopped, it creeps at 0.03 m/s.
- These are the policy as published, under the port's training-scale commands (`limx_policy.CMD_SCALE`). I did not tune
  anything.
- Filmed: robot 8, the most typical tracker (`Tron1Limx.rank`: closest to the median errors), not the best one.

## Changed
- `metalsim/render/film/tron1.py` (new, Apache-2.0).
  - The `Tron1Limx` adapter runs one `Tron1Sim` and one `LimxPolicyController` per robot, stepped by
    `Tron1Sim.run`, which is the evaluation's own path.
  - It checks the sha256 of LimX's files.
  - It writes `rec.json` `adapter.tracking`: per-robot RMSE and fall time, the summary, and the filmed robot's
    per-frame vx, yaw rate, errors, pitch, roll and fallen flag.
  - The file also holds the `TRON1` FilmTask with its notes. Its module level imports no simulator.
- `metalsim/render/film/tasks.py`: 3 lines at the end, an import of `TRON1` and `TASKS[TRON1.name] = TRON1`. The
  TASKS tuple line is unchanged.
- `metalsim/tron1/sim.py`: `build_model` is split into `build_spec(...)` followed by `.compile()`, with no change in
  behaviour. It lets the render twin be the simulated model itself. This file is outside the brief's list.
- `metalsim/render/film/__main__.py`: the doc's Reproduce block prints the task's own Poly Haven fetch command. The
  output for the other 4 tasks is byte-identical (checked). Outside the brief's list.
- `scripts/fetch_polyhaven_assets.sh`: `concrete_floor_worn_001` added to `HELDOUT_TEX`.
- `docs/renders/TRON1.md`: written by the doc stage.
- `docs/renders/README.md`: one film-table row.
- Media:
  - `docs/renders/tron1.mp4` (1.2 MB);
  - `docs/renders/tron1_still{1,2,3}.jpg` (frames 88, 175 and 280);
  - `docs/renders/tron1.webp` (1.02 MB, 150 frames, 320x240, PSNR 36.88 dB). Its sidecar was moved to
    `runs/readme_media/assets/tron1.webp.json`, as for the other cells.
- `runs/film/tron1/`: `MACHINE` (14-inch), `rec/rec.json` (5005 lines, mostly the per-frame series), and
  `{hero,readme}/render.json`, `render_frames.jsonl` and `hero/encode.json`.
- Environment, not in git: I installed `onnxruntime==1.31.0` into `.venv` (main checkout).
  - It is a declared `tron1` extra in `pyproject.toml` and part of `full`. The main venv lacked it, so
    `limx_policy.py` could not import.
  - The pip dry-run showed only additions (onnxruntime, flatbuffers 25.12.19, protobuf 7.36.2), no upgrades.
  - 1.31.0 is the version in `~/metalsim-fresh/MetalSim/.venv`.

## Verified
**Sources**
- LimX upstream at `035e4c9e2f24e916fa5e8363ac6f0f788a2e1e8c`, fetched from raw.githubusercontent:
  - `controllers/model/WF_TRON1A/policy/isaacgym/{policy,encoder}.onnx` and `params.yaml` are byte-identical to
    `assets/tron1/limx_policy/WF_TRON1A/`. sha256 `bfc4617f...` (policy), `f79e29d1...` (encoder), `85b5431a...`
    (params).
  - `main.py` lines 27-39: the default IP 127.0.0.1 means LimX's simulator; an IP argument means the robot.
    `RL_TYPE` is isaacgym or isaaclab with no default; the README's example is isaacgym.
- Licences:
  - LimX's files: Apache-2.0 (`assets/tron1/LICENSE-*`, `THIRD_PARTY_NOTICES.md:212-225`).
  - Poly Haven: CC0 for all assets, per <https://polyhaven.com/license> ("Our assets are all licensed as CC0").
  - The concrete's md5 `8fb2856c...` and size 5,252,392 bytes match `api.polyhaven.com/files`.
- Concrete size: 3 m x 3 m (`api.polyhaven.com/info`).
- Floor scale: `ground_repeat` 2/3 gives one 3 m tile.
  - Source: an infinite plane's UV is 0.5 x position x texrepeat (`metalsim/render/builtin_geoms.py:193-195` and
    `276-296`, computed with those functions).
- Exposure fixed at 1.4. The automatic rule (median luminance to 0.2) gave 2.605 / 2.716 because the concrete's median
  linear albedo is 0.095 (laminate: 0.237); 1.4 = 2.6 x 0.095/0.18. Measured from the texture; the choice was then
  confirmed by eye.

**Commands**
- Measurement (CPU, not queued): `Tron1Limx`, 16 robots, 400 steps, via `python -c` from the worktree. Output:
  `runs/readme_media/scratch/tron1/measure_16x8s_nominal.json`.
- Queue jobs, all of kind render, run from the worktree:
  - `tron1-record`: `--stage record`, 11 s;
  - previews `tron1-preview` and `tron1-preview2`, written to scratch;
  - `tron1-render`: `--stage render`. Cache keys: hero `e1a4c03b590ee6209f78`, readme `9b550a6429345a128742`;
  - `tron1-tests`.
- CPU stages: `--stage encode --shots hero`; the brief's WebP command with `--src-fps 50`; `--stage doc`.
- Tests: `pytest -q tests/test_film.py tests/test_no_machine_paths.py tests/test_tron1.py`: 21 passed, 1 skipped.
  - The skip is `test_newton_twin` (no newton in the main venv).
  - `test_tron1` passed before (16) and after `build_spec`.
- Machine paths: a grep with the test's pattern over every new or changed file found none. `rec/twin.xml` has an
  absolute meshdir but is gitignored.
- Logs: `runs/readme_media/scratch/tron1/logs/`.

**Power**, read live: 14:37 AC, 80 %, not charging, `powermode 2`. Stamps: `rec.json` at 14:38 (80 %, AC); hero
14:45 and readme 14:46 (84 %, charging). Mac16,6. No timing is quoted.

**Frames I looked at** (Read tool; all in `runs/readme_media/scratch/tron1/look/`)
- `hdri_compare.jpg`. `machine_shop_02`, a concrete hall with yellow gantry cranes, reads industrial.
  `artist_workshop` is a painter's studio with parquet, so I chose the heldout HDRI.
- `preview1_hero.jpg` and `preview1_hero140_crop.png`, at automatic exposure 2.6: the robot washed to white, the
  wheels near white.
- `preview2_AB.jpg`, at exposure 1.4:
  - A: the G1 pitch of 21 deg; all floor.
  - B: a 12 deg pitch; the machine shop shows in the top sixth. I chose B.
- `webp_strip.jpg`, every 15th WebP frame:
  - the robot fills about 3/4 of the height;
  - drive (leaning forward, legs splayed), then the turn with the orbit swinging the view, then the stop;
  - the machine shop across the top; no clipping.
- `hero_consecutive_full.jpg`: hero frames 75-86, 12 consecutive full frames. The robot is centred and about 57 % of
  the frame height. The lidar top sits at the horizon line.
- `hero_consecutive_crop.jpg`: the same frames at native resolution around the wheels.
  - The near wheel's three-spoke hub changes orientation every frame, so the wheels visibly turn.
  - The recording says 5.4 deg per frame from 1 to 2 s.
  - The tyres sit on their contact shadows.
- `stills.jpg`, frames 88, 175 and 280: the forward-leaning drive; side view at the end of the turn with bent knees;
  standing.
- `scale_hero.jpg` and `scale_readme.jpg`:
  - each wheel's real 0.1298 m rim, projected from its recorded pose through the film camera, lies on the rendered
    tyre outline (hero frames 88 and 175, readme 150);
  - predicted on-screen wheel height: 98-115 px (hero);
  - a 1 m ruler is drawn for reference.
- Scripts: `scratchpad/{strips,scale_check}.py` in the session scratchpad.

## Open
- **Wording.** This is LimX's *open* policy. LimX's deploy repository runs it on the robot through the SDK. It is not
  the closed controller of the robot's Remote Control Mode (docs/TRON1.md). The doc says so; keep that in the README.
- **Colours.** LimX's `rgba` are read as linear albedo, the films' convention for every MuJoCo model. So the 0.27 grey
  wheels render light grey, and the body white. A renderer-wide decision on sRGB vs linear `rgba` is out of scope.
- **G1 floors (not my stream; from code reading only).** By the same plane mapping, the G1 films' laminate, 1.7 m
  physical with `ground_repeat` 0.5, tiles every 4 m, about 2.35 x too large.
- **README row 2** has three cells today (Franka-Pour plus the parity pair). Where the TRON1 cell goes is the
  orchestrator's call.
- **Generic doc text.** "What is shown" lists 4 kept envs although only env 8 is filmed. The record row says "incl.
  graph capture", which does not apply here.
- **Provenance gaps.**
  - The take ran on uncommitted code (`ed66d8ef3e+dirty`).
  - The pins omit onnxruntime; the notes state it.
- **Not investigated.** Whether the yaw overshoot comes from the port's training-scale commands rather than LimX's
  joystick mapping. TRON1.md records that difference.

## Next
For the orchestrator:

1. Apply `tron1.patch` (or take the worktree diff).

2. Add this README cell:
```html
    <td align="center" width="33%">
      <img width="320" src="docs/renders/tron1.webp" alt="LimX TRON1 wheel-legged robot driving, turning and stopping under LimX's own RL policy in MetalSim's realistic TRON1 simulation, path traced">
      <br><b>python -m metalsim.render.film tron1</b>
    </td>
```

3. Add this sentence to the row-2 paragraph:
   > Then LimX's TRON1 wheel-legged robot driven by LimX's own open RL policy (the ONNX export in LimX's
   > `tron1-rl-deploy-python`, Apache-2.0) in MetalSim's TRON1 simulation at its nominal realism settings (command and
   > sensor latency, quantized encoders, joint friction, IMU noise, a 1.5 kg sensor pack), unchanged: none of 16
   > simulated robots fell over the 8 s drive, turn and stop, with 0.24 m/s and 0.13 rad/s RMS speed and yaw-rate
   > tracking error ([TRON1.md](docs/renders/TRON1.md)).

4. Add a `### TRON1 (tron1.webp)` entry to docs/renders/README.md, "The README's media". The command and sidecar are
   above.

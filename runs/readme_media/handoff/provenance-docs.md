# Hand-off: provenance-docs (2026-10-31)

## Status
**Done.** All six items of the brief are complete, and the coordinator's mid-task change (comparison cells are now videos) is in the index. The test sweep has **194 passed, 0 failed, 8 skipped**, so no worktree comparison was needed. Nothing is committed.

## Changed
- **Canonical takes.** `runs/readme_media/film/g1_{flat,rough}` were moved to `runs/film/g1_{flat,rough}`; nothing was re-recorded or re-rendered.
  - The old `runs/film/g1_flat` (the misattributed 14-inch re-film, plan §0 bug 3) was moved to `runs/readme_media/scratch/provenance-docs/old_runs_film_g1_flat/`, which is ignored. HEAD still has the committed JSON.
  - `runs/readme_media/film/g1_{flat,rough}` are now symlinks to the new location, so older hand-offs and asset JSON paths still resolve.
  - `runs/film/g1_flat/MACHINE` now reads `14-inch` (was `16-inch`). `runs/film/g1_rough/MACHINE` is new.
- **`.gitignore`.**
  - Adds `runs/film/**/stale/` (where `render.py _invalidate` moves old frames).
  - Adds `runs/readme_media/*`, which is everything there except `PROTOCOL.md` and `handoff/*.md`. `handoff/film-render.patch` is now ignored.
- **Trackable now in `runs/film/g1_{flat,rough}`.** `rec/rec.json`, `{hero,readme}/render.json`, `render_frames.jsonl`, `hero/encode.json` and `MACHINE`. `twin.xml`, `rec.npz`, `frames/` and `master.mp4` are ignored (checked with `git check-ignore`).
- **`docs/renders/`.**
  - Re-encoded by the film tool: `g1_flat.mp4` (replaces the bad re-film) and `g1_flat_still{1,2,3}.jpg`.
  - New from the film tool: `g1_rough.mp4` and `g1_rough_still{1,2,3}.jpg`.
  - New from the doc stage: `G1_FLAT.md` and `G1_ROUGH.md`.
  - `README.md` was rewritten as the index of every README cell. Its old content is kept; the G1 rows are updated.
- **`metalsim/render/film/tasks.py`.**
  - `G1_FLAT` and `G1_ROUGH` get a `doc` path, `summary`, `notes` and a checkpoint `AssetRef`.
  - New prose helpers: `G1_CAMERA_NOTES`, `G1_RENDER_NOTES` and `_g1_take_notes()`.
- **`docs/gallery/scripts/14_g1_policy_video.py`.**
  - The default checkpoint is now `runs/g1regress/ckpt14/flat_main_s0.pt`, and the docstring is updated.
  - Beyond the brief, physics dt is now 2.5 ms in both the task and the hero model. That is the checkpoint's training dt; the script's 5 ms default would run the policy off its training setting.
- **`metalsim/parity/side_by_side.py`.** The docstring example now uses `--ckpt runs/g1regress/ckpt14/flat_main_s0.pt`, which is iteration 500 and matches the example's label.

## Verified
- **Encode stage.**
  - Commands: `.venv/bin/python -m metalsim.render.film g1_flat --stage encode --shots hero`, and the same for `g1_rough`.
  - Logs: `scratch/provenance-docs/logs/encode_*.log`.
  - Output: 301 frames each at 1280x720 and 50 fps, 6.0 s. `g1_flat.mp4` is 1.95 MB and `g1_rough.mp4` is 2.28 MB.
  - Stills are frames 75, 150 and 240.
- **What I looked at.**
  - Strips: `scratch/provenance-docs/strip_g1_{flat,rough}_mp4.jpg`, at 1 fps, 6 frames each.
  - All six stills at full size.
  - The existing `scratch/g1_flat_readme_strip.png`, which shows the orbit: the plank direction turns.
  - What they show:
    - Flat: the robot walks upright in every frame, the floor is complete to the horizon, and there is no grey box or wedge.
    - Rough: the robot climbs the steps of the inverted-stairs pit and stays upright, with clean treads and risers.
  - Parity WebPs: a 5-frame strip of each (`strip_parity_webps.jpg`). The robot is upright and in frame in both.
- **Doc stage.**
  - Commands: `.venv/bin/python -m metalsim.render.film g1_{flat,rough} --stage doc` (`logs/doc_*.log`).
  - Every number in the prose was checked against its source:
    - `rec.json` and the checkpoint keys: obs 123/310 and hidden (256,128,128)/(512,256,128).
    - Log headers and the last lines of `runs/g1regress/ckpt14/*.log`, and `replay_main_flat_it500.log`.
    - `ppowarp_checkpoint_load_2026-10-13.md`.
  - The sha256 of each `.pt` file matches `rec.json` (`shasum -a 256`).
- **Orbit (10 deg/s).** google-deepmind/mujoco_warp at `d41a47c0` (clone in this session's scratchpad):
  - `record.py:43`: `cam_azimuth_speed` 0.05 deg per step.
  - `record.py:142`: `azimuth = 90 + (step - nstep/2) * speed`.
  - `benchmarks/unitree_g1/unitree_g1_mjlab.xml:14`: timestep 0.005, included by both `scene_flat.xml` and `scene_hfield.xml`.
  - So the rate is 10 deg/s. The film-record hand-off's "1.5 deg/s at 30 fps" was wrong; `tasks.py` already had 10.
  - `render.py camera()` rotates counter-clockwise about +z, symmetric about t_mid = 3.0 s, which matches MuJoCo's azimuth growing. The direction comes from reading the code, not from an experiment.
- **Reproduce commands.** Re-run on CPU, each byte-identical to the file in `docs/renders` (`md5` / `cmp`):
  - `g1_flat.webp` and `g1_rough.webp`, from `runs/film/*/readme/frames` with `--start 0.5 --seconds 5 --src-fps 50 --fps 30`;
  - `anymal_sand`, `franka_pour` and `lidar_glass_{rgb,depth,lidar}`, with the commands in their JSON;
  - `parity_g1_{isaac,metalsim}`: ffmpeg `-ss 0.5 -t 4 ... -vf crop=540:405:{340|1364}:166 -start_number 0`, then encode at `--src-fps 25 --fps 25`. Script: `scratch/provenance-docs/parity_repro.sh`. ffmpeg gives 540x404 frames, matching the JSON;
  - both chart SVGs (`throughput_chart.py --out <dir>`).
  - Re-runs: `scratch/provenance-docs/{webp_repro,parity_repro,chart_repro}/`.
- **Test sweep.**
  - Command: `scripts/gpu_run.sh provenance-test-sweep render 120 -- runs/readme_media/scratch/provenance-docs/tests/run_sweep.sh`. One queue job.
  - It ran the 26 files in `tests/` whose names match the brief's patterns (`tests/files.txt`).
  - Result: **194 passed, 0 failed, 8 skipped**, 158.6 s (`sweep.log`, `junit.xml`).
  - The 8 skips:
    - Newton is not in the main venv: `test_render_newton_bridge`, `test_render_newton_perworld` and `test_film:102`.
    - 5 in `test_render_splat_public` for absent fetched scenes (InteriorGS, dxgl, NuRec).
  - My first submission ran 0 tests because zsh did not split the file list; the rerun is the one counted.
- **Comparison-cell facts.**
  - PARITY §1.5, lines 314-317: 3.11 / 3.21 m, and pelvis 0.62-0.64 / 0.61-0.63 m for checkpoints 500/1000/1499.
  - The `side_by_side.py` start is Isaac's default state with command (0.5, 0, 0) at 2.5 ms.
  - The source video comes from `scripts/gallery/g1_stage_videos.sh` and was re-rendered in `a14b6a87`.
  - The Isaac checkpoint and play recording are not on this Mac.

## Open
- **Timing note sits in prose, not the table.**
  - It is the summary's last sentence plus a notes paragraph.
  - The doc stage cannot put "on battery, Low Power Mode" inside the Performance table: `doc.py` and `provenance.hardware()` record no power state, and they are library code.
  - The text describes this take, so a re-film on the 16-inch must edit `tasks.py`.
  - Suggested library fix: record `pmset -g batt` and the low-power state in `hardware()`, and print it under the table.
- **`doc.py` / `record.py` quirks**, explained in the prose but not fixed:
  - "What is shown" says "filmed: 6, 2, 3, 0". That is the top four of the ranking; only env 6 (rough: env 8) is filmed.
  - Per-env `base_z`, `level` and `col` are zero placeholders for episodes still running: `record.py:329-331` fills them only at an episode's end.
- **Docs are committable only with the checkpoints.** The checkpoint `AssetRef` says "committed" and the docs point at the `.pt` files, so the two `.pt` files must go into the same commit.
- **`14_g1_policy_video.py`.**
  - It still writes `docs/gallery/g1_tier2_policy.mp4`, which was removed from the gallery today.
  - Not run: it is a GPU job and would recreate that 68 MB file. Only `py_compile` was checked.
- **Asset JSONs.** The `runs/readme_media/assets/*.json` files are now ignored. The index copies each command inline. The G1 WebP JSONs still name `runs/readme_media/film/...`, which the symlinks keep valid locally.
- **Stale elsewhere (not my files).**
  - `docs/DEMO.md:37` ("Film: docs/renders/g1_flat.mp4 ... runs/checks/fix1/") describes the old falling film.
  - Plan §6 describes the re-film that has been replaced.

## README text vs these docs (for your check)
1. **G1 captions.** `python -m metalsim.render.film g1_flat` / `g1_rough` does not write `g1_*.webp`.
   - It writes `g1_*.mp4` and `g1_*_readme.mp4`.
   - A new recording is a different take, because the GPU rollout is not bitwise repeatable.
   - The exact command is the `encode webp` line in `docs/renders/README.md` and G1_*.md.
2. **ANYmal caption.** `python scripts/particle_demo/render_anymal.py` alone does not reproduce the clip.
   - It needs `--sim` from `sim_anymal.py` (Newton venv) and the demo-scale flags; DEMO.md has both.
   - The WebP is cut from a clip outside the repo (`~/metalsim-fresh/runs_render/sand25m/demo/frames`). That clip was rendered on 2026-10-13, a re-run at the settings of the PARTICLES.md 2026-10-12 result.
3. **Franka alt text** says "735 MPM grains". The docs say 735 MPM particles per cup, each drawn as 4 render grains.
4. **MetalSim parity caption.** `python -m metalsim.parity.side_by_side` needs `--isaac/--ckpt/--out`.
   - Isaac's recording and `model_1499.pt` are not in the repo or on this Mac.
   - It writes the two-panel mp4. The WebP is a crop of the tracked `docs/gallery/g1_stage_isaac_it1499.mp4`.
5. **Row 2's "28.8 dB PSNR from RTX's path tracer on the robot"** is correct for §1.7 kinematic replay, but the left cell is RTX real-time. Against real-time the robot figure is 24.0 dB / SSIM 0.86 (PARITY §1.7 table).
6. **README line 11** says "measured on an Apple M4 Max ... unless marked". The G1 and lidar clips were made on the 14-inch on battery in Low Power Mode. The docs say their timings are not figures; the README does not mark this.
7. **README lines 139-140** name "G1 flat and rough" without links. `docs/renders/G1_FLAT.md` and `G1_ROUGH.md` now exist.

## Next
1. Fix the README points above. Then commit together:
   - the `docs/renders` media and docs;
   - `runs/film/g1_{flat,rough}` (the JSON files and `MACHINE`);
   - `runs/g1regress/ckpt14/*.pt`;
   - `.gitignore`, `tasks.py`, the two edited scripts and the two new tests;
   - the hand-offs.
2. Optional library follow-up:
   - record the power state in `provenance.hardware()` and print it in `doc.py`'s Performance section;
   - make "What is shown" list only the filmed env(s);
   - write NaN instead of 0 for episodes still running.
3. Optional: give `14_g1_policy_video.py` an output argument that defaults outside `docs/gallery`.

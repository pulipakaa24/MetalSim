# Hand-off: film-record (2026-10-31)

## Status
**Done.** The G1 done mask, the early-end guard with its opt-out, checkpoint provenance, tracked checkpoints, a new
G1_ROUGH film task, the stray-geometry fix and a 4:3 `readme` shot are in place. Final takes are recorded (record stage
only). The camera orbit is not done: it needs `render.py` (see Next).

## Changed
- `metalsim/render/film/adapters.py`: new `G1Velocity` adapter, with `G1Flat` and `G1Rough` subclasses.
  - Done mask: `done` now comes from the task's own `launch_reward_done_reset` and `launch_timeouts`. The frame is
    captured before the task resets done envs. `info` gains `terminated` and `time_out`.
  - Task settings: the ones saved in the checkpoint, plus `adapter_kw["task"]`. `adapter_kw["terrain_level"]` puts
    every env on one level.
  - Ranking (flat): envs that didn't terminate first, then envs whose pelvis never dropped more than 0.25 m, then the
    longest path. Ranking (rough): envs that didn't terminate first, then the most climbed, then the longest path.
  - `summary()` writes one line per env to the record log.
  - Twin: it carries the task's own terrain geometry. That is the task's collision heightfield, plus Isaac's 14,080
    terrain boxes merged into one mesh, `terrain_boxes`, with planar UVs on each face.
- `metalsim/render/film/record.py`:
  - Early-end guard: raises `FilmEarlyEnd` with the env, frame and the envs that survived.
  - With `envs="auto"`, terminated envs rank last and are never kept as spares.
  - `rec.json` gains `checkpoint` {path, in_repo, sha256, bytes}, `allow_early_end` and `early_end_envs`.
  - New `mujoco_visibility()` hides the twin's colliders and sites. It runs in `mujoco_twin`, and the counts are
    written to `twin.visibility`.
  - The log reports terminated envs.
- `metalsim/render/film/cfg.py`: adds `FilmCfg.allow_early_end` (default False) and `FilmTask.adapter_kw`.
- `metalsim/render/film/cli.py`: adds `--allow-early-end`.
- `metalsim/render/film/tasks.py`:
  - G1_FLAT now uses checkpoint `runs/g1regress/ckpt14/flat_main_s0.pt` with `physics_dt` 0.0025, 16 envs, `envs="auto"`
    and 6 s.
  - New G1_ROUGH: `rough_main_s0.pt`, seed 0, terrain level 5, 40 envs, `auto`, 6 s, the same look, and
    `ground_geoms=("ground", "terrain_boxes")`.
  - Both have shots `hero` (1280x720) and `readme` (640x480, the same fov and pelvis tracking, 0.75x closer).
- `.gitignore`: one line, `!runs/g1regress/ckpt14/*.pt`.
- `runs/g1regress/ckpt14/{flat,rough}_main_s0.pt`: new copies (untracked; the negation makes them trackable).
- Not mine: `__main__.py` and `encode.py` show modifications from the other stream.

## Verified
**Checkpoint copies.** The sha256 of each copy matches its source:

| file | sha256 |
|---|---|
| `flat_main_s0.pt` | `6fbfbfe8b950ab27e82afc912fbc3d5484c5b5050dc288cdecd32350b9917057` |
| `rough_main_s0.pt` | `b5b25ad377ba40dbacc18fd90462a64c20203ee7fb6b97b89c3110755cf9ca6f` |

- Both tracked logs equal `~/metalsim-fresh/agents/g1regress/out/*.log` once `$HOME` is normalised.
- `git status` lists both `.pt` files as `??`, so they are no longer ignored.

**Task settings.** Sources:
- the checkpoint's saved keys, read the same way `scripts/diagnostics/g1_policy_replay.py` reads them;
- the log headers, "physics dt 0.0025 (decimation 8), seed 0";
- `runs/ellip_default/run.sh`: `train $IT ... 0.0025 --reward_cfg flat|rough_isaac --seed $S`.

So rough uses seed 0, which generates the training terrain. Each `rec.json` stores the settings under
`adapter.task_resolved`.

**Termination.**
- Isaac Lab 3 at `ae37b028`, `~/metalsim-fresh/MetalSim/upstream/IsaacLab3/source/isaaclab_tasks/isaaclab_tasks/core/velocity/`:
  - `velocity_env_cfg.py:319-326`: `time_out` (time_out=True) and `base_contact` = `illegal_contact` at 1 N.
  - `config/g1/rough_env_cfg.py:139`: the contact body is `torso_link`.
  - There is no base-height term.
- The port: `metalsim/learn/g1_velocity.py:373-391` (`fell` = torso force > 1 N, or a non-finite state) and `:496-504`
  (`g1_record_timeout`).
- `docs/PARITY.md:49`.

**Collider question.**
- The three `*Cube` geoms are true USD colliders, not visuals. A pxr query on `assets/isaac/G1/g1_minimal.usd` shows
  `/g1/{left,right}_ankle_roll_link/Cube` and `/g1/torso_link/Cube` with `UsdPhysics.CollisionAPI` and
  `visibility=invisible`.
- They are the model's only three colliders (`g1_velocity.py` capacity comment: "3 colliders").
- `usd_to_mjcf.py:779-794` gives a collider group 0 and default grey, and ignores USD visibility.
- The visible grey box was not a collider. It was the `torso_link_touch` sensor site (0.3 x 0.3 x 0.5 m, added by
  `build_g1_model`). The recording poses geoms only, so the site stayed where `qpos0` puts it: world (0, 0, 0.15).
  This was checked with `mj_forward` on `runs/film/g1_flat/rec/twin.xml`.
- Sites are drawn because `MjvOption` site groups 0 to 2 are visible (`metalsim/render/extras.py:157`).
- Fix: a body that has visuals moves its colliders to group 3 (the ray tracer's `max_group=3` skips them,
  `scene_tables.py:775`), and every site moves to group 5.

**Box texture.**
- `builtin_geoms.py:249-311`: a mesh without UVs gets the object-linear mapping in its geom frame. MuJoCo aligns the
  mesh frame to its principal axes (`terrain_boxes` compiled quat 0.40, 0.58, -0.39, 0.59), which smeared the tops.
  This is confirmed by experiment (preview `rough_l5_8s`, frame 200).
- With explicit UVs, `texuniform` scales the texcoords by `geom_size`, so the stored UVs are divided by it.
- A CPU check through the renderer's own `material_uv_transform` matches the heightfield's world scale to 2e-6.

**CPU checks.**
- `pytest tests/test_film.py`: 4 passed, 1 skipped (the Newton venv).
- A stand-in adapter (`scratchpad/guard_check.py`) confirmed:
  - an explicitly filmed env that terminated raises;
  - `auto` returns only survivors;
  - the opt-out records and warns;
  - colliders are hidden only on bodies that have visuals;
  - sites are hidden.

**GPU checks.** Queue, kind render, one at a time. Logs are in `runs/readme_media/scratch/film-record/`.

| check | checkpoint | envs, length | result |
|---|---|---|---|
| (a) | old `g1_flat_1500.pt` | 16, 8 s | Guard tripped. Run 1: 11/16 terminated. Run 2: 10/16, env 0 at frame 147 (2.94 s). All 16 pelvises dropped 0.41–0.57 m, so the 6 survivors were down too. Log: `a_old1500.log` |
| (b) | `flat_main_s0` copy | 16, 8 s | 0/16 terminated. Pelvis drop at most 0.11 m; paths 1.5–7.6 m |
| (c) | `rough_main_s0` | 16, 8 s, task's random levels 0–5 | 0/16 terminated |

- Level 5, 40 envs, 8 s: 1/40 terminated (env 4, descending stairs, at 5.42 s). Env 13 climbed 1.07 m without falling.
- Level choice: training ended at a mean terrain level of 5.86 (`rough_main_s0.log`). Level 5 has risers of
  0.14–0.16 m, computed from the generator's per-row difficulty.
- GPU runs are not bitwise repeatable: (a) gave 11 and then 10 falls.

**Final takes.**

```
scripts/gpu_run.sh film-record-final-g1_flat render 15 -- .venv/bin/python -m metalsim.render.film g1_flat --stage record --skip-preflight --out runs/readme_media/film/g1_flat
scripts/gpu_run.sh film-record-final-g1_rough render 15 -- .venv/bin/python -m metalsim.render.film g1_rough --stage record --skip-preflight --out runs/readme_media/film/g1_rough
```

| take | rec dir | frames | filmed env | first-episode result | checkpoint sha256 |
|---|---|---|---|---|---|
| g1_flat | `runs/readme_media/film/g1_flat/rec` | 301 (6 s) | 6 | 0/16 terminated; path 5.61 m, pelvis drop 0.10 m | `6fbfbfe8…` |
| g1_rough | `runs/readme_media/film/g1_rough/rec` | 301 (6 s) | 8 (terrain type 4, inverted stairs, level 5) | 0/40 terminated; climbed 0.75 m, drop 0.01 m | `b5b25ad3…` |

Both takes are upright for all 6 s.

**Previews.** I rendered `--stage render --frames 0:300:100` from copies in `scratch/film-record/preview_{g1_flat,g1_rough}`.
- I looked at:
  - the strips `scratch/film-record/strip_{g1_flat,g1_rough}_{hero,readme}.jpg` (frames 0, 100, 200);
  - full-size `preview_g1_flat/readme/frame_0000.png`;
  - full-size `preview_g1_rough/hero/frame_0200.png`.
- In every frame the robot is upright.
- There is no grey box, and no light foot plates (the feet show only the dark visual mesh).
- In the rough take the robot climbs from the pit floor up several steps, and the planks are clean on the tops and the
  risers.
- At 4:3 the robot fills about 2/3 to 3/4 of the frame height.
- The flat plane wedge is the known render-side artefact and is still visible.

## Open
- **Orbit not implemented.** The README grid's azimuth orbit needs `render.py`'s `camera()`, which is outside my files.
- **Robots that are down are not refused.** The guard keys on the task's termination (torso contact) only. A robot down
  without torso contact is not refused, but on flat it is ranked last (pelvis drop over 0.25 m) and shows in the
  record log's summary line.
- **Large uncovered bulk files.** `runs/readme_media/film/*/rec` holds bulk files (twin.xml 35 MB flat, 72 MB rough,
  plus rec.npz). They are not covered by `.gitignore`'s `runs/film/**` rules, so don't commit them blindly.
- **Same-dir re-records.** Re-recording into the same `--out` still hits the stale-frame reuse (plan bug 3, render
  stage).
- **Docs still to update.** README, GUIDE and the plan still name `runs/policies/g1_flat_1500.pt`.

## Next
1. Render side (other stream): in `render.py camera()`, add `ViewerCfg.orbit_deg_s` (new field, default 0). For
   `origin_type` asset_*, rotate the `eye - lookat` offset about +z through the tracked point by
   `orbit_deg_s * (k - (F - 1) / 2) / fps` degrees. That is symmetric about mid-clip; MuJoCo Warp uses 0.05 deg per
   step, i.e. 1.5 deg/s at 30 fps. Then set it on `G1_README_VIEWER` in `tasks.py`.
2. After the render-side fixes merge, render and encode both takes, shots `hero` and `readme`, from the rec dirs
   above. Don't re-record: those rec dirs are the checked takes.

# Hand-off: isaac-pretrained (2026-10-31)

## Status
**Partial: stopped at step 3 under the brief's stop rule.** Steps 1, 2, 3 and 5 are done; step 4 (switch defaults,
re-film, encode, docs) was **not** done. Isaac's published flat policy, played in MetalSim, steps at **3.88 Hz with a
5.6 / 7.3 cm swing apex**. That is the "about 4 Hz, low clearance" case: the brief's numbers for the shuffling take were
3.99 Hz and 5.6 / 7.2 cm. So nothing was filmed. `runs/film/g1_*`, `docs/renders/*` and the defaults in `tasks.py` are
unchanged.

The research does not support the stop rule's premise ("then the cause is physics or obs layout"):
- The observation layout is right. It was checked in code and docs, and by three experiments (Verified, B).
- About 4 Hz short steps is how Isaac Lab's G1 velocity policies walk **in PhysX too**. Isaac's own L4-trained flat
  checkpoints step at 3.85 / 3.60 / 3.23 Hz (iterations 500 / 1000 / 1499) in Isaac Sim at 0.5 m/s, with strides of
  0.108 to 0.123 m (`docs/research/speculative_contacts_2026-09-27.md` section 2, cadence table).
- MetalSim's physics adds a documented +4 to 6 % to the cadence (same note; PARITY §1.5).
- The cause is therefore mostly the task's own gait (its rewards), plus about 5 % from physics. Neither training
  length nor the obs layout explains it.

Isaac's checkpoints are still measurably better than ours (table below):
- flat: 15 % lower cadence, 30 to 75 % more swing apex, less slip, longer strides;
- rough: regular, about 3.5 Hz steps with twice our apex, where ours steps irregularly.

Whether that is "clearly better" enough to film is your call (Next, option a).

## Changed
- `metalsim/render/film/adapters.py`:
  - `load_g1_checkpoint` reads either format with `torch.load(weights_only=True)`. For rsl_rl it takes the `actor.*`
    keys, as `side_by_side.load_policy` does.
  - `ISAAC_G1_JOINTS` is Isaac's breadth-first joint order. `G1Velocity` permutes the joint pos / joint vel / last
    action blocks and the actions by joint name, as `side_by_side.py` and `g1_rough_transfer.py` do.
  - `ISAAC_TASK_SETTINGS` holds the task settings an Isaac checkpoint implies: `reward_cfg` flat / rough_isaac and
    `scan_ordering` xy. The adapter raises if an rsl_rl policy meets a non-"xy" scan order.
  - `checkpoint_source` maps a fetched checkpoint back to its URL, published sha256 and licence.
  - Each frame now carries `feet` (foot-collider centres, by forward kinematics on the host), `feet_footprint`
    (bounding-box corner xy), `feet_contact` (touch sensor > 1 N) and `cmd`.
  - `ground_height` is `terrain.collision_top`.
  - `rank(first, gait=None)` adds the natural-gait preference (`natural_gait`, below). `meta()` gains
    `policy_format`, `policy_source`, `policy_joint_order` and `foot_geoms`.
- `metalsim/render/film/record.py`:
  - `gait_metrics` / `gait_lines`: per-env gait for every simulated env. The result is printed, with filmed envs
    marked `*`, and stored as `rec.json` `gait` (`definition`, `per_env`, `summary`, `filmed`,
    `natural_gait_envs`). It is passed to `rank`.
  - `rec.json` `checkpoint` gains `format`, `iteration`, `source`, `source_url`, `published_sha256` and `licence`. A
    published hash that does not match raises.
  - `in_repo` now means "tracked by git": a fetched NVIDIA file under the checkout is `false`.
- `metalsim/render/film/tasks.py`: adds `ISAAC51` and `ISAAC_PRETRAINED` (path, URL, sha256, iteration and actor
  shape for both task ids). The G1 defaults are **unchanged**.
- `scripts/fetch_isaac_assets.sh`: fetches both checkpoints to
  `assets/isaac/pretrained/rsl_rl/<task id>/checkpoint.pt`, SHA-256 checked. They are git-ignored by `*.pt`.

**The gait metric** (`record.GAIT_DEFINITION`). Each env and foot is measured over a window from 1 s to the first
episode's end or the clip's end.
- Foot height: the foot-collider centre above the task's collision surface. On rough terrain that surface is the
  highest point under the foot's footprint.
- `cadence_hz`: the peak of a Hann-windowed, 16× zero-padded FFT of the foot height, searched in 0.5 to 10 Hz.
- `touchdowns_hz`: contact onsets per second. This is a terrain-independent cross-check of the cadence.
- `apex_m`: the 95th percentile of the foot height above the median stance height.
- `slip_mps`: the median horizontal foot speed between consecutive frames that are both in contact.
- `air_frac`: the fraction of frames with no contact.
- `stride_m`: speed divided by cadence.

**Natural-gait preference.** Among envs that did not terminate (flat: and did not go down), an env is preferred when
its lower apex is at or above the group median and its larger slip is at or below the median. The rule is relative to
the take's own envs, with no gate, because no reference value exists. It ranks ahead of path length (flat) and climb
(rough).

## Verified
**Gait table.** Isaac's checkpoint against ours, 16 envs, 8 s, window 1 to 8 s, medians over walking envs.
- Settings: the film's own (the task's random commands, training obs noise, the actor mean).
- Rough terrain: level 5, seed 0.
- Jobs: `runs/readme_media/scratch/isaac-pretrained/measure.sh`, one GPU-queue render job.
- Logs: `.../measure/<task>_<who>.log` (per-env rows); `rec.json` `gait` in `.../measure/<task>_<who>/rec/`.

| | flat, ours (500 it) | flat, Isaac (1499 it) | rough, ours (1000 it) | rough, Isaac (2999 it) |
|---|---|---|---|---|
| walking envs / terminated | 16 / 0 | 16 / 0 | 14 (+2 standing) / 0 | 16 / 0 |
| pelvis speed m/s (command) | 0.55 (0.62) | 0.53 (0.62) | 0.38 (0.39) | 0.48 (0.39) |
| cadence, FFT, L / R Hz | 4.54 / 4.55 | **3.88 / 3.88** | 3.99 / 2.54 (irregular gait: no stable peak) | 3.36 / 3.26 |
| touch-downs, L / R per s | 4.49 / 4.56 | 3.85 / 3.85 | 4.34 / 4.91 | 3.56 / 3.49 |
| swing apex, L / R cm | 4.2 / 4.2 | **7.3 / 5.6** | 5.6 / 6.5 | 11.8 / 12.8 |
| stance slip, L / R m/s | 0.021 / 0.025 | 0.019 / 0.013 | 0.048 / 0.052 | 0.049 / 0.033 |
| airborne fraction, L / R | 0.43 / 0.43 | 0.56 / 0.47 | 0.33 / 0.44 | 0.52 / 0.54 |
| stride m | 0.122 | 0.135 | 0.127 | 0.161 |

- At matched speed on flat ground, about 0.9 m/s (envs 2, 3, 6, 13): Isaac runs at 3.73 to 3.75 Hz with 6.2 to 8.8 cm
  apex; ours at 3.97 to 4.36 Hz with 4.7 to 6.3 cm.
- The FFT cadence and the touch-down rate agree on flat ground within 0.07 Hz.
- On rough terrain, our policy's per-env FFT peaks scatter between 0.5 and 6.6 Hz while its touch-downs run 2 to
  5.6 per s. The gait has no steady period, so read the touch-down row there.
- Rough v1 of these runs, with a single ground point under the foot centre, is in `.../measure_v1_centroid_lookup/`.
  It gave false 1.3 Hz cadences and inflated apexes: the foot centre crossed step edges while the sole stayed on the
  tread. The footprint lookup fixes this.

**A. The metric reproduces your measurement of the old take** (`old_takes/g1_flat/rec.npz`, env 6, the same geoms). It
gives 3.96 Hz on both feet (you measured 3.99; the FFT grid differs) and an apex of 5.61 / 7.43 cm above the clip's
lowest point (5.6 / 7.2).

**B. Looking for an obs-layout mismatch.**
- Sources:
  - Isaac's policy terms, in order: base_lin_vel, base_ang_vel, projected_gravity, velocity_commands, joint_pos_rel,
    joint_vel_rel, last_action, height_scan (`assets/isaac/g1_velocity_env_cfg.py:120-143`; Isaac Lab 2.3.2
    verbatim, `assets/isaac/README.md`). The L4 run's
    `runs/parity/isaac/rough/rsl_rl_g1_rough/2026-09-25_09-54-11/params/env.yaml:582-640` confirms them. Flat drops the
    height scan (`g1_flat_env_cfg.py`).
  - MetalSim's `g1_obs` kernel (`metalsim/learn/g1_velocity.py:170-196`) has the same order. Base velocity is at the
    COM (Isaac 2.3.2 `root_lin_vel_b`), angular velocity is in the body frame, and the noise ranges are the same.
    Also PARITY §1.2 row 1 and `tests/test_g1_task_terms.py::test_observation_layout_and_action_mapping`.
  - Joint order: Isaac's is breadth-first (`left_hip_pitch, right_hip_pitch, torso, ...`; `joint_names` in the tracked
    `runs/parity/isaac/rough/play_rough/L*/isaac_it*/traj.npz`, the same at L3 and L6). MetalSim's is depth-first
    (left leg, right leg, torso, arms, hands). Both are the same 37 names (checked), so the blocks must be permuted.
    The permutation is the one in `metalsim/parity/side_by_side.py:22-27, 54-56, 74-78` and
    `scripts/diagnostics/g1_rough_transfer.py:112-118`. MetalSim stores last_action in its own order, so it is
    permuted the same way.
  - Actions: `JointPositionActionCfg(scale=0.5, use_default_offset=True, joint_names=[".*"])`
    (`g1_velocity_env_cfg.py:112`). This is `g1_apply_action`.
  - Height scan:
    - Isaac (`g1_velocity_env_cfg.py:66-70`, `g1_rough_env_cfg.py` torso prim; `env.yaml:395-424`): on `torso_link`,
      offset z 20, `ray_alignment: yaw`, grid 1.6 x 1.0 m at 0.1 m, `ordering: xy`, which is x fastest
      (`upstream/IsaacLab3/.../ray_caster/patterns/patterns.py:41-53`).
    - MetalSim (`metalsim/learn/terrain.py:338-372`): `HeightScanner(body="torso_link", ordering="xy", offset_z=0.5)`.
    - PARITY §1.5, around "A defect was found and fixed on the way": Isaac's ray order has been the default since
      2026-09-25, and the first-step observation is identical between the two simulators, box cells included.
    - The ij / xy warning in `docs/research/ppowarp_checkpoint_load_2026-10-13.md` is about our checkpoints from
      before 2026-09-25. An Isaac checkpoint needs "xy"; the adapter enforces it.
    - `scan_surface="exact"` and `terrain_collision="boxes_local"` are the defaults (`g1_velocity.py:576-580`). They
      are what `g1_rough_transfer.py` played Isaac's rough checkpoints with: the final checkpoint had 0 / 32 wall-cell
      falls (PARITY §1.5, "Transfer of Isaac's checkpoints on the wall cells").
  - Commands: Isaac's G1 flat command has lin_vel_y ±0.5 and rough has 0 (the "Commands" blocks of the two configs).
    That is `reward_cfg` "flat" / "rough_isaac" (`g1_velocity.py:708`).
  - Learner config (`assets/isaac/g1_rsl_rl_ppo_cfg.py`): ELU, no observation normaliser, 1,500 / 3,000 iterations.
    This matches the checkpoints' iter 1499 / 2999 and their actor shapes.
  - Physics: 2.5 ms x 8 against Isaac's 5 ms x 4. The smaller step keeps the kp-200 drives stable (PARITY §1.5,
    defect 1). It is the default `physics_dt` of `side_by_side.py` and `g1_rough_transfer.py`, the tools that played
    Isaac's checkpoints successfully: 3.09 / 2.96 / 3.21 m against Isaac's 3.19 / 3.02 / 3.11 m (PARITY §1.5,
    "Transfer of the checkpoints across simulators").
- Experiments (`obs_ab.py`, log `obs_ab.log`, `obs_ab_isaac_ours.json`; 16 envs, 8 s, command fixed at (0.5, 0, 0) as
  in `play_policy.py`, observations computed independently in torch from the state). These rest on experiment only:

| flat, command 0.5 m/s | ours, noise on | ours, noise off | Isaac, noise on | Isaac, noise off (Isaac's PLAY) |
|---|---|---|---|---|
| cadence Hz | 4.67 | 4.60 | 3.92 | 3.90 |
| apex L / R cm | 3.8 / 3.8 | 3.2 / 3.6 | 7.1 / 5.5 | 6.8 / 5.3 |
| stride m | 0.090 | 0.086 | 0.103 | 0.099 |
| tracking: vx (cmd 0.5) / vy / yaw rate | 0.420 / +0.006 / -0.019 | 0.397 / +0.010 / -0.033 | 0.403 / -0.007 / -0.012 | 0.386 / -0.012 / -0.012 |

  1. At step 0 the torch observation lies within the noise bounds of the task kernel's observation, for both
     checkpoints. The kernel's layout equals Isaac's definition, computed separately.
  2. Observation noise does not change the gait. MetalSim's task always applies Isaac's training corruption, while
     Isaac's PLAY configs turn it off.
  3. Tracking is about 0.39 to 0.40 m/s at a 0.5 command with no lateral or yaw drift. Isaac's own checkpoints travel
     3.04 to 3.21 m in 8 s in PhysX under the same protocol, which is the same 0.38 to 0.40 m/s. A wrong joint map
     makes the G1 fall at once; here 0 / 16 fell.

**C. PhysX reference for rough** (`physx_reference.py`, `physx_reference.json`). Isaac's own L4 rough checkpoints, played
in Isaac Sim at 0.5 m/s (the tracked `play_rough` traj.npz files), were posed on MetalSim's G1 and measured with
`gait_metrics`.
- Where the robot walks, the hip-pitch cadence is 3.5 to 4.6 Hz: inverted stairs 3.5 to 4.1, boxes 4.0 to 4.2,
  random rough 4.6.
- No contact signal was recorded, so the apex, slip and airborne columns there are not reliable.
- The flat PhysX play recordings (`runs/parity/isaac/parity_out/play/`) are not in this checkout or in
  `~/metalsim-fresh`. The flat reference figures above come from the research note.

**Frames looked at.** Diagnostic previews, not media: 12 consecutive frames, 3.00 to 3.22 s (about one gait cycle), of
each measurement take's first-ranked env. A low side camera on the feet, 960x540, 4 x 4 spp, one GPU-queue job
(`preview.sh`). The contact sheets are `runs/readme_media/scratch/isaac-pretrained/g1_{flat,rough}_{isaac,ours}_feet12_crop.jpg`
(and the uncropped `_feet12.jpg`).
- **Flat, Isaac (env 7, 0.55 m/s).**
  - Deeply bent knees: a crouched stance.
  - In frames 3 to 6 one foot is clearly off the floor, toe down, with a shadow gap under it. In frames 7 to 12 the
    other foot is lifted, heel up and the shin swung back.
  - One foot is airborne at a time, and the feet trade within the 12 frames. The steps are short, with the feet nearly
    side by side and no forward reach.
  - It is a quick, tidy march in place rather than a stride.
- **Flat, ours (env 13, 0.82 m/s).**
  - Straighter legs.
  - The feet barely leave the floor: in frames 4 and 5 the swing foot tips toe-down and skims the boards. The same
    short, quick steps as Isaac's, lower: a shuffle.
- **Rough, Isaac (env 6, column 7, inverted stairs, climbing).**
  - High knee lift.
  - In frames 7 to 12 the swing foot rises above the next riser, toe up, and is placed on the next tread. A proper
    stair climb.
- **Rough, ours (env 6, the same cell).**
  - In frames 1 to 6 the feet stay close together, with toes catching the tread edges.
  - Only in frames 11 and 12 does one foot lift onto the next step. Irregular and scuffing.

**Fetch.** `zsh scripts/fetch_isaac_assets.sh` downloads both checkpoints and checks their hashes:
- flat `fdf0f242...20e7`, 2,027,170 bytes;
- rough `d85a9ade...8851`, 7,842,530 bytes.

They are byte-identical (`cmp`) to the copies you verified. `rec.json` `checkpoint` records the format (rsl_rl), the
iteration (1499 / 2999), `source_url` and `published_sha256`.

**Tests.** `.venv/bin/python -m pytest -q tests/test_film.py tests/test_film_render_cache.py tests/test_no_machine_paths.py`
gives 8 passed, 1 skipped. The skip is `test_newton_twin`, because the main venv has no `newton`.

No timing figure comes from this work (14-inch, Low Power Mode).

## Open
- Not filmed, and the defaults are unchanged. `runs/film/g1_*` and `docs/renders/G1_*` still show our g1regress takes.
- The G1 `notes` in `tasks.py` describe the old env ranking. The next take's notes must describe the natural-gait
  preference and quote `rec.json` `gait`.
- The published checkpoints' own gait in PhysX is not measured: Isaac is not available here. The PhysX reference uses
  Isaac's L4 run's checkpoints of the same task.
- FFT cadence is meaningless for an irregular gait (our rough policy). Use `touchdowns_hz` there.
- `apex_m` is measured above the median stance height. Your measure was above the clip's lowest point, which is 0 to
  5 mm different on flat ground.
- The film runs with training observation noise (the task kernel has no switch). On flat ground this has no measured
  effect on the gait.
- `g1_rough_transfer.py` used the pelvis-origin base velocity; the film uses the COM, which is Isaac 2.3.2's. This was
  not A/B-tested on rough.

## Next
Your decision first. Three options:
- **(a) Switch anyway.** The Isaac policies are better on every measured axis, though they keep Isaac's native about
  4 Hz gait. In `tasks.py`:
  - set `G1_FLAT.checkpoint = ISAAC_PRETRAINED["Isaac-Velocity-Flat-G1-v0"]["path"]` and the same for rough;
  - keep `adapter_kw` as it is;
  - replace each policy `AssetRef` with `AssetRef("Policy: Isaac Lab 2.3 pretrained rsl_rl checkpoint
    (Isaac-Velocity-Flat-G1-v0, iteration 1499)", "NVIDIA Isaac Lab", <url>, NVIDIA_TERMS, FETCH_ISAAC)`;
  - rewrite both `notes` (policy, task settings from `ISAAC_TASK_SETTINGS`, the ranking, gait numbers).

  Then, with GPU jobs one at a time:
  - `python -m metalsim.render.film g1_flat --stage record`, then `--stage render`, then `--stage encode`;
  - the same for `g1_rough`;
  - the README WebPs with the brief's `encode webp` command;
  - `--stage doc` for both.

  Look at a strip of each readme clip, and at 12 full frames of the feet.

  Ours stays selectable: `--checkpoint runs/g1regress/ckpt14/flat_main_s0.pt`.
- **(b) Keep our checkpoints** and describe the gait honestly (wording below).
- **(c) Get a natural-looking gait.** That needs a policy trained with gait shaping (a swing-height or gait-period
  reward). It is outside Isaac's G1 task and outside this stream.

**README wording, if (a).**
- Captions:
  - "G1 flat: Isaac Lab's published G1 flat-terrain policy (NVIDIA, rsl_rl, 1,499 iterations, trained in PhysX),
    played unchanged in MetalSim on MuJoCo Warp."
  - "G1 rough: Isaac Lab's published G1 rough-terrain policy (NVIDIA, 2,999 iterations, trained in PhysX) on Isaac's
    rough terrain in MetalSim."
- Paragraph: "The G1 clips play NVIDIA's pretrained Isaac Lab 2.3 checkpoints for Isaac-Velocity-Flat-G1-v0 and
  Isaac-Velocity-Rough-G1-v0. They were trained in Isaac Sim's PhysX and run here without retraining: MetalSim feeds
  them the task's observation terms in Isaac's joint order and applies their actions. The quick, short steps (about
  3.9 gait cycles per second on flat ground, 3.5 on rough terrain) are how these policies walk in Isaac Sim as well.
  Isaac's own G1 flat checkpoints step at 3.2 to 3.8 Hz in PhysX, and MetalSim's contact physics adds about 5 %. The
  checkpoints are NVIDIA's files under NVIDIA's asset terms, fetched by `scripts/fetch_isaac_assets.sh` and not
  redistributed here."

**README wording, if (b).**
- Flat caption: "G1 flat: a policy trained in MetalSim for 500 iterations of Isaac Lab's G1 flat task; it walks with
  short, quick steps (about 4.5 gait cycles per second, 4 cm foot clearance)."
- Rough caption: "G1 rough: a policy trained in MetalSim for 1,000 iterations; it climbs Isaac's rough terrain with
  irregular, low steps."

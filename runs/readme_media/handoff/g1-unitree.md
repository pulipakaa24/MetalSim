# Hand-off: g1-unitree (2026-10-31)

## Status
**Done.** All five steps are complete:
1. Unitree's own G1 policy (`unitree_rl_gym` `deploy/pre_train/g1/motion.pt`) was played by Unitree's unchanged loop on CPU MuJoCo.
2. The loop was ported to MuJoCo Warp as a film adapter.
3. The `g1_unitree` take was filmed (both shots), the hero was encoded and the README WebP was cut with the brief's
   command.
4. I looked at the output. It walks at **1.25 Hz per foot with clear swing**: 4.2 / 4.4 cm of sole clearance, and the
   ankle-roll link reaches about 8 cm, Unitree's swing target.
5. The doc and the README row are written, and the three test files pass.

Nothing is committed. README.md and the plan doc are untouched.

## Reference vs MetalSim (step 1 vs step 2)
Both runs: 8 s, command (0.5, 0, 0), gait window 1 to 8 s, measured by the same code (`record.gait_metrics` plus
`unitree.unitree_feet` / `unitree_contacts`).
- Reference: Unitree's loop on CPU MuJoCo 3.14.0.
- MetalSim: the `UnitreeG1` adapter on MuJoCo Warp `BatchSim` with 4 worlds, one GPU-queue job.

| | Unitree's loop, CPU MuJoCo | MetalSim, MuJoCo Warp, 4 worlds |
|---|---|---|
| pelvis at 5 s | (2.22441, -0.09875, 0.77027) m | distance 1.3 / 1.7 / 2.2 / 1.5 **um** (worlds 0-3) |
| pelvis at 8 s | (3.64685, -0.25725, 0.77227) m | distance 2.8 to 3.7 um |
| cadence, FFT of foot height, L / R | 1.251 / 1.251 Hz | 1.251 / 1.251 Hz (difference 0) |
| touch-downs per s, L / R | 1.143 / 1.286 | 1.143 / 1.286 |
| swing apex above stance, sole, L / R | 4.17 / 4.38 cm | same |
| ankle-roll link origin in swing (p95), L / R | 7.7 / 8.1 cm | same |
| stance slip, L / R | 0.0037 / 0.0036 m/s | same |
| airborne, L / R | 0.46 / 0.44 | same |
| pelvis speed / stride | 0.480 m/s / 0.384 m | same |
| mean pelvis height (1-8 s) | 0.7698 m | 0.7698 m |
| foot contact per frame | | identical in 400 / 400 frames |
| max abs difference in policy output, any step | | 3.0e-5 |
| max abs joint-angle difference at 1, 2, 5, 8 s | | 1e-6 rad |

- **Not bitwise.** The four GPU worlds also differ from one another by about 1 um.
- **Why so close.** MuJoCo Warp computes in float32 and MuJoCo in float64. The gait is a strongly stable limit cycle,
  so float differences do not grow. This explanation rests on the experiment only; I did not research it further.

## Gait of the filmed take (`runs/film/g1_unitree/rec/rec.json` `gait`)
Window 1 to 6.02 s. World 2 is filmed; all 4 worlds agree at the printed precision.
- Speed 0.48 m/s against a command of 0.50.
- Cadence 1.245 / 1.245 Hz. Touch-downs 1.20 / 1.39 per s (6 and 7 in the window).
- Apex 4.2 / 4.4 cm, slip 0.004 / 0.004 m/s, airborne 0.43 / 0.47, stride 0.386 m.
- No world terminated or went down. Capacity overflow `{}`.
- For comparison, NVIDIA's flat take steps at 3.78 Hz (`isaac-pretrained.md`).

## Changed
- `metalsim/render/film/unitree.py` (new). The `UnitreeG1` adapter and the measurement helpers:
  - foot geoms: the 4 sole spheres of each `*_ankle_roll_link`; foot point = their mean; contact = any sphere touches
    the floor;
  - Unitree's observation and gravity projection;
  - the PD Warp kernel.
  - Header: `SPDX Apache-2.0 AND BSD-3-Clause` with Unitree's copyright line.
- `metalsim/render/film/tasks.py`: adds `UNITREE_RL_GYM` (repo, commit, policy path, URL and sha256), the `G1_UNITREE`
  FilmTask, `G1_UNITREE_NOTES` and `G1_UNITREE_POWER`. The G1_FLAT and G1_ROUGH entries are unchanged.
- `metalsim/render/film/record.py:450`: the sha256-mismatch message names the adapter's `fetch` script.
- `metalsim/render/film/doc.py`: the policy line prints `(iteration N)` only when N is known, else the format. The
  existing docs are unaffected; I checked their current lines.
- `metalsim/render/film/__main__.py`: the doc's Reproduce block lists `scripts/fetch_unitree_rl_gym.sh` when a task
  uses it.
- `scripts/fetch_unitree_rl_gym.sh` (new): a shallow, sparse git checkout at `276801e46c5d433564f24658bac64f254b7d2d4b`
  into `assets/unitree_rl_gym/` (about 108 MB with `.git`). It checks the HEAD SHA and the sha256 of `motion.pt`.
- `.gitignore`: adds `assets/unitree_rl_gym/`. This file is outside my list, but the brief requires the directory to be
  gitignored.
- `licenses/BSD-3-Clause-Unitree.txt` (new): Unitree's LICENSE, byte copy. Also outside my list: BSD-3 requires the
  notice for the derived `unitree.py`, and this follows the `licenses/` convention (Isaac Lab, PhysX).
- `THIRD_PARTY_NOTICES.md`: a summary-table row and a "Unitree Robotics unitree_rl_gym" section.
- `docs/renders/README.md`: a new row in the film table. The "README's media" section is not touched.
- `docs/renders/G1_UNITREE.md`: written by the doc stage.
- New media:
  - `docs/renders/g1_unitree.mp4` (1.5 MB);
  - `docs/renders/g1_unitree_still{1,2,3}.jpg` (frames 75, 150 and 240);
  - `docs/renders/g1_unitree.webp` (0.71 MB, 150 frames, 320x240, PSNR 36.74 dB). Its sidecar is
    `runs/readme_media/assets/g1_unitree.webp.json`, moved there as for the other cells.
- `runs/film/g1_unitree/`:
  - the take: `MACHINE` (14-inch), `rec/rec.json`, `{hero,readme}/render.json` and `render_frames.jsonl`,
    `hero/encode.json`;
  - `sim2sim_check/`:
    - `reference.py`: Unitree's loop verbatim, BSD-3 header;
    - `compare.py`;
    - `reference.json` and `compare.json`: their outputs.
  - The stage logs were moved to `runs/readme_media/scratch/g1-unitree/logs/`.

## Verified
**Live state and hardware**
- Live state, read myself:
  - at the start of the session (time not noted): AC power, 25 % charging, `powermode 2`, GPU lock free;
  - 14:00: AC power, 39 % charging, lock free.
- Stamps:
  - `rec.json`: AC, 36 %, charging, 13:55;
  - hero `render.json`: 36 %, 13:55;
  - readme `render.json`: 37 %, 13:57.
- The machine is a Mac16,6 (`sysctl hw.model`). No timing figure is quoted.

**Sources**
- Fetch: `git ls-remote` gave HEAD = main = `276801e4...` (commit dated 2025-07-25).
- The fetched `g1.yaml`, `deploy_mujoco.py`, `g1_env.py` and `g1_config.py` are identical (`diff`) to the brief's
  copies in `scratchpad/gait_refs/`.
- `motion.pt`:
  - sha256 `cf668f75b90d1abf73d2b87612a6e76bccc61ff7e083b63582d3f6aaa3c1759d`, 145,745 bytes;
  - TorchScript: LSTM 47→64 with `hidden_state` / `cell_state` buffers of batch 1, then an actor of 64→32→12 with ELU.
    So the adapter loads one copy per world (inspected with `named_buffers` and `.code`).
- The real-robot config loads the same file: `git -C assets/unitree_rl_gym show HEAD:deploy/deploy_real/configs/g1.yaml`
  line 10 is `policy_path: .../deploy/pre_train/g1/motion.pt`. That checks the brief's "deployed on the real robot".
- The model, `scene.xml` → `g1_12dof.xml`:
  - nq 19, nu 12;
  - Euler integrator, pyramidal cone, Newton solver, 100 iterations;
  - `<motor>` actuators with no ctrlrange; `actuatorfrcrange` on the joints;
  - joint damping 0.001, armature 0.01, frictionloss 0.1;
  - the arms are geoms of the pelvis, so they are fixed.
- Training sources:
  - `g1_env.py:58-63`: period 0.8, the right leg at offset 0.5;
  - `g1_env.py:98-110`: stance phase < 0.55, swing target 0.08 m on `feet_pos`, the ankle-roll link;
  - `g1_config.py`: `ActorCriticRecurrent`, LSTM 64;
  - `legged_robot_config.py:155`: sim dt 0.005 with decimation 4.

**Actuator form (research, then confirmed by experiment)**
- `metalsim/physics/batch.py:362-385`: `_launch_substeps` runs the substep hooks after every `mjw.step` inside the
  captured step graph, and `add_substep_hook` re-captures the graph. `metalsim/parity/record_g1.py:185` uses the same
  pattern.
- MuJoCo Warp clamps `jnt_actfrcrange` as MuJoCo does: `upstream/mujoco_warp/mujoco_warp/_src/forward.py:1572`.
- The adapter launches the PD kernel before the first step of each policy step, then after each step. So step i uses
  the torque computed from the state that starts step i, exactly as Unitree's `pd_control` before `mj_step` does.
- The comparison table confirms it: identical contacts and µm-level agreement.

**Commands** (all from the repository root)
- Step 1, CPU:
  `.venv/bin/python runs/film/g1_unitree/sim2sim_check/reference.py runs/readme_media/scratch/g1-unitree/ref 8`.
  Re-running it into `ref2` gave a byte-identical `reference.json`.
- Step 2, GPU queue: `scripts/gpu_run.sh g1-unitree-compare render 15 -- .venv/bin/python
  runs/readme_media/scratch/g1-unitree/compare.py <ref> <cmp> 4 8`, rc 0. The script now lives in `sim2sim_check/`.
  Full-precision figures came from `ref/reference.npz` against `cmp/metalsim.npz`.
- Step 3:
  1. `scripts/gpu_run.sh g1-unitree-record render 15 -- .venv/bin/python -m metalsim.render.film g1_unitree --stage
     record`: 301 frames, 60 ms per frame.
  2. `scripts/gpu_run.sh g1-unitree-render render 20 -- ... --stage render`: both shots, 301 frames each, 0 frame
     corrections. Cache keys: hero `66bc4ee77aff6cb51478`, readme `13bd98334555550118fa`.
  3. `.venv/bin/python -m metalsim.render.film g1_unitree --stage encode --shots hero`.
  4. The brief's exact WebP command: `python -m metalsim.render.film.encode webp runs/film/g1_unitree/readme/frames
     docs/renders/g1_unitree.webp --start 0.5 --seconds 5 --src-fps 50 --fps 30`.
  5. `--stage doc`.
- Tests: `.venv/bin/python -m pytest -q tests/test_film.py tests/test_film_render_cache.py
  tests/test_no_machine_paths.py`: 8 passed, 1 skipped.
- Machine paths: a grep of every new or changed tracked-to-be file found none. `rec/twin.xml` holds an absolute
  `meshdir`, but it is gitignored (`runs/film/**/twin*`).

**Frames I looked at** (Read tool)
- `runs/film/g1_unitree/hero/frames/frame_0150.png` and `readme/frames/frame_0100.png`. Framing is right: the readme
  robot fills about 3/4 of the frame height, the floor is complete, and there are no stray collider or site boxes.
- `runs/readme_media/scratch/g1-unitree/look/webp_strip.jpg`: every 15th WebP frame. The orbit is visible and the legs
  alternate.
- `look/hero_consecutive.jpg`: hero frames 150-161, 12 consecutive frames, cropped to the legs at native resolution.
  The near foot finishes its swing and lands flat; the stance foot stays planted.
- `look/hero_every4.jpg`: frames 150-194 in steps of 4, about one gait cycle. Each leg in turn lifts with clear knee
  flexion and passes the stance leg.
- `look/stills.jpg`: the three stills.
- Script: `runs/readme_media/scratch/g1-unitree/strips.py`.

**Description of the gait.** An unhurried, regular walk:
- one step per foot every 0.8 s, legs alternating;
- visible knee lift in swing; the sole clears the floor by about 4 cm;
- flat-footed touch-downs with no visible sliding;
- an upright torso at a steady pelvis height (about 0.77 m), on a narrow track.

It looks like a person walking cautiously, not the sped-up shuffle of the Isaac policies. Two features belong to the
policy and model, not to the port, because Unitree's CPU MuJoCo run shows them identically:
- the arms are rigid in Unitree's 12-dof model, held forward;
- the walk drifts slowly to the right, y -0.16 m over 6 s.

## Open
- I added no unit test for `unitree_obs` / `unitree_contacts`, because `tests/` is not mine. The sim2sim check is a
  script with JSON output, not a pytest.
- `G1_UNITREE.md` says "+dirty" at `dfec9ebf`: the take ran on uncommitted code. A commit that lands this work should
  say so, or re-film after committing.
- docs/renders/README.md "The README's media" section (the per-cell "Shows" bullets and the row-1 cell list) still
  describes `g1_flat.webp` as the flat cell. I didn't change it, because the brief limited me to one row.
- The doc's "Hardware and pins" lists `torch 2.14.0` for the record stage. The CPU reference also used torch 2.14.0 and
  mujoco 3.14.0 in the main venv.

## Next
For the orchestrator, README edit:

1. Replace the G1 flat cell, `README.md` lines 18-21, with:
```html
    <td align="center" width="33%">
      <img width="320" src="docs/renders/g1_unitree.webp" alt="Unitree G1 walking on flat ground with Unitree's own pretrained walking policy, path traced">
      <br><b>python -m metalsim.render.film g1_unitree</b>
    </td>
```
   The rough cell, `g1_rough.webp` with NVIDIA's policy, stays.

2. Add this sentence to the row-1 paragraph, and change its first sentence so it no longer says the flat cell plays
   NVIDIA's flat checkpoint:
   > The flat cell plays Unitree's own pretrained G1 walking policy (`unitree_rl_gym`'s `deploy/pre_train/g1/motion.pt`,
   > BSD-3-Clause, fetched by `scripts/fetch_unitree_rl_gym.sh`) through Unitree's MuJoCo sim2sim loop ported to MuJoCo
   > Warp: it steps at 1.25 gait cycles per second, the 0.8 s gait clock Unitree trains it with, and stays within 3 µm of
   > Unitree's own CPU MuJoCo run of the same loop after 5 s ([G1_UNITREE.md](docs/renders/G1_UNITREE.md)).

   Sources:
   - `docs/renders/G1_UNITREE.md`, "Against Unitree's own run";
   - `runs/film/g1_unitree/sim2sim_check/compare.json`: `at.5s.dpos_m`, max 2.2e-6 m;
   - `rec.json` `gait`: 1.245 Hz;
   - `assets/unitree_rl_gym/legged_gym/envs/g1/g1_env.py:58`: `period = 0.8`.

   The sentence about the "quick, short steps (3.8 gait cycles per second in the flat clip)" then no longer describes
   the flat cell. Drop it, or reword it to refer to the Isaac policies (the rough cell and row 2).

3. Update docs/renders/README.md "The README's media", which needs a `### G1 flat (Unitree) (g1_unitree.webp)`
   entry. The WebP command and sidecar are above and in `runs/readme_media/assets/g1_unitree.webp.json`.

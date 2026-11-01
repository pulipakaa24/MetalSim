# Hand-off: so101-smooth (2026-10-31, second agent, 15:50 to 16:25)

## Status

Done. Both variants were trained and evaluated, then the work stopped as the brief says. **Neither variant reaches a
realistic control style.**

- Variant 2 (`isaac_unscaled_raw`) is the closer of the two, and it is the one filmed:
  - it halves the action change per step, from 0.68 to 0.35;
  - success rises to 86/100 on CPU and 92/100 on MetalSim;
  - but 72 % of its actions still sit at ±1, against 0.9 % for the scripted expert, and its action change per step is
    25 times the expert's.
- The only realistic controller is so101Sim's scripted expert: at 7.4 V it succeeds 20/100 on CPU and 36/100 on
  MetalSim.

Nothing was committed. Untouched: README.md, the plan doc, `tasks.py`, `docs/renders/README.md`, and every tracked
file in so101Sim. The training runs are in so101Sim's git-ignored `rl/runs/lift_ppo_smooth/`.

## Results: 100 seeds (10000–10099), 7.4 V unless marked

Each cell is CPU | MetalSim.

| controller | success | up at 6 s | time to lift | action change / step | actions at limit | torque sat, arm mean / gripper | peak joint speed (rad/s) |
|---|---|---|---|---|---|---|---|
| scripted expert | 20 \| 36 | 17 \| 29 | 4.13 \| 4.20 s | 0.0139 \| 0.0142 | 0.9 \| 0.7 % | 70/90 \| 59/95 % | 5.19 \| 5.16 |
| original | 77 \| 75 | 70 \| 68 | 1.62 \| 1.50 s | 0.682 \| 0.696 | 65.7 \| 66.3 % | 88/99 \| 88/98 % | 5.87 \| 6.33 |
| v1 isaac_scaled | 77 \| 77 | 75 \| 73 | 1.51 \| 1.65 s | 0.756 \| 0.761 | 77.5 \| 77.0 % | 85/99 \| 86/100 % | 5.98 \| 6.23 |
| **v2 isaac_unscaled_raw** | 86 \| 92 | 83 \| 85 | 1.36 \| 1.34 s | 0.348 \| 0.355 | 72.4 \| 72.3 % | 72/100 \| 72/100 % | 6.15 \| 5.96 |
| original, 12 V | 100 \| 98 | 99 \| 97 | 0.95 \| 0.95 s | 0.991 \| 0.982 | 79.6 \| 79.2 % | 83/100 \| 83/100 % | 5.65 \| 5.53 |
| expert, 12 V | 95 \| 98 | 86 \| 94 | 1.60 \| 1.37 s | 0.0051 \| 0.0043 | 0.2 \| 0.1 % | 24/90 \| 14/94 % | 5.39 \| 5.00 |

- The full table, with per-joint saturation and the target change in rad, is in `runs/film/so101_lift_smooth/eval/table.md`
  and `results.json`.
- **MetalSim:** every number comes from job `so101-smooth-evals2`, run 16:13:55–16:14:33. That is after the fork
  move: `upstream/mujoco_warp` reflog shows the fast-forward 07a51a63 → 8eb51fe6 at 15:47:53. Every `metrics.json`
  records mujoco_warp `8eb51fe637`.
- **Repeatability:** an earlier job, also on 8eb51fe6 (15:57:58, kept in `scratch/so101-smooth/metalsim_run1558/`), gave
  successes within ±3 of these. MetalSim is not bitwise repeatable from run to run.
- **v2's style:** its median joint-target change is 0 rad per step, against 0.67 for the original. It holds its targets
  at the joint limits instead of flipping them. The elbow still flips, at 0.89 per step. It lifts higher than the
  original (mean highest box 0.185 m against 0.148 m), and the box never rises above 6 cm in 11 episodes, against 21.

## Variant choice and sources

**Variant 1 failed: action change per step rose from 0.68 to 0.76.**

- `raw_actions.py` (`logs/raw_actions.log`) measured the policy's pre-clip mean:
  - outside [-1, 1] on 45–86 % of steps per joint for the original policy;
  - for v1's wrist flex and wrist roll, median |mean| 20 and 9.8, with SB3 std 15.9 and 103.
- A penalty on the clipped action, which is what v1 used, cannot see a mean outside the clip or noise beyond it.

**Isaac Lab penalises the raw action.** Files downloaded via `gh` at IsaacLab main `6bb65480` into
`runs/film/so101_lift_smooth/eval/upstream/`:

- `managers/action_manager.py:393-394`: `process_action` stores the raw action;
- `isaaclab_rl/rsl_rl/rl_cfg.py:252`: `clip_actions` defaults to None;
- `vecenv_wrapper.py`: clips only if `clip_actions` is set;
- `contrib/lift/lift_env_cfg.py:155-184`: the penalty weights and curriculum;
- `contrib/lift/config/franka/agents/rsl_rl_ppo_cfg.py`: does not set `clip_actions`.

**So v2 = Isaac Lab's weights unscaled (1e-4 → 1e-1, switch at 27.8 %), on the raw policy output.**

- SB3's action space was widened to ±1000; so101Sim's `step` still clips to [-1, 1] (`so101_lift.py:129`).
- One pitfall: `VecNormalize.load` keeps the pickled [-1, 1] action space, so `train_smooth.py` resets it. The first
  smoke run (`logs/smoke_raw.log`) did not, and its "raw" penalty was really clipped.
- Verified before training: in an SB3 learn, raw actions such as -5.2 and 5.8 reached the env.
- Training did change the action rate: right after the switch, the raw rate_l2 fell from 96 to 39, and the
  deterministic action change per step fell from 0.83 to 0.35. Training eval at 10M: 17/20.

**LeRobot `max_relative_target` rejected**:

- **SO-101 default is None:** `lerobot/robots/so_follower/config_so_follower.py`, lerobot 0.6.1 in so101Sim's venv,
  and the `~/lerobot` clone at bf31dd79.
- **Typical values:** 5° is the old Aloha default (`git show 3c0a209f:lerobot/common/robot_devices/robots/configs.py:79-88`);
  Koch was 10/15 in PR #373's first revision (https://github.com/huggingface/lerobot/pull/373).
- **It clamps the goal to present ± the limit** (`lerobot/robots/utils.py` `ensure_safe_goal_position`). 5° = 87 mrad,
  45 times the servo's proportional band (1.912 N·m / kp 998 = 1.9 mrad). So the policy output and the servo torque
  stay bang-bang. This last point is reasoning from the model's numbers, not an experiment.

## Changed

**`metalsim/render/film/so101.py`**

- `SO101_LIFT_SMOOTH` rewritten for v2: title, summary saying it is not realistic, asset text, script paths.
- New `SO101_SMOOTH_SECTION`, appended to `SO101_NOTES`.

**`docs/renders/SO101_LIFT.md`**

- New section "## Smooth 7.4 V retrain", inserted before "Reproduce". Its text is identical to
  `SO101_SMOOTH_SECTION`; checked.

**`docs/renders/` media**

- `so101_lift_smooth.mp4` (0.64 MB).
- `so101_lift_smooth_still{1,2,3}.jpg` (frames 15/30/150).
- `so101_lift_smooth.webp` (0.60 MB, 150 frames, PSNR 35.32 dB).
- `so101_lift_smooth.webp.json`, the sidecar. It is left in `docs/renders/`, as so101_lift_74v's is; I don't own
  `runs/readme_media/assets/`.

**`runs/film/so101_lift_smooth/`**

- `policy/so101_lift_smooth_ppo.pt`: v2, sha256 4593a0ed..., git-ignored by `*.pt`.
- `policy/candidates/{isaac_scaled,isaac_unscaled_raw}.pt`.
- `train/`: `smooth_env.py`, `train_smooth.py`, and `runs/<variant>/{config.json,evals.txt,progress.txt}`.
- `eval/`: `metrics.py`, `eval_cpu.py`, `eval_metalsim.py` (now takes `expert`), `run_metalsim_evals.sh`,
  `raw_actions.py`, `check_extract_raw.py`, `table.py`, `table.md`, `results.json`, per-run `metrics.json` and
  `reference.json`, `upstream/`, and `logs/` (machine paths replaced).
- The take: `rec/`, `hero/`, `readme/`.

**`runs/readme_media/scratch/so101-smooth/`**

- The working copies of the scripts, which produced the results; their output JSONs cite these paths.
- `compare3.mp4`, `take_w13/`, contact strips, `section.md`, `metalsim_stale_07a51a6/` (the earlier agent's evals),
  and `metalsim_run1558/`.

## Verified

**Extraction**

- v1: max diff 4.7e-6 against `predict`.
- v2: `extract_policy.py` reports 12.08, because `predict` clips to ±1000. Against `clip(predict, -1, 1)` the diff is
  1.7e-6 (`check_extract_raw.py`).

**Evals**

- CPU: `eval_cpu.py <so101Sim> <pt|expert> <servo> <out> 100`, in so101Sim's venv, MuJoCo 3.10.0.
- MetalSim:
  `scripts/gpu_run.sh so101-smooth-evals2 render 20 -- bash runs/readme_media/scratch/so101-smooth/run_metalsim_evals.sh <so101Sim> expert_7v4 orig_7v4 isaac_scaled_7v4 isaac_unscaled_raw_7v4 orig_12v expert_12v`.
- The expert on MetalSim runs on CPU kinematics: so101Sim's scene, `pickup` keyframe, `mj_forward` of each world's
  GPU state. Seed-by-seed agreement with CPU is only 56/100 at 7.4 V, where the expert is marginal, and 93/100 at
  12 V.

**Film**

- Record (16:14:44):
  `scripts/gpu_run.sh so101-smooth-record render 15 -- .venv/bin/python -m metalsim.render.film.so101 so101_lift_smooth --stage record`.
  - All 16 first episodes succeed; RMS joint acceleration is 28.8–41.2 rad/s².
  - The ranking filmed world 8 (seed 10008), lifted at 0.86 s.
- Render: `--stage render`.
- Then `--stage encode --shots hero`.
- WebP: `python -m metalsim.render.film.encode webp runs/film/so101_lift_smooth/readme/frames docs/renders/so101_lift_smooth.webp --start 0 --seconds 5 --src-fps 50 --fps 30`.
- `compare3.mp4` (1920×516, 6 s, 50 fps): `compare3.py`, using:
  - so101_lift readme frames (12 V, world 13);
  - so101_lift_74v readme frames (7.4 V unchanged, world 13);
  - a new v2 readme take of world 13 (`--out .../take_w13 --envs 13`; succeeds at 1.16 s).
- Power, from the provenance: **on battery**, 58 → 55 % and discharging, powermode 2, 16:14–16:17. Not a timing run.

**Frames looked at** (Read tool, all in the scratch dir)

- `hero_strip.jpg` (every 25th frame), `grasp_closeup.jpg` (frames 8–60, every 4th), `stills_strip.jpg`,
  `webp_strip.jpg`, `webp_frames_pil.jpg` (WebP frames 0/9/18/27/36/60/90/149 via Pillow), `compare3_strip.jpg`
  (every 30th frame), and `hold_consecutive.jpg` (frames 100–111).
- What they show:
  - the jaws open over the box, and the fixed jaw tips it at 0.3 s;
  - the jaws close around it at 0.6 s;
  - the arm swings the box up and back over its base by about 1.2 s, then holds it about 0.21 m high, roughly upright
    in the jaws, to 6 s;
  - in the hold, the wrist and box turn a few degrees back and forth from frame to frame;
  - the look and framing match the other SO-101 takes;
  - at this scale, I saw no interpenetration.
- `compare3`: at 0.6 s the 12 V arm is already lifting while both 7.4 V arms are still at the box. By 1.2 s all three
  hold it.

**Tests and paths**

- `.venv/bin/python -m pytest -q tests/test_film.py tests/test_no_machine_paths.py`: 5 passed, 1 skipped (newton,
  already skipped before this work).
- A grep for `/Users/` or the local user name in `so101.py`, `SO101_LIFT.md` and the webp sidecar finds nothing, and none in the
  text files of `runs/film/so101_lift_smooth/`. `rec/twin.xml` keeps the absolute meshdir, as the other takes' do.

## README cell, row-2 sentence, tasks.py snippet

The README cell must not claim realism. Recommended text:

```html
    <td align="center" width="33%">
      <img width="320" src="docs/renders/so101_lift_smooth.webp" alt="SO-101 arm on modelled 7.4 V servos grasping a green box and lifting it off a wooden table, so101Sim's PPO teacher retrained with Isaac Lab's smoothness penalties, on MuJoCo Warp, path traced">
      <br><b>python -m metalsim.render.film so101_lift_smooth</b>
    </td>
```

Row-2 sentence: "Then so101Sim's SO-101 cube-lift teacher retrained on a model of the standard follower's 7.4 V servos with
Isaac Lab's lift-task smoothness penalties, 92/100 on MuJoCo Warp (86/100 on CPU); it halves the original policy's
action change per step but still commands the joint limits on 72 % of steps, so it is not yet a realistic control
style ([SO101_LIFT.md](docs/renders/SO101_LIFT.md#smooth-74-v-retrain))."

`tasks.py`, at the bottom after TRON1. so101.py imports no simulator at module level.

```python
# so101Sim's SO-101 lift teacher and its 7.4 V variants: entries live with their adapter
from metalsim.render.film.so101 import SO101_LIFT, SO101_LIFT_74V, SO101_LIFT_SMOOTH  # noqa: E402

for _t in (SO101_LIFT, SO101_LIFT_74V, SO101_LIFT_SMOOTH):
    TASKS[_t.name] = _t
```

## Open

- **Realism goal not met.** The owner's rule ("a realistic simulation that fails beats a working, unrealistic one")
  points to so101Sim's scripted expert on the 7.4 V model, which was not filmed. It is 20/100 on CPU and 36/100 on
  MetalSim, with the box never above 6 cm in 77 CPU episodes. The orchestrator or owner decides.
- **`.gitignore`:** committing the weights needs `!runs/film/so101_lift_smooth/policy/so101_lift_smooth_ppo.pt`.
  Without it, they are rebuilt from so101Sim's `rl/runs/lift_ppo_smooth/isaac_unscaled_raw` with `extract_policy.py`.
- **Seed mismatch in the 7.4 V section.** Its MetalSim column (78/65, on fork 07a51a63, 15:17) disagrees with this
  table's 75/68 (8eb51fe6). The section says so; it was not re-run.
- **Unexplained gap.** Why the expert agrees with CPU on only 56/100 seeds at 7.4 V is not studied. Candidates: the
  marginal grasp, and the kinematics lag of one physics step in the CPU loop (described in `eval_metalsim.py`).
- **Not modelled:** the servo's P loop, current limit, overload protection and backlash.

## Next

1. The orchestrator registers the task with the `tasks.py` snippet above, chooses the README cell (this one, honestly
   labelled, or the 7.4 V unchanged cell), and adds the `.gitignore` line.
2. Optional, owner's call: a `so101_lift_expert74v` take filming the scripted expert at 7.4 V.
   - Add a controller hook to `SO101Lift.step`; the logic is in `runs/film/so101_lift_smooth/eval/eval_metalsim.py`
     (`expert_act`).
   - Record with `envs="auto"` through the GPU queue.
3. Commit with the so101 and so101-74v streams' files.

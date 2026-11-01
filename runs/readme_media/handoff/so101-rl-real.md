# Hand-off: so101-rl-real (2026-10-31, 17:37 to 19:25; second agent from 18:10)

## Status

Done. Training, extraction, the CPU and MetalSim evals (100 seeds), the motion diagnosis and the preview are finished.

- **Result on `7v4_real`, 100 seeds: 3/100 lifts on CPU, 2/100 on MetalSim; 0/100 still up at 6 s on both; time to
  lift 3.22 s on CPU and 1.95 s on MetalSim. The motion is not realistic.**
  - The arm swings shoulder pan / lift / elbow across their range at 1.3-1.7 Hz, with commands at the joint limits on
    59-97 % of steps.
  - Against the deployment-style expert (100/100 CPU, 98/100 MetalSim), per joint: RMS joint acceleration 21-69 rad/s²
    against 2.1-7.6; velocity reversals 2.2-5.9 /s against 0.3-0.9; peak speed 4.6-6.9 rad/s against 2.0-3.1.
  - The old policies on 7v4_real (CPU | MetalSim): original 8 | 5, isaac_scaled 15 | 16, isaac_unscaled_raw 5 | 3,
    with RMS acceleration 21-77 rad/s². The retrain moves somewhat less than they do on CPU at pan / lift (RMS acc 41 / 51 vs
    53-60 / 66-76) and the wrist (41 / 27 vs 56-61 / 35-37), not at the elbow (69 vs 61-67), and lifts less often.
- The Isaac Lab penalty phase (0.1 x action rate + 0.1 x joint velocity, from 4.44M) cut the raw action rate from 175
  to 4 per step (mostly wrist noise) and RMS acceleration by about a quarter (60 -> 46 in training evals); it did not
  change the bang-bang structure, and success fell from 13/100 (pilot at 3M) to 3/100.
- **Not ready to film** as a realistic controller. In the preview, the box is knocked onto its side within about 1 s
  in every seed shown (see **Next**).
- `main_run2` (PID 15482) ran to its end untouched: 17:52:18 to 18:22:49.

## Changed

- `runs/readme_media/scratch/so101-rl-real/` (new, git-ignored scratch):
  - `real_env.py`: `RealLiftEnv(SO101LiftEnv)`: 7v4_real layer per physics step inside `step`, fresh kinematics,
    raw-action Isaac Lab penalties, joint-motion episode stats;
  - `check_env.py`: bitwise check against so101Sim env + `servo_cpu.attach`, fresh-kin check, timing;
  - `train_real.py`: scratch / run2 / resume starts, Isaac Lab curriculum over a fixed plan length;
  - `eval_cpu.py`: so101-servo's harness on `RealLiftEnv` (fresh kinematics by default, `--stale-kin`);
  - `eval_metalsim.py`: so101-servo's MetalSim harness, policies only, fresh kinematics by CPU kinematics of the GPU
    state (default) or `--stale-kin`;
  - `render_policy.py`: `ik_preview/render_expert.py` adapted for a policy;
  - `run_old_evals.sh`, `run_metalsim.sh`, `curves.py`, `cpu/`, `logs/`;
  - second agent: `table.py` (now also reads so101-expert's deployment-expert rows), new `diag_motion.py` (command
    and motion pattern of a CPU eval, typical seed), new `raw_u.py` (pre-clip policy output), `curves_main_run2.md`,
    `curves_pilot_run2.md`, `policy/main_run2.pt`, `cpu/main_run2/`, `metalsim/{main_run2,orig_fresh,isaac_scaled_fresh,isaac_unscaled_raw_fresh}/`, logs `eval_main_run2.log`,
    `raw_u.log`, `metalsim_job.log`, `preview.log`.
- `runs/readme_media/scratch/ik_preview/rl_7v4_real.mp4` and strips `rl_7v4_real_strip.jpg`,
  `rl_7v4_real_10010_first1.5s.jpg`, `rl_7v4_real_10022_f150_185.jpg` (all new).
- `~/so101Sim/rl/runs/lift_ppo_real/` (untracked training outputs): `pilot_scratch/`, `pilot_run2/`,
  `main_run2/` (`model.zip`, `vecnorm.pkl`, `ckpt_{switch,4M..16M}/`, `progress.txt`, `evals.txt`, `config.json`).
- Not touched: README.md, the plan doc, `tasks.py`, `so101_expert.py`, any tracked so101Sim file.

## Verified

Sources read by the second agent: this stream's scripts (`train_real.py`, `real_env.py`, `eval_cpu.py`,
`eval_metalsim.py` header, `render_policy.py`); so101-servo `smoothness.py:1-60` (metric definitions); so101-expert
`cpu/deploy_7v4_real/smooth.json` and its hand-off (real-arm speeds); so101-smooth `train_smooth.py:47-58` (the 9/37
scale); IsaacLab 6bb65480 copies in `runs/film/so101_lift_smooth/eval/upstream/` (`lift_env_cfg.py:128-186` rewards,
curriculum, `decimation` 2 at `sim.dt` 0.01; `config/franka/joint_pos_env_cfg.py:36-43` actions); so101Sim
`rl/native/so101_lift.py:23,141-160` (LIFT_Z, reward, success); `scripts/gpu_run.sh`.

- Action semantics: so101Sim `so101_lift.py` `step` maps a in [-1, 1] linearly onto ctrlrange and writes `d.ctrl` of
  a MuJoCo position actuator: an absolute joint target, like LeRobot's `Goal_Position`.
- `check_env.py`: `RealLiftEnv(fresh_kin=False)` vs so101Sim env + so101-servo `servo_cpu.attach`: max |diff| 0.0 for
  obs, reward and qpos over 3 x 300 steps (random and bang-bang actions); fresh-kin obs equals full `mj_forward` obs
  (0.0). Timing: plain env 47 us, attach 127 us, RealLiftEnv 177 us per control step.
- `eval_cpu.py --stale-kin` on the original policy reproduces so101-servo `cpu/orig_7v4_real` exactly (summary,
  smoothness, all 100 episodes equal).
- Old policies on 7v4_real, fresh kinematics, CPU, 100 seeds (success / box up at 6 s / time to lift): original
  8 / 2 / 3.12 s; isaac_scaled 15 / 1 / 2.84 s; isaac_unscaled_raw 5 / 0 / 2.61 s (stale: 6, 15, 4).
- Final policy extraction (second agent, from so101Sim's `urlab_bridge/`): `.venv/bin/python
  runs/film/so101_lift/port/extract_policy.py <so101Sim>/rl/runs/lift_ppo_real/main_run2
  <scratch>/policy/main_run2.pt` (sha256 3c3236d5...; its own check reports 13.55 because `predict` clips to ±1000);
  so101-smooth's `check_extract_raw.py`: max |clip(predict, -1, 1) - plain| 3.9e-6.

## Training choice: continue from run 2 (decided at 17:52 on the pilots)

Pilots, 3M steps each, run concurrently (17:41-17:52, about 6.4k steps/s each), same env, same reward, curriculum
planned over 16M steps (switch at 4.44M, so both pilots are in the 1e-4 phase):

| at 3M | scratch | run 2 continued |
|---|---|---|
| training success (last 100 episodes, stochastic) | 0.00 (max 0.01 at 2.0M) | 0.16 (0.06-0.16 throughout) |
| deterministic eval, 20 seeds (1M / 2M / 3M) | 0 / 0 / 0 | 1 / 2 / 1 |
| task reward per step | 0.68 (reach only: rising 0.58 -> 0.68) | 0.52 |
| projected penalty after the switch (0.1 x (rate_l2 + vel_l2)) | 2.0 per step | 13.7 per step (rate_l2 92 is wrist-roll noise, std 6.5) |
| deterministic RMS acc arm / reversals per s | 33 / 3.9-4.4 | 60 / 3.8-4.0 |

- 100-seed CPU eval of the pilots at 3M (`eval_cpu.py`, fresh kinematics; `policy/pilot_*.pt`, extraction checked by
  so101-smooth's `check_extract_raw.py`, max diff 1.0e-6 / 1.9e-6): **scratch 0/100** (mean highest box 0.043 m: never
  lifted), **run 2 continued 13/100** (2 up at 6 s, lift at 2.69 s).
- **Choice: continue from run 2** (`resume:pilot_run2` to the 16M plan, 13M more; total steps 3 + 3 + 13 = 19M).
- **Why:** only the run-2 start lifts the box on 7v4_real at all (16 % of training episodes at 3M, rising at the end);
  from scratch, 3M steps gave no lift in any training or eval episode, only reaching. Once the switch puts the 0.1
  penalties in (2.0 per step against 0.68 task reward), exploration noise is penalised and a policy that has not
  found the lift by then is unlikely to (inference, not tested). so101Sim's own history agrees: run 1 from scratch
  with the old reward was at 13 % after 3M on the faster 12 V model (`docs/AGENT_CONTEXT.md` section 6).
- **Cost of the choice:** the run-2 start carries the bang-bang style (RMS acc 60 vs 33 from scratch) into the
  switch; its big rate term is mostly wrist-roll exploration noise, which the 0.1 penalty acts on directly.
- Steps: 16.0M PPO steps on 7v4_real in the chosen line (pilot 3.0M + main 13.0M), on top of run 2's 8.0M on the old
  model (SB3 num_timesteps 24,010,752). Command: `train_real.py ~/so101Sim main_run2
  resume:<...>/pilot_run2 13000000` (so101Sim venv); `config.json` records it.

## Training curves (`scratch/so101-rl-real/curves_main_run2.md`, from `progress.txt` / `evals.txt`)

Deterministic evals during training (20 episodes, seeds 20000-20019, joint-motion metrics of the arm, pooled):

| run steps | weight | success | up at 6 s | RMS acc arm rad/s² | reversals /s arm | >5 Hz vel RMS | peak speed |
|---|---|---|---|---|---|---|---|
| 3.0M (pilot end) | 1e-4 | 1/20 | 1/20 | 60.4 | 3.84 | 0.433 | 7.46 |
| 4.0M | 1e-4 | 2/20 | 0/20 | 62.0 | 3.94 | 0.462 | 7.71 |
| 5.0M (after switch at 4.44M) | 0.1 | 1/20 | 0/20 | 52.2 | 3.94 | 0.398 | 7.28 |
| 7.0M | 0.1 | 2/20 | 0/20 | 51.5 | 3.75 | 0.384 | 6.97 |
| 8.0M-15.0M | 0.1 | 0/20 each | 0/20 | 45.3-47.6 | 3.63-4.51 | 0.33-0.35 | 6.5-7.4 |
| 16.0M (final) | 0.1 | 0/20 | 0/20 | 45.6 | 3.97 | 0.335 | 6.86 |

Training episodes (stochastic, last 100): success 0.09-0.16 before the switch, 0.00-0.06 after; task reward per step
0.45 -> 0.26-0.37; penalty per step 5.5 just after the switch -> 2.8 at the end, of which joint velocity is 0.1 x 24
(qdot² summed over six joints: RMS joint speed about 2 rad/s throughout the episode) and action rate 0.1 x 4.0 (from
175 before the switch); policy std 0.44/0.18/0.12/1.09/6.60/0.22 -> 0.14/0.13/0.10/0.39/0.30/0.11. The penalty
(2.8) stays ten times the task reward (0.3) to the end, so the return is dominated by the penalty, yet PPO did not find
the low-motion behaviour: the arm keeps moving at about 2 rad/s RMS.

## Results: 100 seeds (10000-10099), 7v4_real, fresh kinematics

CPU: `eval_cpu.py ~/so101Sim policy/main_run2.pt 7v4_real cpu/main_run2 100` (so101Sim venv, MuJoCo
3.10). Table: `python3 table.py`; joint columns pan / lift / elbow / wflex / wroll / grip, every 5 ms physics step.

| controller | success | up at 6 s | time to lift s | max box z mean m | RMS acc rad/s² | RMS jerk ×10³ rad/s³ | reversals /s | vel RMS >5 Hz rad/s | peak speed rad/s |
|---|---|---|---|---|---|---|---|---|---|
| deployment-style expert (so101-expert), CPU | 100 | 100 | 2.61 | 0.188 | 2.1 / 6.6 / 5.3 / 5.8 / 3.3 / 7.6 | 0.3 / 1.0 / 1.0 / 0.8 / 0.1 / 0.9 | 0.36 / 0.79 / 0.50 / 0.55 / 0.29 / 0.93 | 0.018 / 0.042 / 0.035 / 0.037 / 0.011 / 0.079 | 2.0 / 2.6 / 2.4 / 2.6 / 3.1 / 2.7 |
| **main_run2 (this retrain), CPU** | **3** | **0** | 3.22 | 0.049 | 41 / 51 / 69 / 41 / 27 / 21 | 8.9 / 13 / 19 / 11 / 6.1 / 4.4 | 2.38 / 4.03 / 5.92 / 4.11 / 3.16 / 2.23 | 0.292 / 0.377 / 0.465 / 0.300 / 0.223 / 0.175 | 6.1 / 6.9 / 5.9 / 5.6 / 6.0 / 4.6 |
| pilot_run2 (3M, before the penalty), CPU | 13 | 2 | 2.69 | 0.079 | 55 / 72 / 71 / 60 / 39 / 24 | 14 / 19 / 19 / 15 / 6.8 / 4.8 | 3.36 / 4.97 / 4.87 / 4.29 / 2.60 / 2.14 | 0.404 / 0.515 / 0.517 / 0.467 / 0.302 / 0.225 | 6.2 / 7.6 / 6.1 / 7.5 / 7.4 / 8.3 |
| original PPO (run 2), CPU | 8 | 2 | 3.12 | 0.063 | 60 / 76 / 67 / 61 / 37 / 21 | 16 / 20 / 18 / 16 / 6.3 / 3.2 | 3.66 / 5.49 / 5.61 / 4.21 / 2.15 / 2.11 | 0.421 / 0.532 / 0.456 / 0.461 / 0.281 / 0.193 | 6.4 / 7.4 / 6.2 / 8.1 / 7.7 / 5.9 |
| isaac_scaled (so101-smooth v1), CPU | 15 | 1 | 2.84 | 0.094 | 55 / 73 / 66 / 57 / 37 / 21 | 14 / 20 / 18 / 15 / 6.3 / 3.8 | 3.41 / 5.66 / 6.18 / 5.21 / 2.44 / 2.03 | 0.391 / 0.518 / 0.452 / 0.428 / 0.273 / 0.202 | 6.5 / 7.6 / 7.8 / 8.0 / 8.4 / 5.5 |
| isaac_unscaled_raw (so101-smooth v2), CPU | 5 | 0 | 2.61 | 0.057 | 53 / 66 / 61 / 56 / 35 / 21 | 14 / 18 / 16 / 14 / 6.0 / 3.9 | 3.41 / 5.37 / 5.13 / 4.94 / 2.27 / 2.06 | 0.362 / 0.458 / 0.410 / 0.398 / 0.269 / 0.193 | 6.4 / 7.3 / 7.0 / 8.4 / 7.4 / 5.6 |
| scripted expert, incremental (so101-servo), CPU | 0 | 0 | – | 0.035 | 4.9 / 8.5 / 12 / 8.5 / 2.9 / 4.3 | 1.4 / 2.5 / 3.4 / 2.5 / 0.4 / 0.3 | 0.21 / 0.29 / 0.36 / 0.88 / 0.23 / 0.18 | 0.027 / 0.050 / 0.066 / 0.049 / 0.025 / 0.034 | 0.9 / 0.9 / 1.5 / 0.9 / 1.3 / 2.7 |

- The deployment expert row is so101-expert's own harness (`scratch/so101-expert/cpu/deploy_7v4_real/smooth.json`, 30 Hz
  absolute goals), the same `smoothness.joint_motion`; its CPU kinematics handling was not re-checked here.
- Command metrics (secondary; `eval_cpu.py`'s row): main_run2 commands a joint limit on 78.7 % of steps (pilot 76.6,
  original 74.2, v1 82.8, v2 74.6), mean action change 0.098 per step (pilot 0.134, original 0.114), commanded
  target change 0.160 rad per step (pilot 0.244). Torque at the duty limit: wrist flex / roll 14.5 / 8.0 % of steps
  (pilot 58.7 / 48.8), shoulder and elbow 48-67 % (pilot 60-68).
- Real arm, for scale (so101-expert hand-off, from LeRobot SO-101 teleop datasets): joint speeds p99 1.1-2.6 rad/s,
  max 2-4. main_run2's peak 4.6-6.9 rad/s is above both that and the servo's 5.29 rad/s firmware cap (gravity and
  coupling back-drive the joints past it, so101-servo `speed_check.py`).
- Outcome spread (`diag_motion.py cpu/main_run2`): highest box z percentiles 10/25/50/75/90 = 0.038 / 0.039 / 0.041 /
  0.050 / 0.059 m; successes on seeds 10024, 10049, 10050; seeds 10000 and 10010 fail (highest 0.040 m, final 0.023
  and 0.020 m: the box is tipped or pushed, not lifted). Typical outcome: seed 10022 (highest 0.041 m, nearest the
  median).

## Motion pattern (why it is not realistic)

`diag_motion.py cpu/main_run2` (100 episodes) and `raw_u.py` (20 episodes, `logs/raw_u.log`), per joint pan / lift /
elbow / wflex / wroll / grip:

| | main_run2 | pilot_run2 |
|---|---|---|
| clipped command at ±1 | 0.80 / 0.69 / 0.59 / 0.91 / 0.97 / 0.74 | 0.86 / 0.67 / 0.64 / 0.82 / 0.84 / 0.75 |
| raw output median \|u\| | 2.58 / 1.49 / 1.29 / 3.23 / 5.14 / 2.05 | 3.42 / 1.63 / 1.52 / 3.07 / 3.73 / 2.24 |
| full-range sign flips per step | 0.057 / 0.089 / 0.087 / 0.014 / 0.010 / 0.058 | 0.059 / 0.097 / 0.078 / 0.065 / 0.075 / 0.055 |
| joint RMS speed rad/s | 2.56 / 2.52 / 1.86 / 1.22 / 1.52 / 1.66 | 2.79 / 2.81 / 2.57 / 2.80 / 3.46 / 2.07 |
| velocity spectrum peak Hz | 1.33 / 1.67 / 1.33 / 0.17 / 0.17 / 0.83 | 0.83 / 1.00 / 1.33 / 1.17 / 0.50 / 0.67 |

- The policy is bang-bang: shoulder pan, lift and elbow flip their targets from one joint limit to the other 2.9-4.5
  times per second (0.057-0.089 per 20 ms step) and the firmware profile turns that into full-range swings at
  1.3-1.7 Hz. The penalty phase calmed only the wrist (flips 0.065-0.075 -> 0.010-0.014 per step; it had the large
  exploration noise, std 6.6 on wrist roll).
- **Inference, not tested:** the raw output sits beyond the clip (median |u| 1.3-5.1), where the env sees ±1 whatever
  the exact value. A held saturated output costs nothing in action rate, and the task reward has no gradient there, so
  the only pressure on a limit-to-limit flip is its one-step rate cost and the velocity cost it causes. This is the
  same structure so101-smooth found for the clipped penalty (`raw_actions.py`), shifted by the raw penalty rather than
  removed.

## MetalSim

Job `so101-rl-real-ms` ran 19:19:51-19:20:19 (mujoco_warp 93e5ce18; log `scratch/so101-rl-real/logs/metalsim_job.log`):
`scripts/gpu_run.sh so101-rl-real-ms render 15 -- bash runs/readme_media/scratch/so101-rl-real/run_metalsim.sh
main_run2=<...>/policy/main_run2.pt orig_fresh=runs/film/so101_lift/policy/so101_lift_ppo.pt
isaac_scaled_fresh=runs/film/so101_lift_smooth/policy/candidates/isaac_scaled.pt
isaac_unscaled_raw_fresh=runs/film/so101_lift_smooth/policy/candidates/isaac_unscaled_raw.pt`. Outputs
`scratch/so101-rl-real/metalsim/<name>/{metrics.json,smooth.json}`. Fresh kinematics (CPU kinematics of the GPU state).

| controller, MetalSim | success | up at 6 s | time to lift s | max box z mean m | RMS acc rad/s² | reversals /s | peak speed rad/s | seeds same as CPU |
|---|---|---|---|---|---|---|---|---|
| deployment-style expert (so101-expert's job) | 98 | 98 | 2.66 | 0.186 | 2.1 / 7.2 / 5.5 / 6.3 / 3.6 / 7.6 | 0.39 / 0.81 / 0.53 / 0.59 / 0.32 / 0.60 | 2.0-3.1 | (that stream) |
| **main_run2** | **2** | **0** | 1.95 | 0.048 | 43 / 56 / 74 / 44 / 29 / 23 | 2.54 / 4.59 / 6.42 / 4.41 / 3.53 / 2.25 | 6.0-6.9 | 95 |
| original PPO | 5 | 0 | 3.04 | 0.059 | 60 / 77 / 67 / 61 / 37 / 21 | 4.09 / 6.12 / 5.65 / 4.38 / 2.15 / 2.14 | 6.2-8.5 | 89 |
| isaac_scaled (v1) | 16 | 2 | 2.62 | 0.096 | 55 / 71 / 66 / 57 / 37 / 21 | 3.68 / 5.51 / 6.51 / 5.77 / 2.43 / 2.12 | 5.7-7.9 | 75 |
| isaac_unscaled_raw (v2) | 3 | 1 | 2.07 | 0.050 | 57 / 70 / 62 / 59 / 36 / 23 | 3.26 / 5.11 / 5.24 / 4.96 / 2.26 / 2.00 | 6.0-7.7 | 92 |

- main_run2: CPU 3 (10024, 10049, 10050), MetalSim 2 (10002, 10043); the joint-motion metrics agree within about 10 %
  (MetalSim slightly higher on every joint). Command metrics: limit share 78.8 % (CPU 78.7), action change 0.104
  (CPU 0.098).
- Solver: ITERATIONS cap hit in 17 worlds, LS_ITERATIONS in all 100 (as so101-servo reported for the expert); no
  overflow.
- Full table with jerk and >5 Hz columns: `python3 runs/readme_media/scratch/so101-rl-real/table.py`.

## Preview

`runs/readme_media/scratch/ik_preview/rl_7v4_real.mp4`: job `so101-rl-real-preview` (render, 19:20; MuJoCo's OpenGL
renderer, so it went through the queue): `render_policy.py ~/so101Sim
runs/readme_media/scratch/so101-rl-real/policy/main_run2.pt "PPO retrain (main_run2), 7v4_real"
runs/readme_media/scratch/ik_preview/rl_7v4_real.mp4 10010 10000 10022`; log `scratch/so101-rl-real/logs/preview.log`.

- Panels left to right: seed 10010, seed 10000, seed 10022 (the typical outcome: highest box z nearest the median).
  No seed in the preview succeeds; that is the typical outcome (97/100 on CPU).
- ffprobe: 1920x480, 50/1 fps, 300 frames, 6.000 s. The frame time is measured from `d.time` (0.0200 s per control
  step, constant): real time.
- Each panel is the seed's `eval_cpu.py` episode: the render log gives highest / final box z 0.040 / 0.020 (10010),
  0.040 / 0.023 (10000), 0.041 / 0.020 (10022), the same as `cpu/main_run2/reference.json`.

**Frames looked at** (Read tool; files in `scratch/ik_preview/`):

- `rl_7v4_real_strip.jpg`: every 50th frame (1 s), all three panels;
- `rl_7v4_real_10010_first1.5s.jpg`: seed 10010, frames 0-72, every 4th (0.08 s);
- `rl_7v4_real_10022_f150_185.jpg`: seed 10022, frames 150-184, every 2nd (3.02-3.70 s).

What they show:

- From the stretched start pose, the arm drops past the box in about 0.3 s, folds up and strikes it: the box rocks
  (z 0.036 m at 0.5-0.66 s) and falls onto its side (z 0.020 m) by 0.82 s (seed 10010). In all three seeds the box is
  on its side by about 1 s and stays there.
- After that, the arm changes posture from second to second: folded upright, arched over the box, swung out
  sideways. The wrist and gripper flip open and shut. It looks like what the metrics say: full-range swings, not a
  reach-and-grasp.
- There are calm stretches: seed 10022 hangs over the box with open jaws, nearly still, from 3.1 to 3.5 s. Then the
  shoulder jerks up within about 0.1 s.
- No visible jitter at the frame rate, and no interpenetration I could see at this scale. The labels and times read
  correctly.

## Open

- **Why PPO keeps the bang-bang style under the penalty is inferred, not tested** (Motion pattern). One structural
  difference from Isaac Lab is documented: its Franka lift uses `JointPositionActionCfg(scale=0.5,
  use_default_offset=True)` (`runs/film/so101_lift_smooth/eval/upstream/isaaclab_tasks_isaaclab_tasks_contrib_lift_config_franka_joint_pos_env_cfg.py:36-38`,
  IsaacLab 6bb65480): target = default pose + 0.5·u rad, and `clip_actions` is None, so no output is ever saturated.
  so101Sim maps u ∈ [-1, 1] onto the whole ctrlrange and clips (`so101_lift.py` `step`), so most of the policy's
  output space (|u| > 1, 59-97 % of steps here) is flat for the task.
- **Penalty scale:** after the switch the penalty (2.8 per step) is about ten times the task reward actually earned
  (0.3, reach only) and a third of so101Sim's largest per-step task reward (9). In Isaac Lab the same 0.1 weights sit
  against a largest task reward of 37 per step (`lift_env_cfg.py` RewardsCfg: 1 + 15 + 16 + 5). so101-smooth's v1
  scaled by 9/37 for this; this run used the weights unscaled (v2's choice, kept for comparability).
- The deployment-expert rows come from so101-expert's harnesses (CPU and MetalSim), not this stream's.
- No timing figure was taken (power at 18:25: AC, charging, 55 %, powermode 2).

## Next

1. **Do not film `main_run2` as a realistic controller.** It fails (3/100), and its arm joints' RMS acceleration is 7-20x the
   deployment-style expert's (reversals per second 5-12x). For a realistic SO-101 7v4_real cell, film so101-expert's `DeployExpert` (100/100 CPU)
   once its MetalSim eval is in (that stream's hand-off).
2. If a learned policy on 7v4_real is still wanted (owner's call), two increments, each from scratch, judged on
   `table.py`'s joint-motion columns against the deployment-expert row:
   - **(a) Isaac Lab's action parameterisation:** in a copy of `real_env.py`, target = home pose + 0.5·u rad per arm
     joint, clipped only to the joint range, and a binary gripper (open 1.4 / close -0.15, so101Sim's values), as
     `JointPositionActionCfg(scale=0.5, use_default_offset=True)` and `BinaryJointPositionActionCfg`; keep the raw
     penalties, scaled by 9/37 as so101-smooth v1 did. First check, before training, that the hypothesis holds:
     after 1-2M steps, the share of steps with the target at a joint limit should be far below 59-97 %.
   - **(b) Distil the deployment expert:** DAgger / BC of `DeployExpert` on 7v4_real (so101Sim has `rl/native/dagger.py`
     as the pattern; the expert is in `metalsim/render/film/so101_expert.py`); the student inherits the expert's
     motion if it tracks it.
3. Owner review: `runs/readme_media/scratch/ik_preview/rl_7v4_real.mp4` (real time, 6 s, three seeds).

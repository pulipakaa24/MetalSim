# Hand-off: so101-film-bam (2026-10-31, 20:44 to 22:10)

## Status

**Done.** `so101_lift_deploy` is re-filmed on `7v4_bam`: record, render, encode, README WebP and doc. The 7v4_real
media are replaced; `7v4_real` stays a servo option. Nothing committed.

- **Metal test:** `tests/test_bam_friction.py`, 8/8 passed in the GPU queue, the BatchSim test included.
- **Take outcome:** 100/100 success, 100 still up at 6 s, time to lift median 2.18 s (mean 2.23), 1 retry (seed 10024).
  It equals so101-bam-gpu's MetalSim eval (`deploy_7v4_bam/episodes.json`) on every seed: outcome, time to lift,
  retries and phase events identical on 100/100.
- **Seed kept: 10051**, typical. It lifts at 2.26 s, the 68th-72nd percentile, inside the IQR 2.10-2.30 s, with no
  retry, the same in every run.
- **actuation.py not touched.**

## Changed

- `metalsim/render/film/so101.py` (sha1 c4e67e8 → 46f1257; diff in `scratch/so101-film-bam/so101.py.diff`):
  - `SO101_LIFT_DEPLOY`: `servo="7v4_bam"`, plus its title, summary and comment;
  - module docstring line 40;
  - `SO101_SERVO_SECTION`:
    - header "(7v4_real, 7v4_bam)" and the intro;
    - a sentence that so101Sim's expert was not run on 7v4_bam;
    - a new subsection "7v4_bam: BAM's identified servo, with load-dependent gearbox friction";
  - `SO101_DEPLOY_SECTION`: rewritten as "Deployment-style scripted expert on 7v4_bam";
  - the `LookCfg` ground line is untouched.
- `so101_expert.py`: unchanged (md5 9884575e). `tasks.py`: not edited by me. Its md5 changed during the session
  (3af80cc → 67e3700), from another stream.
- `docs/renders/SO101_LIFT.md`: regenerated (`python -m metalsim.render.film so101_lift --stage doc`). The generated
  head and the tail ("Reproduce" on) are byte-identical; only the servo and deploy sections changed. Before and after
  copies are in `scratch/so101-film-bam/docgen/`.
- **Replaced:** `docs/renders/so101_lift_deploy.{webp,mp4,webp.json}` and `so101_lift_deploy_still{1,2,3}.jpg`.
- **Re-recorded and re-rendered:** `runs/film/so101_lift_deploy/{rec,hero,readme}`. The render stage moved the
  7v4_real frames to `*/stale`; I moved them to scratch and removed the duplicates.
- **Scratch** `runs/readme_media/scratch/so101-film-bam/` (git-ignored):
  - `real_take/`: the 7v4_real take's rec, hero and readme frames, and media;
  - scripts: `film_job.sh`, `compare_take.py`, `take_measure.py`, `droop.py`, `grasp_cpu.py`, `jaw_depth.py`,
    `side_by_side.py`;
  - their JSON outputs, `logs/`, the contact sheets and `deploy_section.md` (the section's draft).

## Verified

**GPU jobs.** Pins: Warp `52c09cdd82`, MuJoCo 3.14.0.

| job | command | ran | power |
|---|---|---|---|
| Metal test | `scripts/gpu_run.sh so101-bam-test check 10 -- … pytest -q tests/test_bam_friction.py` (`logs/pytest_gpu.log`) | 21:48:19, 8 passed in 3.58 s | AC, charging, 79 %, powermode 2 |
| film (record + render) | `scripts/gpu_run.sh so101-film-bam render 20 -- bash …/film_job.sh` (refuses unless the test log shows 8 passed) | 21:51:14-21:54:08, rc 0 | AC, charging, 79 %, powermode 2 at every stamp |

- **The fork moved between the eval and the take.** `upstream/mujoco_warp` went 93e5ce18 → 787a5b3a (EPA fixes) at
  21:48:31, by another stream (reflog). So the Metal test ran on 93e5ce18 and the take on `787a5b3a37`. The take
  still equals the eval seed for seed.
- **Take facts:** solver flags `LS_ITERATIONS` 100 and `ITERATIONS` 2 (as in the eval); no overflow; 1 frame
  correction.
- **Cache keys:** hero 887877812c6ecff38696, readme ab29464290c5fe6cb10b.

**Seed by seed** (`compare_take.py` → `compare_take.json`):

| reference | same outcome | time to lift |
|---|---|---|
| MetalSim eval | 100/100 | equal on 100; events identical on 100 |
| CPU 3.10 | 100/100 | within 0.04 s on all, equal on 71 |
| CPU 3.14 | 100/100 | within 0.04 s on all, equal on 86 |

**Encode (CPU):** the exact commands are in the doc's Reproduce block.

| output | result |
|---|---|
| mp4 | 0.6 MB; stills at frames 60, 110, 199 (kept: phase times match the old take's within 0.03 s) |
| WebP | 320×240, 150 frames, 0.66 MB, PSNR 34.9 dB, loop 0 |

**Measurements.**
- **Box tilt, measured:**

  | measure | 7v4_real | 7v4_bam |
  |---|---|---|
  | take, seed 10051 at 6 s (MetalSim; recovered from `rec.npz`; FK self-check 1e-14 rad) | 35.5° | 43.5° |
  | 100 seeds at 6 s, CPU 3.14 film scene (`grasp_cpu.py`): median / p90 / max | 29 / 42 / 47° | 40 / 50 / 57° |
  | at lift start, 100 seeds | 5° | 11° |
  | at lift start, the take | 6° | 14° |

  - Ablation: 7v4_bam with the gripper limited to 0.986 N·m gives 38° at 6 s. **The stronger gripper is not the
    cause; the cause is open.**
- **Grasp:**
  - the gripper sits at its limit, 1.71 against 0.99 N·m;
  - jaw normal-force sum 42 against 39 N. The gripper's own load-dependent friction holds 0.57 N·m (inference; the
    ablation agrees: 25 N with 0.34 N·m of friction);
  - jaw penetration median 0.94 mm on both.
  - In the take the visual jaw meshes reach ≤0.59 mm into the box (`jaw_depth.py`, both takes). The apparent overlap
    in the zooms is the jaw in front of the box's face.
  - In-hand motion from 3 to 6 s: 0.9 mm and 1.7° (7v4_real take: 2.9 mm and 2.7°).
- **Droop, seed 10051, hold 3-6 s** (`droop.py`). The takes' joint angles match the evals' trajectories within 8e-5
  rad, so the evals' commands apply to the takes.

  | cmd − q, mrad | shoulder lift | elbow | wrist flex |
  |---|---|---|---|
  | 7v4_real | 18 | 123 | 48 |
  | 7v4_bam | 0.3 | 45 | 12 |

  - The commands differ by ≤9 mrad on those joints. The poses differ by elbow 72-75, wrist flex 44-46 and lift 12 mrad.
  - Box at 6 s: 0.188 against 0.172 m. 7v4_real rises to 0.20 m, then sinks.
- **Lag and speed** (so101-expert `sim_lag.py` on the bam eval): mean lag 1.73° (7v4_real 3.06; datasets 1.94 and
  2.53); speed p99 1.27-2.66 rad/s.

**Frames looked at** (scratch):
- `real_stills.jpg`: the old stills, before replacement.
- `hero_strip_0p5s.jpg`. I first misread its hold tiles as changing. Ruled out by the pixel difference across the hold
  (mean |Δ| ≈ 2/765, frames 150-299) and by `hero_hold_check.jpg` (frames 100, 150, 200, 299 the same pose).
- `hero_grasp_lift_1p12_2p34s.jpg`: frames 56-119, every 3rd. The jaws close at 1.5-1.7 s and the box tips on the
  table, then lifts smoothly; no jumps.
- `side_by_side_full.jpg`: frames 60, 110, 149, 299, real | bam.
- `droop_hold_f299_crop.jpg`: crop plus a 50/50 blend. Base and shoulder are the same; jaws and box are higher on bam.
- `grasp_zoom_bam.jpg` (frames 85, 100, 199, 299) and `grasp_zoom_3x_real_vs_bam.jpg`.
- `bam_stills.jpg`.
- `webp_strip.jpg` and `webp_frames_0_60_90_135.jpg`. The WebP's hold frames differ by mean |Δ| ≈ 5 (codec).
- Framing: the README shot keeps ≥23 px above the arm; the right edge is touched only at frame 0, the shared start pose.

**Tests:** `pytest -q tests/test_film.py tests/test_no_machine_paths.py -k "not newton"` gives 5 passed. No machine
paths in my files, the doc, the sidecar or the take JSONs.

**Sources:**
- hand-offs so101-sag (Verified, Open 2-3, 6) and so101-bam-gpu (Steps 2-3, torque use, Open 1);
- `so101-bam-gpu/table.md`;
- so101-expert `cpu/*/episodes.json` for the 40-43° tilt medians;
- `upstream/mujoco_warp` reflog and log `93e5ce18e7..787a5b3a`;
- `so101.py:159-175` (BAM_M6, SERVOS) and the live `apply_servo` output (kp 10.575, clip 3.3247 / 1.7138, damping
  0.6189).
- **Experiment-only conclusions:** the gripper-ablation reading, and the tilt numbers on CPU.

## Open

1. **Tilt cause, and whether 40-43° is realistic:** not established. No real-arm measurement exists.
2. **Stall torque** 3.32 against 1.91 N·m: unresolved (a bench test). Shoulder residual: 1.6-2.4× real. Both are in
   the doc.
3. **The Metal test has not run on `787a5b3a`.** The take reproducing the eval exactly is indirect evidence only.
4. **`grasp_cpu.py` on 7v4_real gives 99/100 with 11 retries**, against `--cpu-dry`'s 97/10. The harnesses differ in
   float64 vs float32 states; the cause is not traced. 7v4_bam matches exactly.
5. **`tasks.py`'s comment** still says "the deployment-style scripted expert on the real-servo model". It is not wrong,
   but it is vague; left alone, since another stream edits that file.
6. **Not mine:** README.md, `docs/renders/README.md:13`, and other films. The 7v4_real take media survive only in scratch
   `real_take/`.

## Next

1. **README rework stream: the cell.**

   ```html
       <td align="center" width="33%">
         <img width="320" src="docs/renders/so101_lift_deploy.webp" alt="SO-101 arm on Rhoban BAM's identified model of its standard 7.4 V servos, load-dependent gearbox friction included, reaching over a green box, closing its jaws on it and lifting it off a wooden table: a scripted IK expert streaming joint goals at 30 Hz as LeRobot does, on MuJoCo Warp, path traced">
         <br><b>python -m metalsim.render.film so101_lift_deploy</b>
       </td>
   ```

   Caption sentence: "The SO-101 lifting a box with so101Sim's scripted IK plan, streamed as absolute joint goals at
   LeRobot's 30 Hz to Rhoban BAM's identified model of the standard follower's 7.4 V servos (load-dependent gearbox
   friction, LeRobot's registers): 100/100 on MuJoCo Warp and on CPU MuJoCo. Scripted, not learned; the cube pose comes
   from the simulator ([SO101_LIFT.md](docs/renders/SO101_LIFT.md#deployment-style-scripted-expert-on-7v4_bam))."
2. **Owner review** of `docs/renders/so101_lift_deploy.mp4` and `.webp`. For the A/B, use `scratch/so101-film-bam/real_take/docs/` and
   `side_by_side_full.jpg`.
3. **Optional Metal re-check** on the new pin: `scripts/gpu_run.sh <name> check 5 -- .venv/bin/python -m pytest -q tests/test_bam_friction.py`.
4. **Tilt investigation (CPU only):** `grasp_cpu.py 100 7v4_bam '<over json>' <out>.json` takes `apply_servo` overrides.
   Next candidates, one at a time:
   - the gripper's friction (no `friction` on the gripper dof);
   - its speed: the `speed_limit` / `accel_limit` that set how fast the jaws close.
